"""只读核验原始尺度损失历史run，并可导出有限预算结果。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from runoff_training.controlled_data import build_data, load_data, outer_split, inner_splits, sha256
from runoff_training.controlled_train import choose_epoch, write_json
from runoff_training.loss_train import validate_protocol, tag_for, choose, VARIANTS
from runoff_training.loss_model import restore_loss
from verify_runoff_direct import check_audit, check_model
from verify_runoff_strategy import recompute

ROOT = Path(__file__).resolve().parents[1]


def check_loss_audit(audit, ids, data, spec, variant, seed, epoch, phase, h):
    check_audit(audit, ids, data, spec, variant, seed, epoch, phase)
    assert audit['params']['loss_space'] == variant['loss_space']
    raw = StandardScaler().fit(data['y'][ids, h-1:h])
    np.testing.assert_allclose(audit['raw_target_mean'], raw.mean_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(audit['raw_target_scale'], raw.scale_, rtol=1e-12, atol=1e-12)


def verify(run, allow_smoke=False):
    run = Path(run).resolve()
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    spec = manifest['protocol']
    validate_protocol(spec)
    if manifest['status'] != 'completed' or spec['smoke'] and not allow_smoke:
        raise ValueError('只接受完整正式run，预检须显式允许')
    assert manifest['selection_frozen_before_outer']
    assert sha256(run/'selection.json') == manifest['selection_sha256']
    actual = {p.relative_to(run).as_posix() for p in run.rglob('*')
              if p.is_file() and p.name != 'manifest.json'}
    assert actual == set(manifest['frozen_files'])
    for name, digest in manifest['frozen_files'].items():
        assert sha256(run/name) == digest, name
    for name, digest in manifest['source_hashes'].items():
        assert sha256(run/'source'/name) == digest
    assert sha256(run/'source/任务1至7+原始尺度损失历史对照方案_2026-11-11.md') == manifest['plan_sha256']
    raw = ROOT/'Teacher Resources & Requirements/01047000.csv'
    assert sha256(raw) == manifest['data']['raw_sha256']
    dataset = run/'historical.npz'
    assert sha256(dataset) == manifest['data']['dataset_sha256']
    data = load_data(dataset)
    assert data['days'][-1] == np.datetime64('2003-12-31').astype('int64')
    with tempfile.TemporaryDirectory() as temporary:
        rebuilt_path = Path(temporary)/'data.npz'
        build_data(raw, rebuilt_path)
        rebuilt = load_data(rebuilt_path)
        assert set(data) == set(rebuilt)
        for key in data:
            np.testing.assert_equal(data[key], rebuilt[key])
    selection = json.loads((run/'selection.json').read_text(encoding='utf-8'))
    records = selection['records']
    expected = {tag_for(year, h, variant, seed) for year in spec['years'] for h in spec['horizons']
                for variant in spec['variants'] for seed in spec['seeds']}
    assert len(records) == len(expected) == manifest['outer_models'] == spec['outer_models']
    assert {r['tag'] for r in records} == expected
    assert {p.name for p in (run/'models').iterdir()} == expected
    assert {p.stem for p in (run/'cv').glob('*.csv')} == expected
    assert {p.stem for p in (run/'predictions').glob('*.csv')} == expected
    histories = {p.name:json.loads(p.read_text(encoding='utf-8')) for p in (run/'history').glob('*.json')}
    assert len(histories) == spec['fits'] == manifest['fits']
    frame = pd.read_csv(run/'metrics.csv')
    assert len(frame) == len(records)
    used, all_predictions = set(), []
    for record in records:
        year, h, seed, tag = [record[k] for k in ['year', 'task_h', 'seed', 'tag']]
        variant = next(v for v in spec['variants'] if v == record['variant'])
        assert tag == tag_for(year, h, variant, seed)
        train, valid = outer_split(data, year)
        folds = inner_splits(data, train)
        table = pd.read_csv(run/'cv'/f'{tag}.csv')
        assert [json.loads(p)['fixed_epochs'] for p in table.params] == spec['epochs']
        scores = np.column_stack([-table[f'split{k}_test_score'].to_numpy() for k in range(3)])
        assert np.isfinite(scores).all()
        np.testing.assert_allclose(scores.mean(axis=1), -table.mean_test_score.to_numpy(), rtol=1e-12)
        index = choose_epoch(scores.mean(axis=1), spec['epochs'])
        assert record['epochs'] == spec['epochs'][index]
        np.testing.assert_allclose(record['inner_RMSE'], scores[index].mean(), rtol=1e-12)
        np.testing.assert_allclose(record['inner_fold_RMSE'], scores[index], rtol=1e-12)
        own = {n:a for n,a in histories.items() if a['phase'] == tag+'_inner'}
        assert len(own) == 3*len(spec['epochs'])
        for epoch in spec['epochs']:
            for ti, vi in folds:
                ids = train[ti]
                assert data['origin_days'][ids[-1]]+7 < data['origin_days'][train[vi[0]]]
                found = [(n,a) for n,a in own.items() if a['params']['fixed_epochs'] == epoch
                         and a['last_id'] == int(ids[-1])]
                assert len(found) == 1
                name, audit = found[0]
                check_loss_audit(audit, ids, data, spec, variant, seed, epoch, tag+'_inner', h)
                used.add(name)
        model = restore_loss(run/'models'/tag, dataset, spec['device'])
        check_loss_audit(model.audit_, train, data, spec, variant, seed, record['epochs'], tag+'_outer', h)
        check_model(model, data, train, h)
        own = [(n,a) for n,a in histories.items() if a['phase'] == tag+'_outer']
        assert len(own) == 1 and own[0][1] == model.audit_
        used.add(own[0][0])
        assert data['origin_days'][train[-1]]+7 < data['origin_days'][valid[0]]
        pf = pd.read_csv(run/'predictions'/f'{tag}.csv').sort_values('sample_id')
        np.testing.assert_equal(pf.sample_id, valid)
        assert not pf.duplicated(['sample_id', 'h']).any()
        for key, value in dict(year=year, h=h, seed=seed, variant=variant['name'], epochs=record['epochs']).items():
            assert (pf[key] == value).all()
        origin = data['origin_days'][valid]
        np.testing.assert_equal(pf.origin_date, origin.astype('datetime64[D]').astype(str))
        np.testing.assert_equal(pf.target_date, (origin+h).astype('datetime64[D]').astype(str))
        np.testing.assert_equal(pf.truth, data['y'][valid, h-1])
        np.testing.assert_allclose(pf.pred, model.predict(valid[:, None])[:, 0], rtol=1e-7, atol=1e-5)
        metric = frame.loc[(frame.year == year) & (frame.h == h) & (frame.seed == seed)
                           & (frame.variant == variant['name'])]
        assert len(metric) == 1 and int(metric.iloc[0].epochs) == record['epochs']
        for key, value in recompute(pf.truth.to_numpy(), pf.pred.to_numpy()).items():
            np.testing.assert_allclose(metric.iloc[0][key], value, rtol=1e-10, atol=1e-8)
        all_predictions.append(pf)
    assert used == set(histories)
    assert len(selection['selections']) == 6
    assert {(s['year'], s['h']) for s in selection['selections']} == {(y,h) for y in spec['years'] for h in spec['horizons']}
    for selected in selection['selections']:
        own = [r for r in records if r['year'] == selected['year'] and r['task_h'] == selected['h']]
        scores = {v['name']:float(np.mean([r['inner_RMSE'] for r in own if r['variant'] == v])) for v in VARIANTS}
        assert selected['inner_scores'] == scores
        assert selected['selected_variant'] == choose(scores)
        assert selected['selected_log'] == choose({k:v for k,v in scores.items() if k.endswith('_log')}, VARIANTS[:2])
    result = dict(status='passed', smoke=spec['smoke'], fits=spec['fits'], models=len(records),
        prediction_rows=sum(len(p) for p in all_predictions), frozen_files=len(actual),
        max_target='2003-12-31', selection_frozen_before_outer=True,
        checks=['固定协议/工件hash', '因果历史数组重建', '逐fit配置/原尺度归一化统计',
                '实际log scaler/网络/buffer', '全部恢复/日期/指标', 'CV和选择重算'],
        limitations=['审计声明不独立证明随机过程', '内层log scaler未全部保存；重算实际外层scaler'])
    return result, pd.concat(all_predictions, ignore_index=True), selection, frame


def select_rows(frame, selections, key):
    parts = [frame.loc[(frame.year == s['year']) & (frame.h == s['h']) & (frame.variant == s[key])]
             for s in selections]
    if any(p.empty for p in parts):
        raise ValueError('冻结选择没有匹配结果')
    return pd.concat(parts, ignore_index=True)


def gate_result(frame, selections, spec):
    selected = select_rows(frame, selections, 'selected_variant')
    baseline = select_rows(frame, selections, 'selected_log')
    keys = ['year', 'h', 'seed']
    paired = selected.merge(baseline, on=keys, suffixes=('', '_log'), validate='one_to_one')
    paired['RMSE_change_pct'] = 100*(paired.RMSE/paired.RMSE_log-1)
    paired['MAE_change_pct'] = 100*(paired.MAE/paired.MAE_log-1)
    rows = []
    for year, own in paired.groupby('year'):
        gain = 100*(1-own.RMSE.mean()/own.RMSE_log.mean())
        mae = 100*(own.MAE.mean()/own.MAE_log.mean()-1)
        byseed = own.groupby('seed')[['RMSE', 'RMSE_log']].mean()
        byh = own.groupby('h')[['RMSE', 'RMSE_log']].mean()
        seeds = int((byseed.RMSE < byseed.RMSE_log).sum())
        horizons = int((byh.RMSE < byh.RMSE_log).sum())
        worst = float((100*(byh.RMSE/byh.RMSE_log-1)).max())
        gate = spec['gate']
        passed = (gain >= gate['min_year_RMSE_gain_pct'] and seeds >= gate['min_seeds']
                  and horizons >= gate['min_horizons'] and worst <= gate['max_h_RMSE_regression_pct']
                  and mae <= gate['max_year_MAE_regression_pct'])
        rows.append(dict(year=int(year), RMSE=float(own.RMSE.mean()), RMSE_log=float(own.RMSE_log.mean()),
            RMSE_gain_pct=float(gain), MAE_change_pct=float(mae), improved_seeds=seeds,
            improved_horizons=horizons, worst_h_RMSE_change_pct=worst, passed=bool(passed)))
    return dict(status='smoke_only' if spec['smoke'] else 'historical_development',
                advance=bool(not spec['smoke'] and all(r['passed'] for r in rows)), years=rows), paired


def compare_legacy_predictions(current, previous):
    """按调用方已核对的样本顺序比较，不把pandas行索引误作预测身份。"""
    for key in ['origin_date', 'target_date', 'truth']:
        np.testing.assert_equal(current[key].to_numpy(), previous[key].to_numpy())
    np.testing.assert_allclose(current.pred.to_numpy(), previous.pred.to_numpy(), rtol=0, atol=1e-10)
    return float(np.max(np.abs(current.pred.to_numpy()-previous.pred.to_numpy())))


def export(run, out, allow_smoke=False):
    verification, predictions, selection, frame = verify(run, allow_smoke)
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    spec = manifest['protocol']
    gate, paired = gate_result(frame, selection['selections'], spec)
    out.mkdir(parents=True, exist_ok=False)
    for name in ['manifest.json', 'metrics.csv', 'selection.json', 'inner_splits.csv']:
        shutil.copy2(run/name, out/name)
    write_json(out/'verification.json', verification)
    write_json(out/'development_gate.json', gate)
    paired.to_csv(out/'selected_vs_log.csv', index=False, encoding='utf-8-sig')
    frame.groupby(['year', 'h', 'variant'])[['RMSE', 'MAE', 'R2']].mean().reset_index().to_csv(
        out/'annual_summary.csv', index=False, encoding='utf-8-sig')
    diagnostics = []
    for (year, h, variant), group in predictions.groupby(['year', 'h', 'variant']):
        threshold = group.truth.quantile(.9)
        high = group.truth >= threshold
        error = group.pred-group.truth
        diagnostics.append(dict(year=year, h=h, variant=variant, q90=threshold,
            high10_sse_share=float(np.square(error[high]).sum()/np.square(error).sum()),
            high10_mean_bias=float(error[high].mean())))
    pd.DataFrame(diagnostics).to_csv(out/'highflow_diagnostics.csv', index=False, encoding='utf-8-sig')
    select_rows(predictions, selection['selections'], 'selected_variant').to_csv(
        out/'selected_predictions.csv', index=False, encoding='utf-8-sig')
    comparison = []
    old = ROOT/'output/acceptance3_runs/formal_20261111_direct_expand_v1'
    if not spec['smoke']:
        previous = json.loads((old/'manifest.json').read_text(encoding='utf-8'))
        assert previous['status'] == 'completed' and previous['historical_data']['raw_sha256'] == manifest['data']['raw_sha256']
        previous_records = json.loads((old/'selection.json').read_text(encoding='utf-8'))['records']
        for record in selection['records']:
            if record['variant']['loss_space'] != 'log':
                continue
            legacy = 'last_'+record['variant']['name']
            matched = [r for r in previous_records if r['kind'] == 'searched' and r['year'] == record['year']
                       and r['task_h'] == record['task_h'] and r['seed'] == record['seed'] and r['variant']['name'] == legacy]
            assert len(matched) == 1 and matched[0]['epochs'] == record['epochs']
            path = old/'predictions'/f'{matched[0]["tag"]}.csv'
            assert sha256(path) == previous['frozen_files'][path.relative_to(old).as_posix()]
            before = pd.read_csv(path).sort_values('sample_id')
            current = predictions.loc[(predictions.year == record['year']) & (predictions.h == record['task_h'])
                & (predictions.seed == record['seed']) & (predictions.variant == record['variant']['name'])].sort_values('sample_id')
            difference = compare_legacy_predictions(current, before)
            comparison.append(dict(tag=record['tag'], max_abs_prediction_difference=difference))
    pd.DataFrame(comparison).to_csv(out/'legacy_log_baseline_check.csv', index=False, encoding='utf-8-sig')
    text = ['# 原始尺度损失历史对照', '', '日期：2026-11-11。仅2002/2003开发评价，不含2004新成绩。', '',
        f'预检：{spec["smoke"]}；{spec["fits"]}fit、{spec["outer_models"]}模型恢复核验通过。', '',
        '| 年份 | h | 内层所选 | log对照R² | 所选R² | log对照RMSE | 所选RMSE |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for (year,h), own in paired.groupby(['year','h']):
        text.append(f'| {year} | {h} | {own.variant.iloc[0]} | {own.R2_log.mean():.4f} | {own.R2.mean():.4f} | {own.RMSE_log.mean():.2f} | {own.RMSE.mean():.2f} |')
    text += ['', f'预固定推进门槛通过：{gate["advance"]}；不按外层换候选，不自动开展2004评价。', '',
        '[全部固定候选](annual_summary.csv) · [逐seed对照](selected_vs_log.csv) · [推进条件](development_gate.json) · [核验](verification.json)', '',
        '原CSV、Word和旧run未覆盖，未commit/push。多seed不是独立年份；原尺度评分与训练损失空间不是同一个概念。']
    (out/'README.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    (out/'analysis_source').mkdir()
    shutil.copy2(Path(__file__), out/'analysis_source'/Path(__file__).name)
    write_json(out/'analysis_manifest.json', dict(status='completed', run_id=run.name,
        run_manifest_sha256=sha256(run/'manifest.json'),
        files={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()}))
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run = ROOT/'output/acceptance3_runs'/args.run_id
    with threadpool_limits(limits=4):
        result = export(run, args.out, args.allow_smoke) if args.out else verify(run, args.allow_smoke)[0]
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
