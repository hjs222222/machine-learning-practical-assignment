"""第1天专项只读核验：固定预算、因果数组、折内统计、恢复与历史推进条件。"""
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

from runoff_training.controlled_data import build_data, load_data, outer_split, sha256
from runoff_training.controlled_train import metrics, write_json
from runoff_training.day1_train import validate_protocol, model_for, folds_for, pooled_score, choose, FAMILIES
from runoff_training.day1_model import restore_day1, tabular, DeltaNet

ROOT = Path(__file__).resolve().parents[1]


def check_audit(audit, expected, ids, data):
    params = expected.get_params()
    assert audit['params'] == params
    assert audit['phase'] == expected.phase
    assert (audit['n_train'], audit['first_id'], audit['last_id']) == (len(ids), int(ids[0]), int(ids[-1]))
    days = data['origin_days']
    assert audit['train_last_origin'] == str(np.datetime64(int(days[ids[-1]]), 'D'))
    assert audit['train_last_h7'] == str(np.datetime64(int(days[ids[-1]]+7), 'D'))
    rows = np.unique((data['positions'][ids, None]+np.arange(1-expected.window, 1)).ravel())
    assert audit['scaler_last_day'] == str(np.datetime64(int(data['days'][rows[-1]]), 'D'))
    tab = tabular(data, ids)
    for label, values in [('tab', tab), ('delta', data['y'][ids, :1]-tab[:, :1])]:
        scaler = StandardScaler().fit(values)
        for key, value in [('mean', scaler.mean_), ('scale', scaler.scale_)]:
            np.testing.assert_allclose(audit[label+'_'+key], value, rtol=1e-12, atol=1e-12)
    history = audit['history']
    n = expected.epochs if expected.family in ['ANN', 'LSTM'] else 0
    assert len(history) == n
    for i, entry in enumerate(history, 1):
        assert entry['epoch'] == i and entry['gradient_steps'] == i*int(np.ceil(len(ids)/64))
        assert np.isfinite(entry['loss']) and entry['loss'] >= 0


def check_model(model, expected, ids, data):
    assert model.get_params() == expected.get_params()
    rows = np.unique((data['positions'][ids, None]+np.arange(1-model.window, 1)).ravel())
    tab = tabular(data, ids)
    for actual, values in [(model.tab_scaler_, tab), (model.y_scaler_, data['y'][ids, :1]-tab[:, :1]),
                           (model.x_scaler_, data['features'][rows, :9])]:
        rebuilt = StandardScaler().fit(values)
        assert actual.n_samples_seen_ == len(values)
        for key in ['mean_', 'scale_', 'var_']:
            np.testing.assert_allclose(getattr(actual, key), getattr(rebuilt, key), rtol=1e-12, atol=1e-12)
    if model.family == 'RF':
        p = model.model_.get_params()
        for key, value in dict(n_estimators=model.trees, min_samples_leaf=model.leaf, max_depth=10,
                               random_state=model.seed, n_jobs=1).items():
            assert p[key] == value
        assert len(model.model_.estimators_) == model.trees
    elif model.family == 'Ridge':
        assert model.model_.alpha == model.alpha
    else:
        reference = DeltaNet(model.family).state_dict()
        actual = model.model_.state_dict()
        assert set(reference) == set(actual)
        for key in actual:
            assert reference[key].shape == actual[key].shape and torch.isfinite(actual[key]).all()


def verify(run, allow_smoke=False):
    run = Path(run).resolve()
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    spec = manifest['protocol']
    validate_protocol(spec)
    if manifest['status'] != 'completed' or spec['smoke'] and not allow_smoke:
        raise ValueError('仅核验完成的正式run；预检须显式允许')
    assert manifest['selection_frozen_before_outer']
    assert sha256(run/'selection.json') == manifest['selection_sha256']
    actual = {p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    assert actual == set(manifest['frozen_files'])
    for name, digest in manifest['frozen_files'].items():
        assert sha256(run/name) == digest, name
    for name, digest in manifest['source_hashes'].items():
        assert sha256(run/'source'/name) == digest
    assert sha256(run/'source/任务1至7+第一天R2专项目标与方案_2026-11-11.md') == manifest['plan_sha256']
    raw = ROOT/'Teacher Resources & Requirements/01047000.csv'
    dataset = run/'historical.npz'
    assert sha256(raw) == manifest['data']['raw_sha256']
    assert sha256(dataset) == manifest['data']['dataset_sha256']
    data = load_data(dataset)
    assert data['days'][-1] == np.datetime64('2003-12-31').astype('int64')
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'rebuilt.npz'
        build_data(raw, path)
        rebuilt = load_data(path)
        assert set(data) == set(rebuilt)
        for key in data:
            np.testing.assert_equal(data[key], rebuilt[key])
    selection = json.loads((run/'selection.json').read_text(encoding='utf-8'))
    records = selection['records']
    tags = {f'y{y}_{c["name"]}_s{s}' for y in spec['years'] for c in spec['candidates'] for s in spec['seeds']}
    assert len(records) == len(tags) == spec['outer_models'] == manifest['outer_models']
    assert {r['tag'] for r in records} == tags
    assert {p.name for p in (run/'models').iterdir()} == tags
    assert {p.stem for p in (run/'predictions').glob('*.csv')} == tags
    cv_tags = {f'y{y}_{f}_s{s}' for y in spec['years'] for f in FAMILIES for s in spec['seeds']}
    assert {p.stem for p in (run/'cv').glob('*.csv')} == cv_tags
    histories = {p.name:json.loads(p.read_text(encoding='utf-8')) for p in (run/'history').glob('*.json')}
    assert len(histories) == spec['fits'] == manifest['fits']
    frame = pd.read_csv(run/'metrics.csv')
    assert len(frame) == len(records) and not frame.duplicated(['year', 'candidate', 'seed']).any()
    used, predictions, split_rows = set(), [], []
    for year in spec['years']:
        train, valid = outer_split(data, year)
        folds = folds_for(data, train)
        for fold,(ti,vi) in enumerate(folds):
            split_rows.append(dict(year=year, fold=fold, n_train=len(ti), n_valid=len(vi),
                train_last_h7=str(np.datetime64(int(data['origin_days'][train[ti[-1]]]+7), 'D')),
                valid_first=str(np.datetime64(int(data['origin_days'][train[vi[0]]]), 'D'))))
        truths = [data['y'][train[vi], 0] for _,vi in folds]
        for family in FAMILIES:
            configs = [c for c in spec['candidates'] if c['family'] == family]
            for seed in spec['seeds']:
                phase = f'y{year}_{family}_s{seed}_inner'
                table = pd.read_csv(run/'cv'/f'y{year}_{family}_s{seed}.csv')
                assert [json.loads(p) for p in table.params] == [{k:v for k,v in c.items() if k not in ['name', 'family']} for c in configs]
                assert len(table) == len(configs)
                scores = np.array([[table[f'split{k}_test_score'].iloc[i] for k in range(4)] for i in range(len(configs))])
                np.testing.assert_allclose(scores.mean(axis=1), table.mean_test_score, rtol=1e-12, atol=1e-12)
                for i,c in enumerate(configs):
                    record = next(r for r in records if r['year'] == year and r['seed'] == seed and r['candidate'] == c)
                    np.testing.assert_allclose(record['inner_fold_R2'], scores[i], rtol=1e-12, atol=1e-12)
                    value = pooled_score(scores[i], truths)
                    np.testing.assert_allclose([record['inner_R2'], table.pooled_R2.iloc[i]], value, rtol=1e-12, atol=1e-12)
                    for ti,vi in folds:
                        ids = train[ti]
                        assert data['origin_days'][ids[-1]]+7 < data['origin_days'][train[vi[0]]]
                        expected = model_for(dataset, spec, c, seed, run/'history', phase)
                        own = [(n,a) for n,a in histories.items() if a['phase'] == phase and a['last_id'] == int(ids[-1])
                               and a['params'] == expected.get_params()]
                        assert len(own) == 1
                        check_audit(own[0][1], expected, ids, data)
                        used.add(own[0][0])
    pd.testing.assert_frame_equal(pd.read_csv(run/'inner_splits.csv'), pd.DataFrame(split_rows))
    for r in records:
        year, seed, c, tag = [r[k] for k in ['year', 'seed', 'candidate', 'tag']]
        assert c in spec['candidates'] and tag == f'y{year}_{c["name"]}_s{seed}'
        train, valid = outer_split(data, year)
        expected = model_for(dataset, spec, c, seed, run/'history', tag+'_outer')
        model = restore_day1(run/'models'/tag, dataset, expected.device)
        check_audit(model.audit_, expected, train, data)
        check_model(model, expected, train, data)
        own = [(n,a) for n,a in histories.items() if a['phase'] == tag+'_outer']
        assert len(own) == 1 and own[0][1] == model.audit_
        used.add(own[0][0])
        pf = pd.read_csv(run/'predictions'/f'{tag}.csv')
        assert len(pf) == len(valid) and not pf.sample_id.duplicated().any()
        for key,value in dict(year=year, seed=seed, candidate=c['name'], family=c['family']).items():
            assert (pf[key] == value).all()
        for key,value in dict(sample_id=valid, truth=data['y'][valid, 0],
                origin_date=data['origin_days'][valid].astype('datetime64[D]').astype(str),
                target_date=(data['origin_days'][valid]+1).astype('datetime64[D]').astype(str)).items():
            np.testing.assert_equal(pf[key].to_numpy(), value)
        np.testing.assert_allclose(pf.pred, model.predict(valid)[:, 0], rtol=1e-7, atol=1e-5)
        row = frame.loc[(frame.year == year) & (frame.seed == seed) & (frame.candidate == c['name'])]
        assert len(row) == 1 and row.family.iloc[0] == c['family']
        for key,value in metrics(pf.truth.to_numpy()[:, None], pf.pred.to_numpy()[:, None])[0].items():
            np.testing.assert_allclose(row.iloc[0][key], value, rtol=1e-10, atol=1e-8)
        predictions.append(pf)
    assert used == set(histories)
    assert len(selection['selections']) == len(spec['years'])
    assert {s['year'] for s in selection['selections']} == set(spec['years'])
    for s in selection['selections']:
        own = [r for r in records if r['year'] == s['year']]
        scores = {c['name']:float(np.mean([r['inner_R2'] for r in own if r['candidate'] == c])) for c in spec['candidates']}
        winners = {f:choose({c['name']:scores[c['name']] for c in spec['candidates'] if c['family'] == f}) for f in FAMILIES}
        assert s['scores'] == scores and s['family_winners'] == winners
        assert s['selected_variant'] == choose({w:scores[w] for w in winners.values()})
    result = dict(status='passed', smoke=spec['smoke'], fits=spec['fits'], models=len(tags),
        prediction_rows=sum(map(len, predictions)), frozen_files=len(actual), max_target='2003-12-31',
        checks=['固定协议/全部工件hash', '历史数组重建', '四折净化/逐fit配置与审计统计',
                '实际外层scaler/网络结构', '全部模型恢复/日期/真值/指标', '合并R²/配置选择重算'],
        limitations=['内层未保存全部权重及scaler，审计不独立证明随机过程或内层实际预测',
                     '合并R²从逐折R²和相同真值SST还原SSE；不是独立重训'])
    return result, pd.concat(predictions, ignore_index=True), selection, frame


def selected_rows(frame, selections, family=None):
    parts = [frame.loc[(frame.year == s['year']) & (frame.candidate ==
             (s['family_winners'][family] if family else s['selected_variant']))] for s in selections]
    if any(p.empty for p in parts):
        raise ValueError('内层冻结选择无对应结果')
    return pd.concat(parts, ignore_index=True)


def gate_result(paired, spec):
    rows = []
    for year in spec['years']:
        own = paired.loc[paired.year == year]
        if sorted(own.seed.tolist()) != sorted(spec['seeds']):
            raise ValueError('推进比较缺失或重复seed')
        r2 = float(own.R2.mean())
        gain = float((own.R2-own.R2_old).mean())
        mae = float(100*(own.MAE.mean()/own.MAE_old.mean()-1))
        seeds = int((own.R2 > own.R2_old).sum())
        g = spec['gate']
        rows.append(dict(year=year, R2=r2, R2_old=float(own.R2_old.mean()), R2_gain=gain,
            MAE_change_pct=mae, improved_seeds=seeds, target_08_reached=bool(r2 >= spec['target_R2']),
            passed=bool(r2 >= g['min_year_R2'] and gain >= g['min_R2_gain']
                        and seeds >= g['min_improved_seeds'] and mae <= g['max_MAE_regression_pct'])))
    return dict(status='smoke_only' if spec['smoke'] else 'historical_development',
                advance=bool(not spec['smoke'] and all(r['passed'] for r in rows)), years=rows)


def legacy_comparison(run, predictions, selection, frame):
    old = ROOT/'output/acceptance3_runs/formal_20261111_physical_loss_v1'
    manifest = json.loads((old/'manifest.json').read_text(encoding='utf-8'))
    current_manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed' and manifest['data']['raw_sha256'] == current_manifest['data']['raw_sha256']
    baseline, provenance = [], []
    for s in selection['selections']:
        for seed in current_manifest['protocol']['seeds']:
            path = old/'predictions'/f'y{s["year"]}_direct_h1_residual_log_s{seed}.csv'
            assert sha256(path) == manifest['frozen_files'][path.relative_to(old).as_posix()]
            before = pd.read_csv(path).sort_values('origin_date')
            current = selected_rows(predictions, [s], 'LSTM')
            current = current.loc[current.seed == seed].sort_values('origin_date')
            for key in ['origin_date', 'target_date', 'truth']:
                np.testing.assert_equal(current[key].to_numpy(), before[key].to_numpy())
            values = metrics(before.truth.to_numpy()[:, None], before.pred.to_numpy()[:, None])[0]
            baseline.append(dict(year=s['year'], seed=seed, **{k:values[k] for k in ['R2', 'RMSE', 'MAE']}))
            provenance.append(dict(run_id=old.name, path=path.relative_to(old).as_posix(), sha256=sha256(path)))
    paired = selected_rows(frame, selection['selections'], 'LSTM').merge(
        pd.DataFrame(baseline), on=['year', 'seed'], suffixes=('', '_old'), validate='one_to_one')
    return paired, provenance


def export(run, out, allow_smoke=False):
    run, out = Path(run).resolve(), Path(out)
    verification, predictions, selection, frame = verify(run, allow_smoke)
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    spec = manifest['protocol']
    if spec['smoke']:
        gate, paired, provenance = dict(status='smoke_only', advance=False), pd.DataFrame(), []
    else:
        paired, provenance = legacy_comparison(run, predictions, selection, frame)
        gate = gate_result(paired, spec)
    out.mkdir(parents=True, exist_ok=False)
    for name in ['manifest.json', 'selection.json', 'metrics.csv', 'inner_splits.csv']:
        shutil.copy2(run/name, out/name)
    write_json(out/'verification.json', verification)
    write_json(out/'development_gate.json', gate)
    write_json(out/'legacy_sources.json', provenance)
    if not paired.empty:
        paired.to_csv(out/'LSTM_vs_old_log.csv', index=False, encoding='utf-8-sig')
    winners = pd.concat([selected_rows(frame, selection['selections'], f) for f in FAMILIES], ignore_index=True)
    winners.to_csv(out/'family_winners.csv', index=False, encoding='utf-8-sig')
    selected_rows(predictions, selection['selections']).to_csv(out/'selected_predictions.csv', index=False, encoding='utf-8-sig')
    selected_rows(predictions, selection['selections'], 'LSTM').to_csv(out/'LSTM_predictions.csv', index=False, encoding='utf-8-sig')
    frame.groupby(['year', 'family', 'candidate'])[['R2', 'RMSE', 'MAE']].mean().reset_index().to_csv(
        out/'all_candidates.csv', index=False, encoding='utf-8-sig')
    persistence = []
    data = load_data(run/'historical.npz')
    for year in spec['years']:
        _, valid = outer_split(data, year)
        persistence.append(dict(year=year, **metrics(data['y'][valid, :1], tabular(data, valid)[:, :1])[0]))
    pd.DataFrame(persistence).to_csv(out/'persistence.csv', index=False, encoding='utf-8-sig')
    text = ['# 第1天R²专项历史结果', '', '日期：2026-11-11。仅2002/2003开发评价，不含2004新成绩。', '',
        f'预检：{spec["smoke"]}；{spec["fits"]}fit、{spec["outer_models"]}模型全部恢复核验通过。', '',
        '| 年份 | 模型族 | 内层所选配置 | R²三seed均值 | RMSE | MAE |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for (year,family), own in winners.groupby(['year', 'family']):
        text.append(f'| {year} | {family} | {own.candidate.iloc[0]} | {own.R2.mean():.4f} | {own.RMSE.mean():.2f} | {own.MAE.mean():.2f} |')
    text += ['', f'预固定LSTM跨年推进门槛通过：{gate["advance"]}。0.8仍为最终2004第1天目标，不以某历史年份替代。', '',
        '[全部候选](all_candidates.csv) · [各seed所选结果](family_winners.csv) · [推进条件](development_gate.json) · [核验](verification.json) · [真实持久性](persistence.csv)', '',
        '不同族输入形式/容量不同，属于完整方案对照。内层以合并R²选择，非简单平均各折R²；不能按外层重新挑选。', '',
        '固定Q(t)跳连；标准化增量头零初始化实际为Q(t)+训练增量均值，不是严格初始持久性。没有裁负预测。', '',
        '原数据、旧run与Word未覆盖，未commit/push。多seed不是独立年份。']
    (out/'README.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    (out/'analysis_source').mkdir()
    shutil.copy2(__file__, out/'analysis_source'/Path(__file__).name)
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
    with threadpool_limits(limits=4):
        run = ROOT/'output/acceptance3_runs'/args.run_id
        result = export(run, args.out, args.allow_smoke) if args.out else verify(run, args.allow_smoke)[0]
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
