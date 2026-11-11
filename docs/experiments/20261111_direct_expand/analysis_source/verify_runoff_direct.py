"""直接残差实验只读核验：固定协议、实际模型/缩放器、恢复预测和来源。"""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from runoff_training.controlled_data import build_data, load_data, inner_splits, sha256, MAX_WINDOW
from runoff_training.direct_data import build_evaluation_data
from runoff_training.direct_train import validate_protocol, split_for, tag_for
from runoff_training.strategy_model import restore_strategy
from verify_runoff_strategy import recompute

ROOT = Path(__file__).resolve().parents[1]


def check_audit(audit, ids, data, spec, variant, seed, epoch, phase):
    params = audit['params']
    expected = dict(profile='w90_base', seed=seed, fixed_epochs=epoch, device=spec['device'],
        units=32, batch_size=64, lr=.001, pooling='last', residual=variant['residual'],
        target_transform='log', phase=phase)
    if any(params[k] != v for k, v in expected.items()):
        raise ValueError('训练配置/seed声明与固定协议不一致')
    assert audit['phase'] == phase
    assert (audit['n_train'], audit['first_id'], audit['last_id']) == (len(ids), int(ids[0]), int(ids[-1]))
    last = int(data['origin_days'][ids[-1]])
    assert audit['train_last_origin'] == str(np.datetime64(last, 'D'))
    assert audit['train_last_h7_target'] == str(np.datetime64(last+7, 'D'))
    assert audit['scaler_last_day'] == audit['train_last_origin']
    # 单层17输入/32隐层LSTM，加32维head和单输出；残差buffer不计参数。
    assert audit['n_parameters'] == 4*32*(17+32+2) + 32*32+32 + 32+1
    assert len(audit['history']) == epoch
    for i, row in enumerate(audit['history'], 1):
        assert row['epoch'] == i and row['gradient_steps'] == i*int(np.ceil(len(ids)/64))
        assert np.isfinite(row['loss'])


def check_model(model, data, train, h):
    rows = np.unique((data['positions'][train, None]+np.arange(1-MAX_WINDOW, 1)).ravel())
    x = StandardScaler().fit(data['features'][rows])
    y = StandardScaler().fit(np.log(data['y'][train, h-1:h]))
    for actual, reference in [(model.x_scaler_, x), (model.y_scaler_, y)]:
        np.testing.assert_allclose(actual.mean_, reference.mean_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.scale_, reference.scale_, rtol=1e-12, atol=1e-12)
    net = model.model_
    assert net.lstm.input_size == 17 and net.lstm.hidden_size == 32 and net.lstm.num_layers == 1
    assert net.head[0].in_features == net.head[0].out_features == 32
    assert net.head[-1].out_features == model.output_dim_ == 1 and net.drop.p == .1
    assert sum(p.numel() for p in net.parameters()) == model.audit_['n_parameters']
    for actual, expected in [(net.x_mean, x.mean_[0]), (net.x_scale, x.scale_[0]),
                             (net.y_mean, y.mean_), (net.y_scale, y.scale_)]:
        np.testing.assert_equal(actual.detach().cpu().numpy(), np.asarray(expected, dtype='float32'))
    assert net.pooling == model.pooling == 'last' and net.residual == model.residual
    assert net.target_transform == model.target_transform == 'log'
    for k, v in model.get_params().items():
        assert v == model.audit_['params'][k]


def verify(run, allow_smoke=False):
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    spec = manifest['protocol']
    validate_protocol(spec)
    if manifest['status'] != 'completed' or spec['smoke'] and not allow_smoke:
        raise ValueError('只接受完成的正式run，预检须显式允许')
    for rel, digest in manifest['frozen_files'].items():
        assert sha256(run/rel) == digest, rel
    actual_files = {p.relative_to(run).as_posix() for p in run.rglob('*')
                    if p.is_file() and p.name != 'manifest.json'}
    assert actual_files == set(manifest['frozen_files'])
    for name, digest in manifest['source_hashes'].items():
        assert sha256(run/'source'/name) == digest
    assert sha256(run/'source/任务1至7+直接残差扩展与最终对照方案_2026-11-11.md') == manifest['plan_sha256']
    assert manifest['selection_frozen_before_outer']
    assert sha256(run/'selection.json') == manifest['selection_sha256']
    raw = ROOT/'Teacher Resources & Requirements/01047000.csv'
    assert sha256(raw) == manifest['historical_data']['raw_sha256']
    assert sha256(run/'historical.npz') == manifest['historical_data']['dataset_sha256']
    historical = load_data(run/'historical.npz')
    dataset = run/('evaluation.npz' if spec['mode'] == 'final' else 'historical.npz')
    data = load_data(dataset)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'historical.npz'
        build_data(raw, path)
        rebuilt = load_data(path)
        assert set(rebuilt) == set(historical)
        for key in rebuilt:
            np.testing.assert_equal(rebuilt[key], historical[key])
        if spec['mode'] == 'final':
            assert sha256(dataset) == manifest['evaluation_data']['dataset_sha256']
            evaluation = Path(folder)/'evaluation.npz'
            build_evaluation_data(raw, path, evaluation)
            rebuilt = load_data(evaluation)
            for key in rebuilt:
                np.testing.assert_equal(rebuilt[key], data[key])
    selection = json.loads((run/'selection.json').read_text(encoding='utf-8'))
    records = selection['records']
    expected = set()
    for year in spec['years']:
        for h in spec['horizons']:
            for variant in spec['variants']:
                for seed in spec['seeds']:
                    expected.add(tag_for(year, h, variant, seed))
        for epoch in spec['ablation_epochs']:
            for variant in spec['variants']:
                for seed in spec['seeds']:
                    expected.add(tag_for(year, 1, variant, seed, 'fixed', epoch))
    assert len(records) == len(expected) == spec['outer_models'] == manifest['outer_models']
    assert {r['tag'] for r in records} == expected
    assert {p.name for p in (run/'models').iterdir()} == expected
    histories = {p.name:json.loads(p.read_text(encoding='utf-8')) for p in (run/'history').glob('*.json')}
    assert len(histories) == spec['fits'] == manifest['fits']
    used, predictions = set(), []
    frame = pd.read_csv(run/'metrics.csv')
    raw_frame = pd.read_csv(raw).set_index('Date')
    for record in records:
        year, h, seed, tag = [record[k] for k in ['year', 'task_h', 'seed', 'tag']]
        assert year in spec['years'] and h in spec['horizons'] and seed in spec['seeds']
        variant = next(v for v in spec['variants'] if v == record['variant'])
        train, valid = split_for(data, spec['mode'], year)
        if record['kind'] == 'searched':
            assert tag == tag_for(year, h, variant, seed)
            cv = pd.read_csv(run/'cv'/f'{tag}.csv')
            assert [json.loads(v)['fixed_epochs'] for v in cv.params] == spec['epochs']
            scores = np.column_stack([-cv[f'split{k}_test_score'].to_numpy() for k in range(3)])
            assert np.isfinite(scores).all()
            np.testing.assert_allclose(scores.mean(axis=1), -cv.mean_test_score.to_numpy(), rtol=1e-12)
            eligible = [i for i, v in enumerate(scores.mean(axis=1)) if v <= scores.mean(axis=1).min()*1.01]
            index = min(eligible, key=lambda i:spec['epochs'][i])
            assert record['epochs'] == spec['epochs'][index]
            np.testing.assert_allclose(record['inner_RMSE'], scores[index].mean(), rtol=1e-12)
            np.testing.assert_allclose(record['inner_fold_RMSE'], scores[index], rtol=1e-12)
            folds = inner_splits(data, train)
            own = {name:a for name, a in histories.items() if a['phase'] == tag+'_inner'}
            assert len(own) == 3*len(spec['epochs'])
            expected_inner = []
            for epoch in spec['epochs']:
                for ti, vi in folds:
                    ids = train[ti]
                    assert data['origin_days'][ids[-1]]+7 < data['origin_days'][train[vi[0]]]
                    expected_inner.append((epoch, int(ids[-1])))
                    found = [(n,a) for n,a in own.items() if a['params']['fixed_epochs'] == epoch
                             and a['last_id'] == int(ids[-1])]
                    assert len(found) == 1
                    name, audit = found[0]
                    check_audit(audit, ids, data, spec, variant, seed, epoch, tag+'_inner')
                    used.add(name)
            assert sorted(expected_inner) == sorted((a['params']['fixed_epochs'], a['last_id']) for a in own.values())
        else:
            assert record['kind'] == 'fixed' and h == 1 and record['epochs'] in spec['ablation_epochs']
            assert tag == tag_for(year, h, variant, seed, 'fixed', record['epochs'])
            assert record['inner_RMSE'] is None and record['inner_fold_RMSE'] is None
        assert data['origin_days'][train[-1]]+7 < data['origin_days'][valid[0]]
        model = restore_strategy(run/'models'/tag, dataset, spec['device'])
        check_audit(model.audit_, train, data, spec, variant, seed, record['epochs'], tag+'_outer')
        check_model(model, data, train, h)
        own = [(n,a) for n,a in histories.items() if a['phase'] == tag+'_outer']
        assert len(own) == 1 and own[0][1] == model.audit_
        used.add(own[0][0])
        pf = pd.read_csv(run/'predictions'/f'{tag}.csv').sort_values('sample_id')
        assert not pf.duplicated(['sample_id', 'h']).any()
        for key, value in dict(year=year, strategy='direct', task_h=h, h=h, seed=seed,
                               variant=variant['name'], kind=record['kind'], epochs=record['epochs']).items():
            assert (pf[key] == value).all()
        np.testing.assert_equal(pf.sample_id.to_numpy(), valid)
        np.testing.assert_equal(pf.truth.to_numpy(), data['y'][valid, h-1])
        np.testing.assert_allclose(pf.pred.to_numpy(), model.predict(valid[:, None])[:, 0], rtol=1e-7, atol=1e-5)
        for row in pf.itertuples():
            day = np.datetime64(int(data['origin_days'][row.sample_id]), 'D')
            assert row.origin_date == str(day) and row.target_date == str(day+np.timedelta64(h, 'D'))
            assert row.truth == raw_frame.loc[row.target_date, 'Discharge']
        found = frame[(frame.year == year)&(frame.h == h)&(frame.seed == seed)&
                      (frame.variant == variant['name'])&(frame.kind == record['kind'])&
                      (frame.epochs == record['epochs'])]
        assert len(found) == 1
        for key, value in recompute(pf.truth.to_numpy(), pf.pred.to_numpy()).items():
            np.testing.assert_allclose(found.iloc[0][key], value, rtol=1e-10, atol=1e-7)
        predictions.append(pf)
    assert len(frame) == len(records) and used == set(histories)
    assert len(selection['selections']) == len(spec['years'])*7
    for year in spec['years']:
        for h in spec['horizons']:
            own = [r for r in records if r['year'] == year and r['task_h'] == h and r['kind'] == 'searched']
            scores = {v['name']:float(np.mean([r['inner_RMSE'] for r in own if r['variant'] == v]))
                      for v in spec['variants']}
            found = [s for s in selection['selections'] if s['year'] == year and s['task_h'] == h]
            assert len(found) == 1
            for key, value in scores.items():
                np.testing.assert_allclose(found[0]['inner_scores'][key], value, rtol=1e-12)
            chosen = next(k for k, v in scores.items() if v <= min(scores.values())*1.01)
            assert found[0]['selected_variant'] == chosen
    result = dict(status='passed', mode=spec['mode'], smoke=spec['smoke'], fits=len(histories),
        models=len(records), metrics=len(frame), prediction_rows=sum(len(p) for p in predictions),
        frozen_files=len(manifest['frozen_files']), selection_frozen_before_outer=True,
        max_target='2004-12-31' if spec['mode'] == 'final' else '2003-12-31',
        checks=['固定正式协议', '源码/工件hash', '因果数据重建', '逐fit配置/seed声明',
                '时间净化', '实际外层scaler重算', '网络实际容量与buffer', '全部模型恢复', '指标和选择重算'],
        limitations=['不从审计声明独立证明实际随机过程；依据已冻结训练源码与重复性测试'])
    return result, pd.concat(predictions, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with threadpool_limits(limits=4):
        result, _ = verify(ROOT/'output/acceptance3_runs'/args.run_id, args.allow_smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
