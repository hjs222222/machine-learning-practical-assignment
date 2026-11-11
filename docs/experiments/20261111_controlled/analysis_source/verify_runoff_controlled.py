"""只读核验受控实验：hash、全部真值/预测、时间边界、选择规则与冻结模型恢复。"""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from runoff_training.controlled_data import (PROFILES, build_data, load_data, sha256,
                                             outer_split, inner_splits)
from runoff_training.controlled_model import restore

ROOT = Path(__file__).resolve().parents[1]


def metric_rows(truth, pred):
    error = pred - truth
    return dict(RMSE=float(np.sqrt(np.square(error).mean())), MAE=float(np.abs(error).mean()),
                R2=float(1 - np.square(error).sum() / np.square(truth - truth.mean()).sum()),
                bias=float(error.mean()), negative_predictions=int((pred < 0).sum()), n=len(truth))


def verify(run, allow_smoke=False):
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed' or manifest['protocol']['smoke'] and not allow_smoke:
        raise ValueError('只核验已完成正式run；预检须显式--allow-smoke')
    for rel, expected in manifest['frozen_files'].items():
        if sha256(run / rel) != expected:
            raise ValueError(f'冻结文件已改变：{rel}')
    for rel, expected in manifest['source_hashes'].items():
        name = rel.replace('\\', '/').split('/')[-1]
        if sha256(run / 'source' / name) != expected:
            raise ValueError(f'运行快照与源码清单不符：{rel}')
    raw_path = ROOT / 'Teacher Resources & Requirements/01047000.csv'
    if sha256(raw_path) != manifest['data']['raw_sha256']:
        raise ValueError('原CSV已改变')
    data = load_data(run / 'dataset.npz')
    raw = pd.read_csv(raw_path)
    raw.index = pd.to_datetime(raw.Date)
    # 用日期查值而非样本行号，独立核验全部真值。
    origins = data['origin_days'].astype('datetime64[D]')
    for h in range(1, 8):
        dates = origins + np.timedelta64(h, 'D')
        np.testing.assert_equal(raw.loc[pd.to_datetime(dates), 'Discharge'].to_numpy(), data['y'][:, h - 1])
    assert np.max(data['days']) == np.datetime64('2003-12-31').astype('int64')
    with tempfile.TemporaryDirectory() as folder:
        rebuilt_path = Path(folder) / 'dataset.npz'
        build_data(raw_path, rebuilt_path)
        rebuilt = load_data(rebuilt_path)
        for key in data:
            np.testing.assert_equal(data[key], rebuilt[key])
    spec = manifest['protocol']
    frame = pd.read_csv(run / 'metrics.csv')
    rows, selections = [], json.loads((run / 'input_selection.json').read_text(encoding='utf-8'))
    assert len(selections) == len(spec['years']) * len(spec['tasks'])
    all_predictions = []
    history_files = list((run / 'history').glob('*.json'))
    expected_sets = []
    for selection in selections:
        year, strategy, h = selection['year'], selection['strategy'], selection['task_h']
        scores = selection['inner_scores']
        expected = next(p for p in PROFILES if scores[p] <= min(scores.values()) * 1.01)
        assert selection['selected_profile'] == expected and selection['selection_uses_outer'] is False
        train, valid = outer_split(data, year)
        for ti, vi in inner_splits(data, train):
            assert data['origin_days'][train[ti[-1]]] + 7 < data['origin_days'][train[vi[0]]]
            assert data['origin_days'][train[vi[-1]]] + 7 < data['origin_days'][valid[0]]
        for transform, profiles in [('log', list(PROFILES)), ('sqrt', [expected]), ('raw', [expected])]:
            for profile in profiles:
                for seed in spec['seeds']:
                    expected_sets.append((year, strategy, h, profile, transform, seed))
    assert len(expected_sets) == manifest['outer_models']
    for year, strategy, h, profile, transform, seed in expected_sets:
        tag = f'y{year}_{strategy}_h{h}_{profile}_{transform}_s{seed}'
        record = json.loads((run / 'checkpoints' / f'{tag}.json').read_text(encoding='utf-8'))
        cv = pd.read_csv(run / 'cv' / f'{tag}.csv')
        candidates = [json.loads(p)['fixed_epochs'] for p in cv.params]
        scores = -cv.mean_test_score.to_numpy()
        eligible = [i for i, v in enumerate(scores) if v <= scores.min() * 1.01]
        index = min(eligible, key=lambda i: candidates[i])
        assert record['epochs'] == candidates[index]
        fold_scores = np.column_stack([-cv[f'split{k}_test_score'].to_numpy() for k in range(spec['inner_folds'])])
        np.testing.assert_allclose(fold_scores.mean(axis=1), scores, rtol=1e-12)
        np.testing.assert_allclose(record['inner_fold_RMSE'], fold_scores[index], rtol=1e-12)
        np.testing.assert_allclose(record['inner_RMSE'], scores[index], rtol=1e-12)
        selection = next(s for s in selections if (s['year'], s['strategy'], s['task_h']) == (year, strategy, h))
        if transform == 'log':
            own = []
            for own_seed in spec['seeds']:
                own_tag = f'y{year}_{strategy}_h{h}_{profile}_log_s{own_seed}'
                own.append(json.loads((run / 'checkpoints' / f'{own_tag}.json').read_text(encoding='utf-8'))['inner_RMSE'])
            np.testing.assert_allclose(np.mean(own), selection['inner_scores'][profile], rtol=1e-12)
        train, valid = outer_split(data, year)
        prediction_frame = pd.read_csv(run / 'predictions' / f'{tag}.csv')
        assert not prediction_frame.duplicated(['sample_id', 'h']).any()
        own = prediction_frame.pivot(index='sample_id', columns='h', values='pred').sort_index()
        np.testing.assert_equal(own.index.to_numpy(), valid)
        assert own.columns.tolist() == ([h] if strategy == 'direct' else list(range(1, 8)))
        model = restore(run / 'models' / tag, run / 'dataset.npz', spec['device'])
        rebuilt_pred = model.predict(valid[:, None])
        np.testing.assert_allclose(own.to_numpy(), rebuilt_pred, rtol=1e-7, atol=1e-5)
        assert record['audit']['n_train'] == len(train)
        assert record['audit']['last_id'] == int(train[-1])
        assert record['audit']['history'][-1]['epoch'] == record['epochs']
        assert len(record['audit']['history']) == record['epochs']
        n = record['epochs'] * int(np.ceil(len(train) / record['audit']['params']['batch_size']))
        assert record['audit']['history'][-1]['gradient_steps'] == n
        assert record['audit']['train_last_h7_target'] < str(origins[valid[0]])
        for row in prediction_frame.itertuples():
            assert row.origin_date == str(origins[row.sample_id])
            assert row.target_date == str(origins[row.sample_id] + np.timedelta64(row.h, 'D'))
            np.testing.assert_allclose(row.truth, data['y'][row.sample_id, row.h - 1], rtol=0, atol=1e-12)
        for j, horizon in enumerate(own.columns):
            values = metric_rows(data['y'][valid, horizon - 1], own.to_numpy()[:, j])
            identity = dict(year=year, strategy=strategy, task_h=h, profile=profile, transform=transform, seed=seed, h=int(horizon))
            selected = frame
            for k, value in identity.items():
                selected = selected[selected[k] == value]
            assert len(selected) == 1
            for key, value in values.items():
                np.testing.assert_allclose(selected.iloc[0][key], value, rtol=1e-10, atol=1e-7)
            rows.append(dict(**identity, **values))
        all_predictions.append(prediction_frame)
    assert len(rows) == len(frame)
    assert len(history_files) == spec['fit_budget'] == manifest['fits']
    # 每条任务独立查实际内层训练id，防止只核验汇总行数。
    for file in history_files:
        audit = json.loads(file.read_text(encoding='utf-8'))
        phase = audit['phase']
        year = int(phase[1:5])
        train, _ = outer_split(data, year)
        possible = [train] if phase.endswith('_outer') else [train[ti] for ti, _ in inner_splits(data, train)]
        assert any((len(ids), int(ids[0]), int(ids[-1])) ==
                   (audit['n_train'], audit['first_id'], audit['last_id']) for ids in possible)
    result = dict(status='passed', smoke=spec['smoke'], models=len(expected_sets), metrics=len(rows),
        prediction_rows=sum(len(p) for p in all_predictions), fits=len(history_files),
        frozen_files=len(manifest['frozen_files']), max_target='2003-12-31',
        checks=['原CSV/hash', '全部h日期/真值', '特征重建', '时间净化', '内层epoch和输入选择',
                '全部最终模型恢复', '逐h指标重建', '实际fit证据'])
    return result, pd.concat(all_predictions, ignore_index=True)


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
        result, _ = verify(ROOT / 'output/acceptance3_runs' / args.run_id, args.allow_smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
