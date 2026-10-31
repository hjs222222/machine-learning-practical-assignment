# -*- coding: utf-8 -*-
"""第三轮阶段A：全年开发窗口、固定epoch、四方案受控消融。"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset, physical_scorer, restore_model
from acceptance3_train import ROOT, Tee
from acceptance3_v2_train import atomic_json, prepare_v2, split_record
from acceptance3_v3_model import V3Regressor

SOURCES = ['acceptance3_model.py', 'acceptance3_train.py', 'acceptance3_v2_model.py',
           'acceptance3_v2_train.py', 'acceptance3_v3_model.py', 'acceptance3_v3_train.py',
           'feature_engineering.py', 'config.py', 'utils_plot.py']
PROFILES = {
    'A0': dict(feature_mode='plus_Q', target_transform='log', mixed_weight=0.),
    'A1': dict(feature_mode='compact', target_transform='log', mixed_weight=0.),
    'A2': dict(feature_mode='plus_Q', target_transform='raw', mixed_weight=0.),
    'A3': dict(feature_mode='plus_Q', target_transform='log', mixed_weight=.1)}
TASKS = [('direct', 1), ('direct', 3), ('direct', 7), ('multi', 7)]


def development_splits(data):
    days = data['origin_days']
    folds = []
    for year in [2002, 2003]:
        start = np.datetime64(f'{year}-01-01').astype('int64')
        end = np.datetime64(f'{year}-12-31').astype('int64')
        ti = np.flatnonzero(days + 7 < start)
        vi = np.flatnonzero((days >= start) & (days + 7 <= end))
        assert len(ti) >= 665 and len(vi) >= 358
        assert days[ti[-1]] + 7 < days[vi[0]]
        assert (days[vi] + 7 <= np.datetime64('2003-12-31').astype('int64')).all()
        folds.append((year, ti, vi))
    return folds


def fixed_grid(profile, smoke=False):
    return {**{k: [v] for k, v in PROFILES[profile].items()},
            'units': [(8,) if smoke else (64,)], 'dropout': [.2], 'lr': [.001],
            'fixed_epochs': [1] if smoke else [10, 20, 40]}


def frame_metrics(data, train_ids, val_ids, pred, y, strategy, h, profile, epoch, seed, year):
    pred = np.asarray(pred).reshape(len(val_ids), -1)
    truth = np.asarray(y[val_ids]).reshape(len(val_ids), -1)
    rows, values = [], []
    horizons = [h] if strategy == 'direct' else list(range(1, 8))
    quarters = pd.to_datetime(data['origin_dates'][val_ids]).quarter.to_numpy()
    training_y = np.asarray(y[train_ids]).reshape(len(train_ids), -1)
    for col, horizon in enumerate(horizons):
        cuts = np.quantile(training_y[:, col], [.25, .9])
        groups = [('year', np.ones(len(val_ids), dtype=bool))]
        groups += [(f'Q{q}', quarters == q) for q in range(1, 5)]
        groups += [('low_train_q25', truth[:, col] <= cuts[0]), ('high_train_q90', truth[:, col] >= cuts[1])]
        for label, mask in groups:
            if not mask.any():
                continue
            yt, yp = truth[mask, col], pred[mask, col]
            rows.append(dict(strategy=strategy, task_h=h, profile=profile, epoch=epoch, seed=seed,
                year=year, h=horizon, segment=label, n=int(mask.sum()),
                R2=float(r2_score(yt, yp)) if len(yt) > 1 else None,
                RMSE=float(np.sqrt(np.mean((yp - yt) ** 2))), MAE=float(np.mean(np.abs(yp - yt))),
                bias=float(np.mean(yp - yt)), negative_predictions=int((yp < 0).sum())))
        for i, sample_id in enumerate(val_ids):
            values.append(dict(strategy=strategy, task_h=h, profile=profile, epoch=epoch, seed=seed,
                year=year, h=horizon, sample_id=int(sample_id),
                origin_date=str(np.datetime64(int(data['origin_days'][sample_id]), 'D')),
                target_date=str(np.datetime64(int(data['origin_days'][sample_id] + horizon), 'D')),
                truth=float(truth[i, col]), pred=float(pred[i, col])))
    return rows, values


def fit_extra(run, path, data, folds, strategy, h, profile, epoch, seed, smoke):
    rows, predictions = [], []
    y = data['y'][:, h - 1] if strategy == 'direct' else data['y']
    for year, ti, vi in folds:
        label = f'{strategy}_h{h}_{profile}_ep{epoch}_seed{seed}_year{year}'
        checkpoint = run / 'checkpoints' / (label + '.json')
        if checkpoint.exists():
            result = json.loads(checkpoint.read_text(encoding='utf-8'))
        else:
            model = V3Regressor(str(path), family='LSTM', horizon=h,
                units=(8,) if smoke else (64,), fixed_epochs=epoch, seed=seed,
                history_dir=str(run / 'training_history'), phase=label, **PROFILES[profile]).fit(ti[:, None], y[ti])
            model.save(run / 'models' / label)
            pred = model.predict(vi[:, None])
            restored = restore_model(run / 'models' / label, device='cpu').predict(vi[:, None])
            np.testing.assert_allclose(pred, restored, rtol=1e-6, atol=1e-4)
            r, p = frame_metrics(data, ti, vi, pred, y, strategy, h, profile, epoch, seed, year)
            result = dict(metrics=r, predictions=p)
            atomic_json(checkpoint, result)
            print('[三seed复核完成]', label, flush=True)
        rows.extend(result['metrics'])
        predictions.extend(result['predictions'])
    return rows, predictions


def ablation(run, smoke):
    path = run / 'dataset.npz'
    if not path.exists():
        prepare_v2(run)
    data = load_dataset(str(path))
    folds = development_splits(data)
    pd.DataFrame([split_record(str(year), 'annual_development', 0, ti, vi, data['origin_days'])
                  for year, ti, vi in folds]).to_csv(run / 'annual_splits.csv', index=False, encoding='utf-8-sig')
    # GridSearchCV训练部分为截至2003年的索引；测试起报点从未传入fit、scorer或predict。
    pool = np.flatnonzero(data['origin_days'] + 7 <= np.datetime64('2003-12-31').astype('int64'))
    assert np.array_equal(pool, np.arange(len(pool)))
    cv = [(ti, vi) for _, ti, vi in folds]
    selections, metric_rows, prediction_rows = [], [], []
    for strategy, h in TASKS:
        y = data['y'][:, h - 1] if strategy == 'direct' else data['y']
        for profile in PROFILES:
            label = f'{strategy}_h{h}_{profile}'
            cache = run / 'checkpoints' / f'grid_{label}.json'
            if cache.exists():
                chosen = json.loads(cache.read_text(encoding='utf-8'))
            else:
                model = V3Regressor(str(path), family='LSTM', horizon=h,
                    history_dir=str(run / 'training_history'), phase=label, seed=42)
                print('[固定轮数GridSearchCV开始]', label, flush=True)
                search = GridSearchCV(model, fixed_grid(profile, smoke), cv=cv, scoring=physical_scorer,
                    n_jobs=1, refit=False, error_score='raise', verbose=1)
                search.fit(pool[:, None], y[pool])
                table = pd.DataFrame(search.cv_results_)
                table['params'] = table.params.map(lambda x: json.dumps(x))
                table.to_csv(run / 'cv_results' / f'{label}.csv', index=False, encoding='utf-8-sig')
                chosen = dict(strategy=strategy, h=h, profile=profile,
                    epoch=int(search.best_params_['fixed_epochs']), best_score=float(search.best_score_))
                atomic_json(cache, chosen)
            selections.append(chosen)
            # 以所选epoch重新训练并保存，用于独立恢复及三seed同配置比较。
            for seed in ([42] if smoke else [42, 43, 44]):
                r, p = fit_extra(run, path, data, folds, strategy, h, profile, chosen['epoch'], seed, smoke)
                metric_rows.extend(r)
                prediction_rows.extend(p)
    frame = pd.DataFrame(metric_rows)
    frame.to_csv(run / 'development_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(prediction_rows).to_csv(run / 'development_predictions.csv', index=False, encoding='utf-8-sig')
    atomic_json(run / 'ablation_selections.json', selections)
    annual = frame[frame.segment == 'year'].groupby(['strategy', 'task_h', 'profile', 'seed', 'year'])[['RMSE', 'MAE']].mean().reset_index()
    reference = annual[annual.profile == 'A0'].drop(columns='profile')
    paired = annual.merge(reference, on=['strategy', 'task_h', 'seed', 'year'], suffixes=('', '_A0'))
    paired['RMSE_ratio'] = paired.RMSE / paired.RMSE_A0
    paired['MAE_ratio'] = paired.MAE / paired.MAE_A0
    paired.to_csv(run / 'paired_ablation.csv', index=False, encoding='utf-8-sig')
    gates = []
    for (strategy, h, profile), f in paired.groupby(['strategy', 'task_h', 'profile']):
        mean_rmse, mean_mae = float(f.RMSE_ratio.mean()), float(f.MAE_ratio.mean())
        worst_year = float(f.groupby('year').RMSE_ratio.mean().max())
        passed = profile != 'A0' and mean_rmse <= .98 and mean_mae <= 1.02 and worst_year <= 1.02
        gates.append(dict(strategy=strategy, h=int(h), profile=profile, mean_RMSE_ratio=mean_rmse,
            mean_MAE_ratio=mean_mae, worst_year_mean_RMSE_ratio=worst_year, screening_passed=bool(passed)))
    atomic_json(run / 'stage_a_decision.json', dict(stage='A_complete', gates=gates,
        any_screening_passed=any(g['screening_passed'] for g in gates),
        next_step='仅通过筛选的方案进入全部7步及嵌套开发验证；未通过不自动扩大网格。',
        limitation='这些开发年用于选epoch及筛选，不是无偏外层泛化成绩；2004未预测。'))
    # 独立重算落盘预测；检查每条开发标签均不晚于2003年末。
    pred = pd.DataFrame(prediction_rows)
    assert pd.to_datetime(pred.target_date).max() <= pd.Timestamp('2003-12-31')
    assert not pred.duplicated(['strategy', 'task_h', 'profile', 'seed', 'year', 'h', 'sample_id']).any()
    for key, f in pred.groupby(['strategy', 'task_h', 'profile', 'seed', 'year', 'h']):
        strategy, task_h, profile, seed, year, h = key
        r = frame[(frame.strategy == strategy) & (frame.task_h == task_h) & (frame.profile == profile) &
                  (frame.seed == seed) & (frame.year == year) & (frame.h == h) & (frame.segment == 'year')].iloc[0]
        np.testing.assert_allclose([r.RMSE, r.MAE], [np.sqrt(np.mean((f.pred-f.truth)**2)), np.abs(f.pred-f.truth).mean()], rtol=1e-10)
    fits = len(list((run / 'training_history').glob('*.json')))
    expected = 64 if smoke else 192
    assert fits == expected, (fits, expected)
    atomic_json(run / 'verification.json', dict(status='passed', stage='A', fit_records=fits,
        expected_fit_records=expected, annual_development_years=[2002, 2003],
        prediction_rows=len(pred), all_selected_model_predictions_restored=True,
        test_2004_predicted=False, full_training_completed=False))
    print('[阶段A完成，2004未预测]\n', pd.DataFrame(gates).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ['.', '..']:
        parser.error('run-id必须是单个目录名')
    run = ROOT / 'output/acceptance3_runs' / args.run_id
    digest = hashlib.sha256((ROOT / 'Teacher Resources & Requirements/01047000.csv').read_bytes()).hexdigest()
    previous = {}
    if args.resume:
        previous = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
        assert previous['status'] == 'failed' and previous['smoke'] == args.smoke and previous['data_sha256'] == digest
        for name in SOURCES:
            assert (ROOT / 'code' / name).read_bytes() == (run / 'source' / name).read_bytes(), name
    else:
        run.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'logs', 'cv_results', 'models', 'training_history', 'checkpoints']:
        (run / name).mkdir(exist_ok=args.resume)
    for name in SOURCES:
        shutil.copy2(ROOT / 'code' / name, run / 'source' / name)
    manifest = dict(run_id=args.run_id, version=3, phase='ablation', status='running', smoke=args.smoke,
        data_sha256=digest, started=previous.get('started', time.strftime('%Y-%m-%dT%H:%M:%S%z')),
        device='cpu', torch_build=torch.__version__,
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'torch']},
        full_training_completed=False, test_2004_predicted=False)
    atomic_json(run / 'manifest.json', manifest)
    atomic_json(run / 'config.json', dict(profiles=PROFILES, tasks=TASKS, epochs=[1] if args.smoke else [10, 20, 40],
        seeds=[42] if args.smoke else [42,43,44], lookback=30, gap=7, grid_search_seed=42,
        planned_fit_count=64 if args.smoke else 192,
        gate=dict(RMSE_ratio_max=.98, MAE_ratio_max=1.02, worst_year_RMSE_ratio_max=1.02),
        no_2004_selection=True, limitation='2002/2003用于开发筛选，不宣称无偏外层结果'))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    with (run / 'logs/training.log').open('a' if args.resume else 'w', encoding='utf-8', buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
                with threadpool_limits(limits=4):
                    ablation(run, args.smoke)
                manifest['status'] = 'stage_a_completed'
            except Exception:
                import traceback
                traceback.print_exc()
                manifest['status'] = 'failed'
                raise
            finally:
                manifest['seconds'] = previous.get('seconds', 0) + time.perf_counter() - start
                manifest['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
                atomic_json(run / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
