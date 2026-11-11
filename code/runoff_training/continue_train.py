"""第二轮 GPU 训练：随机批次 AdamW / 对数正态条件均值，历史选参与三种子复核。"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV, ParameterGrid
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset, physical_scorer, restore_model
from acceptance3_train import ROOT, Tee, metrics_table, prediction_rows
from acceptance3_v2_train import atomic_json, prepare_v2, split_record
from acceptance3_v3_model import V3Regressor
from acceptance3_v3_stage_b import historical_splits
from runoff_training.distribution import DistributionLSTMRegressor
from runoff_training.train import TASKS, baseline_params, choose_meta, get_target, score, year_folds


def grids(smoke=False):
    return dict(objective=['log_mse', 'lognormal'], units=[(8,)] if smoke else [(32,), (64,)],
        fixed_epochs=[1] if smoke else [15, 40], residual=[False, True],
        dropout=[.1], lr=[.001], batch_size=[64], weight_decay=[.001])


def model_for(run, path, h, params, seed, device, phase):
    return DistributionLSTMRegressor(str(path), horizon=h, seed=seed, device=device,
        history_dir=str(run/'training_history'), phase=phase, **params)


def select(run, path, data, train, strategy, h, args):
    name = f'{strategy}_h{h}'
    file = run/'checkpoints'/f'search_{name}.json'
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    days = data['origin_days']
    ids = train[days[train] + 7 <= np.datetime64('2002-12-31').astype('int64')]
    cv = historical_splits(ids, days)
    if args.smoke:
        cv = [(ti, vi[:12]) for ti, vi in cv[-1:]]
    pd.DataFrame([split_record(name, 'distribution_search', k, ids[ti], ids[vi], days)
        for k, (ti, vi) in enumerate(cv)]).to_csv(run/'splits'/f'search_{name}.csv', index=False)
    estimator = model_for(run, path, h, {}, 42, args.device, f'search_{name}')
    print('[历史网格]', name, '候选', len(list(ParameterGrid(grids(args.smoke)))), '折', len(cv), flush=True)
    search = GridSearchCV(estimator, grids(args.smoke), cv=cv, scoring=physical_scorer,
                          n_jobs=1, refit=False, error_score='raise')
    search.fit(ids[:, None], get_target(data, strategy, h)[ids])
    frame = pd.DataFrame(search.cv_results_)
    frame['params'] = frame.params.map(json.dumps)
    frame.to_csv(run/'cv_results'/f'{name}.csv', index=False, encoding='utf-8-sig')
    result = dict(strategy=strategy, h=h, params=search.best_params_,
                  cv_RMSE=-float(search.best_score_), max_search_target='2002-12-31')
    atomic_json(file, result)
    print('[历史胜出]', result, flush=True)
    return result


def fitted(run, path, data, ti, vi, strategy, h, params, seed, args, label, original=False):
    name = f'{label}_{strategy}_h{h}_seed{seed}' + ('_original' if original else '')
    checkpoint = run/'checkpoints'/f'{name}.json'
    if checkpoint.exists():
        return np.asarray(json.loads(checkpoint.read_text(encoding='utf-8'))['predictions'])
    target = get_target(data, strategy, h)
    if original:
        own = baseline_params(strategy, h, args.smoke)
        own.pop('input_transform', None)
        own.pop('residual', None)
        model = V3Regressor(str(path), family='LSTM', horizon=h, seed=seed, device=args.device,
            history_dir=str(run/'training_history'), phase=name, **own)
    else:
        model = model_for(run, path, h, params, seed, args.device, name)
    model.fit(ti[:, None], target[ti])
    folder = run/'models'/name
    model.save(folder)
    check = vi if len(vi) else ti[-7:]
    pred = np.asarray(model.predict(check[:, None])).reshape(len(check), -1)
    restored = restore_model(folder, device=args.device)
    np.testing.assert_allclose(restored.predict(check[:, None]).reshape(pred.shape), pred,
                               rtol=2e-5, atol=2e-3)
    value = pred if len(vi) else np.empty((0, 1 if strategy == 'direct' else 7))
    atomic_json(checkpoint, dict(predictions=value.tolist(), restored=True))
    print('[拟合完成]', name, flush=True)
    return value


def reused_oof(previous, vi, strategy, h, family, seed):
    part = previous[(previous.strategy == strategy) & (previous.task_h == h) &
                    (previous.model == family) & (previous.seed == seed) & previous.sample_id.isin(vi)]
    frame = part.pivot(index='sample_id', columns='h', values='pred').reindex(vi)
    assert len(frame) == len(vi) and not frame.isna().any().any()
    assert frame.columns.tolist() == ([h] if strategy == 'direct' else list(range(1, 8)))
    return frame.to_numpy()


def gate_table(rows):
    frame = pd.DataFrame(rows)
    records = []
    for (strategy, h), part in frame.groupby(['strategy', 'task_h']):
        for reference in ['original', 'previous']:
            per_seed = []
            for seed, own in part.groupby('seed'):
                ratios = own.candidate_RMSE / own[f'{reference}_RMSE']
                ratio = own.candidate_RMSE.mean() / own[f'{reference}_RMSE'].mean()
                mae_ratio = own.candidate_MAE.mean() / own[f'{reference}_MAE'].mean()
                passed = ratio <= .995 and mae_ratio <= 1.02 and ratios.max() <= 1.15
                per_seed.append(dict(seed=int(seed), RMSE_ratio=float(ratio),
                    MAE_ratio=float(mae_ratio), worst_quarter_ratio=float(ratios.max()), passed=bool(passed)))
            passed = all(value['passed'] for value in per_seed)
            records.append(dict(strategy=strategy, h=int(h), reference=reference,
                passed_all_seeds=passed, seed_results=per_seed,
                policy='参数只用2002选择；2003三种子门槛只决定是否推荐，不反选OOF参数'))
    return records


def run_training(run, args, previous_run):
    path, _, train, test, _ = prepare_v2(run)
    data = load_dataset(str(path))
    previous_data = load_dataset(str(previous_run/'dataset.npz'))
    for key in ['features', 'positions', 'origin_days', 'y']:
        np.testing.assert_array_equal(data[key], previous_data[key])
    previous_manifest = json.loads((previous_run/'manifest.json').read_text(encoding='utf-8'))
    assert previous_manifest['status'] == 'completed' and not previous_manifest['smoke']
    assert previous_manifest['data_sha256'] == hashlib.sha256((ROOT/'Teacher Resources & Requirements/01047000.csv').read_bytes()).hexdigest()
    previous = pd.read_csv(previous_run/'oof_predictions.csv')
    source_hashes = json.loads((previous_run/'model_hashes_before_test.json').read_text(encoding='utf-8'))
    for name, digest in source_hashes.items():
        assert hashlib.sha256((previous_run/name).read_bytes()).hexdigest() == digest
    # Windows保存的相对路径可能含反斜杠，筛选前统一；不改历史run快照。
    source_files = {name for name in source_hashes if name.replace(chr(92), '/').startswith('models/final_') and
                    ('_ANN_' in name or '_RF_' in name)}
    for name in ['manifest.json', 'frozen_config.json', 'model_hashes_before_test.json', 'oof_predictions.csv', 'dataset.npz']:
        source_files.add(name)
    reuse_hashes = {str((previous_run/name).relative_to(ROOT)): hashlib.sha256((previous_run/name).read_bytes()).hexdigest()
                   for name in sorted(source_files)}
    atomic_json(run/'reused_source_hashes.json', reuse_hashes)
    folds = year_folds(data, train, args.smoke)
    pd.DataFrame([split_record(label, '2003_development', 0, ti, vi, data['origin_days'])
        for label, ti, vi in folds]).to_csv(run/'splits/outer.csv', index=False)
    tasks = TASKS if not args.smoke else [('direct', 1), ('multi', 7)]
    seeds = [42] if args.smoke else [42, 43, 44]
    selections = [select(run, path, data, train, strategy, h, args) for strategy, h in tasks]
    atomic_json(run/'lstm_grid_selection.json', selections)
    rows, gate_rows, meta = [], [], []
    for selection in selections:
        strategy, h, params = selection['strategy'], selection['h'], selection['params']
        for seed in seeds:
            ids, matrices = [], []
            for label, ti, vi in folds:
                pred = fitted(run, path, data, ti, vi, strategy, h, params, seed, args, label)
                original = fitted(run, path, data, ti, vi, strategy, h, params, seed, args, label, original=True)
                prior = reused_oof(previous, vi, strategy, h, 'LSTM', seed)
                truth = np.asarray(get_target(data, strategy, h)[vi]).reshape(len(vi), -1)
                gate_rows.append(dict(strategy=strategy, task_h=h, seed=seed, fold=label,
                    candidate_RMSE=score(truth, pred), original_RMSE=score(truth, original),
                    previous_RMSE=score(truth, prior), candidate_MAE=float(np.abs(truth - pred).mean()),
                    original_MAE=float(np.abs(truth - original).mean()), previous_MAE=float(np.abs(truth - prior).mean())))
                block = [reused_oof(previous, vi, strategy, h, family, seed) for family in ['ANN', 'RF']] + [pred]
                ids.append(vi)
                matrices.append(np.column_stack(block))
                for family, values in zip(['ANN', 'RF', 'LSTM'], block):
                    rows.extend(dict(fold=label, strategy=strategy, task_h=h, model=family,
                        seed=seed, sample_id=int(sample), origin_date=str(data['origin_dates'][sample]),
                        h=h if strategy == 'direct' else j + 1,
                        truth=float(data['y'][sample, h - 1 if strategy == 'direct' else j]), pred=float(values[i, j]))
                        for i, sample in enumerate(vi) for j in range(values.shape[1]))
            meta.append(choose_meta(run, data, np.concatenate(ids), np.vstack(matrices), strategy, h, seed, args.smoke))
            fitted(run, path, data, train, np.array([], dtype=int), strategy, h, params, seed, args, 'final')
    pd.DataFrame(rows).to_csv(run/'oof_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(gate_rows).to_csv(run/'development_quarter_metrics.csv', index=False, encoding='utf-8-sig')
    gates = gate_table(gate_rows)
    atomic_json(run/'development_gates.json', gates)
    atomic_json(run/'meta_selection.json', meta)
    frozen = dict(lstm=selections, meta=meta, development_gates=gates, seeds=seeds,
        primary_seed=42, previous_run=previous_run.name, max_search_target='2002-12-31',
        max_train_target='2003-12-31', test_2004_predicted=False,
        reused=['上轮ANN/RF的2003 OOF及最终模型；逐文件SHA256核对'],
        limitations=['2004已被查看，本轮仍为回溯评估', '随机批次仅在当前训练折内部',
            'lognormal是假设边际对数正态，不保证分布或区间已校准',
            'Stacking保留上轮2003 OOF协议，本轮不把重复训练称为新方法'])
    atomic_json(run/'frozen_config.json', frozen)
    hashes = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ['models', 'meta_models'] for p in sorted((run/folder).rglob('*')) if p.is_file()}
    atomic_json(run/'model_hashes_before_test.json', hashes)
    # 只有全部参数、权重、来源和开发成绩落盘后才预测2004。
    output = []
    for seed in seeds:
        collected = {strategy: {family: [] for family in ['ANN', 'RF', 'LSTM', 'Mean', 'Stacking']}
                     for strategy in sorted({strategy for strategy, _ in tasks})}
        for strategy, h in tasks:
            block = []
            for family in ['ANN', 'RF']:
                folder = previous_run/'models'/f'final_{strategy}_{family}_h{h}_seed{seed}'
                model = restore_model(folder, device='cpu')
                block.append(np.asarray(model.predict(test[:, None])).reshape(len(test), -1))
            model = restore_model(run/'models'/f'final_{strategy}_h{h}_seed{seed}', device=args.device)
            block.append(np.asarray(model.predict(test[:, None])).reshape(len(test), -1))
            for family, values in zip(['ANN', 'RF', 'LSTM'], block):
                collected[strategy][family].append(values)
            collected[strategy]['Mean'].append(np.stack(block).mean(axis=0))
            model = joblib.load(run/'meta_models'/f'{strategy}_h{h}_seed{seed}.joblib')
            collected[strategy]['Stacking'].append(np.asarray(model.predict(np.column_stack(block))).reshape(len(test), -1))
        for strategy, families in collected.items():
            for family, blocks in families.items():
                pred = np.column_stack(blocks)
                if args.smoke and strategy == 'direct':
                    own = [dict(run_id=run.name, strategy=strategy, model=family, sample_id=int(sample),
                        origin_date=str(data['origin_dates'][sample]), target_date=str(np.datetime64(int(data['origin_days'][sample]) + 1, 'D')),
                        h=1, truth=float(data['y'][sample, 0]), pred=float(pred[i, 0])) for i, sample in enumerate(test)]
                else:
                    own = prediction_rows(run.name, data, test, pred, strategy, family)
                for row in own:
                    row['seed'] = seed
                output.extend(own)
    frame = pd.DataFrame(output)
    assert np.isfinite(frame[['truth', 'pred']].to_numpy()).all()
    frame.to_csv(run/'test_predictions_all_seeds.csv', index=False, encoding='utf-8-sig')
    tables = []
    for seed, part in frame.groupby('seed'):
        own = metrics_table(part.to_dict('records'))
        own['seed'] = seed
        tables.append(own)
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(run/'metrics_all_seeds.csv', index=False, encoding='utf-8-sig')
    primary = metrics[metrics.seed == 42].drop(columns='seed')
    primary.to_csv(run/'metrics.csv', index=False, encoding='utf-8-sig')
    metrics.groupby(['strategy', 'model'])[['R2', 'RMSE', 'MAE']].mean().to_csv(run/'summary.csv', encoding='utf-8-sig')
    for name, digest in hashes.items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest() == digest
    for name, digest in reuse_hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    required = primary[primary.model.isin(['LSTM', 'Mean', 'Stacking'])]
    assert len(required) == (24 if args.smoke else 42)
    histories = list((run/'training_history').glob('*.json'))
    assert len(histories) == (26 if args.smoke else 728)
    for file in histories:
        record = json.loads(file.read_text(encoding='utf-8'))
        assert data['origin_days'][record['last_id']] + 7 <= np.datetime64('2003-12-31').astype('int64')
        assert len(record['history']) == record['params']['fixed_epochs']
        if 'cv_score' in record:
            assert data['origin_days'][record['last_id']] + 7 < data['origin_days'][record['validation_first_id']]
    atomic_json(run/'verification.json', dict(status='passed', smoke=args.smoke,
        required_metric_rows=len(required), fit_records=len(histories), frozen_files_verified=len(hashes),
        reused_files_verified=len(reuse_hashes), all_models_restored=True, test_type='retrospective'))
    print('[本轮完成]', metrics.groupby(['strategy', 'model'])[['R2', 'RMSE', 'MAE']].mean().to_string(), flush=True)
    print('[开发门槛]', json.dumps(gates, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--previous-run', default='formal_20261111_gpu_lag_residual')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    for value in [args.run_id, args.previous_run]:
        if Path(value).name != value or value in ['.', '..']:
            parser.error('run-id与previous-run必须为单目录名')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA不可用')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run = ROOT/'output/acceptance3_runs'/args.run_id
    previous_run = ROOT/'output/acceptance3_runs'/args.previous_run
    sources = sorted((ROOT/'code/runoff_training').glob('*.py'))
    sources += [ROOT/'code'/name for name in ['acceptance3_model.py', 'acceptance3_v2_model.py',
        'acceptance3_v3_model.py', 'acceptance3_train.py', 'acceptance3_v2_train.py',
        'acceptance3_v3_stage_b.py', 'feature_engineering.py', 'config.py', 'utils_plot.py']]
    data_hash = hashlib.sha256((ROOT/'Teacher Resources & Requirements/01047000.csv').read_bytes()).hexdigest()
    if args.resume:
        old = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        assert old['status'] in ['failed', 'interrupted', 'running']
        assert old['device'] == args.device and old['smoke'] == args.smoke and old['data_sha256'] == data_hash
        assert old['previous_run'] == args.previous_run
        for p in sources:
            assert (run/'source'/p.name).read_bytes() == p.read_bytes(), p
    else:
        run.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'models', 'meta_models', 'training_history', 'splits', 'cv_results', 'checkpoints', 'logs']:
        (run/name).mkdir(exist_ok=args.resume)
    for p in sources:
        shutil.copy2(p, run/'source'/p.name)
    manifest = dict(run_id=args.run_id, status='running', smoke=args.smoke,
        started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), data_sha256=data_hash,
        device=args.device, previous_run=args.previous_run,
        gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        packages={name: importlib.metadata.version(name) for name in ['torch', 'scikit-learn', 'numpy', 'pandas']})
    atomic_json(run/'manifest.json', manifest)
    atomic_json(run/'config.json', dict(grid=grids(args.smoke), tasks=TASKS, seeds=[42, 43, 44],
        expected_new_fits=26 if args.smoke else 728,
        gate=dict(RMSE_ratio=.995, MAE_ratio=1.02, worst_quarter_ratio=1.15, require_all_three_seeds=True)))
    started = time.perf_counter()
    with (run/'logs/training.log').open('a', encoding='utf-8', buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                print(json.dumps(manifest, ensure_ascii=False), flush=True)
                with threadpool_limits(limits=4):
                    run_training(run, args, previous_run)
                manifest['status'] = 'completed'
            except KeyboardInterrupt:
                manifest['status'] = 'interrupted'
                raise
            except Exception:
                manifest['status'] = 'failed'
                raise
            finally:
                manifest['seconds'] = time.perf_counter() - started
                manifest['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
                atomic_json(run/'manifest.json', manifest)


if __name__ == '__main__':
    main()
