"""GPU 改进实验：2002选参、2003前向OOF、冻结后2004回溯评估。"""
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset, physical_scorer, restore_model
from acceptance3_train import ROOT, Tee, metrics_table, prediction_rows
from acceptance3_v2_train import atomic_json, prepare_v2, split_record
from acceptance3_v3_model import V3Regressor
from acceptance3_v3_stage_b import historical_splits
from runoff_training.model import LaggedLSTMRegressor

TASKS = [('direct', h) for h in range(1, 8)] + [('multi', 7)]
FAMILIES = ['ANN', 'RF', 'LSTM']


def grids(smoke=False):
    epochs = [1] if smoke else [10, 30]
    widths = [(8,)] if smoke else [(32,), (64,)]
    common = dict(units=widths, fixed_epochs=epochs, dropout=[.1], lr=[.001])
    profiles = [dict(feature_mode=['compact_lag'], input_transform=['identity'], residual=[False]),
                dict(feature_mode=['compact_lag'], input_transform=['signed_log'], residual=[False]),
                dict(feature_mode=['compact_lag'], input_transform=['signed_log'], residual=[True])]
    result = [{**common, **profile} for profile in profiles]
    result.append(dict(feature_mode=['compact'], input_transform=['identity'], residual=[False],
                       units=[(8,)] if smoke else [(64,)], fixed_epochs=epochs,
                       dropout=[.1], lr=[.001], batch_size=[32]))
    return result


def year_folds(data, train, smoke=False):
    days = data['origin_days']
    folds = []
    for start in pd.date_range('2003-01-01', '2003-10-01', freq='QS'):
        stop = start + pd.offsets.QuarterBegin(startingMonth=1)
        first = np.datetime64(start.date()).astype('int64')
        last = np.datetime64(stop.date()).astype('int64')
        ti = train[days[train] + 7 < first]
        vi = train[(days[train] >= first) & (days[train] < last)]
        if smoke:
            vi = vi[:12]
        assert days[ti[-1]] + 7 < days[vi[0]]
        folds.append((f'{start.year}Q{start.quarter}', ti, vi))
    return folds


def get_target(data, strategy, h):
    return data['y'][:, h - 1] if strategy == 'direct' else data['y']


def make_model(path, h, params, seed, device, run, phase):
    return LaggedLSTMRegressor(str(path), horizon=h, seed=seed, device=device,
        history_dir=str(run/'training_history'), phase=phase, **params)


def score(y, pred):
    return float(np.sqrt(np.square(np.asarray(y) - pred).mean(axis=0)).mean())


def select_lstm(run, path, data, train, strategy, h, args):
    name = f'{strategy}_h{h}'
    checkpoint = run/'checkpoints'/f'search_{name}.json'
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding='utf-8'))
    days = data['origin_days']
    ids = train[days[train] + 7 <= np.datetime64('2002-12-31').astype('int64')]
    cv = historical_splits(ids, days)
    if args.smoke:
        cv = [(ti, vi[:12]) for ti, vi in cv[-1:]]
    pd.DataFrame([split_record(name, 'lstm_search', k, ids[ti], ids[vi], days)
        for k, (ti, vi) in enumerate(cv)]).to_csv(run/'splits'/f'search_{name}.csv', index=False)
    model = make_model(path, h, {}, 42, args.device, run, f'search_{name}')
    print('[网格搜索]', name, '候选', len(list(ParameterGrid(grids(args.smoke)))),
          '折', len(cv), 'device', args.device, flush=True)
    search = GridSearchCV(model, grids(args.smoke), cv=cv, scoring=physical_scorer,
                          n_jobs=1, refit=False, error_score='raise')
    search.fit(ids[:, None], get_target(data, strategy, h)[ids])
    table = pd.DataFrame(search.cv_results_)
    table['params'] = table.params.map(json.dumps)
    table.to_csv(run/'cv_results'/f'{name}.csv', index=False, encoding='utf-8-sig')
    value = dict(strategy=strategy, h=h, params=search.best_params_,
                 cv_RMSE=-float(search.best_score_), latest_search_target='2002-12-31')
    atomic_json(checkpoint, value)
    print('[搜索完成]', name, value, flush=True)
    return value


def baseline_params(strategy, h, smoke):
    if smoke:
        return dict(feature_mode='compact', units=[8], dropout=.1, lr=.001,
                    fixed_epochs=1, input_transform='identity', residual=False)
    if strategy == 'multi':
        return dict(feature_mode='compact', units=[64], dropout=.1, lr=.001,
                    fixed_epochs=10, input_transform='identity', residual=False)
    # 原直接策略七套冻结参数，仅用于同协议基线对照。
    widths = [[64], [64, 32], [64], [64, 32], [64], [64, 32], [32]]
    return dict(feature_mode='plus_Q', units=widths[h - 1],
                dropout=[.2, .2, .1, .1, .1, .1, .1][h - 1],
                lr=.0005 if h == 4 else .001,
                fixed_epochs=[18, 11, 12, 10, 9, 12, 32][h - 1])


def evaluate_gate(run, path, data, folds, selection, args):
    strategy, h = selection['strategy'], selection['h']
    file = run/'checkpoints'/f'gate_{strategy}_h{h}.json'
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    y = get_target(data, strategy, h)
    rows = []
    for label, ti, vi in folds:
        candidate = make_model(path, h, selection['params'], 42, args.device, run,
                               f'gate_{label}_{strategy}_h{h}').fit(ti[:, None], y[ti])
        p = candidate.predict(vi[:, None])
        params = baseline_params(strategy, h, args.smoke)
        params.pop('input_transform', None)
        params.pop('residual', None)
        baseline = V3Regressor(str(path), family='LSTM', horizon=h, seed=42,
            device=args.device, history_dir=str(run/'training_history'),
            phase=f'baseline_{label}_{strategy}_h{h}', **params).fit(ti[:, None], y[ti])
        bp = baseline.predict(vi[:, None])
        truth = np.asarray(y[vi]).reshape(len(vi), -1)
        pred, ref = np.asarray(p).reshape(len(vi), -1), np.asarray(bp).reshape(len(vi), -1)
        rows.append(dict(fold=label, candidate_RMSE=score(truth, pred),
                         baseline_RMSE=score(truth, ref),
                         candidate_MAE=float(np.abs(pred - truth).mean()),
                         baseline_MAE=float(np.abs(ref - truth).mean())))
    f = pd.DataFrame(rows)
    ratio = f.candidate_RMSE / f.baseline_RMSE
    passed = (f.candidate_RMSE.mean() <= .995 * f.baseline_RMSE.mean()
              and f.candidate_MAE.mean() <= 1.02 * f.baseline_MAE.mean()
              and ratio.max() <= 1.15)
    result = dict(strategy=strategy, h=h, passed=bool(passed), folds=rows,
                  RMSE_ratio=float(f.candidate_RMSE.mean() / f.baseline_RMSE.mean()),
                  worst_quarter_ratio=float(ratio.max()),
                  params=selection['params'],
                  policy='本次实验固定2002网格胜出参数；准入仅用于是否推荐替换历史交付，不反选OOF配置')
    atomic_json(file, result)
    print('[2003准入]', strategy, h, 'passed', result['passed'],
          'RMSE比值', result['RMSE_ratio'], flush=True)
    return result


def base_model(run, path, family, strategy, h, params, seed, args, phase):
    if family == 'LSTM' and params['feature_mode'] in ['compact', 'compact_lag']:
        return make_model(path, h, params, seed, args.device, run, phase)
    if family == 'LSTM':
        own = params.copy()
    elif family == 'ANN':
        own = dict(feature_mode='plus_Q', target_transform='log',
                   hidden_layer_sizes=(8,) if args.smoke else (64,), alpha=1.,
                   learning_rate_init=.001, ann_max_iter=10 if args.smoke else 1000)
    else:
        own = dict(feature_mode='plus_Q', target_transform='log',
                   n_estimators=5 if args.smoke else 300, max_depth=10, min_samples_leaf=5)
    return V3Regressor(str(path), family=family, horizon=h, seed=seed,
        device=args.device if family == 'LSTM' else 'cpu',
        history_dir=str(run/'training_history'), phase=phase, **own)


def fit_saved(run, path, data, ti, vi, family, strategy, h, params, seed, args, stage):
    name = f'{stage}_{strategy}_{family}_h{h}_seed{seed}'
    file = run/'checkpoints'/f'{name}.json'
    if file.exists():
        return np.asarray(json.loads(file.read_text(encoding='utf-8'))['predictions'])
    target = get_target(data, strategy, h)
    model = base_model(run, path, family, strategy, h, params, seed, args, name)
    model.fit(ti[:, None], target[ti])
    folder = run/'models'/name
    model.save(folder)
    ids = vi if len(vi) else ti[-7:]
    pred = np.asarray(model.predict(ids[:, None])).reshape(len(ids), -1)
    restored = restore_model(folder, device=args.device if family == 'LSTM' else 'cpu')
    np.testing.assert_allclose(restored.predict(ids[:, None]).reshape(pred.shape), pred,
                               rtol=2e-5, atol=2e-3)
    atomic_json(file, dict(predictions=pred.tolist() if len(vi) else [], restored=True))
    print('[保存并恢复]', name, flush=True)
    return pred if len(vi) else np.empty((0, 1 if strategy == 'direct' else 7))


def choose_meta(run, data, ids, x, strategy, h, seed, smoke):
    days = data['origin_days'][ids]
    y = get_target(data, strategy, h)[ids]
    candidates = [100.] if smoke else [10., 100., 1000., 10000.]
    scores = []
    for alpha in candidates:
        losses = []
        for start in ['2003-07-01', '2003-10-01']:
            begin = np.datetime64(start).astype('int64')
            end = np.datetime64((pd.Timestamp(start) + pd.offsets.QuarterBegin(startingMonth=1)).date()).astype('int64')
            ti = np.flatnonzero(days + 7 < begin)
            vi = np.flatnonzero((days >= begin) & (days < end))
            assert len(ti) and len(vi) and days[ti[-1]] + 7 < days[vi[0]]
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[ti], y[ti])
            p = model.predict(x[vi])
            losses.append(score(np.asarray(y[vi]).reshape(len(vi), -1),
                                np.asarray(p).reshape(len(vi), -1)))
        scores.append(dict(alpha=alpha, RMSE=float(np.mean(losses)), folds=losses))
    winner = min(scores, key=lambda row: row['RMSE'])
    model = make_pipeline(StandardScaler(), Ridge(alpha=winner['alpha'])).fit(x, y)
    file = run/'meta_models'/f'{strategy}_h{h}_seed{seed}.joblib'
    joblib.dump(model, file)
    np.testing.assert_allclose(joblib.load(file).predict(x), model.predict(x))
    return dict(strategy=strategy, h=h, seed=seed, alpha=winner['alpha'], candidates=scores,
                input_dim=x.shape[1], output_dim=1 if strategy == 'direct' else 7)


def run_training(run, args):
    path, _, train, test, _ = prepare_v2(run)
    data = load_dataset(str(path))
    folds = year_folds(data, train, args.smoke)
    pd.DataFrame([split_record(label, 'quarterly_oof', 0, ti, vi, data['origin_days'])
        for label, ti, vi in folds]).to_csv(run/'splits/outer.csv', index=False)
    tasks = TASKS if not args.smoke else [('direct', 1), ('multi', 7)]
    seeds = [42] if args.smoke else [42, 43, 44]
    selections, gates = [], []
    for strategy, h in tasks:
        selected = select_lstm(run, path, data, train, strategy, h, args)
        selections.append(selected)
        gates.append(evaluate_gate(run, path, data, folds, selected, args))
    atomic_json(run/'lstm_grid_selection.json', selections)
    atomic_json(run/'development_gates.json', gates)
    rows, meta = [], []
    for strategy, h in tasks:
        params = next(r['params'] for r in gates if r['strategy'] == strategy and r['h'] == h)
        for seed in seeds:
            ids, blocks = [], []
            for label, ti, vi in folds:
                block = []
                for family in FAMILIES:
                    p = fit_saved(run, path, data, ti, vi, family, strategy, h,
                                  params, seed, args, label)
                    block.append(p)
                    rows.extend(dict(fold=label, strategy=strategy, task_h=h, model=family,
                        seed=seed, sample_id=int(sample), origin_date=str(data['origin_dates'][sample]),
                        h=(h if strategy == 'direct' else j + 1), truth=float(data['y'][sample, h - 1 if strategy == 'direct' else j]),
                        pred=float(p[i, j]))
                        for i, sample in enumerate(vi) for j in range(p.shape[1]))
                ids.append(vi)
                blocks.append(np.column_stack(block))
            meta.append(choose_meta(run, data, np.concatenate(ids), np.vstack(blocks),
                                    strategy, h, seed, args.smoke))
            for family in FAMILIES:
                fit_saved(run, path, data, train, np.array([], dtype=int), family,
                          strategy, h, params, seed, args, 'final')
    oof = pd.DataFrame(rows)
    assert not oof.duplicated(['strategy', 'task_h', 'model', 'seed', 'sample_id', 'h']).any()
    for label, _, vi in folds:
        assert data['origin_days'][vi[-1]] + 7 <= np.datetime64('2003-12-31').astype('int64')
    oof.to_csv(run/'oof_predictions.csv', index=False, encoding='utf-8-sig')
    atomic_json(run/'meta_selection.json', meta)
    frozen = dict(lstm=gates, meta=meta, primary_seed=42, seeds=seeds,
        search_max_target='2002-12-31', oof_max_target='2003-12-31',
        test_2004_predicted=False,
        limitations=['2003准入与元层选参均为开发数据，不是独立泛化证明',
                     '2004此前已查看，仅为回溯评估',
                     'ANN/RF参数固定作参照，不声称本轮重新网格搜索'])
    atomic_json(run/'frozen_config.json', frozen)
    hashes = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((run/'models').rglob('*')) if p.is_file()}
    atomic_json(run/'model_hashes_before_test.json', hashes)
    # 此点之前从不调用模型预测2004，且参数和权重已冻结落盘。
    output = []
    for seed in seeds:
        by_strategy = {strategy: {family: [] for family in FAMILIES + ['Mean', 'Stacking']}
                       for strategy in sorted({s for s, _ in tasks})}
        for strategy, h in tasks:
            block = []
            for family in FAMILIES:
                folder = run/'models'/f'final_{strategy}_{family}_h{h}_seed{seed}'
                model = restore_model(folder, device=args.device if family == 'LSTM' else 'cpu')
                p = np.asarray(model.predict(test[:, None])).reshape(len(test), -1)
                by_strategy[strategy][family].append(p)
                block.append(p)
            mean = np.stack(block).mean(axis=0)
            stack = joblib.load(run/'meta_models'/f'{strategy}_h{h}_seed{seed}.joblib')
            sp = np.asarray(stack.predict(np.column_stack(block))).reshape(len(test), -1)
            by_strategy[strategy]['Mean'].append(mean)
            by_strategy[strategy]['Stacking'].append(sp)
        for strategy, models in by_strategy.items():
            for family, parts in models.items():
                pred = np.column_stack(parts)
                if args.smoke and strategy == 'direct':
                    own_rows = [dict(run_id=run.name, strategy=strategy, model=family,
                        sample_id=int(sample), origin_date=str(data['origin_dates'][sample]),
                        target_date=str(np.datetime64(int(data['origin_days'][sample]) + 1, 'D')),
                        h=1, truth=float(data['y'][sample, 0]), pred=float(pred[i, 0]))
                        for i, sample in enumerate(test)]
                else:
                    own_rows = prediction_rows(run.name, data, test, pred, strategy, family)
                for row in own_rows:
                    row['seed'] = seed
                output.extend(own_rows)
    frame = pd.DataFrame(output)
    assert np.isfinite(frame[['truth', 'pred']].to_numpy()).all()
    assert not frame.duplicated(['strategy', 'model', 'seed', 'sample_id', 'h']).any()
    frame.to_csv(run/'test_predictions_all_seeds.csv', index=False, encoding='utf-8-sig')
    tables = []
    for seed, part in frame.groupby('seed'):
        table = metrics_table(part.to_dict('records'))
        table['seed'] = seed
        tables.append(table)
    all_metrics = pd.concat(tables, ignore_index=True)
    all_metrics.to_csv(run/'metrics_all_seeds.csv', index=False, encoding='utf-8-sig')
    primary = all_metrics[all_metrics.seed == 42].drop(columns='seed')
    primary.to_csv(run/'metrics.csv', index=False, encoding='utf-8-sig')
    summary = all_metrics.groupby(['strategy', 'model'])[['R2', 'RMSE', 'MAE']].mean()
    summary.to_csv(run/'summary.csv', encoding='utf-8-sig')
    for file, digest in hashes.items():
        assert hashlib.sha256((run/file).read_bytes()).hexdigest() == digest
    required = primary[primary.model.isin(['LSTM', 'Mean', 'Stacking'])]
    assert len(required) == (24 if args.smoke else 42)
    histories = list((run/'training_history').glob('*.json'))
    for file in histories:
        record = json.loads(file.read_text(encoding='utf-8'))
        assert data['origin_days'][record['last_id']] + 7 <= np.datetime64('2003-12-31').astype('int64')
        if 'cv_score' in record:
            assert data['origin_days'][record['last_id']] + 7 < data['origin_days'][record['validation_first_id']]
    atomic_json(run/'verification.json', dict(status='passed', required_metric_rows=len(required),
        required_metric_values=len(required)*3, test_origins=len(test), seeds=seeds,
        fit_records=len(histories), all_models_restored=True, frozen_hashes_unchanged=True,
        test_type='retrospective', smoke=args.smoke,
        all_seven_direct_horizons_trained=not args.smoke))
    print('[完成]', summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ['.', '..']:
        parser.error('run-id必须为单目录名')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA不可用，请使用--device cpu')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run = ROOT/'output/acceptance3_runs'/args.run_id
    sources = [Path(__file__), Path(__file__).with_name('model.py')]
    sources += [ROOT/'code'/name for name in ['acceptance3_model.py', 'acceptance3_v2_model.py',
        'acceptance3_v3_model.py', 'acceptance3_train.py', 'acceptance3_v2_train.py',
        'acceptance3_v3_stage_b.py', 'feature_engineering.py', 'config.py', 'utils_plot.py']]
    digest = hashlib.sha256((ROOT/'Teacher Resources & Requirements/01047000.csv').read_bytes()).hexdigest()
    if args.resume:
        old = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        assert old['status'] in ['failed', 'interrupted', 'running']
        assert old['data_sha256'] == digest and old['smoke'] == args.smoke and old['device'] == args.device
        for p in sources:
            assert (run/'source'/p.name).read_bytes() == p.read_bytes(), p
    else:
        run.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'models', 'meta_models', 'training_history', 'splits', 'cv_results', 'checkpoints', 'logs']:
        (run/name).mkdir(exist_ok=args.resume)
    for p in sources:
        shutil.copy2(p, run/'source'/p.name)
    manifest = dict(run_id=args.run_id, status='running', smoke=args.smoke,
        started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), data_sha256=digest, device=args.device,
        gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        packages={name: importlib.metadata.version(name) for name in ['torch', 'scikit-learn', 'numpy', 'pandas']})
    atomic_json(run/'manifest.json', manifest)
    atomic_json(run/'config.json', dict(grids=grids(args.smoke), tasks=TASKS, primary_seed=42,
        gate=dict(quarter_mean_RMSE_ratio=.995, MAE_ratio=1.02, worst_quarter_ratio=1.15),
        meta_alphas=[100.] if args.smoke else [10., 100., 1000., 10000.]))
    started = time.perf_counter()
    with (run/'logs/training.log').open('a', encoding='utf-8', buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                print(json.dumps(manifest, ensure_ascii=False), flush=True)
                with threadpool_limits(limits=4):
                    run_training(run, args)
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
