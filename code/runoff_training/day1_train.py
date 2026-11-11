"""第1天R²专项：历史四折合并R²选参、原始尺度增量和多模型族对照。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from runoff_training.controlled_data import build_data, load_data, outer_split, sha256
from runoff_training.controlled_train import write_json, environment, metrics
from runoff_training.day1_model import Day1Regressor, restore_day1

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'docs/任务1至7+第一天R2专项目标与方案_2026-11-11.md'
FAMILIES = ['Ridge', 'RF', 'ANN', 'LSTM']


def candidates(smoke):
    result = [dict(name=f'Ridge_a{alpha}', family='Ridge', alpha=float(alpha)) for alpha in [10, 1000]]
    result += [dict(name=f'RF_l{leaf}', family='RF', leaf=leaf, trees=200) for leaf in [2, 10]]
    result += [dict(name=f'ANN_e{epochs}', family='ANN', epochs=epochs) for epochs in [30, 100]]
    result += [dict(name=f'LSTM_w{window}_e{epochs}', family='LSTM', window=window, epochs=epochs)
               for window in [7, 30] for epochs in [10, 30]]
    if smoke:
        result = [next(c.copy() for c in result if c['family'] == family) for family in FAMILIES]
        for c in result:
            if c['family'] == 'RF':
                c['trees'] = 5
            if c['family'] in ['ANN', 'LSTM']:
                c['epochs'] = 1
    return result


def protocol(smoke, device):
    if not isinstance(smoke, bool) or device not in ['cpu', 'cuda']:
        raise ValueError('预检标志或设备不合法')
    seeds = [42] if smoke else [42, 43, 44]
    configs = candidates(smoke)
    return dict(version=1, mode='historical_only', smoke=smoke, device=device,
        years=[2002, 2003], h=1, target_R2=.8, seeds=seeds, candidates=configs,
        inner_folds=4, valid_days=90, purge=7, selection_tolerance_R2=.005,
        fits=2*len(configs)*len(seeds)*5, outer_models=2*len(configs)*len(seeds),
        gate=dict(min_year_R2=.70, min_R2_gain=.02, min_improved_seeds=2, max_MAE_regression_pct=10.))


def validate_protocol(spec):
    if spec != protocol(spec['smoke'], spec['device']):
        raise ValueError('第1天专项协议与固定预算不符')


def folds_for(data, train):
    days = data['origin_days'][train]
    result = []
    for stop in range(len(train)-270, len(train)+1, 90):
        vi = np.arange(stop-90, stop)
        if vi[0] < 0:
            raise ValueError('四季内层历史不足')
        ti = np.flatnonzero(days+7 < days[vi[0]])
        if len(ti) < 150 or days[ti[-1]]+7 >= days[vi[0]]:
            raise ValueError('四季内层净化或样本量不足')
        result.append((ti, vi))
    return result


def scorer(model, X, y):
    truth = np.asarray(y, dtype=float).reshape(-1)
    prediction = model.predict(X).reshape(-1)
    return dict(R2=float(r2_score(truth, prediction)), neg_SSE=-float(np.square(truth-prediction).sum()))


def pooled_score(fold_sse, truths):
    errors = np.asarray(fold_sse, dtype=float)
    values = [np.asarray(y, dtype=float).reshape(-1) for y in truths]
    if (errors.shape != (4,) or len(values) != 4 or not np.isfinite(errors).all()
            or (errors < 0).any() or any(not len(y) or not np.isfinite(y).all() for y in values)):
        raise ValueError('合并内层R²输入不合法')
    all_y = np.concatenate(values)
    sst = np.square(all_y-all_y.mean()).sum()
    if sst <= 0:
        raise ValueError('合并真值方差必须为正')
    return float(1-errors.sum()/sst)


def choose(scores):
    if not scores or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('第1天候选R²必须有限')
    best = max(scores.values())
    return next(name for name,value in scores.items() if value >= best-.005)


def model_for(dataset, spec, candidate, seed, history=None, phase='development'):
    params = {k:v for k,v in candidate.items() if k != 'name'}
    return Day1Regressor(str(dataset), seed=seed,
        device=spec['device'] if candidate['family'] in ['ANN', 'LSTM'] else 'cpu',
        history_dir=None if history is None else str(history), phase=phase, **params)


def execute(args):
    if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'} or any(s in args.run_id for s in ['/', chr(92)]):
        raise ValueError('run-id必须为单目录名')
    run = ROOT/'output/acceptance3_runs'/args.run_id
    run.mkdir(parents=True, exist_ok=False)
    for folder in ['source', 'history', 'cv', 'models', 'predictions']:
        (run/folder).mkdir()
    spec = protocol(args.smoke, args.device)
    manifest = dict(status='running', protocol=spec, selection_frozen_before_outer=False)
    start = time.perf_counter()
    try:
        sources = [ROOT/'code/runoff_training'/name for name in ['day1_train.py', 'day1_model.py',
                   'controlled_data.py', 'controlled_train.py', 'controlled_model.py']]
        sources += [ROOT/'code'/name for name in ['verify_runoff_day1.py', 'test_runoff_day1.py']]
        for source in sources+[PLAN]:
            shutil.copy2(source, run/'source'/source.name)
        manifest.update(source_hashes={p.name:sha256(p) for p in sources}, plan_sha256=sha256(PLAN),
            data=build_data(ROOT/'Teacher Resources & Requirements/01047000.csv', run/'historical.npz'),
            environment=environment(args.device), started=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(run/'manifest.json', manifest)
        data = load_data(run/'historical.npz')
        records, selections, split_rows = [], [], []
        for year in spec['years']:
            train, _ = outer_split(data, year)
            folds = folds_for(data, train)
            truths = [data['y'][train[vi], 0] for _,vi in folds]
            for fold,(ti,vi) in enumerate(folds):
                split_rows.append(dict(year=year, fold=fold, n_train=len(ti), n_valid=len(vi),
                    train_last_h7=str(np.datetime64(int(data['origin_days'][train[ti[-1]]]+7), 'D')),
                    valid_first=str(np.datetime64(int(data['origin_days'][train[vi[0]]]), 'D'))))
            for family in FAMILIES:
                own_configs = [c for c in spec['candidates'] if c['family'] == family]
                for seed in spec['seeds']:
                    phase = f'y{year}_{family}_s{seed}_inner'
                    estimator = model_for(run/'historical.npz', spec, own_configs[0], seed, run/'history', phase)
                    grid = [{k:[v] for k,v in c.items() if k not in ['name', 'family']} for c in own_configs]
                    search = GridSearchCV(estimator, grid, cv=folds, scoring=scorer, refit=False, n_jobs=1, error_score='raise')
                    search.fit(train[:, None], data['y'][train, 0])
                    table = pd.DataFrame(search.cv_results_)
                    table['params'] = table.params.map(json.dumps)
                    table['pooled_R2'] = [pooled_score([-search.cv_results_[f'split{k}_test_neg_SSE'][i] for k in range(4)], truths)
                                           for i in range(len(own_configs))]
                    table.to_csv(run/'cv'/f'y{year}_{family}_s{seed}.csv', index=False, encoding='utf-8-sig')
                    for i,c in enumerate(own_configs):
                        records.append(dict(year=year, seed=seed, candidate=c, tag=f'y{year}_{c["name"]}_s{seed}',
                            inner_R2=float(table.pooled_R2.iloc[i]),
                            inner_fold_R2=[float(search.cv_results_[f'split{k}_test_R2'][i]) for k in range(4)],
                            inner_fold_SSE=[-float(search.cv_results_[f'split{k}_test_neg_SSE'][i]) for k in range(4)]))
            own = [r for r in records if r['year'] == year]
            scores = {c['name']:float(np.mean([r['inner_R2'] for r in own if r['candidate'] == c])) for c in spec['candidates']}
            winners = {family:choose({c['name']:scores[c['name']] for c in spec['candidates'] if c['family'] == family})
                       for family in FAMILIES}
            family_scores = {w:scores[w] for w in winners.values()}
            selections.append(dict(year=year, scores=scores, family_winners=winners, selected_variant=choose(family_scores)))
            print('[第1天内层选择]', year, selections[-1], flush=True)
        write_json(run/'selection.json', dict(records=records, selections=selections))
        pd.DataFrame(split_rows).to_csv(run/'inner_splits.csv', index=False, encoding='utf-8-sig')
        digest = sha256(run/'selection.json')
        manifest.update(selection_frozen_before_outer=True, selection_sha256=digest,
                        selection_frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(run/'manifest.json', manifest)
        rows = []
        for record in records:
            if sha256(run/'selection.json') != digest:
                raise ValueError('冻结选择改变')
            year,seed,c,tag = [record[k] for k in ['year', 'seed', 'candidate', 'tag']]
            train, valid = outer_split(data, year)
            model = model_for(run/'historical.npz', spec, c, seed, run/'history', tag+'_outer')
            model.fit(train[:, None], data['y'][train, 0])
            prediction = model.predict(valid[:, None])
            model.save(run/'models'/tag)
            restored = restore_day1(run/'models'/tag, run/'historical.npz', model.device)
            np.testing.assert_allclose(prediction, restored.predict(valid[:, None]), rtol=1e-7, atol=1e-5)
            values = metrics(data['y'][valid, :1], prediction)[0]
            rows.append(dict(year=year, seed=seed, candidate=c['name'], family=c['family'], **values))
            origin = data['origin_days'][valid]
            pd.DataFrame(dict(year=year, seed=seed, candidate=c['name'], family=c['family'], sample_id=valid,
                origin_date=origin.astype('datetime64[D]'), target_date=(origin+1).astype('datetime64[D]'),
                truth=data['y'][valid, 0], pred=prediction[:, 0])).to_csv(run/'predictions'/f'{tag}.csv', index=False, encoding='utf-8-sig')
            print('[第1天外层]', tag, 'R2', round(values['R2'], 4), 'RMSE', round(values['RMSE'], 2), flush=True)
        pd.DataFrame(rows).to_csv(run/'metrics.csv', index=False, encoding='utf-8-sig')
        fits = len(list((run/'history').glob('*.json')))
        if fits != spec['fits'] or len(records) != spec['outer_models']:
            raise ValueError('第1天实际预算不符')
        manifest.update(status='completed', fits=fits, outer_models=len(records),
                        elapsed_seconds=time.perf_counter()-start, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
        manifest['frozen_files'] = {p.relative_to(run).as_posix():sha256(p) for p in run.rglob('*')
                                  if p.is_file() and p.name != 'manifest.json'}
        write_json(run/'manifest.json', manifest)
    except Exception:
        manifest.update(status='failed', traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-start)
        write_json(run/'manifest.json', manifest)
        raise
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with threadpool_limits(limits=4):
        execute(args)


if __name__ == '__main__':
    main()
