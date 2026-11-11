"""原始尺度损失的有限预算历史实验；不含2004评价入口。"""
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
from threadpoolctl import threadpool_limits

from runoff_training.controlled_data import build_data, load_data, outer_split, inner_splits, sha256
from runoff_training.controlled_model import physical_score
from runoff_training.controlled_train import choose_epoch, metrics, environment, write_json
from runoff_training.loss_model import LossLSTM, restore_loss

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'docs/任务1至7+原始尺度损失历史对照方案_2026-11-11.md'
VARIANTS = [dict(name=f'{kind}_{space}', residual=kind == 'residual', loss_space=space)
            for space in ['log', 'physical'] for kind in ['plain', 'residual']]


def protocol(smoke, device):
    if device not in ['cpu', 'cuda'] or not isinstance(smoke, bool):
        raise ValueError('预检标志或设备不合法')
    seeds, epochs = ([42], [1]) if smoke else ([42, 43, 44], [1, 10, 30])
    models = 2*3*4*len(seeds)
    return dict(version=1, mode='historical_only', smoke=smoke, device=device, years=[2002, 2003],
        horizons=[1, 3, 7], seeds=seeds, epochs=epochs, variants=VARIANTS, inner_folds=3,
        tie_tolerance=.01, fixed=dict(profile='w90_base', units=32, batch_size=64, lr=.001,
        pooling='last', target_transform='log', dropout=.1, head=32, shuffle=False),
        fits=models*(3*len(epochs)+1), outer_models=models,
        gate=dict(min_year_RMSE_gain_pct=3., min_seeds=2, min_horizons=2,
                  max_h_RMSE_regression_pct=10., max_year_MAE_regression_pct=10.))


def validate_protocol(spec):
    if spec != protocol(spec['smoke'], spec['device']):
        raise ValueError('协议不符合预固定预算，禁止预检改标正式')


def choose(scores, variants=None):
    names = [v['name'] for v in (VARIANTS if variants is None else variants)]
    if set(scores) != set(names) or not all(np.isfinite(v) and v >= 0 for v in scores.values()):
        raise ValueError('候选分数不合法')
    best = min(scores.values())
    return next(n for n in names if scores[n] <= best*1.01)


def tag_for(year, h, variant, seed):
    return f'y{year}_direct_h{h}_{variant["name"]}_s{seed}'


def model_for(dataset, spec, variant, seed, epochs, history=None, phase='development'):
    return LossLSTM(str(dataset), seed=seed, device=spec['device'], fixed_epochs=epochs,
        residual=variant['residual'], loss_space=variant['loss_space'],
        history_dir=None if history is None else str(history), phase=phase)


def freeze_sources(run):
    sources = [ROOT/'code/runoff_training'/name for name in [
        'loss_train.py', 'loss_model.py', 'controlled_data.py', 'controlled_model.py',
        'controlled_train.py', 'strategy_model.py', 'strategy_train.py', 'direct_train.py', 'direct_data.py']]
    sources += [ROOT/'code'/name for name in [
        'verify_runoff_loss.py', 'test_runoff_loss.py', 'verify_runoff_direct.py', 'verify_runoff_strategy.py']]
    for source in sources + [PLAN]:
        shutil.copy2(source, run/'source'/source.name)
    return {p.name:sha256(p) for p in sources}


def execute(args):
    if (Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}
            or any(c in args.run_id for c in ['/', chr(92)])):
        raise ValueError('run-id必须为单目录名')
    spec = protocol(args.smoke, args.device)
    run = ROOT/'output/acceptance3_runs'/args.run_id
    run.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'history', 'cv', 'models', 'predictions']:
        (run/name).mkdir()
    manifest = dict(status='running', protocol=spec, selection_frozen_before_outer=False)
    started = time.perf_counter()
    try:
        manifest.update(source_hashes=freeze_sources(run), plan_sha256=sha256(PLAN),
            environment=environment(args.device), started=time.strftime('%Y-%m-%d %H:%M:%S'),
            data=build_data(ROOT/'Teacher Resources & Requirements/01047000.csv', run/'historical.npz'))
        write_json(run/'manifest.json', manifest)
        data = load_data(run/'historical.npz')
        records, selections, split_rows = [], [], []
        for year in spec['years']:
            train, _ = outer_split(data, year)
            folds = inner_splits(data, train)
            for fold, (ti, vi) in enumerate(folds):
                split_rows.append(dict(year=year, fold=fold, n_train=len(ti), n_valid=len(vi),
                    train_last_h7=str(np.datetime64(int(data['origin_days'][train[ti[-1]]]+7), 'D')),
                    valid_first_origin=str(np.datetime64(int(data['origin_days'][train[vi[0]]]), 'D'))))
            for h in spec['horizons']:
                own = []
                for variant in spec['variants']:
                    for seed in spec['seeds']:
                        tag = tag_for(year, h, variant, seed)
                        model = model_for(run/'historical.npz', spec, variant, seed, 1,
                                          run/'history', tag+'_inner')
                        search = GridSearchCV(model, dict(fixed_epochs=spec['epochs']), cv=folds,
                            scoring=physical_score, refit=False, n_jobs=1, error_score='raise')
                        search.fit(train[:, None], data['y'][train, h-1:h])
                        scores = -np.asarray(search.cv_results_['mean_test_score'])
                        index = choose_epoch(scores, spec['epochs'])
                        table = pd.DataFrame(search.cv_results_)
                        table['params'] = table.params.map(json.dumps)
                        table.to_csv(run/'cv'/f'{tag}.csv', index=False, encoding='utf-8-sig')
                        record = dict(year=year, task_h=h, seed=seed, tag=tag, variant=variant,
                            epochs=spec['epochs'][index], inner_RMSE=float(scores[index]),
                            inner_fold_RMSE=[-float(search.cv_results_[f'split{k}_test_score'][index]) for k in range(3)])
                        records.append(record)
                        own.append(record)
                scores = {v['name']:float(np.mean([r['inner_RMSE'] for r in own if r['variant'] == v]))
                          for v in spec['variants']}
                logs = {k:v for k,v in scores.items() if k.endswith('_log')}
                selections.append(dict(year=year, h=h, inner_scores=scores,
                    selected_variant=choose(scores), selected_log=choose(logs, VARIANTS[:2])))
                print('[历史内层]', year, h, selections[-1], flush=True)
        write_json(run/'selection.json', dict(records=records, selections=selections))
        pd.DataFrame(split_rows).to_csv(run/'inner_splits.csv', index=False, encoding='utf-8-sig')
        digest = sha256(run/'selection.json')
        manifest.update(selection_frozen_before_outer=True, selection_sha256=digest,
                        selection_frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(run/'manifest.json', manifest)
        metric_rows = []
        for record in records:
            if sha256(run/'selection.json') != digest:
                raise ValueError('冻结选择改变')
            year, h, seed, tag = [record[k] for k in ['year', 'task_h', 'seed', 'tag']]
            train, valid = outer_split(data, year)
            model = model_for(run/'historical.npz', spec, record['variant'], seed, record['epochs'],
                              run/'history', tag+'_outer').fit(train[:, None], data['y'][train, h-1:h])
            prediction = model.predict(valid[:, None])
            model.save(run/'models'/tag)
            restored = restore_loss(run/'models'/tag, run/'historical.npz', args.device)
            np.testing.assert_allclose(prediction, restored.predict(valid[:, None]), rtol=1e-7, atol=1e-5)
            identity = dict(year=year, h=h, seed=seed, variant=record['variant']['name'], epochs=record['epochs'])
            values = metrics(data['y'][valid, h-1:h], prediction)[0]
            values['h'] = h
            metric_rows.append(dict(identity, **values))
            origin = data['origin_days'][valid]
            pf = pd.DataFrame(dict(identity, sample_id=valid, origin_date=origin.astype('datetime64[D]'),
                target_date=(origin+h).astype('datetime64[D]'), truth=data['y'][valid, h-1], pred=prediction[:, 0]))
            pf.to_csv(run/'predictions'/f'{tag}.csv', index=False, encoding='utf-8-sig')
            print('[历史外层]', tag, 'epoch', record['epochs'], 'R2', round(values['R2'], 4),
                  'RMSE', round(values['RMSE'], 3), flush=True)
        pd.DataFrame(metric_rows).to_csv(run/'metrics.csv', index=False, encoding='utf-8-sig')
        fits = len(list((run/'history').glob('*.json')))
        if fits != spec['fits'] or len(records) != spec['outer_models']:
            raise ValueError('fit或模型预算不符')
        manifest.update(status='completed', fits=fits, outer_models=len(records),
            elapsed_seconds=time.perf_counter()-started, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
        manifest['frozen_files'] = {p.relative_to(run).as_posix():sha256(p) for p in run.rglob('*')
                                  if p.is_file() and p.name != 'manifest.json'}
        write_json(run/'manifest.json', manifest)
        print('[训练完成]', args.run_id, fits, len(records), flush=True)
    except Exception:
        manifest.update(status='failed', traceback=traceback.format_exc(),
                        elapsed_seconds=time.perf_counter()-started)
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
