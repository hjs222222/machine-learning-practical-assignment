"""直接残差固定预算实验：七步历史扩展、h1同epoch消融、冻结后2004回溯。"""
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

from runoff_training.controlled_data import build_data, load_data, inner_splits, outer_split, sha256
from runoff_training.controlled_model import physical_score
from runoff_training.controlled_train import choose_epoch, metrics, environment, write_json
from runoff_training.direct_data import build_evaluation_data, final_split
from runoff_training.strategy_model import StrategyLSTM, restore_strategy
from runoff_training.strategy_train import choose_variant

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT/'Teacher Resources & Requirements/01047000.csv'
PLAN = ROOT/'docs/任务1至7+直接残差扩展与最终对照方案_2026-11-11.md'


def protocol(mode, smoke, device):
    if mode not in ['historical', 'final'] or device not in ['cpu', 'cuda']:
        raise ValueError('只接受固定实验模式和设备')
    seeds = [42] if smoke else [42, 43, 44]
    years = [2002, 2003] if mode == 'historical' else [2004]
    epochs = [1] if smoke else [1, 10, 30]
    variants = [dict(name='last_plain_log', pooling='last', residual=False, target_transform='log'),
                dict(name='last_residual_log', pooling='last', residual=True, target_transform='log')]
    searched = len(years)*7*2*len(seeds)
    ablations = len(years)*2*len(seeds)*2 if mode == 'historical' else 0
    return dict(version=1, mode=mode, smoke=bool(smoke), device=device, years=years, seeds=seeds,
        horizons=list(range(1, 8)), epochs=epochs, variants=variants, inner_folds=3,
        tie_tolerance=.01, profile='w90_base', fixed=dict(units=32, batch_size=64, lr=.001,
        dropout=.1, head=32, shuffle=False), ablation_epochs=[10, 30] if ablations else [],
        fits=searched*(3*len(epochs)+1)+ablations, outer_models=searched+ablations,
        selected_models=len(years)*7*len(seeds), selection_before_evaluation=True)


def validate_protocol(spec):
    expected = protocol(spec['mode'], spec['smoke'], spec['device'])
    if spec != expected:
        raise ValueError('运行协议与预固定配置不符；禁止预检改标正式或修改预算')


def model_for(dataset, spec, variant, seed, epochs, history=None, phase='development'):
    return StrategyLSTM(str(dataset), profile='w90_base', fixed_epochs=epochs, seed=seed,
        device=spec['device'], target_transform='log', units=32, batch_size=64, lr=.001,
        history_dir=str(history) if history is not None else None, phase=phase,
        pooling='last', residual=variant['residual'])


def split_for(data, mode, year):
    return final_split(data) if mode == 'final' else outer_split(data, year)


def tag_for(year, h, variant, seed, kind='searched', epoch=None):
    tag = f'y{year}_direct_h{h}_{variant["name"]}_s{seed}'
    return tag if kind == 'searched' else f'{tag}_fixed{epoch}'


def execute(args):
    if (Path(args.run_id).name != args.run_id or any(s in args.run_id for s in ['/', '\\'])
            or args.run_id in {'.', '..'}):
        raise ValueError('run-id必须是单目录名')
    spec = protocol(args.mode, args.smoke, args.device)
    run = ROOT/'output/acceptance3_runs'/args.run_id
    run.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'history', 'cv', 'models', 'predictions']:
        (run/name).mkdir()
    sources = [Path(__file__), Path(__file__).with_name('direct_data.py'),
        Path(__file__).with_name('strategy_model.py'), Path(__file__).with_name('strategy_train.py'),
        Path(__file__).with_name('controlled_data.py'), Path(__file__).with_name('controlled_model.py'),
        Path(__file__).with_name('controlled_train.py'), ROOT/'code/verify_runoff_direct.py',
        ROOT/'code/test_runoff_direct.py', ROOT/'code/analyze_runoff_direct.py',
        ROOT/'code/analyze_runoff_strategy.py', ROOT/'code/analyze_runoff_controlled.py',
        ROOT/'code/verify_runoff_strategy.py']
    for source in sources:
        shutil.copy2(source, run/'source'/source.name)
    shutil.copy2(PLAN, run/'source'/PLAN.name)
    info = build_data(RAW_PATH, run/'historical.npz')
    manifest = dict(status='running', protocol=spec, historical_data=info,
        source_hashes={p.name:sha256(p) for p in sources}, plan_sha256=sha256(PLAN),
        environment=environment(args.device), started=time.strftime('%Y-%m-%d %H:%M:%S'),
        selection_frozen_before_outer=False)
    write_json(run/'manifest.json', manifest)
    started = time.perf_counter()
    try:
        data = load_data(run/'historical.npz')
        records, selections, split_rows = [], [], []
        for year in spec['years']:
            if spec['mode'] == 'final':
                train = np.arange(len(data['positions']))
            else:
                train, _ = outer_split(data, year)
            folds = inner_splits(data, train)
            for k, (ti, vi) in enumerate(folds):
                split_rows.append(dict(year=year, fold=k, n_train=len(ti), n_valid=len(vi),
                    train_last_h7=str(np.datetime64(int(data['origin_days'][train[ti[-1]]]+7), 'D')),
                    valid_first_origin=str(np.datetime64(int(data['origin_days'][train[vi[0]]]), 'D'))))
            for h in spec['horizons']:
                own = []
                y = data['y'][:, h-1:h]
                for variant in spec['variants']:
                    for seed in spec['seeds']:
                        tag = tag_for(year, h, variant, seed)
                        model = model_for(run/'historical.npz', spec, variant, seed, 1,
                                          run/'history', tag+'_inner')
                        search = GridSearchCV(model, dict(fixed_epochs=spec['epochs']), cv=folds,
                            scoring=physical_score, n_jobs=1, refit=False, error_score='raise')
                        search.fit(train[:, None], y[train])
                        scores = -np.asarray(search.cv_results_['mean_test_score'])
                        index = choose_epoch(scores, spec['epochs'])
                        table = pd.DataFrame(search.cv_results_)
                        table['params'] = table.params.map(json.dumps)
                        table.to_csv(run/'cv'/f'{tag}.csv', index=False, encoding='utf-8-sig')
                        record = dict(year=year, strategy='direct', task_h=h, seed=seed, variant=variant,
                            tag=tag, kind='searched', epochs=spec['epochs'][index],
                            inner_RMSE=float(scores[index]), inner_fold_RMSE=[
                                -float(search.cv_results_[f'split{k}_test_score'][index]) for k in range(3)])
                        records.append(record)
                        own.append(record)
                scores = {v['name']:float(np.mean([r['inner_RMSE'] for r in own
                          if r['variant']['name'] == v['name']])) for v in spec['variants']}
                selections.append(dict(year=year, strategy='direct', task_h=h, inner_scores=scores,
                                       selected_variant=choose_variant(scores)))
                print('[内层]', year, h, selections[-1]['selected_variant'], flush=True)
            for epoch in spec['ablation_epochs']:
                for variant in spec['variants']:
                    for seed in spec['seeds']:
                        records.append(dict(year=year, strategy='direct', task_h=1, seed=seed,
                            variant=variant, tag=tag_for(year, 1, variant, seed, 'fixed', epoch),
                            kind='fixed', epochs=epoch, inner_RMSE=None, inner_fold_RMSE=None))
        write_json(run/'selection.json', dict(records=records, selections=selections))
        pd.DataFrame(split_rows).to_csv(run/'inner_splits.csv', index=False, encoding='utf-8-sig')
        digest = sha256(run/'selection.json')
        manifest.update(selection_frozen_before_outer=True, selection_sha256=digest,
            selection_frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(run/'manifest.json', manifest)
        print('[选择全部冻结]', digest, flush=True)
        dataset = run/'historical.npz'
        if spec['mode'] == 'final':
            dataset = run/'evaluation.npz'
            manifest['evaluation_data'] = build_evaluation_data(RAW_PATH, run/'historical.npz', dataset)
            data = load_data(dataset)
            write_json(run/'manifest.json', manifest)
        metric_rows = []
        for record in records:
            if sha256(run/'selection.json') != digest:
                raise ValueError('冻结选择被改变')
            year, h, seed = [record[k] for k in ['year', 'task_h', 'seed']]
            train, valid = split_for(data, spec['mode'], year)
            model = model_for(dataset, spec, record['variant'], seed, record['epochs'],
                run/'history', record['tag']+'_outer').fit(train[:, None], data['y'][train, h-1:h])
            pred = model.predict(valid[:, None])
            model.save(run/'models'/record['tag'])
            rebuilt = restore_strategy(run/'models'/record['tag'], dataset, args.device).predict(valid[:, None])
            np.testing.assert_allclose(pred, rebuilt, rtol=1e-7, atol=1e-5)
            identity = dict(year=year, strategy='direct', task_h=h, variant=record['variant']['name'],
                seed=seed, kind=record['kind'], epochs=record['epochs'])
            values = metrics(data['y'][valid, h-1:h], pred)[0]
            values['h'] = h
            metric_rows.append(dict(**identity, **values))
            rows = []
            for i, sample in enumerate(valid):
                origin = int(data['origin_days'][sample])
                rows.append(dict(**identity, sample_id=int(sample), h=h,
                    origin_date=str(np.datetime64(origin, 'D')),
                    target_date=str(np.datetime64(origin+h, 'D')),
                    truth=float(data['y'][sample, h-1]), pred=float(pred[i, 0])))
            pd.DataFrame(rows).to_csv(run/'predictions'/f'{record["tag"]}.csv', index=False, encoding='utf-8-sig')
            print('[外层]', record['tag'], 'epoch', record['epochs'], 'RMSE', round(values['RMSE'], 3), flush=True)
        pd.DataFrame(metric_rows).to_csv(run/'metrics.csv', index=False, encoding='utf-8-sig')
        count = len(list((run/'history').glob('*.json')))
        if count != spec['fits'] or len(records) != spec['outer_models']:
            raise ValueError('实际fit/模型数与固定预算不同')
        manifest.update(status='completed', fits=count, outer_models=len(records),
            elapsed_seconds=time.perf_counter()-started, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
        manifest['frozen_files'] = {p.relative_to(run).as_posix():sha256(p) for p in run.rglob('*')
                                  if p.is_file() and p.name != 'manifest.json'}
        write_json(run/'manifest.json', manifest)
        print('[完成]', run.name, count, round(manifest['elapsed_seconds'], 1), flush=True)
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc())
        write_json(run/'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--mode', choices=['historical', 'final'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA不可用')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with threadpool_limits(limits=4):
        execute(args)


if __name__ == '__main__':
    main()
