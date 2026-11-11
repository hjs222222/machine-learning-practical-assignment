"""策略增强局部重训：先冻结全部历史内层选择，再计算年度外层成绩。"""
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

from runoff_training.controlled_data import build_data, load_data, sha256, outer_split, inner_splits
from runoff_training.controlled_model import physical_score
from runoff_training.controlled_train import choose_epoch, metrics, environment, write_json
from runoff_training.strategy_model import StrategyLSTM, restore_strategy

ROOT = Path(__file__).resolve().parents[2]
TASKS = [('direct', 1), ('direct', 7), ('multi', 7)]


def variants(strategy, h):
    values = []
    for pooling in ['last', 'last_mean']:
        for residual in [False, True]:
            for transform in (['log', 'sqrt'] if strategy == 'direct' and h == 7 else ['log']):
                name = f'{pooling}_{"residual" if residual else "plain"}_{transform}'
                values.append(dict(name=name, pooling=pooling, residual=residual, target_transform=transform))
    return values


def choose_variant(scores):
    best = min(scores.values())
    return next(name for name, score in scores.items() if score <= best*1.01)


def tag_for(year, strategy, h, variant, seed):
    return f'y{year}_{strategy}_h{h}_{variant["name"]}_s{seed}'


def spec_for(args):
    return dict(version=1, smoke=args.smoke, device=args.device, years=[2002, 2003], tasks=TASKS,
        seeds=[42] if args.smoke else [42,43,44], epochs=[1] if args.smoke else [1,10,30],
        variants={f'{s}_h{h}':variants(s,h) for s,h in TASKS}, profile='w90_base',
        tie_tolerance=.01, inner_folds=3, max_data_day='2003-12-31',
        fixed=dict(units=32, batch_size=64, lr=.001, dropout=.1, head=32, shuffle=False),
        selection='全候选先内层选epoch与策略并冻结，之后才计算外层；策略按三个seed内层均值及固定顺序',
        fits=128 if args.smoke else 960, outer_models=32 if args.smoke else 96)


def target(data, strategy, h):
    return data['y'][:,h-1:h] if strategy == 'direct' else data['y']


def parameters(variant):
    return {k:variant[k] for k in ['pooling','residual','target_transform']}


def execute(args):
    run = ROOT/'output/acceptance3_runs'/args.run_id
    run.mkdir(parents=True, exist_ok=False)
    for name in ['source','history','cv','models','predictions']:
        (run/name).mkdir()
    sources = [Path(__file__), Path(__file__).with_name('strategy_model.py'),
        Path(__file__).with_name('controlled_data.py'), Path(__file__).with_name('controlled_model.py'),
        Path(__file__).with_name('controlled_train.py'), ROOT/'code/test_runoff_strategy.py',
        ROOT/'code/verify_runoff_strategy.py']
    for source in sources:
        shutil.copy2(source, run/'source'/source.name)
    info = build_data(ROOT/'Teacher Resources & Requirements/01047000.csv', run/'dataset.npz')
    spec = spec_for(args)
    manifest = dict(status='running', protocol=spec, data=info, environment=environment(args.device),
        source_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in sources},
        started=time.strftime('%Y-%m-%d %H:%M:%S'), selection_frozen_before_outer=False)
    write_json(run/'manifest.json', manifest)
    started = time.perf_counter()
    try:
        data = load_data(run/'dataset.npz')
        records, selected, split_rows = [], [], []
        # 第一阶段绝不计算外层预测，也不读取旧run的预测/指标。
        for year in spec['years']:
            train, valid = outer_split(data, year)
            folds = inner_splits(data, train)
            for k,(ti,vi) in enumerate(folds):
                split_rows.append(dict(year=year, fold=k, n_train=len(ti), n_valid=len(vi),
                    train_last_h7=str(np.datetime64(int(data['origin_days'][train[ti[-1]]]+7),'D')),
                    valid_first_origin=str(np.datetime64(int(data['origin_days'][train[vi[0]]]),'D'))))
            for strategy,h in TASKS:
                y = target(data,strategy,h)
                own = []
                for variant in variants(strategy,h):
                    for seed in spec['seeds']:
                        tag = tag_for(year,strategy,h,variant,seed)
                        model = StrategyLSTM(str(run/'dataset.npz'), seed=seed, device=args.device,
                            history_dir=str(run/'history'), phase=tag+'_inner', **parameters(variant))
                        search = GridSearchCV(model, dict(fixed_epochs=spec['epochs']), cv=folds,
                            scoring=physical_score, refit=False, n_jobs=1, error_score='raise')
                        search.fit(train[:,None], y[train])
                        scores = -np.asarray(search.cv_results_['mean_test_score'])
                        index = choose_epoch(scores,spec['epochs'])
                        table = pd.DataFrame(search.cv_results_)
                        table['params'] = table.params.map(json.dumps)
                        table.to_csv(run/'cv'/f'{tag}.csv',index=False,encoding='utf-8-sig')
                        record = dict(year=year,strategy=strategy,task_h=h,variant=variant,seed=seed,
                            epochs=spec['epochs'][index],inner_RMSE=float(scores[index]),
                            inner_fold_RMSE=[-float(search.cv_results_[f'split{k}_test_score'][index])
                                             for k in range(3)],tag=tag)
                        records.append(record)
                        own.append(record)
                scores = {v['name']:float(np.mean([r['inner_RMSE'] for r in own
                            if r['variant']['name']==v['name']])) for v in variants(strategy,h)}
                selected.append(dict(year=year,strategy=strategy,task_h=h,inner_scores=scores,
                                     selected_variant=choose_variant(scores)))
                print('[内层完成]',year,strategy,h,'选择',selected[-1]['selected_variant'],flush=True)
        write_json(run/'selection.json',dict(records=records,selections=selected))
        pd.DataFrame(split_rows).to_csv(run/'inner_splits.csv',index=False,encoding='utf-8-sig')
        selection_hash = sha256(run/'selection.json')
        manifest.update(selection_frozen_before_outer=True,selection_sha256=selection_hash,
            selection_frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(run/'manifest.json',manifest)
        print('[全部选择已冻结] sha256',selection_hash,'现在开始外层拟合和评价',flush=True)
        metric_rows = []
        for record in records:
            if sha256(run/'selection.json') != selection_hash:
                raise ValueError('冻结选择已改变')
            year,strategy,h,seed = [record[k] for k in ['year','strategy','task_h','seed']]
            variant,tag = record['variant'],record['tag']
            train,valid = outer_split(data,year)
            y = target(data,strategy,h)
            model = StrategyLSTM(str(run/'dataset.npz'),seed=seed,device=args.device,
                fixed_epochs=record['epochs'],history_dir=str(run/'history'),phase=tag+'_outer',
                **parameters(variant)).fit(train[:,None],y[train])
            pred = model.predict(valid[:,None])
            model.save(run/'models'/tag)
            rebuilt = restore_strategy(run/'models'/tag,run/'dataset.npz',args.device).predict(valid[:,None])
            np.testing.assert_allclose(pred,rebuilt,rtol=1e-7,atol=1e-5)
            identity = dict(year=year,strategy=strategy,task_h=h,variant=variant['name'],seed=seed,
                            epochs=record['epochs'],inner_RMSE=record['inner_RMSE'])
            values = metrics(y[valid],pred)
            if strategy=='direct':
                values[0]['h']=h
            metric_rows.extend([dict(**identity,**v) for v in values])
            prediction_rows = []
            for j in range(pred.shape[1]):
                horizon = h if strategy=='direct' else j+1
                for k,sample in enumerate(valid):
                    origin = int(data['origin_days'][sample])
                    prediction_rows.append(dict(**identity,sample_id=int(sample),h=horizon,
                        origin_date=str(np.datetime64(origin,'D')),
                        target_date=str(np.datetime64(origin+horizon,'D')),
                        truth=float(y[sample,j]),pred=float(pred[k,j])))
            pd.DataFrame(prediction_rows).to_csv(run/'predictions'/f'{tag}.csv',index=False,encoding='utf-8-sig')
            print('[外层完成]',tag,'epoch',record['epochs'],'RMSE',
                  round(float(np.mean([v['RMSE'] for v in values])),3),flush=True)
        pd.DataFrame(metric_rows).to_csv(run/'metrics.csv',index=False,encoding='utf-8-sig')
        count = len(list((run/'history').glob('*.json')))
        if count!=spec['fits'] or len(list((run/'models').iterdir()))!=spec['outer_models']:
            raise ValueError('实际fit或模型数量与预固定预算不同')
        manifest.update(status='completed',fits=count,outer_models=spec['outer_models'],
            finished=time.strftime('%Y-%m-%d %H:%M:%S'),elapsed_seconds=time.perf_counter()-started)
        manifest['frozen_files']={p.relative_to(run).as_posix():sha256(p)
            for p in run.rglob('*') if p.is_file() and p.name!='manifest.json'}
        write_json(run/'manifest.json',manifest)
        print('[完成]',run,'fits',count,'seconds',round(manifest['elapsed_seconds'],1),flush=True)
    except Exception:
        manifest.update(status='failed',error=traceback.format_exc())
        write_json(run/'manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or '/' in args.run_id or '\\' in args.run_id or args.run_id in {'.','..'}:
        raise ValueError('run-id只能是单个目录名')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA不可用')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    with threadpool_limits(limits=4):
        execute(args)


if __name__=='__main__':
    main()
