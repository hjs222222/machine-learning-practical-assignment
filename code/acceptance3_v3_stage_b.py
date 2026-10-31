# -*- coding: utf-8 -*-
"""第三轮阶段B：完整7步、历史内层固定epoch选参、年度外层开发比较。"""
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
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset, physical_scorer, restore_model
from acceptance3_train import ROOT, Tee
from acceptance3_v2_train import atomic_json, prepare_v2, split_record
from acceptance3_v3_model import V3Regressor
from acceptance3_v3_train import SOURCES, development_splits, frame_metrics

TASKS = [('direct', h) for h in range(1, 8)] + [('multi', 7)]
STAGE_SOURCES = SOURCES + ['acceptance3_v3_stage_b.py']


def historical_splits(ids, days):
    # 每个内层验证90个起报点，最多4窗覆盖约一年；最早训练至少365条。
    count = min(4, (len(ids) - 365 - 7) // 90)
    if count < 1:
        raise ValueError('合法历史不足一个90日内层窗')
    pairs = []
    for start in range(len(ids) - 90 * count, len(ids), 90):
        ti, vi = np.arange(start - 7), np.arange(start, start + 90)
        assert len(ti) >= 365
        assert days[ids[ti[-1]]] + 7 < days[ids[vi[0]]]
        pairs.append((ti, vi))
    return pairs


def grid(profile, smoke):
    return dict(feature_mode=['compact' if profile == 'A1' else 'plus_Q'],
        target_transform=['log'], mixed_weight=[0.], units=[(8,)] if smoke else ([(32,), (64,)] if profile == 'A1' else [(64,)]),
        dropout=[.2] if smoke or profile == 'A0' else [.1, .2],
        lr=[.001], fixed_epochs=[1] if smoke else [10, 20, 40])


def choose(run, path, data, ids, strategy, h, profile, year, smoke):
    label = f'year{year}_{strategy}_h{h}_{profile}'
    cache = run / 'checkpoints' / ('choose_' + label + '.json')
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    cv = historical_splits(ids, data['origin_days'])
    audit = [split_record(str(year), f'inner_{strategy}_h{h}_{profile}', k, ids[ti], ids[vi], data['origin_days'])
             for k, (ti, vi) in enumerate(cv)]
    pd.DataFrame(audit).to_csv(run / 'splits' / (label + '.csv'), index=False, encoding='utf-8-sig')
    target = data['y'][:, h - 1] if strategy == 'direct' else data['y']
    model = V3Regressor(str(path), family='LSTM', horizon=h, seed=42,
        history_dir=str(run / 'training_history'), phase=label)
    print('[历史内层搜索]', label, '折数', len(cv), flush=True)
    search = GridSearchCV(model, grid(profile, smoke), cv=cv, scoring=physical_scorer,
        n_jobs=1, refit=False, error_score='raise', verbose=1)
    search.fit(ids[:, None], target[ids])
    table = pd.DataFrame(search.cv_results_)
    table['params'] = table.params.map(lambda x: json.dumps(x))
    table.to_csv(run / 'cv_results' / (label + '.csv'), index=False, encoding='utf-8-sig')
    params = search.best_params_.copy()
    params['units'] = list(params['units'])
    chosen = dict(strategy=strategy, h=h, profile=profile, year=year, params=params,
        n_folds=len(cv), inner_score=float(search.best_score_), selected_without_outer_labels=True)
    atomic_json(cache, chosen)
    return chosen


def refit(run, path, data, ti, vi, record, seed):
    strategy, h, profile, year = [record[k] for k in ['strategy','h','profile','year']]
    label = f'year{year}_{strategy}_h{h}_{profile}_seed{seed}'
    cache = run / 'checkpoints' / (label + '.json')
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    target = data['y'][:, h - 1] if strategy == 'direct' else data['y']
    model = V3Regressor(str(path), family='LSTM', horizon=h, seed=seed,
        history_dir=str(run / 'training_history'), phase=label, **record['params']).fit(ti[:,None], target[ti])
    folder = run / 'models' / label
    model.save(folder)
    pred = model.predict(vi[:,None])
    actual = restore_model(folder, device='cpu').predict(vi[:,None])
    np.testing.assert_allclose(actual,pred,rtol=1e-6,atol=1e-4)
    metrics, predictions = frame_metrics(data,ti,vi,pred,target,strategy,h,profile,
        record['params']['fixed_epochs'],seed,year)
    result = dict(metrics=metrics,predictions=predictions,restored=True)
    atomic_json(cache,result)
    print('[年度外层复核]',label,flush=True)
    return result


def verify(run, data, folds, rows, preds, selections, smoke):
    frame = pd.DataFrame(rows)
    p = pd.DataFrame(preds)
    assert pd.to_datetime(p.target_date).max() <= pd.Timestamp('2003-12-31')
    assert not p.duplicated(['strategy','task_h','profile','seed','year','h','sample_id']).any()
    for key, f in p.groupby(['strategy','task_h','profile','seed','year','h']):
        strategy, task_h, profile, seed, year, h = key
        r = frame[(frame.strategy==strategy)&(frame.task_h==task_h)&(frame.profile==profile)&
            (frame.seed==seed)&(frame.year==year)&(frame.h==h)&(frame.segment=='year')].iloc[0]
        np.testing.assert_allclose([r.RMSE,r.MAE],
            [np.sqrt(np.square(f.pred-f.truth).mean()),np.abs(f.pred-f.truth).mean()],rtol=1e-10)
    counts = sum(len(historical_splits(ti,data['origin_days'])) for _,ti,_ in folds)
    expected_search = len(TASKS)*counts*(2 if smoke else 15)
    expected_refit = len(TASKS)*len(folds)*2*(1 if smoke else 3)
    actual = len(list((run/'training_history').glob('*.json')))
    assert actual == expected_search+expected_refit,(actual,expected_search,expected_refit)
    assert len(selections)==32
    for path in (run/'training_history').glob('*.json'):
        r=json.loads(path.read_text(encoding='utf-8'))
        assert 'early_history' not in r and len(r['history'])==r['params']['fixed_epochs']
        if 'cv_score' in r:
            assert data['origin_days'][r['last_id']]+7<data['origin_days'][r['validation_first_id']]
    atomic_json(run/'verification.json',dict(status='passed',stage='B',fit_records=actual,
        expected_fit_records=expected_search+expected_refit,all_selected_models_restored=True,
        prediction_rows=len(p),test_2004_predicted=False,full_training_completed=False))


def run_stage(run, smoke):
    source = ROOT/'output/acceptance3_runs/formal_20261031_v3_stage_a'
    decision = json.loads((source/'stage_a_decision.json').read_text(encoding='utf-8'))
    assert any(r['profile']=='A1' and r['screening_passed'] for r in decision['gates'])
    atomic_json(run/'stage_a_source_decision.json',decision)
    path,_,_,_,_=prepare_v2(run)
    data=load_dataset(str(path))
    folds=development_splits(data)
    pd.DataFrame([split_record(str(year),'annual_outer',0,ti,vi,data['origin_days']) for year,ti,vi in folds]).to_csv(
        run/'outer_splits.csv',index=False,encoding='utf-8-sig')
    selections,rows,preds=[],[],[]
    for year,ti,vi in folds:
        for strategy,h in TASKS:
            for profile in ['A0','A1']:
                r=choose(run,path,data,ti,strategy,h,profile,year,smoke)
                selections.append(r)
                for seed in ([42] if smoke else [42,43,44]):
                    result=refit(run,path,data,ti,vi,r,seed)
                    rows.extend(result['metrics']);preds.extend(result['predictions'])
                atomic_json(run/'best_parameters.json',selections)
    frame=pd.DataFrame(rows)
    frame.to_csv(run/'development_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(preds).to_csv(run/'development_predictions.csv',index=False,encoding='utf-8-sig')
    annual=frame[frame.segment=='year']
    reference=annual[annual.profile=='A0']
    paired=annual[annual.profile=='A1'].merge(reference,on=['strategy','task_h','seed','year','h','segment'],suffixes=('','_A0'))
    paired['RMSE_ratio']=paired.RMSE/paired.RMSE_A0
    paired['MAE_ratio']=paired.MAE/paired.MAE_A0
    paired.to_csv(run/'paired_reference.csv',index=False,encoding='utf-8-sig')
    gates=[]
    for strategy,f in paired.groupby('strategy'):
        # 对每年每seed先取7个h的原尺度指标均值，再计算与参考之比。
        avg=f.groupby(['seed','year'])[['RMSE','MAE','RMSE_A0','MAE_A0']].mean()
        rmse=float((avg.RMSE/avg.RMSE_A0).mean());mae=float((avg.MAE/avg.MAE_A0).mean())
        byyear=(avg.RMSE/avg.RMSE_A0).groupby('year').mean()
        byh=f.groupby('h').RMSE_ratio.mean()
        seedwins=int(((avg.RMSE/avg.RMSE_A0).groupby('seed').mean()<1).sum())
        passed=rmse<=.98 and mae<=1.02 and byyear.max()<=1.02 and (byh<1).sum()>=4 and byh.max()<=1.10 and seedwins>=(1 if smoke else 2)
        gates.append(dict(strategy=strategy,mean_7h_RMSE_ratio=rmse,mean_7h_MAE_ratio=mae,
            worst_year_RMSE_ratio=float(byyear.max()),improved_horizons=int((byh<1).sum()),
            worst_horizon_RMSE_ratio=float(byh.max()),improved_seeds=seedwins,passed=bool(passed)))
    verify(run,data,folds,rows,preds,selections,smoke)
    atomic_json(run/'stage_b_decision.json',dict(stage='B_complete',gates=gates,
        any_passed=any(r['passed'] for r in gates),
        limitation='结构和epoch选择在各外层之前的历史内完成；A1扩展来自已查看同开发年份的阶段A，仍非完全独立确认，2004未预测。',
        next_step='通过者进入完整基模型OOF及约束元层阶段；未通过不扩大该策略网格。'))
    print('[阶段B年度开发完成，2004未预测]\n',pd.DataFrame(gates).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ['.','..']:
        parser.error('run-id必须是单目录名')
    run=ROOT/'output/acceptance3_runs'/args.run_id
    digest=hashlib.sha256((ROOT/'Teacher Resources & Requirements/01047000.csv').read_bytes()).hexdigest()
    previous={}
    if args.resume:
        previous=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        assert previous['status']=='failed' and previous['smoke']==args.smoke and previous['data_sha256']==digest
        for name in STAGE_SOURCES:
            assert (run/'source'/name).read_bytes()==(ROOT/'code'/name).read_bytes(),name
    else:
        run.mkdir(parents=True,exist_ok=False)
    for folder in ['source','models','training_history','logs','cv_results','splits','checkpoints']:
        (run/folder).mkdir(exist_ok=args.resume)
    for name in STAGE_SOURCES:
        shutil.copy2(ROOT/'code'/name,run/'source'/name)
    manifest=dict(run_id=args.run_id,version=3,phase='annual_nested_development',status='running',smoke=args.smoke,
        started=previous.get('started',time.strftime('%Y-%m-%dT%H:%M:%S%z')),data_sha256=digest,
        device='cpu',torch_build=torch.__version__,
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','torch']},
        full_training_completed=False,test_2004_predicted=False)
    atomic_json(run/'manifest.json',manifest)
    atomic_json(run/'config.json',dict(A0_grid=grid('A0',args.smoke),A1_grid=grid('A1',args.smoke),
        tasks=TASKS,outer_years=[2002,2003],inner_window_days=90,max_inner_folds=4,min_inner_train=365,
        seeds=[42] if args.smoke else [42,43,44],primary_seed=42,
        planned_fit_count=144 if args.smoke else 936,
        gates=dict(RMSE_ratio=.98,MAE_ratio=1.02,worst_year_ratio=1.02,min_improved_horizons=4,
                   worst_horizon_ratio=1.10,min_improved_seeds=1 if args.smoke else 2)))
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    with (run/'logs/training.log').open('a' if args.resume else 'w',encoding='utf-8',buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout,logfile)),contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
            try:
                print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
                with threadpool_limits(limits=4):run_stage(run,args.smoke)
                manifest['status']='stage_b_completed'
            except Exception:
                import traceback
                traceback.print_exc();manifest['status']='failed';raise
            finally:
                manifest['seconds']=previous.get('seconds',0)+time.perf_counter()-start
                manifest['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
                atomic_json(run/'manifest.json',manifest)


if __name__=='__main__':main()
