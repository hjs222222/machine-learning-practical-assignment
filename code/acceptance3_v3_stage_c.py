# -*- coding: utf-8 -*-
"""第三轮阶段C：仅多输出，季度因果OOF、小网格、保守元层与最终参数冻结。"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV, ParameterGrid
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset, physical_scorer, restore_model
from acceptance3_train import ROOT, Tee
from acceptance3_v2_train import atomic_json, prepare_v2, split_record
from acceptance3_v3_model import V3Regressor
from acceptance3_v3_stage_b import historical_splits, grid, STAGE_SOURCES
from acceptance3_v3_fusion import ConservativeFusion, fusion_candidates

FAMILIES = ['ANN', 'RF', 'LSTM']
SOURCES = STAGE_SOURCES + ['acceptance3_v3_fusion.py', 'acceptance3_v3_stage_c.py']


def quarterly_splits(data, train):
    days = data['origin_days']
    folds = []
    last_target = np.datetime64('2003-12-31').astype('int64')
    for start in pd.date_range('2001-07-01', '2003-10-01', freq='QS'):
        end = start + pd.offsets.QuarterBegin(startingMonth=1)
        first = np.datetime64(start.date()).astype('int64')
        stop = np.datetime64(end.date()).astype('int64')
        ti = train[days[train] + 7 < first]
        vi = train[(days[train] >= first) & (days[train] < stop) & (days[train] + 7 <= last_target)]
        assert days[ti[-1]] + 7 < days[vi[0]]
        historical_splits(ti, days)
        folds.append((f'{start.year}Q{start.quarter}', ti, vi))
    return folds


def family_grid(family, smoke):
    if family == 'LSTM':
        return grid('A1', smoke)
    common = dict(feature_mode=['plus_Q'], target_transform=['log'])
    if family == 'ANN':
        return dict(**common, hidden_layer_sizes=[(8,)] if smoke else [(64,)],
            alpha=[.1] if smoke else [.01, .1, 1.], learning_rate_init=[.001])
    return dict(**common, n_estimators=[5] if smoke else [300],
        max_depth=[3] if smoke else [10, 20], min_samples_leaf=[2] if smoke else [1, 5])


def archive_uncommitted_history(run, label, family):
    # 重试只隔离未提交逻辑步骤的历史，不删除证据，也不重复计入成功拟合预算。
    files = list((run/'training_history').glob(f'{label}_{family}_h7_*.json'))
    if files:
        folder = run/'retry_history'/uuid.uuid4().hex
        folder.mkdir(parents=True)
        for file in files:
            file.rename(folder/file.name)


def make_model(run, path, family, phase, smoke, seed=42, **params):
    return V3Regressor(str(path), family=family, horizon=7, seed=seed,
        history_dir=str(run/'training_history'), phase=phase,
        ann_max_iter=10 if smoke else 1000, device='cpu', **params)


def choose(run, path, data, ids, stage, family, smoke):
    label = f'{stage}_{family}'
    checkpoint = run/'checkpoints'/f'choose_{label}.json'
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding='utf-8'))
    archive_uncommitted_history(run,label,family)
    cv = historical_splits(ids, data['origin_days'])
    pd.DataFrame([split_record(stage, family, k, ids[ti], ids[vi], data['origin_days'])
        for k,(ti,vi) in enumerate(cv)]).to_csv(run/'splits'/f'{label}.csv', index=False, encoding='utf-8-sig')
    print('[季度历史搜索]', label, '内折', len(cv), flush=True)
    search = GridSearchCV(make_model(run,path,family,label,smoke), family_grid(family,smoke),
        cv=cv, scoring=physical_scorer, n_jobs=1, refit=False, error_score='raise')
    search.fit(ids[:,None], data['y'][ids])
    table = pd.DataFrame(search.cv_results_)
    table['params'] = table.params.map(json.dumps)
    table.to_csv(run/'cv_results'/f'{label}.csv',index=False,encoding='utf-8-sig')
    params = search.best_params_.copy()
    for name in ['units','hidden_layer_sizes']:
        if name in params: params[name] = list(params[name])
    result = dict(stage=stage,family=family,params=params,n_folds=len(cv),score=float(search.best_score_))
    atomic_json(checkpoint,result)
    return result


def refit(run,path,data,ti,vi,selection,seed,smoke):
    label = f"{selection['stage']}_{selection['family']}_seed{seed}"
    checkpoint = run/'checkpoints'/f'fit_{label}.json'
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding='utf-8'))
    archive_uncommitted_history(run,label,selection['family'])
    model = make_model(run,path,selection['family'],label,smoke,seed,**selection['params'])
    model.fit(ti[:,None],data['y'][ti])
    folder = run/'models'/label
    model.save(folder)
    # final仅在训练历史上检查恢复；阶段C不向predict传入2004。
    check_ids = vi if len(vi) else ti[-7:]
    prediction = model.predict(check_ids[:,None])
    restored = restore_model(folder,device='cpu').predict(check_ids[:,None])
    np.testing.assert_allclose(restored,prediction,rtol=1e-6,atol=1e-4)
    result = dict(restored=True,stage=selection['stage'],family=selection['family'],seed=seed,
        ids=vi.tolist(),predictions=prediction.tolist() if len(vi) else [])
    atomic_json(checkpoint,result)
    print('[模型重训保存]',label,flush=True)
    return result


def meta_splits(ids, days, folds):
    pairs = []
    for label,_,val_ids in folds:
        ti = np.flatnonzero(days[ids] + 7 < days[val_ids[0]])
        vi = np.flatnonzero(np.isin(ids,val_ids))
        if len(ti) >= 180:
            assert days[ids[ti[-1]]] + 7 < days[ids[vi[0]]]
            pairs.append((label,ti,vi))
    if not pairs: raise ValueError('因果元层历史不足180条')
    return pairs


def score(y,p):
    return float(np.sqrt(np.square(p-y).mean(axis=0)).mean()),float(np.abs(p-y).mean())


def build_meta(run,data,matrices,folds,seeds,smoke):
    ids = matrices[42][0]
    cv = meta_splits(ids,data['origin_days'],folds)
    candidates = fusion_candidates(smoke)
    # 无收缩原尺度Ridge只作诊断对照，不参与部署选择。
    candidates.append(dict(kind='ridge_RF',alpha=100.,strength=1.))
    rows, predictions = [], []
    for seed in seeds:
        own_ids,x = matrices[seed]
        assert np.array_equal(ids,own_ids)
        y = data['y'][ids]
        for index,params in enumerate(candidates):
            for label,ti,vi in cv:
                model = ConservativeFusion(**params).fit(x[ti],y[ti])
                p = model.predict(x[vi])
                rmse,mae = score(y[vi],p)
                rows.append(dict(seed=seed,candidate=index,fold=label,RMSE=rmse,MAE=mae,
                    negative_predictions=int((p<0).sum()),n=len(vi),
                    max_train_target=str(np.datetime64(int(data['origin_days'][ids[ti[-1]]]+7),'D')),
                    validation_origin_start=str(data['origin_dates'][ids[vi[0]]])))
                predictions.extend(dict(seed=seed,candidate=index,fold=label,sample_id=int(sample_id),
                    h=h+1,truth=float(y[v,h]),pred=float(p[i,h]))
                    for i,(v,sample_id) in enumerate(zip(vi,ids[vi])) for h in range(7))
    frame = pd.DataFrame(rows)
    frame.to_csv(run/'meta_validation.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(predictions).to_csv(run/'meta_validation_predictions.csv',index=False,encoding='utf-8-sig')
    # 配置只由预先指定seed42选择；其他seed用于稳定性披露，不择优。
    primary = frame[frame.seed==42]
    reference = primary[primary.candidate==0].set_index('fold')
    eligible = [(float(reference.RMSE.mean()),0)]
    gates = []
    for index in range(1,len(candidates)-1):
        f = primary[primary.candidate==index].set_index('fold').loc[reference.index]
        ratio = f.RMSE/reference.RMSE
        mae = f.MAE/reference.MAE
        passed = ratio.mean()<=.99 and mae.mean()<=1.02 and ratio.max()<=1.10
        gates.append(dict(candidate=index,RMSE_ratio=float(ratio.mean()),MAE_ratio=float(mae.mean()),
            worst_quarter_RMSE_ratio=float(ratio.max()),passed=bool(passed)))
        if passed: eligible.append((float(f.RMSE.mean()),index))
    _,winner = min(eligible)
    for seed in seeds:
        own_ids,x = matrices[seed]
        model = ConservativeFusion(**candidates[winner]).fit(x,data['y'][own_ids])
        file = run/'meta_models'/f'multi_seed{seed}.joblib'
        joblib.dump(model,file)
        np.testing.assert_allclose(joblib.load(file).predict(x),model.predict(x))
        stacking = ConservativeFusion(**candidates[-1]).fit(x,data['y'][own_ids])
        stacking_file = run/'meta_models'/f'stacking_seed{seed}.joblib'
        joblib.dump(stacking,stacking_file)
        np.testing.assert_allclose(joblib.load(stacking_file).predict(x),stacking.predict(x))
    decision = dict(selected_index=winner,selected_params=candidates[winner],candidates=candidates,gates=gates,
        selection_seed=42,RF_fallback=winner==0,stacking_params=candidates[-1],
        stacking_policy='独立报告真正21→7原尺度Ridge；RF退路只列SelectedFusion，不冒充Stacking',input_columns=[f'{f}_h{h}' for f in FAMILIES for h in range(1,8)],
        limitation='同一前向季度验证用于元层候选选择和准入，分数为开发选参成绩而非独立确认；2004未预测。')
    atomic_json(run/'meta_decision.json',decision)
    return decision


def run_stage(run,smoke):
    source = ROOT/'output/acceptance3_runs/formal_20261031_v3_stage_b'
    decision = json.loads((source/'stage_b_decision.json').read_text(encoding='utf-8'))
    assert any(r['strategy']=='multi' and r['passed'] for r in decision['gates'])
    atomic_json(run/'stage_b_source_decision.json',decision)
    path,_,train,_,_ = prepare_v2(run)
    data = load_dataset(str(path))
    folds = quarterly_splits(data,train)
    pd.DataFrame([split_record(label,'quarterly_outer',0,ti,vi,data['origin_days'])
        for label,ti,vi in folds]).to_csv(run/'outer_splits.csv',index=False,encoding='utf-8-sig')
    seeds = [42] if smoke else [42,43,44]
    matrices = {seed:[] for seed in seeds}
    selections,oof = [],[]
    for stage,ti,vi in folds:
        block = {seed:[] for seed in seeds}
        for family in FAMILIES:
            selected = choose(run,path,data,ti,stage,family,smoke)
            selections.append(selected)
            for seed in seeds:
                result = refit(run,path,data,ti,vi,selected,seed,smoke)
                p = np.asarray(result['predictions'])
                block[seed].append(p)
                oof.extend(dict(stage=stage,seed=seed,model=family,sample_id=int(sample_id),h=h+1,
                    origin_date=str(data['origin_dates'][sample_id]),
                    target_date=str(np.datetime64(int(data['origin_days'][sample_id]+h+1),'D')),
                    truth=float(data['y'][sample_id,h]),pred=float(p[i,h]))
                    for i,sample_id in enumerate(vi) for h in range(7))
        for seed in seeds: matrices[seed].append((vi,np.column_stack(block[seed])))
        atomic_json(run/'best_parameters.json',selections)
    matrices = {seed:(np.concatenate([r[0] for r in items]),np.vstack([r[1] for r in items]))
                for seed,items in matrices.items()}
    frame = pd.DataFrame(oof)
    frame.to_csv(run/'oof_predictions.csv',index=False,encoding='utf-8-sig')
    assert not frame.duplicated(['seed','model','sample_id','h']).any()
    assert pd.to_datetime(frame.target_date).max()<=pd.Timestamp('2003-12-31')
    meta = build_meta(run,data,matrices,folds,seeds,smoke)
    # 统一冻结完成前不进行2004预测。
    for family in FAMILIES:
        selected = choose(run,path,data,train,'final',family,smoke)
        selections.append(selected)
        for seed in seeds: refit(run,path,data,train,np.array([],dtype=int),selected,seed,smoke)
    atomic_json(run/'best_parameters.json',selections)
    planned = sum(len(historical_splits(ti,data['origin_days'])) for _,ti,_ in folds)
    planned += len(historical_splits(train,data['origin_days']))
    expected = planned*sum(len(list(ParameterGrid(family_grid(f,smoke)))) for f in FAMILIES)
    expected += (len(folds)+1)*len(FAMILIES)*len(seeds)
    records = list((run/'training_history').glob('*.json'))
    assert len(records)==expected,(len(records),expected)
    for file in records:
        r = json.loads(file.read_text(encoding='utf-8'))
        assert data['origin_days'][r['last_id']]+7<=np.datetime64('2003-12-31').astype('int64')
        if 'cv_score' in r:
            assert data['origin_days'][r['last_id']]+7<data['origin_days'][r['validation_first_id']]
        if r['family']=='LSTM':
            assert 'early_history' not in r and len(r['history'])==r['params']['fixed_epochs']
    atomic_json(run/'verification.json',dict(status='passed',stage='C',fit_records=len(records),
        expected_fit_records=expected,retried_fit_records=len(list((run/'retry_history').rglob('*.json'))),
        quarterly_blocks=len(folds),oof_samples=len(matrices[42][0]),
        all_selected_models_restored=True,test_2004_predicted=False,full_training_completed=False))
    atomic_json(run/'frozen_config.json',dict(strategy='multi',families=FAMILIES,seeds=seeds,primary_seed=42,
        final_parameters=[r for r in selections if r['stage']=='final'],meta=meta,
        direct_policy='阶段B未准入，不扩大或替换第二轮直接策略',test_2004_predicted=False))
    print('[阶段C完成：多输出模型和元层已冻结，2004未预测]',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ['.','..']: parser.error('run-id必须为单目录名')
    run = ROOT/'output/acceptance3_runs'/args.run_id
    digest = hashlib.sha256((ROOT/'Teacher Resources & Requirements/01047000.csv').read_bytes()).hexdigest()
    previous = {}
    if args.resume:
        previous = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        assert previous['status'] in ['failed','interrupted'] and previous['smoke']==args.smoke and previous['data_sha256']==digest
        for name in SOURCES: assert (run/'source'/name).read_bytes()==(ROOT/'code'/name).read_bytes(),name
    else: run.mkdir(parents=True,exist_ok=False)
    for name in ['source','models','meta_models','training_history','logs','cv_results','splits','checkpoints']:
        (run/name).mkdir(exist_ok=args.resume)
    for name in SOURCES: shutil.copy2(ROOT/'code'/name,run/'source'/name)
    manifest = dict(run_id=args.run_id,version=3,phase='multi_quarterly_oof',status='running',smoke=args.smoke,
        started=previous.get('started',time.strftime('%Y-%m-%dT%H:%M:%S%z')),data_sha256=digest,
        device='cpu',torch_build=torch.__version__,packages={p:importlib.metadata.version(p)
        for p in ['numpy','pandas','scikit-learn','torch','scipy']},
        full_training_completed=False,test_2004_predicted=False)
    atomic_json(run/'manifest.json',manifest)
    atomic_json(run/'config.json',dict(family_grids={f:family_grid(f,args.smoke) for f in FAMILIES},
        quarter_start='2001-07-01',quarter_end='2003-12-24',seeds=[42] if args.smoke else [42,43,44],
        meta_candidates=fusion_candidates(args.smoke),meta_min_train=180,
        meta_gate=dict(RMSE_ratio=.99,MAE_ratio=1.02,worst_quarter_RMSE_ratio=1.10),
        diagnostic_ridge=dict(kind='ridge_RF',alpha=100.,strength=1.),test_2004_predicted=False))
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    with (run/'logs/training.log').open('a' if args.resume else 'w',encoding='utf-8',buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout,logfile)),contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
            try:
                print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
                with threadpool_limits(limits=4): run_stage(run,args.smoke)
                manifest['status']='stage_c_completed'
            except KeyboardInterrupt:
                manifest['status']='interrupted'
                raise
            except Exception:
                import traceback
                traceback.print_exc();manifest['status']='failed';raise
            finally:
                manifest['seconds']=previous.get('seconds',0)+time.perf_counter()-start
                manifest['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
                atomic_json(run/'manifest.json',manifest)


if __name__=='__main__': main()
