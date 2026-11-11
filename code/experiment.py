"""训练期嵌套时序网格搜索、因果 OOF 和冻结后单次正式评价。"""
from pathlib import Path
import json
import time
import shutil
import warnings
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning
from config import ROOT, DATA_PATH, SEED, save_json, digest
from data import development_splits
from models import RunoffRegressor, simple_average, fit_stacking
from evaluation import metric_rows


# 教师指导书第四.2/四.3 规定的模型清单。这三个常量是整个实验的模型口径来源，
# 报告、清单与自查脚本都从这里取，避免出现与实现对不上的手写数字。
STRATEGIES = ('direct', 'multi')      # 多步直接 / 多输出
FAMILIES = ('RF', 'ANN', 'LSTM')      # 2 种机器学习 + 1 种深度学习
ENSEMBLES = ('Mean', 'Stacking')      # 2 种集成学习


def horizon_groups(strategy):
    """返回 [(预见期, 目标列切片)]。

    多步直接：每个预见期各训 1 个模型，共 7 个；
    多输出：只有 1 个模型，同时输出 t+1..t+7。
    """
    if strategy == 'direct':
        return [(h, slice(h - 1, h)) for h in range(1, 8)]
    if strategy == 'multi':
        return [(7, slice(0, 7))]
    raise ValueError(f'未知预测策略：{strategy}')


def base_model_names():
    """24 个个体模型：3 方法 × (7 直接 + 1 多输出) = 24。"""
    return [f'{s}_{f}_h{h}' for s in STRATEGIES for f in FAMILIES for h, _ in horizon_groups(s)]


def ensemble_model_names():
    """16 个集成模型：2 集成方法 × (7 直接 + 1 多输出) = 16。

    Mean 不训练、在评价时按成员预测求平均；Stacking 需要拟合元学习器。
    两者都计入教师要求的集成模型数量。
    """
    return [f'{s}_{e}_h{h}' for s in STRATEGIES for e in ENSEMBLES for h, _ in horizon_groups(s)]


def frozen_grids():
    grids = {
        'RF': [dict(transform=[t],leaf=[leaf],trees=[120],top_k=[24])
               for t in ('raw','sqrt','log') for leaf in (1,5)],
        'ANN': [dict(transform=[t],hidden=[hidden],alpha=[0.1],iterations=[350],top_k=[24])
                for t in ('raw','sqrt','log') for hidden in (32,64)],
        'LSTM': [dict(transform=[t],hidden=[32],epochs=[epochs],seq_len=[seq],alpha=[0.1],top_k=[16])
                 for t,epochs,seq in [('raw',30,7),('raw',70,14),('sqrt',70,14),('log',70,14)]]
    }
    for family in ('RF','ANN','LSTM'):
        for transform in ('sqrt','log'):
            cfg = grids[family][-1].copy()
            cfg.update(transform=[transform],residual=[True])
            if family == 'RF':
                cfg['leaf'] = [1]
            grids[family].append(cfg)
    return grids


def nested_oof_splits(dates):
    for tr, va in development_splits(dates,3):
        yield tr, va, development_splits(pd.DatetimeIndex(dates)[tr],2)


def require_frozen(run):
    run = Path(run)
    if not (run/'frozen.json').exists() or (run/'test_started.json').exists():
        raise ValueError('必须已冻结且从未正式测试，禁止重复测试选配置')


def score_r2(estimator, X, y):
    return float(r2_score(np.asarray(y).reshape(len(y),-1),estimator.predict(X),multioutput='uniform_average'))


def search(family, X, y, splits, grid, path):
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        gs = GridSearchCV(RunoffRegressor(family=family),grid,scoring=score_r2,cv=splits,
                          refit=True,n_jobs=1,error_score='raise',return_train_score=False)
        gs.fit(X,y)
    pd.DataFrame(gs.cv_results_).to_csv(path,index=False,encoding='utf-8-sig')
    save_json(path.with_suffix('.json'),dict(best_params=gs.best_params_,best_cv_R2=gs.best_score_,
        seconds=time.monotonic()-start,convergence_warnings=sum(isinstance(w.message,ConvergenceWarning) for w in caught)))
    return gs


def benchmark(ds, run):
    train = ds['train']
    rows = []
    for family in ('RF','ANN','LSTM'):
        cfg = {k:v[0] for k,v in frozen_grids()[family][0].items()}
        before = time.monotonic()
        model = RunoffRegressor(family=family,**cfg).fit(ds['X'][train[:500]],ds['y'][train[:500],:1])
        pred = model.predict(ds['X'][train[510:650]])
        rows.append(dict(family=family,seconds=time.monotonic()-before,n_train=500,n_validation=140,
                         h1_R2=float(r2_score(ds['y'][train[510:650],0],pred[:,0])),
                         device=getattr(model,'device_used_','cpu')))
    save_json(run/'benchmark.json',rows)
    return rows


def develop(ds, run):
    grids = frozen_grids()
    save_json(run/'protocol.json',dict(grids=grids,seed=SEED,cv='TimeSeriesSplit n_splits=2 gap=7',
        oof='3 expanding outer folds, each with independent 2-fold inner selection gap=7',
        selection='mean validation R2; multi uniform average across 7 horizons',
        epochs='LSTM fixed budget epochs selected by purged GridSearchCV, not random early-stopping',
        data_sha256=digest(DATA_PATH),test_has_been_evaluated=False,
        versions={m:importlib.metadata.version(m) for m in ('numpy','pandas','scikit-learn','torch','matplotlib','python-docx')}))
    for folder in ('cv','models','oof','features','source'):
        (run/folder).mkdir()
    for path in (ROOT/'code').glob('*.py'):
        shutil.copy2(path,run/'source'/path.name)
    train = ds['train']
    X,y,dates = ds['X'][train],ds['y'][train],ds['dates'][train]
    oof_specs = list(nested_oof_splits(dates))
    pd.DataFrame([dict(fold=i,train_start=str(dates[tr[0]].date()),train_origin_end=str(dates[tr[-1]].date()),
        train_target_end=str((dates[tr[-1]]+pd.Timedelta(days=7)).date()),
        validation_start=str(dates[va[0]].date()),validation_origin_end=str(dates[va[-1]].date()),
        n_train=len(tr),n_validation=len(va)) for i,(tr,va,_) in enumerate(oof_specs)]).to_csv(run/'oof'/'splits.csv',index=False)
    selected, scores = {}, []
    for strategy in STRATEGIES:
        groups = horizon_groups(strategy)
        oof_family, fitted = {}, {}
        for family in FAMILIES:
            oof_family[family] = np.full(y.shape,np.nan)
            for h,sl in groups:
                key = f'{strategy}_{family}_h{h}'
                print('DEVELOP',key,flush=True)
                gs = search(family,X,y[:,sl],development_splits(dates),grids[family],run/'cv'/f'{key}.csv')
                selected[key] = dict(params=gs.best_params_,cv_R2=float(gs.best_score_))
                joblib.dump(gs.best_estimator_,run/'models'/f'{key}.joblib')
                est = gs.best_estimator_
                cat = ds['catalog'].copy()
                cat['MI'] = est.mi_
                cat['selected'] = np.isin(np.arange(len(cat)),est.selected_)
                cat.to_csv(run/'features'/f'{key}.csv',index=False,encoding='utf-8-sig')
                save_json(run/'features'/f'{key}_scalers.json',dict(x_mean=est.x_scaler_.mean_.tolist(),
                    x_scale=est.x_scaler_.scale_.tolist(),y_mean=est.y_scaler_.mean_.tolist(),y_scale=est.y_scaler_.scale_.tolist()))
                if hasattr(est,'loss_history_'):
                    pd.DataFrame({'epoch':np.arange(1,len(est.loss_history_)+1),'loss':est.loss_history_}).to_csv(run/'cv'/f'{key}_loss.csv',index=False)
                for fold,(tr,va,inner) in enumerate(oof_specs):
                    nested = search(family,X[tr],y[tr,sl],inner,grids[family],run/'cv'/f'{key}_oof{fold}.csv')
                    p = nested.predict(X[va])
                    oof_family[family][va,sl] = p
                    for row in metric_rows(strategy,family,y[va,sl],p,1 if strategy=='multi' else h):
                        row.update(fold=fold,phase='nested_oof')
                        scores.append(row)
        valid = np.isfinite(oof_family['RF']).all(1)
        matrices = [oof_family[f][valid] for f in ('RF','ANN','LSTM')]
        pd.DataFrame({'origin':dates[valid].astype(str)}).to_csv(run/'oof'/f'{strategy}_dates.csv',index=False)
        np.savez_compressed(run/'oof'/f'{strategy}.npz',truth=y[valid],**{f:oof_family[f][valid] for f in ('RF','ANN','LSTM')})
        if strategy == 'multi':
            meta = fit_stacking(matrices,y[valid])
            joblib.dump(meta,run/'models'/'multi_Stacking.joblib')
        else:
            for h in range(1,8):
                meta = fit_stacking([p[:,h-1:h] for p in matrices],y[valid,h-1:h])
                joblib.dump(meta,run/'models'/f'direct_Stacking_h{h}.joblib')
    pd.DataFrame(scores).to_csv(run/'development_scores.csv',index=False,encoding='utf-8-sig')
    save_json(run/'frozen.json',dict(selected=selected,model_hashes={p.name:digest(p) for p in (run/'models').glob('*.joblib')},
                                  source_hashes={p.name:digest(p) for p in (run/'source').glob('*.py')},
                                  data_sha256=digest(DATA_PATH),test_evaluated=False))


def evaluate(ds, run):
    require_frozen(run)
    frozen = json.loads((run/'frozen.json').read_text(encoding='utf-8'))
    for name,sha in frozen['model_hashes'].items():
        if digest(run/'models'/name) != sha:
            raise ValueError('冻结模型被改变')
    if frozen['data_sha256'] != digest(DATA_PATH):
        raise ValueError('原始数据被改变')
    save_json(run/'test_started.json',dict(started=pd.Timestamp.now().isoformat(),reason='single frozen retrospective evaluation'))
    test = ds['test']
    X,y,dates = ds['X'][test],ds['y'][test],ds['dates'][test]
    rows, daily = [], []
    all_predictions = {}
    for strategy in ('direct','multi'):
        family_predictions = []
        for family in ('RF','ANN','LSTM'):
            if strategy == 'direct':
                pred = np.concatenate([joblib.load(run/'models'/f'direct_{family}_h{h}.joblib').predict(X) for h in range(1,8)],axis=1)
            else:
                pred = joblib.load(run/'models'/f'multi_{family}_h7.joblib').predict(X)
            family_predictions.append(pred)
            all_predictions[f'{strategy}_{family}'] = pred
        all_predictions[f'{strategy}_Mean'] = simple_average(family_predictions)
        if strategy == 'multi':
            meta = joblib.load(run/'models'/'multi_Stacking.joblib')
            pred = meta.predict(np.concatenate(family_predictions,axis=1))
        else:
            pred = np.concatenate([joblib.load(run/'models'/f'direct_Stacking_h{h}.joblib').predict(
                np.concatenate([p[:,h-1:h] for p in family_predictions],axis=1)).reshape(-1,1) for h in range(1,8)],axis=1)
        all_predictions[f'{strategy}_Stacking'] = np.maximum(pred,0)
        for model in ('RF','ANN','LSTM','Mean','Stacking'):
            p = all_predictions[f'{strategy}_{model}']
            rows.extend(metric_rows(strategy,model,y,p))
            for j,date in enumerate(dates):
                for h in range(1,8):
                    daily.append(dict(strategy=strategy,model=model,h=h,origin=str(date.date()),
                        target_date=str((date+pd.Timedelta(days=h)).date()),truth=y[j,h-1],prediction=p[j,h-1]))
    pd.DataFrame(rows).to_csv(run/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(daily).to_csv(run/'predictions.csv',index=False,encoding='utf-8-sig')
    baseline = np.tile(ds['X'][test,-1,ds['names'].index('Discharge_lag0')][:,None],(1,7))
    pd.DataFrame(metric_rows('baseline','Persistence',y,baseline)).to_csv(run/'baseline.csv',index=False)
    np.savez_compressed(run/'predictions.npz',truth=y,dates=dates.astype(str).to_numpy(),**all_predictions)
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(status='completed',n_test=len(test),
        actual_base_models=len(base_model_names()),actual_ensemble_models=len(ensemble_model_names()),
        best_h1_R2=max(row['R2'] for row in rows if row['h']==1),test_evaluated=True)
    save_json(run/'manifest.json',manifest)
    print('FORMAL COMPLETE',manifest['best_h1_R2'],flush=True)
