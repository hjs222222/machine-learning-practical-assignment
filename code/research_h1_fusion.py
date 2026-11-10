"""Causal h1 fusion of saved training-period predictions, with no new base fits."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ROOT, DATA_PATH, digest, save_json
from data import load_raw, development_splits
from research_h1 import metrics

BASE = ['ANN','RF','LSTM']
ALPHAS = [.1,1.,10.,100.,1000.]


def assemble(raw, frames):
    expected = pd.DatetimeIndex([d for y in (2001,2002,2003)
                                for d in pd.date_range(f'{y}-01-01',f'{y}-12-24')])
    result = pd.DataFrame({'origin':expected})
    for family, source in zip(BASE,frames,strict=True):
        source = source.sort_values('origin')
        if not pd.DatetimeIndex(source.origin).equals(expected):
            raise ValueError('Missing, duplicate, future or mismatched OOF dates')
        result[family] = source.prediction.to_numpy()
    result['truth'] = raw.Discharge.reindex(expected+pd.Timedelta(days=1)).to_numpy()
    if not np.isfinite(result[BASE+['truth']]).all().all():
        raise ValueError('Nonfinite prediction or teacher truth')
    return result


def meta_split(frame, year):
    if year not in (2002,2003):
        raise ValueError('Only historical OOF evaluation years 2002/2003')
    valid = np.flatnonzero(frame.origin.dt.year==year)
    train = np.flatnonzero(frame.origin<frame.origin.iloc[valid[0]]-pd.Timedelta(days=7))
    return train,valid,development_splits(frame.origin.iloc[train])


def score(estimator, X, y):
    return r2_score(y,np.maximum(estimator.predict(X),0))


def run(output):
    parent = ROOT/'output/teacher_selection_20261109'
    sources = [(parent/'nested_ann','selected'),(parent/'nested_rf','selected'),
               (parent/'lstm_selection_seed42','LSTM_mean_k16')]
    frames=[]
    for folder,candidate in sources:
        import json
        protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
        complete=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        if (protocol['data_cutoff']!='2003-12-31' or protocol['seed']!=42 or
                protocol['data_sha256']!=digest(DATA_PATH) or complete['status']!='complete'):
            raise ValueError('Incompatible or incomplete base source')
        source=pd.read_csv(folder/'predictions.csv',parse_dates=['origin'])
        if 'model' in source:
            source=source.loc[source.model==candidate].rename(columns={'prediction_h1':'prediction'})
        else:
            source=source.loc[source.candidate==candidate]
        frames.append(source[['origin','prediction']])
    raw=load_raw().loc[:'2003-12-31']
    frame=assemble(raw,frames)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    save_json(output/'protocol.json',dict(data_cutoff='2003-12-31',data_sha256=digest(DATA_PATH),
        alphas=ALPHAS,inner='TimeSeriesSplit(2),gap7,h1 R2 after clipping',years=[2002,2003],
        inputs=BASE,base_fits=0,seed=42,selection='inner CV only',
        sources=[dict(directory=str(p.relative_to(ROOT)),candidate=c,
            predictions_sha256=digest(p/'predictions.csv'),protocol_sha256=digest(p/'protocol.json')) for p,c in sources],
        code_sha256=digest(__file__),
        limitation='Repeated training-period development; no independent blind generalization claim.'))
    X=frame[BASE].to_numpy();y=frame.truth.to_numpy()
    rows=[];predictions=[];fits=[];split_rows=[]
    for year in (2002,2003):
        train,valid,inner=meta_split(frame,year)
        for it,iv in inner:
            split_rows.append(dict(year=year,train_end=frame.origin.iloc[train[it[-1]]],
                valid_start=frame.origin.iloc[train[iv[0]]]))
        search=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),{'ridge__alpha':ALPHAS},
                            cv=inner,scoring=score,error_score='raise',n_jobs=1)
        search.fit(X[train],y[train])
        fixed=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(X[train],y[train])
        pd.DataFrame(search.cv_results_).to_csv(output/f'cv_{year}.csv',index=False)
        threshold=float(raw.loc[:frame.origin.iloc[train[-1]],'Discharge'].quantile(.95))
        for name,estimator in [('selected',search.best_estimator_),('fixed_alpha10',fixed)]:
            fits.append(dict(year=year,model=name,alpha=estimator[-1].alpha,
                train_count=len(train),train_end=frame.origin.iloc[train[-1]],valid_start=frame.origin.iloc[valid[0]],
                mean=estimator[0].mean_.tolist(),scale=estimator[0].scale_.tolist(),
                coef=estimator[-1].coef_.tolist(),intercept=float(estimator[-1].intercept_)))
        values={'selected':np.maximum(search.predict(X[valid]),0),
                'fixed_alpha10':np.maximum(fixed.predict(X[valid]),0),
                'simple_mean':X[valid].mean(axis=1),**{family:X[valid,i] for i,family in enumerate(BASE)}}
        for name,pred in values.items():
            rows.append(dict(year=year,model=name,**metrics(y[valid],pred,threshold)))
            predictions.extend(dict(year=year,model=name,origin=str(date.date()),prediction=float(p))
                               for date,p in zip(frame.origin.iloc[valid],pred))
    frame.to_csv(output/'inputs.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(split_rows).to_csv(output/'splits.csv',index=False)
    save_json(output/'fits.json',fits)
    save_json(output/'complete.json',dict(status='complete',meta_fits=24,base_fits=0))
    print(pd.DataFrame(rows)[['year','model','r2','rmse','mae']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(**vars(parser.parse_args()))
