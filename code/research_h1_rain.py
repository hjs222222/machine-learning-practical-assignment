"""One predeclared h1 rain feature comparison using existing fold-local MI."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from config import ROOT, DATA_PATH, digest, save_json
from data import load_raw
from models import RunoffRegressor
from research_h1 import prepare_data, annual_splits, metrics


def run(output, feature='Prcp_roll14_max'):
    parent=ROOT/'output/teacher_selection_20261109'
    data=prepare_data(load_raw(),horizons=7)
    X=data['mi']['X'];y=data['y'];dates=data['dates'];names=data['mi']['names']
    rain=names.index(feature)
    sources=[];baselines={};baseline_fits={}
    for seed in (42,13,2026):
        folder=parent/f'ann_depth_seed{seed}'
        protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
        assert protocol['seed']==seed and protocol['data_cutoff']=='2003-12-31'
        assert protocol['data_sha256']==digest(DATA_PATH)
        baseline=pd.read_csv(folder/'predictions.csv',parse_dates=['origin'])
        baselines[seed]=baseline.loc[baseline.candidate=='ANN_first_one'].sort_values('origin')
        baseline_fits[seed]=[f for f in json.loads((folder/'fits.json').read_text(encoding='utf-8'))
                             if f['candidate']=='ANN_first_one']
        assert {f['year'] for f in baseline_fits[seed]}=={2001,2002,2003}
        sources.append(dict(seed=seed,source=str(folder.relative_to(ROOT)),
            predictions_sha256=digest(folder/'predictions.csv'),protocol_sha256=digest(folder/'protocol.json'),
            fits_sha256=digest(folder/'fits.json')))
    planned_fits=sum(feature not in f['selected_names'] for fits in baseline_fits.values() for f in fits)
    if planned_fits==0:
        raise ValueError('全部验证折已包含该特征，没有有效对照，禁止重复训练')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    save_json(output/'protocol.json',dict(seeds=[42,13,2026],data_cutoff='2003-12-31',
        data_sha256=digest(DATA_PATH),feature=feature,top_k=12,forced_union=True,
        model=dict(family='ANN',hidden=32,alpha=1.,iterations=350,seq_len=1,
                   transform='sqrt',residual=True,mi_policy='first',outputs=7),
        sources=sources,planned_base_fits=planned_fits,primary='h1 raw R2',gap=7,code_sha256=digest(__file__),
        limitation='Training-period exploratory feature hypothesis; not an independent blind test.'))
    rows=[];predictions=[];fits=[];warning_rows=[];actual_fits=0
    for seed in (42,13,2026):
        for year,tr,va in annual_splits(dates,horizons=7):
            baseline=baselines[seed].loc[lambda d:d.year==year]
            np.testing.assert_array_equal(baseline.origin,dates[va])
            np.testing.assert_array_equal(baseline.truth,y[va,0])
            original=next(f for f in baseline_fits[seed] if f['year']==year)
            added=rain not in original['selected_indices']
            expected=np.union1d(original['selected_indices'],[rain])
            print('RAIN seed',seed,'year',year,flush=True)
            if added:
                model=RunoffRegressor(family='ANN',hidden=32,alpha=1.,iterations=350,seq_len=1,
                    transform='sqrt',residual=True,mi_policy='first',top_k=12,forced_indices=(rain,),seed=seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');model.fit(X[tr],y[tr])
                warning_rows.extend(dict(seed=seed,year=year,message=str(w.message)) for w in caught)
                np.testing.assert_array_equal(model.selected_,expected)
                forced_prediction=model.predict(X[va])[:,0];actual_fits+=1
            else:
                forced_prediction=baseline.prediction.to_numpy()
            fits.append(dict(seed=seed,year=year,added=added,fit_performed=added,
                selected_names=[names[i] for i in expected],input_width=len(expected),
                train_end=dates[tr[-1]],valid_start=dates[va[0]]))
            threshold=float(data['raw'].Discharge.loc[:dates[tr[-1]]].quantile(.95))
            values={'baseline':baseline.prediction.to_numpy(),'forced_rain':forced_prediction}
            for name,pred in values.items():
                rows.append(dict(seed=seed,year=year,model=name,threshold=threshold,**metrics(y[va,0],pred,threshold)))
                predictions.extend(dict(seed=seed,year=year,model=name,origin=str(d.date()),
                    prediction=float(p),truth=float(t)) for d,p,t in zip(dates[va],pred,y[va,0]))
    pred_frame=pd.DataFrame(predictions)
    for (year,name),group in pred_frame.groupby(['year','model']):
        avg=group.groupby('origin',sort=True).prediction.mean()
        truth=group.drop_duplicates('origin').set_index('origin').truth.reindex(avg.index)
        threshold=next(row['threshold'] for row in rows if row['year']==year)
        rows.append(dict(seed='mean_3',year=year,model=name,threshold=threshold,
            **metrics(truth.to_numpy(),avg.to_numpy(),threshold)))
    pred_frame.to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    save_json(output/'fits.json',fits);save_json(output/'warnings.json',warning_rows)
    assert actual_fits==planned_fits
    save_json(output/'complete.json',dict(status='complete',base_fits=actual_fits))
    print(pd.DataFrame(rows)[['seed','year','model','r2','mae']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--feature',default='Prcp_roll14_max')
    run(**vars(parser.parse_args()))
