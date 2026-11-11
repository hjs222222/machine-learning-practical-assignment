"""Read-only reconstruction of the fixed training-period batch; never fits a model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid

from config import DATA_PATH, digest, save_json


def reconstructed_input(raw):
    raw=raw.loc[:'2003-12-31']
    columns={}
    for col in raw.columns:
        if col=='Swe':
            continue
        for lag in (0,1,2,3,7,14,30):
            columns[f'{col}_lag{lag}']=raw[col].shift(lag)
    for col in ('Discharge','Prcp','Tmax','Tmin'):
        for window in (3,7,14):
            for stat in ('mean','std','max'):
                columns[f'{col}_roll{window}_{stat}']=getattr(raw[col].rolling(window),stat)()
    angle=2*np.pi*(raw.index.month-1)/12
    columns['month_sin']=np.sin(angle)
    columns['month_cos']=np.cos(angle)
    season=(raw.index.month%12)//3
    for i,name in enumerate(('winter','spring','summer','autumn')):
        columns[f'season_{name}']=(season==i).astype(float)
    frame=pd.DataFrame(columns,index=raw.index)
    ready=frame.notna().all(axis=1).to_numpy() & raw.Prcp.shift(1).rolling(30).sum().notna().to_numpy()
    ids=np.array([i for i in range(29,len(raw)-7) if ready[i-29:i+1].all()])
    X=frame.iloc[ids].to_numpy(np.float32)
    y=raw.Discharge.shift(-1).iloc[ids].to_numpy()
    dates=raw.index[ids]
    return X,y,dates,list(frame.columns)


def verify(batch, state=None, raw=None):
    batch=Path(batch)
    teacher_data=raw is None
    protocol=json.loads((batch/'protocol.json').read_text(encoding='utf-8'))
    complete=json.loads((batch/'complete.json').read_text(encoding='utf-8'))
    if raw is None:
        raw=pd.read_csv(DATA_PATH,parse_dates=['Date']).set_index('Date')
        if protocol['data_sha256']!=digest(DATA_PATH):
            raise ValueError('Teacher data hash changed')
        for name,sha in protocol['source_sha256'].items():
            path=Path(__file__).with_name(name)
            if digest(path)!=sha:
                raise ValueError(f'Source hash changed: {name}')
    X,y,dates,names=reconstructed_input(raw)
    actual_hash=hashlib.sha256(X.tobytes()+y.tobytes()+dates.asi8.tobytes()).hexdigest()
    if names!=protocol['names'] or actual_hash!=protocol['input_sha256']:
        raise ValueError('Independently reconstructed input differs')
    if complete['status']!='complete' or complete['test_evaluations']!=0:
        raise ValueError('Batch is incomplete or evaluated 2004')
    predictions=pd.read_csv(batch/'predictions.csv',parse_dates=['origin','target_date'],float_precision='round_trip')
    recorded=pd.read_csv(batch/'metrics.csv')
    split=pd.read_csv(batch/'splits.csv',parse_dates=['train_end','train_label_end','validation_start'])
    fits=pd.read_csv(batch/'fits.csv')
    cv=pd.read_csv(batch/'cv.csv')
    if (len(predictions)!=3*3*358 or len(recorded)!=9 or len(split)!=27 or
            predictions.target_date.max()>=pd.Timestamp('2004-01-01') or
            not (split.train_label_end<split.validation_start).all()):
        raise ValueError('Prediction count, dates or temporal gap differ')
    if complete['new_fits']!=fits.reused_from.isna().sum() or complete['new_fits_by_policy']!=dict(
            fits.loc[fits.reused_from.isna()].groupby('policy',sort=False).size()):
        raise ValueError('Fit/reuse accounting differs')
    for path,sha in complete['model_sha256'].items():
        if digest(batch/'models'/path)!=sha:
            raise ValueError('Saved model hash differs')
    for policy in ('all','recent730','half_life365'):
        for year in (2001,2002,2003):
            group=predictions.loc[(predictions.policy==policy)&(predictions.year==year)].sort_values('origin')
            expected=pd.date_range(f'{year}-01-01',f'{year}-12-24')
            if (not pd.DatetimeIndex(group.origin).equals(expected) or
                    not pd.DatetimeIndex(group.target_date).equals(expected+pd.Timedelta(days=1))):
                raise ValueError('Annual prediction origins differ')
            truth=raw.Discharge.reindex(group.target_date).to_numpy()
            if not np.array_equal(group.truth.to_numpy(),truth):
                raise ValueError('Prediction truth differs from teacher CSV')
            model=joblib.load(batch/'models'/f'{policy}_{year}.joblib')
            x=X[dates.get_indexer(expected)]
            residual=GradientBoostingRegressor.predict(model,x)
            decoded=np.maximum(residual+np.sqrt(x[:,0].astype(np.float64)),0)**2
            np.testing.assert_allclose(decoded,group.prediction.to_numpy(),rtol=1e-10,atol=1e-7)
            row=recorded.loc[(recorded.policy==policy)&(recorded.year==year)].iloc[0]
            pred=group.prediction.to_numpy()
            for key,value in dict(r2=r2_score(truth,pred),rmse=np.sqrt(np.mean((pred-truth)**2)),
                                  mae=np.mean(abs(pred-truth))).items():
                if not np.isclose(row[key],value,rtol=1e-10,atol=1e-10):
                    raise ValueError(f'{policy}/{year} {key} differs')
            search=cv.loc[(cv.policy==policy)&(cv.year==year)].copy()
            ranked=[]
            for params in ParameterGrid(protocol['grid']):
                part=search.loc[search.params==str(params)]
                if len(part)!=2:
                    raise ValueError('Missing inner CV fold')
                ranked.append((part.r2.mean(),params))
            best=max(ranked,key=lambda item:item[0])
            if best[1]!=ast.literal_eval(row.params) or not np.isclose(best[0],row.cv_r2):
                raise ValueError('Inner CV selection differs')
    result=dict(status='passed', gbdt_predictions=len(predictions), gbdt_metric_rows=len(recorded),
        model_restores=len(complete['model_sha256']), new_fits=complete['new_fits'], test_evaluations=0)
    if state is not None:
        state=Path(state)
        gates=pd.read_csv(state/'predictions.csv',parse_dates=['origin','target_date'],float_precision='round_trip')
        records=pd.read_csv(state/'metrics.csv')
        state_complete=json.loads((state/'complete.json').read_text(encoding='utf-8'))
        state_protocol=json.loads((state/'protocol.json').read_text(encoding='utf-8'))
        if state_complete!=dict(status='complete',meta_fits=2,base_fits=0,test_evaluations=0):
            raise ValueError('State gate completion record differs')
        if state_protocol['data_sha256']!=digest(DATA_PATH) and teacher_data:
            raise ValueError('State gate source data differs')
        if len(gates)!=2*6*358 or len(records)!=12:
            raise ValueError('State gate output count differs')
        parent=Path(__file__).resolve().parents[1]/'output/teacher_selection_20261109'
        source_records=json.loads((state/'sources.json').read_text(encoding='utf-8'))
        for record in source_records:
            folder=Path(__file__).resolve().parents[1]/record['directory']
            if (digest(folder/'predictions.csv')!=record['predictions_sha256'] or
                    digest(folder/'protocol.json')!=record['protocol_sha256']):
                raise ValueError('State gate base source hash differs')
        ann=pd.read_csv(parent/'ann_depth_seed42/predictions.csv',parse_dates=['origin'],float_precision='round_trip')
        ann=ann.loc[ann.candidate=='ANN_first_one'].sort_values('origin')
        rf=pd.read_csv(parent/'direct_rf_seed42/predictions.csv',parse_dates=['origin'],float_precision='round_trip')
        rf=rf.loc[rf.model=='selected'].sort_values('origin')
        fit_records=json.loads((state/'fits.json').read_text(encoding='utf-8'))
        for year in (2002,2003):
            group=gates.loc[(gates.year==year)&(gates.model=='gate')].sort_values('origin')
            expected=pd.date_range(f'{year}-01-01',f'{year}-12-24')
            truth=raw.Discharge.reindex(group.target_date).to_numpy()
            if not pd.DatetimeIndex(group.origin).equals(expected) or not np.array_equal(group.truth,truth):
                raise ValueError('State gate date or truth differs')
            tree=joblib.load(state/f'tree_{year}.joblib').tree_
            q=raw.Discharge.reindex(expected).to_numpy()
            p=raw.Prcp.reindex(expected).to_numpy()
            context=np.column_stack([np.log1p(q),p,raw.Prcp.rolling(3).sum().reindex(expected),
                raw.Discharge.diff().reindex(expected),np.sin(2*np.pi*(expected.month-1)/12),
                np.cos(2*np.pi*(expected.month-1)/12)])
            selected=[]
            estimated=[]
            for row in context:
                node=0
                while tree.children_left[node]!=tree.children_right[node]:
                    node=tree.children_left[node] if row[tree.feature[node]]<=tree.threshold[node] else tree.children_right[node]
                selected.append(int(np.argmin(tree.value[node,:,0])))
                estimated.append(float(np.min(tree.value[node,:,0])))
            members=['ANN','RF','persistence']
            if list(group.selected)!=[members[i] for i in selected]:
                raise ValueError('Manual state tree traversal differs')
            np.testing.assert_allclose(group.estimated_loss.to_numpy(),estimated,rtol=1e-10,atol=1e-6)
            sources={name:gates.loc[(gates.year==year)&(gates.model==name)].sort_values('origin').prediction.to_numpy()
                for name in members}
            np.testing.assert_array_equal(sources['ANN'],ann.loc[ann.year==year].prediction.to_numpy())
            np.testing.assert_array_equal(sources['RF'],rf.loc[rf.year==year].prediction_h1.to_numpy())
            np.testing.assert_array_equal(sources['persistence'],q)
            expected_pred=np.array([sources[members[i]][j] for j,i in enumerate(selected)])
            np.testing.assert_allclose(expected_pred,group.prediction.to_numpy(),rtol=1e-12,atol=1e-8)
            mean=gates.loc[(gates.year==year)&(gates.model=='fixed_mean')].sort_values('origin').prediction.to_numpy()
            np.testing.assert_allclose(mean,np.column_stack(list(sources.values())).mean(axis=1),rtol=1e-12,atol=1e-8)
            history=ann.loc[ann.year<year]
            history_rf=rf.loc[rf.year<year]
            if not pd.DatetimeIndex(history.origin).equals(pd.DatetimeIndex(history_rf.origin)):
                raise ValueError('Historical members do not align')
            history_truth=raw.Discharge.reindex(history.origin+pd.Timedelta(days=1)).to_numpy()
            history_q=raw.Discharge.reindex(history.origin).to_numpy()
            prior=int(np.argmin(np.mean((np.column_stack([history.prediction.to_numpy(),
                history_rf.prediction_h1.to_numpy(),history_q])-history_truth[:,None])**2,axis=0)))
            prior_saved=gates.loc[(gates.year==year)&(gates.model=='prior_winner')].sort_values('origin').prediction.to_numpy()
            np.testing.assert_array_equal(prior_saved,sources[members[prior]])
            fit=next(item for item in fit_records if item['year']==year)
            if (fit['prior_winner']!=members[prior] or fit['tree_sha256']!=digest(state/f'tree_{year}.joblib') or
                    pd.Timestamp(fit['train_end'])+pd.Timedelta(days=7)>=pd.Timestamp(fit['validation_start'])):
                raise ValueError('State gate fit history differs')
            score=records.loc[(records.year==year)&(records.model=='gate')].iloc[0]
            if not np.isclose(score.r2,r2_score(truth,expected_pred),rtol=1e-10):
                raise ValueError('State gate metric differs')
            for name in ['gate','fixed_mean','prior_winner',*members]:
                part=gates.loc[(gates.year==year)&(gates.model==name)].sort_values('origin')
                metric=records.loc[(records.year==year)&(records.model==name)].iloc[0]
                err=part.prediction.to_numpy()-truth
                for key,value in dict(r2=r2_score(truth,part.prediction),
                    rmse=np.sqrt(np.mean(err**2)),mae=np.mean(abs(err))).items():
                    if not np.isclose(metric[key],value,rtol=1e-10,atol=1e-8):
                        raise ValueError(f'State gate {year}/{name} metric differs')
        result['state_predictions']=len(gates)
    selection=batch/'selection'
    if selection.exists():
        aligned=pd.read_csv(selection/'aligned_predictions.csv',parse_dates=['origin'],float_precision='round_trip')
        saved=json.loads((selection/'selection.json').read_text(encoding='utf-8'))
        expected=pd.DatetimeIndex([d for year in (2002,2003)
            for d in pd.date_range(f'{year}-01-01',f'{year}-12-24')])
        names=['ANN','GBDT/all','GBDT/recent730','GBDT/half_life365','state_gate']
        if len(expected)!=716 or len(aligned)!=5*716 or saved['comparison_days']!=716:
            raise ValueError('Unified comparison day count differs')
        source_hashes=dict(batch_predictions=digest(batch/'predictions.csv'),
            state_predictions=digest(state/'predictions.csv'),
            ann_predictions=digest(parent/'ann_depth_seed42/predictions.csv'))
        if saved['source_sha256']!=source_hashes:
            raise ValueError('Selection input hashes differ')
        recomputed=[]
        for name in names:
            part=aligned.loc[aligned.candidate==name].sort_values('origin')
            if not pd.DatetimeIndex(part.origin).equals(expected):
                raise ValueError('Selection candidate dates differ')
            truth=raw.Discharge.reindex(expected+pd.Timedelta(days=1)).to_numpy()
            if not np.array_equal(part.truth.to_numpy(),truth):
                raise ValueError('Selection truth differs')
            pred=part.prediction.to_numpy()
            record=dict(candidate=name,combined_r2=float(r2_score(truth,pred)),
                combined_mae=float(np.mean(abs(pred-truth))))
            for year in (2002,2003):
                mask=expected.year==year
                record[f'r2_{year}']=float(r2_score(truth[mask],pred[mask]))
            logged=next(row for row in saved['metrics'] if row['candidate']==name)
            for key in ('combined_r2','combined_mae','r2_2002','r2_2003'):
                if not np.isclose(record[key],logged[key],rtol=1e-10,atol=1e-8):
                    raise ValueError(f'Selection {name}/{key} differs')
            recomputed.append(record)
        base=recomputed[0]
        qualified=[row for row in recomputed[1:]
            if row['combined_r2']>=base['combined_r2']+.01 and
               row['r2_2003']>=base['r2_2003']-.005 and
               row['r2_2002']>=base['r2_2002']-.01 and
               row['combined_mae']<=base['combined_mae']*1.01]
        qualified.sort(key=lambda row:(-row['combined_r2'],-row['r2_2003'],
                                       row['combined_mae'],names.index(row['candidate'])))
        qualified=[row['candidate'] for row in qualified]
        if qualified!=saved['qualified'] or saved['winner']!=(qualified[0] if qualified else None):
            raise ValueError('Selection eligibility differs')
        result['selection_days']=len(expected)
        result['qualified']=qualified
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--state',type=Path)
    args=parser.parse_args()
    result=verify(args.batch,args.state)
    save_json(args.batch/'independent_verification.json',result)
    print(result)
