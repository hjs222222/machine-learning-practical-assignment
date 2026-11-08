"""Training-only scalar offset study; never changes formal model predictions."""
import argparse
import json
import subprocess

import numpy as np
import pandas as pd


def training_offsets(frame):
    required=['candidate','year','origin','target_date','truth','prediction']
    if frame.empty or not set(required).issubset(frame.columns):
        raise ValueError('Missing or empty training records')
    frame=frame[required].copy()
    frame['origin']=pd.to_datetime(frame.origin)
    frame['target_date']=pd.to_datetime(frame.target_date)
    if frame.duplicated(['candidate','year','origin']).any():
        raise ValueError('Duplicate training origins')
    rows=[]
    for (candidate,year),part in frame.groupby(['candidate','year'],dropna=False):
        if year not in (2001,2002,2003) or pd.isna(candidate):
            raise ValueError('Invalid candidate or development year')
        part=part.sort_values('origin')
        expected=pd.date_range('2000-02-29',pd.Timestamp(int(year)-1,12,24))
        if (not np.array_equal(part.origin,expected) or
                not np.array_equal(part.target_date,expected+pd.Timedelta(days=1))):
            raise ValueError('Missing, future, or misaligned training dates')
        values=part[['truth','prediction']].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values<0).any():
            raise ValueError('Invalid original-scale truth/prediction')
        try:
            with np.errstate(over='raise',invalid='raise'):
                error=values[:,0]-values[:,1]
                offset=float(error.mean())
                before=float(np.mean(error**2))
                after=float(np.mean((error-offset)**2))
        except FloatingPointError as exc:
            raise ValueError('Nonfinite residual statistics') from exc
        rows.append(dict(candidate=candidate,year=int(year),n=len(part),offset=offset,
            train_origin_end=str(expected[-1].date()),train_target_end=str((expected[-1]+pd.Timedelta(days=1)).date()),
            mse_before=before,mse_after_unclipped=after))
    return pd.DataFrame(rows)


def apply_raw_offset(prediction,offset):
    if (isinstance(offset,(bool,np.bool_)) or not isinstance(offset,(int,float,np.integer,np.floating))
            or not np.isfinite(offset)):
        raise ValueError('Offset must be a finite real scalar')
    values=np.asarray(prediction,dtype=float)
    if not values.size or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Invalid original predictions')
    try:
        with np.errstate(over='raise',invalid='raise'):
            return np.maximum(values+offset,0)
    except FloatingPointError as exc:
        raise ValueError('Nonfinite corrected prediction') from exc


def main():
    from config import ROOT,DATA_PATH,digest,save_json
    from research_h1 import metrics
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(42,13,2026),default=42)
    seed=parser.parse_args().seed
    parent=ROOT/'output/h1_neural_20261106';output=parent/f'raw_offset_seed{seed}'
    if output.exists():raise ValueError('No overwrite or hot start')
    source=parent/f'ann_training_selection_seed{seed}'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    verification=json.loads((source/'verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='passed' and protocol['record_training']
    assert protocol['data_cutoff']=='2003-12-31' and protocol['gap']==7 and protocol['horizon']==1
    assert digest(DATA_PATH)=='ab09410b23118510bb4e8787a62b97693336a1c46477a01329f587c64bc57151'
    candidates=protocol['candidates']
    if seed!=42:
        gate=json.loads((parent/'raw_offset_seed42/promotion_gate.json').read_text(encoding='utf-8'))
        assert gate['selected_base'] is not None
        candidates=[c for c in candidates if c['id']==gate['selected_base']]
    assert len(candidates)==(3 if seed==42 else 1)
    ids=[c['id'] for c in candidates]
    raw=pd.read_csv(DATA_PATH,parse_dates=['Date'])
    raw=raw.loc[raw.Date<'2004-01-01'].set_index('Date')
    train=pd.read_csv(source/'training_predictions.csv',parse_dates=['origin','target_date'],float_precision='round_trip')
    train=train.loc[train.candidate.isin(ids)]
    np.testing.assert_array_equal(train.truth,raw.loc[train.target_date,'Discharge'])
    offsets=training_offsets(train)
    assert len(offsets)==3*len(candidates)
    output.mkdir()
    offsets.to_csv(output/'offsets.csv',index=False)
    save_json(output/'offsets.json',offsets.to_dict('records'))
    save_json(output/'protocol.json',dict(seed=seed,candidates=candidates,gap=7,horizon=1,data_cutoff='2003-12-31',
        source=str(source),source_hashes={name:digest(source/name) for name in
            ('protocol.json','verification.json','training_predictions.csv','predictions.csv')},
        offsets_sha256=digest(output/'offsets.json'),data_sha256=digest(DATA_PATH),
        code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'code').glob('*.py')},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        rule='b=mean(original train truth-prediction); max(validation prediction+b,0)',
        limitation='In-sample residual mean, not OOF or unbiased retransformation; no new ANN fit or 2004 score'))
    # Coefficients and their provenance are persisted before reading validation values.
    validation=pd.read_csv(source/'predictions.csv',float_precision='round_trip')
    parts=[]
    for candidate in ids:
        for year in (2001,2002,2003):
            frame=validation.loc[(validation.candidate==candidate)&(validation.year==year)].copy()
            b=float(offsets.loc[(offsets.candidate==candidate)&(offsets.year==year),'offset'].iloc[0])
            frame['base_prediction']=frame.prediction;frame['offset']=0.
            parts.append(frame)
            corrected=frame.copy();corrected['candidate']=candidate+'_raw_offset';corrected['offset']=b
            corrected['prediction']=apply_raw_offset(frame.prediction.to_numpy(),b);parts.append(corrected)
    persistence=validation.loc[validation.candidate=='persistence'].copy()
    persistence['base_prediction']=persistence.prediction;persistence['offset']=0.;parts.append(persistence)
    prediction=pd.concat(parts,ignore_index=True)
    scores=[dict(candidate=c,year=int(year),**metrics(frame.truth,frame.prediction,float(frame.threshold.iloc[0])))
            for (c,year),frame in prediction.groupby(['candidate','year'])]
    score_frame=pd.DataFrame(scores)
    summary=score_frame.groupby('candidate',as_index=False).agg(worst_r2=('r2','min'),mean_r2=('r2','mean'),
        mean_rmse=('rmse','mean'),mean_mae=('mae','mean'),mean_peak_rmse=('peak_rmse','mean'),mean_peak_bias=('peak_bias','mean'))
    prediction.to_csv(output/'predictions.csv',index=False);score_frame.to_csv(output/'fold_metrics.csv',index=False)
    summary.to_csv(output/'summary.csv',index=False)
    save_json(output/'complete.json',dict(status='complete',new_ann_fits=0,offset_estimates=len(offsets),
        predictions=len(prediction),source_training_warnings=verification['warnings']))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
