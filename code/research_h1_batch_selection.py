"""Apply the predeclared two-year gates before any 2004 evaluation."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import ROOT, digest, save_json


CANDIDATES = ('GBDT/all', 'GBDT/recent730', 'GBDT/half_life365', 'state_gate')


def choose(frame, expected_dates=None, required=CANDIDATES):
    expected_dates = pd.DatetimeIndex(expected_dates if expected_dates is not None else
        [d for year in (2002, 2003) for d in pd.date_range(f'{year}-01-01', f'{year}-12-24')])
    if len(expected_dates)!=len(set(expected_dates)) or not expected_dates.is_monotonic_increasing:
        raise ValueError('Invalid comparison dates')
    names = ('ANN',)+tuple(required)
    if set(frame.candidate)!=set(names):
        raise ValueError('Missing or extra candidate')
    rows = []
    reference = None
    for name in names:
        part = frame.loc[frame.candidate==name].sort_values('origin').reset_index(drop=True)
        dates = pd.DatetimeIndex(part.origin)
        if not dates.equals(expected_dates) or not np.array_equal(part.year.to_numpy(), dates.year):
            raise ValueError('Missing, duplicate or mismatched candidate dates')
        if not np.isfinite(part[['truth','prediction']].to_numpy(float)).all():
            raise ValueError('Nonfinite truth or prediction')
        if reference is not None and not np.array_equal(part.truth.to_numpy(), reference):
            raise ValueError('Mismatched candidate truth')
        reference = part.truth.to_numpy()
        truth, pred = part.truth.to_numpy(float), part.prediction.to_numpy(float)
        record = dict(candidate=name, combined_r2=float(r2_score(truth,pred)),
            combined_mae=float(np.mean(np.abs(truth-pred))))
        for year in sorted(set(dates.year)):
            mask = dates.year==year
            record[f'r2_{year}']=float(r2_score(truth[mask],pred[mask]))
        rows.append(record)
    base = rows[0]
    for row in rows[1:]:
        row['qualified'] = (row['combined_r2']>=base['combined_r2']+.01 and
            row['combined_mae']<=base['combined_mae']*1.01 and
            all(row[f'r2_{year}']>=base[f'r2_{year}']-(.005 if year==2003 else .01)
                for year in sorted(set(expected_dates.year))))
    rows[0]['qualified']=False
    eligible = [row for row in rows[1:] if row['qualified']]
    eligible.sort(key=lambda row:(-row['combined_r2'],-row.get('r2_2003',-np.inf),
        row['combined_mae'],names.index(row['candidate'])))
    return dict(winner=eligible[0]['candidate'] if eligible else None,
        qualified=[row['candidate'] for row in eligible], metrics=rows,
        comparison_days=len(expected_dates), test_evaluations=0)


def run(output, batch, state):
    output, batch, state = Path(output), Path(batch), Path(state)
    if output.exists():
        raise FileExistsError('Selection output exists')
    b = pd.read_csv(batch/'predictions.csv',parse_dates=['origin'],float_precision='round_trip')
    s = pd.read_csv(state/'predictions.csv',parse_dates=['origin'],float_precision='round_trip')
    ann_path = ROOT/'output/teacher_selection_20261109/ann_depth_seed42/predictions.csv'
    a = pd.read_csv(ann_path,parse_dates=['origin'],float_precision='round_trip')
    ann = a.loc[a.candidate=='ANN_first_one',['year','origin','truth','prediction']].assign(candidate='ANN')
    ann = ann.loc[ann.year.isin((2002,2003))]
    boosted = b.loc[b.year.isin((2002,2003)),['policy','year','origin','truth','prediction']].copy()
    boosted['candidate']='GBDT/'+boosted.policy
    gate = s.loc[s.model=='gate',['year','origin','truth','prediction']].assign(candidate='state_gate')
    frame = pd.concat([ann,boosted.drop(columns='policy'),gate],ignore_index=True)
    result=choose(frame)
    output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(output/'aligned_predictions.csv',index=False)
    pd.DataFrame(result['metrics']).to_csv(output/'metrics.csv',index=False)
    save_json(output/'selection.json',result | dict(source_sha256={
        'batch_predictions':digest(batch/'predictions.csv'),
        'state_predictions':digest(state/'predictions.csv'),
        'ann_predictions':digest(ann_path)}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    args=parser.parse_args()
    print(run(args.output,args.batch,args.state))
