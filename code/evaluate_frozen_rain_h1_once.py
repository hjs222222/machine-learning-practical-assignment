"""Authorized one-shot 2004 h1 backtest of the exact frozen rain ANN ensemble."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from config import ROOT,DATA_PATH,digest,save_json
from data import load_raw
from research_h1 import prepare_data

FROZEN_SHA='3c7b76564f5d7d8b63d787dc24aae2413a633a018e2d6253b03659c1f74ae03a'


def main():
    parent=ROOT/'output/teacher_selection_20261109/frozen_ann_rain3seed'
    output=parent/'evaluation_2004_h1_once'
    if output.exists():
        raise FileExistsError('This candidate evaluation already started; do not repeat predictions')
    assert digest(parent/'frozen.json')==FROZEN_SHA
    frozen=json.loads((parent/'frozen.json').read_text(encoding='utf-8'))
    protocol=json.loads((parent/'protocol.json').read_text(encoding='utf-8'))
    verification=json.loads((parent/'verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='passed' and digest(DATA_PATH)==protocol['data_sha256']
    assert digest(parent/'protocol.json')==frozen['protocol_sha256']
    assert protocol['seeds']==[42,13,2026] and protocol['weights']==[1/3]*3
    for name,sha in protocol['code_hashes'].items():assert digest(ROOT/'code'/name)==sha
    models=[]
    for entry in frozen['models']:
        path=parent/entry['model_file'];assert digest(path)==entry['model_sha256']
        model=joblib.load(path)
        assert model.seed==entry['seed'] and tuple(model.forced_indices)==(14,)
        assert len(model.selected_)==13 and model.outputs_==7
        models.append(model)
    output.mkdir(exist_ok=False)
    save_json(output/'authorization.json',dict(granted_by_user=True,
        user_reply='允许这个候选回溯一次2004',candidate_frozen_sha256=FROZEN_SHA,
        scope='one 2004 h1 ensemble backtest; saved weights only; no refit or tuning',
        supersedes_prior_pending_authorization=True))
    save_json(output/'started.json',dict(status='started',code_sha256=digest(__file__),
        frozen_sha256=FROZEN_SHA,fit_calls=0,planned_predict_calls=3))
    raw=load_raw();data=prepare_data(raw,cutoff='2004-12-31',horizons=7)
    dates=data['dates'];ids=np.flatnonzero((dates>='2004-01-01')&(dates<='2004-12-24'))
    expected=pd.date_range('2004-01-01','2004-12-24')
    assert pd.DatetimeIndex(dates[ids]).equals(expected) and len(ids)==359
    assert data['mi']['names'][14]=='Prcp_lag0'
    X=data['mi']['X'][ids]
    truth=raw.Discharge.reindex(expected+pd.Timedelta(days=1)).to_numpy()
    np.testing.assert_array_equal(truth,data['y'][ids,0])
    frame=pd.DataFrame({'origin':expected,'target_date':expected+pd.Timedelta(days=1),'truth':truth})
    columns=[]
    for model in models:
        prediction=np.asarray(model.predict(X),dtype='float64')
        assert prediction.shape==(359,7) and np.isfinite(prediction).all()
        col=f'seed{model.seed}_h1';frame[col]=prediction[:,0];columns.append(col)
    frame['prediction']=frame[columns].to_numpy().mean(axis=1)
    frame.to_csv(output/'predictions.csv',index=False)
    result=dict(R2=float(r2_score(truth,frame.prediction)),
        RMSE=float(np.sqrt(mean_squared_error(truth,frame.prediction))),
        MAE=float(mean_absolute_error(truth,frame.prediction)))
    save_json(output/'metrics.json',result)
    save_json(output/'complete.json',dict(status='complete',candidate_frozen_sha256=FROZEN_SHA,
        test_origins=359,horizon=1,fit_calls=0,predict_calls=3,
        predictions_sha256=digest(output/'predictions.csv'),data_sha256=digest(DATA_PATH),
        authorization_sha256=digest(output/'authorization.json'),
        limitation='Retrospective evaluation of an already exposed test year; no tuning based on this result'))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
