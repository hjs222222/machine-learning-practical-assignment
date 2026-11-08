import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import mutual_info_regression

import research_h1 as rh
from models import RunoffRegressor


def raw_data():
    dates=pd.date_range('2000-01-01','2004-12-31')
    return pd.DataFrame({c:10+np.sin(np.arange(len(dates))/13)+np.arange(len(dates))*.001
        for c in ['Discharge','Dayl','Prcp','Srad','Swe','Tmax','Tmin','Vp']},index=dates)


@pytest.mark.parametrize('horizons',[1,7])
def test_alias_exact_and_actual_model_target_mi(horizons):
    data=rh.prepare_data(raw_data(),horizons=horizons)
    mode=data['weather_target_mi']
    assert mode['names']==data['weather']['names'] and len(mode['names'])==15
    np.testing.assert_array_equal(mode['X'],data['weather']['X'])
    np.testing.assert_array_equal(mode['dates'],data['dates'])
    x,y=mode['X'][:96],data['y'][:96]
    model=RunoffRegressor(family='RF',trees=2,top_k=8,forced_indices=(0,),mi_target='model_target',
                          transform='sqrt',residual=True,seq_len=1).fit(x,y)
    target=np.sqrt(y)-np.sqrt(x[:,-1,:1])
    score=np.mean([mutual_info_regression(x[:,-1],target[:,h],random_state=42) for h in range(horizons)],axis=0)
    np.testing.assert_array_equal(model.mi_,score)
    expected=np.union1d(np.argsort(score,kind='stable')[-8:],(0,))
    np.testing.assert_array_equal(model.selected_,expected)
    assert len(expected) in (8,9) and 0 in expected
    assert model.predict(x).shape==(96,horizons)


def test_mode_explicit_and_available_boundary_unchanged():
    assert 'weather_target_mi' in rh.MODES
    assert rh.DEFAULT_MODES==('mi','forced','all','hydro')
    available=rh.prepare_data(raw_data(),available_weather=True)
    assert 'weather_target_mi' not in available


def test_alias_no_future_or_2004_information():
    first=rh.prepare_data(raw_data())['weather_target_mi']
    raw=raw_data();raw.loc['2004-01-01':]*=1000
    second=rh.prepare_data(raw)['weather_target_mi']
    np.testing.assert_array_equal(first['X'],second['X'])
    raw.loc['2002-07-01':]*=10
    third=rh.prepare_data(raw)['weather_target_mi']
    past=first['dates']<'2002-07-01'
    np.testing.assert_array_equal(first['X'][past],third['X'][past])


def test_actual_research_mi_policy_and_logs(tmp_path):
    candidate=dict(id='MI',family='ANN',mode='weather_target_mi',transform='sqrt',residual=True,
        params=dict(hidden=2,iterations=1,alpha=1,seq_len=1))
    out=tmp_path/'mi'
    rh.run_stage('features',out,raw=raw_data(),candidates=[candidate])
    fits=json.loads((out/'fits.json').read_text())
    data=rh.prepare_data(raw_data())
    for fit,(year,tr,_) in zip(fits,rh.annual_splits(data['dates'])):
        x=data['weather']['X'][tr,-1]
        y=np.sqrt(data['y'][tr,0])-np.sqrt(x[:,0])
        score=mutual_info_regression(x,y,random_state=42)
        expected=np.union1d(np.argsort(score,kind='stable')[-8:],(0,))
        assert fit['mi_top_k']==8 and fit['mi_target']=='model_target'
        np.testing.assert_array_equal(fit['mi_scores'],score)
        np.testing.assert_array_equal(fit['selected_indices'],expected)
        assert fit['input_width']==len(expected)
