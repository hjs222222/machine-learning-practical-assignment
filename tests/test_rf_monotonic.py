import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from models import RunoffRegressor


@pytest.mark.parametrize('value',[1,'yes',None])
def test_monotonic_strict_bool(value):
    with pytest.raises(ValueError,match='rf_monotonic_flow'):
        RunoffRegressor(rf_monotonic_flow=value).fit(np.ones((20,3,3)),np.ones((20,1)))


@pytest.mark.parametrize('family,outputs',[('ANN',1),('LSTM',1),('RF',7)])
def test_reject_nonrf_or_native_multioutput_constraint(family,outputs):
    with pytest.raises(ValueError,match='rf_monotonic_flow'):
        RunoffRegressor(family=family,rf_monotonic_flow=True).fit(np.ones((20,3,3)),np.ones((20,outputs)))


def test_constraint_rejects_missing_current_flow_after_selection():
    x=np.random.default_rng(7).uniform(1,10,(100,3,3)).astype('float32');x[:,:,0]=1
    with pytest.raises(ValueError,match='rf_monotonic_flow'):
        RunoffRegressor(rf_monotonic_flow=True,top_k=1,trees=2).fit(x,x[:,-1,2:3]*2)


def test_actual_constraint_raw_predictions_and_restore(tmp_path):
    x=np.random.default_rng(7).uniform(1,10,(100,3,3)).astype('float32')
    y=20-x[:,-1,:1]+np.sin(x[:,-1,1:2])
    model=clone(RunoffRegressor(rf_monotonic_flow=True,top_k=None,trees=5,leaf=2,transform='sqrt')).fit(x,y)
    assert model.estimator_.monotonic_cst==[1,0,0]
    grid=np.repeat(x[:1],50,axis=0);grid[:,-1,0]=np.linspace(1,10,50)
    prediction=model.predict(grid)
    assert np.all(np.diff(prediction[:,0])>=-1e-9)
    path=tmp_path/'model.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(prediction,joblib.load(path).predict(grid),atol=1e-9,rtol=0)
    del model.rf_monotonic_flow
    np.testing.assert_allclose(prediction,model.predict(grid),atol=1e-9,rtol=0)


def test_default_false_and_native_seven_remain_unchanged():
    x=np.random.default_rng(7).uniform(1,10,(30,3,3)).astype('float32')
    y=np.column_stack([x[:,-1,0]+i for i in range(7)])
    a=RunoffRegressor(trees=3).fit(x,y)
    b=clone(a).set_params(rf_monotonic_flow=False).fit(x,y)
    assert a.estimator_.n_outputs_==7 and a.estimator_.monotonic_cst is None
    np.testing.assert_allclose(a.predict(x),b.predict(x),atol=1e-9,rtol=0)


def test_research_logs_real_constraint(tmp_path):
    from research_h1 import run_stage
    dates=pd.date_range('2000-01-01','2004-12-31')
    raw=pd.DataFrame({c:10+np.sin(np.arange(len(dates))/13)+np.arange(len(dates))*.001
        for c in ['Discharge','Dayl','Prcp','Srad','Swe','Tmax','Tmin','Vp']},index=dates)
    candidates=[dict(id=str(flag),family='RF',mode='climate',transform='sqrt',residual=False,
        params=dict(trees=2,rf_monotonic_flow=flag)) for flag in (False,True)]
    out=tmp_path/'study';run_stage('features',out,raw=raw,candidates=candidates)
    for row in json.loads((out/'fits.json').read_text()):
        expected=[1]+[0]*38 if row['candidate']=='True' else None
        assert row['rf_monotonic_cst']==expected
