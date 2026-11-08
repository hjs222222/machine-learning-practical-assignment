import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from models import RunoffRegressor


@pytest.mark.parametrize('value',[1,'yes',None])
def test_reject_nonboolean_bootstrap(value):
    with pytest.raises(ValueError,match='rf_bootstrap'):
        RunoffRegressor(rf_bootstrap=value).fit(np.ones((20,3,2)),np.ones((20,1)))


@pytest.mark.parametrize('family',['ANN','LSTM'])
def test_other_families_cannot_change_rf_sampling(family):
    with pytest.raises(ValueError,match='rf_bootstrap'):
        RunoffRegressor(family=family,rf_bootstrap=False).fit(np.ones((20,3,2)),np.ones((20,1)))


@pytest.mark.parametrize('outputs',[1,7])
def test_whole_training_fold_native_outputs_clone_and_restore(outputs,tmp_path):
    x=np.random.default_rng(6).uniform(1,10,(40,3,3)).astype('float32')
    y=np.column_stack([x[:,-1,0]+i for i in range(outputs)])
    model=clone(RunoffRegressor(rf_bootstrap=False,trees=3,leaf=2,max_features=.5,transform='sqrt')).fit(x,y)
    assert model.estimator_.bootstrap is False
    for indices,tree in zip(model.estimator_.estimators_samples_,model.estimator_.estimators_):
        np.testing.assert_array_equal(indices,np.arange(40))
        assert tree.tree_.n_node_samples[0]==40
    prediction=model.predict(x);assert prediction.shape==(40,outputs)
    path=tmp_path/'model.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(prediction,joblib.load(path).predict(x),atol=1e-12,rtol=1e-12)
    del model.rf_bootstrap
    np.testing.assert_allclose(prediction,model.predict(x),atol=1e-12,rtol=1e-12)


def test_default_and_explicit_true_sampling_unchanged():
    x=np.random.default_rng(6).uniform(1,10,(40,3,3)).astype('float32');y=x[:,-1,:1]
    a=RunoffRegressor(trees=3).fit(x,y);b=clone(a).set_params(rf_bootstrap=True).fit(x,y)
    np.testing.assert_allclose(a.predict(x),b.predict(x),atol=1e-9,rtol=0)
    for left,right in zip(a.estimator_.estimators_samples_,b.estimator_.estimators_samples_):
        np.testing.assert_array_equal(left,right)
        assert len(np.unique(left))<40


def test_research_actual_sampling_log(tmp_path):
    from research_h1 import run_stage
    dates=pd.date_range('2000-01-01','2004-12-31')
    raw=pd.DataFrame({c:10+np.sin(np.arange(len(dates))/13)+np.arange(len(dates))*.001
        for c in ['Discharge','Dayl','Prcp','Srad','Swe','Tmax','Tmin','Vp']},index=dates)
    candidates=[dict(id=str(bootstrap),family='RF',mode='climate',transform='sqrt',residual=False,
        params=dict(trees=3,rf_bootstrap=bootstrap)) for bootstrap in (True,False)]
    out=tmp_path/'study';run_stage('features',out,raw=raw,candidates=candidates)
    for row in json.loads((out/'fits.json').read_text()):
        n={2001:300,2002:665,2003:1030}[row['year']]
        assert row['rf_bootstrap']==(row['candidate']=='True')
        assert row['rf_first_tree_sample_count']==n
        assert (row['rf_first_tree_unique_count']<n)==row['rf_bootstrap']
