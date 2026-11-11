"""Synthetic fixtures test software only, never add course observations."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor

from test_weather_target_mi import raw_data


def test_exact_sqrt_residual_manual_and_restore(tmp_path):
    from exact_boosting import SqrtResidualGradientBoostingRegressor
    x = np.column_stack([np.arange(1.,81.), np.cos(np.arange(80))])
    y = (np.sqrt(x[:,0]) + .2*x[:,1])**2
    params = dict(n_estimators=3, max_leaf_nodes=3, min_samples_leaf=5,
                  max_depth=None, n_iter_no_change=None, random_state=42)
    m = SqrtResidualGradientBoostingRegressor(**params).fit(x,y)
    manual = GradientBoostingRegressor(**params).fit(x,np.sqrt(y)-np.sqrt(x[:,0]))
    expected = np.maximum(manual.predict(x)+np.sqrt(x[:,0]),0)**2
    np.testing.assert_array_equal(m.predict(x),expected)
    assert clone(m).get_params() == m.get_params()
    joblib.dump(m,tmp_path/'m.joblib')
    np.testing.assert_array_equal(joblib.load(tmp_path/'m.joblib').predict(x),expected)
    with pytest.raises(ValueError,match='nonnegative'):
        m.fit(x,-y)
    with pytest.raises(ValueError,match='nonnegative'):
        m.predict(-x)
    with pytest.raises(ValueError,match='early'):
        SqrtResidualGradientBoostingRegressor(**(params|{'n_iter_no_change':2})).fit(x,y)


def test_history_window_and_weights_use_fit_end():
    from research_h1_gbdt_batch import history_rows
    dates = pd.date_range('2000-01-01',periods=800)
    keep,w = history_rows(dates,'recent730')
    assert keep.sum() == 730 and np.flatnonzero(keep)[0] == 70
    np.testing.assert_array_equal(w,np.ones(730))
    keep,w = history_rows(dates,'half_life365')
    assert keep.all() and w.mean() == pytest.approx(1)
    assert w[-366]/w[-1] == pytest.approx(.5)
    _,short = history_rows(dates[:100],'half_life365')
    assert short.mean() == pytest.approx(1)
    assert short[-1]/short[-2] == pytest.approx(2**(1/365))
    with pytest.raises(ValueError):
        history_rows(dates[::-1],'all')


def test_gbdt_batch_cache_and_test_year_isolation(tmp_path):
    from research_h1_gbdt_batch import run
    raw = raw_data()
    options = dict(grid=dict(max_leaf_nodes=[3],min_samples_leaf=[5]),n_estimators=2)
    run(tmp_path/'first',raw=raw,**options)
    output=tmp_path/'first'
    complete=json.loads((output/'complete.json').read_text())
    assert complete['new_fits_by_policy'] == dict(all=9,recent730=1,half_life365=9)
    assert complete['new_fits'] == 19 and complete['test_evaluations']==0
    f=pd.read_csv(output/'predictions.csv',parse_dates=['origin','target_date'],float_precision='round_trip')
    assert len(f)==3222 and f.target_date.max()<=pd.Timestamp('2003-12-31')
    for (policy,year),group in f.groupby(['policy','year']):
        assert pd.DatetimeIndex(group.origin).equals(pd.date_range(f'{year}-01-01',f'{year}-12-24'))
        np.testing.assert_array_equal(group.truth,raw.Discharge.reindex(group.target_date))
        assert (group.prediction>=0).all()
    with pytest.raises(FileExistsError): run(output,raw=raw,**options)
    changed=raw.copy(); changed.loc['2004-01-01':]*=100
    run(tmp_path/'second',raw=changed,**options)
    a=json.loads((output/'protocol.json').read_text())
    b=json.loads((tmp_path/'second/protocol.json').read_text())
    assert a['input_sha256']==b['input_sha256']
    pd.testing.assert_frame_equal(f,pd.read_csv(tmp_path/'second/predictions.csv',parse_dates=['origin','target_date'],float_precision='round_trip'))


def test_state_gate_uses_only_origin_state_and_prior_oof(tmp_path):
    from research_h1_state_gate import assemble, evaluate
    raw=raw_data().loc[:'2003-12-31']
    dates=pd.DatetimeIndex([d for year in (2001,2002,2003)
        for d in pd.date_range(f'{year}-01-01',f'{year}-12-24')])
    truth=raw.Discharge.reindex(dates+pd.Timedelta(days=1)).to_numpy()
    ann=pd.DataFrame(dict(origin=dates,target_date=dates+pd.Timedelta(days=1),
                          truth=truth,prediction=truth))
    rf=pd.DataFrame(dict(origin=dates,prediction_h1=truth+2))
    frame=assemble(raw,ann,rf)
    assert len(frame)==1074
    np.testing.assert_array_equal(frame.truth,truth)
    np.testing.assert_array_equal(frame.persistence,raw.Discharge.reindex(dates))
    np.testing.assert_array_equal(frame.prcp,raw.Prcp.reindex(dates))
    np.testing.assert_array_equal(frame.rain3,raw.Prcp.rolling(3).sum().reindex(dates))
    np.testing.assert_array_equal(frame.qdiff,raw.Discharge.diff().reindex(dates))
    result=evaluate(frame,tmp_path/'gate',raw)
    assert result['meta_fits']==2 and result['test_evaluations']==0
    fits=json.loads((tmp_path/'gate/fits.json').read_text())
    assert [f['year'] for f in fits]==[2002,2003]
    assert all(pd.Timestamp(f['train_end'])+pd.Timedelta(days=7)<pd.Timestamp(f['validation_start']) for f in fits)
    pred=pd.read_csv(tmp_path/'gate/predictions.csv')
    assert set(pred.model)=={'gate','fixed_mean','prior_winner','ANN','RF','persistence'}
    assert pred.loc[pred.model=='gate','selected'].eq('ANN').all()
    bad=ann.copy();bad.loc[0,'target_date']=pd.Timestamp('2004-01-01')
    with pytest.raises(ValueError):assemble(raw,bad,rf)
    bad=ann.copy();bad.loc[0,'truth']+=1
    with pytest.raises(ValueError):assemble(raw,bad,rf)


def test_selection_requires_same_dates_and_predeclared_gates():
    from research_h1_batch_selection import choose
    dates=pd.date_range('2002-01-01',periods=20)
    truth=np.arange(20.)+10
    base=pd.DataFrame(dict(year=2002,origin=dates,truth=truth,prediction=truth+2,candidate='ANN'))
    lower=base.assign(candidate='GBDT/all',prediction=truth+1)
    result=choose(pd.concat([base,lower],ignore_index=True),expected_dates=dates,
                  required=('GBDT/all',))
    assert result['winner']=='GBDT/all' and result['qualified']==['GBDT/all']
    bad=lower.copy();bad.loc[0,'origin']=pd.Timestamp('2004-01-01')
    with pytest.raises(ValueError):choose(pd.concat([base,bad],ignore_index=True),
                                            expected_dates=dates,required=('GBDT/all',))


def test_independent_batch_verification_detects_changed_truth(tmp_path):
    from research_h1_gbdt_batch import run
    from verify_h1_generalization_batch import verify
    raw=raw_data()
    out=tmp_path/'batch'
    run(out,raw=raw,grid=dict(max_leaf_nodes=[3],min_samples_leaf=[5]),n_estimators=2)
    assert verify(out,raw=raw)['status']=='passed'
    path=out/'predictions.csv'
    frame=pd.read_csv(path)
    frame.loc[0,'truth']+=1
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError):verify(out,raw=raw)
