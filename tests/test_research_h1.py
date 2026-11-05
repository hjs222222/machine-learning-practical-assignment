import json
import numpy as np
import pandas as pd
import pytest
import research_h1 as rh
from models import RunoffRegressor


def raw_data():
    dates = pd.date_range('2000-01-01', '2004-12-31')
    n = len(dates)
    return pd.DataFrame({c: 10 + np.sin(np.arange(n)/13) + np.arange(n)*.001
                         for c in ['Discharge','Dayl','Prcp','Srad','Swe','Tmax','Tmin','Vp']}, index=dates)


def test_cutoff_causality_and_common_origins():
    raw = raw_data()
    a = rh.prepare_data(raw)
    changed = raw.copy()
    changed.loc['2003-06-01':] *= 99
    b = rh.prepare_data(changed)
    assert a['dates'].max() == pd.Timestamp('2003-12-30')
    for mode in rh.MODES:
        assert np.array_equal(a[mode]['dates'], a['dates'])
        before = a['dates'] < '2003-06-01'
        np.testing.assert_array_equal(a[mode]['X'][before], b[mode]['X'][before])
    assert a['all']['X'].shape[-1] == 91
    assert a['hydro']['names'][0] == 'Discharge_lag0'
    assert a['hydro']['X'].shape[-1] == 101
    only_2004 = raw.copy()
    only_2004.loc['2004-01-01':] = 1e8
    cut = rh.prepare_data(only_2004)
    np.testing.assert_array_equal(a['y'],cut['y'])
    for mode in rh.MODES:
        np.testing.assert_array_equal(a[mode]['X'],cut[mode]['X'])
    date = a['dates'][80]
    names = a['hydro']['names']
    values = a['hydro']['X'][80,-1]
    for days in (1,2,3,7):
        assert values[names.index(f'Discharge_diff{days}')] == pytest.approx(
            raw.Discharge.loc[date]-raw.Discharge.loc[date-pd.Timedelta(days=days)],abs=1e-6)
    for days in (2,3,5,7,14,30):
        assert values[names.index(f'Prcp_prior_sum{days}')] == pytest.approx(
            raw.Prcp.loc[date-pd.Timedelta(days=days):date-pd.Timedelta(days=1)].sum(),rel=1e-6)
    for year, tr, va in rh.annual_splits(a['dates']):
        assert a['dates'][tr[-1]] + pd.Timedelta(days=7) < a['dates'][va[0]]
        assert (a['dates'][va].year == year).all()
        assert ((a['dates'][va] + pd.Timedelta(days=1)).year == year).all()


def test_forced_selection_and_skip_mi(monkeypatch):
    x = np.random.default_rng(4).uniform(1, 10, (40, 3, 8))
    y = x[:, -1, :1]
    monkeypatch.setattr('models.mutual_info_regression', lambda *a, **k: np.arange(8))
    m = RunoffRegressor(top_k=2, forced_indices=(0,1), trees=2).fit(x,y)
    assert set(m.selected_) == {0,1,6,7}
    monkeypatch.setattr('models.mutual_info_regression', lambda *a, **k: pytest.fail('unnecessary MI'))
    assert len(RunoffRegressor(top_k=None, trees=2).fit(x,y).selected_) == 8


@pytest.mark.parametrize('transform', ['raw','sqrt','log'])
@pytest.mark.parametrize('residual', [False,True])
@pytest.mark.parametrize('mi_target', ['raw','transformed','model_target'])
def test_mi_receives_only_actual_fold_training_target(monkeypatch,transform,residual,mi_target):
    x = np.random.default_rng(12).uniform(1,15,(38,3,8)).astype('float32')
    y = np.column_stack([x[:,-1,0]*1.2+2,x[:,-1,0]*.7+1])
    observed = []
    def capture_mi(features,target,**kwargs):
        observed.append((features.copy(),target.copy()))
        return np.arange(features.shape[1])
    monkeypatch.setattr('models.mutual_info_regression',capture_mi)
    model = RunoffRegressor(trees=2,top_k=3,transform=transform,residual=residual,mi_target=mi_target)
    model.fit(x[:25],y[:25])
    target = model.target(y[:25])
    learning_target = target-model.target(x[:25,-1,:1]) if residual else target
    expected = y[:25] if mi_target=='raw' else target if mi_target=='transformed' else learning_target
    assert len(observed)==2
    for h,(features,label) in enumerate(observed):
        np.testing.assert_array_equal(features,x[:25,-1])
        np.testing.assert_allclose(label,expected[:,h],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(model.y_scaler_.mean_,learning_target.mean(axis=0),rtol=1e-6,atol=1e-6)
    assert model.predict(x[25:]).shape==(13,2)


def test_target_mi_research_mode_and_fit_metadata(tmp_path):
    data = rh.prepare_data(raw_data())
    np.testing.assert_array_equal(data['target_mi']['X'],data['mi']['X'])
    assert len(rh.stage_candidates('features'))==12
    candidates = rh.stage_candidates('features',families=['RF'],modes=['target_mi'])
    candidates[0]['params']={'trees':2}
    rh.run_stage('features',tmp_path/'target_mi',raw=raw_data(),candidates=candidates)
    fits=json.loads((tmp_path/'target_mi'/'fits.json').read_text())
    assert len(fits)==3
    for fit in fits:
        assert fit['mi_target']=='model_target'
        assert fit['selected_names'][0]=='Discharge_lag0'
        assert 24<=len(fit['selected_names'])<=25


def test_compact_columns_values_and_explicit_modes():
    raw = raw_data()
    data = rh.prepare_data(raw)
    compact = data['compact']
    names = ([f'{col}_lag{lag}' for col in ('Discharge','Prcp') for lag in (0,1,2,3,7)]
             + [f'Discharge_diff{d}' for d in (1,2,3,7)]
             + [f'Prcp_prior_sum{d}' for d in (2,3,5,7,14,30)] + ['month_sin','month_cos'])
    assert compact['names'] == names and len(names) == 22
    assert np.array_equal(compact['dates'],data['dates'])
    date = data['dates'][100]
    expected = ([raw.loc[date-pd.Timedelta(days=lag),col] for col in ('Discharge','Prcp') for lag in (0,1,2,3,7)]
                + [raw.Discharge.loc[date]-raw.Discharge.loc[date-pd.Timedelta(days=d)] for d in (1,2,3,7)]
                + [raw.Prcp.loc[date-pd.Timedelta(days=d):date-pd.Timedelta(days=1)].sum() for d in (2,3,5,7,14,30)]
                + [np.sin(2*np.pi*(date.month-1)/12),np.cos(2*np.pi*(date.month-1)/12)])
    np.testing.assert_allclose(compact['X'][100,-1],expected,rtol=1e-6,atol=1e-6)
    assert len(rh.stage_candidates('features')) == 12
    candidates = rh.stage_candidates('features',modes=['compact'])
    assert len(candidates) == 3 and {c['mode'] for c in candidates} == {'compact'}


@pytest.mark.parametrize('stage',['targets','tune'])
def test_nonfeature_modes_rejected_before_output(stage,tmp_path):
    with pytest.raises(ValueError,match='modes.*features'):
        rh.stage_candidates(stage,modes=['compact'])
    with pytest.raises(ValueError,match='modes.*features'):
        rh.run_stage(stage,tmp_path/'unused',modes=['compact'],candidates=[{}])
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('transform', ['raw','sqrt','log'])
@pytest.mark.parametrize('residual', [False,True])
def test_target_inverse(transform, residual):
    x = np.random.default_rng(7).uniform(1,10,(24,2,4))
    y = x[:,-1,:1] * 1.1
    model = RunoffRegressor(transform=transform,residual=residual,top_k=None,trees=2).fit(x,y)
    z = model.target(y) - (model.target(x[:,-1,:1]) if residual else 0)
    class Exact:
        def predict(self, values):
            return model.y_scaler_.transform(z)
    model.estimator_ = Exact()
    np.testing.assert_allclose(model.predict(x),y,rtol=1e-6)


def test_metrics_independently():
    y = np.array([1.,2.,3.,10.]); p = np.array([2.,2.,4.,8.])
    m = rh.metrics(y,p,5)
    assert m['r2'] == pytest.approx(1 - 6/np.sum((y-y.mean())**2))
    assert m['rmse'] == pytest.approx(np.sqrt(1.5))
    assert m['mae'] == 1
    assert m['peak_rmse'] == 2 and m['peak_bias'] == -2


def test_small_real_run_artifacts_and_stage_contract(tmp_path):
    candidate = dict(id='RF_all_raw',family='RF',mode='all',transform='raw',residual=False,params={'trees':2})
    out = tmp_path/'run'
    rh.run_stage('features',out,raw=raw_data(),candidates=[candidate],seed=9)
    protocol = json.loads((out/'protocol.json').read_text())
    assert protocol['seed'] == 9 and protocol['data_cutoff'] == '2003-12-31'
    assert protocol['code_hashes'] and protocol['data_sha256']
    folds = pd.read_csv(out/'fold_metrics.csv')
    predictions = pd.read_csv(out/'predictions.csv')
    assert len(folds) == 6 and set(folds.year) == {2001,2002,2003}
    assert predictions.groupby(['candidate','year']).size().min() > 300
    assert (out/'complete.json').exists() and (out/'warnings.json').exists()
    choices = rh.stage_candidates('targets',out,families=['RF'])
    assert len(choices) == 6 and {c['mode'] for c in choices} == {'all'}
    with pytest.raises(ValueError):
        rh.run_stage('features',out,raw=raw_data(),candidates=[candidate])
    with pytest.raises(ValueError):
        rh.stage_candidates('targets',tmp_path/'missing')
