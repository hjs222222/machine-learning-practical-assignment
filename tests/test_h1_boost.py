"""Synthetic arrays below are software fixtures, not coursework observations."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from test_weather_target_mi import raw_data


def test_clipped_score_matches_prediction_policy():
    import research_h1_boost as boost
    assert boost.clipped_r2(np.array([0., 1., 2.]), np.array([-5., 1., 2.])) == 1.
    assert boost.clipped_r2(np.array([0., 1., 2.]), np.array([1., 1., 1.])) == 0.


@pytest.mark.parametrize('target,features', [('raw','all'), ('sqrt_residual','all'),
    ('raw_residual','all'), ('sqrt_residual','lag14')])
def test_training_only_grid_and_saved_predictions(tmp_path, target, features):
    import research_h1_boost as boost
    raw = raw_data()
    grid = dict(max_leaf_nodes=[3], min_samples_leaf=[5])
    first = tmp_path/'first'
    boost.run(first, raw=raw, grid=grid, max_iter=2, target=target, features=features)
    protocol = json.loads((first/'protocol.json').read_text())
    assert protocol['data_cutoff'] == '2003-12-31'
    assert protocol['sample_horizon'] == 7 and protocol['outputs'] == 1
    assert protocol['target_mode'] == target
    assert protocol['feature_mode'] == features
    assert len(protocol['input_features']) == (14 if features == 'lag14' else 91)
    assert protocol['estimator']['early_stopping'] is False
    assert protocol['estimator']['warm_start'] is False
    assert protocol['estimator']['loss'] == 'squared_error'
    assert protocol['estimator']['random_state'] == 42
    splits = pd.read_csv(first/'splits.csv', parse_dates=['train_end', 'valid_start'])
    assert len(splits) == 9
    assert (splits.train_end+pd.Timedelta(days=7) < splits.valid_start).all()
    predictions = pd.read_csv(first/'predictions.csv', parse_dates=['origin'], float_precision='round_trip')
    assert len(predictions) == 2148
    for year in (2001, 2002, 2003):
        group = predictions.query('year == @year and model == "boost"')
        assert pd.DatetimeIndex(group.origin).equals(pd.date_range(f'{year}-01-01', f'{year}-12-24'))
        np.testing.assert_array_equal(group.truth, raw.Discharge.reindex(group.origin+pd.Timedelta(days=1)))
        assert (group.prediction >= 0).all()
        score = pd.read_csv(first/'metrics.csv').query('year == @year and model == "boost"').iloc[0]
        assert score.r2 == pytest.approx(r2_score(group.truth, group.prediction))
        model = joblib.load(first/f'model_{year}.joblib')
        assert model.n_iter_ == 2 and model.do_early_stopping_ is False
        assert model.n_features_in_ == len(protocol['input_features'])
    assert json.loads((first/'complete.json').read_text())['total_training_fits'] == 9
    with pytest.raises(FileExistsError):
        boost.run(first, raw=raw, grid=grid, max_iter=2, target=target, features=features)
    changed = raw.copy()
    changed.loc['2004-01-01':] *= 100
    second = tmp_path/'second'
    boost.run(second, raw=changed, grid=grid, max_iter=2, target=target, features=features)
    assert json.loads((second/'protocol.json').read_text())['input_sha256'] == protocol['input_sha256']
    pd.testing.assert_frame_equal(pd.read_csv(first/'predictions.csv'), pd.read_csv(second/'predictions.csv'))


def test_residual_estimator_matches_manual_target_and_roundtrip(tmp_path):
    from boosting import SqrtResidualHistGradientBoostingRegressor
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    x = np.column_stack([np.arange(1., 81.), np.cos(np.arange(80))])
    y = (np.sqrt(x[:, 0]) + .2*x[:, 1])**2
    params = dict(max_iter=3, max_leaf_nodes=3, min_samples_leaf=5, early_stopping=False, random_state=42)
    with threadpool_limits(limits=2):
        wrapped = SqrtResidualHistGradientBoostingRegressor(**params).fit(x, y)
        manual = HistGradientBoostingRegressor(**params).fit(x, np.sqrt(y)-np.sqrt(x[:, 0]))
        expected = np.maximum(manual.predict(x)+np.sqrt(x[:, 0]), 0)**2
        np.testing.assert_array_equal(wrapped.predict(x), expected)
        assert clone(wrapped).get_params() == wrapped.get_params()
        path = tmp_path/'model.joblib'
        joblib.dump(wrapped, path)
        np.testing.assert_array_equal(joblib.load(path).predict(x), expected)
        with pytest.raises(ValueError, match='nonnegative'):
            wrapped.fit(x, -y)
        with pytest.raises(ValueError, match='nonnegative'):
            wrapped.predict(-x)
        with pytest.raises(ValueError, match='early_stopping'):
            SqrtResidualHistGradientBoostingRegressor(**(params | {'early_stopping':True})).fit(x, y)


def test_invalid_target_rejected_before_output_creation(tmp_path):
    import research_h1_boost as boost
    with pytest.raises(ValueError, match='target'):
        boost.run(tmp_path/'bad', target='unknown')
    assert not (tmp_path/'bad').exists()


def test_raw_residual_matches_manual_and_roundtrip(tmp_path):
    from boosting import RawResidualHistGradientBoostingRegressor
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    x = np.column_stack([np.arange(1., 81.), np.cos(np.arange(80))])
    y = np.maximum(x[:, 0] + 10*x[:, 1], 0)
    params = dict(max_iter=3, max_leaf_nodes=3, min_samples_leaf=5, early_stopping=False, random_state=42)
    with threadpool_limits(limits=2):
        wrapped = RawResidualHistGradientBoostingRegressor(**params).fit(x, y)
        manual = HistGradientBoostingRegressor(**params).fit(x, y-x[:, 0])
        expected = np.maximum(manual.predict(x)+x[:, 0], 0)
        np.testing.assert_array_equal(wrapped.predict(x), expected)
        assert clone(wrapped).get_params() == wrapped.get_params()
        path = tmp_path/'raw_model.joblib'
        joblib.dump(wrapped, path)
        np.testing.assert_array_equal(joblib.load(path).predict(x), expected)
        with pytest.raises(ValueError, match='nonnegative'):
            wrapped.fit(x, -np.ones(len(x)))
        with pytest.raises(ValueError, match='nonnegative'):
            wrapped.predict(-x)
        with pytest.raises(ValueError, match='early_stopping'):
            RawResidualHistGradientBoostingRegressor(**(params | {'early_stopping':True})).fit(x, y)
        zero_target = RawResidualHistGradientBoostingRegressor(**params).fit(x, np.zeros(len(x)))
        np.testing.assert_array_equal(zero_target.predict(np.zeros((2, 2))), [0., 0.])


def test_fixed_lag_inputs_are_exact_original_subset():
    from research_h1 import prepare_data
    from research_h1_boost import select_inputs
    data = prepare_data(raw_data(), horizons=7)
    original = data['mi']['X'][:, -1]
    all_x, all_names = select_inputs(data, 'all')
    np.testing.assert_array_equal(all_x, original)
    assert all_names == data['mi']['names']
    names = [f'{col}_lag{lag}' for col in ['Discharge', 'Prcp'] for lag in [0,1,2,3,7,14,30]]
    reduced, selected = select_inputs(data, 'lag14')
    assert selected == names and selected[0] == 'Discharge_lag0'
    np.testing.assert_array_equal(reduced, original[:, [all_names.index(n) for n in names]])


def test_invalid_feature_mode_rejected_before_output(tmp_path):
    import research_h1_boost as boost
    with pytest.raises(ValueError, match='features'):
        boost.run(tmp_path/'bad_features', features='unknown')
    assert not (tmp_path/'bad_features').exists()
