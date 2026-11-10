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


def test_training_only_grid_and_saved_predictions(tmp_path):
    import research_h1_boost as boost
    raw = raw_data()
    grid = dict(max_leaf_nodes=[3], min_samples_leaf=[5])
    first = tmp_path/'first'
    boost.run(first, raw=raw, grid=grid, max_iter=2)
    protocol = json.loads((first/'protocol.json').read_text())
    assert protocol['data_cutoff'] == '2003-12-31'
    assert protocol['sample_horizon'] == 7 and protocol['outputs'] == 1
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
    assert json.loads((first/'complete.json').read_text())['total_training_fits'] == 9
    with pytest.raises(FileExistsError):
        boost.run(first, raw=raw, grid=grid, max_iter=2)
    changed = raw.copy()
    changed.loc['2004-01-01':] *= 100
    second = tmp_path/'second'
    boost.run(second, raw=changed, grid=grid, max_iter=2)
    assert json.loads((second/'protocol.json').read_text())['input_sha256'] == protocol['input_sha256']
    pd.testing.assert_frame_equal(pd.read_csv(first/'predictions.csv'), pd.read_csv(second/'predictions.csv'))
