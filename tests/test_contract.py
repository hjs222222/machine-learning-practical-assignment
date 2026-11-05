import numpy as np
import pandas as pd
import pytest

from config import DATA_PATH, new_run_dir
from data import load_raw, build_dataset, development_splits
from features import build_features


def test_run_rejects_alternate_data_and_existing_destination(tmp_path):
    with pytest.raises(ValueError):
        new_run_dir(tmp_path / 'wrong.csv', tmp_path / 'run', {})
    new_run_dir(DATA_PATH, tmp_path / 'run', {})
    with pytest.raises(ValueError):
        new_run_dir(DATA_PATH, tmp_path / 'run', {})


def test_raw_data_contract():
    raw = load_raw()
    assert raw.shape == (1827, 8)
    assert raw.index[0] == pd.Timestamp('2000-01-01')
    assert raw.index[-1] == pd.Timestamp('2004-12-31')
    assert not raw.isna().any().any()


def test_features_are_causal_and_season_is_not_quarter():
    raw = load_raw()
    first, catalog = build_features(raw)
    changed = raw.copy()
    changed.iloc[100:] *= 100
    second, _ = build_features(changed)
    pd.testing.assert_frame_equal(first.iloc[:100], second.iloc[:100])
    assert first.loc['2000-02-01', 'season_winter'] == 1
    assert first.loc['2000-03-01', 'season_spring'] == 1
    assert first['Discharge_lag1'].iloc[4] == raw.Discharge.iloc[3]
    assert not any(c.startswith('Swe') for c in first)
    assert len(catalog) == first.shape[1]


def test_labels_dates_and_purged_folds():
    raw = load_raw()
    ds = build_dataset(raw)
    train, test = ds['train'], ds['test']
    assert len(test) == 359
    assert ds['dates'][test[0]] == pd.Timestamp('2004-01-01')
    assert ds['dates'][test[-1]] == pd.Timestamp('2004-12-24')
    assert (ds['dates'][train] + pd.Timedelta(days=7)).max() <= pd.Timestamp('2003-12-31')
    t = ds['dates'][train[0]]
    np.testing.assert_allclose(ds['y'][train[0]], [raw.loc[t + pd.Timedelta(days=h), 'Discharge'] for h in range(1,8)])
    for ti, vi in development_splits(ds['dates'][train]):
        assert ds['dates'][train[ti[-1]]] + pd.Timedelta(days=7) < ds['dates'][train[vi[0]]]
        assert not np.intersect1d(train[vi], test).size
