"""Synthetic fixtures only; never add course observations to results."""
import numpy as np
import pandas as pd
import pytest

from test_weather_target_mi import raw_data
from research_h1_peak_api import augmented_features, build_input, make_candidates, run
from research_h1 import prepare_data


def test_augmented_features_superset_and_alignment():
    raw = raw_data()
    frame, names = augmented_features(raw)
    assert len(names) == 104 and len(set(names)) == 104
    base = prepare_data(raw, horizons=7)['mi']['names']
    assert names[:91] == list(base)
    for col in ('API_a0.8', 'API_a0.85', 'API_a0.9', 'API_a0.95',
                'Discharge_diff1', 'Discharge_diff2', 'Discharge_diff3', 'Discharge_diff7',
                'Prcp_prior_sum2', 'Prcp_prior_sum7', 'Prcp_prior_sum14', 'Prcp_prior_sum30'):
        assert col in names
    X, api_names = build_input(raw, 'api')
    X_mi, mi_names = build_input(raw, 'mi')
    assert X.shape[:2] == X_mi.shape[:2]
    assert np.array_equal(X[:, :, :91], X_mi)
    assert np.isfinite(X).all()


def test_peak_weight_paths():
    rng = np.random.RandomState(0)
    X = rng.uniform(1, 5, (80, 1, 4)).astype(np.float32)
    y = (X[:, 0, 0] ** 2 + rng.uniform(0, 1, 80)).reshape(-1, 1).astype(np.float32)
    from models import RunoffRegressor
    rf = RunoffRegressor(family='RF', transform='raw', residual=False, top_k=None,
                         peak_weight=3.0, trees=3, seed=42).fit(X, y)
    assert rf.predict(X).shape == (80, 1)
    with pytest.raises(ValueError):
        RunoffRegressor(family='ANN', transform='raw', residual=False, top_k=None,
                        peak_weight=3.0, hidden=4, iterations=5, seed=42).fit(X, y)
    ann = RunoffRegressor(family='ANN', ann_backend='torch', transform='raw', residual=False,
                          top_k=None, peak_weight=3.0, hidden=4, iterations=5, seed=42).fit(X, y)
    assert ann.predict(X).shape == (80, 1)


def test_run_writes_isolates_and_rejects_overwrite(tmp_path):
    raw = raw_data()
    candidates = [c for c in make_candidates() if c['id'] in ('RF_mi', 'RF_api')]
    out = tmp_path / 'batch'
    comparison = run(out, raw=raw, candidates=candidates, trees=2)
    assert (out / 'complete.json').is_file() and (out / 'comparison.json').is_file()
    preds = pd.read_csv(out / 'predictions.csv')
    assert set(preds.candidate) == {'RF_mi', 'RF_api'}
    assert preds.target_date.max() <= '2003-12-31'
    assert comparison['test_evaluations'] == 0
    assert comparison['comparison_days'] == 716
    assert {e['candidate'] for e in comparison['entries']} == {'RF_api'}
    with pytest.raises(FileExistsError):
        run(out, raw=raw, candidates=candidates, trees=2)
