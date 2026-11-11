"""Synthetic fixtures only; never add course observations to results."""
import numpy as np
import pandas as pd
import pytest

from test_weather_target_mi import raw_data
from research_h1_two_stage import TwoStageRegressor, run


def _toy(n=240, seed=0):
    rng = np.random.RandomState(seed)
    q = np.linspace(300, 900, n, dtype=np.float32)
    rain = np.where(np.arange(n) % 3 == 0, 0.0, 9.0).astype(np.float32)
    X = np.stack([q, q * .9, rain, np.roll(rain, 1)], axis=1)[:, None, :]
    flow = (q + 40 * rain + rng.normal(0, 15, n)).astype(np.float32)
    return X.astype(np.float32), np.column_stack([flow, rain])


def test_two_stage_fits_and_predicts():
    X, y = _toy()
    model = TwoStageRegressor(trees=3, top_k=None, leaf=3, seed=42, min_regime=10).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X), 1) and np.isfinite(pred).all() and (pred >= 0).all()
    assert model.fallback_ == {'dry': False, 'wet': False}
    p = model.clf_.predict_proba(X[:, -1])[:, 1]
    assert p.min() >= 0 and p.max() <= 1


def test_regime_fallback_when_short():
    X, y = _toy()
    y[:, 1] = 9.0  # every target day wet -> dry regime too small
    model = TwoStageRegressor(trees=2, top_k=None, leaf=3, seed=42, min_regime=10).fit(X, y)
    assert model.fallback_['dry'] is True and model.fallback_['wet'] is False
    assert np.isfinite(model.predict(X)).all()


def test_rejects_wrong_target_shape():
    X, y = _toy()
    with pytest.raises(ValueError):
        TwoStageRegressor(trees=2, top_k=None, leaf=3).fit(X, y[:, :1])


def test_run_writes_no_2004_and_rejects_overwrite(tmp_path):
    raw = raw_data().copy()
    raw['Prcp'] = np.where(np.arange(len(raw)) % 3 == 0, 0.0, 8.0)
    out = tmp_path / 'two'
    comparison = run(out, raw=raw, trees=2)
    assert (out / 'complete.json').is_file()
    preds = pd.read_csv(out / 'predictions.csv')
    assert set(preds.candidate) == {'RF_baseline', 'TwoStage'}
    assert preds.target_date.max() <= '2003-12-31'
    assert comparison['test_evaluations'] == 0 and comparison['comparison_days'] == 716
    assert set(comparison['checks']) == {'combined_r2', 'r2_2003', 'r2_2002', 'mae', 'r2_2001'}
    assert comparison['qualified'] == (comparison['qualified_list'] == ['TwoStage'])
    regime = pd.read_csv(out / 'regime_metrics.csv')
    assert set(regime.regime) <= {'dry', 'wet'}
    with pytest.raises(FileExistsError):
        run(out, raw=raw, trees=2)
