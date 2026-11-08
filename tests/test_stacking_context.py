import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def example():
    # Synthetic unit-test fixtures only; never used in scientific runs.
    dates = pd.date_range('2000-01-01', '2004-12-31')
    k = np.arange(len(dates))
    raw = pd.DataFrame({'Discharge': 100 + k % 31, 'Prcp': k % 7}, index=dates)
    origins = pd.DatetimeIndex([d for year in (2001, 2002, 2003)
                               for d in pd.date_range(f'{year}-01-01', f'{year}-12-30')])
    q = raw.loc[origins, 'Discharge'].to_numpy()
    source = pd.DataFrame({'origin': origins, 'target_date': origins + pd.Timedelta(days=1),
                           'year': origins.year, 'truth': raw.loc[origins + pd.Timedelta(days=1), 'Discharge'].to_numpy(),
                           'RF': q + np.sin(np.arange(len(q))), 'ANN': q * .9, 'LSTM': q * 1.1})
    return raw, source


@pytest.mark.parametrize('year,count', [(2002, 358), (2003, 722)])
@pytest.mark.parametrize('context,width', [(False, 3), (True, 5)])
def test_fold_exact_context_dates_and_gap(example, year, count, context, width):
    module = importlib.import_module('research_stacking_context')
    raw, source = example
    train, valid, x, v = module.context_fold(raw, source, year, context)
    assert x.shape == (count, width) and v.shape == (364, width)
    assert train.origin.max() == pd.Timestamp(year - 1, 12, 24)
    assert train.target_date.max() == pd.Timestamp(year - 1, 12, 25)
    assert valid.origin.min() == pd.Timestamp(year, 1, 1)
    np.testing.assert_array_equal(x[:, :3], train[['RF', 'ANN', 'LSTM']])
    if context:
        np.testing.assert_array_equal(x[:, 3:], raw.loc[train.origin, ['Discharge', 'Prcp']])
        np.testing.assert_array_equal(v[:, 3:], raw.loc[valid.origin, ['Discharge', 'Prcp']])


def test_future_and_2004_do_not_enter_training(example):
    module = importlib.import_module('research_stacking_context')
    raw, source = example
    original = module.context_fold(raw, source, 2002, True)
    changed = raw.copy()
    changed.loc['2004-01-01':] *= 1000
    replay = module.context_fold(changed, source, 2002, True)
    for a, b in zip(original, replay):
        np.testing.assert_array_equal(a, b)
    # Change a later year's original context and its corresponding truth consistently.
    changed = raw.copy()
    changed.loc['2003-01-01':] *= 10
    other = source.copy()
    other['truth'] = changed.loc[other.target_date, 'Discharge'].to_numpy()
    later = module.context_fold(changed, other, 2002, True)
    np.testing.assert_array_equal(original[2], later[2])
    np.testing.assert_array_equal(original[0].truth, later[0].truth)


@pytest.mark.parametrize('defect', ['duplicate', 'missing', 'date', 'truth', 'nan', 'year'])
def test_reject_bad_oof(example, defect):
    module = importlib.import_module('research_stacking_context')
    raw, source = example
    if defect == 'duplicate':
        source = pd.concat([source, source.iloc[[0]]])
    elif defect == 'missing':
        source = source.iloc[1:]
    elif defect == 'date':
        source.loc[0, 'target_date'] += pd.Timedelta(days=1)
    elif defect == 'truth':
        source.loc[0, 'truth'] += 1
    elif defect == 'nan':
        source.loc[0, 'RF'] = np.nan
    else:
        source.loc[0, 'year'] = 2002
    with pytest.raises(ValueError):
        module.context_fold(raw, source, 2002, True)


def test_existing_ridge_with_context_matches_independent_solution(example):
    module = importlib.import_module('research_stacking_context')
    from models import fit_stacking
    raw, source = example
    train, valid, x, v = module.context_fold(raw, source, 2003, True)
    model = fit_stacking([x], train.truth.to_numpy()[:, None])
    mu, sd = x.mean(0), x.std(0)
    z = (x - mu) / sd
    y = train.truth.to_numpy()
    coef = np.linalg.solve(z.T @ z + 10 * np.eye(5), z.T @ (y - y.mean()))
    np.testing.assert_allclose(model.predict(v).ravel(), ((v - mu) / sd) @ coef + y.mean(), atol=1e-9)
    assert model[0].n_samples_seen_ == 722
