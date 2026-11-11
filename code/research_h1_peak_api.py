"""A/B training-period controls: API/momentum features and peak weighting (h=1).

Only 2000-2003 teacher-CSV observations are used; 2004 is never constructed.
This mirrors research_selection_search.run's annual nested protocol and adds two
predeclared knobs: an API/momentum augmented feature set and peak-weighted fitting.
"""
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from config import DATA_PATH, ROOT, digest, save_json
from data import load_raw, development_splits
from features import build_features
from models import RunoffRegressor
from research_h1 import prepare_data, annual_splits, metrics

API_ALPHAS = (0.80, 0.85, 0.90, 0.95)
DIFF_DAYS = (1, 2, 3, 7)
PRIOR_SUM_DAYS = (2, 3, 7, 14, 30)


def augmented_features(raw):
    """Base 91 columns plus API, explicit flow momentum and prior-rain sums.

    All columns use only <=t observations; every new column is an exact superset
    addition to the original 91, none removed.
    """
    base, _ = build_features(raw)
    aug = base.copy()
    prcp = raw.Prcp.to_numpy()
    for a in API_ALPHAS:
        api = np.empty(len(prcp))
        state = 0.0
        for i, value in enumerate(prcp):
            state = a * state + value
            api[i] = state
        aug[f'API_a{a:g}'] = api
    for d in DIFF_DAYS:
        aug[f'Discharge_diff{d}'] = raw.Discharge.diff(d)
    for d in PRIOR_SUM_DAYS:
        aug[f'Prcp_prior_sum{d}'] = raw.Prcp.shift(1).rolling(d).sum()
    return aug, list(aug.columns)


def build_input(raw, features):
    """Return (X, names) for the mi (91) or api (104) feature set on shared origins."""
    source = load_raw() if raw is None else raw
    data = prepare_data(source, horizons=7)
    dates = data['mi']['dates']
    if features == 'mi':
        return data['mi']['X'], list(data['mi']['names'])
    if features != 'api':
        raise ValueError('features must be mi or api')
    frame, names = augmented_features(source)
    positions = source.index.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError('Shared origins missing from augmented frame index')
    X = np.stack([frame.iloc[p - 29:p + 1].to_numpy(np.float32) for p in positions])
    if not np.isfinite(X).all():
        raise ValueError('Augmented input contains non-finite values on shared origins')
    base_X = data['mi']['X']
    if X.shape[:2] != base_X.shape[:2] or not np.array_equal(X[:, :, :91], base_X):
        raise ValueError('Augmented input does not preserve the base 91 columns')
    return X, names


def h1_score(estimator, X, y):
    return r2_score(y[:, 0], estimator.predict(X)[:, 0])


def make_candidates():
    return [
        dict(id='RF_mi', family='RF', features='mi', peak_weight=1.0),
        dict(id='RF_api', family='RF', features='api', peak_weight=1.0),
        dict(id='RF_peak3', family='RF', features='mi', peak_weight=3.0),
        dict(id='RF_peak10', family='RF', features='mi', peak_weight=10.0),
        dict(id='ANN_mi', family='ANN', features='mi', peak_weight=1.0),
        dict(id='ANN_api', family='ANN', features='api', peak_weight=1.0),
        dict(id='ANN_torch1', family='ANN', features='mi', peak_weight=1.0, backend='torch'),
        dict(id='ANN_torch3', family='ANN', features='mi', peak_weight=3.0, backend='torch'),
        dict(id='ANN_torch10', family='ANN', features='mi', peak_weight=10.0, backend='torch'),
    ]


def run(output, raw=None, candidates=None, seed=42, trees=120):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Output exists; inspect it instead of repeating the run')
    candidates = candidates if candidates is not None else make_candidates()
    if not candidates or len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('Candidates must be nonempty and unique')
    outputs = 1
    raw = load_raw() if raw is None else raw
    data = prepare_data(raw, horizons=7)
    y, dates = data['y'][:, :outputs], data['dates']
    inputs = {}
    for features in sorted({c['features'] for c in candidates}):
        inputs[features] = build_input(raw, features)

    def grid_for(family):
        return dict(mi_policy=['mean'], top_k=[12, 24, 36],
                    **(dict(leaf=[1, 5]) if family == 'RF' else dict(alpha=[.1, 1.])))

    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'protocol.json', dict(
        data_cutoff='2003-12-31', test_evaluations=0, seed=seed, outputs=outputs, sample_horizon=7,
        gap=7, outer='2001/2002/2003 annual', inner='TimeSeriesSplit(2), gap=7',
        selection='inner h1 raw R2; each fold refits MI and scaling',
        scoring='unweighted original-flow R2',
        api_alphas=list(API_ALPHAS), diff_days=list(DIFF_DAYS), prior_sum_days=list(PRIOR_SUM_DAYS),
        augmented_columns=[n for n in inputs['api'][1] if n not in inputs['mi'][1]],
        base_columns=inputs['mi'][1], augmented_count=len(inputs['api'][1]),
        candidates=candidates,
        gate=dict(combined_r2_delta=0.01, r2_2003_floor=-0.005, r2_2002_floor=-0.01,
                  mae_ratio_max=1.01, comparison_days=716),
        controls={'A': {'RF_api': 'RF_mi', 'ANN_api': 'ANN_mi'},
                  'B': {'RF_peak3': 'RF_mi', 'RF_peak10': 'RF_mi',
                        'ANN_torch3': 'ANN_torch1', 'ANN_torch10': 'ANN_torch1'}},
        limitation='Repeated training-period development; not an independent blind test.',
        data_sha256=digest(DATA_PATH) if raw is not None else 'unit-test fixture',
        code_sha256={p.name: digest(p) for p in (ROOT / 'code').glob('*.py')}))

    metric_rows, prediction_rows, selected_rows, warning_rows, fit_rows = [], [], [], [], []
    for cand in candidates:
        X, names = inputs[cand['features']]
        settings = dict(family=cand['family'], seed=seed, trees=trees, leaf=1,
                        transform='sqrt', residual=True, seq_len=1, peak_weight=cand['peak_weight'])
        if cand['family'] == 'ANN':
            settings.update(hidden=32, iterations=350, alpha=.1)
            if cand['peak_weight'] != 1:
                settings['ann_backend'] = 'torch'
            elif cand.get('backend') == 'torch':
                settings['ann_backend'] = 'torch'
        grid = grid_for(cand['family'])
        for year, tr, va in annual_splits(dates, horizons=7):
            print(f"[{cand['id']}] year={year} train={len(tr)} valid={len(va)}", flush=True)
            inner = development_splits(dates[tr])
            base = RunoffRegressor(**settings)
            search = GridSearchCV(base, grid, cv=inner, scoring=h1_score, n_jobs=1,
                                  error_score='raise', refit=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                search.fit(X[tr], y[tr])
                base.fit(X[tr], y[tr])
            warning_rows.extend(dict(candidate=cand['id'], year=year, category=w.category.__name__,
                                     message=str(w.message)) for w in caught)
            selected_rows.append(dict(candidate=cand['id'], year=year,
                                      params=search.best_params_, inner_r2=float(search.best_score_),
                                      features=[names[i] for i in search.best_estimator_.selected_],
                                      train_count=len(tr), valid_count=len(va)))
            threshold = float(np.quantile(y[tr, 0], .95))
            for variant, model in [('selected', search.best_estimator_), ('baseline', base)]:
                pred = model.predict(X[va])
                metric_rows.append(dict(candidate=cand['id'], variant=variant, year=year,
                                        horizon=1, **metrics(y[va, 0], pred[:, 0], threshold)))
                prediction_rows.extend(dict(candidate=cand['id'], variant=variant, year=year,
                                            origin=str(date.date()), target_date=str((date + pd.Timedelta(days=1)).date()),
                                            truth=float(truth), prediction=float(value), threshold=threshold,
                                            peak=bool(truth >= threshold))
                                       for date, truth, value in zip(dates[va], y[va, 0], pred[:, 0]))
            fit_rows.append(dict(candidate=cand['id'], year=year, backend=settings.get('ann_backend', 'sklearn'),
                                 peak_weight=cand['peak_weight'], selected_indices=search.best_estimator_.selected_.tolist(),
                                 selected_names=[names[i] for i in search.best_estimator_.selected_]))
        pd.DataFrame(metric_rows).to_csv(output / 'metrics.csv', index=False)
        pd.DataFrame(prediction_rows).to_csv(output / 'predictions.csv', index=False)
        pd.DataFrame(selected_rows).to_csv(output / 'selected.csv', index=False)
        save_json(output / 'warnings.json', warning_rows)
        save_json(output / 'fits.json', fit_rows)

    comparison = evaluate_gates(pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows))
    save_json(output / 'comparison.json', comparison)
    save_json(output / 'complete.json', dict(status='complete', candidates=len(candidates),
                                             fits=len(fit_rows), test_evaluations=0))
    return comparison


def _combined(predictions, candidate, variant='selected'):
    part = predictions[(predictions.candidate == candidate) & (predictions.variant == variant)
                       & (predictions.year.isin([2002, 2003]))]
    if part.empty or len(part) != 716:
        raise ValueError(f'{candidate}/{variant} must have 716 comparison days')
    r2 = r2_score(part.truth, part.prediction)
    mae = float(np.mean(np.abs(part.prediction - part.truth)))
    per_year = {int(year): float(r2_score(g.truth, g.prediction))
                for year, g in part.groupby('year')}
    return dict(combined_r2=float(r2), combined_mae=mae, r2_2002=per_year[2002],
                r2_2003=per_year[2003])


def evaluate_gates(metric_rows, predictions):
    controls = {'RF_api': 'RF_mi', 'ANN_api': 'ANN_mi', 'RF_peak3': 'RF_mi', 'RF_peak10': 'RF_mi',
                'ANN_torch3': 'ANN_torch1', 'ANN_torch10': 'ANN_torch1'}
    present = set(predictions.candidate.unique())
    report = {'comparison_days': 716, 'test_evaluations': 0, 'entries': [], 'qualified': []}
    for candidate, control in controls.items():
        if candidate not in present or control not in present:
            continue
        cand = _combined(predictions, candidate)
        ref = _combined(predictions, control)
        delta_r2 = cand['combined_r2'] - ref['combined_r2']
        mae_ratio = cand['combined_mae'] / ref['combined_mae']
        qualified = (delta_r2 >= 0.01 and cand['r2_2003'] >= ref['r2_2003'] - 0.005
                     and cand['r2_2002'] >= ref['r2_2002'] - 0.01 and mae_ratio <= 1.01)
        report['entries'].append(dict(candidate=candidate, control=control, **cand,
                                      control_combined_r2=ref['combined_r2'],
                                      control_combined_mae=ref['combined_mae'],
                                      control_r2_2002=ref['r2_2002'], control_r2_2003=ref['r2_2003'],
                                      delta_r2=delta_r2, mae_ratio=mae_ratio, qualified=bool(qualified)))
        if qualified:
            report['qualified'].append(candidate)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    run(**vars(parser.parse_args()))
