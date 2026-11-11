"""Fixed ordinary-GBDT and training-history policies; 2004 is never constructed."""
import argparse
import hashlib
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid

from config import DATA_PATH, ROOT, digest, save_json
from data import load_raw, development_splits
from exact_boosting import SqrtResidualGradientBoostingRegressor
from research_h1 import prepare_data, annual_splits, metrics


def history_rows(dates, policy):
    dates = pd.DatetimeIndex(dates)
    if not dates.is_monotonic_increasing or not dates.is_unique or not len(dates):
        raise ValueError('Training dates must be nonempty, unique and increasing')
    age = (dates[-1]-dates).days.to_numpy()
    if policy == 'recent730':
        keep = age < 730
        return keep, np.ones(int(keep.sum()))
    if policy == 'half_life365':
        weight = np.exp2(-age/365)
        return np.ones(len(dates), dtype=bool), weight/weight.mean()
    if policy == 'all':
        return np.ones(len(dates), dtype=bool), np.ones(len(dates))
    raise ValueError('Unknown history policy')


def run(output, raw=None, grid=None, n_estimators=200):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Output exists; inspect it instead of repeating the run')
    grid = grid if grid is not None else dict(max_leaf_nodes=[7, 15], min_samples_leaf=[10, 30])
    candidates = list(ParameterGrid(grid))
    if not candidates:
        raise ValueError('Empty parameter grid')
    data = prepare_data(load_raw() if raw is None else raw, horizons=7)
    X, names = data['mi']['X'][:, -1], data['mi']['names']
    y, dates = data['y'][:, 0], data['dates']
    assert names[0] == 'Discharge_lag0' and len(names) == 91
    assert dates.max()+pd.Timedelta(days=7) <= pd.Timestamp('2003-12-31')
    assert np.isfinite(X).all() and np.isfinite(y).all()
    folds = list(annual_splits(dates, horizons=7))
    output.mkdir(parents=True, exist_ok=False)
    (output/'models').mkdir()
    save_json(output/'protocol.json', dict(data_cutoff='2003-12-31', test_evaluations=0,
        data_sha256=digest(DATA_PATH) if raw is None else 'unit-test fixture',
        input_sha256=hashlib.sha256(X.tobytes()+y.tobytes()+dates.asi8.tobytes()).hexdigest(),
        source_sha256={p.name:digest(p) for p in (ROOT/'code').glob('*.py')},
        names=names, n_estimators=n_estimators, grid=grid, policies=['all','recent730','half_life365'],
        scoring='unweighted original-flow R2', outer='annual 2001/2002/2003',
        inner='TimeSeriesSplit(2), gap=7', shared_origins_per_year=358,
        model='GradientBoostingRegressor sqrt residual, learning_rate=0.05, seed=42, no early stopping'))
    model_args = dict(n_estimators=n_estimators, learning_rate=.05, loss='squared_error',
        subsample=1., max_features=None, max_depth=None, criterion='friedman_mse',
        n_iter_no_change=None, warm_start=False, random_state=42)
    cache, fits, cv_rows, rows, predictions, split_rows, caught = {}, [], [], [], [], [], []
    new_fits = dict(all=0, recent730=0, half_life365=0)
    for policy in new_fits:
        for year, tr, va in folds:
            inner = development_splits(dates[tr])
            split_rows.append(dict(policy=policy, year=year, kind='outer',
                train_end=dates[tr[-1]], train_label_end=dates[tr[-1]]+pd.Timedelta(days=7),
                validation_start=dates[va[0]]))
            for fold, (it, iv) in enumerate(inner):
                split_rows.append(dict(policy=policy, year=year, kind=f'inner{fold}',
                    train_end=dates[tr[it[-1]]], train_label_end=dates[tr[it[-1]]]+pd.Timedelta(days=7),
                    validation_start=dates[tr[iv[0]]]))

            def fitted(indices, params, stage):
                keep, weight = history_rows(dates[indices], policy)
                used = indices[keep]
                key = (tuple(used), tuple(sorted(params.items())), weight.tobytes())
                if key in cache:
                    estimator, source = cache[key]
                else:
                    estimator = SqrtResidualGradientBoostingRegressor(**model_args, **params)
                    with warnings.catch_warnings(record=True) as local:
                        warnings.simplefilter('always')
                        estimator.fit(X[used], y[used], sample_weight=weight)
                    caught.extend(dict(policy=policy, year=year, stage=stage,
                        category=w.category.__name__, message=str(w.message)) for w in local)
                    source = None
                    cache[key] = (estimator, f'{policy}/{year}/{stage}')
                    new_fits[policy] += 1
                fits.append(dict(policy=policy, year=year, stage=stage, params=params,
                    train_count=len(used), train_start=dates[used[0]], train_end=dates[used[-1]],
                    weight_min=float(weight.min()), weight_max=float(weight.max()),
                    weight_mean=float(weight.mean()), reused_from=source))
                return estimator

            ranked = []
            for params in candidates:
                scores = []
                for fold, (it, iv) in enumerate(inner):
                    model = fitted(tr[it], params, f'cv{fold}')
                    score = float(r2_score(y[tr[iv]], model.predict(X[tr[iv]])))
                    scores.append(score)
                    cv_rows.append(dict(policy=policy, year=year, fold=fold, params=params,
                        r2=score, train_count=len(it), validation_count=len(iv)))
                ranked.append((float(np.mean(scores)), params))
            best_score, best = max(ranked, key=lambda pair: pair[0])
            model = fitted(tr, best, 'outer_refit')
            path = output/'models'/f'{policy}_{year}.joblib'
            joblib.dump(model, path)
            prediction = model.predict(X[va])
            assert len(prediction) == len(va) == 358 and np.isfinite(prediction).all()
            threshold = float(data['raw'].Discharge.loc[:dates[tr[-1]]].quantile(.95))
            rows.append(dict(policy=policy, year=year, model='GBDT', cv_r2=best_score,
                params=best, threshold=threshold, **metrics(y[va], prediction, threshold)))
            for date, truth, value in zip(dates[va], y[va], prediction):
                predictions.append(dict(policy=policy, year=year, origin=date.date(),
                    target_date=(date+pd.Timedelta(days=1)).date(), truth=truth, prediction=value))
            pd.DataFrame(fits).to_csv(output/'fits.csv', index=False)
            pd.DataFrame(cv_rows).to_csv(output/'cv.csv', index=False)
            pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
            pd.DataFrame(predictions).to_csv(output/'predictions.csv', index=False)
            pd.DataFrame(split_rows).to_csv(output/'splits.csv', index=False)
            save_json(output/'warnings.json', caught)
            print(f'{policy} {year}: R2={rows[-1]["r2"]:.6f}, new fits={new_fits[policy]}', flush=True)
    save_json(output/'complete.json', dict(status='complete', new_fits_by_policy=new_fits,
        new_fits=sum(new_fits.values()), test_evaluations=0,
        model_sha256={p.name:digest(p) for p in (output/'models').glob('*.joblib')}))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
