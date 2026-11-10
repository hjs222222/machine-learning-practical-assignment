"""Authorized supplementary boosting research; never constructs 2004 inputs."""
import argparse
import hashlib
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from threadpoolctl import threadpool_limits

from config import ROOT, DATA_PATH, digest, save_json
from data import load_raw, development_splits
from research_h1 import prepare_data, annual_splits, metrics
from boosting import SqrtResidualHistGradientBoostingRegressor


def clipped_r2(truth, prediction):
    return r2_score(truth, np.maximum(prediction, 0))


def run(output, raw=None, grid=None, max_iter=200, target='raw'):
    if target not in ('raw', 'sqrt_residual'):
        raise ValueError('target must be raw or sqrt_residual')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Output exists; inspect it instead of repeating the run')
    grid = grid if grid is not None else dict(max_leaf_nodes=[7, 15], min_samples_leaf=[10, 30])
    model_type = HistGradientBoostingRegressor if target == 'raw' else SqrtResidualHistGradientBoostingRegressor
    model = model_type(loss='squared_error', learning_rate=.05,
        max_iter=max_iter, l2_regularization=1., early_stopping=False, warm_start=False,
        categorical_features=None, random_state=42)
    data = prepare_data(load_raw() if raw is None else raw, horizons=7)
    X = data['mi']['X'][:, -1]
    y, dates = data['y'][:, 0], data['dates']
    assert dates.max()+pd.Timedelta(days=7) <= pd.Timestamp('2003-12-31')
    assert np.isfinite(X).all() and np.isfinite(y).all()
    folds = list(annual_splits(dates, horizons=7))
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'protocol.json', dict(data_cutoff='2003-12-31', outputs=1, sample_horizon=7,
        authorized_scope='Supplementary boosting; required ANN/RF/LSTM and ensembles unchanged; no 2004',
        estimator=model.get_params(), grid=grid, seed=42, input_features=data['mi']['names'],
        input_shape=X.shape, input_sha256=hashlib.sha256(X.tobytes()+y.tobytes()+dates.asi8.tobytes()).hexdigest(),
        data_sha256=digest(DATA_PATH) if raw is None else 'unit-test fixture',
        code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'code').glob('*.py')},
        target_mode=target, estimator_class=model_type.__name__,
        target='untransformed h1 flow' if target == 'raw' else 'sqrt(Q(t+1))-sqrt(Q(t)); decoded to original flow',
        prediction_policy='nonnegative original flow, including CV scoring',
        outer='2001/2002/2003 annual', inner='TimeSeriesSplit(2), gap7',
        limitation='Repeated development years; not a blind test or theoretical upper bound.'))
    rows, predictions, fits, splits, caught_rows = [], [], [], [], []
    for year, tr, va in folds:
        print(f'BOOST year={year} train={len(tr)} valid={len(va)}', flush=True)
        inner = development_splits(dates[tr])
        splits.append(dict(year=year, kind='outer', train_end=dates[tr[-1]], valid_start=dates[va[0]]))
        for it, iv in inner:
            splits.append(dict(year=year, kind='inner', train_end=dates[tr[it[-1]]], valid_start=dates[tr[iv[0]]]))
        pd.DataFrame(splits).to_csv(output/'splits.csv', index=False)
        search = GridSearchCV(model, grid, cv=inner, scoring=make_scorer(clipped_r2),
                              n_jobs=1, refit=True, error_score='raise')
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            warnings.simplefilter('always')
            search.fit(X[tr], y[tr])
            prediction = np.maximum(search.predict(X[va]), 0)
        assert prediction.shape == (len(va),) and np.isfinite(prediction).all()
        assert search.best_estimator_.do_early_stopping_ is False
        pd.DataFrame(search.cv_results_).to_csv(output/f'cv_{year}.csv', index=False)
        path = output/f'model_{year}.joblib'
        joblib.dump(search.best_estimator_, path)
        fits.append(dict(year=year, params=search.best_params_, inner_r2=search.best_score_,
            actual_iterations=search.best_estimator_.n_iter_, train_count=len(tr), valid_count=len(va),
            train_end=dates[tr[-1]], train_h1_label_end=dates[tr[-1]]+pd.Timedelta(days=1),
            model_file=path.name, model_sha256=digest(path)))
        caught_rows.extend(dict(year=year, category=w.category.__name__, message=str(w.message)) for w in caught)
        threshold = float(data['raw'].Discharge.loc[:dates[tr[-1]]].quantile(.95))
        for name, pred in [('boost', prediction), ('persistence', X[va, 0])]:
            rows.append(dict(year=year, model=name, threshold=threshold, **metrics(y[va], pred, threshold)))
            predictions.extend(dict(year=year, model=name, origin=str(date.date()),
                target_date=str((date+pd.Timedelta(days=1)).date()), truth=float(truth), prediction=float(value))
                for date, truth, value in zip(dates[va], y[va], pred))
        pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
        pd.DataFrame(predictions).to_csv(output/'predictions.csv', index=False)
        save_json(output/'fits.json', fits)
        save_json(output/'warnings.json', caught_rows)
    save_json(output/'complete.json', dict(status='complete', total_training_fits=3*(2*len(ParameterGrid(grid))+1),
        outer_refits=3, test_evaluations=0, model_sha256={f['model_file']:f['model_sha256'] for f in fits}))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', choices=['raw', 'sqrt_residual'], default='raw')
    print(run(**vars(parser.parse_args())).to_string(index=False))
