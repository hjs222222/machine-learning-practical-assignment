"""Nested training-period feature-count search; never evaluates 2004."""
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from config import DATA_PATH, digest, save_json
from data import load_raw, development_splits
from models import RunoffRegressor
from research_h1 import prepare_data, annual_splits, metrics


def h1_score(estimator, X, y):
    return r2_score(y[:, 0], estimator.predict(X)[:, 0])


def run(output, raw=None, grid=None, trees=120):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    grid = grid or dict(mi_policy=['mean', 'first'], top_k=[12, 24, 36], leaf=[1, 5])
    data = prepare_data(load_raw() if raw is None else raw, horizons=7)
    X, y, dates = data['mi']['X'], data['y'], data['dates']
    save_json(output/'protocol.json', dict(data_cutoff='2003-12-31', grid=grid, seed=42,
        family='RF', trees=trees, transform='sqrt', residual=True, seq_len=1,
        baseline=dict(mi_policy='mean',top_k=24,leaf=1), selection='inner h1 raw R2',
        outer='2001/2002/2003 annual', inner='TimeSeriesSplit(2), gap=7',
        data_sha256=digest(DATA_PATH) if raw is None else 'unit-test fixture',
        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        limitation='Repeated training-period development; not an independent blind test.'))
    scores, predictions, selected, splits, warning_rows = [], [], [], [], []
    for year, tr, va in annual_splits(dates, horizons=7):
        print(f'SEARCH year={year} train={len(tr)} valid={len(va)}', flush=True)
        inner = development_splits(dates[tr])
        splits.append(dict(year=year,kind='outer',train_end=dates[tr[-1]],valid_start=dates[va[0]]))
        for it, iv in inner:
            splits.append(dict(year=year,kind='inner',train_end=dates[tr[it[-1]]],valid_start=dates[tr[iv[0]]]))
        base = RunoffRegressor(family='RF',trees=trees,leaf=1,transform='sqrt',residual=True,seq_len=1)
        search = GridSearchCV(base, grid, cv=inner, scoring=h1_score, n_jobs=1,
                              error_score='raise', refit=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            search.fit(X[tr],y[tr])
            base.fit(X[tr],y[tr])
        warning_rows.extend(dict(year=year,category=w.category.__name__,message=str(w.message)) for w in caught)
        pd.DataFrame(search.cv_results_).to_csv(output/f'cv_{year}.csv',index=False)
        selected.append(dict(year=year,params=search.best_params_,inner_r2=float(search.best_score_),
            features=[data['mi']['names'][i] for i in search.best_estimator_.selected_],
            train_count=len(tr),valid_count=len(va)))
        threshold = float(np.quantile(y[tr,0],.95))
        for name, model in [('selected',search.best_estimator_),('baseline',base)]:
            pred = model.predict(X[va])
            for h in range(7):
                scores.append(dict(year=year,model=name,horizon=h+1,
                    **metrics(y[va,h],pred[:,h],threshold)))
            predictions.extend(dict(year=year,model=name,origin=str(date.date()),
                **{f'prediction_h{h+1}':float(vector[h]) for h in range(7)})
                for date,vector in zip(dates[va],pred))
        pd.DataFrame(scores).to_csv(output/'metrics.csv',index=False)
        pd.DataFrame(predictions).to_csv(output/'predictions.csv',index=False)
        pd.DataFrame(splits).to_csv(output/'splits.csv',index=False)
        save_json(output/'selected.json',selected)
        save_json(output/'warnings.json',warning_rows)
    save_json(output/'complete.json',dict(status='complete',outer_fits=6))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(**vars(parser.parse_args()))
