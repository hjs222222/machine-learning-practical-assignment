"""Verify training-year boosting evidence without fitting or opening test inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingRegressor

from config import ROOT, DATA_PATH, digest, save_json
from data import load_raw
from research_h1 import prepare_data, annual_splits


def raw_scores(truth, prediction, threshold):
    error = prediction-truth
    peak = error[truth >= threshold]
    return dict(r2=float(1-(error**2).sum()/((truth-truth.mean())**2).sum()),
                rmse=float(np.sqrt((error**2).mean())), mae=float(np.abs(error).mean()),
                peak_rmse=float(np.sqrt((peak**2).mean())), peak_bias=float(peak.mean()), peak_count=len(peak))


def main(output, source_ref=None):
    output = Path(output)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    protocol, complete = read(output/'protocol.json'), read(output/'complete.json')
    assert protocol['data_cutoff'] == '2003-12-31' and protocol['data_sha256'] == digest(DATA_PATH)
    assert complete['total_training_fits'] == 27 and complete['test_evaluations'] == 0
    assert complete['status'] == 'complete'
    assert protocol['estimator']['early_stopping'] is False
    source_commit = None
    if source_ref is not None:
        source_commit = subprocess.check_output(['git', 'rev-parse', '--verify', '--end-of-options',
            f'{source_ref}^{{commit}}'], cwd=ROOT, text=True).strip()
    for name, sha in protocol['code_hashes'].items():
        if source_commit is None:
            actual = digest(ROOT/name)
        else:
            blob = subprocess.check_output(['git', 'show', f'{source_commit}:{Path(name).as_posix()}'], cwd=ROOT)
            actual = hashlib.sha256(blob).hexdigest()
        assert actual == sha, name
    raw = load_raw().loc[:'2003-12-31']
    data = prepare_data(raw, horizons=7)
    X, y, dates = data['mi']['X'][:, -1], data['y'][:, 0], data['dates']
    assert hashlib.sha256(X.tobytes()+y.tobytes()+dates.asi8.tobytes()).hexdigest() == protocol['input_sha256']
    predictions = pd.read_csv(output/'predictions.csv', parse_dates=['origin', 'target_date'], float_precision='round_trip')
    assert len(predictions) == 2148
    table = pd.read_csv(output/'metrics.csv')
    fits = read(output/'fits.json')
    splits = pd.read_csv(output/'splits.csv', parse_dates=['train_end', 'valid_start'])
    assert len(splits) == 9 and (splits.train_end+pd.Timedelta(days=7) < splits.valid_start).all()
    comparisons = []
    for year, tr, va in annual_splits(dates, horizons=7):
        fit = next(f for f in fits if f['year'] == year)
        cv = pd.read_csv(output/f'cv_{year}.csv')
        assert len(cv) == 4
        np.testing.assert_allclose(cv.mean_test_score, cv[['split0_test_score', 'split1_test_score']].mean(axis=1))
        best = cv.loc[cv.mean_test_score.idxmax()]
        assert ast.literal_eval(best.params) == fit['params']
        np.testing.assert_allclose(best.mean_test_score, fit['inner_r2'])
        assert fit['train_count'] == len(tr) and fit['valid_count'] == len(va)
        path = output/fit['model_file']
        assert digest(path) == fit['model_sha256'] == complete['model_sha256'][path.name]
        model = joblib.load(path)
        assert type(model).__name__ == protocol.get('estimator_class', 'HistGradientBoostingRegressor')
        assert model.get_params() == dict(protocol['estimator'], **fit['params'])
        assert model.n_iter_ == fit['actual_iterations'] == 200 and model.do_early_stopping_ is False
        with threadpool_limits(limits=2):
            restored = np.maximum(model.predict(X[va]), 0)
            if protocol.get('target_mode') == 'sqrt_residual':
                residual = HistGradientBoostingRegressor.predict(model, X[va])
                manual = np.maximum(residual+np.sqrt(X[va, 0].astype(np.float64)), 0)**2
                np.testing.assert_array_equal(restored, manual)
            elif protocol.get('target_mode') == 'raw_residual':
                residual = HistGradientBoostingRegressor.predict(model, X[va])
                manual = np.maximum(residual+X[va, 0].astype(np.float64), 0)
                np.testing.assert_array_equal(restored, manual)
        threshold = float(raw.Discharge.loc[:dates[tr[-1]]].quantile(.95))
        for label in ('boost', 'persistence'):
            group = predictions.loc[(predictions.year == year) & (predictions.model == label)].sort_values('origin')
            assert pd.DatetimeIndex(group.origin).equals(dates[va])
            assert (group.target_date == group.origin+pd.Timedelta(days=1)).all()
            truth = raw.Discharge.reindex(group.target_date).to_numpy()
            np.testing.assert_array_equal(group.truth, truth)
            np.testing.assert_array_equal(group.prediction, restored if label == 'boost' else X[va, 0])
            assert (group.prediction >= 0).all()
            result = raw_scores(truth, group.prediction.to_numpy(), threshold)
            saved = table.loc[(table.year == year) & (table.model == label)].iloc[0]
            for name, value in result.items():
                np.testing.assert_allclose(value, saved[name], rtol=1e-11, atol=1e-9)
            comparisons.append(dict(year=year, model=label, **result))
    parent = output.parent
    for folder, candidate, origin_col, prediction_col, label in [
        (parent/'direct_rf_seed42', 'selected', 'origin', 'prediction_h1', 'direct_rf_selected'),
        (parent/'ann_depth_seed42', 'ANN_first_one', 'origin', 'prediction', 'ann_first_one')]:
        frame = pd.read_csv(folder/'predictions.csv', parse_dates=[origin_col])
        key = 'model' if label.startswith('direct') else 'candidate'
        for year, group in frame.loc[frame[key] == candidate].groupby('year'):
            group = group.sort_values(origin_col)
            expected = pd.date_range(f'{year}-01-01', f'{year}-12-24')
            assert pd.DatetimeIndex(group[origin_col]).equals(expected)
            truth = raw.Discharge.reindex(expected+pd.Timedelta(days=1)).to_numpy()
            threshold = float(raw.Discharge.loc[:pd.Timestamp(year, 1, 1)-pd.Timedelta(days=8)].quantile(.95))
            comparisons.append(dict(year=int(year), model=label,
                **raw_scores(truth, group[prediction_col].to_numpy(), threshold)))
    verification_file = 'verification.json' if source_commit is None else 'verification_source_review.json'
    comparison_file = 'paired_h1_comparison.csv' if source_commit is None else 'paired_h1_source_review.csv'
    pd.DataFrame(comparisons).to_csv(output/comparison_file, index=False)
    save_json(output/verification_file, dict(status='passed', metric_rows=6, prediction_rows=2148,
        restored_models=3, verifier_training_fits=0, test_evaluations=0,
        data_sha256=digest(DATA_PATH), predictions_sha256=digest(output/'predictions.csv'),
        verifier_sha256=digest(__file__), comparison_rows=len(comparisons), source_commit=source_commit))
    print(pd.DataFrame(comparisons)[['year', 'model', 'r2', 'rmse', 'mae']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-ref')
    main(**vars(parser.parse_args()))
