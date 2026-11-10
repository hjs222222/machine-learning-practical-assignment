"""Audit stored RF direct/multi h1 research against raw teacher observations."""
import ast
import json

import numpy as np
import pandas as pd

from config import ROOT, DATA_PATH, digest, save_json


def main():
    parent = ROOT / 'output/teacher_selection_20261109'
    output = parent / 'direct_rf_seed42'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    protocol = read(output / 'protocol.json')
    assert protocol['family'] == 'RF' and protocol['outputs'] == 1
    assert protocol['sample_horizon'] == 7 and protocol['data_cutoff'] == '2003-12-31'
    assert protocol['grid'] == dict(mi_policy=['mean'], top_k=[12, 24, 36], leaf=[1, 5])
    assert protocol['data_sha256'] == digest(DATA_PATH)
    for name, sha in protocol['code_sha256'].items():
        assert digest(ROOT / 'code' / name) == sha
    selected = read(output / 'selected.json')
    for row in selected:
        cv = pd.read_csv(output / f"cv_{row['year']}.csv")
        assert len(cv) == 6 and np.isfinite(cv.mean_test_score).all()
        np.testing.assert_allclose(cv.mean_test_score, cv[['split0_test_score', 'split1_test_score']].mean(axis=1))
        best = cv.loc[cv.mean_test_score.idxmax()]
        assert ast.literal_eval(best.params) == row['params']
        np.testing.assert_allclose(best.mean_test_score, row['inner_r2'])
        assert len(row['features']) == row['params']['top_k']
    raw = pd.read_csv(DATA_PATH, parse_dates=['Date']).set_index('Date').loc[:'2003-12-31']
    comparisons = []
    hashes = {}
    for label, folder in [('direct', output), ('native7', parent / 'nested_rf')]:
        assert read(folder / 'complete.json')['status'] == 'complete'
        assert read(folder / 'protocol.json')['data_sha256'] == digest(DATA_PATH)
        splits = pd.read_csv(folder / 'splits.csv', parse_dates=['train_end', 'valid_start'])
        assert (splits.train_end + pd.Timedelta(days=7) < splits.valid_start).all()
        frame = pd.read_csv(folder / 'predictions.csv', parse_dates=['origin'])
        scores = pd.read_csv(folder / 'metrics.csv').query('horizon == 1')
        assert len(scores) == 6
        for row in scores.itertuples():
            group = frame.loc[(frame.year == row.year) & (frame.model == row.model)].sort_values('origin')
            expected = pd.date_range(f'{row.year}-01-01', f'{row.year}-12-24')
            assert len(group) == 358 and pd.DatetimeIndex(group.origin).equals(expected)
            truth = raw.Discharge.reindex(expected + pd.Timedelta(days=1)).to_numpy()
            error = group.prediction_h1.to_numpy() - truth
            assert np.isfinite(error).all()
            train_origins = pd.date_range('2000-02-29', pd.Timestamp(row.year, 1, 1)-pd.Timedelta(days=8))
            cutoff = np.quantile(raw.Discharge.reindex(train_origins+pd.Timedelta(days=1)), .95)
            peak = error[truth >= cutoff]
            values = dict(r2=1-(error**2).sum()/((truth-truth.mean())**2).sum(),
                          rmse=np.sqrt((error**2).mean()), mae=np.abs(error).mean(),
                          peak_rmse=np.sqrt((peak**2).mean()), peak_bias=peak.mean(), peak_count=len(peak))
            for name, value in values.items():
                np.testing.assert_allclose(value, getattr(row, name), rtol=1e-11, atol=1e-9)
            comparisons.append(dict(strategy=label, year=row.year, model=row.model, **values))
        hashes[label] = {name: digest(folder / name) for name in ['protocol.json', 'predictions.csv', 'metrics.csv', 'splits.csv']}
    result = pd.DataFrame(comparisons)
    result.to_csv(output / 'paired_h1_comparison.csv', index=False)
    save_json(output / 'verification.json', dict(status='passed', metric_rows=12,
              prediction_rows=4296, cv_candidates_per_year=6, total_training_fits=42,
              verifier_fit_calls=0, verifier_predict_calls=0, data_sha256=digest(DATA_PATH),
              sources=hashes, verifier_sha256=digest(__file__)))
    print(result[['strategy', 'year', 'model', 'r2', 'mae']].to_string(index=False))


if __name__ == '__main__':
    main()
