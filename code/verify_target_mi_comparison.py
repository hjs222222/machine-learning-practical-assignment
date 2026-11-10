"""Audit stored target-MI research, including fold-local MI and raw-flow scores."""
import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from config import ROOT, DATA_PATH, digest, save_json
from data import load_raw
from research_h1 import prepare_data, annual_splits


def scores(truth, prediction, threshold):
    error = prediction-truth
    peak = error[truth >= threshold]
    return dict(r2=float(1-(error**2).sum()/((truth-truth.mean())**2).sum()),
                rmse=float(np.sqrt((error**2).mean())), mae=float(np.abs(error).mean()),
                peak_rmse=float(np.sqrt((peak**2).mean())), peak_bias=float(peak.mean()), peak_count=len(peak))


def main():
    parent = ROOT / 'output/teacher_selection_20261109'
    output = parent / 'ann_target_mi_audit'
    if output.exists():
        raise FileExistsError('Audit already exists; inspect stored evidence')
    raw = load_raw().loc[:'2003-12-31']
    data = prepare_data(raw, horizons=7)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    comparisons, frames, sources = [], [], []
    checked = 0
    warnings_count = 0
    for seed in (42, 13, 2026):
        folder = parent / f'ann_target_mi_seed{seed}'
        protocol = read(folder / 'protocol.json')
        assert protocol['seed'] == seed and protocol['data_cutoff'] == '2003-12-31'
        assert protocol['label_horizon'] == 7 and protocol['data_sha256'] == digest(DATA_PATH)
        assert read(folder / 'complete.json')['fits'] == 3
        for name, sha in protocol['code_hashes'].items():
            assert digest(ROOT / name) == sha
        fits = read(folder / 'fits.json')
        warnings_count += sum(len(w['warnings']) for w in read(folder / 'warnings.json'))
        for year, tr, va in annual_splits(data['dates'], horizons=7):
            fit = next(f for f in fits if f['year'] == year)
            assert fit['mi_target'] == 'model_target' and fit['mi_policy'] == 'first'
            assert data['dates'][tr[-1]]+pd.Timedelta(days=7) < data['dates'][va[0]]
            target = np.sqrt(data['y'][tr, 0])-np.sqrt(data['mi']['X'][tr, -1, 0])
            mi = mutual_info_regression(data['mi']['X'][tr, -1], target, random_state=seed)
            selected = np.union1d(np.argsort(mi, kind='stable')[-12:], [0])
            np.testing.assert_array_equal(selected, fit['selected_indices'])
            assert fit['selected_names'] == [data['mi']['names'][i] for i in selected]
        for label, src, candidate in [('target', folder, 'ANN_first_target_mi'),
                                      ('raw', parent / f'ann_depth_seed{seed}', 'ANN_first_one')]:
            frame = pd.read_csv(src / 'predictions.csv', parse_dates=['origin', 'target_date'])
            include = [candidate, 'persistence'] if label == 'target' else [candidate]
            table = pd.read_csv(src / 'horizon_metrics.csv').query('candidate in @include')
            for row in table.itertuples():
                group = frame.loc[(frame.candidate == row.candidate) & (frame.year == row.year)].sort_values('origin')
                expected = pd.date_range(f'{row.year}-01-01', f'{row.year}-12-24')
                assert len(group) == 358 and pd.DatetimeIndex(group.origin).equals(expected)
                assert (group.target_date == group.origin+pd.Timedelta(days=1)).all()
                truth = raw.Discharge.reindex(expected+pd.Timedelta(days=row.horizon)).to_numpy()
                prediction = group['prediction' if row.horizon == 1 else f'prediction_h{row.horizon}'].to_numpy()
                np.testing.assert_array_equal(group['truth' if row.horizon == 1 else f'truth_h{row.horizon}'], truth)
                threshold = float(raw.Discharge.loc[:pd.Timestamp(row.year, 1, 1)-pd.Timedelta(days=8)].quantile(.95))
                np.testing.assert_allclose(group.threshold, threshold)
                values = scores(truth, prediction, threshold)
                for name, value in values.items():
                    np.testing.assert_allclose(value, getattr(row, name), rtol=1e-11, atol=1e-9)
                checked += 1
                if row.horizon == 1 and row.candidate == candidate:
                    comparisons.append(dict(seed=str(seed), year=row.year, model=label, **values))
            frames.append(frame.loc[frame.candidate == candidate].assign(seed=seed, model=label))
            sources.append(dict(seed=seed, model=label, predictions_sha256=digest(src / 'predictions.csv'),
                                protocol_sha256=digest(src / 'protocol.json')))
    combined = pd.concat(frames)
    for (year, label), group in combined.groupby(['year', 'model']):
        assert len(group) == 3*358
        assert group.groupby('origin').truth.nunique().eq(1).all()
        avg = group.groupby('origin').agg(truth=('truth', 'first'), prediction=('prediction', 'mean'), threshold=('threshold', 'first'))
        comparisons.append(dict(seed='mean_3', year=int(year), model=label,
                                **scores(avg.truth.to_numpy(), avg.prediction.to_numpy(), avg.threshold.iloc[0])))
    assert checked == 135 and len(comparisons) == 24
    output.mkdir()
    pd.DataFrame(comparisons).to_csv(output / 'h1_comparison.csv', index=False)
    save_json(output / 'verification.json', dict(status='passed', metric_rows=checked, fold_mi_checks=9,
              new_training_fits=9, training_warnings=warnings_count, verifier_fit_calls=0, verifier_predict_calls=0,
              sources=sources, data_sha256=digest(DATA_PATH), verifier_sha256=digest(__file__)))
    print(pd.DataFrame(comparisons)[['seed', 'year', 'model', 'r2', 'mae']].to_string(index=False))


if __name__ == '__main__':
    main()
