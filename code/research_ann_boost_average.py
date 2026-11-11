"""Fixed two-model average of saved development predictions; no fitting or test scoring."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT, DATA_PATH, digest, save_json
from data import load_raw


def combine(ann, boost):
    keys = ['year', 'origin', 'target_date', 'truth']
    a, b = ann.reset_index(drop=True), boost.reset_index(drop=True)
    if a.empty or not a[keys].equals(b[keys]) or a['origin'].duplicated().any():
        raise ValueError('Prediction dates/truth must match exactly, once per origin')
    values = np.column_stack([a['truth'], a['prediction'], b['prediction']])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Flow and predictions must be finite and nonnegative')
    result = a[keys].copy()
    result['ann'], result['boost'] = a['prediction'], b['prediction']
    result['mean'] = (result['ann'] + result['boost']) / 2
    return result


def run():
    base = ROOT/'output/teacher_selection_20261109'
    output = base/'h1_ann_boost_fixed_mean_seed42'
    if output.exists():
        raise FileExistsError('Refusing to overwrite saved analysis')
    sources = [base/'ann_depth_seed42', base/'h1_boost_sqrt_residual_seed42']
    frames, hashes = [], {}
    for source, field, candidate in zip(sources, ['candidate', 'model'], ['ANN_first_one', 'boost'], strict=True):
        protocol = json.loads((source/'protocol.json').read_text(encoding='utf-8'))
        assert protocol['data_cutoff'] == '2003-12-31' and protocol['seed'] == 42
        assert protocol['data_sha256'] == digest(DATA_PATH)
        frame = pd.read_csv(source/'predictions.csv', float_precision='round_trip',
                            parse_dates=['origin', 'target_date'])
        frames.append(frame.loc[frame[field] == candidate])
        for name in ['protocol.json', 'predictions.csv']:
            path = source/name
            hashes[str(path.relative_to(ROOT))] = digest(path)
    combined = combine(*frames)
    expected = pd.DatetimeIndex(np.concatenate([pd.date_range(f'{y}-01-01', f'{y}-12-24').values
                                              for y in [2001, 2002, 2003]]))
    assert pd.DatetimeIndex(combined['origin']).equals(expected)
    assert np.array_equal(combined['year'], expected.year)
    assert (combined['target_date'] == combined['origin'] + pd.Timedelta(days=1)).all()
    raw = load_raw().loc[:'2003-12-31']
    np.testing.assert_array_equal(combined['truth'], raw.loc[combined['target_date'], 'Discharge'])
    output.mkdir(exist_ok=False)
    save_json(output/'protocol.json', dict(preregistration_commit='489f376', seed=42,
        weights=dict(ann=.5, boost=.5), data_cutoff='2003-12-31', fits=0, test_evaluations=0,
        sources=hashes, code_sha256=digest(Path(__file__)),
        limitation='Repeated development years; supplementary two-model mean, not required three-family mean.'))
    rows, diagnostics = [], []
    for year, frame in combined.groupby('year', sort=True):
        truth = frame['truth'].to_numpy()
        denominator = np.sum((truth - truth.mean())**2)
        for model in ['ann', 'boost', 'mean']:
            error = frame[model].to_numpy() - truth
            rows.append(dict(year=int(year), model=model, n=len(frame),
                r2=1 - np.sum(error**2)/denominator,
                rmse=np.sqrt(np.mean(error**2)), mae=np.mean(abs(error))))
        a, b, m = [frame[c].to_numpy() - truth for c in ['ann', 'boost', 'mean']]
        # Fixed-mean SSE identity checks both alignment and averaging arithmetic.
        identity = (np.sum(a*a)+np.sum(b*b))/2 - np.sum((a-b)**2)/4
        np.testing.assert_allclose(np.sum(m*m), identity, rtol=1e-12)
        diagnostics.append(dict(year=int(year), error_correlation=float(np.corrcoef(a,b)[0,1]),
            mean_sse=float(np.sum(m*m)), identity_sse=float(identity)))
    combined.to_csv(output/'predictions.csv', index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/'metrics.csv', index=False)
    save_json(output/'verification.json', dict(rows=len(combined), annual_rows=358, metrics=9,
        raw_next_day_truth=True, fixed_mean_identity=True, fits=0, test_evaluations=0,
        diagnostics=diagnostics,
        output_hashes={name:digest(output/name) for name in ['protocol.json','predictions.csv','metrics.csv']}))
    print(scores.to_string(index=False))


if __name__ == '__main__':
    run()
