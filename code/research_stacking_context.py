"""Finite historical OOF study; does not change the formal Stacking interface."""
import argparse
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT, DATA_PATH, digest, save_json
from models import fit_stacking
from research_h1 import metrics

CASE = 'mean_ann_log_lstm_daily'
BASE = ['RF', 'ANN', 'LSTM']
CONTEXT = ['Discharge', 'Prcp']
DATA_SHA = 'ab09410b23118510bb4e8787a62b97693336a1c46477a01329f587c64bc57151'


def context_fold(raw, source, year, include_context):
    """Build historical meta rows with a seven-day origin gap, never future context."""
    if year not in (2002, 2003):
        raise ValueError('Only historical OOF evaluation years 2002/2003 are allowed')
    raw = raw.loc[raw.index < '2004-01-01']
    frame = source.sort_values('origin').reset_index(drop=True)
    expected = pd.DatetimeIndex([d for y in (2001, 2002, 2003)
                                for d in pd.date_range(f'{y}-01-01', f'{y}-12-30')])
    if (not raw.index.is_unique or len(frame) != len(expected)
            or not np.array_equal(frame.origin, expected)
            or not np.array_equal(frame.year, expected.year)
            or not np.array_equal(frame.target_date, expected + pd.Timedelta(days=1))):
        raise ValueError('OOF origins, target dates or years are duplicated, missing or misaligned')
    if not np.array_equal(frame.truth, raw.loc[frame.target_date, 'Discharge']):
        raise ValueError('OOF truths differ from original next-day observations')
    values = frame[BASE].to_numpy(dtype=float)
    context = raw.loc[frame.origin, CONTEXT].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(context).all() or not np.isfinite(frame.truth).all():
        raise ValueError('Nonfinite original context, truth or base prediction')
    if include_context:
        values = np.column_stack([values, context])
    train = frame.origin < pd.Timestamp(year, 1, 1) - pd.Timedelta(days=7)
    valid = frame.year == year
    return frame.loc[train].copy(), frame.loc[valid].copy(), values[train], values[valid]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 13, 2026), default=42)
    args = parser.parse_args()
    parent = ROOT / 'output/h1_neural_20261106'
    output = parent / f'stacking_context_seed{args.seed}'
    if output.exists():
        raise ValueError('Output exists; no overwrite or hot start')
    if digest(DATA_PATH) != DATA_SHA:
        raise ValueError('Teacher CSV hash mismatch')
    import json
    source_dir = parent / f'teacher_mean_h1_seed{args.seed}'
    old_dir = parent / f'teacher_stacking_h1_seed{args.seed}'
    protocol = json.loads((source_dir / 'protocol.json').read_text(encoding='utf-8'))
    old_protocol = json.loads((old_dir / 'protocol.json').read_text(encoding='utf-8'))
    if (protocol['data_cutoff'], protocol['horizon'], protocol['gap'], protocol['seed']) != ('2003-12-31', 1, 7, args.seed):
        raise ValueError('Incompatible historical OOF protocol')
    if (digest(source_dir / 'predictions.csv') != old_protocol['source']['predictions_sha256']
            or digest(source_dir / 'protocol.json') != old_protocol['source']['protocol_sha256']):
        raise ValueError('Historical source hashes changed')
    if args.seed != 42:
        gate = json.loads((parent / 'stacking_context_seed42/promotion_gate.json').read_text())
        if not gate['expansion_gate_passed']:
            raise ValueError('Seed42 did not pass the frozen expansion gate')
    raw = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    raw = raw.loc[raw.Date < '2004-01-01'].set_index('Date')
    source = pd.read_csv(source_dir / 'predictions.csv', parse_dates=['origin', 'target_date'])
    source = source.loc[source.candidate == CASE].copy()
    old = pd.read_csv(old_dir / 'predictions.csv', parse_dates=['origin'])
    old = old.loc[old.candidate == 'stack_ann_log_lstm_daily']
    # Validate every fold before creating the immutable run directory.
    folds = {(year, context): context_fold(raw, source, year, context)
             for year in (2002, 2003) for context in (False, True)}
    output.mkdir()
    save_json(output / 'protocol.json', dict(stage='historical_stacking_context', seed=args.seed,
              case=CASE, context_fields=['Discharge(t)', 'Prcp(t)'], alpha=10.0,
              data_cutoff='2003-12-31', years=[2002, 2003], gap=7, new_base_fits=0,
              planned_meta_fits=4, data_sha256=DATA_SHA,
              git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
              code_hashes={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'code').glob('*.py'))},
              source_hashes={str(p.relative_to(ROOT)): digest(p) for directory in (source_dir, old_dir)
                             for p in (directory / 'protocol.json', directory / 'predictions.csv')},
              gate='worst/mean R2 noninferior 1e-10, one improves >1e-6, MAE/peak RMSE <=1.01 baseline',
              limitation='Repeated research years, non-nested base choices; matched 2002/2003 only; no new 2004 evaluation'))
    predictions, scores, fits, warning_rows = [], [], [], []
    for (year, context), (train, valid, x, v) in folds.items():
        candidate = 'ridge_context' if context else 'ridge_baseline'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = fit_stacking([x], train.truth.to_numpy()[:, None])
        warning_rows.extend(dict(candidate=candidate, year=year, message=str(w.message)) for w in caught)
        before = model.predict(v).ravel()
        prediction = np.maximum(before, 0)
        if not context:
            historical = old.loc[old.year == year].sort_values('origin')
            np.testing.assert_array_equal(valid.origin, historical.origin)
            np.testing.assert_allclose(prediction, historical.prediction, rtol=0, atol=1e-7)
        threshold = float(valid.threshold.iloc[0])
        expected_threshold = float(raw.loc[:train.origin.max(), 'Discharge'].quantile(.95))
        np.testing.assert_allclose(threshold, expected_threshold, rtol=1e-12)
        fits.append(dict(candidate=candidate, year=year, train_count=len(train),
                         train_start=str(train.origin.min().date()), train_end=str(train.origin.max().date()),
                         train_target_end=str(train.target_date.max().date()), input_width=x.shape[1],
                         columns=BASE + (['Discharge(t)', 'Prcp(t)'] if context else []),
                         scaler_mean=model[0].mean_.tolist(), scaler_scale=model[0].scale_.tolist(),
                         coef=model[-1].coef_.ravel().tolist(), intercept=float(model[-1].intercept_[0])))
        frame = valid.copy()
        frame['Discharge_t'], frame['Prcp_t'] = raw.loc[valid.origin, CONTEXT].to_numpy().T
        frame['candidate'], frame['prediction'], frame['prediction_before_clip'] = candidate, prediction, before
        predictions.append(frame)
        scores.append(dict(candidate=candidate, family='Stacking', year=year, threshold=threshold,
                           **metrics(valid.truth.to_numpy(), prediction, threshold)))
    score_frame = pd.DataFrame(scores)
    summary = score_frame.groupby('candidate', as_index=False).agg(
        worst_r2=('r2', 'min'), mean_r2=('r2', 'mean'), mean_mae=('mae', 'mean'),
        mean_rmse=('rmse', 'mean'), mean_peak_rmse=('peak_rmse', 'mean'))
    pd.concat(predictions).to_csv(output / 'predictions.csv', index=False)
    score_frame.to_csv(output / 'fold_metrics.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    save_json(output / 'fits.json', fits)
    save_json(output / 'warnings.json', warning_rows)
    save_json(output / 'complete.json', dict(status='complete', meta_fits=4, base_fits=0,
                                           predictions=1456, warnings=len(warning_rows)))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
