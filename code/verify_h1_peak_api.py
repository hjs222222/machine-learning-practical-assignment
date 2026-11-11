"""Independently recompute recorded h1 peak/API controls; never refits or reads 2004."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import save_json

EXPECTED = {'RF_api': 'RF_mi', 'ANN_api': 'ANN_mi', 'RF_peak3': 'RF_mi',
            'RF_peak10': 'RF_mi', 'ANN_torch3': 'ANN_torch1', 'ANN_torch10': 'ANN_torch1'}
YEARS = [2002, 2003]


def _combined(frame):
    return r2_score(frame.truth, frame.prediction), float(np.mean(np.abs(frame.prediction - frame.truth)))


def verify(output):
    output = Path(output)
    pred = pd.read_csv(output / 'predictions.csv', parse_dates=['origin', 'target_date'])
    metrics = pd.read_csv(output / 'metrics.csv')
    comparison = json.loads((output / 'comparison.json').read_text(encoding='utf-8'))
    problems = []

    if (pred.target_date > pd.Timestamp('2003-12-31')).any():
        problems.append('target_date beyond 2003-12-31')
    if (pred.origin > pd.Timestamp('2003-12-31')).any():
        problems.append('origin beyond 2003-12-31')
    if (pred.prediction < 0).any() or not np.isfinite(pred.prediction).all():
        problems.append('nonfinite or negative prediction')
    if (pred.target_date != pred.origin + pd.Timedelta(days=1)).any():
        problems.append('target_date is not origin+1')
    counts = pred.groupby(['candidate', 'variant', 'year']).size()
    if not counts.eq(358).all():
        problems.append('each candidate/variant/year must have 358 origins')

    for row in metrics.itertuples():
        part = pred[(pred.candidate == row.candidate) & (pred.variant == row.variant)
                    & (pred.year == row.year)]
        error = part.prediction.to_numpy() - part.truth.to_numpy()
        if abs(r2_score(part.truth, part.prediction) - row.r2) > 1e-9:
            problems.append(f'R2 mismatch {row.candidate}/{row.variant}/{row.year}')
        if abs(np.sqrt(np.mean(error ** 2)) - row.rmse) > 1e-6:
            problems.append(f'RMSE mismatch {row.candidate}/{row.variant}/{row.year}')
        if abs(np.mean(np.abs(error)) - row.mae) > 1e-6:
            problems.append(f'MAE mismatch {row.candidate}/{row.variant}/{row.year}')

    entries = {entry['candidate']: entry for entry in comparison['entries']}
    present = set(pred.candidate)
    for candidate, control in EXPECTED.items():
        if candidate not in present or control not in present:
            continue
        part = pred[(pred.candidate == candidate) & (pred.variant == 'selected')
                    & pred.year.isin(YEARS)]
        ref = pred[(pred.candidate == control) & (pred.variant == 'selected')
                   & pred.year.isin(YEARS)]
        if len(part) != 716 or len(ref) != 716:
            problems.append(f'{candidate} combined rows are not 716')
            continue
        r2, mae = _combined(part)
        ref_r2, ref_mae = _combined(ref)
        entry = entries.get(candidate)
        if entry is None:
            problems.append(f'missing comparison entry {candidate}')
            continue
        if abs(entry['combined_r2'] - r2) > 1e-9 or abs(entry['combined_mae'] - mae) > 1e-6:
            problems.append(f'comparison mismatch {candidate}')
        per = {y: r2_score(g.truth, g.prediction) for y, g in part.groupby('year')}
        ref_per = {y: r2_score(g.truth, g.prediction) for y, g in ref.groupby('year')}
        expected = bool(r2 - ref_r2 >= 0.01 and per[2003] >= ref_per[2003] - 0.005
                        and per[2002] >= ref_per[2002] - 0.01 and mae / ref_mae <= 1.01)
        if bool(entry['qualified']) != expected:
            problems.append(f'gate decision mismatch {candidate}')
        if bool(entry['qualified']) != (candidate in comparison['qualified']):
            problems.append(f'qualified list mismatch {candidate}')

    result = dict(status='passed' if not problems else 'failed', problems=problems,
                  candidates=sorted(present), rows=int(len(pred)),
                  test_evaluations=0,
                  qualified=comparison['qualified'])
    save_json(output / 'independent_verification.json', result)
    if problems:
        raise ValueError('; '.join(problems))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(verify(parser.parse_args().output))
