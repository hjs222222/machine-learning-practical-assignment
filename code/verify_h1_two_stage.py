"""Independently recompute recorded two-stage h1 control; never refits or reads 2004."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import save_json


def verify(output):
    output = Path(output)
    pred = pd.read_csv(output / 'predictions.csv', parse_dates=['origin', 'target_date'])
    metrics = pd.read_csv(output / 'metrics.csv')
    regime = pd.read_csv(output / 'regime_metrics.csv')
    comparison = json.loads((output / 'comparison.json').read_text(encoding='utf-8'))
    problems = []

    if (pred.target_date > pd.Timestamp('2003-12-31')).any():
        problems.append('target_date beyond 2003-12-31')
    if (pred.origin > pd.Timestamp('2003-12-31')).any():
        problems.append('origin beyond 2003-12-31')
    if (pred.target_date != pred.origin + pd.Timedelta(days=1)).any():
        problems.append('target_date is not origin+1')
    if (pred.prediction < 0).any() or not np.isfinite(pred.prediction).all():
        problems.append('nonfinite or negative prediction')
    counts = pred.groupby(['candidate', 'year']).size()
    if not counts.eq(358).all():
        problems.append('each candidate/year must have 358 origins')

    for row in metrics.itertuples():
        part = pred[(pred.candidate == row.candidate) & (pred.year == row.year)]
        error = part.prediction.to_numpy() - part.truth.to_numpy()
        if abs(r2_score(part.truth, part.prediction) - row.r2) > 1e-9:
            problems.append(f'R2 mismatch {row.candidate}/{row.year}')
        if abs(np.mean(np.abs(error)) - row.mae) > 1e-6:
            problems.append(f'MAE mismatch {row.candidate}/{row.year}')

    for row in regime.itertuples():
        part = pred[(pred.candidate == row.candidate) & (pred.year == row.year)]
        mask = part.rain_next <= 0.2 if row.regime == 'dry' else part.rain_next > 0.2
        sub = part[mask]
        if len(sub) != row.count:
            problems.append(f'regime count mismatch {row.candidate}/{row.year}/{row.regime}')
        elif abs(r2_score(sub.truth, sub.prediction) - row.r2) > 1e-9:
            problems.append(f'regime R2 mismatch {row.candidate}/{row.year}/{row.regime}')

    def combined(name):
        part = pred[(pred.candidate == name) & pred.year.isin([2002, 2003])]
        if len(part) != 716:
            raise ValueError(f'{name} combined rows are not 716')
        per = {int(y): float(r2_score(g.truth, g.prediction)) for y, g in part.groupby('year')}
        return (float(r2_score(part.truth, part.prediction)),
                float(np.mean(np.abs(part.prediction - part.truth))), per)

    cand_r2, cand_mae, cand_per = combined('TwoStage')
    ref_r2, ref_mae, ref_per = combined('RF_baseline')
    one = pred[(pred.candidate == 'RF_baseline') & (pred.year == 2001)]
    one_two = pred[(pred.candidate == 'TwoStage') & (pred.year == 2001)]
    r2_2001, r2_2001_two = float(r2_score(one.truth, one.prediction)), float(r2_score(one_two.truth, one_two.prediction))
    checks = dict(combined_r2=(cand_r2 - ref_r2) >= 0.01,
                  r2_2003=cand_per[2003] >= ref_per[2003] - 0.005,
                  r2_2002=cand_per[2002] >= ref_per[2002] - 0.01,
                  mae=(cand_mae / ref_mae) <= 1.01,
                  r2_2001=r2_2001_two >= r2_2001 - 0.03)
    if any(abs(comparison[k] - v) > 1e-9 for k, v in
           (('combined_r2', cand_r2), ('control_combined_r2', ref_r2),
            ('combined_mae', cand_mae), ('r2_2001', r2_2001_two),
            ('control_r2_2001', r2_2001))):
        problems.append('comparison values do not match recomputation')
    if comparison['checks'] != checks:
        problems.append('gate checks do not match recomputation')
    if bool(comparison['qualified']) != all(checks.values()):
        problems.append('qualified flag does not match gate')
    if comparison['test_evaluations'] != 0:
        problems.append('test_evaluations must be 0')

    result = dict(status='passed' if not problems else 'failed', problems=problems,
                  rows=int(len(pred)), comparison_days=716, test_evaluations=0,
                  checks=checks, qualified=bool(comparison['qualified']))
    save_json(output / 'independent_verification.json', result)
    if problems:
        raise ValueError('; '.join(problems))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(verify(parser.parse_args().output))
