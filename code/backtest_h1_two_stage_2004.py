"""One authorized 2004 backtest of the frozen two-stage h1 model.

Order is enforced in code: fit on 2000-2003, dump weights and protocol hashes,
and only then construct 2004 features / read 2004 labels. One evaluation only.
"""
import argparse
from datetime import datetime
from pathlib import Path
import warnings
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import DATA_PATH, ROOT, digest, save_json
from data import load_raw
from research_h1 import prepare_data, metrics
from research_h1_two_stage import TwoStageRegressor


def frozen_params(training_run):
    """Majority of the per-year selections made on training-period validation only."""
    import ast
    table = pd.read_csv(Path(training_run) / 'selected.csv')
    rows = table[table.candidate == 'TwoStage']
    if rows.empty:
        raise ValueError('No TwoStage selections recorded in the training run')
    votes = {}
    for row in rows.itertuples():
        chosen = ast.literal_eval(row.params) if isinstance(row.params, str) else row.params
        if not isinstance(chosen, dict) or 'top_k' not in chosen or 'leaf' not in chosen:
            raise ValueError('Recorded selection is not a top_k/leaf grid point')
        votes.setdefault((int(chosen['top_k']), int(chosen['leaf'])), []).append(int(row.year))
    best = max(votes.items(), key=lambda item: (len(item[1]), max(item[1])))
    return dict(top_k=best[0][0], leaf=best[0][1], trees=120, seed=42,
                votes={f'top_k={k[0]},leaf={k[1]}': sorted(v) for k, v in votes.items()})


def run(output, raw=None, params=None, training_run=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Output exists; the single authorized backtest is not repeatable')
    source = load_raw() if raw is None else raw
    if params is None:
        params = frozen_params(training_run or ROOT / 'output/teacher_selection_20261109/h1_two_stage_20261111')

    train_raw = source.loc[:'2003-12-31']
    train = prepare_data(train_raw, horizons=7)
    X_tr, y_tr, d_tr = train['mi']['X'], train['y'][:, 0], train['mi']['dates']
    if d_tr.max() + pd.Timedelta(days=7) > pd.Timestamp('2003-12-31'):
        raise ValueError('Training labels cross the 2003-12-31 freeze boundary')
    rain_tr = train_raw.Prcp.shift(-1).reindex(d_tr).to_numpy()
    if not np.isfinite(rain_tr).all():
        raise ValueError('Training rain labels incomplete')

    model = TwoStageRegressor(**{k: params[k] for k in ('top_k', 'leaf', 'trees', 'seed')})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(X_tr, np.column_stack([y_tr, rain_tr]))
    warnings_out = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]

    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, output / 'model.joblib')
    freeze = dict(
        stage='frozen_before_test_labels', frozen_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        model='TwoStageRegressor(p, wet, dry)', params=params,
        train_origins=int(len(d_tr)), train_start=str(d_tr[0].date()), train_end=str(d_tr[-1].date()),
        train_label_end=str((d_tr[-1] + pd.Timedelta(days=7)).date()),
        rain_rule='Prcp(t+1) > 0.2mm, training label only',
        data_sha256='unit-test fixture' if raw is not None else digest(DATA_PATH),
        model_sha256=digest(output / 'model.joblib'),
        code_sha256={p.name: digest(p) for p in (ROOT / 'code').glob('*.py')},
        selection_note='Params fixed by majority of training-period annual selections; no 2004 input.',
        authorization='User authorized exactly one 2004 backtest on 2026-11-11.')
    save_json(output / 'freeze.json', freeze)

    # --- Only past this line are 2004 features and labels constructed. ---
    test = prepare_data(source, cutoff='2004-12-31', horizons=7)
    X_te, y_te, d_te = test['mi']['X'], test['y'][:, 0], test['mi']['dates']
    keep = np.flatnonzero((d_te.year == 2004) &
                          ((d_te + pd.Timedelta(days=1)).year == 2004))
    if len(keep) != 359:
        raise ValueError(f'Expected 359 2004 origins, got {len(keep)}')
    pred = model.predict(X_te[keep])[:, 0]
    truth = y_te[keep]
    threshold = float(np.quantile(y_tr, .95))
    rain_te = source.Prcp.shift(-1).reindex(d_te[keep]).to_numpy()

    rows = [dict(candidate='TwoStage', strategy='direct', year=2004, horizon=1, n=int(len(keep)),
                 threshold=threshold, **metrics(truth, pred, threshold))]
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
    frame = pd.DataFrame(dict(candidate='TwoStage', origin=[str(d.date()) for d in d_te[keep]],
                              target_date=[str((d + pd.Timedelta(days=1)).date()) for d in d_te[keep]],
                              truth=truth, prediction=pred, rain_next=rain_te,
                              threshold=threshold, peak=truth >= threshold))
    frame.to_csv(output / 'predictions.csv', index=False, float_format='%.17g')

    ref = pd.read_csv(ROOT / 'output/formal_20261105_clean/metrics.csv')
    ref_h1 = ref[(ref.h == 1)].sort_values('R2', ascending=False).iloc[0]
    dry = rain_te <= 0.2
    result = dict(
        status='evaluated_once', evaluations=1, test_year=2004, origins=int(len(keep)),
        r2=float(r2_score(truth, pred)),
        rmse=float(np.sqrt(np.mean((pred - truth) ** 2))),
        mae=float(np.mean(np.abs(pred - truth))),
        peak_rmse=rows[0]['peak_rmse'], peak_bias=rows[0]['peak_bias'], peak_count=rows[0]['peak_count'],
        dry_days=dict(count=int(dry.sum()),
                      r2=float(r2_score(truth[dry], pred[dry])) if dry.sum() > 1 else None,
                      mae=float(np.mean(np.abs(pred[dry] - truth[dry]))) if dry.sum() else None),
        wet_days=dict(count=int((~dry).sum()),
                      r2=float(r2_score(truth[~dry], pred[~dry])) if (~dry).sum() > 1 else None,
                      mae=float(np.mean(np.abs(pred[~dry] - truth[~dry]))) if (~dry).sum() else None),
        formal_reference=dict(strategy=ref_h1.strategy, model=ref_h1.model, r2=float(ref_h1.R2),
                              rmse=float(ref_h1.RMSE), mae=float(ref_h1.MAE)),
        warnings=warnings_out,
        limitation='2004 labels were already inspected in earlier work; this is a retrospective evaluation, not a blind test.')
    save_json(output / 'results.json', result)
    save_json(output / 'complete.json', dict(status='complete', evaluations=1, test_evaluations=1))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(**vars(parser.parse_args()))
