"""Two-stage h=1 control: rain-regime classifier plus dry/wet regressors.

Only 2000-2003 teacher-CSV rows are used; 2004 is never constructed or scored.
The two-stage model is a supplementary research control, not a replacement for the
teacher-required 24 base models or 16 ensembles.
"""
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from config import DATA_PATH, ROOT, digest, save_json
from data import load_raw, development_splits
from models import RunoffRegressor
from research_h1 import prepare_data, annual_splits, metrics


class TwoStageRegressor(RegressorMixin, BaseEstimator):
    """y has two columns: h1 flow and target-day rain. Rain is a training label only."""

    def __init__(self, leaf=5, top_k=36, trees=120, seed=42, rain_threshold=0.2, min_regime=30):
        self.leaf = leaf
        self.top_k = top_k
        self.trees = trees
        self.seed = seed
        self.rain_threshold = rain_threshold
        self.min_regime = min_regime

    def _sub(self):
        return RunoffRegressor(family='RF', transform='sqrt', residual=True, top_k=self.top_k,
                               leaf=self.leaf, trees=self.trees, seed=self.seed, seq_len=1)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError('TwoStageRegressor requires [flow, rain] targets')
        flow, rain = y[:, 0], y[:, 1] > self.rain_threshold
        self.clf_ = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=20,
                                           random_state=self.seed, n_jobs=4).fit(X[:, -1], rain)
        self.fallback_ = {}
        self.all_ = self._sub().fit(X, flow[:, None])
        for name, mask in (('dry', ~rain), ('wet', rain)):
            if int(mask.sum()) < self.min_regime:
                self.fallback_[name] = True
                continue
            self.fallback_[name] = False
            setattr(self, f'{name}_', self._sub().fit(X[mask], flow[mask][:, None]))
        if self.fallback_['dry']:
            self.dry_ = self.all_
        if self.fallback_['wet']:
            self.wet_ = self.all_
        return self

    def predict(self, X):
        X = np.asarray(X)
        if len(self.clf_.classes_) == 1:
            p = np.full(len(X), float(self.clf_.classes_[0]))
        else:
            p = self.clf_.predict_proba(X[:, -1])[:, 1]
        p = p[:, None]
        return p * self.wet_.predict(X) + (1 - p) * self.dry_.predict(X)


def h1_score(estimator, X, y):
    return r2_score(y[:, 0], estimator.predict(X)[:, 0])


def make_candidates():
    return [dict(id='RF_baseline', kind='rf'), dict(id='TwoStage', kind='two')]


def run(output, raw=None, seed=42, trees=120):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Output exists; inspect it instead of repeating the run')
    raw = load_raw() if raw is None else raw
    data = prepare_data(raw, horizons=7)
    X, names, y7, dates = data['mi']['X'], data['mi']['names'], data['y'], data['dates']
    flow = y7[:, 0]
    rain_next = raw.Prcp.shift(-1).reindex(dates).to_numpy()
    if not np.isfinite(rain_next).all():
        raise ValueError('Target-day rain labels are incomplete')
    if (dates + pd.Timedelta(days=1)).max() > pd.Timestamp('2003-12-31'):
        raise ValueError('Target-day rain label crosses 2003-12-31')
    if X.shape[1:] != (30, len(names)):
        raise ValueError('Unexpected input shape')

    grid = dict(top_k=[12, 24, 36], leaf=[1, 5])
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'protocol.json', dict(
        data_cutoff='2003-12-31', test_evaluations=0, seed=seed, outputs=1, sample_horizon=7,
        gap=7, outer='2001/2002/2003 annual', inner='TimeSeriesSplit(2), gap=7',
        selection='inner h1 raw R2; each fold refits MI and scaling',
        scoring='unweighted original-flow R2', grid=grid, trees=trees,
        rain_rule='Prcp(t+1) > 0.2mm, training label only',
        classifier='RandomForestClassifier(100, max_depth=3, min_samples_leaf=20)',
        mixture='p*wet + (1-p)*dry; no post-hoc calibration',
        min_regime_samples=30,
        gate=dict(combined_r2_delta=0.01, r2_2003_floor=-0.005, r2_2002_floor=-0.01,
                  mae_ratio_max=1.01, r2_2001_floor=-0.03, comparison_days=716),
        candidates=make_candidates(),
        limitation='Repeated training-period development; not an independent blind test.',
        data_sha256=digest(DATA_PATH) if raw is not None else 'unit-test fixture',
        code_sha256={p.name: digest(p) for p in (ROOT / 'code').glob('*.py')}))

    rows, preds, selected, fits, warns, regime_rows = [], [], [], [], [], []
    y_two = np.column_stack([flow, rain_next])
    for cand in make_candidates():
        for year, tr, va in annual_splits(dates, horizons=7):
            print(f"[{cand['id']}] year={year} train={len(tr)} valid={len(va)}", flush=True)
            inner = development_splits(dates[tr])
            base = RunoffRegressor(family='RF', transform='sqrt', residual=True, top_k=24,
                                   leaf=5, trees=trees, seed=seed, seq_len=1) if cand['kind'] == 'rf' \
                else TwoStageRegressor(trees=trees, seed=seed)
            y_fit = flow[:, None] if cand['kind'] == 'rf' else y_two
            search = GridSearchCV(base, grid, cv=inner, scoring=h1_score, n_jobs=1,
                                  error_score='raise', refit=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                search.fit(X[tr], y_fit[tr])
            warns.extend(dict(candidate=cand['id'], year=year, category=w.category.__name__,
                              message=str(w.message)) for w in caught)
            model = search.best_estimator_
            pred = model.predict(X[va])[:, 0]
            selected.append(dict(candidate=cand['id'], year=year, params=search.best_params_,
                                 inner_r2=float(search.best_score_), train_count=len(tr),
                                 valid_count=len(va),
                                 fallback=getattr(model, 'fallback_', None)))
            threshold = float(np.quantile(flow[tr], .95))
            rows.append(dict(candidate=cand['id'], year=year, horizon=1, threshold=threshold,
                             **metrics(flow[va], pred, threshold)))
            preds.extend(dict(candidate=cand['id'], year=year, origin=str(date.date()),
                              target_date=str((date + pd.Timedelta(days=1)).date()),
                              truth=float(t), prediction=float(v), rain_next=float(r),
                              threshold=threshold, peak=bool(t >= threshold))
                         for date, t, v, r in zip(dates[va], flow[va], pred, rain_next[va]))
            dry = rain_next[va] <= 0.2
            for label, mask in (('dry', dry), ('wet', ~dry)):
                if mask.sum() > 1:
                    regime_rows.append(dict(candidate=cand['id'], year=year, regime=label,
                                            count=int(mask.sum()),
                                            r2=float(r2_score(flow[va][mask], pred[mask])),
                                            mae=float(np.mean(np.abs(pred[mask] - flow[va][mask])))))
            fits.append(dict(candidate=cand['id'], year=year, params=search.best_params_,
                             rain_rate=float((rain_next[tr] > 0.2).mean()),
                             dry_count=int((rain_next[tr] <= 0.2).sum()),
                             wet_count=int((rain_next[tr] > 0.2).sum())))
        pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
        pd.DataFrame(preds).to_csv(output / 'predictions.csv', index=False)
        pd.DataFrame(selected).to_csv(output / 'selected.csv', index=False)
        pd.DataFrame(regime_rows).to_csv(output / 'regime_metrics.csv', index=False)
        save_json(output / 'warnings.json', warns)
        save_json(output / 'fits.json', fits)

    comparison = evaluate_gate(pd.DataFrame(preds))
    save_json(output / 'comparison.json', comparison)
    save_json(output / 'complete.json', dict(status='complete', fits=len(fits),
                                             test_evaluations=0,
                                             qualified=comparison['qualified']))
    return comparison


def _combined(frame, candidate):
    part = frame[(frame.candidate == candidate) & frame.year.isin([2002, 2003])]
    if len(part) != 716:
        raise ValueError(f'{candidate} must have 716 comparison days')
    per = {int(y): float(r2_score(g.truth, g.prediction)) for y, g in part.groupby('year')}
    return dict(combined_r2=float(r2_score(part.truth, part.prediction)),
                combined_mae=float(np.mean(np.abs(part.prediction - part.truth))),
                r2_2002=per[2002], r2_2003=per[2003])


def evaluate_gate(frame):
    cand, ref = _combined(frame, 'TwoStage'), _combined(frame, 'RF_baseline')
    one = frame[(frame.candidate == 'RF_baseline') & (frame.year == 2001)]
    one_two = frame[(frame.candidate == 'TwoStage') & (frame.year == 2001)]
    r2_2001 = float(r2_score(one.truth, one.prediction))
    r2_2001_two = float(r2_score(one_two.truth, one_two.prediction))
    delta = cand['combined_r2'] - ref['combined_r2']
    ratio = cand['combined_mae'] / ref['combined_mae']
    checks = dict(combined_r2=delta >= 0.01, r2_2003=cand['r2_2003'] >= ref['r2_2003'] - 0.005,
                  r2_2002=cand['r2_2002'] >= ref['r2_2002'] - 0.01, mae=ratio <= 1.01,
                  r2_2001=r2_2001_two >= r2_2001 - 0.03)
    qualified = all(checks.values())
    return dict(comparison_days=716, test_evaluations=0, candidate='TwoStage',
                control='RF_baseline', **cand, control_combined_r2=ref['combined_r2'],
                control_combined_mae=ref['combined_mae'], control_r2_2002=ref['r2_2002'],
                control_r2_2003=ref['r2_2003'], r2_2001=r2_2001_two,
                control_r2_2001=r2_2001, delta_r2=delta, mae_ratio=ratio,
                checks=checks, qualified=qualified, qualified_list=['TwoStage'] if qualified else [])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    run(**vars(parser.parse_args()))
