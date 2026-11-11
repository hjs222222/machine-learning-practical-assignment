"""复用冻结OOF重拟合2003元层并诊断误差相关性；不改变配置、不读取2004预测。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset
from acceptance3_train import ROOT


def score(y, p):
    return float(np.sqrt(np.square(y - p).mean(axis=0)).mean()), float(np.abs(y - p).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    output, correlations, coeffs = [], [], []
    for run_name in ['formal_20261111_gpu_lag_residual', 'formal_20261111_lognormal']:
        run = ROOT/'output/acceptance3_runs'/run_name
        data = load_dataset(str(run/'dataset.npz'))
        oof = pd.read_csv(run/'oof_predictions.csv')
        meta = json.loads((run/'meta_selection.json').read_text(encoding='utf-8'))
        for (strategy, task_h, seed), own in oof.groupby(['strategy', 'task_h', 'seed']):
            pieces = [own[own.model == family].pivot(index='sample_id', columns='h', values='pred').sort_index()
                      for family in ['ANN', 'RF', 'LSTM']]
            ids = pieces[0].index.to_numpy(int)
            assert all(part.index.equals(pieces[0].index) and part.columns.equals(pieces[0].columns)
                       for part in pieces)
            assert (data['origin_days'][ids] + 7 <= np.datetime64('2003-12-31').astype('int64')).all()
            blocks = [part.to_numpy() for part in pieces]
            assert all(np.isfinite(block).all() for block in blocks)
            x = np.column_stack(blocks)
            y = data['y'][ids, task_h - 1:task_h] if strategy == 'direct' else data['y'][ids]
            error = np.column_stack([(p - y).reshape(-1) for p in blocks])
            corr = np.corrcoef(error, rowvar=False)
            correlations.append(dict(run=run_name, strategy=strategy, h=task_h, seed=seed,
                ANN_RF=float(corr[0, 1]), ANN_LSTM=float(corr[0, 2]), RF_LSTM=float(corr[1, 2])))
            selected = next(r for r in meta if r['strategy'] == strategy and r['h'] == task_h and r['seed'] == seed)
            for start, stop in [('2003-07-01', '2003-10-01'), ('2003-10-01', '2004-01-01')]:
                first, last = [np.datetime64(s).astype('int64') for s in [start, stop]]
                ti = np.flatnonzero(data['origin_days'][ids] + 7 < first)
                vi = np.flatnonzero((data['origin_days'][ids] >= first) & (data['origin_days'][ids] < last))
                assert data['origin_days'][ids[ti[-1]]] + 7 < data['origin_days'][ids[vi[0]]]
                model = make_pipeline(StandardScaler(), Ridge(alpha=selected['alpha'])).fit(x[ti], y[ti])
                sp = model.predict(x[vi]).reshape(len(vi), -1)
                outputs = dict(ANN=blocks[0][vi], RF=blocks[1][vi], LSTM=blocks[2][vi],
                    Mean=np.stack([p[vi] for p in blocks]).mean(axis=0), Stacking=sp)
                for family, p in outputs.items():
                    rmse, mae = score(y[vi], p)
                    output.append(dict(run=run_name, strategy=strategy, h=task_h, seed=seed,
                        validation_start=start, model=family, n_meta_train=len(ti), n_validation=len(vi),
                        alpha=selected['alpha'], RMSE=rmse, MAE=mae,
                        negative_predictions=int((p < 0).sum())))
                ridge, scaler = model.steps[-1][1], model.steps[0][1]
                effective = ridge.coef_ / scaler.scale_
                effective_intercept = ridge.intercept_ - effective @ scaler.mean_
                np.testing.assert_allclose(x[vi] @ effective.T + effective_intercept, sp,
                                           rtol=1e-9, atol=1e-7)
                coeffs.append(dict(run=run_name, strategy=strategy, h=int(task_h), seed=int(seed),
                    validation_start=start, alpha=selected['alpha'],
                    max_abs_coefficient=float(np.max(np.abs(effective))),
                    coefficients=effective.tolist(),
                    effective_intercept=np.asarray(effective_intercept).tolist(),
                    standardized_intercept=np.asarray(ridge.intercept_).tolist()))
    frame = pd.DataFrame(output)
    frame.to_csv(args.out/'meta_forward_diagnostics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(correlations).to_csv(args.out/'base_error_correlations.csv', index=False, encoding='utf-8-sig')
    (args.out/'coefficients.json').write_text(json.dumps(coeffs, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.out/'manifest.json').write_text(json.dumps(dict(status='completed', read_2004=False,
        purpose='重建现有alpha的2003开发表现；alpha也由这些窗选择，非独立验证',
        fits=len(coeffs)), ensure_ascii=False, indent=2), encoding='utf-8')
    print(frame.groupby(['run', 'strategy', 'model'])[['RMSE', 'MAE', 'negative_predictions']].mean().to_string())
    print(pd.DataFrame(correlations).groupby(['run', 'strategy'])[['ANN_RF','ANN_LSTM','RF_LSTM']].mean().to_string())


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
