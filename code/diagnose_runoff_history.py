"""只诊断2003历史：训练/验证误差、极端流量贡献和分布漂移；不重新选参、不读取2004预测。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset, restore_model
from acceptance3_train import ROOT
from runoff_training.train import year_folds


RUNS = ['formal_20261111_gpu_lag_residual', 'formal_20261111_lognormal']


def measures(truth, prediction):
    truth, prediction = np.asarray(truth).reshape(len(truth), -1), np.asarray(prediction).reshape(len(truth), -1)
    sse = np.square(truth - prediction).sum(axis=0)
    sst = np.square(truth - truth.mean(axis=0)).sum(axis=0)
    return dict(R2=float((1 - sse / sst).mean()),
        RMSE=float(np.sqrt(np.square(truth - prediction).mean(axis=0)).mean()),
        MAE=float(np.abs(truth - prediction).mean()),
        log_MSE=float(np.square(np.log(truth) - np.log(np.maximum(prediction, 1e-12))).mean()),
        mean_truth=float(truth.mean()), mean_pred=float(prediction.mean()),
        mean_ratio=float(prediction.mean() / truth.mean()))


def diagnose(out, device):
    out.mkdir(parents=True, exist_ok=False)
    rows, segments, drift = [], [], []
    sources = {}
    for run_name in RUNS:
        run = ROOT/'output/acceptance3_runs'/run_name
        manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'completed' and not manifest['smoke']
        data = load_dataset(str(run/'dataset.npz'))
        train = np.flatnonzero(data['origin_days'] + 7 <= np.datetime64('2003-12-31').astype('int64'))
        folds = year_folds(data, train)
        q_index = data['feature_names'].tolist().index('Discharge_current')
        for strategy, h in [('direct', h) for h in range(1, 8)] + [('multi', 7)]:
            y = data['y'][:, h - 1:h] if strategy == 'direct' else data['y']
            for label, ti, vi in folds:
                assert data['origin_days'][ti[-1]] + 7 < data['origin_days'][vi[0]]
                assert data['origin_days'][vi[-1]] + 7 <= np.datetime64('2003-12-31').astype('int64')
                threshold = float(np.quantile(y[ti], .9))
                for seed in [42, 43, 44]:
                    name = (f'{label}_{strategy}_LSTM_h{h}_seed{seed}' if run_name == RUNS[0]
                            else f'{label}_{strategy}_h{h}_seed{seed}')
                    folder = run/'models'/name
                    model = restore_model(folder, device=device)
                    sources[str((folder/'weights.pt').relative_to(ROOT))] = hashlib.sha256((folder/'weights.pt').read_bytes()).hexdigest()
                    for split, ids in [('train_all', ti), ('train_recent90', ti[-90:]), ('validation', vi)]:
                        pred = np.asarray(model.predict(ids[:, None])).reshape(len(ids), -1)
                        result = measures(y[ids], pred)
                        rows.append(dict(run=run_name, strategy=strategy, h=h, fold=label, seed=seed,
                            split=split, n=len(ids), objective=getattr(model, 'objective', 'log_mse'), **result))
                        if split == 'validation':
                            truth = y[ids]
                            squared = np.square(truth - pred)
                            q = data['features'][data['positions'][ids], q_index:q_index + 1]
                            masks = dict(high=truth >= threshold, other=truth < threshold,
                                high_rising=(truth >= threshold) & (truth >= 2 * q),
                                high_recession=(truth >= threshold) & (truth < 2 * q))
                            for segment, mask in masks.items():
                                segments.append(dict(run=run_name, strategy=strategy, h=h, fold=label,
                                    seed=seed, segment=segment, threshold_train_q90=threshold,
                                    n=int(mask.sum()), fraction=float(mask.mean()),
                                    SSE_share=float(squared[mask].sum() / squared.sum()),
                                    RMSE=float(np.sqrt(squared[mask].mean())) if mask.any() else None,
                                    bias=float((pred - truth)[mask].mean()) if mask.any() else None))
                    del model
                persistence = np.repeat(data['features'][data['positions'][vi], q_index:q_index + 1], y.shape[1], axis=1)
                rows.append(dict(run=run_name, strategy=strategy, h=h, fold=label, seed=-1,
                    split='persistence_validation', n=len(vi), objective='none', **measures(y[vi], persistence)))
                drift.append(dict(run=run_name, strategy=strategy, h=h, fold=label,
                    train_n=len(ti), validation_n=len(vi),
                    train_mean=float(y[ti].mean()), validation_mean=float(y[vi].mean()),
                    train_p90=threshold, validation_p90=float(np.quantile(y[vi], .9)),
                    train_max=float(y[ti].max()), validation_max=float(y[vi].max())))
        print('[历史诊断完成]', run_name, flush=True)
    pd.DataFrame(rows).to_csv(out/'train_validation_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(segments).to_csv(out/'validation_error_segments.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(drift).to_csv(out/'quarter_distribution.csv', index=False, encoding='utf-8-sig')
    (out/'model_source_hashes.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(dict(status='completed', kind='historical_diagnosis_only',
        max_target='2003-12-31', read_2004_predictions=False, new_fits=0, model_count=len(sources),
        device=device), ensure_ascii=False, indent=2), encoding='utf-8')
    frame = pd.DataFrame(rows)
    print(frame[frame.seed != -1].groupby(['run', 'strategy', 'split'])[['R2', 'RMSE', 'MAE', 'log_MSE', 'mean_ratio']].mean().round(4).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        diagnose(args.out, args.device)


if __name__ == '__main__':
    main()
