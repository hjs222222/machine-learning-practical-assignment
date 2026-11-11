"""受控学习曲线：固定结构/优化器，只比较轮数；仅2002/2003历史，不访问2004。"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset
from acceptance3_train import ROOT
from runoff_training.model import LaggedLSTMRegressor


PROFILES = dict(compact=dict(feature_mode='compact', input_transform='identity'),
                compact_lag=dict(feature_mode='compact_lag', input_transform='signed_log'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    path = ROOT/'output/acceptance3_runs/formal_20261111_gpu_lag_residual/dataset.npz'
    data = load_dataset(str(path))
    days = data['origin_days']
    results = []
    for year in [2002, 2003]:
        begin = np.datetime64(f'{year}-01-01').astype('int64')
        end = np.datetime64(f'{year}-12-31').astype('int64')
        ti = np.flatnonzero(days + 7 < begin)
        vi = np.flatnonzero((days >= begin) & (days + 7 <= end))
        assert days[ti[-1]] + 7 < days[vi[0]] and days[vi[-1]] + 7 <= np.datetime64('2003-12-31').astype('int64')
        for strategy, h in [('direct', 1), ('direct', 7), ('multi', 7)]:
            y = data['y'][:, h - 1:h] if strategy == 'direct' else data['y']
            for profile, params in PROFILES.items():
                for seed in [42, 43]:
                    model = LaggedLSTMRegressor(str(path), horizon=h, units=(32,), dropout=.1,
                        lr=.001, batch_size=64, fixed_epochs=args.epochs, seed=seed, device='cuda',
                        residual=False, weight_decay=0., **params)
                    model.output_dim_ = y.shape[1]
                    model._preprocess_fit(ti, y[ti])
                    model.model_ = model._network()
                    optimizer = torch.optim.Adam(model.model_.parameters(), lr=model.lr)
                    x = torch.as_tensor(model._inputs(ti), device='cuda')
                    z = torch.as_tensor(model.y_scaler_.transform(np.log(y[ti])).astype('float32'), device='cuda')
                    for epoch in range(1, args.epochs + 1):
                        model.model_.train()
                        total = 0.
                        for start in range(0, len(ti), model.batch_size):
                            optimizer.zero_grad(set_to_none=True)
                            p = model.model_(x[start:start + model.batch_size])
                            loss = torch.nn.functional.mse_loss(p, z[start:start + model.batch_size])
                            if not torch.isfinite(loss):
                                raise ValueError('学习曲线损失非有限')
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.model_.parameters(), 1.)
                            optimizer.step()
                            total += loss.item() * len(p)
                        if epoch == 1 or epoch % 5 == 0:
                            for split, ids in [('train', ti), ('validation', vi)]:
                                pred = np.asarray(model.predict(ids[:, None])).reshape(len(ids), -1)
                                truth = y[ids]
                                results.append(dict(year=year, strategy=strategy, h=h, profile=profile,
                                    seed=seed, epoch=epoch, split=split, n=len(ids),
                                    RMSE=float(np.sqrt(np.square(truth - pred).mean(axis=0)).mean()),
                                    MAE=float(np.abs(truth - pred).mean()),
                                    log_MSE=float(np.square(np.log(truth) - np.log(pred)).mean()),
                                    batch_loss=total / len(ti)))
                    pd.DataFrame(results).to_csv(args.out/'learning_curves.csv', index=False, encoding='utf-8-sig')
                    print('[受控曲线完成]', year, strategy, h, profile, seed, flush=True)
    frame = pd.DataFrame(results)
    primary = frame[frame.split == 'validation']
    columns = ['year', 'strategy', 'h', 'profile', 'seed']
    best = primary.loc[primary.groupby(columns).RMSE.idxmin()].copy()
    best.to_csv(args.out/'best_epoch_diagnostic_only.csv', index=False, encoding='utf-8-sig')
    (args.out/'manifest.json').write_text(json.dumps(dict(status='completed', new_diagnostic_trajectories=24,
        max_epoch=args.epochs, max_validation_target='2003-12-31', read_2004=False,
        kind='非交付选参；只诊断欠拟合/过拟合，不根据本表回填历史OOF',
        constant_config=dict(units=[32], dropout=.1, lr=.001, batch_size=64, optimizer='Adam',
                             residual=False, shuffle=False, weight_decay=0.)), ensure_ascii=False, indent=2), encoding='utf-8')
    print(best[columns + ['epoch', 'RMSE', 'MAE']].to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
