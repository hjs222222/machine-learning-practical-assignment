"""直接残差最终评价的数据契约：独立构造全期因果输入，并严格核对历史前缀。"""
from pathlib import Path

import numpy as np
import pandas as pd

from runoff_training.controlled_data import BASE, STATE, RAW, MAX_WINDOW, WARMUP, load_data, sha256


def build_evaluation_data(csv_path, historical_path, out_path):
    frame = pd.read_csv(csv_path)
    if set(frame.columns) != {'Date', *RAW}:
        raise ValueError('字段不符合原始白名单')
    dates = pd.to_datetime(frame.Date, format='%Y-%m-%d', errors='raise')
    days = dates.to_numpy(dtype='datetime64[D]').astype('int64')
    if (not (np.diff(days) == 1).all() or str(dates.iloc[0].date()) != '2000-01-01'
            or str(dates.iloc[-1].date()) != '2004-12-31'):
        raise ValueError('最终评价要求课程给定完整五年连续日期')
    if not np.isfinite(frame[RAW].to_numpy(float)).all() or (frame.Discharge <= 0).any() or (frame.Prcp < 0).any():
        raise ValueError('原始值不满足数值契约')
    q, p = frame.Discharge, frame.Prcp
    angle = 2 * np.pi * dates.dt.dayofyear.to_numpy() / 365.25
    base = np.column_stack([q, p, frame.Tmax, frame.Tmin, frame.Vp,
                            frame.Srad, frame.Dayl, np.sin(angle), np.cos(angle)])
    api = np.empty(len(frame))
    current = 0.
    for i, rain in enumerate(p):
        current = .95 * current + float(rain)
        api[i] = current
    state = np.column_stack([p.rolling(k, min_periods=k).sum() for k in [7, 30, 90]]
        + [api, q.diff(1), q.diff(7), q.rolling(30).mean(), q.rolling(90).mean()])
    values = np.column_stack([base, state])
    transformed = values.copy()
    logged = [0, 1, *range(len(BASE), len(BASE) + len(STATE))]
    transformed[:, logged] = np.sign(values[:, logged]) * np.log1p(np.abs(values[:, logged]))
    positions = np.arange(WARMUP + MAX_WINDOW - 1, len(frame) - 7, dtype=int)
    result = dict(features=transformed, positions=positions,
        y=q.to_numpy()[positions[:, None] + np.arange(1, 8)], days=days,
        origin_days=days[positions], feature_names=np.asarray(BASE + STATE),
        discharge=q.to_numpy(), max_window=np.asarray(MAX_WINDOW), warmup=np.asarray(WARMUP))
    historical = load_data(historical_path)
    if set(result) != set(historical):
        raise ValueError('历史字段布局改变')
    for key, previous in historical.items():
        actual = result[key]
        prefix = actual[:len(previous)] if previous.ndim else actual
        np.testing.assert_equal(prefix, previous, err_msg=f'历史前缀不一致：{key}')
    if not np.isfinite(transformed[WARMUP:]).all():
        raise ValueError('预热后特征非有限')
    np.savez_compressed(out_path, **result)
    return dict(raw_sha256=sha256(csv_path), dataset_sha256=sha256(out_path),
                historical_sha256=sha256(historical_path), rows=len(frame), samples=len(positions),
                max_target='2004-12-31', historical_prefix_verified=True)


def final_split(data):
    days = data['origin_days']
    first = np.datetime64('2004-01-01').astype('int64')
    last = np.datetime64('2004-12-31').astype('int64')
    train = np.flatnonzero(days + 7 <= np.datetime64('2003-12-31').astype('int64'))
    valid = np.flatnonzero((days >= first) & (days + 7 <= last))
    if len(train) != 1276 or len(valid) != 359 or days[train[-1]] + 7 >= days[valid[0]]:
        raise ValueError('最终训练/评价日期或共同样本数改变')
    return train, valid
