"""窗口/状态受控实验的数据契约：按日期建样本，仅保留2003年及以前。"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


RAW = ['Discharge', 'Prcp', 'Tmax', 'Tmin', 'Vp', 'Srad', 'Dayl', 'Swe']
BASE = ['Q', 'Prcp', 'Tmax', 'Tmin', 'Vp', 'Srad', 'Dayl', 'doy_sin', 'doy_cos']
STATE = ['P_sum7', 'P_sum30', 'P_sum90', 'API095', 'Q_change1', 'Q_change7',
         'Q_mean30', 'Q_mean90']
MAX_WINDOW = 90
WARMUP = 89
PROFILES = {
    'w30_base': (30, False), 'w90_base': (90, False),
    'w30_state': (30, True), 'w90_state': (90, True),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_data(csv_path, out_path):
    frame = pd.read_csv(csv_path)
    if set(frame.columns) != {'Date', *RAW}:
        raise ValueError('原始字段不符合白名单，拒绝意外目标列或缺列')
    dates = pd.to_datetime(frame.Date, format='%Y-%m-%d', errors='raise')
    days = dates.to_numpy(dtype='datetime64[D]').astype('int64')
    if not (np.diff(days) == 1).all():
        raise ValueError('日期必须连续、唯一、递增')
    if not np.isfinite(frame[RAW].to_numpy(float)).all():
        raise ValueError('原始数值必须有限')
    # 读取完整文件以核验契约，但2004数据不进入特征、目标或任何拟合数组。
    frame = frame.loc[dates <= pd.Timestamp('2003-12-31')].reset_index(drop=True)
    dates = pd.to_datetime(frame.Date)
    days = dates.to_numpy(dtype='datetime64[D]').astype('int64')
    if len(frame) < 730 or (frame.Discharge <= 0).any() or (frame.Prcp < 0).any():
        raise ValueError('历史长度不足，或log目标/非负降雨定义域不合法')
    q, p = frame.Discharge, frame.Prcp
    angle = 2 * np.pi * dates.dt.dayofyear.to_numpy() / 365.25
    base = np.column_stack([q, p, frame.Tmax, frame.Tmin, frame.Vp,
                            frame.Srad, frame.Dayl, np.sin(angle), np.cos(angle)])
    api = np.empty(len(frame), dtype=float)
    current = 0.
    for i, rain in enumerate(p):
        current = .95 * current + float(rain)
        api[i] = current
    state = np.column_stack([p.rolling(k, min_periods=k).sum() for k in [7, 30, 90]]
        + [api, q.diff(1), q.diff(7), q.rolling(30).mean(), q.rolling(90).mean()])
    # 统一17列，基准把8个状态通道置零，网络容量/初始化不随开关改变。
    features = np.column_stack([base, state])
    transformed = features.copy()
    logged = [0, 1, *range(len(BASE), len(BASE) + len(STATE))]
    transformed[:, logged] = np.sign(features[:, logged]) * np.log1p(np.abs(features[:, logged]))
    positions = np.arange(WARMUP + MAX_WINDOW - 1, len(frame) - 7, dtype=int)
    y = q.to_numpy()[positions[:, None] + np.arange(1, 8)]
    if not np.isfinite(transformed[WARMUP:]).all():
        raise ValueError('预热后特征仍有非有限值')
    if not len(positions) or days[positions[-1]] + 7 > np.datetime64('2003-12-31').astype('int64'):
        raise ValueError('样本边界错误')
    np.savez_compressed(out_path, features=transformed, positions=positions, y=y,
        days=days, origin_days=days[positions], feature_names=np.asarray(BASE + STATE),
        discharge=q.to_numpy(), max_window=MAX_WINDOW, warmup=WARMUP)
    return dict(rows=len(frame), samples=len(positions), first_origin=str(dates.iloc[positions[0]].date()),
        last_origin=str(dates.iloc[positions[-1]].date()), max_target=str(dates.iloc[-1].date()),
        raw_sha256=sha256(csv_path), dataset_sha256=sha256(out_path))


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_ids(ids, size):
    values = np.asarray(ids)
    if values.ndim not in [1, 2] or values.ndim == 2 and values.shape[1] != 1:
        raise ValueError('索引仅接受一维或(n,1)，禁止把多列静默展平')
    values = values.reshape(-1)
    if not len(values) or not np.issubdtype(values.dtype, np.number):
        raise ValueError('索引不能为空且必须为数值')
    if not np.isfinite(values).all() or not (values == np.floor(values)).all():
        raise ValueError('索引必须为有限整数')
    if (values < 0).any() or (values >= size).any():
        raise ValueError('索引越界')
    return values.astype(int)


def outer_split(data, year):
    days = data['origin_days']
    first = np.datetime64(f'{year}-01-01').astype('int64')
    last = np.datetime64(f'{year}-12-31').astype('int64')
    train = np.flatnonzero(days + 7 < first)
    valid = np.flatnonzero((days >= first) & (days + 7 <= last))
    if not len(train) or not len(valid) or days[train[-1]] + 7 >= days[valid[0]]:
        raise ValueError('年度切分净化失败')
    return train, valid


def inner_splits(data, train):
    """外层历史末尾三块90天验证；返回相对于train的索引。"""
    days = data['origin_days'][train]
    folds = []
    for stop in [len(train) - 180, len(train) - 90, len(train)]:
        vi = np.arange(stop - 90, stop)
        ti = np.flatnonzero(days + 7 < days[vi[0]])
        if len(ti) < 250 or days[ti[-1]] + 7 >= days[vi[0]]:
            raise ValueError('内层数据不足或净化失败')
        folds.append((ti, vi))
    return folds
