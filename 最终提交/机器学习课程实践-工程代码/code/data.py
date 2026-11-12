"""数据校验、标签对齐、按预测原点及标签截止日隔离。"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from config import DATA_PATH
from features import build_features


def load_raw():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date']).set_index('Date')
    expected = ['Discharge','Dayl','Prcp','Srad','Swe','Tmax','Tmin','Vp']
    if df.shape != (1827, 8) or list(df.columns) != expected:
        raise ValueError('CSV 行数或字段与教师数据合同不符')
    if not df.index.equals(pd.date_range('2000-01-01','2004-12-31', name='Date')):
        raise ValueError('日期缺失、重复或不按日递增')
    if not np.isfinite(df.to_numpy()).all() or (df.Discharge < 0).any():
        raise ValueError('无效观测，禁止静默修改原始数据')
    return df


def build_dataset(raw, sequence_length=14):
    features, catalog = build_features(raw)
    targets = pd.concat([raw.Discharge.shift(-h).rename(f'h{h}') for h in range(1,8)], axis=1)
    ready = features.notna().all(axis=1).to_numpy()
    ids = np.array([i for i in range(sequence_length - 1, len(raw) - 7)
                    if ready[i-sequence_length+1:i+1].all()], dtype=int)
    X = np.stack([features.iloc[i-sequence_length+1:i+1].to_numpy(np.float32) for i in ids])
    dates = raw.index[ids]
    train = np.flatnonzero(dates + pd.Timedelta(days=7) <= pd.Timestamp('2003-12-31'))
    test = np.flatnonzero((dates >= '2004-01-01') & (dates <= '2004-12-24'))
    return dict(X=X, y=targets.iloc[ids].to_numpy(), dates=dates, train=train, test=test,
                features=features, catalog=catalog, names=features.columns.to_list())


def development_splits(dates, n_splits=2):
    # gap=7 保证训练末行的 t+7 早于验证首行的 t。
    dates = pd.DatetimeIndex(dates)
    splits = list(TimeSeriesSplit(n_splits=n_splits, gap=7).split(dates))
    for tr, va in splits:
        if dates[tr[-1]] + pd.Timedelta(days=7) >= dates[va[0]]:
            raise ValueError('时序折标签重叠')
    return splits
