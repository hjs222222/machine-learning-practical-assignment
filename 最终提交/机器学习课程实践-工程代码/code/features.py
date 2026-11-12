"""在预测原点 t 构造仅使用 <=t 观测的特征。"""
import numpy as np
import pandas as pd


def build_features(raw):
    values, catalog = {}, []

    def add(name, series, source, lag=0, window=1):
        values[name] = series
        catalog.append(dict(feature=name, source=source, lag_days=lag, window_days=window, latest_source='t' if lag == 0 else f't-{lag}'))

    for col in raw.columns:
        if col == 'Swe':
            continue
        for lag in (0, 1, 2, 3, 7, 14, 30):
            add(f'{col}_lag{lag}', raw[col].shift(lag), col, lag)
    for col in ('Discharge', 'Prcp', 'Tmax', 'Tmin'):
        for window in (3, 7, 14):
            rolling = raw[col].rolling(window)
            for stat in ('mean', 'std', 'max'):
                add(f'{col}_roll{window}_{stat}', getattr(rolling, stat)(), col, window=window)
    for period in (12,):
        angle = 2 * np.pi * (raw.index.month - 1) / period
        add('month_sin', np.sin(angle), 'Date')
        add('month_cos', np.cos(angle), 'Date')
    season = (raw.index.month % 12) // 3
    for i, name in enumerate(('winter','spring','summer','autumn')):
        add(f'season_{name}', (season == i).astype(float), 'Date')
    return pd.DataFrame(values, index=raw.index), pd.DataFrame(catalog)
