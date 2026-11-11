"""历史简单基线：仅Q(t)的log线性回归、AR/ARX；2002内层选择，2003前向检验。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset
from acceptance3_train import ROOT
from acceptance3_v3_stage_b import historical_splits
from runoff_training.train import year_folds


def physical_score(model, x, y):
    pred = np.exp(model.predict(x))
    truth = np.exp(y)
    return -float(np.sqrt(np.square(pred - truth).mean(axis=0)).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    path = ROOT/'output/acceptance3_runs/formal_20261111_gpu_lag_residual/dataset.npz'
    data = load_dataset(str(path))
    names = data['feature_names'].tolist()
    features = data['features'][data['positions']]
    days = data['origin_days']
    train = np.flatnonzero(days + 7 <= np.datetime64('2003-12-31').astype('int64'))
    search_ids = train[days[train] + 7 <= np.datetime64('2002-12-31').astype('int64')]
    cv = historical_splits(search_ids, days)
    representations = {
        'Q_current': ['Discharge_current'],
        'AR': ['Discharge_current'] + [f'Discharge_lag{k}' for k in [1, 2, 3, 7, 14, 30]],
        'ARX': ['Discharge_current'] + [f'Discharge_lag{k}' for k in [1, 2, 3, 7, 14, 30]] +
            ['Prcp', 'Tmax', 'Tmin', 'Vp', 'Srad', 'Dayl', 'doy_sin', 'doy_cos'] +
            [f'Prcp_lag{k}' for k in [1, 2, 3, 7, 14, 30]],
    }
    rows, selections = [], []
    y = np.log(data['y'])
    for representation, selected in representations.items():
        x = features[:, [names.index(name) for name in selected]]
        # 所有所选输入均使用固定符号对数，包括周期变量；保持已完成诊断的变换。
        x = np.sign(x) * np.log1p(np.abs(x))
        estimator = make_pipeline(StandardScaler(), Ridge())
        grid = GridSearchCV(estimator, {'ridge__alpha': [1., 10., 100., 1000.]},
            scoring=physical_score, cv=cv, n_jobs=1, refit=False, error_score='raise')
        grid.fit(x[search_ids], y[search_ids])
        alpha = grid.best_params_['ridge__alpha']
        selections.append(dict(representation=representation, alpha=alpha,
            inner_RMSE=-grid.best_score_, features=selected))
        for label, ti, vi in year_folds(data, train):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[ti], y[ti])
            pred = np.exp(model.predict(x[vi]))
            truth = data['y'][vi]
            for h in range(7):
                sst = np.square(truth[:, h] - truth[:, h].mean()).sum()
                rows.append(dict(representation=representation, fold=label, h=h + 1,
                    R2=1 - np.square(pred[:, h] - truth[:, h]).sum() / sst,
                    RMSE=float(np.sqrt(np.square(pred[:, h] - truth[:, h]).mean())),
                    MAE=float(np.abs(pred[:, h] - truth[:, h]).mean())))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out/'quarter_metrics.csv', index=False, encoding='utf-8-sig')
    (args.out/'selection.json').write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.out/'manifest.json').write_text(json.dumps(dict(status='completed', max_search_target='2002-12-31',
        max_validation_target='2003-12-31', read_2004=False, fits=3 * (4 * 4 + 4),
        purpose='轻量诊断，不替代课程ANN/RF/LSTM，不用2004筛选特征'), ensure_ascii=False, indent=2), encoding='utf-8')
    print(frame.groupby('representation')[['R2', 'RMSE', 'MAE']].mean().to_string())
    print(frame.groupby(['representation', 'h'])[['RMSE', 'MAE']].mean().to_string())


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
