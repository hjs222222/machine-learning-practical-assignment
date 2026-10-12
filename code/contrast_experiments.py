# -*- coding: utf-8 -*-
"""
对照实验（可复现）：用于报告表 4-5 / 表 4-6
============================================
表 4-5  保留特征个数对随机森林精度的影响（对数目标，测试年 2004）
表 4-6  目标变量变换方式对精度的影响（HistGradientBoosting 对照实验）

- 特征池：V6 版 88 个候选特征（含洪水信号特征）
- 表 4-5：按全局互信息（训练集，log 目标）取前 K 个特征，固定 RF 参数
- 表 4-6：按预见期互信息 TOP50（与主流程一致），4 种目标变换
- 严格时序：训练 2000-2003、测试 2004；评估起始日与主流程一致（359 天）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
sys.path.insert(0, r"D:\machine-learning-practical-assignment\code")
import config as C
from data_analysis import load_raw
from feature_engineering import build_features
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from experiment_v6 import select_features_per_horizon

d, feature_cols = build_features(load_raw())
tr_mask = d["Date"] <= pd.Timestamp(C.TRAIN_END)
tr, te = d[tr_mask], d[~tr_mask]
test_start = len(tr)                        # 1431，与主流程 START 对齐
obs_all = np.concatenate([tr[C.TARGET].values, te[C.TARGET].values])
idx = np.arange(test_start, len(obs_all) - 7)   # 359 个评估起始日（2004-01-01 ~ 12-24）

# 全局互信息（log 目标）
y_tr_log = np.log(np.maximum(tr[C.TARGET].values, 1e-6))
mi = mutual_info_regression(tr[feature_cols].values, y_tr_log,
                            random_state=C.RANDOM_STATE)
order = np.argsort(-mi)

def nse(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def ev_origins(h):
    return obs_all[idx + h]

def train_targets(h, mode):
    """训练样本的目标与基准流量（起始日 t_i ∈ [30, test_start-h]）。"""
    yq = obs_all[30 + h: test_start + 1]            # Q_{t+h}
    q0 = obs_all[30: test_start - h + 1]            # Q_t（与特征行对齐）
    if mode == "raw":
        return yq
    if mode == "log":
        return np.log(np.maximum(yq, 1e-6))
    if mode == "logratio":
        return np.log(np.maximum(yq, 1e-6)) - np.log(np.maximum(q0, 1e-6))
    return (yq - q0) / np.maximum(q0, 1e-6)

print("=" * 70)
print("表 4-5 对照：保留特征个数 × RF（对数目标，Grid Search CV + 时序 CV）")
print("=" * 70)
rows5 = []
for K in [10, 20, 40, 60, 88]:
    feats = [feature_cols[i] for i in order[:K]]
    vals = []
    for h in [1, 3, 7]:
        Xt = tr[feats].values[30: test_start - h + 1]
        yt = train_targets(h, "log")
        gs = GridSearchCV(RandomForestRegressor(random_state=C.RANDOM_STATE),
                          {"n_estimators": [300, 500], "max_depth": [None, 15],
                           "min_samples_leaf": [1, 3]},
                          cv=TimeSeriesSplit(3), scoring="neg_root_mean_squared_error",
                          n_jobs=-1)
        gs.fit(Xt, yt)
        ph = np.exp(gs.predict(te[feats].values[:359]))
        vals.append(nse(ev_origins(h), ph))
    rows5.append([str(K), f"{vals[0]:.3f}", f"{vals[1]:.3f}", f"{vals[2]:.3f}"])
    print(f"  K={K:<3}  h1={vals[0]:.3f}  h3={vals[1]:.3f}  h7={vals[2]:.3f}")

print("\n表 4-5 数字:", rows5)

print("=" * 70)
print("表 4-6 对照：目标变量变换 × HistGradientBoosting（TOP50 按预见期）")
print("=" * 70)
feats_h, _ = select_features_per_horizon(d, feature_cols)
rows6 = []
for name, mode in [("原始径流 y", "raw"), ("对数 log(y)", "log"),
                   ("对数比值 log(y/y_t)", "logratio"),
                   ("相对变化 (y-y_t)/y_t", "rel")]:
    vals = []
    for h in [1, 3, 7]:
        feats = feats_h[h]
        yt = train_targets(h, mode)
        hgb = HistGradientBoostingRegressor(max_iter=300, random_state=C.RANDOM_STATE)
        hgb.fit(tr[feats].values[30: len(tr) - h + 1], yt)
        p = hgb.predict(te[feats].values[:359])
        q0_te = obs_all[idx]                        # 测试期当前流量
        if mode == "raw":
            ph = p
        elif mode == "log":
            ph = np.exp(p)
        elif mode == "logratio":
            ph = q0_te * np.exp(p)
        else:
            ph = q0_te * (1 + p)
        vals.append(nse(ev_origins(h), ph))
    rows6.append([name, f"{vals[0]:.3f}", f"{vals[1]:.3f}", f"{vals[2]:.3f}"])
    print(f"  {name:<16}  h1={vals[0]:.3f}  h3={vals[1]:.3f}  h7={vals[2]:.3f}")

print("\n表 4-6 数字:", rows6)
