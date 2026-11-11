# -*- coding: utf-8 -*-
"""梯度提升基模型收益探查：在完全相同的特征/切分/预处理下对比 GB 与 RF。

目的：量化"还能提升多少"——若 GB 显著优于 RF，则值得重训基模型；
若接近，则说明树集成已达平台期，瓶颈在物理信息墙。

诚实约束：只用 train(≤2003) 拟合，2004 测试集只做一次评估；
GB 用合理默认参数（未在 2004 上选参），RF 用第三轮冻结参数。
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

from feature_engineering import build_features

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Teacher Resources & Requirements/01047000.csv"
HORIZONS = list(range(1, 8))


def r2(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    sst = np.square(y - y.mean(axis=0)).sum(axis=0)
    sse = np.square(p - y).sum(axis=0)
    return 1 - sse / sst


def build_data():
    raw = pd.read_csv(SOURCE, parse_dates=["Date"])
    engineered, names = build_features(raw, verbose=False)
    positions = np.arange(29, len(engineered) - 7)
    targets = np.stack([engineered.Discharge.to_numpy()[positions + h] for h in range(1, 8)], axis=1)
    x = engineered[names].to_numpy(float)
    days = engineered.Date.to_numpy()[positions].astype("datetime64[D]").astype("int64")
    last_days = days + 7
    # 追加 prepare_v2 的三列（当日流量 + 年内周期性编码）
    raw_tail = raw.iloc[30:].reset_index(drop=True)
    doy = raw_tail.Date.dt.dayofyear.to_numpy()
    x = np.column_stack([x, raw_tail.Discharge.to_numpy(float),
                         np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
    # positions 从 29 起，features 全表按 engineered 顺序；features[positions] 取对应行
    x = x[positions]
    train = np.flatnonzero(last_days <= np.datetime64("2003-12-31").astype("int64"))
    test = np.flatnonzero((days >= np.datetime64("2004-01-01").astype("int64")) &
                          (last_days <= np.datetime64("2004-12-31").astype("int64")))
    assert np.isfinite(x).all() and np.isfinite(targets).all()
    return x, targets, train, test


def prep(x, y_log_target):
    """MI top-50 + StandardScaler；返回缩放后的 train/test 特征与标记。"""
    try:
        mi = mutual_info_regression(x, y_log_target, random_state=42)
        sel = np.argsort(-mi, kind="stable")[:50]
    except Exception:
        sel = np.arange(x.shape[1])
    scaler = StandardScaler().fit(x[:, sel])
    return sel, scaler


def fit_predict_direct(x, targets, train, test):
    rows = []
    for h in HORIZONS:
        ylog = np.log(targets[:, h - 1])
        sel, scaler = prep(x[train], ylog[train])
        Xtr = scaler.transform(x[train][:, sel])
        Xte = scaler.transform(x[test][:, sel])
        rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                   random_state=42).fit(Xtr, ylog[train])
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=42).fit(Xtr, ylog[train])
        truth = targets[test, h - 1]
        prf = np.exp(rf.predict(Xte))
        pgb = np.exp(gb.predict(Xte))
        rows.append(dict(strategy="direct", h=h, RF=float(r2(truth, prf)), GB=float(r2(truth, pgb))))
    return rows


def fit_predict_multi(x, targets, train, test):
    ylog = np.log(targets)
    sel, scaler = prep(x[train], ylog[train, 0])
    Xtr = scaler.transform(x[train][:, sel])
    Xte = scaler.transform(x[test][:, sel])
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5,
                               random_state=42).fit(Xtr, ylog[train])
    gb = MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
        random_state=42)).fit(Xtr, ylog[train])
    truth = targets[test]
    prf = np.exp(rf.predict(Xte))
    pgb = np.exp(gb.predict(Xte))
    rrf = r2(truth, prf)
    rgb = r2(truth, pgb)
    return [dict(strategy="multi", h=h, RF=float(rrf[h - 1]), GB=float(rgb[h - 1])) for h in HORIZONS]


def main():
    x, targets, train, test = build_data()
    print(f"特征维度={x.shape[1]}  train={len(train)}  test={len(test)}")
    rows = fit_predict_direct(x, targets, train, test) + fit_predict_multi(x, targets, train, test)
    df = pd.DataFrame(rows)
    df["GB-RF"] = (df.GB - df.RF).round(4)

    print()
    print("GB(HistGBR) vs RF 的 2004 测试集 R2 对照（相同特征/切分/预处理，唯一差异是模型）")
    for strategy in ["direct", "multi"]:
        part = df[df.strategy == strategy][["h", "RF", "GB", "GB-RF"]].round(4)
        print(f"\n[{strategy}]")
        print(part.to_string(index=False))

    # 对照现有报告 RF 值（sanity check：我的 RF 复现应接近报告值）
    existing = pd.read_csv(ROOT / "验收/Acceptance 3/data/任务1至7+主seed42完整指标与来源.csv")
    exrf = existing[existing.model == "RF"].pivot_table(index="strategy", columns="h", values="R2").round(4)
    print("\n现有报告的 RF R2（对照 sanity check）")
    print(exrf.to_string())

    df.to_csv(ROOT / "验收/Acceptance 3/data/任务5及7+GB基模型收益探查.csv", index=False, encoding="utf-8-sig")
    print("\n已写出：验收/Acceptance 3/data/任务5及7+GB基模型收益探查.csv")


if __name__ == "__main__":
    main()