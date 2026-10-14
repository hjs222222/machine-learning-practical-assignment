# -*- coding: utf-8 -*-
"""
第二部分：特征工程
==================
1. 时间序列特征提取
   (1) 季节编码（四季 one-hot）与月份编码
   (2) 滞后特征（lag）：径流与气象因子的 1/2/3/7/14/30 天滞后
   (3) 窗口特征（rolling）：径流与降水的 3/7/14/30 天滑动均值、标准差、最大值
2. 特征归一化：实现最小-最大归一化(Min-Max)与 Z-score 标准化，并对比
3. 特征选择：实现皮尔逊相关系数法与互信息法，并给出特征重要性排序

【防数据泄漏】所有统计量（归一化的均值/方差、特征选择的评分）只在训练集上估计，
再应用到测试集；滞后与滑动窗口只向过去取数，不使用未来信息。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

import config as C
from utils_plot import save, PALETTE

SEASON_MAP = {12: "冬", 1: "冬", 2: "冬",
              3: "春", 4: "春", 5: "春",
              6: "夏", 7: "夏", 8: "夏",
              9: "秋", 10: "秋", 11: "秋"}
SEASONS = ["春", "夏", "秋", "冬"]


# ===========================================================================
# 1. 时间序列特征提取
# ===========================================================================
def add_time_features(df):
    """(1) 季节编码与月份编码。"""
    d = df.copy()
    month = d["Date"].dt.month

    # 四季 one-hot
    season = month.map(SEASON_MAP)
    for s in SEASONS:
        d[f"season_{s}"] = (season == s).astype(int)
    d["season"] = season
    d["month"] = month
    return d


def add_lag_features(df):
    """(2) 滞后特征：x(t-1), x(t-2), ... 只使用历史信息。"""
    d = df.copy()
    for col in C.LAG_VARS:
        for k in C.LAG_STEPS:
            d[f"{col}_lag{k}"] = d[col].shift(k)
    return d


def add_rolling_features(df):
    """(3) 窗口特征：滑动均值/标准差/最大值。

    closed='both' + min_periods=w 保证窗口仅覆盖 [t-w+1, t]，不含未来。
    """
    d = df.copy()
    for col in C.ROLL_VARS:
        for w in C.ROLL_WINDOWS:
            r = d[col].rolling(window=w, min_periods=w)
            d[f"{col}_roll{w}_mean"] = r.mean()
            d[f"{col}_roll{w}_std"] = r.std()
            d[f"{col}_roll{w}_max"] = r.max()
    return d


def add_derived_features(df):
    """派生气象因子：日均温、日温差（对融雪型流域的径流过程有指示意义）。"""
    d = df.copy()
    d["Tmean"] = (d["Tmax"] + d["Tmin"]) / 2.0
    d["Trange"] = d["Tmax"] - d["Tmin"]
    # 前期累积降水（反映流域蓄水/土壤含水量，是洪水预报的关键因子）
    d["Prcp_cum7"] = d["Prcp"].rolling(7, min_periods=7).sum()
    d["Prcp_cum30"] = d["Prcp"].rolling(30, min_periods=30).sum()
    d["Tmean_roll7"] = d["Tmean"].rolling(7, min_periods=7).mean()
    # 融雪指数：日均温高于 0 degC 的积温（近似融雪驱动）
    d["melt_index"] = d["Tmean"].clip(lower=0).rolling(7, min_periods=7).sum()
    # ---- 洪水物理信号特征（后续训练增强，全部只用历史信息）----
    # 径流一阶差分：涨水/退水趋势（洪峰上升段的直接指示）
    d["Discharge_diff1"] = d["Discharge"] - d["Discharge"].shift(1)
    d["Discharge_diff3"] = d["Discharge"] - d["Discharge"].shift(3)
    d["Discharge_diff7"] = d["Discharge"] - d["Discharge"].shift(7)
    # 降水累积补全 3/14 天窗口（暴雨产流：短时强降水的直接累积量）
    d["Prcp_roll3_sum"] = d["Prcp"].rolling(3, min_periods=3).sum()
    d["Prcp_roll14_sum"] = d["Prcp"].rolling(14, min_periods=14).sum()
    # 强降水天数计数：过去 3 天降水量 >10mm 的天数（0~3，暴雨频率）
    d["Prcp_heavy3"] = (d["Prcp"] > 10.0).astype(float).rolling(
        3, min_periods=3).sum()
    # 降水突增：当日较前一日降水增量（暴雨启动信号）
    d["Prcp_diff1"] = d["Prcp"] - d["Prcp"].shift(1)
    return d


def build_features(df, verbose=True):
    """构建完整特征表并剔除无效行。"""
    d = add_time_features(df)
    d = add_derived_features(d)
    d = add_lag_features(d)
    d = add_rolling_features(d)

    # 常量列剔除
    drop_const = [c for c in C.CONSTANT_FEATURES if c in d.columns]
    # 基础列（日期/季节标签/目标）不进入特征集合
    meta_cols = ["Date", "season"]

    feature_cols = [c for c in d.columns
                    if c not in meta_cols + [C.TARGET] + drop_const]

    before = len(d)
    d = d.dropna(subset=feature_cols + [C.TARGET]).reset_index(drop=True)
    if verbose:
        print(f"  特征构造：原始 {before} 天 -> 有效 {len(d)} 天"
              f"（前 {before - len(d)} 天因滞后/窗口特征缺测被剔除）")
        print(f"  自变量个数：{len(feature_cols)}"
              f"（剔除常量列 {drop_const}）")
    return d, feature_cols


# ===========================================================================
# 2. 特征归一化
# ===========================================================================
class MinMaxScaler:
    """最小-最大归一化：x' = (x - min) / (max - min)  ->  [0, 1]"""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        rng = self.max_ - self.min_
        rng[rng == 0] = 1.0          # 防止常量列除零
        return (X - self.min_) / rng

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        return X * (self.max_ - self.min_) + self.min_


class ZScoreScaler:
    """Z-score 标准化：x' = (x - mu) / sigma  ->  均值 0、方差 1"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        return X * self.std_ + self.mean_


SCALERS = {"MinMax": MinMaxScaler, "ZScore": ZScoreScaler}


def _feat_category(name):
    """特征类别：用于归一化对比按类别分组展示（含滞后/窗口特征）。"""
    if name.startswith("season_"):
        return "季节编码"
    if name == "month":
        return "月份编码"
    if "_lag" in name:
        return "滞后特征"
    if name in ("Tmean", "Trange", "Prcp_cum7", "Prcp_cum30",
                "Tmean_roll7", "melt_index",
                "Discharge_diff1", "Discharge_diff3", "Discharge_diff7",
                "Prcp_roll3_sum", "Prcp_roll14_sum", "Prcp_heavy3",
                "Prcp_diff1"):
        return "派生气象特征"
    if "_roll" in name:
        return "窗口特征"
    return "原始气象因子"


def compare_scalers(df, feature_cols):
    """对比两种归一化方法：逐特征输出全部特征（含滞后/窗口）归一化前后统计量。

    防泄漏：scaler 仅在【训练集】上拟合统计量，再应用到全量数据。
    03 CSV 为逐特征统计（基于训练集，与实际建模一致）；
    fig7 按特征类别分组展示原始/Min-Max/Z-score 三种取值分布，
    明确覆盖滞后特征、窗口特征与派生气象特征。
    """
    # 仅训练集用于拟合 scaler
    tr_mask = df["Date"] <= pd.Timestamp(C.TRAIN_END)
    X_tr = df.loc[tr_mask, feature_cols].values
    X_all = df[feature_cols].values

    mm = MinMaxScaler().fit(X_tr)          # 仅训练集拟合
    zs = ZScoreScaler().fit(X_tr)
    Xs_mm_tr = mm.transform(X_tr)
    Xs_zs_tr = zs.transform(X_tr)

    # ---- 03 CSV：逐特征归一化前后统计量（训练集口径） ----
    rows = []
    for i, c in enumerate(feature_cols):
        x0, x1, x2 = X_tr[:, i], Xs_mm_tr[:, i], Xs_zs_tr[:, i]
        rows.append({
            "特征": c,
            "类别": _feat_category(c),
            "原始最小值": round(float(x0.min()), 4),
            "原始最大值": round(float(x0.max()), 4),
            "原始均值": round(float(x0.mean()), 4),
            "原始标准差": round(float(x0.std()), 4),
            "MinMax最小值": round(float(x1.min()), 4),
            "MinMax最大值": round(float(x1.max()), 4),
            "MinMax均值": round(float(x1.mean()), 4),
            "MinMax标准差": round(float(x1.std()), 4),
            "ZScore最小值": round(float(x2.min()), 4),
            "ZScore最大值": round(float(x2.max()), 4),
            "ZScore均值": round(float(x2.mean()), 4),
            "ZScore标准差": round(float(x2.std()), 4),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(C.RESULT_DIR, "03_归一化方法对比.csv"),
               index=False, encoding="utf-8-sig")

    # ---- 可视化：按特征类别分组，覆盖滞后/窗口/派生特征 ----
    CATS = [
        ("原始气象因子", [c for c in
                         ["Discharge", "Prcp", "Srad", "Vp", "Tmax"]
                         if c in feature_cols]),
        ("滞后特征", [c for c in feature_cols if "_lag" in c]),
        ("窗口特征", [c for c in feature_cols if "_roll" in c]),
        ("派生气象特征", [c for c in feature_cols if c in
                         ("Tmean", "Trange", "Prcp_cum7", "Prcp_cum30",
                          "Tmean_roll7", "melt_index",
                          "Discharge_diff1", "Discharge_diff3", "Discharge_diff7",
                          "Prcp_roll3_sum", "Prcp_roll14_sum", "Prcp_heavy3",
                          "Prcp_diff1")]),
    ]
    # 每类取 5 个代表（滞后/窗口优先径流与降水，量纲差异最直观）
    def pick(cols, prefs, n=5):
        ord_ = sorted(cols,
                      key=lambda c: (0 if any(p in c for p in prefs) else 1, c))
        return ord_[:n]

    sel_cols = []
    cat_of = {}
    for name, cols in CATS:
        if name == "原始气象因子":
            rep = cols[:5]
        elif name == "滞后特征":
            rep = pick(cols, ["Discharge", "Prcp"])
        elif name == "窗口特征":
            rep = pick(cols, ["Discharge"])
        else:
            rep = cols[:5]
        for c in rep:
            sel_cols.append(c)
            cat_of[c] = name

    idx = [feature_cols.index(c) for c in sel_cols]
    cat_colors = {"原始气象因子": PALETTE[0], "滞后特征": PALETTE[2],
                  "窗口特征": PALETTE[1], "派生气象特征": "#C05A2E"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (name, data) in zip(axes, [
            ("(a) 原始数据", X_all[:, idx]),
            ("(b) Min-Max 归一化", mm.transform(X_all)[:, idx]),
            ("(c) Z-score 标准化", zs.transform(X_all)[:, idx])]):
        bp = ax.boxplot([data[:, i] for i in range(len(sel_cols))],
                        patch_artist=True,
                        flierprops=dict(marker=".", markersize=2, alpha=.3))
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(cat_colors[cat_of[sel_cols[j]]])
            patch.set_alpha(0.65)
        # 类别分隔线与标注
        seen, x_last = {}, -0.5
        for j, c in enumerate(sel_cols):
            cat = cat_of[c]
            if cat not in seen:
                if j > 0:
                    ax.axvline(j - 0.5, color="gray", ls=":", lw=1.0)
                seen[cat] = j
        for cat, j0 in seen.items():
            n0 = sum(1 for c in sel_cols if cat_of[c] == cat)
            ax.text((j0 + j0 + n0 - 1) / 2, ax.get_ylim()[1] * 1.02 if False else 1.0,
                    cat, ha="center", va="bottom", fontsize=9,
                    transform=ax.get_xaxis_transform())
        ax.set_title(name, fontsize=11)
        ax.set_xticks(range(len(sel_cols)))
        ax.set_xticklabels(sel_cols, rotation=45, ha="right", fontsize=7.5)
    plt.tight_layout()
    save(fig, "fig7_归一化对比.png")

    # ---- 07 CSV：全特征（含滞后/窗口）Z-score 归一化后的数据表 ----
    Xs_all = zs.transform(X_all)
    norm = df.copy()
    for i, c in enumerate(feature_cols):
        norm[c] = Xs_all[:, i]
    norm.to_csv(os.path.join(C.RESULT_DIR, "07_归一化后特征数据.csv"),
                index=False, encoding="utf-8-sig")
    return tab


# ===========================================================================
# 3. 特征选择
# ===========================================================================
def select_features(df, feature_cols, top_k=None, verbose=True):
    """在【训练集】上实现皮尔逊相关系数法与互信息法。

    返回
    ----
    report : DataFrame，每个特征的 |Pearson|、互信息得分及排序
    selected : 依据互信息排序取前 top_k 个特征名
    """
    top_k = top_k or C.TOP_K_FEATURES
    tr = df[df["Date"] <= pd.Timestamp(C.TRAIN_END)]

    X = tr[feature_cols].values
    # 皮尔逊相关系数：按课程补充讲义的要求，基于【原始径流序列】计算
    # （以原始序列 {Q_t} 与滞后序列/其他因子序列的相关系数为准）。
    y_raw = tr[C.TARGET].values
    pear = np.array([abs(np.corrcoef(X[:, i], y_raw)[0, 1])
                     if X[:, i].std() > 0 else 0.0
                     for i in range(X.shape[1])])

    # 互信息法：对单调变换不变（两种口径理论等价），
    # 这里沿用建模目标口径 log(y)，与模型训练保持一致。
    y = np.log(np.maximum(y_raw, 1e-6)) if C.TARGET_TRANSFORM == "log" else y_raw
    mi = mutual_info_regression(X, y, random_state=C.RANDOM_STATE)
    mi = np.nan_to_num(mi)

    rep = pd.DataFrame({
        "特征": feature_cols,
        "|皮尔逊相关系数|": pear,
        "互信息得分": mi,
    })
    rep["皮尔逊排名"] = rep["|皮尔逊相关系数|"].rank(ascending=False).astype(int)
    rep["互信息排名"] = rep["互信息得分"].rank(ascending=False).astype(int)
    rep = rep.sort_values("互信息得分", ascending=False).reset_index(drop=True)
    rep.to_csv(os.path.join(C.RESULT_DIR, "04_特征选择得分.csv"),
               index=False, encoding="utf-8-sig")

    selected = rep["特征"].head(top_k).tolist()
    if verbose:
        print(f"  特征选择：共 {len(feature_cols)} 个候选特征，"
              f"按互信息法保留前 {top_k} 个")
        print(f"  互信息 Top10：{rep['特征'].head(10).tolist()}")

    _plot_feature_scores(rep, top_k)
    return rep, selected


def _plot_feature_scores(rep, top_k):
    top = rep.head(min(25, len(rep))).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    colors = [PALETTE[2] if f in set(rep["特征"].head(top_k)) else "#B0B8C1"
              for f in top["特征"]]

    axes[0].barh(top["特征"], top["互信息得分"], color=colors, alpha=0.9)
    axes[0].set_title(f"(a) 互信息法特征重要性 Top{len(top)}", fontsize=11)
    axes[0].set_xlabel("互信息得分 (nats)")
    axes[0].tick_params(axis="y", labelsize=8)

    top2 = rep.head(min(25, len(rep))).sort_values("|皮尔逊相关系数|").iloc[-25:]
    colors2 = [PALETTE[2] if f in set(rep["特征"].head(top_k)) else "#B0B8C1"
               for f in top2["特征"]]
    axes[1].barh(top2["特征"], top2["|皮尔逊相关系数|"], color=colors2, alpha=0.9)
    axes[1].set_title("(b) 皮尔逊相关系数法特征重要性", fontsize=11)
    axes[1].set_xlabel("|皮尔逊相关系数 r|")
    axes[1].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    save(fig, "fig8_特征选择.png")


def _plot_feature_groups(rep):
    """按特征类别汇总互信息贡献。"""
    def group(name):
        if name.startswith("season_"):
            return "季节编码"
        if name == "month":
            return "月份编码"
        if "_lag" in name:
            return "滞后特征"
        if name in ("Tmean", "Trange", "Prcp_cum7", "Prcp_cum30",
                    "Tmean_roll7", "melt_index",
                    "Discharge_diff1", "Discharge_diff3", "Discharge_diff7",
                    "Prcp_roll3_sum", "Prcp_roll14_sum", "Prcp_heavy3",
                    "Prcp_diff1"):
            return "派生气象特征"
        if "_roll" in name:
            return "窗口特征"
        return "原始气象因子"

    rep = rep.copy()
    rep["类别"] = rep["特征"].map(group)
    g = rep.groupby("类别")["互信息得分"].agg(["sum", "count", "mean"]).sort_values(
        "sum", ascending=False).reset_index()
    g.columns = ["特征类别", "互信息总得分", "特征个数", "平均互信息"]
    g.to_csv(os.path.join(C.RESULT_DIR, "05_特征类别贡献.csv"),
             index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b = ax.bar(g["特征类别"], g["互信息总得分"],
               color=PALETTE[:len(g)], alpha=0.85)
    for rect, (_, row) in zip(b, g.iterrows()):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                f"{row['互信息总得分']:.3f}\n({int(row['特征个数'])}个)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("互信息总得分 (nats)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    save(fig, "fig9_特征类别贡献.png")
    return g


# ===========================================================================
# 主流程
# ===========================================================================
def run(df):
    print("\n" + "=" * 70)
    print("第二部分  特征工程")
    print("=" * 70)

    d, feature_cols = build_features(df)

    print("\n【1. 时间序列特征提取】")
    groups = {
        "季节/月编码": [c for c in feature_cols if c.startswith("season_")
                        or c == "month"],
        "滞后特征": [c for c in feature_cols if "_lag" in c],
        "窗口特征": [c for c in feature_cols if "_roll" in c
                     and c != "Tmean_roll7"
                     and c not in ("Prcp_roll3_sum", "Prcp_roll14_sum")],
        "派生气象特征": [c for c in feature_cols if c in
                         ("Tmean", "Trange", "Prcp_cum7", "Prcp_cum30",
                          "Tmean_roll7", "melt_index",
                          "Discharge_diff1", "Discharge_diff3", "Discharge_diff7",
                          "Prcp_roll3_sum", "Prcp_roll14_sum", "Prcp_heavy3",
                          "Prcp_diff1")],
        "原始气象因子": [c for c in feature_cols if c in C.USABLE_RAW_FEATURES],
    }
    for k, v in groups.items():
        print(f"    {k:<12}: {len(v):>3} 个")

    print("\n【2. 特征归一化】")
    tab = compare_scalers(d, feature_cols)
    print(tab.to_string(index=False))

    print("\n【3. 特征选择】")
    rep, selected = select_features(d, feature_cols)
    _plot_feature_groups(rep)

    # 特征与径流的散点（取互信息最高的 6 个特征）
    top6 = rep["特征"].head(6).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, col in zip(axes.ravel(), top6):
        ax.scatter(d[col], d[C.TARGET], s=5, alpha=0.35, color=PALETTE[0])
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("径流 (ft$^3$/s)", fontsize=9)
        r = np.corrcoef(d[col], d[C.TARGET])[0, 1]
        ax.set_title(f"{col}  (r={r:.3f})", fontsize=10)
    plt.tight_layout()
    save(fig, "fig10_特征散点.png")

    # 保存特征表
    d.to_csv(os.path.join(C.RESULT_DIR, "06_特征工程后数据.csv"),
             index=False, encoding="utf-8-sig")

    print("\n第二部分完成")
    return d, feature_cols, selected


if __name__ == "__main__":
    from data_analysis import load_raw
    run(load_raw())
