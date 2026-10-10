# -*- coding: utf-8 -*-
"""
模型精度评价
============
采用水文预报领域通用的评价指标：

  RMSE  均方根误差        —— 对大误差敏感，单位与径流一致
  MAE   平均绝对误差      —— 稳健的绝对误差度量
  NSE   Nash-Sutcliffe 效率系数 —— 水文预报核心指标，1 为完美，>0.75 为很好
  R2    决定系数          —— 回归拟合优度
  PBIAS 百分比偏差        —— 反映系统性高估/低估
  MAPE  平均绝对百分比误差
"""

import numpy as np
import pandas as pd

METRIC_NAMES = ["RMSE", "MAE", "NSE", "R2", "PBIAS(%)", "MAPE(%)",
                "NSE_log", "RMSE_log"]

# 水文预报精度分级（NSE，参考 Moriasi et al., 2007）
NSE_GRADE = [
    (0.75, "很好 (Very Good)"),
    (0.65, "好 (Good)"),
    (0.50, "满意 (Satisfactory)"),
    (-np.inf, "不满意 (Unsatisfactory)"),
]


def nse_grade(v):
    for th, name in NSE_GRADE:
        if v >= th:
            return name
    return "不满意"


def evaluate(y_true, y_pred):
    """计算全部评价指标。"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[m], y_pred[m]

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r2 = float(np.corrcoef(y_true, y_pred)[0, 1] ** 2) if len(y_true) > 1 else np.nan
    pbias = float(100.0 * np.sum(err) / np.sum(y_true)) if np.sum(y_true) != 0 else np.nan
    nz = y_true != 0
    mape = float(100.0 * np.mean(np.abs(err[nz] / y_true[nz]))) if nz.any() else np.nan

    # 对数尺度指标：削弱少数特大洪水对评分的支配，反映模型对中低水过程的拟合能力
    lt = np.log(np.maximum(y_true, 1e-6))
    lp = np.log(np.maximum(y_pred, 1e-6))
    lerr = lp - lt
    nse_log = 1.0 - float(np.sum(lerr ** 2)) / float(np.sum((lt - lt.mean()) ** 2))
    rmse_log = float(np.sqrt(np.mean(lerr ** 2)))

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "NSE": round(nse, 4),
        "R2": round(r2, 4),
        "PBIAS(%)": round(pbias, 2),
        "MAPE(%)": round(mape, 2),
        "NSE_log": round(nse_log, 4),
        "RMSE_log": round(rmse_log, 4),
    }


def metric_table(records, index_name="模型"):
    """把 [{...}, ...] 形式的评价结果整理成 DataFrame。"""
    df = pd.DataFrame(records)
    cols = [c for c in [index_name, "策略", "预见期"] if c in df.columns] + METRIC_NAMES
    cols = [c for c in cols if c in df.columns]
    return df[cols]


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_hydrograph(dates, observed, predictions, title, fname, subdir=None):
    """水文过程线：观测 vs 各模型预测。predictions: {label: array}"""
    import matplotlib.pyplot as plt
    from utils_plot import save, PALETTE

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, observed, color="black", lw=1.6, label="实测径流", zorder=5)
    for i, (label, yp) in enumerate(predictions.items()):
        ax.plot(dates, yp, lw=1.2, alpha=0.9,
                color=PALETTE[i % len(PALETTE)], label=label)
    ax.set_xlabel("日期")
    ax.set_ylabel("径流 (ft$^3$/s)")
    ax.legend(fontsize=9, ncol=2)
    fig.autofmt_xdate()
    return save(fig, fname, subdir)


def plot_scatter(y_true, preds, title, fname, subdir=None):
    """观测—预测散点图（含 1:1 线与 NSE/RMSE 标注）。"""
    import matplotlib.pyplot as plt
    from utils_plot import save, PALETTE

    n = len(preds)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow),
                             squeeze=False)
    axes = axes.ravel()
    hi = float(np.nanmax(np.r_[y_true, *[np.asarray(p).ravel() for p in preds.values()]]))
    for ax, (label, yp) in zip(axes, preds.items()):
        m = evaluate(y_true, yp)
        ax.scatter(y_true, yp, s=10, alpha=0.45, color=PALETTE[0], edgecolors="none")
        ax.plot([0, hi], [0, hi], "k--", lw=1.2, label="1:1 线")
        ax.set_xlabel("实测径流 (ft$^3$/s)", fontsize=9)
        ax.set_ylabel("预测径流 (ft$^3$/s)", fontsize=9)
        ax.set_title(f"{label}\nNSE={m['NSE']:.3f}  RMSE={m['RMSE']:.1f}",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, hi * 1.02)
        ax.set_ylim(0, hi * 1.02)
    for ax in axes[len(preds):]:
        ax.axis("off")
    if title:
        plt.suptitle(title, fontsize=13, y=1.0)
    plt.tight_layout()
    return save(fig, fname, subdir)


def plot_metric_vs_horizon(df, metric, title, fname, subdir=None):
    """指标随预见期的变化曲线。df 需包含 模型 / 预见期 / metric 列。"""
    import matplotlib.pyplot as plt
    from utils_plot import save, PALETTE

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, g) in enumerate(df.groupby("模型", sort=False)):
        g = g.sort_values("预见期")
        ax.plot(g["预见期"], g[metric], "o-", lw=2, ms=6,
                color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xlabel("预见期 (天)")
    ax.set_ylabel(metric)
    ax.set_xticks(range(1, 8))
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return save(fig, fname, subdir)


def plot_model_comparison(df, metric, title, fname, subdir=None, group_col="模型"):
    """各模型指标对比柱状图（带数值标注）。"""
    import matplotlib.pyplot as plt
    from utils_plot import save, PALETTE

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(df)), 5))
    vals = df[metric].values
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    b = ax.bar(df[group_col], vals, color=colors, alpha=0.88)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric)
    if title:
        ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", rotation=20)
    lo, hi = min(0, vals.min()), max(vals) * 1.15
    ax.set_ylim(lo, hi)
    plt.tight_layout()
    return save(fig, fname, subdir)
