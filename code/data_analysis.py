# -*- coding: utf-8 -*-
"""
第一部分：数据分析
==================
1. 读取数据、理解字段含义（Daymet V4 R1 + USGS 径流）
2. 时间序列可视化
3. 直方图分析分布
4. 箱型图分析离散程度与异常值
5. 皮尔逊相关系数热力图分析相关性
6. 月度/季节尺度分析（为后续时间序列特征提取提供依据）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config as C
from utils_plot import save, PALETTE, init


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
def load_raw():
    df = pd.read_csv(C.DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def basic_info(df):
    """输出数据概况表：字段含义、统计量、缺失值。"""
    rows = []
    for col in df.columns:
        if col == "Date":
            continue
        cn, unit = C.COLUMN_INFO[col]
        s = df[col]
        rows.append({
            "字段": col, "中文含义": cn, "单位": unit,
            "样本数": int(s.count()), "缺失值": int(s.isna().sum()),
            "均值": round(s.mean(), 3), "标准差": round(s.std(), 3),
            "最小值": round(s.min(), 3), "25%分位": round(s.quantile(.25), 3),
            "中位数": round(s.median(), 3), "75%分位": round(s.quantile(.75), 3),
            "最大值": round(s.max(), 3),
            "偏度": round(s.skew(), 3), "峰度": round(s.kurt(), 3),
        })
    info = pd.DataFrame(rows)
    info.to_csv(os.path.join(C.RESULT_DIR, "01_数据概况.csv"),
                index=False, encoding="utf-8-sig")
    return info


# ---------------------------------------------------------------------------
# 图 1：时间序列
# ---------------------------------------------------------------------------
def fig_timeseries(df):
    """径流与全部气象因子的逐日过程线（训练/测试分段标注）。

    按阶段验收要求，对数据集中所有列（日期除外）逐一做时间序列可视化：
    Discharge + Dayl/Prcp/Srad/Tmax/Tmin/Vp（Swe 为常量列恒为 0，已在特征工程中剔除）。
    """
    cols = [C.TARGET] + C.USABLE_RAW_FEATURES   # 7 列：Discharge + 6 个可用气象因子
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 3.2 * len(cols)), sharex=True)

    for ax, col in zip(axes, cols):
        if col == C.TARGET:
            ax.plot(df["Date"], df[col], color=PALETTE[0], lw=0.8)
            ax.set_ylabel("ft$^3$/s", fontsize=8, labelpad=4)
            ax.axvspan(pd.Timestamp(C.TRAIN_START), pd.Timestamp(C.TRAIN_END),
                       color="#2E5C8A", alpha=0.06)
            ax.axvspan(pd.Timestamp(C.TEST_START), pd.Timestamp(C.TEST_END),
                       color="#C0392B", alpha=0.08)
            # 训练/测试期划分在图释文字中说明，不在图内标注文字
        else:
            cn = C.COLUMN_INFO[col][0]
            unit = C.COLUMN_INFO[col][1]
            if col == "Prcp":
                ax.bar(df["Date"], df[col], width=1.0, color=PALETTE[5], alpha=0.75)
            else:
                ax.plot(df["Date"], df[col], color=PALETTE[1], lw=0.7)
            ax.set_ylabel(unit, fontsize=8, labelpad=4)
        # 子图标题用文字标注变量（中文名+英文名），置于子图上方，不与曲线重叠
        cn = C.COLUMN_INFO[col][0]
        ax.set_title(f"{cn} ({col})", fontsize=11, pad=6)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # 图号与图名说明统一放在报告图释文字中，图内不加标题
    plt.tight_layout()
    return save(fig, "fig1_时间序列.png")


def fig_seasonal_cycle(df):
    """年内过程线：按月聚合的径流 + 降水 + 气温（气候态）。"""
    d = df.copy()
    d["月"] = d["Date"].dt.month
    g = d.groupby("月").agg(
        径流均值=(C.TARGET, "mean"), 径流上四分位=(C.TARGET, lambda s: s.quantile(.75)),
        径流下四分位=(C.TARGET, lambda s: s.quantile(.25)),
        降水均值=("Prcp", "mean"), 最高气温=("Tmax", "mean"), 最低气温=("Tmin", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.fill_between(g["月"], g["径流下四分位"], g["径流上四分位"],
                     color=PALETTE[0], alpha=0.25, label="径流四分位区间")
    ax1.plot(g["月"], g["径流均值"], "o-", color=PALETTE[0], lw=2, label="径流均值")
    ax1.set_xlabel("月份")
    ax1.set_ylabel("径流 (ft$^3$/s)", color=PALETTE[0])
    ax1.set_xticks(range(1, 13))

    ax2 = ax1.twinx()
    ax2.bar(g["月"] - 0.15, g["降水均值"], width=0.3, color=PALETTE[5],
            alpha=0.7, label="降水均值")
    ax2.plot(g["月"] + 0.15, g["最高气温"], "s--", color=PALETTE[1], lw=1.5,
             label="最高气温均值")
    ax2.plot(g["月"] + 0.15, g["最低气温"], "^--", color=PALETTE[3], lw=1.5,
             label="最低气温均值")
    ax2.set_ylabel("降水 (mm/day) / 气温 (degC)")
    ax2.grid(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, ncol=2)
    plt.tight_layout()
    return save(fig, "fig2_年内分布.png")


# ---------------------------------------------------------------------------
# 图 3：直方图
# ---------------------------------------------------------------------------
def fig_histogram(df):
    cols = [C.TARGET] + C.USABLE_RAW_FEATURES
    n = len(cols)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        cn, unit = C.COLUMN_INFO[col]
        s = df[col].dropna()
        ax.hist(s, bins=40, color=PALETTE[0], alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.axvline(s.mean(), color=PALETTE[1], ls="--", lw=1.5,
                   label=f"均值={s.mean():.2f}")
        ax.axvline(s.median(), color=PALETTE[2], ls=":", lw=1.5,
                   label=f"中位数={s.median():.2f}")
        # 子图标题仅保留英文变量名（中文名/偏度统计在图释文字中说明）
        ax.set_title(col, fontsize=11)
        ax.set_xlabel(unit, fontsize=9)
        ax.set_ylabel("频数", fontsize=9)
        ax.legend(fontsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    plt.tight_layout()
    return save(fig, "fig3_直方图.png")


# ---------------------------------------------------------------------------
# 图 4：箱型图
# ---------------------------------------------------------------------------
def fig_boxplot(df):
    cols = [C.TARGET] + C.USABLE_RAW_FEATURES
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：标准化后的各变量箱线图，便于横向比较离散程度
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    bp = axes[0].boxplot([z[c].dropna() for c in cols], tick_labels=cols,
                         patch_artist=True, showfliers=True,
                         flierprops=dict(marker=".", markersize=3,
                                         markerfacecolor=PALETTE[1],
                                         markeredgecolor=PALETTE[1], alpha=.6))
    for patch, c in zip(bp["boxes"], PALETTE * 3):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    axes[0].axhline(0, color="gray", lw=1, ls="--")
    axes[0].set_title("(a) 标准化后各变量箱型图", fontsize=11)
    axes[0].set_ylabel("Z-score")
    axes[0].tick_params(axis="x", rotation=45)

    # 右：径流按月分组的箱线图 —— 反映季节性
    d = df.copy()
    d["月"] = d["Date"].dt.month
    data = [d.loc[d["月"] == m, C.TARGET].values for m in range(1, 13)]
    bp2 = axes[1].boxplot(data, tick_labels=range(1, 13), patch_artist=True,
                          flierprops=dict(marker=".", markersize=3,
                                          markerfacecolor=PALETTE[1],
                                          markeredgecolor=PALETTE[1], alpha=.6))
    for patch, m in zip(bp2["boxes"], range(1, 13)):
        patch.set_facecolor(PALETTE[0] if m in (4, 5, 6) else
                            (PALETTE[1] if m in (7, 8, 9) else
                             (PALETTE[3] if m in (10, 11) else PALETTE[5])))
        patch.set_alpha(0.6)
    axes[1].set_title("(b) 径流按月分组箱型图", fontsize=11)
    axes[1].set_xlabel("月份")
    axes[1].set_ylabel("径流 (ft$^3$/s)")
    plt.tight_layout()
    return save(fig, "fig4_箱型图.png")


# ---------------------------------------------------------------------------
# 图 5：皮尔逊相关系数热力图
# ---------------------------------------------------------------------------
def pearson_matrix(df, cols=None):
    cols = cols or ([C.TARGET] + C.USABLE_RAW_FEATURES)
    return df[cols].corr(method="pearson")


def fig_corr_heatmap(df):
    cols = [C.TARGET] + C.USABLE_RAW_FEATURES
    corr = pearson_matrix(df, cols)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    im = axes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(cols)))
    axes[0].set_xticklabels(cols, rotation=45, ha="right")
    axes[0].set_yticks(range(len(cols)))
    axes[0].set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if abs(v) > 0.5 else "black")
    axes[0].set_title("(a) 变量间皮尔逊相关系数矩阵", fontsize=11)
    axes[0].grid(False)
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # 右：各因子与径流的相关系数条形图
    s = corr[C.TARGET].drop(C.TARGET).sort_values()
    colors = [PALETTE[1] if v < 0 else PALETTE[2] for v in s.values]
    axes[1].barh(s.index, s.values, color=colors, alpha=0.85)
    axes[1].axvline(0, color="black", lw=1)
    for i, (k, v) in enumerate(s.items()):
        axes[1].text(v + (0.02 if v >= 0 else -0.02), i, f"{v:.3f}",
                     va="center", ha="left" if v >= 0 else "right", fontsize=9)
    axes[1].set_title("(b) 同期气象因子与径流的相关系数", fontsize=11)
    axes[1].set_xlabel("皮尔逊相关系数 r")
    axes[1].set_xlim(-1.05, 1.05)
    plt.tight_layout()
    return save(fig, "fig5_相关系数热力图.png")


def fig_lag_correlation(df, max_lag=45):
    """径流与气象因子、径流自身的滞后相关分析 —— 为滞后/窗口特征提供依据。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    # (a) 径流自相关（ACF）
    y = df[C.TARGET]
    acf = [y.autocorr(lag=k) for k in range(1, max_lag + 1)]
    axes[0].bar(range(1, max_lag + 1), acf, color=PALETTE[0], alpha=0.8)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].axhline(1.96 / np.sqrt(len(y)), color=PALETTE[1], ls="--", lw=1,
                    label="95% 置信界")
    axes[0].axhline(-1.96 / np.sqrt(len(y)), color=PALETTE[1], ls="--", lw=1)
    axes[0].set_title("(a) 径流自相关函数 (ACF)：反映记忆性", fontsize=11)
    axes[0].set_xlabel("滞后天数 (天)")
    axes[0].set_ylabel("自相关系数")
    axes[0].legend(fontsize=9)

    # (b) 气象因子与径流的滞后互相关（全部 6 个可用因子，满足验收要求）
    for i, col in enumerate(C.USABLE_RAW_FEATURES):
        cc = [df[col].corr(y.shift(-k)) for k in range(0, max_lag + 1)]
        axes[1].plot(range(max_lag + 1), cc, "-o", ms=3, lw=1.5,
                     color=PALETTE[i], label=col)   # 图例用英文变量名，中文名在报告图释中说明
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_title("(b) 气象因子对径流的滞后互相关（横轴=气象提前天数）", fontsize=11)
    axes[1].set_xlabel("气象因子提前径流的天数 (天)")
    axes[1].set_ylabel("互相关系数")
    axes[1].legend(fontsize=9, ncol=2)
    plt.tight_layout()
    return save(fig, "fig6_滞后相关.png")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run():
    print("=" * 70)
    print("第一部分  数据分析")
    print("=" * 70)
    print(f"中文字体：{init()}")
    df = load_raw()
    print(f"数据规模：{df.shape[0]} 行 × {df.shape[1]} 列，"
          f"时间范围 {df['Date'].min().date()} ~ {df['Date'].max().date()}")

    info = basic_info(df)
    print("\n【数据概况】")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(info[["字段", "中文含义", "单位", "均值", "标准差",
                    "最小值", "最大值", "偏度", "峰度"]].to_string(index=False))

    # 常量列检测
    const = [c for c in C.RAW_FEATURES if df[c].nunique() <= 1]
    if const:
        print(f"\n[注意] 常量列（无信息量，将在特征工程中剔除）：{const}")

    # 相关性
    corr = pearson_matrix(df)
    print("\n【与径流的皮尔逊相关系数】")
    print(corr[C.TARGET].drop(C.TARGET).sort_values(ascending=False).to_string())
    corr.to_csv(os.path.join(C.RESULT_DIR, "02_皮尔逊相关系数矩阵.csv"),
                encoding="utf-8-sig")

    # 绘制全部图
    fig_timeseries(df)
    fig_seasonal_cycle(df)
    fig_histogram(df)
    fig_boxplot(df)
    fig_corr_heatmap(df)
    fig_lag_correlation(df)

    print("\n第一部分完成，图表已保存至 output/figures/")
    return df


if __name__ == "__main__":
    run()
