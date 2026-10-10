# -*- coding: utf-8 -*-
"""
第四部分：结果可视化与分析
==========================
读取实验产出的指标表与预测文件，绘制：
  图3-1  各模型 1/3/7 天预见期水文过程线（实测 vs 预测）
  图3-2  观测—预测散点图
  图3-3  评价指标随预见期的变化
  图3-4  各模型汇总指标对比
  图3-5  集成模型与个体模型对比
  图3-6  典型洪水事件放大分析
  图3-7  误差分布与误差随流量量级的变化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from utils_plot import save, PALETTE
from evaluation import evaluate


def load_all(prefix):
    """读取某策略下所有个体模型的 7 个预见期预测。prefix: 'direct' / 'multi'
    只加载个体模型（pred_ANN_/pred_RF_/pred_LSTM_），排除集成模型 pred_ens_*。"""
    files = [f for f in os.listdir(C.RESULT_DIR)
             if f.startswith("pred_") and not f.startswith("pred_ens_")
             and f.endswith(f"_{prefix}.npz")]
    out = {}
    for f in files:
        name = f[len("pred_"):-len(f"_{prefix}.npz")]
        z = np.load(os.path.join(C.RESULT_DIR, f))
        out[name] = {int(k[1:]): z[k] for k in z.files}
    return out


def load_index():
    z = np.load(os.path.join(C.RESULT_DIR, "eval_index.npz"))
    return z


def main():
    print("\n" + "=" * 70)
    print("第四部分  结果可视化与分析")
    print("=" * 70)

    z = load_index()
    observed = z["observed"]
    origins = z["origins"]
    dates = pd.to_datetime(z["dates"])

    detail = pd.read_csv(os.path.join(C.RESULT_DIR, "10_各模型各预见期指标.csv"))
    pooled = pd.read_csv(os.path.join(C.RESULT_DIR, "11_模型汇总指标.csv"))

    base = load_all("direct")
    base_m = load_all("multi")

    # 集成模型的预测由 experiment 阶段保存；保留策略后缀，避免 direct/multi 互相覆盖
    ens = {}
    for f in os.listdir(C.RESULT_DIR):
        if f.startswith("pred_ens_") and f.endswith(".npz"):
            key = f[len("pred_ens_"):-4]   # 形如 'Stacking_direct' / '简单平均_multi'
            zz = np.load(os.path.join(C.RESULT_DIR, f))
            ens[key] = {int(k[1:]): zz[k] for k in zz.files}

    def merge_preds(base, ens_map):
        """合并个体与集成预测，固定顺序：ANN, RF, LSTM, 简单平均, Stacking"""
        m = dict(base)
        for k, v in ens_map.items():
            disp = k.replace("_direct", "").replace("_multi", "")
            m[disp] = v
        return {k: m[k] for k in ["ANN", "RF", "LSTM", "简单平均", "Stacking"] if k in m}

    all_direct = merge_preds(base, {k: v for k, v in ens.items() if k.endswith("_direct")})
    all_multi = merge_preds(base_m, {k: v for k, v in ens.items() if k.endswith("_multi")})

    # -----------------------------------------------------------------
    # 图3-1 水文过程线（图内不放图题，图题由报告正文提供）
    # -----------------------------------------------------------------
    for h in [1, 3, 7]:
        obs = observed[origins + h]
        preds = {k: v[h] for k, v in all_direct.items()}
        _hydro(dates, obs, preds,
               f"fig11_过程线_h{h}.png")
    # 多输出策略（h=7 最关键）
    for h in [1, 7]:
        obs = observed[origins + h]
        preds = {k: v[h] for k, v in all_multi.items()}
        _hydro(dates, obs, preds,
               f"fig12_多输出过程线_h{h}.png")

    # -----------------------------------------------------------------
    # 图3-3 散点图（图内不放总标题，子图标注 NSE/RMSE 保留）
    # -----------------------------------------------------------------
    for h in [1, 3, 7]:
        obs = observed[origins + h]
        preds = {k: v[h] for k, v in all_direct.items()}
        from evaluation import plot_scatter
        plot_scatter(obs, preds,
                     None,
                     f"fig13_散点_h{h}.png")

    # -----------------------------------------------------------------
    # 图3-4 指标随预见期变化
    # -----------------------------------------------------------------
    from evaluation import plot_metric_vs_horizon, plot_model_comparison
    d1 = detail[detail["策略"] == "多步直接预测"]
    d2 = detail[detail["策略"] == "多输出预测"]
    plot_metric_vs_horizon(d1, "NSE",
                           None,
                           "fig14_NSE随预见期变化_直接.png")
    plot_metric_vs_horizon(d2, "NSE",
                           None,
                           "fig15_NSE随预见期变化_多输出.png")
    plot_metric_vs_horizon(d1, "RMSE",
                           None,
                           "fig16_RMSE随预见期变化_直接.png")

    # -----------------------------------------------------------------
    # 图3-7 汇总指标对比
    # -----------------------------------------------------------------
    for strat, tag in [("多步直接预测", "直接"), ("多输出预测", "多输出")]:
        sub = pooled[pooled["策略"] == strat].copy()
        plot_model_comparison(sub, "NSE",
                              None,
                              f"fig17_NSE对比_{tag}.png")
        plot_model_comparison(sub, "RMSE",
                              None,
                              f"fig18_RMSE对比_{tag}.png")

    # -----------------------------------------------------------------
    # 图3-9 典型洪水事件放大
    # -----------------------------------------------------------------
    _flood_zoom(dates, observed, origins, all_direct)

    # -----------------------------------------------------------------
    # 图3-10 误差分析
    # -----------------------------------------------------------------
    _error_analysis(observed, origins, all_direct)

    print("\n第四部分完成")
    return detail, pooled


def _hydro(dates, obs, preds, fname):
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.plot(dates, obs, color="black", lw=1.8, label="实测径流", zorder=10)
    for i, (k, v) in enumerate(preds.items()):
        ax.plot(dates, v, lw=1.2, alpha=0.9,
                color=PALETTE[i % len(PALETTE)], label=k)
    ax.set_xlabel("日期")
    ax.set_ylabel("径流 (ft$^3$/s)")
    ax.legend(fontsize=9, ncol=3)
    fig.autofmt_xdate()
    plt.tight_layout()
    return save(fig, fname)


def _flood_zoom(dates, observed, origins, preds, n_events=2, win=12):
    """挑选测试期最大的 n 场洪水，放大展示预报效果。"""
    obs1 = observed[origins + 1]
    idx = np.argsort(obs1)[::-1]
    picked, used = [], []
    for i in idx:
        if len(picked) >= n_events:
            break
        if any(abs(i - u) < win * 2 for u in used):
            continue
        picked.append(i)
        used.append(i)

    fig, axes = plt.subplots(1, len(picked), figsize=(7 * len(picked), 4.6),
                             squeeze=False)
    for ax, i in zip(axes.ravel(), picked):
        lo = max(0, i - win)
        hi = min(len(dates), i + win)
        ax.plot(dates[lo:hi], observed[origins[lo:hi] + 1], "k-o", ms=3, lw=1.8,
                label="实测径流")
        for j, (k, v) in enumerate(preds.items()):
            ax.plot(dates[lo:hi], v[1][lo:hi], lw=1.6, alpha=0.9,
                    color=PALETTE[j % len(PALETTE)], label=k)
        peak = observed[origins[i] + 1]
        ax.axvline(dates[i], color="gray", ls=":", lw=1.2)
        ax.set_title(f"洪峰日期 {str(pd.Timestamp(dates[i]).date())}"
                     f"  实测峰值 {peak:.0f} ft$^3$/s", fontsize=11)
        ax.set_ylabel("径流 (ft$^3$/s)")
        ax.legend(fontsize=8)
        for lb in ax.get_xticklabels():
            lb.set_rotation(30)
    plt.tight_layout()
    return save(fig, "fig19_洪水事件放大.png")


def _error_analysis(observed, origins, preds):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(preds.keys())
    obs1 = observed[origins + 1]

    # (a) 误差箱线图
    errs = [preds[k][1] - obs1 for k in names]
    bp = axes[0].boxplot(errs, tick_labels=names, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[0].axhline(0, color="black", lw=1.2)
    axes[0].set_ylabel("预测误差 (ft$^3$/s)")
    axes[0].set_title("(a) 各模型预测误差分布（预见期 1 天）", fontsize=11)
    axes[0].tick_params(axis="x", rotation=20)

    # (b) 绝对误差随流量量级的变化
    for j, k in enumerate(names):
        ae = np.abs(preds[k][1] - obs1)
        bins = np.quantile(obs1, np.linspace(0, 1, 11))
        bins[-1] += 1e-6
        idx = np.digitize(obs1, bins)
        xs = [obs1[idx == b].mean() for b in range(1, 11) if (idx == b).any()]
        ys = [ae[idx == b].mean() for b in range(1, 11) if (idx == b).any()]
        axes[1].plot(xs, ys, "o-", lw=1.6, ms=5,
                     color=PALETTE[j % len(PALETTE)], label=k)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("实测径流量级 (ft$^3$/s，对数轴)")
    axes[1].set_ylabel("平均绝对误差 (ft$^3$/s)")
    axes[1].set_title("(b) 绝对误差随流量量级的变化", fontsize=11)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    return save(fig, "fig20_误差分析.png")


if __name__ == "__main__":
    main()
