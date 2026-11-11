# -*- coding: utf-8 -*-
"""生成总验收报告用的汇总图（2026-11-11 第1版）。

数据源：验收/Acceptance 3/data/ 下的主 seed42 指标与朴素基线。
输出：验收/总验收报告/figures/ 下的 PNG。
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "验收" / "Acceptance 3" / "data"
OUT = ROOT / "验收" / "总验收报告" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["ANN", "RF", "LSTM", "Mean", "Stacking"]
COLORS = {"ANN": "#4C78A8", "RF": "#54A24B", "LSTM": "#E45756",
          "Mean": "#F58518", "Stacking": "#B279A2", "Persistence": "#8C8C8C"}


def load(name):
    with open(DATA / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


main = load("任务1至7+主seed42完整指标与来源.csv")
base = load("任务2至7+同样本朴素基线指标.csv")


def series(rows, strat, model, key):
    sel = [r for r in rows if r["strategy"] == strat and r["model"] == model]
    sel.sort(key=lambda r: int(r["h"]))
    return [float(r[key]) for r in sel]


def base_series(model, key):
    sel = [r for r in base if r["model"] == model]
    sel.sort(key=lambda r: int(r["h"]))
    return [float(r[key]) for r in sel]


H = list(range(1, 8))

# ---------------- 图1：逐预见期 RMSE ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=200)
for ax, strat, title in zip(axes, ["direct", "multi"],
                            ["(a) 多步直接策略", "(b) 多输出策略"]):
    for m in MODELS:
        ax.plot(H, series(main, strat, m, "RMSE"), marker="o", ms=4.6, lw=2.0,
                color=COLORS[m], label=m)
    ax.plot(H, base_series("Persistence", "RMSE"), ls="--", lw=1.6,
            color=COLORS["Persistence"], label="持久性基线")
    ax.set_title(title, fontsize=11, pad=18)
    ax.set_xlabel("预见期 h / 天")
    ax.set_ylabel("RMSE / (ft³/s)")
    ax.set_xticks(H)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8, ncol=6, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.4)
fig.tight_layout()
fig.savefig(OUT / "图4-9_逐预见期RMSE对比.png", bbox_inches="tight")
plt.close(fig)

# ---------------- 图2：七步平均汇总 ----------------
labels, direct_rmse, multi_rmse, direct_r2, multi_r2 = [], [], [], [], []
for m in MODELS:
    d = series(main, "direct", m, "RMSE")
    u = series(main, "multi", m, "RMSE")
    dr = series(main, "direct", m, "R2")
    ur = series(main, "multi", m, "R2")
    labels.append(m)
    direct_rmse.append(sum(d) / len(d))
    multi_rmse.append(sum(u) / len(u))
    direct_r2.append(sum(dr) / len(dr))
    multi_r2.append(sum(ur) / len(ur))

import numpy as np

x = np.arange(len(labels))
w = 0.36

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1), dpi=200)
b1 = ax1.bar(x - w / 2, direct_rmse, w, label="多步直接", color="#4C78A8")
b2 = ax1.bar(x + w / 2, multi_rmse, w, label="多输出", color="#F58518")
ax1.axhline(sum(base_series("Persistence", "RMSE")) / 7, ls="--", lw=1.4,
            color=COLORS["Persistence"], label="持久性基线均值")
ax1.axhline(sum(base_series("Seasonal_mean", "RMSE")) / 7, ls=":", lw=1.4,
            color="#54A24B", label="季节均值基线")
for b in list(b1) + list(b2):
    ax1.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                 ha="center", va="bottom", fontsize=7.2)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("七步平均 RMSE / (ft³/s)")
ax1.set_title("(a) 七步平均 RMSE（越小越好）", fontsize=11, pad=24)
ax1.set_ylim(0, 1160)
ax1.grid(axis="y", alpha=0.3, ls=":")
ax1.legend(fontsize=7.6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))

b3 = ax2.bar(x - w / 2, direct_r2, w, label="多步直接", color="#4C78A8")
b4 = ax2.bar(x + w / 2, multi_r2, w, label="多输出", color="#F58518")
ax2.axhline(0, color="black", lw=0.8)
for b in list(b3) + list(b4):
    v = b.get_height()
    ax2.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=7.2)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel("七步平均 $R^2$")
ax2.set_title("(b) 七步平均 $R^2$（越大越好，允许为负）", fontsize=11)
ax2.grid(axis="y", alpha=0.3, ls=":")
ax2.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "图4-10_七步平均汇总.png", bbox_inches="tight")
plt.close(fig)

print("已生成：")
for p in sorted(OUT.glob("*.png")):
    print("  ", p.name)
print("直接/多输出七步平均 RMSE：")
for lab, a, b in zip(labels, direct_rmse, multi_rmse):
    print(f"  {lab:9s} {a:8.2f} {b:8.2f}")
print("直接/多输出七步平均 R2：")
for lab, a, b in zip(labels, direct_r2, multi_r2):
    print(f"  {lab:9s} {a:8.4f} {b:8.4f}")

# ---------------- 图4-12：模型 × 预见期 RMSE 热力图 ----------------
from matplotlib.patches import Rectangle
import numpy as np

row_spec = [("direct", "ANN", "直接 · ANN"), ("direct", "LSTM", "直接 · LSTM"),
            ("direct", "RF", "直接 · 随机森林"), ("direct", "Mean", "直接 · 简单平均"),
            ("direct", "Stacking", "直接 · Stacking"),
            ("multi", "ANN", "多输出 · ANN"), ("multi", "LSTM", "多输出 · LSTM"),
            ("multi", "RF", "多输出 · 随机森林"), ("multi", "Mean", "多输出 · 简单平均"),
            ("multi", "Stacking", "多输出 · Stacking")]

mat = np.array([series(main, s, m, "RMSE") for s, m, _ in row_spec])
labels = [lab for _, _, lab in row_spec]

fig, ax = plt.subplots(figsize=(7.8, 3.9), dpi=240)
im = ax.imshow(mat, cmap="OrRd", aspect="auto", vmin=mat.min() * 0.98, vmax=mat.max())

ax.set_xticks(range(len(H)))
ax.set_xticklabels([f"h={h}" for h in H], fontsize=8.4)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8.4)
ax.set_xlabel("预见期 h / 天", fontsize=9.5)
ax.tick_params(length=0)

# 数值标注 + 每列最优方框
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        frac = (v - im.norm.vmin) / (im.norm.vmax - im.norm.vmin)
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.0,
                color="white" if frac > 0.62 else "#333333")
best = mat.argmin(axis=0)
for j, i in enumerate(best):
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                           edgecolor="#1F4E79", lw=1.9))

# 直接/多输出分隔线
ax.axhline(4.5, color="white", lw=2.4)
ax.axhline(4.5, color="#555555", lw=0.9, ls=(0, (4, 3)))

cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
cbar.set_label("RMSE / (ft³/s)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title("各配置逐预见期 RMSE（蓝框为该列最优）", fontsize=10.5, pad=8)
fig.tight_layout()
fig.savefig(OUT / "图4-12_模型预见期RMSE热力图.png", bbox_inches="tight")
plt.close(fig)
print("已生成热力图 图4-12")
