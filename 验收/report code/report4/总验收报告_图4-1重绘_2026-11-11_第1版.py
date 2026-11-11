# -*- coding: utf-8 -*-
"""重绘图 4-1：径流与气象因子逐日时间序列（4×2 紧凑版，便于在报告中一页内排布）。

数据源：Teacher Resources & Requirements/01047000.csv（2000-01-01—2004-12-31，1827 天）
输出：验收/总验收报告/figures/图4-1_径流与气象因子时间序列.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "Teacher Resources & Requirements" / "01047000.csv"
OUT = ROOT / "验收" / "总验收报告" / "figures" / "图4-1_径流与气象因子时间序列.png"

df = pd.read_csv(SRC, parse_dates=["Date"])
split = pd.Timestamp("2003-12-31")

PANELS = [
    ("Discharge", "径流 / (ft³/s)", "#1F4E79"),
    ("Dayl", "日照时长 / (s/day)", "#7F6000"),
    ("Prcp", "降水量 / (mm/day)", "#2E75B6"),
    ("Srad", "短波辐射 / (W/m²)", "#C55A11"),
    ("Tmax", "最高气温 / °C", "#C00000"),
    ("Tmin", "最低气温 / °C", "#2F5597"),
    ("Vp", "水汽压 / Pa", "#548235"),
]

fig = plt.figure(figsize=(11.4, 8.6), dpi=200)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.38, wspace=0.18)
axes = [fig.add_subplot(gs[0, :])]
for i, _ in enumerate(PANELS[1:]):
    axes.append(fig.add_subplot(gs[1 + i // 2, i % 2]))

for ax, (col, label, color) in zip(axes, PANELS):
    ax.plot(df["Date"], df[col], lw=0.75, color=color)
    ax.axvline(split, color="gray", ls="--", lw=1.0)
    ax.set_ylabel(label, fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, ls=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if col != "Discharge":
        ax.set_xlim(df["Date"].min(), df["Date"].max())
    else:
        ax.set_xlim(df["Date"].min(), df["Date"].max())
        ax.set_title("01047000 流域逐日径流与气象因子（虚线左侧为训练期 2000—2003，右侧为测试期 2004）",
                     fontsize=9.5)

axes[-1].set_xlabel("日期", fontsize=8.5)
axes[-2].set_xlabel("日期", fontsize=8.5)
fig.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.98)
fig.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
