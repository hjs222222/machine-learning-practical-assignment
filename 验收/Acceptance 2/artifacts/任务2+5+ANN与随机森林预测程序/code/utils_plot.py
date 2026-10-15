# -*- coding: utf-8 -*-
"""绘图公共工具：中文字体配置、统一风格、图片保存。"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import FIG_DIR

# ---------------------------------------------------------------------------
# 中文字体：按优先级挑选系统中已安装的字体，避免出现方块乱码
# ---------------------------------------------------------------------------
_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
    "Noto Sans CJK SC", "Source Han Sans CN", "WenQuanYi Micro Hei", "Arial Unicode MS",
]


def _pick_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CANDIDATES:
        if name in installed:
            return name
    return None


_FONT = _pick_font()
if _FONT:
    plt.rcParams["font.sans-serif"] = [_FONT]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["font.size"] = 11

# 配色（保证打印为黑白时仍可区分：线型 + 深浅）
PALETTE = ["#2E5C8A", "#C0392B", "#27865B", "#D68910", "#7D3C98", "#117A8B"]


def init():
    """返回实际使用的中文字体名，供报告说明。"""
    return _FONT


def save(fig, name, subdir=None):
    """保存图片并返回相对路径。"""
    d = FIG_DIR if subdir is None else os.path.join(FIG_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path}")
    return path
