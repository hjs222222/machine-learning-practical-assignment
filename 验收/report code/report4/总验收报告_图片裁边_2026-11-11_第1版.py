# -*- coding: utf-8 -*-
"""裁掉报告图片四周的纯白边（保留 8px 留白），减小图与图注之间的空隙。"""
from pathlib import Path

from PIL import Image, ImageChops

FIG = Path(__file__).resolve().parents[3] / "验收" / "总验收报告" / "figures"

for p in sorted(FIG.glob("*.png")):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    # 阈值化，避免压缩噪声被当成内容
    gray = diff.convert("L").point(lambda v: 255 if v > 12 else 0)
    bbox = gray.getbbox()
    if not bbox:
        print("skip", p.name)
        continue
    pad = 8
    left = max(0, bbox[0] - pad)
    upper = max(0, bbox[1] - pad)
    right = min(w, bbox[2] + pad)
    lower = min(h, bbox[3] + pad)
    if (right - left) >= w - 2 and (lower - upper) >= h - 2:
        print("keep", p.name)
        continue
    im.crop((left, upper, right, lower)).save(p, optimize=True)
    print(f"crop {p.name}: {w}x{h} -> {right - left}x{lower - upper}")
