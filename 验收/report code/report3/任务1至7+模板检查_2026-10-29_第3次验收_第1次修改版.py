"""检查第三次验收模板结构与相对附件链接，不把空白模板判为实验完成。"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET
from zipfile import ZipFile


def check(root):
    out = root / "验收" / "Acceptance 3"
    title = "任务1至7+第三次阶段验收内容模板"
    docx = out / f"{title}.docx"
    md = out / f"{title}.md"
    links = []
    with ZipFile(docx) as archive:
        assert archive.testzip() is None, "Word压缩包损坏"
        rels = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        for rel in rels:
            if rel.attrib.get("Type", "").endswith("/hyperlink"):
                links.append(rel.attrib["Target"])
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        tree = ET.fromstring(archive.read("word/document.xml"))
        text = "\n".join(t.text or "" for t in tree.findall(".//w:t", ns))
        for task in range(1, 8):
            assert f"任务{task}+" in text, f"缺少任务{task}标题"
        assert "MultiOutputRegressor" in text
        assert "尚未训练" in text
        assert "2026年11月5日" in text
    markdown = md.read_text(encoding="utf-8")
    links.extend(re.findall(r"\]\(([^)]+)\)", markdown))
    for link in links:
        parsed = urlsplit(link)
        assert not parsed.scheme and not parsed.netloc, f"不是相对链接：{link}"
        target = (out / unquote(parsed.path)).resolve()
        assert target.is_relative_to(out.resolve()), f"链接超出交付目录：{link}"
        assert target.is_file(), f"链接目标缺失：{link}"
    metric_rows = 0
    for task in range(2, 8):
        matches = list((out / "data").glob(f"任务{task}+*精度填写模板.csv"))
        assert len(matches) == 1
        with matches[0].open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row["预见期_d"]) for row in rows] == list(range(1, 8))
        for row in rows:
            assert all(row[key] == "" for key in ["R2", "RMSE", "MAE"]), "空白模板中存在非空精度"
            assert row["状态"] == "待训练/待填写"
        metric_rows += len(rows)
    with (out / "data" / "任务1+LSTM最优超参数填写模板.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 8
    disallowed = {".py", ".pyc", ".log"}
    assert not [p for p in out.rglob("*") if p.suffix in disallowed], "交付目录混有代码或日志"
    print("模板结构检查通过；任务1至7标题齐全。")
    print(f"Word与Markdown相对链接校验通过：{len(links)}处，{len(set(links))}个唯一附件。")
    print(f"精度表共{metric_rows}行，最优参数8行；保持空白且明确标为待训练。")
    print("Word可解析且压缩包完整；未使用Word桌面程序进行视觉排版检查。")
    print("本检查不证明训练完成或模型精度合格。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    check(parser.parse_args().root.resolve())
