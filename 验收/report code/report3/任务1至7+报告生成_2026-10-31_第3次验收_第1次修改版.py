# -*- coding: utf-8 -*-
"""任务1至7+第三次验收报告生成；2026-10-31，第1次修改版。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import quote, unquote
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, Cm, RGBColor

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "验收/Acceptance 3"
TASKS = [(2, "direct", "LSTM", "LSTM直接策略"),
         (3, "multi", "LSTM", "LSTM多输出策略"),
         (4, "direct", "Mean", "简单平均直接策略"),
         (5, "direct", "Stacking", "Stacking直接策略"),
         (6, "multi", "Mean", "简单平均多输出策略"),
         (7, "multi", "Stacking", "Stacking多输出策略")]
COLORS = {"LSTM": "#2a78d6", "Mean": "#eb6834", "Stacking": "#1baf7a"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "grid.color": GRID,
                     "font.size": 10, "lines.linewidth": 1.5})


def rel(path):
    return Path(path).relative_to(OUT).as_posix()


def hyperlink(paragraph, text, path):
    link = OxmlElement("w:hyperlink")
    rid = paragraph.part.relate_to(quote(path, safe="/"),
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    link.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "245C88")
    props.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(underline)
    run.append(props)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    link.append(run)
    paragraph._p.append(link)


class Report:
    def __init__(self):
        self.doc = Document()
        self.md = []
        sec = self.doc.sections[0]
        sec.top_margin = sec.bottom_margin = Cm(1.8)
        sec.left_margin = sec.right_margin = Cm(1.8)
        for name in ("Normal", "Heading 1", "Heading 2", "Title"):
            style = self.doc.styles[name]
            style.font.name = "Microsoft YaHei"
            style.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        self.doc.styles["Normal"].font.size = Pt(10)
        self.doc.styles["Normal"].paragraph_format.space_after = Pt(6)
        footer = sec.footer.paragraphs[0]
        footer.alignment = 2
        footer.add_run("第三次阶段验收 · USGS 01047000 · ")
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        footer._p.append(field)

    def title(self, text):
        self.doc.add_heading(text, 0)
        self.md.extend(["# " + text, ""])

    def heading(self, text, level=1):
        self.doc.add_heading(text, level)
        self.md.extend(["#" * (level + 1) + " " + text, ""])

    def paragraph(self, text):
        self.doc.add_paragraph(text)
        self.md.extend([text, ""])

    def table(self, data, columns=None):
        frame = data.copy()
        if columns:
            frame = frame[columns]
        def fmt(value):
            if isinstance(value, (float, np.floating)):
                return f"{value:.4f}"
            return str(value)
        headers = list(frame.columns)
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.style = "Light Shading Accent 1"
        for cell, text in zip(table.rows[0].cells, headers):
            cell.text = str(text)
        # 跨页表格重复表头，且不将单行拆到两页。
        repeat = OxmlElement("w:tblHeader")
        table.rows[0]._tr.get_or_add_trPr().append(repeat)
        self.md.append("| " + " | ".join(map(str, headers)) + " |")
        self.md.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in frame.itertuples(index=False, name=None):
            values = [fmt(x) for x in row]
            cells = table.add_row().cells
            for cell, value in zip(cells, values):
                cell.text = value
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(8)
            self.md.append("| " + " | ".join(v.replace("|", "/") for v in values) + " |")
        self.md.append("")

    def link(self, label, path):
        p = self.doc.add_paragraph()
        hyperlink(p, label, rel(path))
        self.md.extend([f"[{label}]({quote(rel(path), safe='/')})", ""])

    def image(self, label, path):
        self.doc.add_picture(str(path), width=Inches(6.65))
        self.link(label, path)
        self.md.insert(len(self.md) - 2, f"![{label}]({quote(rel(path), safe='/')})\n")

    def page(self):
        self.doc.add_page_break()


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=.6)
    ax.set_axisbelow(True)


def save_chart(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def trace_plot(predictions, strategy, model, path, run_id):
    frame = predictions[(predictions.strategy == strategy) & (predictions.model == model)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
    for ax, h in zip(axes, (1, 3, 7)):
        part = frame[frame.h == h].sort_values("target_date")
        ax.plot(part.target_date, part.truth, color=MUTED, label="观测", lw=1.3)
        ax.plot(part.target_date, part.pred, color=COLORS[model], label="预测", lw=1.5, ls="--")
        peak = part.loc[part.truth.idxmax()]
        ax.annotate("观测峰值", (peak.target_date, peak.truth), xytext=(8, -18),
                    textcoords="offset points", fontsize=8, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": .5})
        ax.set_title(f"h={h} 天 · 横轴为目标日期", loc="left", fontsize=10)
        ax.set_ylabel("Discharge（原始单位）")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(frameon=False, ncols=2, loc="upper left")
        ax.set_xlim(pd.Timestamp("2004-01-01"), pd.Timestamp("2004-12-31"))
        ax.set_ylim(top=max(float(part.truth.max()), float(part.pred.max())) * 1.12)
        style_axis(ax)
    fig.suptitle(f"{strategy} / {model} · 2004年滚动预测\nrun_id={run_id}", fontsize=12)
    save_chart(fig, path)


def metric_plot(metrics, path, run_id):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for row, strategy in enumerate(("direct", "multi")):
        for col, metric in enumerate(("R2", "RMSE", "MAE")):
            ax = axes[row, col]
            for model, marker, line in (("LSTM", "o", "-"), ("Mean", "s", "--"), ("Stacking", "^", ":")):
                part = metrics[(metrics.strategy == strategy) & (metrics.model == model)].sort_values("h")
                ax.plot(part.h, part[metric], color=COLORS[model], marker=marker,
                        markersize=6, markeredgecolor=SURFACE, markeredgewidth=1,
                        ls=line, label=model)
            ax.set_title(f"{strategy} · {metric}", loc="left")
            ax.set_xlabel("预见期 / 天")
            ax.set_ylabel("无量纲" if metric == "R2" else "原始Discharge单位")
            ax.set_xticks(range(1, 8))
            ax.legend(frameon=False, fontsize=8)
            style_axis(ax)
    fig.suptitle(f"任务2至7 · 逐预见期指标对照\nrun_id={run_id}", fontsize=12)
    save_chart(fig, path)


def render_output(text, path, heading):
    # 逐字渲染真实保存的程序输出；明确标注，不冒充桌面或终端截屏。
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 19)
    title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    lines = text.splitlines()
    width = max(1100, int(max((font.getlength(line) for line in lines), default=0)) + 60)
    image = Image.new("RGB", (width, 100 + 28 * len(lines)), SURFACE)
    draw = ImageDraw.Draw(image)
    draw.text((24, 16), heading + "（真实输出渲染，非桌面截屏）", font=title_font, fill=INK)
    for i, line in enumerate(lines):
        draw.text((24, 65 + 28 * i), line, font=font, fill=INK)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run = ROOT / "output/acceptance3_runs" / args.run_id
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and not manifest["smoke"], "只允许完整正式运行生成最终报告"
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    verification = json.loads((run / "verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "passed" and verification["required_rows"] == 42
    summary = json.loads((run / "data_summary.json").read_text(encoding="utf-8"))
    best = json.loads((run / "best_parameters.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run / "metrics.csv")
    predictions = pd.read_csv(run / "test_predictions.csv", parse_dates=["target_date", "origin_date"])
    required = metrics[metrics.model.isin(["LSTM", "Mean", "Stacking"])]
    assert len(required) == 42 and np.isfinite(required[["R2", "RMSE", "MAE"]]).all().all()
    for folder in ("data", "figures", "artifacts"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    # 新报告另存，不覆盖已存在的填写模板和前两次材料。
    target = OUT / "任务1至7+第三次阶段验收报告.docx"
    if target.exists():
        raise FileExistsError("报告已存在；请先审阅并明确允许替换后再运行")
    copies = {"metrics.csv": "data/任务1至7+全部模型逐预见期指标.csv",
              "test_predictions.csv": "data/任务1至7+测试集逐日预测.csv",
              "oof_predictions.csv": "data/任务5及7+因果OOF预测.csv",
              "time_splits.csv": "data/任务1及5及7+时间划分与净化审计.csv",
              "stacking_coefficients.csv": "data/任务5及7+元学习器系数.csv",
              "cv_results/all_candidates.csv": "data/任务1至7+全部网格候选逐折成绩.csv",
              "manifest.json": "artifacts/任务1至7+运行来源与环境.json",
              "config.json": "artifacts/任务1至7+冻结实验配置.json",
              "best_parameters.json": "artifacts/任务1至7+完整最优参数.json",
              "data_summary.json": "artifacts/任务1至7+样本与特征摘要.json",
              "model_restoration_checks.json": "artifacts/任务1至7+模型恢复核验.json",
              "verification.json": "artifacts/任务1至7+独立结果核验.json"}
    for src, dst in copies.items():
        shutil.copy2(run / src, OUT / dst)
    report = Report()
    report.title("任务1至7+第三次阶段验收报告")
    report.paragraph("基于LSTM与集成学习的1—7天径流预测｜USGS 01047000")
    report.paragraph(f"实验完成：{manifest['finished']}；run_id：{args.run_id}。本报告使用本次独立训练结果，不以历史模型或填写模板替代实验。")
    report.paragraph("验收安排：2026年11月5日13:30—16:50，第12教研楼202。报告未填写未经提供的姓名、学号或分工身份信息。")
    report.heading("任务1至7+验收范围与完成证据")
    overview = [{"任务": "任务1", "内容": "8个LSTM的GridSearchCV", "证据": "完整网格、逐折成绩、最优参数、训练记录"}]
    overview += [{"任务": f"任务{n}", "内容": name, "证据": "7行R2/RMSE/MAE、真实输出与过程线"} for n, _, _, name in TASKS]
    report.table(pd.DataFrame(overview))
    teacher = OUT / "artifacts/任务1至7+第三次阶段验收说明原文.docx"
    report.link("任务1至7+教师验收说明原文", teacher)
    report.heading("任务1至7+数据、样本和公平评价协议")
    report.paragraph(f"原始数据{summary['raw_rows']}天（2000-01-01—2004-12-31），含Discharge及7个Daymet因子。按原项目因果特征工程得到{summary['n_features']}个候选特征；每次fit内按目标ln(Q)的互信息保留前50个，多输出共用h=1选择规则。LSTM输入为(N,30,50)，ANN/RF使用起报日的50维特征。未调用MultiOutputRegressor。")
    report.table(pd.DataFrame([
        {"阶段": "早期校准", "样本数": summary["n_calibration"], "边界": f"目标截至2001-12-31；末起报{summary['calibration_last_origin'][:10]}"},
        {"阶段": "完整训练", "样本数": summary["n_train"], "边界": f"{summary['train_first_origin'][:10]}—{summary['train_last_origin'][:10]}；7列目标均不晚于2003-12-31"},
        {"阶段": "因果OOF", "样本数": summary["n_oof"], "边界": "2002—2003年起报，3个连续验证块，扩张训练窗"},
        {"阶段": "统一测试", "样本数": summary["n_test"], "边界": f"{summary['test_first_origin'][:10]}—{summary['test_last_origin'][:10]}起报，所有目标在2004年"}]))
    report.paragraph("预测场景：t日结束后，用截至t日的观测预测Q(t+1)…Q(t+7)。2004年逐日起报，允许用已经观测到的2004年历史作输入，但不更新模型、特征选择器或标准化器，不使用未来天气实测。所有方法、策略与预见期共用同一组测试起报点。直接任务也保守共用7天合法训练边界。过程线横轴为目标日期，不是起报日期。")
    report.paragraph("R²=1−Σ(y−ŷ)²/Σ(y−ȳ)²，允许为负；RMSE=√mean((y−ŷ)²)，MAE=mean(|y−ŷ|)。全部先逆标准化、exp恢复至原始Discharge尺度再计算。旧项目R2曾使用相关系数平方，因此本报告R²不应与第二次报告的同名列直接作优劣比较。本次ANN/RF也按新协议重训，仅作为同口径集成基模型，不覆盖第二次报告。")
    report.paragraph("单位说明：教师提供的CSV及已读取说明未标注Discharge物理单位。旧config.py写为ft³/s，但缺少该CSV预处理来源证明，本报告谨慎标为“原始Discharge单位”，不未经核实进行单位换算。R²无量纲，RMSE/MAE与源列同单位。")
    report.paragraph("测试独立性限制：项目历史实验已查看过2004年效果，因此本次2004年属于冻结新协议后的回溯测试，不宣称为从未接触的盲测。网格、种子42、特征规则、Ridge alpha=1和不截断负预测规则均在评价前冻结；不按测试分数重新挑选参数。")
    splits = pd.read_csv(run / "time_splits.csv")
    report.table(splits.rename(columns={"stage": "阶段", "fold": "折", "n_train": "训练n", "n_validation": "验证n",
        "max_train_target": "最晚训练目标", "validation_origin_start": "最早验证起报"})[
            ["阶段", "折", "训练n", "验证n", "最晚训练目标", "最早验证起报"]])
    report.paragraph("各折满足max(train_target_date)<min(validation_origin_date)，统一gap=7。预处理仅在当前训练折拟合。LSTM早停另取该折末尾15%作内部验证，再剔除7行标签跨界，选定轮数后重置网络并重训该外折全部训练数据；外层验证和2004不参与早停。")
    report.link("任务1及5及7+完整时间边界审计", OUT / copies["time_splits.csv"])
    report.page()
    report.heading("任务1+LSTM网格搜索与最优超参数")
    report.paragraph("每个直接预见期独立搜索一个单输出网络；多输出独立搜索一个7输出网络。使用sklearn GridSearchCV + 3折TimeSeriesSplit(gap=7)，仅在2000—2001校准区搜索。评分为原始尺度各预见期RMSE均值的负值，越大越好；单输出即负RMSE。每任务8候选、24次CV fit和1次refit，共8任务。")
    report.paragraph("LSTM网格：units={(64,),(64,32)}，dropout={0.2,0.3}，lr={0.001,0.0005}。固定lookback=30、batch_size=32、Adam、标准化ln(Q)尺度MSE、max_epochs=80、patience=6。各层LSTM后Dropout，取末时刻隐藏状态，接ReLU全连接头；单输出头32宽、输出1维，多输出头64宽、输出7维。")
    lstm = [x for x in best if x["model"] == "LSTM"]
    rows = []
    for item in lstm:
        p = item["best_params"]
        rows.append({"模型": "多输出1—7" if item["strategy"] == "multi" else f"直接h={item['h']}",
            "units/dropout/lr": f"{p['units']} / {p['dropout']} / {p['lr']}",
            "CV RMSE": -item["best_score"], "标准差": item["std_test_score"],
            "冻结轮数": item["fixed_epochs"]})
    best_csv = OUT / "data/任务1+8个LSTM最优超参数.csv"
    pd.DataFrame(rows).to_csv(best_csv, index=False, encoding="utf-8-sig")
    report.table(pd.DataFrame(rows))
    report.paragraph("OOF与最终模型不重新选轮数：采用胜出候选3个外折内部早停最佳轮数的中位数（四舍五入），在全部合法训练样本上固定轮数重训。表中“冻结轮数”是最终实际轮数，不是80轮上限。完整候选逐折成绩保留真实best_index、并列名次、耗时及失败记录；本报告不把refit早停轮数混作最终轮数。")
    report.link("任务1+8个LSTM最优超参数", best_csv)
    report.link("任务1至7+全部候选逐折成绩", OUT / copies["cv_results/all_candidates.csv"])
    report.link("任务1至7+完整最优参数与真实best_index", OUT / copies["best_parameters.json"])
    candidates = pd.read_csv(run / "cv_results/all_candidates.csv")
    comparisons = []
    for item in lstm:
        frame = candidates[(candidates.model == "LSTM") & (candidates.strategy == item["strategy"]) & (candidates.h == item["h"])]
        worst = frame.loc[frame.mean_test_score.idxmin()]
        comparisons.append({"模型": f"{item['strategy']} h={item['h']}",
            "最佳CV RMSE": -item["best_score"], "最差CV RMSE": -worst.mean_test_score,
            "候选差距": -worst.mean_test_score + item["best_score"]})
    report.table(pd.DataFrame(comparisons))
    report.paragraph("上表按每个任务的真实8候选统计，差距越大表示网格选择对校准误差越敏感。CV标准差反映不同时间段的不稳定性，不是测试误差置信区间；best_params只表示预设候选网格内最优，不代表全部LSTM结构的全局最优。最佳容量、dropout与学习率均由早期CV决定，未预设深层网络一定更好。")
    search_lines = [line for line in (run / "logs/training.log").read_text(encoding="utf-8").splitlines()
                    if line.startswith("[搜索完成]") and '"model": "LSTM"' in line]
    assert len(search_lines) == 8
    search_text = "run_id=" + args.run_id + "\n" + "\n\n".join(search_lines)
    # 长JSON分行显示，但不重写其中的参数或数值。
    import textwrap
    wrapped = "\n".join("\n".join(textwrap.wrap(line, width=115, replace_whitespace=False))
                         for line in search_text.splitlines())
    search_image = OUT / "figures/任务1+真实网格搜索完成输出.png"
    render_output(wrapped, search_image, "任务1+8个LSTM真实搜索完成输出")
    search_capture = OUT / "figures/任务1+真实程序窗口截图.png"
    if search_capture.is_file():
        report.image("任务1+实际运行结果查看器窗口截图", search_capture)
    else:
        report.image("任务1+真实GridSearchCV结果输出", search_image)
    report.paragraph("ANN与RF也仅在同一早期校准区分别搜索8个任务。ANN为原生多目标MLP，hidden_layer_sizes={(64,),(64,32)}×alpha={1e−5,0.01}×learning_rate_init={0.001,0.0005}，关闭随机内部早停且shuffle=False；RF为原生多目标森林，n_estimators={300,500}×max_depth={10,20}×min_samples_leaf={1,5}。它们的搜索与重训用于生成本次同口径集成输入。")
    # 展示胜出候选首折真实早停历史，不以手工曲线替代训练记录。
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    for ax, strategy in zip(axes, ("direct", "multi")):
        item = next(x for x in lstm if x["strategy"] == strategy)
        chosen = None
        for f in (run / "training_history").glob(f"calibration_{strategy}_LSTM_h{item['h']}_*.json"):
            record = json.loads(f.read_text(encoding="utf-8"))
            if "cv_score" in record and all(record["params"][k] == v for k, v in item["best_params"].items()):
                chosen = record
                break
        assert chosen is not None
        curve = pd.DataFrame(chosen["early_history"])
        ax.plot(curve.epoch, curve.loss, label="内部训练", color=COLORS["LSTM"])
        ax.plot(curve.epoch, curve.val_loss, label="内部验证", color=COLORS["Mean"], ls="--")
        ax.set_title(f"{strategy} h={item['h']} · 胜出候选某校准折")
        ax.set_xlabel("epoch")
        ax.set_ylabel("标准化ln(Q) MSE")
        ax.legend(frameon=False)
        for values, label in ((curve.loss, "训练"), (curve.val_loss, "验证")):
            ax.annotate(label, (curve.epoch.iloc[-1], values.iloc[-1]), xytext=(5, 0),
                        textcoords="offset points", fontsize=8, color=INK)
        ax.margins(x=.12)
        style_axis(ax)
    curve_path = OUT / "figures/任务1+LSTM真实内部训练曲线.png"
    save_chart(fig, curve_path)
    report.image("任务1+LSTM真实内部训练曲线", curve_path)
    report.heading("任务4至7+集成训练方式", level=1)
    report.paragraph("简单平均：分别将ANN、RF、LSTM预测exp回变换后，逐起报点逐h计算(Q_ANN+Q_RF+Q_LSTM)/3，不在对数尺度平均。Stacking只使用2002—2003年有效因果OOF行，首段空缺全部删除，不以0或训练拟合值填充。校准后参数冻结，3个OOF块分别重新拟合选择器、标准化器与基模型，OOF先恢复原始尺度再拼接。")
    report.paragraph("直接Stacking：每h一个StandardScaler→Ridge(alpha=1.0)，输入该h的ANN/RF/LSTM三列预测，共7个元模型。多输出Stacking：按ANN_h1…h7、RF_h1…h7、LSTM_h1…h7拼成21列，单个StandardScaler→原生多输出Ridge联合拟合7列目标，系数为7×21。Ridge目标使用原始流量；保留未截断元输出，并统计负值数量。")
    report.link("任务5及7+OOF逐日证据", OUT / copies["oof_predictions.csv"])
    report.link("任务5及7+全部元模型系数、截距和OOF覆盖", OUT / copies["stacking_coefficients.csv"])
    source_lines = (run / "source/acceptance3_model.py").read_text(encoding="utf-8").splitlines()
    begin = next(i for i, line in enumerate(source_lines) if line.startswith("class SequenceNet"))
    end = next(i for i, line in enumerate(source_lines) if line.startswith("_PREPROCESS_CACHE"))
    source_text = "acceptance3_model.py (actual run source snapshot)\n" + "\n".join(
        f"{i+1:04d} {source_lines[i]}" for i in range(begin, end))
    source_image = OUT / "artifacts/任务1及3+原生多输出LSTM源码证据.png"
    render_output(source_text, source_image, "任务1及3+实际网络定义源码证据")
    report.image("任务1及3+实际运行快照中的原生多输出网络定义", source_image)
    for task, strategy, model, name in TASKS:
        report.page()
        report.heading(f"任务{task}+{name}1至7天预测结果")
        part = metrics[(metrics.strategy == strategy) & (metrics.model == model)].sort_values("h")
        display = part[["h", "R2", "RMSE", "MAE", "n"]].rename(columns={"h": "预见期/天", "n": "样本数"})
        file = OUT / f"data/任务{task}+{name}逐预见期指标.csv"
        display.to_csv(file, index=False, encoding="utf-8-sig")
        report.table(display)
        report.link(f"任务{task}+完整精度CSV", file)
        hi, lo = part.loc[part.R2.idxmax()], part.loc[part.R2.idxmin()]
        monotone = bool((part.RMSE.diff().dropna() >= 0).all())
        report.paragraph(f"本组R²最高为h={int(hi.h)}的{hi.R2:.4f}，最低为h={int(lo.h)}的{lo.R2:.4f}。RMSE随预见期{'单调不减' if monotone else '并非单调不减'}，不将个别波动改写为严格单调规律。负流量预测共{int(part.negative_predictions.sum())}个（跨7个预见期统计），均保留原值计分。")
        other = metrics[(metrics.strategy == ("multi" if strategy == "direct" else "direct")) & (metrics.model == model)].sort_values("h")
        gain = (part.RMSE.to_numpy() < other.RMSE.to_numpy()).sum()
        report.paragraph(f"与另一策略的同模型比较，本策略在{gain}/7个预见期具有更低RMSE。两种策略使用相同测试真值和起报点，因此该比较不受样本数量变化影响。")
        if model in ("Mean", "Stacking"):
            comparisons = []
            for h in range(1, 8):
                individuals = metrics[(metrics.strategy == strategy) & (metrics.h == h) & metrics.model.isin(["ANN", "RF", "LSTM"])]
                baseline = individuals.loc[individuals.RMSE.idxmin()]
                current = part[part.h == h].iloc[0]
                comparisons.append({"h": h, "最佳单模型(按RMSE)": baseline.model,
                    "ΔR²": current.R2 - baseline.R2,
                    "RMSE变化%": 100 * (current.RMSE / baseline.RMSE - 1),
                    "MAE变化%": 100 * (current.MAE / baseline.MAE - 1)})
            cmp = pd.DataFrame(comparisons)
            report.table(cmp)
            improved = int((cmp["RMSE变化%"] < 0).sum())
            report.paragraph(f"相对于逐h按RMSE选出的最佳单模型，本集成仅在{improved}/7个预见期降低RMSE。上表负的误差变化代表改善，正值代表变差；MAE对照也沿用这个RMSE基准模型，不把不同模型的最小值拼成虚构基线。集成不保证优于最佳基模型，误差相关性、OOF样本有限和元模型训练/测试分布差异均可能限制增益。")
            cmp.to_csv(OUT / f"data/任务{task}+集成相对最佳单模型对照.csv", index=False, encoding="utf-8-sig")
            report.link(f"任务{task}+集成增益对照", OUT / f"data/任务{task}+集成相对最佳单模型对照.csv")
        trace = OUT / f"figures/任务{task}+{name}过程线.png"
        trace_plot(predictions, strategy, model, trace, args.run_id)
        report.image(f"任务{task}+h=1、3、7目标日期过程线", trace)
        source = run / "evidence" / f"{strategy}_{model}.txt"
        evidence = OUT / f"figures/任务{task}+真实程序指标输出.png"
        render_output(source.read_text(encoding="utf-8"), evidence, f"任务{task}+真实程序指标输出")
        capture = OUT / f"figures/任务{task}+真实程序窗口截图.png"
        if capture.is_file():
            report.image(f"任务{task}+实际运行结果查看器窗口截图", capture)
        else:
            report.image(f"任务{task}+真实输出记录渲染图", evidence)
    report.page()
    report.heading("任务1至7+综合比较、局限与结论")
    compare = OUT / "figures/任务2至7+逐预见期指标对照.png"
    metric_plot(metrics, compare, args.run_id)
    report.image("任务2至7+逐预见期指标对照", compare)
    agg = metrics.groupby(["strategy", "model"], sort=False)[["R2", "RMSE", "MAE"]].mean().reset_index()
    report.table(agg.rename(columns={"strategy": "策略", "model": "模型", "R2": "平均R²", "RMSE": "平均RMSE", "MAE": "平均MAE"}))
    report.paragraph("上表均值仅用于汇总，不替代任务2—7的42行逐预见期结果。随预见期增加，未来降雨等未知驱动的影响增强，历史径流的可预测性通常减弱；ln目标的MSE训练会压缩洪峰权重，容易出现洪峰幅度偏小。LSTM共享序列表示与多输出共享参数可能减少训练量，但不保证每个h均优于独立模型。")
    for strategy in ("direct", "multi"):
        subset = metrics[metrics.strategy == strategy]
        pivot = subset.pivot(index="h", columns="model", values="RMSE")
        wins = int((pivot.Stacking < pivot.Mean).sum())
        largest = agg[agg.strategy == strategy]
        lowest = largest.loc[largest.RMSE.idxmin()]
        report.paragraph(f"{strategy}策略中，Stacking在{wins}/7个预见期优于简单平均（按RMSE）；按7个h平均RMSE，最低者为{lowest['model']}（{lowest.RMSE:.4f}）。此处是冻结模型后的描述性排名，不据此重新训练或选参。")
    # 从真实洪峰/低流量样本重算误差，避免纯凭图推断。
    extra = []
    for (strategy, model, h), frame in predictions[predictions.model.isin(["LSTM", "Mean", "Stacking"])].groupby(["strategy", "model", "h"]):
        for group, mask in (("观测最高10%", frame.truth >= frame.truth.quantile(.9)),
                            ("观测最低25%", frame.truth <= frame.truth.quantile(.25))):
            cut = frame[mask]
            extra.append({"strategy": strategy, "model": model, "h": h, "segment": group,
                          "n": len(cut), "MAE": float(np.mean(np.abs(cut.pred - cut.truth))),
                          "mean_signed_error": float(np.mean(cut.pred - cut.truth))})
    extremes = pd.DataFrame(extra)
    extreme_path = OUT / "data/任务2至7+高低流量分段误差.csv"
    extremes.to_csv(extreme_path, index=False, encoding="utf-8-sig")
    report.link("任务2至7+高低流量分段真实误差", extreme_path)
    for strategy in ("direct", "multi"):
        cut = extremes[(extremes.strategy == strategy) & (extremes.model == "LSTM") & (extremes.h == 1)]
        high = cut[cut.segment == "观测最高10%"].iloc[0]
        low = cut[cut.segment == "观测最低25%"].iloc[0]
        report.paragraph(f"LSTM {strategy} h=1：最高10%流量段MAE={high.MAE:.4f}，平均有符号误差={high.mean_signed_error:.4f}（负值为低估）；最低25%流量段MAE={low.MAE:.4f}，平均有符号误差={low.mean_signed_error:.4f}。这是逐样本重算的实际分段结果。")
    report.paragraph("分段阈值仅用于冻结模型后的描述性误差分析，不参与选参或训练。原始5年资料较短，早期校准区的洪水事件代表性有限；当前没有未来天气预报输入，长预见期与极端洪峰仍是主要局限。可在后续独立年份验证更长数据、气象预报驱动及更合适的洪峰损失，但本次不根据2004分数追加试验挑优。")
    report.heading("任务1至7+环境、追溯和交付检查")
    report.paragraph(f"后端：PyTorch {manifest['packages']['torch']}，device={manifest['device']}，硬件={manifest['gpu'] or 'CPU'}；Python={manifest['python'].splitlines()[0]}。本次总耗时{manifest['seconds']/3600:.2f}小时，固定seed=42，并启用PyTorch确定性算法；跨版本、硬件仍可能有数值差异。")
    report.paragraph(f"输入SHA256：{manifest['data_sha256']}。Git基线：{manifest['git_commit']}。训练入口code/acceptance3_train.py；完整运行目录output/acceptance3_runs/{args.run_id}/保存源代码快照、配置、真实stdout/stderr、逐次训练历史、模型权重及预处理器。源代码和日志不混入最终交付目录。")
    report.paragraph("已保存24个基模型、7个直接Ridge和1个联合多输出Ridge；简单平均无需fit。模型恢复检查对24个基模型实际重新加载并对同一输入比较预测，不只核对文件存在。LSTM权重与网络构造参数、特征顺序、目标标准化器均有保存。")
    for key in ("metrics.csv", "test_predictions.csv", "manifest.json", "config.json", "data_summary.json", "model_restoration_checks.json", "verification.json"):
        report.link(Path(copies[key]).stem, OUT / copies[key])
    has_captures = all((OUT / f"figures/任务{task}+真实程序窗口截图.png").is_file() for task in range(1, 8))
    if has_captures:
        report.paragraph("程序截图取自实际启动的结果查看器窗口：查看器执行独立Python命令，实时读取正式运行文件并显示原始stdout，随后由Windows PrintWindow捕获本程序窗口；不含其他桌面窗口，也不将手工排版图片冒充截图。截图展示已完成训练的真实结果读取，不声称是长时间训练过程的视频或现场训练。本报告未声称完成Word桌面人工视觉验收。")
    else:
        report.paragraph("运行结果图由真实程序保存文本直接渲染，图中明确标注“非桌面截屏”，不冒充人工终端截图。若验收必须要求操作系统终端截图，应另外在实际终端显示对应运行输出后截取。本报告未声称完成Word桌面人工视觉验收。")
    report.paragraph("任务1—7的训练、预测、逐h指标和图表均已产生；交付包包含内嵌图表的Word及相对链接附件。目录内原有“填写模板”属于训练前准备材料，不作为本报告的结果依据；直接提交本报告及其引用附件即可。")
    report.doc.save(target)
    md_path = OUT / "任务1至7+第三次阶段验收报告.md"
    md_path.write_text("\n".join(report.md), encoding="utf-8")
    # 检查Word外部关系、Markdown链接、指标与落盘预测的一致性。
    loaded = Document(target)
    links = []
    for relationship in loaded.part.rels.values():
        if relationship.reltype.endswith("/hyperlink"):
            resolved = OUT / unquote(relationship.target_ref)
            assert resolved.resolve().is_relative_to(OUT.resolve()), resolved
            assert resolved.is_file(), resolved
            links.append(str(resolved))
    for match in re.findall(r"\]\(([^)]+)\)", md_path.read_text(encoding="utf-8")):
        assert (OUT / unquote(match)).is_file(), match
    sys.path.insert(0, str(ROOT / "code"))
    from acceptance3_train import metrics_table
    recomputed = metrics_table(predictions.to_dict("records"))
    join = recomputed.merge(metrics, on=["strategy", "model", "h"], suffixes=("_new", "_saved"))
    assert len(join) == 70
    for metric in ("R2", "RMSE", "MAE"):
        np.testing.assert_allclose(join[metric + "_new"], join[metric + "_saved"], rtol=1e-10, atol=1e-8)
    import zipfile
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        from xml.etree import ElementTree as ET
        tree = ET.fromstring(archive.read("word/document.xml"))
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        all_text = "\n".join(node.text or "" for node in tree.findall(".//w:t", namespace))
        for task in range(1, 8):
            assert f"任务{task}+" in all_text
        assert "待填写" not in all_text and "待训练" not in all_text
    checks = {"run_id": args.run_id, "required_metric_rows": len(required), "all_metric_rows": len(metrics),
              "word_hyperlinks_checked": len(links), "word_inline_images": len(loaded.inline_shapes),
              "metric_recalculation": "passed", "links": "passed", "word_desktop_visual_check": "not_performed",
              "terminal_screenshot": "actual_result_viewer_window_capture" if has_captures else "output_render_only_not_desktop_capture"}
    check_path = OUT / "artifacts/任务1至7+交付核验.json"
    check_path.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    # 打包只包含本报告直接依赖的最终附件，不带训练前填写模板。
    import zipfile
    archive = ROOT / "验收" / "任务1至7+第三次阶段验收交付包.zip"
    if archive.exists():
        raise FileExistsError("交付压缩包已存在，不自动覆盖")
    package_files = {target, md_path, check_path}
    package_files.update(Path(path) for path in links)
    assert not any(path.suffix in {".py", ".pyc", ".log"} for path in package_files)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(package_files):
            z.write(file, "第三次阶段验收/" + rel(file))
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(target)
    print(archive)


if __name__ == "__main__":
    main()
