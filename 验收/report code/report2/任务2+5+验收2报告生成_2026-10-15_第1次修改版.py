# -*- coding: utf-8 -*-
"""任务2+5+验收2报告生成（2026-10-15，第1次修改版）。"""

import ast
import csv
import shutil
from pathlib import Path
from urllib.parse import unquote

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[3]
ACCEPT = ROOT / "验收" / "Acceptance 2"
OUT = ACCEPT
FIG = OUT / "figures"
DATA = OUT / "data"
ARTIFACTS = OUT / "artifacts"
PROGRAM = ARTIFACTS / "任务2+5+ANN与随机森林预测程序"
SOURCE = ROOT / "output" / "results"
CODE = ROOT / "code"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell(cell, text, bold=False, color=None, size=8.5):
    cell.text = str(text)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for p in cell.paragraphs:
        p.paragraph_format.space_after = Pt(0)
        for r in p.runs:
            r.font.name = "Microsoft YaHei"
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
            r.font.size = Pt(size)
            r.bold = bold
            if color:
                r.font.color.rgb = RGBColor(*color)


def add_table(doc, headers, rows, widths=None, font_size=8.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    for i, label in enumerate(headers):
        set_cell(table.rows[0].cells[i], label, True, (255, 255, 255), font_size)
        shade(table.rows[0].cells[i], "24476B")
    for n, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell(cells[i], value, size=font_size)
            if n % 2:
                shade(cells[i], "F1F5F9")
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_hyperlink(paragraph, label, target):
    rel = paragraph.part.relate_to(
        target, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rel)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    props.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(underline)
    run.append(props)
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    link.append(run)
    paragraph._p.append(link)


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)


def add_bullet(doc, text):
    doc.add_paragraph(text, style="List Bullet")


def main():
    for folder in (FIG, DATA, ARTIFACTS, PROGRAM / "code", PROGRAM / "Teacher Resources & Requirements"):
        folder.mkdir(parents=True, exist_ok=True)

    metrics = [r for r in read_csv(SOURCE / "10_各模型各预见期指标.csv")
               if r["模型"] in ("人工神经网络 (ANN/MLP)", "随机森林 (RF)")]
    metric_headers = ["模型", "策略", "预见期", "R2", "RMSE", "MAE"]
    write_csv(DATA / "任务3+4+6+7_ANN与随机森林预测指标.csv", metric_headers,
              [{k: r[k] for k in metric_headers} for r in metrics])

    cv_rows = [r for r in read_csv(SOURCE / "13_网格搜索CV结果.csv") if r["模型"] in ("ANN", "RF")]
    cv_headers = ["模型", "策略", "预见期", "params", "std_test_score", "rank_test_score", "mean_test_RMSE"]
    write_csv(DATA / "任务2+5_网格搜索CV结果.csv", cv_headers,
              [{k: r[k] for k in cv_headers} for r in cv_rows])

    best = {}
    for r in cv_rows:
        key = (r["模型"], r["策略"], r["预见期"])
        if r["rank_test_score"] == "1" and key not in best:
            best[key] = r
    best_rows = []
    for key in sorted(best, key=lambda k: (k[0], k[1], int(k[2]))):
        r = best[key]
        params = ast.literal_eval(r["params"])
        best_rows.append({"模型": key[0], "策略": key[1], "预见期": key[2],
                          "最优参数": str(params), "CV均值RMSE(标准化目标)": r["mean_test_RMSE"],
                          "CV标准差": r["std_test_score"]})
    best_headers = ["模型", "策略", "预见期", "最优参数", "CV均值RMSE(标准化目标)", "CV标准差"]
    write_csv(DATA / "任务2+5_最优超参数.csv", best_headers, best_rows)

    figure_names = ["fig18_RMSE对比_直接.png", "fig18_RMSE对比_多输出.png"]
    for name in figure_names:
        shutil.copy2(ROOT / "output" / "figures" / name, FIG / ("任务3+4+6+7_" + name))
    for old_name in ("任务3+4+6+7_fig17_NSE对比_直接.png",
                     "任务3+4+6+7_fig17_NSE对比_多输出.png"):
        (FIG / old_name).unlink(missing_ok=True)
    render_tmp = OUT / "rendered"
    if render_tmp.exists() and not any(render_tmp.iterdir()):
        render_tmp.rmdir()

    # Reproducible source package: only modules required to rerun the tested ANN/RF pipeline.
    required = ["config.py", "utils_plot.py", "data_analysis.py", "feature_engineering.py",
                "models.py", "evaluation.py", "experiment_v6.py"]
    for name in required:
        shutil.copy2(CODE / name, PROGRAM / "code" / name)
    shutil.copy2(ROOT / "Teacher Resources & Requirements" / "01047000.csv",
                 PROGRAM / "Teacher Resources & Requirements" / "01047000.csv")
    runner = PROGRAM / "code" / "run_task2_acceptance.py"
    runner.write_text(
        '# -*- coding: utf-8 -*-\n'
        '"""任务2+5 ANN与RF网格搜索及多步预测复现入口。"""\n'
        'import sys\nfrom pathlib import Path\n'
        'ROOT = Path(__file__).resolve().parents[1]\n'
        'sys.path.insert(0, str(Path(__file__).resolve().parent))\n'
        'from data_analysis import run as analysis_run\n'
        'from feature_engineering import run as feature_run\n'
        'import experiment_v6 as experiment\n'
        'if __name__ == "__main__":\n'
        '    data = analysis_run()\n'
        '    engineered, features, _ = feature_run(data)\n'
        '    # MLPRegressor 与 RandomForestRegressor 原生支持多目标，直接拟合7列目标。\n'
        '    # 本流程不使用 MultiOutputRegressor 包装器。\n'
        '    experiment.BASE_MODELS = ["ANN", "RF"]\n'
        '    experiment.run(engineered, features, fast=False)\n',
        encoding="utf-8",
    )
    (PROGRAM / "运行说明.txt").write_text(
        "任务2+5 ANN与随机森林预测程序\n\n"
        "本目录包含复现网格搜索与多步预测所需的代码模块和教师提供的数据。\n"
        "在已安装 numpy、pandas、scikit-learn、matplotlib 的 Python 环境中，\n"
        "于本目录运行：python code/run_task2_acceptance.py\n"
        "完整网格搜索耗时较长；运行结果写入本目录 output/。\n",
        encoding="utf-8",
    )

    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    styles = doc.styles
    styles["Normal"].font.name = "Microsoft YaHei"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    styles["Normal"].font.size = Pt(10)
    for style_name in ("Title", "Heading 1", "Heading 2"):
        styles[style_name].font.name = "Microsoft YaHei"
        styles[style_name]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        styles[style_name].font.color.rgb = RGBColor(0, 0, 0)

    title = doc.add_paragraph()
    title.style = "Title"
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("任务1至7 第二次阶段验收报告\n基于 ANN 与随机森林的径流多步预测")
    p = doc.add_paragraph("机器学习课程实践｜USGS 01047000｜2026年10月15日整理")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].font.size = Pt(10)

    doc.add_heading("验收范围与实验设置", level=1)
    doc.add_paragraph(
        "本报告逐项对应第二次阶段验收子任务1至7。实验使用 USGS 01047000 流域逐日数据，"
        "时间范围为2000-01-01至2004-12-31，共1827天；2000—2003年用于训练，2004年用于测试。"
        "预测未来1—7天径流，ANN和随机森林均采用多步直接策略及多输出策略；验收指标按新说明列 R²、RMSE、MAE。"
    )
    add_bullet(doc, "输入特征由气象因子、季节与日历编码、滞后项、滑动统计及派生气象特征构成；Swe 为常量列并剔除。")
    add_bullet(doc, "径流目标取对数后建模；输入和目标的标准化统计量只在训练期拟合，评价指标在原始径流尺度计算。")
    add_bullet(doc, "直接策略为每个预见期单独训练模型；多输出策略由一个模型同时输出未来7天。")
    add_bullet(doc, "ANN 与随机森林直接接收7列目标完成多输出预测；未使用 MultiOutputRegressor。")
    add_bullet(doc, "统一测试起报日，保证不同预见期的结果可比；各指标和候选参数详见随报告 CSV。")

    doc.add_heading("子任务1 适用于径流预测的机器学习方法", level=1)
    add_table(doc, ["方法", "适用方式"], [
        ("人工神经网络（ANN/MLP）", "学习降水、温度、历史径流和时间特征与未来径流之间的非线性关系；本次完成预测与评价。"),
        ("随机森林（RF）", "通过多棵决策树集成刻画非线性及特征交互，对异常值和特征尺度较稳健；本次完成预测与评价。"),
        ("长短期记忆网络（LSTM）", "使用历史时间序列学习较长时间依赖；项目已有实现，可作为径流预测方法。"),
        ("支持向量回归（SVR）", "适用于中小样本的非线性回归，可配合径向基核函数建模。"),
        ("梯度提升树（如 XGBoost）", "逐步拟合残差以表达复杂非线性；可用于扩展对比。"),
    ], font_size=9)
    doc.add_paragraph("本次验收的子任务2—7按说明聚焦 ANN 与随机森林；SVR、梯度提升树仅列为适用方法，不报告本次未运行的指标。")

    doc.add_heading("子任务2与5 网格搜索确定超参数", level=1)
    doc.add_paragraph(
        "使用 GridSearchCV 和 3 折 TimeSeriesSplit 前向验证，以负 RMSE 为搜索评分，"
        "不打乱时间顺序。ANN 网格包含 hidden_layer_sizes={(64,), (128,), (64, 32)}、"
        "alpha={1e-5, 1e-2}、learning_rate_init={0.001, 0.01}；随机森林网格包含"
        "n_estimators={300, 500}、max_depth={None, 10, 15, 20}、min_samples_leaf={1, 5}。"
        "表中 CV 均值 RMSE 在标准化目标尺度上计算。"
    )
    best_display = []
    for r in best_rows:
        m = "ANN" if r["模型"] == "ANN" else "RF"
        strategy = "直接" if r["策略"] == "多步直接预测" else "多输出"
        best_display.append((m, strategy, r["预见期"], r["最优参数"],
                             f'{float(r["CV均值RMSE(标准化目标)"]):.4f}'))
    add_table(doc, ["模型", "策略", "预见期", "最优超参数", "CV均值RMSE"], best_display, font_size=8)
    doc.add_paragraph(
        "随机森林个别预见期存在并列最优候选时，表格记录按 GridSearchCV 候选顺序首先出现的 rank=1 组合；"
        "完整候选及评分保存在网格搜索 CSV。"
    )

    metric_map = {"R2": "R²", "RMSE": "RMSE", "MAE": "MAE"}
    for task, model, strategy_name in [
        ("子任务3 ANN 直接策略", "人工神经网络 (ANN/MLP)", "多步直接预测"),
        ("子任务4 ANN 多输出策略", "人工神经网络 (ANN/MLP)", "多输出预测"),
        ("子任务6 随机森林直接策略", "随机森林 (RF)", "多步直接预测"),
        ("子任务7 随机森林多输出策略", "随机森林 (RF)", "多输出预测"),
    ]:
        doc.add_heading(task, level=1)
        selected = [r for r in metrics if r["模型"] == model and r["策略"] == strategy_name]
        rows = [(r["预见期"], *(r[k] for k in metric_map)) for r in selected]
        add_table(doc, ["预见期（天）", *metric_map.values()], rows, font_size=9)
        doc.add_paragraph(
            "评价结果显示，预见期增长时误差总体增大、R²总体下降。结果仅代表本数据划分和当前实验设置。"
        )

    doc.add_heading("误差指标变化图", level=1)
    for i, name in enumerate(figure_names):
        image_path = FIG / ("任务3+4+6+7_" + name)
        doc.add_picture(str(image_path), width=Inches(6.8))
        add_caption(doc, f"图{i+1} {name.replace('.png', '').replace('_', ' ')}")
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_hyperlink(p, "打开原图", "figures/" + image_path.name)

    doc.add_heading("结论与交付附件", level=1)
    doc.add_paragraph(
        "本次已整理子任务1的适用方法说明、ANN与随机森林的网格搜索参数，以及两类策略下1—7天的 R²、RMSE 和 MAE。"
        "预测期延长后，误差总体增大且拟合能力总体下降。所有指标来源于项目已保存的实验结果 CSV；"
        "报告附有指标表、完整 ANN/RF 网格搜索候选结果、最优参数表、对比图和可复现程序包。"
    )
    links = [
        ("ANN与随机森林预测指标 CSV", "data/任务3+4+6+7_ANN与随机森林预测指标.csv"),
        ("ANN与随机森林完整网格搜索结果 CSV", "data/任务2+5_网格搜索CV结果.csv"),
        ("最优超参数 CSV", "data/任务2+5_最优超参数.csv"),
        ("任务2+5预测程序与数据包", "artifacts/任务2+5+ANN与随机森林预测程序/运行说明.txt"),
        ("复现入口程序", "artifacts/任务2+5+ANN与随机森林预测程序/code/run_task2_acceptance.py"),
        ("ANN与随机森林模型代码", "artifacts/任务2+5+ANN与随机森林预测程序/code/models.py"),
        ("实验流程代码", "artifacts/任务2+5+ANN与随机森林预测程序/code/experiment_v6.py"),
    ]
    for label, target in links:
        p = doc.add_paragraph(style="List Bullet")
        add_hyperlink(p, label, target)
    p = doc.add_paragraph("教师原始说明：第二次阶段验收说明（10月22日 周四）。")
    p.paragraph_format.space_before = Pt(6)

    output = OUT / "任务1至7+第二次阶段验收报告.docx"
    doc.save(output)
    # 小型交付自检：覆盖要求中的全部指标、图表和可点击附件。
    check = Document(output)
    links = [r.target_ref for r in check.part.rels.values()
             if r.reltype.endswith("/hyperlink")]
    missing = [target for target in links
               if not (OUT / unquote(target.replace("/", "\\"))).exists()]
    assert len(metrics) == 28 and len(cv_rows) == 476 and len(best_rows) == 28
    assert len(figure_names) == 2 and len(check.tables) == 6
    assert len(links) == 9 and not missing, f"未解析的附件链接：{missing}"
    print(f"报告：{output}")
    print(f"指标行：{len(metrics)}；CV候选行：{len(cv_rows)}；最优参数组：{len(best_rows)}；图表：{len(figure_names)}")


if __name__ == "__main__":
    main()
