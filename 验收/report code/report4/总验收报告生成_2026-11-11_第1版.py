# -*- coding: utf-8 -*-
"""生成《机器学习课程实践》实验报告（总验收报告）2026-11-11 第1版。

数据来源（全部为落盘 CSV / JSON，可追溯）：
  验收/Acceptance 3/data/任务1至7+主seed42完整指标与来源.csv
  验收/Acceptance 3/data/任务2至7+同样本朴素基线指标.csv
  验收/Acceptance 3/artifacts/任务1至7+样本与特征摘要.json
  验收/Acceptance 3/data/任务5+Stacking元层修复对比.csv（git 分支 main: 11d87ab）
  docs/任务1至7+直接残差最终新旧对比结论_2026-11-11.md

输出：验收/总验收报告/机器学习课程实践-实验报告_2026-11-11_第1版.docx
"""
import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[3]
ACC3 = ROOT / "验收" / "Acceptance 3"
DATA = ACC3 / "data"
OUTDIR = ROOT / "验收" / "总验收报告"
FIG = OUTDIR / "figures"
OUTDOC = OUTDIR / "机器学习课程实践-实验报告_2026-11-11_第1版.docx"

CN = "宋体"
CN_HEI = "黑体"
EN = "Times New Roman"


# ---------------------------------------------------------------- 数据读取
def load_csv(name):
    with open(DATA / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


main_rows = load_csv("任务1至7+主seed42完整指标与来源.csv")
base_rows = load_csv("任务2至7+同样本朴素基线指标.csv")
feature_json = json.loads((ACC3 / "artifacts" / "任务1至7+样本与特征摘要.json").read_text(encoding="utf-8-sig"))


def get(strategy, model, key):
    sel = [r for r in main_rows if r["strategy"] == strategy and r["model"] == model]
    sel.sort(key=lambda r: int(r["h"]))
    return [float(r[key]) for r in sel]


def get_base(model, key):
    sel = [r for r in base_rows if r["model"] == model]
    sel.sort(key=lambda r: int(r["h"]))
    return [float(r[key]) for r in sel]


def avg(xs):
    return sum(xs) / len(xs)


def f2(x):
    return f"{x:.2f}"


def f4(x):
    return f"{x:.4f}"


# ---------------------------------------------------------------- 排版工具
def set_run_font(run, size=10.5, bold=False, cn=CN, en=EN, color=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = en
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), cn)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    if color is not None:
        run.font.color.rgb = color


def body(text, indent=True, size=10.5, bold=False, space_after=3, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         first_line=0.74, before=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = align
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.36
    pf.space_before = Pt(before)
    pf.space_after = Pt(space_after)
    if indent:
        pf.first_line_indent = Cm(first_line)
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold)
    return p


def h1(text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(14)
    pf.space_after = Pt(8)
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=15, bold=True, cn=CN_HEI)
    # 大纲级别，便于导航
    p.style = doc.styles["Heading 1"]
    for run in p.runs:
        set_run_font(run, size=15, bold=True, cn=CN_HEI)
        run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def h2(text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(9)
    pf.space_after = Pt(5)
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=12, bold=True, cn=CN_HEI)
    p.style = doc.styles["Heading 2"]
    for run in p.runs:
        set_run_font(run, size=12, bold=True, cn=CN_HEI)
        run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def h3(text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(3)
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=10.5, bold=True)
    return p


def bullet(text, size=10.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.74)
    pf.first_line_indent = Cm(-0.0)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.32
    pf.space_after = Pt(2)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    set_run_font(r, size=size)
    return p


def caption(text, before=3, after=6, keep_next=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.1
    pf.keep_with_next = keep_next
    r = p.add_run(text)
    set_run_font(r, size=9, cn=CN)
    return p


def shade(cell, color="DCE6F1"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def table(headers, rows, widths=None, title=None, note=None, font=8.5, first_col_left=True):
    if title:
        caption(title, before=6, after=3, keep_next=True)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0]
    repeat_header(hdr)
    for i, htxt in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(str(htxt))
        set_run_font(r, size=font, bold=True)
        shade(cell)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cell = cells[i]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (i == 0 and first_col_left) else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(str(val))
            set_run_font(r, size=font)
    if widths:
        total = sum(widths)
        wtxt = 15.4
        if total > wtxt:
            widths = [w * wtxt / total for w in widths]
        # 固定列宽布局，避免 Word 自动重排导致比例失调
        tblPr = t._tbl.tblPr
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tblPr.append(layout)
        for i, w in enumerate(widths):
            t.columns[i].width = Cm(w)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    # 禁止单行跨页拆分
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
    # 小表整体不跨页：除最后一行外，让每行与下一行同页
    for row in t.rows[:-1]:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(6)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(note)
        set_run_font(r, size=8.5, cn=CN)
    return t


def figure(path, title, width=14.6, note=None, max_h=9.0):
    if not Path(path).exists():
        body(f"（图片缺失：{Path(path).name}）", indent=False)
        return
    # 按图片长宽比限制最大高度，避免整页空白
    try:
        from PIL import Image
        w_px, h_px = Image.open(path).size
        ratio = h_px / w_px
        if width * ratio > max_h:
            width = max_h / ratio
    except Exception:
        pass
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width))
    caption(title, before=1, after=6)
    if note:
        body(note, indent=False, size=8.5, space_after=8)


def code_block(lines, size=8.2):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.6)
    pf.line_spacing = 1.05
    pf.space_before = Pt(2)
    pf.space_after = Pt(6)
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.keep_together = True
    for i, line in enumerate(lines):
        r = p.add_run(line)
        set_run_font(r, size=size, cn=CN, en="Consolas")
        if i != len(lines) - 1:
            r.add_break()
    # 浅灰底纹
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F2F2F2")
    pPr.append(shd)


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=9)


def add_header(section, text):
    hdr = section.header
    p = hdr.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_run_font(r, size=9, cn=CN)
    r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)


# ---------------------------------------------------------------- 文档初始化
doc = Document()

# 页面设置
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.5)
sec.bottom_margin = Cm(2.3)
sec.left_margin = Cm(3.0)
sec.right_margin = Cm(2.6)
sec.header_distance = Cm(1.4)
sec.footer_distance = Cm(1.3)
sec.different_first_page_header_footer = True

normal = doc.styles["Normal"]
normal.font.name = EN
normal.font.size = Pt(10.5)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), CN)
normal.paragraph_format.line_spacing = 1.36
normal.paragraph_format.space_after = Pt(3)

for sname in ("Heading 1", "Heading 2", "Heading 3"):
    st = doc.styles[sname]
    st.font.name = CN_HEI
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.font.bold = True
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CN_HEI)

add_header(sec, "杭州电子科技大学《机器学习课程实践》实验报告")
add_page_number_footer(sec)

# ================================================================ 封面
for _ in range(2):
    doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("杭州电子科技大学")
set_run_font(r, size=22, bold=True, cn=CN, en=EN)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(10)
r = p.add_run("《机器学习课程实践》")
set_run_font(r, size=18, bold=True, cn=CN_HEI)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
r = p.add_run("实 验 报 告")
set_run_font(r, size=30, bold=True, cn=CN_HEI)

for _ in range(3):
    doc.add_paragraph()

cover_rows = [
    ("题　　目", "基于机器学习的流域径流多步预测"),
    ("专　　业", ""),
    ("学　　号", ""),
    ("姓　　名", ""),
    ("指导教师", ""),
    ("成　　绩", ""),
]
ct = doc.add_table(rows=0, cols=2)
ct.alignment = WD_TABLE_ALIGNMENT.CENTER
ct.autofit = False
for label, value in cover_rows:
    cells = ct.add_row().cells
    for i, txt in enumerate((label, value)):
        cells[i].text = ""
        p = cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(5)
        r = p.add_run(txt)
        set_run_font(r, size=12, cn=CN)
    cells[0].width = Cm(4.2)
    cells[1].width = Cm(9.6)

# 封面表格只保留底边框（下划线式）
for row in ct.rows:
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "none")
            borders.append(el)
        tcPr.append(borders)
    bottom = row.cells[1]._tc.get_or_add_tcPr()
    bd = OxmlElement("w:tcBorders")
    el = OxmlElement("w:bottom")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), "6")
    el.set(qn("w:color"), "808080")
    bd.append(el)
    bottom.append(bd)

for _ in range(3):
    doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("二〇二六年十一月")
set_run_font(r, size=14, cn=CN)

doc.add_page_break()

print("封面完成")
# ================================================================ 一、任务要求
h1("一、任务要求")

h2("1.1 题目")
body("本实践以美国地质调查局（USGS）01047000 号水文站 2000—2004 年的逐日观测数据为对象，要求综合运用特征工程、机器学习、深度学习和集成学习方法，分别采用"
     "多步直接预测与多输出预测两种策略，预测该流域未来 1—7 天的日径流（Discharge），并对各模型精度作出定量评价与对比分析。所用气象因子来自 Daymet V4 R1 "
     "逐日格点数据集，共 1 827 天、9 个字段。")

h2("1.2 题目解析")
body("（1）任务类型。目标变量为连续型径流量，属于监督学习中的回归任务；样本按时间先后排列，且要求一次给出未来 7 天的值，因此进一步属于时间序列多步预测"
     "（multi-step-ahead forecasting）。这一判断决定了三件事：评价指标必须用回归指标而非分类指标；数据切分必须按时间顺序、不能随机打乱；交叉验证必须用前向链式"
     "切分（TimeSeriesSplit），不能用普通 K 折。")
body("（2）预测场景。t 日观测结束后，利用截至 t 日已经发生的径流与气象信息，输出 Q(t+1)…Q(t+7)。模型参数在测试年不更新，也不输入未来的实测气象，因此这是一个"
     "“逐日起报、固定参数”的滚动预测场景，而不是“年初一次性预报全年”。")
body("（3）数据划分。2000—2003 年共 4 年用于训练，2004 年用于测试；按 30 天回看窗与 7 天预见期对齐后，可建模起报点共 1 797 个，其中训练 1 395 个、测试 359 个。"
     "所有模型使用完全相同的 359 个测试起报点，保证横向可比。")
body("（4）主要难点。第一，径流强右偏（偏度 4.96，均值 610.89、中位数仅 244.00 ft³/s），洪峰是稀疏但量级极大的事件，误差被少数高值主导；第二，各变量量纲差异"
     "大（降水量 0—50 mm/day、短波辐射数百 W/m²、径流数千 ft³/s），不归一化无法直接建模；第三，样本量受年份限制（仅 4 年），深度模型容易过拟合；第四，未来 3—7 天"
     "的降水无法获知，而降水恰是径流的主要驱动，这构成远期精度的信息上限。")
body("（5）解题思路。先做数据分析摸清分布与相关性，据此确定目标变换与滞后结构；再做特征工程构造候选特征并做因果化的归一化与选择；然后用人工神经网络、随机森林、"
     "LSTM 三类个体模型分别在两种策略下建模，用 Grid Search CV 定超参数；最后用简单平均与 Stacking 融合，并在统一口径下与朴素基线对照，判断模型是否真正有效。")

h2("1.3 工作分解与模型规模")
body("按实验指导书的四段式要求，全部工作可分解为表 1-1 所示环节。其中个体模型总数为 3（方法）×8（7 个直接策略模型 + 1 个多输出模型）= 24，集成模型总数为 "
     "2（集成方法）×8 = 16，合计 40 个模型，与指导书“注意事项”的数量要求一致。")
table(
    ["环节", "主要工作", "关键产出"],
    [
        ["数据分析", "字段含义解析；时间序列、直方图、箱型图、皮尔逊相关系数热力图；自相关（ACF）与滞后互相关（CCF）", "8 类图表与统计量，确定 log 变换、滞后阶数与归一化方案"],
        ["特征工程", "季节/月份编码、滞后特征、滑动窗口特征、派生气象特征；归一化方法对比；特征选择方法对比", "84 列工程特征、87 列候选；plus_Q（51 维）/compact（9 维）两套建模输入"],
        ["个体建模", "ANN、随机森林、LSTM ×（多步直接 7 个 + 多输出 1 个）；Grid Search CV 定超参数", "24 个个体模型及逐预见期指标"],
        ["集成学习", "简单平均；Stacking（Ridge 元学习器，基于因果 OOF 预测训练）", "16 个集成模型及逐预见期指标"],
        ["精度评价", "R²、RMSE、MAE；三 seed 随机性复核；持久性等 4 类朴素基线对照；洪峰过程线误差分析", "6 组主表、基线表、对照表与 12 幅图"],
    ],
    widths=[2.4, 8.2, 5.2],
    title="表 1-1　工作分解与产出",
    note="注：数据划分、评价口径与模型数量严格按《实验指导书》执行；表中“24 + 16”仅统计指导书要求的必做模型，不含消融与改进实验。",
)

h2("1.4 技术路线")
body("整体技术路线如图 1-1：数据经分析与特征工程后按时间切分，进入“三类个体模型 × 两种策略 + 两类集成”的建模主干，最后统一评价并回到误差分析。图中同时标出"
     "了防泄漏的三处关键控制点（切分、折内缩放、折外预测）。", indent=True)
figure(FIG / "图1-1_技术路线.png", "图 1-1　总体技术路线", width=13.6, max_h=10.6)

# ================================================================ 二、详细分工
h1("二、详细分工")
body("本实践按“方案与建模—数据分析—特征工程”三条线分工，每人对所负责模块的代码与结论负责，交叉复核对方的中间结果。具体分工见表 2-1。")
table(
    ["成员", "承担任务", "对应章节"],
    [
        ["组长（姓名待填）", "总体方案与技术路线设计；径流预测建模与集成（两种策略、ANN/RF/LSTM、简单平均与 Stacking）；训练与评价流程实现；结果汇总与报告统稿", "三、四、六章"],
        ["组员 1（姓名待填）", "数据分析：字段解析、时间序列/直方图/箱型图/相关性可视化与 ACF、CCF 分析；可视化规范与中文字体配置；数据分析结论撰写", "3.1、4.2、五章部分"],
        ["组员 2（姓名待填）", "特征工程：季节与月份编码、滞后/窗口/派生特征、归一化与特征选择实现及方法对比；实验数据、图表与附件的整理归档", "3.2、4.3、附录"],
    ],
    widths=[3.0, 9.6, 3.2],
    title="表 2-1　小组成员分工",
    note="注：姓名、学号、专业等信息按实际填写；本表按实际承担内容分工，不以挂名方式分配。",
)
body("协作方式上，三人共用同一套数据切分与评价代码，任何人修改特征或切分协议必须同步更新实验记录，避免“各跑各的、结果不可比”。所有中间结果以 CSV 落盘，"
     "报告中的每个数字都能追溯到对应的结果文件。")

# ================================================================ 三、过程设计
h1("三、过程设计")
body("过程设计按“数据—特征—模型—集成—评价”五步展开。每一步都给出输入、处理目标与所用工具，使第四章的实现与本章的步骤一一对应。")

h2("3.1 数据分析")
body("第一步是理解数据。数据集共 9 个字段（表 3-1），其中雪水当量 Swe 在全部 1 827 天恒为 0，属常量列，建模前剔除，实际参与分析的变量为 7 个。"
     "径流为 USGS 实测流量，气象因子来自 Daymet V4 R1 逐日格点数据集[8]。")
table(
    ["字段", "中文含义", "单位", "说明"],
    [
        ["Date", "日期", "—", "2000-01-01 至 2004-12-31，逐日"],
        ["Discharge", "径流（流量）", "ft³/s", "预测目标，强右偏，偏度 4.96"],
        ["Prcp", "降水量", "mm/day", "右偏，多数日为无降水日，偏度 3.54"],
        ["Tmax", "最高气温", "°C", "与 Tmin 高度相关（r=0.93）"],
        ["Tmin", "最低气温", "°C", "与 Tmax 高度相关"],
        ["Srad", "短波辐射", "W/m²", "与气温中等相关"],
        ["Vp", "水汽压", "Pa", "与气温高度相关（r=0.88—0.93）"],
        ["Dayl", "日照时长", "s/day", "强季节周期"],
        ["Swe", "雪水当量", "kg/m²", "全期为 0，常量列，剔除"],
    ],
    widths=[2.3, 3.6, 2.2, 7.3],
    title="表 3-1　数据字段含义",
)
body("在此基础上完成四类可视化与三类相关性分析：时间序列图用于观察季节波动与洪峰；直方图用于判断偏度与是否需要目标变换；箱型图用于识别异常值与月度差异；"
     "相关系数热力图用于筛选线性相关变量。相关性分析分三层——径流自身的自相关（ACF）、变量间同期相关、气象因子对径流的滞后互相关（CCF），"
     "目的是回答“滞后特征该取到多少阶、气象因子是否值得作为输入”这两个直接决定特征工程的问题。")

h2("3.2 特征工程")
body("特征工程的目标是把一张 9 列的日表，变成机器学习模型可用的“因果特征矩阵”。按构造方式分为七组（表 3-2），共 84 列工程特征；若再加上当日实测流量"
     "与年内 sin/cos 两列，则为 87 列候选。")
table(
    ["特征组", "构造方式", "数量"],
    [
        ["原始当期气象", "Dayl、Prcp、Srad、Tmax、Tmin、Vp 的当日观测值", "6"],
        ["四季编码", "season_春/夏/秋/冬 的 one-hot 编码", "4"],
        ["月份标签", "month 整数（仅作分组与审计，不进入建模矩阵，避免与周期编码重复）", "1"],
        ["派生气象", "Tmean、Trange、Prcp_cum7、Prcp_cum30、Tmean_roll7、melt_index（融雪指数）", "6"],
        ["派生水文", "Discharge_diff1/3/7、Prcp_diff1、Prcp_roll3_sum、Prcp_roll14_sum、Prcp_heavy3（强降水指示）", "7"],
        ["滞后特征", "Discharge、Prcp、Tmax、Tmin、Vp、Srad 六变量 × 滞后 1/2/3/7/14/30 天", "36"],
        ["窗口特征", "Discharge、Prcp 两变量 × 窗口 3/7/14/30 天 × 均值/标准差/最大值", "24"],
        ["当日流量", "Discharge_current（当日实测流量，因果可用）", "1"],
        ["年内周期", "doy_sin、doy_cos 三角编码（供 LSTM 序列输入）", "2"],
    ],
    widths=[3.0, 10.6, 1.8],
    title="表 3-2　特征体系构成",
    note="注：合计 87 列候选，其中工程特征 84 列。滞后与窗口特征严格只向过去取数（shift/rolling 均不含当日之后的信息）。",
)
body("归一化方面，实现并对比了最小-最大归一化与 Z-score 标准化。Min-Max 把特征线性压缩到 [0,1]，在径流这类含极端洪峰的强右偏数据上，绝大多数样本会被压到接近 0 的窄区间，"
     "信息几乎被极端值吞掉；Z-score 标准化后各特征分布更均衡、对方差差异不敏感，因此最终采用 Z-score。关键约束是：缩放统计量只在当次训练集上估计，"
     "绝不使用测试期数据。")
body("特征选择方面，实现并对比了皮尔逊相关系数法与互信息法。前者只度量线性相关，后者基于信息熵、能捕捉非线性依赖。考虑到径流与气象因子之间明显是非线性关系，"
     "最终以互信息为准则，在每个训练折内部计算得分并保留 TOP 50，形成 plus_Q 输入（84 列折内 TOP 50 + 当日流量，共 51 维）；另设 compact 输入（当日流量、降水、"
     "最高/最低气温、水汽压、短波辐射、日照时长与年内 sin/cos，共 9 维）供多输出 LSTM 使用。折内选择保证同一折内“选特征”与“训练模型”看到的是同一段历史。")

h2("3.3 径流预测建模")
body("两种预测策略的差别在于“模型与预见期的对应关系”（表 3-3）。多步直接策略为每个预见期单独训练一个模型，模型只需专注一个目标；多输出策略由单个模型一次输出 7 列，"
     "可共享表示，但也更容易被远期噪声拖累。指导书要求两种策略都做，正是为了对比这一权衡。")
table(
    ["策略", "输入", "输出", "模型个数（每种方法）", "实现要点"],
    [
        ["多步直接", "截至 t 的因果特征（plus_Q）", "Q(t+h)，h=1…7 各一个模型", "7", "每个 h 独立选参、独立训练；h 越远样本可用区间越短"],
        ["多输出", "compact 9 维或工程特征", "Q(t+1)…Q(t+7) 共 7 列", "1", "ANN/RF 直接 fit 二维目标；LSTM 末端 Linear 原生输出 7 列"],
    ],
    widths=[2.4, 4.6, 4.4, 2.6, 4.6],
    title="表 3-3　两种多步预测策略",
)
body("三类个体模型的实现见表 3-4。三者刻意选择了不同的归纳偏置：ANN 学非线性映射、随机森林学分段常数与特征交互、LSTM 学序列依赖，从而使后续集成具备互补性。"
     "三种方法及集成的原始定义分别见文献[3][4][5]。")
table(
    ["模型", "实现", "关键设置", "多输出实现"],
    [
        ["ANN（MLP）", "scikit-learn MLPRegressor", "隐藏层与正则系数由网格搜索确定；关闭 early_stopping 以避免随机验证集造成的时间泄漏，max_iter=1000", "直接拟合 (n,7) 目标，未使用 MultiOutputRegressor"],
        ["随机森林", "scikit-learn RandomForestRegressor", "树数、最大深度、叶最小样本数由网格搜索确定；对量纲不敏感但仍统一缩放", "同上，原生支持多输出"],
        ["LSTM", "PyTorch 自定义 SequenceNet", "lookback=30，batch=32，Adam，梯度裁剪 1.0；目标取对数", "末端 Linear 直接输出 7 列，单次前向返回 (n,7)"],
    ],
    widths=[2.6, 4.4, 8.0, 4.2],
    title="表 3-4　三类个体模型",
)
body("超参数一律用 Grid Search CV 确定，搜索空间见表 3-5。树模型与 ANN 使用 3 折 TimeSeriesSplit 前向验证；LSTM 直接策略每阶段用 2 折，并对候选配置做 pilot 选择表示与目标损失；"
     "多输出 LSTM 按季度滚动的前向验证选择参数，候选中的 epoch（10/20/40）作为显式超参数参与搜索，不使用隐式早停，避免“选出来的轮数不可复现”。")
table(
    ["模型", "搜索空间", "验证方式", "评分"],
    [
        ["ANN", "hidden_layer_sizes {(64,), (128,), (64,32)} × alpha {1e-5, 1e-2} × learning_rate_init {0.001, 0.01}", "3 折 TimeSeriesSplit", "负 RMSE（标准化尺度）"],
        ["随机森林", "n_estimators {300, 500} × max_depth {None, 10, 15, 20} × min_samples_leaf {1, 5}", "3 折 TimeSeriesSplit", "负 RMSE"],
        ["LSTM（直接）", "units {32 / 64 / 64+32} × dropout {0.1, 0.2} × lr {0.001, 0.0005}，共 12 组；另选 log/原尺度目标损失", "每阶段 2 折 + pilot", "原尺度逐 h RMSE 均值"],
        ["LSTM（多输出）", "units {32, 64} × dropout {0.1, 0.2} × fixed_epochs {10, 20, 40}，共 12 组，lr=0.001", "季度 90 日起报窗前向验证", "原尺度逐 h RMSE 均值"],
    ],
    widths=[2.8, 8.6, 4.0, 3.8],
    title="表 3-5　超参数搜索空间",
    note="注：每次搜索只使用该次训练区间内、且经过 7 天净化的历史数据；测试年不参与任何选参。",
)

h2("3.4 集成学习")
body("集成部分实现两种方法。简单平均（Mean）在原始尺度直接取 (ANN + RF + LSTM)/3，不做加权，是最朴素也最不容易过拟合的融合。Stacking 用基模型预测作为元层输入、"
     "以 Ridge 为元学习器：直接策略为每个预见期各拟合一个元模型，输入是同一 h 下三个基模型的预测；多输出策略把 ANN、RF、LSTM 的 h1—h7 预测拼成 21 列，"
     "经标准化后拟合一个原生七输出 Ridge。")
body("Stacking 的关键在于元层的训练数据必须来自“基模型没见过的样本”，即折外预测（OOF）。本实践按时间分块构造因果 OOF：直接策略用 2002-01-01 至 2003-12-24 的 723 个"
     "起报点、分 3 块前向生成；多输出策略用 2001-07-01 至 2003-12-24 的 907 个起报点、按 10 个季度前向生成。元层只在这批 OOF 上训练，再作用于 2004 年基模型预测。"
     "此外，多输出任务还比较了 RF、简单平均、非负凸融合与“向 RF 收缩”的 Ridge 等候选，按各季度原尺度 RMSE 的算术平均择优，作为部署选择（SelectedFusion）。")

h2("3.5 评价口径与防泄漏设计")
body("评价指标定义为：R² = 1 − Σ(y−ŷ)² / Σ(y−ȳ)²，允许为负（负值说明不如“用均值预测”）；RMSE = √mean[(y−ŷ)²]；MAE = mean|y−ŷ|。三者均在指数逆变换后的原始径流尺度上计算，"
     "并明确：逐步指标的算术平均不等于把所有步长合并后的一个 RMSE，报告中的“七步平均”一律指先算各 h 指标再取算术平均。原始 Discharge 的单位以数据说明为准，"
     "本报告不做换算。指标定义与水文模型评价规范参照文献[9][10]。")
body("防泄漏是本实践最重视的工程纪律，具体落实为三条：其一，归一化与特征选择的统计量只在当次训练集上估计，测试集只做 transform；其二，交叉验证一律用前向链式的 "
     "TimeSeriesSplit，且训练与验证之间净化 7 天，保证训练集最晚的 h=7 目标日严格早于验证段首个起报日；其三，Stacking 元学习器只用 OOF 预测训练，绝不用基模型在训练集上的"
     "拟合值。此外，2004 年在前期实验中已被查看，因此本报告明确将其定位为回溯评价而非独立盲测，相关结论不外推为未来泛化性能。")
# ================================================================ 四、编码实现及结果
h1("四、编码实现及结果")

h2("4.1 开发环境与代码组织")
body("全部实验在 Windows 11 上完成，开发语言为 Python 3.12，主要依赖为 PyTorch 2.6.0、scikit-learn 1.5.1、NumPy 1.26.4、pandas 2.2.2 与 Matplotlib。"
     "代码按“配置—数据—特征—模型—训练—集成—评价—报告”分层组织，见表 4-1；所有中间结果以 CSV/JSON 落盘，图表以 PNG 保存，报告中的每个数字都可回溯到对应文件。")
table(
    ["模块", "职责", "关键内容"],
    [
        ["config.py", "全局配置", "路径、变量对照、滞后阶数、窗口长度、随机种子"],
        ["data_analysis.py", "数据分析", "统计量、时间序列/直方图/箱型图、相关性与 ACF/CCF"],
        ["feature_engineering.py", "特征工程", "编码、滞后、窗口、派生特征；Min-Max/Z-score；皮尔逊/互信息"],
        ["models.py / acceptance3_model.py", "模型定义", "ANN、RF 封装；PyTorch SequenceNet 与统一 RunoffRegressor"],
        ["acceptance3_v2_train.py", "直接策略训练", "7 个 h 的网格搜索、训练、折外预测与评估"],
        ["acceptance3_v3_stage_c.py", "多输出训练", "季度 OOF、原生七输出训练与冻结评估"],
        ["acceptance3_v3_fusion.py", "集成与部署选择", "简单平均、Stacking、凸融合、向 RF 收缩的元层"],
        ["evaluation.py / utils_plot.py", "评价与绘图", "R²/RMSE/MAE/NSE/PBIAS/MAPE；中文字体与统一配色"],
    ],
    widths=[5.0, 3.6, 7.4],
    title="表 4-1　代码模块与职责",
)
body("规模上，第二轮直接策略完整运行约 92 分钟；第三轮多输出分三阶段执行，阶段 A（输入筛选）约 6.5 分钟、阶段 B（结构选参）约 25.9 分钟、阶段 C（季度 OOF 与冻结）约 17.0 分钟。")
body("为便于复核，表 4-3—表 4-8、表 4-10、表 4-11 的原始数据均随报告提交于 data 目录（文件名以表号开头），报告中的每位数字都可以在这些 CSV 中找到对应行；"
     "表 4-9 的百分比取自第三次阶段验收报告的“集成相对基模型”对照表。")

# ------------------------------------------------ 4.2 数据分析结果
h2("4.2 数据分析结果")
body("本阶段按“时间序列—分布形态—相关性”三组图展开（图 4-1—图 4-5）。三组图给出的结论是：径流强右偏、季节性显著、短期记忆强，而气象因子的同期线性相关性很弱。")
figure(FIG / "图4-1_径流与气象因子时间序列.png", "图 4-1　01047000 流域径流与气象因子逐日时间序列", width=14.6, max_h=10.8)
figure(FIG / "图4-2_变量分布直方图.png", "图 4-2　各变量分布直方图", width=14.6)
figure(FIG / "图4-3_箱型图.png", "图 4-3　标准化后各变量箱型图（a）与径流按月分布（b）", width=14.2)
body("从图 4-1 看，径流呈清晰的季节波动，并叠加若干突发洪峰：冬春基流低而平稳，3—5 月与 11—12 月出现量级突增的洪水过程，全期最大日径流达 13 700 ft³/s"
     "（2003-12-18），2004 年测试期峰值为 7 470 ft³/s（11 月末），而中位数仅 244 ft³/s，极值可达中位数的 50 余倍。这一形态直接决定了后续两点处理："
     "目标取对数以压缩动态范围，评价必须同时看 RMSE 与 MAE。")
body("分布形态见图 4-2 与图 4-3。径流高度右偏（偏度 4.96），降水同样右偏且大量日期为 0，气温类变量接近对称。箱型图进一步显示径流的离散度远大于其他变量，"
     "并按月分组呈现明显的季节性：3—5 月整体抬升。量纲差异（降水 0—50、辐射数百、径流数千）说明不归一化无法直接送入距离敏感的模型。")
body("相关性分析结果（图 4-4、图 4-5）是本阶段最有信息量的部分。同期相关中，只有降水与径流存在中等正相关（r = 0.431），其余气象因子的 |r| 均不超过 0.10"
     "（日照 0.042、辐射 0.038、最低气温 −0.007、最高气温 −0.040、水汽压 −0.097），线性关系极弱；气温三兄弟之间则高度共线（r = 0.88—0.93）。"
     "径流自相关 lag1 = 0.77、lag7 = 0.40，衰减较快但记忆显著。滞后互相关在 0—40 天内没有出现显著峰值，说明气象因子对未来径流不存在“固定滞后的线性先行关系”。")
figure(FIG / "图4-4_皮尔逊相关系数热力图.png", "图 4-4　变量间皮尔逊相关系数矩阵（a）与同期气象因子相关系数（b）", width=14.4)
figure(FIG / "图4-5_自相关与滞后互相关.png", "图 4-5　径流自相关 ACF（a）与气象因子对径流的滞后互相关 CCF（b）", width=14.4)
body("这三点结论直接指导了特征工程：滞后特征必须做（lag1=0.77 说明昨日流量对今日有强预测力）；气象因子不能只靠同期线性项（|r| 太小），需要非线性模型与多阶滞后组合；"
     "气温类变量高度共线，应避免重复膨胀特征维度，改用派生量（日均温、日温差、融雪指数）表达其物理含义。")

# ------------------------------------------------ 4.3 特征工程结果
h2("4.3 特征工程结果")
body("按第 3.2 节方案构造的特征共 84 列，加当日流量与年内 sin/cos 后为 87 列候选。图 4-6 给出各类特征对径流的互信息总得分（以 80 个基础候选统计）：窗口特征以 18.854 居首，"
     "滞后特征 11.865 次之，派生气象 2.032、月份/日序编码 1.625、原始当期气象 1.524、季节编码 0.383。结论很明确——径流自身的近期状态是最强的预测信息，"
     "气象因子更多是“通过径流的历史状态间接起作用”。")
figure(FIG / "图4-6_特征类别互信息贡献.png", "图 4-6　各类特征对径流的互信息总得分", width=11.4,
       note="注：该图为 80 个基础候选的统计；后续补充日流量差分、累积降水与强降水指示等派生特征后，工程特征定为 84 列。")

body("归一化方法对比见图 4-7。原始数据量纲差异大且含极端值；Min-Max 归一化后大部分样本被压缩到接近 0 的窄区间（分布明显左挤），Z-score 标准化后各特征分布更均衡、"
     "对异常值更稳健。因此最终采用 Z-score，并在实现上强制“训练集 fit、验证/测试集 transform”。")
figure(FIG / "图4-7_归一化方法对比.png", "图 4-7　原始数据与两种归一化方法的取值分布对比", width=14.4)

body("特征选择对比见图 4-8。互信息法的 TOP 25 几乎全部是径流自身的滚动窗口与滞后项（Discharge_roll3_max、roll3_mean、lag1、roll7_max 等），皮尔逊法则给出高度相似的排序，"
     "但把部分非线性重要的特征（如 roll3_std、Prcp_roll30_max）排得更靠后。最终采用互信息 TOP 50 作为 plus_Q 输入，既保留非线性依赖，又避免 84 维全量输入在 1 395 个"
     "样本上过拟合。")
figure(FIG / "图4-8_特征选择方法对比.png", "图 4-8　互信息法（a）与皮尔逊相关系数法（b）特征重要性对比", width=14.4)

# ------------------------------------------------ 4.4 超参数
h2("4.4 超参数搜索结果")
body("经 Grid Search CV 确定的 8 组 LSTM 最终参数见表 4-2（直接策略 7 组 + 多输出 1 组）。7 组直接模型一致选择了 plus_Q 输入与 log 目标，说明折内互信息选出的特征与对数变换"
     "在短预见期与长预见期上都被验证有效；结构上多为单层 64 单元，说明在 1 395 个训练样本下，更宽的网络并没有带来收益。多输出模型选择 compact 9 维输入、64 单元、dropout 0.1、"
     "固定 10 轮，与直接策略的输入方案不同，原因是多输出共享一个表示，低维精简输入更不易过拟合。")
lstm_rows = [
    ["direct", "1", "[64]", "0.20", "0.0010", "log", "plus_Q", "0.0", "18"],
    ["direct", "2", "[64, 32]", "0.20", "0.0010", "log", "plus_Q", "0.0", "11"],
    ["direct", "3", "[64]", "0.10", "0.0010", "log", "plus_Q", "0.0", "12"],
    ["direct", "4", "[64, 32]", "0.10", "0.0005", "log", "plus_Q", "0.0", "10"],
    ["direct", "5", "[64]", "0.10", "0.0010", "log", "plus_Q", "0.0", "9"],
    ["direct", "6", "[64, 32]", "0.10", "0.0010", "log", "plus_Q", "0.0", "12"],
    ["direct", "7", "[32]", "0.10", "0.0010", "log", "plus_Q", "0.0", "32"],
    ["multi", "1—7", "[64]", "0.10", "0.0010", "log", "compact", "0.0", "10"],
]
table(
    ["策略", "预见期 h", "隐藏单元", "dropout", "学习率", "目标变换", "输入方案", "混合损失 λ", "epoch"],
    lstm_rows,
    widths=[1.9, 2.0, 2.6, 1.9, 2.2, 2.0, 2.4, 2.4, 1.6],
    title="表 4-2　8 组 LSTM 最终超参数",
    note="注：batch=32，优化器 Adam，梯度裁剪 1.0，lookback=30。“多输出策略”的 7 个列共享同一组参数。",
)
body("ANN 与随机森林的搜索结果呈现两点规律：其一，ANN 的隐藏层在 (64,)、(128,)、(64,32) 之间切换，更大的网络并未稳定胜出，正则系数多选 0.01、学习率多为 0.01，"
     "说明在小样本上强正则比大容量更重要；其二，随机森林在短预见期倾向较浅的树（max_depth 10—15）与较大的叶节点（min_samples_leaf 5），在长预见期则放开深度限制"
     "（max_depth=None），说明远期目标更依赖多棵深树的平均来降方差。")

# ------------------------------------------------ 4.5 直接策略结果
h2("4.5 多步直接策略预测结果")
body("直接策略的逐预见期结果见表 4-3（R²）与表 4-4（RMSE），图 4-9(a) 给出对应的折线。主表固定 seed42，每行均为相同的 359 个测试起报点。")
direct_models = ["ANN", "RF", "LSTM", "Mean", "Stacking"]
rows_r2, rows_rmse = [], []
for h in range(1, 8):
    idx = h - 1
    rows_r2.append([str(h)] + [f4(get("direct", m, "R2")[idx]) for m in direct_models])
    rows_rmse.append([str(h)] + [f2(get("direct", m, "RMSE")[idx]) for m in direct_models])
table(["h / 天", "ANN", "随机森林", "LSTM", "简单平均", "Stacking"], rows_r2,
      widths=[1.8, 2.6, 2.6, 2.6, 2.6, 2.6], title="表 4-3　多步直接策略各预见期 R²（seed42，n=359）")
table(["h / 天", "ANN", "随机森林", "LSTM", "简单平均", "Stacking"], rows_rmse,
      widths=[1.8, 2.6, 2.6, 2.6, 2.6, 2.6], title="表 4-4　多步直接策略各预见期 RMSE（原始尺度，ft³/s）")
body("直接策略的结论有三条。第一，随机森林在七步平均上最优（RMSE 765.08、R² 0.1728），且是唯一在全部 7 个预见期都保持正 R² 的个体模型，说明树模型对高偏、稀疏洪峰的"
     "稳健性最好。第二，LSTM 在 h=3、h=5 出现负 R²（−0.157、−0.195），表现为个别年份的相位错位：模型倾向于把前一周的洪峰“平移”到当期，一旦实际洪峰提前或推迟，"
     "单点误差会被放大。第三，ANN 的整体表现最差（七步平均 RMSE 835.16、R² 仅为 0.0115），且在 h=4、h=6 为负，说明在 1 395 个样本上，全连接网络不如树模型稳定。")
body("从 h=1 到 h=7 的整体趋势看，五种方法的 R² 均从 0.45—0.51 单调衰减到 0.00—0.10，RMSE 从约 590 升到约 820。这一衰减速度与“未来降水未知”的物理约束一致："
     "预见期越长，越多的径流变化由尚未发生的降水决定，任何只用历史信息的模型都无法补齐这部分。")
figure(FIG / "图4-9_逐预见期RMSE对比.png", "图 4-9　多步直接（a）与多输出（b）策略下各模型的逐预见期 RMSE", width=15.0,
       note="注：虚线为持久性基线（各 h 均预测起报日流量），用于判断模型是否优于“最简单的参照”。")

# ------------------------------------------------ 4.6 多输出结果
h2("4.6 多输出策略预测结果")
body("多输出策略的结果见表 4-5（R²）与表 4-6（RMSE），图 4-9(b) 为对应折线。")
rows_r2m, rows_rmse_m = [], []
for h in range(1, 8):
    idx = h - 1
    rows_r2m.append([str(h)] + [f4(get("multi", m, "R2")[idx]) for m in direct_models])
    rows_rmse_m.append([str(h)] + [f2(get("multi", m, "RMSE")[idx]) for m in direct_models])
table(["h / 天", "ANN", "随机森林", "LSTM", "简单平均", "Stacking"], rows_r2m,
      widths=[1.8, 2.6, 2.6, 2.6, 2.6, 2.6], title="表 4-5　多输出策略各预见期 R²（seed42，n=359）")
table(["h / 天", "ANN", "随机森林", "LSTM", "简单平均", "Stacking"], rows_rmse_m,
      widths=[1.8, 2.6, 2.6, 2.6, 2.6, 2.6], title="表 4-6　多输出策略各预见期 RMSE（原始尺度，ft³/s）")
body("多输出策略下随机森林同样最优（七步平均 RMSE 757.10、R² 0.1916），且比其直接策略版本更好（757.10 对 765.08）。原因在于多输出让 7 个目标共享同一棵树的分裂结构，"
     "相当于用 7 个相关任务互相“借力”，在小样本下起到了隐式正则的作用。LSTM 多输出的七步平均 RMSE 为 793.71（R² 0.1152），明显优于其直接策略版本 819.62，"
     "但 h=7 的 R² 仍为 −0.0012，说明 7 天的联合输出对最远一步仍然无力。ANN 多输出最差（854.10、R² −0.0280），在 h=6、h=7 的 R² 分别为 −0.183、−0.239，出现系统性低估。")
body("对比两种策略可以看出一个清晰的分工：随机森林和 LSTM 更适合多输出（共享表示带来正则收益），ANN 则在两种策略下都不稳定。这与样本量直接相关——"
     "多输出把 7 个预测任务绑在一个目标函数里，只有参数量与表达能力匹配时才是正则，否则反而互相干扰。")

# ------------------------------------------------ 4.7 汇总
h2("4.7 集成效果、基线对照与误差分析")
body("十种“策略 × 模型”组合的七步平均指标汇总于表 4-7，图 4-10 以柱状图直观给出 RMSE 与 R² 的对比。")
summary_rows = []
for m in direct_models:
    d_r2, d_rmse, d_mae = get("direct", m, "R2"), get("direct", m, "RMSE"), get("direct", m, "MAE")
    m_r2, m_rmse, m_mae = get("multi", m, "R2"), get("multi", m, "RMSE"), get("multi", m, "MAE")
    summary_rows.append([m, f4(avg(d_r2)), f2(avg(d_rmse)), f2(avg(d_mae)),
                         f4(avg(m_r2)), f2(avg(m_rmse)), f2(avg(m_mae))])
table(["模型", "直接 R²", "直接 RMSE", "直接 MAE", "多输出 R²", "多输出 RMSE", "多输出 MAE"], summary_rows,
      widths=[2.4, 2.2, 2.4, 2.2, 2.2, 2.4, 2.4],
      title="表 4-7　各模型七步平均指标汇总（seed42）",
      note="注：七步平均指先对每个 h 计算指标，再对 7 个 h 取算术平均，不等价于把 2 513 个预测混在一起算一个指标。")
figure(FIG / "图4-10_七步平均汇总.png", "图 4-10　七步平均 RMSE（a）与 R²（b）汇总", width=15.0)

body("模型是否“有用”，不能只看模型之间的排名，还要看是否优于朴素基线。表 4-8 给出四类基线：持久性（各 h 均预测起报日流量）、训练均值、季节均值、7 日流量 Ridge。"
     "持久性基线在 h≥3 时 R² 已转负，七步平均 RMSE 达 911.38；季节均值与 7 日 Ridge 分别稳定在 791.55 与 780.07。对照之下，随机森林（757.10 / 765.08）与两种集成"
     "（768—786）确实优于全部基线，但优势有限；ANN 直接策略（835.16）甚至不如季节均值基线（791.55）。这提醒我们：在小样本、强噪声的水文序列上，“模型更复杂”并不自动等于“更准”。")
base_rows_show = []
for model, label in [("Persistence", "持久性"), ("Training_mean", "训练均值"), ("Seasonal_mean", "季节均值"), ("AR7_Ridge", "7 日流量 Ridge")]:
    r2s = get_base(model, "R2")
    rm = get_base(model, "RMSE")
    base_rows_show.append([label] + [f4(x) for x in r2s] + [f2(avg(rm))])
table(["基线", "h=1", "h=2", "h=3", "h=4", "h=5", "h=6", "h=7", "平均 RMSE"], base_rows_show,
      widths=[3.0, 1.55, 1.55, 1.55, 1.55, 1.55, 1.55, 1.55, 2.1],
      title="表 4-8　朴素基线的逐预见期 R² 与七步平均 RMSE",
      note="注：基线与本报告模型使用完全相同的 359 个测试起报点与真值，可直接作为技能参照。")

body("集成是否带来增益，要看它相对“同一预见期最好的基模型”表现如何（表 4-9）。结论并不乐观：简单平均只在直接策略的 2 个预见期优于最佳基模型，Stacking 同样为 2/7，"
     "而多输出策略下两种集成的 7 个预见期全部不如最佳基模型（0/7）。也就是说，本实践中的集成主要起“降低方差、避免单模型踩坑”的作用，而不是“提升精度”。"
     "这与三个基模型误差高度相关（它们共享同一套特征与同一段训练历史）直接相关——集成要有增益，前提是基模型“错得不一样”。")
ens_rows = [
    ["直接 · 简单平均", "-0.32", "+4.08", "+8.29", "+4.20", "+0.92", "+2.37", "-0.48", "2 / 7"],
    ["直接 · Stacking", "-0.11", "+2.63", "+9.75", "+10.86", "+0.51", "+1.42", "-2.32", "2 / 7"],
    ["多输出 · 简单平均", "+2.99", "+5.12", "+2.66", "+3.15", "+2.11", "+2.53", "+1.79", "0 / 7"],
    ["多输出 · Stacking", "+2.48", "+2.16", "+1.26", "+0.82", "+1.54", "+0.71", "+1.61", "0 / 7"],
]
table(["集成", "h=1", "h=2", "h=3", "h=4", "h=5", "h=6", "h=7", "优于最佳基模型的步数"], ens_rows,
      widths=[3.6, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3.2],
      title="表 4-9　集成相对同预见期最佳基模型的 RMSE 变化（%，负值表示更好）",
      note="注：逐 h 最佳基模型在直接策略下为 RF（h=1—4、h=6）、ANN（h=5）、LSTM（h=7）；多输出策略下 7 个 h 均为 RF。该“逐 h 最佳”只是事后诊断参照，不是一套已部署的选择器。数据取自第三次阶段验收报告的“集成相对基模型”对照表。")

figure(FIG / "图4-11_多输出Stacking过程线.png", "图 4-11　多输出 Stacking 策略在 2004 年的滚动预测过程线（h=1/3/7）", width=14.6,
       note="注：run_id = formal_20261101_v3_stage_d；横轴为目标日期，观测与预测逐日对齐。")
body("过程线最能暴露误差的来源。图 4-11 给出多输出 Stacking 在 2004 年的逐日滚动预测：h=1 时预测能跟上季节与基流，但洪峰已开始被低估；h=3 时峰值被明显削平；"
     "h=7 时不仅峰值被削平，峰现时间也出现前后错位，11 月末与 12 月下旬的两次大洪水（观测峰值约 7 470 与 5 900 ft³/s）被系统性低估三到五成。这正是“R² 在远期转负、"
     "但 MAE 增长相对温和”的原因——模型抓住了季节与基流，抓不住未发生的降水所带来的峰量。")


# ------------------------------------------------ 4.8 改进验证
h2("4.8 结课后的改进验证")
body("在三次阶段验收完成后，又针对两个已定位的问题做了补充实验，用于验证改进方向是否成立。两项结果均单独列出，不与第 4.5—4.7 节的冻结结果混用。")
body("（1）Stacking 元层过拟合的修复。直接策略 Stacking 在 h=4 的 R² 为 −0.0699，是典型的元层在小样本 OOF 上过拟合。修正在不重训任何基模型、不触碰测试集的前提下进行："
     "在净化前向 OOF 上对候选元层做因果交叉验证选参，最终选中的是“向随机森林收缩”的 Ridge，即 最终预测 = RF + 0.25 × (Ridge 预测 − RF)。结果见表 4-10。")
meta_rows = [
    ["1", "0.5051", "0.5111", "592.35", "588.72"],
    ["2", "0.2303", "0.3115", "741.57", "701.36"],
    ["3", "0.0402", "0.2544", "828.80", "730.45"],
    ["4", "-0.0699", "0.1598", "875.39", "775.72"],
    ["5", "0.0338", "0.0732", "832.04", "814.89"],
    ["6", "0.0344", "0.1177", "831.85", "795.16"],
    ["7", "0.1033", "0.0815", "801.65", "811.33"],
    ["七步平均", "0.1253", "0.2156", "786.24", "745.38"],
]
table(["h / 天", "修复前 R²", "修复后 R²", "修复前 RMSE", "修复后 RMSE"], meta_rows,
      widths=[2.4, 3.2, 3.2, 3.4, 3.4],
      title="表 4-10　直接策略 Stacking 元层收缩修复前后对比（seed42，n=359）",
      note="注：元层候选与选择规则在 2004 年评价数组构造前冻结；修复后 2 513 个预测中仅 1 个为负值。h=7 略有退化，不按 h 事后拼接“最优”结果。")
body("修复后直接 Stacking 的七步平均 RMSE 由 786.24 降至 745.38（−5.20%），R² 由 0.1253 升至 0.2156，h=4 由负转正（−0.0699 → 0.1598）。需要说明的是，这是“修正过拟合”，"
     "不是“突破信息上限”，因此并未改变第 4.7 节关于远期精度受限的判断。")
body("（2）直接 LSTM 的残差跳连改进。原直接 LSTM 采用 log 目标、普通输出头，在 h≥3 上出现较大波动。改进方案把“以当日流量为基准的残差”作为学习目标，"
     "即网络只需要学 Q(t+h) 相对 Q(t) 的变化量，末层零初始化使初始预测恰为持久性。在与原报告完全相同的 359 个起报点上，三 seed 的对照结果见表 4-11。")
res_rows = [
    ["1", "611.02", "647.52", "+5.97", "0.4733", "0.4086"],
    ["2", "762.65", "717.33", "−5.94", "0.1855", "0.2798"],
    ["3", "852.98", "733.19", "−14.04", "-0.0194", "0.2488"],
    ["4", "845.02", "753.70", "−10.81", "0.0030", "0.2069"],
    ["5", "871.75", "813.34", "−6.70", "-0.0627", "0.0767"],
    ["6", "863.52", "819.74", "−5.07", "-0.0413", "0.0622"],
    ["7", "833.01", "824.79", "−0.99", "0.0317", "0.0508"],
    ["七步平均", "805.71", "758.51", "−5.86", "0.0814", "0.1905"],
]
table(["h / 天", "原正式 RMSE", "改进后 RMSE", "RMSE 变化 / %", "原平均 R²", "改进后平均 R²"], res_rows,
      widths=[2.2, 3.0, 3.0, 2.8, 2.6, 3.0],
      title="表 4-11　直接 LSTM 残差跳连改进与原正式结果对照（三 seed 七步平均）",
      note="注：两列 RMSE 均为“先算各 h、再对 h 与 seed 取算术平均”，非扁平 RMSE；改善 seed 数为 h=1 的 0/3、h=2—h=6 的 3/3、h=7 的 2/3。")
body("改进使三 seed 七步平均 RMSE 从 805.71 降至 758.51（−5.86%），MAE 从 387.93 降至 325.68（−16.05%），R² 由 0.0814 升至 0.1905。但必须同时报告两点不利事实："
     "其一，h=1 的 RMSE 反而退化 5.97%（611.02 → 647.52），即“以当日流量为基准”的残差结构在最短预见期上不如直接回归；其二，h=7 的均值改善仅 0.99%，且只有 2/3 的 seed 改善。"
     "因此这项改进不能被表述为“全面升级”，只能作为后续优化方向的一项证据。")
body("综合两项补充实验，可以得出一个对后续工作更有价值的结论：在不引入新数据的前提下，模型层面的收益主要来自“修正过拟合”和“改变学习目标”，量级在 5% 上下；"
     "而真正决定 h≥3 精度的，仍是能否获得未来降水信息。这一点将在 6.2 节展开。")
# ================================================================ 五、程序调试
h1("五、程序调试")
body("本实践在实现过程中遇到并解决了七类问题。它们的共同点是：都不会让程序报错退出，而是让程序“跑出看起来很正常的错误结果”，因此必须靠专门的检查才能发现。")

h3("（1）目标强右偏，模型只学到基流")
body("现象：早期直接用原始流量做回归时，训练损失下降但验证集 R² 在 h≥3 上长期为负，预测曲线几乎是一条贴着基流的平线。原因：平方损失被少数洪峰主导，模型为降低总体损失"
     "选择“宁可不预测洪峰”。处理：目标取 log 后再建模，预测时用 exp 逆变换（保证预测恒为正），评价仍在原始尺度进行。验证：后续受控实验在相同训练预算下对比了 log、sqrt 与"
     "原尺度三种损失空间，2002 年 h=1 的 RMSE 分别为 342.51、483.07、543.46，log 空间最好，说明该处理不是“习惯做法”，而是有对照支撑的选择。")

h3("（2）归一化统计量用到了测试期数据")
body("现象：归一化方法对比表中，测试期的分布看起来“过分离谱”，且同一样本在训练/测试两次运行中的标准化值不一致。原因：对比脚本图省事，用全量数据 fit 了 scaler。"
     "处理：改为仅在训练集上 fit，测试集只 transform；并补写单元测试，人为把测试期流量放大 5 倍，验证 scaler 的均值/方差不受测试期影响。验证：测试通过，且报告中"
     "所有标准化结果均来自“折内 fit”。")

h3("（3）ANN 的 early_stopping 悄悄破坏了时间顺序")
body("现象：ANN 的验证曲线异常平滑，但测试表现明显不如交叉验证成绩。原因：scikit-learn 的 early_stopping 默认从训练数据中随机抽取 10% 作验证集，等于把未来的样本"
     "混进了训练过程，既造成泄漏，也让“停止轮数”不可复现。处理：统一设置 early_stopping=False，并把 max_iter 提高到 1000 以保证收敛；停止条件改由外层网格搜索决定。"
     "验证：补写断言测试，确认模型实例的 early_stopping 为 False 且 max_iter ≥ 1000。")

h3("（4）多输出 Stacking 的 OOF 预测维度对不上")
body("现象：多输出策略的 Stacking 结果异常差，元层输入的形状与 7 列目标不匹配，程序靠广播“凑”出了结果。原因：生成 OOF 时复用了直接策略的模型构造，输出维度仍是 1，"
     "而多输出基模型应当输出 7 列。处理：为多输出 OOF 单独指定 output_dim=7，并补齐形状断言。验证：修复后 Stacking 的指标回到合理区间（七步平均 RMSE 768.22），"
     "与简单平均、RF 的相对关系符合预期。")

h3("（5）Stacking 元层在 h=4 严重过拟合")
body("现象：直接策略 Stacking 在 h=4 的 R² 为 −0.0699，比任一基模型都差。原因：元层 Ridge 在仅 723 条因果 OOF 上学习了过多噪声，迁移到 2004 年即失效。处理：在 OOF 上"
     "做因果交叉验证，比较 RF、简单平均、非负凸融合、不同 alpha 的 Ridge 与“向 RF 收缩”的变体，最终选择收缩系数 0.25 的 shrinkRF；选择规则在测试集评价前冻结。"
     "验证：h=4 的 R² 由 −0.0699 修复到 0.1598，七步平均 R² 由 0.1253 升至 0.2156，且 2 513 个预测中仅 1 个负值（表 4-10）。")

h3("（6）时间编码重复进入特征矩阵")
body("现象：特征重要性排序中 month、doy_sin、doy_cos 与四季 one-hot 同时出现，树模型在同一信息上反复分裂，且不同轮次的特征数对不上（80 列与 84 列混用）。原因：早期版本"
     "同时保留了整数月份标签与三角编码。处理：V6 起移除周期正余弦列，只保留 season 与 month 用于分组；第三轮为 LSTM 序列输入单独补回 doy_sin/doy_cos 两列，并在文档"
     "中固定“工程特征 84 列、候选 87 列”的口径。验证：补写测试，断言特征构造函数不再生成 doy_sin/month_sin 等列，同时保留 season/month。")

h3("（7）两套策略的训练轮数协议不一致")
body("现象：直接策略用“两内折最佳轮数的中位数”作为最终轮数，多输出策略却直接固定 10 轮，导致两者的“训练预算”不可比，也无法解释 10 轮是否早停。原因：两套代码由不同轮次"
     "迭代而来，选轮逻辑没有统一。处理：把 epoch 提升为网格搜索的显式超参数（10/20/40），由同一套前向验证选择，不再用折间中位数替代；报告中明确说明“10 轮是候选之一，"
     "不代表程序异常提前结束”。验证：8 组最终参数（表 4-2）均可由候选表复现。")

body("除上述七项外，还处理了绘图中文乱码与负号显示（统一设置字体与 unicode_minus）、以及训练中断后的结果恢复（以 manifest 与 verification 文件校验模型完整性）等工程问题。"
     "这些工作不产生精度提升，但决定了结果是否可信、是否可复现。")

# ================================================================ 六、总结
h1("六、总结")

h2("6.1 本实践所用的技术与模型")
table(
    ["环节", "技术与方法"],
    [
        ["数据分析", "时间序列折线图、直方图（偏度/峰度）、箱型图（分组与异常值）、皮尔逊相关系数矩阵、自相关 ACF、滞后互相关 CCF"],
        ["特征工程", "四季 one-hot、月份与年内三角编码、滞后特征（1/2/3/7/14/30 天）、滑动窗口统计（均值/标准差/最大值）、派生气象与水文特征；Min-Max 与 Z-score 归一化；皮尔逊与互信息特征选择"],
        ["个体模型", "人工神经网络（MLP）、随机森林、LSTM（PyTorch，lookback=30）；多步直接策略与多输出策略；目标 log 变换与逆变换"],
        ["集成学习", "简单平均（原尺度算术平均）、Stacking（Ridge 元学习器 + 因果 OOF；含向 RF 收缩的 shrinkRF 变体）"],
        ["调参与评估", "Grid Search CV + TimeSeriesSplit（含 7 天净化）；R²、RMSE、MAE（另算 NSE、PBIAS、MAPE 作参考）；三 seed 随机性复核；四类朴素基线对照"],
    ],
    widths=[3.0, 12.6],
    title="表 6-1　技术、模型与算法清单",
)

h2("6.2 当前模型存在的问题与改进方案")
body("第一，远期精度受信息上限约束。h≥3 的 R² 普遍低于 0.2 并出现负值，而持久性基线在 h≥3 也转负，说明瓶颈不在模型容量，而在“t 日看不到 t+3—t+7 的降水”。"
     "探查实验也支持这一判断：在相同特征与切分下，梯度提升（HistGradientBoosting）的 R² 在直接与多输出的全部 14 个“策略 × h”组合上都低于随机森林（差值恒为负）。"
     "改进方向是引入数值天气预报（NWP）降水作为输入，或把预测目标改为“给定预报降水的条件预测”；LSTM 在径流预测中的典型用法见文献[6]，"
     "融合气象预报做全球洪水预警的实践见文献[7]。")
body("第二，样本量不足。仅 4 年、1 395 个训练起报点，其中 2002—2003 还要抽作 OOF 供元层训练，实际可用于基模型拟合的历史更短；洪水事件样本尤其稀少。"
     "改进方向是扩充年份或引入邻近流域做迁移学习，也可用时间序列增广（分块重采样、加噪）缓解，但必须保持因果顺序。")
body("第三，集成收益有限。三个基模型共享同一套特征与训练历史，误差高度相关，因此简单平均与 Stacking 多数预见期不如最佳基模型。改进方向是提高基模型多样性——"
     "例如让不同基模型使用不同输入方案或不同损失空间，而不是继续在同一套特征上叠加元学习器。")
body("第四，洪峰系统性低估。过程线显示大洪水峰值被削平三到五成，这是平方损失下的必然结果。改进方向包括：对高流量样本加权、改用分位数回归或 pinball 损失输出区间、"
     "对洪水事件单独建模，并在评价中加入峰值误差与洪峰时序误差等分档指标。")
body("第五，元层与选参的跨期稳定性不足。直接策略的元层在 2002/2003 与 2004 上表现不一致，说明“在历史内层选出的最优”未必在下一段历史上仍最优。改进方向是引入"
     "更强的正则（收缩、贝叶斯岭）与多期滚动验证，把“选参风险”本身纳入模型选择准则。上述方向汇总于表 6-2。")
table(
    ["问题", "证据", "改进方向", "可行性"],
    [
        ["远期受未来降水制约", "h≥3 的 R²<0.2 且持久性基线同区间为负；梯度提升全面劣于 RF", "引入 NWP 降水预报或条件预测", "需外部数据"],
        ["样本量不足", "1 395 个训练起报点，洪水事件数十个", "扩充年份/流域、迁移学习、因果增广", "需外部数据或额外实验"],
        ["集成增益有限", "多输出两种集成 0/7 步优于最佳基模型", "提高基模型多样性（不同输入/损失）", "可立即实验"],
        ["洪峰低估", "2004 年两次大洪水峰值低估 30%—50%", "样本加权、分位数回归、事件单独建模", "可立即实验"],
        ["元层跨期不稳", "h=4 由 −0.07 修复到 0.16；h=7 略退化", "收缩正则、滚动多期验证", "已完成部分验证"],
    ],
    widths=[3.4, 6.4, 5.0, 3.2],
    title="表 6-2　存在问题与改进方向",
)

h2("6.3 认识与心得体会")
body("第一，模型复杂度与效果之间没有单调关系。本次实践中，结构最简单的随机森林在两种策略下都取得了最好的七步平均成绩（757.10 / 765.08），而参数量更大的 LSTM 与"
     "全连接网络反而更差；集成方法也没有带来预期增益。这让我意识到，“上更复杂的模型”往往是在用参数量掩盖数据问题——在 1 395 个样本上，能学到的规律本来就有限，"
     "模型越复杂，越容易把噪声当规律。")
body("第二，数据决定精度上限，方法只能逼近上限。径流 lag1 高达 0.77，说明“昨天到今天”的变化几乎被历史信息解释完了；而 h≥3 之后，决定径流的是尚未发生的降水。"
     "任何只依赖历史观测的模型，在这一段都注定低效。把“模型不行”与“信息不够”区分开，是这次实践最重要的方法论收获：前者靠调参，后者只能靠换数据。")
body("第三，评估纪律比调参技巧更能决定结论的真伪。时间序列里，只要归一化统计量、特征选择或早停验证集有一处“看到了未来”，指标就会虚高，而且不会报错。"
     "本次为此专门写了三个回归测试（缩放泄漏、早停随机验证集、OOF 维度），并坚持前向切分 + 7 天净化 + OOF 元层。可以说，报告里那些“不好看”的数字（负 R²、集成不增益、"
     "远期不如基线），恰恰是评估纪律到位的结果。")
body("第四，工程的规范性直接决定结果能否被复核。把每次训练的参数、数据切分、随机种子与模型哈希写进 manifest，把指标从落盘预测重算而不是从内存变量抄写，"
     "才使得“三 seed 复核”“冻结后只评价一次”这些要求能够真正执行。这一点在今后的科研与工程中同样适用：可复现不是附加项，而是结论成立的前提。")
body("最后，从课程思政的角度看，径流预测直接服务于防洪抗旱与水资源调度。我国流域面积广、洪水灾害频发，把机器学习用在这样的国计民生问题上，让我更直观地体会到"
     "“技术为用”的含义：模型的价值不在于指标多漂亮，而在于能否诚实地告诉使用者——未来几天哪一段可信、哪一段不可信、风险在哪里。这也是我们小组在报告中坚持"
     "如实呈现负 R² 与失败案例的原因。")

# ================================================================ 参考文献
h1("参考文献")
refs = [
    "[1] 周志华. 机器学习[M]. 北京: 清华大学出版社, 2016.",
    "[2] 李航. 统计学习方法[M]. 2 版. 北京: 清华大学出版社, 2019.",
    "[3] BREIMAN L. Random forests[J]. Machine Learning, 2001, 45(1): 5-32.",
    "[4] HOCHREITER S, SCHMIDHUBER J. Long short-term memory[J]. Neural Computation, 1997, 9(8): 1735-1780.",
    "[5] WOLPERT D H. Stacked generalization[J]. Neural Networks, 1992, 5(2): 241-259.",
    "[6] KRATZERT F, KLOTZ D, BRENNER C, et al. Rainfall–runoff modelling using long short-term memory (LSTM) networks[J]. Hydrology and Earth System Sciences, 2018, 22(11): 6005-6022.",
    "[7] NEARING G, COHEN D, DUBE V, et al. Global prediction of extreme floods in ungauged watersheds[J]. Nature, 2024, 627: 559-563.",
    "[8] THORNTON M M, SHRESTHA R, WEI Y, et al. Daymet: daily surface weather data on a 1-km grid for North America, version 4 R1[DS]. Oak Ridge: ORNL DAAC, 2022.",
    "[9] NASH J E, SUTCLIFFE J V. River flow forecasting through conceptual models, part I: a discussion of principles[J]. Journal of Hydrology, 1970, 10(3): 282-290.",
    "[10] MORIASI D N, ARNOLD J G, VAN LIEW M W, et al. Model evaluation guidelines for systematic quantification of accuracy in watershed simulations[J]. Transactions of the ASABE, 2007, 50(3): 885-900.",
    "[11] 王贺, 刘鹏, 钱乾. 机器学习算法竞赛实战[M]. 北京: 人民邮电出版社, 2021.",
]
for ref in refs:
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.74)
    pf.first_line_indent = Cm(-0.74)
    pf.line_spacing = 1.25
    pf.space_after = Pt(2)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(ref)
    set_run_font(r, size=9.5)

# ================================================================ 附注
h1("附注")
body("（1）本报告的分工见表 2-1，每位同学对所负责模块的代码与结论负责；姓名、学号、专业、指导教师等信息按实际填写，未提供的信息未作编造。", indent=True, size=9.5)
body("（2）本报告电子版与全部工程代码按“组号+组长姓名+组员1姓名+组员2姓名”命名，打包提交至网络教学平台。", indent=True, size=9.5)
body("（3）报告中的全部数值均来自工程代码落盘的 CSV/JSON 结果文件，可按附录 B 的命令复现；未执行的实验、未验证的结论未写入本报告。", indent=True, size=9.5)

# ================================================================ 附录
h1("附录 A　核心代码")
body("受篇幅限制，此处列出与第三章步骤对应的关键代码片段（均摘自工程源码，含必要注释）。完整代码见随报告提交的工程包，目录清单见附录 B。")

h3("A.1 滞后特征与窗口特征（严格只向过去取数）")
code_block([
    "def add_lag_features(df):",
    "    \"\"\"滞后特征：x(t-1), x(t-2), ... 只使用历史信息。\"\"\"",
    "    d = df.copy()",
    "    for col in C.LAG_VARS:              # Discharge/Prcp/Tmax/Tmin/Vp/Srad",
    "        for k in C.LAG_STEPS:           # 1, 2, 3, 7, 14, 30",
    "            d[f\"{col}_lag{k}\"] = d[col].shift(k)",
    "    return d",
    "",
    "def add_rolling_features(df):",
    "    \"\"\"窗口特征：滑动均值/标准差/最大值。",
    "    min_periods=w 保证窗口仅覆盖 [t-w+1, t]，不含未来。\"\"\"",
    "    d = df.copy()",
    "    for col in C.ROLL_VARS:             # Discharge, Prcp",
    "        for w in C.ROLL_WINDOWS:        # 3, 7, 14, 30",
    "            r = d[col].rolling(window=w, min_periods=w)",
    "            d[f\"{col}_roll{w}_mean\"] = r.mean()",
    "            d[f\"{col}_roll{w}_std\"]  = r.std()",
    "            d[f\"{col}_roll{w}_max\"]  = r.max()",
    "    return d",
])

h3("A.2 LSTM 网络定义（原生七输出）")
code_block([
    "class SequenceNet(torch.nn.Module):",
    "    def __init__(self, n_features, units, dropout, output_dim):",
    "        super().__init__()",
    "        self.layers = torch.nn.ModuleList()",
    "        for i, width in enumerate(units):",
    "            self.layers.append(torch.nn.LSTM(",
    "                n_features if i == 0 else units[i - 1],",
    "                width, batch_first=True))",
    "        self.drop = torch.nn.Dropout(dropout)",
    "        self.head = torch.nn.Sequential(",
    "            torch.nn.Linear(units[-1], 32 if output_dim == 1 else 64),",
    "            torch.nn.ReLU(),",
    "            torch.nn.Linear(32 if output_dim == 1 else 64, output_dim))",
    "",
    "    def forward(self, x):",
    "        for layer in self.layers:",
    "            x, _ = layer(x)",
    "            x = self.drop(x)",
    "        return self.head(x[:, -1, :])   # 取末时刻隐状态；output_dim=7 即原生七输出",
])

h3("A.3 折内预处理与因果特征选择")
code_block([
    "def _preprocess_fit(self, ids, y):",
    "    data = load_dataset(self.dataset_path)",
    "    positions = data[\"positions\"][ids]",
    "    last = data[\"features\"][positions]",
    "    target = np.log(y[:, 0] if y.ndim == 2 else y)",
    "    # 特征选择只用当次训练行，且目标在 log 尺度上计算互信息",
    "    mi = mutual_info_regression(last, target, random_state=self.seed)",
    "    self.selected_ = np.argsort(-mi, kind=\"stable\")[:self.top_k]",
    "    # 缩放统计量同样只在训练行上估计；测试行只 transform",
    "    self.x_scaler_ = StandardScaler().fit(last[:, self.selected_])",
    "    self.y_scaler_ = StandardScaler().fit(target.reshape(-1, 1))",
])

h3("A.4 简单平均与 Stacking（元层只用 OOF 训练）")
code_block([
    "def mean_ensemble(preds):",
    "    \"\"\"preds: dict{model: ndarray(n,7)}，原尺度算术平均。\"\"\"",
    "    return np.mean(np.stack([preds[m] for m in (\"ANN\", \"RF\", \"LSTM\")]), axis=0)",
    "",
    "def stack_fit(oof_x, oof_y, alpha=100.0):",
    "    \"\"\"oof_x: (n_oof, 21) 三模型各 7 步的折外预测；oof_y: (n_oof, 7)。",
    "    元层只在 OOF 上训练，绝不使用基模型在训练集上的拟合值。\"\"\"",
    "    scaler = StandardScaler().fit(oof_x)",
    "    ridge = Ridge(alpha=alpha).fit(scaler.transform(oof_x), oof_y)",
    "    return scaler, ridge",
    "",
    "def stack_predict(scaler, ridge, test_x, rf_only, shrink=0.25):",
    "    # 向稳健基模型收缩，抑制小样本元层过拟合",
    "    raw = ridge.predict(scaler.transform(test_x))",
    "    return rf_only + shrink * (raw - rf_only)",
])

# ================================================================ 附录 B
h1("附录 B　代码清单与复现方式")
body("工程代码按职责分目录组织，主要文件见表 B-1。完整代码、训练记录与结果文件随报告打包提交；报告引用的图片见图 1-1 与图 4-1—4-11，数据表见工程 data 目录。")
table(
    ["文件/目录", "说明"],
    [
        ["config.py", "全局配置：路径、变量分组、滞后阶数与窗口长度、随机种子"],
        ["data_analysis.py", "数据分析与全部统计图（对应 4.2 节）"],
        ["feature_engineering.py", "特征构造、归一化对比、特征选择（对应 4.3 节）"],
        ["models.py、acceptance3_model.py", "ANN/RF 封装、PyTorch SequenceNet 与统一训练器（对应 3.3 节）"],
        ["acceptance3_v2_train.py", "多步直接策略训练与评估（对应 4.5 节）"],
        ["acceptance3_v3_stage_c.py", "多输出策略：季度 OOF、原生七输出训练与冻结评价（对应 4.6 节）"],
        ["acceptance3_v3_fusion.py、acceptance3_v3_evaluate.py", "集成与部署选择、冻结后的回溯评估（对应 3.4、4.7 节）"],
        ["evaluation.py、utils_plot.py", "指标计算与绘图工具"],
        ["runoff_training/", "结课后的改进实验（残差跳连、元层收缩等，对应 4.8 节）"],
        ["data/、figures/", "报告引用的结果 CSV 与图片"],
    ],
    widths=[5.6, 9.2],
    title="表 B-1　主要代码与数据文件",
)
body("复现方式：在工程根目录执行以下命令，即可重跑特征工程、模型训练与评价（完整重训耗时较长，报告中给出的是已冻结的正式结果）。")
code_block([
    "# 1) 数据分析与特征工程",
    "python code/data_analysis.py",
    "python code/feature_engineering.py",
    "",
    "# 2) 多步直接策略：网格搜索 + 训练 + 评估",
    "python code/acceptance3_v2_train.py --run-id formal_direct",
    "",
    "# 3) 多输出策略：季度 OOF + 原生七输出训练 + 冻结评估",
    "python code/acceptance3_v3_stage_c.py --run-id formal_multi",
    "python code/acceptance3_v3_evaluate.py --run-id formal_multi",
    "",
    "# 4) 集成与部署选择",
    "python code/acceptance3_v3_fusion.py",
], size=8.4)

# ---------------------------------------------------------------- 保存
OUTDIR.mkdir(parents=True, exist_ok=True)
doc.save(OUTDOC)
print("已生成：", OUTDOC)

# 粗略统计正文字数
import re
texts = []
for p in doc.paragraphs:
    texts.append(p.text)
for t in doc.tables:
    for row in t.rows:
        for cell in row.cells:
            texts.append(cell.text)
all_text = "".join(texts)
cn_chars = len(re.findall(r"[\u4e00-\u9fff]", all_text))
print("中文字符数：", cn_chars)
print("段落数：", len(doc.paragraphs), " 表格数：", len(doc.tables))
