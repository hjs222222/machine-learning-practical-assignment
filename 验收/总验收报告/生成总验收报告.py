# -*- coding: utf-8 -*-
"""生成总验收报告 .docx —— 与三次阶段验收报告保持一致的 Word 格式。"""
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = Path(__file__).resolve().parent
TARGET = OUT / "机器学习课程实践-总验收报告.docx"
FONT = "微软雅黑"


def set_east_asia(run, name=FONT):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def center(text, size=12, bold=False, before=0, after=0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    set_east_asia(r)
    return p


def para(text, after=6):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(after)
    for r in p.runs:
        set_east_asia(r)
    return p


def heading(text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        set_east_asia(r)
    return p


def table(headers, rows, font_size=8):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Shading Accent 1"
    for cell, text in zip(t.rows[0].cells, headers):
        cell.text = str(text)
        set_cell(cell, font_size, bold=True)
    repeat = OxmlElement("w:tblHeader")
    t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        cells = t.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
            set_cell(cell, font_size)
    return t


def set_cell(cell, size, bold=False):
    for p in cell.paragraphs:
        for run in p.runs:
            run.font.size = Pt(size)
            run.bold = bold
            set_east_asia(run)


def code_block(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    r.font.size = Pt(8.5)
    return p


doc = Document()
sec = doc.sections[0]
sec.top_margin = sec.bottom_margin = Cm(1.8)
sec.left_margin = sec.right_margin = Cm(1.8)
for name in ("Normal", "Heading 1", "Heading 2", "Title"):
    style = doc.styles[name]
    style.font.name = "微软雅黑"
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
doc.styles["Normal"].font.size = Pt(10)
doc.styles["Normal"].paragraph_format.space_after = Pt(6)

# ---- 封面 ----
for _ in range(3):
    doc.add_paragraph()
center("杭州电子科技大学", size=22, bold=True, after=4)
center("机器学习课程实践", size=22, bold=True, after=4)
center("实 验 报 告", size=26, bold=True, after=40)
center("题目：基于机器学习的流域径流多步预测（USGS 01047000）", size=13, after=30)
info_rows = [
    ("组号", "____"),
    ("组长（姓名）", "____"),
    ("组员1（姓名）", "____"),
    ("组员2（姓名）", "____"),
    ("指导教师", "____"),
    ("日期", "2026年11月"),
]
for label, value in info_rows:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(f"{label}：{value}")
    r.font.size = Pt(12)
    set_east_asia(r)
doc.add_page_break()

# ---- 一、任务要求 ----
heading("一、任务要求", 1)
para("题目：使用美国地质调查局（USGS）01047000 流域站的逐日观测数据，构建机器学习模型，预测未来 1—7 天的河道径流（流量）。")
para("题目解析：")
para("1. 任务类型：这是一个监督学习中的回归任务，并且是时间序列的多步（multi-step-ahead）预测问题。输入是截至当日 t 为止的已观测气象与径流信息，输出是未来 t+1 至 t+7 共 7 天的径流值。")
para("2. 数据规模：2000-01-01 至 2004-12-31 共 1827 天；数据字段为日期、径流（Discharge）以及日照时长、降水、短波辐射、雪水当量、最高/最低气温、水汽压共 6 个气象因子。雪水当量（Swe）恒为 0，作常量列剔除。")
para("3. 划分方式：按时间顺序，2000—2003 年（4 年）用于训练，2004 年（1 年，共 359 个起报点）用于测试，全程禁止打乱时间顺序、禁止泄漏未来信息。")
para("4. 解题思路：首先做数据分析，理解字段含义并用时间序列图、直方图、箱型图、相关性热力图刻画数据分布；其次做特征工程，构造季节/月份编码、滞后特征、滑动窗口特征、派生气象特征并做归一化与特征选择；再用人工神经网络（ANN/MLP）、随机森林（RF）、长短期记忆网络（LSTM）三种模型，分别按多步直接策略与多输出策略建模，以简单平均与 Stacking 两种集成方法融合；最后在原始径流尺度计算 R²、RMSE、MAE 评估。")

# ---- 二、详细分工 ----
heading("二、详细分工", 1)
para("下表成员姓名请据实填写；分工对应本报告第三部分的三个环节。")
table(
    ["成员", "承担任务"],
    [
        ["组长 ____",
         "总体方案与技术路线设计、径流预测建模与集成（直接/多输出策略、ANN/RF/LSTM、简单平均与 Stacking）、结果汇总"],
        ["组员1 ____", "数据分析（字段解析、时间序列/直方图/箱型图/相关性可视化）、报告数据分析章节撰写"],
        ["组员2 ____", "特征工程（季节与月份编码、滞后/窗口/派生特征、归一化与特征选择）、实验数据与图表整理"],
    ],
)

# ---- 三、过程设计 ----
heading("三、过程设计", 1)
para("本课题按三次阶段验收逐层推进，总过程分三个环节。")
heading("3.1 数据分析（第一次验收 · 子任务 1—5）", 2)
para("1. 字段中文含义：明确 9 个字段的名称、含义与单位（Discharge 为径流，单位 ft³/s；Prcp 为降水量，单位 mm/day；Tmax/Tmin 为最高/最低气温，单位 ℃；Vp 为水汽压，单位 Pa；Dayl 为日照时长，单位 s/day；Srad 为短波辐射，单位 W/m²；Swe 为雪水当量，单位 kg/m²）。")
para("2. 时间序列可视化：绘制径流与 6 个气象因子（Swe 常量列剔除后）的逐日序列，径流呈明显季节波动并叠加突发洪水脉冲。")
para("3. 直方图：描述各变量分布。径流高度右偏（偏度 4.96，均值 610.89、中位数 244.00 ft³/s），降水右偏（多数日无降水），气温类近似对称；量纲差异大，建模前须归一化。")
para("4. 箱型图：径流离散度最大、存在大量高值异常点（洪水事件）；按月分组显示春季（3—5 月）径流明显偏高，季节性显著。")
para("5. 皮尔逊相关系数热力图（含滞后）：分径流自相关（ACF）、变量同期相关、气象因子对径流的滞后互相关（CCF）三类。降水与径流同期相关性最强（r≈0.43），其余气象因子 |r|≤0.10；径流自相关 lag1=0.77、lag7=0.40，存在显著记忆效应；气象因子在 0—40 天滞后内无显著峰值，说明其对未来径流的直接线性预测能力有限。")
heading("3.2 特征工程（第一次验收 · 子任务 6—8）", 2)
para("1. 时间序列特征提取：构造 81 个候选特征——季节/月份编码（四季 one-hot、月份、doy_sin/doy_cos 三角周期编码）、滞后特征（Discharge/Prcp/Tmax/Tmin/Vp/Srad 的 1/2/3/7/14/30 天滞后）、窗口特征（Discharge/Prcp 的 3/7/14/30 天滑动均值/标准差/最大值）、派生气象特征（Tmean、Trange、Prcp_cum、melt_index 等）。")
para("2. 特征归一化：实现并对比 Min-Max 归一化与 Z-score 标准化，最终采用对异常值更稳健的 Z-score 标准化；归一化统计量仅在训练集上估计。")
para("3. 特征选择：实现并对比皮尔逊相关系数法与互信息法，最终按互信息法保留得分前 50 的特征。互信息最高的始终是径流自身的短窗口与滞后特征，说明径流近期状态是最强预测因子。")
heading("3.3 径流预测建模与集成（第二次、第三次验收）", 2)
para("基础模型：ANN（MLPRegressor）、随机森林（RandomForestRegressor）、LSTM（PyTorch SequenceNet，lookback=30）。目标取对数（log）建模，预测时指数逆变换回原始径流尺度。")
para("两种预测策略：")
table(
    ["策略", "做法", "模型个数/家族"],
    [
        ["多步直接策略（direct）", "对每个预见期 h=1…7 单独训练一个模型，输出当天 h 步后的径流", "7 个"],
        ["多输出策略（multi）", "一个模型同时输出未来 7 天，ANN/RF 直接拟合二维目标，LSTM 末尾 Linear 原生输出 7 列", "1 个"],
    ],
)
para("超参数确定：GridSearchCV + 3 折 TimeSeriesSplit 前向验证（不打乱时间顺序），以负 RMSE 为评分。")
para("两种集成方法：")
table(
    ["集成", "做法"],
    [
        ["简单平均（Mean）", "在原尺度对 ANN、RF、LSTM 三模型预测求平均"],
        ["Stacking", "用基模型预测作输入，拟合 Ridge 元学习器；直接策略每 h 一个元模型，多输出策略用 21 列基预测拟合单个原生七输出 Ridge"],
    ],
)
para("评价口径：R² = 1 − Σ(y−ŷ)²/Σ(y−ȳ)²（允许为负），RMSE/MAE 在逆变换后的原始径流尺度计算；全程防泄漏——监督特征选择、缩放统计量只在当次训练集上拟合，时间层间净化 7 天。")

# ---- 四、编码实现及结果 ----
heading("四、编码实现及结果", 1)
heading("4.1 环境与代码组织", 2)
para("操作系统：Windows 11；开发环境：Python 3 + PyTorch（CPU，版本 2.6.0+cpu）、scikit-learn、NumPy、pandas、matplotlib。")
para("核心代码模块：data_analysis.py（数据分析）、feature_engineering.py（特征工程）、acceptance3_model.py（RunoffRegressor 统一封装 LSTM/ANN/RF）、acceptance3_train.py 与 acceptance3_v2_train.py、acceptance3_v3_stage_c.py（直接/多输出训练与元层构建）、acceptance3_v3_fusion.py（融合策略）、acceptance3_meta_boost.py、acceptance3_meta_deploy.py（元层修复与部署）、acceptance3_gb_probe.py（梯度提升收益探查）。所有中间数值落盘为 CSV，图表保存为 PNG，结论可追溯、可复现。")
heading("4.2 结果总览", 2)
para("三次阶段验收全部子任务均已按要求完成。最终冻结主种子（seed42）在 2004 年测试集（359 个起报点）上的结果摘要如下。")
para("各模型与策略七步平均 RMSE（原始径流尺度，越小越好）：")
table(
    ["策略", "ANN", "LSTM", "RF", "简单平均", "Stacking"],
    [
        ["直接策略", "835.16", "819.62", "765.08", "782.22", "786.24"],
        ["多输出策略", "854.10", "793.71", "757.10", "778.72", "768.22"],
    ],
)
para("交叉验证与朴素基线：为判断模型是否“有用”，另以持久性基线（各 h 均预测起报日径流）作参照：")
table(
    ["基线/模型", "h=1", "h=2", "h=3", "h=4", "h=5", "h=6", "h=7"],
    [
        ["持久性基线 R²", "0.399", "0.060", "−0.053", "−0.245", "−0.399", "−0.478", "−0.582"],
        ["持久性基线 RMSE", "652.83", "819.42", "868.24", "944.19", "1001.26", "1029.08", "1064.63"],
        ["RF 直接策略 R²", "0.504", "0.269", "0.203", "0.130", "0.031", "0.061", "0.012"],
        ["RF 多输出 R²", "0.472", "0.304", "0.217", "0.155", "0.097", "0.063", "0.034"],
    ],
)
para("主要结论：")
para("1. 随机森林在七步平均 RMSE 上最优（直接 765.08、多输出 757.10），且优于简单平均与 Stacking 两条集成线。")
para("2. 预见期越长精度越低：h=1 的 R² 约 0.5，到 h≥3 时普遍跌至 0.2 以下甚至为负，与“未来降水未知”的物理信息墙一致。")
para("3. 集成不必然带来增益：简单平均与 Stacking 在多数预见期未超过最佳基模型；多输出 Stacking 经反向验证后确认 RF 更优，部署时选择 RF。")
para("4. 元层过拟合已被定位并诚实修正：直接策略 Stacking 在 h=4 出现严重过拟合（R²=−0.070）。通过“向 RF 收缩”的正则化元层（shrinkRF：RF + 0.25×(Ridge 预测 − RF)，元层 Ridge alpha=100）修复后，直接 Stacking 七步平均 R² 由 0.125 提升到 0.216，h=4 由 −0.070 修复到 +0.160，负预测数仅 1/2513。")

# ---- 五、程序调试 ----
heading("五、程序调试", 1)
para("以下是解题过程中遇到的关键问题及解决方法：")
para("1. 目标与特征量纲差异大、数据强右偏：目标取对数建模、特征 Z-score 标准化，指标在逆变换后的原始尺度计算。")
para("2. 数据泄漏风险（时间序列）：归一化统计量与特征选择评分只在训练集上估计；滞后/窗口特征只向过去取数；时间序裂采用 TimeSeriesSplit 前向验证，训练与验证之间净化 7 天，训练最晚 h7 目标严格早于验证首起报日。")
para("3. Stacking 元层在少量 OOF 样本上过拟合（直接策略 h=4 R²=−0.070）：定位为元层 Ridge 对基模型预测过度拟合，采用向稳健基模型（RF）收缩的 shrinkRF 方案，并仅在净化前向 OOF 上做因果选择，2004 测试集只评估一次。")
para("4. 多输出 Stacking 无增益：对 RF、Mean、非负凸融合、向 RF 收缩的 Ridge 做反向前向验证，按季度原尺度 RMSE 择优，结果为 RF（635.20）优于最佳合格候选（636.74），故部署选择 RF，不把 RF 回退冒称为“融合提升”。")
para("5. 换更强模型（梯度提升）并不能突破瓶颈：探查实验在相同特征/切分/预处理下对比 HistGradientBoosting 与 RF，GB 的 R² 在直接/多输出各 h 上全部低于 RF（GB−RF 恒为负），说明当前瓶颈是数据信息与样本量，而非模型容量。")
para("6. 只要“更复杂”不能证明有效：以持久性基线作参照，发现远期预测与“直接沿用今日径流”接近甚至更差，避免用复杂模型间的排名冒充真实提升。")

# ---- 六、总结 ----
heading("六、总结", 1)
heading("6.1 本实践所用的技术/模型/算法", 2)
table(
    ["环节", "技术/模型/算法"],
    [
        ["数据分析", "折线图、直方图、箱型图、皮尔逊相关系数矩阵、自相关（ACF）、滞后互相关（CCF）"],
        ["特征工程", "季节/月份编码、三角周期编码、滞后特征（1/2/3/7/14/30 天）、滑动窗口统计、派生气象特征；Min-Max 与 Z-score；皮尔逊与互信息特征选择"],
        ["建模", "MLP（ANN）、RandomForest、LSTM（PyTorch，lookback 30）；多步直接策略、多输出策略；目标 log 变换"],
        ["集成", "简单平均（Mean）、Stacking（Ridge 元学习器）、向 RF 收缩的 shrinkRF"],
        ["评估与调参", "GridSearchCV + TimeSeriesSplit、R²/RMSE/MAE、三 seed 随机性复核、朴素基线（持久性/均值/季节均值/7 日 Ridge）"],
    ],
)
heading("6.2 当前模型存在的问题与后续改进方案", 2)
para("1. 远期（h≥3）预测精度低，且接近甚至差于持久性基线：根本原因是在 t 时刻无法获知未来 3—7 天的降水，而降水是径流的主要驱动。改进方向：引入数值天气预报（NWP）降水作为额外输入（数据许可时）；对洪水峰值单独建模；采用分位数回归输出区间而非单点值。")
para("2. 训练样本过少：仅 4 年约 1395 个训练样本，2002—2003 又用于 OOF。树模型与神经网络都受限于样本量。改进方向：扩充多年份数据、增广、多流域迁移学习。")
para("3. 元层易过拟合：Stacking 元层在少量 OOF 上学习不稳定。已用 shrinkRF 向稳健基模型收缩缓解，可进一步尝试贝叶斯岭、按 h 加权收缩、集成选择等更稳的元层正则化。")
heading("6.3 认识与心得体会", 2)
para("1. “模型越复杂、集成越多”不等于“效果越好”：在 359 个测试点的小数据集上，RF 稳定优于 LSTM 与 Stacking，梯度提升反而更差；集成能否带来收益必须用朴素基线和诚实的前向验证说话。")
para("2. 数据信息上限决定了精度的天花板：径流具有很强的短期记忆（lag1=0.77），但未来降水不可知，导致远期预测难以超过“沿用今日径流”。模型能做的上限受物理信息约束。")
para("3. 评估与泄漏纪律远重于调参技巧：时间序列一旦打乱顺序或碰撞未来信息，任何“高精度”都不可信；全程前向切分、净化、只评估一次测试集，是结论可信的根基。")
para("4. 机器学习的实际价值在于“可靠地说明能力边界”：即便 R² 在远期不高，能定量指出“第几天之前可信、哪些事件会失败”，对防洪调度等应用仍有真实意义。")

# ---- 附注 ----
heading("附注", 1)
para("1. 报告中已写明各组员分工（见“二、详细分工”，请据实填写姓名）。")
para("2. 本报告电子版及全部代码按“组号+组长姓名+组员1姓名+组员2姓名”方式命名并打包提交至网络教学平台。")
para("3. 未提供的信息（姓名、学号、指导教师）不作编造，留空待填。")

# ---- 附录 ----
heading("附录", 1)
para("完整代码清单（运行入口与核心模块）：")
code_block(
    "code/\n"
    "├── data_analysis.py            # 数据分析：字段解析、可视化、相关性\n"
    "├── feature_engineering.py      # 特征工程：编码、滞后、窗口、派生、归一化、选择\n"
    "├── acceptance3_model.py        # 统一模型封装：LSTM/ANN/RF（折内预处理、净化早停）\n"
    "├── acceptance3_train.py        # 第三次验收：直接策略训练与评估\n"
    "├── acceptance3_v2_train.py     # 直接策略元层构建 / OOF\n"
    "├── acceptance3_v3_stage_c.py   # 多输出策略元层构建（季度 OOF）\n"
    "├── acceptance3_v3_fusion.py    # 融合策略（RF/Mean/凸融合/ridge_RF）\n"
    "├── acceptance3_meta_boost.py   # 元层候选的前向因果选择（shrinkRF 等）\n"
    "├── acceptance3_meta_deploy.py  # shrinkRF 元层修复部署与逐日预测\n"
    "└── acceptance3_gb_probe.py     # 梯度提升基模型收益探查（GB vs RF）"
)
para("详细结果与来源见三次阶段验收报告（验收/Acceptance 1/2/3）及随附 CSV/PNG。")

# 页脚页码
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = footer.add_run("机器学习课程实践 · 总验收报告 · ")
set_east_asia(fr, FONT)
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)

doc.save(TARGET)
print("saved:", TARGET)