# -*- coding: utf-8 -*-
"""
生成《机器学习课程实践》实验报告（.docx）
=========================================
按照实验报告模板的六个章节 + 附录组织，自动读入前面各阶段产出的
指标表与图片，保证报告中的数字与代码运行结果完全一致。
"""

import os
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

import config as C

CONTENT_W = Cm(16.0)      # A4 页面去掉 2.5cm 边距后的可用宽度
FIG = C.FIG_DIR
RES = C.RESULT_DIR
REPORT_PATH = os.path.join(C.BASE_DIR, "实验报告-机器学习课程实践-径流预测.docx")


# ===========================================================================
# python-docx 工具函数
# ===========================================================================
def _set_cn_font(run, name="宋体", size=None, bold=None, color=None):
    run.font.name = name
    # 安全获取 rPr/rFonts，避免首次设置时子元素不存在导致 AttributeError
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def h(doc, text, level=1):
    p = doc.add_heading("", level=level)
    r = p.add_run(text)
    _set_cn_font(r, "黑体" if level <= 2 else "楷体",
                 {1: 15, 2: 13.5, 3: 12, 4: 11.5}[level], True)
    return p


def para(doc, text, size=10.5, indent=True, bold=False, align=None, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.4
    if indent:
        p.paragraph_format.first_line_indent = Pt(21)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    _set_cn_font(r, "宋体", size, bold)
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.35
    r = p.add_run(text)
    _set_cn_font(r, "宋体", size)
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    _set_cn_font(r, "楷体", 9.5)
    return p


def figure(doc, path, cap, width=None):
    if not os.path.exists(path):
        para(doc, f"[缺失图片：{os.path.basename(path)}]", indent=False)
        return
    doc.add_picture(path, width=width or CONTENT_W)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(doc, cap)


def table(doc, df, cap=None, size=9, index=False, widths=None):
    cols = ([df.index.name or ""] if index else []) + list(df.columns)
    t = doc.add_table(rows=1, cols=len(cols))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, cn in enumerate(cols):
        hdr[i].text = ""
        r = hdr[i].paragraphs[0].add_run(str(cn))
        _set_cn_font(r, "黑体", size, True)
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    body = df.reset_index() if index else df
    for _, row in body.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row.tolist()):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(_fmt(v))
            _set_cn_font(r, "宋体", size)
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if widths:
        for r_ in t.rows:
            for c_, w_ in zip(r_.cells, widths):
                c_.width = w_
    if cap:
        caption(doc, cap)
    return t


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}" if abs(v) < 1e4 else f"{v:.0f}"
    return str(v)


def code_block(doc, text, size=7.5):
    for line in text.split("\n"):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.left_indent = Cm(0.4)
        r = p.add_run(line if line.strip() else " ")
        _set_cn_font(r, "Consolas", size)
    return doc


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def read_csv(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def src(fname):
    """读取源码用于附录。"""
    p = os.path.join(C.BASE_DIR, "code", fname)
    with open(p, encoding="utf-8") as f:
        return f.read()


# ===========================================================================
# 封面
# ===========================================================================
def cover(doc):
    for _ in range(3):
        para(doc, "", indent=False)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("杭州电子科技大学"); _set_cn_font(r, "黑体", 22, True)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("《机器学习课程实践》"); _set_cn_font(r, "黑体", 20, True)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("实 验 报 告"); _set_cn_font(r, "黑体", 26, True)
    for _ in range(3):
        para(doc, "", indent=False)

    rows = [("题    目", "基于机器学习的径流预测（USGS 01047000 流域）"),
            ("专    业", C.REPORT_META["专业"]),
            ("学    号", C.REPORT_META["学号"]),
            ("姓    名", C.REPORT_META["姓名"]),
            ("指导教师", C.REPORT_META["指导教师"]),
            ("成    绩", "")]
    t = doc.add_table(rows=len(rows), cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (k, v) in enumerate(rows):
        for j, txt in enumerate((k, v)):
            c = t.cell(i, j)
            c.text = ""
            pr = c.paragraphs[0]
            pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = pr.add_run(txt)
            _set_cn_font(r, "宋体", 13)
        t.rows[i].cells[0].width = Cm(3.6)
        t.rows[i].cells[1].width = Cm(9.6)
    # 去掉封面表格边框
    for row in t.rows:
        for c in row.cells:
            c._element.get_or_add_tcPr().append(_no_border())

    for _ in range(3):
        para(doc, "", indent=False)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("二〇二六 年    月    日"); _set_cn_font(r, "宋体", 13)
    page_break(doc)


def _no_border():
    from docx.oxml import OxmlElement
    b = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "nil")
        b.append(e)
    return b


# ===========================================================================
# 一、任务要求
# ===========================================================================
def sec1(doc):
    h(doc, "一、任务要求", 1)

    h(doc, "1.1 题目解析", 2)
    para(doc, "本实践题目为「基于机器学习的径流预测」。径流是指降水扣除蒸发、下渗等"
              "消耗后沿地表与地下汇入河道的水流，通常以流域出口断面的流量表示。"
              "径流预测即依据流域已观测到的水文与气象要素，推求未来若干天的出口断面流量，"
              "在防洪调度、抗旱供水、水力发电与灌溉管理中具有重要作用。")
    para(doc, "从机器学习角度看，本题属于：")
    bullet(doc, "监督学习（Supervised Learning）——训练样本同时给定了输入特征与目标真值；")
    bullet(doc, "回归任务（Regression）——目标变量「径流」是连续型数值，而非离散类别；")
    bullet(doc, "时间序列预测（Time-Series Forecasting）——样本之间存在严格的时间先后"
                "依赖关系，不能随机打乱；")
    bullet(doc, "多步预测（Multi-Step Forecasting）——需要一次性给出未来 1~7 天的结果，"
                "而不是只预测下一个时刻。")
    para(doc, "本题的难点在于：① 输入仅为历史径流与历史气象要素，未来 1~7 天的降水是"
              "未知的，因此随着预见期延长，可用的信息量迅速衰减；② 径流序列呈强右偏分布，"
              "少数特大洪水的量级远高于常遇洪水，评价指标容易被极端事件主导；"
              "③ 必须严格按时间顺序划分数据，否则会造成「用未来的数据预测过去」的数据泄漏。")

    h(doc, "1.2 数据说明", 2)
    para(doc, "数据文件为 01047000.csv，包含 2000-01-01 至 2004-12-31 共 1827 天的逐日记录，"
              "共 9 个字段，无缺失值。其中：")
    bullet(doc, "目标变量：Discharge（径流，ft³/s），来源于 USGS 01047000 号水文站的实测流量；")
    bullet(doc, "气象驱动因子：Dayl（日照时长）、Prcp（降水）、Srad（短波辐射）、"
                "Swe（雪水当量）、Tmax（最高气温）、Tmin（最低气温）、Vp（水汽压），"
                "共 7 个，来源于 Daymet V4 R1 北美 1 km 逐日地表气象数据集。")
    para(doc, "上述字段的中文含义依据《Daymet_Daily_V4R1 用户指南》Table 1 确定，"
              "完整对照见下表。")

    h(doc, "1.3 输入与输出定义", 2)
    para(doc, "按照任务要求，多输出预测方式下模型的输入输出为：")
    para(doc, "　　输入：第 t-s, …, t-1, t 天的径流；第 t-m, …, t-1, t 天的降水；"
              "第 t-p, …, t-1, t 天的气温。", indent=False)
    para(doc, "　　输出：第 t+1, t+2, …, t+7 天的径流。", indent=False)
    para(doc, "多步直接预测方式下，则为每个预见期 h（h = 1, 2, …, 7）单独训练一个模型，"
              "第 h 个模型的输出为第 t+h 天的径流。")

    h(doc, "1.4 建模要求汇总", 2)
    df = pd.DataFrame([
        ["个体模型", "人工神经网络(ANN)、随机森林(RF)", "2 种机器学习方法"],
        ["", "LSTM 长短期记忆网络", "1 种深度学习方法"],
        ["预测策略", "多步直接预测", "每种方法 7 个模型"],
        ["", "多输出预测", "每种方法 1 个模型"],
        ["超参数", "Grid Search CV（网格搜索 + 交叉验证）", "逐个模型寻优"],
        ["集成学习", "简单平均、Stacking", "2 种，将 3 个个体模型结果结合"],
        ["数据划分", "前 4 年（2000—2003）训练，第 5 年（2004）测试", "严格按时间顺序，不可打乱"],
        ["评价指标", "RMSE、MAE、NSE、R²、PBIAS、MAPE", "以水文通用的 NSE 为主"],
    ], columns=["类别", "内容", "说明"])
    table(doc, df, "表 1-1  建模任务要求汇总", size=9.5)
    para(doc, "模型总数核算：个体模型 3 种 × 8（多步直接 7 个 + 多输出 1 个）= 24 个；"
              "集成模型 2 种 × 8 = 16 个；合计 40 个模型。", space_after=4)
    page_break(doc)


# ===========================================================================
# 二、详细分工
# ===========================================================================
def sec2(doc):
    h(doc, "二、详细分工", 1)
    para(doc, "本实践由 3 人组队完成。为保证 40 个模型的构建、调试与结果整理能够高效推进，"
              "按「数据与特征 → 个体模型 → 集成与评价」三段式分工，各成员职责如下：")
    df = pd.DataFrame([
        [C.REPORT_META["组员1"], "组长", "数据与特征工程",
         "数据读取与字段释义；时间序列/直方图/箱型图/相关系数热力图可视化分析；"
         "季节与月份编码、滞后特征、窗口特征提取；Min-Max 与 Z-score 归一化实现；"
         "皮尔逊相关系数法与互信息法特征选择；负责报告第一、二章撰写与全文统稿"],
        [C.REPORT_META["组员2"], "组员", "个体预测模型",
         "ANN（多层感知机）多步直接与多输出模型构建；随机森林多步直接与多输出模型构建；"
         "LSTM 网络结构设计与 PyTorch 模型封装；Grid Search CV 超参数寻优；"
         "多步直接/多输出两种预测策略的数据集构造；负责报告第三章「过程设计」撰写"],
        [C.REPORT_META["组员3"], "组员", "集成学习与评价",
         "简单平均集成建模；Stacking 集成（含折外预测 OOF 生成与 Ridge 元学习器）；"
         "RMSE/MAE/NSE/R²/PBIAS/MAPE 指标实现；预测结果可视化与误差分析；"
         "负责报告第四章结果整理、第五章调试记录与第六章总结撰写"],
    ], columns=["姓名", "角色", "负责模块", "具体工作内容"])
    table(doc, df, "表 2-1  小组成员分工", size=9,
          widths=[Cm(2.0), Cm(1.5), Cm(2.6), Cm(9.9)])
    para(doc, "说明：全体成员共同参与选题讨论、模型方案论证与结果复核；"
              "代码由组长统一整合为可一键复现的工程，并完成整体联调与验证。",
         space_after=4)
    page_break(doc)


# ===========================================================================
# 三、过程设计
# ===========================================================================
def sec3(doc):
    h(doc, "三、过程设计", 1)
    para(doc, "整个实践按「数据探索 → 特征构造 → 单模型建模 → 超参数寻优 → 集成融合 → "
              "精度评价」六个环节展开，整体流程如下图。")
    figure(doc, os.path.join(FIG, "fig_流程.png"),
           "图 3-1  径流多步预测总体流程", Cm(15))

    h(doc, "3.1 步骤一：数据分析", 2)
    df = pd.DataFrame([
        ["1.1", "字段释义", "查 Daymet V4 R1 用户指南与 USGS 站点说明",
         "明确 8 个变量的中文含义、单位与物理意义"],
        ["1.2", "时间序列可视化", "Matplotlib 折线图/柱状图",
         "观察径流的年内与年际变化、丰枯交替、洪峰出现时间"],
        ["1.3", "分布分析", "直方图 + 偏度/峰度",
         "判断各变量分布形态，识别右偏与长尾特征"],
        ["1.4", "离散程度与异常值", "箱型图",
         "比较各变量离散程度，识别极端洪水与异常观测"],
        ["1.5", "相关性分析", "皮尔逊相关系数热力图",
         "量化各气象因子与径流的相关性强弱，筛选候选特征"],
        ["1.6", "滞后相关分析", "自相关(ACF)/互相关(CCF)",
         "确定径流的记忆长度与降水-径流的响应时间，为滞后阶数提供依据"],
    ], columns=["序号", "分析内容", "工具/方法", "处理目标"])
    table(doc, df, "表 3-1  数据分析步骤设计", size=9)

    h(doc, "3.2 步骤二：特征工程", 2)
    df = pd.DataFrame([
        ["2.1", "季节编码", "四季 one-hot（春/夏/秋/冬）", "刻画径流的季节性差异"],
        ["2.2", "月份编码", "年内日序与月份的 sin/cos 三角编码",
         "周期变量连续化，避免 12 月与 1 月被人为拉远"],
        ["2.3", "滞后特征", "Lag(1,2,3,7,14,30)，对径流及主要气象因子",
         "引入历史信息，体现流域调蓄与汇流滞后"],
        ["2.4", "窗口特征", "滑动均值/标准差/最大值（窗口 3,7,14,30 天）",
         "刻画前期平均状态与波动剧烈程度"],
        ["2.5", "派生特征", "Tmean、Trange、累积降水、融雪指数",
         "补充具有水文物理意义的中间变量"],
        ["2.6", "特征归一化", "Min-Max 归一化 与 Z-score 标准化（均实现并对比）",
         "消除量纲差异，加速神经网络收敛"],
        ["2.7", "特征选择", "皮尔逊相关系数法 与 互信息法（均实现并对比）",
         "剔除冗余与无关特征，降低过拟合风险"],
    ], columns=["序号", "特征类别", "方法", "处理目标"])
    table(doc, df, "表 3-2  特征工程步骤设计", size=9)

    h(doc, "3.3 步骤三：多步预测策略设计", 2)
    para(doc, "多步预测要求同时给出未来 1~7 天的径流，本题采用两种策略分别建模：")
    para(doc, "（1）多步直接预测（Direct Multi-Step）。对每个预见期 h 单独训练一个模型，"
              "第 h 个模型直接建立「当前特征 → 第 t+h 天径流」的映射：y(t+h) = f_h(X(t))。"
              "该策略下 1 种方法需要 7 个模型，共 7 个；各预见期模型相互独立，"
              "不会把前一步的预测误差传递下去。", space_after=4)
    para(doc, "（2）多输出预测（Multi-Output）。只训练一个模型，令其输出层同时给出 7 个值："
              "[y(t+1),…,y(t+7)] = F(X(t))。该策略下 1 种方法只需要 1 个模型；"
              "7 个输出共享同一套隐层表示，能够利用各预见期之间的相关性，"
              "但模型容量被 7 个任务分摊。", space_after=4)
    para(doc, "本实践中，随机森林与多层感知机原生支持多输出回归；LSTM 则通过把输出层"
              "设为 7 个神经元实现多输出。")

    h(doc, "3.4 步骤四：超参数寻优", 2)
    para(doc, "三种个体模型均使用 Grid Search CV 确定超参数。关键设计是："
              "由于数据是时间序列，交叉验证必须使用 TimeSeriesSplit（前向链式切分），"
              "而不能使用普通 K 折交叉验证。普通 K 折会让验证折中的数据在时间上早于"
              "训练折，造成用未来数据预测过去的泄漏。TimeSeriesSplit 保证每一折的训练集"
              "在时间上始终早于验证集。")
    df = pd.DataFrame([
        ["ANN", "hidden_layer_sizes", "(64,), (128,), (128,64)", "隐层结构与宽度"],
        ["ANN", "alpha", "1e-4, 1e-2", "L2 正则化强度"],
        ["ANN", "learning_rate_init", "1e-3, 1e-2", "初始学习率"],
        ["RF", "n_estimators", "200, 400", "决策树数量"],
        ["RF", "max_depth", "None, 12, 20", "树的最大深度"],
        ["RF", "min_samples_leaf", "1, 3", "叶节点最小样本数"],
        ["LSTM", "units", "(32,), (64,), (32,32)", "LSTM 层结构与单元数"],
        ["LSTM", "dropout", "0.1, 0.3", "Dropout 丢弃率"],
        ["LSTM", "lr", "5e-3", "Adam 学习率"],
    ], columns=["模型", "超参数", "搜索范围", "含义"])
    table(doc, df, "表 3-3  各模型超参数搜索网格", size=9)
    para(doc, "搜索准则为负均方根误差（neg_root_mean_squared_error），"
              "交叉验证折数取 3。")

    h(doc, "3.5 步骤五：集成学习", 2)
    para(doc, "将 ANN、RF、LSTM 三个个体模型的预测结果按两种方式融合：")
    para(doc, "（1）简单平均（Simple Average）：对同一预见期上三个个体模型的预测值"
              "取算术平均，ŷ = (ŷ_ANN + ŷ_RF + ŷ_LSTM) / 3。方法简单、稳健，"
              "通过平均降低单个模型的方差。", space_after=4)
    para(doc, "（2）Stacking：以三个个体模型的预测作为元特征，用 Ridge 回归作为元学习器"
              "学习最优组合权重。这里有一个容易出错的关键点——元学习器的训练样本"
              "必须来自个体模型的折外预测（Out-of-Fold, OOF），而不能直接用个体模型"
              "在训练集上的拟合值，否则个体模型在训练集上的过拟合会被元学习器误认为"
              "是真实精度，导致融合权重失真。本实践采用与超参数搜索相同的时间序列"
              "交叉验证生成 OOF 预测。", space_after=4)

    h(doc, "3.6 步骤六：精度评价", 2)
    para(doc, "采用水文预报领域通用的 6 项指标，评价均在原始径流尺度上进行：")
    df = pd.DataFrame([
        ["RMSE", "均方根误差", "对大误差敏感，量纲与径流一致", "越小越好"],
        ["MAE", "平均绝对误差", "稳健的绝对误差度量", "越小越好"],
        ["NSE", "Nash-Sutcliffe 效率系数", "水文预报核心指标，衡量相对实测均值的解释能力", "越接近 1 越好，>0.75 为很好"],
        ["R²", "决定系数", "预测值与实测值的线性相关程度", "越接近 1 越好"],
        ["PBIAS", "百分比偏差", "反映系统性高估或低估", "越接近 0 越好"],
        ["MAPE", "平均绝对百分比误差", "相对误差的平均水平", "越小越好"],
    ], columns=["指标", "名称", "含义", "评价标准"])
    table(doc, df, "表 3-4  模型精度评价指标", size=9)
    para(doc, "为便于比较，所有模型在完全相同的测试起始日集合上进行评价"
              "（保证 t+7 仍落在测试期内），使各预见期的样本数一致。")

    h(doc, "3.7 工具与算法汇总", 2)
    df = pd.DataFrame([
        ["开发环境", "Windows 11 + Python 3.12", "操作系统与解释器"],
        ["数据处理", "pandas、numpy", "数据读取、特征构造与变换"],
        ["机器学习", "scikit-learn", "MLPRegressor、RandomForestRegressor、Ridge、"
                                    "GridSearchCV、TimeSeriesSplit"],
        ["深度学习", "PyTorch", "LSTM 网络构建与训练（GPU 加速）"],
        ["可视化", "matplotlib", "全部统计图表绘制"],
        ["评价指标", "自实现 + scikit-learn", "NSE、PBIAS 等水文专用指标"],
    ], columns=["环节", "工具/库", "用途"])
    table(doc, df, "表 3-5  工具与算法汇总", size=9)
    page_break(doc)


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    from report_sections import (make_flowchart, sec4, sec4_conclusions,
                                 sec5, sec6, sec7, appendix)

    print("生成实验报告 ...")
    make_flowchart()

    doc = Document()
    # 页面设置：A4，2.5cm 页边距
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.top_margin = s.bottom_margin = Cm(2.5)
    s.left_margin = s.right_margin = Cm(2.5)
    st = doc.styles["Normal"]
    st.font.name = "宋体"
    st.font.size = Pt(10.5)
    st.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")

    cover(doc)
    sec1(doc)
    sec2(doc)
    sec3(doc)
    sec4(doc)
    sec4_conclusions(doc)
    sec5(doc)
    sec6(doc)
    sec7(doc)
    appendix(doc)

    doc.save(REPORT_PATH)
    print(f"报告已生成：{REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    main()
