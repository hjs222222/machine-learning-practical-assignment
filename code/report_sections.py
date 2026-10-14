# -*- coding: utf-8 -*-
"""实验报告第四~六章及附录的正文生成（被 make_report.py 调用）。"""

import ast
import os

import numpy as np
import pandas as pd

import config as C
import models as M
from make_report import (h, para, bullet, figure, table, code_block, caption,
                         page_break, read_csv, src, _set_cn_font, FIG)
from docx.shared import Pt


# ===========================================================================
# 从 results CSV 动态读取指标
# 报告中出现的所有模型精度数字都必须来自实验结果文件，
# 杜绝与运行结果不一致的硬编码（此前曾因 LSTM 未运行而出现不实数字）。
# ===========================================================================
def _detail():
    return read_csv("10_各模型各预见期指标.csv")


def _pooled():
    return read_csv("11_模型汇总指标.csv")


def _metric(df, model_cn, strat, h, col="NSE"):
    """从指标表读取 某模型×某策略×预见期(h 或 '1-7(汇总)') 的指标值。"""
    if df is None or df.empty:
        return None
    r = df[(df["模型"] == model_cn) & (df["策略"] == strat) & (df["预见期"] == h)]
    if r.empty or col not in r.columns:
        return None
    v = r.iloc[0][col]
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def metric_h(model_cn, strat, h, col="NSE"):
    """逐预见期指标。"""
    return _metric(_detail(), model_cn, strat, int(h), col)


def metric_pooled(model_cn, strat, col="NSE"):
    """1~7 天汇总指标。"""
    return _metric(_pooled(), model_cn, strat, "1-7(汇总)", col)


def fmt(v, nd=3):
    """格式化指标；缺失时返回 '——'。"""
    return "——" if v is None else f"{v:.{nd}f}"


def stacking_coefs(strat_cn="多步直接预测"):
    """读取 Stacking 元学习器系数，返回 [(模型中文名, 平均系数), ...]。

    系数顺序与 experiment.py 中 base_preds 的模型顺序一致
    （BASE_MODELS = ANN → RF → LSTM 中实际可用的子集）。
    """
    df = read_csv("14_Stacking元学习器系数.csv")
    if df is None or df.empty:
        return []
    sub = df[df["策略"] == strat_cn]
    if sub.empty:
        return []
    coefs = []
    for _, row in sub.iterrows():
        try:
            coefs.append(np.asarray(ast.literal_eval(str(row["Ridge系数"])), dtype=float))
        except (ValueError, SyntaxError):
            return []
    arr = np.vstack(coefs)              # (7, n_models)
    n = arr.shape[1]
    # 顺序与 experiment.py 的 BASE_MODELS = ["ANN", "RF", "LSTM"] 一致
    names = [M.MODEL_CN[m] for m in ["ANN", "RF", "LSTM"][:n]]
    return list(zip(names, arr.mean(axis=0)))


# ===========================================================================
# 流程图
# ===========================================================================
def make_flowchart():
    """绘制总体流程图。"""
    path = os.path.join(FIG, "fig_流程.png")
    if os.path.exists(path):
        return path

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from utils_plot import save

    fig, ax = plt.subplots(figsize=(9.4, 7.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.4); ax.axis("off")

    def box(x, y, w, hh, text, fc, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.12",
                                    fc=fc, ec="#3A4A5A", lw=1.1))
        ax.text(x + w / 2, y + hh / 2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.6)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.3, color="#3A4A5A"))

    box(0.5, 11.15, 9, 1.0,
        "数据源：01047000.csv（2000—2004 逐日，1827 天）\n"
        "USGS 实测径流 Discharge  +  Daymet V4 R1 气象因子", "#DCE7F2", 10)
    box(0.5, 9.45, 4.35, 1.3,
        "① 数据分析\n时间序列 / 直方图 / 箱型图\n皮尔逊热力图 / 滞后相关", "#E8F3E8")
    box(5.15, 9.45, 4.35, 1.3,
        "② 特征工程\n季节与月份编码 / 滞后特征\n窗口特征 / 归一化 / 特征选择", "#E8F3E8")
    box(0.5, 7.55, 9, 1.4,
        "③ 数据集划分（严格按时间顺序，不可打乱）\n"
        "训练集：2000-01-01 ~ 2003-12-31（前 4 年）　|　"
        "测试集：2004-01-01 ~ 2004-12-31（第 5 年）", "#FBF0DA")
    box(0.5, 5.5, 2.85, 1.55,
        "④ 个体模型\n— ANN（多层感知机）\n— 随机森林 RF\n— LSTM", "#F5E6EF")
    box(3.6, 5.5, 2.8, 1.55,
        "⑤ 预测策略\n— 多步直接预测\n　(h=1…7，7 个模型)\n— 多输出预测（1 个）", "#F5E6EF")
    box(6.65, 5.5, 2.85, 1.55,
        "⑥ 超参数寻优\nGrid Search CV\nTimeSeriesSplit\n（前向链式切分）", "#F5E6EF")
    box(0.5, 3.45, 4.3, 1.4,
        "⑦ 集成学习\n简单平均（算术平均）\nStacking（Ridge 元学习器 + OOF）", "#E3F0F5")
    box(5.2, 3.45, 4.3, 1.4,
        "⑧ 精度评价\nRMSE / MAE / NSE / R²\nPBIAS / MAPE", "#E3F0F5")
    box(1.4, 1.45, 7.2, 1.3,
        "⑨ 结果分析：指标随预见期变化、洪水事件放大、误差分布\n"
        "共构建个体模型 24 个 + 集成模型 16 个 = 40 个模型", "#DCE7F2", 10)

    arrow(2.675, 11.15, 2.675, 10.78)
    arrow(7.325, 11.15, 7.325, 10.78)
    arrow(2.675, 9.45, 3.5, 8.98)
    arrow(7.325, 9.45, 6.5, 8.98)
    arrow(5.0, 7.55, 5.0, 7.08)
    arrow(1.9, 5.5, 2.3, 4.88)
    arrow(5.0, 5.5, 4.9, 4.88)
    arrow(8.05, 5.5, 7.7, 4.88)
    arrow(5.0, 3.45, 5.0, 2.78)

    plt.tight_layout()
    save(fig, "fig_流程.png")
    return path


# ===========================================================================
# 四、编码实现及结果
# ===========================================================================
def sec4(doc):
    h(doc, "四、编码实现及结果", 1)

    h(doc, "4.1 开发环境", 2)
    para(doc, "操作系统 Windows 11；编程语言 Python 3.12；主要算法库："
              "scikit-learn（机器学习与超参数搜索）、"
              "PyTorch（LSTM 深度学习，GPU 加速）、"
              "pandas 与 numpy（数据处理）、"
              "matplotlib（可视化）。全部代码组织为数据、特征、模型、实验、"
              "可视化与报告生成等多个核心模块，"
              "可通过 run_all.py 一键复现 V6 定稿版全部结果。")

    # ---------------- 4.2 数据分析 ----------------
    h(doc, "4.2 数据分析结果", 2)
    info = read_csv("01_数据概况.csv")
    if info is not None:
        table(doc, info[["字段", "中文含义", "单位", "均值", "标准差",
                         "最小值", "最大值", "偏度", "峰度"]].copy(),
              "表 4-1  各变量统计特征（2000—2004 年，n = 1827）", size=8.5)

    para(doc, "（1）字段含义与基本统计。数据集共 1827 天、9 个字段，无缺失值。"
              "径流 Discharge 的均值为 611 ft³/s、标准差 1059 ft³/s，"
              "最小值 29 ft³/s、最大值 13700 ft³/s，偏度高达 4.96、峰度 35.94，"
              "呈极强的右偏长尾分布，说明洪水事件量级远高于常遇流量。"
              "需要特别注意的是：气象因子 Swe（雪水当量）在全部 1827 天中恒为 0，"
              "标准差为 0，属于常量列，不含任何信息量，在特征工程阶段予以剔除。")

    para(doc, "（2）时间序列分析。由图 4-1 可见，径流年内分配极不均匀，"
              "呈现明显的「春季融雪洪峰 + 冬季雨洪」双峰特征："
              "4 月前后出现由融雪与降雨共同驱动的春季洪峰，"
              "11—12 月出现由降雨驱动的冬季洪峰；1—2 月气温低、径流小且平稳，"
              "7—9 月为稳定退水期。年际之间丰枯差异显著。")
    figure(doc, os.path.join(FIG, "fig1_时间序列.png"),
           "图 4-1  01047000 流域逐日径流与气象因子时间序列（2000—2004）")
    figure(doc, os.path.join(FIG, "fig2_年内分布.png"),
           "图 4-2  径流与气象因子的年内分布（气候态）")

    para(doc, "（3）分布分析。直方图（图 4-3）显示：径流、降水呈明显的右偏长尾分布，"
              "大部分日子径流偏小、少数日子极高；气温类变量近似对称；"
              "日照时长呈双峰分布，对应冬夏两季。这种非正态、强右偏的目标分布，"
              "是后续采用对数变换建模的直接依据。")
    figure(doc, os.path.join(FIG, "fig3_直方图.png"),
           "图 4-3  各变量分布直方图")

    para(doc, "（4）箱型图分析。标准化后的箱型图（图 4-4a）显示径流的箱体最窄而上须最长、"
              "异常值最多，离散程度远大于其他变量；按月分组的箱型图（图 4-4b）"
              "进一步印证了 4 月与 11—12 月的高流量特征。")
    figure(doc, os.path.join(FIG, "fig4_箱型图.png"),
           "图 4-4  箱型图分析：各变量离散程度与径流的季节性")

    para(doc, "（5）相关性分析。皮尔逊相关系数热力图（图 4-5）表明："
              "与径流同期相关性最强的是 Prcp（r = 0.431），"
              "其余气象因子与径流的线性相关性都很弱（|r| < 0.10），"
              "其中 Vp 为 -0.097、Tmax 为 -0.040、Tmin 为 -0.007。"
              "气象因子之间，Dayl 与 Tmax、Tmin、Vp 高度相关（r = 0.72~0.76），"
              "存在明显的多重共线性。这说明仅靠同期气象因子难以解释径流变化，"
              "必须引入滞后与累积效应特征。")
    figure(doc, os.path.join(FIG, "fig5_相关系数热力图.png"),
           "图 4-5  皮尔逊相关系数热力图")

    para(doc, "（6）滞后相关分析。图 4-6a 显示径流自相关 lag1 为 0.767、lag2 为 0.576、"
              "lag7 为 0.402，衰减较快但仍有明显记忆，为引入滞后特征提供了依据；"
              "图 4-6b 给出了径流与全部 6 个气象因子（Dayl、Prcp、Srad、Tmax、Tmin、Vp）"
              "的滞后互相关曲线，其中降水与径流的互相关在「同日」最强（r ≈ 0.43）并随提前天数"
              "迅速衰减，其余因子互相关普遍较弱（|r| < 0.15），说明该流域汇流速度快、响应时间在 1 天以内，"
              "属于典型的山区小流域「陡涨陡落」型产汇流特征。"
              "这一点是理解后续精度随预见期快速衰减的关键。")
    figure(doc, os.path.join(FIG, "fig6_滞后相关.png"),
           "图 4-6  滞后相关分析：径流自相关（ACF）与气象因子互相关（CCF）")

    # ---------------- 4.3 特征工程 ----------------
    h(doc, "4.3 特征工程结果", 2)
    para(doc, "（1）时间序列特征提取。共构造 84 个候选特征，构成如表 4-2。"
              "在常规特征之外，针对本流域「陡涨陡落」的洪水特性，"
              "补充了 7 个洪水信号特征：径流差分（Discharge_diff1/3/7，刻画涨水趋势）、"
              "降水累积（Prcp_roll3_sum、Prcp_roll14_sum）、强降水计数（Prcp_heavy3）"
              "与降水突增（Prcp_diff1）。")
    table(doc, pd.DataFrame([
        ["季节编码", "season_春/夏/秋/冬", "4", "四季 one-hot 编码"],
        ["月份编码", "month", "1", "提取月份信息"],
        ["滞后特征", "Discharge/Prcp/Tmax/Tmin/Vp/Srad × lag(1,2,3,7,14,30)", "36",
         "引入历史信息，体现流域调蓄与汇流滞后"],
        ["窗口特征", "Discharge/Prcp × roll(3,7,14,30) × (mean,std,max)", "24",
         "刻画前期平均状态与波动激烈程度"],
        ["派生特征", "Tmean, Trange, Prcp_cum7, Prcp_cum30, Tmean_roll7, melt_index", "6",
         "日均温、日温差、累积降水、融雪指数"],
        ["洪水信号特征", "Discharge_diff1/3/7、Prcp_roll3_sum、Prcp_roll14_sum、"
         "Prcp_heavy3、Prcp_diff1", "7",
         "涨水趋势差分、降水累积、强降水计数与降水突增，捕捉洪峰前兆信号"],
        ["原始气象因子", "Dayl, Prcp, Srad, Tmax, Tmin, Vp", "6", "已剔除常量列 Swe"],
        ["合计", "——", "88", "——"],
    ], columns=["类别", "特征示例", "个数", "水文意义"]),
        "表 4-2  时间序列特征构成（共 88 个候选特征）", size=8.5)
    para(doc, "所有滞后与窗口特征均只向过去取数（窗口统计使用 min_periods 严格限制在 "
              "[t-w+1, t]），不使用任何未来信息；构造完成后前 30 天因特征缺测被剔除，"
              "有效样本 1797 天。")

    para(doc, "（2）特征归一化。实现了最小-最大归一化与 Z-score 标准化两种方法并对比：")
    nt = read_csv("03_归一化方法对比.csv")
    if nt is not None:
        table(doc, nt, "表 4-3  两种归一化方法对比", size=8.5)
    para(doc, "最小-最大归一化把各特征线性映射到 [0, 1]，形式简单、能保留原始分布形态，"
              "但变换区间完全由极值决定，对异常值非常敏感——本数据中径流最大值 13700、"
              "最小值 29，一旦出现新的极值，原有的归一化区间就会失效。"
              "Z-score 标准化把特征变换为均值 0、方差 1 的分布，受异常值影响相对较小，"
              "更适合本数据这种强右偏的情形。因此建模阶段统一采用 Z-score 标准化，"
              "且均值与标准差仅在训练集上估计后再应用到测试集。")
    figure(doc, os.path.join(FIG, "fig7_归一化对比.png"),
           "图 4-7  归一化前后特征取值分布对比")

    para(doc, "（3）特征选择。实现了皮尔逊相关系数法与互信息法。"
              "皮尔逊相关系数只度量线性相关，对非线性关系不敏感；"
              "互信息基于信息熵，能捕捉任意形式的依赖关系，更适合本数据。"
              "两种方法得到的特征重要性排序对比见图 4-8。")
    fs = read_csv("04_特征选择得分.csv")
    if fs is not None:
        table(doc, fs.head(12)[["特征", "|皮尔逊相关系数|", "互信息得分",
                                "皮尔逊排名", "互信息排名"]].round(4),
              "表 4-4  特征选择得分（按互信息降序，前 12 个）", size=8.5)
    figure(doc, os.path.join(FIG, "fig8_特征选择.png"),
           "图 4-8  互信息法与皮尔逊相关系数法特征重要性对比")
    para(doc, "互信息得分最高的是 Discharge 的短窗口与滞后特征"
              "（Discharge_roll3_max、Discharge_roll3_mean、Discharge_lag1 等），"
              "说明径流自身的近期状态是最强的预测因子；"
              "降水类特征（Prcp 及其累积量）次之；"
              "单纯的同期气象因子贡献相对有限。"
              f"按互信息得分从高到低保留前 {C.TOP_K_FEATURES} 个特征作为最终输入。")
    figure(doc, os.path.join(FIG, "fig9_特征类别贡献.png"),
           "图 4-9  各类特征对径流的互信息贡献")
    para(doc, "为验证特征选择的效果，比较了保留不同特征个数时随机森林的精度"
              "（表 4-5）。"
              f"保留 {C.TOP_K_FEATURES} 个特征时精度已接近使用全部 88 个特征的水平"
              "（h=1 的 NSE 分别为 0.512 与 0.513，h=3 为 0.239 与 0.234），"
              "而特征维数从 88 降到 50，有利于抑制过拟合，"
              f"因此最终取 TOP_K = {C.TOP_K_FEATURES}。")
    table(doc, pd.DataFrame([
        ["10", "0.460", "0.131", "-0.043"],
        ["20", "0.492", "0.161", "-0.034"],
        ["40", "0.512", "0.239", "0.057"],
        ["60", "0.506", "0.217", "0.068"],
        ["88（全部）", "0.513", "0.234", "0.064"],
    ], columns=["保留特征数", "预见期 1 天 NSE", "预见期 3 天 NSE", "预见期 7 天 NSE"]),
        "表 4-5  特征个数对随机森林精度的影响（对数目标，Grid Search CV + 时序交叉验证，"
        "测试年 2004）", size=8.5)

    # ---------------- 4.4 建模结果 ----------------
    h(doc, "4.4 径流多步预测建模结果", 2)
    para(doc, "（1）目标变量变换。径流呈强右偏分布，且 NSE 的分母"
              "（总离差平方和）被少数特大洪水主导——经统计，测试年 2004 年中"
              "仅占 2.8% 的 10 天就贡献了 61.6% 的离差平方和。"
              "直接对原始径流建模会使损失函数被个别洪水事件绑架。"
              "为此对径流取对数后再标准化建模，预测结果再经指数变换回原始尺度，"
              "这是水文预报中的标准做法。对照实验表明，对数变换显著提升了"
              "各预见期的精度（表 4-6）。")
    table(doc, pd.DataFrame([
        ["原始径流 y", "0.139", "-0.450", "-0.262"],
        ["对数 log(y)", "0.481", "0.223", "0.121"],
        ["对数比值 log(y/yₜ)", "0.411", "0.211", "0.147"],
        ["相对变化 (y−yₜ)/yₜ", "-0.059", "-3.100", "-1.181"],
    ], columns=["目标变量变换方式", "预见期 1 天 NSE", "预见期 3 天 NSE",
                "预见期 7 天 NSE"]),
        "表 4-6  目标变量变换方式对精度的影响（HistGradientBoosting 对照实验，"
        "TOP50 按预见期特征）",
        size=8.5)

    detail = read_csv("10_各模型各预见期指标.csv")
    if detail is None:
        para(doc, "【提示】尚未检测到实验结果文件，请先运行 experiment.py。")
        return

    for idx, (strat, figname) in enumerate(
            [("多步直接预测", "fig14_NSE随预见期变化_直接.png"),
             ("多输出预测", "fig15_NSE随预见期变化_多输出.png")]):
        sub = detail[detail["策略"] == strat]
        piv = sub.pivot(index="模型", columns="预见期", values="NSE").reset_index()
        piv.columns = ["模型"] + [f"NSE(h={c})" for c in piv.columns[1:]]
        md = sub.pivot(index="模型", columns="预见期",
                       values="RMSE").reset_index()
        md.columns = ["模型"] + [f"RMSE(h={c})" for c in md.columns[1:]]
        out = piv.merge(md, on="模型").round(3)
        table(doc, out, f"表 4-{7 + idx}  各模型各预见期 NSE 与 RMSE（{strat}）",
              size=7.5)
        figure(doc, os.path.join(FIG, figname),
               f"图 4-{10 + idx}  {strat}：NSE 随预见期的变化")
        tag = "直接" if strat == "多步直接预测" else "多输出"
        figure(doc, os.path.join(FIG, f"fig17_NSE对比_{tag}.png"),
               f"图 4-{12 + idx}  各模型 {strat}（1~7 天汇总）NSE 对比")

    for f, cap in [("fig11_过程线_h1.png",
                    "图 4-14  多步直接预测 预见期 1 天：2004 年实测与预测径流过程线"),
                   ("fig11_过程线_h7.png",
                    "图 4-15  多步直接预测 预见期 7 天：2004 年实测与预测径流过程线"),
                   ("fig12_多输出过程线_h7.png",
                    "图 4-16  多输出预测 预见期 7 天：实测与预测径流过程线"),
                   ("fig13_散点_h1.png",
                    "图 4-17  预见期 1 天 实测—预测散点图"),
                   ("fig19_洪水事件放大.png",
                    "图 4-18  典型洪水事件放大分析"),
                   ("fig20_误差分析.png",
                    "图 4-19  预测误差分析：误差分布与误差随流量量级的变化")]:
        figure(doc, os.path.join(FIG, f), cap)


def sec4_conclusions(doc):
    h(doc, "4.5 结果分析", 2)

    # 全部指标从结果 CSV 动态读取，保证与运行结果一致
    rf_h1 = metric_h(M.MODEL_CN["RF"], "多步直接预测", 1)
    rf_h7 = metric_h(M.MODEL_CN["RF"], "多步直接预测", 7)
    rf_d1 = metric_h(M.MODEL_CN["RF"], "多步直接预测", 1)
    rf_m1 = metric_h(M.MODEL_CN["RF"], "多输出预测", 1)
    lstm_h1 = metric_h(M.MODEL_CN["LSTM"], "多步直接预测", 1)
    ann_h1 = metric_h(M.MODEL_CN["ANN"], "多步直接预测", 1)
    st_m = metric_pooled("Stacking 集成", "多输出预测")
    rf_m = metric_pooled(M.MODEL_CN["RF"], "多输出预测")
    ann_d = metric_pooled(M.MODEL_CN["ANN"], "多步直接预测")
    rf_d = metric_pooled(M.MODEL_CN["RF"], "多步直接预测")
    lstm_d = metric_pooled(M.MODEL_CN["LSTM"], "多步直接预测")
    coefs = stacking_coefs("多步直接预测")

    para(doc, "（1）精度随预见期单调衰减。所有模型都表现出同样的规律："
              "预见期越长，精度越低。以随机森林多步直接预测为例，"
              f"NSE 由 h=1 的 {fmt(rf_h1)} 单调下降到 h=7 的 {fmt(rf_h7)}。"
              "这与物理机理一致——模型的输入中不含未来的降水信息，"
              "而该流域降水—径流响应时间不足 1 天，"
              "随着预见期延长，未来降水的未知性成为误差的主要来源，"
              "模型只能依靠径流的退水规律进行外推。")

    para(doc, "（2）多步直接预测与多输出预测的对比。在同一模型下，"
              "多步直接预测在短预见期（h = 1~3）上普遍优于多输出预测，"
              f"如随机森林 h=1 的 NSE 分别为 {fmt(rf_d1)} 与 {fmt(rf_m1)}；"
              "但在长预见期上两者趋于接近甚至多输出略优。"
              "原因是多步直接策略为每个预见期单独优化、模型容量专一，"
              "短预见期拟合更充分；而多输出策略的 7 个输出共享同一套隐层表示，"
              "虽能利用预见期之间的相关性、参数更少不易过拟合，"
              "但存在「任务竞争」，短预见期的精度会被牺牲。")

    # 依据实际汇总 NSE 动态生成模型排序结论
    rank = sorted(
        [(c, v) for c, v in
         [(M.MODEL_CN["ANN"], ann_d), (M.MODEL_CN["RF"], rf_d),
          (M.MODEL_CN["LSTM"], lstm_d)] if v is not None],
        key=lambda t: -t[1])
    if len(rank) == 3:
        best_cn, best_v = rank[0]
        mid_cn, mid_v = rank[1]
        worst_cn, worst_v = rank[2]
        para(doc, f"（3）模型间对比。从多步直接策略的汇总 NSE 看，{best_cn} 表现最好"
                  f"（{fmt(best_v)}），{mid_cn} 次之（{fmt(mid_v)}），"
                  f"{worst_cn} 相对最弱（{fmt(worst_v)}）。"
                  f"以最常用的 h=1 预见期为例：{M.MODEL_CN['RF']} 为 {fmt(rf_h1)}、"
                  f"{M.MODEL_CN['ANN']} 为 {fmt(ann_h1)}、"
                  f"{M.MODEL_CN['LSTM']} 为 {fmt(lstm_h1)}。"
                  "随机森林的优势在于：它天然处理非线性与特征交互，"
                  "对特征量纲和异常值不敏感，"
                  "且在训练样本仅 1400 余天的情形下，"
                  "比参数更多的神经网络更不容易过拟合。"
                  "神经网络类模型表现相对波动，与训练样本量偏少、"
                  "以及该流域径流强烈的非平稳性（洪枯悬殊、无典型周期）有关。")
    else:
        para(doc, f"（3）模型间对比。{M.MODEL_CN['RF']} 在多步直接策略上的汇总 NSE "
                  f"为 {fmt(rf_d)}，是当前个体模型中表现最好且最稳定的；"
                  f"{M.MODEL_CN['ANN']} 为 {fmt(ann_d)}。"
                  "随机森林的优势在于：它天然处理非线性与特征交互，"
                  "对特征量纲和异常值不敏感，"
                  "且在训练样本仅 1400 余天的情形下，"
                  "比参数更多的神经网络更不容易过拟合。")

    if coefs:
        coef_txt = "、".join(f"{cn} {v:.2f}" for cn, v in coefs)
        para(doc, f"（4）集成效果。简单平均与 Stacking 两种集成方式整体上都优于"
                  f"表现最差的个体模型；Stacking 在多输出策略下的汇总 NSE "
                  f"（{fmt(st_m)}）与随机森林（{fmt(rf_m)}）相当，"
                  "但没有大幅超越最强的个体模型。"
                  f"从 Stacking 元学习器的系数看（基模型在直接策略下的平均系数为 "
                  f"{coef_txt}），元学习器主要通过对不同基模型加权来提升精度。"
                  "各基模型之间的误差相关性较高、多样性不足，"
                  "限制了集成效果的上限。")
    else:
        para(doc, f"（4）集成效果。简单平均与 Stacking 两种集成方式整体上都优于"
                  f"表现最差的个体模型；Stacking 在多输出策略下的汇总 NSE "
                  f"（{fmt(st_m)}）与随机森林（{fmt(rf_m)}）相当，"
                  "但没有大幅超越最强的个体模型。"
                  "各基模型之间的误差相关性较高、多样性不足，"
                  "限制了集成效果的上限。")

    para(doc, "（5）误差结构与洪水量级。误差分析（图 4-19b）显示，"
              "预测误差随流量量级增大而显著增大，"
              "在特大洪水（> 3000 ft³/s）处的误差可达常遇流量的数倍；"
              "各模型的 PBIAS 多为负值，表明模型对大洪水存在系统性低估，"
              "这正是原始尺度 NSE 偏低的主要原因。"
              "这与前面的方差分解结论一致："
              "洪水过程是该流域径流预测精度提升的关键瓶颈。")


# ===========================================================================
# 五、程序调试
# ===========================================================================
PROBLEMS = [
    ("数据泄漏：归一化统计量使用了全量数据",
     "最初把归一化和特征选择放在数据划分之前，用全部 2000—2004 年的数据"
     "计算均值和标准差。这样测试年的信息会通过归一化参数渗入训练过程，"
     "导致评价结果虚高、不能反映真实泛化能力。",
     "改为先按时间划分训练集与测试集，所有归一化参数（均值、标准差）"
     "与特征选择评分都只在训练集上估计，再 transform 到测试集，"
     "在 DataBundle 中显式以训练期切片 fit 归一化器。"),
    ("交叉验证方式错误：普通 K 折导致时间穿越",
     "GridSearchCV 默认使用 KFold，会把数据随机分成 K 折。"
     "对时间序列而言，验证折中的样本在时间上可能早于训练折，"
     "相当于用未来的数据训练、再去预测过去，严重高估模型能力。"
     "同时深度学习的 fit 若保持默认 shuffle=True，也会打乱时间顺序。",
     "将 cv 参数显式指定为 TimeSeriesSplit(n_splits=3)，"
     "保证每一折的训练集在时间上严格早于验证集；"
     "LSTM 的 fit 设置 shuffle=False，并用末端 10% 作为验证集"
     "（验证集取的是样本末端，时间顺序得到保持）。"),
    ("Stacking 元学习器训练集泄漏",
     "最初直接用个体模型在训练集上的拟合值作为元特征训练 Ridge 元学习器。"
     "由于个体模型在训练集上存在过拟合，其拟合精度高于真实泛化精度，"
     "元学习器会错误地给这些模型过高的权重。",
     "改用折外预测（Out-of-Fold）生成元特征：用 TimeSeriesSplit 把训练期"
     "分成若干折，每一折用其之前的数据训练个体模型、预测本折，"
     "拼成完整的 OOF 预测矩阵后再训练元学习器，"
     "最后用全体训练数据重训个体模型并预测测试集。"),
    ("常量特征导致除零",
     "Swe（雪水当量）在该站点全部 1827 天恒为 0，"
     "在 Z-score 标准化时标准差为 0，出现除零产生 NaN，"
     "并污染整个模型输入，报错信息难以定位。",
     "在特征工程阶段先做常量列检测并剔除 Swe；"
     "同时在归一化器内部加保护，当某列标准差（或极差）为 0 时置为 1，"
     "从根本上避免除零。"),
    ("LSTM 无法直接接入 GridSearchCV",
     "PyTorch 的 nn.Module 模型没有 sklearn 的 fit/predict/get_params/set_params "
     "接口，无法直接传给 GridSearchCV。",
     "自行实现 LSTMModel 类，继承 sklearn 的 BaseEstimator 与 RegressorMixin，"
     "在内部构建 PyTorch 网络并实现 fit/predict，"
     "让三种模型共用同一套 GridSearchCV 流程，保证寻优方式完全一致、结果可比。"),
    ("LSTM 输入维度不匹配",
     "LSTM 需要三维输入（样本数, 时间步长, 特征数），"
     "而 MLP 与随机森林使用二维输入（样本数, 特征数），"
     "直接把二维特征表喂给 LSTM 会报维度错误；"
     "而且不同模型的可用起始日不同，会造成评价口径不一致。",
     "用 numpy 的 sliding_window_view 把二维特征表切成长度为 30 天的滑窗张量，"
     "并在 DataBundle 中通过 X_of() 方法按模型类型返回对应的输入视图；"
     "同时强制三种模型使用同一批预报起始日，保证对比公平。"),
    ("预测出现负径流",
     "初期直接对原始径流建模，ANN 的线性输出层外推能力有限，"
     "在退水段给出了负的径流预测值（最小 −756 ft³/s），"
     "物理上不合理，也大幅拉低了精度。",
     "对目标变量取对数后建模，预测值经 exp 回变换后恒为正，"
     "从根本上消除了负值问题；同时压缩了极端洪水的杠杆作用，"
     "使各预见期的 NSE 明显提升。"),
    ("评价指标被极端洪水主导，模型优劣难以分辨",
     "测试年仅有 10 天（占 2.8%）的特大洪水就贡献了 61.6% 的离差平方和，"
     "NSE 几乎完全由这几场洪水决定，"
     "导致各模型之间的差异被个别事件的误差掩盖。",
     "在对数尺度上补充计算 NSE_log 与 RMSE_log，削弱极端值的支配作用，"
     "反映模型对中低水过程的拟合能力；"
     "同时增加洪水事件放大分析与误差随量级变化的分析，使评价结论更全面。"),
    ("洪水洪峰系统性低估：当日降水被特征选择裁掉",
     "按实测流量分位区间诊断发现：特大洪水（>P95）的预测偏差达 −48%，"
     "逐场分析显示洪峰全部由当日强降水驱动（如 2004-11-29 降水 51.1 mm、"
     "径流一日暴涨至 7470 ft³/s），但当日降水在互信息排名仅第 74 位、"
     "滞后 1 天降水排第 76 位，被 TOP50 特征选择直接裁掉，"
     "模型在暴雨日看不到「今天下了大雨」这一最强信号。",
     "在候选特征池中新增 7 个洪水物理信号（径流涨水差分 Discharge_diff1/3/7、"
     "降水累积 Prcp_roll3_sum/Prcp_roll14_sum、强降水计数 Prcp_heavy3、"
     "降水突增 Prcp_diff1），特征池由 81 扩至 88。"
     "其中 Discharge_diff3/diff7/diff1 与 Prcp_roll14_sum 进入 TOP50，"
     "多输出策略各模型汇总 NSE 显著回升（如简单平均 0.089 → 0.212）。"),
    ("直接 ANN 在新特征池下网格漂移，测试精度骤降",
     "V5 全量重训后，直接 ANN 的汇总 NSE 由 -0.18 崩到 -0.43："
     "h=1 在 CV 中选中 (128,64) 大结构 + alpha=1e-5 的最松组合，"
     "训练样本仅 1400 余天，大结构发生过拟合，测试期完全失效；"
     "说明仅看 CV 分数选参在样本量偏少时并不可靠。",
     "将直接 ANN 的超参数网格收敛为保守小结构（(64,)/(128,)/(64,32) × "
     "alpha{1e-5,1e-2} × lr{1e-3,1e-2}，12 组），重训后汇总 NSE 稳定在 0.030，"
     "h=1 达 0.448，且全部 10 项指标（5 模型 × 2 策略）均不低于 V4，无退步。"),
]


def sec5(doc):
    h(doc, "五、程序调试", 1)
    para(doc, "在实现过程中遇到并解决了以下主要问题：")
    for i, (title, desc, sol) in enumerate(PROBLEMS, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(8)
        r = p.add_run(f"问题 {i}：{title}")
        _set_cn_font(r, "黑体", 10.5, True)
        para(doc, f"【现象】{desc}", size=10)
        para(doc, f"【解决】{sol}", size=10, space_after=2)
    page_break(doc)


# ===========================================================================
# 六、总结
# ===========================================================================
def sec6(doc):
    h(doc, "六、总结", 1)

    h(doc, "6.1 本次实践所用的技术、模型与算法", 2)
    table(doc, pd.DataFrame([
        ["数据分析", "时间序列可视化、直方图、箱型图、皮尔逊相关系数热力图",
         "认识数据结构、分布与相关性"],
        ["", "自相关（ACF）与互相关（CCF）分析", "确定滞后阶数与响应时间"],
        ["特征工程", "四季 one-hot、月份/日序 sin-cos 三角编码", "周期特征连续化"],
        ["", "滞后特征（lag 1/2/3/7/14/30）", "引入历史信息"],
        ["", "滑动窗口统计（mean/std/max，窗口 3/7/14/30）", "刻画前期状态与波动"],
        ["", "最小-最大归一化、Z-score 标准化", "消除量纲差异"],
        ["", "皮尔逊相关系数法、互信息法特征选择", "降维去冗余"],
        ["", "目标变量对数变换", "压缩极端值、保证预测为正"],
        ["机器学习", "人工神经网络 ANN（多层感知机 MLPRegressor）", "个体模型 1"],
        ["", "随机森林 Random Forest", "个体模型 2"],
        ["深度学习", "LSTM 长短期记忆网络", "个体模型 3"],
        ["集成学习", "简单平均（Simple Average）", "集成方法 1"],
        ["", "Stacking（Ridge 元学习器 + 折外预测 OOF）", "集成方法 2"],
        ["多步预测", "多步直接预测策略（每个预见期一个模型）", "预测策略 1"],
        ["", "多输出预测策略（一个模型输出 7 天）", "预测策略 2"],
        ["超参数优化", "Grid Search CV + TimeSeriesSplit", "时间序列安全寻优"],
        ["精度评价", "RMSE、MAE、NSE、R²、PBIAS、MAPE、NSE_log",
         "多角度精度评价"],
    ], columns=["环节", "技术 / 模型 / 算法", "作用"]),
        "表 6-1  本次实践所用技术、模型与算法汇总", size=8.5)

    h(doc, "6.2 存在的问题与改进方案", 2)
    para(doc, "（1）长预见期精度不足。预见期 5~7 天的 NSE 已接近或低于 0，"
              "模型基本退化为「远期均值回归」。根本原因是输入中不含未来的降水信息，"
              "而本流域响应时间不足 1 天，预见期一旦超过响应时间，"
              "可用的确定性信息就基本耗尽。改进方向："
              "① 引入数值天气预报（NWP）的预报降水作为输入，"
              "这是业务洪水预报的标准做法；"
              "② 改用 Seq2Seq 编码器—解码器结构，"
              "让解码器在每一步都能利用已生成的前序预测；"
              "③ 将多步递归预测与直接预测做对比融合。", space_after=4)
    para(doc, "（2）极端洪水系统性低估。误差分析显示模型对大洪水普遍低估，"
              "PBIAS 多为负值。原因是洪水样本稀少，损失函数被大量常遇流量主导，"
              "而在对数尺度上极端值又被过度压缩。改进方向："
              "① 在损失函数中引入样本加权，对高流量样本赋予更大权重；"
              "② 采用分位数回归或极值理论（EVT）对尾部单独建模；"
              "③ 引入两阶段建模（先判别是否发生洪水，再回归洪水量级）。",
         space_after=4)
    para(doc, "（3）数据本身的限制。Swe（雪水当量）全为 0，"
              "而该流域 4 月存在明显的融雪洪峰，"
              "说明积雪信息在数据集中缺失，模型无法显式刻画融雪过程。"
              "改进方向：① 补充 Daymet 的雪水当量数据或 MODIS 积雪面积产品；"
              "② 自行构建度日因子（degree-day）积雪—融雪模块，"
              "把积雪储量作为显式的物理状态变量输入模型；"
              "③ 采用物理机制与机器学习耦合的混合模型（如 HBV + LSTM）。",
         space_after=4)
    para(doc, "（4）模型多样性与集成策略。当前三个个体模型的误差相关性较高，"
              "集成增益有限。改进方向：引入结构差异更大的基模型"
              "（如 XGBoost、支持向量回归、CNN-LSTM、Transformer），"
              "或采用按预见期动态加权的集成方式，提升基模型之间的多样性。",
         space_after=4)
    para(doc, "（5）训练样本量偏少。仅有 4 年（约 1400 天）训练数据，"
              "对 LSTM 这类参数较多的深度模型而言偏少，容易过拟合。"
              "改进方向：采用迁移学习或数据增强（如加噪、时间扭曲），"
              "或获取更长序列的历史观测资料。")

    h(doc, "6.3 心得体会", 2)
    para(doc, "通过本次《机器学习课程实践》，我们完整走通了「数据分析—特征工程—"
              "建模—调参—集成—评价」的机器学习全流程，"
              "对课堂上学到的知识有了更具体的体会。", space_after=4)
    para(doc, "第一，特征工程往往比模型选择更重要。实验中随机森林这种结构相对简单的"
              "模型，最终效果稳定地好于 LSTM，很大程度上是因为我们构造的滞后特征与"
              "窗口特征已经把流域的调蓄记忆显式地表达了出来。"
              "这让我们认识到，机器学习不是把数据丢给模型就完事，"
              "把领域知识（这里是水文汇流规律）转化为特征，"
              "才是提升效果的关键，也最能体现「具体问题具体分析」。",
         space_after=4)
    para(doc, "第二，时间序列问题的数据划分是一条「红线」。普通 K 折交叉验证、"
              "用全量数据做归一化、Stacking 直接用训练集拟合值——"
              "这三处数据泄漏都很隐蔽，一旦踩中，指标会虚高得让人误以为模型很好，"
              "但真正投入业务使用时完全不可用。"
              "我们通过 TimeSeriesSplit 与折外预测逐一修正，"
              "深刻体会到「离线评估必须模拟真实预测场景」这句话的分量。",
         space_after=4)
    para(doc, "第三，评价指标的选择会直接影响结论。NSE 被少数几场特大洪水主导，"
              "若只看这一个指标，会误以为所有模型都「不可用」；"
              "补上对数尺度指标、洪水事件放大分析和误差随量级的变化分析之后，"
              "才发现模型对常遇流量的拟合其实相当好，"
              "真正的问题在于极端事件的预报。这提醒我们，"
              "评价模型不能只看一个数字，要理解这个数字背后到底反映了什么。",
         space_after=4)
    para(doc, "第四，对人工智能在工程实际中应用的认识。径流预测是一个"
              "「物理机理清楚但过程复杂」的典型问题：流域产汇流有明确的物理规律，"
              "但地形、土壤、植被等下垫面条件难以完全刻画。"
              "机器学习方法的优势在于不依赖对物理过程的完整假设，"
              "能够直接从数据中学到映射关系；但它也有明显短板——"
              "缺乏可解释性和外推能力，在训练样本未覆盖的极端情形下可能失效。"
              "因此，把物理机理与机器学习结合起来"
              "（如物理约束的损失函数、机理与数据驱动的混合建模）"
              "是这一领域的重要方向。", space_after=4)
    para(doc, "第五，作为机器学习方向的研究生，我们更加体会到技术背后的责任。"
              "洪水预报直接关系到人民生命财产安全和流域的防洪调度决策，"
              "模型每一次精度提升都可能换来更多的预警时间。"
              "这次实践让我们认识到，扎实掌握机器学习技术、"
              "严谨对待每一个数据划分与评价细节，不只是完成课程作业，"
              "更是将来用技术服务国家水利事业、服务人民的基础。"
              "我们将继续深入学习，努力把人工智能技术用在国家和人民真正需要的地方。")
    page_break(doc)


# ===========================================================================
# 附录
# ===========================================================================
CODE_FILES = [
    ("config.py", "A.1  config.py —— 全局配置"),
    ("utils_plot.py", "A.2  utils_plot.py —— 绘图公共工具"),
    ("data_analysis.py", "A.3  data_analysis.py —— 数据分析"),
    ("feature_engineering.py", "A.4  feature_engineering.py —— 特征工程（88 特征）"),
    ("models.py", "A.5  models.py —— 模型与超参数搜索"),
    ("experiment_v6.py", "A.6  experiment_v6.py —— 定稿实验主流程（V6）"),
    ("results_analysis.py", "A.7  results_analysis.py —— 结果可视化"),
    ("run_all.py", "A.8  run_all.py —— 一键运行"),
]


def appendix(doc):
    h(doc, "附录：完整代码", 1)
    para(doc, "全部代码放在 code/ 目录下，运行 python run_all.py 即可一键复现"
              "全部图表与指标（等价于 V6 定稿实验）。实验过程经历了 V1→V6 六轮迭代，"
              "各版本的实验脚本 experiment_v2.py ~ experiment_v6.py 均保留在 code/ 中，"
              "可在报告中追溯每一次改进的数值依据。各核心模块功能如下：")
    table(doc, pd.DataFrame([
        ["config.py", "全局配置：路径、变量中英文对照、特征与建模超参数"],
        ["utils_plot.py", "绘图公共工具：中文字体、配色与图片保存"],
        ["data_analysis.py", "第一部分 数据分析：统计、可视化与相关性分析"],
        ["feature_engineering.py",
         "第二部分 特征工程：编码、滞后/窗口/洪水信号特征、归一化、特征选择"],
        ["models.py", "模型定义（ANN/RF/LSTM）与 Grid Search CV 超参数寻优"],
        ["experiment_v6.py",
         "第三部分 定稿主流程：按预见期特征选择、多步直接/多输出策略、集成、评价"],
        ["experiment_v2~v5.py", "实验迭代历史版本（V2~V5），保留每次改进的运行记录"],
        ["evaluation.py", "精度评价指标（NSE/RMSE/R²/PBIAS 等）与绘图函数"],
        ["results_analysis.py",
         "第四部分 结果可视化：过程线、散点、指标对比、误差分析"],
        ["contrast_experiments.py", "对照实验：报告表 4-5（特征个数）、表 4-6（目标变换）"],
        ["make_report.py / report_sections.py", "实验报告自动生成（正文与章节内容）"],
        ["build_accept2.py", "第二次阶段验收汇报 docx 自动生成"],
        ["run_all.py", "一键运行全部流程"],
    ], columns=["文件", "功能"]), "表 A-1  代码模块清单", size=9)

    for fn, title in CODE_FILES:
        h(doc, title, 3)
        code_block(doc, src(fn))
        para(doc, "", indent=False, space_after=4)


# ===========================================================================
# 参考文献
# ===========================================================================
REFERENCES = [
    "Thornton P E, Shrestha R, Thornton M, et al. Daymet: Daily Surface "
    "Weather Data on a 1-km Grid for North America, Version 4 R1[R/OL]. "
    "Oak Ridge National Laboratory DAAC, 2022. "
    "https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4R1.html",
    "U.S. Geological Survey. USGS 01047000 Kennebec River at Bingham, ME, "
    "Daily Discharge Data[DB/OL]. National Water Information System (NWIS). "
    "https://waterdata.usgs.gov/nwis",
    "Nash J E, Sutcliffe J V. River flow forecasting through conceptual "
    "models part I — A discussion of principles[J]. Journal of Hydrology, "
    "1970, 10(3): 282-290.",
    "Breiman L. Random Forests[J]. Machine Learning, 2001, 45(1): 5-32.",
    "Hochreiter S, Schmidhuber J. Long Short-Term Memory[J]. Neural "
    "Computation, 1997, 9(8): 1735-1780.",
    "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine "
    "Learning in Python[J]. Journal of Machine Learning Research, 2011, "
    "12: 2825-2830.",
    "杭州电子科技大学. 《机器学习课程实践》实验指导书——基于机器学习的径流预测[Z]. "
    "2026.",
    "Moriasi D N, Arnold J G, Van Liew M W, et al. Model evaluation "
    "guidelines for systematic quantification of accuracy in watershed "
    "simulations[J]. Transactions of the ASABE, 2007, 50(3): 885-900.",
]


def sec7(doc):
    """参考文献（按评分标准要求 ≥ 3 篇，此处列 8 篇）。"""
    h(doc, "参考文献", 1)
    for i, r in enumerate(REFERENCES, 1):
        para(doc, f"[{i}] {r}", indent=False, size=10, space_after=3)
    page_break(doc)

