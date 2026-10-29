"""生成第三次验收的 Word 内容模板、Markdown 阅读版和空白 CSV 附件。

生成日期：2026-10-29；验收次数：第3次；版本：第1次修改版。
本脚本不训练模型，不读取或冒用历史精度，不生成虚构图表。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import quote

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.opc.constants import RELATIONSHIP_TYPE as RT


VERSION = "2026-10-29_第3次验收_第1次修改版"
TASKS = {
    2: "LSTM直接策略",
    3: "LSTM多输出策略",
    4: "简单平均直接策略",
    5: "Stacking直接策略",
    6: "简单平均多输出策略",
    7: "Stacking多输出策略",
}
METRIC_COLUMNS = [
    "预见期_d", "R2", "RMSE", "MAE", "样本数", "起报时间范围",
    "目标时间范围", "run_id", "状态",
]
PARAM_COLUMNS = [
    "策略", "预见期", "lookback", "hidden_size", "num_layers", "dropout",
    "learning_rate", "batch_size", "max_epochs", "patience", "seed",
    "CV折数", "CV_gap", "CV评分名称", "CV最优均分", "CV标准差",
    "特征方案", "最终训练轮数", "run_id", "状态",
]
CV_COLUMNS = [
    "策略", "预见期", "候选编号", "参数JSON", "split0_test_score",
    "split1_test_score", "split2_test_score", "mean_test_score",
    "std_test_score", "rank_test_score", "mean_fit_time", "run_id",
]


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def add_link(paragraph, label, relative_path):
    relationship = paragraph.part.relate_to(quote(relative_path, safe="/"), RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship)
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
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def build(root, source):
    out = root / "验收" / "Acceptance 3"
    data = out / "data"
    artifacts = out / "artifacts"
    title = "任务1至7+第三次阶段验收内容模板"
    metrics = {task: f"任务{task}+{name}精度填写模板.csv" for task, name in TASKS.items()}
    parameter_file = "任务1+LSTM最优超参数填写模板.csv"
    cv_file = "任务1+LSTM网格搜索明细填写模板.csv"
    checklist_file = "任务1至7+交付检查清单.csv"
    source_file = "任务1至7+第三次阶段验收说明原文.docx"
    provenance_file = "任务1至7+模板来源与版本.json"
    targets = [out / f"{title}.docx", out / f"{title}.md"]
    targets += [data / name for name in metrics.values()]
    targets += [data / parameter_file, data / cv_file, data / checklist_file]
    targets += [artifacts / source_file, artifacts / provenance_file]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise FileExistsError("为保护已有交付物，拒绝覆盖：" + "；".join(existing))
    data.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, artifacts / source_file)

    for task, name in metrics.items():
        rows = [[h, "", "", "", "", "", "", "", "待训练/待填写"] for h in range(1, 8)]
        write_csv(data / name, METRIC_COLUMNS, rows)
    params = []
    for strategy, horizons in [("直接", range(1, 8)), ("多输出", ["1至7联合输出"])]:
        for horizon in horizons:
            row = {column: "" for column in PARAM_COLUMNS}
            row.update({"策略": strategy, "预见期": horizon, "状态": "待训练/待填写"})
            params.append([row[column] for column in PARAM_COLUMNS])
    write_csv(data / parameter_file, PARAM_COLUMNS, params)
    write_csv(data / cv_file, CV_COLUMNS, [])
    checklist = [
        ("范围", "任务1至7分别有独立结果和证据"),
        ("任务1", "8个LSTM模型有真实GridSearchCV候选、逐折成绩与最优参数"),
        ("任务2至7", "6组各7个预见期，共42行指标且每行包含R2/RMSE/MAE"),
        ("多输出", "LSTM/ANN/RF及元学习器均未调用MultiOutputRegressor"),
        ("数据", "训练目标日期不晚于2003-12-31，测试目标均位于2004年"),
        ("防泄漏", "特征选择和缩放仅在各折训练段拟合，边界按目标时间隔离"),
        ("Stacking", "元学习器只使用因果有效的OOF样本，无样本内预测和未来标签"),
        ("公平性", "同一预见期各模型起报时间、目标时间与样本数一致"),
        ("指标", "先回到原始径流尺度，再计算R2/RMSE/MAE，R2不是相关系数平方"),
        ("集成", "简单平均在原始尺度计算，输入恰为ANN、RF、LSTM三列"),
        ("图表", "图表来自本次run_id，全文和典型过程图均注明目标日期"),
        ("复现", "输入哈希、环境版本、随机种子、完整命令、配置与日志已保存"),
        ("交付", "一个Word/PPT集中展示，附件相对链接能在另一台机器打开"),
        ("完整性", "Word和CSV不存在待填写内容、空指标、NaN或无说明的失败候选"),
    ]
    write_csv(data / checklist_file, ["检查范围", "检查项", "检查状态", "证据或备注"],
              [[group, item, "待检查", ""] for group, item in checklist])

    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = section.right_margin = Cm(2)
    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    for style_name in ["Title", "Heading 1", "Heading 2"]:
        style = doc.styles[style_name]
        style.font.name = "黑体"
        style.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    markdown = []

    def heading(text, level=1):
        doc.add_heading(text, level=level)
        markdown.append("#" * (level + 1) + " " + text + "\n")

    def paragraph(text):
        doc.add_paragraph(text)
        markdown.append(text + "\n")

    def table(headers, rows):
        grid = doc.add_table(rows=1, cols=len(headers))
        grid.style = "Table Grid"
        for cell, value in zip(grid.rows[0].cells, headers):
            cell.text = str(value)
        for row in rows:
            for cell, value in zip(grid.add_row().cells, row):
                cell.text = str(value)
        markdown.append("| " + " | ".join(headers) + " |")
        markdown.append("| " + " | ".join(["---"] * len(headers)) + " |")
        markdown.extend("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in rows)
        markdown.append("")

    def link(label, relative_path):
        add_link(doc.add_paragraph(), label, relative_path)
        markdown.append(f"[{label}]({quote(relative_path, safe='/')})\n")

    doc.add_heading(title, level=0)
    markdown.append("# " + title + "\n")
    paragraph("模板状态：尚未训练，非已完成验收报告。所有“待填写”和空白指标必须用本次真实运行结果替换；本模板没有声明任何任务已完成。")
    paragraph(f"版本：{VERSION}。适用对象：第三阶段验收教师与汇报小组。")
    paragraph("验收安排：2026年11月5日（周四）13:30—16:50，第12教研楼202。请于11月5日前完成。")
    paragraph("小组：待填写；成员及学号：待填写；汇报人：待填写；实验日期：待填写；run_id / Git提交：待填写。")
    link("任务1至7+教师验收说明原文", f"artifacts/{source_file}")
    link("任务1至7+模板来源与版本", f"artifacts/{provenance_file}")

    heading("任务1至7+要求与证据总览")
    table(["任务", "教师要求", "需要展示的证据", "当前状态"], [
        ["任务1", "Grid Search CV确定LSTM超参数", "候选网格、时间CV、最优参数、逐折成绩、真实日志截图", "待训练"],
        *[[f"任务{task}", f"{name}的1至7天R2、RMSE、MAE", "7行精度表、过程线、真实结果截图", "待训练"] for task, name in TASKS.items()],
    ])
    paragraph("硬性要求：不得调用MultiOutputRegressor实现多输出；现场无需运行程序；结果图、表与程序运行截图需统一放入一个Word或PPT。原文集成部分标题编号为“3”，本模板按子任务1至7对应，不据此新增任务。")

    heading("任务1至7+公共实验条件")
    table(["项目", "填写内容"], [
        ["数据来源", "USGS 01047000；Date、Discharge及7个Daymet因子；原始1827行，2000-01-01至2004-12-31"],
        ["训练/测试", "2000—2003年训练；2004年测试；建样本后的实际行数、目标日期和边界处理：待填写"],
        ["起报和目标", "使用截至t日已知信息预测Q(t+1)…Q(t+7)；记录逐样本origin_date与target_date"],
        ["窗口与特征", "lookback、变量列表、特征选择规则与特征个数：待填写"],
        ["目标变换", "是否使用ln(Q)：待填写；如使用，exp回变换后评价；不能把log1p与ln混用"],
        ["单位", "R2无量纲；RMSE、MAE与原始Discharge同单位；原始径流单位及依据：待核实"],
        ["CV与预处理", "TimeSeriesSplit折数、gap、时间区间；各折训练内拟合scaler与selector：待填写"],
        ["后端与环境", "Python、NumPy、scikit-learn、TensorFlow/PyTorch及硬件：待填写"],
        ["复现信息", "run_id、随机种子、输入SHA256、命令、配置文件与日志归档位置：待填写"],
        ["多输出证明", "LSTM输出层为7维；ANN/RF使用原生多输出；Stacking使用联合7输出元学习器：待填写实际结构"],
    ])
    paragraph("指标定义：R2 = 1 − Σ(y−ŷ)²/Σ(y−ȳ)²；RMSE = √mean((y−ŷ)²)；MAE = mean(|y−ŷ|)。每个预见期单独计算，不用一个7输出平均分代替7行结果。R2可为负，不能改为相关系数平方或截断为0。")
    paragraph("评估协议：在2004年逐日起报的滚动预测；允许使用截至起报日已观测的2004年历史，不允许未来真值、未来天气实测或测试集拟合预处理；如模型固定不更新，应明确说明。")

    heading("任务1+LSTM网格搜索与最优超参数")
    paragraph("实现说明：待填写LSTM结构、序列输入形状、输出维数、损失函数、优化器、目标变换和sklearn兼容封装；附模型结构及真实GridSearchCV运行结果截图。截图不能只显示手填参数或文件存在。")
    table(["项目", "实际设置或结果"], [
        ["直接策略", "h=1…7分别搜索7个单输出LSTM；各模型网格与最优参数：待填写"],
        ["多输出策略", "单个LSTM联合输出7天，独立网格搜索：待填写"],
        ["候选网格", "lookback / hidden_size / learning_rate等实际候选：待填写"],
        ["CV", "折数、每折训练/验证日期、gap及边界检查：待填写"],
        ["评分", "实际scoring、优化方向及原始/对数尺度：待填写"],
        ["早停", "只使用每折训练段末尾内验证，不使用外层CV验证或2004测试集：待填写"],
        ["最优结果", "best_params_、best_score_、std_test_score、最终轮数和耗时：待填写"],
    ])
    table(["模型", "最优参数摘要", "CV均分±标准差", "最终轮数"],
          [[f"LSTM直接h={h}", "待填写", "待填写", "待填写"] for h in range(1, 8)] +
          [["LSTM多输出h=1…7", "待填写", "待填写", "待填写"]])
    link("任务1+8个LSTM最优超参数填写模板", f"data/{parameter_file}")
    link("任务1+网格搜索逐候选逐折明细填写模板（当前仅表头）", f"data/{cv_file}")
    paragraph("分析填写：参数选择依据、不同容量/学习率的CV差异、训练与验证曲线、失败候选原因、直接与多输出的结构差异。结论：待填写，未训练前不得声称已得到最优模型。")

    for task, name in TASKS.items():
        doc.add_page_break()
        heading(f"任务{task}+{name}1至7天径流预测")
        if task == 2:
            paragraph("方法填写：7个独立单输出LSTM，各模型预测一个预见期；给出对应lookback、特征、最优参数、训练样本数及测试样本数。")
        elif task == 3:
            paragraph("方法填写：一个共享LSTM编码器与7维输出层一次预测未来7天；提供模型结构摘要和(N,7)预测形状证据，禁止以7个独立回归器包装冒充多输出。")
        elif task in (4, 6):
            paragraph("方法填写：对ANN、RF、LSTM在同一起报点、同一预见期的原始径流预测做等权算术平均，即(Q_ANN + Q_RF + Q_LSTM)/3。若目标取ln，先各自exp回变换再平均；不能平均对数后exp。")
        elif task == 5:
            paragraph("方法填写：每个h有一个Ridge元学习器，以该h的ANN/RF/LSTM三列因果OOF预测为输入；7个元学习器对应直接策略。报告OOF覆盖、样本数、alpha及系数。")
        else:
            paragraph("方法填写：将3个原生多输出基模型的(N,7) OOF预测按固定列序拼为(N,21)，使用一个原生多输出Ridge联合拟合(N,7)目标。报告21列顺序、7×21系数、alpha与OOF覆盖；不得以7个独立元学习器声称联合多输出。")
        table(["预见期/天", "R2", "RMSE", "MAE", "样本数"],
              [[str(h), "待填写", "待填写", "待填写", "待填写"] for h in range(1, 8)])
        link(f"任务{task}+{name}精度填写模板", f"data/{metrics[task]}")
        paragraph("图表占位（不是已生成图表）：在此插入本次真实h=1、3、7的观测—预测过程线，以及R2/RMSE/MAE随预见期变化图；条件允许补充全预见期图或典型洪峰放大图。每图注明run_id、目标日期、单位、图例和相应原图相对链接。")
        paragraph("程序结果截图占位：插入本次真实运行完成及7行指标输出截图；截图中的run_id、模型和数值须与本节CSV一致。")
        paragraph("结果分析填写：最好/最差预见期、随预见期变化是否单调、洪峰/低流量段误差、直接与多输出差异：待填写。")
        if task in (4, 5, 6, 7):
            paragraph("集成对照填写：与同策略同预见期的ANN、RF、LSTM比较，给出R2绝对差和RMSE/MAE相对变化；如未改善，保留真实结论并分析基模型相关误差、OOF样本量及过拟合风险。不得预设集成一定更优。")
        if task in (5, 7):
            paragraph("OOF证据填写：每折训练/验证的起报日期、最晚训练目标日期、最早验证起报日期；逐折重新拟合预处理、基模型；剔除OOF首段空缺，不以训练拟合值补齐；元模型调参也隔离未来标签。最终基模型仅在2000—2003目标范围重新拟合后预测测试集。")
        paragraph("本任务结论：待填写。完成状态：待训练/待核验。")

    heading("任务1至7+综合对照与验收结论")
    table(["对照问题", "需填入的真实结论"], [
        ["LSTM直接 vs 多输出", "按相同h与同一测试样本比较R2/RMSE/MAE：待填写"],
        ["简单平均 vs 最佳单模型", "分别比较两种策略，不能只报告平均优势：待填写"],
        ["Stacking vs 简单平均", "分别比较两种策略与OOF风险：待填写"],
        ["最终结论", "已完成任务、失败或不足、改进方向：待填写"],
    ])
    paragraph("推荐汇报顺序：公共数据口径 → 任务1真实网格搜索 → 任务2/3的14行指标 → 任务4至7的28行指标 → 对照图与失败分析 → 交付检查。教师要求的是最终结果，不需要现场启动训练。")
    heading("任务1至7+交付前检查与附件填写规则")
    link("任务1至7+交付检查清单", f"data/{checklist_file}")
    paragraph("填写规则：CSV空白代表未知，绝不代表0；保留足够精度并统一展示位数。任务1网格明细当前仅有表头，必须补真实候选行；任务2至7当前共42条空指标行，必须全部补齐。所有任务填完后，再形成新的最终报告，不要将本文件直接作为已完成报告提交。")
    paragraph("图表规则：此模板没有绑定不存在的图片；添加真实图片时应将图片内嵌到Word并归档至figures，同时附可点击的相对链接。需要检查的代码以文本截图或PDF证据归档至artifacts；源代码仍保存在code或report code，不混入最终交付目录。")
    paragraph("附件规则：保留Word及其同目录data、figures、artifacts后整体打包。所有新附件都须实际存在，不使用本机绝对路径。模板阶段的CSV可点击打开不等于实验完成；最终应清除全部占位并复核所有数据与图表。")
    doc.save(out / f"{title}.docx")
    (out / f"{title}.md").write_text("\n".join(markdown), encoding="utf-8")
    provenance = {
        "生成日期": "2026-10-29", "验收次数": 3, "修改版本": "第1次修改版",
        "用途": "仅训练前填写模板，不含已训练结果", "训练已执行": False,
        "验收说明文件": source.name,
        "验收说明SHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "生成脚本": Path(__file__).name,
        "任务2至7指标预留行数": 42,
        "任务1最优参数预留行数": 8,
    }
    (artifacts / provenance_file).write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已生成第三阶段模板；未执行训练；指标保持空白。")
    print("交付文件数：", len(targets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="项目根目录")
    parser.add_argument("--source", type=Path, required=True, help="第三次阶段验收说明DOCX")
    args = parser.parse_args()
    build(args.root.resolve(), args.source.resolve())
