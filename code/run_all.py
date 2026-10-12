# -*- coding: utf-8 -*-
"""
《机器学习课程实践》—— 基于机器学习的径流预测
一键运行全部流程

用法：
    python run_all.py              # 完整运行（含全部 GridSearchCV，耗时较长）
    python run_all.py --fast       # 小规模快速验证（缩小网格与训练轮数）
    python run_all.py --skip-model # 跳过建模，仅重绘图表/重新生成报告

流程：
    第一部分 数据分析      data_analysis.run()
    第二部分 特征工程      feature_engineering.run()
    第三部分 径流预测      experiment.run()
    第四部分 结果可视化    results_analysis.main()
    报告生成              make_report.main()
"""

import os
import sys
import time
import warnings

# Windows 控制台默认 GBK，重定向到文件时中文乱码；统一为 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config as C


def _banner(title):
    print("\n" + "#" * 74)
    print(f"#  {title}")
    print("#" * 74)


def main():
    fast = "--fast" in sys.argv
    skip_model = "--skip-model" in sys.argv
    t0 = time.time()

    _banner("《机器学习课程实践》基于机器学习的径流预测 —— 全流程运行")
    print(f"数据文件：{C.DATA_PATH}")
    print(f"输出目录：{C.OUTPUT_DIR}")
    if fast:
        print("运行模式：FAST（快速验证，超参数网格与训练轮数已缩小）")

    # ---------------- 第一部分：数据分析 ----------------
    _banner("第一部分  数据分析")
    from data_analysis import run as analysis_run
    df = analysis_run()

    # ---------------- 第二部分：特征工程 ----------------
    _banner("第二部分  特征工程")
    from feature_engineering import run as fe_run
    d, feature_cols, selected = fe_run(df)

    # ---------------- 第三部分：径流预测 ----------------
    if not skip_model:
        _banner("第三部分  径流多步预测（V6 定稿实验）")
        from experiment_v6 import run as exp_run
        exp_run(d, feature_cols, fast=fast)
    else:
        print("\n[跳过] 建模阶段")

    # ---------------- 第四部分：结果可视化 ----------------
    _banner("第四部分  结果可视化")
    from results_analysis import main as viz_main
    viz_main()

    # ---------------- 报告生成 ----------------
    _banner("生成实验报告")
    from make_report import main as report_main
    report_main()

    _banner(f"全部完成，总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"图表：{C.FIG_DIR}")
    print(f"结果表：{C.RESULT_DIR}")
    print(f"报告：{os.path.join(C.BASE_DIR, '实验报告-机器学习课程实践-径流预测.docx')}")


if __name__ == "__main__":
    main()
