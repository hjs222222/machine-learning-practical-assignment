# -*- coding: utf-8 -*-
"""任务2+5 ANN与RF网格搜索及多步预测复现入口。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_analysis import run as analysis_run
from feature_engineering import run as feature_run
import experiment_v6 as experiment
if __name__ == "__main__":
    data = analysis_run()
    engineered, features, _ = feature_run(data)
    # MLPRegressor 与 RandomForestRegressor 原生支持多目标，直接拟合7列目标。
    # 本流程不使用 MultiOutputRegressor 包装器。
    experiment.BASE_MODELS = ["ANN", "RF"]
    experiment.run(engineered, features, fast=False)
