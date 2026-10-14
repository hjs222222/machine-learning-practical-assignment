# -*- coding: utf-8 -*-
"""周期正余弦编码不再作为特征工程输出。"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import add_time_features


df = pd.DataFrame({"Date": pd.to_datetime(["2000-01-01", "2000-06-01"])})
result = add_time_features(df)

removed = {"doy_sin", "doy_cos", "month_sin", "month_cos"}
assert removed.isdisjoint(result.columns), "不应生成周期正余弦编码列"
assert {"season", "month"}.issubset(result.columns), "保留季节和月份编码"

print("PASS: 未生成周期正余弦编码列，季节和月份编码仍保留")
