# -*- coding: utf-8 -*-
"""项目数据文件应能从当前资料目录定位。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

assert os.path.isfile(C.DATA_PATH), f"未找到数据文件：{C.DATA_PATH}"
print(f"PASS: 数据文件已定位：{C.DATA_PATH}")
