"""2026-11-05 最终报告 第1次修改版；从已完成冻结结果生成分析附件。"""
from pathlib import Path
import sys
import json
import joblib
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from data import load_raw,build_dataset
from data_analysis import make_analysis
from config import save_json


if __name__=='__main__':
    run=Path(sys.argv[1]).resolve()
    assert json.loads((run/'manifest.json').read_text(encoding='utf-8'))['status']=='completed'
    raw=load_raw()
    make_analysis(raw,build_dataset(raw),run)
    models=[joblib.load(p) for p in (run/'models').glob('*_LSTM_*.joblib')]
    devices=sorted({m.device_used_ for m in models})
    save_json(run/'analysis'/'devices.json',dict(training_devices=devices,
        current_gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        note='training_devices读取保存模型的训练设备；current_gpu为报告生成时机器核验'))
