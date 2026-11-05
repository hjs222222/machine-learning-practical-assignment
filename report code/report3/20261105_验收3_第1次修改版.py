"""2026-11-05 验收3 第1次修改版；只生成LSTM与两类集成验收材料。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'final'))
from build_final_report import build_acceptance

if __name__=='__main__':
    print(build_acceptance(Path(sys.argv[1]).resolve(),3))
