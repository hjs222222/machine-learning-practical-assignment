"""唯一实验入口。基准计时与正式运行均从零开始，拒绝覆盖。"""
import argparse
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from config import ROOT, DATA_PATH, new_run_dir
from data import load_raw, build_dataset
from experiment import benchmark, develop, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark',action='store_true',help='仅训练期短跑，不读取或输出测试指标')
    parser.add_argument('--name',default=None,help='全新运行目录名，不可与历史运行相同')
    args = parser.parse_args()
    name = args.name or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run = new_run_dir(DATA_PATH,ROOT/'output'/name,dict(benchmark=args.benchmark,sequence_length=14,gap_days=7))
    before = time.monotonic()
    ds = build_dataset(load_raw())
    if args.benchmark:
        print(benchmark(ds,run),flush=True)
    else:
        develop(ds,run)
        evaluate(ds,run)
    print('OUTPUT',str(run),'SECONDS',time.monotonic()-before,flush=True)


if __name__ == '__main__':
    main()
