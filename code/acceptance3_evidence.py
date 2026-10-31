# -*- coding: utf-8 -*-
"""任务1至7+实际结果查看器，供终端运行和真实窗口截图。"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", type=int, required=True, choices=range(1, 8))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / "output/acceptance3_runs" / args.run_id
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and not manifest["smoke"]
    print("RUN_ID:", args.run_id)
    print("TRAINING STATUS:", manifest["status"])
    print("DATA SHA256:", manifest["data_sha256"])
    print("TASK:", args.task)
    if args.task == 1:
        best = json.loads((run / "best_parameters.json").read_text(encoding="utf-8"))
        print("Actual GridSearchCV results: 8 candidates x 3 time folds per task")
        print("Calibration: 2000-2001; gap=7; score=-original-scale RMSE")
        print("LSTM input=(N,30,50); direct output=(N,1); native multi output=(N,7)")
        print("Native multi: stacked LSTM -> last timestep -> Linear(64) -> ReLU -> Linear(7)")
        for row in best:
            if row["model"] != "LSTM":
                continue
            print(f"\n{row['strategy']} h={row['h']} best_index={row['best_index']}")
            print("params:", row["best_params"])
            print(f"CV RMSE={-row['best_score']:.6f}; std={row['std_test_score']:.6f}; epochs={row['fixed_epochs']}")
    else:
        mapping = {2: ("direct", "LSTM"), 3: ("multi", "LSTM"),
                   4: ("direct", "Mean"), 5: ("direct", "Stacking"),
                   6: ("multi", "Mean"), 7: ("multi", "Stacking")}
        strategy, model = mapping[args.task]
        table = pd.read_csv(run / "metrics.csv")
        part = table[(table.strategy == strategy) & (table.model == model)].sort_values("h")
        print(f"Strategy: {strategy}; model: {model}")
        print("R2: standard coefficient of determination, not squared correlation")
        print("RMSE/MAE: original Discharge scale; predictions are not clipped")
        print(part.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nEvidence source:", str(run / ("best_parameters.json" if args.task == 1 else "metrics.csv")))
    print("RESULT VIEW COMPLETED")


if __name__ == "__main__":
    main()
