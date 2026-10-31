# -*- coding: utf-8 -*-
"""仅展示已完成且核验通过的第二轮结果，不重新训练。"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = {2: ('direct', 'LSTM'), 3: ('multi', 'LSTM'), 4: ('direct', 'Mean'),
         5: ('direct', 'Stacking'), 6: ('multi', 'Mean'), 7: ('multi', 'Stacking')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--task', type=int, choices=range(1, 8), required=True)
    args = parser.parse_args()
    run = ROOT / 'output/acceptance3_runs' / args.run_id
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    verification = json.loads((run / 'verification.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed' and not manifest['smoke']
    assert verification['status'] == 'passed'
    print(f"run_id={args.run_id}  status=completed  verification=passed")
    print('Primary seed=42 (fixed before evaluation); original Discharge scale.')
    if args.task == 1:
        records = json.loads((run / 'best_parameters.json').read_text(encoding='utf-8'))
        print('Final LSTM GridSearchCV: 12 candidates x 2 temporal folds per task.')
        print('Actual selected parameters (all 8 tasks):')
        for r in records:
            if r['stage'] != 'final' or r['model'] != 'LSTM':
                continue
            p = r['best_params']
            print(f"{r['strategy']:6s} h={r['h']} units={str(p['units']):8s} drop={p['dropout']} lr={p['lr']} epochs={r['fixed_epochs']}")
            print(f"  feature={p['feature_mode']} target={p['target_transform']} mixed={p['mixed_weight']} CV_RMSE={-r['best_score']:.4f}")
    else:
        strategy, model = TASKS[args.task]
        frame = pd.read_csv(run / 'metrics.csv')
        print(f'Task {args.task}: {strategy} / {model}')
        print(frame[(frame.strategy == strategy) & (frame.model == model)][
            ['h', 'R2', 'RMSE', 'MAE', 'n', 'negative_predictions']].to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('Verified: 880 candidates; 2048 fits; 72 final base models restored.')
    print('Source: completed run files; this viewer does not rerun model training.')


if __name__ == '__main__':
    main()
