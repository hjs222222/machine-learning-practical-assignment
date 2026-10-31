# -*- coding: utf-8 -*-
"""第三轮冻结后真实结果只读展示；不训练、不修改参数或指标。"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--task',type=int,choices=range(1,8),required=True)
    args = parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ['.','..']:
        parser.error('run-id必须为单目录名')
    run = ROOT/'output/acceptance3_runs'/args.run_id
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    check = json.loads((run/'verification.json').read_text(encoding='utf-8'))
    assert manifest['status']=='completed' and not manifest['smoke']
    assert check['status']=='passed' and check['test_2004_predicted']
    source = ROOT/'output/acceptance3_runs'/manifest['source_run']
    frozen = json.loads((source/'frozen_config.json').read_text(encoding='utf-8'))
    print('任务%d | 第三轮只读结果展示，不是重新训练 | primary_seed=42' % args.task)
    print('2004为回溯评估；直接策略未通过第三轮准入，保留第二轮已完成结果。')
    if args.task==1:
        print('第三轮仅更新multi；LSTM固定epoch参与GridSearchCV，原生七输出。')
        direct_run = ROOT/'output/acceptance3_runs/formal_20261031_v2'
        direct = json.loads((direct_run/'best_parameters.json').read_text(encoding='utf-8'))
        print('策略/h  units  dropout  lr  loss_lambda  epochs  features/target')
        for row in direct:
            if row['stage']=='final' and row['strategy']=='direct' and row['model']=='LSTM':
                p = row['best_params']
                print('direct/'+str(row['h']),p['units'],p['dropout'],p['lr'],
                      p['mixed_weight'],row['fixed_epochs'],p['feature_mode']+'/'+p['target_transform'])
        p = next(r['params'] for r in frozen['final_parameters'] if r['family']=='LSTM')
        print('multi/1-7',p['units'],p['dropout'],p['lr'],p['mixed_weight'],p['fixed_epochs'],
              p['feature_mode']+'/'+p['target_transform'])
        print('真正Stacking：',json.dumps(frozen['meta']['stacking_params'],ensure_ascii=False))
        print('SelectedFusion：',json.dumps(frozen['meta']['selected_params'],ensure_ascii=False))
        print('direct来自第二轮；multi来自第三轮，均按完整策略保留，不逐h挑分数。')
        return
    strategy,model = {2:('direct','LSTM'),3:('multi','LSTM'),4:('direct','Mean'),
                      5:('direct','Stacking'),6:('multi','Mean'),7:('multi','Stacking')}[args.task]
    metrics_path = run/'metrics.csv' if strategy=='multi' else ROOT/'output/acceptance3_runs/formal_20261031_v2/metrics.csv'
    frame = pd.read_csv(metrics_path)
    frame = frame[(frame.strategy==strategy)&(frame.model==model)].sort_values('h')
    assert len(frame)==7 and (frame.n==359).all()
    print('来源：',metrics_path.parent.name,'|',strategy,model)
    print(frame[['h','R2','RMSE','MAE','n','negative_predictions']].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    if args.task==7:
        print('以上为真正21→7 Ridge Stacking；不以RF退路分数冒充融合。')
        print('独立SelectedFusion：',frozen['meta']['selected_params'])


if __name__=='__main__': main()
