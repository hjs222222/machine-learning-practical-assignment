# -*- coding: utf-8 -*-
"""第三轮阶段D：仅对已冻结的多输出流程进行一次2004回溯评估，不训练或选参。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset, restore_model
from acceptance3_train import ROOT, prediction_rows, metrics_table
from acceptance3_v2_train import atomic_json

FAMILIES = ['ANN','RF','LSTM']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(source, output, smoke=False):
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    verification = json.loads((source/'verification.json').read_text(encoding='utf-8'))
    assert manifest['status']=='stage_c_completed' and manifest['smoke']==smoke
    assert verification['status']=='passed' and not verification['test_2004_predicted']
    frozen = json.loads((source/'frozen_config.json').read_text(encoding='utf-8'))
    assert frozen['strategy']=='multi' and frozen['primary_seed']==42
    raw = ROOT/'Teacher Resources & Requirements/01047000.csv'
    assert digest(raw)==manifest['data_sha256']
    # 读取正式快照并校验，防止保存后替换模型定义而不留来源。
    for file in (source/'source').glob('*.py'):
        assert file.read_bytes()==(ROOT/'code'/file.name).read_bytes(),file.name
    data = load_dataset(str(source/'dataset.npz'))
    days = data['origin_days']
    test = np.flatnonzero((days>=np.datetime64('2004-01-01').astype('int64')) &
        (days+7<=np.datetime64('2004-12-31').astype('int64')))
    assert len(test)==359
    model_files = [p for directory in ['models','meta_models'] for p in (source/directory).rglob('*') if p.is_file()]
    atomic_json(output/'selection_frozen.json',dict(source_run=source.name,source_manifest=manifest,
        frozen_config_sha256=digest(source/'frozen_config.json'),
        dataset_sha256=digest(source/'dataset.npz'),
        model_sha256={str(p.relative_to(source)):digest(p) for p in model_files},
        evaluated_after_freeze=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        limitation='2004曾在前两轮查看，本次仍为回溯评估，不是独立盲测；不使用本次分数选模型。'))
    rows = []
    for seed in frozen['seeds']:
        preds = {}
        for family in FAMILIES:
            model = restore_model(source/'models'/f'final_{family}_seed{seed}',device='cpu')
            pred = np.asarray(model.predict(test[:,None]))
            assert pred.shape==(359,7) and np.isfinite(pred).all()
            preds[family] = pred
        preds['Mean'] = np.mean([preds[f] for f in FAMILIES],axis=0)
        x = np.column_stack([preds[f] for f in FAMILIES])
        meta = joblib.load(source/'meta_models'/f'multi_seed{seed}.joblib')
        assert meta.kind==frozen['meta']['selected_params']['kind']
        preds['SelectedFusion'] = meta.predict(x)
        if meta.kind=='RF': np.testing.assert_array_equal(preds['SelectedFusion'],preds['RF'])
        # 验收任务7单列真正21→7 Ridge Stacking，不能把RF退路冒充融合。
        stacking = joblib.load(source/'meta_models'/f'stacking_seed{seed}.joblib')
        assert stacking.kind=='ridge_RF' and stacking.strength==1. and stacking.alpha==100.
        preds['Stacking'] = stacking.predict(x)
        for family,pred in preds.items():
            rows.extend(dict(**r,seed=seed) for r in prediction_rows(output.name,data,test,pred,'multi',family))
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(['seed','strategy','model','h','sample_id']).any()
    frame.to_csv(output/'test_predictions_all_seeds.csv',index=False,encoding='utf-8-sig')
    tables = []
    for seed,f in frame.groupby('seed'):
        table = metrics_table(f.to_dict('records'))
        table['seed'] = seed
        tables.append(table)
    metrics = pd.concat(tables,ignore_index=True)
    metrics.to_csv(output/'metrics_all_seeds.csv',index=False,encoding='utf-8-sig')
    metrics[metrics.seed==42].drop(columns='seed').to_csv(output/'metrics.csv',index=False,encoding='utf-8-sig')
    frame[frame.seed==42].to_csv(output/'test_predictions.csv',index=False,encoding='utf-8-sig')
    metrics.groupby(['strategy','model','h'])[['R2','RMSE','MAE']].agg(['mean','std']).to_csv(output/'seed_summary.csv',encoding='utf-8-sig')
    # 从落盘预测独立重算，并重建算术平均与元层预测。
    saved = pd.read_csv(output/'test_predictions_all_seeds.csv')
    for seed,f in saved.groupby('seed'):
        rebuilt = metrics_table(f.to_dict('records'))
        expected = metrics[metrics.seed==seed].drop(columns='seed')
        np.testing.assert_allclose(rebuilt[['R2','RMSE','MAE']],expected[['R2','RMSE','MAE']],rtol=1e-10,atol=1e-9)
        matrices = {family:f[f.model==family].pivot(index='sample_id',columns='h',values='pred').sort_index().to_numpy()
                    for family in ['ANN','RF','LSTM','Mean','Stacking','SelectedFusion']}
        np.testing.assert_allclose(matrices['Mean'],np.mean([matrices[k] for k in FAMILIES],axis=0),rtol=1e-12)
        meta = joblib.load(source/'meta_models'/f'multi_seed{seed}.joblib')
        meta_x = np.column_stack([matrices[k] for k in FAMILIES])
        np.testing.assert_allclose(matrices['SelectedFusion'],meta.predict(meta_x),rtol=1e-10,atol=1e-7)
        stacking = joblib.load(source/'meta_models'/f'stacking_seed{seed}.joblib')
        np.testing.assert_allclose(matrices['Stacking'],stacking.predict(meta_x),rtol=1e-10,atol=1e-7)
    if not smoke:
        old = pd.read_csv(ROOT/'output/acceptance3_runs/formal_20261031_v2/metrics_all_seeds.csv')
        comparison = metrics.merge(old,on=['seed','strategy','model','h'],suffixes=('_v3','_v2'),validate='one_to_one')
        comparison['RMSE_change_pct'] = (comparison.RMSE_v3/comparison.RMSE_v2-1)*100
        comparison['MAE_change_pct'] = (comparison.MAE_v3/comparison.MAE_v2-1)*100
        comparison.to_csv(output/'comparison_v2.csv',index=False,encoding='utf-8-sig')
    atomic_json(output/'verification.json',dict(status='passed',phase='multi_test_evaluation',
        test_samples=359,metric_rows=len(metrics),prediction_rows=len(saved),native_outputs=7,
        test_2004_predicted=True,multi_training_completed=True,all_tasks_third_round_completed=False,
        direct_policy='阶段B未准入，不扩网格；第二轮直接策略交付保持不变',
        meta_params=frozen['meta']['selected_params'],metrics_recomputed=True,mean_and_meta_reconstructed=True))
    print(metrics[metrics.seed==42].to_string(index=False),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    for name in [args.source_run,args.run_id]:
        if Path(name).name!=name or name in ['.','..']: parser.error('运行名必须为单目录名')
    source = ROOT/'output/acceptance3_runs'/args.source_run
    output = ROOT/'output/acceptance3_runs'/args.run_id
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    manifest = dict(run_id=args.run_id,source_run=args.source_run,phase='multi_frozen_evaluation',
        status='running',smoke=args.smoke,started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    atomic_json(output/'manifest.json',manifest)
    (output/'source').mkdir()
    import shutil
    shutil.copy2(Path(__file__),output/'source'/Path(__file__).name)
    torch.set_num_threads(4)
    try:
        with threadpool_limits(limits=4): evaluate(source,output,args.smoke)
        manifest['status']='completed'
    except Exception:
        manifest['status']='failed'
        raise
    finally:
        manifest['seconds']=time.perf_counter()-started
        manifest['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        atomic_json(output/'manifest.json',manifest)


if __name__=='__main__': main()
