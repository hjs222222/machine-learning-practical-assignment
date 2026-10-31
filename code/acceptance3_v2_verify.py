# -*- coding: utf-8 -*-
"""第二轮独立核验：预测、时间、原生多输出、元层与保存恢复。"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch

from acceptance3_train import ROOT, dump
from acceptance3_model import restore_model
from acceptance3_v2_train import meta_predict


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    run=ROOT/'output/acceptance3_runs'/args.run_id
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='completed'
    frozen=json.loads((run/'selection_frozen.json').read_text(encoding='utf-8'))
    assert frozen['status']=='frozen_before_test'
    seeds=[42] if manifest['smoke'] else [42,43,44]
    p=pd.read_csv(run/'test_predictions_all_seeds.csv',parse_dates=['origin_date','target_date'])
    o=pd.read_csv(run/'oof_predictions_all_seeds.csv',parse_dates=['origin_date','target_date'])
    m=pd.read_csv(run/'metrics_all_seeds.csv')
    assert len(p)==25130*len(seeds) and len(o)==30366*len(seeds)
    assert len(m)==70*len(seeds)
    assert set(p.seed)==set(seeds)
    assert not p.duplicated(['seed','strategy','model','h','origin_date']).any()
    assert not o.duplicated(['seed','strategy','model','h','origin_date']).any()
    assert p.target_date.dt.year.eq(2004).all()
    assert (p.target_date-p.origin_date).dt.days.eq(p.h).all()
    aligned={}
    for key,f in p.groupby(['seed','strategy','model','h']):
        seed,strategy,model,h=key
        assert len(f)==359
        values=[r2_score(f.truth,f.pred),np.sqrt(np.mean((f.truth-f.pred)**2)),np.mean(np.abs(f.truth-f.pred))]
        row=m[(m.seed==seed)&(m.strategy==strategy)&(m.model==model)&(m.h==h)].iloc[0]
        np.testing.assert_allclose(values,row[['R2','RMSE','MAE']].to_numpy(float),rtol=1e-10,atol=1e-7)
        truth=f.sort_values('origin_date')[['origin_date','target_date','truth']].reset_index(drop=True)
        if h in aligned:
            pd.testing.assert_frame_equal(truth,aligned[h])
        aligned[h]=truth
    with np.load(run/'dataset.npz',allow_pickle=False) as data:
        days=data['origin_days']
        final_ids=np.flatnonzero(days+7<=np.datetime64('2003-12-31').astype('int64'))
        for file in (run/'splits').glob('*.csv'):
            f=pd.read_csv(file)
            assert (pd.to_datetime(f.max_train_target)<pd.to_datetime(f.validation_origin_start)).all()
            assert f.n_train.ge(478).all()
        outer=pd.read_csv(run/'outer_splits.csv')
        assert (pd.to_datetime(outer.max_train_target)<pd.to_datetime(outer.validation_origin_start)).all()
        for file in (run/'training_history').glob('*.json'):
            r=json.loads(file.read_text(encoding='utf-8'))
            assert days[r['last_id']]+7<=np.datetime64('2003-12-31').astype('int64')
            if 'cv_score' in r:
                assert days[r['last_id']]+7<days[r['validation_first_id']]
            if 'early_validation_first_id' in r:
                assert days[r['early_train_last_id']]+7<days[r['early_validation_first_id']]
                assert all(np.isfinite(x['val_RMSE']) for x in r['early_history'])
                assert r['selected_epochs']<= (2 if manifest['smoke'] else 150)
            if r['family']=='LSTM':
                assert len(r['history'])==r['selected_epochs']
        metas=pd.read_csv(run/'meta_validation.csv')
        assert (pd.to_datetime(metas.max_train_target)<pd.to_datetime(metas.validation_origin_start)).all()
        restoration=[]
        for rec in frozen['base_selections']:
            if rec['stage']!='final':
                continue
            for seed in seeds:
                name=f"final_{rec['strategy']}_{rec['model']}_h{rec['h']}_seed{seed}"
                model=restore_model(run/'models'/name,device='cpu')
                ids=p[(p.seed==seed)&(p.strategy==rec['strategy'])&(p.model==rec['model'])].sample_id.unique()
                actual=model.predict(ids[:,None])
                expected=p[(p.seed==seed)&(p.strategy==rec['strategy'])&(p.model==rec['model'])].pivot(index='sample_id',columns='h',values='pred').sort_index()
                if rec['strategy']=='direct':
                    expected=expected[[rec['h']]]
                np.testing.assert_allclose(np.asarray(actual).reshape(len(ids),-1),expected.to_numpy(),rtol=1e-6,atol=1e-3)
                if rec['strategy']=='multi':
                    assert model.output_dim_==7
                    if rec['model']=='LSTM': assert model.model_.head[-1].out_features==7
                    if rec['model']=='RF': assert model.model_.n_outputs_==7
                restoration.append(name)
    for seed in seeds:
        for strategy in ['direct','multi']:
            f=p[(p.seed==seed)&(p.strategy==strategy)]
            matrices=[f[f.model==fam].pivot(index='sample_id',columns='h',values='pred').sort_index().to_numpy() for fam in ['ANN','RF','LSTM']]
            mean=f[f.model=='Mean'].pivot(index='sample_id',columns='h',values='pred').sort_index().to_numpy()
            np.testing.assert_allclose(mean,np.mean(matrices,axis=0),rtol=1e-10)
            x=np.column_stack(matrices)
            expected=f[f.model=='Stacking'].pivot(index='sample_id',columns='h',values='pred').sort_index().to_numpy()
            for rec in frozen['meta_selections']:
                if rec['seed']!=seed or rec['strategy']!=strategy: continue
                model=joblib.load(run/'models'/f"{rec['label']}.joblib")
                actual=meta_predict(model,x[:,rec['input_cols']],rec['kind'])
                target=expected[:,rec['h']-1] if strategy=='direct' else expected
                np.testing.assert_allclose(actual,target,rtol=1e-9,atol=1e-6)
                if strategy=='multi': assert model[-1].coef_.shape==(7,21)
    cv=pd.read_csv(run/'all_cv_candidates.csv')
    assert len(cv)==(104 if manifest['smoke'] else 880),len(cv)
    # 4阶段×(2策略pilot×6候选+8任务×(LSTM12+ANN6+RF8))=880候选。
    assert cv.filter(regex=r'split\d+_test_score').notna().all().all()
    assert len(frozen['base_selections'])==96
    assert len(restoration)==24*len(seeds)
    fits=list((run/'training_history').glob('*.json'))
    assert len(fits)>=(304 if manifest['smoke'] else 2048),len(fits)
    result=dict(run_id=args.run_id,status='passed',seeds=seeds,metric_rows=len(m),
                required_rows_per_seed=42,test_prediction_rows=len(p),oof_prediction_rows=len(o),
                cv_candidate_rows=len(cv),fit_records=len(fits),final_restored_models=len(restoration),
                checks=['逐样本重算指标','共同真值和目标日期','外层和内层时间净化','内部早停原尺度监控',
                        '全部最终模型恢复并重建359起报点预测','原尺度简单平均','落盘元层重建预测',
                        '原生多输出结构','完整候选表','冻结记录存在'])
    dump(run/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
