"""独立只读核验策略增强run：冻结选择、历史边界、全部恢复与逐h指标。"""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from runoff_training.controlled_data import build_data, load_data, outer_split, inner_splits, sha256
from runoff_training.strategy_model import restore_strategy

ROOT=Path(__file__).resolve().parents[1]


def recompute(truth,pred):
    error=np.asarray(pred)-np.asarray(truth)
    return dict(n=len(error),RMSE=float(np.sqrt(np.square(error).mean())),
        MAE=float(np.abs(error).mean()),R2=float(1-np.square(error).sum()/np.square(truth-np.mean(truth)).sum()),
        bias=float(error.mean()),negative_predictions=int((np.asarray(pred)<0).sum()))


def verify(run,allow_smoke=False):
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    spec=manifest['protocol']
    if manifest['status']!='completed' or spec['smoke'] and not allow_smoke:
        raise ValueError('只接受完成的正式run；smoke须显式允许')
    for rel,expected in manifest['frozen_files'].items():
        if sha256(run/rel)!=expected:
            raise ValueError(f'冻结文件改变：{rel}')
    for rel,expected in manifest['source_hashes'].items():
        if sha256(run/'source'/Path(rel).name)!=expected:
            raise ValueError('源码快照hash不一致')
    if not manifest['selection_frozen_before_outer'] or sha256(run/'selection.json')!=manifest['selection_sha256']:
        raise ValueError('选择冻结记录不符')
    raw_path=ROOT/'Teacher Resources & Requirements/01047000.csv'
    if sha256(raw_path)!=manifest['data']['raw_sha256'] or sha256(run/'dataset.npz')!=manifest['data']['dataset_sha256']:
        raise ValueError('数据hash改变')
    data=load_data(run/'dataset.npz')
    assert data['days'][-1]==np.datetime64('2003-12-31').astype('int64')
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'data.npz'
        build_data(raw_path,path)
        rebuilt=load_data(path)
        assert set(rebuilt)==set(data)
        for key in data:
            np.testing.assert_equal(data[key],rebuilt[key])
    raw=pd.read_csv(raw_path).set_index('Date')
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    frame=pd.read_csv(run/'metrics.csv')
    records=selection['records']
    assert len(records)==spec['outer_models']==manifest['outer_models']
    expected_keys=[]
    for year in spec['years']:
        for strategy,h in spec['tasks']:
            for variant in spec['variants'][f'{strategy}_h{h}']:
                for seed in spec['seeds']:
                    expected_keys.append((year,strategy,h,variant['name'],seed))
    keys=[(r['year'],r['strategy'],r['task_h'],r['variant']['name'],r['seed']) for r in records]
    assert len(set(keys))==len(keys) and set(keys)==set(expected_keys)
    histories={p.name:json.loads(p.read_text(encoding='utf-8')) for p in (run/'history').glob('*.json')}
    assert len(histories)==spec['fits']==manifest['fits']
    used=set()
    predictions=[]
    n_metrics=0
    for record in records:
        year,strategy,h,seed=[record[k] for k in ['year','strategy','task_h','seed']]
        variant,tag=record['variant'],record['tag']
        train,valid=outer_split(data,year)
        folds=inner_splits(data,train)
        assert data['origin_days'][train[-1]]+7<data['origin_days'][valid[0]]
        cv=pd.read_csv(run/'cv'/f'{tag}.csv')
        epochs=[json.loads(v)['fixed_epochs'] for v in cv.params]
        assert epochs==spec['epochs']
        fold_scores=np.column_stack([-cv[f'split{k}_test_score'].to_numpy() for k in range(3)])
        scores=fold_scores.mean(axis=1)
        np.testing.assert_allclose(scores,-cv.mean_test_score.to_numpy(),rtol=1e-12)
        eligible=[i for i,v in enumerate(scores) if v<=scores.min()*1.01]
        index=min(eligible,key=lambda i:epochs[i])
        assert record['epochs']==epochs[index]
        np.testing.assert_allclose(record['inner_RMSE'],scores[index],rtol=1e-12)
        np.testing.assert_allclose(record['inner_fold_RMSE'],fold_scores[index],rtol=1e-12)
        own_hist={name:a for name,a in histories.items() if a['phase']==tag+'_inner'}
        assert len(own_hist)==len(epochs)*3
        expected=[]
        for epoch in epochs:
            for ti,vi in folds:
                ids=train[ti]
                assert data['origin_days'][ids[-1]]+7<data['origin_days'][train[vi[0]]]
                expected.append((epoch,len(ids),int(ids[0]),int(ids[-1])))
        actual=[]
        for name,audit in own_hist.items():
            actual.append((audit['params']['fixed_epochs'],audit['n_train'],audit['first_id'],audit['last_id']))
            assert len(audit['history'])==audit['params']['fixed_epochs']
            assert audit['history'][-1]['gradient_steps']==len(audit['history'])*int(np.ceil(audit['n_train']/64))
            used.add(name)
        assert sorted(expected)==sorted(actual)
        model=restore_strategy(run/'models'/tag,run/'dataset.npz',spec['device'])
        audit=model.audit_
        assert (audit['n_train'],audit['first_id'],audit['last_id'])==(len(train),int(train[0]),int(train[-1]))
        assert audit['params']['fixed_epochs']==record['epochs'] and len(audit['history'])==record['epochs']
        for k in ['pooling','residual','target_transform']:
            assert audit['params'][k]==variant[k]
        assert audit['train_last_h7_target']<str(np.datetime64(int(data['origin_days'][valid[0]]),'D'))
        assert audit['scaler_last_day']==audit['train_last_origin']
        assert audit['history'][-1]['gradient_steps']==record['epochs']*int(np.ceil(len(train)/64))
        outer_hist=[(name,a) for name,a in histories.items() if a['phase']==tag+'_outer']
        assert len(outer_hist)==1 and outer_hist[0][1]==audit
        used.add(outer_hist[0][0])
        pf=pd.read_csv(run/'predictions'/f'{tag}.csv')
        assert not pf.duplicated(['sample_id','h']).any()
        own=pf.pivot(index='sample_id',columns='h',values='pred').sort_index()
        np.testing.assert_equal(own.index.to_numpy(),valid)
        assert own.columns.tolist()==([h] if strategy=='direct' else list(range(1,8)))
        np.testing.assert_allclose(own.to_numpy(),model.predict(valid[:,None]),rtol=1e-7,atol=1e-5)
        for row in pf.itertuples():
            day=np.datetime64(int(data['origin_days'][row.sample_id]),'D')
            assert row.origin_date==str(day) and row.target_date==str(day+np.timedelta64(row.h,'D'))
            np.testing.assert_allclose(row.truth,raw.loc[row.target_date,'Discharge'],rtol=0,atol=1e-12)
        for horizon in own.columns:
            values=recompute(data['y'][valid,horizon-1],own[horizon].to_numpy())
            found=frame[(frame.year==year)&(frame.strategy==strategy)&(frame.task_h==h)&
                (frame.variant==variant['name'])&(frame.seed==seed)&(frame.h==horizon)]
            assert len(found)==1
            for k,v in values.items():
                np.testing.assert_allclose(found.iloc[0][k],v,rtol=1e-10,atol=1e-7)
            n_metrics+=1
        predictions.append(pf)
    assert used==set(histories) and n_metrics==len(frame)
    assert len(selection['selections'])==len(spec['years'])*len(spec['tasks'])
    for item in selection['selections']:
        own=[r for r in records if (r['year'],r['strategy'],r['task_h'])==
             (item['year'],item['strategy'],item['task_h'])]
        order=spec['variants'][f'{item["strategy"]}_h{item["task_h"]}']
        scores={v['name']:np.mean([r['inner_RMSE'] for r in own if r['variant']['name']==v['name']]) for v in order}
        for key,val in scores.items():
            np.testing.assert_allclose(item['inner_scores'][key],val,rtol=1e-12)
        expected=next(name for name,val in scores.items() if val<=min(scores.values())*1.01)
        assert item['selected_variant']==expected
    result=dict(status='passed',smoke=spec['smoke'],models=len(records),fits=len(histories),metrics=n_metrics,
        prediction_rows=sum(len(p) for p in predictions),frozen_files=len(manifest['frozen_files']),
        selection_frozen_before_outer=True,max_target='2003-12-31',
        checks=['源码/工件hash','按日期重建真值','全部特征重建','内外层净化','每次fit证据',
                '内层epoch/策略选择','全部模型恢复','逐h指标重建'])
    return result,pd.concat(predictions,ignore_index=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--allow-smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    with threadpool_limits(limits=4):
        result,_=verify(ROOT/'output/acceptance3_runs'/args.run_id,args.allow_smoke)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
