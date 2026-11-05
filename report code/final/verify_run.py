"""2026-11-05 第1次修改版：只读复算正式文件，断言同源、边界和指标。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from config import digest,DATA_PATH,save_json


def verify(run):
    run=Path(run)
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    frozen=json.loads((run/'frozen.json').read_text(encoding='utf-8'))
    assert manifest['status']=='completed' and len(frozen['selected'])==24
    assert manifest['data_sha256']==frozen['data_sha256']==digest(DATA_PATH)
    assert len(frozen['model_hashes'])==32
    for name,sha in frozen['model_hashes'].items():assert digest(run/'models'/name)==sha,name
    for name,sha in frozen['source_hashes'].items():assert digest(run/'source'/name)==sha,name
    scores=pd.read_csv(run/'metrics.csv')
    daily=pd.read_csv(run/'predictions.csv',parse_dates=['origin','target_date'])
    assert len(scores)==70 and len(daily)==25130
    assert not daily.duplicated(['strategy','model','h','origin']).any()
    assert set(scores.model)=={'ANN','RF','LSTM','Mean','Stacking'}
    assert set(scores.strategy)=={'direct','multi'}
    origins=pd.date_range('2004-01-01','2004-12-24')
    for row in scores.itertuples():
        d=daily[(daily.strategy==row.strategy)&(daily.model==row.model)&(daily.h==row.h)]
        assert pd.DatetimeIndex(d.origin).equals(origins)
        assert ((d.target_date-d.origin).dt.days==row.h).all()
        assert d.target_date.max()<=pd.Timestamp('2004-12-31')
        actual=[r2_score(d.truth,d.prediction),np.sqrt(mean_squared_error(d.truth,d.prediction)),mean_absolute_error(d.truth,d.prediction)]
        np.testing.assert_allclose(actual,[row.R2,row.RMSE,row.MAE],rtol=1e-9,atol=1e-8)
    arrays=np.load(run/'predictions.npz')
    for strategy in ('direct','multi'):
        np.testing.assert_allclose(arrays[f'{strategy}_Mean'],np.mean([arrays[f'{strategy}_{f}'] for f in ('RF','ANN','LSTM')],axis=0))
    folds=pd.read_csv(run/'oof'/'splits.csv')
    assert (pd.to_datetime(folds.train_target_end)<pd.to_datetime(folds.validation_start)).all()
    assert (pd.to_datetime(folds.validation_origin_end)<=pd.Timestamp('2003-12-24')).all()
    cv=list((run/'cv').glob('*.json'))
    assert len(cv)==96
    warnings=sum(json.loads(p.read_text(encoding='utf-8'))['convergence_warnings'] for p in cv)
    result=dict(status='passed',base_models=24,ensemble_models=16,metrics_recomputed=70,daily_predictions=25130,
                test_origins=359,independent_nested_searches=96,formal_convergence_warnings=warnings,
                note='仅复算冻结结果文件，没有重新预测、训练或按测试选参')
    save_json(run/'verification.json',result)
    return result


if __name__=='__main__':print(verify(Path(sys.argv[1])))
