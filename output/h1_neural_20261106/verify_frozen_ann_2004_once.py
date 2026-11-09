"""Independent persisted score/scaler/NumPy-network verification; no fit/predict."""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'code'))  # Only needed to unpickle this run's own model class.
parent = Path(__file__).resolve().parent
run = parent/'frozen_ann_batch62_2004_once'
dest = run/'verification.json'
assert not dest.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
protocol, fit, complete = (read(run/n) for n in ('protocol.json','fit.json','complete.json'))
assert protocol['test_authorized'] is True and protocol['planned_fits'] == fit['fits'] == complete['fits'] == 1
assert protocol['authorization_answer'] == '允许冻结一个候选后回溯一次2004'
assert sha(parent/'run_frozen_ann_2004_once.py') == protocol['launcher_sha256']
for name, value in protocol['source_hashes'].items(): assert sha(root/name) == value
assert sha(parent/'candidate_freeze_audit/proposal.json') == protocol['proposal_sha256']
assert sha(parent/'proposed_evaluation_preflight.json') == protocol['preflight_sha256']
assert sha(run/'model.joblib') == fit['model_sha256'] == read(run/'test_started.json')['model_sha256']
assert read(run/'fit_started.json')['protocol_sha256'] == sha(run/'protocol.json')
assert read(run/'test_started.json')['protocol_sha256'] == sha(run/'protocol.json')
raw_path = root/'Teacher Resources & Requirements/01047000.csv'
assert sha(raw_path) == protocol['data_sha256']
raw = pd.read_csv(raw_path,parse_dates=['Date']).set_index('Date')
train_dates = pd.date_range('2000-02-29','2003-12-24')
dates = pd.date_range('2004-01-01','2004-12-24')
assert len(train_dates) == 1395 and len(dates) == 359
q95 = np.quantile(raw.loc[:'2003-12-24','Discharge'],.95)
predictions = pd.read_csv(run/'predictions.csv',float_precision='round_trip')
scores = pd.read_csv(run/'metrics.csv',float_precision='round_trip').set_index('candidate')
assert len(predictions) == 1077 and len(scores) == 3
assert not predictions.duplicated(['candidate','origin']).any()
cid = protocol['candidate']['id']
groups = {}
for name, frame in predictions.groupby('candidate',sort=False):
    assert len(frame) == 359
    np.testing.assert_array_equal(pd.to_datetime(frame.origin),dates)
    np.testing.assert_array_equal(pd.to_datetime(frame.target_date),dates+pd.Timedelta(days=1))
    y = raw.loc[dates+pd.Timedelta(days=1),'Discharge'].to_numpy()
    np.testing.assert_array_equal(frame.truth,y)
    np.testing.assert_allclose(frame.threshold,q95,rtol=0,atol=1e-10)
    np.testing.assert_array_equal(frame.peak,y>=q95)
    error = frame.prediction.to_numpy()-y
    peak = error[y>=q95]
    expected = dict(r2=1-np.sum(error**2)/np.sum((y-y.mean())**2),rmse=np.sqrt(np.mean(error**2)),
        mae=np.mean(abs(error)),peak_rmse=np.sqrt(np.mean(peak**2)),peak_bias=peak.mean(),peak_count=len(peak),n=359)
    for key,value in expected.items(): np.testing.assert_allclose(value,scores.loc[name,key],rtol=1e-12,atol=1e-8)
    groups[name] = frame
np.testing.assert_array_equal(groups['persistence'].prediction, raw.loc[dates,'Discharge'])
formal_path = root/'output/formal_20261105_clean/predictions.csv'
assert sha(formal_path) == protocol['formal_predictions_sha256']
formal = pd.read_csv(formal_path,float_precision='round_trip')
formal = formal.loc[(formal.strategy=='multi')&(formal.model=='RF')&(formal.h==1)]
np.testing.assert_array_equal(groups['formal_multi_RF_h1'].prediction,formal.prediction)
assert sha(root/'output/formal_20261105_clean/manifest.json') == protocol['formal_manifest_sha256']

# Rebuild weather15 directly from raw fields and nonnegative lag differences, not research_h1.
weather = raw[['Discharge','Dayl','Prcp','Srad','Tmax','Tmin','Vp']].copy()
for col in ('Vp','Srad'):
    for lag in (1,3): weather[f'{col}_diff{lag}'] = raw[col]-raw[col].shift(lag)
tavg = (raw.Tmax+raw.Tmin)/2
for lag in (1,3): weather[f'Tavg_diff{lag}'] = tavg-tavg.shift(lag)
weather['Trange_lag0'] = raw.Tmax-raw.Tmin
weather['Trange_diff1'] = weather.Trange_lag0-weather.Trange_lag0.shift(1)
model = joblib.load(run/'model.joblib')
assert json.loads(json.dumps(model.get_params())) == protocol['full_constructor'] == fit['parameters']
np.testing.assert_array_equal(model.selected_,np.arange(15))
assert model.seq_len == 1 and not model.ann_history and model.outputs_ == 1
assert model.x_scaler_.n_samples_seen_ == model.y_scaler_.n_samples_seen_ == 1395
train_x = weather.loc[train_dates].to_numpy(np.float32)
np.testing.assert_allclose(model.x_scaler_.mean_,train_x.astype(float).mean(axis=0),rtol=1e-12,atol=1e-10)
np.testing.assert_allclose(model.x_scaler_.var_,train_x.astype(float).var(axis=0),rtol=1e-12,atol=1e-8)
target = np.sqrt(raw.loc[train_dates+pd.Timedelta(days=1),'Discharge'].to_numpy()[:,None])-np.sqrt(train_x[:,:1])
np.testing.assert_allclose(model.y_scaler_.mean_,target.mean(axis=0),rtol=1e-12,atol=1e-10)
np.testing.assert_allclose(model.y_scaler_.var_,target.var(axis=0),rtol=1e-12,atol=1e-10)
x = weather.loc[dates].to_numpy(np.float32)
current = x[:,:1].copy()
x -= model.x_scaler_.mean_
x /= model.x_scaler_.scale_
hidden = np.maximum(x@model.estimator_.coefs_[0]+model.estimator_.intercepts_[0],0)
out = hidden@model.estimator_.coefs_[1]+model.estimator_.intercepts_[1]
out *= model.y_scaler_.scale_
out += model.y_scaler_.mean_
out += np.sqrt(current)
manual = np.maximum(out,0)**2
np.testing.assert_array_equal(manual[:,0],groups[cid].prediction)
assert fit['n_iter'] == model.estimator_.n_iter_ == len(fit['loss_history']) == 350
np.testing.assert_array_equal(fit['loss_history'],model.estimator_.loss_curve_)
caught = read(run/'warnings.json')
assert len(caught) == complete['warnings'] == 1 and caught[0]['category'] == 'ConvergenceWarning'
delivery = read(root/'最终提交/交付核验.json')
assert sha(root/'最终提交/机器学习课程实践实验报告.docx') == delivery['report_sha256']
assert sha(root/'最终提交/机器学习课程实践代码.zip') == delivery['code_zip_sha256']
report = dict(status='passed',fits=1,test_origins=359,persisted_records=1077,actual_iterations=350,warnings=1,
    model_manual_forward_exact=True,no_refit_or_predict_during_verification=True,
    candidate_r2=float(scores.loc[cid,'r2']),formal_rf_r2=float(scores.loc['formal_multi_RF_h1','r2']),
    persistence_r2=float(scores.loc['persistence','r2']),
    checks=['raw dates/truth/Q95 and all metrics','authorized frozen protocol/source/data/model hashes',
        'raw weather15 lag formula; train-only scaler statistics','exact NumPy two-layer forward and sqrt-residual decode',
        'full constructor, selected columns, loss curve and actual warning', 'formal predictions/manifest and delivered Word/ZIP unchanged'],
    verifier_sha256=sha(Path(__file__)),
    artifact_hashes={n:sha(run/n) for n in ('protocol.json','fit.json','predictions.csv','metrics.csv','warnings.json','complete.json','model.joblib')},
    limitations=['No independent second fit; once-only retrospective not blind','Does not promote candidate or establish 0.8; no further test tuning'])
dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
