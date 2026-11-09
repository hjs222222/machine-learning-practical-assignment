"""Explicitly authorized one-fit retrospective evaluation; never overwrite/restart."""
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'code'))
from config import save_json
from models import RunoffRegressor
from research_h1 import prepare_data, metrics

parent = Path(__file__).resolve().parent
out = parent / 'frozen_ann_batch62_2004_once'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
proposal_path = parent / 'candidate_freeze_audit/proposal.json'
proposal = read(proposal_path)
preflight = read(parent / 'proposed_evaluation_preflight.json')
assert sha(proposal_path) == preflight['proposal_sha256'] == '6d5662cb87b275ec86ebe26dfac92b8c7953640dedcde6d5bfe2c3217d8931ff'
assert proposal['test_authorized'] is False  # Preserve historical unauthorised proposal; new protocol grants this run only.
for name, value in preflight['source_hashes'].items():
    assert sha(root / name) == value
assert (platform.python_version(), np.__version__, pd.__version__, sklearn.__version__) == ('3.13.5','2.1.3','2.2.3','1.6.1')
candidate = proposal['candidate']
model = RunoffRegressor(family=candidate['family'], transform=candidate['transform'],
    residual=candidate['residual'], top_k=None, seed=42, **candidate['params'])
assert json.loads(json.dumps(model.get_params())) == proposal['full_current_constructor_parameters']
data_path = root / 'Teacher Resources & Requirements/01047000.csv'
assert sha(data_path) == preflight['data_sha256']
formal_path = root / 'output/formal_20261105_clean/predictions.csv'
formal_manifest = root / 'output/formal_20261105_clean/manifest.json'
assert sha(formal_manifest) == preflight['formal_manifest_sha256']
out.mkdir()  # Existing run, including failed one, is an error; no overwrite or hot start.
protocol = dict(status='frozen_before_fit_and_test_processing', test_authorized=True,
    authorization_date='2026-11-09', authorization_question_item='call_242a9d759b0d49f58bcc988c68873a82/0',
    authorization_answer='允许冻结一个候选后回溯一次2004',
    scope='One fixed ANN retrospective; retain all outcomes, no test-based tuning or formal report changes',
    candidate=candidate, full_constructor=proposal['full_current_constructor_parameters'],
    fixed_seed=42, planned_fits=1, train_start=preflight['train_start'],
    train_end=preflight['train_end'], train_target_end=preflight['train_target_end'], train_count=1395,
    origin_start='2004-01-01', origin_end='2004-12-24', target_start='2004-01-02', target_end='2004-12-25',
    test_count=359, horizon=1, available_weather=False, public_window=30, model_seq_len=1,
    peak_threshold_policy='Q95 of original training raw flow through 2003-12-24',
    proposal_sha256=sha(proposal_path), preflight_sha256=sha(parent/'proposed_evaluation_preflight.json'),
    data_sha256=sha(data_path), source_hashes=preflight['source_hashes'],
    launcher_sha256=sha(Path(__file__)), formal_predictions_sha256=sha(formal_path),
    formal_manifest_sha256=sha(formal_manifest),
    environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
    limitations=['Already exposed 2004; not blind', 'Candidate 1395 vs formal 1411 training origins',
                'Rolling next-day observed inputs; not year-ahead forecast', 'No configuration selection after this run'])
save_json(out/'protocol.json', protocol)
started = time.perf_counter()
try:
    raw_train = pd.read_csv(data_path, parse_dates=['Date']).set_index('Date').loc[:'2003-12-31'].copy()
    data = prepare_data(raw_train)
    train = np.flatnonzero(data['dates'] < pd.Timestamp('2004-01-01')-pd.Timedelta(days=7))
    X, y = data['weather']['X'][train], data['y'][train]
    assert len(train) == 1395
    assert hashlib.sha256(X.tobytes()).hexdigest() == preflight['train_X_sha256']
    assert hashlib.sha256(y.tobytes()).hexdigest() == preflight['train_y_sha256']
    save_json(out/'fit_started.json', dict(fit_number=1, planned_fits=1, protocol_sha256=sha(out/'protocol.json')))
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(X, y)
    finally:
        save_json(out/'warnings.json', [dict(category=w.category.__name__,message=str(w.message)) for w in caught])
    joblib.dump(model, out/'model.joblib')
    save_json(out/'fit.json', dict(fits=1, parameters=model.get_params(), n_iter=int(model.estimator_.n_iter_),
        loss_history=[float(x) for x in model.estimator_.loss_curve_], train_count=len(train),
        selected=model.selected_.tolist(), x_scaler_samples=int(model.x_scaler_.n_samples_seen_),
        model_sha256=sha(out/'model.joblib')))
    save_json(out/'test_started.json', dict(model_sha256=sha(out/'model.joblib'),
        protocol_sha256=sha(out/'protocol.json'), no_refit=True))
    raw = pd.read_csv(data_path, parse_dates=['Date']).set_index('Date')
    full = prepare_data(raw, cutoff='2004-12-31')
    ids = np.flatnonzero((full['dates'] >= protocol['origin_start']) & (full['dates'] <= protocol['origin_end']))
    dates = full['dates'][ids]
    np.testing.assert_array_equal(dates, pd.date_range(protocol['origin_start'], protocol['origin_end']))
    truth = full['y'][ids,0]
    prediction = model.predict(full['weather']['X'][ids])[:,0]
    persistence = raw.loc[dates, 'Discharge'].to_numpy()
    formal = pd.read_csv(formal_path, float_precision='round_trip')
    formal = formal.loc[(formal.strategy=='multi') & (formal.model=='RF') & (formal.h==1)].reset_index(drop=True)
    assert len(formal) == 359
    np.testing.assert_array_equal(pd.to_datetime(formal.origin), dates)
    np.testing.assert_array_equal(formal.truth, truth)
    threshold = float(np.quantile(raw_train.loc[:'2003-12-24','Discharge'], .95))
    frames, scores = [], []
    for name, pred in ((candidate['id'],prediction),('persistence',persistence),('formal_multi_RF_h1',formal.prediction.to_numpy())):
        assert np.isfinite(pred).all() and (pred >= 0).all()
        frames.append(pd.DataFrame(dict(candidate=name, origin=dates, target_date=dates+pd.Timedelta(days=1),
            truth=truth, prediction=pred, threshold=threshold, peak=truth>=threshold)))
        scores.append(dict(candidate=name, n=359, **metrics(truth,pred,threshold)))
    pd.concat(frames, ignore_index=True).to_csv(out/'predictions.csv', index=False)
    pd.DataFrame(scores).to_csv(out/'metrics.csv', index=False)
    save_json(out/'complete.json', dict(status='complete_pending_independent_verification',fits=1,
        candidate_test_predictions=359, total_records=1077, warnings=len(caught),
        seconds=time.perf_counter()-started, no_test_based_tuning=True))
    print(pd.DataFrame(scores).to_string(index=False))
    print(f'Exactly one fit; warnings={len(caught)}; no formal artifacts modified.')
except BaseException as error:
    save_json(out/'failure.json', dict(error_type=type(error).__name__,error=str(error),
        fit_started=(out/'fit_started.json').exists(),test_started=(out/'test_started.json').exists(),
        instruction='Retain failed run; do not rerun, change configuration or refit automatically'))
    raise
