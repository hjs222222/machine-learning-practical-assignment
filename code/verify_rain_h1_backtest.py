"""Recompute saved one-shot scores without fitting or predicting any model."""
import json

import numpy as np
import pandas as pd

from config import ROOT, DATA_PATH, digest, save_json


def main():
    parent = ROOT / 'output/teacher_selection_20261109/frozen_ann_rain3seed'
    output = parent / 'evaluation_2004_h1_once'
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    complete = read(output / 'complete.json')
    frozen = read(parent / 'frozen.json')
    authorization = read(output / 'authorization.json')
    started = read(output / 'started.json')
    protocol = read(parent / 'protocol.json')
    assert complete['status'] == 'complete'
    assert complete['fit_calls'] == 0 and complete['predict_calls'] == 3
    assert digest(parent / 'frozen.json') == complete['candidate_frozen_sha256']
    assert digest(parent / 'protocol.json') == frozen['protocol_sha256']
    assert digest(DATA_PATH) == complete['data_sha256'] == protocol['data_sha256']
    assert digest(output / 'authorization.json') == complete['authorization_sha256']
    assert authorization['granted_by_user'] is True
    assert authorization['candidate_frozen_sha256'] == complete['candidate_frozen_sha256']
    assert digest(ROOT / 'code/evaluate_frozen_rain_h1_once.py') == started['code_sha256']
    assert digest(output / 'predictions.csv') == complete['predictions_sha256']
    for entry in frozen['models']:
        assert digest(parent / entry['model_file']) == entry['model_sha256']
    for name, sha in protocol['code_hashes'].items():
        assert digest(ROOT / 'code' / name) == sha
    frame = pd.read_csv(output / 'predictions.csv', parse_dates=['origin', 'target_date'])
    origins = pd.date_range('2004-01-01', '2004-12-24')
    assert len(frame) == complete['test_origins'] == 359
    assert pd.DatetimeIndex(frame.origin).equals(origins)
    assert pd.DatetimeIndex(frame.target_date).equals(origins + pd.Timedelta(days=1))
    raw = pd.read_csv(DATA_PATH, parse_dates=['Date']).set_index('Date')
    truth = raw.Discharge.reindex(frame.target_date).to_numpy()
    np.testing.assert_array_equal(truth, frame.truth)
    components = frame[['seed42_h1', 'seed13_h1', 'seed2026_h1']].to_numpy()
    np.testing.assert_allclose(components.mean(axis=1), frame.prediction, rtol=1e-14)
    assert np.isfinite(components).all() and (components >= 0).all()
    errors = frame.prediction.to_numpy() - truth
    metrics = dict(R2=float(1 - (errors**2).sum() / ((truth-truth.mean())**2).sum()),
                   RMSE=float(np.sqrt((errors**2).mean())), MAE=float(np.abs(errors).mean()))
    saved = read(output / 'metrics.json')
    for name, value in metrics.items():
        np.testing.assert_allclose(value, saved[name], rtol=1e-12, atol=1e-12)
    result = dict(status='passed', rows=359, metrics=metrics, fit_calls=0, predict_calls=0,
                  checks='raw truth, dates, equal weights, metrics, data/model/code/authorization hashes',
                  verifier_sha256=digest(__file__))
    save_json(output / 'independent_verification.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
