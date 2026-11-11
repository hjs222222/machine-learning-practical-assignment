"""Fixtures only; the real single backtest is not repeatable."""
import json

import numpy as np
import pandas as pd
import pytest

from test_weather_target_mi import raw_data
from backtest_h1_two_stage_2004 import frozen_params, run


def test_frozen_params_majority_and_tiebreak(tmp_path):
    rows = [dict(candidate='TwoStage', year=2001, params="{'leaf': 5, 'top_k': 24}"),
            dict(candidate='TwoStage', year=2002, params="{'leaf': 5, 'top_k': 36}"),
            dict(candidate='TwoStage', year=2003, params="{'leaf': 5, 'top_k': 36}")]
    pd.DataFrame(rows).to_csv(tmp_path / 'selected.csv', index=False)
    params = frozen_params(tmp_path)
    assert params['top_k'] == 36 and params['leaf'] == 5 and params['trees'] == 120
    assert params['votes']['top_k=36,leaf=5'] == [2002, 2003]


def test_backtest_freezes_then_evaluates_once(tmp_path):
    raw = raw_data().copy()
    raw['Prcp'] = np.where(np.arange(len(raw)) % 3 == 0, 0.0, 8.0)
    out = tmp_path / 'bt'
    result = run(out, raw=raw, params=dict(top_k=12, leaf=5, trees=2, seed=42))
    freeze = json.loads((out / 'freeze.json').read_text(encoding='utf-8'))
    assert freeze['stage'] == 'frozen_before_test_labels'
    assert freeze['train_label_end'] <= '2003-12-31'
    assert (out / 'model.joblib').is_file()
    preds = pd.read_csv(out / 'predictions.csv')
    assert len(preds) == 359
    assert preds.origin.min() == '2004-01-01' and preds.origin.max() == '2004-12-24'
    assert preds.target_date.max() == '2004-12-25'
    assert result['evaluations'] == 1 and result['origins'] == 359
    assert 0 <= result['r2'] <= 1
    assert result['dry_days']['count'] + result['wet_days']['count'] == 359
    with pytest.raises(FileExistsError):
        run(out, raw=raw, params=dict(top_k=12, leaf=5, trees=2, seed=42))
