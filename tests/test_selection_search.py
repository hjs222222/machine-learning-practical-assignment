import json
import numpy as np
import pandas as pd
import pytest
import research_selection_search as search
from test_weather_target_mi import raw_data


def test_h1_score_does_not_average_other_outputs():
    class Estimator:
        def predict(self, x):
            return np.column_stack([np.arange(10), np.zeros(10)])
    truth = np.column_stack([np.arange(10), np.arange(10)+100])
    assert search.h1_score(Estimator(), None, truth) == 1


def test_nested_search_is_training_only_and_reproducible(tmp_path):
    output = tmp_path/'run'
    grid = dict(mi_policy=['mean','first'], top_k=[3], leaf=[1])
    search.run(output, raw=raw_data(), grid=grid, trees=2)
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['data_cutoff'] == '2003-12-31'
    splits = pd.read_csv(output/'splits.csv')
    for row in splits.itertuples():
        assert pd.Timestamp(row.train_end)+pd.Timedelta(days=7) < pd.Timestamp(row.valid_start)
    metrics = pd.read_csv(output/'metrics.csv')
    assert len(metrics) == 42
    assert set(metrics.model) == {'selected','baseline'}
    pred = pd.read_csv(output/'predictions.csv')
    assert pd.to_datetime(pred.origin).max()+pd.Timedelta(days=7) < pd.Timestamp('2004-01-01')
    selected = json.loads((output/'selected.json').read_text())
    assert len(selected) == 3
    assert all(row['params'] in [dict(mi_policy=p,top_k=3,leaf=1) for p in ['mean','first']] for row in selected)
    assert json.loads((output/'complete.json').read_text())['outer_fits'] == 6
    with pytest.raises(FileExistsError):
        search.run(output, raw=raw_data(), grid=grid, trees=2)
