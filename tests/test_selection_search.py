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


@pytest.mark.parametrize('family,outputs', [('RF',7),('ANN',1)])
def test_nested_search_is_training_only_and_reproducible(tmp_path, family, outputs):
    output = tmp_path/'run'
    extra = dict(leaf=[1]) if family == 'RF' else dict(alpha=[.1],iterations=[1])
    grid = dict(mi_policy=['mean','first'], top_k=[3], **extra)
    search.run(output, raw=raw_data(), grid=grid, trees=2, family=family, seed=13, outputs=outputs)
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['data_cutoff'] == '2003-12-31'
    assert protocol['family'] == family
    assert protocol['estimator']['family'] == family
    assert protocol['seed'] == protocol['estimator']['seed'] == 13
    assert protocol['outputs'] == outputs
    splits = pd.read_csv(output/'splits.csv')
    for row in splits.itertuples():
        assert pd.Timestamp(row.train_end)+pd.Timedelta(days=7) < pd.Timestamp(row.valid_start)
    metrics = pd.read_csv(output/'metrics.csv')
    assert len(metrics) == 6*outputs
    assert set(metrics.horizon) == set(range(1, outputs+1))
    assert set(metrics.model) == {'selected','baseline'}
    pred = pd.read_csv(output/'predictions.csv')
    assert len([c for c in pred if c.startswith('prediction_h')]) == outputs
    assert pd.to_datetime(pred.origin).max()+pd.Timedelta(days=7) < pd.Timestamp('2004-01-01')
    selected = json.loads((output/'selected.json').read_text())
    assert len(selected) == 3
    expected = [dict(mi_policy=p,top_k=3,**{k:v[0] for k,v in extra.items()}) for p in ['mean','first']]
    assert all(row['params'] in expected for row in selected)
    assert json.loads((output/'complete.json').read_text())['outer_fits'] == 6
    with pytest.raises(FileExistsError):
        search.run(output, raw=raw_data(), grid=grid, trees=2)


def test_unknown_family_rejected_before_creating_output(tmp_path):
    with pytest.raises(ValueError,match='family'):
        search.run(tmp_path/'bad', family='unknown')
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('outputs',[2,0,True])
def test_invalid_outputs_rejected(tmp_path, outputs):
    with pytest.raises(ValueError,match='outputs'):
        search.run(tmp_path/'bad', outputs=outputs)
    assert not (tmp_path/'bad').exists()
