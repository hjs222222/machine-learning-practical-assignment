import numpy as np
import pytest
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_regression
from models import RunoffRegressor


@pytest.mark.parametrize('policy', ['mean', 'intersection', 'union', 'first'])
@pytest.mark.parametrize('outputs', [1, 7])
def test_policy_matches_training_only_scores(policy, outputs):
    rng = np.random.RandomState(16)
    x = rng.uniform(1, 10, (100, 1, 5)).astype('float32')
    y = np.column_stack([x[:, 0, h % 5] ** 2 for h in range(outputs)])
    scores = np.array([mutual_info_regression(x[:, -1], y[:, h], random_state=42)
                       for h in range(outputs)])
    sets = [set(np.argsort(s, kind='stable')[-3:]) for s in scores]
    expected = {'mean': set(np.argsort(scores.mean(0), kind='stable')[-3:]),
                'intersection': set.intersection(*sets), 'union': set.union(*sets),
                'first': sets[0]}[policy] | {0}
    model = clone(RunoffRegressor(trees=2, top_k=3, mi_policy=policy, forced_indices=(0,)))
    model.fit(x, y)
    np.testing.assert_array_equal(model.selected_, sorted(expected))
    np.testing.assert_array_equal(model.mi_, scores.mean(0))
    assert model.predict(x[:3]).shape == (3, outputs)


@pytest.mark.parametrize('policy', ['bad', None, 3])
def test_invalid_policy(policy):
    with pytest.raises(ValueError, match='mi_policy'):
        RunoffRegressor(mi_policy=policy).fit(np.ones((10, 1, 2)), np.ones((10, 1)))


def test_empty_intersection_is_not_silently_replaced():
    rng = np.random.RandomState(6)
    x = rng.uniform(1, 10, (200, 1, 2)).astype('float32')
    with pytest.raises(ValueError, match='交集为空'):
        RunoffRegressor(mi_policy='intersection', top_k=1).fit(x, x[:, 0])


def test_research_records_explicit_selection(tmp_path):
    import json
    import research_h1 as rh
    from test_weather_target_mi import raw_data
    candidate = dict(id='selection', family='RF', mode='mi', outputs=7,
                     transform='sqrt', residual=True, top_k=3,
                     params=dict(trees=2, mi_policy='first', seq_len=1))
    rh.run_stage('features', tmp_path/'run', raw=raw_data(), candidates=[candidate])
    fits = json.loads((tmp_path/'run'/'fits.json').read_text())
    assert len(fits) == 3
    for fit in fits:
        assert fit['mi_policy'] == 'first'
        assert fit['mi_top_k'] == 3
        assert len(fit['selected_indices']) == 3
