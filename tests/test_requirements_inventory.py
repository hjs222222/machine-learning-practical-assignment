"""锁定教师指导书要求的模型口径：24 个个体模型 + 16 个集成模型。"""
import pytest

from experiment import (base_model_names, ensemble_model_names, horizon_groups,
                        ENSEMBLES, FAMILIES, STRATEGIES)


def test_model_counts_match_teacher_requirement():
    # 指导书四.2：3 方法 × (7 直接 + 1 多输出) = 24
    assert len(base_model_names()) == 24
    # 指导书四.3：2 集成方法 × (7 直接 + 1 多输出) = 16
    assert len(ensemble_model_names()) == 16


def test_names_are_unique_and_follow_naming():
    base = base_model_names()
    ensemble = ensemble_model_names()
    assert len(set(base)) == 24
    assert len(set(ensemble)) == 16
    assert not set(base) & set(ensemble)
    assert all(name.count('_') == 2 for name in base)


def test_direct_has_seven_models_multi_has_one():
    assert len(horizon_groups('direct')) == 7
    assert len(horizon_groups('multi')) == 1
    assert horizon_groups('multi')[0][1] == slice(0, 7)
    with pytest.raises(ValueError):
        horizon_groups('recursive')


def test_counts_are_derived_from_strategy_constants():
    # 换策略数或方法数时数量应随之变化，而不是写死在函数里
    per_family = sum(len(horizon_groups(s)) for s in STRATEGIES)   # 7 直接 + 1 多输出 = 8
    assert per_family == 8
    assert len(base_model_names()) == len(FAMILIES) * per_family
    assert len(ensemble_model_names()) == len(ENSEMBLES) * per_family
