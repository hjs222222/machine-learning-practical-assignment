import numpy as np
import pytest
import pandas as pd
from experiment import nested_oof_splits, frozen_grids, require_frozen


def test_nested_oof_never_sees_outer_validation():
    dates = pd.date_range('2000-02-14','2003-12-24')
    for tr, va, inner in nested_oof_splits(dates):
        assert dates[tr[-1]] + pd.Timedelta(days=7) < dates[va[0]]
        for it, iv in inner:
            assert set(tr[iv]).isdisjoint(va)
            assert dates[tr[it[-1]]] + pd.Timedelta(days=7) < dates[tr[iv[0]]]


def test_frozen_config_and_no_test_retry(tmp_path):
    with pytest.raises(ValueError):
        require_frozen(tmp_path)
    (tmp_path/'frozen.json').write_text('{}')
    require_frozen(tmp_path)
    (tmp_path/'test_started.json').write_text('{}')
    with pytest.raises(ValueError):
        require_frozen(tmp_path)


def test_grids_are_finite_and_have_real_choices():
    grids = frozen_grids()
    assert set(grids) == {'ANN','RF','LSTM'}
    assert all(2 <= len(g) <= 12 for g in grids.values())
    assert any(c['transform'] == ['raw'] for c in grids['RF'])
