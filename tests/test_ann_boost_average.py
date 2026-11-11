"""Small software fixture only; no scientific observations added."""
import numpy as np
import pandas as pd
import pytest


def test_fixed_mean_and_alignment():
    from research_ann_boost_average import combine
    a = pd.DataFrame(dict(year=[2001, 2001], origin=pd.to_datetime(['2001-01-01', '2001-01-02']),
        target_date=pd.to_datetime(['2001-01-02', '2001-01-03']), truth=[3., 4.], prediction=[2., 6.]))
    b = a.copy()
    b['prediction'] = [4., 2.]
    result = combine(a, b)
    np.testing.assert_array_equal(result['mean'], [3., 4.])
    for column, value in [('truth', 99.), ('origin', pd.Timestamp('2002-01-01')),
                          ('prediction', np.nan), ('prediction', -1.)]:
        bad = b.copy()
        bad.loc[0, column] = value
        with pytest.raises(ValueError):
            combine(a, bad)
    with pytest.raises(ValueError):
        combine(a, b.iloc[::-1])
    with pytest.raises(ValueError):
        combine(a, pd.concat([b, b.iloc[:1]]))
