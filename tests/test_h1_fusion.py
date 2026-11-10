import numpy as np
import pandas as pd
import pytest
from research_h1_fusion import assemble, meta_split


def example():
    dates = pd.DatetimeIndex([d for y in (2001,2002,2003)
                             for d in pd.date_range(f'{y}-01-01',f'{y}-12-24')])
    raw = pd.DataFrame({'Discharge':np.arange(1461)+100.},
                       index=pd.date_range('2000-01-01','2003-12-31'))
    frames = [pd.DataFrame({'origin':dates,'prediction':100.+np.arange(len(dates))+i}) for i in range(3)]
    return raw,frames


def test_assembly_and_causal_splits():
    raw,frames=example();data=assemble(raw,frames)
    assert list(data.columns)==['origin','ANN','RF','LSTM','truth']
    np.testing.assert_array_equal(data.truth,raw.Discharge.reindex(data.origin+pd.Timedelta(days=1)))
    for year in (2002,2003):
        train,valid,inner=meta_split(data,year)
        assert data.origin.iloc[train[-1]]+pd.Timedelta(days=7)<data.origin.iloc[valid[0]]
        for it,iv in inner:
            assert data.origin.iloc[train[it[-1]]]+pd.Timedelta(days=7)<data.origin.iloc[train[iv[0]]]


@pytest.mark.parametrize('bad',['missing','duplicate','future','nan'])
def test_reject_bad_sources(bad):
    raw,frames=example()
    if bad=='missing':frames[0]=frames[0].iloc[1:]
    elif bad=='duplicate':frames[0]=pd.concat([frames[0],frames[0].iloc[:1]])
    elif bad=='future':frames[0].loc[0,'origin']=pd.Timestamp('2004-01-01')
    else:frames[0].loc[0,'prediction']=np.nan
    with pytest.raises(ValueError):assemble(raw,frames)
