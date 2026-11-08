import json

import numpy as np
import pandas as pd
import pytest

import research_h1 as rh


def raw_data():
    dates=pd.date_range('2000-01-01','2004-12-31')
    return pd.DataFrame({c:10+np.sin(np.arange(len(dates))/13)+np.arange(len(dates))*.001
        for c in ['Discharge','Dayl','Prcp','Srad','Swe','Tmax','Tmin','Vp']},index=dates)


@pytest.mark.parametrize('horizons',[1,7])
def test_only_redundant_temperature_range_is_omitted(horizons):
    raw=raw_data();before=raw.copy(deep=True)
    data=rh.prepare_data(raw,horizons=horizons)
    old,new=data['weather'],data['weather_no_trange0']
    indices=[i for i,n in enumerate(old['names']) if n!='Trange_lag0']
    assert new['names']==[old['names'][i] for i in indices] and len(indices)==14
    np.testing.assert_array_equal(new['X'],old['X'][:,:,indices])
    np.testing.assert_array_equal(new['dates'],data['dates'])
    pd.testing.assert_frame_equal(raw,before)
    for name in ('Tmax_lag0','Tmin_lag0','Tavg_diff1','Tavg_diff3','Trange_diff1'):
        assert name in new['names']


def test_mode_default_and_future_boundaries():
    assert rh.WEATHER_ABLATIONS['weather_no_trange0']==('Trange_lag0',)
    assert rh.DEFAULT_MODES==('mi','forced','all','hydro')
    raw=raw_data();first=rh.prepare_data(raw)['weather_no_trange0']
    raw.loc['2004-01-01':]*=1000;second=rh.prepare_data(raw)['weather_no_trange0']
    np.testing.assert_array_equal(first['X'],second['X'])
    raw.loc['2002-07-01':]*=10;third=rh.prepare_data(raw)['weather_no_trange0']
    past=first['dates']<'2002-07-01'
    np.testing.assert_array_equal(first['X'][past],third['X'][past])
    assert 'weather_no_trange0' not in rh.prepare_data(raw,available_weather=True)


def test_research_keeps_all_other_temperature_features(tmp_path):
    out=tmp_path/'study'
    candidate=dict(id='reduced',family='ANN',mode='weather_no_trange0',transform='sqrt',residual=True,
                   params=dict(hidden=2,iterations=1,seq_len=1))
    rh.run_stage('features',out,raw=raw_data(),candidates=[candidate])
    fits=json.loads((out/'fits.json').read_text())
    assert len(fits)==3
    for fit in fits:
        assert fit['input_width']==14 and fit['mi_top_k'] is None and fit['mi_scores'] is None
        assert 'Trange_lag0' not in fit['selected_names']
        assert 'Trange_diff1' in fit['selected_names'] and 'Tmax_lag0' in fit['selected_names'] and 'Tmin_lag0' in fit['selected_names']
