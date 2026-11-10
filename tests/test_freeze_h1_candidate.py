import numpy as np
from freeze_h1_candidate import training_data
from test_weather_target_mi import raw_data


def test_freeze_inputs_ignore_2004_and_keep_seven_day_boundary():
    raw=raw_data();a=training_data(raw)
    raw.loc['2004-01-01':]*=10000
    b=training_data(raw)
    assert a['dates'].max().strftime('%Y-%m-%d')=='2003-12-24'
    assert a['y'].shape[1]==7
    assert len(a['dates'])==1395
    np.testing.assert_array_equal(a['y'],b['y'])
    np.testing.assert_array_equal(a['mi']['X'],b['mi']['X'])
