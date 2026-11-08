import importlib

import numpy as np
import pandas as pd
import pytest


def training_frame(year=2001):
    dates=pd.date_range('2000-02-29',f'{year-1}-12-24')
    truth=100+np.arange(len(dates))%17
    return pd.DataFrame(dict(candidate='ANN',year=year,origin=dates,
        target_date=dates+pd.Timedelta(days=1),truth=truth,prediction=truth-3.))


@pytest.mark.parametrize('year,n',[(2001,300),(2002,665),(2003,1030)])
def test_training_intercept_and_exact_mse_identity(year,n):
    module=importlib.import_module('research_raw_offset')
    frame=training_frame(year)
    result=module.training_offsets(frame).iloc[0]
    assert result.n==n and result.offset==3.
    assert result.mse_before-result.mse_after_unclipped==pytest.approx(result.offset**2)
    np.testing.assert_array_equal(module.apply_raw_offset(frame.prediction.to_numpy(),result.offset),frame.truth)


def test_extra_validation_values_and_no_file_io_do_not_affect_estimate(monkeypatch):
    module=importlib.import_module('research_raw_offset')
    frame=training_frame();expected=module.training_offsets(frame)
    frame['validation_truth']=1e200;frame['validation_prediction']=-1e200

    def forbidden(*args,**kwargs):
        raise AssertionError('Pure coefficient estimation must not read files')

    monkeypatch.setattr(pd,'read_csv',forbidden)
    pd.testing.assert_frame_equal(module.training_offsets(frame),expected)


@pytest.mark.parametrize('defect',['duplicate','missing','future','target','nan','negative','overflow','empty'])
def test_invalid_training_table_rejected(defect):
    module=importlib.import_module('research_raw_offset')
    frame=training_frame()
    if defect=='duplicate':frame=pd.concat([frame,frame.iloc[[0]]])
    elif defect=='missing':frame=frame.iloc[1:]
    elif defect=='future':frame.loc[0,'origin']=pd.Timestamp('2001-01-01')
    elif defect=='target':frame.loc[0,'target_date']+=pd.Timedelta(days=1)
    elif defect=='nan':frame.loc[0,'prediction']=np.nan
    elif defect=='negative':frame.loc[0,'truth']=-1
    elif defect=='overflow':frame['truth']=1e308
    else:frame=frame.iloc[:0]
    with pytest.raises(ValueError):module.training_offsets(frame)


def test_negative_offset_clips_and_rejects_invalid_values():
    module=importlib.import_module('research_raw_offset')
    np.testing.assert_array_equal(module.apply_raw_offset(np.array([1.,3.]),-2.),[0.,1.])
    for value in (True,np.nan,np.inf,'3'):
        with pytest.raises(ValueError):module.apply_raw_offset(np.array([1.]),value)
    with pytest.raises(ValueError):module.apply_raw_offset(np.array([np.nan]),1.)
