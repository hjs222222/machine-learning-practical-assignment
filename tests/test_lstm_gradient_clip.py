import joblib
import numpy as np
import pytest
import torch
from sklearn.base import clone

from models import RunoffRegressor


@pytest.mark.parametrize('value', [True, None, '5', 0, -1, np.nan, np.inf])
def test_reject_invalid_lstm_clip(value):
    with pytest.raises(ValueError, match='lstm_grad_clip'):
        RunoffRegressor(family='LSTM', lstm_grad_clip=value).fit(np.ones((10,3,2)), np.ones((10,1)))


@pytest.mark.parametrize('family', ['RF', 'ANN'])
def test_other_families_cannot_set_lstm_clip(family):
    with pytest.raises(ValueError, match='lstm_grad_clip'):
        RunoffRegressor(family=family, lstm_grad_clip=5).fit(np.ones((10,3,2)), np.ones((10,1)))


@pytest.mark.parametrize('outputs', [1, 7])
def test_actual_clip_native_outputs_clone_and_restore(outputs, monkeypatch, tmp_path):
    values = []
    original = torch.nn.utils.clip_grad_norm_

    def observed(parameters, value):
        values.append(value)
        return original(parameters, value)

    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', observed)
    x = np.random.default_rng(8).uniform(1,10,(18,3,2)).astype('float32')
    y = np.column_stack([x[:,-1,0]+i for i in range(outputs)])
    model = clone(RunoffRegressor(family='LSTM', lstm_grad_clip=5., hidden=2, epochs=2, seq_len=3,
                                  transform='log', lstm_raw_loss=.25))
    model.fit(x, y)
    assert values == [5., 5.]
    prediction = model.predict(x)
    assert prediction.shape == (18,outputs) and np.isfinite(prediction).all()
    path = tmp_path/'model.joblib'
    joblib.dump(model,path)
    np.testing.assert_array_equal(prediction,joblib.load(path).predict(x))
    del model.lstm_grad_clip
    np.testing.assert_array_equal(prediction,model.predict(x))


def test_default_and_explicit_one_are_exact():
    x = np.random.default_rng(8).uniform(1,10,(18,3,2)).astype('float32')
    y = x[:,-1,:1]
    first = RunoffRegressor(family='LSTM', hidden=2, epochs=2).fit(x,y)
    second = clone(first).set_params(lstm_grad_clip=1.).fit(x,y)
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    assert first.loss_history_ == second.loss_history_


def test_research_actual_clip_log(tmp_path):
    import json
    import pandas as pd
    from research_h1 import run_stage
    dates = pd.date_range('2000-01-01','2004-12-31')
    raw = pd.DataFrame({c:10+np.sin(np.arange(len(dates))/13)+np.arange(len(dates))*.001
                       for c in ['Discharge','Dayl','Prcp','Srad','Swe','Tmax','Tmin','Vp']}, index=dates)
    candidates = [dict(id=str(clip), family='LSTM', mode='daily', transform='log', residual=False,
                       params=dict(hidden=2,epochs=1,seq_len=3,lstm_grad_clip=clip)) for clip in (1.,5.)]
    out = tmp_path/'study'
    run_stage('features', out, raw=raw, candidates=candidates)
    fits = json.loads((out/'fits.json').read_text())
    assert len(fits) == 6
    for fit in fits:
        assert fit['lstm_grad_clip'] == fit['gradient_clip'] == float(fit['candidate'])
