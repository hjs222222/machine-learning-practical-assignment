import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone

from models import SequenceNet, RunoffRegressor


def test_bidirectional_uses_final_hidden_states_not_last_output():
    torch.manual_seed(7)
    net = SequenceNet(3, 2, 1, bidirectional=True)
    x = torch.arange(30, dtype=torch.float32).reshape(2,5,3)/10
    sequence, (hidden, _) = net.lstm(x)
    expected = net.head(torch.cat((hidden[-2], hidden[-1]), dim=1))
    torch.testing.assert_close(net(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden[-1], sequence[:,0,2:], rtol=0, atol=0)
    assert not torch.allclose(net(x), net.head(sequence[:,-1]), rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('hidden,bidirectional,count,width', [(16,False,1617,16),(32,False,5281,32),(16,True,3233,32)])
def test_actual_parameter_capacity(hidden, bidirectional, count, width):
    net = SequenceNet(7, hidden, 1, bidirectional=bidirectional)
    assert sum(p.numel() for p in net.parameters()) == count
    assert net.head.in_features == width


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_sequence_rejects_nonboolean_direction(value):
    with pytest.raises(ValueError, match='bidirectional'):
        SequenceNet(3,2,1,bidirectional=value)


@pytest.mark.parametrize('kwargs', [dict(family='RF'),dict(family='ANN'),dict(lstm_readout='last_mean'),dict(lstm_linear_skip=True)])
def test_reject_unstudied_regressor_combinations(kwargs):
    params = dict(family='LSTM',lstm_bidirectional=True)
    params.update(kwargs)
    with pytest.raises(ValueError, match='lstm_bidirectional'):
        RunoffRegressor(**params).fit(np.ones((12,3,2)),np.ones((12,1)))


@pytest.mark.parametrize('kwargs', [dict(readout='last_mean'),dict(linear_skip=True)])
def test_sequence_rejects_unstudied_combinations(kwargs):
    with pytest.raises(ValueError, match='bidirectional'):
        SequenceNet(3,2,1,bidirectional=True,**kwargs)


@pytest.mark.parametrize('outputs', [1,7])
def test_native_fit_clone_save_and_restore(outputs, tmp_path):
    x=np.random.default_rng(7).uniform(1,10,(18,4,3)).astype('float32')
    y=np.column_stack([x[:,-1,0]+i for i in range(outputs)])
    model=clone(RunoffRegressor(family='LSTM',lstm_bidirectional=True,hidden=2,epochs=2,seq_len=3,transform='log'))
    model.fit(x,y)
    prediction=model.predict(x)
    assert prediction.shape==(18,outputs) and np.isfinite(prediction).all()
    path=tmp_path/'model.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(prediction,joblib.load(path).predict(x))


def test_default_rng_output_and_legacy_are_exact():
    torch.manual_seed(7); old_lstm=torch.nn.LSTM(3,2,batch_first=True);old_head=torch.nn.Linear(2,1)
    state=torch.get_rng_state().clone()
    torch.manual_seed(7);net=SequenceNet(3,2,1,bidirectional=False)
    assert torch.equal(state,torch.get_rng_state())
    x=torch.arange(36,dtype=torch.float32).reshape(3,4,3)/10
    expected=old_head(old_lstm(x)[0][:,-1])
    torch.testing.assert_close(net(x),expected,rtol=0,atol=0)
    del net.bidirectional
    torch.testing.assert_close(net(x),expected,rtol=0,atol=0)
    a=np.random.default_rng(7).uniform(1,10,(18,4,3)).astype('float32'); y=a[:,-1,:1]
    model=RunoffRegressor(family='LSTM',hidden=2,epochs=2).fit(a,y)
    explicit=clone(model).set_params(lstm_bidirectional=False).fit(a,y)
    assert model.loss_history_==explicit.loss_history_
    np.testing.assert_array_equal(model.predict(a),explicit.predict(a))
    prediction=model.predict(a);del model.lstm_bidirectional
    np.testing.assert_array_equal(prediction,model.predict(a))


def raw_data():
    dates=pd.date_range('2000-01-01','2004-12-31')
    return pd.DataFrame({c:10+np.sin(np.arange(len(dates))/13)+np.arange(len(dates))*.001
        for c in ['Discharge','Dayl','Prcp','Srad','Swe','Tmax','Tmin','Vp']},index=dates)


def test_future_observations_cannot_change_bidirectional_history_prediction():
    from research_h1 import prepare_data
    raw=raw_data();first=prepare_data(raw);origin=pd.Timestamp('2002-01-05')
    changed=raw.copy();changed.loc[origin+pd.Timedelta(days=1):]*=100
    second=prepare_data(changed)
    index=np.flatnonzero(first['dates']==origin)[0]
    train=first['dates']<'2001-01-01'
    model=RunoffRegressor(family='LSTM',lstm_bidirectional=True,hidden=2,epochs=1,seq_len=7).fit(
        first['daily']['X'][train],first['y'][train])
    np.testing.assert_array_equal(first['daily']['X'][index],second['daily']['X'][index])
    np.testing.assert_array_equal(model.predict(first['daily']['X'][index:index+1]),model.predict(second['daily']['X'][index:index+1]))


def test_research_records_actual_direction_and_capacity(tmp_path):
    import json
    from research_h1 import run_stage
    candidates=[dict(id=str(bi),family='LSTM',mode='daily',transform='log',residual=False,
        params=dict(hidden=2,epochs=1,seq_len=3,lstm_bidirectional=bi)) for bi in (False,True)]
    out=tmp_path/'study';run_stage('features',out,raw=raw_data(),candidates=candidates)
    for row in json.loads((out/'fits.json').read_text()):
        bi=row['candidate']=='True'
        assert row['lstm_bidirectional']==bi and row['lstm_directions']==(2 if bi else 1)
        assert row['lstm_hidden_per_direction']==2
        assert row['lstm_readout_width']==(4 if bi else 2)
        assert row['lstm_total_parameters']==(181 if bi else 91)
