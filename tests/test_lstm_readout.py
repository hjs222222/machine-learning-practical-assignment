"""Real recurrent outputs and independent readout/gradient checks."""
import joblib
import numpy as np
import pytest
import torch
from sklearn.base import clone
from models import RunoffRegressor, SequenceNet

pytestmark=pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')


@pytest.mark.parametrize('outputs',[1,7])
@pytest.mark.parametrize('length',[1,5])
def test_lstm_readout_matches_formula_and_has_mean_gradients(outputs,length):
    torch.manual_seed(941);plain=SequenceNet(4,3,outputs)
    torch.manual_seed(941);pooled=SequenceNet(4,3,outputs,readout='last_mean')
    for name,value in plain.lstm.state_dict().items():
        torch.testing.assert_close(value,pooled.lstm.state_dict()[name],rtol=0,atol=0)
    x=torch.arange(2*length*4,dtype=torch.float32).reshape(2,length,4)/20
    sequence,_=pooled.lstm(x)
    expected=pooled.head(torch.cat([sequence[:,-1],sequence.mean(1)],dim=1))
    torch.testing.assert_close(pooled(x),expected,rtol=0,atol=0)
    assert pooled.head.in_features==6 and sum(p.numel() for p in pooled.head.parameters())==7*outputs
    captured=[]
    def retain(module,inputs,result):
        result[0].retain_grad();captured.append(result[0])
    hook=pooled.lstm.register_forward_hook(retain)
    with torch.no_grad():
        pooled.head.weight[:,:3].zero_();pooled.head.weight[:,3:].fill_(1.);pooled.head.bias.zero_()
    pooled(x).sum().backward();hook.remove()
    torch.testing.assert_close(captured[0].grad,torch.full_like(captured[0],outputs/length))


@pytest.mark.parametrize('readout',[None,True,1,'mean','',[]])
def test_lstm_readout_rejects_invalid_values(readout):
    with pytest.raises(ValueError,match='readout'):
        RunoffRegressor(family='LSTM',lstm_readout=readout).fit(np.ones((12,2,3)),np.ones((12,1)))
    with pytest.raises(ValueError,match='readout'):SequenceNet(3,2,1,readout=readout)


@pytest.mark.parametrize('options',[dict(family='RF'),dict(family='ANN'),dict(family='ANN',ann_backend='torch')])
def test_lstm_readout_rejects_other_families(options):
    with pytest.raises(ValueError,match='lstm_readout'):
        RunoffRegressor(**options,lstm_readout='last_mean').fit(np.ones((12,2,3)),np.ones((12,1)))


@pytest.mark.parametrize('outputs',[1,7])
def test_lstm_readout_native_training_clone_restore(outputs,tmp_path):
    x=np.random.default_rng(942).uniform(1,5,(24,3,4)).astype('float32')
    y=x[:,-1,:1]*np.arange(1,outputs+1)[None,:]+1
    model=RunoffRegressor(family='LSTM',lstm_readout='last_mean',hidden=3,epochs=3,seq_len=3,
        top_k=None,transform='log',lstm_raw_loss=.25,first_day_weight=4 if outputs==7 else 1).fit(x,y)
    assert model.estimator_.head.in_features==6 and len(model.loss_history_)==3
    assert model.predict(x).shape==(24,outputs) and np.isfinite(model.predict(x)).all()
    assert clone(model).get_params()==model.get_params()
    path=tmp_path/'readout.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(joblib.load(path).predict(x),model.predict(x))


def test_lstm_readout_default_and_legacy_inference_are_exact(tmp_path):
    x=np.random.default_rng(943).uniform(1,5,(24,3,4)).astype('float32');y=x[:,-1,:1]+1
    args=dict(family='LSTM',hidden=3,epochs=3,seq_len=3,top_k=None)
    first=RunoffRegressor(**args).fit(x,y);second=RunoffRegressor(**args,lstm_readout='last').fit(x,y)
    np.testing.assert_array_equal(first.predict(x),second.predict(x))
    np.testing.assert_array_equal(first.loss_history_,second.loss_history_)
    for key,value in first.estimator_.state_dict().items():
        torch.testing.assert_close(value,second.estimator_.state_dict()[key],rtol=0,atol=0)
    del first.lstm_readout;del first.estimator_.readout
    path=tmp_path/'legacy.joblib';joblib.dump(first,path)
    np.testing.assert_array_equal(joblib.load(path).predict(x),second.predict(x))


def test_lstm_readout_chrono_probe_and_refit_keep_head():
    x=np.random.default_rng(944).uniform(1,5,(90,3,4)).astype('float32');y=x[:,-1,:1]+1
    model=RunoffRegressor(family='LSTM',lstm_readout='last_mean',lstm_chrono_stop=True,
        hidden=3,epochs=5,seq_len=3,top_k=None,transform='log').fit(x[:80],y[:80])
    info=model.lstm_training_;assert info['probe_train_count']==57 and info['validation_start']==64
    assert model.estimator_.head.in_features==6 and len(model.loss_history_)==info['best_epoch']
    replay=clone(model).set_params(lstm_chrono_stop=False,epochs=info['best_epoch']).fit(x[:80],y[:80])
    np.testing.assert_array_equal(replay.predict(x[80:]),model.predict(x[80:]))
