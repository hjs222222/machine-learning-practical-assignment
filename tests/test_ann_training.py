"""Small iteration caps are intentional in these training-contract tests."""
import joblib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import r2_score
from models import RunoffRegressor, DenseNet

pytestmark = pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')


@pytest.mark.parametrize('history',[False,True])
@pytest.mark.parametrize('outputs',[1,7])
def test_torch_ann_native_shapes_clone_and_serialization(history,outputs,tmp_path):
    x=np.random.default_rng(121).uniform(1,5,(35,3,4)).astype('float32')
    y=x[:,-1,:1]*np.arange(1,outputs+1)[None,:]+1
    model=RunoffRegressor(family='ANN',ann_backend='torch',ann_raw_loss=.25,
        hidden=3,iterations=3,top_k=None,ann_history=history).fit(x,y)
    assert model.predict(x).shape==(35,outputs)
    assert model.estimator_.hidden.in_features==4*(3 if history else 1)
    assert len(model.loss_history_)==3
    assert model.raw_loss_scale_==pytest.approx(y.std())
    assert model.get_params()==clone(model).get_params()
    path=tmp_path/'torch_ann.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(model.predict(x),joblib.load(path).predict(x))


@pytest.mark.parametrize('options',[{'ann_backend':'invalid'},
    {'ann_raw_loss':.25},{'ann_backend':'torch','ann_raw_loss':1.1},
    {'ann_backend':'torch','ann_solver':'lbfgs'},
    {'ann_backend':'torch','ann_chrono_stop':True}])
def test_torch_ann_rejects_unsupported_settings(options):
    with pytest.raises(ValueError,match='ann_backend|ann_raw_loss'):
        RunoffRegressor(family='ANN',top_k=None,**options).fit(np.ones((80,3,4)),np.ones((80,1)))


@pytest.mark.parametrize('activation',['relu','tanh','logistic'])
def test_torch_ann_initial_weights_match_untrained_sklearn(activation):
    from sklearn.neural_network import MLPRegressor
    reference=MLPRegressor(activation=activation)
    reference._random_state=np.random.RandomState(42)
    net=DenseNet(7,16,1,activation,False,42)
    for layer,inputs,outputs in [(net.hidden,7,16),(net.head,16,1)]:
        weights,bias=reference._init_coef(inputs,outputs,np.float32)
        np.testing.assert_array_equal(layer.weight.detach().numpy(),weights.T)
        np.testing.assert_array_equal(layer.bias.detach().numpy(),bias)


@pytest.mark.parametrize('scaling',['standard','minmax'])
@pytest.mark.parametrize('compression',['raw','signed_log'])
def test_lstm_chrono_preprocessing_is_inner_only_then_full_refit(monkeypatch,tmp_path,scaling,compression):
    x=np.random.default_rng(90).uniform(1,8,(90,3,5)).astype('float32')
    y=x[:,-1,:1]*1.2+2
    calls=[]
    original=RunoffRegressor._fit_preprocessing
    def capture(self,X,Y):
        calls.append((X.copy(),Y.copy()))
        return original(self,X,Y)
    monkeypatch.setattr(RunoffRegressor,'_fit_preprocessing',capture)
    model=RunoffRegressor(family='LSTM',top_k=None,hidden=2,epochs=5,
                          transform='log',lstm_chrono_stop=True,x_scaling=scaling,input_transform=compression)
    params=model.get_params()
    model.fit(x[:80],y[:80])
    info=model.lstm_training_
    assert [len(X) for X,Y in calls]==[57,80]
    np.testing.assert_array_equal(calls[0][1],y[:57])
    assert info['validation_start']==64 and info['gap']==7
    assert info['probe_scaler_samples']==57*3
    assert info['best_epoch'] in (1,5) and len(model.loss_history_)==info['best_epoch']
    assert model.x_scaler_.n_samples_seen_==80*3
    assert model.get_params()==params==clone(model).get_params()
    path=tmp_path/'lstm.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(model.predict(x[80:]),joblib.load(path).predict(x[80:]))
    probe=clone(model).set_params(lstm_chrono_stop=False,epochs=info['best_epoch']).fit(x[:57],y[:57])
    score=r2_score(y[64:80],probe.predict(x[64:80]))
    assert score==pytest.approx(info['best_inner_r2'],abs=1e-5)


def test_lstm_chrono_patience_uses_raw_flow_r2():
    model=RunoffRegressor(family='LSTM',top_k=None,hidden=2,epochs=100,
        lstm_learning_rate=1e-30,lstm_chrono_stop=True).fit(np.ones((80,3,4)),np.ones((80,1)))
    info=model.lstm_training_
    assert info['best_epoch']==1 and info['stopped_epoch']==35
    assert len(model.loss_history_)==1


@pytest.mark.parametrize('changes',[{'family':'RF'},{'epochs':4},{'epochs':301}])
def test_lstm_chrono_rejects_unsupported_settings(changes):
    args=dict(family='LSTM',top_k=None,epochs=5,lstm_chrono_stop=True);args.update(changes)
    with pytest.raises(ValueError,match='lstm_chrono_stop'):
        RunoffRegressor(**args).fit(np.ones((80,3,4)),np.ones((80,1)))


@pytest.mark.parametrize('n,outputs',[(40,1),(80,7)])
def test_lstm_chrono_rejects_short_or_multioutput(n,outputs):
    with pytest.raises(ValueError,match='lstm_chrono_stop'):
        RunoffRegressor(family='LSTM',lstm_chrono_stop=True).fit(np.ones((n,3,4)),np.ones((n,outputs)))


@pytest.mark.parametrize('rate',[.0003,.001,.003])
def test_lstm_learning_rate_reaches_optimizer(monkeypatch,rate):
    observed=[]
    original=torch.optim.Adam
    def capture(params,**kwargs):
        observed.append(kwargs['lr'])
        return original(params,**kwargs)
    monkeypatch.setattr(torch.optim,'Adam',capture)
    model=RunoffRegressor(family='LSTM',top_k=None,hidden=2,epochs=1,lstm_learning_rate=rate)
    model.fit(np.ones((12,3,4)),np.ones((12,1)))
    assert observed==[rate]


@pytest.mark.parametrize('rate',[0,-.001,float('inf')])
def test_lstm_learning_rate_rejects_invalid(rate):
    with pytest.raises(ValueError,match='lstm_learning_rate'):
        RunoffRegressor(family='LSTM',lstm_learning_rate=rate).fit(np.ones((12,3,4)),np.ones((12,1)))


@pytest.mark.parametrize('transform',['raw','sqrt','log'])
@pytest.mark.parametrize('residual',[False,True])
@pytest.mark.parametrize('family',['ANN','LSTM'])
def test_torch_raw_loss_matches_original_flow_error_and_has_gradient(transform,residual,family):
    model=RunoffRegressor(family=family,transform=transform,residual=residual,
        ann_backend='torch' if family=='ANN' else 'sklearn',ann_raw_loss=.5 if family=='ANN' else 0.,
        lstm_raw_loss=.5 if family=='LSTM' else 0.)
    truth=np.array([[10.],[20.],[40.]])
    current=np.array([[8.],[21.],[35.]])
    training_target=model.target(truth)-(model.target(current) if residual else 0)
    model.y_scaler_=StandardScaler().fit(training_target)
    model.raw_loss_scale_=float(truth.std())
    pred_flow=truth+2
    target_pred=model.target(pred_flow)-(model.target(current) if residual else 0)
    pred=torch.tensor(model.y_scaler_.transform(target_pred),dtype=torch.float64,requires_grad=True)
    z=torch.tensor(model.y_scaler_.transform(training_target),dtype=torch.float64)
    loss=model._torch_objective(pred,z,torch.tensor(current),torch.tensor(truth))
    expected=.5*np.mean((pred.detach().numpy()-z.numpy())**2)+.5*np.mean(((pred_flow-truth)/truth.std())**2)
    assert loss.item()==pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0


def test_lstm_raw_loss_scale_uses_fit_labels_only():
    x=np.random.default_rng(18).uniform(1,5,(30,3,4)).astype('float32')
    y=np.arange(30,dtype=float)[:,None]+1
    model=RunoffRegressor(family='LSTM',top_k=None,epochs=1,hidden=3,lstm_raw_loss=.5).fit(x[:20],y[:20])
    assert model.raw_loss_scale_==pytest.approx(y[:20].std())
    assert np.isfinite(model.predict(x[20:])).all()


def test_weighted_multioutput_lstm_raw_loss_matches_weighted_original_error():
    model=RunoffRegressor(family='LSTM',transform='sqrt',residual=True,
                          first_day_weight=4,lstm_raw_loss=.5)
    truth=np.array([[10.],[20.],[40.]])*np.arange(1,8)[None,:]
    current=np.array([[8.],[21.],[35.]])
    target=model.target(truth)-model.target(current)
    model.y_scaler_=StandardScaler().fit(target)
    model.output_scale_=np.array([2.,1.,1.,1.,1.,1.,1.])
    model.raw_loss_scale_=float(truth.std())
    z=model.y_scaler_.transform(target)*model.output_scale_
    pp=model.y_scaler_.transform(model.target(truth+2)-model.target(current))*model.output_scale_
    prediction=torch.tensor(pp,requires_grad=True)
    loss=model._torch_objective(prediction,torch.tensor(z),torch.tensor(current),torch.tensor(truth))
    expected=.5*np.mean((pp-z)**2)+.5*np.mean(((2/model.raw_loss_scale_)*model.output_scale_)**2)
    assert loss.item()==pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize('options',[{'lstm_raw_loss':-1},{'lstm_raw_loss':1.1},
    {'lstm_raw_loss':float('nan')},{'lstm_raw_loss':.5,'family':'ANN'}])
def test_lstm_raw_loss_rejects_invalid_settings(options):
    args=dict(family='LSTM',top_k=None,epochs=1);args.update(options)
    with pytest.raises(ValueError,match='lstm_raw_loss'):
        RunoffRegressor(**args).fit(np.ones((20,3,4)),np.ones((20,1)))


@pytest.mark.parametrize('solver',['adam','lbfgs'])
@pytest.mark.parametrize('rate',[.0003,.003])
def test_ann_optimizer_parameters_are_used(solver,rate):
    x=np.random.default_rng(14).uniform(1,5,(40,3,5)).astype('float32')
    y=x[:,-1,:1]+2
    model=RunoffRegressor(family='ANN',top_k=None,hidden=4,iterations=5,
                         ann_solver=solver,ann_learning_rate=rate).fit(x,y)
    assert model.estimator_.solver==solver
    assert model.estimator_.learning_rate_init==rate
    assert model.predict(x).shape==(40,1)


@pytest.mark.parametrize('activation',['relu','tanh','logistic'])
def test_ann_hidden_activation_is_used(activation):
    x=np.random.default_rng(35).uniform(1,5,(40,3,5)).astype('float32')
    model=RunoffRegressor(family='ANN',top_k=None,hidden=4,iterations=5,
                          ann_activation=activation).fit(x,x[:,-1,:1]+2)
    assert model.estimator_.activation==activation
    assert model.predict(x).shape==(40,1)


@pytest.mark.parametrize('history',[False,True])
@pytest.mark.parametrize('scaling',['standard','minmax'])
def test_chrono_probe_is_fold_local_then_full_refit(monkeypatch,tmp_path,history,scaling):
    x=np.random.default_rng(19).uniform(1,8,(90,5,8)).astype('float32')
    y=x[:,-1,:1]*1.2+2
    y[80:]=100000
    observed=[]
    def capture(features,target,**kwargs):
        observed.append((features.copy(),target.copy()))
        return np.arange(features.shape[1])
    monkeypatch.setattr('models.mutual_info_regression',capture)
    model=RunoffRegressor(family='ANN',hidden=4,iterations=10,top_k=3,
                          transform='log',residual=True,mi_target='model_target',
                          ann_chrono_stop=True,ann_history=history,x_scaling=scaling)
    parameters=model.get_params()
    model.fit(x[:80],y[:80])
    info=model.ann_training_
    assert info['probe_train_count']==57
    assert info['validation_start']==64 and info['validation_count']==16 and info['gap']==7
    assert len(observed)==2
    for (features,label),n in zip(observed,[57,80]):
        np.testing.assert_array_equal(features,x[:n,-1])
        np.testing.assert_allclose(label,np.log1p(y[:n,0])-np.log1p(x[:n,-1,0]),rtol=1e-6)
    assert model.x_scaler_.n_samples_seen_==80*5
    assert model.get_params()==parameters==clone(model).get_params()
    assert info['best_epoch'] in [1,5,10]
    assert 1<=info['best_epoch']<=info['stopped_epoch']<=10
    assert len(model.estimator_.loss_curve_)==info['best_epoch']
    assert model.estimator_.n_features_in_==len(model.selected_)*(5 if history else 1)
    path=tmp_path/'ann.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(model.predict(x[80:]),joblib.load(path).predict(x[80:]))


def test_ann_history_fit_and_predict_share_time_major_layout():
    x=np.random.default_rng(27).uniform(1,5,(35,5,4)).astype('float32')
    y=x[:,-1,:1]+x[:,2,:1]
    model=RunoffRegressor(family='ANN',top_k=None,hidden=4,iterations=5,
                         ann_history=True,seq_len=3).fit(x,y)
    assert model.estimator_.n_features_in_==3*4
    scaled=model.x_scaler_.transform(x[:,-3:].reshape(-1,4)).reshape(35,-1).astype('float32')
    expected=model.y_scaler_.inverse_transform(model.estimator_.predict(scaled).reshape(-1,1))
    np.testing.assert_allclose(model.predict(x),np.maximum(expected,0))


def test_ann_history_rejects_other_families():
    with pytest.raises(ValueError,match='ann_history'):
        RunoffRegressor(family='RF',ann_history=True).fit(np.ones((20,3,4)),np.ones((20,1)))


def test_chrono_patience_stops_without_random_split():
    x=np.ones((80,3,4),dtype='float32');y=np.ones((80,1))
    model=RunoffRegressor(family='ANN',hidden=2,iterations=100,top_k=None,
                          ann_learning_rate=1e-30,ann_chrono_stop=True).fit(x,y)
    assert model.ann_training_['best_epoch']==1
    assert model.ann_training_['stopped_epoch']==35
    assert [v['epoch'] for v in model.ann_training_['history']]==[1,5,10,15,20,25,30,35]


@pytest.mark.parametrize('changes',[
    {'ann_solver':'lbfgs'}, {'family':'RF'}, {'iterations':3}, {'iterations':401},
])
def test_chrono_rejects_unsupported_settings(changes):
    args=dict(family='ANN',top_k=None,iterations=10,ann_chrono_stop=True)
    args.update(changes)
    with pytest.raises(ValueError,match='ann_chrono_stop'):
        RunoffRegressor(**args).fit(np.ones((80,3,4)),np.ones((80,1)))


@pytest.mark.parametrize('n,outputs',[(40,1),(80,7)])
def test_chrono_rejects_short_or_multioutput_training(n,outputs):
    with pytest.raises(ValueError,match='ann_chrono_stop'):
        RunoffRegressor(family='ANN',top_k=None,ann_chrono_stop=True).fit(
            np.ones((n,3,4)),np.ones((n,outputs)))
