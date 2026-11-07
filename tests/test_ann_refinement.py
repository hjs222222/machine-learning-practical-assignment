"""Real full-batch optimizers and explicit failure/rollback boundaries."""
import joblib
import numpy as np
import pytest
import torch
from sklearn.base import clone
from models import DenseNet,RunoffRegressor

pytestmark=pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')


@pytest.mark.parametrize('mode',[None,True,1,'','lbfgs',[]])
def test_ann_refinement_rejects_invalid_modes(mode):
    with pytest.raises(ValueError,match='ann_refinement'):
        RunoffRegressor(family='ANN',ann_backend='torch',ann_refinement=mode).fit(np.ones((12,1,3)),np.ones((12,1)))


@pytest.mark.parametrize('options',[dict(family='RF'),dict(family='LSTM'),dict(family='ANN'),
    dict(family='ANN',ann_backend='torch',ann_lr_schedule='half_linear')])
def test_ann_refinement_requires_constant_torch_ann(options):
    with pytest.raises(ValueError,match='ann_refinement'):
        RunoffRegressor(**options,ann_refinement='lbfgs20').fit(np.ones((12,1,3)),np.ones((12,1)))


def test_ann_refinement_full_objective_matches_unequal_weighted_batches():
    x=np.random.default_rng(980).uniform(1,4,(11,1,3)).astype('float32');y=x[:,-1,:1]+1
    model=RunoffRegressor(family='ANN',ann_backend='torch',ann_refinement='none',top_k=None,ann_batch_size=4,alpha=1.)
    scaled,z=model._fit_preprocessing(x,y);net=DenseNet(3,2,1,'relu',False,42).double()
    tx=torch.tensor(scaled,dtype=torch.float64);tz=torch.tensor(z,dtype=torch.float64)
    q=torch.tensor(x[:,-1,:1],dtype=torch.float64);truth=torch.tensor(y,dtype=torch.float64)
    penalty=sum(p.square().sum() for name,p in net.named_parameters() if name.endswith('weight'))
    expected=0
    for start in (0,4,8):
        n=min(4,11-start)
        expected=expected+(.5*model._torch_objective(net(tx[start:start+4]),tz[start:start+4],q[start:start+4],truth[start:start+4])+penalty/(2*n))*n/11
    actual=model._ann_full_objective(net,tx,tz,q,truth)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    first=torch.autograd.grad(actual,list(net.parameters()),retain_graph=True)
    second=torch.autograd.grad(expected,list(net.parameters()))
    for a,b in zip(first,second):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)


@pytest.mark.parametrize('mode',['adam_full45','lbfgs20'])
@pytest.mark.parametrize('outputs',[1,7])
def test_ann_refinement_real_training_counts_and_restore(mode,outputs,monkeypatch,tmp_path):
    x=np.random.default_rng(981).uniform(1,4,(25,3,4)).astype('float32')
    y=x[:,-1,:1]*np.arange(1,outputs+1)[None,:]+1
    steps=[];original=torch.optim.Adam.step
    def observe(optimizer,*args,**kwargs):
        state=next(iter(optimizer.state.values()),{})
        steps.append(float(state.get('step',0)))
        return original(optimizer,*args,**kwargs)
    monkeypatch.setattr(torch.optim.Adam,'step',observe)
    model=RunoffRegressor(family='ANN',ann_backend='torch',ann_refinement=mode,ann_batch_size=7,
        hidden=3,iterations=2,top_k=None,ann_history=True,ann_raw_loss=.25,
        first_day_weight=4 if outputs==7 else 1).fit(x,y)
    state=model.ann_refinement_
    assert state['mode']==mode and state['gradient_budget']==45 and 1<=state['gradient_evaluations']<=45
    assert len(state['objective_history'])==len(state['gradient_norm_history'])==state['gradient_evaluations']
    assert state['full_objective_evaluations']==state['gradient_evaluations']+2+int(state['rolled_back'])
    assert state['full_l2_coefficient']==pytest.approx(model.alpha*4/(2*25)) and state['unclipped_gradients']
    assert state['selected_final_objective']<=state['initial_objective']+1e-6
    assert state['first_stage_epochs']==len(model.loss_history_)==len(model.learning_rate_history_)==2
    assert len(steps)==(53 if mode=='adam_full45' else 8)
    if mode=='adam_full45':assert steps[8]==8 and state['gradient_evaluations']==state['optimizer_iterations']==45
    scaled=model.x_scaler_.transform(x.reshape(-1,4)).reshape(x.shape).astype('float32')
    target=model.y_scaler_.transform(y).astype('float32')*model.output_scale_
    with torch.no_grad():
        selected=model._ann_full_objective(model.estimator_,torch.from_numpy(scaled),torch.from_numpy(target),
            torch.from_numpy(x[:,-1,:1]),torch.tensor(y,dtype=torch.float32))
    assert float(selected)==pytest.approx(state['selected_final_objective'],rel=1e-5,abs=1e-6)
    assert model.predict(x).shape==(25,outputs) and np.isfinite(model.predict(x)).all()
    assert clone(model).get_params()==model.get_params()
    path=tmp_path/'refine.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(joblib.load(path).predict(x),model.predict(x))


def test_ann_refinement_rolls_back_only_using_training_objective(monkeypatch):
    x=np.ones((16,1,3));y=np.arange(1,17)[:,None]
    args=dict(family='ANN',ann_backend='torch',hidden=2,iterations=1,top_k=None)
    baseline=RunoffRegressor(**args).fit(x,y)
    original=torch.optim.Adam.step;calls=0
    def worsen(optimizer,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==1:return original(optimizer,*args,**kwargs)
        with torch.no_grad():
            for group in optimizer.param_groups:
                for p in group['params']:p.add_(1.)
    monkeypatch.setattr(torch.optim.Adam,'step',worsen)
    refined=RunoffRegressor(**args,ann_refinement='adam_full45').fit(x,y)
    state=refined.ann_refinement_
    assert state['rolled_back'] and state['proposed_final_objective']>state['initial_objective']
    assert state['selected_final_objective']==state['initial_objective']
    np.testing.assert_array_equal(refined.predict(x),baseline.predict(x))


def test_ann_refinement_rejects_46th_closure(monkeypatch):
    original=RunoffRegressor._ann_full_objective;calls=0
    def observe(self,*args):
        nonlocal calls
        calls+=1;return original(self,*args)
    def exceed(optimizer,closure):
        for _ in range(46):closure()
    monkeypatch.setattr(RunoffRegressor,'_ann_full_objective',observe)
    monkeypatch.setattr(torch.optim.LBFGS,'step',exceed)
    with pytest.raises(ValueError,match='45'):
        RunoffRegressor(family='ANN',ann_backend='torch',ann_refinement='lbfgs20',
            hidden=2,iterations=1,top_k=None).fit(np.ones((16,1,3)),np.arange(1,17)[:,None])
    assert calls==46  # One initial forward plus 45 real gradient evaluations.


@pytest.mark.parametrize('failure',['objective','gradient','norm'])
def test_ann_refinement_nonfinite_failures_are_not_silently_rolled_back(failure,monkeypatch):
    def invalid(self,net,*args):
        if failure=='objective':return net.head.weight.sum()*float('nan')
        if failure=='norm':return net.head.weight.sum()*1e30
        return torch.sqrt((net.head.weight-net.head.weight.detach()).sum())
    monkeypatch.setattr(RunoffRegressor,'_ann_full_objective',invalid)
    with pytest.raises(ValueError,match='梯度范数' if failure=='norm' else '精修'):
        RunoffRegressor(family='ANN',ann_backend='torch',ann_refinement='lbfgs20',
            hidden=2,iterations=1,top_k=None).fit(np.ones((16,1,3)),np.arange(1,17)[:,None])


@pytest.mark.parametrize('family,backend',[('ANN','torch'),('ANN','sklearn'),('LSTM','sklearn'),('RF','sklearn')])
def test_ann_refinement_none_default_and_legacy_inference(family,backend,tmp_path):
    x=np.random.default_rng(982).uniform(1,4,(20,2,3)).astype('float32');y=x[:,-1,:1]+1
    args=dict(family=family,ann_backend=backend,hidden=2,iterations=2,epochs=2,trees=2,top_k=None)
    first=RunoffRegressor(**args).fit(x,y);second=RunoffRegressor(**args,ann_refinement='none').fit(x,y)
    np.testing.assert_array_equal(first.predict(x),second.predict(x))
    np.testing.assert_array_equal(getattr(first,'loss_history_',[]),getattr(second,'loss_history_',[]))
    assert getattr(first,'ann_refinement_',None) is None
    del first.ann_refinement
    path=tmp_path/'legacy.joblib';joblib.dump(first,path)
    np.testing.assert_array_equal(joblib.load(path).predict(x),second.predict(x))
