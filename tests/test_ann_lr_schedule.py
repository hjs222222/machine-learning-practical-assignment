"""Observe actual Adam steps, not a substitute scheduler."""
import joblib
import numpy as np
import pytest
import torch
from sklearn.base import clone
from models import RunoffRegressor

pytestmark=pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')


@pytest.mark.parametrize('schedule',[None,True,1,'plateau','',[]])
def test_ann_lr_schedule_rejects_invalid_values(schedule):
    with pytest.raises(ValueError,match='ann_lr_schedule'):
        RunoffRegressor(family='ANN',ann_backend='torch',ann_lr_schedule=schedule).fit(
            np.ones((12,1,2)),np.ones((12,1)))


@pytest.mark.parametrize('options',[dict(family='RF'),dict(family='LSTM'),dict(family='ANN')])
def test_ann_lr_schedule_rejects_unsupported_backends(options):
    with pytest.raises(ValueError,match='ann_lr_schedule'):
        RunoffRegressor(**options,ann_lr_schedule='half_linear').fit(np.ones((12,1,2)),np.ones((12,1)))


@pytest.mark.parametrize('epochs,expected',[(1,[1.]),(3,[1.,.55,.1]),(6,[1.,1.,1.,.7,.4,.1])])
def test_ann_lr_schedule_reaches_real_adam_steps(epochs,expected,monkeypatch):
    original=torch.optim.Adam.step;steps=[]
    def observe(optimizer,*args,**kwargs):
        steps.append(optimizer.param_groups[0]['lr'])
        return original(optimizer,*args,**kwargs)
    monkeypatch.setattr(torch.optim.Adam,'step',observe)
    x=np.random.default_rng(901).uniform(1,4,(13,1,3)).astype('float32');y=x[:,-1,:1]+1
    model=RunoffRegressor(family='ANN',ann_backend='torch',ann_lr_schedule='half_linear',
        hidden=2,iterations=epochs,ann_batch_size=5,top_k=None,ann_learning_rate=.01).fit(x,y)
    np.testing.assert_allclose(steps,np.repeat(np.array(expected)*.01,3),rtol=1e-12)
    np.testing.assert_allclose(model.learning_rate_history_,np.array(expected)*.01,rtol=1e-12)
    assert len(model.loss_history_)==epochs


def test_ann_lr_schedule_does_not_reset_early_stop_before_decay():
    x=np.ones((16,1,3));y=np.arange(1,17)[:,None]
    args=dict(family='ANN',ann_backend='torch',hidden=2,iterations=30,top_k=None,alpha=0,ann_learning_rate=1e-30)
    constant=RunoffRegressor(**args).fit(x,y)
    scheduled=RunoffRegressor(**args,ann_lr_schedule='half_linear').fit(x,y)
    assert len(constant.loss_history_)==len(scheduled.loss_history_)==12
    np.testing.assert_array_equal(constant.loss_history_,scheduled.loss_history_)
    np.testing.assert_array_equal(constant.predict(x),scheduled.predict(x))
    assert scheduled.learning_rate_history_==[1e-30]*12


@pytest.mark.parametrize('outputs',[1,7])
@pytest.mark.parametrize('history',[False,True])
def test_ann_lr_schedule_trains_native_outputs_and_restores(outputs,history,tmp_path):
    x=np.random.default_rng(902).uniform(1,5,(20,3,4)).astype('float32')
    y=x[:,-1,:1]*np.arange(1,outputs+1)[None,:]+1
    model=RunoffRegressor(family='ANN',ann_backend='torch',ann_lr_schedule='half_linear',
        hidden=3,iterations=4,top_k=None,ann_history=history,ann_raw_loss=.25,
        first_day_weight=4 if outputs==7 else 1,transform='sqrt',residual=True).fit(x,y)
    assert model.predict(x).shape==(20,outputs) and np.isfinite(model.predict(x)).all()
    np.testing.assert_allclose(model.learning_rate_history_,[.003,.003,.00165,.0003],rtol=1e-12)
    assert clone(model).get_params()==model.get_params()
    path=tmp_path/'schedule.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(joblib.load(path).predict(x),model.predict(x))


@pytest.mark.parametrize('family,backend',[('ANN','torch'),('ANN','sklearn'),('LSTM','sklearn'),('RF','sklearn')])
def test_ann_lr_schedule_constant_default_and_legacy_inference(family,backend,tmp_path):
    x=np.random.default_rng(903).uniform(1,4,(20,2,3)).astype('float32');y=x[:,-1,:1]+1
    args=dict(family=family,ann_backend=backend,hidden=2,iterations=3,epochs=3,trees=2,top_k=None)
    first=RunoffRegressor(**args).fit(x,y)
    second=RunoffRegressor(**args,ann_lr_schedule='constant').fit(x,y)
    np.testing.assert_array_equal(first.predict(x),second.predict(x))
    np.testing.assert_array_equal(getattr(first,'loss_history_',[]),getattr(second,'loss_history_',[]))
    if family=='LSTM' or backend=='torch':assert first.learning_rate_history_==[.003]*3
    else:assert not hasattr(first,'learning_rate_history_')
    del first.ann_lr_schedule
    path=tmp_path/'legacy.joblib';joblib.dump(first,path)
    np.testing.assert_array_equal(joblib.load(path).predict(x),second.predict(x))
