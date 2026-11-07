"""Small real fits exercise clipping without replacing the optimizer."""
import joblib
import numpy as np
import pytest
import torch
from sklearn.base import clone
from models import DenseNet, RunoffRegressor

pytestmark = pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')


@pytest.mark.parametrize('value',[0,-1,True,np.bool_(False),None,'5',5+0j,np.inf,np.nan,[]])
def test_ann_grad_clip_rejects_invalid_values(value):
    with pytest.raises(ValueError,match='ann_grad_clip'):
        RunoffRegressor(family='ANN',ann_backend='torch',ann_grad_clip=value).fit(
            np.ones((12,1,2)),np.ones((12,1)))


@pytest.mark.parametrize('options',[dict(family='RF'),dict(family='LSTM'),dict(family='ANN')])
def test_ann_grad_clip_nondefault_rejects_other_backends(options):
    with pytest.raises(ValueError,match='ann_grad_clip'):
        RunoffRegressor(**options,ann_grad_clip=5).fit(np.ones((12,1,2)),np.ones((12,1)))


@pytest.mark.parametrize('history',[False,True])
@pytest.mark.parametrize('outputs',[1,7])
def test_ann_grad_clip_trains_native_outputs_and_restores(history,outputs,monkeypatch,tmp_path):
    original=torch.nn.utils.clip_grad_norm_
    thresholds=[]
    def observe(parameters,threshold):
        parameters=list(parameters);norm=original(parameters,threshold)
        after=np.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None))
        assert after<=threshold+1e-5
        thresholds.append(threshold)
        return norm
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',observe)
    x=np.random.default_rng(831).uniform(1,5,(25,3,4)).astype('float32')
    y=x[:,-1,:1]*np.arange(1,outputs+1)[None,:]+1
    model=RunoffRegressor(family='ANN',ann_backend='torch',ann_grad_clip=np.float64(5.),
        ann_history=history,ann_linear_skip=True,ann_raw_loss=.25,first_day_weight=4 if outputs==7 else 1,
        hidden=3,iterations=2,ann_batch_size=7,top_k=None,seq_len=3,transform='sqrt',residual=True).fit(x,y)
    assert thresholds==[5.]*8 and len(model.loss_history_)==2
    assert model.predict(x).shape==(25,outputs) and np.isfinite(model.predict(x)).all()
    assert clone(model).get_params()==model.get_params()
    path=tmp_path/'clip.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(joblib.load(path).predict(x),model.predict(x))


def test_ann_grad_clip_changes_actual_large_gradient_norm(monkeypatch):
    original=DenseNet
    def controlled(*args,**kwargs):
        model=original(*args,**kwargs)
        with torch.no_grad():
            for parameter in model.parameters():parameter.zero_()
            model.head.bias.fill_(100.)
        return model
    monkeypatch.setattr('models.DenseNet',controlled)
    original_clip=torch.nn.utils.clip_grad_norm_
    observed=[]
    def observe(parameters,threshold):
        parameters=list(parameters);before=original_clip(parameters,threshold)
        after=np.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None))
        observed.append((float(before),after,threshold))
        return before
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',observe)
    for threshold in (1.,5.):
        RunoffRegressor(family='ANN',ann_backend='torch',ann_grad_clip=threshold,hidden=2,
            iterations=1,ann_learning_rate=1e-30,top_k=None,seq_len=1).fit(
                np.ones((4,1,2)),np.arange(1,5)[:,None])
    assert len(observed)==2
    for before,after,threshold in observed:
        assert before>5 and after==pytest.approx(threshold,rel=1e-5)


@pytest.mark.parametrize('family,backend',[('ANN','torch'),('ANN','sklearn'),('LSTM','sklearn'),('RF','sklearn')])
def test_ann_grad_clip_default_one_and_legacy_inference(family,backend,monkeypatch,tmp_path):
    original=torch.nn.utils.clip_grad_norm_;thresholds=[]
    def observe(parameters,threshold):
        thresholds.append(threshold)
        return original(parameters,threshold)
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',observe)
    x=np.random.default_rng(832).uniform(1,5,(20,2,3)).astype('float32');y=x[:,-1,:1]+1
    args=dict(family=family,ann_backend=backend,hidden=2,iterations=2,epochs=2,trees=2,top_k=None)
    first=RunoffRegressor(**args).fit(x,y);second=RunoffRegressor(**args,ann_grad_clip=1.).fit(x,y)
    np.testing.assert_array_equal(first.predict(x),second.predict(x))
    np.testing.assert_array_equal(getattr(first,'loss_history_',[]),getattr(second,'loss_history_',[]))
    assert thresholds==([1.]*4 if family=='LSTM' or backend=='torch' else [])
    del first.ann_grad_clip
    path=tmp_path/'legacy.joblib';joblib.dump(first,path)
    np.testing.assert_array_equal(joblib.load(path).predict(x),second.predict(x))
