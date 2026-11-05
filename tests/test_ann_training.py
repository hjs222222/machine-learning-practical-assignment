"""Small iteration caps are intentional in these training-contract tests."""
import joblib
import numpy as np
import pytest
from sklearn.base import clone
from models import RunoffRegressor

pytestmark = pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')


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


def test_chrono_probe_is_fold_local_then_full_refit(monkeypatch,tmp_path):
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
                          ann_chrono_stop=True)
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
    path=tmp_path/'ann.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(model.predict(x[80:]),joblib.load(path).predict(x[80:]))


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
