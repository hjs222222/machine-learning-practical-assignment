import joblib
import numpy as np
import pytest
from sklearn.base import clone
from models import RunoffRegressor

pytestmark=pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')


@pytest.mark.parametrize('outputs',[1,7])
def test_multilayer_native_outputs_clone_and_restore(outputs,tmp_path):
    rng=np.random.RandomState(3);x=rng.uniform(1,5,(40,1,4)).astype('float32')
    y=x[:,0,:1]*np.arange(1,outputs+1)
    model=clone(RunoffRegressor(family='ANN',hidden=(5,3),iterations=3,top_k=None)).fit(x,y)
    assert [a.shape for a in model.estimator_.coefs_]==[(4,5),(5,3),(3,outputs)]
    assert model.predict(x).shape==(40,outputs)
    path=tmp_path/'m.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(model.predict(x),joblib.load(path).predict(x))


@pytest.mark.parametrize('hidden',[(),(0,3),(True,3),(3.5,2)])
def test_invalid_layer_tuple(hidden):
    with pytest.raises(ValueError,match='hidden'):
        RunoffRegressor(family='ANN',hidden=hidden).fit(np.ones((10,1,2)),np.ones((10,1)))


@pytest.mark.parametrize('family,backend',[('LSTM','sklearn'),('RF','sklearn'),('ANN','torch')])
def test_layer_tuple_does_not_silently_change_other_backends(family,backend):
    with pytest.raises(ValueError,match='hidden'):
        RunoffRegressor(family=family,ann_backend=backend,hidden=(5,3)).fit(np.ones((10,1,2)),np.ones((10,1)))
