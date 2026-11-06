import numpy as np
import pytest
from sklearn.base import clone
from models import RunoffRegressor, simple_average, fit_stacking
from evaluation import metric_rows


@pytest.mark.parametrize('family',['ANN','RF','LSTM'])
@pytest.mark.parametrize('transform',['raw','sqrt','log','power025','power075'])
def test_native_output_scaling_and_restore(family, transform, tmp_path):
    import joblib
    rng = np.random.default_rng(1)
    X = rng.uniform(1,10,(70,5,8)).astype('float32')
    y = np.column_stack([X[:,-1,0] + i + 1 for i in range(7)])
    model = RunoffRegressor(family=family, transform=transform, top_k=4, epochs=2, trees=5, iterations=10)
    clone(model).fit(X[:50],y[:50])
    model.fit(X[:50],y[:50])
    pred = model.predict(X[50:])
    assert pred.shape == (20,7) and np.isfinite(pred).all()
    assert len(model.selected_) == 4
    np.testing.assert_allclose(model.x_scaler_.mean_, X[:50,:,model.selected_].reshape(-1,4).mean(0),rtol=1e-5)
    assert model.y_scaler_.n_samples_seen_ == 50
    path = tmp_path/'model.joblib'
    joblib.dump(model,path)
    np.testing.assert_allclose(pred, joblib.load(path).predict(X[50:]),rtol=1e-5,atol=1e-5)


def test_stacking_is_actual_meta_model_and_metrics_are_r2():
    rng = np.random.default_rng(1)
    y = rng.uniform(0,10,(100,7))
    predictions = [y + rng.normal(0,1,y.shape) for _ in range(3)]
    np.testing.assert_allclose(simple_average(predictions), np.stack(predictions).mean(0))
    meta = fit_stacking(predictions,y)
    assert meta.coef_.shape == (7,21)
    assert meta.predict(np.concatenate(predictions,axis=1)).shape == y.shape
    rows = metric_rows('multi','fake',y,y+10)
    assert rows[0]['R2'] < 0 and len(rows) == 7


def test_single_output_is_one_column():
    X = np.random.default_rng(2).normal(size=(40,3,6)).astype('float32')
    y = np.arange(40).reshape(-1,1)+1
    model = RunoffRegressor(family='RF',trees=5,top_k=3).fit(X,y)
    assert model.predict(X).shape == (40,1)


@pytest.mark.parametrize('transform',['raw','sqrt','log'])
def test_residual_prediction_reconstructs_current_observed_flow(transform):
    X = np.random.default_rng(4).uniform(1,10,(40,3,6)).astype('float32')
    y = np.tile(X[:,-1,:1],(1,7))
    model = RunoffRegressor(family='RF',trees=5,top_k=3,transform=transform,residual=True).fit(X,y)
    np.testing.assert_allclose(model.predict(X),y,rtol=1e-5,atol=1e-5)
