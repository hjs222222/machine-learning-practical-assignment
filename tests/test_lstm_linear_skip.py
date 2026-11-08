import numpy as np
import pytest
import torch
from sklearn.base import clone

from models import SequenceNet, RunoffRegressor


@pytest.mark.parametrize('outputs', [1, 7])
@pytest.mark.parametrize('readout', ['last', 'last_mean'])
def test_zero_skip_preserves_old_weights_output_rng_and_gradients(outputs, readout):
    torch.manual_seed(19)
    old = SequenceNet(7, 3, outputs, readout=readout)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    new = SequenceNet(7, 3, outputs, readout=readout, linear_skip=True)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name])
    x = torch.arange(4*5*7, dtype=torch.float32).reshape(4, 5, 7)/100
    assert torch.equal(old(x), new(x))
    old(x).square().sum().backward()
    new(x).square().sum().backward()
    for name, value in old.named_parameters():
        assert torch.equal(value.grad, dict(new.named_parameters())[name].grad)
    assert torch.count_nonzero(new.skip_weight.grad) > 0
    assert torch.count_nonzero(new.skip_bias.grad) > 0
    optimizer = torch.optim.Adam(new.parameters(), lr=.001)
    optimizer.step()
    assert torch.count_nonzero(new.skip_weight) > 0
    assert torch.count_nonzero(new.skip_bias) > 0


def test_skip_uses_last_standardized_observation_only():
    net = SequenceNet(2, 3, 1, linear_skip=True)
    for parameter in net.lstm.parameters():
        parameter.data.zero_()
    net.head.weight.data.zero_()
    net.head.bias.data.zero_()
    net.skip_weight.data.copy_(torch.tensor([[2., 3.]]))
    net.skip_bias.data.fill_(5.)
    x = torch.tensor([[[100., 200.], [4., 7.]]])
    assert net(x).item() == 34.


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_sequence_rejects_nonboolean_skip(value):
    with pytest.raises(ValueError, match='linear_skip'):
        SequenceNet(2, 3, 1, linear_skip=value)


@pytest.mark.parametrize('family,value', [('RF', True), ('ANN', True), ('LSTM', 1), ('LSTM', 'yes')])
def test_regressor_rejects_invalid_skip(family, value):
    model = RunoffRegressor(family=family, lstm_linear_skip=value)
    with pytest.raises(ValueError, match='lstm_linear_skip'):
        model.fit(np.ones((12, 3, 2)), np.ones((12, 1)))


@pytest.mark.parametrize('outputs', [1, 7])
def test_native_fit_clone_and_restore(outputs, tmp_path):
    import joblib
    rng = np.random.default_rng(2)
    x = rng.uniform(1, 10, (18, 4, 3)).astype('float32')
    y = np.column_stack([x[:, -1, 0] + i for i in range(outputs)])
    model = clone(RunoffRegressor(family='LSTM', lstm_linear_skip=True, top_k=None,
                                  hidden=2, epochs=2, seq_len=3, transform='log'))
    model.fit(x[:12], y[:12])
    prediction = model.predict(x[12:])
    assert prediction.shape == (6, outputs) and np.isfinite(prediction).all()
    assert model.estimator_.skip_weight.numel() + model.estimator_.skip_bias.numel() == outputs*4
    assert model.estimator_.skip_weight.norm() > 0
    path = tmp_path/'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_array_equal(prediction, joblib.load(path).predict(x[12:]))


def test_default_false_and_legacy_prediction_are_exact():
    x = np.random.default_rng(2).uniform(1, 10, (18, 4, 3)).astype('float32')
    y = x[:, -1, :1]
    old = RunoffRegressor(family='LSTM', hidden=2, epochs=2, seq_len=3).fit(x, y)
    explicit = clone(old).set_params(lstm_linear_skip=False).fit(x, y)
    assert not hasattr(old.estimator_, 'skip_weight')
    np.testing.assert_array_equal(old.predict(x), explicit.predict(x))
    assert old.loss_history_ == explicit.loss_history_
    prediction = old.predict(x)
    del old.lstm_linear_skip
    np.testing.assert_array_equal(prediction, old.predict(x))


def test_research_records_actual_lstm_skip(tmp_path):
    from research_h1 import run_stage
    import pandas as pd
    candidates = [dict(id=str(skip), family='LSTM', mode='daily', transform='log', residual=False,
                       params=dict(hidden=2, epochs=1, seq_len=3, lstm_linear_skip=skip)) for skip in (False, True)]
    dates = pd.date_range('2000-01-01', '2004-12-31')
    raw = pd.DataFrame({name: 10 + np.sin(np.arange(len(dates))/13) + np.arange(len(dates))*.001
                        for name in ['Discharge','Dayl','Prcp','Srad','Swe','Tmax','Tmin','Vp']}, index=dates)
    run_stage('features', tmp_path/'study', seed=42, candidates=candidates, raw=raw)
    import json
    fits = json.loads((tmp_path/'study/fits.json').read_text())
    for row in fits:
        assert row['lstm_linear_skip'] == (row['candidate'] == 'True')
        assert row['lstm_skip_parameters'] == (8 if row['lstm_linear_skip'] else 0)
        assert (row['lstm_skip_weight_norm'] is not None) == row['lstm_linear_skip']
        assert row['lstm_readout_width'] == 2
