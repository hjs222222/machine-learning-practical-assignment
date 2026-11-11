"""Single-output boosting with an explicit causal current-flow baseline."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def _current_flow(X):
    values = np.asarray(X)
    if (values.ndim != 2 or values.shape[1] < 1 or
            not np.isfinite(values[:, 0]).all() or (values[:, 0] < 0).any()):
        raise ValueError('First input must be finite nonnegative current flow')
    return values[:, 0].astype(np.float64)


class SqrtResidualHistGradientBoostingRegressor(HistGradientBoostingRegressor):
    def fit(self, X, y, sample_weight=None):
        if self.early_stopping is not False:
            raise ValueError('Residual target requires early_stopping=False')
        current = _current_flow(X)
        truth = np.asarray(y, dtype=np.float64)
        if truth.ndim != 1 or not np.isfinite(truth).all() or (truth < 0).any():
            raise ValueError('Target must be finite nonnegative single-output flow')
        return super().fit(X, np.sqrt(truth)-np.sqrt(current), sample_weight=sample_weight)

    def predict(self, X):
        current = _current_flow(X)
        return np.maximum(super().predict(X)+np.sqrt(current), 0)**2


class RawResidualHistGradientBoostingRegressor(HistGradientBoostingRegressor):
    def fit(self, X, y, sample_weight=None):
        if self.early_stopping is not False:
            raise ValueError('Residual target requires early_stopping=False')
        current = _current_flow(X)
        truth = np.asarray(y, dtype=np.float64)
        if truth.ndim != 1 or not np.isfinite(truth).all() or (truth < 0).any():
            raise ValueError('Target must be finite nonnegative single-output flow')
        return super().fit(X, truth-current, sample_weight=sample_weight)

    def predict(self, X):
        current = _current_flow(X)
        return np.maximum(super().predict(X)+current, 0)
