"""Ordinary gradient boosting on the causal square-root flow increment."""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from boosting import _current_flow


class SqrtResidualGradientBoostingRegressor(GradientBoostingRegressor):
    def fit(self, X, y, sample_weight=None, monitor=None):
        if self.n_iter_no_change is not None or self.warm_start:
            raise ValueError('Residual target requires early stopping and warm start disabled')
        current = _current_flow(X)
        truth = np.asarray(y, dtype=np.float64)
        if truth.ndim != 1 or len(truth) != len(current) or not np.isfinite(truth).all() or (truth < 0).any():
            raise ValueError('Target must be finite nonnegative single-output flow')
        return super().fit(X, np.sqrt(truth)-np.sqrt(current), sample_weight=sample_weight, monitor=monitor)

    def predict(self, X):
        current = _current_flow(X)
        return np.maximum(super().predict(X)+np.sqrt(current), 0)**2
