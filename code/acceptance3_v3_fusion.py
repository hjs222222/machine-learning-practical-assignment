# -*- coding: utf-8 -*-
"""第三轮原生21输入、7输出的保守元层；不包装单输出估计器。"""
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class ConservativeFusion(RegressorMixin, BaseEstimator):
    def __init__(self, kind='RF', alpha=100., strength=.25):
        self.kind = kind
        self.alpha = alpha
        self.strength = strength

    @staticmethod
    def blocks(X):
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != 21 or not np.isfinite(x).all():
            raise ValueError('元层必须接收有限的ANN/RF/LSTM各7列，共21列')
        return x.reshape(len(x), 3, 7)

    def fit(self, X, y):
        b = self.blocks(X)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(b), 7):
            raise ValueError('元层目标必须为原生七输出')
        self.n_features_in_ = 21
        if self.kind == 'convex':
            scale = np.maximum(y.std(axis=0), 1e-6)
            def loss(w):
                return np.square((np.einsum('nkh,k->nh', b, w) - y) / scale).mean()
            result = minimize(loss, np.ones(3) / 3, method='SLSQP',
                bounds=[(0., 1.)] * 3,
                constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
                options={'ftol': 1e-10, 'maxiter': 300})
            if not result.success:
                raise ValueError('凸融合求解失败：' + result.message)
            self.weights_ = np.maximum(result.x, 0)
            self.weights_ /= self.weights_.sum()
        elif self.kind == 'ridge_RF':
            self.scaler_ = StandardScaler().fit(X)
            self.ridge_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(X), y)
        elif self.kind not in ['RF', 'Mean']:
            raise ValueError(self.kind)
        return self

    def predict(self, X):
        b = self.blocks(X)
        if self.kind == 'RF':
            return b[:, 1, :]
        if self.kind == 'Mean':
            return b.mean(axis=1)
        if self.kind == 'convex':
            return np.einsum('nkh,k->nh', b, self.weights_)
        # 原尺度Ridge保留负值；预先固定收缩，不按测试误差截断。
        ridge = self.ridge_.predict(self.scaler_.transform(X))
        return b[:, 1, :] + self.strength * (ridge - b[:, 1, :])


def fusion_candidates(smoke=False):
    candidates = [dict(kind='RF'), dict(kind='Mean'), dict(kind='convex')]
    candidates += [dict(kind='ridge_RF', alpha=a, strength=s)
                   for a in ([100.] if smoke else [10., 100., 1000.])
                   for s in ([.25] if smoke else [.25, .5])]
    return candidates
