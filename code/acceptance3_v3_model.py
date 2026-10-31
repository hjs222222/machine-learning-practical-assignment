# -*- coding: utf-8 -*-
"""第三轮：固定epoch训练与原尺度目标；不改动前两轮实现。"""
import json
from pathlib import Path

import numpy as np

from acceptance3_v2_model import V2Regressor


class V3Regressor(V2Regressor):
    def _target(self, y):
        if self.target_transform == 'raw':
            return np.asarray(y).reshape(len(y), -1)
        return super()._target(y)

    def _inverse(self, z):
        if self.target_transform == 'raw':
            # 原尺度分支不根据验证或测试误差截断预测。
            return self.y_scaler_.inverse_transform(np.asarray(z))
        return super()._inverse(z)

    def fit(self, X, y):
        if self.family == 'LSTM' and self.fixed_epochs is None:
            raise ValueError('第三轮必须显式指定固定epoch，禁止隐式内部早停')
        return super().fit(X, y)

    def save(self, folder):
        super().save(folder)
        path = Path(folder) / 'inference.json'
        metadata = json.loads(path.read_text(encoding='utf-8'))
        metadata['inverse_domain_rule'] = ('raw: inverse_standardize(z), no clipping' if self.target_transform == 'raw'
            else 'sqrt: max(inverse_standardize(z),0)^2; log: exp(inverse_standardize(z))')
        metadata['fixed_epoch_protocol'] = True
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
