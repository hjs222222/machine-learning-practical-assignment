"""新训练入口的因果性、残差初值和模型恢复测试。"""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from acceptance3_model import load_dataset, restore_model
from acceptance3_v2_train import prepare_v2
from runoff_training.model import LaggedLSTMRegressor
from runoff_training.train import grids, year_folds, get_target


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path, _, cls.train, cls.test, _ = prepare_v2(Path(cls.temp.name))
        cls.data = load_dataset(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        load_dataset.cache_clear()
        cls.temp.cleanup()

    def test_grid_clone(self):
        self.assertEqual(len(list(ParameterGrid(grids()))), 14)
        model = LaggedLSTMRegressor(str(self.path))
        self.assertEqual(clone(model).get_params(), model.get_params())

    def test_time_purging(self):
        days = self.data['origin_days']
        folds = year_folds(self.data, self.train)
        self.assertEqual(len(folds), 4)
        self.assertEqual(sum(len(vi) for _, _, vi in folds), 358)
        for _, ti, vi in folds:
            self.assertLess(days[ti[-1]] + 7, days[vi[0]])
            self.assertLessEqual(days[vi[-1]] + 7, np.datetime64('2003-12-31').astype('int64'))

    def test_residual_starts_at_persistence(self):
        ids = self.train[:400]
        for h in [1, 7]:
            y = get_target(self.data, 'direct' if h == 1 else 'multi', h)[ids]
            for transform in ['identity', 'signed_log']:
                model = LaggedLSTMRegressor(str(self.path), horizon=h, units=(8,),
                    fixed_epochs=1, input_transform=transform)
                model.output_dim_ = 1 if y.ndim == 1 else 7
                model._preprocess_fit(ids, y)
                model.model_ = model._network().eval()
                q = self.data['features'][self.data['positions'][ids[-5:]], -3]
                p = model.predict(ids[-5:, None]).reshape(5, -1)
                np.testing.assert_allclose(p, np.repeat(q[:, None], model.output_dim_, axis=1),
                                           rtol=3e-6, atol=.002)

    def test_fold_only_scaler_and_position_mapping(self):
        ids = self.train[:400]
        model = LaggedLSTMRegressor(str(self.path), fixed_epochs=1)
        model._preprocess_fit(ids, self.data['y'][ids])
        positions = self.data['positions'][ids]
        rows = np.unique((positions[:, None] + np.arange(-29, 1)).ravel())
        expected = model._transform_x(self.data['features'][rows][:, model.selected_])
        np.testing.assert_allclose(model.x_scaler_.mean_, expected.mean(axis=0))
        self.assertEqual(model._inputs(ids[:5]).shape, (5, 30, 17))

    def test_fit_native_shape_and_restore(self):
        ids = self.train[:400]
        for strategy, h in [('direct', 1), ('multi', 7)]:
            y = get_target(self.data, strategy, h)[ids]
            model = LaggedLSTMRegressor(str(self.path), horizon=h, units=(8,), fixed_epochs=1)
            model.fit(ids[:, None], y)
            p = model.predict(self.test[:7, None])
            self.assertEqual(p.shape, (7,) if strategy == 'direct' else (7, 7))
            with tempfile.TemporaryDirectory() as tmp:
                model.save(tmp)
                np.testing.assert_allclose(restore_model(tmp).predict(self.test[:7, None]), p,
                                           rtol=1e-6, atol=1e-4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
