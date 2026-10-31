# -*- coding: utf-8 -*-
"""第三轮固定epoch、全年开发边界及原尺度模型恢复测试。"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.base import clone
import torch

from acceptance3_model import load_dataset, restore_model
from acceptance3_v2_train import prepare_v2
from acceptance3_v3_model import V3Regressor
from acceptance3_v3_train import development_splits, fixed_grid, frame_metrics


class V3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.path, cls.y, cls.train, cls.test, _ = prepare_v2(cls.folder)
        cls.data = load_dataset(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        load_dataset.cache_clear()
        cls.temp.cleanup()

    def test_annual_purged_splits_and_no_test(self):
        folds = development_splits(self.data)
        self.assertEqual([len(va) for _, _, va in folds], [358, 358])
        self.assertEqual([len(ti) for _, ti, _ in folds], [665, 1030])
        days = self.data['origin_days']
        for year, ti, vi in folds:
            self.assertLess(days[ti[-1]] + 7, days[vi[0]])
            self.assertTrue((days[vi] + 7 <= np.datetime64(f'{year}-12-31').astype('int64')).all())
            self.assertFalse(np.isin(vi, self.test).any())

    def test_requires_explicit_epochs_and_clone(self):
        model = V3Regressor(str(self.path), units=(8,), fixed_epochs=2)
        self.assertEqual(clone(model).fixed_epochs, 2)
        model.fit(self.train[:100,None], self.y[:100, 0])
        self.assertEqual(len(model.audit_['history']), 2)
        self.assertNotIn('early_history', model.audit_)
        with self.assertRaises(ValueError):
            V3Regressor(str(self.path)).fit(self.train[:100,None], self.y[:100,0])

    def test_raw_native_multi_restore(self):
        model = V3Regressor(str(self.path), units=(8,), fixed_epochs=2,
                            target_transform='raw', feature_mode='compact', horizon=7)
        model.fit(self.train[:100,None], self.y[:100])
        np.testing.assert_allclose(model.y_scaler_.mean_, self.y[:100].mean(axis=0))
        pred = model.predict(self.train[100:104,None])
        self.assertEqual(pred.shape, (4,7))
        self.assertEqual(model.model_.head[-1].out_features,7)
        folder = self.folder / 'raw_multi'
        model.save(folder)
        np.testing.assert_allclose(pred,restore_model(folder).predict(self.train[100:104,None]))

    def test_grid_explicit_epoch_and_raw_profile(self):
        self.assertEqual(fixed_grid('A0')['fixed_epochs'],[10,20,40])
        self.assertEqual(fixed_grid('A2')['target_transform'],['raw'])
        self.assertEqual(fixed_grid('A3')['mixed_weight'],[.1])

    def test_segment_metrics_training_thresholds(self):
        year, ti, vi = development_splits(self.data)[0]
        rows, predictions = frame_metrics(self.data,ti,vi,self.y[vi],self.y,'multi',7,'A0',10,42,year)
        yearly = [r for r in rows if r['segment']=='year']
        self.assertEqual(len(yearly),7)
        self.assertTrue(all(r['RMSE']==0 for r in yearly))
        self.assertEqual(len(predictions),358*7)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
