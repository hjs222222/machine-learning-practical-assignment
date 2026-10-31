# -*- coding: utf-8 -*-
"""第二轮训练的时间边界、原尺度早停和模型恢复测试。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.base import clone

from acceptance3_model import load_dataset, restore_model
from acceptance3_v2_model import V2Regressor
from acceptance3_v2_train import inner_splits, meta_fit, meta_predict, prepare_v2


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.path, cls.y, cls.train, cls.test, cls.folds = prepare_v2(cls.folder)
        cls.data = load_dataset(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        load_dataset.cache_clear()
        cls.temp.cleanup()

    def test_nested_date_boundaries(self):
        days = self.data['origin_days']
        for _, ids, val in self.folds:
            self.assertLess(days[ids[-1]]+7, days[val[0]])
            for ti, vi in inner_splits(ids, days):
                self.assertGreaterEqual(len(ti),478)
                self.assertGreaterEqual(len(vi),90)
                self.assertLess(days[ids[ti[-1]]]+7, days[ids[vi[0]]])
        self.assertEqual(len(np.unique(np.concatenate([v for _,_,v in self.folds]))),723)

    def test_explicit_current_q_and_fold_scale(self):
        ids=self.train[:478]
        model=V2Regressor(str(self.path),family='RF',horizon=7,n_estimators=3,max_depth=2)
        clone(model)
        model.fit(ids[:,None],self.y[ids])
        self.assertIn('Discharge_current',model.feature_names_)
        np.testing.assert_allclose(model.y_scaler_.mean_,np.log(self.y[ids]).mean(axis=0))
        self.assertEqual(model.predict(self.test[:3,None]).shape,(3,7))

    def test_sqrt_native_multi_restore(self):
        ids=self.train[:478]
        model=V2Regressor(str(self.path),family='RF',horizon=7,n_estimators=3,max_depth=2,
                          target_transform='sqrt',feature_mode='compact')
        model.fit(ids[:,None],self.y[ids])
        np.testing.assert_allclose(model.y_scaler_.mean_,np.sqrt(self.y[ids]).mean(axis=0))
        pred=model.predict(self.test[:3,None])
        self.assertTrue((pred>=0).all())
        folder=self.folder/'sqrt_model'; model.save(folder)
        np.testing.assert_allclose(pred,restore_model(folder).predict(self.test[:3,None]))

    def test_early_stop_original_scale_metrics(self):
        ids=self.train[:478]
        model=V2Regressor(str(self.path),family='LSTM',horizon=7,units=(8,),max_epochs=2,patience=1)
        model.fit(ids[:,None],self.y[ids])
        audit=model.audit_
        days=self.data['origin_days']
        self.assertLess(days[audit['early_train_last_id']]+7,days[audit['early_validation_first_id']])
        for row in audit['early_history']:
            self.assertTrue(np.isfinite(row['val_RMSE']))
            self.assertIn('val_MAE',row)
        self.assertEqual(len(audit['history']),model.epochs_)
        folder=self.folder/'lstm_model'; model.save(folder)
        np.testing.assert_allclose(model.predict(self.test[:3,None]),restore_model(folder).predict(self.test[:3,None]))

    def test_log_meta_positive_native_multi(self):
        rng=np.random.default_rng(42)
        x=rng.uniform(1,10,(120,21)); y=rng.uniform(1,10,(120,7))
        model=meta_fit(x,y,'log',100)
        pred=meta_predict(model,x,'log')
        self.assertEqual(pred.shape,(120,7))
        self.assertEqual(model[-1].coef_.shape,(7,21))
        self.assertTrue((pred>0).all())


if __name__=='__main__':
    import torch
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
