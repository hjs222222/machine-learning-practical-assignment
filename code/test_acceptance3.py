# -*- coding: utf-8 -*-
"""任务1至7+第三阶段时间边界、预处理与预测定义检查。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from acceptance3_model import RunoffRegressor, load_dataset, restore_model
from acceptance3_train import ROOT, metrics_table, prepare
from feature_engineering import build_features


class Acceptance3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.path, cls.y, cls.calibration, cls.train, cls.test, cls.folds = prepare(cls.folder)
        cls.data = load_dataset(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        load_dataset.cache_clear()
        cls.temporary.cleanup()

    def test_sample_boundaries(self):
        days = self.data["origin_days"]
        self.assertLessEqual(int((days[self.train] + 7).max()),
                             int(np.datetime64("2003-12-31").astype("int64")))
        self.assertEqual(len(self.test), 359)
        for ti, vi in self.folds:
            self.assertLess(days[ti[-1]] + 7, days[vi[0]])
        for ti, vi in TimeSeriesSplit(n_splits=3, gap=7).split(self.calibration):
            self.assertLess(days[self.calibration[ti[-1]]] + 7,
                            days[self.calibration[vi[0]]])

    def test_feature_causality(self):
        raw = pd.read_csv(ROOT / "Teacher Resources & Requirements/01047000.csv", parse_dates=["Date"])
        first, names = build_features(raw, verbose=False)
        changed = raw.copy()
        changed.loc[changed.Date >= "2003-01-01", changed.columns[1:]] *= 3
        second, _ = build_features(changed, verbose=False)
        mask = first.Date < "2003-01-01"
        np.testing.assert_array_equal(first.loc[mask, names], second.loc[mask, names])

    def test_fold_local_preprocessing_and_native_multi(self):
        ids = self.calibration[:120]
        est = RunoffRegressor(str(self.path), family="RF", horizon=7,
                               n_estimators=3, max_depth=2, device="cpu")
        clone(est)
        est.fit(ids[:, None], self.y[ids])
        np.testing.assert_allclose(est.y_scaler_.mean_, np.log(self.y[ids]).mean(axis=0))
        actual = self.data["features"][self.data["positions"][ids]][:, est.selected_]
        np.testing.assert_allclose(est.x_scaler_.mean_, actual.mean(axis=0))
        pred = est.predict(self.test[:5, None])
        self.assertEqual(pred.shape, (5, 7))
        self.assertEqual(est.model_.n_outputs_, 7)
        folder = self.folder / "rf_model"
        est.save(folder)
        recovered = restore_model(folder, device="cpu")
        np.testing.assert_allclose(pred, recovered.predict(self.test[:5, None]))

    def test_r2_is_not_squared_correlation(self):
        truth = [1., 2., 3., 4.]
        predicted = [11., 12., 13., 14.]
        rows = [{"strategy": "direct", "model": "check", "h": 1,
                 "truth": y, "pred": p} for y, p in zip(truth, predicted)]
        score = metrics_table(rows).iloc[0]
        self.assertLess(score.R2, 0)
        self.assertAlmostEqual(score.RMSE, 10)
        self.assertAlmostEqual(score.MAE, 10)

    def test_lstm_early_stop_is_purged(self):
        ids = self.calibration[:120]
        est = RunoffRegressor(str(self.path), family="LSTM", horizon=7,
                               units=(8,), max_epochs=2, patience=1, device="cpu")
        est.fit(ids[:, None], self.y[ids])
        audit = est.audit_
        days = self.data["origin_days"]
        self.assertLess(days[audit["early_train_last_id"]] + 7,
                        days[audit["early_validation_first_id"]])
        self.assertIn(est.epochs_, [1, 2])
        self.assertEqual(len(audit["history"]), est.epochs_)
        np.testing.assert_allclose(est.y_scaler_.mean_, np.log(self.y[ids]).mean(axis=0))

    def test_lstm_native_multi_and_recovery(self):
        ids = self.calibration[:90]
        est = RunoffRegressor(str(self.path), family="LSTM", horizon=7,
                               units=(8,), fixed_epochs=1, device="cpu")
        est.fit(ids[:, None], self.y[ids])
        pred = est.predict(self.test[:3, None])
        self.assertEqual(pred.shape, (3, 7))
        self.assertEqual(est.model_.head[-1].out_features, 7)
        folder = self.folder / "lstm_model"
        est.save(folder)
        recovered = restore_model(folder, device="cpu")
        np.testing.assert_allclose(pred, recovered.predict(self.test[:3, None]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
