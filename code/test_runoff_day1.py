"""第1天专项：因果趋势、增量恢复、季节四折、合并R²和固定推进门槛。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import r2_score

from runoff_training.controlled_data import build_data, load_data, outer_split
from runoff_training.day1_model import Day1Regressor, DeltaNet, tabular, restore_day1
from runoff_training.day1_train import protocol, validate_protocol, folds_for, pooled_score, choose, model_for, scorer
from verify_runoff_day1 import verify, check_audit, check_model, selected_rows, gate_result

ROOT = Path(__file__).resolve().parents[1]


class Day1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.dataset = cls.folder/'historical.npz'
        build_data(ROOT/'Teacher Resources & Requirements/01047000.csv', cls.dataset)
        cls.data = load_data(cls.dataset)
        cls.ids = np.arange(160)
        cls.y = cls.data['y'][cls.ids, 0]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_protocol_and_smoke_relabelling(self):
        formal, smoke = protocol(False, 'cpu'), protocol(True, 'cpu')
        self.assertEqual((formal['fits'], formal['outer_models']), (300, 60))
        self.assertEqual((smoke['fits'], smoke['outer_models']), (40, 8))
        for spec in [formal, smoke]:
            validate_protocol(spec)
        bad = copy.deepcopy(smoke)
        bad['smoke'] = False
        with self.assertRaises(ValueError):
            validate_protocol(bad)
        bad = copy.deepcopy(formal)
        bad['gate']['min_R2_gain'] = 0
        with self.assertRaises(ValueError):
            validate_protocol(bad)

    def test_tabular_positions_values_and_future_causality(self):
        ids = np.array([20, 40])
        t = self.data['positions'][ids]
        q = self.data['discharge']
        result = tabular(self.data, ids)
        self.assertEqual(result.shape, (2, 22))
        np.testing.assert_equal(result[:, 0], q[t])
        np.testing.assert_equal(result[:, 5], q[t]-q[t-1])
        np.testing.assert_equal(result[:, 10], [q[k-89:k+1].mean() for k in t])
        rain = np.expm1(self.data['features'][:, 1])
        np.testing.assert_allclose(result[:, 14], [rain[k-29:k+1].sum() for k in t])
        other = {k:v.copy() for k,v in self.data.items()}
        other['discharge'][t[-1]+1:] *= 100
        other['features'][t[-1]+1:] *= 100
        np.testing.assert_equal(tabular(other, ids), result)

    def test_2004_perturbation_cannot_affect_historical_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            raw = pd.read_csv(ROOT/'Teacher Resources & Requirements/01047000.csv')
            raw.loc[raw.Date >= '2004-01-01', ['Discharge', 'Prcp']] *= 10
            raw.to_csv(path/'future.csv', index=False)
            build_data(path/'future.csv', path/'other.npz')
            other = load_data(path/'other.npz')
            for key in other:
                np.testing.assert_equal(other[key], self.data[key])

    def test_four_folds_purge_minimum_and_unique_validation(self):
        for year in [2002, 2003]:
            train, valid = outer_split(self.data, year)
            folds = folds_for(self.data, train)
            self.assertEqual(len(folds), 4)
            self.assertEqual(len(np.unique(np.concatenate([vi for _,vi in folds]))), 360)
            for ti,vi in folds:
                self.assertGreaterEqual(len(ti), 150)
                self.assertEqual(len(vi), 90)
                self.assertLess(self.data['origin_days'][train[ti[-1]]]+7, self.data['origin_days'][train[vi[0]]])
            self.assertLess(self.data['origin_days'][train[-1]]+7, self.data['origin_days'][valid[0]])

    def test_pooled_R2_matches_concatenated_predictions_not_fold_mean(self):
        ys = [np.array([1., 2., 3.])+offset for offset in [0., 10., 20., 30.]]
        ps = [y+error for y,error in zip(ys, [.1, .2, .5, 1.])]
        scores = [r2_score(y,p) for y,p in zip(ys,ps)]
        errors = [np.square(y-p).sum() for y,p in zip(ys,ps)]
        pooled = pooled_score(errors, ys)
        self.assertAlmostEqual(pooled, r2_score(np.concatenate(ys), np.concatenate(ps)), places=12)
        self.assertGreater(abs(pooled-np.mean(scores)), .1)
        with self.assertRaises(ValueError):
            pooled_score([np.nan]*4, ys)

    def test_constant_fold_direct_SSE_is_not_lost(self):
        class FixedPrediction:
            def predict(self, X):
                return np.asarray(X, dtype=float).reshape(-1, 1)
        ys = [np.ones(3), np.arange(3.), np.arange(3.)+2, np.arange(3.)+4]
        ps = [ys[0]+5, *ys[1:]]
        score = [scorer(FixedPrediction(), p, y) for p,y in zip(ps,ys)]
        self.assertEqual(score[0]['R2'], 0.)
        self.assertEqual(score[0]['neg_SSE'], -75.)
        pooled = pooled_score([-s['neg_SSE'] for s in score], ys)
        self.assertAlmostEqual(pooled, r2_score(np.concatenate(ys), np.concatenate(ps)), places=12)
        self.assertLess(pooled, 1.)

    def test_all_families_scaler_audit_clone_and_restore(self):
        spec = protocol(True, 'cpu')
        for c in spec['candidates']:
            with self.subTest(family=c['family']), tempfile.TemporaryDirectory() as directory:
                model = model_for(self.dataset, spec, c, 42)
                self.assertEqual(model.get_params(), clone(model).get_params())
                model.fit(self.ids, self.y)
                check_audit(model.audit_, model_for(self.dataset, spec, c, 42), self.ids, self.data)
                check_model(model, model_for(self.dataset, spec, c, 42), self.ids, self.data)
                prediction = model.predict(np.arange(170, 174))
                model.save(Path(directory)/'model')
                restored = restore_day1(Path(directory)/'model', self.dataset, 'cpu')
                np.testing.assert_equal(prediction, restored.predict(np.arange(170, 174)))
                altered = copy.deepcopy(model.audit_)
                altered['delta_mean'][0] += 1
                with self.assertRaises(AssertionError):
                    check_audit(altered, model_for(self.dataset, spec, c, 42), self.ids, self.data)

    def test_verifier_recomputes_actual_scaler_not_only_audit(self):
        spec = protocol(True, 'cpu')
        c = spec['candidates'][0]
        model = model_for(self.dataset, spec, c, 42).fit(self.ids, self.y)
        expected = model_for(self.dataset, spec, c, 42)
        model.tab_scaler_.mean_[0] += 1
        with self.assertRaises(AssertionError):
            check_model(model, expected, self.ids, self.data)
        model = model_for(self.dataset, spec, c, 42).fit(self.ids, self.y)
        model.x_scaler_.scale_[0] *= 2
        with self.assertRaises(AssertionError):
            check_model(model, expected, self.ids, self.data)

    def test_delta_zero_head_is_Q_plus_training_delta_mean(self):
        model = Day1Regressor(str(self.dataset), family='ANN', epochs=1).fit(self.ids, self.y)
        for p in model.model_.head[-1].parameters():
            torch.nn.init.zeros_(p)
        ids = np.arange(170, 174)
        expected = tabular(self.data, ids)[:, :1]+model.y_scaler_.mean_
        np.testing.assert_allclose(model.predict(ids), expected, rtol=0, atol=1e-12)
        for family in ['ANN', 'LSTM']:
            net = DeltaNet(family)
            output = net(torch.ones(2, 7, 9), torch.ones(2, 22))
            torch.testing.assert_close(output, torch.zeros(2, 1), rtol=0, atol=0)

    def test_invalid_configuration_target_and_ids_rejected(self):
        for params in [dict(epochs=True), dict(window=7.), dict(trees=5.), dict(leaf=0), dict(alpha=-1.), dict(seed=True)]:
            with self.assertRaises(ValueError):
                Day1Regressor(str(self.dataset), **params).fit(self.ids, self.y)
        with self.assertRaises(ValueError):
            Day1Regressor(str(self.dataset)).fit(self.ids[::-1], self.y)
        with self.assertRaises(ValueError):
            Day1Regressor(str(self.dataset)).fit(self.ids, np.column_stack([self.y, self.y]))
        with self.assertRaises(AssertionError):
            Day1Regressor(str(self.dataset)).fit(self.ids, self.data['y'][self.ids, 1])

    def test_selection_uses_inner_scores_and_fixed_order(self):
        self.assertEqual(choose(dict(a=.75, b=.754)), 'a')
        self.assertEqual(choose(dict(a=.75, b=.756)), 'b')
        frame = pd.DataFrame([dict(year=2002, candidate='a', R2=.99), dict(year=2002, candidate='b', R2=.1)])
        selection = [dict(year=2002, selected_variant='b', family_winners=dict(LSTM='b'))]
        self.assertEqual(selected_rows(frame, selection).R2.tolist(), [.1])
        self.assertEqual(selected_rows(frame, selection, 'LSTM').R2.tolist(), [.1])

    def test_gate_needs_both_years_gain_seed_and_MAE(self):
        spec = protocol(False, 'cpu')
        rows = [dict(year=y, seed=s, R2=.85, R2_old=.8, MAE=50., MAE_old=50.)
                for y in spec['years'] for s in spec['seeds']]
        frame = pd.DataFrame(rows)
        self.assertTrue(gate_result(frame, spec)['advance'])
        bad = frame.copy()
        bad.loc[bad.year == 2003, 'R2'] = .69
        self.assertFalse(gate_result(bad, spec)['advance'])
        bad = frame.copy()
        bad.loc[bad.year == 2002, 'MAE'] = 56.
        self.assertFalse(gate_result(bad, spec)['advance'])
        bad = frame.copy()
        bad.loc[(bad.year == 2003) & (bad.seed != 42), 'R2'] = .79
        self.assertFalse(gate_result(bad, spec)['advance'])
        with self.assertRaises(ValueError):
            gate_result(frame.iloc[:-1], spec)

    def test_verifier_rejects_incomplete_or_unapproved_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for status,smoke in [('running', False), ('completed', True)]:
                (path/'manifest.json').write_text(json.dumps(dict(status=status, protocol=protocol(smoke, 'cpu'))), encoding='utf-8')
                with self.assertRaises(ValueError):
                    verify(path)


if __name__ == '__main__':
    unittest.main()
