"""原始尺度损失：梯度、基线等价、折内统计、恢复与选择门槛回归。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

from runoff_training.controlled_data import build_data, load_data, outer_split, inner_splits
from runoff_training.strategy_model import StrategyLSTM
from runoff_training.loss_model import LossLSTM, restore_loss, training_loss
from runoff_training.loss_train import protocol, validate_protocol, choose
from verify_runoff_loss import gate_result, select_rows, verify, compare_legacy_predictions

ROOT = Path(__file__).resolve().parents[1]


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.dataset = cls.folder/'historical.npz'
        build_data(ROOT/'Teacher Resources & Requirements/01047000.csv', cls.dataset)
        cls.data = load_data(cls.dataset)
        cls.ids = np.arange(300)
        cls.y = cls.data['y'][cls.ids, :1]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_fixed_budgets_and_smoke_cannot_be_relabelled(self):
        formal, smoke = protocol(False, 'cpu'), protocol(True, 'cpu')
        self.assertEqual((formal['fits'], formal['outer_models']), (720, 72))
        self.assertEqual((smoke['fits'], smoke['outer_models']), (96, 24))
        bad = copy.deepcopy(smoke)
        bad['smoke'] = False
        with self.assertRaises(ValueError):
            validate_protocol(bad)
        bad = copy.deepcopy(formal)
        bad['fixed']['units'] = 64
        with self.assertRaises(ValueError):
            validate_protocol(bad)
        self.assertEqual(formal['horizons'], [1, 3, 7])
        self.assertEqual(formal['years'], [2002, 2003])

    def test_physical_loss_value_and_gradient(self):
        output = torch.tensor([[.2], [-.3]], dtype=torch.float32, requires_grad=True)
        mean, scale, raw_scale = [torch.tensor([v], dtype=torch.float64) for v in [2., .7, 4.]]
        raw = torch.tensor([[9.], [3.]], dtype=torch.float64)
        loss = training_loss(output, output.detach(), raw, mean, scale, raw_scale, 'physical')
        expected = torch.square((torch.exp(output.double()*scale+mean)-raw)/raw_scale).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        physical = torch.exp(output.detach().double()*scale+mean)
        grad = (2/len(raw)*(physical-raw)*physical*scale/raw_scale.square()).float()
        torch.testing.assert_close(output.grad, grad)
        self.assertGreater(float(output.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            training_loss(output, output.detach(), raw, mean, scale, raw_scale, 'unknown')

    def test_log_candidate_exactly_matches_frozen_implementation(self):
        for residual in [False, True]:
            old = StrategyLSTM(str(self.dataset), fixed_epochs=2, seed=42, residual=residual).fit(self.ids, self.y)
            new = LossLSTM(str(self.dataset), fixed_epochs=2, seed=42, residual=residual, loss_space='log').fit(self.ids, self.y)
            np.testing.assert_equal(old.predict(np.arange(301, 305)), new.predict(np.arange(301, 305)))
            for key, value in old.model_.state_dict().items():
                torch.testing.assert_close(value, new.model_.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(old.audit_['history'], new.audit_['history'])

    def test_physical_model_training_scaling_and_restore(self):
        model = LossLSTM(str(self.dataset), fixed_epochs=2, residual=True, loss_space='physical').fit(self.ids, self.y)
        np.testing.assert_allclose(model.y_scaler_.mean_, np.log(self.y).mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(model.audit_['raw_target_scale'], self.y.std(axis=0), rtol=1e-12)
        self.assertEqual(model.audit_['train_last_h7_target'], str(np.datetime64(int(self.data['origin_days'][299]+7), 'D')))
        self.assertEqual(model.audit_['n_parameters'], 7617)
        prediction = model.predict(np.arange(301, 305))
        self.assertTrue(np.isfinite(prediction).all() and (prediction > 0).all())
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/'model'
            model.save(folder)
            restored = restore_loss(folder, self.dataset)
            np.testing.assert_equal(prediction, restored.predict(np.arange(301, 305)))
            self.assertEqual(restored.loss_space, 'physical')

    def test_reject_invalid_configuration_and_target(self):
        for kwargs in [dict(batch_size=64.), dict(units=True), dict(loss_space='raw'), dict(pooling='last_mean')]:
            with self.assertRaises(ValueError):
                LossLSTM(str(self.dataset), **kwargs).fit(self.ids, self.y)
        with self.assertRaises(ValueError):
            LossLSTM(str(self.dataset)).fit(self.ids[::-1], self.y)
        with self.assertRaises(ValueError):
            LossLSTM(str(self.dataset)).fit(self.ids, np.zeros_like(self.y))
        estimator = LossLSTM(str(self.dataset), loss_space='physical', residual=True)
        self.assertEqual(clone(estimator).get_params(), estimator.get_params())

    def test_historical_boundaries_and_future_perturbation(self):
        for year, size in [(2002, 546), (2003, 911)]:
            train, valid = outer_split(self.data, year)
            self.assertEqual((len(train), len(valid)), (size, 358))
            for ti, vi in inner_splits(self.data, train):
                self.assertLess(self.data['origin_days'][train[ti[-1]]]+7, self.data['origin_days'][train[vi[0]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            raw = pd.read_csv(ROOT/'Teacher Resources & Requirements/01047000.csv')
            raw.loc[raw.Date >= '2004-01-01', ['Discharge', 'Prcp']] *= 10
            raw.to_csv(path/'perturbed.csv', index=False)
            build_data(path/'perturbed.csv', path/'other.npz')
            other = load_data(path/'other.npz')
            for key in self.data:
                np.testing.assert_equal(other[key], self.data[key])

    def test_inner_choice_and_frozen_row_selection(self):
        self.assertEqual(choose(dict(plain_log=100., residual_log=99.5, plain_physical=99.3, residual_physical=99.2)), 'plain_log')
        self.assertEqual(choose(dict(plain_log=100., residual_log=100., plain_physical=90., residual_physical=89.5)), 'plain_physical')
        frame = pd.DataFrame([dict(year=2002, h=1, variant='plain_log', RMSE=1.),
                              dict(year=2002, h=1, variant='plain_physical', RMSE=100.)])
        selected = select_rows(frame, [dict(year=2002, h=1, selected_variant='plain_physical')], 'selected_variant')
        self.assertEqual(selected.RMSE.tolist(), [100.])

    def test_gate_reports_seed_horizon_and_mae_constraints(self):
        spec = protocol(False, 'cpu')
        rows, selections = [], []
        for year in spec['years']:
            for h in spec['horizons']:
                selections.append(dict(year=year, h=h, selected_variant='plain_physical', selected_log='plain_log'))
                for seed in spec['seeds']:
                    for variant, rmse in [('plain_log', 100.), ('plain_physical', 95.)]:
                        rows.append(dict(year=year, h=h, seed=seed, variant=variant, RMSE=rmse, MAE=50., R2=.2))
        frame = pd.DataFrame(rows)
        gate, _ = gate_result(frame, selections, spec)
        self.assertTrue(gate['advance'])
        frame.loc[frame.variant == 'plain_physical', 'MAE'] = 56.
        self.assertFalse(gate_result(frame, selections, spec)[0]['advance'])
        frame.loc[frame.variant == 'plain_physical', 'MAE'] = 50.
        frame.loc[(frame.variant == 'plain_physical') & (frame.h == 7), 'RMSE'] = 111.
        self.assertFalse(gate_result(frame, selections, spec)[0]['advance'])

    def test_legacy_comparison_ignores_dataframe_index_not_dates(self):
        previous = pd.DataFrame(dict(origin_date=['2002-01-01', '2002-01-02'],
            target_date=['2002-01-02', '2002-01-03'], truth=[2., 3.], pred=[2.1, 3.1]))
        current = previous.copy()
        current.index = [700, 701]
        self.assertEqual(compare_legacy_predictions(current, previous), 0.)
        current.loc[700, 'truth'] = 5.
        with self.assertRaises(AssertionError):
            compare_legacy_predictions(current, previous)

    def test_verifier_rejects_unfinished_and_implicit_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for status, smoke in [('running', False), ('completed', True)]:
                (run/'manifest.json').write_text(json.dumps(dict(status=status, protocol=protocol(smoke, 'cpu'))), encoding='utf-8')
                with self.assertRaises(ValueError):
                    verify(run)


if __name__ == '__main__':
    unittest.main()
