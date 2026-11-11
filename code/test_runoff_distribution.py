"""分布 LSTM 的损失、条件均值、训练边界与恢复检查。"""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from acceptance3_model import load_dataset, restore_model
from acceptance3_v2_train import prepare_v2
from runoff_training.distribution import DistributionLSTMRegressor, bounded_variance, gaussian_nll


class DistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path, _, cls.train, _, _ = prepare_v2(Path(cls.temp.name))
        cls.data = load_dataset(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        load_dataset.cache_clear()
        cls.temp.cleanup()

    def test_clone(self):
        model = DistributionLSTMRegressor(str(self.path), objective='lognormal')
        self.assertEqual(model.get_params(), clone(model).get_params())

    def test_budget_and_three_seed_gate(self):
        from runoff_training.continue_train import grids, gate_table
        self.assertEqual(len(list(ParameterGrid(grids()))), 16)
        rows = []
        for seed in [42, 43, 44]:
            for fold in range(4):
                rows.append(dict(strategy='multi', task_h=7, seed=seed, fold=fold,
                    candidate_RMSE=90. if seed != 44 else 120., candidate_MAE=40.,
                    original_RMSE=100., original_MAE=50., previous_RMSE=100., previous_MAE=50.))
        gates = gate_table(rows)
        self.assertEqual(len(gates), 2)
        self.assertFalse(any(g['passed_all_seeds'] for g in gates))
        self.assertEqual([r['passed'] for r in gates[0]['seed_results']], [True, True, False])

    def test_verify_rejects_smoke_and_unfinished(self):
        import json
        from verify_runoff_continued import verify
        for state in [dict(status='completed', smoke=True), dict(status='running', smoke=False)]:
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp)/'manifest.json').write_text(json.dumps(state), encoding='utf-8')
                with self.assertRaises(AssertionError):
                    verify(Path(tmp))

    def test_variance_and_nll(self):
        raw = torch.tensor([[-100., 0., 100.]], requires_grad=True)
        variance = bounded_variance(raw)
        self.assertTrue((variance >= .02).all() and (variance <= 2.).all())
        output = torch.cat([torch.zeros_like(raw), raw], dim=-1)
        loss = gaussian_nll(output, torch.ones_like(raw))
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(raw.grad).all())

    def test_conditional_mean_formula_and_native_output(self):
        ids = self.train[:400]
        for strategy in ['direct', 'multi']:
            y = self.data['y'][ids, 0] if strategy == 'direct' else self.data['y'][ids]
            model = DistributionLSTMRegressor(str(self.path), residual=True, units=(8,))
            model.output_dim_ = 1 if y.ndim == 1 else 7
            model._preprocess_fit(ids, y)
            model.model_ = model._network().eval()
            q = self.data['features'][self.data['positions'][ids[:5]], -3]
            expected = q[:, None] * np.exp(.125 * model.y_scaler_.scale_ ** 2)
            np.testing.assert_allclose(model.predict(ids[:5, None]).reshape(5, -1), expected,
                                       rtol=4e-6, atol=.002)

    def test_fit_determinism_and_restore(self):
        ids, vi = self.train[:400], self.train[420:425]
        for objective in ['log_mse', 'lognormal']:
            for outputs in [1, 7]:
                y = self.data['y'][ids, 0] if outputs == 1 else self.data['y'][ids]
                model = DistributionLSTMRegressor(str(self.path), units=(8,), fixed_epochs=1,
                                                  objective=objective, residual=False)
                model.fit(ids[:, None], y)
                pred = model.predict(vi[:, None])
                self.assertEqual(pred.shape, (5,) if outputs == 1 else (5, 7))
                repeat = clone(model).fit(ids[:, None], y)
                np.testing.assert_array_equal(repeat.predict(vi[:, None]), pred)
                with tempfile.TemporaryDirectory() as tmp:
                    model.save(tmp)
                    np.testing.assert_allclose(restore_model(tmp).predict(vi[:, None]), pred,
                                               rtol=1e-6, atol=1e-4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
