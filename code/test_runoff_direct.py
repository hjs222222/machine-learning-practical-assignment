"""直接残差收尾测试：固定预算、因果数据前缀、真实缩放器/参数核验与旧预测对齐。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from analyze_runoff_direct import original_comparison, selected_rows, OLD_PREDICTIONS
from runoff_training.controlled_data import build_data, load_data, inner_splits, outer_split
from runoff_training.direct_data import build_evaluation_data, final_split
from runoff_training.direct_train import protocol, validate_protocol, model_for
from verify_runoff_direct import verify, check_audit, check_model

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'Teacher Resources & Requirements/01047000.csv'


class DirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.history = cls.folder/'historical.npz'
        cls.evaluation = cls.folder/'evaluation.npz'
        build_data(RAW, cls.history)
        build_evaluation_data(RAW, cls.history, cls.evaluation)
        cls.historical = load_data(cls.history)
        cls.data = load_data(cls.evaluation)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixed_budget_and_protocol_rejects_smoke_relabel(self):
        self.assertEqual(protocol('historical', False, 'cpu')['fits'], 864)
        self.assertEqual(protocol('final', False, 'cpu')['fits'], 420)
        self.assertEqual(protocol('historical', True, 'cpu')['fits'], 120)
        self.assertEqual(protocol('final', True, 'cpu')['fits'], 56)
        self.assertEqual(protocol('historical', False, 'cpu')['outer_models'], 108)
        self.assertEqual(protocol('final', False, 'cpu')['outer_models'], 42)
        bad = protocol('final', True, 'cpu')
        bad['smoke'] = False
        with self.assertRaises(ValueError):
            validate_protocol(bad)
        bad = protocol('historical', False, 'cpu')
        bad['fixed']['units'] = 64
        with self.assertRaises(ValueError):
            validate_protocol(bad)

    def test_final_split_same_common_dates_and_purge(self):
        train, valid = final_split(self.data)
        self.assertEqual(len(train), len(self.historical['positions']))
        self.assertEqual((len(train), len(valid)), (1276, 359))
        days = self.data['origin_days']
        self.assertEqual(str(np.datetime64(int(days[valid[0]]), 'D')), '2004-01-01')
        self.assertEqual(str(np.datetime64(int(days[valid[-1]]), 'D')), '2004-12-24')
        self.assertLess(days[train[-1]]+7, days[valid[0]])
        for ti, vi in inner_splits(self.data, train):
            self.assertLess(days[train[ti[-1]]]+7, days[train[vi[0]]])
        for year in [2002, 2003]:
            _, ids = outer_split(self.historical, year)
            self.assertEqual(len(ids), 358)

    def test_future_raw_change_preserves_historical_prefix(self):
        frame = pd.read_csv(RAW)
        mask = frame.Date >= '2004-01-01'
        frame.loc[mask, 'Discharge'] *= 2
        frame.loc[mask, 'Prcp'] += 10
        raw, destination = self.folder/'changed.csv', self.folder/'changed.npz'
        frame.to_csv(raw, index=False)
        build_evaluation_data(raw, self.history, destination)
        changed = load_data(destination)
        for key, value in self.historical.items():
            np.testing.assert_equal(changed[key][:len(value)] if value.ndim else changed[key], value)
        self.assertFalse(np.array_equal(changed['y'], self.data['y']))

    def test_historical_and_full_fit_exact_same_and_scaler_check(self):
        ids = np.arange(35)
        spec = protocol('final', True, 'cpu')
        variant = spec['variants'][1]
        a = model_for(self.history, spec, variant, 42, 1).fit(ids[:, None], self.historical['y'][ids, :1])
        b = model_for(self.evaluation, spec, variant, 42, 1).fit(ids[:, None], self.data['y'][ids, :1])
        np.testing.assert_equal(a.predict([[40], [41]]), b.predict([[40], [41]]))
        check_model(b, self.data, ids, 1)
        check_audit(b.audit_, ids, self.data, spec, variant, 42, 1, 'development')
        bad = copy.deepcopy(b.audit_)
        bad['params']['seed'] = 43
        with self.assertRaises(ValueError):
            check_audit(bad, ids, self.data, spec, variant, 42, 1, 'development')
        b.x_scaler_.mean_[0] += 1
        with self.assertRaises(AssertionError):
            check_model(b, self.data, ids, 1)

    def test_historical_prefix_mismatch_rejected(self):
        changed = {k:v.copy() for k,v in self.historical.items()}
        changed['features'][200, 0] += 1
        wrong = self.folder/'wrong_history.npz'
        np.savez_compressed(wrong, **changed)
        with self.assertRaises(AssertionError):
            build_evaluation_data(RAW, wrong, self.folder/'should_not_write.npz')

    def test_selected_rows_uses_inner_choice_not_external_error(self):
        frame = pd.DataFrame([dict(year=2002, task_h=1, variant=v, kind='searched', pred=p)
                              for v,p in [('plain', 100), ('residual', 1)]])
        chosen = [dict(year=2002, task_h=1, selected_variant='plain')]
        self.assertEqual(selected_rows(frame, chosen).pred.tolist(), [100])
        with self.assertRaises(ValueError):
            selected_rows(frame.iloc[:0], chosen)

    def test_original_comparison_by_dates_not_new_sample_ids(self):
        old = pd.read_csv(OLD_PREDICTIONS)
        old = old[(old.strategy == 'direct')&(old.model == 'LSTM')].copy()
        old['sample_id'] += 9999
        old['variant'] = 'selected'
        table, merged = original_comparison(old)
        self.assertEqual(len(table), 21)
        np.testing.assert_allclose(table.RMSE_change_pct, 0, atol=1e-10)
        self.assertEqual(len(merged), 359*21)
        old.iloc[0, old.columns.get_loc('truth')] += 1
        with self.assertRaises(AssertionError):
            original_comparison(old)

    def test_verifier_rejects_incomplete_and_smoke_without_permission(self):
        folder = self.folder/'bad'
        folder.mkdir()
        for status, smoke in [('running', False), ('completed', True)]:
            value = dict(status=status, protocol=protocol('final', smoke, 'cpu'))
            (folder/'manifest.json').write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaises(ValueError):
                verify(folder)


if __name__ == '__main__':
    unittest.main(verbosity=2)
