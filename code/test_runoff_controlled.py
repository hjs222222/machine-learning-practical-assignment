"""受控实验：日期契约、因果状态、折内缩放、容量对齐与保存恢复。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

from runoff_training.controlled_data import (BASE, STATE, PROFILES, build_data, load_data,
                                             outer_split, inner_splits, validate_ids)
from runoff_training.controlled_model import (ControlledLSTM, ControlledNet, restore,
                                              target_values, inverse_values)
from runoff_training.controlled_train import choose_epoch, choose_profile, protocol, candidate_record

ROOT = Path(__file__).resolve().parents[1]


class ControlledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.csv = ROOT / 'Teacher Resources & Requirements/01047000.csv'
        cls.path = cls.folder / 'dataset.npz'
        cls.info = build_data(cls.csv, cls.path)
        cls.data = load_data(cls.path)
        cls.raw = pd.read_csv(cls.csv)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_date_mapping_all_targets(self):
        data = self.data
        self.assertEqual(self.info['rows'], 1461)
        self.assertEqual(self.info['samples'], 1276)
        self.assertEqual(self.info['first_origin'], '2000-06-27')
        q = self.raw.Discharge.to_numpy()
        np.testing.assert_equal(data['y'], q[data['positions'][:, None] + np.arange(1, 8)])
        np.testing.assert_equal(data['origin_days'], data['days'][data['positions']])
        self.assertEqual(str(np.datetime64(int(data['days'][-1]), 'D')), '2003-12-31')

    def test_state_values_against_raw_history(self):
        features = self.data['features']
        q = self.raw.Discharge.to_numpy()
        p = self.raw.Prcp.to_numpy()
        api = 0.
        expected_api = []
        for rain in p:
            api = .95 * api + rain
            expected_api.append(api)
        for row in [178, 500, 1000, 1460]:
            expected = [p[row-6:row+1].sum(), p[row-29:row+1].sum(), p[row-89:row+1].sum(),
                expected_api[row], q[row]-q[row-1], q[row]-q[row-7], q[row-29:row+1].mean(),
                q[row-89:row+1].mean()]
            expected = np.sign(expected) * np.log1p(np.abs(expected))
            np.testing.assert_allclose(features[row, len(BASE):], expected, rtol=1e-12, atol=1e-12)
        model = ControlledLSTM(str(self.path), profile='w90_state', fixed_epochs=1, units=4)
        ids = np.arange(40)
        model.fit(ids[:, None], self.data['y'][ids, :1])
        restored = model.x_scaler_.inverse_transform(model._inputs([[50]])[0])
        position = self.data['positions'][50]
        np.testing.assert_allclose(restored, features[position-89:position+1], rtol=1e-5, atol=1e-5)

    def test_feature_and_2004_invariance(self):
        changed = self.raw.copy()
        changed.loc[changed.Date >= '2002-01-01', ['Discharge', 'Prcp', 'Tmax', 'Srad']] = 1e6
        csv, dataset = self.folder / 'future.csv', self.folder / 'future.npz'
        changed.to_csv(csv, index=False)
        build_data(csv, dataset)
        other = load_data(dataset)
        before = self.data['days'] < np.datetime64('2002-01-01').astype('int64')
        np.testing.assert_equal(other['features'][before], self.data['features'][before])
        changed = self.raw.copy()
        changed.loc[changed.Date >= '2004-01-01', 'Discharge'] = 1e8
        changed.to_csv(csv, index=False)
        build_data(csv, dataset)
        other = load_data(dataset)
        for key in self.data:
            np.testing.assert_equal(other[key], self.data[key])

    def test_future_perturbation_leaves_fitted_scaler_unchanged(self):
        data = {key: value.copy() for key, value in self.data.items()}
        first = np.datetime64('2002-01-01').astype('int64')
        data['features'][data['days'] >= first] = 1e8
        altered = self.folder / 'poisoned.npz'
        np.savez_compressed(altered, **data)
        train, _ = outer_split(self.data, 2002)
        original = ControlledLSTM(str(self.path), fixed_epochs=1, units=4).fit(train[:, None], self.data['y'][train, :1])
        poisoned = ControlledLSTM(str(altered), fixed_epochs=1, units=4).fit(train[:, None], self.data['y'][train, :1])
        np.testing.assert_equal(original.x_scaler_.mean_, poisoned.x_scaler_.mean_)
        np.testing.assert_equal(original.x_scaler_.scale_, poisoned.x_scaler_.scale_)
        np.testing.assert_equal(original.y_scaler_.mean_, poisoned.y_scaler_.mean_)
        np.testing.assert_equal(original.predict(train[-3:, None]), poisoned.predict(train[-3:, None]))

    def test_raw_contract_and_ids(self):
        for values in [[-1], [1.9], [np.nan], [len(self.data['y'])], []]:
            with self.assertRaises(ValueError):
                validate_ids(values, len(self.data['y']))
        for kind in ['reversed', 'extra']:
            raw = self.raw.iloc[::-1] if kind == 'reversed' else self.raw.assign(Discharge_h1=1.)
            path = self.folder / f'{kind}.csv'
            raw.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                build_data(path, self.folder / f'{kind}.npz')

    def test_purged_nested_splits(self):
        days = self.data['origin_days']
        for year in [2002, 2003]:
            ti, vi = outer_split(self.data, year)
            self.assertEqual(len(vi), 358)
            self.assertLess(days[ti[-1]] + 7, days[vi[0]])
            folds = inner_splits(self.data, ti)
            self.assertEqual(len(folds), 3)
            for inner_t, inner_v in folds:
                self.assertEqual(len(inner_v), 90)
                self.assertLess(days[ti[inner_t[-1]]] + 7, days[ti[inner_v[0]]])
                self.assertLess(days[ti[inner_v[-1]]] + 7, days[vi[0]])

    def test_fixed_capacity_scaler_and_state_switch(self):
        ti, _ = outer_split(self.data, 2002)
        ids = ti[:40]
        models = []
        for profile in PROFILES:
            model = ControlledLSTM(str(self.path), profile=profile, fixed_epochs=1, units=4)
            self.assertEqual(clone(model).get_params(), model.get_params())
            model.fit(ids[:, None], self.data['y'][ids, :1])
            x = model._inputs(ids[:3])
            self.assertEqual(x.shape, (3, PROFILES[profile][0], len(BASE) + len(STATE)))
            if not PROFILES[profile][1]:
                np.testing.assert_equal(x[:, :, len(BASE):], 0.)
            rows = np.unique((self.data['positions'][ids, None] + np.arange(-89, 1)).ravel())
            np.testing.assert_allclose(model.x_scaler_.mean_, self.data['features'][rows].mean(axis=0))
            self.assertEqual(rows[-1], self.data['positions'][ids[-1]])
            models.append(model)
        self.assertEqual(len({m.audit_['n_parameters'] for m in models}), 1)
        for model in models[1:]:
            np.testing.assert_equal(model.x_scaler_.mean_, models[0].x_scaler_.mean_)
        # 同seed、同维数，未训练初始化完全一致，不因状态开关改变容量。
        initial = []
        for _ in PROFILES:
            torch.manual_seed(42)
            initial.append(ControlledNet(17, 4, 1).state_dict())
        for state in initial[1:]:
            for key in state:
                torch.testing.assert_close(state[key], initial[0][key], rtol=0, atol=0)

    def test_target_inverse_and_native_restore(self):
        ids = np.arange(40)
        for transform in ['log', 'sqrt', 'raw']:
            truth = self.data['y'][ids]
            transformed = target_values(truth, transform)
            np.testing.assert_allclose(inverse_values(transformed, 0, 1, transform), truth, rtol=1e-12)
        for transform in ['log', 'sqrt', 'raw']:
            for dim in [1, 7]:
                model = ControlledLSTM(str(self.path), fixed_epochs=1, units=4, target_transform=transform)
                model.fit(ids[:, None], self.data['y'][ids, :dim])
                pred = model.predict([[50], [51]])
                self.assertEqual(pred.shape, (2, dim))
                folder = self.folder / f'model_{transform}_{dim}'
                model.save(folder)
                restored = restore(folder, self.path)
                np.testing.assert_allclose(restored.predict([[50], [51]]), pred, rtol=1e-7, atol=1e-5)
                broken = self.folder / 'different.npz'
                np.savez(broken, feature_names=['changed'])
                with self.assertRaises(ValueError):
                    restore(folder, broken)
        np.testing.assert_equal(inverse_values([[-1.]], 0, 1, 'raw'), [[-1.]])
        np.testing.assert_equal(inverse_values([[-1.]], 0, 1, 'sqrt'), [[0.]])

    def test_restore_resave_and_invalid_api(self):
        ids = np.arange(20)
        y = self.data['y'][ids, :1]
        for values in [dict(batch_size=-64), dict(fixed_epochs=0), dict(lr=0)]:
            with self.assertRaises(ValueError):
                ControlledLSTM(str(self.path), **values).fit(ids[:, None], y)
        with self.assertRaises(ValueError):
            validate_ids([[1, 2], [3, 4]], len(self.data['y']))
        model = ControlledLSTM(self.path, fixed_epochs=1, units=4,
            history_dir=self.folder / 'path_history').fit(ids[:, None], y)
        a, b = self.folder / 'resave_a', self.folder / 'resave_b'
        model.save(a)
        restored = restore(a, self.path)
        restored.save(b)
        np.testing.assert_equal(restored.predict([[30]]), restore(b, self.path).predict([[30]]))

    def test_uncommitted_and_changed_candidate_fail_fast(self):
        from runoff_training.controlled_data import sha256
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            for name in ['history', 'models', 'checkpoints', 'cv', 'predictions']:
                (run / name).mkdir()
            tag = 'candidate'
            self.assertIsNone(candidate_record(run, tag))
            history = run / 'history' / f'{tag}_123.json'
            history.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '未提交'):
                candidate_record(run, tag)
            (run / 'models' / tag).mkdir()
            artifacts = [history]
            for name in ['weights.pt', 'scalers.npz', 'inference.json']:
                path = run / 'models' / tag / name
                path.write_text('fixture', encoding='utf-8')
                artifacts.append(path)
            for name in ['cv', 'predictions']:
                path = run / name / f'{tag}.csv'
                path.write_text('fixture', encoding='utf-8')
                artifacts.append(path)
            value = {'artifacts': {p.relative_to(run).as_posix(): sha256(p) for p in artifacts}}
            (run / 'checkpoints' / f'{tag}.json').write_text(json.dumps(value), encoding='utf-8')
            self.assertEqual(candidate_record(run, tag), value)
            artifacts[-1].write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '改变'):
                candidate_record(run, tag)

    def test_predeclared_selection_and_budget(self):
        self.assertEqual(choose_epoch([101, 100], [10, 30]), 0)
        self.assertEqual(choose_epoch([102, 100], [10, 30]), 1)
        self.assertEqual(choose_profile(dict(w30_base=101, w90_base=100, w30_state=105, w90_state=110)), 'w30_base')
        from argparse import Namespace
        for smoke in [False, True]:
            spec = protocol(Namespace(smoke=smoke, device='cpu'))
            n = len(spec['years']) * len(spec['tasks']) * (len(PROFILES) + 2) * len(spec['seeds'])
            self.assertEqual(n * (len(spec['epochs']) * spec['inner_folds'] + 1), spec['fit_budget'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
