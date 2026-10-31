# -*- coding: utf-8 -*-
"""第三轮年度外层与历史内层选参边界测试。"""
import tempfile
from pathlib import Path
import unittest

from sklearn.model_selection import ParameterGrid

from acceptance3_model import load_dataset
from acceptance3_v2_train import prepare_v2
from acceptance3_v3_train import development_splits
from acceptance3_v3_stage_b import historical_splits, grid, TASKS


class StageBTests(unittest.TestCase):
    def test_historical_inner_never_sees_outer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,*_=prepare_v2(Path(tmp))
            data=load_dataset(str(path))
            folds=development_splits(data)
            self.assertEqual([len(historical_splits(ti,data['origin_days'])) for _,ti,_ in folds],[3,4])
            for _,ids,outer in folds:
                for ti,vi in historical_splits(ids,data['origin_days']):
                    self.assertGreaterEqual(len(ti),365)
                    self.assertEqual(len(vi),90)
                    self.assertLess(data['origin_days'][ids[ti[-1]]]+7,data['origin_days'][ids[vi[0]]])
                    self.assertLess(data['origin_days'][ids[vi[-1]]]+7,data['origin_days'][outer[0]])
            load_dataset.cache_clear()

    def test_budget_and_fixed_profiles(self):
        self.assertEqual(len(TASKS),8)
        self.assertEqual(len(list(ParameterGrid(grid('A0',False)))),3)
        self.assertEqual(len(list(ParameterGrid(grid('A1',False)))),12)
        self.assertEqual(8*7*15+8*2*2*3,936)
        self.assertEqual(grid('A1',False)['target_transform'],['log'])
        self.assertEqual(grid('A1',False)['feature_mode'],['compact'])


if __name__=='__main__':unittest.main(verbosity=2)
