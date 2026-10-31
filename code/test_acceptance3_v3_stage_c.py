# -*- coding: utf-8 -*-
"""季度因果OOF与原生保守多输出元层测试。"""
import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from acceptance3_model import load_dataset
from acceptance3_v2_train import prepare_v2
from acceptance3_v3_stage_b import historical_splits
from acceptance3_v3_stage_c import quarterly_splits, meta_splits, family_grid, archive_uncommitted_history
from acceptance3_v3_fusion import ConservativeFusion, fusion_candidates


class StageCTests(unittest.TestCase):
    def test_quarterly_and_meta_causality(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,train,_,_ = prepare_v2(Path(tmp))
            data = load_dataset(str(path))
            days = data['origin_days']
            folds = quarterly_splits(data,train)
            self.assertEqual(len(folds),10)
            self.assertEqual(folds[0][0],'2001Q3')
            self.assertEqual(folds[-1][0],'2003Q4')
            ids = np.concatenate([vi for _,_,vi in folds])
            self.assertEqual(len(ids),len(np.unique(ids)))
            self.assertEqual(str(data['origin_dates'][ids[-1]]),'2003-12-24')
            for _,ti,vi in folds:
                self.assertLess(days[ti[-1]]+7,days[vi[0]])
                for inner_train,inner_val in historical_splits(ti,days):
                    self.assertLess(days[ti[inner_train[-1]]]+7,days[ti[inner_val[0]]])
                    self.assertLess(days[ti[inner_val[-1]]]+7,days[vi[0]])
            for _,ti,vi in meta_splits(ids,days,folds):
                self.assertGreaterEqual(len(ti),180)
                self.assertLess(days[ids[ti[-1]]]+7,days[ids[vi[0]]])
            load_dataset.cache_clear()

    def test_native_fusion_and_restore(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(10,100,(80,21))
        y = .2*x[:,:7]+.7*x[:,7:14]+.1*x[:,14:]
        for params in fusion_candidates():
            model = clone(ConservativeFusion(**params)).fit(x,y)
            p = model.predict(x)
            self.assertEqual(p.shape,(80,7))
            self.assertTrue(np.isfinite(p).all())
            with tempfile.TemporaryDirectory() as tmp:
                file = Path(tmp)/'meta.joblib'
                joblib.dump(model,file)
                np.testing.assert_allclose(joblib.load(file).predict(x),p)
            if params['kind']=='convex':
                b = x.reshape(80,3,7)
                self.assertTrue((p>=b.min(axis=1)-1e-8).all())
                self.assertTrue((p<=b.max(axis=1)+1e-8).all())
                self.assertAlmostEqual(model.weights_.sum(),1.)
        np.testing.assert_allclose(ConservativeFusion('RF').fit(x,y).predict(x),x[:,7:14])
        np.testing.assert_allclose(ConservativeFusion('Mean').fit(x,y).predict(x),x.reshape(80,3,7).mean(axis=1))

    def test_strict_shape(self):
        with self.assertRaises(ValueError): ConservativeFusion().fit(np.ones((3,3)),np.ones((3,7)))
        with self.assertRaises(ValueError): ConservativeFusion().fit(np.ones((3,21)),np.ones(3))

    def test_uncommitted_retry_history_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            folder = run/'training_history'
            folder.mkdir()
            unfinished = folder/'2001Q3_ANN_ANN_h7_uuid.json'
            committed = folder/'2001Q3_ANN_seed42_ANN_h7_other.json'
            unfinished.write_text('{}',encoding='utf-8')
            committed.write_text('{}',encoding='utf-8')
            archive_uncommitted_history(run,'2001Q3_ANN','ANN')
            self.assertFalse(unfinished.exists())
            self.assertTrue(committed.exists())
            archived = list((run/'retry_history').rglob('*.json'))
            self.assertEqual(len(archived),1)
            self.assertEqual(archived[0].read_text(encoding='utf-8'),'{}')

    def test_small_grids(self):
        self.assertEqual([len(list(ParameterGrid(family_grid(f,False)))) for f in ['ANN','RF','LSTM']],[3,4,12])
        for f in ['ANN','RF','LSTM']:
            self.assertEqual(family_grid(f,False)['target_transform'],['log'])
        self.assertEqual(len(fusion_candidates()),9)


if __name__=='__main__': unittest.main(verbosity=2)
