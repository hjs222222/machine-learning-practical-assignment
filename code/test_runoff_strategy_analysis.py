"""策略分析回归测试：seed/h对齐、选择不看外层、配对方向与聚合。"""
import unittest

import numpy as np
import pandas as pd

from analyze_runoff_strategy import arrays, compare
from analyze_runoff_controlled import summary_metrics, paired_interval


class StrategyAnalysisTests(unittest.TestCase):
    def test_common_seed_sample_horizon_order(self):
        rows=[]
        for seed in [44,42,43]:
            for sample in [3,1,2]:
                for h in [7,1]:
                    rows.append(dict(variant='a',seed=seed,sample_id=sample,h=h,
                                     truth=10*sample+h,pred=10*sample+h+seed-42))
        truth,pred,ids=arrays(pd.DataFrame(rows),'a')
        np.testing.assert_equal(ids,[1,2,3])
        np.testing.assert_equal(truth,[[11,17],[21,27],[31,37]])
        self.assertEqual(pred.shape,(3,3,2))
        np.testing.assert_equal(pred[0],truth)
        np.testing.assert_equal(pred[2],truth+2)

    def test_paired_comparison_and_interval_direction(self):
        truth=np.arange(40,dtype=float).reshape(20,2)
        base=np.repeat((truth+2)[None],3,axis=0)
        candidate=np.repeat((truth+1)[None],3,axis=0)
        values=compare(truth,base,candidate)
        self.assertAlmostEqual(values['RMSE_change_pct'],-50)
        self.assertAlmostEqual(values['MAE_change_pct'],-50)
        self.assertEqual(values['improved_seeds'],3)
        lo,hi=paired_interval(truth,base,candidate,7,repeats=20)
        self.assertAlmostEqual(lo,-50)
        self.assertAlmostEqual(hi,-50)

    def test_rmse_aggregation_is_not_flattened(self):
        truth=np.zeros((4,2))
        pred=np.array([[[1.,3.]]*4,[[2.,6.]]*4])
        self.assertEqual(summary_metrics(truth,pred)['RMSE'],3.)
        self.assertNotEqual(summary_metrics(truth,pred)['RMSE'],float(np.sqrt(np.square(pred).mean())))

    def test_missing_variant_rejected(self):
        frame=pd.DataFrame([dict(variant='a',seed=42,sample_id=0,h=1,truth=1.,pred=1.)])
        with self.assertRaises(ValueError):
            arrays(frame,'missing')


if __name__=='__main__':
    unittest.main(verbosity=2)
