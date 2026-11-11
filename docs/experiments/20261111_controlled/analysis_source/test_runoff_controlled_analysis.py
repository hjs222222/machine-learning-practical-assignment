"""受控实验分析计算：恒等配对、变好方向及多seed/多h聚合口径。"""
import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from analyze_runoff_controlled import arrays, paired_interval, summary_metrics, write_markdown


class AnalysisTests(unittest.TestCase):
    def test_block_identity_and_improvement(self):
        truth = np.arange(1, 101, dtype=float).reshape(50, 2)
        baseline = np.stack([truth + 2, truth + 3, truth + 4])
        for block in [7, 14, 30]:
            low, high = paired_interval(truth, baseline, baseline, block, repeats=30)
            self.assertEqual((low, high), (0., 0.))
            low, high = paired_interval(truth, baseline, baseline - 1, block, repeats=30)
            self.assertLess(high, 0.)
            self.assertLessEqual(low, high)

    def test_average_rmse_not_flattened_rmse(self):
        truth = np.zeros((3, 2))
        pred = np.array([[[1., 3.]] * 3, [[2., 4.]] * 3])
        result = summary_metrics(truth, pred)
        self.assertEqual(result['RMSE'], 2.5)
        self.assertEqual(result['MAE'], 2.5)
        self.assertNotAlmostEqual(result['RMSE'], np.sqrt(np.square(pred).mean()))

    def test_report_table_accesses_transform_column(self):
        annual = pd.DataFrame([dict(year=2002, strategy='direct', task_h=1, profile=p, transform=t, RMSE=1., MAE=1.)
            for p in ['w30_base', 'w90_base', 'w30_state', 'w90_state'] for t in ['log', 'sqrt', 'raw']])
        pair = pd.DataFrame([dict(year=2002, strategy='direct', task_h=1, comparison='same',
            RMSE_change_pct=0., seed_min_change_pct=0., seed_max_change_pct=0., improved_seeds=0)])
        selected = [dict(year=2002, strategy='direct', task_h=1, selected_profile='w30_base')]
        with tempfile.TemporaryDirectory() as folder:
            write_markdown(Path(folder), annual, pair, pd.DataFrame(), selected,
                           dict(fits=756, models=108, prediction_rows=1, metrics=1, status='passed'))
            text = (Path(folder) / 'README.md').read_text(encoding='utf-8')
            self.assertIn('| 2002 | h1 |', text)
            self.assertIn('w30_base | 1.00 | 1.00 | 1.00', text)

    def test_seed_pivot_uses_common_order(self):
        rows = []
        for seed in [44, 42, 43]:
            for sample in [9, 5, 7]:
                for h in [2, 1]:
                    rows.append(dict(profile='w30_base', transform='log', seed=seed,
                        sample_id=sample, h=h, truth=float(sample + h), pred=float(sample + h + seed)))
        truth, pred, ids = arrays(pd.DataFrame(rows), 'w30_base', 'log')
        np.testing.assert_equal(ids, [5, 7, 9])
        np.testing.assert_equal(truth, [[6, 7], [8, 9], [10, 11]])
        for i, seed in enumerate([42, 43, 44]):
            np.testing.assert_equal(pred[i] - truth, seed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
