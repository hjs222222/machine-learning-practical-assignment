# -*- coding: utf-8 -*-
"""冻结模型评估必须拒绝未完成、错误类型或未核验来源。"""
import json
import tempfile
from pathlib import Path
import unittest

from acceptance3_v3_evaluate import evaluate


class EvaluateTests(unittest.TestCase):
    def test_refuse_unfinished_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'manifest.json').write_text(json.dumps(dict(status='running',smoke=False)),encoding='utf-8')
            (path/'verification.json').write_text(json.dumps(dict(status='passed',test_2004_predicted=False)),encoding='utf-8')
            with self.assertRaises(AssertionError): evaluate(path,path)
            self.assertFalse((path/'selection_frozen.json').exists())

    def test_refuse_smoke_as_formal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'manifest.json').write_text(json.dumps(dict(status='stage_c_completed',smoke=True)),encoding='utf-8')
            (path/'verification.json').write_text(json.dumps(dict(status='passed',test_2004_predicted=False)),encoding='utf-8')
            with self.assertRaises(AssertionError): evaluate(path,path,smoke=False)
            self.assertFalse((path/'selection_frozen.json').exists())


if __name__=='__main__': unittest.main(verbosity=2)
