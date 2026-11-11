"""策略增强回归测试：严格基准等价、残差初值、容量、因果性、恢复与预算。"""
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from diagnose_runoff_memory import perturb
from runoff_training.controlled_data import build_data, load_data
from runoff_training.controlled_model import ControlledLSTM, ControlledNet, inverse_values
from runoff_training.strategy_model import StrategyLSTM, restore_strategy
from runoff_training.strategy_train import variants, spec_for, choose_variant
from verify_runoff_strategy import verify

ROOT=Path(__file__).resolve().parents[1]


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp=tempfile.TemporaryDirectory()
        cls.folder=Path(cls.tmp.name)
        cls.dataset=cls.folder/'data.npz'
        build_data(ROOT/'Teacher Resources & Requirements/01047000.csv',cls.dataset)
        cls.data=load_data(cls.dataset)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def initialized(self,pooling='last',residual=False,transform='log',dim=7):
        model=StrategyLSTM(str(self.dataset),pooling=pooling,residual=residual,target_transform=transform)
        model.data_=self.data
        model.x_scaler_=StandardScaler().fit(self.data['features'][89:250])
        y=self.data['y'][:40,:dim]
        transformed=np.log(y) if transform=='log' else np.sqrt(y)
        model.y_scaler_=StandardScaler().fit(transformed)
        model.output_dim_=dim
        torch.manual_seed(42)
        model.model_=model._network()
        model.model_.eval()
        return model

    def test_plain_baseline_exact_and_capacity(self):
        models=[self.initialized(pooling,residual) for pooling in ['last','last_mean'] for residual in [False,True]]
        self.assertEqual(len({sum(p.numel() for p in m.model_.parameters()) for m in models}),1)
        plain=models[0]
        torch.manual_seed(42)
        reference=ControlledNet(17,32,7).eval()
        x=torch.as_tensor(plain._inputs([[80],[81]]))
        torch.testing.assert_close(reference(x),plain.model_(x),rtol=0,atol=0)
        for name,param in plain.model_.named_parameters():
            torch.testing.assert_close(param,dict(models[2].model_.named_parameters())[name],rtol=0,atol=0)
        self.assertFalse(torch.equal(models[2].model_(x),plain.model_(x)))

    def test_residual_initial_prediction_persistence(self):
        for transform in ['log','sqrt']:
            for dim in [1,7]:
                for pooling in ['last','last_mean']:
                    model=self.initialized(pooling,True,transform,dim)
                    x=torch.as_tensor(model._inputs([[80],[81]]))
                    with torch.no_grad():
                        z=model.model_(x).numpy()
                    prediction=inverse_values(z,model.y_scaler_.mean_,model.y_scaler_.scale_,transform)
                    expected=self.data['discharge'][self.data['positions'][[80,81]]]
                    np.testing.assert_allclose(prediction,np.repeat(expected[:,None],dim,axis=1),rtol=2e-6,atol=2e-4)

    def test_training_matches_controlled_plain(self):
        ids=np.arange(40)
        y=self.data['y'][ids,:1]
        controlled=ControlledLSTM(str(self.dataset),profile='w90_base',fixed_epochs=2,units=4).fit(ids[:,None],y)
        strategy=StrategyLSTM(str(self.dataset),fixed_epochs=2,units=4).fit(ids[:,None],y)
        np.testing.assert_equal(controlled.predict([[60],[61]]),strategy.predict([[60],[61]]))

    def test_native_fit_clone_restore_resave(self):
        ids=np.arange(40)
        for dim,transform in [(1,'sqrt'),(7,'log')]:
            model=StrategyLSTM(self.dataset,fixed_epochs=2,units=4,pooling='last_mean',residual=True,
                               target_transform=transform,history_dir=self.folder/f'history_{dim}')
            self.assertEqual(clone(model).get_params(),model.get_params())
            model.fit(ids[:,None],self.data['y'][ids,:dim])
            first,second=self.folder/f'model_{dim}',self.folder/f'resave_{dim}'
            model.save(first)
            restored=restore_strategy(first,self.dataset)
            self.assertEqual(restored.predict([[60],[61]]).shape,(2,dim))
            np.testing.assert_equal(model.predict([[60],[61]]),restored.predict([[60],[61]]))
            restored.save(second)
            np.testing.assert_equal(restored.predict([[60]]),restore_strategy(second,self.dataset).predict([[60]]))

    def test_future_data_does_not_affect_fit(self):
        ids=np.arange(40)
        changed={k:v.copy() for k,v in self.data.items()}
        boundary=int(self.data['positions'][ids[-1]])+1
        changed['features'][boundary:]=1e8
        path=self.folder/'future.npz'
        np.savez_compressed(path,**changed)
        y=self.data['y'][ids,:1]
        a=StrategyLSTM(str(self.dataset),fixed_epochs=1,units=4,residual=True).fit(ids[:,None],y)
        b=StrategyLSTM(str(path),fixed_epochs=1,units=4,residual=True).fit(ids[:,None],y)
        np.testing.assert_equal(a.x_scaler_.mean_,b.x_scaler_.mean_)
        np.testing.assert_equal(a.predict(ids[-3:,None]),b.predict(ids[-3:,None]))

    def test_budget_and_fixed_selection_priority(self):
        self.assertEqual(len(variants('direct',1)),4)
        self.assertEqual(len(variants('direct',7)),8)
        self.assertEqual(len(variants('multi',7)),4)
        for smoke in [False,True]:
            spec=spec_for(Namespace(smoke=smoke,device='cpu'))
            count=len(spec['years'])*len(spec['seeds'])*sum(len(v) for v in spec['variants'].values())
            self.assertEqual(count,spec['outer_models'])
            self.assertEqual(count*(len(spec['epochs'])*3+1),spec['fits'])
        self.assertEqual(choose_variant(dict(plain=101,residual=100)),'plain')
        self.assertEqual(choose_variant(dict(plain=102,residual=100)),'residual')

    def test_memory_perturbations_preserve_recent_data(self):
        x=np.arange(2*90*3,dtype='float32').reshape(2,90,3)
        for kind in ['old60_zero','old60_reverse']:
            np.testing.assert_equal(perturb(x,kind)[:,60:],x[:,60:])
        np.testing.assert_equal(perturb(x,'recent7_zero')[:,:83],x[:,:83])
        self.assertTrue(np.any(x))

    def test_verify_rejects_incomplete_and_unapproved_smoke(self):
        folder=self.folder/'bad_run'
        folder.mkdir()
        for status,smoke in [('failed',False),('completed',True)]:
            (folder/'manifest.json').write_text(json.dumps(dict(status=status,protocol=dict(smoke=smoke))),encoding='utf-8')
            with self.assertRaises(ValueError):
                verify(folder)


if __name__=='__main__':
    unittest.main(verbosity=2)
