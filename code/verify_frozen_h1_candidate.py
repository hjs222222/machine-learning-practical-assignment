"""Independent NumPy forward checks of training probes; no fit or predict calls."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import ROOT, DATA_PATH, digest, save_json
from data import load_raw
from research_h1 import prepare_data


def verify(output):
    output=Path(output)
    frozen=json.loads((output/'frozen.json').read_text(encoding='utf-8'))
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    assert digest(output/'protocol.json')==frozen['protocol_sha256']
    assert digest(DATA_PATH)==protocol['data_sha256']
    assert protocol['seeds']==[42,13,2026] and protocol['weights']==[1/3]*3
    assert protocol['test_evaluated'] is False and frozen['test_evaluated'] is False
    for name,sha in protocol['code_hashes'].items():
        assert digest(ROOT/'code'/name)==sha,name
    data=prepare_data(load_raw().loc[:'2003-12-31'],horizons=7)
    X=data['mi']['X'][:,-1:];y=data['y']
    assert len(X)==protocol['training_origins']==1395
    assert data['dates'].max()+pd.Timedelta(days=7)<=pd.Timestamp('2003-12-31')
    probes=pd.read_csv(output/'training_probes.csv',parse_dates=['origin'])
    largest=0.
    for entry in frozen['models']:
        path=output/entry['model_file'];assert digest(path)==entry['model_sha256']
        model=joblib.load(path)
        for name,value in entry['params'].items():assert getattr(model,name)==value
        assert model.estimator_.n_iter_==entry['actual_iterations']
        assert (model.family,model.ann_backend,model.transform,model.residual)==('ANN','sklearn','sqrt',True)
        assert model.outputs_==7 and model.estimator_.activation=='relu'
        assert model.x_scaler_.n_samples_seen_==model.y_scaler_.n_samples_seen_==len(X)
        selected=X[:,:,model.selected_].reshape(len(X),-1)
        np.testing.assert_allclose(selected.mean(0,dtype='float64'),model.x_scaler_.mean_,rtol=1e-10)
        target=np.sqrt(y)-np.sqrt(X[:,-1,:1])
        np.testing.assert_allclose(target.mean(0),model.y_scaler_.mean_,rtol=1e-10)
        np.testing.assert_allclose(target.var(0),model.y_scaler_.var_,rtol=1e-10)
        for row in probes.loc[probes.seed==entry['seed']].itertuples():
            i=np.flatnonzero(data['dates']==row.origin)[0]
            a=X[i,-1,model.selected_].copy().reshape(1,-1)
            a-=model.x_scaler_.mean_;a/=model.x_scaler_.scale_
            for layer,(weight,bias) in enumerate(zip(model.estimator_.coefs_,model.estimator_.intercepts_)):
                a=a@weight+bias
                if layer<len(model.estimator_.coefs_)-1:a=np.maximum(a,0)
            a/=model.output_scale_;a*=model.y_scaler_.scale_;a+=model.y_scaler_.mean_
            actual=np.maximum(a+np.sqrt(X[i,-1,:1]),0)**2
            expected=np.array([getattr(row,f'h{h}') for h in range(1,8)])
            np.testing.assert_allclose(actual.ravel(),expected,rtol=1e-6,atol=1e-4)
            largest=max(largest,float(np.max(np.abs(actual.ravel()-expected))))
    result=dict(status='passed',models=3,training_origins=len(X),probe_values=63,
                largest_probe_difference=largest,test_evaluated=False,fit_calls=0,predict_calls=0,
                verifier_sha256=digest(__file__))
    save_json(output/'verification.json',result)
    frozen['status']='frozen_verified_pending_test_authorization'
    frozen['verification_sha256']=digest(output/'verification.json')
    save_json(output/'frozen.json',frozen)
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    verify(**vars(parser.parse_args()))
