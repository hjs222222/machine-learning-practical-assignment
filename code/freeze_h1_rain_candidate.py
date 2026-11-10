"""Freeze the fixed rain-feature h1 hypothesis without opening the test year."""
import argparse
import copy
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from config import ROOT, DATA_PATH, digest, save_json
from data import load_raw
from freeze_h1_candidate import training_data
from models import RunoffRegressor


def run(output):
    parent=ROOT/'output/teacher_selection_20261109/frozen_ann_search_3seed'
    frozen=json.loads((parent/'frozen.json').read_text(encoding='utf-8'))
    protocol=copy.deepcopy(json.loads((parent/'protocol.json').read_text(encoding='utf-8')))
    assert digest(DATA_PATH)==protocol['data_sha256']
    data=training_data(load_raw());X=data['mi']['X'];y=data['y']
    rain=data['mi']['names'].index('Prcp_lag0')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    protocol.update(parent_frozen_sha256=digest(parent/'frozen.json'),grid=None,cv=None,
        selection='Fixed inherited first/k12/alpha1 plus Prcp_lag0; no new parameter search',
        forced_feature='Prcp_lag0',test_evaluated=False,test_authorized=False,
        code_hashes={p.name:digest(p) for p in (ROOT/'code').glob('*.py')})
    save_json(output/'protocol.json',protocol)
    selected=[];probes=[];warning_rows=[]
    for entry in frozen['models']:
        path=parent/entry['model_file'];assert digest(path)==entry['model_sha256']
        params=joblib.load(path).get_params();params['forced_indices']=(rain,)
        model=RunoffRegressor(**params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(X,y)
        assert rain in model.selected_ and len(model.selected_)==13
        warning_rows.extend(dict(seed=model.seed,message=str(w.message)) for w in caught)
        target=output/entry['model_file'];joblib.dump(model,target)
        selected.append(dict(seed=model.seed,params=entry['params'],forced_indices=[rain],
            model_file=target.name,model_sha256=digest(target),actual_iterations=model.estimator_.n_iter_,
            features=[data['mi']['names'][i] for i in model.selected_]))
        ids=np.array([0,len(X)//2,len(X)-1])
        for i,pred in zip(ids,model.predict(X[ids])):
            probes.append(dict(seed=model.seed,origin=str(data['dates'][i].date()),
                               **{f'h{h+1}':float(p) for h,p in enumerate(pred)}))
    pd.DataFrame(probes).to_csv(output/'training_probes.csv',index=False)
    save_json(output/'selected.json',selected);save_json(output/'warnings.json',warning_rows)
    save_json(output/'frozen.json',dict(status='frozen_pending_independent_verification',
        protocol_sha256=digest(output/'protocol.json'),models=selected,base_fits=3,
        test_evaluated=False,test_authorized=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(**vars(parser.parse_args()))
