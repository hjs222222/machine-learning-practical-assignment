"""Fit a reviewable h1 candidate on 2000--2003 only; no test evaluation."""
import argparse
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from config import ROOT, DATA_PATH, digest, save_json
from data import load_raw, development_splits
from models import RunoffRegressor
from research_h1 import prepare_data
from research_selection_search import h1_score


def training_data(raw):
    data=prepare_data(raw,horizons=7)
    if data['dates'].max()+pd.Timedelta(days=7)>pd.Timestamp('2003-12-31'):
        raise ValueError('Training labels exceed cutoff')
    return data


def run(output):
    source=ROOT/'output/teacher_selection_20261109/nested_ann/protocol.json'
    reference=json.loads(source.read_text(encoding='utf-8'))
    if reference['data_sha256']!=digest(DATA_PATH) or reference['family']!='ANN':
        raise ValueError('Candidate source does not match teacher data and ANN')
    data=training_data(load_raw());X=data['mi']['X'];y=data['y']
    folds=development_splits(data['dates'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    save_json(output/'protocol.json',dict(seeds=[42,13,2026],weights=[1/3]*3,outputs=7,
        primary_metric='h1 original-flow R2',grid=reference['grid'],source=str(source.relative_to(ROOT)),
        source_sha256=digest(source),data_sha256=digest(DATA_PATH),data_cutoff='2003-12-31',
        training_origins=len(X),origin_start=data['dates'].min(),origin_end=data['dates'].max(),
        label_end=data['dates'].max()+pd.Timedelta(days=7),test_authorized=False,
        cv='TimeSeriesSplit(2),gap7',test_evaluated=False,
        code_hashes={p.name:digest(p) for p in (ROOT/'code').glob('*.py')}))
    selected=[];warning_rows=[];probe_rows=[]
    for seed in (42,13,2026):
        print('FREEZE seed',seed,flush=True)
        params=dict(reference['estimator'],seed=seed)
        search=GridSearchCV(RunoffRegressor(**params),reference['grid'],cv=folds,
                            scoring=h1_score,n_jobs=1,error_score='raise')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');search.fit(X,y)
        warning_rows.extend(dict(seed=seed,category=w.category.__name__,message=str(w.message)) for w in caught)
        model=search.best_estimator_;path=output/f'ann_seed{seed}.joblib';joblib.dump(model,path)
        pd.DataFrame(search.cv_results_).to_csv(output/f'cv_seed{seed}.csv',index=False)
        selected.append(dict(seed=seed,params=search.best_params_,inner_cv_h1_r2=float(search.best_score_),
            model_file=path.name,model_sha256=digest(path),actual_iterations=model.estimator_.n_iter_,
            features=[data['mi']['names'][i] for i in model.selected_]))
        # Probes are training rows only, for independent forward verification.
        indices=np.array([0,len(X)//2,len(X)-1])
        for i,pred in zip(indices,model.predict(X[indices])):
            probe_rows.append(dict(seed=seed,origin=str(data['dates'][i].date()),
                                   **{f'h{h+1}':float(p) for h,p in enumerate(pred)}))
    pd.DataFrame(probe_rows).to_csv(output/'training_probes.csv',index=False)
    save_json(output/'selected.json',selected);save_json(output/'warnings.json',warning_rows)
    save_json(output/'frozen.json',dict(status='frozen_pending_independent_verification',
        protocol_sha256=digest(output/'protocol.json'),models=selected,base_fits=75,
        test_evaluated=False,test_authorized=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    run(**vars(parser.parse_args()))
