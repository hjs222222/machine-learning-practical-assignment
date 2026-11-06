"""Training-only, pre-registered h=1 annual comparisons. Never consumes 2004 labels."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import DATA_PATH, ROOT, digest, save_json
from data import load_raw
from features import build_features
from models import RunoffRegressor

DEFAULT_MODES = ('mi', 'forced', 'all', 'hydro')
MODES = DEFAULT_MODES + ('compact','target_mi','climate')
BASE = {'RF': dict(trees=120, leaf=1),
        'ANN': dict(hidden=64, iterations=350, alpha=.1),
        'LSTM': dict(hidden=32, seq_len=14, epochs=70, alpha=.1)}


def fixed_h1_ensemble_members():
    """Predeclared members; weights are never selected on a validation year."""
    return [
        dict(id='climate_rf', candidate='RF_tune_05', family='RF', mode='climate', weight=1 / 3),
        dict(id='compact_ann', candidate='ANN_tune_01', family='ANN', mode='compact', weight=1 / 3),
        dict(id='persistence', candidate='persistence', family='persistence', mode='all', weight=1 / 3),
    ]


def prepare_data(raw):
    raw = raw.loc[:'2003-12-31'].copy()
    base, _ = build_features(raw)
    hydro = base.copy()
    for days in (1, 2, 3, 7):
        hydro[f'Discharge_diff{days}'] = raw.Discharge.diff(days)
    for days in (2, 3, 5, 7, 14, 30):
        hydro[f'Prcp_prior_sum{days}'] = raw.Prcp.shift(1).rolling(days).sum()
    compact_names = ([f'{col}_lag{lag}' for col in ('Discharge','Prcp') for lag in (0,1,2,3,7)]
                     + [f'Discharge_diff{d}' for d in (1,2,3,7)]
                     + [f'Prcp_prior_sum{d}' for d in (2,3,5,7,14,30)] + ['month_sin','month_cos'])
    compact = hydro[compact_names]
    climate = compact.copy()
    avg = (raw.Tmax+raw.Tmin)/2
    climate['Tmax_lag0'],climate['Tmin_lag0'],climate['Tavg_lag0'] = raw.Tmax,raw.Tmin,avg
    for days in (1,3,7):
        climate[f'Tavg_diff{days}'] = avg.diff(days)
    for days in (3,7,14):
        climate[f'Tavg_roll{days}_mean'] = avg.rolling(days).mean()
        climate[f'Tavg_positive_sum{days}'] = avg.clip(lower=0).rolling(days).sum()
    for days in (7,14,30):
        climate[f'ColdPrcp_prior_sum{days}'] = (raw.Prcp*(avg<0)).shift(1).rolling(days).sum()
    climate['warming_wet7'] = avg.diff(3).clip(lower=0)*hydro.Prcp_prior_sum7
    climate['warm_rain0'] = raw.Prcp*avg.clip(lower=0)
    ready = hydro.notna().all(axis=1).to_numpy()
    ids = np.array([i for i in range(29, len(raw)-1) if ready[i-29:i+1].all()])
    dates = raw.index[ids]
    result = dict(dates=dates, y=raw.Discharge.shift(-1).iloc[ids].to_numpy()[:,None],
                  raw=raw)
    for mode in MODES:
        frame = climate if mode == 'climate' else compact if mode == 'compact' else hydro if mode == 'hydro' else base
        result[mode] = dict(X=np.stack([frame.iloc[i-29:i+1].to_numpy(np.float32) for i in ids]),
                            names=list(frame.columns), dates=dates)
    return result


def annual_splits(dates):
    dates = pd.DatetimeIndex(dates)
    for year in (2001, 2002, 2003):
        va = np.flatnonzero((dates.year == year) & ((dates + pd.Timedelta(days=1)).year == year))
        tr = np.flatnonzero(dates < dates[va[0]] - pd.Timedelta(days=7))
        if not len(tr) or dates[tr[-1]] + pd.Timedelta(days=1) >= dates[va[0]]:
            raise ValueError('Invalid annual temporal boundary')
        yield year, tr, va


def metrics(truth, prediction, threshold):
    truth, prediction = np.asarray(truth).ravel(), np.asarray(prediction).ravel()
    error = prediction - truth
    peak = error[truth >= threshold]
    return dict(r2=float(r2_score(truth,prediction)), rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(abs(error))), peak_rmse=float(np.sqrt(np.mean(peak**2))) if len(peak) else None,
                peak_bias=float(np.mean(peak)) if len(peak) else None, peak_count=int(len(peak)))


def stage_candidates(stage, previous=None, families=('RF','ANN','LSTM'), modes=DEFAULT_MODES):
    if stage != 'features' and tuple(modes) != DEFAULT_MODES:
        raise ValueError('--modes is supported only for features')
    if stage == 'features':
        if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
            raise ValueError('Feature modes must be nonempty, unique and supported')
        return [dict(id=f'{family}_{mode}_log_delta', family=family, mode=mode,
                     transform='log', residual=True, params=BASE[family])
                for family in families for mode in modes]
    previous = Path(previous) if previous is not None else None
    if previous is None or not (previous/'complete.json').is_file():
        raise ValueError('Previous stage must be complete')
    protocol = json.loads((previous/'protocol.json').read_text(encoding='utf-8'))
    if protocol['stage'] != {'targets':'features','tune':'targets'}[stage]:
        raise ValueError('Wrong previous stage')
    summary = pd.read_csv(previous/'summary.csv').sort_values(['worst_r2','mean_r2'],ascending=False)
    candidates = []
    for family in families:
        rows = summary[summary.family == family]
        if rows.empty:
            raise ValueError(f'No completed {family} candidate in previous stage')
        best = next(c for c in protocol['candidates'] if c['id'] == rows.iloc[0].candidate)
        if stage == 'targets':
            settings = [dict(transform=t,residual=r,params=BASE[family])
                        for t,r in itertools.product(('raw','sqrt','log'),(False,True))]
        else:
            grids = {'RF': [dict(trees=120,leaf=l,max_features=m) for l,m in itertools.product((1,3,5,10),(1.,.5))],
                     'ANN': [dict(hidden=h,iterations=i,alpha=a) for h,i,a in itertools.product((32,64,128),(350,800),(.1,1.))],
                     'LSTM': [dict(hidden=h,seq_len=s,epochs=e,alpha=.1) for h,s,e in itertools.product((32,64),(7,14,30),(70,150))]}
            settings = [dict(transform=best['transform'],residual=best['residual'],params=p) for p in grids[family]]
        for i, setting in enumerate(settings):
            candidates.append(dict(id=f'{family}_{stage}_{i:02d}',family=family,mode=best['mode'],**setting))
    return candidates


def run_stage(stage, output, previous=None, seed=42, families=('RF','ANN','LSTM'), limit=None,
              raw=None, candidates=None, modes=DEFAULT_MODES):
    if stage != 'features' and tuple(modes) != DEFAULT_MODES:
        raise ValueError('--modes is supported only for features')
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; no overwrite or hot start')
    if candidates is None:
        candidates = stage_candidates(stage,previous,families,modes)
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        candidates = [c for family in families for c in [v for v in candidates if v['family']==family][:limit]]
    if not candidates or len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('Candidates must be nonempty and unique')
    data = prepare_data(load_raw() if raw is None else raw)
    splits = list(annual_splits(data['dates']))
    output.mkdir(parents=True)
    protocol = dict(stage=stage,seed=seed,data_cutoff='2003-12-31',horizon=1,gap=7,
                    sequence_length=30,selection='worst annual R2, then mean annual R2',
                    limitation='Training-period selection is biased; no independent generalization claim',
                    sample_note='Common 30-day sequences for new annual comparisons; not reproduction of old formal folds',
                    data_sha256=digest(DATA_PATH) if raw is None else hashlib.sha256(raw.loc[:'2003-12-31'].to_csv().encode()).hexdigest(),
                    code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'code').glob('*.py')},
                    previous=str(Path(previous).resolve()) if previous else None,candidates=candidates,
                    folds=[dict(year=year,train_start=data['dates'][tr[0]],train_end=data['dates'][tr[-1]],
                                train_label_end=data['dates'][tr[-1]]+pd.Timedelta(days=1),
                                valid_start=data['dates'][va[0]],valid_end=data['dates'][va[-1]],
                                valid_label_end=data['dates'][va[-1]]+pd.Timedelta(days=1),
                                train_count=len(tr),valid_count=len(va)) for year,tr,va in splits])
    save_json(output/'protocol.json',protocol)
    rows, predictions, warning_rows, fits = [], [], [], []
    for candidate in [dict(id='persistence',family='persistence',mode='all')] + candidates:
        for year,tr,va in splits:
            print(f"[{stage}] {candidate['id']} year={year} train={len(tr)} valid={len(va)}",flush=True)
            threshold = float(np.quantile(data['raw'].Discharge.loc[:data['dates'][tr[-1]]],.95))
            current = data[candidate['mode']]
            if candidate['family'] == 'persistence':
                pred = current['X'][va,-1,:1]
            else:
                mode = candidate['mode']
                forced = tuple(current['names'].index(n) for n in ('Discharge_lag0','Prcp_lag0','Prcp_lag1','Prcp_lag2','Prcp_lag3')) if mode == 'forced' else ()
                if mode == 'target_mi':
                    forced = (0,)
                model = RunoffRegressor(family=candidate['family'],transform=candidate['transform'],
                    residual=candidate['residual'],seed=seed,top_k=24 if mode in ('mi','forced','target_mi') else None,
                    forced_indices=forced,mi_target='model_target' if mode == 'target_mi' else 'raw',
                    **candidate['params'])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(current['X'][tr],data['y'][tr])
                    pred = model.predict(current['X'][va])
                warning_rows.append(dict(candidate=candidate['id'],year=year,
                    warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]))
                ann_training = getattr(model,'ann_training_',None)
                if ann_training:
                    ann_training = dict(ann_training,
                        inner_train_end=data['dates'][tr[ann_training['probe_train_count']-1]],
                        inner_validation_start=data['dates'][tr[ann_training['validation_start']]],
                        inner_validation_end=data['dates'][tr[-1]])
                fits.append(dict(candidate=candidate['id'],year=year,selected_indices=model.selected_.tolist(),
                    selected_names=[current['names'][i] for i in model.selected_],
                    mi_target=model.mi_target,
                    peak_weight=model.peak_weight,peak_training_threshold=model.peak_threshold_,
                    ann_solver=model.ann_solver,ann_learning_rate=model.ann_learning_rate,
                    ann_activation=model.ann_activation,
                    ann_training=ann_training,
                    threshold=threshold,n_iter=getattr(model.estimator_,'n_iter_',None),
                    convergence_warning=any(w.category.__name__=='ConvergenceWarning' for w in caught),
                    loss_history=getattr(model,'loss_history_',getattr(model.estimator_,'loss_curve_',[]))))
            if pred.shape != data['y'][va].shape or not np.isfinite(pred).all():
                raise ValueError('Nonfinite or misaligned prediction')
            rows.append(dict(candidate=candidate['id'],family=candidate['family'],year=year,
                             threshold=threshold,**metrics(data['y'][va],pred,threshold)))
            predictions.extend(dict(candidate=candidate['id'],year=year,origin=str(date.date()),
                target_date=str((date+pd.Timedelta(days=1)).date()),truth=float(y),prediction=float(p),
                threshold=threshold,peak=bool(y>=threshold))
                for date,y,p in zip(data['dates'][va],data['y'][va,0],pred[:,0]))
            pd.DataFrame(rows).to_csv(output/'fold_metrics.csv',index=False)
            pd.DataFrame(predictions).to_csv(output/'predictions.csv',index=False)
            save_json(output/'warnings.json',warning_rows)
            save_json(output/'fits.json',fits)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['candidate','family'],as_index=False).agg(
        mean_r2=('r2','mean'),worst_r2=('r2','min'),mean_rmse=('rmse','mean'),
        mean_mae=('mae','mean'),mean_peak_rmse=('peak_rmse','mean'),mean_peak_bias=('peak_bias','mean'))
    summary.sort_values(['worst_r2','mean_r2'],ascending=False).to_csv(output/'summary.csv',index=False)
    save_json(output/'complete.json',dict(stage=stage,candidates=len(candidates),fits=len(fits),status='complete'))
    return summary


def run_fixed_h1_ensemble(output, climate_source, ann_source, ann_candidate='ANN_tune_01'):
    """Fuse already-recorded annual predictions with predeclared equal weights."""
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; no overwrite or hot start')
    sources = {'climate_rf': Path(climate_source), 'compact_ann': Path(ann_source),
               'persistence': Path(climate_source)}
    frames, protocols = [], {}
    keys = ['year','origin','target_date','truth','threshold','peak']
    for member in fixed_h1_ensemble_members():
        source = sources[member['id']]
        protocol_path, prediction_path = source/'protocol.json', source/'predictions.csv'
        if not protocol_path.is_file() or not prediction_path.is_file():
            raise ValueError(f"Missing recorded source for {member['id']}")
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        if protocol['data_cutoff'] != '2003-12-31' or protocol['horizon'] != 1 or protocol['gap'] != 7:
            raise ValueError(f"Incompatible source protocol for {member['id']}")
        protocols[member['id']] = dict(path=str(source.resolve()), protocol_sha256=digest(protocol_path),
                                       seed=protocol['seed'])
        candidate = ann_candidate if member['id'] == 'compact_ann' else member['candidate']
        rows = pd.read_csv(prediction_path)
        rows = rows[rows.candidate.eq(candidate)][keys+['prediction']].copy()
        if rows.empty or rows.duplicated(keys).any():
            raise ValueError(f"Invalid source predictions for {member['id']}")
        frames.append(rows.rename(columns={'prediction':member['id']}))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how='inner', validate='one_to_one')
    if len(merged) != len(frames[0]) or not np.isfinite(merged[['climate_rf','compact_ann','persistence']]).all().all():
        raise ValueError('Component predictions are not exactly aligned')
    if len({record['seed'] for record in protocols.values()}) != 1:
        raise ValueError('Component sources must use the same seed')
    merged['prediction'] = merged[['climate_rf','compact_ann','persistence']].mean(axis=1)
    rows = []
    for year, frame in merged.groupby('year', sort=True):
        threshold = float(frame.threshold.iloc[0])
        rows.append(dict(candidate='fixed_equal_rf_ann_persistence', family='fixed_equal_ensemble',
                         year=int(year), threshold=threshold, **metrics(frame.truth, frame.prediction, threshold)))
    summary = pd.DataFrame(rows).groupby(['candidate','family'],as_index=False).agg(
        mean_r2=('r2','mean'),worst_r2=('r2','min'),mean_rmse=('rmse','mean'),
        mean_mae=('mae','mean'),mean_peak_rmse=('peak_rmse','mean'),mean_peak_bias=('peak_bias','mean'))
    output.mkdir(parents=True)
    protocol = dict(stage='fixed_h1_ensemble', seed=next(iter(protocols.values()))['seed'],
                    data_cutoff='2003-12-31', horizon=1, gap=7,
                    selection='Predeclared equal 1/3 weights; no yearly weight fitting',
                    limitation='Training-period comparison is not an independent generalization claim',
                    members=fixed_h1_ensemble_members(), ann_source_candidate=ann_candidate, sources=protocols,
                    code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'code').glob('*.py')})
    save_json(output/'protocol.json', protocol)
    merged.assign(candidate='fixed_equal_rf_ann_persistence')[
        ['candidate']+keys+['prediction','climate_rf','compact_ann','persistence']].to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'fold_metrics.csv',index=False)
    summary.to_csv(output/'summary.csv',index=False)
    save_json(output/'complete.json',dict(stage='fixed_h1_ensemble',candidates=1,fits=0,status='complete'))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('features','targets','tune'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--families',nargs='+',choices=('RF','ANN','LSTM'),default=['RF','ANN','LSTM'])
    parser.add_argument('--modes',nargs='+',choices=MODES,default=DEFAULT_MODES,
                        help='Feature-stage input modes; default keeps the original four')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--limit',type=int,help='Maximum candidates per family, in predeclared grid order')
    args = parser.parse_args()
    run_stage(**vars(args))


if __name__ == '__main__':
    main()
