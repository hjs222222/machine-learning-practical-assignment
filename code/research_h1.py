"""Pre-registered h=1 research; only the explicit frozen backtest consumes 2004 labels."""
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
WEATHER_ABLATIONS = {'weather_no_vp':('Vp_diff1','Vp_diff3'),
                    'weather_no_srad':('Srad_diff1','Srad_diff3'),
                    'weather_no_temp':('Tavg_diff1','Tavg_diff3','Trange_lag0','Trange_diff1')}
RAIN_WINDOWS = {'weather_rain_sum30':1.,'weather_rain_decay05':.5,
                'weather_rain_decay08':.8,'weather_rain_decay095':.95}
FLOW_INTERACTIONS = ('weather_qdiff1','weather_rain_q','weather_rain_qdiff1')
CALENDAR_INPUTS = {'weather_month':('month_sin','month_cos'),
                   'weather_season':('season_winter','season_spring','season_summer','season_autumn')}
MODES = DEFAULT_MODES + ('compact','target_mi','climate','daily','weather') + tuple(WEATHER_ABLATIONS) + tuple(RAIN_WINDOWS) + FLOW_INTERACTIONS + tuple(CALENDAR_INPUTS) + ('weather_radiation','climate_radiation','daily_radiation','climate_solar_raw')
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


def prepare_data(raw, cutoff='2003-12-31', horizons=1, available_weather=False):
    if horizons not in (1,7):
        raise ValueError('horizons仅支持1或7')
    raw = raw.loc[:cutoff].copy()
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
    daily = base[[f'{col}_lag0' for col in ('Discharge','Dayl','Prcp','Srad','Tmax','Tmin','Vp')]]
    climate = compact.copy()
    avg = (raw.Tmax+raw.Tmin)/2
    weather = daily.copy()
    for col in ('Vp','Srad'):
        for days in (1,3):
            weather[f'{col}_diff{days}'] = raw[col].diff(days)
    for days in (1,3):
        weather[f'Tavg_diff{days}'] = avg.diff(days)
    weather['Trange_lag0'] = raw.Tmax-raw.Tmin
    weather['Trange_diff1'] = (raw.Tmax-raw.Tmin).diff(1)
    if available_weather:
        ids = np.flatnonzero(weather.notna().all(axis=1).to_numpy())
        ids = ids[ids < len(raw)-horizons]
        dates = raw.index[ids]
        return dict(dates=dates,raw=raw,
            y=np.column_stack([raw.Discharge.shift(-h).iloc[ids].to_numpy() for h in range(1,horizons+1)]),
            weather=dict(X=weather.iloc[ids].to_numpy(np.float32)[:,None,:],names=list(weather.columns),dates=dates))
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
    ids = np.array([i for i in range(29, len(raw)-horizons) if ready[i-29:i+1].all()])
    dates = raw.index[ids]
    result = dict(dates=dates, y=np.column_stack([raw.Discharge.shift(-h).iloc[ids].to_numpy() for h in range(1,horizons+1)]),
                  raw=raw)
    frames = dict(hydro=hydro,compact=compact,climate=climate,daily=daily,weather=weather)
    energy = raw.Srad*raw.Dayl/1_000_000
    for mode in ('weather','climate','daily'):
        frames[f'{mode}_radiation'] = frames[mode].assign(Srad_energy_MJ_m2_lag0=energy)
    frames['climate_solar_raw'] = climate.assign(Srad_lag0=raw.Srad,Dayl_lag0=raw.Dayl)
    frames.update({mode:pd.concat([weather,base[list(columns)]],axis=1) for mode,columns in CALENDAR_INPUTS.items()})
    frames.update(dict(weather_qdiff1=weather.assign(Discharge_diff1=raw.Discharge.diff(1)),
        weather_rain_q=weather.assign(Prcp_times_Discharge=raw.Prcp*raw.Discharge),
        weather_rain_qdiff1=weather.assign(Prcp_times_Discharge_diff1=raw.Prcp*raw.Discharge.diff(1))))
    frames.update({mode:weather.drop(columns=list(removed)) for mode,removed in WEATHER_ABLATIONS.items()})
    for mode,decay in RAIN_WINDOWS.items():
        name = 'Prcp_prior_sum30' if decay == 1 else f'Prcp_prior_decay30_{decay:g}'
        frames[mode] = weather.assign(**{name:sum(decay**(k-1)*raw.Prcp.shift(k) for k in range(1,31))})
    for mode in MODES:
        frame = frames.get(mode,base)
        result[mode] = dict(X=np.stack([frame.iloc[i-29:i+1].to_numpy(np.float32) for i in ids]),
                            names=list(frame.columns), dates=dates)
    return result


def annual_splits(dates, horizons=1):
    dates = pd.DatetimeIndex(dates)
    for year in (2001, 2002, 2003):
        va = np.flatnonzero((dates.year == year) & ((dates + pd.Timedelta(days=horizons)).year == year))
        tr = np.flatnonzero(dates < dates[va[0]] - pd.Timedelta(days=7))
        if not len(tr) or dates[tr[-1]] + pd.Timedelta(days=horizons) >= dates[va[0]]:
            raise ValueError('Invalid annual temporal boundary')
        yield year, tr, va


def metrics(truth, prediction, threshold):
    truth, prediction = np.asarray(truth).ravel(), np.asarray(prediction).ravel()
    error = prediction - truth
    peak = error[truth >= threshold]
    return dict(r2=float(r2_score(truth,prediction)), rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(abs(error))), peak_rmse=float(np.sqrt(np.mean(peak**2))) if len(peak) else None,
                peak_bias=float(np.mean(peak)) if len(peak) else None, peak_count=int(len(peak)))


def select_initializations(training_predictions, seeds=(13,42,2026)):
    """Choose one seed per configuration/year using only final raw-flow training MSE."""
    if (not isinstance(seeds,(tuple,list)) or len(seeds) != 3 or
            any(isinstance(s,(bool,np.bool_)) or not isinstance(s,(int,np.integer)) or not 0 <= s < 2**32 for s in seeds) or
            len(set(seeds)) != 3):
        raise ValueError('seeds须为三个不同的非bool整数，范围0..2^32-1')
    columns = ['candidate','year','seed','origin','target_date','truth','prediction']
    if (not isinstance(training_predictions,pd.DataFrame) or training_predictions.empty or
            not set(columns).issubset(training_predictions.columns)):
        raise ValueError('Incomplete training prediction table')
    frame = training_predictions[columns].copy()
    if frame.isna().any().any() or not np.isfinite(frame[['truth','prediction']].to_numpy(float)).all():
        raise ValueError('Nonfinite training prediction table')
    if frame.duplicated(['candidate','year','seed','origin']).any():
        raise ValueError('Duplicate training origins')
    chosen = []
    for (candidate,year),part in frame.groupby(['candidate','year'],sort=True):
        if set(part.seed) != set(seeds):
            raise ValueError('Expected training predictions for declared seeds')
        origin,target = pd.to_datetime(part.origin),pd.to_datetime(part.target_date)
        if not (target == origin+pd.Timedelta(days=1)).all() or (target >= pd.Timestamp(int(year),1,1)).any():
            raise ValueError('Training target dates cross validation boundary or are misaligned')
        reference = None
        scores = []
        for seed,group in part.groupby('seed',sort=True):
            group = group.sort_values('origin').reset_index(drop=True)
            keys = group[['origin','target_date','truth']]
            if reference is not None and not reference.equals(keys):
                raise ValueError('Incomparable training origins or truth across seeds')
            reference = keys
            with np.errstate(over='ignore',invalid='ignore'):
                mse = float(np.mean((group.prediction.to_numpy(float)-group.truth.to_numpy(float))**2))
            if not np.isfinite(mse):
                raise ValueError('Nonfinite training MSE')
            scores.append((mse,int(seed)))
        mse,seed = min(scores)
        chosen.append(dict(candidate=candidate,year=int(year),seed=seed,training_mse=mse))
    return pd.DataFrame(chosen)


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
              raw=None, candidates=None, modes=DEFAULT_MODES, available_weather=False, record_training=False):
    if not isinstance(record_training,bool):
        raise ValueError('record_training须为bool')
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
    if any(candidate.get('outputs',1) not in (1,7) for candidate in candidates):
        raise ValueError('outputs仅支持1或7')
    if available_weather and any(c.get('family') != 'ANN' or c.get('mode') != 'weather' or
            c.get('params',{}).get('seq_len') != 1 for c in candidates):
        raise ValueError('available_weather仅支持显式seq_len=1的weather ANN候选')
    label_horizon = max(candidate.get('outputs',1) for candidate in candidates)
    if record_training and label_horizon != 1:
        raise ValueError('record_training仅支持h1')
    data = prepare_data(load_raw() if raw is None else raw,horizons=label_horizon,available_weather=available_weather)
    splits = list(annual_splits(data['dates'],horizons=label_horizon))
    output.mkdir(parents=True)
    protocol = dict(stage=stage,seed=seed,data_cutoff='2003-12-31',horizon=1,label_horizon=label_horizon,gap=7,
                    sequence_length=1 if available_weather else 30,available_weather=available_weather,record_training=record_training,
                    selection='worst annual R2, then mean annual R2',
                    limitation='Training-period selection is biased; no independent generalization claim',
                    sample_note=('Available original weather origins, one-step ANN; same annual validation dates' if available_weather else
                                 'Common 30-day sequences for new annual comparisons; not reproduction of old formal folds'),
                    data_sha256=digest(DATA_PATH) if raw is None else hashlib.sha256(raw.loc[:'2003-12-31'].to_csv().encode()).hexdigest(),
                    code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'code').glob('*.py')},
                    previous=str(Path(previous).resolve()) if previous else None,candidates=candidates,
                    folds=[dict(year=year,train_start=data['dates'][tr[0]],train_end=data['dates'][tr[-1]],
                                train_label_end=data['dates'][tr[-1]]+pd.Timedelta(days=label_horizon),
                                valid_start=data['dates'][va[0]],valid_end=data['dates'][va[-1]],
                                valid_label_end=data['dates'][va[-1]]+pd.Timedelta(days=label_horizon),
                                train_count=len(tr),valid_count=len(va)) for year,tr,va in splits])
    save_json(output/'protocol.json',protocol)
    rows, predictions, warning_rows, fits, horizon_rows = [], [], [], [], []
    training_predictions = []
    for candidate in [dict(id='persistence',family='persistence',mode='weather' if available_weather else 'all')] + candidates:
        for year,tr,va in splits:
            print(f"[{stage}] {candidate['id']} year={year} train={len(tr)} valid={len(va)}",flush=True)
            threshold = float(np.quantile(data['raw'].Discharge.loc[:data['dates'][tr[-1]]],.95))
            current = data[candidate['mode']]
            outputs = candidate.get('outputs',1)
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
                    model.fit(current['X'][tr],data['y'][tr,:outputs])
                    pred = model.predict(current['X'][va])
                    train_pred = model.predict(current['X'][tr]) if record_training else None
                if record_training:
                    if train_pred.shape != (len(tr),1) or not np.isfinite(train_pred).all():
                        raise ValueError('Nonfinite or misaligned training prediction')
                    training_predictions.extend(dict(candidate=candidate['id'],seed=seed,year=year,
                        origin=str(date.date()),target_date=str((date+pd.Timedelta(days=1)).date()),
                        truth=float(y),prediction=float(p))
                        for date,y,p in zip(data['dates'][tr],data['y'][tr,0],train_pred[:,0]))
                warning_rows.append(dict(candidate=candidate['id'],year=year,
                    warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]))
                ann_training = getattr(model,'ann_training_',None)
                if ann_training:
                    ann_training = dict(ann_training,
                        inner_train_end=data['dates'][tr[ann_training['probe_train_count']-1]],
                        inner_validation_start=data['dates'][tr[ann_training['validation_start']]],
                        inner_validation_end=data['dates'][tr[-1]])
                lstm_training = getattr(model,'lstm_training_',None)
                if lstm_training:
                    lstm_training = dict(lstm_training,
                        inner_train_end=data['dates'][tr[lstm_training['probe_train_count']-1]],
                        inner_validation_start=data['dates'][tr[lstm_training['validation_start']]],
                        inner_validation_end=data['dates'][tr[-1]])
                linear_skip = getattr(model.estimator_,'skip',None)
                lstm_head = model.estimator_.head if model.family == 'LSTM' else None
                batch_size = (min(model.ann_batch_size,len(tr)) if model.family == 'ANN' and model.ann_solver == 'adam' else
                              min(128,len(tr)) if model.family == 'LSTM' else None)
                fits.append(dict(candidate=candidate['id'],year=year,outputs=outputs,selected_indices=model.selected_.tolist(),
                    selected_names=[current['names'][i] for i in model.selected_],
                    mi_target=model.mi_target,
                    x_scaling=model.x_scaling,
                    input_transform=model.input_transform,
                    first_day_weight=model.first_day_weight,output_scale=model.output_scale_.tolist(),
                    peak_weight=model.peak_weight,peak_training_threshold=model.peak_threshold_,
                    peak_weight_normalizer=model.peak_weight_normalizer_,
                    ann_solver=model.ann_solver,ann_learning_rate=model.ann_learning_rate,
                    ann_backend=model.ann_backend,ann_raw_loss=model.ann_raw_loss,
                    ann_refinement=model.ann_refinement,
                    ann_refinement_state=getattr(model,'ann_refinement_',None),
                    ann_lr_schedule=model.ann_lr_schedule,
                    learning_rate_history=getattr(model,'learning_rate_history_',None),
                    ann_grad_clip=model.ann_grad_clip,
                    lstm_grad_clip=model.lstm_grad_clip,
                    gradient_clip=(model.ann_grad_clip if model.family == 'ANN' and model.ann_backend == 'torch' else
                                   model.lstm_grad_clip if model.family == 'LSTM' else None),
                    ann_linear_skip=model.ann_linear_skip,
                    linear_skip_parameters=sum(p.numel() for p in linear_skip.parameters()) if linear_skip is not None else 0,
                    linear_skip_weight_norm=float(linear_skip.weight.detach().norm().item()) if linear_skip is not None else None,
                    linear_skip_bias_norm=float(linear_skip.bias.detach().norm().item()) if linear_skip is not None else None,
                    ann_batch_size=model.ann_batch_size,batch_size=batch_size,
                    batches_per_epoch=(len(tr)+batch_size-1)//batch_size if batch_size else None,
                    last_batch_size=(len(tr)-1)%batch_size+1 if batch_size else None,
                    ann_activation=model.ann_activation,ann_history=model.ann_history,
                    input_width=getattr(model.estimator_,'n_features_in_',None),
                    rf_max_features_per_split=model.estimator_.estimators_[0].max_features_ if model.family == 'RF' else None,
                    lstm_raw_loss=model.lstm_raw_loss,lstm_learning_rate=model.lstm_learning_rate,
                    lstm_readout=model.lstm_readout,
                    lstm_bidirectional=model.lstm_bidirectional,
                    lstm_directions=(2 if model.estimator_.lstm.bidirectional else 1) if model.family == 'LSTM' else None,
                    lstm_hidden_per_direction=model.estimator_.lstm.hidden_size if model.family == 'LSTM' else None,
                    lstm_total_parameters=sum(p.numel() for p in model.estimator_.parameters()) if model.family == 'LSTM' else None,
                    lstm_linear_skip=model.lstm_linear_skip,
                    lstm_skip_parameters=(model.estimator_.skip_weight.numel()+model.estimator_.skip_bias.numel()
                                          if model.family == 'LSTM' and hasattr(model.estimator_,'skip_weight') else 0),
                    lstm_skip_weight_norm=(float(model.estimator_.skip_weight.detach().norm().item())
                                          if model.family == 'LSTM' and hasattr(model.estimator_,'skip_weight') else None),
                    lstm_skip_bias_norm=(float(model.estimator_.skip_bias.detach().norm().item())
                                        if model.family == 'LSTM' and hasattr(model.estimator_,'skip_weight') else None),
                    lstm_readout_width=lstm_head.in_features if lstm_head is not None else None,
                    lstm_head_parameters=sum(p.numel() for p in lstm_head.parameters()) if lstm_head is not None else None,
                    raw_loss_scale=getattr(model,'raw_loss_scale_',None),
                    training_mse=float(np.mean((train_pred-data['y'][tr,:1])**2)) if record_training else None,
                    ann_training=ann_training,lstm_training=lstm_training,
                    threshold=threshold,n_iter=getattr(model.estimator_,'n_iter_',None),
                    convergence_warning=any(w.category.__name__=='ConvergenceWarning' for w in caught),
                    loss_history=getattr(model,'loss_history_',getattr(model.estimator_,'loss_curve_',[]))))
            if pred.shape != (len(va),outputs) or not np.isfinite(pred).all():
                raise ValueError('Nonfinite or misaligned prediction')
            rows.append(dict(candidate=candidate['id'],family=candidate['family'],year=year,
                             threshold=threshold,**metrics(data['y'][va,0],pred[:,0],threshold)))
            for h in range(outputs):
                horizon_rows.append(dict(candidate=candidate['id'],family=candidate['family'],year=year,
                    horizon=h+1,threshold=threshold,**metrics(data['y'][va,h],pred[:,h],threshold)))
            predictions.extend(dict(candidate=candidate['id'],year=year,origin=str(date.date()),
                target_date=str((date+pd.Timedelta(days=1)).date()),truth=float(y),prediction=float(p),
                threshold=threshold,peak=bool(y>=threshold),
                **{f'prediction_h{h+1}':float(vector[h]) for h in range(1,outputs)},
                **{f'truth_h{h+1}':float(truth[h]) for h in range(1,outputs)})
                for date,y,p,vector,truth in zip(data['dates'][va],data['y'][va,0],pred[:,0],pred,data['y'][va]))
            pd.DataFrame(rows).to_csv(output/'fold_metrics.csv',index=False)
            pd.DataFrame(predictions).to_csv(output/'predictions.csv',index=False)
            if record_training and training_predictions:
                pd.DataFrame(training_predictions).to_csv(output/'training_predictions.csv',index=False)
            if label_horizon > 1:
                pd.DataFrame(horizon_rows).to_csv(output/'horizon_metrics.csv',index=False)
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


def run_h1_2004_backtest(output, raw=None, seed=42):
    """One retrospective evaluation of the frozen h=1 equal-weight ensemble."""
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; no overwrite or hot start')
    source = load_raw() if raw is None else raw
    data = prepare_data(source, cutoff='2004-12-31')
    train = np.flatnonzero(data['dates'] <= pd.Timestamp('2003-12-30'))
    test = np.flatnonzero((data['dates'].year == 2004) &
                          ((data['dates'] + pd.Timedelta(days=1)).year == 2004))
    if not len(train) or not len(test) or data['dates'][train[-1]] + pd.Timedelta(days=1) > data['dates'][test[0]]:
        raise ValueError('Invalid frozen train/test boundary')
    specs = [
        ('climate_rf', 'climate', dict(family='RF', transform='sqrt', residual=False, top_k=None,
                                       trees=120, leaf=5, max_features=.5)),
        ('compact_ann', 'compact', dict(family='ANN', transform='raw', residual=True, top_k=None,
                                        hidden=32, iterations=350, alpha=1.0)),
    ]
    members, warnings_by_member = {}, {}
    for name, mode, params in specs:
        model = RunoffRegressor(seed=seed, **params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(data[mode]['X'][train], data['y'][train])
            members[name] = model.predict(data[mode]['X'][test]).ravel().astype(np.float64)
        warnings_by_member[name] = [dict(category=item.category.__name__, message=str(item.message)) for item in caught]
    members['persistence'] = data['compact']['X'][test,-1,0].astype(np.float64)
    prediction = np.mean(np.column_stack([members['climate_rf'],members['compact_ann'],members['persistence']]),axis=1)
    threshold = float(np.quantile(data['raw'].Discharge.loc[:data['dates'][train[-1]]],.95))
    output.mkdir(parents=True)
    protocol = dict(stage='frozen_h1_2004_backtest', seed=seed, train_end=str(data['dates'][train[-1]].date()),
                    test_start=str(data['dates'][test[0]].date()), test_end=str(data['dates'][test[-1]].date()),
                    horizon=1, selection='Frozen before reading 2004 labels; no 2004 tuning',
                    members=fixed_h1_ensemble_members(), data_sha256=digest(DATA_PATH) if raw is None else
                    hashlib.sha256(source.to_csv().encode()).hexdigest(),
                    code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'code').glob('*.py')})
    save_json(output/'protocol.json',protocol)
    frame = pd.DataFrame(dict(candidate='fixed_equal_rf_ann_persistence', year=2004,
        origin=[str(value.date()) for value in data['dates'][test]],
        target_date=[str((value+pd.Timedelta(days=1)).date()) for value in data['dates'][test]],
        truth=data['y'][test,0], prediction=prediction, threshold=threshold,
        peak=data['y'][test,0] >= threshold, **members))
    frame.to_csv(output/'predictions.csv',index=False,float_format='%.17g')
    row = dict(candidate='fixed_equal_rf_ann_persistence', family='fixed_equal_ensemble', year=2004,
               threshold=threshold, **metrics(frame.truth,frame.prediction,threshold))
    pd.DataFrame([row]).to_csv(output/'fold_metrics.csv',index=False)
    summary = pd.DataFrame([row]).drop(columns=['year','threshold'])
    summary.rename(columns={'r2':'mean_r2','rmse':'mean_rmse','mae':'mean_mae',
                            'peak_rmse':'mean_peak_rmse','peak_bias':'mean_peak_bias'},inplace=True)
    summary['worst_r2'] = summary['mean_r2']
    summary = summary[['candidate','family','mean_r2','worst_r2','mean_rmse','mean_mae','mean_peak_rmse','mean_peak_bias']]
    summary.to_csv(output/'summary.csv',index=False)
    save_json(output/'warnings.json',warnings_by_member)
    save_json(output/'complete.json',dict(stage='frozen_h1_2004_backtest',candidates=1,fits=2,status='complete'))
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
