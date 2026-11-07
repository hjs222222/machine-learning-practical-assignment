import json
import numpy as np
import pandas as pd
import pytest
import research_h1 as rh
from models import RunoffRegressor


def raw_data():
    dates = pd.date_range('2000-01-01', '2004-12-31')
    n = len(dates)
    return pd.DataFrame({c: 10 + np.sin(np.arange(n)/13) + np.arange(n)*.001
                         for c in ['Discharge','Dayl','Prcp','Srad','Swe','Tmax','Tmin','Vp']}, index=dates)


def test_cutoff_causality_and_common_origins():
    raw = raw_data()
    a = rh.prepare_data(raw)
    changed = raw.copy()
    changed.loc['2003-06-01':] *= 99
    b = rh.prepare_data(changed)
    assert a['dates'].max() == pd.Timestamp('2003-12-30')
    for mode in rh.MODES:
        assert np.array_equal(a[mode]['dates'], a['dates'])
        before = a['dates'] < '2003-06-01'
        np.testing.assert_array_equal(a[mode]['X'][before], b[mode]['X'][before])
    assert a['all']['X'].shape[-1] == 91
    assert a['hydro']['names'][0] == 'Discharge_lag0'
    assert a['hydro']['X'].shape[-1] == 101
    only_2004 = raw.copy()
    only_2004.loc['2004-01-01':] = 1e8
    cut = rh.prepare_data(only_2004)
    np.testing.assert_array_equal(a['y'],cut['y'])
    for mode in rh.MODES:
        np.testing.assert_array_equal(a[mode]['X'],cut[mode]['X'])
    date = a['dates'][80]
    names = a['hydro']['names']
    values = a['hydro']['X'][80,-1]
    for days in (1,2,3,7):
        assert values[names.index(f'Discharge_diff{days}')] == pytest.approx(
            raw.Discharge.loc[date]-raw.Discharge.loc[date-pd.Timedelta(days=days)],abs=1e-6)
    for days in (2,3,5,7,14,30):
        assert values[names.index(f'Prcp_prior_sum{days}')] == pytest.approx(
            raw.Prcp.loc[date-pd.Timedelta(days=days):date-pd.Timedelta(days=1)].sum(),rel=1e-6)
    for year, tr, va in rh.annual_splits(a['dates']):
        assert a['dates'][tr[-1]] + pd.Timedelta(days=7) < a['dates'][va[0]]
        assert (a['dates'][va].year == year).all()
        assert ((a['dates'][va] + pd.Timedelta(days=1)).year == year).all()


def test_optional_cutoff_exposes_test_origins_without_changing_training_rows():
    raw = raw_data()
    train = rh.prepare_data(raw)
    full = rh.prepare_data(raw, cutoff='2004-12-31')
    keep = full['dates'] <= pd.Timestamp('2003-12-30')
    assert full['dates'].max() == pd.Timestamp('2004-12-30')
    np.testing.assert_array_equal(full['dates'][keep], train['dates'])
    np.testing.assert_array_equal(full['y'][keep], train['y'])
    for mode in rh.MODES:
        np.testing.assert_array_equal(full[mode]['X'][keep], train[mode]['X'])


@pytest.mark.parametrize('horizons',[1,7])
def test_available_weather_uses_original_valid_origins_and_preserves_overlap(horizons):
    raw=raw_data();common=rh.prepare_data(raw,horizons=horizons)
    explicit=rh.prepare_data(raw,horizons=horizons,available_weather=False)
    available=rh.prepare_data(raw,horizons=horizons,available_weather=True)
    assert common['dates'][0]==pd.Timestamp('2000-02-29')
    for mode in rh.MODES:
        np.testing.assert_array_equal(common[mode]['X'],explicit[mode]['X'])
    assert available['dates'][0]==pd.Timestamp('2000-01-04')
    assert available['dates'][-1]==pd.Timestamp('2003-12-31')-pd.Timedelta(days=horizons)
    assert available['weather']['X'].shape==(len(available['dates']),1,15)
    assert np.isfinite(available['weather']['X']).all()
    assert available['weather']['names']==common['weather']['names']
    ids=available['dates'].get_indexer(common['dates']);assert (ids>=0).all()
    np.testing.assert_array_equal(available['weather']['X'][ids,0],common['weather']['X'][:,-1])
    np.testing.assert_array_equal(available['y'][ids],common['y'])
    expected=np.column_stack([raw.Discharge.shift(-h).reindex(available['dates']) for h in range(1,horizons+1)])
    np.testing.assert_array_equal(available['y'],expected)
    for (year,tr,va),(_,ctr,cva) in zip(rh.annual_splits(available['dates'],horizons),rh.annual_splits(common['dates'],horizons)):
        assert len(tr)=={2001:356,2002:721,2003:1086}[year] and len(tr)==len(ctr)+56
        np.testing.assert_array_equal(available['dates'][va],common['dates'][cva])
        assert available['dates'][tr[-1]]+pd.Timedelta(days=7)<available['dates'][va[0]]


@pytest.mark.parametrize('horizons',[1,7])
def test_available_weather_keeps_future_and_2004_out_of_features(horizons):
    raw=raw_data();a=rh.prepare_data(raw,horizons=horizons,available_weather=True)
    future=raw.copy();future.loc['2003-06-01':]*=99
    b=rh.prepare_data(future,horizons=horizons,available_weather=True)
    before=a['dates']<'2003-06-01'
    np.testing.assert_array_equal(a['weather']['X'][before],b['weather']['X'][before])
    heldout=raw.copy();heldout.loc['2004-01-01':]=1e9
    c=rh.prepare_data(heldout,horizons=horizons,available_weather=True)
    np.testing.assert_array_equal(a['dates'],c['dates'])
    np.testing.assert_array_equal(a['y'],c['y'])
    np.testing.assert_array_equal(a['weather']['X'],c['weather']['X'])


@pytest.mark.parametrize('family,mode,params',[
    ('RF','weather',dict(seq_len=1)),('ANN','climate',dict(seq_len=1)),
    ('ANN','weather',dict(seq_len=7)),('ANN','weather',{})])
def test_available_weather_rejects_unsupported_candidate_before_output(family,mode,params,tmp_path):
    c=dict(id='invalid',family=family,mode=mode,transform='sqrt',residual=True,params=params)
    out=tmp_path/'invalid'
    with pytest.raises(ValueError,match='available_weather'):
        rh.run_stage('features',out,raw=raw_data(),candidates=[c],available_weather=True)
    assert not out.exists()


@pytest.mark.parametrize('available',[False,True])
@pytest.mark.parametrize('outputs',[1,7])
def test_available_weather_stage_records_true_dates_and_persistence(available,outputs,tmp_path):
    raw=raw_data();out=tmp_path/f'weather_{available}_{outputs}'
    c=dict(id='ANN',family='ANN',mode='weather',transform='power025',residual=True,
           outputs=outputs,params=dict(hidden=2,iterations=1,seq_len=1))
    rh.run_stage('features',out,raw=raw,candidates=[c],available_weather=available)
    protocol=json.loads((out/'protocol.json').read_text());p=pd.read_csv(out/'predictions.csv')
    assert protocol['available_weather']==available
    assert protocol['sequence_length']==(1 if available else 30)
    assert protocol['folds'][0]['train_start'].startswith('2000-01-04' if available else '2000-02-29')
    assert [f['train_count'] for f in protocol['folds']]==([356,721,1086] if available else [300,665,1030])
    count=364 if outputs==1 else 358
    assert len(p)==2*3*count
    baseline=p[p.candidate=='persistence']
    expected=raw.Discharge.reindex(pd.to_datetime(baseline.origin)).to_numpy().astype('float32')
    np.testing.assert_array_equal(baseline.prediction.to_numpy().astype('float32'),expected)
    np.testing.assert_allclose(p.truth,raw.Discharge.reindex(pd.to_datetime(p.target_date)),rtol=1e-12)
    if outputs==7:
        ann=p[p.candidate=='ANN'];last=pd.to_datetime(ann.origin)+pd.Timedelta(days=7)
        assert last.max()<=pd.Timestamp('2003-12-31')
        np.testing.assert_allclose(ann.truth_h7,raw.Discharge.reindex(last),rtol=1e-12)


def test_forced_selection_and_skip_mi(monkeypatch):
    x = np.random.default_rng(4).uniform(1, 10, (40, 3, 8))
    y = x[:, -1, :1]
    monkeypatch.setattr('models.mutual_info_regression', lambda *a, **k: np.arange(8))
    m = RunoffRegressor(top_k=2, forced_indices=(0,1), trees=2).fit(x,y)
    assert set(m.selected_) == {0,1,6,7}
    monkeypatch.setattr('models.mutual_info_regression', lambda *a, **k: pytest.fail('unnecessary MI'))
    assert len(RunoffRegressor(top_k=None, trees=2).fit(x,y).selected_) == 8


def initialization_training_table():
    dates=pd.date_range('2000-01-01',periods=3)
    return pd.DataFrame([dict(candidate='A',year=2001,seed=seed,origin=str(date.date()),
        target_date=str((date+pd.Timedelta(days=1)).date()),truth=float(i+1),
        prediction=float(i+1+(1 if seed==13 else .5)))
        for seed in (13,42,2026) for i,date in enumerate(dates)])


@pytest.mark.parametrize('seeds',[(7,101,2027),(0,101,2**32-1),(np.int64(7),101,2027)])
def test_select_initializations_accepts_explicit_group_and_preserves_default(seeds):
    original=initialization_training_table()
    pd.testing.assert_frame_equal(rh.select_initializations(original),
        rh.select_initializations(original,seeds=(13,42,2026)))
    other=original.copy();other['seed']=other.seed.map({13:seeds[2],42:seeds[1],2026:seeds[0]})
    result=rh.select_initializations(other,seeds=seeds)
    assert result.to_dict('records')==[dict(candidate='A',year=2001,seed=int(seeds[0]),training_mse=.25)]
    pd.testing.assert_frame_equal(result,rh.select_initializations(other,seeds=list(reversed(seeds))))
    with pytest.raises(ValueError):rh.select_initializations(other)


@pytest.mark.parametrize('seeds',[None,'7,101,2027',[],[7,101],[7,101,101],[7,101,True],
                                  [7,101,-1],[7,101,2**32],[7,101,1.5],[7,101,2027.0]])
def test_select_initializations_rejects_invalid_declared_seed_groups(seeds):
    with pytest.raises(ValueError,match='seeds'):
        rh.select_initializations(initialization_training_table(),seeds=seeds)


def test_select_initializations_uses_raw_training_mse_ties_and_independent_groups():
    a=initialization_training_table();b=a.copy();b['candidate']='B'
    b['prediction']=b.truth+np.where(b.seed==13,0.,2.)
    result=rh.select_initializations(pd.concat([a,b],ignore_index=True))
    assert result.to_dict('records')==[dict(candidate='A',year=2001,seed=42,training_mse=.25),
                                      dict(candidate='B',year=2001,seed=13,training_mse=0.)]
    changed=pd.concat([a,b],ignore_index=True).sample(frac=1,random_state=8)
    changed['validation_r2']=np.where(changed.seed==13,1e9,-1e9)
    pd.testing.assert_frame_equal(result,rh.select_initializations(changed))


@pytest.mark.parametrize('bad',['missing_seed','missing_row','duplicate','truth','date','future',
                               'nonfinite','overflow','empty','missing_column'])
def test_select_initializations_rejects_incomparable_or_invalid_training_tables(bad):
    frame=initialization_training_table()
    if bad=='missing_seed':frame=frame[frame.seed!=2026]
    elif bad=='missing_row':frame=frame.iloc[:-1]
    elif bad=='duplicate':frame=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
    elif bad=='truth':frame.loc[0,'truth']+=1
    elif bad=='date':frame.loc[0,'target_date']='2000-01-03'
    elif bad=='future':
        frame['origin']='2001-01-01';frame['target_date']='2001-01-02'
        frame=frame.groupby('seed',as_index=False).head(1)
    elif bad=='nonfinite':frame.loc[0,'prediction']=np.nan
    elif bad=='overflow':frame.loc[0,'prediction']=1e308
    elif bad=='empty':frame=frame.iloc[:0]
    elif bad=='missing_column':frame=frame.drop(columns='origin')
    with pytest.raises(ValueError):rh.select_initializations(frame)


@pytest.mark.parametrize('family',['RF','ANN','LSTM'])
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_record_training_uses_final_fit_predictions_and_keeps_validation_unchanged(family,tmp_path):
    raw=raw_data();c=dict(id=family,family=family,mode='weather',transform='sqrt',residual=True,
        params=dict(trees=2,hidden=2,epochs=1,iterations=1,seq_len=1))
    plain=tmp_path/'plain';recorded=tmp_path/'recorded'
    rh.run_stage('features',plain,raw=raw,candidates=[c],seed=13)
    rh.run_stage('features',recorded,raw=raw,candidates=[c],seed=13,record_training=True)
    assert not (plain/'training_predictions.csv').exists()
    assert not json.loads((plain/'protocol.json').read_text())['record_training']
    assert json.loads((recorded/'protocol.json').read_text())['record_training']
    np.testing.assert_allclose(pd.read_csv(plain/'predictions.csv').prediction,
        pd.read_csv(recorded/'predictions.csv').prediction,rtol=1e-10,atol=1e-8)
    train=pd.read_csv(recorded/'training_predictions.csv',float_precision='round_trip')
    fits=json.loads((recorded/'fits.json').read_text());old_fits=json.loads((plain/'fits.json').read_text())
    assert set(train.candidate)=={family} and set(train.seed)=={13}
    data=rh.prepare_data(raw)
    for (year,tr,_),fit,old in zip(rh.annual_splits(data['dates']),fits,old_fits):
        part=train[train.year==year]
        assert len(part)==len(tr) and fit['loss_history']==old['loss_history']
        assert old['training_mse'] is None
        np.testing.assert_array_equal(pd.to_datetime(part.origin),data['dates'][tr])
        np.testing.assert_array_equal(pd.to_datetime(part.target_date),data['dates'][tr]+pd.Timedelta(days=1))
        np.testing.assert_array_equal(part.truth,data['y'][tr,0])
        assert pd.to_datetime(part.target_date).max()<pd.Timestamp(year,1,1)
        manual=RunoffRegressor(family=family,top_k=None,transform='sqrt',residual=True,
            seed=13,**c['params']).fit(data['weather']['X'][tr],data['y'][tr])
        expected=manual.predict(data['weather']['X'][tr])[:,0]
        np.testing.assert_allclose(part.prediction,expected,rtol=1e-10,atol=1e-8)
        assert fit['training_mse']==pytest.approx(np.mean((expected-data['y'][tr,0])**2))


@pytest.mark.parametrize('flag',[1,'yes',None])
def test_record_training_rejects_nonboolean_flag_before_output(flag,tmp_path):
    out=tmp_path/'invalid'
    with pytest.raises(ValueError,match='record_training'):
        rh.run_stage('features',out,raw=raw_data(),candidates=[dict(id='RF',family='RF',mode='weather',
            transform='sqrt',residual=False,params=dict(trees=2))],record_training=flag)
    assert not out.exists()


def test_record_training_rejects_seven_outputs_before_output(tmp_path):
    out=tmp_path/'invalid'
    with pytest.raises(ValueError,match='record_training'):
        rh.run_stage('features',out,raw=raw_data(),record_training=True,candidates=[dict(id='RF',
            family='RF',mode='weather',transform='sqrt',residual=False,outputs=7,params=dict(trees=2))])
    assert not out.exists()


@pytest.mark.parametrize('bad',['shape','nonfinite'])
def test_record_training_rejects_invalid_final_training_prediction(bad,tmp_path,monkeypatch):
    original=RunoffRegressor.predict
    def capture(self,X):
        if len(X)==300:
            return np.ones((len(X),2)) if bad=='shape' else np.full((len(X),1),np.nan)
        return original(self,X)
    monkeypatch.setattr(RunoffRegressor,'predict',capture)
    with pytest.raises(ValueError,match='training prediction'):
        rh.run_stage('features',tmp_path/'bad',raw=raw_data(),record_training=True,candidates=[dict(
            id='RF',family='RF',mode='weather',transform='sqrt',residual=False,params=dict(trees=2))])


def test_minmax_only_fits_training_rows_and_does_not_clip_future_extremes():
    x=np.random.default_rng(106).uniform(1,10,(40,3,4)).astype('float32')
    x[30:]=1e5
    model=RunoffRegressor(trees=2,top_k=None,x_scaling='minmax').fit(x[:30],x[:30,-1,:1])
    np.testing.assert_allclose(model.x_scaler_.data_max_,x[:30].reshape(-1,4).max(axis=0))
    assert model.x_scaler_.n_samples_seen_==90
    assert (model.x_scaler_.transform(x[30:,-1])>1).all()
    assert np.isfinite(model.predict(x[30:])).all()


def test_unknown_input_scaling_rejected():
    with pytest.raises(ValueError,match='x_scaling'):
        RunoffRegressor(x_scaling='invalid').fit(np.ones((20,3,4)),np.ones((20,1)))


def test_signed_log_inputs_preserve_raw_flow_residual_and_predict_layout():
    x=np.random.default_rng(117).uniform(-10,10,(40,3,4)).astype('float32')
    x[:,:,0]=abs(x[:,:,0])+1
    y=x[:,-1,:1]*1.2+2
    model=RunoffRegressor(trees=2,top_k=None,input_transform='signed_log',
                          transform='sqrt',residual=True).fit(x[:30],y[:30])
    compressed=np.sign(x)*np.log1p(abs(x))
    np.testing.assert_allclose(model.x_scaler_.mean_,compressed[:30].reshape(-1,4).mean(axis=0),rtol=1e-6,atol=1e-6)
    expected_target=np.sqrt(y[:30])-np.sqrt(x[:30,-1,:1])
    np.testing.assert_allclose(model.y_scaler_.mean_,expected_target.mean(axis=0),rtol=1e-6)
    scaled=model.x_scaler_.transform(compressed[30:].reshape(-1,4)).reshape(10,3,4).astype('float32')
    z=model.y_scaler_.inverse_transform(model.estimator_.predict(scaled[:,-1]).reshape(-1,1))
    expected=np.maximum(z+np.sqrt(x[30:,-1,:1]),0)**2
    np.testing.assert_allclose(model.predict(x[30:]),expected,rtol=1e-6)


def test_unknown_input_transform_rejected():
    with pytest.raises(ValueError,match='input_transform'):
        RunoffRegressor(input_transform='invalid').fit(np.ones((20,3,4)),np.ones((20,1)))


def test_daily_mode_contains_only_observed_daily_values():
    raw=raw_data()
    data=rh.prepare_data(raw)
    cols=['Discharge','Dayl','Prcp','Srad','Tmax','Tmin','Vp']
    assert data['daily']['names']==[f'{col}_lag0' for col in cols]
    date=data['dates'][100]
    expected=raw.loc[date-pd.Timedelta(days=29):date,cols].to_numpy()
    np.testing.assert_allclose(data['daily']['X'][100],expected,rtol=1e-6)


def test_weather_changes_are_exact_causal_derivatives_of_teacher_fields():
    raw=raw_data();i=np.arange(len(raw))
    raw['Vp']=500+30*np.sin(i/3)
    raw['Srad']=200+20*np.cos(i/4)
    raw['Tmax']=10+8*np.sin(i/5)
    raw['Tmin']=3+4*np.cos(i/7)
    data=rh.prepare_data(raw);mode=data['weather']
    avg=(raw.Tmax+raw.Tmin)/2
    expected={f'{col}_diff{days}':raw[col].diff(days) for col in ('Vp','Srad') for days in (1,3)}
    expected.update({f'Tavg_diff{days}':avg.diff(days) for days in (1,3)})
    expected['Trange_lag0']=raw.Tmax-raw.Tmin
    expected['Trange_diff1']=(raw.Tmax-raw.Tmin).diff(1)
    assert len(mode['names'])==15
    np.testing.assert_array_equal(mode['X'][:,:,:7],data['daily']['X'])
    for name,values in expected.items():
        np.testing.assert_allclose(mode['X'][:,-1,mode['names'].index(name)],values.reindex(data['dates']),rtol=1e-6,atol=1e-6)


def test_weather_ablation_removes_only_named_derivative_group():
    data=rh.prepare_data(raw_data())
    groups={'vp':['Vp_diff1','Vp_diff3'],'srad':['Srad_diff1','Srad_diff3'],
            'temp':['Tavg_diff1','Tavg_diff3','Trange_lag0','Trange_diff1']}
    for group,removed in groups.items():
        mode=data[f'weather_no_{group}']
        names=[name for name in data['weather']['names'] if name not in removed]
        assert mode['names']==names and names[0]=='Discharge_lag0'
        ids=[data['weather']['names'].index(name) for name in names]
        np.testing.assert_array_equal(mode['X'],data['weather']['X'][:,:,ids])


@pytest.mark.parametrize('horizons',[1,7])
@pytest.mark.parametrize('mode',['weather_month','weather_season'])
def test_calendar_inputs_match_independent_month_and_season_tables(horizons,mode):
    raw=raw_data();data=rh.prepare_data(raw,horizons=horizons);part=data[mode]
    columns=['month_sin','month_cos'] if mode=='weather_month' else [f'season_{s}' for s in ('winter','spring','summer','autumn')]
    assert part['names']==data['weather']['names']+columns
    assert part['X'].shape==(len(data['dates']),30,15+len(columns))
    np.testing.assert_array_equal(part['dates'],data['dates'])
    np.testing.assert_array_equal(part['X'][:,:,:15],data['weather']['X'])
    root3=np.sqrt(3)/2
    sine=np.array([0,.5,root3,1,root3,.5,0,-.5,-root3,-1,-root3,-.5])
    cosine=np.array([1,root3,.5,0,-.5,-root3,-1,-root3,-.5,0,.5,root3])
    months=raw.index.month.to_numpy()
    expected=np.column_stack([sine[months-1],cosine[months-1]]) if mode=='weather_month' else np.column_stack([
        np.isin(months,m) for m in ((12,1,2),(3,4,5),(6,7,8),(9,10,11))])
    ends=raw.index.get_indexer(data['dates']);steps=ends[:,None]+np.arange(-29,1)[None,:]
    np.testing.assert_allclose(part['X'][:,:,15:],expected[steps].astype('float32'),rtol=1e-6,atol=1e-7)
    if mode=='weather_season':
        assert np.all(part['X'][:,:,15:].sum(axis=-1)==1)
        for date,wanted in [('2000-03-01',[0,1,0,0]),('2000-12-01',[1,0,0,0])]:
            np.testing.assert_array_equal(part['X'][data['dates'].get_loc(date),-1,15:],wanted)
    np.testing.assert_array_equal(data['y'],np.column_stack([
        raw.Discharge.shift(-h).reindex(data['dates']) for h in range(1,horizons+1)]))


def test_calendar_run_records_columns_native_outputs_and_default_modes(tmp_path):
    modes=['weather_month','weather_season']
    candidates=[dict(id=m,family='RF',mode=m,transform='sqrt',residual=True,outputs=7,
        params=dict(trees=2,seq_len=1)) for m in modes]
    out=tmp_path/'calendar';rh.run_stage('features',out,raw=raw_data(),candidates=candidates)
    protocol=json.loads((out/'protocol.json').read_text());fits=json.loads((out/'fits.json').read_text())
    assert [c['mode'] for c in protocol['candidates']]==modes and protocol['label_horizon']==7
    assert not protocol['available_weather'] and len(fits)==6
    for row in fits:
        columns=['month_sin','month_cos'] if row['candidate']=='weather_month' else [f'season_{s}' for s in ('winter','spring','summer','autumn')]
        assert row['outputs']==7 and row['input_width']==len(row['selected_names'])==15+len(columns)
        assert row['selected_names'][15:]==columns and row['selected_names'][0]=='Discharge_lag0'
    assert {c['mode'] for c in rh.stage_candidates('features',families=['RF'])}=={'mi','forced','all','hydro'}
    assert {c['mode'] for c in rh.stage_candidates('features',families=['RF'],modes=modes)}==set(modes)
    assert json.loads((out/'complete.json').read_text())['status']=='complete'


@pytest.mark.parametrize('horizons',[1,7])
@pytest.mark.parametrize('mode,column',[
    ('weather_qdiff1','Discharge_diff1'),
    ('weather_rain_q','Prcp_times_Discharge'),
    ('weather_rain_qdiff1','Prcp_times_Discharge_diff1')])
def test_flow_interactions_match_independent_original_field_formula(horizons,mode,column):
    raw=raw_data();rng=np.random.default_rng(916)
    raw['Discharge']=rng.choice([0.,1.,20.,100.,10000.],len(raw))
    raw['Prcp']=rng.choice([0.,1.,5.,30.],len(raw))
    data=rh.prepare_data(raw,horizons=horizons);part=data[mode]
    assert part['names']==data['weather']['names']+[column]
    assert part['X'].shape==(len(data['dates']),30,16)
    np.testing.assert_array_equal(part['dates'],data['dates'])
    np.testing.assert_array_equal(part['X'][:,:,:15],data['weather']['X'])
    q=raw.Discharge.to_numpy();p=raw.Prcp.to_numpy();expected=[]
    for i in range(1,len(raw)):
        expected.append(q[i]-q[i-1] if mode=='weather_qdiff1' else
                        p[i]*q[i] if mode=='weather_rain_q' else p[i]*(q[i]-q[i-1]))
    values=np.asarray(expected)
    for sample,date in enumerate(data['dates']):
        end=raw.index.get_loc(date)
        np.testing.assert_array_equal(part['X'][sample,:,-1],values[end-30:end].astype('float32'))
    assert np.any(part['X'][:,:,-1]==0) and np.isfinite(part['X']).all()
    if mode!='weather_rain_q':
        assert np.any(part['X'][:,:,-1]>0) and np.any(part['X'][:,:,-1]<0)
    expected_y=np.column_stack([raw.Discharge.shift(-h).reindex(data['dates']) for h in range(1,horizons+1)])
    np.testing.assert_array_equal(data['y'],expected_y)


def test_flow_interaction_run_records_columns_native_outputs_and_default_modes(tmp_path):
    modes=['weather_qdiff1','weather_rain_q','weather_rain_qdiff1']
    candidates=[dict(id=mode,family='RF',mode=mode,transform='sqrt',residual=True,
        outputs=7,params=dict(trees=2,seq_len=1)) for mode in modes]
    out=tmp_path/'flow_interaction'
    rh.run_stage('features',out,raw=raw_data(),candidates=candidates)
    protocol=json.loads((out/'protocol.json').read_text())
    fits=json.loads((out/'fits.json').read_text())
    assert [c['mode'] for c in protocol['candidates']]==modes and protocol['label_horizon']==7
    assert not protocol['available_weather'] and len(fits)==9
    for row in fits:
        assert row['outputs']==7 and row['input_width']==len(row['selected_names'])==16
        assert row['selected_names'][0]=='Discharge_lag0'
        assert row['selected_names'][-1]==dict(weather_qdiff1='Discharge_diff1',
            weather_rain_q='Prcp_times_Discharge',weather_rain_qdiff1='Prcp_times_Discharge_diff1')[row['candidate']]
    assert {c['mode'] for c in rh.stage_candidates('features',families=['RF'])}=={'mi','forced','all','hydro'}
    assert {c['mode'] for c in rh.stage_candidates('features',families=['RF'],modes=modes)}==set(modes)
    assert json.loads((out/'complete.json').read_text())['status']=='complete'


def test_rain_windows_are_exact_prior_only_weighted_sums():
    raw=raw_data()
    raw['Prcp']=np.random.default_rng(218).uniform(0,30,len(raw))
    data=rh.prepare_data(raw)
    modes={'weather_rain_sum30':1.,'weather_rain_decay05':.5,
           'weather_rain_decay08':.8,'weather_rain_decay095':.95}
    for mode,decay in modes.items():
        part=data[mode]
        assert len(part['names'])==16 and part['names'][:15]==data['weather']['names']
        np.testing.assert_array_equal(part['X'][:,:,:15],data['weather']['X'])
        for sample in (0,100,len(data['dates'])-1):
            end=raw.index.get_loc(data['dates'][sample])
            expected=[np.dot(raw.Prcp.iloc[i-30:i].to_numpy()[::-1],decay**np.arange(30))
                      for i in range(end-29,end+1)]
            np.testing.assert_allclose(part['X'][sample,:,-1],expected,rtol=1e-6,atol=1e-6)
    origin=data['dates'][100]
    changed=raw.copy();changed.loc[origin,'Prcp']+=1e5
    other=rh.prepare_data(changed)
    for mode in modes:
        assert other[mode]['X'][100,-1,-1]==data[mode]['X'][100,-1,-1]
    candidates=rh.stage_candidates('features',families=['RF'],modes=list(modes))
    assert {candidate['mode'] for candidate in candidates}==set(modes)


def test_rain_window_run_records_feature_names_and_keeps_default_modes(tmp_path):
    modes=['weather_rain_sum30','weather_rain_decay05','weather_rain_decay08','weather_rain_decay095']
    candidates=[dict(id=mode,family='RF',mode=mode,transform='sqrt',residual=True,
                     params=dict(trees=2,seq_len=1)) for mode in modes]
    out=tmp_path/'rain'
    rh.run_stage('features',out,raw=raw_data(),candidates=candidates)
    protocol=json.loads((out/'protocol.json').read_text())
    fits=json.loads((out/'fits.json').read_text())
    assert [c['mode'] for c in protocol['candidates']]==modes
    assert len(fits)==12
    for fit in fits:
        assert len(fit['selected_names'])==fit['input_width']==16
        assert fit['selected_names'][0]=='Discharge_lag0'
        assert fit['selected_names'][-1].startswith('Prcp_prior_')
    assert {c['mode'] for c in rh.stage_candidates('features',families=['RF'])}=={'mi','forced','all','hydro'}


def test_ann_research_records_actual_linear_skip_parameters(tmp_path):
    candidates=[dict(id=f'ANN_skip{skip}',family='ANN',mode='weather',transform='sqrt',residual=True,
        params=dict(ann_backend='torch',ann_linear_skip=skip,hidden=2,iterations=1,seq_len=1)) for skip in (False,True)]
    out=tmp_path/'linear_skip';rh.run_stage('features',out,raw=raw_data(),candidates=candidates)
    protocol=json.loads((out/'protocol.json').read_text());fits=json.loads((out/'fits.json').read_text())
    assert len(fits)==6 and [c['params']['ann_linear_skip'] for c in protocol['candidates']]==[False,True]
    for row in fits:
        assert row['ann_linear_skip']==row['candidate'].endswith('True')
        if row['ann_linear_skip']:
            assert row['linear_skip_parameters']==16 and row['linear_skip_weight_norm']>0
            assert np.isfinite(row['linear_skip_bias_norm'])
        else:
            assert row['linear_skip_parameters']==0 and row['linear_skip_weight_norm'] is row['linear_skip_bias_norm'] is None


def test_ann_research_records_effective_gradient_clip(tmp_path):
    candidates=[dict(id='torch5',family='ANN',mode='weather',transform='sqrt',residual=True,
        params=dict(ann_backend='torch',ann_grad_clip=5.,hidden=2,iterations=1,seq_len=1)),
        dict(id='sklearn',family='ANN',mode='weather',transform='sqrt',residual=True,params=dict(hidden=2,iterations=1,seq_len=1)),
        dict(id='LSTM',family='LSTM',mode='weather',transform='sqrt',residual=True,params=dict(hidden=2,epochs=1,seq_len=1)),
        dict(id='RF',family='RF',mode='weather',transform='sqrt',residual=True,params=dict(trees=2,seq_len=1))]
    out=tmp_path/'gradient_clip';rh.run_stage('features',out,raw=raw_data(),candidates=candidates)
    fits=json.loads((out/'fits.json').read_text())
    assert len(fits)==12
    for row in fits:
        assert row['ann_grad_clip']==(5. if row['candidate']=='torch5' else 1.)
        assert row['gradient_clip']==dict(torch5=5.,LSTM=1.,sklearn=None,RF=None)[row['candidate']]


def test_ann_research_records_actual_learning_rate_schedule(tmp_path):
    candidates=[dict(id='scheduled',family='ANN',mode='weather',transform='sqrt',residual=True,
        params=dict(ann_backend='torch',ann_lr_schedule='half_linear',hidden=2,iterations=4,seq_len=1)),
        dict(id='sklearn',family='ANN',mode='weather',transform='sqrt',residual=True,params=dict(hidden=2,iterations=1,seq_len=1)),
        dict(id='LSTM',family='LSTM',mode='weather',transform='sqrt',residual=True,params=dict(hidden=2,epochs=2,seq_len=1))]
    out=tmp_path/'lr_schedule';rh.run_stage('features',out,raw=raw_data(),candidates=candidates)
    fits=json.loads((out/'fits.json').read_text());assert len(fits)==9
    for row in fits:
        if row['candidate']=='scheduled':
            assert row['ann_lr_schedule']=='half_linear'
            np.testing.assert_allclose(row['learning_rate_history'],[.003,.003,.00165,.0003],rtol=1e-12)
        elif row['candidate']=='LSTM':
            assert row['ann_lr_schedule']=='constant' and row['learning_rate_history']==[.003,.003]
        else:assert row['ann_lr_schedule']=='constant' and row['learning_rate_history'] is None


def test_lstm_research_records_actual_readout_head(tmp_path):
    candidates=[dict(id=readout,family='LSTM',mode='daily',transform='log',residual=False,
        params=dict(lstm_readout=readout,hidden=2,epochs=1,seq_len=3)) for readout in ('last','last_mean')]
    candidates.append(dict(id='RF',family='RF',mode='daily',transform='sqrt',residual=False,params=dict(trees=2)))
    out=tmp_path/'readout';rh.run_stage('features',out,raw=raw_data(),candidates=candidates)
    fits=json.loads((out/'fits.json').read_text());assert len(fits)==9
    for row in fits:
        if row['candidate']=='RF':
            assert row['lstm_readout']=='last' and row['lstm_readout_width'] is row['lstm_head_parameters'] is None
        else:
            assert row['lstm_readout']==row['candidate']
            assert row['lstm_readout_width']==(4 if row['candidate']=='last_mean' else 2)
            assert row['lstm_head_parameters']==row['lstm_readout_width']+1


def test_ann_research_records_actual_batch_geometry(tmp_path):
    candidates=[dict(id=f'ANN_{backend}',family='ANN',mode='weather',transform='sqrt',
        residual=True,params=dict(ann_backend=backend,ann_batch_size=62,hidden=2,
                                 iterations=1,seq_len=1)) for backend in ('sklearn','torch')]
    candidates.append(dict(id='ANN_lbfgs',family='ANN',mode='weather',transform='sqrt',residual=True,
        params=dict(ann_solver='lbfgs',ann_batch_size=62,hidden=2,iterations=1,seq_len=1)))
    out=tmp_path/'ann_batch';raw=raw_data()
    rh.run_stage('features',out,raw=raw,candidates=candidates)
    splits={year:len(tr) for year,tr,_ in rh.annual_splits(rh.prepare_data(raw)['dates'])}
    for row in json.loads((out/'fits.json').read_text()):
        assert row['ann_batch_size']==62
        if row['ann_solver']=='lbfgs':
            assert row['batch_size'] is row['batches_per_epoch'] is row['last_batch_size'] is None
        else:
            n=splits[row['year']];b=min(62,n)
            assert (row['batch_size'],row['batches_per_epoch'],row['last_batch_size'])==(b,(n+b-1)//b,(n-1)%b+1)


def test_weighted_ann_research_records_actual_fit_threshold_and_normalizer(tmp_path):
    candidates=[dict(id=f'ANN_peak{w}',family='ANN',mode='weather',transform='power025',
        residual=True,params=dict(ann_backend='torch',peak_weight=w,hidden=2,
                                 iterations=1,seq_len=1)) for w in (1.,5.)]
    raw=raw_data();out=tmp_path/'ann_peak'
    rh.run_stage('features',out,raw=raw,candidates=candidates)
    data=rh.prepare_data(raw);splits={year:tr for year,tr,_ in rh.annual_splits(data['dates'])}
    fits=json.loads((out/'fits.json').read_text());assert len(fits)==6
    for row in fits:
        if row['peak_weight']==1:
            assert row['peak_training_threshold'] is None and row['peak_weight_normalizer'] is None
        else:
            y=data['y'][splits[row['year']],0];q=np.quantile(y,.95)
            assert row['peak_training_threshold']==pytest.approx(q)
            assert row['peak_weight_normalizer']==pytest.approx(np.where(y>q,5.,1.).mean())


def test_joint_labels_and_year_boundaries_keep_all_seven_targets_inside_year():
    raw=raw_data()
    data=rh.prepare_data(raw,horizons=7)
    assert data['dates'].max()==pd.Timestamp('2003-12-24')
    date=data['dates'][90]
    np.testing.assert_allclose(data['y'][90],raw.Discharge.loc[date+pd.Timedelta(days=1):date+pd.Timedelta(days=7)])
    changed=raw.copy();changed.loc['2004-01-01':]=1e9
    np.testing.assert_array_equal(data['y'],rh.prepare_data(changed,horizons=7)['y'])
    for year,tr,va in rh.annual_splits(data['dates'],horizons=7):
        assert len(va)==358
        assert data['dates'][tr[-1]]+pd.Timedelta(days=7)<data['dates'][va[0]]
        assert ((data['dates'][va]+pd.Timedelta(days=7)).year==year).all()


def test_direct_and_joint_run_share_dates_and_score_first_day(tmp_path):
    candidates=[dict(id=f'RF_outputs{outputs}',family='RF',mode='daily',transform='sqrt',
        residual=True,outputs=outputs,params=dict(trees=2,seq_len=1)) for outputs in (1,7)]
    out=tmp_path/'joint'
    rh.run_stage('features',out,raw=raw_data(),candidates=candidates)
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['label_horizon']==7 and protocol['horizon']==1
    p=pd.read_csv(out/'predictions.csv');m=pd.read_csv(out/'horizon_metrics.csv')
    a=p[p.candidate.eq('RF_outputs1')];b=p[p.candidate.eq('RF_outputs7')]
    np.testing.assert_array_equal(a.origin.to_numpy(),b.origin.to_numpy())
    assert len(a)==len(b)==3*358
    assert set(m[m.candidate.eq('RF_outputs7')].horizon)==set(range(1,8))
    assert b.prediction_h7.notna().all() and b.truth_h7.notna().all()


@pytest.mark.parametrize('transform',['raw','sqrt','log'])
@pytest.mark.parametrize('residual',[False,True])
def test_first_day_weight_scales_training_and_is_reversed_in_prediction(transform,residual,monkeypatch):
    from sklearn.ensemble import RandomForestRegressor
    observed=[]
    original=RandomForestRegressor.fit
    def capture(self,X,z,**kwargs):
        observed.append(z.copy())
        return original(self,X,z,**kwargs)
    monkeypatch.setattr(RandomForestRegressor,'fit',capture)
    x=np.random.default_rng(102).uniform(1,8,(30,3,4)).astype('float32')
    y=x[:,-1,:1]*np.arange(1,8)[None,:]+1
    model=RunoffRegressor(trees=2,top_k=None,transform=transform,residual=residual,
                          first_day_weight=4.).fit(x,y)
    target=model.target(y)-(model.target(x[:,-1,:1]) if residual else 0)
    z=model.y_scaler_.transform(target)*np.array([2.,1.,1.,1.,1.,1.,1.])
    np.testing.assert_allclose(observed[0],z,rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(model.output_scale_,[2,1,1,1,1,1,1])
    class Exact:
        def predict(self,X):return z
    model.estimator_=Exact()
    np.testing.assert_allclose(model.predict(x),y,rtol=1e-6)


@pytest.mark.parametrize('weight,outputs',[(0,7),(float('nan'),7),(4.,1)])
def test_first_day_weight_rejects_invalid_or_single_output(weight,outputs):
    with pytest.raises(ValueError,match='first_day_weight'):
        RunoffRegressor(first_day_weight=weight).fit(np.ones((20,3,4)),np.ones((20,outputs)))


@pytest.mark.parametrize('transform', ['raw','sqrt','log'])
@pytest.mark.parametrize('residual', [False,True])
@pytest.mark.parametrize('mi_target', ['raw','transformed','model_target'])
def test_mi_receives_only_actual_fold_training_target(monkeypatch,transform,residual,mi_target):
    x = np.random.default_rng(12).uniform(1,15,(38,3,8)).astype('float32')
    y = np.column_stack([x[:,-1,0]*1.2+2,x[:,-1,0]*.7+1])
    observed = []
    def capture_mi(features,target,**kwargs):
        observed.append((features.copy(),target.copy()))
        return np.arange(features.shape[1])
    monkeypatch.setattr('models.mutual_info_regression',capture_mi)
    model = RunoffRegressor(trees=2,top_k=3,transform=transform,residual=residual,mi_target=mi_target)
    model.fit(x[:25],y[:25])
    target = model.target(y[:25])
    learning_target = target-model.target(x[:25,-1,:1]) if residual else target
    expected = y[:25] if mi_target=='raw' else target if mi_target=='transformed' else learning_target
    assert len(observed)==2
    for h,(features,label) in enumerate(observed):
        np.testing.assert_array_equal(features,x[:25,-1])
        np.testing.assert_allclose(label,expected[:,h],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(model.y_scaler_.mean_,learning_target.mean(axis=0),rtol=1e-6,atol=1e-6)
    assert model.predict(x[25:]).shape==(13,2)


def test_target_mi_research_mode_and_fit_metadata(tmp_path):
    data = rh.prepare_data(raw_data())
    np.testing.assert_array_equal(data['target_mi']['X'],data['mi']['X'])
    assert len(rh.stage_candidates('features'))==12
    candidates = rh.stage_candidates('features',families=['RF'],modes=['target_mi'])
    candidates[0]['params']={'trees':2}
    rh.run_stage('features',tmp_path/'target_mi',raw=raw_data(),candidates=candidates)
    fits=json.loads((tmp_path/'target_mi'/'fits.json').read_text())
    assert len(fits)==3
    for fit in fits:
        assert fit['mi_target']=='model_target'
        assert fit['selected_names'][0]=='Discharge_lag0'
        assert 24<=len(fit['selected_names'])<=25


@pytest.mark.parametrize('weight',[1.,3.,10.])
def test_rf_peak_weights_use_only_training_labels(monkeypatch,weight):
    from sklearn.ensemble import RandomForestRegressor
    x=np.random.default_rng(71).uniform(1,10,(60,2,4)).astype('float32')
    y=x[:,-1,:1].copy()
    y[38:40]=100
    y[40:]=10000
    observed=[]
    original=RandomForestRegressor.fit
    def capture(self,X,target,sample_weight=None):
        observed.append(sample_weight)
        return original(self,X,target,sample_weight=sample_weight)
    monkeypatch.setattr(RandomForestRegressor,'fit',capture)
    model=RunoffRegressor(trees=2,top_k=None,peak_weight=weight).fit(x[:40],y[:40])
    if weight==1:
        assert observed==[None] and model.peak_threshold_ is None
    else:
        threshold=np.quantile(y[:40,0],.95)
        assert model.peak_threshold_==threshold
        np.testing.assert_array_equal(observed[0],np.where(y[:40,0]>threshold,weight,1.))
    assert model.predict(x[40:]).shape==(20,1)


@pytest.mark.parametrize('options',[
    {'peak_weight':.5},{'peak_weight':float('inf')},
    {'peak_weight':3,'family':'ANN'},{'peak_weight':3,'family':'LSTM'},
])
def test_invalid_peak_weight_contract(options):
    x=np.ones((20,2,4));y=np.ones((20,1))
    with pytest.raises(ValueError,match='peak_weight'):
        RunoffRegressor(top_k=None,**options).fit(x,y)


def test_peak_weight_is_h1_rf_only():
    with pytest.raises(ValueError,match='peak_weight'):
        RunoffRegressor(top_k=None,peak_weight=3).fit(np.ones((20,2,4)),np.ones((20,7)))


def test_compact_columns_values_and_explicit_modes():
    raw = raw_data()
    data = rh.prepare_data(raw)
    compact = data['compact']
    names = ([f'{col}_lag{lag}' for col in ('Discharge','Prcp') for lag in (0,1,2,3,7)]
             + [f'Discharge_diff{d}' for d in (1,2,3,7)]
             + [f'Prcp_prior_sum{d}' for d in (2,3,5,7,14,30)] + ['month_sin','month_cos'])
    assert compact['names'] == names and len(names) == 22
    assert np.array_equal(compact['dates'],data['dates'])
    date = data['dates'][100]
    expected = ([raw.loc[date-pd.Timedelta(days=lag),col] for col in ('Discharge','Prcp') for lag in (0,1,2,3,7)]
                + [raw.Discharge.loc[date]-raw.Discharge.loc[date-pd.Timedelta(days=d)] for d in (1,2,3,7)]
                + [raw.Prcp.loc[date-pd.Timedelta(days=d):date-pd.Timedelta(days=1)].sum() for d in (2,3,5,7,14,30)]
                + [np.sin(2*np.pi*(date.month-1)/12),np.cos(2*np.pi*(date.month-1)/12)])
    np.testing.assert_allclose(compact['X'][100,-1],expected,rtol=1e-6,atol=1e-6)
    assert len(rh.stage_candidates('features')) == 12
    candidates = rh.stage_candidates('features',modes=['compact'])
    assert len(candidates) == 3 and {c['mode'] for c in candidates} == {'compact'}


def test_climate_features_are_observed_causal_derivatives():
    raw = raw_data()
    i=np.arange(len(raw))
    raw['Tmax']=8*np.sin(i/5)+1
    raw['Tmin']=4*np.sin(i/5)-3
    raw['Prcp']=i%7+1.
    data=rh.prepare_data(raw)
    mode=data['climate']
    avg=(raw.Tmax+raw.Tmin)/2
    expected={'Tmax_lag0':raw.Tmax,'Tmin_lag0':raw.Tmin,'Tavg_lag0':avg}
    for days in (1,3,7):
        expected[f'Tavg_diff{days}']=avg-avg.shift(days)
    for days in (3,7,14):
        expected[f'Tavg_roll{days}_mean']=avg.rolling(days).mean()
        expected[f'Tavg_positive_sum{days}']=np.maximum(avg,0).rolling(days).sum()
    for days in (7,14,30):
        expected[f'ColdPrcp_prior_sum{days}']=(raw.Prcp*(avg<0)).shift(1).rolling(days).sum()
    expected['warming_wet7']=np.maximum(avg-avg.shift(3),0)*raw.Prcp.shift(1).rolling(7).sum()
    expected['warm_rain0']=raw.Prcp*np.maximum(avg,0)
    assert len(expected)==17 and len(mode['names'])==39
    assert mode['names'][:22]==data['compact']['names']
    np.testing.assert_array_equal(mode['X'][:,:,:22],data['compact']['X'])
    for name,series in expected.items():
        np.testing.assert_allclose(mode['X'][:,-1,mode['names'].index(name)],
                                   series.reindex(data['dates']),rtol=1e-6,atol=1e-6)
    assert np.isfinite(mode['X']).all()
    assert (expected['ColdPrcp_prior_sum7'].dropna()>0).any()
    assert (expected['warming_wet7'].dropna()>0).any()
    assert {c['mode'] for c in rh.stage_candidates('features',modes=['climate'])}=={'climate'}


@pytest.mark.parametrize('stage',['targets','tune'])
def test_nonfeature_modes_rejected_before_output(stage,tmp_path):
    with pytest.raises(ValueError,match='modes.*features'):
        rh.stage_candidates(stage,modes=['compact'])
    with pytest.raises(ValueError,match='modes.*features'):
        rh.run_stage(stage,tmp_path/'unused',modes=['compact'],candidates=[{}])
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('transform', ['raw','sqrt','log','power025','power075'])
@pytest.mark.parametrize('residual', [False,True])
def test_target_inverse(transform, residual):
    x = np.random.default_rng(7).uniform(1,10,(24,2,4))
    y = x[:,-1,:1] * 1.1
    model = RunoffRegressor(transform=transform,residual=residual,top_k=None,trees=2).fit(x,y)
    z = model.target(y) - (model.target(x[:,-1,:1]) if residual else 0)
    class Exact:
        def predict(self, values):
            return model.y_scaler_.transform(z)
    model.estimator_ = Exact()
    np.testing.assert_allclose(model.predict(x),y,rtol=1e-6)


def test_metrics_independently():
    y = np.array([1.,2.,3.,10.]); p = np.array([2.,2.,4.,8.])
    m = rh.metrics(y,p,5)
    assert m['r2'] == pytest.approx(1 - 6/np.sum((y-y.mean())**2))
    assert m['rmse'] == pytest.approx(np.sqrt(1.5))
    assert m['mae'] == 1
    assert m['peak_rmse'] == 2 and m['peak_bias'] == -2


def test_fixed_h1_ensemble_is_predeclared_and_equal_weighted():
    members = rh.fixed_h1_ensemble_members()
    assert [member['id'] for member in members] == ['climate_rf', 'compact_ann', 'persistence']
    assert sum(member['weight'] for member in members) == pytest.approx(1.0)
    assert all(member['weight'] == pytest.approx(1 / 3) for member in members)
    assert members[0]['mode'] == 'climate' and members[0]['family'] == 'RF'
    assert members[1]['mode'] == 'compact' and members[1]['family'] == 'ANN'


def test_fixed_h1_ensemble_requires_aligned_recorded_predictions(tmp_path):
    def source(name, ann=False):
        path=tmp_path/name;path.mkdir()
        (path/'protocol.json').write_text(json.dumps({'data_cutoff':'2003-12-31','horizon':1,'gap':7,'seed':9}),encoding='utf-8')
        rows=[]
        for year in (2001,2002,2003):
            for day in (1,2,3):
                rows.append(dict(candidate='ANN_tune_01' if ann else 'RF_tune_05',year=year,
                    origin=f'{year}-01-0{day}',target_date=f'{year}-01-0{day+1}',truth=float(day),
                    threshold=.5,peak=True,prediction=float(day+1)))
            if not ann:
                rows.extend(dict(candidate='persistence',year=year,origin=f'{year}-01-0{day}',
                    target_date=f'{year}-01-0{day+1}',truth=float(day),threshold=.5,peak=True,prediction=float(day))
                    for day in (1,2,3))
        pd.DataFrame(rows).to_csv(path/'predictions.csv',index=False)
        return path
    climate,ann=source('climate'),source('ann',ann=True)
    result=rh.run_fixed_h1_ensemble(tmp_path/'fused',climate,ann)
    predictions=pd.read_csv(tmp_path/'fused'/'predictions.csv')
    assert len(predictions)==9 and result.iloc[0].candidate=='fixed_equal_rf_ann_persistence'
    np.testing.assert_allclose(predictions.prediction,[5/3,8/3,11/3]*3)


def test_frozen_h1_backtest_writes_only_2004_predictions(tmp_path):
    out = tmp_path/'backtest'
    summary = rh.run_h1_2004_backtest(out, raw=raw_data(), seed=9)
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(out/'predictions.csv')
    assert protocol['stage'] == 'frozen_h1_2004_backtest'
    assert protocol['selection'] == 'Frozen before reading 2004 labels; no 2004 tuning'
    assert protocol['train_end'] == '2003-12-30'
    assert protocol['test_start'] == '2004-01-01'
    assert set(pd.to_datetime(predictions.origin).dt.year) == {2004}
    assert pd.to_datetime(predictions.target_date).max() == pd.Timestamp('2004-12-31')
    np.testing.assert_allclose(predictions.prediction,
                               predictions[['climate_rf','compact_ann','persistence']].mean(axis=1))
    assert (predictions.prediction - predictions[['climate_rf','compact_ann','persistence']].mean(axis=1)).abs().max() < 1e-10
    assert summary.iloc[0].candidate == 'fixed_equal_rf_ann_persistence'


def test_small_real_run_artifacts_and_stage_contract(tmp_path):
    candidate = dict(id='RF_all_raw',family='RF',mode='all',transform='raw',residual=False,params={'trees':2})
    out = tmp_path/'run'
    rh.run_stage('features',out,raw=raw_data(),candidates=[candidate],seed=9)
    protocol = json.loads((out/'protocol.json').read_text())
    assert protocol['seed'] == 9 and protocol['data_cutoff'] == '2003-12-31'
    assert protocol['code_hashes'] and protocol['data_sha256']
    folds = pd.read_csv(out/'fold_metrics.csv')
    predictions = pd.read_csv(out/'predictions.csv')
    assert len(folds) == 6 and set(folds.year) == {2001,2002,2003}
    assert predictions.groupby(['candidate','year']).size().min() > 300
    assert (out/'complete.json').exists() and (out/'warnings.json').exists()
    choices = rh.stage_candidates('targets',out,families=['RF'])
    assert len(choices) == 6 and {c['mode'] for c in choices} == {'all'}
    with pytest.raises(ValueError):
        rh.run_stage('features',out,raw=raw_data(),candidates=[candidate])
    with pytest.raises(ValueError):
        rh.stage_candidates('targets',tmp_path/'missing')
