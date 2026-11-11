"""Choose among fixed ANN, RF and persistence using only prior OOF losses and known origin state."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from config import DATA_PATH, ROOT, digest, save_json
from data import load_raw
from research_h1 import metrics


MEMBERS = ['ANN', 'RF', 'persistence']
STATE = ['log_q', 'prcp', 'rain3', 'qdiff', 'month_sin', 'month_cos']


def assemble(raw, ann, rf):
    expected = pd.DatetimeIndex([d for year in (2001, 2002, 2003)
        for d in pd.date_range(f'{year}-01-01', f'{year}-12-24')])
    ann = ann.sort_values('origin').reset_index(drop=True)
    rf = rf.sort_values('origin').reset_index(drop=True)
    if (not pd.DatetimeIndex(ann.origin).equals(expected) or
            not pd.DatetimeIndex(rf.origin).equals(expected) or
            not pd.DatetimeIndex(ann.target_date).equals(expected+pd.Timedelta(days=1))):
        raise ValueError('Missing, duplicate, future or misaligned OOF dates')
    truth = raw.Discharge.reindex(expected+pd.Timedelta(days=1)).to_numpy()
    if not np.array_equal(ann.truth.to_numpy(), truth):
        raise ValueError('ANN truth does not match teacher next-day flow')
    result = pd.DataFrame(dict(origin=expected, target_date=expected+pd.Timedelta(days=1), truth=truth,
        ANN=ann.prediction.to_numpy(), RF=rf.prediction_h1.to_numpy(),
        persistence=raw.Discharge.reindex(expected).to_numpy(),
        log_q=np.log1p(raw.Discharge.reindex(expected).to_numpy()),
        prcp=raw.Prcp.reindex(expected).to_numpy(),
        rain3=raw.Prcp.rolling(3).sum().reindex(expected).to_numpy(),
        qdiff=raw.Discharge.diff().reindex(expected).to_numpy(),
        month_sin=np.sin(2*np.pi*(expected.month-1)/12),
        month_cos=np.cos(2*np.pi*(expected.month-1)/12)))
    if not np.isfinite(result[MEMBERS+STATE+['truth']].to_numpy(float)).all():
        raise ValueError('Nonfinite source predictions, state or truth')
    return result


def evaluate(frame, output, raw, data_sha256='unit-test fixture'):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Output exists; inspect it instead of repeating the run')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'protocol.json', dict(data_cutoff='2003-12-31', years=[2002, 2003],
        members=MEMBERS, state=STATE, max_depth=2, min_samples_leaf=60, seed=42,
        training_target='three member daily squared errors', test_evaluations=0,
        data_sha256=data_sha256))
    rows, predictions, fits = [], [], []
    for year in (2002, 2003):
        valid = np.flatnonzero(frame.origin.dt.year == year)
        train = np.flatnonzero(frame.origin < frame.origin.iloc[valid[0]]-pd.Timedelta(days=7))
        if not len(train) or frame.origin.iloc[train[-1]]+pd.Timedelta(days=7)>=frame.origin.iloc[valid[0]]:
            raise ValueError('OOF history overlaps validation')
        truth = frame.truth.to_numpy(float)
        member = frame[MEMBERS].to_numpy(float)
        loss = (member[train]-truth[train,None])**2
        tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=60, random_state=42)
        tree.fit(frame.loc[train, STATE], loss)
        estimated = tree.predict(frame.loc[valid, STATE])
        selected = np.argmin(estimated, axis=1)
        gate = member[valid, selected]
        prior = int(np.argmin(loss.mean(axis=0)))
        values = dict(gate=gate, fixed_mean=member[valid].mean(axis=1),
            prior_winner=member[valid,prior], **{name:member[valid,i] for i,name in enumerate(MEMBERS)})
        model_path = output/f'tree_{year}.joblib'
        joblib.dump(tree, model_path)
        fits.append(dict(year=year, train_count=len(train), train_end=frame.origin.iloc[train[-1]],
            validation_start=frame.origin.iloc[valid[0]], validation_count=len(valid),
            prior_winner=MEMBERS[prior], tree_sha256=digest(model_path)))
        threshold = float(raw.Discharge.loc[:frame.origin.iloc[train[-1]]].quantile(.95))
        for name, pred in values.items():
            rows.append(dict(year=year, model=name, threshold=threshold,
                **metrics(truth[valid], pred, threshold)))
            for j, idx in enumerate(valid):
                predictions.append(dict(year=year, model=name, origin=frame.origin.iloc[idx].date(),
                    target_date=frame.target_date.iloc[idx].date(), truth=truth[idx],
                    prediction=pred[j], selected=MEMBERS[selected[j]] if name=='gate' else '',
                    estimated_loss=estimated[j,selected[j]] if name=='gate' else np.nan))
        pd.DataFrame(fits).to_json(output/'fits.json', orient='records', indent=2, date_format='iso')
        pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
        pd.DataFrame(predictions).to_csv(output/'predictions.csv', index=False)
        print(f'state gate {year}: R2={rows[-6]["r2"]:.6f}', flush=True)
    result = dict(status='complete', meta_fits=2, base_fits=0, test_evaluations=0)
    save_json(output/'complete.json', result)
    return result


def run(output):
    parent = ROOT/'output/teacher_selection_20261109'
    sources = [(parent/'ann_depth_seed42', 'ANN_first_one'),
        (parent/'direct_rf_seed42', 'selected')]
    frames = []
    for folder, candidate in sources:
        import json
        protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
        complete = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        if (protocol['data_cutoff']!='2003-12-31' or protocol['seed']!=42 or
                protocol['data_sha256']!=digest(DATA_PATH) or complete['status']!='complete'):
            raise ValueError('Incompatible or incomplete base predictions')
        source = pd.read_csv(folder/'predictions.csv', parse_dates=['origin','target_date'] if candidate.startswith('ANN') else ['origin'],
            float_precision='round_trip')
        source = source.loc[source.candidate.eq(candidate) if 'candidate' in source else source.model.eq(candidate)]
        frames.append(source)
    raw = load_raw().loc[:'2003-12-31']
    frame = assemble(raw, *frames)
    result = evaluate(frame, output, raw, data_sha256=digest(DATA_PATH))
    save_json(Path(output)/'sources.json', [dict(directory=str(p.relative_to(ROOT)), candidate=c,
        predictions_sha256=digest(p/'predictions.csv'), protocol_sha256=digest(p/'protocol.json')) for p,c in sources])
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
