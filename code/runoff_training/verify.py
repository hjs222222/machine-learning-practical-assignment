"""只读重建指标并对照旧交付；不重新训练、不修改冻结配置。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from acceptance3_model import load_dataset
from acceptance3_train import ROOT, metrics_table


KEY = ['strategy', 'model', 'seed', 'sample_id', 'h']


def verify_run(run):
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed' or manifest['smoke']:
        raise ValueError('只允许检查已经完成的正式run，不能把smoke当作正式结果')
    data_file = ROOT/'Teacher Resources & Requirements/01047000.csv'
    assert hashlib.sha256(data_file.read_bytes()).hexdigest() == manifest['data_sha256']
    data = load_dataset(str(run/'dataset.npz'))
    frame = pd.read_csv(run/'test_predictions_all_seeds.csv')
    metrics = pd.read_csv(run/'metrics_all_seeds.csv')
    assert len(frame) == 2 * 5 * 3 * 7 * 359
    assert len(metrics) == 2 * 5 * 3 * 7
    assert set(frame.seed) == {42, 43, 44}
    assert set(frame.strategy) == {'direct', 'multi'}
    assert set(frame.model) == {'ANN', 'RF', 'LSTM', 'Mean', 'Stacking'}
    assert not frame.duplicated(KEY).any()
    ids = frame.sample_id.to_numpy(int)
    np.testing.assert_allclose(frame.truth, data['y'][ids, frame.h.to_numpy(int) - 1], rtol=0, atol=1e-10)
    assert (frame.origin_date.to_numpy() == data['origin_dates'][ids]).all()
    expected_dates = (data['origin_days'][ids] + frame.h.to_numpy(int)).astype('datetime64[D]').astype(str)
    assert (frame.target_date.to_numpy() == expected_dates).all()
    expected_ids = np.flatnonzero((data['origin_days'] >= np.datetime64('2004-01-01').astype('int64')) &
        (data['origin_days'] + 7 <= np.datetime64('2004-12-31').astype('int64')))
    assert len(expected_ids) == 359
    for _, part in frame.groupby(['strategy', 'model', 'seed', 'h']):
        np.testing.assert_array_equal(np.sort(part.sample_id), expected_ids)
    all_tables = []
    for seed, part in frame.groupby('seed'):
        table = metrics_table(part.to_dict('records'))
        table['seed'] = seed
        all_tables.append(table)
    actual = pd.concat(all_tables).sort_values(['strategy', 'model', 'seed', 'h'])
    expected = metrics.sort_values(['strategy', 'model', 'seed', 'h'])
    np.testing.assert_allclose(actual[['R2', 'RMSE', 'MAE']], expected[['R2', 'RMSE', 'MAE']], rtol=1e-10)
    pivot = frame.pivot(index=['strategy', 'seed', 'sample_id', 'h'], columns='model', values='pred')
    np.testing.assert_allclose(pivot.Mean, pivot[['ANN', 'RF', 'LSTM']].mean(axis=1), rtol=1e-10)
    for (strategy, seed), own in frame.groupby(['strategy', 'seed']):
        for h in (range(1, 8) if strategy == 'direct' else [7]):
            part = own[own.h == h] if strategy == 'direct' else own
            pieces = [part[part.model == family].pivot(index='sample_id', columns='h', values='pred').sort_index()
                      for family in ['ANN', 'RF', 'LSTM']]
            x = np.column_stack([p.to_numpy() for p in pieces])
            model = joblib.load(run/'meta_models'/f'{strategy}_h{h}_seed{seed}.joblib')
            pred = np.asarray(model.predict(x)).reshape(len(x), -1)
            expected_stack = part[part.model == 'Stacking'].pivot(index='sample_id', columns='h', values='pred').sort_index()
            np.testing.assert_allclose(pred, expected_stack.to_numpy(), rtol=1e-9, atol=1e-7)
    histories = list((run/'training_history').glob('*.json'))
    assert len(histories) == 872
    for p in histories:
        record = json.loads(p.read_text(encoding='utf-8'))
        assert data['origin_days'][record['last_id']] + 7 <= np.datetime64('2003-12-31').astype('int64')
        if record['family'] == 'LSTM':
            assert len(record['history']) == record['params']['fixed_epochs']
        if 'cv_score' in record:
            assert data['origin_days'][record['last_id']] + 7 < data['origin_days'][record['validation_first_id']]
    hashes = json.loads((run/'model_hashes_before_test.json').read_text(encoding='utf-8'))
    for name, digest in hashes.items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest() == digest
    for file in (run/'splits').glob('*.csv'):
        part = pd.read_csv(file)
        assert (pd.to_datetime(part.max_train_target) < pd.to_datetime(part.validation_origin_start)).all()
    result = dict(status='passed', all_seed_metric_rows=len(metrics),
        all_seed_prediction_rows=len(frame), required_primary_rows=42,
        mean_reconstructed=True, stacking_reconstructed=True,
        metrics_reconstructed=True, dates_and_truth_verified=True,
        fit_records=len(histories), frozen_files_verified=len(hashes), test_origins=359)
    return result, frame, metrics


def compare_to_delivery(frame, metrics, delivery):
    previous = pd.read_csv(delivery/'data/任务1至7+三seed逐日预测与来源.csv')
    old_metrics = pd.read_csv(delivery/'data/任务1至7+三seed完整指标与来源.csv')
    joined = frame.merge(previous, on=KEY, suffixes=('', '_previous'), validate='one_to_one')
    assert len(joined) == len(frame)
    np.testing.assert_allclose(joined.truth, joined.truth_previous, rtol=0, atol=1e-9)
    assert (joined.origin_date == joined.origin_date_previous).all()
    if 'target_date_previous' in joined:
        assert (joined.target_date == joined.target_date_previous).all()
    old = old_metrics[['strategy', 'model', 'seed', 'h', 'R2', 'RMSE', 'MAE']]
    compared = metrics.merge(old, on=['strategy', 'model', 'seed', 'h'],
                             suffixes=('', '_previous'), validate='one_to_one')
    assert len(compared) == len(metrics)
    for name in ['RMSE', 'MAE']:
        compared[f'{name}_change_pct'] = (compared[name] / compared[f'{name}_previous'] - 1) * 100
    primary = compared[compared.seed == 42].groupby(['strategy', 'model'])[
        ['R2', 'R2_previous', 'RMSE', 'RMSE_previous', 'MAE', 'MAE_previous']].mean()
    primary['RMSE_change_pct'] = (primary.RMSE / primary.RMSE_previous - 1) * 100
    return compared, primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--export-dir', type=Path,
                        help='可选：将核验和比较另存到新的空目录，不修改原run')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ['.', '..']:
        parser.error('run-id必须为单目录名')
    run = ROOT/'output/acceptance3_runs'/args.run_id
    result, frame, metrics = verify_run(run)
    compared, primary = compare_to_delivery(frame, metrics, ROOT/'验收/Acceptance 3')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(primary.to_string())
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=False)
        (args.export_dir/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        compared.to_csv(args.export_dir/'comparison_by_horizon.csv', index=False, encoding='utf-8-sig')
        primary.to_csv(args.export_dir/'comparison_primary_summary.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()
