"""只读核验继续训练的全部结果，并可向新目录导出轻量比较记录。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset, restore_model
from acceptance3_train import ROOT, metrics_table
from runoff_training.continue_train import gate_table
from runoff_training.verify import compare_to_delivery


KEY = ['strategy', 'model', 'seed', 'sample_id', 'h']
METRIC_KEY = ['strategy', 'model', 'seed', 'h']


def verify(run):
    torch.set_num_threads(4)
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed' and not manifest['smoke']
    assert hashlib.sha256((ROOT/'Teacher Resources & Requirements/01047000.csv').read_bytes()).hexdigest() == manifest['data_sha256']
    data = load_dataset(str(run/'dataset.npz'))
    frame = pd.read_csv(run/'test_predictions_all_seeds.csv')
    metrics = pd.read_csv(run/'metrics_all_seeds.csv')
    assert len(frame) == 75390 and len(metrics) == 210
    assert not frame.duplicated(KEY).any() and not metrics.duplicated(METRIC_KEY).any()
    assert np.isfinite(frame[['pred', 'truth']].to_numpy()).all()
    assert np.isfinite(metrics[['R2', 'RMSE', 'MAE']].to_numpy()).all()
    assert set(frame.seed) == {42, 43, 44}
    assert set(frame.strategy) == {'direct', 'multi'}
    assert set(frame.model) == {'ANN', 'RF', 'LSTM', 'Mean', 'Stacking'}
    ids = frame.sample_id.to_numpy(int)
    np.testing.assert_allclose(frame.truth, data['y'][ids, frame.h.to_numpy(int) - 1], rtol=0, atol=1e-10)
    assert (frame.origin_date.to_numpy() == data['origin_dates'][ids]).all()
    target_dates = (data['origin_days'][ids] + frame.h.to_numpy(int)).astype('datetime64[D]').astype(str)
    assert (frame.target_date.to_numpy() == target_dates).all()
    expected_ids = np.flatnonzero((data['origin_days'] >= np.datetime64('2004-01-01').astype('int64')) &
        (data['origin_days'] + 7 <= np.datetime64('2004-12-31').astype('int64')))
    for _, own in frame.groupby(METRIC_KEY):
        np.testing.assert_array_equal(np.sort(own.sample_id), expected_ids)
    rebuilt = []
    for seed, own in frame.groupby('seed'):
        table = metrics_table(own.to_dict('records'))
        table['seed'] = seed
        rebuilt.append(table)
    rebuilt = pd.concat(rebuilt).sort_values(METRIC_KEY)
    expected = metrics.sort_values(METRIC_KEY)
    np.testing.assert_allclose(rebuilt[['R2', 'RMSE', 'MAE']], expected[['R2', 'RMSE', 'MAE']], rtol=1e-10)
    primary = pd.read_csv(run/'metrics.csv')
    expected_primary = metrics[metrics.seed == 42].drop(columns='seed').sort_values(['strategy', 'model', 'h'])
    assert len(primary) == 70
    np.testing.assert_allclose(primary.sort_values(['strategy', 'model', 'h'])[['R2', 'RMSE', 'MAE']],
                               expected_primary[['R2', 'RMSE', 'MAE']], rtol=1e-10)
    pivot = frame.pivot(index=['strategy', 'seed', 'sample_id', 'h'], columns='model', values='pred')
    np.testing.assert_allclose(pivot.Mean, pivot[['ANN', 'RF', 'LSTM']].mean(axis=1), rtol=1e-10)
    previous_run = ROOT/'output/acceptance3_runs'/manifest['previous_run']
    previous_hashes = json.loads((previous_run/'model_hashes_before_test.json').read_text(encoding='utf-8'))
    final_reference_files = 0
    for name, digest in previous_hashes.items():
        assert hashlib.sha256((previous_run/name).read_bytes()).hexdigest() == digest
        normalized = name.replace(chr(92), '/')
        if normalized.startswith('models/final_') and ('_ANN_' in normalized or '_RF_' in normalized):
            final_reference_files += 1
    assert final_reference_files == 96
    oof = pd.read_csv(run/'oof_predictions.csv')
    previous_oof = pd.read_csv(previous_run/'oof_predictions.csv')
    assert len(oof) == 2 * 3 * 3 * 7 * 358
    assert not oof.duplicated(KEY).any()
    for family in ['ANN', 'RF']:
        own = oof[oof.model == family].sort_values(KEY)
        prior = previous_oof[previous_oof.model == family].sort_values(KEY)
        assert own[KEY].reset_index(drop=True).equals(prior[KEY].reset_index(drop=True))
        np.testing.assert_allclose(own[['truth', 'pred']], prior[['truth', 'pred']], rtol=1e-12)
    rebuilt_quarters = []
    for (strategy, seed, task_h, fold), own in oof.groupby(['strategy', 'seed', 'task_h', 'fold']):
        candidate = own[own.model == 'LSTM'].pivot(index='sample_id', columns='h', values='pred').sort_index()
        valid_ids = candidate.index.to_numpy(int)
        truth = data['y'][valid_ids, task_h - 1:task_h] if strategy == 'direct' else data['y'][valid_ids]
        prior = previous_oof[(previous_oof.strategy == strategy) & (previous_oof.seed == seed) &
            (previous_oof.task_h == task_h) & (previous_oof.fold == fold) & (previous_oof.model == 'LSTM')]
        prior_pred = prior.pivot(index='sample_id', columns='h', values='pred').reindex(valid_ids).to_numpy()
        # 原参照含双层LSTM；CPU与CUDA内核差异可被exp放大，使用训练时同设备核验。
        original = restore_model(run/'models'/f'{fold}_{strategy}_h{task_h}_seed{seed}_original', device=manifest['device'])
        original_pred = np.asarray(original.predict(valid_ids[:, None])).reshape(len(valid_ids), -1)
        candidate_model = restore_model(run/'models'/f'{fold}_{strategy}_h{task_h}_seed{seed}', device='cpu')
        predicted = np.asarray(candidate_model.predict(valid_ids[:, None])).reshape(len(valid_ids), -1)
        np.testing.assert_allclose(predicted, candidate.to_numpy(), rtol=3e-5, atol=.005)
        def rmse(p):
            return float(np.sqrt(np.square(p - truth).mean(axis=0)).mean())
        rebuilt_quarters.append(dict(strategy=strategy, task_h=task_h, seed=seed, fold=fold,
            candidate_RMSE=rmse(candidate.to_numpy()), original_RMSE=rmse(original_pred),
            previous_RMSE=rmse(prior_pred), candidate_MAE=float(np.abs(candidate.to_numpy() - truth).mean()),
            original_MAE=float(np.abs(original_pred - truth).mean()), previous_MAE=float(np.abs(prior_pred - truth).mean())))
    for (strategy, seed), own in frame.groupby(['strategy', 'seed']):
        for h in (range(1, 8) if strategy == 'direct' else [7]):
            part = own[own.h == h] if strategy == 'direct' else own
            matrices = [part[part.model == family].pivot(index='sample_id', columns='h', values='pred').sort_index()
                        for family in ['ANN', 'RF', 'LSTM']]
            x = np.column_stack([value.to_numpy() for value in matrices])
            meta = joblib.load(run/'meta_models'/f'{strategy}_h{h}_seed{seed}.joblib')
            predicted = np.asarray(meta.predict(x)).reshape(len(x), -1)
            saved = part[part.model == 'Stacking'].pivot(index='sample_id', columns='h', values='pred').sort_index()
            np.testing.assert_allclose(predicted, saved.to_numpy(), rtol=1e-9, atol=1e-7)
            # 从最终权重重新预测，与CSV逐项对照，不只重算指标。
            model = restore_model(run/'models'/f'final_{strategy}_h{h}_seed{seed}', device='cpu')
            predicted = np.asarray(model.predict(expected_ids[:, None])).reshape(len(expected_ids), -1)
            np.testing.assert_allclose(predicted, matrices[2].to_numpy(), rtol=3e-5, atol=.005)
    histories = list((run/'training_history').glob('*.json'))
    assert len(histories) == 728
    for file in histories:
        record = json.loads(file.read_text(encoding='utf-8'))
        assert data['origin_days'][record['last_id']] + 7 <= np.datetime64('2003-12-31').astype('int64')
        assert len(record['history']) == record['params']['fixed_epochs']
        if 'cv_score' in record:
            assert data['origin_days'][record['last_id']] + 7 < data['origin_days'][record['validation_first_id']]
            assert data['origin_days'][record['validation_last_id']] + 7 <= np.datetime64('2002-12-31').astype('int64')
    for file in (run/'splits').glob('*.csv'):
        own = pd.read_csv(file)
        assert (pd.to_datetime(own.max_train_target) < pd.to_datetime(own.validation_origin_start)).all()
    hashes = json.loads((run/'model_hashes_before_test.json').read_text(encoding='utf-8'))
    reused = json.loads((run/'reused_source_hashes.json').read_text(encoding='utf-8'))
    for name, digest in hashes.items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest() == digest
    for name, digest in reused.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    quarters = pd.read_csv(run/'development_quarter_metrics.csv')
    assert len(quarters) == 96
    quarter_keys = ['strategy', 'task_h', 'seed', 'fold']
    actual_quarters = pd.DataFrame(rebuilt_quarters).sort_values(quarter_keys)
    expected_quarters = quarters.sort_values(quarter_keys)
    columns = ['candidate_RMSE', 'original_RMSE', 'previous_RMSE',
               'candidate_MAE', 'original_MAE', 'previous_MAE']
    np.testing.assert_allclose(actual_quarters[columns], expected_quarters[columns], rtol=3e-5, atol=.005)
    gates = json.loads((run/'development_gates.json').read_text(encoding='utf-8'))
    computed_gates = gate_table(quarters.to_dict('records'))
    for actual, wanted in zip(computed_gates, gates):
        assert actual['strategy'] == wanted['strategy'] and actual['h'] == wanted['h']
        assert actual['passed_all_seeds'] == wanted['passed_all_seeds']
        for a, b in zip(actual['seed_results'], wanted['seed_results']):
            assert a['seed'] == b['seed'] and a['passed'] == b['passed']
            np.testing.assert_allclose([a[key] for key in ['RMSE_ratio', 'MAE_ratio', 'worst_quarter_ratio']],
                                       [b[key] for key in ['RMSE_ratio', 'MAE_ratio', 'worst_quarter_ratio']], rtol=1e-10)
    previous_frame = pd.read_csv(previous_run/'test_predictions_all_seeds.csv')
    merged = frame.merge(previous_frame, on=KEY, suffixes=('', '_previous'), validate='one_to_one')
    assert len(merged) == len(frame)
    np.testing.assert_allclose(merged.truth, merged.truth_previous, rtol=0, atol=1e-10)
    assert (merged.origin_date == merged.origin_date_previous).all()
    assert (merged.target_date == merged.target_date_previous).all()
    reused_rows = merged[merged.model.isin(['ANN', 'RF'])]
    np.testing.assert_allclose(reused_rows.pred, reused_rows.pred_previous, rtol=1e-12, atol=1e-9)
    previous_metrics = pd.read_csv(previous_run/'metrics_all_seeds.csv')
    compared = metrics.merge(previous_metrics[METRIC_KEY + ['R2', 'RMSE', 'MAE']], on=METRIC_KEY,
                              suffixes=('', '_previous'), validate='one_to_one')
    for key in ['RMSE', 'MAE']:
        compared[key + '_change_pct'] = (compared[key] / compared[key + '_previous'] - 1) * 100
    delivery, _ = compare_to_delivery(frame, metrics, ROOT/'验收/Acceptance 3')
    result = dict(status='passed', all_seed_prediction_rows=len(frame), all_seed_metric_rows=len(metrics),
        required_primary_rows=42, fits=len(histories), frozen_files_verified=len(hashes),
        reused_files_verified=len(reused), previous_frozen_files_verified=len(previous_hashes),
        reused_final_model_files_verified=final_reference_files, test_origins=len(expected_ids),
        metrics_rebuilt=True, mean_rebuilt=True, stacking_rebuilt=True,
        final_lstm_predictions_rebuilt=True, development_gates_rebuilt=True,
        dates_and_truth_verified=True, ann_rf_identical_to_previous=True)
    return result, metrics, compared, delivery


def summary(compared):
    columns = ['RMSE', 'RMSE_previous', 'MAE', 'MAE_previous']
    frame = compared.groupby(['strategy', 'model'])[columns].mean().reset_index()
    for key in ['RMSE', 'MAE']:
        frame[key + '_change_pct'] = (frame[key] / frame[key + '_previous'] - 1) * 100
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='formal_20261111_lognormal')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ['.', '..']:
        parser.error('run-id必须为单目录名')
    run = ROOT/'output/acceptance3_runs'/args.run_id
    with threadpool_limits(limits=4):
        result, metrics, compared, delivery = verify(run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('相对上轮GPU：\n' + summary(compared).to_string(index=False))
    print('相对原正式交付：\n' + summary(delivery).to_string(index=False))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
        names = ['manifest.json', 'config.json', 'verification.json', 'lstm_grid_selection.json',
            'development_gates.json', 'development_quarter_metrics.csv', 'frozen_config.json',
            'meta_selection.json', 'model_hashes_before_test.json', 'reused_source_hashes.json',
            'metrics.csv', 'metrics_all_seeds.csv']
        for name in names:
            shutil.copy2(run/name, args.out/name)
        for name in ['cv_results', 'splits']:
            shutil.copytree(run/name, args.out/name)
        (args.out/'independent_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        compared.to_csv(args.out/'comparison_to_previous_by_horizon.csv', index=False, encoding='utf-8-sig')
        delivery.to_csv(args.out/'comparison_to_delivery_by_horizon.csv', index=False, encoding='utf-8-sig')
        summary(compared).to_csv(args.out/'summary_to_previous.csv', index=False, encoding='utf-8-sig')
        summary(delivery).to_csv(args.out/'summary_to_delivery.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
