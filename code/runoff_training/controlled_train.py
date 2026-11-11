"""受控窗口/状态与目标损失实验：历史内层选epoch，年度外层仅评价。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from runoff_training.controlled_data import (BASE, STATE, PROFILES, build_data, load_data,
                                             outer_split, inner_splits, sha256)
from runoff_training.controlled_model import ControlledLSTM, physical_score, restore

ROOT = Path(__file__).resolve().parents[2]
TASKS = [('direct', 1), ('direct', 7), ('multi', 7)]
TOLERANCE = .01


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def environment(device):
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        pandas=pd.__version__, sklearn=sklearn.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None)


def candidate_record(run, tag):
    checkpoint = run / 'checkpoints' / f'{tag}.json'
    if not checkpoint.exists():
        if ((run / 'models' / tag).exists() or (run / 'cv' / f'{tag}.csv').exists()
                or (run / 'predictions' / f'{tag}.csv').exists()
                or any((run / 'history').glob(f'{tag}_*.json'))):
            raise ValueError(f'候选{tag}存在未提交工件；保留痕迹，请新建run，禁止重复fit')
        return None
    record = json.loads(checkpoint.read_text(encoding='utf-8'))
    required = [f'models/{tag}/weights.pt', f'models/{tag}/scalers.npz',
                f'models/{tag}/inference.json', f'cv/{tag}.csv', f'predictions/{tag}.csv']
    artifacts = record.get('artifacts', {})
    if not set(required).issubset(artifacts):
        raise ValueError(f'候选{tag}缺少工件hash，不支持旧协议resume')
    for rel, expected in artifacts.items():
        if not (run / rel).is_file() or sha256(run / rel) != expected:
            raise ValueError(f'候选工件缺失或改变：{rel}')
    return record


def choose_epoch(scores, epochs):
    """预注册的1%近似平手：选择更少epoch，不按外层成绩调整。"""
    scores = np.asarray(scores)
    eligible = np.flatnonzero(scores <= scores.min() * (1 + TOLERANCE))
    return int(min(eligible, key=lambda i: epochs[i]))


def choose_profile(scores):
    best = min(scores.values())
    return next(profile for profile in PROFILES if scores[profile] <= best * (1 + TOLERANCE))


def metrics(truth, pred):
    rows = []
    for j in range(truth.shape[1]):
        y, p = truth[:, j], pred[:, j]
        error = p - y
        sst = float(np.square(y - y.mean()).sum())
        rows.append(dict(h=j + 1, n=len(y), RMSE=float(np.sqrt(np.square(error).mean())),
            MAE=float(np.abs(error).mean()), R2=float(1 - np.square(error).sum() / sst),
            bias=float(error.mean()), negative_predictions=int((p < 0).sum())))
    return rows


def select_and_evaluate(run, data, train, valid, year, strategy, h, profile,
                        transform, seed, args):
    tag = f'y{year}_{strategy}_h{h}_{profile}_{transform}_s{seed}'
    checkpoint = run / 'checkpoints' / f'{tag}.json'
    saved = candidate_record(run, tag)
    if saved is not None:
        return saved
    target = data['y'][:, h - 1:h] if strategy == 'direct' else data['y']
    folds = inner_splits(data, train)
    epochs = [1] if args.smoke else [10, 30]
    estimator = ControlledLSTM(str(run / 'dataset.npz'), profile=profile, seed=seed,
        device=args.device, target_transform=transform, history_dir=str(run / 'history'), phase=tag)
    search = GridSearchCV(estimator, dict(fixed_epochs=epochs), cv=folds, scoring=physical_score,
                          n_jobs=1, refit=False, error_score='raise')
    search.fit(train[:, None], target[train])
    scores = -np.asarray(search.cv_results_['mean_test_score'])
    index = choose_epoch(scores, epochs)
    chosen = epochs[index]
    table = pd.DataFrame(search.cv_results_)
    table['params'] = table.params.map(json.dumps)
    table.to_csv(run / 'cv' / f'{tag}.csv', index=False, encoding='utf-8-sig')
    selected = dict(year=year, strategy=strategy, task_h=h, profile=profile, transform=transform,
        seed=seed, epochs=chosen, inner_RMSE=float(scores[index]), inner_best_RMSE=float(scores.min()),
        inner_fold_RMSE=[-float(search.cv_results_[f'split{k}_test_score'][index]) for k in range(len(folds))])
    model = ControlledLSTM(str(run / 'dataset.npz'), profile=profile, seed=seed, device=args.device,
        target_transform=transform, fixed_epochs=chosen, history_dir=str(run / 'history'), phase=tag + '_outer')
    model.fit(train[:, None], target[train])
    prediction = model.predict(valid[:, None])
    model.save(run / 'models' / tag)
    rebuilt = restore(run / 'models' / tag, run / 'dataset.npz', args.device).predict(valid[:, None])
    np.testing.assert_allclose(rebuilt, prediction, rtol=1e-7, atol=1e-5)
    selected['metrics'] = metrics(target[valid], prediction)
    if strategy == 'direct':
        selected['metrics'][0]['h'] = h
    selected['audit'] = model.audit_
    rows = []
    days = data['origin_days']
    for j in range(prediction.shape[1]):
        horizon = h if strategy == 'direct' else j + 1
        for k, sample_id in enumerate(valid):
            rows.append(dict(year=year, strategy=strategy, task_h=h, profile=profile,
                transform=transform, seed=seed, sample_id=int(sample_id), h=horizon,
                origin_date=str(np.datetime64(int(days[sample_id]), 'D')),
                target_date=str(np.datetime64(int(days[sample_id] + horizon), 'D')),
                truth=float(target[sample_id, j]), pred=float(prediction[k, j])))
    pd.DataFrame(rows).to_csv(run / 'predictions' / f'{tag}.csv', index=False, encoding='utf-8-sig')
    artifacts = [run / 'cv' / f'{tag}.csv', run / 'predictions' / f'{tag}.csv']
    artifacts += list((run / 'models' / tag).glob('*'))
    artifacts += list((run / 'history').glob(f'{tag}_*.json'))
    selected['artifacts'] = {p.relative_to(run).as_posix(): sha256(p) for p in artifacts}
    write_json(checkpoint, selected)
    outer_score = np.mean([row['RMSE'] for row in selected['metrics']])
    print('[受控完成]', tag, 'epoch', chosen, 'inner', round(scores[index], 3),
          'outer', round(outer_score, 3), flush=True)
    return selected


def protocol(args):
    return dict(version=2, kind='历史开发受控实验，非验收替换，不访问2004预测',
        years=[2002, 2003], tasks=TASKS, seeds=[42] if args.smoke else [42, 43, 44],
        profiles=PROFILES, base_features=BASE, state_features=STATE,
        fixed=dict(units=32, dropout=.1, head=32, batch_size=64, lr=.001, optimizer='Adam',
                   shuffle=False, residual=False, loss='standardized_target_MSE'),
        epochs=[1] if args.smoke else [10, 30], inner_folds=3, inner_validation_days=90,
        purge_days=7, tie_tolerance=TOLERANCE, priority=list(PROFILES),
        scaler_rule='相同训练id覆盖的90日历史并集，无未来行；基准状态通道固定置零',
        profile_selection='各外层仅按三个seed的内层分数平均选择；1%内按priority，不使用外层分数',
        loss_phase='第一阶段选定输入上比较log/sqrt/raw；log复用，其他设置不变',
        max_data_day='2003-12-31', smoke=args.smoke, device=args.device,
        fit_budget=144 if args.smoke else 756)


def source_files():
    return [Path(__file__), Path(__file__).with_name('controlled_data.py'),
            Path(__file__).with_name('controlled_model.py'), ROOT / 'code/test_runoff_controlled.py']


def run_experiment(args):
    run = ROOT / 'output/acceptance3_runs' / args.run_id
    raw = ROOT / 'Teacher Resources & Requirements/01047000.csv'
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_files()}
    spec = protocol(args)
    if args.resume:
        old = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
        if old['status'] == 'completed':
            raise ValueError('已完成run不能继续覆盖')
        if old['protocol'] != json.loads(json.dumps(spec)) or old['source_hashes'] != hashes or old['data']['raw_sha256'] != sha256(raw):
            raise ValueError('恢复要求源码、数据、设备和协议全部一致')
        if sha256(run / 'dataset.npz') != old['data']['dataset_sha256']:
            raise ValueError('数据集hash改变')
        if old.get('environment') != environment(args.device):
            raise ValueError('依赖版本或设备环境改变，不允许混合环境恢复')
        # 恢复前全局预检，不到候选执行时才发现晚期残留或丢失权重。
        committed = set()
        for folder in (run / 'models').iterdir():
            if folder.is_dir():
                candidate_record(run, folder.name)
        for file in (run / 'checkpoints').glob('*.json'):
            record = candidate_record(run, file.stem)
            committed.update(record['artifacts'])
        for name in ['history', 'models', 'cv', 'predictions']:
            for file in (run / name).rglob('*'):
                if file.is_file() and file.relative_to(run).as_posix() not in committed:
                    raise ValueError(f'发现未提交半候选工件{file}，请新建run')
        manifest = old
    else:
        run.mkdir(parents=True, exist_ok=False)
        for name in ['source', 'cv', 'history', 'models', 'predictions', 'checkpoints']:
            (run / name).mkdir()
        for path in source_files():
            shutil.copy2(path, run / 'source' / path.name)
        info = build_data(raw, run / 'dataset.npz')
        manifest = dict(status='running', started=time.strftime('%Y-%m-%d %H:%M:%S'),
            protocol=spec, source_hashes=hashes, data=info, raw_2004_rows_used_in_arrays=False,
            environment=environment(args.device))
    manifest['status'] = 'running'
    write_json(run / 'manifest.json', manifest)
    start = time.perf_counter()
    try:
        data = load_data(run / 'dataset.npz')
        results, selections, splits = [], [], []
        for year in spec['years']:
            train, valid = outer_split(data, year)
            for k, (ti, vi) in enumerate(inner_splits(data, train)):
                days = data['origin_days']
                splits.append(dict(year=year, fold=k, n_train=len(ti), n_validation=len(vi),
                    train_last_target=str(np.datetime64(int(days[train[ti[-1]]] + 7), 'D')),
                    validation_first_origin=str(np.datetime64(int(days[train[vi[0]]]), 'D')),
                    validation_last_target=str(np.datetime64(int(days[train[vi[-1]]] + 7), 'D'))))
            for strategy, h in TASKS:
                initial = []
                for profile in PROFILES:
                    for seed in spec['seeds']:
                        value = select_and_evaluate(run, data, train, valid, year, strategy, h,
                                                    profile, 'log', seed, args)
                        initial.append(value)
                        results.append(value)
                scores = {p: float(np.mean([r['inner_RMSE'] for r in initial if r['profile'] == p])) for p in PROFILES}
                chosen = choose_profile(scores)
                selections.append(dict(year=year, strategy=strategy, task_h=h,
                    selected_profile=chosen, inner_scores=scores, selection_uses_outer=False))
                write_json(run / 'input_selection.json', selections)
                # 损失对照的输入仅由内层确定，不根据刚输出的外层成绩改变实验。
                for transform in ['sqrt', 'raw']:
                    for seed in spec['seeds']:
                        results.append(select_and_evaluate(run, data, train, valid, year, strategy, h,
                            chosen, transform, seed, args))
        metric_rows = []
        for value in results:
            identity = {k: value[k] for k in ['year', 'strategy', 'task_h', 'profile', 'transform', 'seed', 'epochs', 'inner_RMSE']}
            metric_rows.extend([dict(**identity, **row) for row in value['metrics']])
        pd.DataFrame(metric_rows).to_csv(run / 'metrics.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(splits).to_csv(run / 'inner_splits.csv', index=False, encoding='utf-8-sig')
        histories = list((run / 'history').glob('*.json'))
        if len(histories) != spec['fit_budget']:
            raise ValueError(f'实际fit {len(histories)}不等于预算{spec["fit_budget"]}；不要复用有半完成候选的run')
        manifest.update(status='completed', finished=time.strftime('%Y-%m-%d %H:%M:%S'),
            elapsed_this_invocation_seconds=time.perf_counter() - start, fits=len(histories),
            outer_models=len(results), prediction_files=len(list((run / 'predictions').glob('*.csv'))))
        manifest['frozen_files'] = {str(p.relative_to(run)): sha256(p) for name in
            ['source', 'cv', 'history', 'models', 'predictions', 'checkpoints'] for p in (run / name).rglob('*') if p.is_file()}
        for name in ['dataset.npz', 'input_selection.json', 'inner_splits.csv', 'metrics.csv']:
            manifest['frozen_files'][name] = sha256(run / name)
        write_json(run / 'manifest.json', manifest)
        print('[完成]', run, 'fits', len(histories), 'seconds', round(time.perf_counter() - start, 1), flush=True)
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc())
        write_json(run / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or '/' in args.run_id or '\\' in args.run_id or args.run_id in {'.', '..'}:
        raise ValueError('run-id只能为单个目录名')
    args.device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA不可用')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with threadpool_limits(limits=4):
        run_experiment(args)


if __name__ == '__main__':
    main()
