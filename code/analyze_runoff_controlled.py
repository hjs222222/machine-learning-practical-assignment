"""核验后导出受控实验轻量指标及配对时间块敏感性，不按外层成绩选模型。"""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from verify_runoff_controlled import verify, ROOT, metric_rows
from runoff_training.controlled_data import PROFILES, load_data, outer_split, sha256


BOOTSTRAP_REPEATS = 600
BOOTSTRAP_SEED = 20261111


def paired_interval(truth, baseline, candidate, block, repeats=BOOTSTRAP_REPEATS):
    """所有seed同步重采样连续起报块，保留跨seed/跨h误差依赖；非独立样本置信保证。"""
    n = len(truth)
    rng = np.random.default_rng(BOOTSTRAP_SEED + block)
    deltas = []
    for _ in range(repeats):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        ids = (starts[:, None] + np.arange(block)).ravel()[:n]
        base = np.sqrt(np.square(baseline[:, ids] - truth[None, ids]).mean(axis=1)).mean()
        own = np.sqrt(np.square(candidate[:, ids] - truth[None, ids]).mean(axis=1)).mean()
        deltas.append(100 * (own / base - 1))
    low, high = np.quantile(deltas, [.025, .975])
    return float(low), float(high)


def arrays(frame, profile, transform):
    own = frame[(frame.profile == profile) & (frame['transform'] == transform)]
    seeds = sorted(own.seed.unique())
    truth, predictions = None, []
    ids = None
    for seed in seeds:
        piece = own[own.seed == seed]
        p = piece.pivot(index='sample_id', columns='h', values='pred').sort_index()
        t = piece.pivot(index='sample_id', columns='h', values='truth').sort_index()
        if truth is not None:
            np.testing.assert_equal(t.to_numpy(), truth)
            np.testing.assert_equal(p.index.to_numpy(), ids)
        truth, ids = t.to_numpy(), p.index.to_numpy()
        predictions.append(p.to_numpy())
    return truth, np.stack(predictions), ids


def summary_metrics(truth, pred):
    squared = np.square(pred - truth[None])
    return dict(RMSE=float(np.sqrt(squared.mean(axis=1)).mean()),
                MAE=float(np.abs(pred - truth[None]).mean()))


def write_markdown(out, annual, paired, blocks, selections, verification):
    lines = ['# 任务1至7：受控窗口、状态与损失实验结果', '',
        '日期：2026-11-11。供项目维护者决定后续训练；不是新的正式验收报告。', '',
        '本轮只包含direct h1、direct h7、原生multi三个哨兵任务。全部评价止于2003年；没有2004新成绩。', '',
        '## 1. 执行与独立核验', '',
        f'- 正式完成{verification["fits"]}次fit、{verification["models"]}个外层模型。',
        f'- 已恢复全部模型，核对{verification["prediction_rows"]}行预测和{verification["metrics"]}项逐h指标；核验状态：{verification["status"]}。',
        '- 共同起报样本、固定17通道和32单层LSTM、三个seed；窗口/状态比较统一log损失。',
        '- 为保证90日状态和90日窗口完整预热，起报从2000-06-27开始，比旧30日工程序列更晚；不能把本表直接与旧run数字当配对提升比较。',
        '- 训练预算仅由各外层历史内层的3块90日GridSearchCV选择10/30轮，1%近似平手选少轮数。',
        '- 输入按三个seed内层均值选择；不同年度可选不同输入。外层只评价，不用于换输入。', '',
        '## 2. 同协议窗口/状态结果', '',
        'RMSE为各seed/逐h RMSE算术平均；不是把所有列扁平后计算。年度模型固定在年初训练，未季度更新。', '',
        '| 年度 | 任务 | 30日基础 | 90日基础 | 30日状态 | 90日状态 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for (year, strategy, h), part in annual[annual['transform'] == 'log'].groupby(['year', 'strategy', 'task_h']):
        lookup = part.set_index('profile').RMSE
        label = f'h{h}' if strategy == 'direct' else 'multi'
        lines.append(f'| {year} | {label} | ' + ' | '.join(f'{lookup[p]:.2f}' for p in PROFILES) + ' |')
    lines += ['', '## 3. 内层所选输入与损失对照', '',
        '| 年度 | 任务 | 内层所选输入 | log | sqrt | raw |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for selection in selections:
        year, strategy, h = selection['year'], selection['strategy'], selection['task_h']
        part = annual[(annual.year == year) & (annual.strategy == strategy) & (annual.task_h == h)
                      & (annual.profile == selection['selected_profile'])].set_index('transform')
        label = f'h{h}' if strategy == 'direct' else 'multi'
        lines.append(f'| {year} | {label} | {selection["selected_profile"]} | ' +
                     ' | '.join(f'{part.loc[t, "RMSE"]:.2f}' for t in ['log', 'sqrt', 'raw']) + ' |')
    lines += ['', '输入内层选择细节见[input_selection.json](input_selection.json)。本轮不根据此处外层最小值宣布下一轮已冻结配置。'
        '各方案可在相同10/30轮预算中选择不同epoch，因此是同搜索预算的策略比较，不是固定训练轮数的纯损失/窗口梯度消融。'
        '状态方案也可能通过累计统计访问早于窗口首日的信息；窗口长度与状态开关分别按2×2对照解释，不将90日窗口等同于唯一历史范围。', '',
        '## 4. 逐seed、季度与不确定性', '',
        '下表每项为年度配对RMSE变化百分比，负数表示改善；seed范围是三个配对变化的最小/最大值。', '',
        '| 年度 | 任务 | 对照 | 平均变化 | seed范围 | 改善seed数 |',
        '| --- | --- | --- | ---: | --- | ---: |']
    for row in paired.itertuples():
        label = f'h{row.task_h}' if row.strategy == 'direct' else 'multi'
        lines.append(f'| {row.year} | {label} | {row.comparison} | {row.RMSE_change_pct:+.2f}% | '
                     f'{row.seed_min_change_pct:+.2f}%～{row.seed_max_change_pct:+.2f}% | {row.improved_seeds}/3 |')
    lines += ['', '7/14/30日移动块同步重采样所有seed，600次，报告2.5%～97.5%经验分位区间。'
              '这保留块内相关，但只是这两年历史误差的敏感性分析，不是未来泛化保证；不能将重叠h当独立样本。', '',
        '- [年度逐h/逐seed指标](metrics.csv)',
        '- [年度三seed汇总](annual_summary.csv)',
        '- [季度逐h/逐seed指标](quarter_metrics.csv)',
        '- [严格持久性基线](persistence.csv)',
        '- [训练P90高流量/普通期分段](error_segments.csv)',
        '- [年度配对变化](paired_changes.csv)',
        '- [时间块敏感性区间](paired_block_sensitivity.csv)',
        '- [完整独立核验结果](verification.json)',
        '- [冻结运行记录](manifest.json)',
        '- [运行前方案](../../任务1至7+受控窗口状态与损失实验方案_2026-11-11.md)', '',
        '## 5. 解读边界', '',
        '这里不同输入、损失和年度的平均数属于开发对照，不是额外独立测试。'
        '不将某个最佳seed、最佳h或最佳季度拼成新模型；不把本轮结果归因于单一水文机理。', '',
        '状态是固定API/累计雨量/流量统计，未重建真实积雪或土壤水分；90日也不等于覆盖所有长期状态。'
        '若提升不稳定，应保留失败证据，再设计新的小对照，而不是用2004继续筛选。', '',
        '本轮未改变ANN/RF、未重新训练Mean/Stacking、未完成direct h2～h6，不能据此宣布全部七项任务已提升。'
        '原CSV、历史run、正式Word保持不变，未提交或推送。', '']
    (out / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run = ROOT / 'output/acceptance3_runs' / args.run_id
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with threadpool_limits(limits=4):
        verification, predictions = verify(run)
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ['metrics.csv', 'input_selection.json', 'inner_splits.csv', 'manifest.json']:
        shutil.copy2(run / name, args.out / name)
    (args.out / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    selections = json.loads((run / 'input_selection.json').read_text(encoding='utf-8'))
    annual, paired, intervals = [], [], []
    for (year, strategy, h), part in predictions.groupby(['year', 'strategy', 'task_h']):
        for (profile, transform), own in part.groupby(['profile', 'transform']):
            truth, pred, _ = arrays(part, profile, transform)
            annual.append(dict(year=year, strategy=strategy, task_h=h, profile=profile,
                transform=transform, **summary_metrics(truth, pred)))
        selected = next(s['selected_profile'] for s in selections if
                        (s['year'], s['strategy'], s['task_h']) == (year, strategy, h))
        pairs = [(p, 'log', 'w30_base', 'log', f'{p}/w30_base') for p in list(PROFILES)[1:]]
        pairs += [('w90_state', 'log', 'w30_state', 'log', 'w90_state/w30_state'),
                  ('w90_state', 'log', 'w90_base', 'log', 'w90_state/w90_base')]
        pairs += [(selected, t, selected, 'log', f'{t}/log@{selected}') for t in ['sqrt', 'raw']]
        for profile, transform, ref_p, ref_t, label in pairs:
            truth, candidate, _ = arrays(part, profile, transform)
            _, baseline, _ = arrays(part, ref_p, ref_t)
            base_score, own_score = summary_metrics(truth, baseline), summary_metrics(truth, candidate)
            own_seeds = np.sqrt(np.square(candidate - truth[None]).mean(axis=1)).mean(axis=1)
            base_seeds = np.sqrt(np.square(baseline - truth[None]).mean(axis=1)).mean(axis=1)
            changes = 100 * (own_seeds / base_seeds - 1)
            identity = dict(year=year, strategy=strategy, task_h=h, comparison=label)
            paired.append(dict(**identity, RMSE_change_pct=100 * (own_score['RMSE'] / base_score['RMSE'] - 1),
                MAE_change_pct=100 * (own_score['MAE'] / base_score['MAE'] - 1),
                seed_min_change_pct=float(changes.min()), seed_max_change_pct=float(changes.max()),
                improved_seeds=int((changes < 0).sum())))
            for block in [7, 14, 30]:
                low, high = paired_interval(truth, baseline, candidate, block)
                intervals.append(dict(**identity, block_days=block, repeats=BOOTSTRAP_REPEATS,
                                      change_pct_low=low, change_pct_high=high))
    quarter_rows = []
    dated = pd.to_datetime(predictions.origin_date)
    predictions['quarter'] = dated.dt.quarter
    for keys, own in predictions.groupby(['year', 'strategy', 'task_h', 'profile', 'transform', 'seed', 'quarter', 'h']):
        identity = dict(zip(['year', 'strategy', 'task_h', 'profile', 'transform', 'seed', 'quarter', 'h'], keys))
        quarter_rows.append(dict(**identity, **metric_rows(own.truth.to_numpy(), own.pred.to_numpy())))
    # 严格持久性Q_hat(t+h)=Q(t)，与仅Q(t)拟合回归区分；另按训练P90诊断高流量。
    data = load_data(run / 'dataset.npz')
    persistence_rows, segment_rows = [], []
    for (year, strategy, h), part in predictions.groupby(['year', 'strategy', 'task_h']):
        train, valid = outer_split(data, year)
        horizons = [h] if strategy == 'direct' else list(range(1, 8))
        q = data['discharge'][data['positions'][valid]]
        for horizon in horizons:
            persistence_rows.append(dict(year=year, strategy=strategy, task_h=h, h=horizon,
                **metric_rows(data['y'][valid, horizon - 1], q)))
        for (profile, transform, seed, horizon), own in part.groupby(['profile', 'transform', 'seed', 'h']):
            threshold = float(np.quantile(data['y'][train, horizon - 1], .9))
            own = own.sort_values('sample_id')
            high = own.truth.to_numpy() >= threshold
            total_sse = float(np.square(own.pred.to_numpy() - own.truth.to_numpy()).sum())
            for label, mask in [('high', high), ('other', ~high)]:
                y, p = own.truth.to_numpy()[mask], own.pred.to_numpy()[mask]
                segment_rows.append(dict(year=year, strategy=strategy, task_h=h, profile=profile,
                    transform=transform, seed=seed, h=horizon, segment=label, threshold=threshold,
                    n=int(mask.sum()), RMSE=float(np.sqrt(np.square(p-y).mean())),
                    MAE=float(np.abs(p-y).mean()), bias=float((p-y).mean()),
                    sse_share=float(np.square(p-y).sum()/total_sse)))
    pd.DataFrame(persistence_rows).to_csv(args.out / 'persistence.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(segment_rows).to_csv(args.out / 'error_segments.csv', index=False, encoding='utf-8-sig')
    annual, paired, intervals = pd.DataFrame(annual), pd.DataFrame(paired), pd.DataFrame(intervals)
    annual.to_csv(args.out / 'annual_summary.csv', index=False, encoding='utf-8-sig')
    paired.to_csv(args.out / 'paired_changes.csv', index=False, encoding='utf-8-sig')
    intervals.to_csv(args.out / 'paired_block_sensitivity.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(quarter_rows).to_csv(args.out / 'quarter_metrics.csv', index=False, encoding='utf-8-sig')
    write_markdown(args.out, annual, paired, intervals, selections, verification)
    source = args.out / 'analysis_source'
    source.mkdir()
    for path in [Path(__file__), ROOT / 'code/verify_runoff_controlled.py', ROOT / 'code/test_runoff_controlled_analysis.py']:
        shutil.copy2(path, source / path.name)
    output_manifest = dict(status='completed', run_id=args.run_id, bootstrap_repeats=BOOTSTRAP_REPEATS,
        bootstrap_seed=BOOTSTRAP_SEED, block_days=[7, 14, 30],
        run_manifest_sha256=sha256(run / 'manifest.json'), verification=verification,
        files={str(p.relative_to(args.out)): sha256(p) for p in args.out.rglob('*') if p.is_file()})
    (args.out / 'analysis_manifest.json').write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(annual.to_string(index=False))
    print(paired.to_string(index=False))
    print('export', args.out, flush=True)


if __name__ == '__main__':
    main()
