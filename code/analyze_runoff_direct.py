"""核验后导出直接残差扩展，最终评价与原报告按日期/seed/h严格对齐。"""
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

from analyze_runoff_controlled import paired_interval, summary_metrics
from analyze_runoff_strategy import arrays, compare
from runoff_training.controlled_data import load_data, sha256
from runoff_training.controlled_train import write_json
from runoff_training.direct_train import split_for
from verify_runoff_direct import verify, ROOT
from verify_runoff_strategy import recompute

BASELINE = 'last_plain_log'
OLD_METRICS = ROOT/'验收/Acceptance 3/data/任务1至7+三seed完整指标与来源.csv'
OLD_PREDICTIONS = ROOT/'验收/Acceptance 3/data/任务1至7+三seed逐日预测与来源.csv'


def selected_rows(predictions, selections):
    result = []
    for item in selections:
        part = predictions[(predictions.year == item['year'])&(predictions.task_h == item['task_h'])&
                           (predictions.variant == item['selected_variant'])&(predictions.kind == 'searched')]
        if part.empty:
            raise ValueError('缺少预冻结所选策略')
        result.append(part)
    return pd.concat(result, ignore_index=True)


def original_comparison(selected):
    old = pd.read_csv(OLD_PREDICTIONS)
    old = old[(old.strategy == 'direct')&(old.model == 'LSTM')]
    current = selected.copy()
    keys = ['origin_date', 'target_date', 'h', 'seed']
    previous = old[keys+['truth', 'pred']].rename(columns={'truth':'truth_previous', 'pred':'pred_previous'})
    assert not current.duplicated(keys).any() and not previous.duplicated(keys).any()
    merged = current.merge(previous, on=keys, how='left', validate='one_to_one')
    if merged.pred_previous.isna().any() or len(merged) != len(current):
        raise ValueError('原正式预测缺少共同起报/目标日期')
    np.testing.assert_equal(merged.truth.to_numpy(), merged.truth_previous.to_numpy())
    metrics = pd.read_csv(OLD_METRICS)
    rows = []
    for (seed, h), part in merged.groupby(['seed', 'h']):
        before = recompute(part.truth.to_numpy(), part.pred_previous.to_numpy())
        after = recompute(part.truth.to_numpy(), part.pred.to_numpy())
        saved = metrics[(metrics.strategy == 'direct')&(metrics.model == 'LSTM')&
                        (metrics.seed == seed)&(metrics.h == h)]
        assert len(saved) == 1
        for key in ['RMSE', 'MAE', 'R2', 'n']:
            np.testing.assert_allclose(before[key], saved.iloc[0][key], rtol=1e-10, atol=1e-7)
        row = dict(seed=seed, h=h, variant=part.variant.iloc[0], n=len(part))
        for key in ['RMSE', 'MAE', 'R2']:
            row[key+'_previous'], row[key] = before[key], after[key]
        row['RMSE_change_pct'] = 100*(after['RMSE']/before['RMSE']-1)
        row['MAE_change_pct'] = 100*(after['MAE']/before['MAE']-1)
        rows.append(row)
    return pd.DataFrame(rows), merged


def analyze(run, out):
    verification, predictions = verify(run)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    spec = manifest['protocol']
    selection = json.loads((run/'selection.json').read_text(encoding='utf-8'))
    data = load_data(run/('evaluation.npz' if spec['mode'] == 'final' else 'historical.npz'))
    searched = predictions[predictions.kind == 'searched'].copy()
    selected = selected_rows(predictions, selection['selections'])
    annual, pairs, blocks, persistence, ablations, quarters = [], [], [], [], [], []
    for (year, h), part in searched.groupby(['year', 'h']):
        _, valid = split_for(data, spec['mode'], year)
        truth, plain, ids = arrays(part, BASELINE)
        np.testing.assert_equal(ids, valid)
        q = data['discharge'][data['positions'][ids]][:, None]
        repeated = np.repeat(q[None], len(spec['seeds']), axis=0)
        persistence.append(dict(year=year, h=h, **recompute(truth[:, 0], q[:, 0])))
        chosen = next(s['selected_variant'] for s in selection['selections']
                      if s['year'] == year and s['task_h'] == h)
        for variant in [v['name'] for v in spec['variants']]:
            own_truth, own, _ = arrays(part, variant)
            np.testing.assert_equal(truth, own_truth)
            identity = dict(year=year, h=h, variant=variant, selected=variant == chosen)
            annual.append(dict(**identity, **summary_metrics(truth, own)))
            for label, reference in [('plain', plain), ('persistence', repeated)]:
                pairs.append(dict(**identity, reference=label, **compare(truth, reference, own)))
                if variant != BASELINE or label == 'persistence':
                    for block in [7, 14, 30]:
                        low, high = paired_interval(truth, reference, own, block)
                        blocks.append(dict(**identity, reference=label, block_days=block,
                                           change_pct_low=low, change_pct_high=high))
    fixed = predictions[predictions.kind == 'fixed']
    for (year, epoch), part in fixed.groupby(['year', 'epochs']):
        truth, plain, _ = arrays(part, BASELINE)
        _, residual, _ = arrays(part, 'last_residual_log')
        ablations.append(dict(year=year, h=1, epochs=epoch,
            plain_RMSE=summary_metrics(truth, plain)['RMSE'],
            residual_RMSE=summary_metrics(truth, residual)['RMSE'], **compare(truth, plain, residual)))
    predictions['quarter'] = pd.to_datetime(predictions.origin_date).dt.quarter
    for keys, part in predictions.groupby(['year', 'h', 'seed', 'variant', 'kind', 'epochs', 'quarter']):
        identity = dict(zip(['year', 'h', 'seed', 'variant', 'kind', 'epochs', 'quarter'], keys))
        quarters.append(dict(**identity, **recompute(part.truth.to_numpy(), part.pred.to_numpy())))
    tables = dict(annual_summary=annual, paired_changes=pairs, paired_block_sensitivity=blocks,
                  persistence=persistence, same_epoch_ablation=ablations, quarter_metrics=quarters)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out/f'{name}.csv', index=False, encoding='utf-8-sig')
    summary = []
    for year, part in selected.groupby('year'):
        # variant标签仅为排序工具，保留实际逐h选择另表，不重新按评价分数选。
        own = part.assign(variant='selected')
        truth, pred, _ = arrays(own, 'selected')
        reference = searched[searched.year == year].assign(variant=lambda x:x.variant)
        _, plain, _ = arrays(reference, BASELINE)
        summary.append(dict(year=year, **summary_metrics(truth, pred), **compare(truth, plain, pred)))
    pd.DataFrame(summary).to_csv(out/'selected_summary.csv', index=False, encoding='utf-8-sig')
    selected.to_csv(out/'selected_predictions.csv', index=False, encoding='utf-8-sig')
    old_hashes = None
    comparison_summary = None
    if spec['mode'] == 'final':
        comparison, merged = original_comparison(selected)
        comparison.to_csv(out/'comparison_by_horizon.csv', index=False, encoding='utf-8-sig')
        means = comparison.groupby('seed')[['RMSE_previous', 'RMSE', 'MAE_previous', 'MAE', 'R2_previous', 'R2']].mean()
        means['RMSE_change_pct'] = 100*(means.RMSE/means.RMSE_previous-1)
        means['MAE_change_pct'] = 100*(means.MAE/means.MAE_previous-1)
        means.to_csv(out/'comparison_by_seed.csv', encoding='utf-8-sig')
        before = comparison[['RMSE_previous', 'MAE_previous', 'R2_previous']].mean()
        after = comparison[['RMSE', 'MAE', 'R2']].mean()
        comparison_summary = dict(RMSE_previous=float(before.RMSE_previous), RMSE=float(after.RMSE),
            MAE_previous=float(before.MAE_previous), MAE=float(after.MAE),
            R2_previous=float(before.R2_previous), R2=float(after.R2),
            RMSE_change_pct=float(100*(after.RMSE/before.RMSE_previous-1)),
            MAE_change_pct=float(100*(after.MAE/before.MAE_previous-1)))
        write_json(out/'comparison_summary.json', comparison_summary)
        old_hashes = {str(p.relative_to(ROOT)):sha256(p) for p in [OLD_METRICS, OLD_PREDICTIONS]}
        interval_rows = []
        for horizon in [0, *range(1, 8)]:
            piece = merged if horizon == 0 else merged[merged.h == horizon]
            piece = piece.assign(variant='selected')
            truth, pred, _ = arrays(piece, 'selected')
            _, previous, _ = arrays(piece.assign(pred=piece.pred_previous), 'selected')
            for block in [7, 14, 30]:
                low, high = paired_interval(truth, previous, pred, block)
                interval_rows.append(dict(h=horizon, block_days=block, change_pct_low=low, change_pct_high=high))
        pd.DataFrame(interval_rows).to_csv(out/'original_block_sensitivity.csv', index=False, encoding='utf-8-sig')
    for name in ['metrics.csv', 'selection.json', 'inner_splits.csv', 'manifest.json']:
        shutil.copy2(run/name, out/name)
    write_json(out/'verification.json', verification)
    lines = ['# 直接残差扩展与最终对照', '', '日期：2026-11-11。内部实验，不替换正式Word。', '',
        f'模式：{spec["mode"]}；{verification["fits"]}fit、{verification["models"]}模型全部恢复核验通过。',
        '普通/残差候选及历史选择规则在评价前冻结，不按2004逐h挑选。', '',
        '## 历史内层所选策略', '', '| 年份 | h | 所选策略 | RMSE | MAE |',
        '| --- | --- | --- | ---: | ---: |']
    for row in pd.DataFrame(annual).query('selected').itertuples():
        lines.append(f'| {row.year} | {row.h} | {row.variant} | {row.RMSE:.2f} | {row.MAE:.2f} |')
    lines += ['', '## 七步平均汇总', '']
    for row in summary:
        lines.append(f'- {row["year"]}：所选RMSE {row["RMSE"]:.4f}、MAE {row["MAE"]:.4f}；'
                     f'相对本轮普通LSTM RMSE变化{row["RMSE_change_pct"]:+.2f}%。')
    if comparison_summary:
        c = comparison_summary
        lines += ['', '## 与原正式报告同一2004样本比较', '',
            f'三seed七步平均RMSE：{c["RMSE_previous"]:.4f} → {c["RMSE"]:.4f}（{c["RMSE_change_pct"]:+.2f}%）。',
            f'MAE：{c["MAE_previous"]:.4f} → {c["MAE"]:.4f}（{c["MAE_change_pct"]:+.2f}%）。',
            '原正式预测按起报日/目标日/h/seed合并，真值一致；原指标另由原预测重算核对。',
            '这是已经查看过的2004回溯评价，不是盲测；不同训练起始日/容量等限制纯结构归因。',
            '[逐h/seed对比](comparison_by_horizon.csv) · [逐seed七步均值](comparison_by_seed.csv) · '
            '[对原报告的时间块敏感性](original_block_sensitivity.csv)']
    lines += ['', '## 证据与边界', '',
        '- [逐h/seed指标](metrics.csv)', '- [全部固定候选](annual_summary.csv)',
        '- [配对变化](paired_changes.csv)', '- [时间块敏感性](paired_block_sensitivity.csv)',
        '- [严格持久性](persistence.csv)', '- [同epoch消融](same_epoch_ablation.csv)',
        '- [季度](quarter_metrics.csv)', '- [内层冻结选择](selection.json)',
        '- [独立核验](verification.json)',
        '- [训练前固定方案](../../任务1至7+直接残差扩展与最终对照方案_2026-11-11.md)', '',
        'RMSE先对每个h计算，再算术平均h及seed，不是扁平RMSE。',
        '7/14/30日连续块同步抽样seed/h、600次；区间只是已观测历史敏感性，不保证未来。',
        '本轮未新训ANN/RF/多输出/集成，不按外层拼最优，不保证0.8。原CSV/Word/旧run未覆盖，未commit/push。', '']
    (out/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    (out/'analysis_source').mkdir()
    for source in [Path(__file__), ROOT/'code/verify_runoff_direct.py', ROOT/'code/analyze_runoff_strategy.py',
                   ROOT/'code/analyze_runoff_controlled.py']:
        shutil.copy2(source, out/'analysis_source'/source.name)
    write_json(out/'analysis_manifest.json', dict(status='completed', run_id=run.name,
        run_manifest_sha256=sha256(run/'manifest.json'), original_sources=old_hashes,
        bootstrap_seed=20261111, block_days=[7, 14, 30], repeats=600,
        files={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(pd.DataFrame(annual).to_string(index=False), flush=True)
    print('所选七步汇总', summary, flush=True)
    if comparison_summary:
        print('与原正式报告', comparison_summary, flush=True)
    print('导出完成', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with threadpool_limits(limits=4):
        analyze(ROOT/'output/acceptance3_runs'/args.run_id, args.out)


if __name__ == '__main__':
    main()
