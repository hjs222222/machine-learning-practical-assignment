"""独立核验后生成继续训练的轻量结果记录，不覆盖已有目录或正式报告。"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset
from acceptance3_train import ROOT
from runoff_training.export import markdown_table
from runoff_training.train import year_folds, get_target, score
from verify_runoff_continued import verify, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='formal_20261111_lognormal')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ['.', '..']:
        parser.error('run-id必须为单目录名')
    run = ROOT/'output/acceptance3_runs'/args.run_id
    with threadpool_limits(limits=4):
        verified, metrics, previous, delivery = verify(run)
    if args.out.exists() and not args.out.is_dir():
        raise ValueError('导出目标必须为新目录')
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ['manifest.json', 'config.json', 'verification.json', 'lstm_grid_selection.json',
                 'development_gates.json', 'development_quarter_metrics.csv', 'frozen_config.json',
                 'meta_selection.json', 'model_hashes_before_test.json', 'reused_source_hashes.json',
                 'metrics.csv', 'metrics_all_seeds.csv']:
        shutil.copy2(run/name, args.out/name)
    for folder in ['cv_results', 'splits']:
        shutil.copytree(run/folder, args.out/folder)
    (args.out/'independent_verification.json').write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, frame in [('comparison_to_previous_by_horizon', previous), ('comparison_to_delivery_by_horizon', delivery),
                        ('summary_to_previous', summary(previous)), ('summary_to_delivery', summary(delivery))]:
        frame.to_csv(args.out/(name + '.csv'), index=False, encoding='utf-8-sig')
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    selections = json.loads((run/'lstm_grid_selection.json').read_text(encoding='utf-8'))
    gates = json.loads((run/'development_gates.json').read_text(encoding='utf-8'))
    parameter_rows = pd.DataFrame([dict(strategy=r['strategy'], h=r['h'],
        objective=r['params']['objective'], units=str(r['params']['units']),
        epoch=r['params']['fixed_epochs'], residual=r['params']['residual'], CV_RMSE=r['cv_RMSE'])
        for r in selections])
    gate_rows = pd.DataFrame([dict(strategy=g['strategy'], h=g['h'], reference=g['reference'],
        seed=r['seed'], passed=r['passed'], RMSE_ratio=r['RMSE_ratio'], MAE_ratio=r['MAE_ratio'],
        worst_quarter_ratio=r['worst_quarter_ratio']) for g in gates for r in g['seed_results']])
    gate_rows.to_csv(args.out/'development_gates_by_seed.csv', index=False, encoding='utf-8-sig')
    parameter_rows.to_csv(args.out/'parameters.csv', index=False, encoding='utf-8-sig')
    both = [dict(strategy=s, h=h, passed_both_references_all_seeds=all(
        g['passed_all_seeds'] for g in gates if g['strategy'] == s and g['h'] == h))
        for s, h in [('direct', h) for h in range(1, 8)] + [('multi', 7)]]
    both_frame = pd.DataFrame(both)
    passed = both_frame[both_frame.passed_both_references_all_seeds]
    passed_names = '、'.join(f"{r.strategy} h{r.h}" for r in passed.itertuples()) or '无'
    aggregate = summary(previous)
    aggregate = aggregate[aggregate.model.isin(['LSTM', 'Mean', 'Stacking'])]
    lognormal = parameter_rows[parameter_rows.objective == 'lognormal']
    by_seed = previous.groupby(['strategy', 'model', 'seed'])[['RMSE', 'RMSE_previous', 'MAE', 'MAE_previous']].mean().reset_index()
    by_seed['RMSE_change_pct'] = (by_seed.RMSE / by_seed.RMSE_previous - 1) * 100
    by_seed.to_csv(args.out/'comparison_to_previous_by_seed.csv', index=False, encoding='utf-8-sig')
    candidate_rows = []
    for record in selections:
        table = pd.read_csv(run/'cv_results'/f"{record['strategy']}_h{record['h']}.csv")
        for objective, own in table.groupby('param_objective'):
            candidate_rows.append(dict(strategy=record['strategy'], h=record['h'], objective=objective,
                best_CV_RMSE=-float(own.mean_test_score.max()), candidates=len(own)))
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(args.out/'objective_search_summary.csv', index=False, encoding='utf-8-sig')
    primary = metrics[(metrics.seed == 42) & metrics.model.isin(['LSTM', 'Mean', 'Stacking'])].drop(columns='seed')
    data = load_dataset(str(run/'dataset.npz'))
    train = np.flatnonzero(data['origin_days'] + 7 <= np.datetime64('2003-12-31').astype('int64'))
    q = data['features'][data['positions'], list(data['feature_names']).index('Discharge_current')]
    persistence = []
    for label, _, vi in year_folds(data, train):
        for strategy, h in [('direct', h) for h in range(1, 8)] + [('multi', 7)]:
            truth = np.asarray(get_target(data, strategy, h)[vi]).reshape(len(vi), -1)
            pred = np.repeat(q[vi, None], truth.shape[1], axis=1)
            persistence.append(dict(strategy=strategy, task_h=h, fold=label,
                persistence_RMSE=score(truth, pred), persistence_MAE=float(np.abs(truth - pred).mean())))
    quarters = pd.read_csv(run/'development_quarter_metrics.csv')
    persistence_comparison = quarters.merge(pd.DataFrame(persistence), on=['strategy', 'task_h', 'fold'], validate='many_to_one')
    persistence_comparison.to_csv(args.out/'development_persistence_comparison.csv', index=False, encoding='utf-8-sig')
    persistence_summary = persistence_comparison.groupby(['strategy', 'task_h'])[
        ['candidate_RMSE', 'persistence_RMSE', 'candidate_MAE', 'persistence_MAE']].mean().reset_index()
    lines = ['# 任务1至7+继续训练结果', '', '2026-11-11｜独立继续训练记录；不替换原正式Word。', '',
        '## 结论与适用边界', '',
        f"本轮新增{verified['fits']}次拟合，设备{manifest['gpu']}，耗时{manifest['seconds']/60:.2f}分钟；种子42/43/44全部保留。与前轮GPU实验相比，只替换LSTM及由其派生的集成；ANN/RF复用且预测逐项一致。", '',
        '**本轮整体未提升，不采用为新的整套配置。** 相对前轮，三种子平均七步RMSE的LSTM、Mean与Stacking在两种策略下均升高；不按2004局部有利步长拼接新结果。', '',
        f'八项任务中，历史网格选择了{len(lognormal)}项lognormal、{8-len(lognormal)}项log-MSE。运行名含lognormal不代表该分支必定胜出。', '',
        f'同时对照原LSTM参数参照与前轮LSTM、并要求三种子全部通过开发门槛的任务：**{passed_names}**。门槛未过的配置不因2004回溯分数变好就推荐替换。', '',
        '2004已在此前多轮查看，以下均为回溯评估，不是独立盲测。多种子反映初始化随机性，不是独立年份，也不证明统计显著性。', '',
        '## 相对前轮GPU：三种子平均七步指标', '', markdown_table(aggregate), '',
        'RMSE/MAE先逐h算指标，再按7个h与3个seed作算术平均。change_pct为相对变化，负数表示误差降低；不是合并全部步长重新计算的RMSE。', '',
        '## 相对原正式交付：三种子平均', '',
        markdown_table(summary(delivery)[lambda f: f.model.isin(['LSTM', 'Mean', 'Stacking'])]), '',
        '原交付的ANN/RF及OOF协议与本轮可能不同；此对照不能将全部差异归因于分布目标。', '',
        '## 逐种子与前轮对照', '', markdown_table(by_seed[by_seed.model.isin(['LSTM', 'Mean', 'Stacking'])]), '',
        '## 历史搜索胜出参数', '', markdown_table(parameter_rows), '',
        '所有参数只使用最晚目标≤2002-12-31的4个90天净化前向折选择；16候选/任务。AdamW、训练折内固定种子随机批次、weight_decay0.001、batch64、dropout0.1、lr0.001。没有用2003全年或2004选择这些参数。', '',
        '## 目标函数对照（仅2002历史CV）', '', markdown_table(candidate_frame), '',
        '各分支最优历史CV分数属于选参数据，不能当作独立验证。lognormal输出边际条件均值exp(mu_log+variance_log/2)，方差范围预先固定；未证实分布假设或区间校准。输出维数翻倍还使直接lognormal分支head隐藏宽度为64，普通直接log-MSE为32，因此不是纯损失消融。', '',
        '## 2003开发准入', '', markdown_table(both_frame), '', markdown_table(gate_rows), '',
        'reference=original为原正式LSTM参数的同协议重新训练；previous为前轮GPU的已冻结LSTM。门槛是季度平均RMSE比≤0.995、MAE比≤1.02、最差季度RMSE比≤1.15，三个种子均需满足。开发门槛不反选早期OOF参数。', '',
        '## 历史2003持久性基线', '', markdown_table(persistence_summary), '',
        '持久性基线固定为未来各步均等于Q(t)，没有可调参数。表中候选先平均季度再平均种子，仅用历史2003数据作合理性检查，不按此表反选模型。', '',
        '## 任务2～7：主seed42全部结果', '', markdown_table(primary), '',
        'R²=1−SSE/SST，允许为负；Discharge单位未可靠确认，不擅自补单位。负预测如实保留，不事后裁剪。', '',
        '## 独立核验与文件', '',
        f"独立核验重建{verified['all_seed_prediction_rows']}行预测对应的{verified['all_seed_metric_rows']}行指标，重建Mean、Stacking、24个最终LSTM及季度候选模型预测，核对{verified['fits']}条训练记录、{verified['frozen_files_verified']}个冻结文件和{verified['reused_files_verified']}个直接引用文件，并独立核对前轮全部{verified['previous_frozen_files_verified']}个冻结文件（含96个复用最终ANN/RF文件）；开发门槛从季度指标重建一致。", '',
        '本轮完成后修复了未来运行的Windows引用路径筛选；run内源码快照保持原样。独立核验补查前轮全部冻结文件及复用预测，不修改训练结果。核验原双层LSTM参照使用原CUDA设备，避免跨设备数值差异。', '',
        '- [完整指标](metrics_all_seeds.csv)',
        '- [相对前轮逐h对照](comparison_to_previous_by_horizon.csv)',
        '- [相对原交付逐h对照](comparison_to_delivery_by_horizon.csv)',
        '- [开发准入逐seed](development_gates_by_seed.csv)',
        '- [独立核验](independent_verification.json)',
        '- [来源与耗时](manifest.json)',
        '- [冻结配置](frozen_config.json)',
        '- [继续训练方案](../../任务1至7+继续训练方案_2026-11-11.md)', '',
        '模型、逐日预测与完整日志仍在本地独立run，轻量记录不包含大权重，不表示已推送远程。原正式Word、图片和截图未变。', '']
    (args.out/'任务1至7+继续训练结果_2026-11-11.md').write_text('\n'.join(lines), encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
