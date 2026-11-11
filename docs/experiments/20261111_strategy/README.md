# 任务1至7：策略增强局部重训结果

日期：2026-11-11。内部开发对照，不替换正式验收报告。

完成960fit、96个外层模型，全部恢复独立核验通过。
模型数组和性能评价止于2003年；2004未用于本轮选择或评价。全部内层选择已在外层拟合/预测前冻结。
任务仅含direct h1、direct h7、原生multi；不是完整七任务新交付，也未改变ANN/RF及集成。

## 1. 固定候选与口径

last_plain_log为同协议普通90日LSTM；last_mean为末状态与全部状态均值等权平均，不增加可训练参数。
residual为目标变换尺度的Q(t)跳连，末层零初始化，初始预测恰为持久性；不声称残差与普通头初始化相同。
三seed42/43/44，原尺度逐h RMSE平均，再对seed平均，不是扁平RMSE。
各策略由净化历史内层选1/10/30轮；因此是同搜索预算策略比较，不是固定epoch纯消融。

## 2. 三seed年度结果

| 年度 | 任务 | 策略 | RMSE | MAE | 内层选中 |
| --- | --- | --- | ---: | ---: | --- |
| 2002 | h1 | last_plain_log | 332.47 | 158.23 | 否 |
| 2002 | h1 | last_residual_log | 242.46 | 98.22 | 是 |
| 2002 | h1 | last_mean_plain_log | 488.99 | 223.15 | 否 |
| 2002 | h1 | last_mean_residual_log | 259.05 | 106.89 | 否 |
| 2002 | h7 | last_plain_log | 679.67 | 288.30 | 否 |
| 2002 | h7 | last_plain_sqrt | 641.77 | 271.23 | 否 |
| 2002 | h7 | last_residual_log | 599.83 | 285.65 | 是 |
| 2002 | h7 | last_residual_sqrt | 648.99 | 316.53 | 否 |
| 2002 | h7 | last_mean_plain_log | 688.32 | 296.09 | 否 |
| 2002 | h7 | last_mean_plain_sqrt | 656.50 | 284.07 | 否 |
| 2002 | h7 | last_mean_residual_log | 746.38 | 353.95 | 否 |
| 2002 | h7 | last_mean_residual_sqrt | 747.37 | 354.28 | 否 |
| 2002 | multi | last_plain_log | 560.54 | 275.86 | 否 |
| 2002 | multi | last_residual_log | 519.37 | 259.03 | 是 |
| 2002 | multi | last_mean_plain_log | 631.19 | 274.68 | 否 |
| 2002 | multi | last_mean_residual_log | 559.71 | 262.83 | 否 |
| 2003 | h1 | last_plain_log | 982.03 | 310.86 | 否 |
| 2003 | h1 | last_residual_log | 918.66 | 254.95 | 是 |
| 2003 | h1 | last_mean_plain_log | 994.53 | 344.37 | 否 |
| 2003 | h1 | last_mean_residual_log | 897.42 | 254.43 | 否 |
| 2003 | h7 | last_plain_log | 1363.76 | 634.82 | 否 |
| 2003 | h7 | last_plain_sqrt | 1371.77 | 649.47 | 是 |
| 2003 | h7 | last_residual_log | 1254.81 | 567.38 | 否 |
| 2003 | h7 | last_residual_sqrt | 1274.10 | 601.14 | 否 |
| 2003 | h7 | last_mean_plain_log | 1334.79 | 594.79 | 否 |
| 2003 | h7 | last_mean_plain_sqrt | 1342.29 | 609.19 | 否 |
| 2003 | h7 | last_mean_residual_log | 1304.12 | 610.02 | 否 |
| 2003 | h7 | last_mean_residual_sqrt | 1291.54 | 657.57 | 否 |
| 2003 | multi | last_plain_log | 1270.58 | 544.28 | 否 |
| 2003 | multi | last_residual_log | 1167.18 | 461.48 | 是 |
| 2003 | multi | last_mean_plain_log | 1195.84 | 477.79 | 否 |
| 2003 | multi | last_mean_residual_log | 1195.02 | 483.75 | 否 |

## 3. 仅按内层选择的策略表现

此表不按外层最优值挑选。负变化表示改善。

| 年度 | 任务 | 内层所选 | 对普通log变化 | 对持久性变化 |
| --- | --- | --- | ---: | ---: |
| 2002 | h1 | last_residual_log | -27.07% | -6.91% |
| 2002 | h7 | last_residual_log | -11.75% | -19.80% |
| 2002 | multi | last_residual_log | -7.35% | -7.37% |
| 2003 | h1 | last_residual_log | -6.45% | -3.86% |
| 2003 | h7 | last_plain_sqrt | +0.59% | +3.45% |
| 2003 | multi | last_residual_log | -8.14% | -11.55% |

## 4. 证据和限制

- [逐h/逐seed R²、RMSE、MAE](metrics.csv)
- [全部内层选择与CV汇总](selection.json)
- [年度汇总](annual_summary.csv)
- [配对变化](paired_changes.csv)
- [7/14/30日块重采样](paired_block_sensitivity.csv)
- [季度结果](quarter_metrics.csv)
- [严格持久性](persistence.csv)
- [训练P90高流量分段](error_segments.csv)
- [独立核验](verification.json)
- [运行方案与指导书边界](../../任务1至7+策略增强与局部重训方案_2026-11-11.md)

时间块同步重采样所有seed和h，600次，报告2.5%～97.5%经验分位；这些不是独立未来年份的泛化保证。
两个外层年份均是已查看的开发历史；最早内折训练样本有限。更多seed不是更多独立年份。
不得按外层逐h、季度或seed拼最优成绩；固定策略的局部好结果不等于内层选策略整条流程稳定有效。
标准课程模型、真Mean和Stacking仍须保留；本轮增强单列，不替代规定任务。
原CSV、正式Word、旧run均未改，未commit/push。
