# 直接残差扩展与最终对照

日期：2026-11-11。内部实验，不替换正式Word。

模式：final；420fit、42模型全部恢复核验通过。
普通/残差候选及历史选择规则在评价前冻结，不按2004逐h挑选。

## 历史内层所选策略

| 年份 | h | 所选策略 | RMSE | MAE |
| --- | --- | --- | ---: | ---: |
| 2004 | 1 | last_residual_log | 647.52 | 181.74 |
| 2004 | 2 | last_residual_log | 717.33 | 274.61 |
| 2004 | 3 | last_residual_log | 733.19 | 319.01 |
| 2004 | 4 | last_residual_log | 753.70 | 352.37 |
| 2004 | 5 | last_plain_log | 813.34 | 376.80 |
| 2004 | 6 | last_plain_log | 819.74 | 385.04 |
| 2004 | 7 | last_plain_log | 824.79 | 390.18 |

## 七步平均汇总

- 2004：所选RMSE 758.5149、MAE 325.6771；相对本轮普通LSTM RMSE变化-0.98%。

## 与原正式报告同一2004样本比较

三seed七步平均RMSE：805.7074 → 758.5149（-5.86%）。
MAE：387.9261 → 325.6771（-16.05%）。
原正式预测按起报日/目标日/h/seed合并，真值一致；原指标另由原预测重算核对。
这是已经查看过的2004回溯评价，不是盲测；不同训练起始日/容量等限制纯结构归因。
[逐h/seed对比](comparison_by_horizon.csv) · [逐seed七步均值](comparison_by_seed.csv) · [对原报告的时间块敏感性](original_block_sensitivity.csv)

## 证据与边界

- [逐h/seed指标](metrics.csv)
- [全部固定候选](annual_summary.csv)
- [配对变化](paired_changes.csv)
- [时间块敏感性](paired_block_sensitivity.csv)
- [严格持久性](persistence.csv)
- [同epoch消融](same_epoch_ablation.csv)
- [季度](quarter_metrics.csv)
- [内层冻结选择](selection.json)
- [独立核验](verification.json)
- [训练前固定方案](../../任务1至7+直接残差扩展与最终对照方案_2026-11-11.md)

RMSE先对每个h计算，再算术平均h及seed，不是扁平RMSE。
7/14/30日连续块同步抽样seed/h、600次；区间只是已观测历史敏感性，不保证未来。
本轮未新训ANN/RF/多输出/集成，不按外层拼最优，不保证0.8。原CSV/Word/旧run未覆盖，未commit/push。
