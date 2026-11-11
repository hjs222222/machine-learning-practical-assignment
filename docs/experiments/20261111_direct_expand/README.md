# 直接残差扩展与最终对照

日期：2026-11-11。内部实验，不替换正式Word。

模式：historical；864fit、108模型全部恢复核验通过。
普通/残差候选及历史选择规则在评价前冻结，不按2004逐h挑选。

## 历史内层所选策略

| 年份 | h | 所选策略 | RMSE | MAE |
| --- | --- | --- | ---: | ---: |
| 2002 | 1 | last_residual_log | 242.46 | 98.22 |
| 2002 | 2 | last_residual_log | 417.77 | 179.17 |
| 2002 | 3 | last_residual_log | 519.37 | 244.55 |
| 2002 | 4 | last_residual_log | 561.31 | 287.28 |
| 2002 | 5 | last_residual_log | 605.26 | 304.05 |
| 2002 | 6 | last_residual_log | 576.32 | 274.10 |
| 2002 | 7 | last_residual_log | 599.83 | 285.65 |
| 2003 | 1 | last_residual_log | 918.66 | 254.95 |
| 2003 | 2 | last_residual_log | 1116.13 | 377.35 |
| 2003 | 3 | last_residual_log | 1205.33 | 451.46 |
| 2003 | 4 | last_residual_log | 1237.79 | 501.01 |
| 2003 | 5 | last_plain_log | 1326.98 | 596.92 |
| 2003 | 6 | last_plain_log | 1351.12 | 618.16 |
| 2003 | 7 | last_plain_log | 1363.76 | 634.82 |

## 七步平均汇总

- 2002：所选RMSE 503.1871、MAE 239.0054；相对本轮普通LSTM RMSE变化-5.85%。
- 2003：所选RMSE 1217.1123、MAE 490.6670；相对本轮普通LSTM RMSE变化-2.91%。

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
