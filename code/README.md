# 代码入口与版本说明

本目录保留历史实现，避免重命名模块后破坏旧模型的 `joblib` 恢复、源码快照和报告生成依赖。新实验使用明确入口与职责分离的子目录。

| 用途 | 入口／模块 | 注意事项 |
| --- | --- | --- |
| 第1天R²专项 | [day1_train.py](runoff_training/day1_train.py)、[day1_model.py](runoff_training/day1_model.py) | 原始尺度增量与Q(t)跳连、四折合并R²、三seed；300fit/60模型，仅历史；不是完整七项验收替代 |
| 第1天专项核验/导出 | [verify_runoff_day1.py](verify_runoff_day1.py)、[测试](test_runoff_day1.py) | 实际外层scaler/四族恢复/时间净化/冻结选择/旧log残差对照；正式300fit/60模型全部恢复核验完成，13项专项单测/全部94项回归通过；未过跨年推进条件，无新2004成绩 |
| 原始尺度损失历史对照 | [loss_train.py](runoff_training/loss_train.py)、[loss_model.py](runoff_training/loss_model.py) | 保持log正输出头，普通/残差×log/原始尺度MSE，h1/h3/h7；仅历史、720fit固定预算，不含2004评价入口 |
| 损失对照核验/导出 | [verify_runoff_loss.py](verify_runoff_loss.py)、[测试](test_runoff_loss.py) | 实际原始尺度归一化统计、log scaler与网络核验；三seed/跨年推进门槛，不事后挑外层候选 |
| 直接残差扩展与最终对照 | [direct_train.py](runoff_training/direct_train.py)、[direct_data.py](runoff_training/direct_data.py) | 七步普通/残差，h1同epoch消融；historical/final分阶段，冻结后才构造2004评价数组；新run、不支持resume |
| 直接残差核验/分析 | [verify_runoff_direct.py](verify_runoff_direct.py)、[analyze_runoff_direct.py](analyze_runoff_direct.py) | 固定协议、实际scaler/容量重算，按日期与原正式预测对齐；[测试](test_runoff_direct.py) |
| 策略增强局部重训 | [strategy_train.py](runoff_training/strategy_train.py)、[strategy_model.py](runoff_training/strategy_model.py) | 先冻结全部历史内层选择，再评价；等容量池化×持久性残差，h7另比较sqrt；不覆盖旧run |
| 策略增强核验/分析 | [verify_runoff_strategy.py](verify_runoff_strategy.py)、[analyze_runoff_strategy.py](analyze_runoff_strategy.py) | 全部模型恢复、日期/指标/fit核验；单列持久性和内层所选策略 |
| 冻结历史记忆诊断 | [diagnose_runoff_memory.py](diagnose_runoff_memory.py) | 36模型、0fit，早60日/近7日固定扰动及梯度占比；非新精度成绩 |
| 受控窗口/状态/损失实验 | [runoff_training/controlled_train.py](runoff_training/controlled_train.py) | 日期驱动、共同样本、等容量2×2输入对照；仅历史内层选参，三seed |
| 受控实验独立数据与模型 | [controlled_data.py](runoff_training/controlled_data.py)、[controlled_model.py](runoff_training/controlled_model.py) | 不修改旧prepare；固定字段契约、状态因果递推、严格id与恢复hash |
| 受控实验核验/分析 | [verify_runoff_controlled.py](verify_runoff_controlled.py)、[analyze_runoff_controlled.py](analyze_runoff_controlled.py) | 全部模型恢复及日期/指标核验，再导出年度/季度指标与时间块敏感性 |
| 受控实验测试 | [test_runoff_controlled.py](test_runoff_controlled.py)、[分析测试](test_runoff_controlled_analysis.py) | 11项训练测试+4项分析测试：目标映射、未来扰动、容量/缩放、时间净化、恢复与聚合口径等 |
| 2026-11-11 GPU改进训练 | `train_runoff.py` → `runoff_training/train.py` | 独立run，不覆盖历史报告；支持auto/cpu/cuda |
| 新实验模型 | `runoff_training/model.py` | 精简滞后输入、符号对数变换、可选持久性残差，原生单/七输出 |
| 新实验测试 | `test_runoff_training.py` | 折内缩放、索引映射、时间净化、残差初值及恢复 |
| 新实验独立核验 | `runoff_training/verify.py` | 重建全部指标、简单平均与Stacking，核对目标日期、真值及冻结权重，再对照旧交付 |
| 新实验轻量导出 | `runoff_training/export.py` | 先独立核验，再导出指标、配置、对照与实验说明；不复制大模型或覆盖旧Word |
| 继续训练：分布目标对照 | `runoff_training/continue_train.py`、`distribution.py` | log-MSE与异方差lognormal、AdamW、训练折内随机批次；复用前轮ANN/RF |
| 继续训练独立核验 | `verify_runoff_continued.py` | 重建最终LSTM预测、全部指标与集成，核验来源hash和三种子开发门槛 |
| 冻结模型历史诊断 | [diagnose_runoff_history.py](diagnose_runoff_history.py) | 192个历史LSTM推理、真实持久性对照及误差分段；0次新拟合，不读2004预测 |
| 受控学习曲线 | [diagnose_runoff_learning_curve.py](diagnose_runoff_learning_curve.py) | 24条从零至120轮的历史诊断；最优epoch只供诊断，不能倒填OOF |
| 历史线性基线 | [diagnose_runoff_baselines.py](diagnose_runoff_baselines.py) | 仅Q(t)回归、AR、ARX；60次轻量拟合，不替代课程模型 |
| 融合历史重建 | [diagnose_runoff_fusion.py](diagnose_runoff_fusion.py) | 复用冻结OOF、96次元层拟合；重建已有alpha，非独立验证，不读2004预测 |
| 第一次验收分析 | `data_analysis.py`、`feature_engineering.py` | 历史分析和当前数据准备共用；修改会影响后续输入 |
| 历史一键流程 | `run_all.py` → `experiment_v6.py`、`models.py` | 当前LSTM已为PyTorch；写共享output和根报告，不是当前独立实验入口 |
| 更早实验入口 | `experiment.py` | 支持加载旧权重热启动；不作为新训练依据 |
| 历史V2～V6探索 | `experiment_v2.py`～`experiment_v6.py` | 会覆盖共享output，不作为最新结果依据；文件版本不同于acceptance3训练轮次 |
| 第三次验收第1轮 | `acceptance3_train.py`、`acceptance3_model.py` | 独立run，后续轮次复用数据与模型辅助函数 |
| 第三次验收第2轮 | `acceptance3_v2_train.py`、`acceptance3_v2_model.py` | 直接策略历史正式结果来自此轮 |
| 第3轮输入筛选 | `acceptance3_v3_train.py` | 实际为阶段A，不等于第3轮全流程入口 |
| 第3轮年度复核 | `acceptance3_v3_stage_b.py` | 阶段B；按历史内层选参、年度外层比较 |
| 第3轮季度OOF与冻结 | `acceptance3_v3_stage_c.py` | 阶段C；原生多输出，CPU设备为历史固定配置 |
| 第3轮回溯评估 | `acceptance3_v3_evaluate.py` | 阶段D；只接受已冻结的正式来源 |
| 旧结果截图与核验 | `acceptance3*_evidence.py`、`acceptance3*_verify.py` | 不等同于重新训练；部分依赖本地大模型目录 |
| 报告生成 | [验收/report code/](../验收/report%20code/) | 此处的早期 `make_report.py`／`report_sections.py` 属于原始总报告流程 |

## 当前第1天专项

[训练前方案](../docs/任务1至7+第一天R2专项目标与方案_2026-11-11.md)固定第1天R²为主指标、目标0.8。正式历史300fit/60模型已全部恢复核验，见[完整结果](../docs/experiments/20261111_day1_delta/README.md)与[结论](../docs/任务1至7+第一天R2专项训练结论_2026-11-11.md)。历史所选LSTM R²：2002为0.8802→0.8813，2003为0.4699→0.4766；MAE增加，未通过推进条件，不开展该方案2004评价、不替换模型。[预检v2](../docs/experiments/20261111_day1_delta_smoke_v2/README.md)40fit/8模型仅作流程证据。13项专项单测、69项runoff+25项acceptance回归共94项通过。正式训练前补齐直接SSE留档和常量验证折测试；旧预检快照保留不改。四族各保留三seed，全部内层选择先冻结再外层拟合。

以下仅用于新目录复现，固定预算已经结束，不是仍有训练待完成：

```powershell
$env:PYTHONPATH = 'code'
python -m runoff_training.day1_train --run-id 自定义唯一第1天历史名称 --device cuda
python code/verify_runoff_day1.py --run-id 自定义唯一第1天历史名称 --out docs/experiments/自定义唯一第1天结果目录
```

本入口仅历史评价。须先通过方案中的跨年LSTM推进门槛，才另行冻结全前四年选择并评价2004；不以历史某年R²≥0.8替代最终达标，不按外层挑冠军，不自动改Word。

## 当前原始尺度损失实验

[预固定历史方案](../docs/任务1至7+原始尺度损失历史对照方案_2026-11-11.md)已完成正式720fit/72模型和全部恢复核验，仅2002/2003开发评价，不含2004入口；另有96fit预检/24模型。56项runoff+25项acceptance测试，共81项通过。2002第3/7天R²小幅提高、2003所选流程无收益，未过预固定跨年门槛，不推进至完整七步或2004评价。见[结论](../docs/任务1至7+原始尺度损失历史对照结论_2026-11-11.md)与[最终完整结果](../docs/experiments/20261111_physical_loss_verified/README.md)。本轮不是首次尝试原始尺度误差，新增点为90日w90_base、相同log参数化下的损失空间对照。分析阶段修复了旧基线DataFrame索引比较错误；训练源码/冻结工件不改，首次不完整导出不作为证据。

```powershell
$env:PYTHONPATH = 'code'
python -m runoff_training.loss_train --run-id 自定义唯一损失实验名称 --device cuda
python code/verify_runoff_loss.py --run-id 自定义唯一损失实验名称 --out docs/experiments/自定义唯一损失结果目录
```

## 最新完成结果与复现入口

[直接残差扩展与最终对照](../docs/任务1至7+直接残差扩展与最终对照方案_2026-11-11.md)已完成：正式历史864fit+最终420fit，150模型全部恢复核验通过；另完成两阶段176fit预检、50模型核验和71项回归测试。与原正式直接LSTM的同一2004评价样本相比，三seed七步平均RMSE805.71→758.51（−5.86%）、MAE387.93→325.68（−16.05%）；h1 RMSE退化5.97%、h7收益弱，不代表全部课程模型或集成都升级。见[新旧对比结论](../docs/任务1至7+直接残差最终新旧对比结论_2026-11-11.md)、[最终结果](../docs/experiments/20261111_direct_final/README.md)、[历史消融](../docs/experiments/20261111_direct_expand/README.md)。本轮固定预算已结束，不按2004继续改选或追加网格，不改旧快照和正式Word。

以下命令用于复现协议，使用新目录，不是仍有训练待完成：

```powershell
$env:PYTHONPATH = 'code'
python -m runoff_training.direct_train --run-id 自定义历史名称 --mode historical --device cuda
python code/verify_runoff_direct.py --run-id 自定义历史名称
python -m runoff_training.direct_train --run-id 自定义最终名称 --mode final --device cuda
python code/analyze_runoff_direct.py --run-id 自定义最终名称 --out docs/experiments/自定义目录
```

上一轮为[策略增强局部重训](../docs/任务1至7+策略增强与局部重训方案_2026-11-11.md)：等容量读取方式×持久性残差，h7另设sqrt，所有内层选择先冻结再评价。正式`formal_20261111_strategy_v1`完成960fit、96模型全部恢复核验；[结果](../docs/experiments/20261111_strategy/README.md)与[结论](../docs/任务1至7+策略增强重训结论_2026-11-11.md)记录了h1局部收益、多输出不稳及h7选择失误。新增8模型+4分析测试，总63项回归测试通过。仅新run、不支持resume；不覆盖旧Word/CSV或替代必做模型。

从项目根目录执行新的局部实验：

```powershell
$env:PYTHONPATH = 'code'
python -m runoff_training.strategy_train --run-id 自定义唯一名称 --device cuda
python code/verify_runoff_strategy.py --run-id 自定义唯一名称
python code/analyze_runoff_strategy.py --run-id 自定义唯一名称 --out docs/experiments/自定义唯一目录
```

此前全面诊断见[全面复核与下一轮调整方案](../docs/diagnostics/20261111_full_review/README.md)，其中列出教师依据、学习曲线、误差分段、选参稳定性及融合重建证据。四个诊断脚本均支持`--out`，输出必须使用尚不存在的新目录；具体必需参数以各脚本`--help`为准。融合的已完成产物位于`fusion_verified/`，`fusion/`保留首次序列化失败记录，不应混用。

部分短预见期可从增训获益，但长预见期/多输出已有过拟合迹象。不建议原样扩大前两轮网格；新的[受控窗口/状态/损失方案](../docs/任务1至7+受控窗口状态与损失实验方案_2026-11-11.md)已完成正式756fit和全部108模型独立恢复核验；见[结果](../docs/experiments/20261111_controlled/README.md)与[结论](../docs/任务1至7+受控实验结论与后续_2026-11-11.md)。没有跨年整体稳定升级，不能把局部最优值写成交付成绩。运行后已修复后续v2入口的半候选拒绝、工件hash和环境恢复保护，另一次144fit预检通过；旧v1快照不改，也不能跨源码resume。

```powershell
$env:PYTHONPATH = 'code'
python -m runoff_training.controlled_train --run-id 自定义唯一名称 --device cuda
python code/verify_runoff_controlled.py --run-id 自定义唯一名称
# 核验通过后导出轻量分析，目标目录必须不存在
python code/analyze_runoff_controlled.py --run-id 自定义唯一名称 --out docs/experiments/自定义唯一目录
```

## 既有训练流程使用方式

以下命令用于复现已记录的第一轮流程，并非执行新诊断方案。从项目根目录执行：

```powershell
python -m unittest discover -s code -p test_runoff_training.py -v
python -m unittest discover -s code -p "test_acceptance3*.py" -v
python code/train_runoff.py --run-id 自定义唯一运行名 --device auto
```

`--smoke`只检查缩小后的流程，直接策略仅训练h1，不得当作完整七项任务结果。`--resume`用于数据、源码及设备不变的中断恢复，不会用旧模型权重热启动新候选。

## 已完成结果与独立核验

[2026-11-11实验记录](../docs/experiments/20261111_gpu_lag_residual/任务1至7+GPU改进训练结果_2026-11-11.md)保留了872次fit、三seed及新旧逐h对照。直接LSTM/Mean部分改善，但Stacking不稳定，八项配置只有direct h1通过开发准入，因此不推荐整体替换旧交付。

以下命令从项目根目录执行，核验与导出依赖本地完整run（包括未上传的模型权重）：

```powershell
$env:PYTHONPATH = 'code'
python -m runoff_training.verify --run-id formal_20261111_gpu_lag_residual
# 导出目标必须是尚不存在的新目录，避免覆盖已有记录
python -m runoff_training.export --run-id formal_20261111_gpu_lag_residual --out docs/experiments/自定义唯一导出目录
```

## 继续训练（分布目标对照）

详见[继续训练方案](../docs/任务1至7+继续训练方案_2026-11-11.md)与[已完成结果](../docs/experiments/20261111_lognormal/任务1至7+继续训练结果_2026-11-11.md)。这是独立新run，不在旧权重上累计轮数；728次新fit，ANN/RF复用前轮已核验产物。本轮整体退化，不采用为新整套配置；文件名不表示lognormal分支必定胜出。

```powershell
$env:PYTHONPATH = 'code'
python -m runoff_training.continue_train --run-id 自定义唯一运行名 --device cuda
# 完成后只读核验；可加 --out 尚不存在的目录 导出轻量记录
python code/verify_runoff_continued.py --run-id 自定义唯一运行名
```

有关数据边界、候选、开发门槛、预算及课程要求，见[本轮方案](../docs/任务1至7+训练方案与要求_2026-11-11.md)。当前正式Word仍以[第三次验收目录](../验收/Acceptance%203/)中的现有版本为准，新run完成不表示Word已自动更新。
