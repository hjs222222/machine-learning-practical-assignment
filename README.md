# 基于机器学习的径流预测

《机器学习课程实践》实验 —— USGS 01047000 流域逐日径流多步预测。

## 目录结构

```
├── Teacher Resources & Requirements/ # 教师原件、辅助阅读版与 01047000.csv
├── docs/                         # 当前任务要求、训练方案与项目说明
├── code/                         # 建模、测试和历史实现
│   ├── config.py                 # 全局配置（路径、变量对照、特征与建模参数）
│   ├── utils_plot.py             # 绘图公共工具（中文字体、配色、保存）
│   ├── data_analysis.py          # 第一部分：数据分析（统计 + 可视化 + 相关性）
│   ├── feature_engineering.py    # 第二部分：特征工程（编码/滞后/窗口/归一化/特征选择）
│   ├── models.py                 # 模型定义（ANN/RF/LSTM）+ GridSearchCV
│   ├── experiment.py             # 第三部分：多步直接/多输出策略、集成、评价
│   ├── results_analysis.py       # 第四部分：结果可视化与误差分析
│   ├── make_report.py            # 报告生成（封面 + 第一~三章）
│   ├── report_sections.py        # 报告生成（第四~六章 + 附录）
│   ├── run_all.py                # 原始一键流程（非最新验收训练入口）
│   ├── train_runoff.py           # 新 GPU 改进实验统一入口
│   └── runoff_training/          # 新实验模型与训练流程
├── output/
│   ├── figures/                  # 全部统计图表（PNG）
│   ├── results/                  # 全部结果表（CSV）与预测文件（NPZ）
│   ├── acceptance3_runs/          # 各轮独立训练目录（大部分只保留在本地）
│   └── run_full.log              # 原始流程运行日志
├── 实验报告模版/                  # 最终课程报告与个人思政报告模板
├── 验收/
│   ├── Acceptance 1/             # 第一次验收材料
│   ├── Acceptance 2/             # 第二次验收材料
│   ├── Acceptance 3/             # 第三次验收正式报告及附件
│   │   ├── 第x次阶段验收报告.md    # 也可以是 .html 或 .docx
│   │   ├── figures/               # 报告引用的最终图片
│   │   ├── data/                  # 报告引用的最终 CSV 等数据
│   │   └── artifacts/             # 其他最终交付附件
│   ├── report code/
│   │   └── reportx/               # 生成第 x 次报告的代码与非交付性中间文件
│   └── acceptance.md              # 阶段验收归档规范
├── AGENT.MD                       # AI 编写与检查阶段验收报告的协作规范
└── README.md
```

## 当前改进训练（2026-11-11）

- **第1天R²专项已完成，仍未达跨年0.8目标：**[结论](docs/任务1至7+第一天R2专项训练结论_2026-11-11.md)、[完整结果](docs/experiments/20261111_day1_delta/README.md)、[预固定方案](docs/任务1至7+第一天R2专项目标与方案_2026-11-11.md)。原始尺度增量/因果趋势输入/四折合并R²，对照Ridge/RF/ANN/LSTM；正式300fit、60模型全部恢复核验，94项回归测试通过。所选LSTM历史第1天R²：2002为0.8802→0.8813，2003为0.4699→0.4766，仅小幅改善，MAE分别增加16.49%/7.54%，未过推进条件。**不推进本方案至2004，不替换模型。** 2004年直接LSTM三seed第1天R²仍为原正式约0.4733、上轮残差所选约0.4086；不能以2002的高分替代。入口与边界见[代码说明](code/README.md)。
- **原始尺度损失历史对照已完成，不推进：**[结论](docs/任务1至7+原始尺度损失历史对照结论_2026-11-11.md)、[完整结果](docs/experiments/20261111_physical_loss_verified/README.md)、[预固定方案](docs/任务1至7+原始尺度损失历史对照方案_2026-11-11.md)。正式720fit、72模型全部恢复核验，81项回归测试通过。2002年第3/7天R²小幅提高，三h平均RMSE下降1.71%；2003内层仍全选log，无新增收益，未通过跨年推进门槛。不扩展该轮网格、不新增2004评价、不替换上轮模型。首次分析导出失败后已修复并仅重做核验/导出，最终证据以带_verified的目录为准。
- **直接残差扩展与最终对照已完成：**[新旧对比结论](docs/任务1至7+直接残差最终新旧对比结论_2026-11-11.md)、[2004完整结果](docs/experiments/20261111_direct_final/README.md)、[历史扩展与消融](docs/experiments/20261111_direct_expand/README.md)、[训练前固定方案](docs/任务1至7+直接残差扩展与最终对照方案_2026-11-11.md)。正式1284fit、150模型全部恢复核验通过，71项回归测试通过；同一2004年359个起报点上，直接LSTM三seed七步平均RMSE805.71→758.51（−5.86%）、MAE387.93→325.68（−16.05%）。**h1 RMSE退化5.97%，h7收益弱，不是逐h或全项目最优升级。** 只由前四年选参并冻结，2004为回溯评价；不按它追加网格，不新训多输出/集成，不覆盖原Word。入口见[代码说明](code/README.md)。
- **策略增强局部重训已完成：**[指导书边界与方案](docs/任务1至7+策略增强与局部重训方案_2026-11-11.md)、[结果](docs/experiments/20261111_strategy/README.md)、[结论与后续](docs/任务1至7+策略增强重训结论_2026-11-11.md)。正式960fit、96模型全部恢复核验通过；内层所选h1残差模型在2002/2003的RMSE较同协议普通LSTM下降27.07%/6.45%，但多输出seed/远期h不稳、2003 h7选择流程失效，不能称整套升级。63项回归测试通过，无2004新成绩。[记忆诊断](docs/diagnostics/20261111_memory/README.md)为36模型、0fit。增强单列，不替代课程必做项。
- **新受控实验已完成：**[方案](docs/任务1至7+受控窗口状态与损失实验方案_2026-11-11.md)、[完整结果](docs/experiments/20261111_controlled/README.md)、[结论与后续](docs/任务1至7+受控实验结论与后续_2026-11-11.md)。756fit、108模型全部独立核验；仅2002—2003历史评价。90日与状态输入未获得跨年整体稳定收益，h7的sqrt有局部信号但证据不足，不替换正式交付；无新的2004成绩。
- **此前全面复核：**[全面诊断与下一轮调整方案](docs/diagnostics/20261111_full_review/README.md)。已重新核对教师原件、数据对齐、历史学习曲线和融合；主要问题不是统一少训，而是跨期选参稳定性、输入状态表达及集成对照。其后已依次完成上方受控实验、局部策略与直接最终对照；该诊断本身不含新2004模型成绩，最新2004成绩见“直接残差扩展与最终对照”。
- [任务要求与完整训练方案](docs/任务1至7+训练方案与要求_2026-11-11.md)：明确七项验收内容、历史验证边界、候选与实验预算。
- [已完成的改进训练结果](docs/experiments/20261111_gpu_lag_residual/任务1至7+GPU改进训练结果_2026-11-11.md)：872次fit、三seed、独立核验通过；LSTM/Mean部分改善，Stacking不稳定，多数配置未通过开发准入，暂不替换原Word。
- [继续训练方案](docs/任务1至7+继续训练方案_2026-11-11.md)与[继续训练结果](docs/experiments/20261111_lognormal/任务1至7+继续训练结果_2026-11-11.md)：另完成728次fit，对照log-MSE与异方差lognormal；整体退化，不采用、不覆盖前轮结果。
- 首轮GPU流程入口为 `python code/train_runoff.py`，支持 `--device auto/cpu/cuda`；最新直接对照入口为 `python -m runoff_training.direct_train`（historical/final分阶段），上一轮策略入口 `python -m runoff_training.strategy_train` 保留；模块入口需先设 `$env:PYTHONPATH = 'code'`，使用方式见[代码说明](code/README.md)。这些独立入口不会调用下文原始流程的热启动，也不会覆盖现有Word报告。
- 新实验代码集中在 `code/runoff_training/`；历史 `acceptance3*.py` 保留，以免破坏旧模型恢复和训练来源。
- 新训练不能事先保证提升；正式结果以独立run中的 `manifest.json`、`verification.json`、`metrics.csv` 为准，smoke不能当作正式结果。

```powershell
python -m unittest discover -s code -p test_runoff_training.py -v
python code/train_runoff.py --run-id smoke_20261111_gpu_lag_residual --device cuda --smoke
python code/train_runoff.py --run-id formal_20261111_gpu_lag_residual --device cuda
# 中断后须使用相同数据、源码和设备
python code/train_runoff.py --run-id formal_20261111_gpu_lag_residual --device cuda --resume
```

本机实测：Python 3.12.3、PyTorch 2.6.0+cu126、RTX 4060 Laptop GPU（8GB，CUDA可用）、scikit-learn 1.5.1、NumPy 1.26.4、pandas 2.2.2。既有环境已满足新训练需求，不必为了训练重装TensorFlow或整套依赖。CUDA版PyTorch须按本机驱动选择安装方式，CPU环境使用 `--device cpu`。

## 历史共享输出流程（不作为本轮训练入口）

当前源码中的 `models.py` 已使用 PyTorch，`run_all.py` 实际调用 `experiment_v6.py`；原先这里标注的TensorFlow环境与当前代码不符，旧版本号不再作为已验证依赖推荐。当前独立训练使用上方实测环境，无需安装TensorFlow。

这套历史流程会写共享 `output/` 并生成根目录报告，部分历史入口支持热启动；其交叉验证、特征选择和早停协议也不同于当前独立run。以下命令仅供定位旧流程，**不要用来替代当前改进训练或覆盖正式验收材料**。历史耗时不是本机本轮实测承诺。

### 历史一键命令

```bash
cd code
python run_all.py              # 完整运行（含全部 GridSearchCV，约 1.5~2 小时）
python run_all.py --fast       # 小规模快速验证（缩小网格与训练轮数，约 10 分钟）
python run_all.py --skip-model # 跳过建模，仅基于已有结果重绘图表与报告
```

## 第三次验收现有正式交付（11月1日结果，11月11日排版整理）

- [第3次修改版报告（Word）](验收/Acceptance%203/任务1至7+第三次阶段验收报告_第3次修改版.docx)
- [第3次修改版报告（Markdown）](验收/Acceptance%203/任务1至7+第三次阶段验收报告_第3次修改版.md)
- [教师第三次验收说明](第三次阶段验收说明（11月5日%20周四）.docx)

最新版已将正式报告和附件放到 `验收/Acceptance 3/` 顶层，旧ZIP已由上游移除，不再列出失效下载链接。本版只包含已完成实验及结果，不包含本轮训练方案。直接策略保留第二轮冻结结果，多输出采用第三轮冻结结果；2004年为回溯评估，不是独立盲测。“第三次验收”“第三轮训练”“第3次修改版”分别是验收阶段、实验轮次和文档版本，不应混为一谈。

训练实现及测试位于 `code/acceptance3*.py` 与 `code/test_acceptance3*.py`，报告生成与复核代码位于 `验收/report code/report3/`。第三次验收历史正式训练记录使用 PyTorch 2.6.0+cpu；本次GPU实验使用上方CUDA环境，当前历史一键流程源码中的LSTM也已是PyTorch，不能再按旧TensorFlow说明安装。训练包版本见各自报告来源附件。Git仅归档最新交付、相关代码和测试，体积较大的本地 `output/acceptance3_runs/` 模型及运行缓存未上传；当前归档不能直接恢复未上传的模型权重。

## 阶段验收交付

每次阶段验收必须依据教师原文完成；说明原件目前分散保存在根目录与验收目录，并非全部位于教师资料目录。详细规则见 [AGENT.MD](AGENT.MD) 和 [验收/acceptance.md](验收/acceptance.md)。

- 第一次：[教师说明](验收/Acceptance%201/第一次阶段验收说明（10月8日%20周四%29.docx)、[原验收汇报](验收/Acceptance%201/机器学习课程实践_第一次阶段验收汇报.docx)、[根目录修订版v2](机器学习课程实践_第一次阶段验收汇报_修订版v2.docx)。两个汇报版本均保留，不互相覆盖。
- 第二次：[教师说明](第二次阶段验收说明（10月22日%20周四）%20.docx)、[正式报告](验收/Acceptance%202/任务1至7+第二次阶段验收报告.docx)。
- 第三次：见上方现有正式交付入口。
- 最终课程报告与个人思政报告：使用 `实验报告模版/` 中教师模板，不能只提交第三次阶段报告替代。

交付第 `x` 次验收时：

1. 在实际的 `验收/Acceptance X/`（X为1、2、3）中存放对应报告及其直接引用的图片、CSV和其他附件；
2. 报告可使用 `.md`、`.html` 或 `.docx`，格式以教师要求为准；
3. 报告中的每个图片、CSV、代码或其他验收附件必须使用可直接打开的相对链接；
4. `Acceptance X/` 只放最终交付物，不放报告生成源码、草稿、缓存或训练日志；历史已被报告引用的程序快照属于交付附件，不因目录整理而删除；
5. 报告生成代码放在 `验收/report code/reportx/`。最终图表、CSV和报告输出或复制到对应的 `Acceptance X/`，不能只留在代码目录。

提交前应确认：报告逐项回应了阶段验收说明、所有链接目标真实存在且可打开、结论能追溯到实际实验结果。

## 读取教师资料

编写或修改验收报告前，先读取 `Teacher Resources & Requirements/` 中与本次验收有关的资料。对于 PDF、Word、PowerPoint、Excel 或其他不便直接阅读的文件，可使用已安装的 MarkItDown 转为 Markdown 以便提取要求：

```powershell
conda run -n markitdown markitdown "C:\文件路径\源文件.pdf" -o "C:\文件路径\提取结果.md"
```

- 不需要激活或切换 Conda 环境。
- 当前仅安装 MarkItDown 本体及 `[all]` 格式依赖，未安装 OCR 或第三方插件；扫描件和图片型 PDF 可能无法提取文字。
- 提取出的临时 Markdown 仅用于阅读和整理要求，不应放入 `验收/acceptance x/`。

## 原始流程的模型热启动（不适用于新改进训练入口）

`output/models/` 保存了上一次训练的最优模型。再次运行 `run_all.py` 时，
程序会自动检测并加载已保存模型，在其基础上**继续训练**（而非从零开始）：

- ANN：`warm_start=True`，继续迭代 500 轮；
- RF：`n_estimators += 200`（在原树数上增加）；
- LSTM：加载已保存权重，继续训练 30 个 epochs；
- 训练完成后再次保存更新后的模型，供下一次热启动使用。

如需完全从零训练，删除 `output/models/` 目录下的文件即可。

## 实验设计要点

- **数据划分**：前 4 年（2000-2003）训练，第 5 年（2004）测试，严格按时间顺序，不可打乱。
- **特征工程**：四季 one-hot、月份/日序三角编码、滞后特征（lag 1/2/3/7/14/30）、
  滑动窗口统计（mean/std/max）、Min-Max 与 Z-score 归一化、皮尔逊相关系数法与互信息法特征选择。
- **目标变换**：径流强右偏（偏度 4.96），取对数后建模（预测经 exp 回变换恒为正）。
- **多步预测**：
  - 多步直接预测：每个预见期 h=1..7 一个模型，共 7 个；
  - 多输出预测：一个模型同时输出 7 天。
- **个体模型**：ANN（MLPRegressor）、随机森林、LSTM；均用 GridSearchCV + TimeSeriesSplit 寻优。
- **集成学习**：简单平均 + Stacking（Ridge 元学习器，基于折外预测 OOF 训练）。
- **模型总数**：个体 3×8=24，集成 2×8=16，合计 40。
- **评价指标**：RMSE、MAE、NSE、R²、PBIAS、MAPE（另附对数尺度 NSE_log）。

## 防数据泄漏的三处关键设计

1. 归一化与特征选择统计量**只在训练集上估计**；
2. 交叉验证用 **TimeSeriesSplit**（前向链式），不用普通 K 折；
3. Stacking 元学习器用**折外预测（OOF）**训练，不用训练集拟合值。
