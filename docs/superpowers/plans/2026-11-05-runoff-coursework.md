# 径流预测课程实践 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不引入外部数据和不泄漏未来信息的前提下，生成可复现的径流预测实验、三次验收证据和最终可编辑 Word 实验报告。

**Architecture:** 一个 Python 命令从原始 CSV 构造严格因果的特征与切分，训练/评价 40 个教师要求模型并导出统一指标。报告构建器只读取已落盘的结果文件，分别复制每次验收需要的附件并填充最终模板，报告不直接触发训练。

**Tech Stack:** Python 3.13、pandas、numpy、scikit-learn 1.6.1、PyTorch 2.7.1、matplotlib、python-docx、pytest。

## 执行结论 2026年11月5日

Task 0--6的实现与本地交付核验已完成，精度目标未达成：本次同源冻结运行最高h=1测试R²为0.5103。24项自动测试通过，70行指标和25130条逐日预测独立复算通过；三份Markdown共68个附件链接通过；最终Word 17页已逐页检查。个人信息和真实三人分工待填写，用户下载及教学平台提交未确认。详细证据和交付路径以`docs/2026-11-05-对话与决策交接.md`为准。

下文保留最初的步骤与拟定提交消息。实际采用更少的共享接口和测试文件，避免重复：

| 原计划位置 | 实际实现与验证 |
|---|---|
| Task 0--2 | `config.new_run_dir(data_path,destination,config)`、`data.load_raw()`、`data.build_dataset()`、`features.build_features()`；`tests/test_contract.py`、`tests/test_analysis.py` |
| Task 3--4 | `models.RunoffRegressor(family=...).fit/predict`统一直接/原生多输出；`simple_average`、`fit_stacking`；`tests/test_models.py` |
| Task 5 | `experiment.develop/evaluate`、嵌套OOF和冻结哈希；`tests/test_experiment.py`；`report code/final/verify_run.py`实际文件复算 |
| Task 6 | `build_final_report.py`、`analyse_run.py`及Word原生渲染替代；`tests/test_delivery.py`与逐页视觉检查 |

LSTM采用30/70轮候选的purged GridSearchCV，替代原计划中未明确拆分方式的早停；该变更在正式运行前冻结。代码ZIP包含报告重建脚本及教师模板转换脚本，模板本体需使用教师提供资料。没有复制旧项目内容。

## Global Constraints

- 原始数据仅为 `Teacher Resources & Requirements/01047000.csv`，不修改该文件且不接入外部数据。
- `C:\Users\何\Desktop\机器学习课程实践` 未达到本次目标要求，只可用于风险警戒；不得直接复用其中的代码、结果、指标、图表、报告文字或运行产物。
- 旧项目警戒事实：其最高已记录直接 h=1 测试 R² 为 0.5217，未达 0.8；直接和多输出来自不同冻结轮次；2004 已被反复查看；自动热启动和多候选数据路径妨碍从零复现；早期版本曾有预处理/时序交叉验证/随机打乱风险。
- 训练期为 2000--2003，测试期为 2004；不得随机打乱。
- 本项目只有 `code/run_all.py` 一个正式入口；只接受唯一原始数据路径；默认拒绝加载任何历史模型或输出；一次正式运行必须产出 24 个基模型、16 个集成模型及统一指标表。
- 缩放、互信息选择、网格搜索和 Stacking 元学习器均只能使用训练期。
- 多输出模型不得使用 `MultiOutputRegressor`。
- 全部模型输出 R²、RMSE、MAE；h=1 追求高 R²，h=2--h=7 允许下降。
- 预测原点为 t：允许 t 当日已观测字段和历史值，禁止 t+1 之后字段；所有正式指标统一使用 2004-01-01 至 2004-12-24 的 359 个预测原点。
- 时序交叉验证使用 7 天 `gap`；正式测试只在冻结配置后运行一次。未达 h=1 R²≈0.8 时如实报告，不以 2004 年测试结果调参。
- 验收目录只存可直接交付的报告和附件；生成脚本放入 `report code/`。
- 三份验收 Markdown 从同一冻结正式运行按任务筛出证据；最终报告保留未提供的个人封面信息及三人分工为“待填写”，不得捏造。
- 最终 Word 不粘贴代码；附录改为代码提交说明。最终代码 ZIP 仅含源码、测试、依赖、运行说明和哈希清单，不含教师数据、权重、缓存、旧项目或大体积中间输出。
- 先在训练期执行计时试跑，再冻结候选网格和运行预算；GPU 不可用或超时仅缩小 LSTM 候选网格，不删减 40 模型。
- 初始化仅本地 Git；每个独立阶段完成后提交，不推送，且不提交原始数据副本、模型权重或缓存。

---

## Planned file map

| Path | Responsibility |
|---|---|
| `code/config.py` | 单一数据路径、时间边界、随机种子、预测范围和输出路径。 |
| `code/data.py` | CSV 校验、日期切分、无泄漏缩放及序列样本索引。 |
| `code/features.py` | 仅过去/当日信息的日历、滞后、滚动特征与训练期互信息选择。 |
| `code/models.py` | MLP、RF、原生多输出 PyTorch LSTM 的训练和预测接口。 |
| `code/experiment.py` | 直接/多输出训练、训练期时序 CV、OOF Stacking 和结果落盘。 |
| `code/evaluation.py` | R²、RMSE、MAE、逐预测步指标和图表。 |
| `code/run_all.py` | 唯一复现入口；依次调用特征、模型、评价与导出。 |
| `tests/` | 小型合成数据测试，覆盖因果性、时间边界、输出形状和 OOF 边界。 |
| `report code/` | 每次验收及最终报告构建器。 |
| `验收/acceptance 1..3/` | 三次可交付阶段材料。 |

### Task 0: 建立从零运行与旧项目警戒线

**Files:**
- Create: `README.md`, `tests/test_run_contract.py`
- Produces: `new_run_dir(config) -> Path`，其中运行目录包含 `manifest.json`、唯一 CSV 的 SHA-256、配置哈希、随机种子和模型计数。

- [x] 写失败测试：数据路径必须精确等于 `Teacher Resources & Requirements/01047000.csv`；传入旧目录路径、历史模型路径或已存在的运行目录必须抛出 `ValueError`。
- [x] 运行 `pytest tests/test_run_contract.py -v`，确认接口不存在而失败。
- [x] 实现唯一入口和运行清单：每次正式运行创建新目录，记录但不读取旧输出；命令不提供热启动、恢复权重、多路径回退或“跳过模型并复用已有结果”选项。
- [x] 在 README 写明旧项目的六项警戒事实及本项目对应防线：未达 h=1 目标、混合冻结轮次、测试期反复暴露、热启动/多路径、早期泄漏风险、入口碎片化。
- [x] 重新运行 `pytest tests/test_run_contract.py -v`，确认通过。
- [x] 初始化 Git 后提交本任务：`git commit -m "chore: establish clean run contract"`。

### Task 1: 建立可重复且可验证的数据边界

**Files:**
- Create: `code/config.py`, `code/data.py`, `tests/test_data.py`
- Produces: `load_raw(path) -> pd.DataFrame`、`split_by_year(df) -> (train, test)`、`fit_transform_train_test(train, test, columns) -> (X_train, X_test)`。

- [x] 写失败测试：CSV 必须有 1,827 行、九列、无缺失；训练最大日期为 2003-12-31，测试最小日期为 2004-01-01；缩放器只从训练行计算。
- [x] 运行 `pytest tests/test_data.py -v`，确认缺少模块或接口导致失败。
- [x] 实现最小加载、日期解析和断言；`Swe` 仅在数据概览中保留，建模特征排除。
- [x] 重新运行该测试并确认通过。
- [x] 提交本任务（若仓库初始化后）：`git commit -m "feat: add leakage-safe data boundary"`。

### Task 2: 实现因果特征工程与第一验收证据

**Files:**
- Create: `code/features.py`, `code/data_analysis.py`, `tests/test_features.py`
- Produces: `build_causal_features(df, horizons) -> (X, targets, feature_catalog)`，其中任一行特征的最大源日期不晚于该行预测基准日。

- [x] 写失败测试：`lag_1` 在 t 行等于 t-1；滚动窗口不含 t+1；7 个目标分别是 t+1 至 t+7；特征选择拟合行均来自训练期。
- [x] 运行 `pytest tests/test_features.py -v`，确认测试因函数缺失失败。
- [x] 实现季度/月编码、对 `Discharge`、`Prcp`、`Tmax`、`Tmin` 等非恒定字段的滞后和过去窗口统计；保存特征字典、互信息排名、标准化参数和所选列。
- [x] 实现每个非日期原始字段的时间序列图、直方图、箱型图，以及含径流滞后和因子滞后的 Pearson 热力图。
- [x] 重新运行 `pytest tests/test_features.py -v`，并人工检查首尾索引不越界。
- [x] 将第一验收所需图、CSV、特征字典和代码快照标记为待正式运行筛选的候选附件；此任务不生成阶段报告。
- [x] 提交本任务：`git commit -m "feat: add causal feature pipeline"`。

### Task 3: 实现 ANN、RF 的训练期开发流程

**Files:**
- Create: `code/models.py`, `code/experiment.py`, `code/evaluation.py`, `tests/test_models.py`
- Produces: `fit_direct(model_name, X_train, y_train, h)`、`fit_multi(model_name, X_train, Y_train)` 和统一的 `(n_samples, 7)` 预测矩阵。

- [x] 写失败测试：直接策略为 7 个独立目标；ANN/RF 多输出产生 7 列且没有 `MultiOutputRegressor`；开发评分表含模型、策略、h、R2、RMSE、MAE。
- [x] 运行 `pytest tests/test_models.py -v`，确认未实现接口失败。
- [x] 为 ANN 和 RF 用 `TimeSeriesSplit` 搜索有限、预先声明的参数网格；在训练期内记录每组参数和交叉验证分数。
- [x] 在训练期滚动验证中训练 7 个直接模型及 1 个原生多输出模型；此任务不得读取或生成 2004 测试指标。
- [x] 重新运行测试；检查模型计数为 ANN 8 + RF 8，开发指标覆盖 h=1..7。
- [x] 提交本任务：`git commit -m "feat: add ANN RF development workflow"`。

### Task 4: 实现 LSTM 与两类集成的训练期开发流程

**Files:**
- Modify: `code/models.py`, `code/experiment.py`, `code/evaluation.py`
- Create: `tests/test_lstm_and_ensembles.py`
- Produces: `fit_lstm_direct(...)`、`fit_lstm_multi(...)`、`simple_average(predictions)`、`fit_oof_stacking(train_predictions, train_targets)`。

- [x] 写失败测试：LSTM 输入为三维序列张量、原生多输出头输出 7 个值；简单平均逐元素平均；Stacking 拟合行是训练期 OOF 行且无测试标签。
- [x] 运行 `pytest tests/test_lstm_and_ensembles.py -v`，确认失败。
- [x] 实现带固定随机种子、早停和训练期参数选择的 PyTorch LSTM；实现 7 个直接与 1 个多输出模型。
- [x] 用三个基模型的同一日期预测构建简单平均；仅用滚动 OOF 预测拟合每步 Ridge Stacking，并只在训练期开发折内推理。
- [x] 导出训练期 LSTM 参数搜索表和 OOF 边界说明；此任务不得读取或生成 2004 测试指标。
- [x] 重新运行测试；检查总数为 24 个基模型、16 个集成模型，且源码不导入 `MultiOutputRegressor`。
- [x] 提交本任务：`git commit -m "feat: add LSTM and ensemble development workflow"`。

### Task 5: 以训练期验证选择合法的 h=1 改进

**Files:**
- Modify: `code/experiment.py`, `code/run_all.py`
- Create: `tests/test_selection_protocol.py`
- Produces: `output/selection_protocol.json`、`output/development_scores.csv`、`output/final_metrics.csv`。

- [x] 写失败测试：开发期分数来自 2000--2003 的滚动验证；选择配置与测试指标文件分离；测试集不出现在参数候选比较表。
- [x] 运行 `pytest tests/test_selection_protocol.py -v`，确认失败。
- [x] 在不读取 2004 测试集的短计时试跑中确定运行预算；随后冻结有限、可解释的序列长度、滞后/窗口组合、模型网格和随机种子，并按训练期 h=1 验证 R²选定最终配置。
- [x] 用 7 天 gap 的时序切分复核全部开发候选；固定配置后一次性完成 24 个基模型、16 个集成模型的 2004 正式评价，所有 h 使用相同的 359 个预测原点。
- [x] 若 h=1 R² 未接近 0.8，保留真实分数并在调试记录中写明训练期证据和误差来源，不改动测试集或数据。
- [x] 从这一份正式运行生成三份 Markdown：`验收/acceptance 1/任务1+数据分析与特征工程.md`、`验收/acceptance 2/任务2+ANN与随机森林径流预测.md`、`验收/acceptance 3/任务3+LSTM与集成学习径流预测.md`，并各自复制仅本报告引用的附件。
- [x] 重新运行协议测试和全量测试；保存随机种子、候选、选择依据、最终指标及预测。
- [x] 提交本任务：`git commit -m "feat: record leakage-safe model selection"`。

### Task 6: 生成最终实验报告并进行版面和链接验证

**Files:**
- Create: `report code/final/build_final_report.py`, `report code/final/report_content.md`, `tests/test_report_artifacts.py`
- Produces: `最终提交/机器学习课程实践实验报告.docx` 和 `最终提交/机器学习课程实践代码.zip`。

- [x] 写失败测试：报告引用的每个图表/CSV 均存在；六个模板章节与“代码提交说明”均有内容；封面身份字段和三人分工仍为“待填写”；代码 ZIP 不含教师数据、权重或缓存。
- [x] 运行 `pytest tests/test_report_artifacts.py -v`，确认失败。
- [x] 只读取 `output/` 中的真实产物，按模板填充“一至六”；将附录改为代码提交说明，记录遇到的实际调试问题及解决方式。
- [x] 从 `.doc` 模板生成可编辑 `.docx`，不改变模板主体章节顺序；生成只含 `code/`、`tests/`、`requirements.txt`、运行说明、配置/数据哈希清单的代码 ZIP。
- [x] 使用文档渲染器导出每页 PNG，逐页检查标题、表格、图像、中文字体、页码和代码提交说明无截断；有问题则修改并重新渲染。
- [x] 重新运行报告附件测试，并提交：`git commit -m "docs: add final coursework report"`。

## Self-review

- 覆盖性：任务 1--2 准备第一次验收 8 项；任务 3 准备第二次验收 7 项；任务 4 准备第三次验收 7 项；任务 5 从同一冻结运行生成三份验收 Markdown；任务 6 对应最终 `.doc` 模板六章和代码提交说明。
- 反泄漏：任务 1、2、4、5 分别测试数据切分、特征来源、OOF 元学习器和选择协议。
- 范围：未加入外部数据、网页、额外模型或测试集调优。
- 前置缺口：用户尚未提供封面身份资料及真实三人分工；最终报告必须保留待填写项。
