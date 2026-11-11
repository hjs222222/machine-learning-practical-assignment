# 基于机器学习的径流预测 从零实验

工作目录：`D:\machinelearning`。教师原始 CSV 是唯一数据来源，原始资料保持只读。

## 当前工作：按三次验收要求提升预测精度

2026-11-11：两阶段模型（雨情分类+旱/雨分训）训练期晋级。2001/2002/2003 R²为0.887902/0.864902/0.510596，2002/03合并0.604547（RF基线0.584249，+0.020298），合并MAE比0.9542，五项预注册门槛（含2001稳定性）全部通过，是首个三年同步改善的候选。代价是2001峰值RMSE与峰值偏差变差，改善主要在中低流量段。未冻结、未打开2004。见`docs/三次验收突破实验-2026-11-09.md`第十九轮及`output/teacher_selection_20261109/h1_two_stage_20261111/`。

2026-11-11：三路线普通GBDT、历史窗口/衰减、当日状态选择器的57次训练期拟合已完成；2002/03共716日统一筛选，无候选超过预定ANN门槛，未做新2004评价。第一天测试R²依次0.6/0.7/0.8的目标仍未达到，后六天保留老师必做结果。详见`docs/三次验收突破实验-2026-11-09.md`第十七轮及`output/teacher_selection_20261109/h1_generalization_batch/`。

2026-11-11：用户允许增加仅训练期的梯度提升树补充对照，不替代教师必做模型。原流量目标困难年R²0.385059，固定网格改sqrt残差后升至0.481806，但仍低于既有ANN验证0.549640，未晋级、无新2004评价。新入口`code/research_h1_boost.py --target sqrt_residual --output output/新的训练期目录`；不要复写已存在目录。历史源码核验可用`code/verify_h1_boost.py <旧目录> --source-ref <对应Git提交>`，需要原Git对象，不是重新训练或新测试。

2026-11-10最新结果：加入当天降雨的固定三seed ANN经新授权完成一次2004回溯，R²0.5030792581，未超过正式RF0.5103305386；原无降雨候选A未测试。随后RF单输出与七输出ANN目标一致MI均只做训练期对照，没有稳定晋级证据。研究全部保留，不继续据2004调参。源码包范围及历史脚本使用边界见`docs/研究代码包使用边界-2026-11-10.md`。

2026-11-10用户更新：第一天原始R²是首要指标，后六天只保留教师必做项，不要求同步改善。`--outputs 1`可运行第一天单输出联合搜索；默认7输出，两者使用相同日期掩码，禁止把年度高分当2004成绩。三份验收Word展示版已经生成，MD保留。

2026-11-09：重点仍是第一天，但同时保留七天完整指标和教师规定的全部模型/策略。最新方案与结果见 `docs/三次验收突破实验-2026-11-09.md`，执行计划见 `docs/superpowers/plans/2026-11-09-teacher-feature-selection.md`。中间以MD记录，教师要求的Word/PPT展示版另行整理；最终Word保留，身份与分工按用户要求待填写。尚未完成的交付项不得写成已完成。

训练期联合网格搜索入口（输出目录必须尚不存在）：

```powershell
python code/research_selection_search.py --family RF --output output/新的RF研究目录
python code/research_selection_search.py --family ANN --output output/新的ANN研究目录
python code/research_selection_search.py --family ANN --outputs 1 --seed 42 --output output/新的第一天研究目录
```

上述命令仅在2000—2003内部做嵌套时序验证，不运行2004测试。研究结果不自动覆盖正式报告、模型或代码包。2004的已授权单次ANN回溯已完成，不应通过另起目录规避一次性评价边界。

### 早期一天研究入口

暂不修改最终报告。先阅读`docs/一天预测精度提升记录.md`和`docs/superpowers/plans/2026-11-05-h1-accuracy-research.md`。研究入口`code/research_h1.py`在构造样本前切断2004，逐年验证2001/2002/2003；输出为训练期探索结果，不是新的独立测试成绩。原正式运行和交付文件保持不变。

```powershell
python code/research_h1.py --stage features --output output/新研究/features
python code/research_h1.py --stage targets --previous output/新研究/features --output output/新研究/targets
python code/research_h1.py --stage tune --previous output/新研究/targets --output output/新研究/tune
```

特征阶段默认比较原MI、强制保留近期降雨、全部91列、101列因果派生；显式`--modes compact`可单独测试22列紧凑输入。每一运行目录拒绝覆盖，候选协议在训练前写入。参数扩大前先做输入和目标对照；指标回到原始流量单位，汇总优先最差年R²，再看平均R²，同时检查洪峰误差。

## 旧项目警戒

旧目录 `C:\Users\何\Desktop\机器学习课程实践` 未达到本次目标要求。其已核验的 v4 直接 RF h=1 R²为 0.5217；旧报告混合不同冻结轮次；2004 已被多次查看；主流程自动热启动、数据来源有路径回退；早期版本的调试文字提到预处理与时间切分问题；并行版本入口过多。以上用于警戒，禁止复用旧代码、指标、图表、文字或模型。

本项目使用唯一数据路径、新运行目录、固定随机种子和配置哈希，全部预处理在各训练折拟合。2004 的结果仍应如实称为回溯测试；重建项目并不能使已看过的数据成为新盲测。

## 命令

环境为 Windows、Python 3.13.5。按 `requirements.txt` 安装本次实际依赖；LSTM 会自动使用可用 CUDA，否则使用 CPU。PyTorch 固定 CUDA 11.8 版本；不同设备的浮点结果可能有细微差别。

在项目目录运行：

```powershell
python -m pytest tests -q
python code/run_all.py --benchmark --name 新的计时目录名
python code/run_all.py --name 新的正式目录名
```

正式运行默认从零训练，约含96次网格搜索（每个24个基模型有一个全训练期搜索及3个独立外层OOF内搜索），不读取历史权重。目录已存在时拒绝覆盖。每折重新拟合互信息和Z-score。LSTM轮数通过带7天隔离的CV选择，避免随机早停切分。简单平均严格三模型等权；Stacking只拟合嵌套OOF预测。直接与原生多输出共24个基模型、16个集成模型。

配置冻结后，才正式读取2004标签进行指标评价；评估开始标记禁止在同目录重复打开测试。其他年份/月份、不同时刻的观测不得拼成新数据；不允许擅自删洪水日或生成气象预报。训练原点的最后七天标签均在2003内；测试为359个共同预测原点，t当日观测可作为随后预报输入。

输出 `output/正式目录名/` 含冻结参数、96次搜索详情、折内缩放和MI字典、32个序列化对象（24个基模型+8个Ridge元模型；简单平均无需权重文件）、逐日预测、70行指标、持续性基线和运行来源哈希。

在正式结果已完成后生成图与报告：

```powershell
python "report code/final/analyse_run.py" "output/正式目录名"
python "report code/final/verify_run.py" "output/正式目录名"
python "report code/final/build_final_report.py" "output/正式目录名"
```

Word构建需要教师`.doc`用Word另存的模板副本 `report code/final/template.docx`（教师材料不放在代码ZIP内）。有Microsoft Word时，运行 `powershell -File "report code/final/prepare_template.ps1"` 创建该副本；已存在时拒绝覆盖。附录只写代码提交说明，不粘代码。报告构建脚本使用单独的文档环境；本机实际调用打包的Python运行。`report code/report1..3/` 的脚本可以分别重建对应验收Markdown。

## 交接

最新进度见 `docs/2026-11-05-对话与决策交接.md`。最终 Word 不粘贴源码，代码以独立 ZIP 提交。封面身份与真实三人分工等待用户填写。
