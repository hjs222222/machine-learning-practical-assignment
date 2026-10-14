# 基于机器学习的径流预测

《机器学习课程实践》实验 —— USGS 01047000 流域逐日径流多步预测。

## 目录结构

```
├── 01047000.csv                  # 数据：2000-2004 逐日，9 字段（Discharge + 7 个 Daymet 气象因子 + 日期）
├── code/                         # 全部源代码
│   ├── config.py                 # 全局配置（路径、变量对照、特征与建模参数）
│   ├── utils_plot.py             # 绘图公共工具（中文字体、配色、保存）
│   ├── data_analysis.py          # 第一部分：数据分析（统计 + 可视化 + 相关性）
│   ├── feature_engineering.py    # 第二部分：特征工程（编码/滞后/窗口/归一化/特征选择）
│   ├── models.py                 # 模型定义（ANN/RF/LSTM）+ GridSearchCV
│   ├── experiment.py             # 第三部分：多步直接/多输出策略、集成、评价
│   ├── results_analysis.py       # 第四部分：结果可视化与误差分析
│   ├── make_report.py            # 报告生成（封面 + 第一~三章）
│   ├── report_sections.py        # 报告生成（第四~六章 + 附录）
│   └── run_all.py                # 一键运行全部流程
├── output/
│   ├── figures/                  # 全部统计图表（PNG）
│   ├── results/                  # 全部结果表（CSV）与预测文件（NPZ）
│   └── run_full.log              # 完整运行日志
├── Teacher Resources & Requirements/ # 教师提供的阶段验收说明、模板与资料
├── 验收/
│   ├── acceptance x/              # 第 x 次验收可直接交付的最终材料
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

## 运行环境

推荐使用独立虚拟环境（避免与系统 Python 包冲突；本机已创建于 D 盘）：

```bash
# 首次创建（可选，已在 D:\ml_venv 创建好）
python -m venv D:\ml_venv

# 激活并使用
D:\ml_venv\Scripts\python.exe -m pip install numpy scipy scikit-learn pandas \
    matplotlib tensorflow-cpu python-docx openpyxl joblib
```

- Windows 11 / Python 3.12+
- 已测试版本：numpy 2.5、scipy 1.18、scikit-learn 1.9、pandas 3.0、
  matplotlib 3.11、tensorflow-cpu 2.21 / keras 3.15
- 运行命令把下面的 `python` 替换为 `D:\ml_venv\Scripts\python.exe`

## 一键运行

```bash
cd code
python run_all.py              # 完整运行（含全部 GridSearchCV，约 1.5~2 小时）
python run_all.py --fast       # 小规模快速验证（缩小网格与训练轮数，约 10 分钟）
python run_all.py --skip-model # 跳过建模，仅基于已有结果重绘图表与报告
```

## 阶段验收交付

每次阶段验收必须严格依据 `Teacher Resources & Requirements/` 中的《第 x 次阶段验收说明》完成；详细协作规则见 [AGENT.MD](AGENT.MD) 和 [验收/acceptance.md](验收/acceptance.md)。

交付第 `x` 次验收时：

1. 在 `验收/acceptance x/` 中存放最终验收报告及其直接引用的图片、CSV 和其他附件；
2. 报告可使用 `.md`、`.html` 或 `.docx`，格式以教师要求为准；
3. 报告中的每个图片、CSV、代码或其他验收附件必须使用可直接打开的相对链接；
4. `acceptance x/` 只保留可直接提交的最终材料，不放源代码、草稿、缓存、临时文件或运行日志；
5. 生成报告的代码放在 `验收/report code/reportx/`。代码产生的最终图表、CSV 和报告必须输出或复制到对应的 `acceptance x/`，不能只留在代码目录。

提交前应确认：报告逐项回应了阶段验收说明、所有链接目标真实存在且可打开、结论能追溯到实际实验结果。

## 读取教师资料

编写或修改验收报告前，先读取 `Teacher Resources & Requirements/` 中与本次验收有关的资料。对于 PDF、Word、PowerPoint、Excel 或其他不便直接阅读的文件，可使用已安装的 MarkItDown 转为 Markdown 以便提取要求：

```powershell
conda run -n markitdown markitdown "C:\文件路径\源文件.pdf" -o "C:\文件路径\提取结果.md"
```

- 不需要激活或切换 Conda 环境。
- 当前仅安装 MarkItDown 本体及 `[all]` 格式依赖，未安装 OCR 或第三方插件；扫描件和图片型 PDF 可能无法提取文字。
- 提取出的临时 Markdown 仅用于阅读和整理要求，不应放入 `验收/acceptance x/`。

## 模型热启动（在已有最优模型上继续训练）

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
