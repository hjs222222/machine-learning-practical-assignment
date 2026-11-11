# 三路线有限批次执行计划

**Goal:** 实施用户已确认的三路线57拟合批次，按固定规则筛选唯一候选；测试0.60是阶段门槛而非验证成功声明。

**Architecture:** 新exact_boosting.py封装普通GBDT的sqrt残差；research_h1_gbdt_batch.py按原内层时序折手工展开固定网格并逐折计算历史政策，缓存完全相同的拟合；research_h1_state_gate.py只读历史OOF训练固定误差树；verify_h1_generalization_batch.py独立复算；research_h1_batch_selection.py按同716日规则筛选。全部新文件，不修改旧模型/冻结依赖。

**Tech Stack:** 已安装Python3.13/sklearn1.6.1/pandas/numpy/joblib，无新依赖。内联执行，不使用子代理。

**Constraints:** 完全继承2026-11-11-h1-generalization-batch-design.md：教师CSV、截止2003、91输入、三年度同原点、4网格/200树/seed42；all27/recent730新增1/half_life365新增27/状态2。没有新2004授权，不删峰/改指标/增候选。

## Task 1：普通GBDT与历史政策

Files: code/exact_boosting.py、code/research_h1_gbdt_batch.py、tests/test_h1_generalization_batch.py。

- [x] 先写失败测试：普通GBDT手工sqrt残差拟合/解码一致、clone/joblib、负值/早停拒绝；history_rows(dates,policy)返回掩码与按本fit最新日计算的单位均值权重，730天边界严格age<730；半衰期365为0.5比例。
- [x] 运行pytest确认缺新模块失败，再实现最小类及运行器。模型保持200树/lr.05/max_depthNone/n_iter_no_changeNone；每个内层fit和最终fit都单独调用政策函数。缓存键为参数+实际训练索引+实际权重，输入数据在运行内固定；不缓存不同科学干预。
- [x] 软件fixture真实小GBDT、2树/单网格，验证三政策实际9/1/9=19次新拟合、时间gap、相同原点、拒绝覆盖、修改2004不改变输入SHA及全部预测。人工数组不进入课程结果。
- [x] 相关测试通过后Git预存，运行`python code/research_h1_gbdt_batch.py --output output/teacher_selection_20261109/h1_generalization_batch`，逐步保存protocol/搜索日志/fit记录/模型/预测/警告/complete。

## Task 2：固定水文状态组合

Files: code/research_h1_state_gate.py、上述测试文件。

- [x] 先失败测试状态公式、原点因果性、预测日期/真值错位拒绝、只用先前年份OOF、固定并列顺序ANN/RF/persistence。
- [x] 实现固定六状态输入与深2叶60的多输出误差树，2002/03各1次fit。每次损失标签=(三成员预测-实际次日流量)^2，仅历史OOF参与fit；保存树、选择成员、估计损失和控制预测，三家族必做Stacking不变。
- [x] 软件测试通过后Git预存，运行`python code/research_h1_state_gate.py --output output/teacher_selection_20261109/h1_generalization_batch/state_gate`；基预测只读，不重训ANN/RF。

## Task 3：独立核验与固定筛选

Files: code/verify_h1_generalization_batch.py、code/research_h1_batch_selection.py、上述测试文件、研究/交接MD。

- [x] 核验器从教师截止2003的原始数据重建91输入、六状态、次日真值和gap；独立计算窗口/权重及CV均值/选择；恢复树用父类输出手算sqrt解码，核对保存预测/指标/SHA。核验器不调用训练fit。
- [x] 状态树独立读取tree_节点手工遍历并对三损失argmin，核对逐日选择及预测、训练标签/上下文、全局历史赢家和三成员固定均值。没有第二次训练或测试评价。
- [x] 四候选统一2002/03的716日，合并R²主排序；相对ANN合并R²+0.01、2003≥ANN-.005、2002≥ANN-.01、MAE≤ANN×1.01。先测试门槛边界/拒绝错位，再计算真实结果。无人合格不放宽；合格只保留第一位，冻结训练另做记录，2004仍需单独确认。本轮无人合格，不进行冻结或2004评价。
- [x] 更新所有任务状态和研究/交接MD，保存源码/测试/全部结果及权重至Git。原正式Word/ZIP SHA保持；后续研究代码快照若本轮未生成，明确待补，不冒称旧包已含新实现。目标未达不标记完成。
