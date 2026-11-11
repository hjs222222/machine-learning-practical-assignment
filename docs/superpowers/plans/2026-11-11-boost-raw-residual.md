# 原流量增量提升树执行计划

> 使用executing-plans在当前codex/runoff-clean分支内联执行，逐任务核验和Git保存。用户已明确同意本具体实验，不另请求相同设计确认、不创建子代理。

**Goal:** 补齐原流量增量目标对照，检验保留原点流量基准且不压缩量纲是否改善h1。

**Architecture:** code/boosting.py新增RawResidualHistGradientBoostingRegressor，保留原sqrt类；research_h1_boost.py增加raw_residual目标；verify_h1_boost.py手算父类输出解码。既有输出、默认目标与冻结依赖不动。

**Tech Stack:** 已安装Python3.13/sklearn1.6/numpy/pandas/joblib/threadpoolctl，不新增依赖。

## 固定边界

仅教师CSV、2000—2003年、相同91因果输入/七日可用性掩码/各358年度原点。seed42、lr0.05、L2=1、200轮、关闭早停和warm_start；4网格叶数7/15×最小叶样本10/30，内2折gap7，每年9次，共27拟合。CV原始流量R²，解码max(Q(t)+delta,0)。无2004、无权重搜索、无增网格，保留全部结果。目标影响选参，不声称相同最终参数的纯因果对照。

## Task 1：先失败测试、再最小实现

Files: tests/test_h1_boost.py、code/boosting.py、code/research_h1_boost.py、code/verify_h1_boost.py。

- [x] 将现有真实小模型fixture目标参数化列表加入raw_residual；新增手工父类对照、clone/joblib、负输入/负y与早停拒绝测试。fixture的y使用max(x[:,0]+10*cos(index),0)保证合法非负流量，对照模型fit(x,y-x[:,0])，包装预测逐项等于max(parent.predict(x)+x[:,0],0)。另以零目标拟合、零原点输入验证非负截断。
- [x] 运行`D:\anaconda3\python.exe -m pytest tests/test_h1_boost.py -q`，2失败/5通过，失败明确是缺新类/缺目标选项。
- [x] 新类fit执行`super().fit(X,truth-current,sample_weight=sample_weight)`，predict执行`np.maximum(super().predict(X)+current,0)`。复用_current_flow与原目标验证语义，禁止early_stopping非False。运行器改成明确目标到类/描述映射，默认raw不变；新增CLI枚举；验证器对raw_residual调用父类未解码输出加回X[:,0]并逐项核对。
- [x] 重跑上述测试及tests/test_selection_search.py、tests/test_ann_boost_average.py，15通过（38.96秒），Git56462e7预存实现。保留两种旧目标和修改fixture2004不影响输入SHA/预测的回归。未重跑此前548项全量测试，不将其当作本轮后的全量成绩。

## Task 2：27次真实拟合与独立核验

- [x] 确认输出不存在后运行`D:\anaconda3\python.exe code/research_h1_boost.py --target raw_residual --output output/teacher_selection_20261109/h1_boost_raw_residual_seed42`。27拟合完成，拟合前协议落盘；CV/3权重/零警告/各200轮/2148预测（包含持久性）完整保存。
- [x] 运行`D:\anaconda3\python.exe code/verify_h1_boost.py output/teacher_selection_20261109/h1_boost_raw_residual_seed42`。数据/源码/权重SHA、9个gap记录、4候选的CV选择、原始次日真值与6行指标、3模型恢复和父类输出手算解码均通过，12行含旧ANN/RF的配对表已保存。
- [x] 三种提升树保存预测的1074个日期/真值完全相同，独立重算全部9行指标。raw_residual的2001/02/03 R²0.649289/0.587549/0.384160，均低于持久性及sqrt残差，不晋级、不搜索或打开2004。2001/02选7叶/最小叶30，2003选15叶/最小叶10；不是同一最终参数的纯目标因果比较。
- [x] 更新本计划、三次验收研究记录和对话交接；强制跟踪被忽略的本轮结果及3份模型权重，Git存档。旧Word/正式ZIP原SHA核对不变，旧258432b包不冒称包含新代码。

自查：所有设计范围均覆盖；个人信息仍待填；目标是实测改善而非保证0.8。若失败保留证据，不替换较强ANN或宣称测试提高。
