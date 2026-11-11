# 原流量增量提升树执行计划

> 使用executing-plans在当前codex/runoff-clean分支内联执行，逐任务核验和Git保存。用户已明确同意本具体实验，不另请求相同设计确认、不创建子代理。

**Goal:** 补齐原流量增量目标对照，检验保留原点流量基准且不压缩量纲是否改善h1。

**Architecture:** code/boosting.py新增RawResidualHistGradientBoostingRegressor，保留原sqrt类；research_h1_boost.py增加raw_residual目标；verify_h1_boost.py手算父类输出解码。既有输出、默认目标与冻结依赖不动。

**Tech Stack:** 已安装Python3.13/sklearn1.6/numpy/pandas/joblib/threadpoolctl，不新增依赖。

## 固定边界

仅教师CSV、2000—2003年、相同91因果输入/七日可用性掩码/各358年度原点。seed42、lr0.05、L2=1、200轮、关闭早停和warm_start；4网格叶数7/15×最小叶样本10/30，内2折gap7，每年9次，共27拟合。CV原始流量R²，解码max(Q(t)+delta,0)。无2004、无权重搜索、无增网格，保留全部结果。目标影响选参，不声称相同最终参数的纯因果对照。

## Task 1：先失败测试、再最小实现

Files: tests/test_h1_boost.py、code/boosting.py、code/research_h1_boost.py、code/verify_h1_boost.py。

- [ ] 将现有真实小模型fixture目标参数化列表加入raw_residual；新增手工父类对照、clone/joblib、负输入/负y与早停拒绝测试。固定训练y=x[:,0]+10*cos(index)，对照模型fit(x,y-x[:,0])，断言包装预测逐项等于max(parent.predict(x)+x[:,0],0)。用归零手工父类预测校验截断。
- [ ] 运行`D:\anaconda3\python.exe -m pytest tests/test_h1_boost.py -q`，确认因缺新类/缺目标选项失败，而非其他原因。
- [ ] 新类fit执行`super().fit(X,truth-current,sample_weight=sample_weight)`，predict执行`np.maximum(super().predict(X)+current,0)`。复用_current_flow与原目标验证语义，禁止early_stopping非False。运行器改成明确目标到类/描述映射，默认raw不变；新增CLI枚举；验证器对raw_residual调用父类未解码输出加回X[:,0]并逐项核对。
- [ ] 重跑上述测试及tests/test_selection_search.py、tests/test_ann_boost_average.py，全部通过才Git保存实现。保留两种旧目标和修改fixture2004不影响输入SHA/预测的回归。

## Task 2：27次真实拟合与独立核验

- [ ] 先确认输出不存在；运行`D:\anaconda3\python.exe code/research_h1_boost.py --target raw_residual --output output/teacher_selection_20261109/h1_boost_raw_residual_seed42`。拟合前协议落盘；保留CV/权重/警告/轮数/2148预测（包含持久性）。
- [ ] 运行`D:\anaconda3\python.exe code/verify_h1_boost.py output/teacher_selection_20261109/h1_boost_raw_residual_seed42`。验证数据/源码/权重SHA、9个gap记录、4候选的CV选择、原始次日真值与指标、3模型恢复和父类输出手算解码。
- [ ] 从三个旧新提升树与ANN保存文件按相同日期复算比较；不按结果再搜索或打开2004。真实警告如实保留。
- [ ] 更新本计划、三次验收研究记录和对话交接；强制跟踪被忽略的本轮结果及3份模型权重，Git存档。旧Word/正式ZIP不变，旧258432b包不冒称包含新代码。

自查：所有设计范围均覆盖；个人信息仍待填；目标是实测改善而非保证0.8。若失败保留证据，不替换较强ANN或宣称测试提高。
