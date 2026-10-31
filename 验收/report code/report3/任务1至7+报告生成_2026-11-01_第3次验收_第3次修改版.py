# -*- coding: utf-8 -*-
"""第三次验收第3次修改版，2026-11-01：仅已完成实验，无后续训练计划。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote
import zipfile

import numpy as np
import pandas as pd
from docx import Document
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'验收/Acceptance 3/第三轮结果修订版'
BASE = Path(__file__).with_name('任务1至7+报告生成_2026-10-31_第3次验收_第2次修改版.py')
spec = importlib.util.spec_from_file_location('report_helpers_v2',BASE)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
v = helper.v
helper.OUT = OUT
v.OUT = OUT
Report = helper.Report
RUNS = ROOT/'output/acceptance3_runs'
DIRECT = RUNS/'formal_20261031_v2'
MULTI = RUNS/'formal_20261101_v3_stage_d'
TRAIN = RUNS/'formal_20261031_v3_stage_c'
NAME = '任务1至7+第三次阶段验收报告_第3次修改版'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv(frame,name):
    path = OUT/'data'/name
    frame.to_csv(path,index=False,encoding='utf-8-sig')
    return path


def copy(source,name):
    path = OUT/name
    assert not path.exists(),path
    shutil.copy2(source,path)
    return path


def combined(name):
    a,b = pd.read_csv(DIRECT/name),pd.read_csv(MULTI/name)
    a = a[a.strategy=='direct'].copy()
    b = b[b.strategy=='multi'].copy()
    a['source_run'] = DIRECT.name
    b['source_run'] = MULTI.name
    return pd.concat([a,b],ignore_index=True)


def final_parameters():
    rows = []
    for r in load(DIRECT/'best_parameters.json'):
        if r['stage']=='final' and r['model']=='LSTM' and r['strategy']=='direct':
            p = r['best_params']
            rows.append(dict(策略='direct',h=str(r['h']),units=str(p['units']),dropout=p['dropout'],
                lr=p['lr'],目标=p['target_transform'],输入=p['feature_mode'],
                λ=p['mixed_weight'],epoch=r['fixed_epochs']))
    p = next(r['params'] for r in load(TRAIN/'frozen_config.json')['final_parameters'] if r['family']=='LSTM')
    rows.append(dict(策略='multi',h='1—7',units=str(p['units']),dropout=p['dropout'],lr=p['lr'],
        目标=p['target_transform'],输入=p['feature_mode'],λ=p['mixed_weight'],epoch=p['fixed_epochs']))
    assert len(rows)==8
    return pd.DataFrame(rows)


def comparison_plot(old,new,path):
    fig,axes = v.plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,model in zip(axes,['LSTM','Mean','Stacking']):
        for data,label,color,marker,line in [(old,'第二轮',v.MUTED,'s','--'),
                (new,'第三轮',v.COLORS['LSTM'],'o','-')]:
            f = data[(data.strategy=='multi')&(data.model==model)].sort_values('h')
            ax.plot(f.h,f.RMSE,label=label,color=color,marker=marker,markersize=6,
                    markeredgecolor=v.SURFACE,ls=line,lw=2)
            ax.annotate(f'{f.RMSE.iloc[-1]:.0f}',(7,f.RMSE.iloc[-1]),
                xytext=(4,-12 if label=='第三轮' else 5),textcoords='offset points',fontsize=8,color=v.INK)
        ax.set_title(model,loc='left');ax.set_xlabel('预见期 / 天');ax.set_ylabel('RMSE（原始单位）')
        ax.set_xticks(range(1,8));ax.set_xlim(.8,7.8);ax.legend(frameon=False);v.style_axis(ax)
    fig.suptitle('多输出第二、三轮同seed42对照 · 共同359个起报点')
    v.save_chart(fig,path)


def main():
    target,md = OUT/(NAME+'.docx'),OUT/(NAME+'.md')
    archive = ROOT/'验收/任务1至7+第三次阶段验收交付包_第3次修改版.zip'
    assert not any(p.exists() for p in [target,md,archive]),'不覆盖已有交付物'
    for folder in ['data','artifacts','figures']: (OUT/folder).mkdir(parents=True,exist_ok=True)
    assert load(DIRECT/'verification.json')['status']=='passed'
    assert load(MULTI/'verification.json')['status']=='passed'
    assert load(TRAIN/'verification.json')['status']=='passed'
    metrics = combined('metrics.csv')
    all_metrics = combined('metrics_all_seeds.csv')
    predictions = combined('test_predictions.csv')
    all_predictions = combined('test_predictions_all_seeds.csv')
    predictions['target_date'] = pd.to_datetime(predictions.target_date)
    required = metrics[metrics.model.isin(['LSTM','Mean','Stacking'])]
    assert len(required)==42
    files = []
    def attach(source,name):
        path = copy(source,name);files.append(path);return path
    metric_path = csv(metrics,'任务1至7+主seed42完整指标与来源.csv')
    all_path = csv(all_metrics,'任务1至7+三seed完整指标与来源.csv')
    pred_path = csv(predictions,'任务1至7+主seed42逐日预测与来源.csv')
    all_pred_path = csv(all_predictions,'任务1至7+三seed逐日预测与来源.csv')
    params = final_parameters()
    param_path = csv(params,'任务1+八组最终LSTM参数.csv')
    summary = all_metrics.groupby(['strategy','model','h'])[['R2','RMSE','MAE']].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    seed_path = csv(summary,'任务2至7+三seed均值与样本标准差.csv')
    artifacts = [
        (DIRECT/'manifest.json','artifacts/任务2及4及5+第二轮运行来源.json'),
        (DIRECT/'verification.json','artifacts/任务2及4及5+第二轮核验.json'),
        (DIRECT/'best_parameters.json','artifacts/任务1+第二轮完整基模型参数.json'),
        (DIRECT/'meta_parameters.json','artifacts/任务5+第二轮元层参数.json'),
        (DIRECT/'selection_frozen.json','artifacts/任务2及4及5+第二轮冻结记录.json'),
        (TRAIN/'manifest.json','artifacts/任务3及6及7+第三轮训练来源.json'),
        (TRAIN/'verification.json','artifacts/任务3及6及7+第三轮训练核验.json'),
        (TRAIN/'frozen_config.json','artifacts/任务3及6及7+第三轮最终参数.json'),
        (TRAIN/'meta_decision.json','artifacts/任务7+真实Stacking与部署选择.json'),
        (MULTI/'manifest.json','artifacts/任务3及6及7+第三轮评估来源.json'),
        (MULTI/'verification.json','artifacts/任务3及6及7+第三轮评估核验.json'),
        (MULTI/'selection_frozen.json','artifacts/任务3及6及7+测试前模型校验摘要.json'),
        (TRAIN/'data_summary.json','artifacts/任务1至7+样本与特征摘要.json'),
        (OUT.parent/'artifacts/任务1至7+第三次阶段验收说明原文.docx','artifacts/任务1至7+第三次阶段验收说明原文.docx'),
        (DIRECT/'all_cv_candidates.csv','data/任务1+第二轮完整网格候选.csv'),
        (TRAIN/'outer_splits.csv','data/任务7+第三轮季度OOF划分.csv'),
        (TRAIN/'meta_validation.csv','data/任务7+第三轮元层前向验证.csv'),
        (TRAIN/'oof_predictions.csv','data/任务7+第三轮三seed因果OOF预测.csv'),
        (DIRECT/'oof_predictions_all_seeds.csv','data/任务5+第二轮三seed因果OOF预测.csv'),
        (DIRECT/'baseline_metrics.csv','data/任务2至7+同样本朴素基线指标.csv')]
    for source,name in artifacts: attach(source,name)
    cv = pd.concat([pd.read_csv(p).assign(source=p.name) for p in sorted((TRAIN/'cv_results').glob('*.csv'))],ignore_index=True)
    cv_path = csv(cv,'任务1+第三轮完整网格候选.csv')
    splits = pd.concat([pd.read_csv(p).assign(source=p.name) for p in sorted((TRAIN/'splits').glob('*.csv'))],ignore_index=True)
    split_path = csv(splits,'任务1及7+第三轮内层时间净化审计.csv')
    old = pd.read_csv(DIRECT/'metrics.csv')
    first = pd.read_csv(RUNS/'formal_20261031_v1/metrics.csv')
    compare = metrics.merge(old,on=['strategy','model','h'],suffixes=('_current','_v2'))
    compare['RMSE变化%'] = (compare.RMSE_current/compare.RMSE_v2-1)*100
    cmp_path = csv(compare,'任务2至7+与第二轮逐h对照.csv')
    first_path = csv(first,'任务2至7+第一轮同口径指标.csv')
    report = Report()
    report.title('任务1至7+第三次阶段验收报告')
    report.paragraph('第3次修改版｜2026-11-01｜USGS 01047000｜LSTM与集成学习的1—7天径流预测')
    report.paragraph('验收时间地点：11月5日13:30—16:50，第12教研楼202。本版仅报告已完成实验及结果，不包含后续训练计划。姓名、学号等未提供信息不作编造。')
    report.heading('任务1至7+结果摘要与来源')
    report.paragraph('七项验收内容全部给出：任务1包含直接h1—7与原生七输出LSTM共8组参数；任务2—7共42行、126个R²/RMSE/MAE值。直接策略整体保留第二轮，原生多输出整体采用第三轮，不按测试集逐h挑选较优值，也不把两者冒称同一轮训练。')
    report.table(pd.DataFrame([{'任务':'任务1','内容':'GridSearchCV与8组LSTM参数','来源':'direct第二轮；multi第三轮'}]+
        [{'任务':f'任务{n}','内容':label,'来源':'第二轮' if strategy=='direct' else '第三轮'}
        for n,strategy,_,label in v.TASKS]))
    report.paragraph('第三轮阶段B直接策略未达到事先规定的跨年和步长稳定性门槛，因此没有扩展其后续网格；多输出通过开发门槛后完成季度OOF、最终训练及冻结评估。第三轮阶段A/B/C共1949次正式基模型fit，阶段C821次；这些是已执行记录，不是拟议方案。')
    report.paragraph('主结果预先固定seed42；seed43/44仅报告随机性波动。多输出主seed七步平均RMSE：LSTM793.7147、Mean778.7158、Stacking768.2216、RF757.0957。LSTM较第二轮仅改善0.21%，Stacking改善5.31%，但并非所有结果优于第一轮或RF。')
    report.link('任务1至7+教师验收说明原文',OUT/artifacts[13][1])
    report.heading('任务1至7+数据、预测场景与评价口径')
    report.paragraph('教师CSV为2000—2004年1827天。Swe恒为0剔除；原84列因果工程特征加当日Discharge及年内sin/cos后共87列候选。plus_Q为原84列折内互信息TOP50加当日Q，共51维，不含周期列；compact为当日Q、Prcp、Tmax、Tmin、Vp、Srad、Dayl、doy_sin、doy_cos共9列。监督特征选择和缩放仅fit当次训练集。')
    report.table(pd.DataFrame([{'数据用途':'最终训练','起报点数':1395,'边界':'2000-02-29—2003-12-24，七步目标均≤2003-12-31'},
        {'数据用途':'第二轮直接策略OOF','起报点数':723,'边界':'2002-01-01—2003-12-24，分3块'},
        {'数据用途':'第三轮多输出OOF','起报点数':907,'边界':'2001-07-01—2003-12-24，分10季度'},
        {'数据用途':'共同测试','起报点数':359,'边界':'2004-01-01—2004-12-24，全部目标在2004'}]))
    report.paragraph('t日结束后，使用截至t的已观测信息预测Q(t+1)…Q(t+7)。LSTM输入(N,30,p)，ANN/RF用当日工程特征。测试可使用此前已观测的2004历史，不更新参数、不使用未来天气实测，不是年初预测全年。直接任务亦共用h7合法边界。各层时间划分净化7天，训练最晚h7目标严格早于验证首起报日。')
    report.paragraph('R²=1−Σ(y−ŷ)²/Σ(y−ȳ)²，允许为负，不使用相关系数平方；RMSE=√mean((y−ŷ)²)，MAE=mean(|y−ŷ|)。指标均在逆变换后的原始Discharge尺度，原始单位未可靠确认，不擅自标注或换算。逐步指标的算术均值不等于将全部步长合并后的一个RMSE或R²。')
    report.paragraph('2004已在此前实验中查看，本版为回溯评估，不称独立盲测。第三轮阶段A已用2002/2003筛选输入，阶段B结构选参虽只用每年之前历史，研究层面仍不是独立确认。元层候选比较也是开发选参成绩，不作统计显著性证明。直接与多输出的特征模式、选参及轮数协议不同，因此这里只对照各自冻结策略的结果，不属于同特征、同预算的纯网络结构消融。')
    report.link('任务1及7+第三轮内层划分与净化审计',split_path)
    report.page()
    report.heading('任务1+已执行网格搜索与最终LSTM参数')
    report.paragraph('第二轮直接LSTM采用每阶段2折GridSearchCV，搜索units32/64/64+32、dropout0.1/0.2、lr0.001/0.0005共12候选；每阶段pilot选择表示/目标损失。内部最多150epoch、patience15，以原尺度RMSE监控，选轮后重新初始化全折训练。最终固定轮数为胜出候选两内折最佳轮数的中位数四舍五入。此轮数策略存在选参和部署差异，直接结果据实保留。')
    report.paragraph('第三轮多输出固定compact/log/无混合损失，units32/64×dropout0.1/0.2×fixed_epochs10/20/40，共12候选；lr0.001。epoch直接作为GridSearchCV参数，不隐式早停，不用折间中位数替代。每季度只用该季度之前的历史选择参数；最终全历史4个90日起报验证窗，最小训练365、净化7天。最早OOF只有1个内折，季节覆盖不足仍存在。评分为原尺度逐h RMSE算术均值的负值。')
    report.table(params)
    report.paragraph('LSTM使用Adam、batch32、顺序batch和梯度裁剪1；lookback30。7组直接LSTM均采用plus_Q/log，λ为混合原尺度损失权重；多输出采用compact/log、64units、dropout0.1、固定10轮。10轮来自预先列入网格的候选，不代表程序异常提前结束。')
    report.link('任务1+八组最终参数CSV',param_path)
    report.link('任务1+第二轮完整网格候选',OUT/artifacts[14][1])
    report.link('任务1+第三轮完整网格候选',cv_path)
    report.image('任务1+八组参数真实程序窗口截图',OUT/'figures/任务1+正式结果程序窗口截图.png')
    report.heading('任务1及3+原生多输出实现',2)
    report.paragraph('SequenceNet最终Linear直接输出7列，单次前向返回(N,7)。ANN和RF直接fit二维目标，未使用MultiOutputRegressor。多输出Stacking是21列基模型预测到7列目标的单个原生Ridge。Ridge并不因此自动学习目标协方差。')
    source = (TRAIN/'source/acceptance3_model.py').read_text(encoding='utf-8').splitlines()
    evidence = '\n'.join(f'{i+1}: {source[i]}' for i in range(23,41))
    source_image = OUT/'figures/任务1及3+原生七输出网络证据.png'
    v.render_output(evidence,source_image,'任务1及3+训练源码快照摘录')
    report.image('任务1及3+原生网络源码摘录（文字渲染，非桌面截图）',source_image)
    report.heading('任务4至7+已实现集成方法')
    report.paragraph('Mean严格在原尺度计算(ANN+RF+LSTM)/3。直接Stacking每h用同h三模型预测拟合7个元模型，第二轮raw/log×alpha1/10/100/1000在净化前向OOF中选择；各seed独立选元层。多输出Stacking按ANN h1—7、RF h1—7、LSTM h1—7组成21列，经StandardScaler后拟合原生七输出Ridge(alpha100)，在907条OOF上重训。alpha100是已冻结诊断配置，不声称它由第三轮GridSearchCV选出。原尺度Ridge不裁剪负预测。')
    report.paragraph('第三轮另验证RF、Mean、三权重非负和为1凸融合和向RF收缩的Ridge。部分收缩候选通过平均相对比值门槛，但部署选择按各季度原尺度RMSE的算术平均，RF为635.2039，最佳合格候选636.7351，故SelectedFusion选择RF。二者不可混淆：任务7报告真正Stacking，RF退路只作附加部署结果，不能声称融合提升。')
    report.link('任务5+第二轮元层参数',OUT/artifacts[3][1])
    report.link('任务7+第三轮元层前向验证',OUT/artifacts[16][1])
    report.link('任务7+真实Stacking与部署选择',OUT/artifacts[8][1])
    for task,strategy,model,title in v.TASKS:
        report.page();report.heading(f'任务{task}+{title}结果')
        part = metrics[(metrics.strategy==strategy)&(metrics.model==model)].sort_values('h')
        report.paragraph(f'来源：{DIRECT.name if strategy=="direct" else MULTI.name}。主表固定seed42，每行359个相同测试起报点；负R²保留，本组预测负值数为0。')
        report.table(part[['h','R2','RMSE','MAE','n']].rename(columns={'R2':'R²'}))
        report.link(f'任务{task}+逐h完整结果',csv(part,f'任务{task}+{title}指标.csv'))
        s = summary[(summary.strategy==strategy)&(summary.model==model)].sort_values('h')
        stable = pd.DataFrame({'h':s.h})
        for metric in ['R2','RMSE','MAE']:
            stable[metric+'均值±s'] = [f'{a:.4f} ± {b:.4f}' for a,b in zip(s[metric+'_mean'],s[metric+'_std'])]
        report.table(stable)
        report.paragraph('三seed为相同年份与配置下随机性复核，s为样本标准差ddof=1，不是独立年份置信区间，未按测试结果挑选seed。')
        if model!='LSTM':
            gain = []
            for h in range(1,8):
                base = metrics[(metrics.strategy==strategy)&(metrics.h==h)&metrics.model.isin(['ANN','RF','LSTM'])]
                best = base.loc[base.RMSE.idxmin()];actual = part[part.h==h].iloc[0]
                gain.append({'h':h,'最低RMSE基模型':best.model,'基模型RMSE':best.RMSE,'集成RMSE':actual.RMSE,
                             '变化%':(actual.RMSE/best.RMSE-1)*100})
            gain = pd.DataFrame(gain)
            report.table(gain)
            report.paragraph(f'本集成在{int((gain["变化%"]<0).sum())}/7个预见期优于同h最佳基模型。逐h最佳只是事后诊断参照，不是一套已部署选择器。')
            report.link(f'任务{task}+集成相对基模型数值表',csv(gain,f'任务{task}+集成相对基模型.csv'))
        trace = OUT/f'figures/任务{task}+{title}过程线.png'
        v.trace_plot(predictions,strategy,model,trace,DIRECT.name if strategy=='direct' else MULTI.name)
        report.image(f'任务{task}+h1/h3/h7过程线（目标日期）',trace)
        report.image(f'任务{task}+真实结果程序窗口截图',OUT/f'figures/任务{task}+正式结果程序窗口截图.png')
    report.page();report.heading('任务2至7+总体对照与误差分析')
    chart = OUT/'figures/任务2至7+当前逐预见期指标.png'
    v.metric_plot(metrics,chart,'direct:v2 / multi:v3')
    report.image('任务2至7+逐预见期精度对照',chart)
    comparison = OUT/'figures/任务3及6及7+第二三轮RMSE对照.png'
    comparison_plot(old,metrics,comparison)
    report.image('任务3及6及7+第二三轮同seed42对照',comparison)
    agg = metrics.groupby(['strategy','model'])[['RMSE','MAE']].mean().reset_index()
    oldagg = old.groupby(['strategy','model'])[['RMSE','MAE']].mean().reset_index()
    totals = agg.merge(oldagg,on=['strategy','model'],suffixes=('_current','_v2'))
    totals['RMSE变化%'] = (totals.RMSE_current/totals.RMSE_v2-1)*100
    report.table(totals[['strategy','model','RMSE_current','RMSE_v2','RMSE变化%']])
    report.link('任务2至7+与第二轮逐h完整对照',cmp_path)
    report.link('任务2至7+第一轮完整指标',first_path)
    report.paragraph('第三轮多输出seed42：LSTM七步平均RMSE仅下降0.21%，h1从675.1497退化为679.5188（R²从0.3570降至0.3487）；Mean下降1.02%，Stacking下降5.31%。三seed各自平均七步再跨seed平均，RMSE分别下降1.00%、0.73%、3.14%。不能把开发期相对A0的5.92%当作测试提升。')
    report.paragraph('对第一轮的比较也不支持全面领先：第一轮多输出Mean为767.4847，仍低于本版778.7158；第一轮Stacking767.4846也略低于本版768.2216。Stacking主要修复第二轮校准退化，当前主seed仍差于RF757.0957，MAE423.0160也高于RF357.6963。ANN主seed七步RMSE854.1012，但三seed波动明显，融合不保证增益。')
    baseline = pd.read_csv(DIRECT/'baseline_metrics.csv')
    report.table(baseline[baseline.model=='Persistence'][['h','R2','RMSE','MAE']])
    report.paragraph('持久性基线为各h均预测起报日Q，完整基线还含训练均值、季节均值、7日Q的Ridge。基线与本版使用相同359起报点，可作技能参照；不能只用复杂模型间排名证明有效。远期缺少未来降雨信息，洪峰幅度和时序偏差仍明显。')
    report.link('任务2至7+朴素基线完整指标',OUT/artifacts[19][1])
    report.heading('任务1至7+完成性、来源及交付核验')
    report.paragraph('第二轮完整运行约92分钟；第三轮A约6.54分钟、B约25.94分钟、C约17.02分钟，阶段C821次基模型fit全部保存恢复通过。训练时长不是精度证据；本版不承诺不存在于实测结果中的提升。CPU/PyTorch2.6.0+cpu，包版本和数据SHA256见来源附件。')
    report.paragraph('直接策略保留第二轮结果不等于第三轮重新完成所有任务；多输出第三轮独立完成。2004的359个共同起报点、目标日期和真值一致，Mean与两种元层由落盘预测重建。源码、模型和历史保留在本地运行目录；交付包只包含报告直接引用的最终附件，不含训练计划、源码、缓存或日志。')
    report.paragraph('七张程序图来自实际执行只读结果命令并以Windows PrintWindow捕获本程序窗口，不冒充训练过程截图。所有图内嵌且附原图链接、完整数值表。自动检查Word结构、数值、相对链接、ZIP完整性；未进行Word/WPS桌面逐页检查，因此不声称完成桌面分页视觉验收。')
    for path in [metric_path,all_path,pred_path,all_pred_path,param_path,seed_path,cv_path,split_path]+files:
        report.link(path.stem,path)
    report.doc.save(target)
    md.write_text('\n'.join(report.md),encoding='utf-8')
    loaded = Document(target)
    links = set()
    for r in loaded.part.rels.values():
        if r.reltype.endswith('/hyperlink'):
            path = (OUT/unquote(r.target_ref)).resolve()
            assert path.is_relative_to(OUT.resolve()) and path.is_file(),path
            links.add(path)
    for ref in re.findall(r'\]\(([^)]+)\)',md.read_text(encoding='utf-8')):
        path = (OUT/unquote(ref)).resolve()
        assert path.is_relative_to(OUT.resolve()) and path.is_file(),path
    tables = [t for t in loaded.tables if [c.text for c in t.rows[0].cells]==['h','R²','RMSE','MAE','n']]
    assert len(tables)==6
    for table,(_,strategy,model,_) in zip(tables,v.TASKS):
        actual = np.array([[float(c.text) for c in row.cells] for row in table.rows[1:]])
        expected = required[(required.strategy==strategy)&(required.model==model)].sort_values('h')[['h','R2','RMSE','MAE','n']].to_numpy()
        np.testing.assert_allclose(actual,expected,rtol=0,atol=.000051)
    assert len(params)==8 and len(loaded.inline_shapes)==16
    headings = [p.text for p in loaded.paragraphs if p.style.name.startswith('Heading')]
    assert not any('方案' in h or '计划' in h for h in headings)
    text = '\n'.join(p.text for p in loaded.paragraphs)
    assert all(f'任务{i}+' in text for i in range(1,8))
    assert not any(word in text for word in ['待填写','待训练','下一阶段方案','优先级','P0','P1','P2'])
    with zipfile.ZipFile(target) as z: assert z.testzip() is None
    check = dict(status='passed',required_metric_rows=42,required_metric_values=126,parameter_rows=8,
        combined_all_model_metric_rows=len(all_metrics),word_inline_images=len(loaded.inline_shapes),
        checked_relative_links=len(links),word_metric_rounding='passed',links='passed',
        training_plan_included=False,word_desktop_visual_check='not_performed',
        direct_source=DIRECT.name,multi_source=MULTI.name)
    check_path = OUT/'artifacts/任务1至7+第3次修改版交付核验.json'
    check_path.write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(links|{target,md,check_path}):
            assert path.suffix.lower() not in ['.py','.pyc','.log']
            z.write(path,'第三次阶段验收_第3次修改版/'+path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    images = sorted((OUT/'figures').glob('*.png'))
    contact = Image.new('RGB',(1200,260*((len(images)+2)//3)),'white')
    draw = ImageDraw.Draw(contact)
    for i,path in enumerate(images):
        image = Image.open(path).convert('RGB')
        image.thumbnail((390,225))
        x,y = (i%3)*400,(i//3)*260
        contact.paste(image,(x,y+25))
        draw.text((x+5,y+5),path.name,fill='black')
    contact.save(Path(__file__).parent/'tmp/任务1至7+第三轮全部图表检查缩略图.png')
    print(json.dumps(check,ensure_ascii=False,indent=2));print(target);print(archive)


if __name__=='__main__': main()
