# -*- coding: utf-8 -*-
"""第二轮验收报告与附件生成；2026-10-31，第三次验收，第2次修改版。"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import unquote
import zipfile

import numpy as np
import pandas as pd
from docx import Document
from docx.oxml import OxmlElement
from docx.shared import Inches
from PIL import Image
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '验收/Acceptance 3/第二轮修订版'
BASE = Path(__file__).with_name('任务1至7+报告生成_2026-10-31_第3次验收_第1次修改版.py')
spec = importlib.util.spec_from_file_location('report_v1_helpers', BASE)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
v.OUT = OUT
plt = v.plt


class Report(v.Report):
    def table(self, frame, columns=None):
        super().table(frame, columns)
        for row in self.doc.tables[-1].rows:
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))

    def image(self, label, path):
        with Image.open(path) as image:
            width = min(6.3, 7.0 * image.width / image.height)
        self.doc.add_picture(str(path), width=Inches(width))
        self.link(label, path)
        from urllib.parse import quote
        self.md.insert(len(self.md) - 2, f'![{label}]({quote(v.rel(path), safe="/")})\n')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv(frame, name):
    path = OUT / 'data' / name
    frame.to_csv(path, index=False, encoding='utf-8-sig')
    return path


def comparison_plot(old, current, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for i, strategy in enumerate(['direct', 'multi']):
        for j, model in enumerate(['LSTM', 'Mean', 'Stacking']):
            ax = axes[i, j]
            for data, label, color, marker, line in [(old, '第一轮 seed42', v.MUTED, 's', '--'),
                    (current, '第二轮 seed42', v.COLORS['LSTM'], 'o', '-')]:
                f = data[(data.strategy == strategy) & (data.model == model)].sort_values('h')
                ax.plot(f.h, f.RMSE, color=color, marker=marker, markersize=6,
                        markeredgecolor=v.SURFACE, markeredgewidth=1, ls=line, label=label)
                ax.annotate(f'{f.RMSE.iloc[-1]:.0f}', (7, f.RMSE.iloc[-1]),
                            xytext=(5, 6 if label.startswith('第二') else -12),
                            textcoords='offset points', color=v.INK, fontsize=8)
            ax.set_title(f'{strategy} / {model}', loc='left')
            ax.set_xlabel('预见期 / 天')
            ax.set_ylabel('RMSE（原始Discharge单位）')
            ax.set_xticks(range(1, 8))
            ax.set_xlim(.8, 7.9)
            ax.legend(frameon=False, fontsize=8)
            v.style_axis(ax)
    fig.suptitle('两轮同一359起报点对照 · 更低RMSE更好', fontsize=13)
    v.save_chart(fig, path)


def history_plot(run, selections):
    rows, records = [], []
    for strategy in ['direct', 'multi']:
        best = next(r for r in selections if r['stage'] == 'final' and r['model'] == 'LSTM'
                    and r['strategy'] == strategy and r['h'] == (1 if strategy == 'direct' else 7))
        for path in sorted((run / 'training_history').glob(f'final_grid_{strategy}_LSTM_h{best["h"]}_*.json')):
            r = load(path)
            if all(r['params'].get(k) == value for k, value in best['best_params'].items()):
                records.append((strategy, r))
                for h in r['early_history']:
                    rows.append(dict(strategy=strategy, source_record=path.name, n_train=r['n_train'],
                                     selected_epochs=r['selected_epochs'], **h))
    assert len(records) == 4
    data = pd.DataFrame(rows)
    path = csv(data, '任务1+胜出候选内部早停真实历史.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for i, strategy in enumerate(['direct', 'multi']):
        for k, (_, r) in enumerate(x for x in records if x[0] == strategy):
            f = pd.DataFrame(r['early_history'])
            color = [v.COLORS['LSTM'], v.COLORS['Mean']][k]
            label = f'内折{k + 1} · 选{r["selected_epochs"]}轮'
            for j, metric in enumerate(['val_RMSE', 'loss']):
                ax = axes[i, j]
                ax.plot(f.epoch, f[metric], color=color, ls=['-', '--'][k], label=label)
                ax.set_title(f'{strategy} / {"原尺度验证RMSE" if j == 0 else "变换尺度训练损失"}', loc='left')
                ax.set_xlabel('epoch')
                ax.set_ylabel('原始Discharge单位' if j == 0 else '无量纲')
                ax.legend(frameon=False, fontsize=8)
                v.style_axis(ax)
    fig.suptitle('最终阶段胜出LSTM的两个内部早停过程 · direct h1 / multi h1—7', fontsize=12)
    figure = OUT / 'figures/任务1+第二轮真实早停历史.png'
    v.save_chart(fig, figure)
    return path, figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='formal_20261031_v2')
    args = parser.parse_args()
    run = ROOT / 'output/acceptance3_runs' / args.run_id
    oldrun = run.with_name('formal_20261031_v1')
    manifest, verification = load(run / 'manifest.json'), load(run / 'verification.json')
    assert manifest['status'] == 'completed' and not manifest['smoke']
    assert verification['status'] == 'passed' and verification['required_rows_per_seed'] == 42
    name = '任务1至7+第三次阶段验收报告_第2次修改版'
    target, md = OUT / (name + '.docx'), OUT / (name + '.md')
    archive = ROOT / '验收/任务1至7+第三次阶段验收交付包_第2次修改版.zip'
    assert not any(p.exists() for p in [target, md, archive]), '不覆盖已有最终交付物'
    for folder in ['data', 'artifacts', 'figures']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    copies = {
        'metrics.csv': 'data/任务1至7+主seed42全部指标.csv',
        'metrics_all_seeds.csv': 'data/任务1至7+三seed全部指标.csv',
        'test_predictions.csv': 'data/任务1至7+主seed42逐日预测.csv',
        'test_predictions_all_seeds.csv': 'data/任务1至7+三seed逐日预测.csv',
        'oof_predictions_all_seeds.csv': 'data/任务5及7+三seed因果OOF预测.csv',
        'all_cv_candidates.csv': 'data/任务1至7+880行网格候选逐折成绩.csv',
        'outer_splits.csv': 'data/任务1及5及7+外层时间划分.csv',
        'meta_validation.csv': 'data/任务5及7+元层开发期验证.csv',
        'baseline_metrics.csv': 'data/任务1至7+朴素基线指标.csv',
        'baseline_predictions.csv': 'data/任务1至7+朴素基线逐日预测.csv',
        'manifest.json': 'artifacts/任务1至7+运行来源与环境.json',
        'config.json': 'artifacts/任务1至7+实验配置.json',
        'best_parameters.json': 'artifacts/任务1至7+四阶段完整最优参数.json',
        'meta_parameters.json': 'artifacts/任务5及7+三seed元层最优参数.json',
        'selection_frozen.json': 'artifacts/任务1至7+测试前冻结记录.json',
        'data_summary.json': 'artifacts/任务1至7+样本与特征摘要.json',
        'verification.json': 'artifacts/任务1至7+独立结果核验.json'}
    for source, destination in copies.items():
        dst = OUT / destination
        assert not dst.exists(), dst
        shutil.copy2(run / source, dst)
    teacher = OUT / 'artifacts/任务1至7+第三次阶段验收说明原文.docx'
    shutil.copy2(OUT.parent / 'artifacts' / teacher.name, teacher)
    metrics = pd.read_csv(run / 'metrics.csv')
    old = pd.read_csv(oldrun / 'metrics.csv')
    all_metrics = pd.read_csv(run / 'metrics_all_seeds.csv')
    predictions = pd.read_csv(run / 'test_predictions.csv', parse_dates=['origin_date', 'target_date'])
    required = metrics[metrics.model.isin(['LSTM', 'Mean', 'Stacking'])]
    assert len(required) == 42
    parameters = load(run / 'best_parameters.json')
    metas = load(run / 'meta_parameters.json')
    aggregate = metrics.groupby(['strategy', 'model'])[['R2', 'RMSE', 'MAE']].mean().reset_index()
    comparison = old.merge(metrics, on=['strategy', 'model', 'h'], suffixes=('_v1', '_v2'))
    comparison['RMSE变化%'] = 100 * (comparison.RMSE_v2 / comparison.RMSE_v1 - 1)
    cmp_path = csv(comparison, '任务1至7+两轮同seed逐预见期对照.csv')
    old_path = csv(old, '任务1至7+第一轮同口径指标对照.csv')
    seed_summary = all_metrics.groupby(['strategy', 'model', 'h'])[['R2', 'RMSE', 'MAE']].agg(['mean', 'std'])
    seed_summary.columns = ['_'.join(c) for c in seed_summary.columns]
    seed_path = csv(seed_summary.reset_index(), '任务1至7+三seed逐预见期均值与样本标准差.csv')
    splits = pd.concat([pd.read_csv(p).assign(source=p.name) for p in sorted((run / 'splits').glob('*.csv'))])
    split_path = csv(splits, '任务1+全部阶段内层时间划分.csv')
    hist_path, hist_image = history_plot(run, parameters)
    report = Report()
    report.title('任务1至7+第三次阶段验收报告')
    report.paragraph('第2次修改版｜2026-10-31｜LSTM与集成学习的1—7天径流预测｜USGS 01047000')
    report.paragraph(f'正式运行：{args.run_id}；完成时间：{manifest["finished"]}；耗时{manifest["seconds"] / 60:.2f}分钟。主结果固定使用seed42，另报告seed43、44，不按测试分数挑选随机种子。')
    report.paragraph('验收：2026年11月5日13:30—16:50，第12教研楼202。本修订版保留第一轮作为对照，不覆盖原报告，不拼接两轮较优分数。姓名、学号等未提供信息不作编造。')
    report.heading('任务1至7+摘要与验收完成情况')
    report.paragraph('第二轮完成880行候选、2048次fit，72个最终基模型全部重新加载并重建359起报点预测；独立核验通过。七项任务所需42行、126个精度值全部给出。训练更充分不等于全面更准确：仅10/42行RMSE优于第一轮。直接h1 LSTM的RMSE由639.0101降至601.9554（改善5.80%），但直接LSTM和两种Stacking的七步平均误差上升；多输出LSTM仅小幅变化。')
    report.table(pd.DataFrame([{'任务': '任务1', '内容': '8个LSTM的GridSearchCV', '结果': '12候选×2时间折/任务、最优参数及早停历史'}] +
        [{'任务': f'任务{n}', '内容': label, '结果': '7行R²/RMSE/MAE、真实程序截图及过程线'} for n, _, _, label in v.TASKS]))
    report.link('任务1至7+教师验收说明原文', teacher)
    report.heading('任务1至7+数据与公平评价协议')
    report.paragraph('教师CSV含2000—2004年1827天数据。Swe恒为0，剔除；原84列因果特征保留，新增当日Q与年内周期sin/cos，候选列共87。折内先对原84列做互信息TOP50，再按表示方案加入Q，或使用精简9列；plus_Q为51维，并不包含额外两个周期列。多输出选择共用h1目标的特征排序。序列长度30，LSTM输入(N,30,p)，ANN/RF使用起报日p列。输入缩放和监督特征选择均仅在该次fit训练集拟合。')
    report.table(pd.DataFrame([{'阶段': '完整训练', '起报点': 1395, '边界': '2000-02-29—2003-12-24；全部7列目标≤2003-12-31'},
        {'阶段': '外层OOF', '起报点': 723, '边界': '2002全年365、2003上半年181、下半年177'},
        {'阶段': '统一测试', '起报点': 359, '边界': '2004-01-01—12-24；所有目标均在2004'}]))
    report.paragraph('场景为t日结束后，用截至t日已观测信息预测Q(t+1)…Q(t+7)。测试时可使用已观测到的2004历史，不更新参数，不使用未来气象实测；不是年初一次性预报全年。所有方法共享起报点，直接任务也共用最大h7合法边界。外层、内层、早停和元层均净化7天，训练最晚目标严格早于验证首起报日。')
    report.paragraph('指标定义：R²=1−Σ(y−ŷ)²/Σ(y−ȳ)²，可为负；RMSE=√mean((y−ŷ)²)，MAE=mean(|y−ŷ|)。均在逆目标变换后的原始Discharge尺度计算，原始单位未获可靠证明，不擅自标为m³/s或换算。旧报告部分R2采用相关系数平方，与本次标准R²不可直接比较。')
    report.paragraph('历史实验已经查看2004反馈，本轮属于回溯开发评估，而非全新盲测。本轮所有基模型和元层选择在预测2004前冻结，但这不能消除历史反馈带来的研究层面适应。')
    report.link('任务1及5及7+外层划分', OUT / copies['outer_splits.csv'])
    report.link('任务1+全部内层划分', split_path)
    report.link('任务1至7+测试前冻结记录', OUT / copies['selection_frozen.json'])
    report.page()
    report.heading('任务1+LSTM网格搜索与最终参数')
    report.paragraph('四阶段分别为3个OOF阶段及最终训练阶段；每阶段独立pilot，不用未来阶段选择的表示回填早期OOF。pilot比较6种表示/损失：原TOP50/log、加Q/log、精简9列/log、加Q/sqrt、加Q/log混合权重0.25或0.5；只运行direct h1与multi七输出，再将同策略表示用于各h。此设计节省预算，但h1选出的表示未必适合长预见期。')
    report.table(pd.DataFrame([{'模型': 'LSTM', '搜索空间': 'units32、64、64/32 × dropout0.1/0.2 × lr0.001/0.0005', '候选/任务': 12},
        {'模型': 'ANN', '搜索空间': 'hidden64；alpha0.01/0.1/1 × log/sqrt；lr0.001', '候选/任务': 6},
        {'模型': 'RF', '搜索空间': '300树；depth10/20 × leaf1/5 × log/sqrt', '候选/任务': 8}]))
    report.paragraph('GridSearchCV采用2折扩张时间窗，refit=False，评价为原尺度逐h RMSE均值的负值；多输出不是先混合所有平方误差再开方。最小内折训练478，内部再留90天和7天净化后至少381条梯度训练样本。4×[2×6+8×(12+6+8)]=880行候选，2折共1760次搜索fit；96个阶段胜出配置×3seed再fit288次，共2048次。')
    report.paragraph('LSTM使用Adam、batch32、顺序batch、梯度裁剪1，最多150轮、patience15，以原尺度验证RMSE监控；须相对最佳值改善超过0.1%才更新最佳epoch。内部尾段长度max(90,min(180,ceil(0.15n)))，只用于选轮数，之后重新初始化并用整个折训练。最终采用胜出候选两折最佳轮数的中位数四舍五入。三seed共用seed42选出的配置和轮数，未做三seed网格搜索。')
    final_lstm = [r for r in parameters if r['stage'] == 'final' and r['model'] == 'LSTM']
    rows = []
    for r in final_lstm:
        p = r['best_params']
        rows.append({'策略/h': f'{r["strategy"]}/{r["h"] if r["strategy"] == "direct" else "1—7"}',
            'units': str(p['units']), 'dropout': p['dropout'], 'lr': p['lr'],
            'λ': p['mixed_weight'], '轮数': r['fixed_epochs'], 'CV RMSE': -r['best_score']})
    report.table(pd.DataFrame(rows))
    report.paragraph('以上8组最终LSTM均采用plus_Q及log目标；λ=0为标准化log MSE，λ>0加入以训练目标标准差归一化的原尺度MSE，λ为其权重。sqrt候选逆变换预先规定max(逆标准化预测,0)²；log候选用exp恢复。并非根据测试误差事后截断。')
    report.link('任务1至7+完整四阶段最优参数', OUT / copies['best_parameters.json'])
    report.link('任务1至7+880行候选逐折成绩', OUT / copies['all_cv_candidates.csv'])
    report.image('任务1+最终8组参数真实结果窗口截图', OUT / 'figures/任务1+第二轮真实程序窗口截图.png')
    report.image('任务1+真实早停历史（两个量分面，不使用双纵轴）', hist_image)
    report.link('任务1+早停曲线的逐epoch数值表', hist_path)
    report.heading('任务1及3+原生多输出实现证据', 2)
    report.paragraph('SequenceNet最后一层直接输出output_dim=7，单次前向返回(N,7)，不是7个单输出模型套装。ANN和RF均直接fit二维目标；多输出元层为21输入到7输出的单个Ridge。未调用MultiOutputRegressor。Ridge平方损失的各输出系数仍分别估计，不因此宣称自动建模目标协方差。')
    source = (run / 'source/acceptance3_model.py').read_text(encoding='utf-8').splitlines()
    evidence = '\n'.join(f'{i + 1}: {source[i]}' for i in range(23, 41))
    source_image = OUT / 'figures/任务1及3+原生多输出网络源码证据.png'
    v.render_output(evidence, source_image, '任务1及3+训练快照源码摘录')
    report.image('任务1及3+训练时网络源码摘录（文字渲染，非桌面截屏）', source_image)
    report.heading('任务4至7+集成方法与元层选参')
    report.paragraph('简单平均严格为原尺度(ANN+RF+LSTM)/3，不是log预测均值再exp。直接Stacking每h输入同h三个模型预测，训练7个Ridge；多输出Stacking按ANN h1—7、RF h1—7、LSTM h1—7组成21列，训练原生7输出Ridge。基模型的OOF仅使用当前验证起报前合法训练标签。')
    report.paragraph('元层在前段OOF训练、后段OOF验证并净化7天，比较raw/log输入输出×alpha1/10/100/1000，最终在全部OOF重拟合。log分支预先以1e-12处理零值，再标准化及Ridge，预测exp恢复；raw分支不裁剪负值。每seed单独选择元层，主表seed42如下。')
    report.table(pd.DataFrame([{'策略/h': f'{r["strategy"]}/{r["h"] if r["strategy"] == "direct" else "1—7"}',
        '变换': r['kind'], 'alpha': r['alpha'], '验证RMSE': r['mean_validation_RMSE']}
        for r in metas if r['seed'] == 42]))
    report.link('任务5及7+三seed元层参数', OUT / copies['meta_parameters.json'])
    report.link('任务5及7+元层逐折验证', OUT / copies['meta_validation.csv'])
    report.link('任务5及7+三seed因果OOF预测', OUT / copies['oof_predictions_all_seeds.csv'])
    for task, strategy, model, title in v.TASKS:
        report.page()
        report.heading(f'任务{task}+{title}结果')
        part = metrics[(metrics.strategy == strategy) & (metrics.model == model)].sort_values('h')
        report.paragraph('主表：预先固定seed42；每行359个共同测试起报点。负R²如实保留；本组全部预测负值数为0。')
        report.table(part[['h', 'R2', 'RMSE', 'MAE', 'n']].rename(columns={'R2': 'R²'}))
        report.link(f'任务{task}+逐预见期结果CSV', csv(part, f'任务{task}+{title}指标.csv'))
        s = seed_summary.reset_index()
        s = s[(s.strategy == strategy) & (s.model == model)]
        stable = pd.DataFrame({'h': s.h})
        for metric in ['R2', 'RMSE', 'MAE']:
            stable[metric + ' 均值±s'] = [f'{a:.4f} ± {b:.4f}' for a, b in zip(s[metric + '_mean'], s[metric + '_std'])]
        report.table(stable)
        report.paragraph('三seed表为同一数据、同一配置下初始化/随机性波动，s为样本标准差（ddof=1），不是独立年份置信区间，不代表三seed重新搜索超参数。')
        compare = comparison[(comparison.strategy == strategy) & (comparison.model == model)]
        report.paragraph(f'与第一轮同seed同样本比较，本组{int((compare.RMSE变化.values < 0).sum()) if "RMSE变化" in compare else int((compare["RMSE变化%"] < 0).sum())}/7个预见期RMSE降低；七步RMSE算术平均为{part.RMSE.mean():.4f}，MAE平均为{part.MAE.mean():.4f}。该比较为事后描述，不参与本轮选参。')
        if model != 'LSTM':
            gains = []
            for h in range(1, 8):
                individuals = metrics[(metrics.strategy == strategy) & (metrics.h == h) & metrics.model.isin(['ANN', 'RF', 'LSTM'])]
                best = individuals.loc[individuals.RMSE.idxmin()]
                actual = part[part.h == h].iloc[0]
                gains.append({'h': h, '同h最低RMSE基模型': best.model, '基模型RMSE': best.RMSE,
                              '集成RMSE': actual.RMSE, 'RMSE变化%': 100 * (actual.RMSE / best.RMSE - 1)})
            gain = pd.DataFrame(gains)
            report.table(gain)
            report.paragraph(f'本集成在{int((gain["RMSE变化%"] < 0).sum())}/7个预见期优于同h最低RMSE基模型。此处逐h最佳基模型仅为诊断参照，不是一套已部署模型；集成不保证优于基模型。')
            report.link(f'任务{task}+集成增益明细', csv(gain, f'任务{task}+集成相对基模型对照.csv'))
        trace = OUT / f'figures/任务{task}+{title}过程线.png'
        v.trace_plot(predictions, strategy, model, trace, args.run_id)
        report.image(f'任务{task}+h1、h3、h7过程线（横轴为目标日期）', trace)
        report.image(f'任务{task}+已完成结果查看器真实窗口截图', OUT / f'figures/任务{task}+第二轮真实程序窗口截图.png')
    report.page()
    report.heading('任务1至7+两轮对照、朴素基线与总体结论')
    chart = OUT / 'figures/任务2至7+第二轮逐预见期指标.png'
    v.metric_plot(metrics, chart, args.run_id)
    report.image('任务2至7+第二轮三方法精度对照', chart)
    comparison_image = OUT / 'figures/任务2至7+两轮同seed误差对照.png'
    comparison_plot(old, metrics, comparison_image)
    report.image('任务2至7+两轮同seed逐h RMSE对照', comparison_image)
    agg_old = old.groupby(['strategy', 'model'])[['RMSE', 'MAE']].mean().reset_index()
    totals = agg_old.merge(aggregate, on=['strategy', 'model'], suffixes=('_v1', '_v2'))
    totals['RMSE变化%'] = 100 * (totals.RMSE_v2 / totals.RMSE_v1 - 1)
    report.table(totals[['strategy', 'model', 'RMSE_v1', 'RMSE_v2', 'MAE_v1', 'MAE_v2', 'RMSE变化%']])
    report.paragraph('以上为各h指标的算术平均，不是把7步预测堆在一起计算一个R²或RMSE。第一轮只有seed42，因此只用同seed作对照，不把第二轮三seed均值与第一轮单seed混称统计显著改善。三seed×七步各自平均后再跨seed汇总的完整数据见附件。')
    report.link('任务1至7+两轮逐h完整对照', cmp_path)
    report.link('任务1至7+第一轮同口径完整指标', old_path)
    baseline = pd.read_csv(run / 'baseline_metrics.csv')
    report.table(baseline[baseline.model == 'Persistence'][['h', 'R2', 'RMSE', 'MAE']])
    report.paragraph('持久性基线为所有h均预测当日Q，其他基线含训练均值、起报月份季节均值和7日历史Q的Ridge(alpha10)。LSTM h1 RMSE优于持久性，但MAE 213.70与持久性214.46接近；direct h3/h5出现负R²，说明并非所有预见期都形成稳定技能。不能只用复杂模型之间排名证明有效性。')
    report.link('任务1至7+四类朴素基线完整指标', OUT / copies['baseline_metrics.csv'])
    report.link('任务1至7+四类朴素基线逐日预测', OUT / copies['baseline_predictions.csv'])
    report.heading('任务1至7+退化定位：已证据与因果边界')
    report.paragraph('1. 不能简单归因于训练时间短。第二轮已运行约92分钟，直接h1轮数4→18且改善，但其他h退化。final h5两折轮数14/3后冻结9，h6为22/2后冻结12；内部早停与外折评分属于不同季节。搜索时各折自行早停，部署时取中位固定轮数，该固定轮数策略未在同阶段内折重新评分。此为可证实的验证/部署差异，不代表多训练一定有益。')
    report.paragraph('2. 时间合规不等于验证代表性。最早OOF阶段搜索验证仅覆盖2001年6—12月，h1两窗最大真值422/476；最终后半验证窗却包含13700洪峰。final h5候选两折排名Spearman相关约−0.538，最好与次好平均RMSE仅差2.705，显示选参不稳定。最终CV RMSE与2004误差受年份难度影响，不能直接视为同分布泛化差距。')
    rep_rows = []
    for r in parameters:
        if r['model'] == 'LSTM' and (r['strategy'] == 'multi' or r['h'] == 1):
            p = r['best_params']
            rep_rows.append({'阶段': r['stage'], '策略': r['strategy'], '表示': p['feature_mode'],
                             '目标': p['target_transform'], 'λ': p['mixed_weight']})
    report.table(pd.DataFrame(rep_rows))
    report.paragraph('3. 上表证实OOF→最终配置变化。嵌套OOF评估完整自适应选参算法本身合理，不构成天然标签泄漏；但元层混合不同表示/目标变换/训练规模的预测，校准不保证稳定。RF的OOF均为log，而最终direct h1—5与multi转为sqrt；multi ANN也由OOF log转sqrt。配置和年份变化同时存在，不能仅凭预测均值归因于某一个改动。')
    # 从本轮落盘预测产生可追溯的偏差及极端流量诊断。
    diagnostic = []
    for (strategy, model, h), f in predictions[predictions.model.isin(['LSTM', 'Mean', 'Stacking'])].groupby(['strategy', 'model', 'h']):
        for label, mask in [('全部', np.ones(len(f), dtype=bool)), ('真值<500', f.truth < 500),
                            ('真值最高10%', f.truth >= f.truth.quantile(.9))]:
            s = f.loc[mask]
            diagnostic.append({'strategy': strategy, 'model': model, 'h': h, 'segment': label,
                'n': len(s), 'MAE': np.abs(s.pred - s.truth).mean(),
                'bias': (s.pred - s.truth).mean(), 'SSE': np.square(s.pred - s.truth).sum()})
    diag_path = csv(pd.DataFrame(diagnostic), '任务2至7+分流量区间偏差与误差.csv')
    report.paragraph('4. 元层发生外推式校准，而非简单平均。OOF第三块仅177/723条样本，却贡献direct/multi LSTM约76.11%/77.62%的平方误差，三基模型该块整体低估。主seed42 direct h3原尺度元层约为114.40+0.4061ANN+0.6644RF+0.0383LSTM；它不满足非负且和为1的凸融合约束。h3有297/359个Stacking输出超出三模型预测区间，不等同于全部输入超出OOF历史支持。')
    report.paragraph('测试七步总体偏差由Mean的−49.91/−38.16变为Stacking的+118.67/+155.84（direct/multi）；真值<500时MAE由177.24/189.61上升至312.38/347.19。元层以部分高流量误差下降换取低流量过报。开发期选参所用验证上，Stacking相对RF的七步平均RMSE改善仅约0.15%/1.42%，不是独立元层泛化证据。三seed所有方法负预测为0，本轮不足不能归咎于负值。')
    report.paragraph('5. 测试事后定位显示年初集中过报，不能只用洪峰低估解释。direct h3相对第一轮新增SSE约89.76%来自1月前30个起报点，h5为61.46%；2004-01-01起报h3真值705，预测4353.25。该发现仅说明问题位置，不删除这些点，不据此为2004重新调参；真正因果仍需开发期受控复现。原84列已满足Discharge_diff1+Discharge_lag1=当日Q，新增Q是显式表示而不是首次获得Q信息。')
    report.link('任务2至7+分流量区间误差与偏差', diag_path)
    report.heading('任务1至7+能否进一步提升与修改后的实验方案')
    report.paragraph('判断：仍有提升空间，但现有证据不支持继续简单加轮数、扩网格或承诺某个R²。优先解决验证代表性和融合校准，再讨论更大模型。以下为尚未执行的下一阶段方案，不作为本报告已完成的第三轮结果。')
    report.table(pd.DataFrame([
        {'优先级': 'P0', '改进': '完整水文年/季节滚动开发验证', '具体做法': '用2000—2003构造覆盖不同季节的合法历史窗；样本不足处明确披露，争取新增年份作盲测', '判据': '原尺度RMSE、MAE、季节bias及事件误差同时报告'},
        {'优先级': 'P0', '改进': '原样验证部署epoch策略', '具体做法': '将固定轮数选择与全量重训作为完整算法在开发期外层评估，不直接把季节中位数当全年最优', '判据': '跨块稳定性；不以2004选epoch'},
        {'优先级': 'P1', '改进': '稳定OOF→final及约束融合', '具体做法': '比较冻结表示与自适应表示；加入RF/Mean退路、非负和为1融合、收缩校准', '判据': '开发期同时改善误差及低流量bias，否则保留退路'},
        {'优先级': 'P1', '改进': '少量受控消融', '具体做法': '固定其他设置，分别改早停、显式Q、log/sqrt、混合loss、epoch；检查跨h与3seed', '判据': '逐项归因，不做十几项同时改变'},
        {'优先级': 'P2', '改进': '新增信息与不确定性', '具体做法': '有条件增加历史年份及真实可用气象预报；成块bootstrap报告误差差值不确定性', '判据': '无未来实测泄漏；新年份盲测前冻结'}]))
    report.paragraph('此前h1 R²约0.50—0.65的工程目标带只能作待检验目标，不是理论保证或置信区间。本轮主seed42 h1 R²0.4889、三seed均值0.4733，尚不能声称达到目标。长预见期缺少未来降雨信息、仅5年观测且极端事件有限，无法给出可靠可保证的提升幅度。')
    report.heading('任务1至7+追溯、环境与交付检查')
    report.paragraph(f'训练后端PyTorch {manifest["torch_build"]}，device={manifest["device"]}；Python {manifest["python"].splitlines()[0]}。训练包版本以运行环境JSON为准，未安装CUDA包。数据SHA256={manifest["data_sha256"]}；Git基线={manifest["git_commit"]}。')
    report.paragraph(f'完整运行保留于output/acceptance3_runs/{args.run_id}/，包含模型、预处理器、训练源码快照和历史。核验实际恢复72个最终基模型，重建全部359起报点、原尺度Mean及落盘元层预测，逐样本重算210行三seed指标通过；结构与时间边界核验通过不代表性能达标或验证季节代表性充分。')
    report.paragraph('七张程序截图来自实际执行结果读取命令的stdout，经Windows PrintWindow仅捕获本程序窗口；并非长时间训练过程截图，不含其他桌面窗口。所有图表附原图链接及数值表。Word结构、压缩完整性和附件链接可自动核验；本机未完成Word/WPS桌面逐页视觉检查，不声称已完成该项。')
    for source, destination in copies.items():
        report.link(Path(destination).stem, OUT / destination)
    report.link('任务1至7+三seed逐h均值与样本标准差', seed_path)
    report.paragraph('提交本Word与配套附件即可查看任务1—7全部结果；Markdown为同内容便读副本。第一、二次验收材料及第一轮最终报告均保留，不混入本修订版交付包。')
    report.doc.save(target)
    md.write_text('\n'.join(report.md), encoding='utf-8')
    # 重算三seed所有指标；链接必须位于本修订版目录内。
    p = pd.read_csv(run / 'test_predictions_all_seeds.csv')
    for key, f in p.groupby(['seed', 'strategy', 'model', 'h']):
        seed, strategy, model, h = key
        r = all_metrics[(all_metrics.seed == seed) & (all_metrics.strategy == strategy) &
                        (all_metrics.model == model) & (all_metrics.h == h)].iloc[0]
        actual = [r2_score(f.truth, f.pred), np.sqrt(np.square(f.truth - f.pred).mean()), np.abs(f.truth - f.pred).mean()]
        np.testing.assert_allclose(actual, r[['R2', 'RMSE', 'MAE']].to_numpy(float), rtol=1e-10, atol=1e-7)
    loaded = Document(target)
    links = set()
    for r in loaded.part.rels.values():
        if r.reltype.endswith('/hyperlink'):
            path = (OUT / unquote(r.target_ref)).resolve()
            assert path.is_relative_to(OUT.resolve()) and path.is_file(), path
            links.add(path)
    for link in re.findall(r'\]\(([^)]+)\)', md.read_text(encoding='utf-8')):
        path = (OUT / unquote(link)).resolve()
        assert path.is_relative_to(OUT.resolve()) and path.is_file()
    text = '\n'.join(x.text for x in loaded.paragraphs)
    assert all(f'任务{i}+' in text for i in range(1, 8))
    assert not any(x in text for x in ['待填写', '待训练'])
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    checks = dict(run_id=args.run_id, required_metric_rows=42, all_seed_metric_rows=210,
        final_lstm_parameter_rows=len(final_lstm), word_hyperlinks_checked=len(links),
        word_inline_images=len(loaded.inline_shapes), metric_recalculation='passed',
        links='passed', docx_integrity='passed', word_desktop_visual_check='not_performed',
        screenshot='actual_result_viewer_window_capture', third_round_training='not_performed')
    check_path = OUT / 'artifacts/任务1至7+第2次修改版交付核验.json'
    check_path.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    files = links | {target, md, check_path}
    assert not any(p.suffix.lower() in ['.py', '.pyc', '.log'] for p in files)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            z.write(path, '第三次阶段验收_第2次修改版/' + path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(target)
    print(archive)


if __name__ == '__main__':
    main()
