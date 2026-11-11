"""先独立核验，再导出策略增强结果；不从外层成绩拼最优模型。"""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from analyze_runoff_controlled import paired_interval, summary_metrics
from runoff_training.controlled_data import load_data, outer_split, sha256
from runoff_training.controlled_train import write_json
from verify_runoff_strategy import verify, recompute, ROOT

BASELINE='last_plain_log'


def arrays(frame,variant):
    piece=frame[frame.variant==variant]
    truth,ids,predictions=None,None,[]
    for seed in sorted(piece.seed.unique()):
        part=piece[piece.seed==seed]
        p=part.pivot(index='sample_id',columns='h',values='pred').sort_index()
        t=part.pivot(index='sample_id',columns='h',values='truth').sort_index()
        if truth is not None:
            np.testing.assert_equal(t.to_numpy(),truth)
            np.testing.assert_equal(p.index.to_numpy(),ids)
        truth,ids=t.to_numpy(),p.index.to_numpy()
        predictions.append(p.to_numpy())
    if not predictions:
        raise ValueError('没有所需策略预测')
    return truth,np.stack(predictions),ids


def compare(truth,reference,candidate):
    baseline=summary_metrics(truth,reference)
    own=summary_metrics(truth,candidate)
    base_seed=np.sqrt(np.square(reference-truth[None]).mean(axis=1)).mean(axis=1)
    own_seed=np.sqrt(np.square(candidate-truth[None]).mean(axis=1)).mean(axis=1)
    changes=100*(own_seed/base_seed-1)
    return dict(RMSE_change_pct=100*(own['RMSE']/baseline['RMSE']-1),
        MAE_change_pct=100*(own['MAE']/baseline['MAE']-1),improved_seeds=int((changes<0).sum()),
        seed_min_change_pct=float(changes.min()),seed_max_change_pct=float(changes.max()))


def analyze(run,out):
    verification,predictions=verify(run)
    out.mkdir(parents=True,exist_ok=False)
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    data=load_data(run/'dataset.npz')
    annual,paired,intervals,persistence,quarters,segments=[],[],[],[],[],[]
    predictions['quarter']=pd.to_datetime(predictions.origin_date).dt.quarter
    for (year,strategy,h),part in predictions.groupby(['year','strategy','task_h']):
        train,valid=outer_split(data,year)
        selected=next(v['selected_variant'] for v in selection['selections'] if
                      (v['year'],v['strategy'],v['task_h'])==(year,strategy,h))
        truth,reference,ids=arrays(part,BASELINE)
        np.testing.assert_equal(ids,valid)
        q=data['discharge'][data['positions'][ids]]
        persistent=np.repeat(q[:,None],truth.shape[1],axis=1)
        repeated=np.repeat(persistent[None],reference.shape[0],axis=0)
        for j,horizon in enumerate([h] if strategy=='direct' else range(1,8)):
            persistence.append(dict(year=year,strategy=strategy,task_h=h,h=horizon,
                                    **recompute(truth[:,j],persistent[:,j])))
        for variant in part.variant.unique():
            own_truth,own,own_ids=arrays(part,variant)
            np.testing.assert_equal(own_truth,truth)
            np.testing.assert_equal(own_ids,ids)
            identity=dict(year=year,strategy=strategy,task_h=h,variant=variant,selected=variant==selected)
            annual.append(dict(**identity,**summary_metrics(truth,own)))
            for label,ref in [('plain_log',reference),('persistence',repeated)]:
                values=compare(truth,ref,own)
                paired.append(dict(**identity,reference=label,**values))
                if variant!=BASELINE or label=='persistence':
                    for block in [7,14,30]:
                        low,high=paired_interval(truth,ref,own,block)
                        intervals.append(dict(**identity,reference=label,block_days=block,
                            repeats=600,change_pct_low=low,change_pct_high=high))
        for (variant,seed,horizon),own in part.groupby(['variant','seed','h']):
            threshold=float(np.quantile(data['y'][train,horizon-1],.9))
            for label,mask in [('high',own.truth.to_numpy()>=threshold),('other',own.truth.to_numpy()<threshold)]:
                subset=own[mask]
                values=recompute(subset.truth.to_numpy(),subset.pred.to_numpy())
                segments.append(dict(year=year,strategy=strategy,task_h=h,variant=variant,seed=seed,
                                     h=horizon,segment=label,threshold=threshold,**values))
    for keys,part in predictions.groupby(['year','strategy','task_h','variant','seed','quarter','h']):
        identity=dict(zip(['year','strategy','task_h','variant','seed','quarter','h'],keys))
        quarters.append(dict(**identity,**recompute(part.truth.to_numpy(),part.pred.to_numpy())))
    tables=dict(annual_summary=annual,paired_changes=paired,paired_block_sensitivity=intervals,
                persistence=persistence,quarter_metrics=quarters,error_segments=segments)
    for name,rows in tables.items():
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    for name in ['metrics.csv','selection.json','manifest.json','inner_splits.csv']:
        shutil.copy2(run/name,out/name)
    write_json(out/'verification.json',verification)
    frame=pd.DataFrame(annual)
    pairs=pd.DataFrame(paired)
    lines=['# 任务1至7：策略增强局部重训结果','','日期：2026-11-11。内部开发对照，不替换正式验收报告。','',
        f'完成{verification["fits"]}fit、{verification["models"]}个外层模型，全部恢复独立核验通过。',
        '模型数组和性能评价止于2003年；2004未用于本轮选择或评价。全部内层选择已在外层拟合/预测前冻结。',
        '任务仅含direct h1、direct h7、原生multi；不是完整七任务新交付，也未改变ANN/RF及集成。','',
        '## 1. 固定候选与口径','',
        'last_plain_log为同协议普通90日LSTM；last_mean为末状态与全部状态均值等权平均，不增加可训练参数。',
        'residual为目标变换尺度的Q(t)跳连，末层零初始化，初始预测恰为持久性；不声称残差与普通头初始化相同。',
        '三seed42/43/44，原尺度逐h RMSE平均，再对seed平均，不是扁平RMSE。',
        '各策略由净化历史内层选1/10/30轮；因此是同搜索预算策略比较，不是固定epoch纯消融。','',
        '## 2. 三seed年度结果','',
        '| 年度 | 任务 | 策略 | RMSE | MAE | 内层选中 |',
        '| --- | --- | --- | ---: | ---: | --- |']
    for row in frame.itertuples():
        task=f'h{row.task_h}' if row.strategy=='direct' else 'multi'
        lines.append(f'| {row.year} | {task} | {row.variant} | {row.RMSE:.2f} | {row.MAE:.2f} | {"是" if row.selected else "否"} |')
    lines+=['','## 3. 仅按内层选择的策略表现','','此表不按外层最优值挑选。负变化表示改善。','',
        '| 年度 | 任务 | 内层所选 | 对普通log变化 | 对持久性变化 |',
        '| --- | --- | --- | ---: | ---: |']
    for item in selection['selections']:
        part=pairs[(pairs.year==item['year'])&(pairs.strategy==item['strategy'])&
                   (pairs.task_h==item['task_h'])&(pairs.variant==item['selected_variant'])].set_index('reference')
        task=f'h{item["task_h"]}' if item['strategy']=='direct' else 'multi'
        lines.append(f'| {item["year"]} | {task} | {item["selected_variant"]} | '
                     f'{part.loc["plain_log","RMSE_change_pct"]:+.2f}% | {part.loc["persistence","RMSE_change_pct"]:+.2f}% |')
    lines+=['','## 4. 证据和限制','',
        '- [逐h/逐seed R²、RMSE、MAE](metrics.csv)',
        '- [全部内层选择与CV汇总](selection.json)',
        '- [年度汇总](annual_summary.csv)',
        '- [配对变化](paired_changes.csv)',
        '- [7/14/30日块重采样](paired_block_sensitivity.csv)',
        '- [季度结果](quarter_metrics.csv)',
        '- [严格持久性](persistence.csv)',
        '- [训练P90高流量分段](error_segments.csv)',
        '- [独立核验](verification.json)',
        '- [运行方案与指导书边界](../../任务1至7+策略增强与局部重训方案_2026-11-11.md)','',
        '时间块同步重采样所有seed和h，600次，报告2.5%～97.5%经验分位；这些不是独立未来年份的泛化保证。',
        '两个外层年份均是已查看的开发历史；最早内折训练样本有限。更多seed不是更多独立年份。',
        '不得按外层逐h、季度或seed拼最优成绩；固定策略的局部好结果不等于内层选策略整条流程稳定有效。',
        '标准课程模型、真Mean和Stacking仍须保留；本轮增强单列，不替代规定任务。',
        '原CSV、正式Word、旧run均未改，未commit/push。','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'analysis_source').mkdir()
    for path in [Path(__file__),ROOT/'code/verify_runoff_strategy.py',ROOT/'code/analyze_runoff_controlled.py']:
        shutil.copy2(path,out/'analysis_source'/path.name)
    write_json(out/'analysis_manifest.json',dict(status='completed',run_id=run.name,
        run_manifest_sha256=sha256(run/'manifest.json'),bootstrap_repeats=600,bootstrap_seed=20261111,
        block_days=[7,14,30],files={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(frame.to_string(index=False))
    print('仅内层选择策略的配对变化：')
    print(pairs[pairs.selected].to_string(index=False))
    print('导出完成：',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    with threadpool_limits(limits=4):
        analyze(ROOT/'output/acceptance3_runs'/args.run_id,args.out)


if __name__=='__main__':
    main()
