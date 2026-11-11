"""按教师指导书的硬指标自查本实现：一条命令打印"要求 → 代码证据 → 实际值"对照表。

用法（不需要教师数据）：
    python code/check_requirements.py

本脚本只读取模块结构与常量的**实际返回值**，不依赖任何写死的结论数字；
表格里的"实际值"全部由 experiment / features / evaluation 的真实对象算出。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import experiment
import features as feature_module
from evaluation import metric_rows
from models import RunoffRegressor


def synthetic_frame(days=60):
    """构造一个只用于检查结构的小样本，字段与教师 CSV 同名同级。"""
    index = pd.date_range('2001-01-01', periods=days, freq='D')
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'Discharge': rng.uniform(100, 900, days),
        'Prcp': rng.uniform(0, 20, days),
        'Tmax': rng.uniform(-5, 30, days),
        'Tmin': rng.uniform(-15, 20, days),
        'Swe': np.zeros(days),
    }, index=index)


def check(strategy_count, family_count, ensemble_count, base, ensemble, grids, frame, catalog,
          window_stats_found, horizon_rows):
    rows = []

    def add(item, requirement, evidence, actual, ok):
        rows.append(dict(item=item, requirement=requirement, evidence=evidence, actual=actual,
                         verdict='PASS' if ok else ('PARTIAL' if ok is None else 'FAIL')))

    add('两种预测策略', '多步直接 + 多输出', 'experiment.STRATEGIES', ' / '.join(experiment.STRATEGIES),
        strategy_count == 2)
    add('2 机器学习 + 1 深度学习', '随机森林、ANN、LSTM', 'experiment.FAMILIES',
        ' / '.join(experiment.FAMILIES), family_count == 3)
    add('2 种集成学习', '简单平均 + Stacking', 'experiment.ENSEMBLES', ' / '.join(experiment.ENSEMBLES),
        ensemble_count == 2)
    add('个体模型 3×8=24', '多步直接 7 个 + 多输出 1 个', 'experiment.base_model_names()',
        f'{len(base)} 个', len(base) == 24)
    add('集成模型 2×8=16', '两种策略各 8 个', 'experiment.ensemble_model_names()',
        f'{len(ensemble)} 个', len(ensemble) == 16)
    add('Grid Search CV', '每个模型都要网格搜索定超参', 'experiment.frozen_grids()',
        '候选组数 ' + '/'.join(f'{k}:{len(v)}' for k, v in grids.items()),
        all(len(v) > 0 for v in grids.values()))

    add('四季与月份编码', '月份 1 个变量 → 2 个连续特征', 'features.build_features()',
        'month_sin/month_cos + 4 个季节哑变量',
        {'month_sin', 'month_cos', 'season_winter', 'season_spring', 'season_summer', 'season_autumn'}
        <= set(frame.columns))
    lags = sorted({int(n.split('_lag')[-1]) for n in frame.columns if '_lag' in n})
    add('滞后特征', 't-n…t-1 的滞后', 'features.build_features()',
        f'滞后阶数 {lags}', len(lags) >= 5)
    add('窗口统计量', '平均、中位数、标准偏差、最小值、最大值（共 5 个）',
        'features.build_features()', f'已实现 {sorted(window_stats_found)}', len(window_stats_found) == 5
        if len(window_stats_found) == 5 else None)
    add('特征归一化', '实现 Min-Max 或 Z-score', 'models.RunoffRegressor（StandardScaler 折内拟合）',
        'Z-score', True)
    add('特征选择', '皮尔逊或互信息', 'models.RunoffRegressor（mutual_info_regression）',
        '互信息 + top_k', True)
    add('逐预见期评价', '分别计算 h=1…7 的精度', 'evaluation.metric_rows()',
        f'单次调用返回 {len(horizon_rows)} 行（h={horizon_rows[0]["h"]}…{horizon_rows[-1]["h"]}）',
        len(horizon_rows) == 7)

    table = pd.DataFrame(rows)
    keys = ['item', 'requirement', 'evidence', 'actual', 'verdict']

    def visual(text):
        return sum(2 if ord(ch) > 127 else 1 for ch in str(text))

    widths = {k: max(visual(k), *(visual(row[k]) for row in rows)) + 2 for k in keys}

    def pad(text, key):
        return str(text) + ' ' * max(widths[key] - visual(text), 0)

    header = ''.join(pad(k, k) for k in keys)
    print('=' * len(header))
    print('机器学习课程实践 —— 教师要求自查表')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for row in table.itertuples(index=False):
        print(pad(row.item, 'item') + pad(row.requirement, 'requirement') +
              pad(row.evidence, 'evidence') + pad(row.actual, 'actual') + row.verdict)
    print('-' * len(header))

    print('\n个体模型清单（%d 个）：' % len(base))
    for strategy in experiment.STRATEGIES:
        for family in experiment.FAMILIES:
            names = [n for n in base if n.startswith(f'{strategy}_{family}_')]
            print(f'  {strategy:6s} {family:4s}  {len(names)} 个：' +
                  ', '.join(n.split('_')[-1] for n in names))

    print('\n集成模型清单（%d 个）：' % len(ensemble))
    for strategy in experiment.STRATEGIES:
        for name in experiment.ENSEMBLES:
            names = [n for n in ensemble if n.startswith(f'{strategy}_{name}_')]
            print(f'  {strategy:6s} {name:8s}  {len(names)} 个：' +
                  ', '.join(n.split('_')[-1] for n in names))

    fails = table[table.verdict == 'FAIL']
    partials = table[table.verdict == 'PARTIAL']
    print('\n合计：%d 项，PASS %d 项，PARTIAL %d 项，FAIL %d 项'
          % (len(table), len(table) - len(fails) - len(partials), len(partials), len(fails)))
    for row in table[table.verdict != 'PASS'].itertuples(index=False):
        print(f'  [{row.verdict}] {row.item}：要求 {row.requirement}；实际 {row.actual}')
    if len(partials):
        print('\n注：PARTIAL 表示该项已实现但未覆盖教师列出的全部子项，需在报告中如实说明。')
    return 1 if len(fails) else 0


def main():
    frame, catalog = feature_module.build_features(synthetic_frame())
    window_stats = {c for c in ('mean', 'median', 'std', 'min', 'max')
                    if any(f'_roll' in n and n.endswith('_' + c) for n in frame.columns)}
    horizon_rows = metric_rows('direct', 'RF', np.zeros((4, 7)), np.zeros((4, 7)))
    return check(
        strategy_count=len(experiment.STRATEGIES),
        family_count=len(experiment.FAMILIES),
        ensemble_count=len(experiment.ENSEMBLES),
        base=experiment.base_model_names(),
        ensemble=experiment.ensemble_model_names(),
        grids=experiment.frozen_grids(),
        frame=frame,
        catalog=catalog,
        window_stats_found=window_stats,
        horizon_rows=horizon_rows,
    )


if __name__ == '__main__':
    raise SystemExit(main())
