# -*- coding: utf-8 -*-
"""元层融合提升实验：不重训基模型，仅用 OOF 重训元层，2004 测试集诚实评估。

原则：
- 参数选择只看 OOF（训练期预测），不看 2004 测试集。
- 元层候选在 OOF 上做时间前向(因果)交叉验证选参，选中后在完整 OOF 上重训，
  再应用到 2004 测试集基模型预测，重算 R2/RMSE/MAE。
- 只做诚实对比，不伪造、不"美化"数据。
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.optimize import nnls

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "验收" / "Acceptance 3" / "data"

FAMILIES = ["ANN", "RF", "LSTM"]
HORIZONS = list(range(1, 8))


def r2_rmse_mae(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    sse = np.square(p - y).sum(axis=0)
    sst = np.square(y - y.mean(axis=0)).sum(axis=0)
    r2 = 1 - sse / np.where(sst == 0, np.nan, sst)
    rmse = np.sqrt(np.square(p - y).mean(axis=0))
    mae = np.abs(p - y).mean(axis=0)
    return r2, rmse, mae


def build_matrix(df):
    """返回 (ids, 21 列矩阵 [ANN_h1..h7, RF_h1..h7, LSTM_h1..h7], truth[n,7])"""
    fams = {f: df[df.model == f].pivot(index="sample_id", columns="h", values="pred").sort_index()
            for f in FAMILIES}
    ids = fams[FAMILIES[0]].index.to_numpy()
    for f in FAMILIES:
        assert np.array_equal(ids, fams[f].index.to_numpy()), "样本 id 不对齐"
        assert list(fams[f].columns) == HORIZONS, "h 列不对齐"
    matrix = np.column_stack([fams[f].to_numpy() for f in FAMILIES])
    truth = df[df.model == FAMILIES[0]].pivot(index="sample_id", columns="h", values="truth").sort_index().to_numpy()
    return ids, matrix, truth


def oof_causal_splits(ids, days, n_splits=3):
    """时间前向因果切分：验证段只含时间靠后的样本。"""
    order = np.argsort(days)
    n = len(order)
    denom = n_splits + 1
    splits = []
    for k in range(1, denom):
        cut = int(round(n * k / denom))
        ti, vi = order[:cut], order[cut:]
        if len(ti) and len(vi):
            splits.append((ti, vi))
    return splits


def ridge_raw(train_x, train_y, test_x, alpha):
    return Ridge(alpha=alpha).fit(train_x, train_y).predict(test_x)


def ridge_log(train_x, train_y, test_x, alpha):
    tx = np.log(np.maximum(train_x, 1e-12))
    ty = np.log(np.maximum(train_y, 1e-12))
    m = Ridge(alpha=alpha).fit(tx, ty)
    return np.exp(np.clip(m.predict(np.log(np.maximum(test_x, 1e-12))), -50, 50))


def convex_weights(train_x, train_y, normalize):
    """对每个输出列独立做非负最小二乘；normalize=True 时归一化为凸组合。"""
    k = train_x.shape[1]
    W = np.zeros((k, train_y.shape[1]))
    for j in range(train_y.shape[1]):
        w, _ = nnls(train_x, train_y[:, j])
        s = w.sum()
        W[:, j] = (w / s if s > 1e-12 else np.ones(k) / k) if normalize else w
    return W


def build_candidates():
    """返回 [(tag, fit_predict(tx, ty, px)->pred), ...]"""
    out = []
    # 单模型 / 简单平均基底
    out.append(("RF-only", lambda tx, ty, px: px[:, 7:14]))
    out.append(("Mean", lambda tx, ty, px: px.reshape(len(px), 3, 7).mean(axis=1)))
    # 非负凸组合 / 非负线性
    out.append(("convex", lambda tx, ty, px: px @ convex_weights(tx, ty, True)))
    out.append(("nnls", lambda tx, ty, px: px @ convex_weights(tx, ty, False)))
    # 原尺度 Ridge，加大 alpha 范围（大 alpha 收缩到训练均值）
    for a in [10., 100., 1000., 1e4, 1e5]:
        out.append((f"ridge_raw_a{a:g}", lambda tx, ty, px, a=a: ridge_raw(tx, ty, px, a)))
    # log 尺度 Ridge
    for a in [1., 10., 100., 1000., 1e4]:
        out.append((f"ridge_log_a{a:g}", lambda tx, ty, px, a=a: ridge_log(tx, ty, px, a)))
    # 向 RF 收缩
    for a in [100., 1000., 1e4]:
        for s in [0.25, 0.5]:
            out.append((f"shrinkRF_a{a:g}_s{s:g}",
                        lambda tx, ty, px, a=a, s=s: px[:, 7:14] + s * (ridge_raw(tx, ty, px, a) - px[:, 7:14])))
    # 向 Mean 收缩
    for a in [100., 1000., 1e4]:
        for s in [0.25, 0.5]:
            out.append((f"shrinkMean_a{a:g}_s{s:g}",
                        lambda tx, ty, px, a=a, s=s:
                        px.reshape(len(px), 3, 7).mean(axis=1)
                        + s * (ridge_raw(tx, ty, px, a) - px.reshape(len(px), 3, 7).mean(axis=1))))
    return out


def run_strategy(name, oof_ids, oof_days, oof_x, oof_y, test_x, test_y):
    splits = oof_causal_splits(oof_ids, oof_days)
    cands = build_candidates()

    cv_scores = {}
    for tag, f in cands:
        rmses = []
        for ti, vi in splits:
            p = np.asarray(f(oof_x[ti], oof_y[ti], oof_x[vi]), float).reshape(len(vi), -1)
            rmses.append(np.sqrt(np.square(p - oof_y[vi]).mean()))
        cv_scores[tag] = float(np.mean(rmses))
    best_tag = min(cv_scores, key=cv_scores.get)

    results = []
    for tag, f in cands:
        p = np.asarray(f(oof_x, oof_y, test_x), float).reshape(len(test_x), 7)
        r2, rmse, mae = r2_rmse_mae(test_y, p)
        for i, h in enumerate(HORIZONS):
            results.append(dict(strategy=name, meta=tag, h=h,
                                R2=float(r2[i]), RMSE=float(rmse[i]), MAE=float(mae[i]),
                                cv_rmse=cv_scores[tag]))
    ranked = sorted(cv_scores.items(), key=lambda kv: kv[1])
    return results, best_tag, ranked


def main():
    test = pd.read_csv(DATA / "任务1至7+主seed42逐日预测与来源.csv")
    test = test[test.seed == 42]
    oof_direct = pd.read_csv(DATA / "任务5+第二轮三seed因果OOF预测.csv")
    oof_direct = oof_direct[oof_direct.seed == 42]
    oof_multi = pd.read_csv(DATA / "任务7+第三轮三seed因果OOF预测.csv")
    oof_multi = oof_multi[oof_multi.seed == 42]

    all_rows = []
    summary = {}
    for strategy, oof in [("direct", oof_direct), ("multi", oof_multi)]:
        if strategy == "direct":
            # 任务5 CSV 同时含 direct/multi 两种策略，取 direct 部分
            oof = oof[oof.strategy == "direct"]
        tst = test[test.strategy == strategy]
        oof_ids, oof_x, oof_y = build_matrix(oof)
        test_ids, test_x, test_y = build_matrix(tst)
        oday = pd.to_datetime(oof[oof.model == "ANN"].groupby("sample_id")["origin_date"].first())
        oof_days = oday.reindex(oof_ids).astype("int64").to_numpy()
        rows, best_tag, ranked = run_strategy(strategy, oof_ids, oof_days, oof_x, oof_y, test_x, test_y)
        all_rows.extend(rows)
        summary[strategy] = dict(best_tag=best_tag, top=ranked[:8],
                                 n_oof=len(oof_ids), n_test=len(test_ids))

    frame = pd.DataFrame(all_rows)

    print("=" * 110)
    print("样本规模")
    for s, v in summary.items():
        print(f"  {s}: OOF={v['n_oof']}  测试={v['n_test']}")
    print()
    print("OOF 因果 CV 选参排序（只看 OOF，不看 2004）：")
    for s, v in summary.items():
        print(f"  [{s}]  最优={v['best_tag']}")
        for tag, rmse in v["top"]:
            print(f"        cv_rmse={rmse:8.1f}  {tag}")

    rel = ["RF-only", "Mean", "convex", "nnls", "ridge_raw_a10", "ridge_raw_a100",
           "ridge_raw_a1000", "ridge_raw_a1e+04", "ridge_raw_a1e+05",
           "ridge_log_a1", "ridge_log_a10", "ridge_log_a100",
           "shrinkRF_a100_s0.25", "shrinkRF_a1000_s0.5",
           "shrinkMean_a1000_s0.25", "shrinkMean_a1000_s0.5"]
    rel = [c for c in rel if c in frame.meta.unique()]

    cross = frame[frame.meta.isin(rel)].pivot_table(index=["strategy", "h"], columns="meta", values="R2").round(4)
    print()
    print("=" * 110)
    print("2004 测试集 R2（strategy x h，重点候选）")
    print(cross.to_string())

    existing = pd.read_csv(DATA / "任务1至7+主seed42完整指标与来源.csv")
    existing = existing[existing.model.isin(["Mean", "RF", "Stacking", "SelectedFusion"])]
    ex = existing.pivot_table(index=["strategy", "h"], columns="model", values="R2").round(4)
    print()
    print("现有报告对照（Mean / RF / Stacking / SelectedFusion）")
    print(ex.to_string())

    # 每个 strategy 每个 h 的 OOF 选参最优候选在测试集的表现 vs 现有 Stacking
    print()
    print("=" * 110)
    print("OOF 选出的单一最优元层（整体）在测试集 R2，对比现有 Stacking：")
    for s, v in summary.items():
        best = v["best_tag"]
        sub = frame[(frame.strategy == s) & (frame.meta == best)][["h", "R2", "RMSE", "MAE"]].round(4)
        print(f"\n  [{s}] 最优元层={best}")
        print(sub.to_string(index=False))

    out_csv = DATA / "任务5及7+元层提升实验_2004测试指标.csv"
    frame.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print()
    print("已写出：", out_csv)


if __name__ == "__main__":
    main()