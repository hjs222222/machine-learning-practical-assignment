# -*- coding: utf-8 -*-
"""元层修复部署：把 direct 新 Stacking(shrinkRF a100 s0.25) 落地成逐日预测 + 指标。

严格复用 acceptance3_meta_boost 的拟合逻辑（无标准化的 Ridge + 向 RF 收缩 0.25），
OOF 选参已在 meta_boost 完成（shrinkRF_a100_s0.25 在 OOF 因果 CV 上最优）。
这里只在完整 direct OOF 上重训，应用到 2004 测试集，写出可交付 CSV。
multi 结论：OOF 选参为 RF-only（Stacking 无增益），不另造元层。
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "验收" / "Acceptance 3" / "data"

FAMILIES = ["ANN", "RF", "LSTM"]
HORIZONS = list(range(1, 8))


def build_matrix(df):
    fams = {f: df[df.model == f].pivot(index="sample_id", columns="h", values="pred").sort_index()
            for f in FAMILIES}
    ids = fams[FAMILIES[0]].index.to_numpy()
    for f in FAMILIES:
        assert np.array_equal(ids, fams[f].index.to_numpy())
        assert list(fams[f].columns) == HORIZONS
    matrix = np.column_stack([fams[f].to_numpy() for f in FAMILIES])
    truth = df[df.model == FAMILIES[0]].pivot(index="sample_id", columns="h", values="truth").sort_index().to_numpy()
    return ids, matrix, truth


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    sse = np.square(p - y).sum(axis=0)
    sst = np.square(y - y.mean(axis=0)).sum(axis=0)
    r2 = 1 - sse / sst
    rmse = np.sqrt(np.square(p - y).mean(axis=0))
    mae = np.abs(p - y).mean(axis=0)
    return r2, rmse, mae


def main():
    # direct OOF（seed42）拟合 shrinkRF_a100_s0.25
    oof = pd.read_csv(DATA / "任务5+第二轮三seed因果OOF预测.csv")
    oof = oof[(oof.seed == 42) & (oof.strategy == "direct")]
    oof_ids, oof_x, oof_y = build_matrix(oof)

    test = pd.read_csv(DATA / "任务1至7+主seed42逐日预测与来源.csv")
    test = test[test.seed == 42]
    tst = test[test.strategy == "direct"]
    test_ids, test_x, test_y = build_matrix(tst)

    ridge = Ridge(alpha=100.).fit(oof_x, oof_y)
    rf_cols = test_x[:, 7:14]
    pred = rf_cols + 0.25 * (ridge.predict(test_x) - rf_cols)  # 向 RF 收缩 0.25

    r2, rmse, mae = metrics(test_y, pred)
    neg = int((pred < 0).sum())
    print("[direct] shrinkRF_a100_s0.25 测试集指标：")
    for i, h in enumerate(HORIZONS):
        print(f"  h={h}: R2={r2[i]:.4f}  RMSE={rmse[i]:.2f}  MAE={mae[i]:.2f}")
    print(f"  负预测总数={neg} / {pred.size}")
    print(f"  平均R2={r2.mean():.4f}（现有 direct Stacking 平均R2=0.1253）")

    # 落地逐日预测：对齐现有交付格式
    meta = test[(test.strategy == "direct") & (test.model == "ANN")][
        ["sample_id", "origin_date", "target_date", "h"]].copy()
    meta = meta.sort_values(["sample_id", "h"])
    pred_flat = pred.flatten()  # 按 sample_id 排序、h 升序展开
    # build_matrix 按 sample_id sort_index、h=1..7 升序，与 meta 排序一致
    ann = test[(test.strategy == "direct") & (test.model == "ANN")].pivot(
        index="sample_id", columns="h", values=["pred", "truth", "origin_date", "target_date"])
    # 直接构造：按 test_ids 的顺序展开（sort_index 后的 sample_id）
    rows = []
    order_ids = test_ids
    pmat = pred  # [n_test, 7] 与 order_ids 对齐
    dates = ann["origin_date"].sort_index()  # 需要 origin_date per sample
    targets = ann["target_date"].sort_index()
    for i, sid in enumerate(order_ids):
        for j, h in enumerate(HORIZONS):
            rows.append(dict(
                run_id="acceptance3_meta_deploy",
                fold="",
                sample_id=int(sid),
                origin_date=str(dates.iloc[i, j]),
                target_date=str(targets.iloc[i, j]),
                h=h,
                strategy="direct",
                model="Stacking",
                truth=float(test_y[i, j]),
                pred=float(pmat[i, j]),
                seed=42,
                source_run="acceptance3_meta_boost_shrinkRF_a100_s0.25",
            ))
    frame = pd.DataFrame(rows)
    out_pred = DATA / "任务5+新Stacking_shrinkRF逐日预测.csv"
    frame.to_csv(out_pred, index=False, encoding="utf-8-sig")
    print("已写出：", out_pred)

    # 对比指标表
    existing = pd.read_csv(DATA / "任务1至7+主seed42完整指标与来源.csv")
    old = existing[(existing.strategy == "direct") & (existing.model == "Stacking")][
        ["h", "R2", "RMSE", "MAE"]].copy()
    old.columns = ["h", "R2_old", "RMSE_old", "MAE_old"]
    new = pd.DataFrame(dict(h=HORIZONS, R2_new=r2, RMSE_new=rmse, MAE_new=mae))
    cmp = new.merge(old, on="h")
    cmp["R2_增益"] = cmp.R2_new - cmp.R2_old
    cmp = cmp[["h", "R2_old", "R2_new", "R2_增益", "RMSE_old", "RMSE_new", "MAE_old", "MAE_new"]]
    cmp.to_csv(DATA / "任务5+Stacking元层修复对比.csv", index=False, encoding="utf-8-sig")
    print("已写出：", DATA / "任务5+Stacking元层修复对比.csv")
    print(cmp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()