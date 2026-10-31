# -*- coding: utf-8 -*-
"""任务1至7+正式运行独立核验，不启动训练，不修改预测。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from acceptance3_train import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run = ROOT / "output/acceptance3_runs" / args.run_id
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    metrics = pd.read_csv(run / "metrics.csv")
    predictions = pd.read_csv(run / "test_predictions.csv", parse_dates=["origin_date", "target_date"])
    oof = pd.read_csv(run / "oof_predictions.csv", parse_dates=["origin_date", "target_date"])
    splits = pd.read_csv(run / "time_splits.csv")
    assert len(metrics) == 70
    required = metrics[metrics.model.isin(["LSTM", "Mean", "Stacking"])]
    assert len(required) == 42
    assert np.isfinite(metrics[["R2", "RMSE", "MAE"]]).all().all()
    assert predictions.target_date.dt.year.eq(2004).all()
    assert (predictions.target_date - predictions.origin_date).dt.days.eq(predictions.h).all()
    assert not predictions.duplicated(["strategy", "model", "h", "origin_date"]).any()
    assert not oof.duplicated(["strategy", "model", "h", "origin_date"]).any()
    base_keys = None
    for (strategy, model, h), frame in predictions.groupby(["strategy", "model", "h"]):
        frame = frame.sort_values("origin_date")
        assert len(frame) == 359
        key = frame[["origin_date", "target_date", "truth"]].reset_index(drop=True)
        if base_keys is None:
            base_keys = {}
        if h not in base_keys:
            base_keys[h] = key
        else:
            pd.testing.assert_frame_equal(key, base_keys[h])
        row = metrics[(metrics.strategy == strategy) & (metrics.model == model) & (metrics.h == h)].iloc[0]
        values = [r2_score(frame.truth, frame.pred),
                  np.sqrt(np.mean((frame.truth - frame.pred) ** 2)),
                  np.mean(np.abs(frame.truth - frame.pred))]
        np.testing.assert_allclose(values, row[["R2", "RMSE", "MAE"]].to_numpy(float), rtol=1e-10, atol=1e-8)
    for strategy in ("direct", "multi"):
        pivot = predictions[predictions.strategy == strategy].pivot(
            index=["origin_date", "h"], columns="model", values="pred")
        np.testing.assert_allclose(pivot.Mean, pivot[["ANN", "RF", "LSTM"]].mean(axis=1), rtol=1e-12)
    for _, row in splits.iterrows():
        assert pd.Timestamp(row.max_train_target) < pd.Timestamp(row.validation_origin_start)
        if row.stage == "oof":
            part = oof[oof.fold == row.fold]
            assert part.origin_date.min() == pd.Timestamp(row.validation_origin_start)
            assert part.origin_date.max() == pd.Timestamp(row.validation_origin_end)
    records = [json.loads(p.read_text(encoding="utf-8")) for p in (run / "training_history").glob("*.json")]
    with np.load(run / "dataset.npz", allow_pickle=False) as data:
        days = data["origin_days"]
        for record in records:
            if record["phase"].startswith("calibration"):
                assert days[record["last_id"]] + 7 <= np.datetime64("2001-12-31").astype("int64")
            if "cv_score" in record:
                assert days[record["last_id"]] + record["horizon"] < days[record["validation_first_id"]]
            if "early_validation_first_id" in record:
                assert days[record["early_train_last_id"]] + record["horizon"] < days[record["early_validation_first_id"]]
            if record["phase"].startswith("final"):
                assert days[record["last_id"]] + 7 <= np.datetime64("2003-12-31").astype("int64")
    coef = pd.read_csv(run / "stacking_coefficients.csv")
    assert len(coef[coef.strategy == "direct"]) == 21
    assert len(coef[coef.strategy == "multi"]) == 147
    restored = json.loads((run / "model_restoration_checks.json").read_text(encoding="utf-8"))
    assert len(restored) == 24
    cv = pd.read_csv(run / "cv_results/all_candidates.csv")
    expected = 24 if manifest["smoke"] else 192
    assert len(cv) == expected
    assert cv.filter(regex=r"split\d+_test_score").notna().all().all()
    assert len(records) == (168 if manifest["smoke"] else 696)
    # 使用已保存元模型独立重建Stacking预测，核对输入列序及落盘结果。
    import joblib
    for strategy in ("direct", "multi"):
        selected = predictions[predictions.strategy == strategy]
        model_matrices = []
        for family in ("ANN", "RF", "LSTM"):
            model_matrices.append(selected[selected.model == family].pivot(
                index="origin_date", columns="h", values="pred").sort_index().to_numpy())
        matrix = np.column_stack(model_matrices)
        saved_stacking = selected[selected.model == "Stacking"].pivot(
            index="origin_date", columns="h", values="pred").sort_index().to_numpy()
        if strategy == "multi":
            meta = joblib.load(run / "models/stacking_multi_h7.joblib")
            assert meta[-1].coef_.shape == (7, 21)
            np.testing.assert_allclose(meta.predict(matrix), saved_stacking, rtol=1e-10, atol=1e-7)
        else:
            for h in range(1, 8):
                meta = joblib.load(run / "models" / f"stacking_direct_h{h}.joblib")
                np.testing.assert_allclose(meta.predict(matrix[:, [h-1, h+6, h+13]]),
                                           saved_stacking[:, h-1], rtol=1e-10, atol=1e-7)
    result = {"run_id": args.run_id, "status": "passed", "metric_rows": 70,
              "required_rows": 42, "test_origin_n": 359, "test_prediction_rows": len(predictions),
              "oof_prediction_rows": len(oof), "cv_candidate_rows": len(cv),
              "actual_fit_records": len(records), "restored_models": len(restored),
              "checks": ["逐h独立重算指标", "目标日期与真值对齐", "原尺度等权平均", "训练/CV/内部早停边界",
                         "OOF覆盖与唯一性", "多输出元层21到7系数形状", "模型恢复记录", "完整网格成绩"]}
    (run / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
