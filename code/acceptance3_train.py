# -*- coding: utf-8 -*-
"""任务1至7+第三次验收训练入口，2026-10-31，第1次修改版。

独立运行目录；校准后冻结参数；因果OOF；保存全部结果与推理证据。
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from acceptance3_model import RunoffRegressor, load_dataset, physical_scorer, restore_model
from feature_engineering import build_features

ROOT = Path(__file__).resolve().parents[1]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def prepare(run):
    source = ROOT / "Teacher Resources & Requirements/01047000.csv"
    raw = pd.read_csv(source, parse_dates=["Date"])
    assert raw["Date"].is_unique and raw["Date"].is_monotonic_increasing
    assert raw["Date"].diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert np.isfinite(raw.select_dtypes("number").to_numpy()).all()
    assert (raw.Discharge > 0).all()
    engineered, names = build_features(raw)
    # 保留原项目的特征设计，不根据2004分数新增特征。
    positions = np.arange(29, len(engineered) - 7)
    origins = engineered.Date.to_numpy()[positions]
    targets = np.stack([engineered.Discharge.to_numpy()[positions + h] for h in range(1, 8)], axis=1)
    x = engineered[names].to_numpy(float)
    assert np.isfinite(x).all() and np.isfinite(targets).all()
    days = origins.astype("datetime64[D]").astype("int64")
    path = run / "dataset.npz"
    np.savez_compressed(path, features=x, positions=positions, y=targets,
                        origin_days=days, origin_dates=origins.astype("datetime64[D]").astype(str),
                        feature_names=np.asarray(names))
    last_days = days + 7
    calibration = np.flatnonzero(last_days <= np.datetime64("2001-12-31").astype("int64"))
    train = np.flatnonzero(last_days <= np.datetime64("2003-12-31").astype("int64"))
    test = np.flatnonzero((days >= np.datetime64("2004-01-01").astype("int64")) &
                          (last_days <= np.datetime64("2004-12-31").astype("int64")))
    oof_ids = np.flatnonzero((days >= np.datetime64("2002-01-01").astype("int64")) &
                             (last_days <= np.datetime64("2003-12-31").astype("int64")))
    folds = []
    for block in np.array_split(oof_ids, 3):
        prior = train[last_days[train] < days[block[0]]]
        assert last_days[prior[-1]] < days[block[0]]
        folds.append((prior, block))
    split_rows = []
    def add_split(stage, fold, ti, vi):
        assert last_days[ti[-1]] < days[vi[0]]
        split_rows.append({"stage": stage, "fold": fold,
            "n_train": len(ti), "n_validation": len(vi),
            "train_origin_start": str(np.datetime64(int(days[ti[0]]), "D")),
            "train_origin_end": str(np.datetime64(int(days[ti[-1]]), "D")),
            "max_train_target": str(np.datetime64(int(last_days[ti[-1]]), "D")),
            "validation_origin_start": str(np.datetime64(int(days[vi[0]]), "D")),
            "validation_origin_end": str(np.datetime64(int(days[vi[-1]]), "D"))})
    for i, (ti, vi) in enumerate(TimeSeriesSplit(n_splits=3, gap=7).split(calibration)):
        add_split("calibration", i, calibration[ti], calibration[vi])
    for i, (ti, vi) in enumerate(folds):
        add_split("oof", i, ti, vi)
    pd.DataFrame(split_rows).to_csv(run / "time_splits.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"sample_id": np.arange(len(days)), "origin_date": origins,
                  "last_target_date": (days + 7).astype("datetime64[D]")}).to_csv(
                      run / "sample_index.csv", index=False, encoding="utf-8-sig")
    info = {"raw_rows": len(raw), "engineered_rows": len(engineered), "features": names,
            "n_features": len(names), "n_calibration": len(calibration), "n_train": len(train),
            "n_test": len(test), "n_oof": len(oof_ids), "first_origin": str(origins[0]),
            "train_first_origin": str(origins[train[0]]), "train_last_origin": str(origins[train[-1]]),
            "calibration_last_origin": str(origins[calibration[-1]]),
            "test_first_origin": str(origins[test[0]]), "test_last_origin": str(origins[test[-1]])}
    dump(run / "data_summary.json", info)
    return path, targets, calibration, train, test, folds


def prediction_rows(run_id, data, ids, predictions, strategy, model, fold=None):
    pred = np.asarray(predictions).reshape(len(ids), -1)
    truth = data["y"][ids]
    rows = []
    for j in range(7):
        for i, sample in enumerate(ids):
            rows.append({"run_id": run_id, "fold": fold,
                         "sample_id": int(sample), "origin_date": data["origin_dates"][sample],
                         "target_date": str(np.datetime64(int(data["origin_days"][sample]) + j + 1, "D")),
                         "h": j + 1, "strategy": strategy, "model": model,
                         "truth": float(truth[i, j]), "pred": float(pred[i, j])})
    return rows


def metrics_table(rows):
    df = pd.DataFrame(rows)
    metrics = []
    for (strategy, model, h), part in df.groupby(["strategy", "model", "h"], sort=False):
        truth, pred = part.truth.to_numpy(), part.pred.to_numpy()
        metrics.append({"strategy": strategy, "model": model, "h": h,
                        "R2": r2_score(truth, pred), "RMSE": np.sqrt(np.mean((truth - pred) ** 2)),
                        "MAE": mean_absolute_error(truth, pred), "n": len(truth),
                        "negative_predictions": int((pred < 0).sum())})
    return pd.DataFrame(metrics)


def run_training(run, smoke=False, device="cpu"):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    path, targets, calibration, train, test, folds = prepare(run)
    data = load_dataset(str(path))
    grids = {
        "LSTM": {"units": [(64,), (64, 32)], "dropout": [.2, .3], "lr": [.001, .0005]},
        "ANN": {"hidden_layer_sizes": [(64,), (64, 32)], "alpha": [1e-5, .01],
                "learning_rate_init": [.001, .0005]},
        "RF": {"n_estimators": [300, 500], "max_depth": [10, 20], "min_samples_leaf": [1, 5]}}
    if smoke:
        grids = {"LSTM": {"units": [(8,)], "dropout": [.2], "lr": [.001]},
                 "ANN": {"hidden_layer_sizes": [(8,)], "alpha": [.01], "learning_rate_init": [.001]},
                 "RF": {"n_estimators": [5], "max_depth": [3], "min_samples_leaf": [2]}}
    cfg = {"grids": grids, "smoke": smoke, "seed": 42, "seq_len": 30, "top_k": 50,
           "cv_folds": 2 if smoke else 3, "gap": 7, "max_epochs": 2 if smoke else 80,
           "patience": 6, "batch_size": 32, "score": "negative_mean_per_horizon_original_scale_RMSE",
           "calibration_target_end": "2001-12-31", "train_target_end": "2003-12-31",
           "test_origin_start": "2004-01-01", "test_target_end": "2004-12-31",
           "feature_selection_multi": "mutual_information_with_ln_h1",
           "feature_selection_direct": "mutual_information_with_ln_h",
           "lstm_epoch_policy": "inner_purged_15pct_early_stop_then_refit; frozen_median_selected_cv_epochs_for_oof_final",
           "stacking_alpha": 1.0, "stacking_scale": "original_discharge", "clip_negative": False,
           "ann_max_iter": 10 if smoke else 1000, "device": device,
           "common_training_rows": "all_7_targets_in_training_range_even_for_direct",
           "test_reuse_limitation": "2004已被历史实验查看，本次为冻结协议后的回溯测试，非全新盲测"}
    dump(run / "config.json", cfg)
    best_records, cv_frames = [], []
    tasks = [("direct", h) for h in range(1, 8)] + [("multi", 7)]
    frozen = {}
    # 校准完成的检查点恢复时仍重建完整清单，后续阶段不读取测试分数。
    print("阶段A：仅在2000—2001目标范围校准选参", flush=True)
    for strategy, h in tasks:
        y = targets[:, h - 1] if strategy == "direct" else targets
        for family in ("LSTM", "ANN", "RF"):
            label = f"{strategy}_{family}_h{h}"
            cache = run / "checkpoints" / f"calibration_{label}.json"
            if cache.exists():
                record = json.loads(cache.read_text(encoding="utf-8"))
                frozen[label] = record.pop("frozen_params")
                best_records.append(record)
                cv_frames.append(pd.read_csv(run / "cv_results" / f"{label}.csv"))
                print(f"[恢复校准检查点] {label}", flush=True)
                continue
            print(f"[搜索开始] {label}", flush=True)
            start = time.perf_counter()
            est = RunoffRegressor(str(path), family=family, horizon=h,
                max_epochs=cfg["max_epochs"], ann_max_iter=cfg["ann_max_iter"], device=device,
                history_dir=str(run / "training_history"), phase=f"calibration_{strategy}")
            cv = TimeSeriesSplit(n_splits=cfg["cv_folds"], gap=7)
            gs = GridSearchCV(est, grids[family], cv=cv, scoring=physical_scorer,
                              n_jobs=1, refit=True, error_score="raise", return_train_score=False,
                              verbose=1)
            gs.fit(calibration[:, None], y[calibration])
            table = pd.DataFrame(gs.cv_results_)
            table["strategy"], table["model"], table["h"] = strategy, family, h
            table["candidate_index"] = np.arange(len(table))
            table["params"] = table.params.map(lambda x: json.dumps(x, ensure_ascii=False))
            table.to_csv(run / "cv_results" / f"{label}.csv", index=False, encoding="utf-8-sig")
            cv_frames.append(table)
            params = gs.best_params_.copy()
            epoch = None
            if family == "LSTM":
                candidates = {}
                evidence_files = sorted((run / "training_history").glob(
                    f"calibration_{strategy}_LSTM_h{h}_*.json"), key=lambda p: p.stat().st_mtime)
                for evidence in evidence_files:
                    record = json.loads(evidence.read_text(encoding="utf-8"))
                    if "cv_score" in record and all(record["params"][k] == (list(v) if isinstance(v, tuple) else v)
                                                     for k, v in params.items()):
                        candidates[record["validation_first_id"]] = record["selected_epochs"]
                assert len(candidates) == cfg["cv_folds"], candidates
                epoch = max(1, int(np.floor(np.median(list(candidates.values())) + .5)))
                params["fixed_epochs"] = epoch
            frozen[label] = params
            record = {"strategy": strategy, "model": family, "h": h,
                "best_index": int(gs.best_index_), "best_params": gs.best_params_,
                "best_score": float(gs.best_score_),
                "std_test_score": float(gs.cv_results_["std_test_score"][gs.best_index_]),
                "fixed_epochs": epoch, "seconds": time.perf_counter() - start}
            best_records.append(record)
            cache_tmp = cache.with_suffix(".tmp")
            dump(cache_tmp, {**record, "frozen_params": params})
            cache_tmp.replace(cache)
            dump(run / "best_parameters.json", best_records)
            dump(run / "frozen_parameters.json", frozen)
            print("[搜索完成] " + json.dumps(record, ensure_ascii=False), flush=True)
    dump(run / "best_parameters.json", best_records)
    dump(run / "frozen_parameters.json", frozen)
    pd.concat(cv_frames, ignore_index=True).to_csv(run / "cv_results/all_candidates.csv", index=False, encoding="utf-8-sig")
    print("参数冻结完成；开始因果OOF与最终训练，不读取测试分数选参。", flush=True)
    oof_rows, test_rows, coefficients, restoration = [], [], [], []
    for strategy in ("direct", "multi"):
        oof = {family: np.full((len(targets), 7), np.nan) for family in ("ANN", "RF", "LSTM")}
        test_predictions = {family: np.zeros((len(test), 7)) for family in oof}
        horizons = range(1, 8) if strategy == "direct" else [7]
        for h in horizons:
            y = targets[:, h - 1] if strategy == "direct" else targets
            for family in ("ANN", "RF", "LSTM"):
                label = f"{strategy}_{family}_h{h}"
                base = dict(dataset_path=str(path), family=family, horizon=h, device=device,
                            max_epochs=cfg["max_epochs"], ann_max_iter=cfg["ann_max_iter"],
                            history_dir=str(run / "training_history"), **frozen[label])
                for fold, (ti, vi) in enumerate(folds):
                    print(f"[OOF] {label} 折{fold + 1} train={len(ti)} validation={len(vi)}", flush=True)
                    pred_cache = run / "checkpoints" / f"oof{fold}_{label}.npy"
                    if pred_cache.exists():
                        pred = np.load(pred_cache)
                    else:
                        est = RunoffRegressor(**base, phase=f"oof{fold}_{strategy}").fit(ti[:, None], y[ti])
                        pred = est.predict(vi[:, None])
                        np.save(pred_cache, pred)
                    if strategy == "direct":
                        oof[family][vi, h - 1] = pred
                    else:
                        oof[family][vi] = pred
                print(f"[最终训练] {label} train={len(train)}", flush=True)
                folder = run / "models" / label
                final_cache = run / "checkpoints" / f"final_{label}.json"
                if final_cache.exists():
                    est = restore_model(folder, device=device)
                else:
                    est = RunoffRegressor(**base, phase=f"final_{strategy}").fit(train[:, None], y[train])
                    est.save(folder)
                    dump(final_cache, {"saved": True})
                # 真正从落盘模型恢复，并核对预测；不是只检查文件存在。
                sample = test[:4, None]
                before = est.predict(sample)
                restored = restore_model(folder, device=device)
                after = restored.predict(sample)
                np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-5)
                restoration.append({"model": label, "max_absolute_difference": float(np.max(np.abs(before - after)))})
                pred = est.predict(test[:, None])
                if strategy == "direct":
                    test_predictions[family][:, h - 1] = pred
                else:
                    test_predictions[family][:] = pred
        valid = np.flatnonzero(np.isfinite(np.column_stack(list(oof.values()))).all(axis=1))
        expected = np.concatenate([vi for _, vi in folds])
        np.testing.assert_array_equal(valid, expected)
        for family in oof:
            for fold, (_, vi) in enumerate(folds):
                oof_rows.extend(prediction_rows(run.name, data, vi, oof[family][vi], strategy, family, fold))
            test_rows.extend(prediction_rows(run.name, data, test, test_predictions[family], strategy, family))
        average = np.mean(list(test_predictions.values()), axis=0)
        stacking = np.zeros_like(average)
        names = [f"{family}_h{h}" for family in ("ANN", "RF", "LSTM") for h in range(1, 8)]
        z_oof = np.column_stack([oof[family][valid] for family in ("ANN", "RF", "LSTM")])
        z_test = np.column_stack([test_predictions[family] for family in ("ANN", "RF", "LSTM")])
        for h in (range(1, 8) if strategy == "direct" else [7]):
            cols = [h - 1, h + 6, h + 13] if strategy == "direct" else list(range(21))
            target = targets[valid, h - 1] if strategy == "direct" else targets[valid]
            meta = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(z_oof[:, cols], target)
            prediction = meta.predict(z_test[:, cols])
            if strategy == "direct":
                stacking[:, h - 1] = prediction
            else:
                stacking[:] = prediction
            joblib.dump(meta, run / "models" / f"stacking_{strategy}_h{h}.joblib")
            ridge = meta[-1]
            coef = np.asarray(ridge.coef_).reshape(-1, len(cols))
            intercept = np.atleast_1d(ridge.intercept_)
            for j in range(coef.shape[0]):
                for k, col in enumerate(cols):
                    coefficients.append({"strategy": strategy, "target_h": h if strategy == "direct" else j + 1,
                                         "input": names[col], "coefficient_standardized_X": coef[j, k],
                                         "intercept": intercept[j], "alpha": 1.0, "n_oof": len(valid)})
        test_rows.extend(prediction_rows(run.name, data, test, average, strategy, "Mean"))
        test_rows.extend(prediction_rows(run.name, data, test, stacking, strategy, "Stacking"))
        pd.DataFrame(oof_rows).to_csv(run / "oof_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(test_rows).to_csv(run / "test_predictions.csv", index=False, encoding="utf-8-sig")
    dump(run / "model_restoration_checks.json", restoration)
    pd.DataFrame(coefficients).to_csv(run / "stacking_coefficients.csv", index=False, encoding="utf-8-sig")
    result = metrics_table(test_rows)
    assert len(result) == 70 and result[["R2", "RMSE", "MAE"]].notna().all().all()
    result.to_csv(run / "metrics.csv", index=False, encoding="utf-8-sig")
    print("[真实测试指标] 标准决定系数R2，原始流量尺度RMSE/MAE", flush=True)
    print(result.to_string(index=False), flush=True)
    for strategy in ("direct", "multi"):
        for family in ("LSTM", "Mean", "Stacking"):
            part = result[(result.strategy == strategy) & (result.model == family)]
            text = f"run_id={run.name}\n{strategy} / {family}\n" + part.to_string(index=False)
            (run / "evidence" / f"{strategy}_{family}.txt").write_text(text, encoding="utf-8")
    print("[完成] 搜索、OOF、最终训练、落盘恢复与70行指标全部完成。", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id 必须是单个目录名")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    run = ROOT / "output/acceptance3_runs" / args.run_id
    run.mkdir(parents=True, exist_ok=args.resume)
    previous = None
    if args.resume:
        previous = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if previous["status"] == "completed" or previous["smoke"] != args.smoke:
            parser.error("只允许恢复同类型的未完成运行")
        for file in ("acceptance3_train.py", "acceptance3_model.py", "feature_engineering.py", "config.py"):
            if (run / "source" / file).read_bytes() != (ROOT / "code" / file).read_bytes():
                parser.error("源代码与原运行快照不同，不能混合检查点")
        if previous["device"] != device:
            parser.error("恢复运行必须使用相同设备，不能混合CPU和GPU检查点")
        data_file = ROOT / "Teacher Resources & Requirements/01047000.csv"
        if hashlib.sha256(data_file.read_bytes()).hexdigest() != previous["data_sha256"]:
            parser.error("输入数据hash发生变化，不能恢复旧检查点")
    for sub in ("logs", "models", "cv_results", "training_history", "source", "evidence", "checkpoints"):
        (run / sub).mkdir(exist_ok=args.resume)
    for file in ("acceptance3_train.py", "acceptance3_model.py", "feature_engineering.py", "config.py"):
        shutil.copy2(ROOT / "code" / file, run / "source" / file)
    data = ROOT / "Teacher Resources & Requirements/01047000.csv"
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "matplotlib", "python-docx", "joblib"):
        packages[package] = importlib.metadata.version(package)
    manifest = {"run_id": args.run_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "status": "running", "smoke": args.smoke, "command": subprocess.list2cmdline(sys.argv),
                "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                "packages": packages, "device": device, "torch_build": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "plan": "任务1至7+第三阶段训练方案_2026-10-29_第1次修改版.md"}
    (run / "source/git_diff.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
    (run / "source/git_status.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=ROOT))
    if previous is not None:
        manifest["started"] = previous["started"]
        manifest["previous_seconds"] = previous.get("seconds", 0)
        manifest["resume_count"] = previous.get("resume_count", 0) + 1
    dump(run / "manifest.json", manifest)
    start = time.perf_counter()
    with (run / "logs/training.log").open("a" if args.resume else "w", encoding="utf-8", buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
                with threadpool_limits(limits=4):
                    run_training(run, args.smoke, device)
                manifest["status"] = "completed"
            except Exception:
                import traceback
                traceback.print_exc()
                manifest["status"] = "failed"
                raise
            finally:
                manifest["seconds"] = manifest.get("previous_seconds", 0) + time.perf_counter() - start
                manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                dump(run / "manifest.json", manifest)


if __name__ == "__main__":
    main()
