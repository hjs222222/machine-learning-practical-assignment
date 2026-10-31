# -*- coding: utf-8 -*-
"""任务1至7+第二轮嵌套时间CV训练，2026-10-31，第2次修改版。"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from acceptance3_model import load_dataset, physical_scorer, restore_model
from acceptance3_train import ROOT, Tee, dump, metrics_table, prediction_rows, prepare
from acceptance3_v2_model import V2Regressor

SOURCES = ["acceptance3_model.py", "acceptance3_train.py", "acceptance3_v2_model.py",
           "acceptance3_v2_train.py", "feature_engineering.py", "config.py"]
SEEDS = [42, 43, 44]


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    dump(temp, value)
    temp.replace(path)


def prepare_v2(run):
    path, targets, _, train, test, _ = prepare(run)
    data = load_dataset(str(path))
    raw = pd.read_csv(ROOT / "Teacher Resources & Requirements/01047000.csv", parse_dates=["Date"])
    dates = pd.to_datetime(data["origin_dates"])
    # 特征行由原特征工程丢弃30天后的原数据构成；不使用未来实测值。
    raw_features = raw.iloc[30:].reset_index(drop=True)
    assert len(raw_features) == len(data["features"])
    day = raw_features.Date.dt.dayofyear.to_numpy()
    original = data["features"].shape[1]
    data["features"] = np.column_stack([data["features"], raw_features.Discharge,
                                        np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25)])
    data["feature_names"] = np.append(data["feature_names"], ["Discharge_current", "doy_sin", "doy_cos"])
    data["original_feature_count"] = np.asarray(original)
    np.savez_compressed(path, **data)
    load_dataset.cache_clear()
    summary = json.loads((run / "data_summary.json").read_text(encoding="utf-8"))
    summary.update(original_feature_count=original, n_features=data["features"].shape[1],
                   features=data["feature_names"].tolist(),
                   feature_selection="原84列折内TOP50，可选另加当日Q；精简序列9列")
    dump(run / "data_summary.json", summary)
    days = data["origin_days"]
    blocks = [("oof0", "2002-01-01", "2002-12-31"),
              ("oof1", "2003-01-01", "2003-06-30"),
              ("oof2", "2003-07-01", "2003-12-24")]
    folds, rows = [], []
    for label, start, end in blocks:
        vi = train[(days[train] >= np.datetime64(start).astype("int64")) &
                   (days[train] <= np.datetime64(end).astype("int64"))]
        ti = train[days[train] + 7 < days[vi[0]]]
        folds.append((label, ti, vi))
        rows.append(split_record(label, "outer", 0, ti, vi, days))
    pd.DataFrame(rows).to_csv(run / "outer_splits.csv", index=False, encoding="utf-8-sig")
    return path, targets, train, test, folds


def split_record(stage, kind, fold, ti, vi, days):
    assert len(ti) and len(vi) and days[ti[-1]] + 7 < days[vi[0]]
    date = lambda i: str(np.datetime64(int(i), "D"))
    return dict(stage=stage, kind=kind, fold=fold, n_train=len(ti), n_validation=len(vi),
                train_last_id=int(ti[-1]), validation_first_id=int(vi[0]),
                max_train_target=date(days[ti[-1]] + 7), validation_origin_start=date(days[vi[0]]),
                validation_origin_end=date(days[vi[-1]]))


def inner_splits(ids, days, smoke=False):
    # 第一内折含478个样本，内部再留90天+7天后仍有381条训练样本。
    width = 90 if len(ids) < 900 else 180
    splits = []
    for start in [len(ids) - 2 * width, len(ids) - width]:
        tr, va = np.arange(start - 7), np.arange(start, start + width)
        assert len(tr) >= 478
        assert days[ids[tr[-1]]] + 7 < days[ids[va[0]]]
        splits.append((tr, va))
    return splits


def estimator(path, family, h, run, phase, smoke, **params):
    return V2Regressor(str(path), family=family, horizon=h, history_dir=str(run / "training_history"),
                       phase=phase, max_epochs=2 if smoke else 150,
                       patience=1 if smoke else 15, ann_max_iter=10 if smoke else 1000,
                       device="cpu", **params)


def search(run, path, ids, y, stage, family, strategy, h, grid, smoke, role="grid"):
    label = f"{stage}_{role}_{strategy}_{family}_h{h}"
    cache = run / "checkpoints" / f"{label}.json"
    if cache.exists():
        print("[恢复搜索]", label, flush=True)
        return json.loads(cache.read_text(encoding="utf-8"))
    print("[开始搜索]", label, "训练样本", len(ids), flush=True)
    start = time.perf_counter()
    days = load_dataset(str(path))["origin_days"]
    cv = inner_splits(ids, days, smoke)
    rows = [split_record(stage, f"{role}_{strategy}_{family}_h{h}", k, ids[ti], ids[vi], days)
            for k, (ti, vi) in enumerate(cv)]
    pd.DataFrame(rows).to_csv(run / "splits" / f"{label}.csv", index=False, encoding="utf-8-sig")
    gs = GridSearchCV(estimator(path, family, h, run, label, smoke), grid, cv=cv,
                      scoring=physical_scorer, n_jobs=1, error_score="raise", refit=False,
                      return_train_score=False, verbose=1)
    gs.fit(ids[:, None], y[ids])
    table = pd.DataFrame(gs.cv_results_)
    table["stage"], table["role"], table["strategy"], table["model"], table["h"] = stage, role, strategy, family, h
    table["params"] = table.params.map(lambda v: json.dumps(v, ensure_ascii=False))
    table.to_csv(run / "cv_results" / f"{label}.csv", index=False, encoding="utf-8-sig")
    params, epochs = gs.best_params_.copy(), None
    if family == "LSTM":
        matches = {}
        for file in sorted((run / "training_history").glob(f"{label}_LSTM_h{h}_*.json"), key=lambda p: p.stat().st_mtime):
            record = json.loads(file.read_text(encoding="utf-8"))
            if "cv_score" in record and all(record["params"][k] == (list(v) if isinstance(v, tuple) else v) for k, v in params.items()):
                matches[record["validation_first_id"]] = record["selected_epochs"]
        assert len(matches) == len(cv), matches
        epochs = max(1, int(np.floor(np.median(list(matches.values())) + .5)))
    value = dict(stage=stage, role=role, strategy=strategy, model=family, h=h,
                 best_params=params, fixed_epochs=epochs, best_index=int(gs.best_index_),
                 best_score=float(gs.best_score_), seconds=time.perf_counter() - start)
    atomic_json(cache, value)
    print("[搜索完成]", json.dumps(value, ensure_ascii=False), flush=True)
    return value


def pilot(run, path, ids, targets, stage, strategy, smoke):
    h = 1 if strategy == "direct" else 7
    y = targets[:, 0] if strategy == "direct" else targets
    configs = [("engineered", "log", 0.), ("plus_Q", "log", 0.),
               ("compact", "log", 0.), ("plus_Q", "sqrt", 0.),
               ("plus_Q", "log", .25), ("plus_Q", "log", .5)]
    grid = [{"feature_mode": [f], "target_transform": [t], "mixed_weight": [w],
             "units": [(8,) if smoke else (64,)], "dropout": [.2], "lr": [.001]}
            for f, t, w in (configs[:1] if smoke else configs)]
    result = search(run, path, ids, y, stage, "LSTM", strategy, h, grid, smoke, "pilot")
    # pilot只决定表示与损失；结构参数仍由完整GridSearchCV选择。
    return {k: result["best_params"][k] for k in ("feature_mode", "target_transform", "mixed_weight")}


def model_grid(family, representation, smoke):
    if family == "LSTM":
        return {**{k: [v] for k, v in representation.items()},
                "units": [(8,)] if smoke else [(32,), (64,), (64, 32)],
                "dropout": [.2] if smoke else [.1, .2], "lr": [.001] if smoke else [.001, .0005]}
    if family == "ANN":
        return dict(feature_mode=["plus_Q"], target_transform=["log"] if smoke else ["log", "sqrt"],
                    hidden_layer_sizes=[(8,)] if smoke else [(64,)], alpha=[.1] if smoke else [.01, .1, 1.],
                    learning_rate_init=[.001])
    return dict(feature_mode=["plus_Q"], target_transform=["log"] if smoke else ["log", "sqrt"],
                n_estimators=[5] if smoke else [300], max_depth=[3] if smoke else [10, 20],
                min_samples_leaf=[2] if smoke else [1, 5])


def fit_model(run, path, ids, targets, record, stage, seed, smoke):
    strategy, family, h = record["strategy"], record["model"], record["h"]
    label = f"{stage}_{strategy}_{family}_h{h}_seed{seed}"
    cache, folder = run / "checkpoints" / f"fit_{label}.json", run / "models" / label
    if cache.exists():
        return folder
    params = record["best_params"].copy()
    if record["fixed_epochs"] is not None:
        params["fixed_epochs"] = record["fixed_epochs"]
    y = targets[:, h - 1] if strategy == "direct" else targets
    model = estimator(path, family, h, run, label, smoke, seed=seed, **params).fit(ids[:, None], y[ids])
    model.save(folder)
    restored = restore_model(folder, device="cpu")
    before, after = model.predict(ids[-4:, None]), restored.predict(ids[-4:, None])
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-5)
    atomic_json(cache, dict(label=label, restored=True, max_difference=float(np.max(np.abs(before - after)))))
    print("[模型保存]", label, "epochs=", record["fixed_epochs"], flush=True)
    return folder


def meta_fit(x, y, kind, alpha):
    # log元层保证非负，原尺度元层保留未裁剪输出，两者均在看测试前固定。
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
        np.log(np.maximum(x, 1e-12)) if kind == "log" else x,
        np.log(np.maximum(y, 1e-12)) if kind == "log" else y)


def meta_predict(model, x, kind):
    p = model.predict(np.log(np.maximum(x, 1e-12)) if kind == "log" else x)
    p = np.exp(p) if kind == "log" else p
    if not np.isfinite(p).all():
        raise ValueError("元层预测非有限")
    return p


def build_meta(run, data, oof, folds, smoke):
    selected, validation = [], []
    for seed in ([42] if smoke else SEEDS):
        for strategy in ("direct", "multi"):
            f = oof[(oof.seed == seed) & (oof.strategy == strategy)]
            matrices = [f[f.model == family].pivot(index="sample_id", columns="h", values="pred").sort_index()
                        for family in ("ANN", "RF", "LSTM")]
            ids = matrices[0].index.to_numpy()
            assert all(np.array_equal(ids, m.index) for m in matrices)
            matrix = np.column_stack([m.to_numpy() for m in matrices])
            targets = data["y"][ids]
            cv = []
            for _, _, vi in folds[1:]:
                tr = np.flatnonzero(data["origin_days"][ids] + 7 < data["origin_days"][vi[0]])
                va = np.flatnonzero(np.isin(ids, vi))
                assert len(tr) and len(va)
                cv.append((tr, va))
            for h in (range(1, 8) if strategy == "direct" else [7]):
                cols = [h-1, h+6, h+13] if strategy == "direct" else list(range(21))
                x = matrix[:, cols]
                y = targets[:, h-1] if strategy == "direct" else targets
                candidates = []
                for kind in ("raw", "log"):
                    for alpha in ([1.] if smoke else [1., 10., 100., 1000.]):
                        scores = []
                        for k, (ti, vi) in enumerate(cv):
                            model = meta_fit(x[ti], y[ti], kind, alpha)
                            pred = np.asarray(meta_predict(model, x[vi], kind)).reshape(len(vi), -1)
                            truth = np.asarray(y[vi]).reshape(len(vi), -1)
                            rmse = float(np.sqrt(np.mean((truth-pred)**2, axis=0)).mean())
                            scores.append(rmse)
                            validation.append(dict(seed=seed, strategy=strategy, h=h, kind=kind, alpha=alpha,
                                                   fold=k, RMSE=rmse, MAE=float(np.abs(truth-pred).mean()),
                                                   negative_predictions=int((pred < 0).sum()), n_train=len(ti),
                                                   max_train_target=str(np.datetime64(int(data['origin_days'][ids[ti[-1]]] + 7), 'D')),
                                                   validation_origin_start=str(np.datetime64(int(data['origin_days'][ids[vi[0]]]), 'D'))))
                        candidates.append((float(np.mean(scores)), kind, alpha))
                score, kind, alpha = min(candidates, key=lambda c:c[0])
                model = meta_fit(x, y, kind, alpha)
                label = f"stacking_{strategy}_h{h}_seed{seed}"
                joblib.dump(model, run / "models" / f"{label}.joblib")
                selected.append(dict(seed=seed, strategy=strategy, h=h, kind=kind, alpha=alpha,
                                     mean_validation_RMSE=score, label=label, input_cols=cols))
    pd.DataFrame(validation).to_csv(run / "meta_validation.csv", index=False, encoding="utf-8-sig")
    atomic_json(run / "meta_parameters.json", selected)
    return selected


def evaluate(run, path, targets, train, test, selections, meta_records, smoke):
    data = load_dataset(str(path))
    rows = []
    for seed in ([42] if smoke else SEEDS):
        for strategy in ("direct", "multi"):
            preds = {family: np.zeros((len(test), 7)) for family in ("ANN", "RF", "LSTM")}
            for record in selections:
                if record["stage"] != "final" or record["strategy"] != strategy:
                    continue
                h, family = record["h"], record["model"]
                model = restore_model(run / "models" / f"final_{strategy}_{family}_h{h}_seed{seed}", device="cpu")
                pred = model.predict(test[:, None])
                if strategy == "direct":
                    preds[family][:, h-1] = pred
                else:
                    preds[family][:] = pred
            preds["Mean"] = np.mean([preds[f] for f in ("ANN", "RF", "LSTM")], axis=0)
            matrix = np.column_stack([preds[f] for f in ("ANN", "RF", "LSTM")])
            stacking = np.zeros((len(test), 7))
            for rec in meta_records:
                if rec["seed"] != seed or rec["strategy"] != strategy:
                    continue
                model = joblib.load(run / "models" / f"{rec['label']}.joblib")
                pred = meta_predict(model, matrix[:, rec["input_cols"]], rec["kind"])
                if strategy == "direct":
                    stacking[:, rec["h"]-1] = pred
                else:
                    stacking[:] = pred
            preds["Stacking"] = stacking
            for family, pred in preds.items():
                new = prediction_rows(run.name, data, test, pred, strategy, family)
                rows.extend([{**r, "seed": seed} for r in new])
    frame = pd.DataFrame(rows)
    frame.to_csv(run / "test_predictions_all_seeds.csv", index=False, encoding="utf-8-sig")
    tables = []
    for seed, f in frame.groupby("seed"):
        table = metrics_table(f.to_dict("records"))
        table["seed"] = seed
        tables.append(table)
    all_metrics = pd.concat(tables, ignore_index=True)
    all_metrics.to_csv(run / "metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    all_metrics[all_metrics.seed == 42].drop(columns="seed").to_csv(run / "metrics.csv", index=False, encoding="utf-8-sig")
    frame[frame.seed == 42].to_csv(run / "test_predictions.csv", index=False, encoding="utf-8-sig")
    all_metrics.groupby(["strategy", "model", "h"])[["R2", "RMSE", "MAE"]].agg(["mean", "std"]).to_csv(
        run / "seed_summary.csv", encoding="utf-8-sig")
    # 基线仅在各训练范围估计；共同起报点一致。
    current = data["features"][data["positions"], data["feature_names"].tolist().index("Discharge_current")]
    months = pd.to_datetime(data["origin_dates"]).month.to_numpy()
    baselines = {}
    baselines["Persistence"] = np.repeat(current[test, None], 7, axis=1)
    baselines["Training_mean"] = np.repeat(targets[train].mean(axis=0)[None, :], len(test), axis=0)
    seasonal = np.zeros((len(test), 7))
    for month in range(1, 13):
        seasonal[months[test] == month] = targets[train[months[train] == month]].mean(axis=0)
    baselines["Seasonal_mean"] = seasonal
    positions = data["positions"]
    qcol = data["feature_names"].tolist().index("Discharge_current")
    arx = data["features"][positions[:, None] + np.arange(-6, 1), qcol]
    ar = make_pipeline(StandardScaler(), Ridge(alpha=10.)).fit(arx[train], targets[train])
    baselines["AR7_Ridge"] = ar.predict(arx[test])
    br = []
    for family, pred in baselines.items():
        br.extend(prediction_rows(run.name, data, test, pred, "baseline", family))
    pd.DataFrame(br).to_csv(run / "baseline_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_table(br).to_csv(run / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    print("[全部配置冻结后的真实测试指标 seed42]\n", all_metrics[all_metrics.seed == 42].to_string(index=False), flush=True)


def train_run(run, smoke):
    path, targets, train, test, folds = prepare_v2(run)
    data = load_dataset(str(path))
    selections, oof_rows = [], []
    stages = folds + [("final", train, None)]
    for stage, ids, vi in stages:
        for strategy in ("direct", "multi"):
            representation = pilot(run, path, ids, targets, stage, strategy, smoke)
            for h in (range(1, 8) if strategy == "direct" else [7]):
                y = targets[:, h-1] if strategy == "direct" else targets
                for family in ("LSTM", "ANN", "RF"):
                    record = search(run, path, ids, y, stage, family, strategy, h,
                                    model_grid(family, representation, smoke), smoke)
                    selections.append(record)
                    for seed in ([42] if smoke else SEEDS):
                        folder = fit_model(run, path, ids, targets, record, stage, seed, smoke)
                        if vi is not None:
                            cache = run / "checkpoints" / f"pred_{stage}_{strategy}_{family}_h{h}_seed{seed}.npy"
                            if cache.exists():
                                pred = np.load(cache)
                            else:
                                pred = restore_model(folder, device="cpu").predict(vi[:, None])
                                np.save(cache, pred)
                            # 直接预测需聚齐7列后再写长表。
        if vi is not None:
            for seed in ([42] if smoke else SEEDS):
                for strategy in ("direct", "multi"):
                    for family in ("ANN", "RF", "LSTM"):
                        if strategy == "direct":
                            pred = np.column_stack([np.load(run / "checkpoints" / f"pred_{stage}_{strategy}_{family}_h{h}_seed{seed}.npy") for h in range(1,8)])
                        else:
                            pred = np.load(run / "checkpoints" / f"pred_{stage}_{strategy}_{family}_h7_seed{seed}.npy")
                        new = prediction_rows(run.name, data, vi, pred, strategy, family, int(stage[-1]))
                        oof_rows.extend([{**r, "seed":seed} for r in new])
        atomic_json(run / "best_parameters.json", selections)
    oof = pd.DataFrame(oof_rows)
    oof.to_csv(run / "oof_predictions_all_seeds.csv", index=False, encoding="utf-8-sig")
    oof[oof.seed == 42].to_csv(run / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    meta = build_meta(run, data, oof, folds, smoke)
    atomic_json(run / "selection_frozen.json", dict(status="frozen_before_test", timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                seeds=[42] if smoke else SEEDS, base_selections=selections, meta_selections=meta,
                limitation="历史已使用2004反馈，本轮为回溯开发；不宣称盲测。"))
    print("[配置冻结] 所有基模型和元层均已确定，现在才开始2004预测评价。", flush=True)
    evaluate(run, path, targets, train, test, selections, meta, smoke)
    pd.concat([pd.read_csv(p) for p in (run / "cv_results").glob("*.csv")], ignore_index=True).to_csv(
        run / "all_cv_candidates.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id必须为单个目录名")
    run = ROOT / "output/acceptance3_runs" / args.run_id
    data_file = ROOT / "Teacher Resources & Requirements/01047000.csv"
    digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
    if args.resume:
        previous = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if previous["status"] == "completed" or previous["smoke"] != args.smoke or previous["data_sha256"] != digest:
            parser.error("只支持相同数据和运行类型的未完成运行")
        for name in SOURCES:
            if (run / "source" / name).read_bytes() != (ROOT / "code" / name).read_bytes():
                parser.error("源码已变化，禁止混合旧检查点")
    else:
        run.mkdir(parents=True, exist_ok=False)
        previous = {}
    for sub in ("models", "source", "logs", "splits", "cv_results", "checkpoints", "training_history"):
        (run / sub).mkdir(exist_ok=args.resume)
    for name in SOURCES:
        shutil.copy2(ROOT / "code" / name, run / "source" / name)
    plan = ROOT / "验收/report code/report3/任务1至7+训练诊断与改进方案_2026-10-31_第2次修改版.md"
    shutil.copy2(plan, run / "source" / "training_plan.md")
    manifest = dict(run_id=args.run_id, version=2, status="running", smoke=args.smoke,
                    started=previous.get("started", time.strftime("%Y-%m-%dT%H:%M:%S%z")),
                    command=subprocess.list2cmdline(sys.argv), data_sha256=digest, device="cpu",
                    python=sys.version, torch_build=torch.__version__, seeds=[42] if args.smoke else SEEDS,
                    packages={p:importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","torch","joblib"]},
                    git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    atomic_json(run / "manifest.json", manifest)
    dump(run / "config.json", dict(max_epochs=2 if args.smoke else 150, patience=1 if args.smoke else 15,
         min_relative_improvement=.001, inner_cv_folds=2, min_inner_train=478,
         early_stop_validation_days="max(90,min(180,ceil(0.15*n_train)))", min_early_train=365,
         score="negative_mean_per_horizon_original_scale_RMSE", gap=7, lookback=30,
         primary_seed=42, seeds=manifest["seeds"], representations="fold_local_pilot_6_candidates",
         lstm_grid=model_grid("LSTM", dict(feature_mode="plus_Q",target_transform="log",mixed_weight=0.),args.smoke),
         ann_grid=model_grid("ANN",{},args.smoke), rf_grid=model_grid("RF",{},args.smoke),
         meta="purged_forward_OOF_validation; Ridge alpha1/10/100/1000; raw/log; no_test_selection",
         inverse_sqrt="max(transformed_prediction,0)^2; fixed_before_test",
         test_limitation="回溯开发，2004历史反馈已知；测试不用于本轮搜索"))
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    with (run / "logs/training.log").open("a" if args.resume else "w", encoding="utf-8", buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
                with threadpool_limits(limits=4):
                    train_run(run, args.smoke)
                manifest["status"] = "completed"
            except Exception:
                import traceback
                traceback.print_exc()
                manifest["status"] = "failed"
                raise
            finally:
                manifest["seconds"] = previous.get("seconds",0) + time.perf_counter() - start
                manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                atomic_json(run / "manifest.json", manifest)


if __name__ == "__main__":
    main()
