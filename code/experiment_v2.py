# -*- coding: utf-8 -*-
"""
实验V2：第二次阶段验收版 —— 按预测策略/预见期重做特征选择并重训全部模型
=====================================================================

相比 V1（experiment.py）的改进：
1. 按预见期特征选择（补充 PPT「不同多步预测策略下的特征选择」）：
   - 多步直接预测：对每个 h=1..7，在训练集上对 y(t+h) 用互信息法重新选 TOP-K 特征，
     使每个预见期模型使用最适合该预见期的输入；
   - 多输出预测：采用 PPT 方法(3)——简单起见，以第 t+1 天径流对应的特征作为共用特征。
2. 精简 LSTM 输入（TOP30 特征 × 30 天），并缩小 LSTM 网格、加大训练轮数；
   缩小 ANN/RF 网格，控制总训练时长。
3. 其余保持与 V1 一致：对数目标变换、Z-score（仅训练集拟合）、
   TimeSeriesSplit + GridSearchCV、Stacking 用 OOF 元特征防泄漏、评价在原始尺度。

用法：python experiment_v2.py
输出：覆盖 output/results 下 pred_*.npz 与 10~14 号结果表；模型保存到 output/models
（运行前请将旧模型移出 output/models，特征子集变化后旧权重不兼容）。
"""

import os
import sys
import time
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import models as M
from feature_engineering import ZScoreScaler
from evaluation import evaluate, nse_grade

BASE_MODELS = ["ANN", "RF", "LSTM"]
STRATEGIES = [("direct", "多步直接预测"), ("multi", "多输出预测")]
ENSEMBLES = ["简单平均", "Stacking"]
TOP_K = 50          # 按预见期选择时保留的特征数（与V1特征量一致，避免信息损失）


# ===========================================================================
# 1. 按预见期特征选择（补充 PPT 要求）
# ===========================================================================
def select_features_per_horizon(d, feature_cols, top_k=TOP_K):
    """对 h=1..7 分别在训练集上对 y(t+h)（对数）做互信息特征选择。

    返回 {1..7: [特征名...], "multi": [特征名...]}；"multi" 采用 h=1 的特征（PPT 方法3）。
    """
    tr = d[d["Date"] <= pd.Timestamp(C.TRAIN_END)]
    X = tr[feature_cols].to_numpy(dtype=float)
    feats_h, info_h = {}, {}
    for h in range(1, C.HORIZON + 1):
        yh = tr[C.TARGET].shift(-h)
        m = yh.notna().to_numpy()
        ylog = np.log(np.maximum(yh[m].to_numpy(dtype=float), 1e-6))
        mi = mutual_info_regression(X[m], ylog, random_state=C.RANDOM_STATE)
        mi = np.nan_to_num(mi)
        order = np.argsort(mi)[::-1][:top_k]
        feats_h[h] = [feature_cols[i] for i in order]
        info_h[h] = dict(zip(feats_h[h], np.round(mi[order], 4)))
    feats_h["multi"] = list(feats_h[1])          # PPT 方法(3)：t+1 特征共用
    return feats_h, info_h


# ===========================================================================
# 2. 数据装配（按预见期分特征集）
# ===========================================================================
def make_sequences(X2d, seq_len):
    win = sliding_window_view(X2d, seq_len, axis=0)
    return np.ascontiguousarray(win.transpose(0, 2, 1))


class DataBundle:
    """per-h 特征子集 + 各自归一化（统计量仅训练期拟合）。"""

    def __init__(self, d, feats_h):
        self.d = d
        self.feats_h = feats_h
        self.N = len(d)
        self.y = d[C.TARGET].to_numpy(dtype=float)
        self.dates = d["Date"].to_numpy()

        pos = np.arange(self.N)
        self.train_end = int(pos[d["Date"] <= pd.Timestamp(C.TRAIN_END)].max())
        self.test_start = int(pos[d["Date"] >= pd.Timestamp(C.TEST_START)].min())
        self.START = C.SEQ_LEN - 1
        self.eval_origins = np.arange(self.test_start, self.N - C.HORIZON)
        self.eval_dates = self.dates[self.eval_origins]

        self.use_log = (C.TARGET_TRANSFORM == "log")
        self.y_model = np.log(self.y) if self.use_log else self.y.copy()
        tr = slice(self.START, self.train_end + 1)

        self.x_scalers = {}
        self.Xs_h = {}
        self.Xseq_h = {}
        for h, cols in feats_h.items():
            Xraw = d[cols].to_numpy(dtype=float)
            sc = ZScoreScaler().fit(Xraw[tr])
            self.x_scalers[h] = sc
            self.Xs_h[h] = sc.transform(Xraw)
            self.Xseq_h[h] = make_sequences(self.Xs_h[h], C.SEQ_LEN)

        self.y_scaler = ZScoreScaler().fit(self.y_model[tr].reshape(-1, 1))
        self.ys = self.y_scaler.transform(self.y_model.reshape(-1, 1)).ravel()

    def X_of(self, origins, model_name, h):
        if model_name == "LSTM":
            return self.Xseq_h[h][origins - self.START]
        return self.Xs_h[h][origins]

    def y_direct(self, origins, h):
        return self.ys[origins + h]

    def y_multi(self, origins):
        return np.column_stack([self.ys[origins + k] for k in range(1, C.HORIZON + 1)])

    def inv_y(self, v):
        v = np.asarray(v, dtype=float)
        shp = v.shape
        z = self.y_scaler.inverse_transform(v.reshape(-1, 1)).ravel()
        out = np.exp(z) if self.use_log else z
        return out.reshape(shp)

    def true_at(self, origins, h):
        return self.y[origins + h]


# ===========================================================================
# 3. 训练（从头，无热启动；特征子集已变化）
# ===========================================================================
def _fmt_params(p):
    s = ", ".join(f"{k}={v}" for k, v in sorted(p.items()))
    return s if len(s) <= 46 else s[:43] + "..."


def train_direct(ds, model_name):
    results = {}
    for h in range(1, C.HORIZON + 1):
        t0 = time.time()
        tr_org = np.arange(ds.START, ds.train_end - h + 1)
        Xtr = ds.X_of(tr_org, model_name, h)
        ytr = ds.y_direct(tr_org, h)
        est, best, cvtab = M.grid_search(model_name, Xtr, ytr)
        M.save_model(est, model_name, "direct", h)

        yp_scaled = np.asarray(est.predict(ds.X_of(ds.eval_origins, model_name, h)),
                               dtype=float).ravel()
        yp = ds.inv_y(yp_scaled)
        results[h] = {"pred": yp, "pred_scaled": yp_scaled,
                      "best_params": best, "cv_table": cvtab,
                      "n_train": len(tr_org), "sec": time.time() - t0}
        print(f"    [直接] {model_name:<5} h={h}  特征={len(ds.feats_h[h]):<3}"
              f" 训练样本={len(tr_org):<5} 最优={_fmt_params(best):<40} "
              f"用时 {time.time()-t0:6.1f}s", flush=True)
    return results


def train_multioutput(ds, model_name):
    t0 = time.time()
    h_key = 1                                    # 多输出共用 h=1 特征
    tr_org = np.arange(ds.START, ds.train_end - C.HORIZON + 1)
    Xtr = ds.X_of(tr_org, model_name, h_key)
    ytr = ds.y_multi(tr_org)
    est, best, cvtab = M.grid_search(model_name, Xtr, ytr)
    M.save_model(est, model_name, "multi", h=1)

    pm_scaled = np.asarray(est.predict(ds.X_of(ds.eval_origins, model_name, h_key)),
                           dtype=float)
    if pm_scaled.ndim == 1:
        pm_scaled = pm_scaled.reshape(-1, C.HORIZON)
    pm = ds.inv_y(pm_scaled)

    print(f"    [多输出] {model_name:<5} 特征={len(ds.feats_h[h_key]):<3}"
          f" 训练样本={len(tr_org):<5} 最优={_fmt_params(best):<40} "
          f"用时 {time.time()-t0:6.1f}s", flush=True)
    return {h: {"pred": pm[:, h - 1], "pred_scaled": pm_scaled[:, h - 1],
                "best_params": best, "cv_table": cvtab,
                "n_train": len(tr_org), "sec": time.time() - t0}
            for h in range(1, C.HORIZON + 1)}


# ===========================================================================
# 4. 集成学习
# ===========================================================================
def ensemble_simple_average(base_preds, strategy, model_names):
    out = {}
    for h in range(1, C.HORIZON + 1):
        stack = np.vstack([base_preds[m][strategy][h]["pred"] for m in model_names])
        out[h] = {"pred": stack.mean(axis=0),
                  "best_params": {"融合方式": f"{len(model_names)}个模型算术平均",
                                  "参与模型": "+".join(model_names)}}
    return out


def ensemble_stacking(ds, strategy, base_preds, base_est_params, model_names):
    n_splits = C.N_CV_SPLITS
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out, meta_coefs = {}, []
    for h in range(1, C.HORIZON + 1):
        is_direct = (strategy == "direct")
        if is_direct:
            tr_org = np.arange(ds.START, ds.train_end - h + 1)
            ytr = ds.y_direct(tr_org, h)
        else:
            tr_org = np.arange(ds.START, ds.train_end - C.HORIZON + 1)
            ytr = ds.y_multi(tr_org)[:, h - 1]
        n_tr = len(tr_org)
        oof = np.full((n_tr, len(model_names)), np.nan)

        for tr_i, va_i in tscv.split(np.zeros(n_tr)):
            for j, mname in enumerate(model_names):
                kw = dict(base_est_params[mname][strategy][h])
                kw.pop("output_dim", None)
                if is_direct:
                    est = M.MODEL_FACTORY[mname](output_dim=1, **kw)
                    yy = ds.y_direct(tr_org[tr_i], h)
                    est.fit(ds.X_of(tr_org[tr_i], mname, h), yy)
                    oof[va_i, j] = est.predict(ds.X_of(tr_org[va_i], mname, h))
                else:
                    est = M.MODEL_FACTORY[mname](output_dim=C.HORIZON, **kw)
                    yy = ds.y_multi(tr_org[tr_i])
                    est.fit(ds.X_of(tr_org[tr_i], mname, 1), yy)
                    p = np.asarray(est.predict(ds.X_of(tr_org[va_i], mname, 1)))
                    oof[va_i, j] = p.ravel() if p.ndim == 1 else p[:, h - 1]

        ok = ~np.isnan(oof).any(axis=1)
        meta = Ridge(alpha=1.0, random_state=C.RANDOM_STATE)
        meta.fit(oof[ok], ytr[ok])
        test_feats = np.column_stack([base_preds[m][strategy][h]["pred_scaled"]
                                      for m in model_names])
        out[h] = {
            "pred": ds.inv_y(meta.predict(test_feats)),
            "best_params": {"元学习器": "Ridge(alpha=1.0)",
                            "元特征": "+".join(model_names),
                            "系数": np.round(meta.coef_, 3).tolist(),
                            "截距": round(float(meta.intercept_), 4)},
        }
        meta_coefs.append({"预见期": h, "Ridge系数": np.round(meta.coef_, 4).tolist(),
                           "截距": round(float(meta.intercept_), 4)})
        print(f"    [Stacking-{strategy}] h={h} 系数={np.round(meta.coef_,3)}"
              f" 截距={meta.intercept_:.4f}", flush=True)
    return out, pd.DataFrame(meta_coefs)


# ===========================================================================
# 5. 指标汇总
# ===========================================================================
def collect_metrics(preds_by_model, ds, strategy_cn):
    rows = []
    for mname, per_h in preds_by_model.items():
        for h in range(1, C.HORIZON + 1):
            yt = ds.true_at(ds.eval_origins, h)
            m = evaluate(yt, per_h[h]["pred"])
            rows.append({"模型": mname, "策略": strategy_cn, "预见期": h, **m})
    df = pd.DataFrame(rows)
    df["NSE等级"] = df["NSE"].map(nse_grade)
    return df


def pooled_metrics(preds_by_model, ds, strategy_cn):
    rows = []
    yt_all = np.concatenate([ds.true_at(ds.eval_origins, h)
                             for h in range(1, C.HORIZON + 1)])
    for mname, per_h in preds_by_model.items():
        yp_all = np.concatenate([per_h[h]["pred"] for h in range(1, C.HORIZON + 1)])
        rows.append({"模型": mname, "策略": strategy_cn, "预见期": "1-7(汇总)",
                     **evaluate(yt_all, yp_all)})
    df = pd.DataFrame(rows)
    df["NSE等级"] = df["NSE"].map(nse_grade)
    return df


# ===========================================================================
# 6. 主流程
# ===========================================================================
def run(d, feature_cols):
    print("=" * 70)
    print("实验V2：按预见期特征选择 + 全量重训（第二次验收版）")
    print("=" * 70)

    # ---- 网格调整（控制总时长 + 加大 LSTM 训练）----
    M.PARAM_GRIDS["ANN"] = {"hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
                            "alpha": [1e-5, 1e-4, 1e-2],
                            "learning_rate_init": [5e-4, 1e-3, 1e-2]}
    M.PARAM_GRIDS["RF"] = {"n_estimators": [300, 500],
                           "max_depth": [None, 10, 15, 20],
                           "min_samples_leaf": [1, 5]}
    M.PARAM_GRIDS["LSTM"] = {"units": [(64,), (64, 32)],
                             "dropout": [0.2],
                             "lr": [1e-3, 5e-4],
                             "epochs": [80, 120]}
    C.N_CV_SPLITS = 3

    print("\n【0. 按预见期特征选择】（互信息 TOP%d，仅训练集）" % TOP_K)
    feats_h, info_h = select_features_per_horizon(d, feature_cols)
    for h in range(1, C.HORIZON + 1):
        print(f"  h={h} 特征数={len(feats_h[h])}  Top5={feats_h[h][:5]}", flush=True)
    print(f"  多输出(共用 h=1 特征) 特征数={len(feats_h['multi'])}")

    ds = DataBundle(d, feats_h)
    print(f"\n测试期：{str(pd.Timestamp(ds.eval_dates[0]).date())} ~ "
          f"{str(pd.Timestamp(ds.eval_dates[-1]).date())}，"
          f"评估起始日 {len(ds.eval_origins)} 天")

    base_preds = {m: {} for m in BASE_MODELS}
    base_est_params = {}
    all_cv = []

    print("\n【1. 个体模型训练（GridSearchCV + TimeSeriesSplit）】")
    for mname in BASE_MODELS:
        if mname == "LSTM" and not M.tf_available():
            print(f"    [跳过] {mname}：TensorFlow 不可用")
            continue
        for strat, strat_cn in STRATEGIES:
            print(f"\n  --- {M.MODEL_CN[mname]} × {strat_cn} ---", flush=True)
            res = (train_direct(ds, mname) if strat == "direct"
                   else train_multioutput(ds, mname))
            base_preds[mname][strat] = res
            base_est_params.setdefault(mname, {})[strat] = {
                h: r["best_params"] for h, r in res.items()}
            for h, r in res.items():
                t = r["cv_table"].copy()
                t.insert(0, "模型", mname)
                t.insert(1, "策略", strat_cn)
                t.insert(2, "预见期", h)
                all_cv.append(t)
            np.savez(os.path.join(C.RESULT_DIR, f"pred_{mname}_{strat}.npz"),
                     **{f"h{h}": res[h]["pred"] for h in res})

    available = [m for m in BASE_MODELS if base_preds[m]]

    print("\n【2. 集成学习模型】")
    ens_preds = {name: {} for name in ENSEMBLES}
    stack_coef_tables = []
    for strat, strat_cn in STRATEGIES:
        print(f"\n  --- {strat_cn} ---", flush=True)
        ens_preds["简单平均"][strat] = ensemble_simple_average(
            base_preds, strat, available)
        print(f"    [集成] 简单平均 —— 对 {len(available)} 个个体模型的预测取算术平均", flush=True)
        st, coef = ensemble_stacking(ds, strat, base_preds, base_est_params, available)
        ens_preds["Stacking"][strat] = st
        coef.insert(0, "策略", strat_cn)
        stack_coef_tables.append(coef)
        for name in ENSEMBLES:
            np.savez(os.path.join(C.RESULT_DIR, f"pred_ens_{name}_{strat}.npz"),
                     **{f"h{h}": ens_preds[name][strat][h]["pred"]
                        for h in range(1, C.HORIZON + 1)})

    print("\n【3. 模型精度评价】")
    all_models = {**{M.MODEL_CN[m]: base_preds[m] for m in available},
                  "简单平均集成": ens_preds["简单平均"],
                  "Stacking 集成": ens_preds["Stacking"]}
    detail_frames, pooled_frames = [], []
    for strat, strat_cn in STRATEGIES:
        sub = {k: v[strat] for k, v in all_models.items() if strat in v}
        detail_frames.append(collect_metrics(sub, ds, strat_cn))
        pooled_frames.append(pooled_metrics(sub, ds, strat_cn))
    detail = pd.concat(detail_frames, ignore_index=True)
    pooled = pd.concat(pooled_frames, ignore_index=True)

    detail.to_csv(os.path.join(C.RESULT_DIR, "10_各模型各预见期指标.csv"),
                  index=False, encoding="utf-8-sig")
    pooled.to_csv(os.path.join(C.RESULT_DIR, "11_模型汇总指标.csv"),
                  index=False, encoding="utf-8-sig")

    bp_rows = []
    for m in available:
        for strat, strat_cn in STRATEGIES:
            for h, r in base_preds[m][strat].items():
                bp_rows.append({"模型": M.MODEL_CN[m], "策略": strat_cn, "预见期": h,
                                "最优超参数": _fmt_params(r["best_params"]),
                                "训练样本数": r.get("n_train"),
                                "寻优耗时(s)": round(r.get("sec", 0), 1)})
    pd.DataFrame(bp_rows).to_csv(os.path.join(C.RESULT_DIR, "12_最优超参数.csv"),
                                 index=False, encoding="utf-8-sig")

    if all_cv:
        pd.concat(all_cv, ignore_index=True).to_csv(
            os.path.join(C.RESULT_DIR, "13_网格搜索CV结果.csv"),
            index=False, encoding="utf-8-sig")
    if stack_coef_tables:
        pd.concat(stack_coef_tables, ignore_index=True).to_csv(
            os.path.join(C.RESULT_DIR, "14_Stacking元学习器系数.csv"),
            index=False, encoding="utf-8-sig")

    np.savez(os.path.join(C.RESULT_DIR, "eval_index.npz"),
             origins=ds.eval_origins, dates=ds.eval_dates.astype("datetime64[D]"),
             observed=ds.y, test_start=ds.test_start, train_end=ds.train_end,
             START=ds.START)

    # 按预见期选择特征表（供报告引用）
    fe_rows = [{"预见期": h, "特征数": len(feats_h[h]),
                "Top5特征": "; ".join(feats_h[h][:5])}
               for h in range(1, C.HORIZON + 1)]
    pd.DataFrame(fe_rows).to_csv(os.path.join(C.RESULT_DIR, "15_按预见期特征选择.csv"),
                                 index=False, encoding="utf-8-sig")

    print("\n【多输出预测 · 汇总指标（1~7 天拼接）】")
    p = pooled[pooled["策略"] == "多输出预测"]
    print(p[["模型", "RMSE", "MAE", "NSE", "R2", "PBIAS(%)", "NSE等级"]]
          .to_string(index=False))
    print("\n【多步直接预测 · 汇总指标（1~7 天拼接）】")
    p2 = pooled[pooled["策略"] == "多步直接预测"]
    print(p2[["模型", "RMSE", "MAE", "NSE", "R2", "PBIAS(%)", "NSE等级"]]
          .to_string(index=False))
    print("\n实验V2完成")
    return ds, all_models, detail, pooled


if __name__ == "__main__":
    from data_analysis import load_raw
    from feature_engineering import run as fe_run
    raw = load_raw()
    d, feature_cols, selected = fe_run(raw)
    run(d, feature_cols)
