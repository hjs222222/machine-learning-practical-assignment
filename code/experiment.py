# -*- coding: utf-8 -*-
"""
第三部分：径流多步预测实验主流程
================================

预测策略
--------
  A. 多步直接预测 (Direct Multi-Step)
     对 h = 1,2,...,7 每个预见期单独训练一个模型，共 7 个模型：
         y(t+h) = f_h( X(t) )
  B. 多输出预测 (Multi-Output)
     只训练 1 个模型，一次性输出未来 7 天：
         [y(t+1),...,y(t+7)] = F( X(t) )

个体模型：ANN、RF、LSTM                                           3 种 × 8 = 24 个
集成模型：简单平均 (Simple Average)、Stacking                    2 种 × 8 = 16 个

评估：统一在同一组测试起始日上评估，保证各预见期样本数一致、可比。
"""

import os
import json
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

import config as C
import models as M
from feature_engineering import ZScoreScaler
from evaluation import evaluate, metric_table, nse_grade

BASE_MODELS = ["ANN", "RF", "LSTM"]
STRATEGIES = [("direct", "多步直接预测"), ("multi", "多输出预测")]
ENSEMBLES = ["简单平均", "Stacking"]


# ===========================================================================
# 数据装配
# ===========================================================================
def make_sequences(X2d, seq_len):
    """把二维特征表切成长度为 seq_len 的滑窗，供 LSTM 使用。"""
    win = sliding_window_view(X2d, seq_len, axis=0)      # (n-seq+1, p, seq)
    return np.ascontiguousarray(win.transpose(0, 2, 1))  # (n-seq+1, seq, p)


class DataBundle:
    """封装训练/测试索引、归一化与三种模型所需的输入视图。"""

    def __init__(self, d, feats):
        self.d = d
        self.feats = feats
        self.N = len(d)
        self.y = d[C.TARGET].to_numpy(dtype=float)
        self.Xraw = d[feats].to_numpy(dtype=float)
        self.dates = d["Date"].to_numpy()

        pos = np.arange(self.N)
        self.train_end = int(pos[d["Date"] <= pd.Timestamp(C.TRAIN_END)].max())
        self.test_start = int(pos[d["Date"] >= pd.Timestamp(C.TEST_START)].min())
        self.START = C.SEQ_LEN - 1        # 三种模型统一使用同一批起始日

        # 待评估的预报起始日：保证 t+7 落在测试期内
        self.eval_origins = np.arange(self.test_start, self.N - C.HORIZON)
        self.eval_dates = self.dates[self.eval_origins]

        # ---- 目标变换：对径流取对数，压缩极值杠杆、保证预测为正 ----
        self.use_log = (C.TARGET_TRANSFORM == "log")
        self.y_model = np.log(self.y) if self.use_log else self.y.copy()

        # ---- 归一化：统计量仅在训练期上估计，防止数据泄漏 ----
        tr = slice(self.START, self.train_end + 1)
        self.x_scaler = ZScoreScaler().fit(self.Xraw[tr])
        self.y_scaler = ZScoreScaler().fit(self.y_model[tr].reshape(-1, 1))
        self.Xs = self.x_scaler.transform(self.Xraw)
        self.ys = self.y_scaler.transform(self.y_model.reshape(-1, 1)).ravel()
        self.Xseq = make_sequences(self.Xs, C.SEQ_LEN)

        self.train_end_label = str(pd.Timestamp(C.TRAIN_END).date())
        self.test_start_label = str(pd.Timestamp(C.TEST_START).date())

    # -- 不同模型的输入视图 -------------------------------------------------
    # 注意：三种模型输入形式不同但信息对等：
    #   ANN/RF：当天特征向量（已含预构造的 lag/rolling 统计），利用人工特征
    #   LSTM：过去 30 天 × 全部特征的序列，由循环网络自行学习时序依赖
    # 预构造的 lag/rolling 特征对 LSTM 有冗余（它能从序列中学到），
    # 但保留它们不影响正确性，且保证三种模型使用同一套特征子集。
    def X_of(self, origins, model_name):
        if model_name == "LSTM":
            return self.Xseq[origins - self.START]
        return self.Xs[origins]

    def y_direct(self, origins, h):
        return self.ys[origins + h]

    def y_multi(self, origins):
        return np.column_stack([self.ys[origins + k] for k in range(1, C.HORIZON + 1)])

    def inv_y(self, v):
        """标准化空间 -> 原始径流尺度（对数变换的逆变换）。"""
        v = np.asarray(v, dtype=float)
        shp = v.shape
        z = self.y_scaler.inverse_transform(v.reshape(-1, 1)).ravel()
        out = np.exp(z) if self.use_log else z
        return out.reshape(shp)

    def true_at(self, origins, h):
        return self.y[origins + h]


# ===========================================================================
# 单个模型的训练（含 GridSearchCV）
# ===========================================================================
def train_direct(ds, model_name, verbose=0):
    """多步直接预测：为每个预见期 h 训练一个模型，返回 7 个模型的预测与超参数。

    热启动逻辑：若 output/models/ 下已有该模型的已保存文件，
    则在已有权重基础上继续训练（warm start），而非每次从零开始。
    继续训练后再次保存模型，下次运行时在更新后的权重上继续。
    """
    results = {}
    for h in range(1, C.HORIZON + 1):
        t0 = time.time()
        tr_org = np.arange(ds.START, ds.train_end - h + 1)
        Xtr = ds.X_of(tr_org, model_name)
        ytr = ds.y_direct(tr_org, h)

        # --- 热启动：检查是否有已保存的模型 ---
        saved = M.load_model(model_name, "direct", h)
        if saved is not None:
            if model_name == "LSTM":
                est = M.warm_start_train(
                    model_name, Xtr, ytr, saved, "direct", h,
                    seq_len=C.SEQ_LEN, extra_iter=30)
                if est is None:
                    est, best, cvtab = M.grid_search(
                        model_name, Xtr, ytr, verbose=verbose)
                    M.save_model(est, model_name, "direct", h)
                else:
                    best = saved
                    cvtab = pd.DataFrame([{"params": "warm-start",
                                           "rank_test_score": 1,
                                           "mean_test_RMSE": np.nan}])
            else:
                saved.warm_start = True
                if model_name == "ANN":
                    saved.max_iter = 500
                elif model_name == "RF":
                    saved.n_estimators += 200
                saved.fit(Xtr, ytr)
                est = saved
                best = {k: v for k, v in saved.get_params().items()
                        if k in M.PARAM_GRIDS[model_name]}
                cvtab = pd.DataFrame([{"params": str(best), "rank_test_score": 1,
                                       "mean_test_RMSE": np.nan}])
            M.save_model(est, model_name, "direct", h)
            print(f"    [热启动] {model_name:<5} h={h}  从已保存模型继续训练")
        else:
            est, best, cvtab = M.grid_search(model_name, Xtr, ytr, verbose=verbose)
            M.save_model(est, model_name, "direct", h)

        yp_scaled = np.asarray(est.predict(ds.X_of(ds.eval_origins, model_name)),
                               dtype=float).ravel()
        yp = ds.inv_y(yp_scaled)

        results[h] = {
            "pred": yp, "pred_scaled": yp_scaled, "best_params": best,
            "cv_table": cvtab, "n_train": len(tr_org), "sec": time.time() - t0,
        }
        print(f"    [直接] {model_name:<5} h={h}  训练样本={len(tr_org):<5}"
              f" 最优={_fmt_params(best):<48} 用时 {time.time()-t0:6.1f}s")
    return results


def train_multioutput(ds, model_name, verbose=0):
    """多输出预测：训练 1 个模型，同时输出未来 1~7 天径流。

    热启动逻辑：若 output/models/ 下已有该模型的已保存文件，
    则在已有权重基础上继续训练（warm start），而非每次从零开始。
    """
    t0 = time.time()
    tr_org = np.arange(ds.START, ds.train_end - C.HORIZON + 1)
    Xtr = ds.X_of(tr_org, model_name)
    ytr = ds.y_multi(tr_org)

    saved = M.load_model(model_name, "multi", h=1)
    if saved is not None:
        if model_name == "LSTM":
            est = M.warm_start_train(
                model_name, Xtr, ytr, saved, "multi", 1,
                seq_len=C.SEQ_LEN, extra_iter=30)
            if est is None:
                est, best, cvtab = M.grid_search(
                    model_name, Xtr, ytr, verbose=verbose)
                M.save_model(est, model_name, "multi", 1)
            else:
                best = saved
                cvtab = pd.DataFrame([{"params": "warm-start",
                                       "rank_test_score": 1,
                                       "mean_test_RMSE": np.nan}])
        else:
            saved.warm_start = True
            if model_name == "ANN":
                saved.max_iter = 500
            elif model_name == "RF":
                saved.n_estimators += 200
            saved.fit(Xtr, ytr)
            est = saved
            best = {k: v for k, v in saved.get_params().items()
                    if k in M.PARAM_GRIDS[model_name]}
            cvtab = pd.DataFrame([{"params": str(best), "rank_test_score": 1,
                                   "mean_test_RMSE": np.nan}])
        M.save_model(est, model_name, "multi", 1)
        print(f"    [热启动] {model_name:<5} 多输出  从已保存模型继续训练")
    else:
        est, best, cvtab = M.grid_search(model_name, Xtr, ytr, verbose=verbose)
        M.save_model(est, model_name, "multi", h=1)

    pm_scaled = np.asarray(est.predict(ds.X_of(ds.eval_origins, model_name)),
                           dtype=float)
    if pm_scaled.ndim == 1:
        pm_scaled = pm_scaled.reshape(-1, C.HORIZON)
    pm = ds.inv_y(pm_scaled)

    print(f"    [多输出] {model_name:<5} 训练样本={len(tr_org):<5}"
          f" 最优={_fmt_params(best):<48} 用时 {time.time()-t0:6.1f}s")
    return {h: {"pred": pm[:, h - 1], "pred_scaled": pm_scaled[:, h - 1],
                "best_params": best, "cv_table": cvtab,
                "n_train": len(tr_org), "sec": time.time() - t0}
            for h in range(1, C.HORIZON + 1)}


def _fmt_params(p):
    s = ", ".join(f"{k}={v}" for k, v in sorted(p.items()))
    return s if len(s) <= 46 else s[:43] + "..."


# ===========================================================================
# 集成学习
# ===========================================================================
def ensemble_simple_average(base_preds, strategy, model_names):
    """简单平均集成：对个体模型在同一预见期上的预测取算术平均。"""
    out = {}
    for h in range(1, C.HORIZON + 1):
        stack = np.vstack([base_preds[m][strategy][h]["pred"] for m in model_names])
        out[h] = {"pred": stack.mean(axis=0),
                  "best_params": {"融合方式": f"{len(model_names)}个模型算术平均",
                                  "参与模型": "+".join(model_names)}}
    return out


def ensemble_stacking(ds, strategy, base_preds, base_est_params,
                      model_names, verbose=0):
    """Stacking 集成：以个体模型预测为元特征，用 Ridge 作为元学习器。

    元学习器的训练样本必须来自【个体模型在训练期内的折外预测(OOF)】，
    否则个体模型在训练集上的拟合优度会被元学习器高估（数据泄漏）。
    这里用 TimeSeriesSplit 生成 OOF，与前面的超参数搜索保持同一套切分逻辑。
    """
    n_splits = C.N_CV_SPLITS
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out = {}
    meta_coefs = []

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
                    # 直接策略：OOF 与测试预测都来自单输出模型，分布一致
                    est = M.MODEL_FACTORY[mname](output_dim=1, **kw)
                    yy = ds.y_direct(tr_org[tr_i], h)
                    est.fit(ds.X_of(tr_org[tr_i], mname), yy)
                    oof[va_i, j] = est.predict(ds.X_of(tr_org[va_i], mname))
                else:
                    # 多输出策略：OOF 必须与测试预测来自同一多输出模型，
                    # 否则元学习器学到的权重作用于不同分布的特征（train-serving skew）
                    est = M.MODEL_FACTORY[mname](output_dim=C.HORIZON, **kw)
                    yy = ds.y_multi(tr_org[tr_i])
                    est.fit(ds.X_of(tr_org[tr_i], mname), yy)
                    p = np.asarray(est.predict(ds.X_of(tr_org[va_i], mname)))
                    oof[va_i, j] = p.ravel() if p.ndim == 1 else p[:, h - 1]

        ok = ~np.isnan(oof).any(axis=1)
        meta = Ridge(alpha=1.0, random_state=C.RANDOM_STATE)
        meta.fit(oof[ok], ytr[ok])

        # 元特征必须与个体模型在测试期的输出处于同一尺度：OOF 与测试预测
        # 都由"标准化空间"的模型给出，这里统一在标准化空间完成融合后再反变换
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
        print(f"    [Stacking-{strategy}] h={h} 元学习器系数={np.round(meta.coef_,3)}"
              f" 截距={meta.intercept_:.4f}")
    return out, pd.DataFrame(meta_coefs)


# ===========================================================================
# 汇总指标
# ===========================================================================
def collect_metrics(preds_by_model, ds, strategy_cn):
    """把某策略下所有模型、所有预见期的预测整理成指标表。"""
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
    """把 1~7 天的预测汇总（拼接）后计算整体指标。"""
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
# 主流程
# ===========================================================================
def run(d, feats, fast=False):
    print("\n" + "=" * 70)
    print("第三部分  径流多步预测")
    print("=" * 70)

    if fast:                      # 小规模验证模式：缩小网格与训练轮数
        M.PARAM_GRIDS["ANN"] = {"hidden_layer_sizes": [(64,)], "alpha": [1e-4]}
        M.PARAM_GRIDS["RF"] = {"n_estimators": [50], "max_depth": [12]}
        M.PARAM_GRIDS["LSTM"] = {"units": [(16,)], "dropout": [0.1], "lr": [5e-2]}
        C.N_CV_SPLITS = 2
        print("[FAST MODE] 缩小超参数网格用于流程验证")

    ds = DataBundle(d, feats)
    print(f"特征数：{len(feats)}   序列长度(LSTM)：{C.SEQ_LEN}")
    print(f"训练期：{ds.d['Date'].iloc[ds.START].date()} ~ {ds.train_end_label}"
          f"（{ds.train_end - ds.START + 1} 天）")
    print(f"测试期：{ds.test_start_label} ~ {ds.d['Date'].iloc[-1].date()}"
          f"（{ds.N - ds.test_start} 天）")
    print(f"评估起始日：{len(ds.eval_origins)} 天"
          f"（{str(pd.Timestamp(ds.eval_dates[0]).date())} ~ "
          f"{str(pd.Timestamp(ds.eval_dates[-1]).date())}）")

    base_preds = {m: {} for m in BASE_MODELS}
    base_est_params = {}
    all_cv = []

    # ---------------- 个体模型 ----------------
    print("\n【1. 个体模型训练（含 GridSearchCV 超参数寻优）】")
    for mname in BASE_MODELS:
        if mname == "LSTM" and not M.torch_available():
            print(f"    [跳过] {mname}：PyTorch 不可用")
            continue
        for strat, strat_cn in STRATEGIES:
            print(f"\n  --- {M.MODEL_CN[mname]} × {strat_cn} ---")
            if strat == "direct":
                res = train_direct(ds, mname)
            else:
                res = train_multioutput(ds, mname)
            base_preds[mname][strat] = res
            base_est_params.setdefault(mname, {})[strat] = {
                h: r["best_params"] for h, r in res.items()}

            # 记录 CV 结果
            for h, r in res.items():
                t = r["cv_table"].copy()
                t.insert(0, "模型", mname)
                t.insert(1, "策略", strat_cn)
                t.insert(2, "预见期", h)
                all_cv.append(t)

            # 保存该模型该策略的预测
            np.savez(os.path.join(C.RESULT_DIR,
                                  f"pred_{mname}_{strat}.npz"),
                     **{f"h{h}": res[h]["pred"] for h in res})

    available = [m for m in BASE_MODELS if base_preds[m]]

    # ---------------- 集成模型 ----------------
    print("\n【2. 集成学习模型】")
    ens_preds = {name: {} for name in ENSEMBLES}
    stack_coef_tables = []
    for strat, strat_cn in STRATEGIES:
        print(f"\n  --- {strat_cn} ---")
        ens_preds["简单平均"][strat] = ensemble_simple_average(
            base_preds, strat, available)
        print(f"    [集成] 简单平均 —— 对 {len(available)} 个个体模型的预测取算术平均")
        st, coef = ensemble_stacking(ds, strat, base_preds, base_est_params,
                                     available)
        ens_preds["Stacking"][strat] = st
        coef.insert(0, "策略", strat_cn)
        stack_coef_tables.append(coef)

        for name in ENSEMBLES:
            np.savez(os.path.join(C.RESULT_DIR, f"pred_ens_{name}_{strat}.npz"),
                     **{f"h{h}": ens_preds[name][strat][h]["pred"]
                        for h in range(1, C.HORIZON + 1)})

    # ---------------- 汇总评价 ----------------
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

    # 最优超参数
    bp_rows = []
    for m in available:
        for strat, strat_cn in STRATEGIES:
            for h, r in base_preds[m][strat].items():
                bp_rows.append({"模型": M.MODEL_CN[m], "策略": strat_cn, "预见期": h,
                                "最优超参数": _fmt_params(r["best_params"]),
                                "训练样本数": r.get("n_train"),
                                "寻优耗时(s)": round(r.get("sec", 0), 1)})
    pd.DataFrame(bp_rows).to_csv(
        os.path.join(C.RESULT_DIR, "12_最优超参数.csv"),
        index=False, encoding="utf-8-sig")

    if all_cv:
        pd.concat(all_cv, ignore_index=True).to_csv(
            os.path.join(C.RESULT_DIR, "13_网格搜索CV结果.csv"),
            index=False, encoding="utf-8-sig")
    if stack_coef_tables:
        pd.concat(stack_coef_tables, ignore_index=True).to_csv(
            os.path.join(C.RESULT_DIR, "14_Stacking元学习器系数.csv"),
            index=False, encoding="utf-8-sig")

    print("\n【多输出预测策略 · 汇总指标（1~7 天拼接）】")
    p = pooled[pooled["策略"] == "多输出预测"]
    print(p[["模型", "RMSE", "MAE", "NSE", "R2", "PBIAS(%)", "NSE等级"]]
          .to_string(index=False))

    np.savez(os.path.join(C.RESULT_DIR, "eval_index.npz"),
             origins=ds.eval_origins, dates=ds.eval_dates.astype("datetime64[D]"),
             observed=ds.y, test_start=ds.test_start, train_end=ds.train_end,
             START=ds.START)

    return ds, all_models, detail, pooled


if __name__ == "__main__":
    import sys
    from data_analysis import load_raw
    from feature_engineering import run as fe_run

    fast = "--fast" in sys.argv
    raw = load_raw()
    d, feature_cols, selected = fe_run(raw)
    run(d, selected, fast=fast)
