# -*- coding: utf-8 -*-
"""第三阶段独立模型：折内预处理、净化早停与可恢复的原生多输出。"""
import copy
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@lru_cache(maxsize=4)
def load_dataset(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


class SequenceNet(torch.nn.Module):
    def __init__(self, n_features, units, dropout, output_dim):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i, width in enumerate(units):
            self.layers.append(torch.nn.LSTM(
                n_features if i == 0 else units[i - 1], width, batch_first=True))
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(units[-1], 32 if output_dim == 1 else 64),
            torch.nn.ReLU(),
            torch.nn.Linear(32 if output_dim == 1 else 64, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
            x = self.drop(x)
        return self.head(x[:, -1, :])


_PREPROCESS_CACHE = {}


class RunoffRegressor(RegressorMixin, BaseEstimator):
    """X 为样本索引；数据集只含因果输入，所有监督处理均在 fit 内。"""
    def __init__(self, dataset_path, family="LSTM", horizon=1, units=(64,),
                 dropout=0.2, lr=0.001, hidden_layer_sizes=(64,), alpha=0.01,
                 learning_rate_init=0.001, n_estimators=300, max_depth=10,
                 min_samples_leaf=1, max_epochs=80, patience=6, fixed_epochs=None,
                 batch_size=32, top_k=50, seed=42, device="cuda", history_dir=None,
                 phase="calibration", ann_max_iter=1000):
        self.dataset_path = dataset_path
        self.family = family
        self.horizon = horizon
        self.units = units
        self.dropout = dropout
        self.lr = lr
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_epochs = max_epochs
        self.patience = patience
        self.fixed_epochs = fixed_epochs
        self.batch_size = batch_size
        self.top_k = top_k
        self.seed = seed
        self.device = device
        self.history_dir = history_dir
        self.phase = phase
        self.ann_max_iter = ann_max_iter

    def _preprocess_fit(self, ids, y):
        data = load_dataset(self.dataset_path)
        # 参数候选共享的确定性折内计算可以缓存；键包含全部训练行和目标。
        key = (self.dataset_path, self.horizon, self.family == "LSTM",
               self.top_k, self.seed, ids.tobytes(), np.asarray(y).tobytes())
        if key in _PREPROCESS_CACHE:
            saved = copy.deepcopy(_PREPROCESS_CACHE[key])
            self.selected_, self.x_scaler_, self.y_scaler_, self.mi_ = saved
            return
        positions = data["positions"][ids]
        last = data["features"][positions]
        target = np.log(y[:, 0] if y.ndim == 2 else y)
        mi = mutual_info_regression(last, target, random_state=self.seed)
        self.mi_ = mi
        self.selected_ = np.argsort(-mi, kind="stable")[:min(self.top_k, len(mi))]
        if self.family == "LSTM":
            rows = np.unique((positions[:, None] + np.arange(-29, 1)).ravel())
        else:
            rows = positions
        self.x_scaler_ = StandardScaler().fit(data["features"][rows][:, self.selected_])
        self.y_scaler_ = StandardScaler().fit(np.log(y).reshape(len(y), -1))
        _PREPROCESS_CACHE[key] = copy.deepcopy((
            self.selected_, self.x_scaler_, self.y_scaler_, self.mi_))

    def _inputs(self, ids):
        data = load_dataset(self.dataset_path)
        positions = data["positions"][ids]
        if self.family == "LSTM":
            x = data["features"][positions[:, None] + np.arange(-29, 1)]
            x = x[:, :, self.selected_]
            return self.x_scaler_.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype("float32")
        return self.x_scaler_.transform(data["features"][positions][:, self.selected_])

    def _network(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        return SequenceNet(len(self.selected_), self.units, self.dropout,
                           self.output_dim_).to(self.device)

    def _train_network(self, ids, y, epochs, validation=None):
        net = self._network()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        x = torch.as_tensor(self._inputs(ids), device=self.device)
        z = torch.as_tensor(self.y_scaler_.transform(
            np.log(y).reshape(len(y), -1)).astype("float32"), device=self.device)
        if validation is not None:
            vi, vy = validation
            vx = torch.as_tensor(self._inputs(vi), device=self.device)
            vz = torch.as_tensor(self.y_scaler_.transform(
                np.log(vy).reshape(len(vy), -1)).astype("float32"), device=self.device)
        history, best_loss, best_epoch, wait = [], float("inf"), 1, 0
        for epoch in range(1, epochs + 1):
            net.train()
            total = 0.0
            for start in range(0, len(x), self.batch_size):
                xb, yb = x[start:start + self.batch_size], z[start:start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(net(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)
            row = {"epoch": epoch, "loss": total / len(x)}
            if validation is not None:
                net.eval()
                with torch.no_grad():
                    val = loss_fn(net(vx), vz).item()
                row["val_loss"] = val
                if val < best_loss - 1e-5:
                    best_loss, best_epoch, wait = val, epoch, 0
                else:
                    wait += 1
            history.append(row)
            if validation is not None and wait >= self.patience:
                break
        net.eval()
        return net, history, best_epoch

    def fit(self, X, y):
        ids = np.asarray(X).reshape(-1).astype(int)
        y = np.asarray(y, dtype=float)
        self.output_dim_ = 1 if y.ndim == 1 else y.shape[1]
        self.n_features_in_ = 1
        self.audit_ = {"phase": self.phase, "family": self.family,
                       "horizon": self.horizon, "n_train": len(ids),
                       "first_id": int(ids[0]), "last_id": int(ids[-1])}
        if self.family == "LSTM":
            if self.fixed_epochs is None:
                n_val = max(12, int(np.ceil(len(ids) * .15)))
                cut = len(ids) - n_val
                ti, vi = ids[:cut - 7], ids[cut:]
                assert len(ti) > 5
                data = load_dataset(self.dataset_path)
                assert data["origin_days"][ti[-1]] + self.horizon < data["origin_days"][vi[0]]
                self._preprocess_fit(ti, y[:cut - 7])
                _, early, best = self._train_network(
                    ti, y[:cut - 7], self.max_epochs, (vi, y[cut:]))
                self.audit_["early_history"] = early
                self.audit_["early_train_last_id"] = int(ti[-1])
                self.audit_["early_validation_first_id"] = int(vi[0])
                self.epochs_ = best
            else:
                self.epochs_ = int(self.fixed_epochs)
            self._preprocess_fit(ids, y)
            self.model_, history, _ = self._train_network(ids, y, self.epochs_)
            self.audit_["history"] = history
            self.audit_["selected_epochs"] = self.epochs_
        else:
            self._preprocess_fit(ids, y)
            x = self._inputs(ids)
            z = self.y_scaler_.transform(np.log(y).reshape(len(y), -1))
            if self.output_dim_ == 1:
                z = z.ravel()
            if self.family == "ANN":
                self.model_ = MLPRegressor(
                    hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                    learning_rate_init=self.learning_rate_init, max_iter=self.ann_max_iter,
                    early_stopping=False, shuffle=False, random_state=self.seed)
            else:
                self.model_ = RandomForestRegressor(
                    n_estimators=self.n_estimators, max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf, max_features="sqrt",
                    n_jobs=8, random_state=self.seed)
            self.model_.fit(x, z)
            if self.family == "ANN":
                self.audit_["n_iter"] = int(self.model_.n_iter_)
        self.feature_names_ = load_dataset(self.dataset_path)["feature_names"][self.selected_].tolist()
        if self.history_dir:
            folder = Path(self.history_dir)
            folder.mkdir(parents=True, exist_ok=True)
            # 每次实际 fit 都有独立证据，不把上限轮数当作真实训练轮数。
            import uuid
            self.audit_file_ = folder / f"{self.phase}_{self.family}_h{self.horizon}_{uuid.uuid4().hex}.json"
            self.audit_["params"] = self.get_params(deep=False)
            self.audit_file_.write_text(json.dumps(self.audit_, ensure_ascii=False, indent=2), encoding="utf-8")
        return self

    def predict(self, X):
        ids = np.asarray(X).reshape(-1).astype(int)
        x = self._inputs(ids)
        if self.family == "LSTM":
            self.model_.eval()
            parts = []
            with torch.no_grad():
                for start in range(0, len(x), 128):
                    parts.append(self.model_(torch.as_tensor(
                        x[start:start + 128], device=self.device)).cpu().numpy())
            z = np.concatenate(parts)
        else:
            z = np.asarray(self.model_.predict(x)).reshape(len(ids), -1)
        pred = np.exp(self.y_scaler_.inverse_transform(z))
        if not np.isfinite(pred).all():
            raise ValueError("预测出现非有限值，不截断掩盖失败")
        return pred.ravel() if self.output_dim_ == 1 else pred

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.family == "LSTM":
            torch.save(self.model_.cpu().state_dict(), folder / "weights.pt")
            net = self.model_
            del self.model_
            try:
                joblib.dump(self, folder / "estimator.joblib")
            finally:
                self.model_ = net.to(self.device)
        else:
            joblib.dump(self, folder / "estimator.joblib")
        (folder / "inference.json").write_text(json.dumps({
            "params": self.get_params(), "features": self.feature_names_,
            "output_dim": self.output_dim_, "weights": "weights.pt" if self.family == "LSTM" else None,
            "target_transform": "ln_then_standardize", "inverse": "inverse_standardize_then_exp"
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def restore_model(folder, dataset_path=None, device="cpu"):
    folder = Path(folder)
    est = joblib.load(folder / "estimator.joblib")
    if dataset_path is not None:
        est.dataset_path = str(dataset_path)
    if est.family == "LSTM":
        est.device = device
        est.model_ = est._network()
        est.model_.load_state_dict(torch.load(folder / "weights.pt", map_location=device, weights_only=True))
        est.model_.eval()
    return est


def physical_scorer(estimator, X, y):
    p = np.asarray(estimator.predict(X)).reshape(len(y), -1)
    true = np.asarray(y).reshape(len(y), -1)
    score = -float(np.sqrt(np.mean((p - true) ** 2, axis=0)).mean())
    if hasattr(estimator, "audit_file_"):
        record = estimator.audit_.copy()
        record["cv_score"] = score
        record["validation_first_id"] = int(np.asarray(X).reshape(-1)[0])
        record["validation_last_id"] = int(np.asarray(X).reshape(-1)[-1])
        estimator.audit_file_.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return score
