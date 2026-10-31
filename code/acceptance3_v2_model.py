# -*- coding: utf-8 -*-
"""第三次验收第二轮模型：原尺度早停、显式当日流量与可选目标损失。"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from acceptance3_model import RunoffRegressor, load_dataset


_CACHE = {}


class V2Regressor(RunoffRegressor):
    def __init__(self, dataset_path, family="LSTM", horizon=1, units=(64,),
                 dropout=0.2, lr=0.001, hidden_layer_sizes=(64,), alpha=0.1,
                 learning_rate_init=0.001, n_estimators=300, max_depth=10,
                 min_samples_leaf=1, max_epochs=150, patience=15,
                 fixed_epochs=None, batch_size=32, top_k=50, seed=42,
                 device="cpu", history_dir=None, phase="development",
                 ann_max_iter=1000, feature_mode="plus_Q", target_transform="log",
                 mixed_weight=0.0, min_relative_improvement=0.001):
        super().__init__(dataset_path, family, horizon, units, dropout, lr,
                         hidden_layer_sizes, alpha, learning_rate_init,
                         n_estimators, max_depth, min_samples_leaf, max_epochs,
                         patience, fixed_epochs, batch_size, top_k, seed,
                         device, history_dir, phase, ann_max_iter)
        self.feature_mode = feature_mode
        self.target_transform = target_transform
        self.mixed_weight = mixed_weight
        self.min_relative_improvement = min_relative_improvement

    def _target(self, y):
        y = np.asarray(y).reshape(len(y), -1)
        if self.target_transform == "log":
            return np.log(y)
        if self.target_transform == "sqrt":
            return np.sqrt(y)
        raise ValueError(self.target_transform)

    def _inverse(self, z):
        transformed = self.y_scaler_.inverse_transform(np.asarray(z))
        if self.target_transform == "log":
            return np.exp(transformed)
        # 平方根目标的逆变换预先限定其定义域，不按测试误差事后截断。
        return np.maximum(transformed, 0) ** 2

    def _preprocess_fit(self, ids, y):
        data = load_dataset(self.dataset_path)
        key = (self.dataset_path, self.horizon, self.family == "LSTM", self.top_k,
               self.seed, self.feature_mode, self.target_transform,
               ids.tobytes(), np.asarray(y).tobytes())
        if key in _CACHE:
            self.selected_, self.x_scaler_, self.y_scaler_, self.mi_ = copy.deepcopy(_CACHE[key])
            return
        names = data["feature_names"].tolist()
        original = int(data["original_feature_count"])
        positions = data["positions"][ids]
        mi = mutual_info_regression(data["features"][positions, :original],
                                   np.log(y[:, 0] if y.ndim == 2 else y), random_state=self.seed)
        if self.feature_mode == "compact":
            selected_names = ["Discharge_current", "Prcp", "Tmax", "Tmin", "Vp",
                              "Srad", "Dayl", "doy_sin", "doy_cos"]
            selected = np.asarray([names.index(n) for n in selected_names])
        else:
            selected = np.argsort(-mi, kind="stable")[:min(self.top_k, original)]
            if self.feature_mode == "plus_Q":
                selected = np.append(selected, names.index("Discharge_current"))
            elif self.feature_mode != "engineered":
                raise ValueError(self.feature_mode)
        rows = np.unique((positions[:, None] + np.arange(-29, 1)).ravel()) if self.family == "LSTM" else positions
        self.selected_, self.mi_ = selected, mi
        self.x_scaler_ = StandardScaler().fit(data["features"][rows][:, selected])
        # 基类ANN/RF训练调用np.log(y)，借助独立覆盖fit处理不同目标。
        self.y_scaler_ = StandardScaler().fit(self._target(y))
        _CACHE[key] = copy.deepcopy((selected, self.x_scaler_, self.y_scaler_, mi))

    def _train_network(self, ids, y, epochs, validation=None):
        net = self._network()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        x = torch.as_tensor(self._inputs(ids), device=self.device)
        z = torch.as_tensor(self.y_scaler_.transform(self._target(y)).astype("float32"), device=self.device)
        physical_y = torch.as_tensor(np.asarray(y).reshape(len(y), -1).astype("float32"), device=self.device)
        physical_scale = torch.as_tensor(np.maximum(np.std(np.asarray(y).reshape(len(y), -1), axis=0), 1e-6).astype("float32"), device=self.device)
        mean = torch.as_tensor(self.y_scaler_.mean_.astype("float32"), device=self.device)
        scale = torch.as_tensor(self.y_scaler_.scale_.astype("float32"), device=self.device)
        if validation is not None:
            vi, vy = validation
            vx = torch.as_tensor(self._inputs(vi), device=self.device)
            truth = np.asarray(vy).reshape(len(vy), -1)
            vz = torch.as_tensor(self.y_scaler_.transform(self._target(vy)).astype("float32"), device=self.device)
        history, best_score, best_epoch, wait = [], float("inf"), 1, 0
        for epoch in range(1, epochs + 1):
            net.train()
            total = 0.0
            for start in range(0, len(x), self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                output = net(x[start:start + self.batch_size])
                loss = torch.nn.functional.mse_loss(output, z[start:start + self.batch_size])
                if self.mixed_weight:
                    transformed = output * scale + mean
                    prediction = torch.exp(transformed) if self.target_transform == "log" else torch.relu(transformed).square()
                    raw_loss = ((prediction - physical_y[start:start + self.batch_size]) / physical_scale).square().mean()
                    loss = (1 - self.mixed_weight) * loss + self.mixed_weight * raw_loss
                if not torch.isfinite(loss):
                    raise ValueError("训练损失非有限，保留失败而不修改测试预测")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(x[start:start + self.batch_size])
            row = {"epoch": epoch, "loss": total / len(x)}
            if validation is not None:
                net.eval()
                with torch.no_grad():
                    output = net(vx)
                    row["val_loss"] = float(torch.nn.functional.mse_loss(output, vz).item())
                prediction = self._inverse(output.cpu().numpy())
                if not np.isfinite(prediction).all():
                    raise ValueError("验证预测非有限")
                score = float(np.sqrt(np.mean((prediction - truth) ** 2, axis=0)).mean())
                row.update(val_RMSE=score, val_MAE=float(np.abs(prediction - truth).mean()))
                if np.isinf(best_score) or score < best_score * (1 - self.min_relative_improvement):
                    best_score, best_epoch, wait = score, epoch, 0
                else:
                    wait += 1
            history.append(row)
            if validation is not None and wait >= self.patience:
                break
        net.eval()
        return net, history, best_epoch

    def fit(self, X, y):
        if self.family == "LSTM":
            # 基类15%内部验证在本轮较大内折上至少90天；末尾留7天净化。
            ids = np.asarray(X).reshape(-1).astype(int)
            y = np.asarray(y, dtype=float)
            self.output_dim_, self.n_features_in_ = (1 if y.ndim == 1 else y.shape[1]), 1
            self.audit_ = {"phase": self.phase, "family": self.family, "horizon": self.horizon,
                           "n_train": len(ids), "first_id": int(ids[0]), "last_id": int(ids[-1])}
            if self.fixed_epochs is None:
                n_val = max(90, min(180, int(np.ceil(len(ids) * .15))))
                cut = len(ids) - n_val
                ti, vi = ids[:cut - 7], ids[cut:]
                if len(ti) < (20 if self.max_epochs <= 2 else 365):
                    raise ValueError("内部早停训练样本不足365，请扩大内折或使用固定epoch")
                days = load_dataset(self.dataset_path)["origin_days"]
                assert days[ti[-1]] + 7 < days[vi[0]]
                self._preprocess_fit(ti, y[:cut - 7])
                _, early, self.epochs_ = self._train_network(ti, y[:cut - 7], self.max_epochs, (vi, y[cut:]))
                self.audit_.update(early_history=early, early_train_last_id=int(ti[-1]),
                                   early_validation_first_id=int(vi[0]), early_validation_last_id=int(vi[-1]))
            else:
                self.epochs_ = int(self.fixed_epochs)
            self._preprocess_fit(ids, y)
            self.model_, history, _ = self._train_network(ids, y, self.epochs_)
            self.audit_.update(history=history, selected_epochs=self.epochs_)
        else:
            from sklearn.neural_network import MLPRegressor
            from sklearn.ensemble import RandomForestRegressor
            ids = np.asarray(X).reshape(-1).astype(int)
            y = np.asarray(y, dtype=float)
            self.output_dim_, self.n_features_in_ = (1 if y.ndim == 1 else y.shape[1]), 1
            self.audit_ = {"phase": self.phase, "family": self.family, "horizon": self.horizon,
                           "n_train": len(ids), "first_id": int(ids[0]), "last_id": int(ids[-1])}
            self._preprocess_fit(ids, y)
            z = self.y_scaler_.transform(self._target(y))
            if self.output_dim_ == 1:
                z = z.ravel()
            if self.family == "ANN":
                self.model_ = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                    learning_rate_init=self.learning_rate_init, max_iter=self.ann_max_iter,
                    early_stopping=False, shuffle=False, random_state=self.seed)
            else:
                self.model_ = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf, max_features="sqrt", n_jobs=8, random_state=self.seed)
            self.model_.fit(self._inputs(ids), z)
            if self.family == "ANN":
                self.audit_["n_iter"] = int(self.model_.n_iter_)
        self.feature_names_ = load_dataset(self.dataset_path)["feature_names"][self.selected_].tolist()
        if self.history_dir:
            import uuid
            folder = Path(self.history_dir)
            folder.mkdir(parents=True, exist_ok=True)
            self.audit_file_ = folder / f"{self.phase}_{self.family}_h{self.horizon}_{uuid.uuid4().hex}.json"
            self.audit_["params"] = self.get_params(deep=False)
            self.audit_file_.write_text(json.dumps(self.audit_, ensure_ascii=False, indent=2), encoding="utf-8")
        return self

    def predict(self, X):
        ids = np.asarray(X).reshape(-1).astype(int)
        x = self._inputs(ids)
        if self.family == "LSTM":
            self.model_.eval()
            with torch.no_grad():
                z = np.concatenate([self.model_(torch.as_tensor(x[i:i + 128], device=self.device)).cpu().numpy()
                                    for i in range(0, len(x), 128)])
        else:
            z = np.asarray(self.model_.predict(x)).reshape(len(ids), -1)
        pred = self._inverse(z)
        if not np.isfinite(pred).all():
            raise ValueError("预测非有限")
        return pred.ravel() if self.output_dim_ == 1 else pred

    def save(self, folder):
        super().save(folder)
        (Path(folder) / "inference.json").write_text(json.dumps({
            "params": self.get_params(), "features": self.feature_names_, "output_dim": self.output_dim_,
            "target_transform": self.target_transform,
            "inverse_domain_rule": "sqrt: max(inverse_standardize(z),0)^2; log: exp(inverse_standardize(z))",
            "weights": "weights.pt" if self.family == "LSTM" else None}, ensure_ascii=False, indent=2), encoding="utf-8")
