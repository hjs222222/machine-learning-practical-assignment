# -*- coding: utf-8 -*-
"""
第三部分：模型与超参数搜索
==========================
个体模型（3 种）：
  1. ANN     —— 人工神经网络（sklearn MLPRegressor，多层感知机）
  2. RF      —— 随机森林（sklearn RandomForestRegressor）
  3. LSTM    —— 长短期记忆网络（PyTorch，支持 GPU 训练）

超参数搜索：GridSearchCV。由于是时间序列，交叉验证必须使用 TimeSeriesSplit
（前向链式切分），普通 K 折会把未来信息泄漏到训练集。

说明：PyTorch 模型没有 sklearn 官方封装，因此实现 LSTMModel 使其具备
fit/predict/get_params/set_params 接口，可直接接入 sklearn 的 GridSearchCV，
从而三种模型使用完全一致的超参数搜索流程。
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import copy

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

import config as C

# ---------------------------------------------------------------------------
# PyTorch 延迟导入：未安装 torch 时其余模型仍可运行
# ---------------------------------------------------------------------------
torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch


def torch_available():
    """检查 PyTorch 是否可用（GPU 版为 torch.cuda.is_available()）。"""
    try:
        _ensure_torch()
        return True
    except Exception as e:      # noqa: BLE001
        print(f"[warn] PyTorch 不可用：{e}")
        return False


def _default_device():
    """优先使用 GPU，否则回退 CPU。"""
    _ensure_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# 1. LSTM —— sklearn 风格封装（PyTorch 实现，支持 GPU）
# ===========================================================================
class LSTMModel(BaseEstimator, RegressorMixin):
    """PyTorch LSTM 回归器，接口兼容 sklearn，可直接用于 GridSearchCV。

    输入 X 的形状为 (样本数, 时间步长, 特征数)。
    输出维度 output_dim：1 表示单步预测（多步直接策略）；7 表示多输出策略。

    网络结构与原 Keras 版等价：
      LSTM 层 × len(units)（非末层输出全序列，层间 Dropout）
      → Dense(32, ReLU) → Dense(output_dim)
    训练行为与原 Keras 版保持一致：
      - 末端 10% 作为验证集（保持时间顺序，不打乱）
      - EarlyStopping（monitor=val_loss, patience, restore_best_weights）
      - Adam 优化器 + MSE 损失，batch 按顺序切分（shuffle=False）
    """

    def __init__(self, seq_len=C.SEQ_LEN, units=(64,), dropout=0.1, lr=5e-3,
                 epochs=60, batch_size=32, patience=6, output_dim=1,
                 head_units=None, random_state=C.RANDOM_STATE, verbose=0,
                 device=None):
        self.seq_len = seq_len
        self.units = units
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.output_dim = output_dim
        self.head_units = head_units
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

    # -- 构建网络 ----------------------------------------------------------
    def _build(self, n_features):
        torch = _ensure_torch()
        nn = torch.nn
        units = tuple(self.units)
        output_dim = self.output_dim
        dropout = self.dropout
        head_units = self.head_units

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstms = nn.ModuleList()
                for i, u in enumerate(units):
                    in_size = n_features if i == 0 else units[i - 1]
                    self.lstms.append(nn.LSTM(in_size, u, batch_first=True))
                self.drop = nn.Dropout(dropout)
                # 输出头中间层宽度：多输出(7)默认 64 以增强表达力，单输出保持 32
                hw = head_units if head_units is not None else (64 if output_dim > 1 else 32)
                self.head = nn.Sequential(
                    nn.Linear(units[-1], hw),
                    nn.ReLU(),
                    nn.Linear(hw, output_dim),
                )

            def forward(self, x):
                for lstm in self.lstms:
                    x, _ = lstm(x)      # 每层输出全序列，等价 Keras return_sequences
                    x = self.drop(x)
                x = x[:, -1, :]         # 取最后时间步（等价末层 return_sequences=False）
                return self.head(x)

        return _Net()

    def _run_training(self, net, opt, loss_fn, Xtr_t, ytr_t, Xva_t, yva_t,
                      epochs, dev):
        """训练循环：按顺序切 batch（不打乱）、末端验证集 EarlyStopping。"""
        best_loss = float("inf")
        best_state = None
        wait = 0
        history = {"loss": [], "val_loss": []}

        for epoch in range(epochs):
            net.train()
            epoch_loss, nb = 0.0, 0
            for i in range(0, len(Xtr_t), self.batch_size):
                xb = Xtr_t[i:i + self.batch_size]
                yb = ytr_t[i:i + self.batch_size]
                opt.zero_grad()
                out = net(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(xb)
                nb += len(xb)
            history["loss"].append(epoch_loss / nb)

            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(Xva_t), yva_t).item()
            history["val_loss"].append(val_loss)

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = copy.deepcopy(net.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    if self.verbose:
                        print(f"  [LSTM] early stop @ epoch {epoch + 1}")
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net, history

    def fit(self, X, y):
        torch = _ensure_torch()
        nn = torch.nn
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        # 末端 10% 验证（保持时间顺序，等价 Keras validation_split=0.1 + shuffle=False）
        n = len(X)
        n_val = max(1, int(round(n * 0.1)))
        Xtr, Xva = X[:-n_val], X[-n_val:]
        ytr, yva = y[:-n_val], y[-n_val:]

        dev = torch.device(self.device) if self.device else _default_device()
        self.device_ = dev
        net = self._build(X.shape[2]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.from_numpy(Xtr).to(dev)
        ytr_t = torch.from_numpy(ytr).to(dev)
        Xva_t = torch.from_numpy(Xva).to(dev)
        yva_t = torch.from_numpy(yva).to(dev)

        self.model_, self.history_ = self._run_training(
            net, opt, loss_fn, Xtr_t, ytr_t, Xva_t, yva_t, self.epochs, dev)
        return self

    def predict(self, X):
        torch = _ensure_torch()
        X = np.asarray(X, dtype=np.float32)
        dev = getattr(self, "device_", _default_device())
        net = self.model_.to(dev)
        net.eval()
        with torch.no_grad():
            p = net(torch.from_numpy(X).to(dev)).cpu().numpy()
        if self.output_dim == 1:
            return p.ravel()
        return p

    @property
    def n_epochs_run(self):
        return len(getattr(self, "history_", {}).get("loss", []))


# ===========================================================================
# 2. 模型工厂
# ===========================================================================
def make_ann(output_dim=1, **kw):
    """人工神经网络（多层感知机）。

    注意：sklearn MLPRegressor 的 early_stopping=True 内部调用
    train_test_split(shuffle=True) 随机划分验证集，对时间序列构成数据泄漏。
    因此这里关闭内置 early_stopping，依赖：
      (1) L2 正则化 alpha（由 GridSearchCV 选优）防止过拟合；
      (2) 充分的 max_iter 保证收敛；
      (3) 外层 TimeSeriesSplit 选出最优泛化模型。
    """
    params = dict(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=False,         # 禁用随机划分验证集，防时间序列泄漏
        random_state=C.RANDOM_STATE,
    )
    params.update(kw)
    return MLPRegressor(**params)


def make_rf(output_dim=1, **kw):
    """随机森林（原生支持多输出，无需 MultiOutputRegressor 包装）。"""
    params = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=C.N_JOBS,
        random_state=C.RANDOM_STATE,
    )
    params.update(kw)
    return RandomForestRegressor(**params)


def make_lstm(output_dim=1, seq_len=C.SEQ_LEN, **kw):
    """LSTM 深度神经网络。"""
    params = dict(seq_len=seq_len, output_dim=output_dim)
    params.update(kw)
    return LSTMModel(**params)


# ---------------------------------------------------------------------------
# 超参数网格（为控制总计算量，取适中规模的网格）
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "ANN": {
        "hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
        "alpha": [1e-5, 1e-4, 1e-2],
        "learning_rate_init": [5e-4, 1e-3, 1e-2],
    },
    "RF": {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [None, 10, 12, 15, 20, 25],
        "min_samples_leaf": [1, 3, 5],
    },
    "LSTM": {
        "units": [(32,), (64,), (32, 32), (64, 32)],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [5e-3, 1e-3],
        "epochs": [60, 80],
    },
}

MODEL_FACTORY = {"ANN": make_ann, "RF": make_rf, "LSTM": make_lstm}

MODEL_CN = {
    "ANN": "人工神经网络 (ANN/MLP)",
    "RF": "随机森林 (RF)",
    "LSTM": "长短期记忆网络 (LSTM)",
}


# ===========================================================================
# 3. 网格搜索
# ===========================================================================
def grid_search(model_name, X, y, seq_len=None, n_splits=None, verbose=0):
    """在时间序列交叉验证下做 GridSearchCV 超参数寻优。

    参数
    ----
    X : ndarray, (n, p) 或 (n, seq_len, p)
    y : ndarray, (n,) 或 (n, horizon)
    返回
    ----
    best_estimator, best_params, cv_table(DataFrame)
    """
    import pandas as pd
    n_splits = n_splits or C.N_CV_SPLITS
    cv = TimeSeriesSplit(n_splits=n_splits)

    grid = PARAM_GRIDS[model_name]
    output_dim = 1 if np.ndim(y) == 1 else np.asarray(y).shape[1]
    kwargs = {"output_dim": output_dim}
    if seq_len is not None:
        kwargs["seq_len"] = seq_len

    est = MODEL_FACTORY[model_name](**kwargs)
    gs = GridSearchCV(
        est, grid, cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,          # Keras 模型不可多进程；统一单进程便于复现
        verbose=verbose, refit=True,
    )
    gs.fit(X, y)

    cv_table = pd.DataFrame(gs.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score").reset_index(drop=True)
    cv_table["mean_test_RMSE"] = -cv_table["mean_test_score"]
    cv_table = cv_table.drop(columns=["mean_test_score"])
    return gs.best_estimator_, gs.best_params_, cv_table


# ===========================================================================
# 4. 模型持久化与热启动（warm start）
#    训练好的最优模型保存到磁盘，下次运行时在已有权重基础上继续训练，
#    而非每次随机初始化从零开始。
# ===========================================================================
def _model_path(model_name, strategy, h):
    """生成模型保存路径。"""
    fname = f"{model_name}_{strategy}_h{h}.joblib"
    return os.path.join(C.MODEL_DIR, fname)


def save_model(model, model_name, strategy, h):
    """保存训练好的模型到磁盘。"""
    import joblib
    path = _model_path(model_name, strategy, h)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if model_name == "LSTM":
        # PyTorch 权重单独保存为 .pt
        torch_path = path.replace(".joblib", ".pt")
        torch.save(model.model_.state_dict(), torch_path)
        joblib.dump({"_torch_weights_path": torch_path}, path)
    else:
        joblib.dump(model, path)
    return path


def load_model(model_name, strategy, h):
    """从磁盘加载已保存的模型；不存在返回 None。"""
    import joblib
    path = _model_path(model_name, strategy, h)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def warm_start_train(model_name, X, y, best_params, strategy, h,
                     seq_len=None, extra_iter=500):
    """在已有最优模型基础上继续训练（热启动）。

    - ANN：设置 warm_start=True，在已有权重上继续梯度下降
    - RF：设置 warm_start=True，在已有树基础上增加新树
    - LSTM：加载已保存的 PyTorch 权重，继续训练更多 epochs

    返回训练后的模型；没有已保存模型时返回 None。
    """
    saved = load_model(model_name, strategy, h)
    if saved is None:
        return None

    output_dim = 1 if np.ndim(y) == 1 else np.asarray(y).shape[1]
    kw = dict(best_params)

    if model_name == "ANN":
        est = make_ann(output_dim=output_dim, **kw)
        est.warm_start = True       # sklearn 关键参数：继续训练而非重新初始化
        est.max_iter = extra_iter
        est.fit(X, y)
        print(f"    [热启动] ANN h={h}  从已保存模型继续训练 {extra_iter} 轮")
        return est

    elif model_name == "RF":
        est = make_rf(output_dim=output_dim, **kw)
        est.warm_start = True       # 在已有树基础上增加新树
        est.n_estimators = extra_iter
        est.fit(X, y)
        print(f"    [热启动] RF  h={h}  从已保存模型增加 {extra_iter} 棵树")
        return est

    elif model_name == "LSTM":
        torch_path = _model_path(model_name, strategy, h).replace(
            ".joblib", ".pt")
        if not os.path.exists(torch_path):
            return None
        seq_len_val = seq_len or C.SEQ_LEN
        est = make_lstm(output_dim=output_dim, seq_len=seq_len_val, **kw)
        X_sample = np.asarray(X[:1], dtype=np.float32)
        dev = torch.device(est.device) if est.device else _default_device()
        est.device_ = dev
        est.model_ = est._build(X_sample.shape[2]).to(dev)
        est.model_.load_state_dict(
            torch.load(torch_path, map_location=dev, weights_only=True))
        # 继续训练 extra_iter 轮（末端 10% 验证 + EarlyStopping）
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        n = len(X)
        n_val = max(1, int(round(n * 0.1)))
        opt = torch.optim.Adam(est.model_.parameters(), lr=est.lr)
        loss_fn = torch.nn.MSELoss()
        Xtr_t = torch.from_numpy(np.asarray(X[:-n_val], dtype=np.float32)).to(dev)
        ytr_t = torch.from_numpy(y_arr[:-n_val]).to(dev)
        Xva_t = torch.from_numpy(np.asarray(X[-n_val:], dtype=np.float32)).to(dev)
        yva_t = torch.from_numpy(y_arr[-n_val:]).to(dev)
        est.model_, est.history_ = est._run_training(
            est.model_, opt, loss_fn, Xtr_t, ytr_t, Xva_t, yva_t,
            extra_iter, dev)
        print(f"    [热启动] LSTM h={h}  从已保存权重继续训练 {extra_iter} epochs")
        return est

    return None
