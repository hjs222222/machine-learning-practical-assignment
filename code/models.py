"""原生单/多输出模型；每次 fit 内重新做选择、缩放和训练。"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import torch
from torch import nn

torch.set_num_threads(2)


class SequenceNet(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.lstm = nn.LSTM(inputs, hidden, batch_first=True)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1])


class RunoffRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, family='RF', transform='raw', top_k=24, hidden=32, epochs=40,
                 trees=120, leaf=2, iterations=300, alpha=0.1, seed=42, seq_len=14, residual=False,
                 forced_indices=(), mi_target='raw', max_features=1.0):
        self.family = family
        self.transform = transform
        self.top_k = top_k
        self.hidden = hidden
        self.epochs = epochs
        self.trees = trees
        self.leaf = leaf
        self.iterations = iterations
        self.alpha = alpha
        self.seed = seed
        self.seq_len = seq_len
        self.residual = residual
        self.forced_indices = forced_indices
        self.mi_target = mi_target
        self.max_features = max_features

    def target(self, y):
        if self.transform == 'log':
            return np.log1p(y)
        if self.transform == 'sqrt':
            return np.sqrt(y)
        if self.transform != 'raw':
            raise ValueError('未知目标变换')
        return y

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)[:, -self.seq_len:]
        y = np.asarray(y).reshape(len(X), -1)
        if not np.isfinite(X).all() or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('模型输入/目标无效')
        self.outputs_ = y.shape[1]
        target = self.target(y)
        if self.residual:
            target = target - self.target(X[:,-1,:1])
        # 多输出用逐 h MI 的均值，不偷用验证标签，也不只偏向单一输出。
        if self.mi_target not in ('raw', 'transformed', 'model_target'):
            raise ValueError('未知 MI 目标')
        if any(i < 0 or i >= X.shape[-1] for i in self.forced_indices):
            raise ValueError('强制特征索引越界')
        if self.top_k is None:
            self.mi_ = None
            self.selected_ = np.arange(X.shape[-1])
        else:
            mi_y = y if self.mi_target == 'raw' else self.target(y) if self.mi_target == 'transformed' else target
            self.mi_ = np.mean([mutual_info_regression(X[:,-1], mi_y[:,h], random_state=self.seed)
                                for h in range(y.shape[1])], axis=0)
            self.selected_ = np.union1d(np.argsort(self.mi_, kind='stable')[-min(self.top_k,X.shape[-1]):],
                                        self.forced_indices).astype(int)
        part = X[:,:,self.selected_]
        self.x_scaler_ = StandardScaler().fit(part.reshape(-1,len(self.selected_)))
        self.y_scaler_ = StandardScaler().fit(target)
        x = self.x_scaler_.transform(part.reshape(-1,len(self.selected_))).reshape(part.shape).astype('float32')
        z = self.y_scaler_.transform(target).astype('float32')
        if self.family == 'RF':
            self.estimator_ = RandomForestRegressor(n_estimators=self.trees, min_samples_leaf=self.leaf,
                max_features=self.max_features, random_state=self.seed, n_jobs=4)
            self.estimator_.fit(x[:,-1], z.ravel() if self.outputs_ == 1 else z)
        elif self.family == 'ANN':
            self.estimator_ = MLPRegressor(hidden_layer_sizes=(self.hidden,), max_iter=self.iterations,
                early_stopping=False, shuffle=False, alpha=self.alpha, random_state=self.seed,
                learning_rate_init=0.003, batch_size=min(64,len(x)), tol=1e-5)
            self.estimator_.fit(x[:,-1], z.ravel() if self.outputs_ == 1 else z)
        elif self.family == 'LSTM':
            self._fit_lstm(x,z)
        else:
            raise ValueError('未知模型家族')
        return self

    def _fit_lstm(self, x, z):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = SequenceNet(x.shape[-1],self.hidden,z.shape[1]).to(device)
        optimizer = torch.optim.Adam(net.parameters(),lr=0.003,weight_decay=self.alpha*0.001)
        tx, ty = torch.tensor(x,device=device), torch.tensor(z,device=device)
        self.loss_history_ = []
        # epochs 由外部 purged 时序 CV 选择；不启用 sklearn 的随机验证拆分。
        for _ in range(self.epochs):
            net.train()
            loss_sum = 0.0
            for start in range(0,len(x),128):
                optimizer.zero_grad()
                pred = net(tx[start:start+128])
                loss = nn.functional.mse_loss(pred,ty[start:start+128])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.0)
                optimizer.step()
                loss_sum += loss.item()*len(pred)
            self.loss_history_.append(loss_sum/len(x))
        self.estimator_ = net.cpu().eval()
        self.device_used_ = device

    def predict(self, X):
        current = np.asarray(X)[:,-1,:1]
        X = np.asarray(X,dtype=np.float32)[:, -self.seq_len:, self.selected_]
        x = self.x_scaler_.transform(X.reshape(-1,len(self.selected_))).reshape(X.shape).astype('float32')
        if self.family == 'LSTM':
            with torch.no_grad():
                prediction = self.estimator_(torch.from_numpy(x)).numpy()
        else:
            prediction = self.estimator_.predict(x[:,-1])
        z = self.y_scaler_.inverse_transform(np.asarray(prediction).reshape(len(X),-1))
        if self.residual:
            z += self.target(current)
        if self.transform == 'log':
            z = np.expm1(z)
        elif self.transform == 'sqrt':
            z = np.maximum(z,0)**2
        return np.maximum(z,0)


def simple_average(predictions):
    if len(predictions) != 3 or len({p.shape for p in predictions}) != 1:
        raise ValueError('简单平均需要三个同形状基模型结果')
    return np.mean(predictions,axis=0)


def fit_stacking(predictions, truth):
    X = np.concatenate(predictions,axis=1)
    if not np.isfinite(X).all() or len(X) != len(truth):
        raise ValueError('OOF 元特征缺失或错位')
    # 输入量纲相同但标准化使正则项不被流量单位左右。
    model = make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(X,truth)
    model.coef_ = model[-1].coef_
    return model
