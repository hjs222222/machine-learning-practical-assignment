"""原生单/七输出 LSTM：因果滞后输入、折内缩放和可选持久性残差。"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from acceptance3_model import SequenceNet, load_dataset
from acceptance3_v3_model import V3Regressor


COMPACT = ['Discharge_current', 'Prcp', 'Tmax', 'Tmin', 'Vp', 'Srad',
           'Dayl', 'doy_sin', 'doy_cos']
LAGS = [f'{variable}_lag{lag}' for variable in ['Discharge', 'Prcp']
        for lag in [1, 3, 7, 14]]


class ResidualSequenceNet(SequenceNet):
    def __init__(self, n_features, units, dropout, output_dim, q_index,
                 x_mean, x_scale, y_mean, y_scale, input_transform):
        super().__init__(n_features, units, dropout, output_dim)
        self.q_index = q_index
        self.input_transform = input_transform
        self.register_buffer('x_mean', torch.tensor(float(x_mean), dtype=torch.float32))
        self.register_buffer('x_scale', torch.tensor(float(x_scale), dtype=torch.float32))
        self.register_buffer('y_mean', torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer('y_scale', torch.tensor(y_scale, dtype=torch.float32))
        # 零初始化使初始预测恰为起报日流量，而不是随机流量。
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        q = x[:, -1, self.q_index:self.q_index + 1] * self.x_scale + self.x_mean
        if self.input_transform == 'signed_log':
            q = torch.sign(q) * torch.expm1(q.abs())
        baseline = (torch.log(q.clamp_min(1e-6)) - self.y_mean) / self.y_scale
        return baseline + super().forward(x)


class LaggedLSTMRegressor(V3Regressor):
    def __init__(self, dataset_path, horizon=7, units=(64,), dropout=.1,
                 lr=.001, fixed_epochs=20, batch_size=64, seed=42, device='cpu',
                 history_dir=None, phase='development', feature_mode='compact_lag',
                 input_transform='signed_log', residual=True, weight_decay=0.):
        super().__init__(dataset_path, family='LSTM', horizon=horizon, units=units,
                         dropout=dropout, lr=lr, fixed_epochs=fixed_epochs,
                         batch_size=batch_size, seed=seed, device=device,
                         history_dir=history_dir, phase=phase,
                         feature_mode=feature_mode, target_transform='log')
        self.input_transform = input_transform
        self.residual = residual
        self.weight_decay = weight_decay

    def _transform_x(self, x):
        if self.input_transform == 'identity':
            return x
        if self.input_transform != 'signed_log':
            raise ValueError(self.input_transform)
        # 符号对数不估计数据统计量；正流量/降雨压缩，负温度符号保留。
        return np.sign(x) * np.log1p(np.abs(x))

    def _preprocess_fit(self, ids, y):
        data = load_dataset(self.dataset_path)
        names = data['feature_names'].tolist()
        if self.feature_mode not in ['compact', 'compact_lag']:
            raise ValueError(self.feature_mode)
        chosen = COMPACT + (LAGS if self.feature_mode == 'compact_lag' else [])
        self.selected_ = np.asarray([names.index(n) for n in chosen])
        positions = data['positions'][ids]
        rows = np.unique((positions[:, None] + np.arange(-29, 1)).ravel())
        self.x_scaler_ = StandardScaler().fit(self._transform_x(
            data['features'][rows][:, self.selected_]))
        self.y_scaler_ = StandardScaler().fit(self._target(y))
        self.mi_ = np.array([])

    def _inputs(self, ids):
        data = load_dataset(self.dataset_path)
        positions = data['positions'][ids]
        x = data['features'][positions[:, None] + np.arange(-29, 1)]
        x = self._transform_x(x[:, :, self.selected_])
        return self.x_scaler_.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype('float32')

    def _network(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if not self.residual:
            return super()._network()
        return ResidualSequenceNet(len(self.selected_), self.units, self.dropout,
            self.output_dim_, 0, self.x_scaler_.mean_[0], self.x_scaler_.scale_[0],
            self.y_scaler_.mean_.astype('float32'), self.y_scaler_.scale_.astype('float32'),
            self.input_transform).to(self.device)

    def _train_network(self, ids, y, epochs, validation=None):
        if validation is not None:
            raise ValueError('本轮只允许网格显式选择固定 epoch')
        net = self._network()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        x = torch.as_tensor(self._inputs(ids), device=self.device)
        z = torch.as_tensor(self.y_scaler_.transform(self._target(y)).astype('float32'),
                            device=self.device)
        history = []
        for epoch in range(1, epochs + 1):
            net.train()
            total = 0.
            for start in range(0, len(x), self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                output = net(x[start:start + self.batch_size])
                loss = torch.nn.functional.mse_loss(output, z[start:start + self.batch_size])
                if not torch.isfinite(loss):
                    raise ValueError('训练损失非有限')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                optimizer.step()
                total += loss.item() * len(output)
            history.append(dict(epoch=epoch, loss=total / len(x)))
        net.eval()
        return net, history, epochs
