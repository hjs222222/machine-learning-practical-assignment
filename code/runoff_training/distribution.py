"""继续训练：随机批次与异方差对数正态 LSTM，不改写上轮实现。"""
import json
from pathlib import Path

import numpy as np
import torch

from acceptance3_model import SequenceNet
from runoff_training.model import LaggedLSTMRegressor


def bounded_variance(raw):
    """方差定义于标准化 ln(Q) 空间；边界是预先固定的模型约束。"""
    return .02 + 1.98 * torch.sigmoid(raw)


def gaussian_nll(output, target):
    mu, raw = output.chunk(2, dim=-1)
    variance = bounded_variance(raw)
    return .5 * (torch.log(variance) + (target - mu).square() / variance).mean()


class DistributionSequenceNet(SequenceNet):
    def __init__(self, n_features, units, dropout, output_dim, residual,
                 x_mean, x_scale, y_mean, y_scale):
        super().__init__(n_features, units, dropout, 2 * output_dim)
        self.residual = residual
        self.register_buffer('x_mean', torch.tensor(float(x_mean), dtype=torch.float32))
        self.register_buffer('x_scale', torch.tensor(float(x_scale), dtype=torch.float32))
        self.register_buffer('y_mean', torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer('y_scale', torch.tensor(y_scale, dtype=torch.float32))
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            # 初始标准化 log 方差为 .25，且不由验证或测试标签估计。
            self.head[-1].bias[output_dim:] = np.log((.25 - .02) / (2. - .25))

    def forward(self, x):
        mu, raw = super().forward(x).chunk(2, dim=-1)
        if self.residual:
            signed_log_q = x[:, -1, :1] * self.x_scale + self.x_mean
            q = torch.sign(signed_log_q) * torch.expm1(signed_log_q.abs())
            mu = mu + (torch.log(q.clamp_min(1e-6)) - self.y_mean) / self.y_scale
        return torch.cat([mu, raw], dim=-1)


class DistributionLSTMRegressor(LaggedLSTMRegressor):
    def __init__(self, dataset_path, horizon=7, units=(32,), dropout=.1,
                 lr=.001, fixed_epochs=20, batch_size=64, seed=42, device='cpu',
                 history_dir=None, phase='development', residual=False,
                 objective='lognormal', weight_decay=.001):
        super().__init__(dataset_path, horizon=horizon, units=units, dropout=dropout,
                         lr=lr, fixed_epochs=fixed_epochs, batch_size=batch_size,
                         seed=seed, device=device, history_dir=history_dir, phase=phase,
                         feature_mode='compact_lag', input_transform='signed_log',
                         residual=residual, weight_decay=weight_decay)
        self.objective = objective

    def _network(self):
        if self.objective == 'log_mse':
            return super()._network()
        if self.objective != 'lognormal':
            raise ValueError(self.objective)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        return DistributionSequenceNet(len(self.selected_), self.units, self.dropout,
            self.output_dim_, self.residual, self.x_scaler_.mean_[0],
            self.x_scaler_.scale_[0], self.y_scaler_.mean_, self.y_scaler_.scale_).to(self.device)

    def _train_network(self, ids, y, epochs, validation=None):
        if validation is not None:
            raise ValueError('只允许历史网格选择固定 epoch')
        net = self._network()
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x = torch.as_tensor(self._inputs(ids), device=self.device)
        z = torch.as_tensor(self.y_scaler_.transform(self._target(y)).astype('float32'), device=self.device)
        rng = np.random.default_rng(self.seed)
        history = []
        for epoch in range(1, epochs + 1):
            net.train()
            total = 0.
            # 仅打乱当前训练折的独立历史窗口，不打乱时间划分、不接触验证标签。
            order = rng.permutation(len(ids))
            for start in range(0, len(x), self.batch_size):
                batch = torch.as_tensor(order[start:start + self.batch_size], device=self.device)
                optimizer.zero_grad(set_to_none=True)
                output = net(x[batch])
                loss = (gaussian_nll(output, z[batch]) if self.objective == 'lognormal'
                        else torch.nn.functional.mse_loss(output, z[batch]))
                if not torch.isfinite(loss):
                    raise ValueError('训练损失非有限；不裁剪预测掩盖失败')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                optimizer.step()
                total += loss.item() * len(batch)
            history.append(dict(epoch=epoch, loss=total / len(ids)))
        net.eval()
        return net, history, epochs

    def predict(self, X):
        if self.objective == 'log_mse':
            return super().predict(X)
        ids = np.asarray(X).reshape(-1).astype(int)
        x = self._inputs(ids)
        self.model_.eval()
        with torch.no_grad():
            output = torch.cat([self.model_(torch.as_tensor(x[i:i + 128], device=self.device))
                                for i in range(0, len(x), 128)])
            mu, raw = output.chunk(2, dim=-1)
            mu = mu.cpu().numpy().astype('float64')
            variance = bounded_variance(raw).cpu().numpy().astype('float64')
        # E[Q|x] = exp(mu_log + sigma_log^2 / 2)，而不是对数空间的中位数。
        log_mu = self.y_scaler_.inverse_transform(mu)
        log_variance = variance * np.square(self.y_scaler_.scale_)
        pred = np.exp(log_mu + .5 * log_variance)
        if not np.isfinite(pred).all():
            raise ValueError('对数正态条件均值非有限')
        return pred.ravel() if self.output_dim_ == 1 else pred

    def save(self, folder):
        super().save(folder)
        file = Path(folder) / 'inference.json'
        metadata = json.loads(file.read_text(encoding='utf-8'))
        metadata.update(objective=self.objective, batch_order='seeded_shuffle_within_train_only',
            optimizer='AdamW', variance_bounds_standardized_log=[.02, 2.],
            inverse_domain_rule=('exp(mu_log + variance_log / 2)' if self.objective == 'lognormal'
                                 else 'exp(mu_log)'),
            distribution_assumption='七个边际分布；不声称预测联合协方差或已校准区间')
        file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
