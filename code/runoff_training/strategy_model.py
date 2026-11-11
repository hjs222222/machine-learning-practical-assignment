"""局部增强LSTM：等容量时间池化、持久性跳连；不修改已冻结受控模型。"""
import json
from pathlib import Path
import uuid

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from runoff_training.controlled_data import BASE, MAX_WINDOW, load_data, sha256, validate_ids
from runoff_training.controlled_model import ControlledLSTM, target_values


class StrategyNet(torch.nn.Module):
    def __init__(self, n_features, units, output_dim, pooling, residual,
                 x_mean, x_scale, y_mean, y_scale, target_transform):
        super().__init__()
        if pooling not in ['last', 'last_mean'] or target_transform not in ['log', 'sqrt']:
            raise ValueError('只接受预固定池化和目标变换')
        self.pooling, self.residual, self.target_transform = pooling, residual, target_transform
        self.lstm = torch.nn.LSTM(n_features, units, batch_first=True)
        self.drop = torch.nn.Dropout(.1)
        self.head = torch.nn.Sequential(torch.nn.Linear(units, 32), torch.nn.ReLU(),
                                        torch.nn.Linear(32, output_dim))
        self.register_buffer('x_mean', torch.as_tensor(float(x_mean), dtype=torch.float32))
        self.register_buffer('x_scale', torch.as_tensor(float(x_scale), dtype=torch.float32))
        self.register_buffer('y_mean', torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer('y_scale', torch.as_tensor(y_scale, dtype=torch.float32))
        if residual:
            # 与普通头初始化不同，是明确披露的残差训练策略；初始预测严格为Q(t)。
            torch.nn.init.zeros_(self.head[-1].weight)
            torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        states, _ = self.lstm(x)
        context = states[:, -1]
        if self.pooling == 'last_mean':
            context = .5 * (context + states.mean(dim=1))
        correction = self.head(self.drop(context))
        if not self.residual:
            return correction
        q = torch.expm1(x[:, -1, :1]*self.x_scale + self.x_mean).clamp_min(1e-6)
        baseline = torch.log(q) if self.target_transform == 'log' else torch.sqrt(q)
        return correction + (baseline-self.y_mean)/self.y_scale


class StrategyLSTM(ControlledLSTM):
    def __init__(self, dataset_path, profile='w90_base', fixed_epochs=10,
                 seed=42, device='cpu', target_transform='log', units=32,
                 batch_size=64, lr=.001, history_dir=None, phase='development',
                 pooling='last', residual=False):
        super().__init__(dataset_path, profile, fixed_epochs, seed, device, target_transform,
                         units, batch_size, lr, history_dir, phase)
        self.pooling, self.residual = pooling, residual

    def _network(self):
        return StrategyNet(len(self.data_['feature_names']), self.units, self.output_dim_,
            self.pooling, self.residual, self.x_scaler_.mean_[0], self.x_scaler_.scale_[0],
            self.y_scaler_.mean_, self.y_scaler_.scale_, self.target_transform).to(self.device)

    def fit(self, X, y):
        for name in ['fixed_epochs', 'batch_size', 'units']:
            value = getattr(self, name)
            if not np.isfinite(value) or int(value) != value or value < 1:
                raise ValueError(f'{name}必须为正整数')
        if self.profile != 'w90_base' or self.target_transform not in ['log', 'sqrt']:
            raise ValueError('本实验只使用90日基础输入和log/sqrt目标')
        if self.pooling not in ['last', 'last_mean'] or not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError('训练配置不合法')
        self.data_ = load_data(self.dataset_path)
        ids = validate_ids(X, len(self.data_['positions']))
        if len(ids) != len(y) or not (np.diff(ids) > 0).all():
            raise ValueError('训练id必须唯一递增并与目标等长')
        y = np.asarray(y).reshape(len(ids), -1)
        self.output_dim_, self.n_features_in_ = y.shape[1], 1
        rows = np.unique((self.data_['positions'][ids, None]+np.arange(1-MAX_WINDOW, 1)).ravel())
        self.x_scaler_ = StandardScaler().fit(self.data_['features'][rows])
        self.y_scaler_ = StandardScaler().fit(target_values(y, self.target_transform))
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.model_ = self._network()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        x = torch.as_tensor(self._inputs(ids), device=self.device)
        z = torch.as_tensor(self.y_scaler_.transform(target_values(y, self.target_transform)).astype('float32'),
                            device=self.device)
        history = []
        for epoch in range(1, int(self.fixed_epochs)+1):
            self.model_.train()
            total = 0.
            for start in range(0, len(ids), int(self.batch_size)):
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(x[start:start+self.batch_size])
                loss = torch.nn.functional.mse_loss(pred, z[start:start+self.batch_size])
                if not torch.isfinite(loss):
                    raise ValueError('损失非有限')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.)
                optimizer.step()
                total += loss.item()*len(pred)
            history.append(dict(epoch=epoch, loss=total/len(ids),
                gradient_steps=epoch*int(np.ceil(len(ids)/self.batch_size))))
        self.model_.eval()
        params = self.get_params().copy()
        for name in ['dataset_path', 'history_dir']:
            if params[name] is not None:
                params[name] = str(params[name])
        days = self.data_['origin_days']
        self.audit_ = dict(phase=self.phase, params=params, n_train=len(ids), first_id=int(ids[0]),
            last_id=int(ids[-1]), train_last_origin=str(np.datetime64(int(days[ids[-1]]), 'D')),
            train_last_h7_target=str(np.datetime64(int(days[ids[-1]]+7), 'D')),
            scaler_last_day=str(np.datetime64(int(self.data_['days'][rows[-1]]), 'D')),
            history=history, n_parameters=sum(p.numel() for p in self.model_.parameters()))
        if self.history_dir:
            folder = Path(self.history_dir)
            folder.mkdir(parents=True, exist_ok=True)
            (folder/f'{self.phase}_{uuid.uuid4().hex}.json').write_text(
                json.dumps(self.audit_, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        return self


def restore_strategy(folder, dataset_path, device='cpu'):
    folder = Path(folder)
    meta = json.loads((folder/'inference.json').read_text(encoding='utf-8'))
    if sha256(dataset_path) != meta['dataset_sha256']:
        raise ValueError('恢复数据hash不同')
    model = StrategyLSTM(**dict(meta['params'], dataset_path=str(dataset_path), device=device))
    model.data_ = load_data(dataset_path)
    if model.data_['feature_names'].tolist() != meta['feature_names']:
        raise ValueError('特征布局改变')
    with np.load(folder/'scalers.npz', allow_pickle=False) as scalers:
        for attr, mean, scale in [('x_scaler_', 'x_mean', 'x_scale'), ('y_scaler_', 'y_mean', 'y_scale')]:
            scaler = StandardScaler()
            scaler.mean_, scaler.scale_ = scalers[mean], scalers[scale]
            scaler.n_features_in_ = len(scaler.mean_)
            setattr(model, attr, scaler)
    model.output_dim_, model.n_features_in_, model.audit_ = meta['output_dim'], 1, meta['audit']
    model.model_ = model._network()
    model.model_.load_state_dict(torch.load(folder/'weights.pt', map_location=device, weights_only=True))
    model.model_.eval()
    return model
