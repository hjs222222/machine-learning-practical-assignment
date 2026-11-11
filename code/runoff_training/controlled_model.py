"""受控原生LSTM：统一输入维数/初始化、明确目标逆变换和折内预处理。"""
import json
from pathlib import Path
import uuid

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from runoff_training.controlled_data import BASE, MAX_WINDOW, PROFILES, load_data, sha256, validate_ids


class ControlledNet(torch.nn.Module):
    def __init__(self, n_features, units, output_dim):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_features, units, batch_first=True)
        self.drop = torch.nn.Dropout(.1)
        self.head = torch.nn.Sequential(torch.nn.Linear(units, 32), torch.nn.ReLU(),
                                        torch.nn.Linear(32, output_dim))

    def forward(self, x):
        states, _ = self.lstm(x)
        return self.head(self.drop(states[:, -1]))


def target_values(y, transform):
    values = np.asarray(y, dtype=float).reshape(len(y), -1)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('目标必须有限且非负')
    if transform == 'log':
        if (values <= 0).any():
            raise ValueError('log目标要求正流量')
        return np.log(values)
    if transform == 'sqrt':
        return np.sqrt(values)
    if transform == 'raw':
        return values
    raise ValueError(transform)


def inverse_values(z, mean, scale, transform):
    values = np.asarray(z, dtype=float) * scale + mean
    if transform == 'log':
        result = np.exp(values)
    elif transform == 'sqrt':
        result = np.maximum(values, 0) ** 2
    elif transform == 'raw':
        result = values
    else:
        raise ValueError(transform)
    if not np.isfinite(result).all():
        raise ValueError('预测非有限，不裁剪掩盖失败')
    return result


class ControlledLSTM(RegressorMixin, BaseEstimator):
    def __init__(self, dataset_path, profile='w30_base', fixed_epochs=10,
                 seed=42, device='cpu', target_transform='log', units=32,
                 batch_size=64, lr=.001, history_dir=None, phase='development'):
        self.dataset_path = dataset_path
        self.profile = profile
        self.fixed_epochs = fixed_epochs
        self.seed = seed
        self.device = device
        self.target_transform = target_transform
        self.units = units
        self.batch_size = batch_size
        self.lr = lr
        self.history_dir = history_dir
        self.phase = phase

    def _inputs(self, ids):
        data = self.data_
        ids = validate_ids(ids, len(data['positions']))
        window, states = PROFILES[self.profile]
        rows = data['positions'][ids, None] + np.arange(1 - window, 1)
        x = data['features'][rows]
        x = self.x_scaler_.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        if not states:
            x[:, :, len(BASE):] = 0.
        return x.astype('float32')

    def fit(self, X, y):
        for name in ['fixed_epochs', 'batch_size', 'units']:
            value = getattr(self, name)
            if not np.isfinite(value) or int(value) != value or value < 1:
                raise ValueError(f'{name}必须为正整数')
        if self.profile not in PROFILES or not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError('配置不合法')
        self.data_ = load_data(self.dataset_path)
        ids = validate_ids(X, len(self.data_['positions']))
        if len(ids) != len(y) or not (np.diff(ids) > 0).all():
            raise ValueError('拟合id必须唯一递增并与目标等长')
        y = np.asarray(y).reshape(len(ids), -1)
        self.output_dim_ = y.shape[1]
        self.n_features_in_ = 1
        # 四种方案用相同训练id覆盖的90天历史并集拟合缩放器，避免窗口比较混入缩放差异。
        rows = np.unique((self.data_['positions'][ids, None] + np.arange(1 - MAX_WINDOW, 1)).ravel())
        self.scaler_rows_ = rows
        self.x_scaler_ = StandardScaler().fit(self.data_['features'][rows])
        self.y_scaler_ = StandardScaler().fit(target_values(y, self.target_transform))
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.model_ = ControlledNet(len(self.data_['feature_names']), self.units, self.output_dim_).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        x = torch.as_tensor(self._inputs(ids), device=self.device)
        z = torch.as_tensor(self.y_scaler_.transform(target_values(y, self.target_transform)).astype('float32'),
                            device=self.device)
        history = []
        for epoch in range(1, int(self.fixed_epochs) + 1):
            self.model_.train()
            total = 0.
            for start in range(0, len(ids), self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                predicted = self.model_(x[start:start + self.batch_size])
                loss = torch.nn.functional.mse_loss(predicted, z[start:start + self.batch_size])
                if not torch.isfinite(loss):
                    raise ValueError('训练损失非有限')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.)
                optimizer.step()
                total += loss.item() * len(predicted)
            history.append(dict(epoch=epoch, loss=total / len(ids),
                gradient_steps=epoch * int(np.ceil(len(ids) / self.batch_size))))
        self.model_.eval()
        days = self.data_['origin_days']
        params = self.get_params().copy()
        for name in ['dataset_path', 'history_dir']:
            if params[name] is not None:
                params[name] = str(params[name])
        self.audit_ = dict(phase=self.phase, params=params, n_train=len(ids),
            first_id=int(ids[0]), last_id=int(ids[-1]),
            train_last_origin=str(np.datetime64(int(days[ids[-1]]), 'D')),
            train_last_h7_target=str(np.datetime64(int(days[ids[-1]] + 7), 'D')),
            scaler_last_day=str(np.datetime64(int(self.data_['days'][rows[-1]]), 'D')),
            history=history, n_parameters=sum(p.numel() for p in self.model_.parameters()))
        if self.history_dir:
            folder = Path(self.history_dir)
            folder.mkdir(parents=True, exist_ok=True)
            self.audit_path_ = folder / f'{self.phase}_{uuid.uuid4().hex}.json'
            self.audit_path_.write_text(json.dumps(self.audit_, ensure_ascii=False, indent=2), encoding='utf-8')
        return self

    def predict(self, X):
        x = self._inputs(X)
        self.model_.eval()
        with torch.no_grad():
            z = np.concatenate([self.model_(torch.as_tensor(x[i:i + 128], device=self.device)).cpu().numpy()
                                for i in range(0, len(x), 128)])
        return inverse_values(z, self.y_scaler_.mean_, self.y_scaler_.scale_, self.target_transform)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        torch.save(self.model_.state_dict(), folder / 'weights.pt')
        np.savez_compressed(folder / 'scalers.npz', x_mean=self.x_scaler_.mean_, x_scale=self.x_scaler_.scale_,
            y_mean=self.y_scaler_.mean_, y_scale=self.y_scaler_.scale_)
        params = self.get_params().copy()
        for name in ['dataset_path', 'history_dir']:
            if params[name] is not None:
                params[name] = str(params[name])
        meta = dict(params=params, feature_names=self.data_['feature_names'].tolist(),
            dataset_sha256=sha256(self.dataset_path), output_dim=self.output_dim_, audit=self.audit_)
        (folder / 'inference.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


def restore(folder, dataset_path, device='cpu'):
    folder = Path(folder)
    meta = json.loads((folder / 'inference.json').read_text(encoding='utf-8'))
    if meta['dataset_sha256'] != sha256(dataset_path):
        raise ValueError('恢复数据hash不同，拒绝静默更换输入布局')
    params = dict(meta['params'], dataset_path=str(dataset_path), device=device)
    model = ControlledLSTM(**params)
    model.data_ = load_data(dataset_path)
    if model.data_['feature_names'].tolist() != meta['feature_names']:
        raise ValueError('特征布局不同')
    with np.load(folder / 'scalers.npz', allow_pickle=False) as scalers:
        model.x_scaler_, model.y_scaler_ = StandardScaler(), StandardScaler()
        for name, mean, scale in [('x_scaler_', 'x_mean', 'x_scale'), ('y_scaler_', 'y_mean', 'y_scale')]:
            scaler = getattr(model, name)
            scaler.mean_, scaler.scale_ = scalers[mean], scalers[scale]
            scaler.n_features_in_ = len(scaler.mean_)
    model.output_dim_ = meta['output_dim']
    model.audit_ = meta['audit']
    model.n_features_in_ = 1
    model.model_ = ControlledNet(len(meta['feature_names']), model.units, model.output_dim_).to(device)
    model.model_.load_state_dict(torch.load(folder / 'weights.pt', map_location=device, weights_only=True))
    model.model_.eval()
    return model


def physical_score(estimator, X, y):
    truth = np.asarray(y).reshape(len(y), -1)
    return -float(np.sqrt(np.square(estimator.predict(X) - truth).mean(axis=0)).mean())
