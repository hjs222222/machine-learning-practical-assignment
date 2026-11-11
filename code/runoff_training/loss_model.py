"""保持log输出参数化，独立对照log/原始尺度MSE；不改冻结模型实现。"""
import json
from pathlib import Path
import uuid

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from runoff_training.controlled_data import MAX_WINDOW, load_data, sha256, validate_ids
from runoff_training.strategy_model import StrategyLSTM


def training_loss(output, target_log, target_raw, log_mean, log_scale, raw_scale, space):
    if space == 'log':
        return torch.nn.functional.mse_loss(output, target_log)
    if space != 'physical':
        raise ValueError('损失空间必须为log或physical')
    # exp和平方用float64；不裁剪或用验证目标估计归一化尺度。
    physical = torch.exp(output.double() * log_scale + log_mean)
    return torch.square((physical - target_raw) / raw_scale).mean()


class LossLSTM(StrategyLSTM):
    def __init__(self, dataset_path, profile='w90_base', fixed_epochs=10,
                 seed=42, device='cpu', target_transform='log', units=32,
                 batch_size=64, lr=.001, history_dir=None, phase='development',
                 pooling='last', residual=False, loss_space='log'):
        super().__init__(dataset_path, profile, fixed_epochs, seed, device, target_transform,
                         units, batch_size, lr, history_dir, phase, pooling, residual)
        self.loss_space = loss_space

    def fit(self, X, y):
        for name in ['fixed_epochs', 'batch_size', 'units']:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name}必须为正整数类型')
        if (self.profile != 'w90_base' or self.target_transform != 'log' or self.pooling != 'last'
                or self.loss_space not in ['log', 'physical'] or self.device not in ['cpu', 'cuda']
                or not np.isfinite(self.lr) or self.lr <= 0 or not isinstance(self.residual, bool)):
            raise ValueError('损失对照配置不合法')
        self.data_ = load_data(self.dataset_path)
        ids = validate_ids(X, len(self.data_['positions']))
        if len(ids) != len(y) or not (np.diff(ids) > 0).all():
            raise ValueError('训练id必须唯一递增并与目标等长')
        y = np.asarray(y, dtype=float).reshape(len(ids), -1)
        if y.shape[1] != 1 or not np.isfinite(y).all() or (y <= 0).any():
            raise ValueError('本轮仅接受有限正值单输出目标')
        self.output_dim_, self.n_features_in_ = 1, 1
        rows = np.unique((self.data_['positions'][ids, None] + np.arange(1-MAX_WINDOW, 1)).ravel())
        self.x_scaler_ = StandardScaler().fit(self.data_['features'][rows])
        self.y_scaler_ = StandardScaler().fit(np.log(y))
        raw_scaler = StandardScaler().fit(y)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.model_ = self._network()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        x = torch.as_tensor(self._inputs(ids), device=self.device)
        z = torch.as_tensor(self.y_scaler_.transform(np.log(y)).astype('float32'), device=self.device)
        raw = torch.as_tensor(y, dtype=torch.float64, device=self.device)
        mean = torch.as_tensor(self.y_scaler_.mean_, dtype=torch.float64, device=self.device)
        scale = torch.as_tensor(self.y_scaler_.scale_, dtype=torch.float64, device=self.device)
        raw_scale = torch.as_tensor(raw_scaler.scale_, dtype=torch.float64, device=self.device)
        history = []
        for epoch in range(1, self.fixed_epochs+1):
            self.model_.train()
            total = 0.
            for start in range(0, len(ids), self.batch_size):
                batch = slice(start, start+self.batch_size)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model_(x[batch])
                loss = training_loss(prediction, z[batch], raw[batch], mean, scale, raw_scale, self.loss_space)
                if not torch.isfinite(loss):
                    raise ValueError('训练损失非有限；拒绝静默裁剪')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                total += loss.item()*len(prediction)
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
            raw_target_mean=raw_scaler.mean_.tolist(), raw_target_scale=raw_scaler.scale_.tolist(),
            history=history, n_parameters=sum(p.numel() for p in self.model_.parameters()))
        if self.history_dir:
            folder = Path(self.history_dir)
            folder.mkdir(parents=True, exist_ok=True)
            (folder/f'{self.phase}_{uuid.uuid4().hex}.json').write_text(
                json.dumps(self.audit_, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        return self


def restore_loss(folder, dataset_path, device='cpu'):
    folder = Path(folder)
    meta = json.loads((folder/'inference.json').read_text(encoding='utf-8'))
    if sha256(dataset_path) != meta['dataset_sha256']:
        raise ValueError('恢复数据hash不同')
    model = LossLSTM(**dict(meta['params'], dataset_path=str(dataset_path), device=device))
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
