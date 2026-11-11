"""第1天原始尺度增量模型：严格因果趋势输入、折内缩放与固定Q(t)跳连。"""
import json
from pathlib import Path
import uuid

import joblib
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from runoff_training.controlled_data import load_data, validate_ids, sha256

TAB_NAMES = ['Q0', 'Q1', 'Q2', 'Q3', 'Q7', 'dQ1', 'dQ2', 'dQ7',
             'Qmean7', 'Qmean30', 'Qmean90', 'P0', 'Psum3', 'Psum7', 'Psum30',
             'Tmax', 'Tmin', 'Vp', 'Srad', 'Dayl', 'sin', 'cos']


def tabular(data, ids):
    ids = validate_ids(ids, len(data['positions']))
    t = data['positions'][ids]
    q = data['discharge']
    rain = np.expm1(data['features'][:, 1])
    values = [q[t-k] for k in [0, 1, 2, 3, 7]]
    values += [q[t]-q[t-k] for k in [1, 2, 7]]
    values += [q[t[:, None]+np.arange(1-k, 1)].mean(axis=1) for k in [7, 30, 90]]
    values += [rain[t]] + [rain[t[:, None]+np.arange(1-k, 1)].sum(axis=1) for k in [3, 7, 30]]
    values += [data['features'][t, k] for k in range(2, 9)]
    result = np.column_stack(values)
    if result.shape[1] != len(TAB_NAMES) or not np.isfinite(result).all():
        raise ValueError('第1天因果趋势特征不合法')
    return result


class DeltaNet(torch.nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        if family == 'LSTM':
            self.lstm = torch.nn.LSTM(9, 32, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(len(TAB_NAMES)+(32 if family == 'LSTM' else 0), 32),
            torch.nn.ReLU(), torch.nn.Linear(32, 1))
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, tab):
        if self.family == 'LSTM':
            states, _ = self.lstm(x)
            tab = torch.cat([states[:, -1], tab], dim=1)
        return self.head(tab)


class Day1Regressor(RegressorMixin, BaseEstimator):
    def __init__(self, dataset_path, family='LSTM', window=7, epochs=10,
                 alpha=10., leaf=2, trees=200, seed=42, device='cpu', history_dir=None, phase='development'):
        self.dataset_path, self.family, self.window, self.epochs = dataset_path, family, window, epochs
        self.alpha, self.leaf, self.trees, self.seed = alpha, leaf, trees, seed
        self.device, self.history_dir, self.phase = device, history_dir, phase

    def _inputs(self, ids):
        ids = validate_ids(ids, len(self.data_['positions']))
        tab = self.tab_scaler_.transform(tabular(self.data_, ids)).astype('float32')
        rows = self.data_['positions'][ids, None]+np.arange(1-self.window, 1)
        values = self.data_['features'][rows, :9]
        x = self.x_scaler_.transform(values.reshape(-1, 9)).reshape(values.shape).astype('float32')
        return x, tab

    def fit(self, X, y):
        for name in ['window', 'epochs', 'leaf', 'trees', 'seed']:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (0 if name == 'seed' else 1):
                raise ValueError(f'{name}必须为合法整数')
        if self.family not in ['Ridge', 'RF', 'ANN', 'LSTM'] or self.window not in [7, 30]:
            raise ValueError('只接受预固定模型族和窗口')
        if (self.device not in ['cpu', 'cuda'] or isinstance(self.alpha, (bool, np.bool_))
                or not np.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError('设备或alpha不合法')
        self.data_ = load_data(self.dataset_path)
        ids = validate_ids(X, len(self.data_['positions']))
        truth = np.asarray(y, dtype=float)
        if truth.ndim not in [1, 2] or truth.ndim == 2 and truth.shape[1] != 1:
            raise ValueError('第1天只接受单输出目标')
        truth = truth.reshape(-1, 1)
        if len(ids) != len(truth) or not (np.diff(ids) > 0).all() or not np.isfinite(truth).all():
            raise ValueError('训练id/目标不合法')
        # 拒绝调用方静默传入别的h、变换后目标或错位标签。
        np.testing.assert_equal(truth[:, 0], self.data_['y'][ids, 0])
        tab = tabular(self.data_, ids)
        self.tab_scaler_ = StandardScaler().fit(tab)
        delta = truth-tab[:, :1]
        self.y_scaler_ = StandardScaler().fit(delta)
        rows = np.unique((self.data_['positions'][ids, None]+np.arange(1-self.window, 1)).ravel())
        self.x_scaler_ = StandardScaler().fit(self.data_['features'][rows, :9])
        x, t = self._inputs(ids)
        z = self.y_scaler_.transform(delta).ravel()
        history = []
        if self.family == 'Ridge':
            self.model_ = Ridge(alpha=self.alpha).fit(t, z)
        elif self.family == 'RF':
            self.model_ = RandomForestRegressor(n_estimators=self.trees, max_depth=10,
                min_samples_leaf=self.leaf, random_state=self.seed, n_jobs=1).fit(t, z)
        else:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            self.model_ = DeltaNet(self.family).to(self.device)
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=.001)
            x, t = [torch.as_tensor(a, device=self.device) for a in [x, t]]
            z = torch.as_tensor(z[:, None].astype('float32'), device=self.device)
            for epoch in range(1, int(self.epochs)+1):
                self.model_.train()
                total = 0.
                for start in range(0, len(ids), 64):
                    batch = slice(start, start+64)
                    optimizer.zero_grad(set_to_none=True)
                    output = self.model_(x[batch], t[batch])
                    loss = torch.nn.functional.mse_loss(output, z[batch])
                    if not torch.isfinite(loss):
                        raise ValueError('增量训练损失非有限')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    total += loss.item()*len(output)
                history.append(dict(epoch=epoch, loss=total/len(ids), gradient_steps=epoch*int(np.ceil(len(ids)/64))))
            self.model_.eval()
        params = self.get_params().copy()
        for k in ['dataset_path', 'history_dir']:
            if params[k] is not None:
                params[k] = str(params[k])
        days = self.data_['origin_days']
        self.n_features_in_ = 1
        self.audit_ = dict(params=params, phase=self.phase, n_train=len(ids), first_id=int(ids[0]), last_id=int(ids[-1]),
            train_last_origin=str(np.datetime64(int(days[ids[-1]]), 'D')),
            train_last_h7=str(np.datetime64(int(days[ids[-1]]+7), 'D')),
            scaler_last_day=str(np.datetime64(int(self.data_['days'][rows[-1]]), 'D')),
            tab_mean=self.tab_scaler_.mean_.tolist(), tab_scale=self.tab_scaler_.scale_.tolist(),
            delta_mean=self.y_scaler_.mean_.tolist(), delta_scale=self.y_scaler_.scale_.tolist(), history=history)
        if self.history_dir:
            path = Path(self.history_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path/f'{self.phase}_{uuid.uuid4().hex}.json').write_text(
                json.dumps(self.audit_, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
        return self

    def predict(self, X):
        ids = validate_ids(X, len(self.data_['positions']))
        x, tab = self._inputs(ids)
        if self.family in ['Ridge', 'RF']:
            z = self.model_.predict(tab).reshape(-1, 1)
        else:
            self.model_.eval()
            with torch.no_grad():
                z = np.concatenate([self.model_(torch.as_tensor(x[k:k+128], device=self.device),
                    torch.as_tensor(tab[k:k+128], device=self.device)).cpu().numpy() for k in range(0, len(ids), 128)])
        result = tabular(self.data_, ids)[:, :1] + self.y_scaler_.inverse_transform(z.astype(float))
        if not np.isfinite(result).all():
            raise ValueError('增量预测非有限')
        return result

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        state = dict(tab=self.tab_scaler_, x=self.x_scaler_, y=self.y_scaler_)
        if self.family in ['Ridge', 'RF']:
            state['model'] = self.model_
        else:
            torch.save(self.model_.state_dict(), folder/'weights.pt')
        joblib.dump(state, folder/'state.joblib')
        meta = dict(params=self.audit_['params'], audit=self.audit_, dataset_sha256=sha256(self.dataset_path), tab_names=TAB_NAMES)
        (folder/'inference.json').write_text(json.dumps(meta, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def restore_day1(folder, dataset_path, device):
    folder = Path(folder)
    meta = json.loads((folder/'inference.json').read_text(encoding='utf-8'))
    if sha256(dataset_path) != meta['dataset_sha256'] or meta['tab_names'] != TAB_NAMES:
        raise ValueError('恢复数据或趋势特征布局改变')
    model = Day1Regressor(**dict(meta['params'], dataset_path=str(dataset_path), device=device))
    model.data_ = load_data(dataset_path)
    state = joblib.load(folder/'state.joblib')
    model.tab_scaler_, model.x_scaler_, model.y_scaler_ = state['tab'], state['x'], state['y']
    model.audit_, model.n_features_in_ = meta['audit'], 1
    if model.family in ['Ridge', 'RF']:
        model.model_ = state['model']
    else:
        model.model_ = DeltaNet(model.family).to(device)
        model.model_.load_state_dict(torch.load(folder/'weights.pt', map_location=device, weights_only=True))
        model.model_.eval()
    return model
