"""原生单/多输出模型；每次 fit 内重新做选择、缩放和训练。"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
                 forced_indices=(), mi_target='raw', max_features=1.0, peak_weight=1.0,
                 ann_solver='adam', ann_learning_rate=0.003, ann_chrono_stop=False,
                 ann_activation='relu', lstm_raw_loss=0.0, lstm_learning_rate=0.003,
                 lstm_chrono_stop=False, ann_history=False, first_day_weight=1.0,
                 x_scaling='standard', input_transform='raw'):
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
        self.peak_weight = peak_weight
        self.ann_solver = ann_solver
        self.ann_learning_rate = ann_learning_rate
        self.ann_chrono_stop = ann_chrono_stop
        self.ann_activation = ann_activation
        self.lstm_raw_loss = lstm_raw_loss
        self.lstm_learning_rate = lstm_learning_rate
        self.lstm_chrono_stop = lstm_chrono_stop
        self.ann_history = ann_history
        self.first_day_weight = first_day_weight
        self.x_scaling = x_scaling
        self.input_transform = input_transform

    def target(self, y):
        if self.transform == 'log':
            return np.log1p(y)
        if self.transform == 'sqrt':
            return np.sqrt(y)
        if self.transform != 'raw':
            raise ValueError('未知目标变换')
        return y

    def _fit_preprocessing(self, X, y):
        self.outputs_ = y.shape[1]
        if (not np.isfinite(self.peak_weight) or self.peak_weight < 1 or
                (self.peak_weight != 1 and (self.family != 'RF' or self.outputs_ != 1))):
            raise ValueError('peak_weight 必须有限且>=1；非1仅适用于单输出RF')
        self.peak_threshold_ = None
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
        part = self._transform_inputs(X[:,:,self.selected_])
        self.x_scaler_ = (StandardScaler() if self.x_scaling == 'standard' else MinMaxScaler()).fit(
            part.reshape(-1,len(self.selected_)))
        self.y_scaler_ = StandardScaler().fit(target)
        x = self.x_scaler_.transform(part.reshape(-1,len(self.selected_))).reshape(part.shape).astype('float32')
        z = self.y_scaler_.transform(target).astype('float32')
        self.output_scale_ = np.ones(self.outputs_,dtype=np.float32)
        self.output_scale_[0] = np.sqrt(self.first_day_weight)
        z *= self.output_scale_
        return x,z

    def _transform_inputs(self, values):
        return np.sign(values)*np.log1p(np.abs(values)) if self.input_transform == 'signed_log' else values

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)[:, -self.seq_len:]
        y = np.asarray(y).reshape(len(X), -1)
        if not np.isfinite(X).all() or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('模型输入/目标无效')
        if self.x_scaling not in ('standard','minmax'):
            raise ValueError('x_scaling仅支持standard/minmax')
        if self.input_transform not in ('raw','signed_log'):
            raise ValueError('input_transform仅支持raw/signed_log')
        if (not np.isfinite(self.first_day_weight) or self.first_day_weight < 1 or
                (self.first_day_weight != 1 and y.shape[1] != 7)):
            raise ValueError('first_day_weight须为有限且>=1，非1仅用于七输出')
        if self.ann_history and self.family != 'ANN':
            raise ValueError('ann_history仅适用于ANN')
        if self.family == 'LSTM' and (not np.isfinite(self.lstm_learning_rate) or self.lstm_learning_rate <= 0):
            raise ValueError('lstm_learning_rate必须为有限正数')
        if (not np.isfinite(self.lstm_raw_loss) or not 0 <= self.lstm_raw_loss <= 1 or
                (self.lstm_raw_loss != 0 and self.family != 'LSTM')):
            raise ValueError('lstm_raw_loss须在0至1且非零仅适用于LSTM')
        if self.family == 'ANN' and (self.ann_solver not in ('adam','lbfgs') or
                self.ann_activation not in ('relu','tanh','logistic') or
                not np.isfinite(self.ann_learning_rate) or self.ann_learning_rate <= 0):
            raise ValueError('ANN optimizer参数无效')
        self.ann_training_ = None
        self.lstm_training_ = None
        if self.lstm_chrono_stop:
            if self.family != 'LSTM' or len(X) < 60 or y.shape[1] != 1 or not 5 <= self.epochs <= 300:
                raise ValueError('lstm_chrono_stop仅支持>=60行、5至300轮单输出LSTM')
            return self._fit_lstm_chronological(X,y)
        if self.ann_chrono_stop:
            if (self.family != 'ANN' or self.ann_solver != 'adam' or y.shape[1] != 1 or
                    len(X) < 60 or not 5 <= self.iterations <= 400):
                raise ValueError('ann_chrono_stop仅支持>=60行、5至400轮的单输出Adam ANN')
            return self._fit_ann_chronological(X,y)
        x,z = self._fit_preprocessing(X,y)
        if self.family == 'RF':
            self.estimator_ = RandomForestRegressor(n_estimators=self.trees, min_samples_leaf=self.leaf,
                max_features=self.max_features, random_state=self.seed, n_jobs=4)
            weights = None
            if self.peak_weight != 1:
                self.peak_threshold_ = float(np.quantile(y[:,0],.95))
                weights = np.where(y[:,0] > self.peak_threshold_,self.peak_weight,1.)
            self.estimator_.fit(x[:,-1], z.ravel() if self.outputs_ == 1 else z,sample_weight=weights)
        elif self.family == 'ANN':
            self.estimator_ = self._ann_estimator(len(x))
            self.estimator_.fit(self._ann_inputs(x), z.ravel() if self.outputs_ == 1 else z)
        elif self.family == 'LSTM':
            self._fit_lstm(x,z,X[:,-1,:1],y)
        else:
            raise ValueError('未知模型家族')
        return self

    def _ann_inputs(self, x):
        return x.reshape(len(x),-1) if self.ann_history else x[:,-1]

    def _ann_estimator(self, samples):
        return MLPRegressor(hidden_layer_sizes=(self.hidden,),max_iter=self.iterations,
            early_stopping=False,shuffle=False,alpha=self.alpha,random_state=self.seed,
            solver=self.ann_solver,learning_rate_init=self.ann_learning_rate,
            activation=self.ann_activation,
            batch_size=min(64,samples),tol=1e-5)

    def _fit_ann_chronological(self, X, y):
        start = len(X)-int(np.ceil(len(X)*.2))
        count = start-7
        probe = clone(self).set_params(ann_chrono_stop=False)
        px,pz = probe._fit_preprocessing(X[:count],y[:count])
        probe.estimator_ = probe._ann_estimator(count)
        best_score,best_epoch,history = -np.inf,1,[]
        for epoch in range(1,self.iterations+1):
            probe.estimator_.partial_fit(probe._ann_inputs(px),pz.ravel())
            if epoch == 1 or epoch % 5 == 0:
                score = float(r2_score(y[start:],probe.predict(X[start:])))
                if not np.isfinite(score):
                    raise ValueError('ANN内部时间验证分数无效')
                history.append(dict(epoch=epoch,r2=score,training_loss=float(probe.estimator_.loss_)))
                if score > best_score+1e-4:
                    best_score,best_epoch = score,epoch
                elif epoch-best_epoch >= 30:
                    break
        # epoch已冻结后才允许全部外层训练行进入MI和缩放。
        full = clone(self).set_params(ann_chrono_stop=False)
        fx,fz = full._fit_preprocessing(X,y)
        full.estimator_ = full._ann_estimator(len(X))
        for _ in range(best_epoch):
            full.estimator_.partial_fit(full._ann_inputs(fx),fz.ravel())
        self.__dict__.update({k:v for k,v in vars(full).items() if k.endswith('_')})
        self.ann_training_ = dict(probe_train_count=count,validation_start=start,
            validation_count=len(X)-start,gap=7,best_epoch=best_epoch,stopped_epoch=epoch,
            best_inner_r2=best_score,history=history,probe_selected_indices=probe.selected_.tolist(),
            probe_scaler_samples=int(probe.x_scaler_.n_samples_seen_))
        return self

    def _lstm_objective(self, pred, target, current, truth):
        transformed_loss = nn.functional.mse_loss(pred,target)
        if self.lstm_raw_loss == 0:
            return transformed_loss
        flow = self._lstm_decode(pred,current)
        raw_loss = (((flow-truth)/self.raw_loss_scale_)*pred.new_tensor(getattr(self,'output_scale_',1.))).square().mean()
        return (1-self.lstm_raw_loss)*transformed_loss+self.lstm_raw_loss*raw_loss

    def _lstm_decode(self, pred, current):
        unweighted = pred/pred.new_tensor(getattr(self,'output_scale_',1.))
        flow = unweighted*pred.new_tensor(self.y_scaler_.scale_)+pred.new_tensor(self.y_scaler_.mean_)
        if self.residual:
            baseline = torch.log1p(current) if self.transform == 'log' else torch.sqrt(current) if self.transform == 'sqrt' else current
            flow = flow+baseline
        if self.transform == 'log':
            flow = torch.expm1(flow)
        elif self.transform == 'sqrt':
            flow = flow.clamp(min=0).square()
        return flow.clamp(min=0)

    def _fit_lstm_chronological(self, X, y):
        start = len(X)-int(np.ceil(len(X)*.2))
        count = start-7
        probe = clone(self).set_params(lstm_chrono_stop=False)
        px,pz = probe._fit_preprocessing(X[:count],y[:count])
        probe._fit_lstm(px,pz,X[:count,-1,:1],y[:count],validation=(X[start:],y[start:]))
        info = dict(probe.lstm_training_,probe_train_count=count,validation_start=start,
                    validation_count=len(X)-start,gap=7,probe_selected_indices=probe.selected_.tolist(),
                    probe_scaler_samples=int(probe.x_scaler_.n_samples_seen_))
        full = clone(self).set_params(lstm_chrono_stop=False,epochs=info['best_epoch'])
        full.fit(X,y)
        self.__dict__.update({key:value for key,value in vars(full).items() if key.endswith('_')})
        self.lstm_training_ = info
        return self

    def _fit_lstm(self, x, z, current, truth, validation=None):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = SequenceNet(x.shape[-1],self.hidden,z.shape[1]).to(device)
        optimizer = torch.optim.Adam(net.parameters(),lr=self.lstm_learning_rate,weight_decay=self.alpha*0.001)
        tx, ty = torch.tensor(x,device=device), torch.tensor(z,device=device)
        tq = torch.tensor(current,dtype=tx.dtype,device=device)
        truth_tensor = torch.tensor(truth,dtype=tx.dtype,device=device)
        self.raw_loss_scale_ = max(float(np.std(truth)),1e-6)
        self.loss_history_ = []
        if validation is not None:
            vx,vy = validation
            part = self._transform_inputs(vx[:,:,self.selected_])
            scaled = self.x_scaler_.transform(part.reshape(-1,len(self.selected_))).reshape(part.shape)
            valid_x = torch.tensor(scaled,dtype=tx.dtype,device=device)
            valid_q = torch.tensor(vx[:,-1,:1],dtype=tx.dtype,device=device)
            best_score,best_epoch,history = -np.inf,1,[]
        # epochs 由外部 purged 时序 CV 选择；不启用 sklearn 的随机验证拆分。
        for epoch in range(1,self.epochs+1):
            net.train()
            loss_sum = 0.0
            for start in range(0,len(x),128):
                optimizer.zero_grad()
                pred = net(tx[start:start+128])
                loss = self._lstm_objective(pred,ty[start:start+128],tq[start:start+128],truth_tensor[start:start+128])
                if not torch.isfinite(loss):
                    raise ValueError('LSTM训练损失出现非有限值')
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.0)
                optimizer.step()
                loss_sum += loss.item()*len(pred)
            self.loss_history_.append(loss_sum/len(x))
            if validation is not None and (epoch == 1 or epoch % 5 == 0):
                net.eval()
                with torch.no_grad():
                    prediction = self._lstm_decode(net(valid_x),valid_q).cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ValueError('LSTM内部时间验证预测非有限')
                score = float(r2_score(vy,prediction))
                if not np.isfinite(score):
                    raise ValueError('LSTM内部时间验证R2非有限')
                history.append(dict(epoch=epoch,r2=score,training_loss=self.loss_history_[-1]))
                if score > best_score+1e-4:
                    best_score,best_epoch = score,epoch
                elif epoch-best_epoch >= 30:
                    break
        if validation is not None:
            self.lstm_training_ = dict(best_epoch=best_epoch,stopped_epoch=epoch,
                                       best_inner_r2=best_score,history=history)
        self.estimator_ = net.cpu().eval()
        self.device_used_ = device

    def predict(self, X):
        current = np.asarray(X)[:,-1,:1]
        X = np.asarray(X,dtype=np.float32)[:, -self.seq_len:, self.selected_]
        X = self._transform_inputs(X)
        x = self.x_scaler_.transform(X.reshape(-1,len(self.selected_))).reshape(X.shape).astype('float32')
        if self.family == 'LSTM':
            with torch.no_grad():
                prediction = self.estimator_(torch.from_numpy(x)).numpy()
        elif self.family == 'ANN':
            prediction = self.estimator_.predict(self._ann_inputs(x))
        else:
            prediction = self.estimator_.predict(x[:,-1])
        prediction = np.asarray(prediction).reshape(len(X),-1)/getattr(self,'output_scale_',1.)
        z = self.y_scaler_.inverse_transform(prediction)
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
