"""原生单/多输出模型；每次 fit 内重新做选择、缩放和训练。"""
import numpy as np
import warnings
import time
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning
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
    def __init__(self, inputs, hidden, outputs, readout='last', linear_skip=False, bidirectional=False):
        super().__init__()
        if not isinstance(readout,str) or readout not in ('last','last_mean'):
            raise ValueError('readout仅支持last/last_mean')
        if not isinstance(linear_skip,(bool,np.bool_)):
            raise ValueError('linear_skip须为bool')
        if (not isinstance(bidirectional,(bool,np.bool_)) or
                (bidirectional and (readout != 'last' or linear_skip))):
            raise ValueError('bidirectional须为bool，True仅支持last且无linear_skip')
        self.readout = readout
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(inputs, hidden, batch_first=True, bidirectional=self.bidirectional)
        self.head = nn.Linear(hidden*(2 if readout == 'last_mean' or self.bidirectional else 1), outputs)
        if linear_skip:
            # Zero Parameters preserve the old initialization and RNG state.
            self.skip_weight = nn.Parameter(torch.zeros(outputs,inputs))
            self.skip_bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, x):
        sequence, (hidden, _) = self.lstm(x)
        if getattr(self,'bidirectional',False):
            representation = torch.cat((hidden[-2],hidden[-1]),dim=1)
        else:
            last = sequence[:, -1]
            representation = torch.cat((last,sequence.mean(dim=1)),dim=1) if getattr(self,'readout','last') == 'last_mean' else last
        prediction = self.head(representation)
        weight = getattr(self,'skip_weight',None)
        return prediction if weight is None else prediction+nn.functional.linear(x[:,-1],weight,self.skip_bias)


class DenseNet(nn.Module):
    def __init__(self, inputs, hidden, outputs, activation, history, seed, linear_skip=False):
        super().__init__()
        self.history = history
        self.n_features_in_ = inputs
        self.hidden = nn.Linear(inputs,hidden)
        self.head = nn.Linear(hidden,outputs)
        self.activation = {'relu':nn.ReLU,'tanh':nn.Tanh,'logistic':nn.Sigmoid}[activation]()
        # Match sklearn's untrained Glorot initialization, never reuse trained weights.
        rng = np.random.RandomState(seed)
        with torch.no_grad():
            for layer in (self.hidden,self.head):
                bound = np.sqrt((2. if activation == 'logistic' else 6.)/(layer.in_features+layer.out_features))
                weight = rng.uniform(-bound,bound,(layer.in_features,layer.out_features)).astype('float32')
                bias = rng.uniform(-bound,bound,layer.out_features).astype('float32')
                layer.weight.copy_(torch.from_numpy(weight.T))
                layer.bias.copy_(torch.from_numpy(bias))
        if linear_skip:
            self.skip = nn.Linear(inputs,outputs)
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def forward(self,x):
        x = x.flatten(1) if self.history else x[:,-1]
        prediction = self.head(self.activation(self.hidden(x)))
        skip = getattr(self,'skip',None)
        return prediction if skip is None else prediction+skip(x)


class RunoffRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, family='RF', transform='raw', top_k=24, hidden=32, epochs=40,
                 trees=120, leaf=2, iterations=300, alpha=0.1, seed=42, seq_len=14, residual=False,
                 forced_indices=(), mi_target='raw', max_features=1.0, peak_weight=1.0,
                 ann_solver='adam', ann_learning_rate=0.003, ann_chrono_stop=False,
                 ann_activation='relu', lstm_raw_loss=0.0, lstm_learning_rate=0.003,
                 lstm_chrono_stop=False, ann_history=False, first_day_weight=1.0,
                 x_scaling='standard', input_transform='raw',
                 ann_backend='sklearn', ann_raw_loss=0.0, ann_batch_size=64, ann_linear_skip=False,
                 ann_grad_clip=1.0, ann_lr_schedule='constant', lstm_readout='last',
                 ann_refinement='none', lstm_linear_skip=False, lstm_grad_clip=1.0, lstm_bidirectional=False,
                 rf_bootstrap=True, rf_monotonic_flow=False, mi_policy='mean'):
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
        self.ann_backend = ann_backend
        self.ann_raw_loss = ann_raw_loss
        self.ann_batch_size = ann_batch_size
        self.ann_linear_skip = ann_linear_skip
        self.ann_grad_clip = ann_grad_clip
        self.ann_lr_schedule = ann_lr_schedule
        self.lstm_readout = lstm_readout
        self.ann_refinement = ann_refinement
        self.lstm_linear_skip = lstm_linear_skip
        self.lstm_grad_clip = lstm_grad_clip
        self.lstm_bidirectional = lstm_bidirectional
        self.rf_bootstrap = rf_bootstrap
        self.rf_monotonic_flow = rf_monotonic_flow
        self.mi_policy = mi_policy

    def target(self, y):
        if self.transform == 'log':
            return np.log1p(y)
        if self.transform == 'sqrt':
            return np.sqrt(y)
        if self.transform in ('power025','power075'):
            return np.power(y,.25 if self.transform == 'power025' else .75)
        if self.transform != 'raw':
            raise ValueError('未知目标变换')
        return y

    def _fit_preprocessing(self, X, y):
        self.outputs_ = y.shape[1]
        if (not np.isfinite(self.peak_weight) or self.peak_weight < 1 or
                (self.peak_weight != 1 and (self.outputs_ != 1 or not
                 (self.family == 'RF' or (self.family == 'ANN' and self.ann_backend == 'torch'))))):
            raise ValueError('peak_weight 必须有限且>=1；非1仅适用于单输出RF或torch ANN')
        self.peak_threshold_ = None
        self.peak_weight_normalizer_ = None
        if self.family == 'ANN' and self.peak_weight != 1:
            self.peak_threshold_ = float(np.quantile(y[:,0],.95))
            self.peak_weight_normalizer_ = float(np.where(y[:,0]>self.peak_threshold_,self.peak_weight,1.).mean())
        target = self.target(y)
        if self.residual:
            target = target - self.target(X[:,-1,:1])
        # Each policy is fitted on this training fold only.
        if self.mi_policy not in ('mean', 'intersection', 'union', 'first'):
            raise ValueError('mi_policy仅支持mean/intersection/union/first')
        if self.mi_target not in ('raw', 'transformed', 'model_target'):
            raise ValueError('未知 MI 目标')
        if any(i < 0 or i >= X.shape[-1] for i in self.forced_indices):
            raise ValueError('强制特征索引越界')
        if self.top_k is None:
            self.mi_ = None
            self.selected_ = np.arange(X.shape[-1])
        else:
            mi_y = y if self.mi_target == 'raw' else self.target(y) if self.mi_target == 'transformed' else target
            scores = np.array([mutual_info_regression(X[:,-1], mi_y[:,h], random_state=self.seed)
                               for h in range(y.shape[1])])
            self.mi_ = scores.mean(axis=0)
            k = min(self.top_k, X.shape[-1])
            selected = np.argsort(self.mi_, kind='stable')[-k:]
            if self.mi_policy != 'mean':
                sets = [set(np.argsort(score, kind='stable')[-k:]) for score in scores]
                selected = sorted(sets[0] if self.mi_policy == 'first' else
                                  set.intersection(*sets) if self.mi_policy == 'intersection' else
                                  set.union(*sets))
            self.selected_ = np.union1d(selected, self.forced_indices).astype(int)
            if not len(self.selected_):
                raise ValueError('特征交集为空；禁止静默回退')
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
        if isinstance(self.hidden,tuple) and (self.family != 'ANN' or self.ann_backend != 'sklearn' or
                not self.hidden or any(isinstance(h,(bool,np.bool_)) or not isinstance(h,(int,np.integer)) or h < 1
                                       for h in self.hidden)):
            raise ValueError('hidden tuple须为正整数层宽，仅支持sklearn ANN')
        if (not isinstance(self.rf_bootstrap,(bool,np.bool_)) or
                (not self.rf_bootstrap and self.family != 'RF')):
            raise ValueError('rf_bootstrap须为bool，False仅适用于RF')
        X = np.asarray(X, dtype=np.float32)[:, -self.seq_len:]
        y = np.asarray(y).reshape(len(X), -1)
        if (not isinstance(self.rf_monotonic_flow,(bool,np.bool_)) or
                (self.rf_monotonic_flow and (self.family != 'RF' or y.shape[1] != 1))):
            raise ValueError('rf_monotonic_flow须为bool，True仅支持单输出RF')
        if not np.isfinite(X).all() or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('模型输入/目标无效')
        if self.x_scaling not in ('standard','minmax'):
            raise ValueError('x_scaling仅支持standard/minmax')
        if self.input_transform not in ('raw','signed_log'):
            raise ValueError('input_transform仅支持raw/signed_log')
        if (isinstance(self.ann_batch_size,(bool,np.bool_)) or
                not isinstance(self.ann_batch_size,(int,np.integer)) or self.ann_batch_size < 1 or
                (self.family != 'ANN' and self.ann_batch_size != 64)):
            raise ValueError('ann_batch_size须为非bool正整数；非64仅适用于ANN')
        if (not isinstance(self.ann_linear_skip,bool) or
                (self.ann_linear_skip and (self.family != 'ANN' or self.ann_backend != 'torch'))):
            raise ValueError('ann_linear_skip须为bool，True仅适用于torch ANN')
        if (isinstance(self.ann_grad_clip,(bool,np.bool_)) or
                not isinstance(self.ann_grad_clip,(int,float,np.integer,np.floating)) or
                not np.isfinite(self.ann_grad_clip) or self.ann_grad_clip <= 0 or
                (self.ann_grad_clip != 1 and (self.family != 'ANN' or self.ann_backend != 'torch'))):
            raise ValueError('ann_grad_clip须为有限非bool正实数，非1仅适用于torch ANN')
        if (isinstance(self.lstm_grad_clip,(bool,np.bool_)) or
                not isinstance(self.lstm_grad_clip,(int,float,np.integer,np.floating)) or
                not np.isfinite(self.lstm_grad_clip) or self.lstm_grad_clip <= 0 or
                (self.lstm_grad_clip != 1 and self.family != 'LSTM')):
            raise ValueError('lstm_grad_clip须为有限非bool正实数，非1仅适用于LSTM')
        if (not isinstance(self.ann_lr_schedule,str) or self.ann_lr_schedule not in ('constant','half_linear') or
                (self.ann_lr_schedule != 'constant' and (self.family != 'ANN' or self.ann_backend != 'torch'))):
            raise ValueError('ann_lr_schedule仅支持constant/half_linear；非constant仅适用于torch ANN')
        if (not isinstance(self.lstm_readout,str) or self.lstm_readout not in ('last','last_mean') or
                (self.lstm_readout != 'last' and self.family != 'LSTM')):
            raise ValueError('lstm_readout仅支持last/last_mean，非last仅适用于LSTM')
        if (not isinstance(self.lstm_linear_skip,(bool,np.bool_)) or
                (self.lstm_linear_skip and self.family != 'LSTM')):
            raise ValueError('lstm_linear_skip须为bool，True仅适用于LSTM')
        if (not isinstance(self.lstm_bidirectional,(bool,np.bool_)) or
                (self.lstm_bidirectional and
                 (self.family != 'LSTM' or self.lstm_readout != 'last' or self.lstm_linear_skip))):
            raise ValueError('lstm_bidirectional须为bool，True仅适用于last且无linear_skip的LSTM')
        if (not isinstance(self.ann_refinement,str) or self.ann_refinement not in ('none','adam_full45','lbfgs20') or
                (self.ann_refinement != 'none' and
                 (self.family != 'ANN' or self.ann_backend != 'torch' or self.ann_lr_schedule != 'constant'))):
            raise ValueError('ann_refinement仅支持none/adam_full45/lbfgs20；非none仅适用于constant的torch ANN')
        if (self.ann_backend not in ('sklearn','torch') or
                (self.ann_backend == 'torch' and (self.family != 'ANN' or self.ann_solver != 'adam' or self.ann_chrono_stop))):
            raise ValueError('ann_backend torch仅支持无ann_chrono_stop的Adam ANN')
        if (not np.isfinite(self.ann_raw_loss) or not 0 <= self.ann_raw_loss <= 1 or
                (self.ann_raw_loss != 0 and (self.family != 'ANN' or self.ann_backend != 'torch'))):
            raise ValueError('ann_raw_loss须为0至1，非零仅支持torch ANN')
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
            if self.rf_monotonic_flow and 0 not in self.selected_:
                raise ValueError('rf_monotonic_flow需要选中原始Qlag0特征')
            constraint = [int(i == 0) for i in self.selected_] if self.rf_monotonic_flow else None
            self.estimator_ = RandomForestRegressor(n_estimators=self.trees, min_samples_leaf=self.leaf,
                max_features=self.max_features, random_state=self.seed, n_jobs=4,bootstrap=bool(self.rf_bootstrap),
                monotonic_cst=constraint)
            weights = None
            if self.peak_weight != 1:
                self.peak_threshold_ = float(np.quantile(y[:,0],.95))
                weights = np.where(y[:,0] > self.peak_threshold_,self.peak_weight,1.)
            self.estimator_.fit(x[:,-1], z.ravel() if self.outputs_ == 1 else z,sample_weight=weights)
        elif self.family == 'ANN' and self.ann_backend == 'sklearn':
            self.estimator_ = self._ann_estimator(len(x))
            self.estimator_.fit(self._ann_inputs(x), z.ravel() if self.outputs_ == 1 else z)
        elif self.family in ('ANN','LSTM'):
            self._fit_torch(x,z,X[:,-1,:1],y)
        else:
            raise ValueError('未知模型家族')
        return self

    def _ann_inputs(self, x):
        return x.reshape(len(x),-1) if self.ann_history else x[:,-1]

    def _ann_estimator(self, samples):
        return MLPRegressor(hidden_layer_sizes=self.hidden if isinstance(self.hidden,tuple) else (self.hidden,),max_iter=self.iterations,
            early_stopping=False,shuffle=False,alpha=self.alpha,random_state=self.seed,
            solver=self.ann_solver,learning_rate_init=self.ann_learning_rate,
            activation=self.ann_activation,
            batch_size=min(self.ann_batch_size,samples),tol=1e-5)

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

    def _torch_objective(self, pred, target, current, truth):
        transformed_loss = nn.functional.mse_loss(pred,target)
        weights = None
        if self.family == 'ANN' and self.peak_weight != 1:
            weights = torch.where(truth>self.peak_threshold_,pred.new_tensor(self.peak_weight),pred.new_tensor(1.))/self.peak_weight_normalizer_
            transformed_loss = (weights*(pred-target).square()).mean()
        mix = self.ann_raw_loss if self.family == 'ANN' else self.lstm_raw_loss
        if mix == 0:
            return transformed_loss
        flow = self._torch_decode(pred,current)
        raw_loss = (((flow-truth)/self.raw_loss_scale_)*pred.new_tensor(getattr(self,'output_scale_',1.))).square().mean()
        if weights is not None:
            raw_loss = (weights*(((flow-truth)/self.raw_loss_scale_)*pred.new_tensor(getattr(self,'output_scale_',1.))).square()).mean()
        return (1-mix)*transformed_loss+mix*raw_loss

    def _torch_decode(self, pred, current):
        unweighted = pred/pred.new_tensor(getattr(self,'output_scale_',1.))
        flow = unweighted*pred.new_tensor(self.y_scaler_.scale_)+pred.new_tensor(self.y_scaler_.mean_)
        if self.residual:
            baseline = torch.log1p(current) if self.transform == 'log' else torch.sqrt(current) if self.transform == 'sqrt' else current
            if self.transform in ('power025','power075'):
                baseline = current.pow(.25 if self.transform == 'power025' else .75)
            flow = flow+baseline
        if self.transform == 'log':
            flow = torch.expm1(flow)
        elif self.transform == 'sqrt':
            flow = flow.clamp(min=0).square()
        elif self.transform in ('power025','power075'):
            flow = flow.clamp(min=0).pow(4. if self.transform == 'power025' else 4./3)
        return flow.clamp(min=0)

    def _fit_lstm_chronological(self, X, y):
        start = len(X)-int(np.ceil(len(X)*.2))
        count = start-7
        probe = clone(self).set_params(lstm_chrono_stop=False)
        px,pz = probe._fit_preprocessing(X[:count],y[:count])
        probe._fit_torch(px,pz,X[:count,-1,:1],y[:count],validation=(X[start:],y[start:]))
        info = dict(probe.lstm_training_,probe_train_count=count,validation_start=start,
                    validation_count=len(X)-start,gap=7,probe_selected_indices=probe.selected_.tolist(),
                    probe_scaler_samples=int(probe.x_scaler_.n_samples_seen_))
        full = clone(self).set_params(lstm_chrono_stop=False,epochs=info['best_epoch'])
        full.fit(X,y)
        self.__dict__.update({key:value for key,value in vars(full).items() if key.endswith('_')})
        self.lstm_training_ = info
        return self

    def _ann_full_objective(self, net, x, z, current, truth):
        batches = (len(x)+self.ann_batch_size-1)//self.ann_batch_size
        penalty = sum(p.square().sum() for name,p in net.named_parameters() if name.endswith('weight'))
        return .5*self._torch_objective(net(x),z,current,truth)+self.alpha*batches*penalty/(2*len(x))

    def _refine_ann(self, net, optimizer, x, z, current, truth):
        started = time.perf_counter()
        snapshot = {name:value.detach().clone() for name,value in net.state_dict().items()}
        calls, objective_calls = 0,0
        history,norms = [],[]
        def objective():
            nonlocal objective_calls
            objective_calls += 1
            value = self._ann_full_objective(net,x,z,current,truth)
            if not torch.isfinite(value):
                raise ValueError('ANN精修目标出现非有限值')
            return value
        with torch.no_grad():
            initial = float(objective().item())
        refiner = (optimizer if self.ann_refinement == 'adam_full45' else
                   torch.optim.LBFGS(net.parameters(),lr=1.,max_iter=20,max_eval=20,line_search_fn='strong_wolfe'))
        def closure():
            nonlocal calls
            if calls >= 45:
                raise ValueError('ANN精修超出45次完整梯度预算')
            calls += 1
            refiner.zero_grad()
            value = objective()
            value.backward()
            gradients = [p.grad for p in net.parameters() if p.grad is not None]
            if any(not torch.isfinite(g).all() for g in gradients):
                raise ValueError('ANN精修梯度出现非有限值')
            norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]))
            if not torch.isfinite(norm):
                raise ValueError('ANN精修梯度范数出现非有限值')
            norms.append(float(norm.item()))
            history.append(float(value.detach().item()))
            return value
        if self.ann_refinement == 'adam_full45':
            for _ in range(45):
                closure()
                refiner.step()
            iterations = 45
        else:
            refiner.step(closure)
            state = refiner.state[next(iter(net.parameters()))]
            if state['func_evals'] != calls:
                raise ValueError('ANN精修闭包计数与优化器状态不一致')
            iterations = state['n_iter']
        with torch.no_grad():
            proposed = float(objective().item())
            rolled_back = proposed > initial
            if rolled_back:
                net.load_state_dict(snapshot)
                selected = float(objective().item())
            else:
                selected = proposed
        batches = (len(x)+self.ann_batch_size-1)//self.ann_batch_size
        self.ann_refinement_ = dict(mode=self.ann_refinement,full_train_rows=len(x),batch_count=batches,
            full_l2_coefficient=self.alpha*batches/(2*len(x)),first_stage_epochs=len(self.loss_history_),
            gradient_budget=45,gradient_evaluations=calls,full_objective_evaluations=objective_calls,
            optimizer_iterations=iterations,objective_history=history,gradient_norm_history=norms,
            initial_objective=initial,proposed_final_objective=proposed,selected_final_objective=selected,
            rolled_back=rolled_back,unclipped_gradients=True,elapsed_seconds=time.perf_counter()-started)

    def _fit_torch(self, x, z, current, truth, validation=None):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        is_ann = self.family == 'ANN'
        if is_ann:
            inputs = x.shape[-1]*(x.shape[1] if self.ann_history else 1)
            net = DenseNet(inputs,self.hidden,z.shape[1],self.ann_activation,self.ann_history,self.seed,
                           linear_skip=self.ann_linear_skip).to(device)
        else:
            net = SequenceNet(x.shape[-1],self.hidden,z.shape[1],readout=self.lstm_readout,
                              linear_skip=self.lstm_linear_skip,bidirectional=self.lstm_bidirectional).to(device)
        optimizer = torch.optim.Adam(net.parameters(),
            lr=self.ann_learning_rate if is_ann else self.lstm_learning_rate,
            weight_decay=0. if is_ann else self.alpha*0.001)
        tx, ty = torch.tensor(x,device=device), torch.tensor(z,device=device)
        tq = torch.tensor(current,dtype=tx.dtype,device=device)
        # Weighted peak membership uses the original labels, not float32-rounded ties.
        truth_tensor = torch.tensor(truth,dtype=torch.float64 if is_ann and self.peak_weight != 1 else tx.dtype,device=device)
        self.raw_loss_scale_ = max(float(np.std(truth)),1e-6)
        self.loss_history_ = []
        self.learning_rate_history_ = []
        batch_size = self.ann_batch_size if is_ann else 128
        max_epochs = self.iterations if is_ann else self.epochs
        half = max(1,max_epochs//2)
        best_loss,stale_epochs = np.inf,0
        if validation is not None:
            vx,vy = validation
            part = self._transform_inputs(vx[:,:,self.selected_])
            scaled = self.x_scaler_.transform(part.reshape(-1,len(self.selected_))).reshape(part.shape)
            valid_x = torch.tensor(scaled,dtype=tx.dtype,device=device)
            valid_q = torch.tensor(vx[:,-1,:1],dtype=tx.dtype,device=device)
            best_score,best_epoch,history = -np.inf,1,[]
        # epochs 由外部 purged 时序 CV 选择；不启用 sklearn 的随机验证拆分。
        for epoch in range(1,max_epochs+1):
            if is_ann and self.ann_lr_schedule == 'half_linear' and epoch > half:
                optimizer.param_groups[0]['lr'] = self.ann_learning_rate*(1-.9*(epoch-half)/(max_epochs-half))
            self.learning_rate_history_.append(float(optimizer.param_groups[0]['lr']))
            net.train()
            loss_sum = 0.0
            for start in range(0,len(x),batch_size):
                optimizer.zero_grad()
                pred = net(tx[start:start+batch_size])
                loss = self._torch_objective(pred,ty[start:start+batch_size],tq[start:start+batch_size],truth_tensor[start:start+batch_size])
                if is_ann:
                    penalty = sum(parameter.square().sum() for name,parameter in net.named_parameters() if name.endswith('weight'))
                    loss = .5*loss+self.alpha*penalty/(2*len(pred))
                if not torch.isfinite(loss):
                    raise ValueError('神经网络训练损失出现非有限值')
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),self.ann_grad_clip if is_ann else self.lstm_grad_clip)
                optimizer.step()
                loss_sum += loss.item()*len(pred)
            self.loss_history_.append(loss_sum/len(x))
            if is_ann:
                current_loss = self.loss_history_[-1]
                stale_epochs = stale_epochs+1 if current_loss > best_loss-1e-5 else 0
                best_loss = min(best_loss,current_loss)
                if stale_epochs > 10:
                    break
            if validation is not None and (epoch == 1 or epoch % 5 == 0):
                net.eval()
                with torch.no_grad():
                    prediction = self._torch_decode(net(valid_x),valid_q).cpu().numpy()
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
        if is_ann and epoch == max_epochs and stale_epochs <= 10:
            warnings.warn('Torch ANN reached its iteration budget without training-loss convergence',ConvergenceWarning)
        self.ann_refinement_ = None
        if is_ann and self.ann_refinement != 'none':
            self._refine_ann(net,optimizer,tx,ty,tq,truth_tensor)
        self.estimator_ = net.cpu().eval()
        if is_ann:
            self.estimator_.n_iter_ = len(self.loss_history_)
        self.device_used_ = device

    def predict(self, X):
        current = np.asarray(X)[:,-1,:1]
        X = np.asarray(X,dtype=np.float32)[:, -self.seq_len:, self.selected_]
        X = self._transform_inputs(X)
        x = self.x_scaler_.transform(X.reshape(-1,len(self.selected_))).reshape(X.shape).astype('float32')
        if self.family == 'LSTM' or (self.family == 'ANN' and self.ann_backend == 'torch'):
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
        elif self.transform in ('power025','power075'):
            z = np.maximum(z,0)**(4. if self.transform == 'power025' else 4./3)
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
