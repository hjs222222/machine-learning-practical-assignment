"""冻结90日LSTM的记忆敏感性：只改模型输入，不训练、不用误差选择特征。"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from runoff_training.controlled_data import load_data, outer_split, sha256
from runoff_training.controlled_model import restore
from runoff_training.controlled_train import write_json, environment

ROOT = Path(__file__).resolve().parents[1]
PERTURBATIONS = ['old60_zero', 'old60_reverse', 'recent7_zero']


def perturb(x, kind):
    """在标准化输入上固定扰动；零表示训练历史均值，不表示原始观测为零。"""
    if x.ndim != 3 or x.shape[1] != 90:
        raise ValueError('诊断只接受90日窗口')
    changed = x.copy()
    if kind == 'old60_zero':
        changed[:, :60] = 0
    elif kind == 'old60_reverse':
        changed[:, :60] = changed[:, :60][:, ::-1].copy()
    elif kind == 'recent7_zero':
        changed[:, -7:] = 0
    else:
        raise ValueError(kind)
    return changed


def standardized_predictions(model, x):
    model.model_.eval()
    with torch.no_grad():
        return np.concatenate([model.model_(torch.as_tensor(x[i:i+128], device=model.device)).cpu().numpy()
                               for i in range(0, len(x), 128)])


def gradient_mass(model, x):
    """各日期块绝对梯度占比；在标准化输入/输出尺度，禁用cuDNN以支持eval反传。"""
    total = np.zeros(90)
    model.model_.eval()
    with torch.backends.cudnn.flags(enabled=False):
        for start in range(0, len(x), 64):
            inputs = torch.tensor(x[start:start+64], device=model.device, requires_grad=True)
            output = model.model_(inputs)
            for j in range(output.shape[1]):
                grad, = torch.autograd.grad(output[:, j].sum(), inputs,
                                           retain_graph=j+1 < output.shape[1])
                total += grad.detach().abs().sum(dim=(0, 2)).cpu().numpy()
    denominator = float(total.sum())
    if denominator <= 0 or not np.isfinite(total).all():
        raise ValueError('梯度无效，不能解释为记忆证据')
    return dict(old60_gradient_share=float(total[:60].sum()/denominator),
                recent23_gradient_share=float(total[60:83].sum()/denominator),
                recent7_gradient_share=float(total[83:].sum()/denominator))


def diagnose(run, out):
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed' or manifest['protocol']['smoke']:
        raise ValueError('只接受已完成正式受控run')
    for rel, expected in manifest['frozen_files'].items():
        if sha256(run/rel) != expected:
            raise ValueError(f'冻结工件已改变：{rel}')
    out.mkdir(parents=True, exist_ok=False)
    data = load_data(run/'dataset.npz')
    rows, gradients = [], []
    for year in manifest['protocol']['years']:
        _, valid = outer_split(data, year)
        for strategy, h in manifest['protocol']['tasks']:
            for profile in ['w90_base', 'w90_state']:
                for seed in manifest['protocol']['seeds']:
                    tag = f'y{year}_{strategy}_h{h}_{profile}_log_s{seed}'
                    model = restore(run/'models'/tag, run/'dataset.npz', manifest['protocol']['device'])
                    x = model._inputs(valid)
                    base = standardized_predictions(model, x)
                    identity = dict(year=year, strategy=strategy, task_h=h, profile=profile, seed=seed)
                    gradients.append(dict(**identity, **gradient_mass(model, x)))
                    for kind in PERTURBATIONS:
                        changed = standardized_predictions(model, perturb(x, kind))
                        delta = np.abs(changed-base)
                        # exp逆变换后的相对变化无单位，不读取真实目标来给模型排优劣。
                        relative = np.abs(np.expm1((changed-base)*model.y_scaler_.scale_))
                        rows.append(dict(**identity, perturbation=kind,
                            standardized_abs_mean=float(delta.mean()), standardized_abs_max=float(delta.max()),
                            relative_prediction_change_mean=float(relative.mean()),
                            relative_prediction_change_p95=float(np.quantile(relative, .95))))
    pd.DataFrame(rows).to_csv(out/'perturbations.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(gradients).to_csv(out/'gradients.csv', index=False, encoding='utf-8-sig')
    text = ['# 90日模型历史记忆诊断', '', '日期：2026-11-11。内部训练诊断，不是新性能评价。', '',
        '0次fit；对36个冻结模型的2002/2003共同外层输入作固定扰动和梯度检查。',
        'old60_zero：最早60日标准化输入置零；old60_reverse：只反转最早60日；recent7_zero：最后7日置零。',
        '置零表示该模型训练历史均值，不是原始观测为零。扰动可能离开真实输入分布，不能直接作因果重要性解释。',
        '状态通道包含递推历史，因此保持最近30日状态不变不等于移除其长期信息。',
        '梯度在标准化输入与输出上计算绝对值占比，不等于物理变量重要性，也不能据此排除其他结构。', '',
        '[扰动结果](perturbations.csv)、[梯度结果](gradients.csv)、[来源及hash](manifest.json)。', '']
    (out/'README.md').write_text('\n'.join(text), encoding='utf-8')
    write_json(out/'manifest.json', dict(status='completed', fits=0, models=len(gradients),
        run_id=run.name, source_run_manifest_sha256=sha256(run/'manifest.json'),
        source_sha256=sha256(__file__), environment=environment(manifest['protocol']['device']),
        files={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
    frame = pd.DataFrame(rows)
    print(frame.groupby(['profile', 'perturbation'])[
        ['standardized_abs_mean', 'relative_prediction_change_mean']].mean().to_string())
    print(pd.DataFrame(gradients).groupby('profile')[
        ['old60_gradient_share', 'recent23_gradient_share', 'recent7_gradient_share']].mean().to_string())
    print('完成，0次fit：', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='formal_20261111_controlled_v1')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with threadpool_limits(limits=4):
        diagnose(ROOT/'output/acceptance3_runs'/args.run_id, args.out)


if __name__ == '__main__':
    main()
