"""Fixed target-aligned MI comparison; reuses training-only annual runner."""
import argparse
import json

from config import ROOT
from research_h1 import run_stage


def run(seed):
    if seed not in (42, 13, 2026):
        raise ValueError('Only the three preregistered seeds are allowed')
    parent = ROOT / 'output/teacher_selection_20261109'
    source = parent / f'ann_depth_seed{seed}'
    protocol = json.loads((source / 'protocol.json').read_text(encoding='utf-8'))
    baseline = next(c for c in protocol['candidates'] if c['id'] == 'ANN_first_one')
    assert baseline == dict(id='ANN_first_one', family='ANN', mode='mi', outputs=7,
                           transform='sqrt', residual=True, top_k=12,
                           params=dict(hidden=32, seq_len=1, iterations=350, alpha=1., mi_policy='first'))
    fits = json.loads((source / 'fits.json').read_text(encoding='utf-8'))
    baseline_fits = [f for f in fits if f['candidate'] == baseline['id']]
    assert len(baseline_fits) == 3 and all('Discharge_lag0' in f['selected_names'] for f in baseline_fits)
    candidate = dict(baseline, id='ANN_first_target_mi', mode='target_mi')
    return run_stage('features', parent / f'ann_target_mi_seed{seed}', seed=seed,
                     families=('ANN',), candidates=[candidate])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[42, 13, 2026], default=42)
    print(run(**vars(parser.parse_args())).to_string(index=False))
