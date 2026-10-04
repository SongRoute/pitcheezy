"""Torch-free registry of the DRAFT weight-averaging family (DRAFT-CUDA-EMA-20261004).

Question: does an exponential moving average (EMA) of the weights, used for early
stopping and as the saved model, lower the calibrated direct NLL of the flatten
head compared with the identical training that keeps the raw last-step weights?
Prior basis: Izmailov et al. 2018 (UAI, arXiv:1803.05407) and Morales-Brotons,
Vogels, Hendrikx 2024 (TMLR, arXiv:2411.18704) report better generalization and
calibration from weight averaging.  Not tried in any earlier family (P0-P32).
Exposure: dev / mlb_dev are repeated DEV (P24-P32); descriptive only, no verdict,
ranking, promotion or 2026.

2 x 2 factorial, 4 arms, one immutable export, fixed budget (30 epochs, patience 5,
lr 5e-4, batch 1024, AdamW weight decay .01, uniform loss), external calibration,
no workload:

                 raw weights (fresh control)                EMA decay .999
  dropout 0      control = MatrixModel flatten_mlp 128      ema
  dropout .1     dropout = frontier ARMS['dropout'] exact   dropout_ema

EMA decay is fixed BEFORE any run: the P32 control fit used 13,464 updates in 11
epochs (about 1,224 per epoch), so .999 averages over about 0.8 epoch.  Warm-up
decay_t = min(decay, (1 + t) / (10 + t)) (the num_updates rule of TensorFlow's
ExponentialMovingAverage) so the random initialization is forgotten quickly.
No decay sweep.  EMA has no parameters: each arm has its base parameter count.
The EMA arms consume no extra randomness, so for one seed their raw trajectory
is the base arm's trajectory; only the evaluated / stopped / saved weights differ.
"""
from __future__ import annotations

import json

EMA_DECAY = .999
DROPOUT = .1
WEIGHT_DECAY = .01
WIDTH = 128
SEEDS = [19, 20, 21, 22, 23]  # disjoint from 0-2 (P24/P25/P27/P29), 3-7 (P30), 8-12 (reserved P30 repeat),
FITS = 20                     # 13-15 (P31) and 16-18 (P32); new model RNG seeds, not new evaluation data

# name -> exact config entry without name / priority / ema / workload
ARMS = {
    'ema': {'family': 'weight_averaging', 'model': 'ema_flatten_mlp', 'base_model': 'flatten_mlp',
            'width': WIDTH, 'dropout': 0., 'weight_decay': WEIGHT_DECAY, 'ema_decay': EMA_DECAY},
    'dropout_ema': {'family': 'weight_averaging', 'model': 'ema_flatten_mlp', 'base_model': 'regularized_flatten_mlp',
                    'width': WIDTH, 'dropout': DROPOUT, 'weight_decay': WEIGHT_DECAY, 'ema_decay': EMA_DECAY},
}
CONTRAST_KEY = 'ema_contrasts'
CONTROL = {'name': 'control', 'priority': 1, 'model': 'flatten_mlp', 'width': WIDTH, 'workload': []}
CONTRASTS = [
    {'name': 'ema_vs_control', 'candidate': 'ema', 'reference': 'control', 'isolates': 'ema',
     'held': {'dropout': 0.0}},
    {'name': 'dropout_ema_vs_dropout', 'candidate': 'dropout_ema', 'reference': 'dropout', 'isolates': 'ema',
     'held': {'dropout': DROPOUT}},
    {'name': 'dropout_vs_control', 'candidate': 'dropout', 'reference': 'control', 'isolates': 'dropout',
     'held': {'weights': 'raw'}},
]
ARM_ORDER = ('control', 'ema', 'dropout', 'dropout_ema')


def same(a, b):
    """Exact equality including types (0 vs 0.0, True vs 1)."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def ema_decay_at(decay, step):
    """Decay used for the update after optimizer step number `step` (0-based)."""
    if not 0 < decay < 1 or step < 0:
        raise ValueError('EMA decay in (0, 1) and a nonnegative step required')
    return min(float(decay), (1. + step) / (10. + step))


def arm_spec(name):
    if name not in ARMS:
        raise ValueError('Unknown weight-averaging arm: ' + str(name))
    return dict(ARMS[name])


def check_arm(arm):
    """An EMA arm is exactly its registry entry (+ name, priority 1, flag, no workload)."""
    name = arm.get('name')
    rest = {k: v for k, v in arm.items() if k not in ('name', 'priority', 'ema', 'workload')}
    if arm.get('ema') is not True or name not in ARMS or not same(rest, ARMS[name]):
        raise ValueError('EMA arm differs from the fixed registry: ' + str(name))
    if arm.get('workload') != [] or arm.get('priority') != 1:
        raise ValueError('EMA arms keep the control inputs (no workload) and priority 1')
    return arm


def check_family(config):
    """The whole family is the four fixed arms in order (control and frontier 'dropout' exact)."""
    from pitchmdp.direct_frontier_registry import ARMS as FRONTIER
    if [a.get('name') for a in config['arms']] != list(ARM_ORDER):
        raise ValueError('EMA family must be exactly ' + ', '.join(ARM_ORDER))
    arms = {a['name']: a for a in config['arms']}
    if not same(arms['control'], CONTROL):
        raise ValueError('control differs from the fresh flatten control spec')
    dropout = {k: v for k, v in arms['dropout'].items() if k not in ('name', 'priority', 'frontier', 'workload')}
    if arms['dropout'].get('frontier') is not True or not same(dropout, FRONTIER['dropout']) \
            or arms['dropout'].get('workload') != [] or arms['dropout'].get('priority') != 1 \
            or (FRONTIER['dropout']['dropout'], FRONTIER['dropout']['weight_decay']) != (DROPOUT, WEIGHT_DECAY):
        raise ValueError('dropout must be the exact frontier regularization arm (p .1, wd .01)')
    for name in ARMS:
        check_arm(arms[name])
    check_seeds(config)
    return config


def check_seeds(config):
    """Exactly seeds [19..23], no repeat_tag, 20 fits."""
    if 'repeat_tag' in config or not same(config.get('seeds'), SEEDS):
        raise ValueError('EMA seeds must be exactly [19, 20, 21, 22, 23] with no repeat_tag')
    if not same(config.get('fits'), FITS):
        raise ValueError(f'EMA family is exactly {FITS} fits')
    return config['seeds']


def check_contrasts(config):
    """comparison.ema_contrasts (or None): exactly the fixed list, checked before any read."""
    contrasts = config.get('comparison', {}).get(CONTRAST_KEY)
    if contrasts is None:
        return None
    if not same(contrasts, CONTRASTS):
        raise ValueError('ema_contrasts differ from the fixed factorial pairs')
    check_family(config)
    return contrasts
