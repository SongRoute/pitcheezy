"""Torch-free registry of the DRAFT TabM-mini x periodic-residual family (DRAFT-CUDA-TABM-RESIDUAL-20261004).

Pre-registration: docs/DRAFT-TABM-RESIDUAL-DESIGN-20261004.md (committed before any module
code, wiring, smoke or fit of this family).  Designed after P30-P33 scores were read on
repeated DEV: post-selection, exploratory and descriptive only (no promotion, verdict,
ranking, winner or 2026).  Fresh MODEL RNG seeds are NOT new data (same export, same DEV rows).

Question: with the same periodic input + workload ``counts``, data, loop, epoch cap and
calibration chain, does the TabM-mini k=8 shared-weight ensemble on the frozen residual
backbone (direct_tabm_residual_model.DirectTabMResidualModel) change calibrated direct
NLL / Brier relative to each parent?

  backbone \\ head        single network                        TabM-mini k=8
  flatten (dropout 0)    -                                     tabm_counts (EXP-P31-001 entry)
  flatten (head .1)      -                                     tabm_counts_d10 (EXP-P31-001 entry; Amendment 1)
  residual (blocks .1)   periodic_residual_counts (EXP-P32-001) tabm_residual_counts (new)

plus control = MatrixModel flatten_mlp 128, no workload.  The three earlier arms are the
EXACT registered config entries (same flags, same code path).  5 arms x seeds exactly
[24, 25, 26, 27, 28] = 25 fits (Amendment 1 added tabm_counts_d10 before any smoke/fit), one immutable export, the cuda_direct_v1 budget EXCEPT the
epoch cap: 60 for every arm (fixed in the pre-registration before any fit; no learning-rate
schedule, so a run that stops by patience 5 before epoch 30 is algorithmically unchanged).
Numbers are not compared with P31/P32 (other cap, seeds, run batch).  Seeds 24-28 are
disjoint from every read or reserved block (0-2, P30 3-7, P30 repeat 8-12, P31 13-15,
P32 16-18, P33 19-23); no repeat tag.

Fixed contrasts (candidate - reference; ISOLATES gives the exact changed factor set):
  ensemble_head          residual backbone, TabM-mini vs single network
  backbone_bundle_d10    TabM-mini head, residual vs flatten backbone at dropout .1 on both
                         sides (structure + member-head width + dropout placement together;
                         not one-factor, disclosed)
  backbone_bundle        TabM-mini head, residual vs flatten backbone (structure + dropout
                         0 -> blocks .1 together; disclosed, not separated)
  *_bundle_vs_control    each non-control arm vs the fresh flatten control (several factors)
"""
from __future__ import annotations

import json

from pitchmdp import direct_embed_residual_registry as er
from pitchmdp import direct_tabm_registry as tr
from pitchmdp import direct_workload_dropout_registry as wr

FLAG = 'tabm_residual'
CONTRAST_KEY = 'tabm_residual_contrasts'
K, SCALING_INIT = tr.K, tr.SCALING_INIT  # == direct_tabm_residual_model.ENSEMBLE (tested; this module stays torch-free)
SEEDS = [24, 25, 26, 27, 28]
LEVEL = 'counts'
WORKLOAD = list(wr.WORKLOADS[LEVEL])
CONTINUOUS = list(wr.CONTINUOUS[LEVEL])
EPOCH_CAP = 60
# cuda_direct_v1 budget with the pre-registered epoch cap (every other entry unchanged).
TRAINING = {'epochs': EPOCH_CAP, 'patience': 5, 'batch_size': 1024, 'learning_rate': 0.0005,
            'loss_weighting': 'uniform', 'hpo_trials': 0}

# Exact earlier registered config entries (EXP-P31-001 / EXP-P32-001; old flags, old behavior).
BASELINES = {
    'control': wr.BASELINES['control'],
    'periodic_residual_counts': {'name': 'periodic_residual_counts', 'priority': 1, er.FLAG: True,
                                 'workload': WORKLOAD, **er.ARMS['periodic_residual_counts']},
    'tabm_counts': {'name': 'tabm_counts', 'priority': 1, tr.FLAG: True, 'workload': WORKLOAD,
                    **tr.ARMS['tabm_counts']},
    'tabm_counts_d10': {'name': 'tabm_counts_d10', 'priority': 1, tr.FLAG: True, 'workload': WORKLOAD,
                        **tr.ARMS['tabm_counts_d10']},
}
# New arm name -> exact config entry without name / priority / flag / workload.
ARMS = {
    'tabm_residual_counts': {'family': FLAG, 'model': 'periodic_residual_tabm_mini_mlp',
                             'base_model': 'periodic_residual_mlp', 'embedding': 'periodic', 'width': 128,
                             'k': K, 'scaling_init': SCALING_INIT, 'workload_level': LEVEL,
                             'context_continuous': CONTINUOUS},
}
ARM_ORDER = ('control', 'periodic_residual_counts', 'tabm_counts', 'tabm_residual_counts', 'tabm_counts_d10')
FITS = len(ARM_ORDER) * len(SEEDS)
FACTOR_NAMES = ('input', 'workload', 'backbone', 'dropout_rate', 'dropout_position', 'head')
FACTORS = {'control': ('raw', 'none', 'flatten', 0., 'none', 'single'),
           'periodic_residual_counts': ('periodic', LEVEL, 'residual', .1, 'blocks', 'single'),
           'tabm_counts': ('periodic', LEVEL, 'flatten', 0., 'none', 'tabm_mini'),
           'tabm_residual_counts': ('periodic', LEVEL, 'residual', .1, 'blocks', 'tabm_mini'),
           'tabm_counts_d10': ('periodic', LEVEL, 'flatten', .1, 'head', 'tabm_mini')}
ISOLATES = {'ensemble_head': {'head'},
            'backbone_bundle_d10': {'backbone', 'dropout_position'},
            'backbone_bundle': {'backbone', 'dropout_rate', 'dropout_position'},
            'tabm_bundle_vs_control': {'input', 'workload', 'head'},
            'tabm_d10_bundle_vs_control': {'input', 'workload', 'dropout_rate', 'dropout_position', 'head'},
            'residual_bundle_vs_control': {'input', 'workload', 'backbone', 'dropout_rate', 'dropout_position'},
            'tabm_residual_bundle_vs_control': set(FACTOR_NAMES)}


def same(value, expected):
    """JSON-exact equality (True != 1, 8.0 != 8), so 'exact earlier entry' means the same config text."""
    return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)


def pair(candidate, reference, isolates, held):
    return {'name': f'{candidate}_vs_{reference}', 'candidate': candidate, 'reference': reference,
            'isolates': isolates, 'held': held}


# Order and pairs exactly as in the pre-registration "Fixed contrasts" (Amendment 1: 7 pairs).
CONTRASTS = [
    pair('tabm_residual_counts', 'periodic_residual_counts', 'ensemble_head',
         {'input': 'periodic', 'workload': LEVEL, 'backbone': 'residual'}),
    pair('tabm_residual_counts', 'tabm_counts_d10', 'backbone_bundle_d10',
         {'input': 'periodic', 'workload': LEVEL, 'head': 'tabm_mini', 'dropout_rate': .1}),
    pair('tabm_residual_counts', 'tabm_counts', 'backbone_bundle',
         {'input': 'periodic', 'workload': LEVEL, 'head': 'tabm_mini'}),
    pair('tabm_counts', 'control', 'tabm_bundle_vs_control', {'backbone': 'flatten', 'dropout_rate': 0.}),
    pair('tabm_counts_d10', 'control', 'tabm_d10_bundle_vs_control', {'backbone': 'flatten'}),
    pair('periodic_residual_counts', 'control', 'residual_bundle_vs_control', {'head': 'single'}),
    pair('tabm_residual_counts', 'control', 'tabm_residual_bundle_vs_control', {}),
]


def check_arm(arm):
    """A new arm is exactly its registry entry (+ name, priority 1, flag, the counts workload)."""
    name = arm.get('name')
    rest = {k: v for k, v in arm.items() if k not in ('name', 'priority', FLAG, 'workload')}
    if arm.get(FLAG) is not True or name not in ARMS or not same(rest, ARMS[name]):
        raise ValueError('TabM-residual arm differs from the fixed registry: ' + str(name))
    if not same(arm.get('workload'), WORKLOAD) or not same(arm.get('priority'), 1):
        raise ValueError(name + ': workload is not counts or priority is not 1')
    return arm


def check_seeds(config):
    """Exactly seeds [24, 25, 26, 27, 28], no repeat_tag, 25 fits."""
    if 'repeat_tag' in config or not same(config.get('seeds'), SEEDS):
        raise ValueError('TabM-residual seeds must be exactly [24, 25, 26, 27, 28] with no repeat_tag')
    if not same(config.get('fits'), FITS):
        raise ValueError(f'TabM-residual family is exactly {FITS} fits')
    return config['seeds']


def check_training(config):
    """The cuda_direct_v1 budget with the pre-registered epoch cap 60 for every arm (nothing else changed)."""
    if not same(config.get('training'), TRAINING):
        raise ValueError(f'TabM-residual training must be the cuda_direct_v1 budget with epochs {EPOCH_CAP}')
    return config['training']


def check_family(config):
    """The whole family is the five fixed arms in order; earlier arms are the exact registered entries."""
    if [a.get('name') for a in config['arms']] != list(ARM_ORDER):
        raise ValueError('TabM-residual family must be exactly ' + ', '.join(ARM_ORDER))
    for arm in config['arms']:
        if arm['name'] in BASELINES:
            if not same(arm, BASELINES[arm['name']]):
                raise ValueError(arm['name'] + ': earlier arm differs from its exact registered spec')
        else:
            check_arm(arm)
    check_seeds(config)
    check_training(config)
    return config


def check_contrasts(config):
    """comparison.tabm_residual_contrasts (or None): exactly the fixed list, each changing its declared factors."""
    contrasts = config.get('comparison', {}).get(CONTRAST_KEY)
    if contrasts is None:
        return None
    if not same(contrasts, CONTRASTS):
        raise ValueError(CONTRAST_KEY + ' differ from the fixed pre-registered pairs')
    check_family(config)
    arms = {a['name']: a for a in config['arms']}
    for c in contrasts:
        cand, ref = (dict(zip(FACTOR_NAMES, FACTORS[c[k]])) for k in ('candidate', 'reference'))
        workload = {n: WORKLOAD if FACTORS[n][1] == LEVEL else [] for n in (c['candidate'], c['reference'])}
        if any(not same(arms[n]['workload'], w) for n, w in workload.items()):
            raise ValueError(c['name'] + ': workload differs from the registered arms')
        changed = {k for k in FACTOR_NAMES if cand[k] != ref[k]}
        if changed != ISOLATES[c['isolates']] or any(cand.get(k) != v or ref.get(k) != v for k, v in c['held'].items()):
            raise ValueError(c['name'] + ': contrast must change exactly the declared factors vs its reference')
    return contrasts
