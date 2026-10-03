"""Torch-free registry of the DRAFT TabM-mini family (DRAFT-CUDA-TABM-MINI-20261004).

Design note: docs/DRAFT-TABM-MINI-DESIGN-20261004.md (fixed before any TabM fit or P30
score was read by this worker).  Every family so far is exposed repeated DEV and
workload ``counts`` was chosen after reading P27: this family is post-selection,
exploratory and descriptive only (no promotion, verdict, ranking, winner or 2026).
Fresh MODEL RNG seeds are NOT new data (same export, same DEV rows).

Question: with the same periodic input + workload ``counts``, data, loop and
calibration chain, does a TabM-mini k=8 shared-weight ensemble head
(direct_tabm_model.DirectTabMModel) change calibrated direct NLL / Brier?

  head \\ head dropout   0                               .1
  single (frozen)       periodic_counts (P27/P30 entry)  periodic_counts_d10 (P30 entry)
  TabM-mini k=8         tabm_counts                      tabm_counts_d10

plus control = MatrixModel flatten_mlp 128, no workload.  The three single-network
arms are the EXACT earlier config entries (same flags, same code path).  5 arms x
seeds exactly [13, 14, 15] = 15 fits, one immutable export, the cuda_direct_v1
budget (30 epochs, patience 5, lr 5e-4, batch 1024, AdamW weight decay .01,
uniform loss), external calibration.  Seeds 13-15 are disjoint from every read or
reserved block (P24/P25/P27/P29 0-2, P30 3-7, P30 repeat 8-12); no repeat tag.

Fixed contrasts (candidate - reference, each changes exactly one factor):
  ensemble_head  same head dropout, TabM-mini vs single network   (2)
  head_dropout   single network, dropout .1 vs 0                  (1)
"""
from __future__ import annotations

import json

from pitchmdp import direct_workload_dropout_registry as wr

FLAG = 'tabm'
CONTRAST_KEY = 'tabm_contrasts'
K, SCALING_INIT = 8, 'normal'  # == direct_tabm_model.ENSEMBLE (tested; this module stays torch-free)
SEEDS = [13, 14, 15]
DROPOUTS = (0., .1)
LEVEL = 'counts'
WORKLOAD = list(wr.WORKLOADS[LEVEL])
CONTINUOUS = list(wr.CONTINUOUS[LEVEL])
SINGLE = {0.: 'periodic_counts', .1: 'periodic_counts_d10'}
TABM = {0.: 'tabm_counts', .1: 'tabm_counts_d10'}

# Exact earlier config entries of the single-network arms (old behavior, old flags).
BASELINES = {
    'control': wr.BASELINES['control'],
    'periodic_counts': wr.BASELINES['periodic_counts'],
    'periodic_counts_d10': {'name': 'periodic_counts_d10', 'priority': 1, wr.FLAG: True, 'workload': WORKLOAD,
                            **wr.ARMS['periodic_counts_d10']},
}
# New arm name -> exact config entry without name / priority / flag / workload.
ARMS = {TABM[d]: {'family': 'tabm_mini', 'model': 'periodic_tabm_mini_mlp', 'base_model': 'periodic_flatten_mlp',
                  'width': 128, 'head_dropout': d, 'k': K, 'scaling_init': SCALING_INIT, 'workload_level': LEVEL,
                  'context_continuous': CONTINUOUS}
        for d in DROPOUTS}
ARM_ORDER = ('control', SINGLE[0.], SINGLE[.1], TABM[0.], TABM[.1])
FITS = len(ARM_ORDER) * len(SEEDS)


def same(value, expected):
    """JSON-exact equality (True != 1, 8.0 != 8), so 'exact earlier entry' means the same config text."""
    return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)


CONTRASTS = [
    *({'name': f'{TABM[d]}_vs_{SINGLE[d]}', 'candidate': TABM[d], 'reference': SINGLE[d],
       'isolates': 'ensemble_head', 'held': {'head_dropout': d}} for d in DROPOUTS),
    {'name': f'{SINGLE[.1]}_vs_{SINGLE[0.]}', 'candidate': SINGLE[.1], 'reference': SINGLE[0.],
     'isolates': 'head_dropout', 'held': {'head': 'single'}},
]


def check_arm(arm):
    """A TabM arm is exactly its registry entry (+ name, priority 1, flag, the counts workload)."""
    name = arm.get('name')
    rest = {k: v for k, v in arm.items() if k not in ('name', 'priority', FLAG, 'workload')}
    if arm.get(FLAG) is not True or name not in ARMS or not same(rest, ARMS[name]):
        raise ValueError('TabM-mini arm differs from the fixed registry: ' + str(name))
    if not same(arm.get('workload'), WORKLOAD) or not same(arm.get('priority'), 1):
        raise ValueError(name + ': workload is not counts or priority is not 1')
    return arm


def check_seeds(config):
    """Exactly seeds [13, 14, 15], no repeat_tag, 15 fits."""
    if 'repeat_tag' in config or not same(config.get('seeds'), SEEDS):
        raise ValueError('TabM-mini seeds must be exactly [13, 14, 15] with no repeat_tag')
    if not same(config.get('fits'), FITS):
        raise ValueError(f'TabM-mini family is exactly {FITS} fits')
    return config['seeds']


def check_family(config):
    """The whole family is the five fixed arms in order; single-network arms are the exact earlier entries."""
    if [a.get('name') for a in config['arms']] != list(ARM_ORDER):
        raise ValueError('TabM-mini family must be exactly ' + ', '.join(ARM_ORDER))
    for arm in config['arms']:
        if arm['name'] in BASELINES:
            if not same(arm, BASELINES[arm['name']]):
                raise ValueError(arm['name'] + ': single-network arm differs from its exact earlier spec')
        else:
            check_arm(arm)
    check_seeds(config)
    return config


def factors(arm):
    """(head, head dropout, continuous map, workload) of a family arm other than control."""
    name = arm['name']
    if name in ARMS:
        return 'tabm_mini', ARMS[name]['head_dropout'], arm['context_continuous'], arm['workload']
    if name in SINGLE.values():
        return 'single', {v: d for d, v in SINGLE.items()}[name], arm['context_continuous'], arm['workload']
    raise ValueError(name + ': not a TabM family contrast arm')


def check_contrasts(config):
    """comparison.tabm_contrasts (or None): exactly the fixed list, each changing one factor."""
    contrasts = config.get('comparison', {}).get(CONTRAST_KEY)
    if contrasts is None:
        return None
    if not same(contrasts, CONTRASTS):
        raise ValueError(CONTRAST_KEY + ' differ from the fixed ensemble-head / head-dropout pairs')
    check_family(config)
    arms = {a['name']: a for a in config['arms']}
    for c in contrasts:
        (ch, cd, cc, cw), (rh, rd, rc, rw) = factors(arms[c['candidate']]), factors(arms[c['reference']])
        if cc != CONTINUOUS or rc != CONTINUOUS or cw != WORKLOAD or rw != WORKLOAD:
            raise ValueError(c['name'] + ': input differs from periodic + counts')
        changed = ({'ensemble_head'} if ch != rh else set()) | ({'head_dropout'} if cd != rd else set())
        held = {'head_dropout': cd} if c['isolates'] == 'ensemble_head' else {'head': ch}
        reference_is_base = rh == 'single' if c['isolates'] == 'ensemble_head' else rd == 0.
        if changed != {c['isolates']} or c['held'] != held or not reference_is_base:
            raise ValueError(c['name'] + ': contrast must change exactly the declared factor vs its reference')
    return contrasts
