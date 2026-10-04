"""Torch-free registry of the DRAFT embedded-input residual family (DRAFT-CUDA-EMBED-RESIDUAL-20261004).

Design note: docs/DRAFT-PERIODIC-RESIDUAL-DESIGN-20261004.md (fixed before any fit of
this family).  Designed after reading P24/P27/P29/P31 on repeated DEV: post-hoc,
exploratory and descriptive only (no promotion, verdict, ranking, winner or 2026).
Fresh MODEL RNG seeds are NOT new data (same export, same DEV rows).

Question: with workload ``counts``, the same data, loop and calibration chain, does
feeding the frozen PLR-lite periodic embedding into the residual backbone
(direct_periodic_residual_model) change calibrated direct NLL / Brier relative to
each parent, and is a difference due to the periodic map or to per-feature
embedding as such (LR control)?

  input \\ backbone      flatten_mlp head                  residual
  raw                   -                                 residual_counts (P27 entry)
  PLR-lite periodic     periodic_counts (P27/P30 entry)   periodic_residual_counts
  LR linear             -                                 linear_residual_counts

plus control = MatrixModel flatten_mlp 128, no workload.  The three baseline arms
are the EXACT earlier config entries (same flags, same code path).  5 arms x seeds
exactly [16, 17, 18] = 15 fits.  Seeds 16-18 are disjoint from every read or
reserved block (0-2, P30 3-7, P30 repeat 8-12, P31 13-15); no repeat tag.

Fixed contrasts (candidate - reference, each changes exactly one factor):
  backbone        PLR-lite input, residual vs flatten head      (1)
  input           residual backbone, PLR-lite vs raw            (1)
  embedding_type  residual backbone, PLR-lite vs LR             (1)
  embedding       residual backbone, LR vs raw                  (1)
"""
from __future__ import annotations

import json

from pitchmdp import direct_workload_dropout_registry as wr

FLAG = 'embed_residual'
CONTRAST_KEY = 'embed_residual_contrasts'
SEEDS = [16, 17, 18]
LEVEL = 'counts'
WORKLOAD = list(wr.WORKLOADS[LEVEL])
CONTINUOUS = list(wr.CONTINUOUS[LEVEL])

# Exact earlier config entries (DRAFT-CUDA-DATA-COMBO-RUN-20261004 / EXP-P30-001; old flags, old behavior).
BASELINES = {
    'control': wr.BASELINES['control'],
    'residual_counts': {'name': 'residual_counts', 'priority': 1, 'combo': True, 'workload': WORKLOAD,
                        'family': 'combination', 'model': 'residual_mlp', 'width': 128},
    'periodic_counts': wr.BASELINES['periodic_counts'],
}
# New arm name -> exact config entry without name / priority / flag / workload.
ARMS = {
    'periodic_residual_counts': {'family': FLAG, 'model': 'periodic_residual_mlp', 'embedding': 'periodic',
                                 'width': 128, 'workload_level': LEVEL, 'context_continuous': CONTINUOUS},
    'linear_residual_counts': {'family': FLAG, 'model': 'linear_residual_mlp', 'embedding': 'linear',
                               'width': 128, 'workload_level': LEVEL, 'context_continuous': CONTINUOUS},
}
ARM_ORDER = ('control', 'residual_counts', 'periodic_counts', 'periodic_residual_counts', 'linear_residual_counts')
FITS = len(ARM_ORDER) * len(SEEDS)
# (input, backbone) of every contrast arm; control is not in any contrast.
FACTORS = {'residual_counts': ('raw', 'residual'), 'periodic_counts': ('periodic', 'flatten'),
           'periodic_residual_counts': ('periodic', 'residual'), 'linear_residual_counts': ('linear', 'residual')}


def same(value, expected):
    """JSON-exact equality (True != 1, 8.0 != 8), so 'exact earlier entry' means the same config text."""
    return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)


CONTRASTS = [
    {'name': 'periodic_residual_counts_vs_periodic_counts', 'candidate': 'periodic_residual_counts',
     'reference': 'periodic_counts', 'isolates': 'backbone', 'held': {'input': 'periodic'}},
    {'name': 'periodic_residual_counts_vs_residual_counts', 'candidate': 'periodic_residual_counts',
     'reference': 'residual_counts', 'isolates': 'input', 'held': {'backbone': 'residual'}},
    {'name': 'periodic_residual_counts_vs_linear_residual_counts', 'candidate': 'periodic_residual_counts',
     'reference': 'linear_residual_counts', 'isolates': 'embedding_type', 'held': {'backbone': 'residual'}},
    {'name': 'linear_residual_counts_vs_residual_counts', 'candidate': 'linear_residual_counts',
     'reference': 'residual_counts', 'isolates': 'embedding', 'held': {'backbone': 'residual'}},
]


def check_arm(arm):
    """A new arm is exactly its registry entry (+ name, priority 1, flag, the counts workload)."""
    name = arm.get('name')
    rest = {k: v for k, v in arm.items() if k not in ('name', 'priority', FLAG, 'workload')}
    if arm.get(FLAG) is not True or name not in ARMS or not same(rest, ARMS[name]):
        raise ValueError('Embedded-residual arm differs from the fixed registry: ' + str(name))
    if not same(arm.get('workload'), WORKLOAD) or not same(arm.get('priority'), 1):
        raise ValueError(name + ': workload is not counts or priority is not 1')
    return arm


def check_seeds(config):
    """Exactly seeds [16, 17, 18], no repeat_tag, 15 fits."""
    if 'repeat_tag' in config or not same(config.get('seeds'), SEEDS):
        raise ValueError('Embedded-residual seeds must be exactly [16, 17, 18] with no repeat_tag')
    if not same(config.get('fits'), FITS):
        raise ValueError(f'Embedded-residual family is exactly {FITS} fits')
    return config['seeds']


def check_family(config):
    """The whole family is the five fixed arms in order; baseline arms are the exact earlier entries."""
    if [a.get('name') for a in config['arms']] != list(ARM_ORDER):
        raise ValueError('Embedded-residual family must be exactly ' + ', '.join(ARM_ORDER))
    for arm in config['arms']:
        if arm['name'] in BASELINES:
            if not same(arm, BASELINES[arm['name']]):
                raise ValueError(arm['name'] + ': baseline arm differs from its exact earlier spec')
        else:
            check_arm(arm)
    check_seeds(config)
    return config


def check_contrasts(config):
    """comparison.embed_residual_contrasts (or None): exactly the fixed list, each changing one factor."""
    contrasts = config.get('comparison', {}).get(CONTRAST_KEY)
    if contrasts is None:
        return None
    if not same(contrasts, CONTRASTS):
        raise ValueError(CONTRAST_KEY + ' differ from the fixed backbone / input / embedding pairs')
    check_family(config)
    arms = {a['name']: a for a in config['arms']}
    for c in contrasts:
        (ci, cb), (ri, rb) = FACTORS[c['candidate']], FACTORS[c['reference']]
        if any(arms[n]['workload'] != WORKLOAD for n in (c['candidate'], c['reference'])):
            raise ValueError(c['name'] + ': workload differs from counts')
        changed = ({'backbone'} if cb != rb else set()) | ({'input'} if ci != ri else set())
        held = {'input': ci} if c['isolates'] == 'backbone' else {'backbone': cb}
        expected = {'backbone'} if c['isolates'] == 'backbone' else {'input'}
        if changed != expected or c['held'] != held or (ci == ri and cb == rb):
            raise ValueError(c['name'] + ': contrast must change exactly the declared factor vs its reference')
    return contrasts
