"""Torch-free registry of the DRAFT embedding-map control family (DRAFT-CUDA-EMBED-TYPE-20261005, tentative EXP-P37-001).
Pre-registration: docs/PREREG-DRAFT-EXP-P37-001-embed-type-2026-10-05.md, written 2026-10-05 after the P36 report
(ce2861b) and before any P37 fit.  Designed after P24-P36 scores were read on repeated DEV: post-selection,
exploratory and descriptive only (no promotion, verdict, ranking, winner, selection or 2026).  Fresh MODEL RNG seeds
are NOT new data (same export, same DEV rows).
Question: P36 periodic_residual_counts vs control (-0.00379 dev / -0.00324 mlb_dev) bundles the periodic map, extra
embedding parameters / backbone input width and more compute.  Holding the residual backbone, the embedded columns,
the embedding width 8 and therefore the backbone input width fixed, does the periodic (PLR-lite) map lower calibrated
direct NLL / Brier vs the per-feature linear (LR) map with ~the same parameter count?
  embedding map        residual backbone, single head, counts workload, same 81 embedded scalars
  PLR-lite periodic    periodic_residual_counts (EXP-P32/P34/P35/P36 entry, embed_residual flag)
  LR linear            linear_residual_counts   (EXP-P32-001 entry, embed_residual flag)
plus control = MatrixModel flatten_mlp 128, no workload (runner requirement).  All three arms are the EXACT earlier
registered config entries (same flags, same frozen code path, no new model code).  Both reference arms keep the
embed_residual flag, so this family is recognised by the top-level marker MARKER and its predicate is tested FIRST
(config_check, training_budget, scorer).  3 arms x seeds exactly [39, 40, 41, 42, 43] = 15 fits, the cuda_direct_v1
budget with the epoch cap 60 (as P34-P36).  Seeds 39-43 are disjoint from every read or reserved block (0-38).
Fixed contrast (candidate - reference): embedding_map periodic_residual_counts - linear_residual_counts (primary,
only one).  PARAMETER-NEAR-MATCHED, NOT compute-matched: the periodic arm has PARAMETERS['periodic_residual_counts'] -
PARAMETERS['linear_residual_counts'] more parameters (tested from the frozen networks at the registered shapes) and
does more embedding work per row (2 x 16 cos/sin + a shared Linear(32, 8) per scalar vs one affine map per scalar);
early stopping can also change epochs and optimizer updates.  Cost is therefore reported per fit (seconds, updates,
seconds per update, epochs, parameters, peak device bytes), never used to adjust the NLL contrast.
Every arm is also reported vs the same-family CUDA control by the scorer (runner requirement), not as a contrast.
Hardware (Song 2026-10-05 single-GPU rule): all 15 fits on physical GPU3 only (UUID RETAINED_GPU), one at a time;
the released GPU4 is refused by check_device (called from check_family, hence by the runner's config_check).
"""
from __future__ import annotations

import json
import math

from pitchmdp import direct_embed_residual_registry as er
from pitchmdp import direct_robust_clip_registry as robust_clip

MARKER = 'embed_type_family'
CONTRAST_KEY = 'embed_type_contrasts'
SEEDS = [39, 40, 41, 42, 43]
EARLIER_SEEDS = {**robust_clip.EARLIER_SEEDS, 'EXP-P36-001': list(robust_clip.SEEDS)}
LEVEL = er.LEVEL
WORKLOAD = list(er.WORKLOAD)
CONTINUOUS = list(er.CONTINUOUS)
EPOCH_CAP = robust_clip.EPOCH_CAP
TRAINING = dict(robust_clip.TRAINING)  # cuda_direct_v1 budget with the epoch cap 60 (== P34/P35/P36)
# Exact earlier registered config entries (EXP-P32-001 / EXP-P36-001; old flags, old behavior).
BASELINES = {
    'control': robust_clip.BASELINES['control'],
    'periodic_residual_counts': robust_clip.BASELINES['periodic_residual_counts'],
    'linear_residual_counts': {'name': 'linear_residual_counts', 'priority': 1, er.FLAG: True, 'workload': WORKLOAD,
                               **er.ARMS['linear_residual_counts']},
}
ARM_ORDER = ('control', 'periodic_residual_counts', 'linear_residual_counts')
FITS = len(ARM_ORDER) * len(SEEDS)
# Allocated parameters at the registered shapes (length 6, n_token 38, n_context 52 + 1 counts scalar);
# re-derived from the frozen networks by the tests; identical to EXP-P32-001 fit.json values.
PARAMETERS = {'control': 141834, 'periodic_residual_counts': 244898, 'linear_residual_counts': 244634}
FACTOR_NAMES = ('embedding_map', 'embedded_columns', 'embedding_width', 'workload', 'backbone', 'dropout_rate',
                'dropout_position', 'head')
FACTORS = {'periodic_residual_counts': ('periodic', 'context_continuous+past_physics', 8, LEVEL, 'residual', .1,
                                        'blocks', 'single'),
           'linear_residual_counts': ('linear', 'context_continuous+past_physics', 8, LEVEL, 'residual', .1,
                                      'blocks', 'single')}
HELD = {'embedded_columns': 'context_continuous+past_physics', 'embedding_width': 8, 'workload': LEVEL,
        'backbone': 'residual', 'dropout_rate': .1, 'head': 'single'}
CONTRASTS = [
    {'name': 'periodic_residual_counts_vs_linear_residual_counts', 'candidate': 'periodic_residual_counts',
     'reference': 'linear_residual_counts', 'isolates': 'embedding_map', 'role': 'primary',
     'parameter_matched': False, 'parameter_near_matched': True, 'compute_matched': False, 'held': dict(HELD)},
]
PRIMARY = CONTRASTS[0]['name']
CHANGED = {PRIMARY: {'embedding_map'}}
# Single-GPU resource rule (Song 2026-10-05, supersedes the P36 two-GPU allow-list for this family): physical GPU3 by
# UUID only, at most one fit at a time; GPU4 is released to another project and must never be allowed or used as fallback.
RETAINED_GPU = 'GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9'
RELEASED_GPUS = ('GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99',)
DEVICE = {'kind': 'cuda', 'allowed_gpu_uuids': [RETAINED_GPU], 'max_parallel': 1, 'cpu_fallback': False}
COST_KEYS = ('parameter_count', 'fit_seconds', 'seconds_total', 'optimizer_updates', 'epochs_run', 'best_epoch',
             'seconds_per_update', 'peak_device_bytes')


def same(value, expected):
    """JSON-exact equality (True != 1, 8.0 != 8), so 'exact earlier entry' means the same config text."""
    return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)


def is_family(config):
    """The P37 DRAFT family predicate (top-level marker; its arms keep the P32 embed_residual flag)."""
    return MARKER in config


def check_seeds(config):
    """Exactly seeds [39, 40, 41, 42, 43], no repeat_tag, 15 fits."""
    if 'repeat_tag' in config or not same(config.get('seeds'), SEEDS):
        raise ValueError('Embed-type seeds must be exactly [39, 40, 41, 42, 43] with no repeat_tag')
    if set(SEEDS) & {s for block in EARLIER_SEEDS.values() for s in block}:
        raise ValueError('Embed-type seeds overlap an earlier seed block')
    if not same(config.get('fits'), FITS):
        raise ValueError(f'Embed-type family is exactly {FITS} fits')
    return config['seeds']


def check_family(config):
    """Marker true, exactly the three earlier registered arm entries in order, seeds 39-43, epoch cap 60."""
    if config.get(MARKER) is not True:
        raise ValueError(MARKER + ' must be exactly true')
    if [a.get('name') for a in config['arms']] != list(ARM_ORDER):
        raise ValueError('Embed-type family must be exactly ' + ', '.join(ARM_ORDER))
    for arm in config['arms']:
        if not same(arm, BASELINES[arm['name']]):
            raise ValueError(arm['name'] + ': arm differs from its exact earlier registered entry')
    check_seeds(config)
    check_device(config)
    if not same(config.get('training'), TRAINING):
        raise ValueError(f'Embed-type training must be the cuda_direct_v1 budget with epochs {EPOCH_CAP}')
    return config


def check_device(config):
    """Exactly DEVICE: CUDA, allow-list == [RETAINED_GPU], max_parallel 1, no CPU fallback; a released GPU is refused."""
    device = config.get('device') or {}
    if any(g in RELEASED_GPUS for g in device.get('allowed_gpu_uuids') or []):
        raise ValueError('Embed-type family must not allow a released GPU (GPU4 belongs to another project)')
    if not same(device, DEVICE):
        raise ValueError('Embed-type device must be exactly one GPU (GPU3 UUID) with max_parallel 1, no fallback')
    return device


def check_contrasts(config):
    """comparison.embed_type_contrasts (or None): exactly the fixed list, changing exactly the embedding map."""
    contrasts = config.get('comparison', {}).get(CONTRAST_KEY)
    if contrasts is None:
        return None
    if not same(contrasts, CONTRASTS):
        raise ValueError(CONTRAST_KEY + ' differ from the fixed pre-registered pair')
    check_family(config)
    for c in contrasts:
        cand, ref = (dict(zip(FACTOR_NAMES, FACTORS[c[k]])) for k in ('candidate', 'reference'))
        changed = {k for k in FACTOR_NAMES if cand[k] != ref[k]}
        if changed != CHANGED[c['name']] or any(cand.get(k) != v or ref.get(k) != v for k, v in c['held'].items()):
            raise ValueError(c['name'] + ': contrast must change exactly the embedding map')
    return contrasts


def cost_record(fit):
    """Per-fit cost from fit.json (runner fields, no prediction read); fail closed on missing or malformed values.

    fit_seconds is fit wall (export load/hash + training); train_seconds is the frozen training-loop report.seconds
    only (review 6b66436 M1/m1). The two are reported separately, never mixed."""
    report = fit.get('report') or {}
    try:
        updates, seconds = int(report['optimizer_updates']), float(fit['fit_seconds'])
        train, epochs = float(report['seconds']), int(report['epochs_run'])
        record = {'parameter_count': int(fit['parameter_count']), 'fit_seconds': seconds,
                  'seconds_total': float(fit['seconds_total']), 'optimizer_updates': updates,
                  'epochs_run': epochs, 'best_epoch': int(report['best_epoch']),
                  'seconds_per_update': seconds / updates,
                  'train_seconds': train, 'train_seconds_per_epoch': train / epochs,
                  'train_seconds_per_update': train / updates,
                  'peak_device_bytes': None if fit.get('peak_device_bytes') is None else int(fit['peak_device_bytes'])}
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f'Embed-type cost record malformed ({error!r})') from error
    if updates <= 0 or epochs <= 0 or not seconds > 0 or not (math.isfinite(train) and train > 0):
        raise ValueError('Embed-type cost record needs positive updates, epochs, fit seconds and finite train seconds')
    return record


def check_parameters(arm, record):
    """Allocated parameters equal the registered value for the arm (same frozen network at the registered shapes)."""
    if record['parameter_count'] != PARAMETERS[arm]:
        raise ValueError(f'{arm}: parameter_count {record["parameter_count"]} != registered {PARAMETERS[arm]}')
    return record
