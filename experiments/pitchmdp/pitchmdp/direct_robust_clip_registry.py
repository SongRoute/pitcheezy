"""Torch-free registry of the DRAFT robust-clip family (DRAFT-CUDA-ROBUST-CLIP-20261005, tentative EXP-P36-001).

Pre-registration: docs/PREREG-DRAFT-EXP-P36-001-robust-clip-2026-10-04.md (§2 question and contrasts, §3 fixed
design), written before any P35 score and before this wiring.  Designed after P24-P34 scores were read on repeated
DEV: post-selection, exploratory and descriptive only (no promotion, verdict, ranking, winner, selection or 2026).
Fresh MODEL RNG seeds are NOT new data (same export, same DEV rows).

Question: with identical data, residual backbone, counts workload, loop, epoch cap 60, early stop, calibration chain
and seeds, does the fixed RealMLP robust scale + smooth clip (direct_robust_clip_model, statistics from the TRAIN
argument only) on the 81 scalars the periodic arm embeds change calibrated direct NLL / Brier / ECE?

  numeric input treatment       residual backbone, single head, counts workload
  standardized only (raw)       residual_counts (EXP-P32-001 entry, combo flag, frozen direct_residual_model)
  PLR-lite periodic embedding   periodic_residual_counts (EXP-P32/P34/P35 entry, embed_residual flag)
  robust scale + smooth clip    robust_clip_residual_counts (new)

plus control = MatrixModel flatten_mlp 128, no workload.  The three earlier arms are the EXACT registered config
entries (same flags, same code path); periodic_residual_counts keeps its embed_residual flag and residual_counts its
combo flag, which is why the runner and scorer test this family's predicate FIRST.  4 arms x seeds exactly
[34, 35, 36, 37, 38] = 20 fits, the cuda_direct_v1 budget with the pre-registered epoch cap 60 (as P34/P35).
Seeds 34-38 are disjoint from every read or reserved block (0-33); no repeat tag.

Fixed contrasts (candidate - reference, PREREG §2 order):
  input_preprocessing      robust_clip_residual_counts - residual_counts: primary, one factor (same seed => identical
                           initial parameters; the module only registers buffers)
  numeric_input_treatment  robust_clip_residual_counts - periodic_residual_counts: secondary, descriptive, NOT
                           compute-matched (the periodic arm adds embedding parameters); never used to pick
Every arm is also reported vs the same-family CUDA control by the scorer (runner requirement), not as a contrast.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from pitchmdp import direct_embed_residual_registry as er
from pitchmdp import direct_ple_registry as ple
from pitchmdp import direct_workload_dropout_registry as wr

FLAG = 'robust_clip_residual'
CONTRAST_KEY = 'robust_clip_contrasts'
MODULE = 'direct_robust_clip_model.py'
# direct_robust_clip_model.py sha256: the ONE pin (author pin 5f1864b; re-pin here after the module review disposition).
MODULE_SHA256 = '22841f2c82b75099cc58ac5e78cda6b1fc4a387366ccb0944b7b6f41cbbee3f2'
KIND = 'robust_clip_residual_mlp'  # == direct_robust_clip_model.KIND (tested; this module stays torch-free)
CLIP = 3.                          # == direct_robust_clip_model.CLIP (tested)
SEEDS = [34, 35, 36, 37, 38]
EARLIER_SEEDS = {**ple.EARLIER_SEEDS, 'EXP-P35-001': list(ple.SEEDS)}
LEVEL = 'counts'
WORKLOAD = list(wr.WORKLOADS[LEVEL])
CONTINUOUS = list(wr.CONTINUOUS[LEVEL])
EPOCH_CAP = 60
# cuda_direct_v1 budget with the pre-registered epoch cap (every other entry unchanged; == the P34/P35 budget).
TRAINING = {'epochs': EPOCH_CAP, 'patience': 5, 'batch_size': 1024, 'learning_rate': 0.0005,
            'loss_weighting': 'uniform', 'hpo_trials': 0}

# Exact earlier registered config entries (EXP-P32-001 / EXP-P35-001 DRAFT; old flags, old behavior).
BASELINES = {'control': ple.BASELINES['control'], 'residual_counts': er.BASELINES['residual_counts'],
             'periodic_residual_counts': ple.BASELINES['periodic_residual_counts']}
# New arm name -> exact config entry without name / priority / flag / workload.
ARMS = {
    'robust_clip_residual_counts': {'family': 'robust_clip', 'model': KIND, 'base_model': 'residual_mlp',
                                    'preprocessing': 'robust_scale_smooth_clip', 'clip': CLIP, 'width': 128,
                                    'workload_level': LEVEL, 'context_continuous': CONTINUOUS},
}
ARM_ORDER = ('control', 'residual_counts', 'periodic_residual_counts', 'robust_clip_residual_counts')
FITS = len(ARM_ORDER) * len(SEEDS)
FACTOR_NAMES = ('numeric_input', 'workload', 'backbone', 'dropout_rate', 'dropout_position', 'head')
FACTORS = {'control': ('raw', 'none', 'flatten', 0., 'none', 'single'),
           'residual_counts': ('raw', LEVEL, 'residual', .1, 'blocks', 'single'),
           'periodic_residual_counts': ('periodic', LEVEL, 'residual', .1, 'blocks', 'single'),
           'robust_clip_residual_counts': ('robust_clip', LEVEL, 'residual', .1, 'blocks', 'single')}
HELD = {'workload': LEVEL, 'backbone': 'residual', 'dropout_rate': .1, 'head': 'single'}


def same(value, expected):
    """JSON-exact equality (True != 1, 8.0 != 8), so 'exact earlier entry' means the same config text."""
    return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)


def pair(candidate, reference, isolates, role, compute_matched):
    return {'name': f'{candidate}_vs_{reference}', 'candidate': candidate, 'reference': reference,
            'isolates': isolates, 'role': role, 'compute_matched': compute_matched, 'held': dict(HELD)}


# Order, pairs and isolates labels exactly as PREREG §2 (2 rows; primary first).
CONTRASTS = [
    pair('robust_clip_residual_counts', 'residual_counts', 'input_preprocessing', 'primary', True),
    pair('robust_clip_residual_counts', 'periodic_residual_counts', 'numeric_input_treatment', 'secondary', False),
]
PRIMARY, SECONDARY = (c['name'] for c in CONTRASTS)
CHANGED = {PRIMARY: {'numeric_input'}, SECONDARY: {'numeric_input'}}


def module_sha256():
    return hashlib.sha256(Path(__file__).with_name(MODULE).read_bytes()).hexdigest()


def check_module():
    """Fail closed unless direct_robust_clip_model.py is the pinned file."""
    digest = module_sha256()
    if digest != MODULE_SHA256:
        raise ValueError(f'{MODULE} sha256 {digest} differs from the registry pin')
    return digest


def is_robust_clip_arm(arm):
    return FLAG in arm


def check_arm(arm):
    """The new arm is exactly its registry entry (+ name, priority 1, its flag, the counts workload)."""
    name = arm.get('name')
    rest = {k: v for k, v in arm.items() if k not in ('name', 'priority', FLAG, 'workload')}
    if arm.get(FLAG) is not True or name not in ARMS or not same(rest, ARMS[name]):
        raise ValueError('Robust-clip arm differs from the fixed registry: ' + str(name))
    if not same(arm.get('workload'), WORKLOAD) or not same(arm.get('priority'), 1):
        raise ValueError(name + ': workload is not counts or priority is not 1')
    return arm


def check_seeds(config):
    """Exactly seeds [34, 35, 36, 37, 38], no repeat_tag, 20 fits."""
    if 'repeat_tag' in config or not same(config.get('seeds'), SEEDS):
        raise ValueError('Robust-clip seeds must be exactly [34, 35, 36, 37, 38] with no repeat_tag')
    if set(SEEDS) & {s for block in EARLIER_SEEDS.values() for s in block}:
        raise ValueError('Robust-clip seeds overlap an earlier seed block')
    if not same(config.get('fits'), FITS):
        raise ValueError(f'Robust-clip family is exactly {FITS} fits')
    return config['seeds']


def check_training(config):
    """The cuda_direct_v1 budget with the pre-registered epoch cap 60 for every arm (nothing else changed)."""
    if not same(config.get('training'), TRAINING):
        raise ValueError(f'Robust-clip training must be the cuda_direct_v1 budget with epochs {EPOCH_CAP}')
    return config['training']


def check_family(config):
    """The whole family is the four fixed arms in order; earlier arms are the exact registered entries."""
    check_module()
    if [a.get('name') for a in config['arms']] != list(ARM_ORDER):
        raise ValueError('Robust-clip family must be exactly ' + ', '.join(ARM_ORDER))
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
    """comparison.robust_clip_contrasts (or None): exactly the fixed list, each changing exactly the numeric input."""
    contrasts = config.get('comparison', {}).get(CONTRAST_KEY)
    if contrasts is None:
        return None
    if not same(contrasts, CONTRASTS):
        raise ValueError(CONTRAST_KEY + ' differ from the fixed pre-registered pairs')
    check_family(config)
    arms = {a['name']: a for a in config['arms']}
    for c in contrasts:
        cand, ref = (dict(zip(FACTOR_NAMES, FACTORS[c[k]])) for k in ('candidate', 'reference'))
        if any(not same(arms[n]['workload'], WORKLOAD) for n in (c['candidate'], c['reference'])):
            raise ValueError(c['name'] + ': workload differs from counts')
        changed = {k for k in FACTOR_NAMES if cand[k] != ref[k]}
        if changed != CHANGED[c['name']] or any(cand.get(k) != v or ref.get(k) != v for k, v in c['held'].items()):
            raise ValueError(c['name'] + ': contrast must change exactly the declared factors vs its reference')
    return contrasts


# ------------------------------------------------------------------ fit.json robust-clip record (runner and scorer)

HEX64 = re.compile(r'[0-9a-f]{64}')
RULES = ('iqr', 'range', 'constant')


def n_scalars(length, context_continuous=CONTINUOUS):
    """Transformed scalars: past positions x 8 physics channels + the continuous map (81 for the counts export)."""
    return (length - 1) * len(ple.PHYSICAL_CHANNELS) + len(context_continuous)


def fit_record(report, length):
    """Top-level fit.json fields from a robust-clip fit report, re-derived and checked (fail closed)."""
    network = report.get('network') or {}
    digest, counts = report.get('rc_stats_sha256'), report.get('rc_rule_counts')
    if not isinstance(digest, str) or not HEX64.fullmatch(digest) or network.get('rc_stats_sha256') != digest \
            or network.get('kind') != KIND or network.get('context_continuous') != CONTINUOUS:
        raise ValueError('Robust-clip network record is malformed')
    if not isinstance(counts, dict) or set(counts) != set(RULES) \
            or not all(type(counts[r]) is int and counts[r] >= 0 for r in RULES) \
            or sum(counts.values()) != n_scalars(length):
        raise ValueError('Robust-clip rule counts do not cover the transformed scalars')
    return {'rc_stats_sha256': digest, 'rc_rule_counts': {r: counts[r] for r in RULES}}


def check_stats_rows(record, length):
    """fit.json ``rc_stats_rows`` (runner): TRAIN-argument rows and per-past-position (oldest first) valid rows."""
    if not isinstance(record, dict) or set(record) != {'train_rows', 'past_valid_rows'} \
            or type(record['train_rows']) is not int or record['train_rows'] <= 0 \
            or not isinstance(record['past_valid_rows'], list) or len(record['past_valid_rows']) != length - 1 \
            or not all(type(v) is int and 0 < v <= record['train_rows'] for v in record['past_valid_rows']):
        raise ValueError('Robust-clip statistics rows record is malformed')
    return record
