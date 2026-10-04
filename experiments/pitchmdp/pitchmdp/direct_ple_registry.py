"""Torch-free registry of the DRAFT PLE-embedding family (DRAFT-CUDA-PLE-RESIDUAL-20261004, EXP-P35-001 DRAFT).

Pre-registration: docs/PREREG-DRAFT-EXP-P35-001-ple-embedding-2026-10-04.md (§5 family, §6 timing, §10 wiring),
review OPUS-review-p35-prereg-4ccd13f and its disposition (N1 dispatch first, N12 one bins hash).  Designed after
P24-P34 scores were read on repeated DEV: post-selection, exploratory and descriptive only (no promotion, verdict,
ranking, winner or 2026).  Fresh MODEL RNG seeds are NOT new data (same export, same DEV rows).

Question: with identical data, embedded scalars (5 past positions x 8 physics channels + the counts continuous
map), d = 8, backbone, loop, epoch cap 60 and calibration chain, does replacing the periodic PLR-lite embedding by
a piecewise-linear embedding (PLE version B, quantile bins on TRAIN only; direct_ple_model) change calibrated
direct NLL / Brier / ECE

  embedding \\ head      single network                          TabM-mini k=8
  periodic PLR-lite     periodic_residual_counts (EXP-P32/P34)  tabm_residual_counts (EXP-P34-001 entry)
  PLE (version B)       ple_residual_counts (new)               ple_tabm_residual_counts (new)

plus control = MatrixModel flatten_mlp 128, no workload.  The three earlier arms are the EXACT registered config
entries (same flags, same code path; tabm_residual_counts keeps its P34 flag, which is why the runner and scorer
test this family's predicate BEFORE the TabM-residual one).  5 arms x seeds exactly [29, 30, 31, 32, 33] = 25 fits,
one immutable export, the cuda_direct_v1 budget with the pre-registered epoch cap 60 (as P34).  Seeds 29-33 are
disjoint from every read or reserved block (0-2, P30 3-7, P30 repeat 8-12, P31 13-15, P32 16-18, P33 19-23,
P34 24-28); no repeat tag.

Fixed contrasts (candidate - reference, DRAFT §5 order; the two embedding_type rows are the headline rows):
  embedding_type_single  residual backbone, single head, PLE vs periodic (the embedding bundle of DRAFT §2)
  embedding_type_tabm    residual backbone, TabM-mini k=8, PLE vs periodic
  ensemble_head_ple      PLE, residual backbone, TabM-mini vs single head
  bundle                 each non-control arm vs the fresh flatten control (several factors; the DRAFT labels all
                         four ``bundle``, so the exact changed set is keyed by contrast name in CHANGED)
"""
from __future__ import annotations

import json
import re

from pitchmdp import direct_tabm_residual_registry as trr
from pitchmdp import direct_workload_dropout_registry as wr

FLAGS = {'ple_residual_counts': 'ple_residual', 'ple_tabm_residual_counts': 'ple_tabm_residual'}
FLAG_KEYS = tuple(FLAGS.values())
CONTRAST_KEY = 'ple_contrasts'
# direct_ple_model.py sha256: the ONE pin (author pin 1894a0d; re-pin here after the module review disposition).
MODULE_SHA256 = '6a1541a33f36d02787ba33c9061ef55ab2c48030b00f3aa7992be0876c637425'
K, SCALING_INIT = trr.K, trr.SCALING_INIT  # == direct_tabm_residual_model.ENSEMBLE (tested; this module stays torch-free)
N_BINS, D_EMBEDDING = 48, 8                 # == direct_ple_model.N_BINS / D_EMBEDDING (tested)
KINDS = {'ple_residual_counts': 'ple_residual_mlp', 'ple_tabm_residual_counts': 'ple_residual_tabm_mini_mlp'}
SEEDS = [29, 30, 31, 32, 33]
EARLIER_SEEDS = {'P24/P25/P27/P29': [0, 1, 2], 'EXP-P30-001': [3, 4, 5, 6, 7], 'EXP-P30-002': [8, 9, 10, 11, 12],
                 'EXP-P31-001': [13, 14, 15], 'EXP-P32-001': [16, 17, 18], 'EXP-P33-001': [19, 20, 21, 22, 23],
                 'EXP-P34-001': [24, 25, 26, 27, 28]}
LEVEL = 'counts'
WORKLOAD = list(wr.WORKLOADS[LEVEL])
CONTINUOUS = list(wr.CONTINUOUS[LEVEL])
EPOCH_CAP = 60
# cuda_direct_v1 budget with the pre-registered epoch cap (every other entry unchanged; == the P34 budget).
TRAINING = {'epochs': EPOCH_CAP, 'patience': 5, 'batch_size': 1024, 'learning_rate': 0.0005,
            'loss_weighting': 'uniform', 'hpo_trials': 0}

# Exact earlier registered config entries (EXP-P34-001 = EXP-P32-001 / P34 new arm; old flags, old behavior).
BASELINES = {name: trr.BASELINES[name] for name in ('control', 'periodic_residual_counts')}
BASELINES['tabm_residual_counts'] = {'name': 'tabm_residual_counts', 'priority': 1, trr.FLAG: True,
                                     'workload': WORKLOAD, **trr.ARMS['tabm_residual_counts']}
# New arm name -> exact config entry without name / priority / flag / workload.
ARMS = {
    'ple_residual_counts': {'family': 'ple_embedding', 'model': KINDS['ple_residual_counts'],
                            'base_model': 'periodic_residual_mlp', 'embedding': 'ple', 'n_bins': N_BINS,
                            'd_embedding': D_EMBEDDING, 'width': 128, 'workload_level': LEVEL,
                            'context_continuous': CONTINUOUS},
    'ple_tabm_residual_counts': {'family': 'ple_embedding', 'model': KINDS['ple_tabm_residual_counts'],
                                 'base_model': 'periodic_residual_tabm_mini_mlp', 'embedding': 'ple',
                                 'n_bins': N_BINS, 'd_embedding': D_EMBEDDING, 'width': 128, 'k': K,
                                 'scaling_init': SCALING_INIT, 'workload_level': LEVEL,
                                 'context_continuous': CONTINUOUS},
}
ARM_ORDER = ('control', 'periodic_residual_counts', 'ple_residual_counts', 'tabm_residual_counts',
             'ple_tabm_residual_counts')
FITS = len(ARM_ORDER) * len(SEEDS)
FACTOR_NAMES = ('input', 'workload', 'backbone', 'dropout_rate', 'dropout_position', 'head')
FACTORS = {'control': ('raw', 'none', 'flatten', 0., 'none', 'single'),
           'periodic_residual_counts': ('periodic', LEVEL, 'residual', .1, 'blocks', 'single'),
           'ple_residual_counts': ('ple', LEVEL, 'residual', .1, 'blocks', 'single'),
           'tabm_residual_counts': ('periodic', LEVEL, 'residual', .1, 'blocks', 'tabm_mini'),
           'ple_tabm_residual_counts': ('ple', LEVEL, 'residual', .1, 'blocks', 'tabm_mini')}
HELD_RESIDUAL = {'workload': LEVEL, 'backbone': 'residual', 'dropout_rate': .1}


def same(value, expected):
    """JSON-exact equality (True != 1, 8.0 != 8), so 'exact earlier entry' means the same config text."""
    return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)


def pair(candidate, reference, isolates, held):
    return {'name': f'{candidate}_vs_{reference}', 'candidate': candidate, 'reference': reference,
            'isolates': isolates, 'held': held}


# Order, pairs and isolates labels exactly as DRAFT §5 "Contrasts" (7 rows).
CONTRASTS = [
    pair('ple_residual_counts', 'periodic_residual_counts', 'embedding_type_single', {**HELD_RESIDUAL, 'head': 'single'}),
    pair('ple_tabm_residual_counts', 'tabm_residual_counts', 'embedding_type_tabm', {**HELD_RESIDUAL, 'head': 'tabm_mini'}),
    pair('ple_tabm_residual_counts', 'ple_residual_counts', 'ensemble_head_ple', {**HELD_RESIDUAL, 'input': 'ple'}),
    pair('periodic_residual_counts', 'control', 'bundle', {}),
    pair('ple_residual_counts', 'control', 'bundle', {}),
    pair('tabm_residual_counts', 'control', 'bundle', {}),
    pair('ple_tabm_residual_counts', 'control', 'bundle', {}),
]
HEADLINE = ('ple_residual_counts_vs_periodic_residual_counts', 'ple_tabm_residual_counts_vs_tabm_residual_counts')
BUNDLE = set(FACTOR_NAMES) - {'head'}
CHANGED = {'ple_residual_counts_vs_periodic_residual_counts': {'input'},
           'ple_tabm_residual_counts_vs_tabm_residual_counts': {'input'},
           'ple_tabm_residual_counts_vs_ple_residual_counts': {'head'},
           'periodic_residual_counts_vs_control': BUNDLE, 'ple_residual_counts_vs_control': BUNDLE,
           'tabm_residual_counts_vs_control': set(FACTOR_NAMES), 'ple_tabm_residual_counts_vs_control': set(FACTOR_NAMES)}


def is_ple_arm(arm):
    return any(key in arm for key in FLAG_KEYS)


def check_arm(arm):
    """A new arm is exactly its registry entry (+ name, priority 1, its own flag, the counts workload)."""
    name = arm.get('name')
    flag = FLAGS.get(name)
    rest = {k: v for k, v in arm.items() if k not in ('name', 'priority', flag, 'workload')}
    if flag is None or arm.get(flag) is not True or not same(rest, ARMS[name]):
        raise ValueError('PLE arm differs from the fixed registry: ' + str(name))
    if not same(arm.get('workload'), WORKLOAD) or not same(arm.get('priority'), 1):
        raise ValueError(name + ': workload is not counts or priority is not 1')
    return arm


def check_seeds(config):
    """Exactly seeds [29, 30, 31, 32, 33], no repeat_tag, 25 fits."""
    if 'repeat_tag' in config or not same(config.get('seeds'), SEEDS):
        raise ValueError('PLE seeds must be exactly [29, 30, 31, 32, 33] with no repeat_tag')
    if not same(config.get('fits'), FITS):
        raise ValueError(f'PLE family is exactly {FITS} fits')
    return config['seeds']


def check_training(config):
    """The cuda_direct_v1 budget with the pre-registered epoch cap 60 for every arm (nothing else changed)."""
    if not same(config.get('training'), TRAINING):
        raise ValueError(f'PLE training must be the cuda_direct_v1 budget with epochs {EPOCH_CAP}')
    return config['training']


def check_family(config):
    """The whole family is the five fixed arms in order; earlier arms are the exact registered entries."""
    if [a.get('name') for a in config['arms']] != list(ARM_ORDER):
        raise ValueError('PLE family must be exactly ' + ', '.join(ARM_ORDER))
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
    """comparison.ple_contrasts (or None): exactly the fixed list, each changing exactly its expected factors."""
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
        if changed != CHANGED[c['name']] or (c['isolates'] == 'bundle') != (len(changed) > 1) \
                or any(cand.get(k) != v or ref.get(k) != v for k, v in c['held'].items()):
            raise ValueError(c['name'] + ': contrast must change exactly the declared factors vs its reference')
    return contrasts


# ------------------------------------------------------------------ fit.json PLE record (runner and scorer)

# == sequence_data.PHYSICAL_CHANNELS, the first 8 token channels (tested; copied so this module stays pandas-free).
PHYSICAL_CHANNELS = ('effective_speed', 'release_spin_rate', 'spin_axis_sin', 'spin_axis_cos', 'pfx_x', 'pfx_z',
                     'plate_x', 'plate_z')
HEX64 = re.compile(r'[0-9a-f]{64}')
LABEL = re.compile(r'past(\d+)\.phys(\d+)|context(\d+)')


def feature_labels(length, context_continuous):
    """Embedded-scalar labels in embedding order (== direct_ple_model.feature_labels; tested)."""
    return [f'past{p}.phys{c}' for p in range(length - 1) for c in range(len(PHYSICAL_CHANNELS))] + \
        [f'context{int(i)}' for i in context_continuous]


def column_name(label, length, context_names, workload):
    """Export column name of one embedded-scalar label: past{p}.<physics channel> or the context / workload name.

    Past positions are the exported history order (oldest to newest, sequence_data.build_history_indices)."""
    match = LABEL.fullmatch(label)
    if match is None:
        raise ValueError('Unknown PLE feature label: ' + str(label))
    if match.group(3) is None:
        p, c = int(match.group(1)), int(match.group(2))
        if not (p < length - 1 and c < len(PHYSICAL_CHANNELS)):
            raise ValueError('PLE physics label outside the token layout: ' + label)
        return f'past{p}.{PHYSICAL_CHANNELS[c]}'
    i, names = int(match.group(3)), [*context_names, *workload]
    if i >= len(names):
        raise ValueError('PLE context label outside the context + workload layout: ' + label)
    return names[i]


def fit_record(network, length, context_names, workload):
    """Top-level fit.json fields from report['network'] of a PLE fit, re-derived and checked (fail closed)."""
    labels = feature_labels(length, network.get('context_continuous', ()))
    digest, counts, degenerate = (network.get(k) for k in ('ple_bins_sha256', 'ple_n_bins', 'ple_degenerate_columns'))
    if not isinstance(digest, str) or not HEX64.fullmatch(digest) or not isinstance(counts, list) \
            or len(counts) != len(labels) or not all(type(t) is int and 0 <= t <= N_BINS for t in counts) \
            or network.get('n_bins') != N_BINS:
        raise ValueError('PLE network record is malformed')
    if degenerate != [label for label, t in zip(labels, counts) if t == 0]:
        raise ValueError('PLE degenerate columns differ from the recorded bin counts')
    names = [column_name(label, length, context_names, workload) for label in degenerate]
    return {'ple_bins_sha256': digest, 'ple_n_bins': list(counts), 'ple_degenerate_columns': names,
            'ple_degenerate_count': len(names)}
