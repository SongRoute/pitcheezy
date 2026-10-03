"""Torch-free registry of the DRAFT data-quantity / pitcher-information / combination arms.

Family DRAFT-CUDA-DATA-COMBO-RUN-20261004 (root assigns the final ID).  Same-family
CUDA control (MatrixModel flatten_mlp 128, full TRAIN, no workload), same immutable
export, budget, early stop and external calibration.  Data and pitcher-information
arms are ONE disclosed change from the control.  The combination arms change BOTH
architecture and workload information vs the control; the config also carries the
base counts / block / residual / periodic arms (non-registry, exact DRAFT-CUDA-DIRECT
specs) so architecture x workload is a complete 3x3 factorial, and joint arms are
attributed only through the config-owned factorial_contrasts (vs the same-architecture
no-workload arm = workload; vs the same-workload flatten arm = architecture):

  data_quantity  data25 / data50 / data75: a fixed nested subset of the TRAIN rows
                 only.  u(row) = blake2b-64(salt, int64 keys row) / 2**64; a row is
                 kept iff u < fraction.  The same u for every fraction makes the
                 subsets nested; u depends on the keys and the config salt only (not
                 on model seeds, labels, DEV or outcomes); the boolean mask keeps the
                 original row order.  Early stop / temperature / blend / dev / mlb_dev
                 rows and every frozen TRAIN-fitted transform and profile are unchanged.
                 This isolates optimizer EXAMPLE COUNT only: the historical
                 batter/pitcher profile summaries in the context were built from the
                 full TRAIN and are retained, and 30 epochs over fewer rows means fewer
                 optimizer steps, so it is not a compute-matched comparison.
  pitcher_information  no_pitcher_profile (ctx 28:51), no_pitcher_support (ctx 51),
                 no_pitcher_all (ctx 28:52): the existing frontier MaskedFlattenModel
                 zeroing NAMED columns (names verified against the export manifest
                 lineage); batter, game state and pitch history are kept.  Tests the
                 dependence on the EXISTING pitcher profile, not a new clustering.
  combination    residual_counts / residual_block: the existing DirectResidualModel
                 plus the first / all four exported workload columns;
                 periodic_counts / periodic_block: the existing DirectPeriodicModel
                 plus workload, the appended columns 52 / 52-55 embedded as numeric
                 (rest_missing included) after the base periodic map.
"""
from __future__ import annotations

import hashlib

import numpy as np

from pitchmdp.direct_frontier_registry import (BASE_CONTEXT_NAMES, PITCHER_CLUSTER_FEATURES, SUPPORT_NAME,
                                               direct_context_names)

PROTOCOL_TAG = 'cuda_data_combo_v1'
WORKLOAD_COLUMNS = ('prior_records_per100', 'thruorder_minus1_over3', 'log_rest_days', 'rest_missing')
N_CONTEXT = 52
PROFILE = (len(BASE_CONTEXT_NAMES), N_CONTEXT - 1)  # 28:51
# Base periodic map of DRAFT-CUDA-DIRECT-20261004 / direct_periodic_model.context_map: 11..51 without left_share (32).
BASE_PERIODIC = tuple(i for i in range(11, N_CONTEXT) if i != PROFILE[0] + len(PITCHER_CLUSTER_FEATURES))
COUNTS, BLOCK = list(WORKLOAD_COLUMNS[:1]), list(WORKLOAD_COLUMNS)


def pitcher_names(cluster_columns):
    """(profile names 28:51, support name 51) from the export's pitcher-profile columns."""
    names = direct_context_names(cluster_columns)
    if len(names) != N_CONTEXT or names[-1] != SUPPORT_NAME:
        raise ValueError('Direct context is not the 52-column layout')
    return names[PROFILE[0]:PROFILE[1]], [SUPPORT_NAME]


def build_arms(cluster_columns):
    """The fixed arm registry (name -> spec without name/priority/combo/workload/train_fraction)."""
    profile, support = pitcher_names(cluster_columns)
    masked = {'family': 'pitcher_information', 'model': 'masked_flatten_mlp', 'width': 128, 'drop_history': False}
    data = {'family': 'data_quantity', 'model': 'flatten_mlp', 'width': 128, 'train_subset': 'nested_key_hash'}
    return {
        'data25': data, 'data50': data, 'data75': data,
        'no_pitcher_profile': {**masked, 'drop_context': profile},
        'no_pitcher_support': {**masked, 'drop_context': support},
        'no_pitcher_all': {**masked, 'drop_context': profile + support},
        'residual_counts': {'family': 'combination', 'model': 'residual_mlp', 'width': 128},
        'residual_block': {'family': 'combination', 'model': 'residual_mlp', 'width': 128},
        'periodic_counts': {'family': 'combination', 'model': 'periodic_flatten_mlp', 'width': 128,
                            'context_continuous': [*BASE_PERIODIC, 52]},
        'periodic_block': {'family': 'combination', 'model': 'periodic_flatten_mlp', 'width': 128,
                           'context_continuous': [*BASE_PERIODIC, 52, 53, 54, 55]},
    }


WORKLOAD = {'data25': [], 'data50': [], 'data75': [], 'no_pitcher_profile': [], 'no_pitcher_support': [],
            'no_pitcher_all': [], 'residual_counts': COUNTS, 'residual_block': BLOCK,
            'periodic_counts': COUNTS, 'periodic_block': BLOCK}
FRACTIONS = {'data25': .25, 'data50': .5, 'data75': .75}
ARM_NAMES = tuple(WORKLOAD)


def check_arm(arm, cluster_columns, subset):
    """An arm is exactly its registry entry (+ name, priority, combo flag, fixed workload, fraction for data arms)."""
    name = arm.get('name')
    arms = build_arms(cluster_columns)
    if arm.get('combo') is not True or name not in arms:
        raise ValueError('Combination arm differs from the fixed registry: ' + str(name))
    rest = {k: v for k, v in arm.items() if k not in ('name', 'priority', 'combo', 'workload', 'train_fraction')}
    if rest != arms[name] or arm.get('workload') != WORKLOAD[name] or arm.get('priority') not in (1, 2):
        raise ValueError('Combination arm differs from the fixed registry: ' + name)
    if name in FRACTIONS:
        if arm.get('train_fraction') != FRACTIONS[name] or subset.get('fractions', {}).get(name) != FRACTIONS[name]:
            raise ValueError(name + ': train fraction differs from the config train_subset')
    elif 'train_fraction' in arm:
        raise ValueError(name + ': only data arms subsample TRAIN')
    return arm


def check_subset(subset):
    """Config-owned subset rule: method, salt and the three fractions."""
    if subset.get('method') != 'nested_key_hash' or subset.get('hash') != 'blake2b-64 keyed by salt over int64 keys row':
        raise ValueError('Unknown TRAIN subset method')
    salt = subset.get('salt')
    if not isinstance(salt, str) or not salt or len(salt.encode()) > 64:
        raise ValueError('A fixed non-empty salt (<= 64 bytes) is required')
    if subset.get('fractions') != FRACTIONS or subset.get('split') != 'train':
        raise ValueError('Subset fractions or split differ from the registry')
    return subset


def row_uniforms(keys, salt):
    """u in [0, 1) per row from the int64 keys and the salt only (portable, order preserving)."""
    keys = np.ascontiguousarray(keys, dtype=np.int64)
    if keys.ndim != 2 or not len(keys):
        raise ValueError('Expected nonempty 2-D int64 keys')
    key = salt.encode()
    width = keys.shape[1] * 8
    raw = keys.astype('<i8', copy=False).tobytes()
    out = np.empty(len(keys), dtype=np.uint64)
    for i in range(len(keys)):
        out[i] = int.from_bytes(hashlib.blake2b(raw[i * width:(i + 1) * width], digest_size=8, key=key).digest(), 'little')
    return out.astype(np.float64) / 2. ** 64


def subset_mask(keys, salt, fraction):
    if not 0 < fraction < 1:
        raise ValueError('Fraction must be in (0, 1)')
    if len(np.unique(np.ascontiguousarray(keys, dtype=np.int64), axis=0)) != len(keys):
        raise ValueError('TRAIN keys must be unique for a key-hash subset')
    return row_uniforms(keys, salt) < fraction


def rows_sha256(keys):
    return hashlib.sha256(np.ascontiguousarray(keys, dtype=np.int64).tobytes()).hexdigest()


def subset_split(arrays, salt, fraction):
    """(arrays restricted to the kept TRAIN rows in their original order, provenance)."""
    keys = np.asarray(arrays['keys'])
    mask = subset_mask(keys, salt, fraction)
    if not mask.any():
        raise ValueError('Empty TRAIN subset')
    index = np.flatnonzero(mask)
    kept = {name: np.asarray(value)[index] for name, value in arrays.items()}
    record = {'method': 'nested_key_hash', 'salt': salt, 'fraction': fraction, 'n_source': int(len(keys)),
              'n_selected': int(len(index)), 'source_rows_sha256': rows_sha256(keys),
              'selected_rows_sha256': rows_sha256(kept['keys']),
              'selected_mask_sha256': hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
              'depends_on': 'keys and salt only (no labels, model seed, DEV or outcomes)'}
    return kept, record
