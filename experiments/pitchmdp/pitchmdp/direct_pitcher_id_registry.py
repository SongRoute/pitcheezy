"""Torch-free registry of the DRAFT pitcher-ID family (DRAFT-CUDA-PITCHER-ID-20261005, tentative EXP-P38-001).
Pre-registration: docs/PREREG-DRAFT-EXP-P38-001-pitcher-id-2026-10-05.md, written 2026-10-05 after the P36 report and
before any P38 fit.  Designed after P24-P36 scores were read on repeated DEV: post-selection, exploratory and
descriptive only (no promotion, verdict, ranking, winner, selection or 2026).  Fresh MODEL RNG seeds are NOT new data.
Question: does a TRAIN-only pitcher-ID embedding (export column `pitcher`, never read by a direct head before) added
to the frozen periodic_residual_counts arm lower calibrated direct NLL / Brier beyond the pitcher profile / support
context the arm already reads?
  periodic_residual_counts             EXP-P36-001 entry verbatim (reference; embed_residual flag)
  pitcher_id_periodic_residual_counts  same entry + `pitcher_id` block (candidate; direct_pitcher_id_model)
Two arms only, NO flatten control (the reference is the comparator; the runner's control requirement is waived for
this marker family only).  2 arms x seeds exactly [44, 45, 46, 47, 48] = 10 fits, cuda_direct_v1 budget with the
epoch cap 60 (as P34-P37).  Seeds 44-48 are disjoint from every registered block 0-38 and the P37 DRAFT 39-43.
Fixed contrast (candidate - reference): pitcher_id_vs_periodic_residual_counts (primary, only one), reported on all
rows and on the in-vocabulary / UNK row subsets, every seed and the seed-mean ensemble.  NOT parameter-matched and
NOT compute-matched: the candidate adds (n_vocab + 1) x D_ID + WIDTH x D_ID parameters (n_vocab is measured from
TRAIN at run time, recorded per fit and checked by the scorer).  A capacity-matched information-free control is NOT
INCLUDED in this bounded two-arm study; that limits causal attribution of any difference to identity information vs
added capacity (not a claim that such a control is impossible).
Vocabulary: TRAIN `pitcher` rows only (labels never read), dates <= LAST_TRAIN_DATE, ids with >= MIN_TRAIN_PITCHES
TRAIN rows sorted ascending; everything else (rare TRAIN pitchers, ids first seen after TRAIN) is UNK 0.
Hardware (Song 2026-10-05 single-GPU rule): GPU3 UUID only, one fit at a time, GPU4 refused (check_device).
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from pitchmdp import direct_embed_type_registry as et

MARKER = 'pitcher_id_family'
FLAG = 'pitcher_id'
CONTRAST_KEY = 'pitcher_id_contrasts'
SEEDS = [44, 45, 46, 47, 48]
EARLIER_SEEDS = {**et.EARLIER_SEEDS, 'EXP-P37-001 (DRAFT)': list(et.SEEDS)}
EPOCH_CAP = et.EPOCH_CAP
TRAINING = dict(et.TRAINING)  # cuda_direct_v1 budget with the epoch cap 60 (== P34-P37)
MIN_TRAIN_PITCHES = 100
D_ID = 8
UNK = 0
WIDTH = 128  # backbone width of the frozen periodic_residual_counts network (id projection rows)
LAST_TRAIN_DATE = 20250430  # TRAIN ends 2025-04-30; earlystop 05-01..15, temperature 05-16..31 (frozen splits)
PITCHER_ID = {'min_train_pitches': MIN_TRAIN_PITCHES, 'd_id': D_ID, 'unk_index': UNK, 'vocab_source': 'train',
              'last_train_date': LAST_TRAIN_DATE}
REFERENCE = 'periodic_residual_counts'
CANDIDATE = 'pitcher_id_periodic_residual_counts'
BASELINES = {
    REFERENCE: et.BASELINES[REFERENCE],
    CANDIDATE: {**et.BASELINES[REFERENCE], 'name': CANDIDATE, 'family': 'pitcher_id', FLAG: dict(PITCHER_ID)},
}
ARM_ORDER = (REFERENCE, CANDIDATE)
FITS = len(ARM_ORDER) * len(SEEDS)
BASE_PARAMETERS = et.PARAMETERS[REFERENCE]  # 244898 at the registered shapes (tested from the frozen network)
CONTRASTS = [
    {'name': 'pitcher_id_vs_periodic_residual_counts', 'candidate': CANDIDATE, 'reference': REFERENCE,
     'isolates': 'pitcher_identity_with_id_parameters', 'role': 'primary', 'parameter_matched': False,
     'compute_matched': False, 'information_free_capacity_control': 'not included in this bounded study'},
]
PRIMARY = CONTRASTS[0]['name']
SUBSETS = ('all', 'in_vocab', 'unk')
VOCAB_SPLITS = ('train', 'earlystop', 'temperature', 'blend', 'dev', 'mlb_dev')
DEVICE = et.DEVICE
RETAINED_GPU, RELEASED_GPUS = et.RETAINED_GPU, et.RELEASED_GPUS
# Pre-specified TRAIN-only resource gate (prereg section 4, fixed before any measurement): on the 10-task TRAIN smoke,
# median over the 5 seeds of report.seconds / report.epochs_run, candidate / reference <= RESOURCE_RATIO_MAX.  The ID
# path adds one table gather and one 8 -> 128 projection per row (no per-scalar work), so a larger ratio means a
# wiring or device problem, not a property worth paying GPU3 time for.  The P37 linear/periodic 1.5 gate is NOT reused.
RESOURCE_RATIO_MAX = 1.25
P36_PERIODIC_MAX_TASK_SECONDS = 235.20  # P36 actual per-task wall max of periodic_residual_counts (extract-p36 ce2861b)
SAFETY = 2.0
TASK_BOUND_SECONDS = {REFERENCE: SAFETY * P36_PERIODIC_MAX_TASK_SECONDS,
                      CANDIDATE: RESOURCE_RATIO_MAX * SAFETY * P36_PERIODIC_MAX_TASK_SECONDS}
FAMILY_PROJECTION_SECONDS = len(SEEDS) * sum(TASK_BOUND_SECONDS.values())  # serial, one GPU: 5,292.0 s
# ------------------------------------------------------------------ P40 controls (tentative EXP-P40-001 DRAFT)
# Pre-registration docs/PREREG-DRAFT-EXP-P40-001-pitcher-id-controls-2026-10-06.md, written 2026-10-06 00:50 KST after
# the P38 score (post-selection, exploratory, repeated DEV) and before any P40 output.  Four arms, fresh seeds 60-64:
#   periodic_residual_counts                          reference (P36/P38 entry verbatim)
#   pitcher_id_periodic_residual_counts               P38 candidate verbatim (replication on fresh seeds)
#   pitcher_id_row_shuffle_periodic_residual_counts   same network, parameters and compute as the candidate; the
#       pitcher column is permuted ACROSS ROWS within every split by a fixed seed-independent permutation before the
#       vocabulary lookup, so index frequencies / UNK share are kept and the row -> pitcher association is destroyed
#       (information-free, parameter- and compute-matched control of the id path)
#   pitcher_id_dropout_periodic_residual_counts       candidate + ID dropout: during training each row's index is
#       replaced by UNK with probability 0.2 (private device generator; eval uses the true index), a fixed
#       cold-start / shrinkage regulariser (DropoutNet-style input dropout), not tuned
DESIGN_KEY = 'pitcher_id_design'
P40_DESIGN = 'p40_controls_v1'
SHUFFLE = 'pitcher_id_row_shuffle_periodic_residual_counts'
DROPOUT = 'pitcher_id_dropout_periodic_residual_counts'
SHUFFLE_SEED = 40_001
ID_DROPOUT = 0.2
P40_SEEDS = [60, 61, 62, 63, 64]
P40_EARLIER_SEEDS = {**EARLIER_SEEDS, 'EXP-P38-001': list(SEEDS), 'EXP-P39-001 (MPS)': [49, 50, 51],
                     'reserved for EXP-P41 (MPS sibling, 2026-10-06)': [52, 53, 54, 55, 56, 57, 58, 59]}
BASELINES[SHUFFLE] = {**BASELINES[CANDIDATE], 'name': SHUFFLE,
                      FLAG: {**PITCHER_ID, 'index_source': 'row_shuffle', 'shuffle_seed': SHUFFLE_SEED}}
BASELINES[DROPOUT] = {**BASELINES[CANDIDATE], 'name': DROPOUT, FLAG: {**PITCHER_ID, 'id_dropout': ID_DROPOUT}}
ID_ARMS = (CANDIDATE, SHUFFLE, DROPOUT)
P40_ARM_ORDER = (REFERENCE, CANDIDATE, SHUFFLE, DROPOUT)
P40_FITS = len(P40_ARM_ORDER) * len(P40_SEEDS)
P40_CONTRASTS = [
    {'name': 'pitcher_id_vs_row_shuffle', 'candidate': CANDIDATE, 'reference': SHUFFLE,
     'isolates': 'pitcher_identity_information', 'role': 'primary', 'parameter_matched': True,
     'compute_matched': True, 'information_free_capacity_control': 'this contrast (reference = row-shuffled ids)'},
    {'name': 'pitcher_id_vs_periodic_residual_counts', 'candidate': CANDIDATE, 'reference': REFERENCE,
     'isolates': 'pitcher_identity_with_id_parameters', 'role': 'replication_of_p38', 'parameter_matched': False,
     'compute_matched': False, 'information_free_capacity_control': 'see pitcher_id_vs_row_shuffle'},
    {'name': 'row_shuffle_vs_periodic_residual_counts', 'candidate': SHUFFLE, 'reference': REFERENCE,
     'isolates': 'id_parameters_without_information', 'role': 'capacity_control', 'parameter_matched': False,
     'compute_matched': False, 'information_free_capacity_control': 'candidate side is information-free'},
    {'name': 'id_dropout_vs_pitcher_id', 'candidate': DROPOUT, 'reference': CANDIDATE,
     'isolates': 'id_dropout_regularisation', 'role': 'secondary', 'parameter_matched': True,
     'compute_matched': True, 'information_free_capacity_control': 'not applicable (both arms read ids)'},
]
P40_PRIMARY = P40_CONTRASTS[0]['name']
P38_MAX_TASK_SECONDS = 244.70  # P38 actual per-task wall max (runs/EXP-P38-001 fit.json seconds_total, ref seed 48)
P40_SAFETY = 1.25
P40_TASK_BOUND_SECONDS = P40_SAFETY * P38_MAX_TASK_SECONDS
P40_FAMILY_PROJECTION_SECONDS = P40_FITS * P40_TASK_BOUND_SECONDS  # serial, one GPU: 6,117.5 s
DESIGNS = {
    'p38': {'arms': ARM_ORDER, 'seeds': SEEDS, 'fits': FITS, 'contrasts': CONTRASTS, 'primary': PRIMARY,
            'earlier_seeds': EARLIER_SEEDS, 'projection_seconds': FAMILY_PROJECTION_SECONDS},
    P40_DESIGN: {'arms': P40_ARM_ORDER, 'seeds': P40_SEEDS, 'fits': P40_FITS, 'contrasts': P40_CONTRASTS,
                 'primary': P40_PRIMARY, 'earlier_seeds': P40_EARLIER_SEEDS,
                 'projection_seconds': P40_FAMILY_PROJECTION_SECONDS},
}
VOCAB_KEYS = ('n_vocab', 'ids_sha256', 'min_count', 'train_rows', 'train_pitchers', 'train_unk_rows',
              'train_max_date', 'last_train_date')


def same(value, expected):
    return et.same(value, expected)


def is_family(config):
    return MARKER in config


def is_arm(arm):
    return FLAG in arm


def id_parameters(n_vocab):
    """Extra allocated parameters of the candidate: id table (n_vocab + 1) x D_ID + projection WIDTH x D_ID."""
    return (int(n_vocab) + 1) * D_ID + WIDTH * D_ID


def check_arm(arm):
    """A pitcher-ID arm is exactly one registered ID entry (P38 candidate or a P40 control; old flags + fixed block)."""
    name = arm.get('name')
    if name not in ID_ARMS or not same(arm, BASELINES[name]):
        raise ValueError('Pitcher-ID arm differs from the fixed registry entry: ' + str(name))
    return arm


def id_spec(arm):
    """(index_source, shuffle_seed, id_dropout) of a checked pitcher-ID arm; P38 candidate = ('identity', None, 0.)."""
    block = check_arm(arm)[FLAG]
    return block.get('index_source', 'identity'), block.get('shuffle_seed'), float(block.get('id_dropout', 0.))


def check_device(config):
    device = config.get('device') or {}
    if any(g in RELEASED_GPUS for g in device.get('allowed_gpu_uuids') or []):
        raise ValueError('Pitcher-ID family must not allow a released GPU (GPU4 belongs to another project)')
    if not same(device, DEVICE):
        raise ValueError('Pitcher-ID device must be exactly one GPU (GPU3 UUID) with max_parallel 1, no fallback')
    return device


def design(config):
    """The fixed design of a pitcher-ID config: P38 (no design key) or P40 controls (design key == P40_DESIGN)."""
    key = config.get(DESIGN_KEY)
    if key is None:
        return DESIGNS['p38']
    if key != P40_DESIGN:
        raise ValueError(DESIGN_KEY + ' must be absent (P38) or exactly ' + P40_DESIGN)
    return DESIGNS[P40_DESIGN]


def check_family(config):
    """Marker true, exactly the design's arms in order and seeds (P38: 2 arms 44-48; P40: 4 arms 60-64), cap 60, GPU3."""
    if config.get(MARKER) is not True:
        raise ValueError(MARKER + ' must be exactly true')
    d = design(config)
    if [a.get('name') for a in config['arms']] != list(d['arms']):
        raise ValueError('Pitcher-ID family must be exactly ' + ', '.join(d['arms']))
    for arm in config['arms']:
        if not same(arm, BASELINES[arm['name']]):
            raise ValueError(arm['name'] + ': arm differs from its fixed registry entry')
    if 'repeat_tag' in config or not same(config.get('seeds'), d['seeds']):
        raise ValueError(f"Pitcher-ID seeds must be exactly {d['seeds']} with no repeat_tag")
    if set(d['seeds']) & {s for block in d['earlier_seeds'].values() for s in block}:
        raise ValueError('Pitcher-ID seeds overlap an earlier seed block')
    if not same(config.get('fits'), d['fits']):
        raise ValueError(f"Pitcher-ID family is exactly {d['fits']} fits")
    check_device(config)
    if not same(config.get('training'), TRAINING):
        raise ValueError(f'Pitcher-ID training must be the cuda_direct_v1 budget with epochs {EPOCH_CAP}')
    return config


def check_contrasts(config):
    """comparison.pitcher_id_contrasts (or None): exactly the fixed list; only inside the marker family."""
    contrasts = config.get('comparison', {}).get(CONTRAST_KEY)
    if contrasts is None:
        return None
    if not same(contrasts, design(config)['contrasts']):
        raise ValueError(CONTRAST_KEY + ' differ from the fixed pre-registered contrasts')
    check_family(config)
    return contrasts


# ------------------------------------------------------------------ TRAIN-only vocabulary (numpy only)

def build_vocab(train_pitcher, train_date=None, last_train_date=None, min_count=MIN_TRAIN_PITCHES):
    """TRAIN-only vocabulary: sorted pitcher ids with >= min_count TRAIN rows (labels never read)."""
    pitcher = np.asarray(train_pitcher, dtype=np.int64)
    if pitcher.ndim != 1 or not len(pitcher):
        raise ValueError('Nonempty one-dimensional TRAIN pitcher ids required')
    max_date = None
    if last_train_date is not None:
        date = np.asarray(train_date, dtype=np.int64)
        if date.shape != pitcher.shape:
            raise ValueError('TRAIN dates must align with TRAIN pitcher ids')
        max_date = int(date.max())
        if max_date > int(last_train_date):
            raise ValueError('A row after the TRAIN cutoff reached the vocabulary')
    ids, counts = np.unique(pitcher, return_counts=True)
    keep = counts >= int(min_count)
    ids = ids[keep]
    return {'ids': ids, 'min_count': int(min_count), 'n_vocab': int(len(ids)),
            'ids_sha256': hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest(),
            'train_rows': int(len(pitcher)), 'train_pitchers': int(len(counts)),
            'train_unk_rows': int(counts[~keep].sum()), 'train_max_date': max_date,
            'last_train_date': None if last_train_date is None else int(last_train_date)}


def map_ids(pitcher, ids):
    """1-based vocabulary index, UNK (0) for ids outside the TRAIN vocabulary."""
    pitcher = np.asarray(pitcher, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) and np.any(np.diff(ids) <= 0):
        raise ValueError('Vocabulary ids must be strictly increasing')
    if not len(ids):
        return np.zeros(len(pitcher), dtype=np.int64)
    pos = np.searchsorted(ids, pitcher)
    hit = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == pitcher)
    return np.where(hit, pos + 1, UNK).astype(np.int64)


def vocab_record(vocab):
    """The fit.json record of a vocabulary (ids themselves stay in the model archive; hash here)."""
    return {k: vocab[k] for k in VOCAB_KEYS}


def check_split_dates(manifest, last_train_date=LAST_TRAIN_DATE):
    """Export pin: TRAIN ends on/before the cutoff and every later split starts after it (no future rows in vocab)."""
    splits = manifest['splits']
    if splits['train'].get('max_date') is None or int(splits['train']['max_date']) > int(last_train_date):
        raise ValueError('Export TRAIN max_date is after the pitcher-ID TRAIN cutoff')
    for name in VOCAB_SPLITS[1:]:
        if splits[name].get('min_date') is None or int(splits[name]['min_date']) <= int(last_train_date):
            raise ValueError(f'{name}: starts on/before the TRAIN cutoff; vocabulary split order broken')
    return {name: [splits[name].get('min_date'), splits[name].get('max_date')] for name in VOCAB_SPLITS}


def check_fit(config, arm, fit):
    """Per-fit record: vocabulary (candidate only), allocated parameters and cost; fail closed."""
    record = et.cost_record(fit)
    vocab = fit.get(FLAG)
    if arm not in design(config)['arms']:
        raise ValueError(f'{arm}: not an arm of this pitcher-ID design')
    if arm == REFERENCE:
        if vocab is not None:
            raise ValueError(f'{arm}: reference fit carries a pitcher-ID record')
        expected = BASE_PARAMETERS
    else:
        if not isinstance(vocab, dict) or set(VOCAB_KEYS) - set(vocab) or vocab.get('min_count') != MIN_TRAIN_PITCHES \
                or vocab.get('last_train_date') != LAST_TRAIN_DATE or vocab.get('train_max_date') is None \
                or int(vocab['train_max_date']) > LAST_TRAIN_DATE or int(vocab.get('n_vocab', 0)) < 1:
            raise ValueError(f'{arm}: pitcher-ID vocabulary record missing or malformed')
        shares = vocab.get('unk_share') or {}
        if set(shares) != set(VOCAB_SPLITS) or not all(isinstance(v, float) and math.isfinite(v) and 0 <= v <= 1
                                                       for v in shares.values()):
            raise ValueError(f'{arm}: per-split UNK shares missing or malformed')
        expected = BASE_PARAMETERS + id_parameters(vocab['n_vocab'])
    if record['parameter_count'] != expected:
        raise ValueError(f'{arm}: parameter_count {record["parameter_count"]} != registered {expected}')
    return {'cost': record, 'vocab': None if vocab is None else {**{k: vocab[k] for k in VOCAB_KEYS},
                                                                    'unk_share': dict(vocab['unk_share'])}}


def check_one_vocab(tasks):
    """All candidate fits carry one vocabulary record (TRAIN-only, seed-independent); the shared record."""
    records = [t['vocab'] for arm in tasks if arm != REFERENCE for t in tasks[arm].values()]
    if not records or len({json.dumps(r, sort_keys=True) for r in records}) != 1:
        raise ValueError('Pitcher-ID fits of one family must carry one vocabulary record')
    return {**records[0], 'n_fits': len(records), 'id_parameters': id_parameters(records[0]['n_vocab'])}
