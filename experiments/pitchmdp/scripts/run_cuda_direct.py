"""DRAFT portable CUDA runner for the direct pre-pitch head (DRAFT-CUDA-DIRECT-20261004).

Linux/newsrv side.  Consumes ONE immutable numeric export written on the Mac by
export_cuda_direct.py (frozen EXP-P4-001/P15 loaders, already TRAIN-normalized
arrays, workload scalars, keys, labels, SHA256 manifest).  No parquet, pandas or
raw Statcast is read here.  Per task (arm, seed):

  fit      MatrixModel flatten_mlp 128 (or the frozen residual/periodic network),
           30 epochs, patience 5, lr 5e-4, batch 1024, uniform loss, early stop on
           the May 1-15 rows (same loop as EXP-P15-001)
  calibrate per-seed temperature on the May 16-31 Cpanel rows, minimize_scalar
           bounded (.5, 2.5) over the G0 SharingPredictor chain
  predict  calibrated/raw probabilities for blend (June), dev (Cpanel), mlb_dev in
           the Mac member npz layout; June blend and scoring stay on the Mac

Frontier arms (next batch, DRAFT-CUDA-FRONTIER-RUN-20261004): an arm marked
"frontier": true must equal the fixed direct_frontier_registry.ARMS entry of its
name (capacity / information / regularization; no workload, one change vs the
flatten 128 control).  Information arms zero NAMED context columns, so the 52
names are derived from the config's expected layout and must match the export
manifest's recorded context_layout before the model is built.  Base arms keep the
cuda_direct_v1 checks unchanged.

Combination arms (DRAFT-CUDA-DATA-COMBO-RUN-20261004): an arm marked "combo": true
must equal direct_combo_registry's entry.  Data arms fit on a nested key-hash subset
of the TRAIN rows only (config train_subset salt/fractions; provenance in fit.json);
pitcher-information arms zero verified named context columns; residual/periodic
combination arms append workload like the base counts/block arms.

Dropout arms (DRAFT-CUDA-DROPOUT-RUN-20261004): an arm marked "dropout_head": true
must equal direct_dropout_registry.ARMS (frozen residual / periodic network plus
head Dropout(.1), no workload); a config with such arms must be exactly the fixed
six-arm architecture x dropout family.

Workload-dropout arms (DRAFT-CUDA-WORKLOAD-DROPOUT-20261004): an arm marked
"workload_dropout": true must equal direct_workload_dropout_registry.ARMS (frozen
periodic network + head Dropout(.1/.2), P27 workload and continuous map); a config
with such arms must be exactly the fixed ten-arm family, and only that family may
use its registered fresh seeds ([3..7], or [8..12] with the explicit repeat_tag).
Every other config keeps seeds 0-2 unchanged.

TabM-mini arms (DRAFT-CUDA-TABM-MINI-20261004): an arm marked "tabm": true must equal
direct_tabm_registry.ARMS (frozen periodic network + TabM-mini k=8 head, counts
workload); a config with such arms must be exactly the fixed five-arm family
(control and the single-network arms are the exact earlier P27/P30 entries) on
seeds exactly [13, 14, 15].  The family check runs before the workload-dropout one
because periodic_counts_d10 keeps its P30 flag.

Embedded-residual arms (DRAFT-CUDA-EMBED-RESIDUAL-20261004): an arm marked
"embed_residual": true must equal direct_embed_residual_registry.ARMS (frozen PLR-lite
or LR embedding in front of the frozen-architecture residual backbone, counts
workload); a config with such arms must be exactly the fixed six-arm family (control,
residual_counts, periodic_counts and periodic_counts_d10 are the exact earlier entries)
on seeds exactly [16, 17, 18].  Its family check also runs before the workload-dropout
one.  After load the archived continuous map must equal the registered one.

EMA weight-averaging arms (DRAFT-CUDA-EMA-20261004): an arm marked "ema": true must
equal direct_ema_registry.ARMS (frontier flatten network, dropout 0 or .1, EMA decay
.999 for early stop, best state and saved model); a config with such arms must be
exactly the fixed four-arm family (control and the frontier dropout arm exact) on
seeds exactly [19, 20, 21, 22, 23], no workload.  Its family check runs first.

TabM-residual arms (DRAFT-CUDA-TABM-RESIDUAL-20261004): an arm marked "tabm_residual":
true must equal direct_tabm_residual_registry.ARMS (TabM-mini k=8 on the frozen PLR-lite
+ residual backbone, counts workload); a config with such arms must be exactly the fixed
five-arm family (control, periodic_residual_counts = EXP-P32-001, tabm_counts and
tabm_counts_d10 = EXP-P31-001 exact entries) on seeds exactly [24, 25, 26, 27, 28] with the pre-registered
epoch cap 60 (every other training entry unchanged).  Its family check runs before the
EMA / TabM / embedded-residual ones because two of its arms keep those flags.  fit.json
of this family also records the epoch cap passed to fit, epochs run, best epoch and
whether the cap or patience stopped the fit.

PLE-embedding arms (DRAFT-CUDA-PLE-RESIDUAL-20261004, EXP-P35-001 DRAFT): an arm marked
"ple_residual": true or "ple_tabm_residual": true must equal direct_ple_registry.ARMS
(version-B piecewise-linear embedding, quantile bins from the TRAIN argument only, in front
of the frozen residual backbone, single head or TabM-mini k=8; counts workload); a config
with such arms must be exactly the fixed five-arm family (control, periodic_residual_counts
and tabm_residual_counts = EXP-P34-001 exact entries) on seeds exactly [29, 30, 31, 32, 33]
with the epoch cap 60.  Its family predicate is tested FIRST (config_check, training_budget,
run_task, scorer), because tabm_residual_counts keeps its P34 flag.  fit.json records the
epoch cap as P34 and, for PLE arms, the bins hash, per-feature bin counts and the degenerate
(one-valued, T_j = 0) columns by export column name, and ple_effective_parameter_count next to
the allocated parameter_count, and ple_bin_rows (TRAIN-argument rows and per-past-position
valid rows the bins were fitted on, also printed); load_model checks the bins against that recorded hash.

Robust-clip arms (DRAFT-CUDA-ROBUST-CLIP-20261005, tentative EXP-P36-001 DRAFT): an arm marked
"robust_clip_residual": true must equal direct_robust_clip_registry.ARMS (RealMLP robust scale +
smooth clip, statistics from the TRAIN argument only, in front of the frozen residual backbone;
counts workload); a config with such arms must be exactly the fixed four-arm family (control,
residual_counts = EXP-P32-001 combo entry, periodic_residual_counts = EXP-P32/P34/P35 entry) on
seeds exactly [34, 35, 36, 37, 38] with the epoch cap 60, and direct_robust_clip_model.py must be
the registry-pinned file.  Its family predicate is tested FIRST (config_check, training_budget,
run_task, scorer), because periodic_residual_counts keeps its embed_residual flag.  fit.json
records the epoch cap as P34/P35 and, for the robust-clip arm, rc_stats_sha256 and rc_rule_counts
(re-derived from the fit report), rc_stats_rows (TRAIN-argument rows and per-past-position valid
rows the statistics were computed on, also printed) and rc_module_sha256; load_model checks the
archived statistics against that recorded hash (statistics are never recomputed at predict time).

Embedding-map control (DRAFT-CUDA-EMBED-TYPE-20261005, tentative EXP-P37-001 DRAFT): a config with the
top-level marker "embed_type_family": true must be exactly the fixed three-arm family (control,
periodic_residual_counts and linear_residual_counts = the EXP-P32-001 entries, no new arm or model code) on
seeds exactly [39, 40, 41, 42, 43] with the epoch cap 60 (direct_embed_type_registry).  Its predicate is
tested FIRST (config_check, training_budget, scorer) because both embedded arms keep the embed_residual flag;
run_task, build and load are the unchanged embedded-residual paths.  fit.json records the epoch cap as P34-P36.

Pitcher-ID family (DRAFT-CUDA-PITCHER-ID-20261005, tentative EXP-P38-001 DRAFT): top-level marker "pitcher_id_family"
true = exactly periodic_residual_counts (EXP-P36-001 entry) + pitcher_id_periodic_residual_counts on seeds 44-48, cap
60, GPU3 only (direct_pitcher_id_registry); tested before every other family (both arms keep embed_residual); no
flatten control (the reference arm is the comparator; the control requirement is waived for this marker only).  The
candidate's vocabulary is built in run_task from the loaded TRAIN rows' `pitcher` (labels never read; TRAIN dates <=
20250430 and later splits starting after it, checked on the manifest and the rows), ids with >= 100 TRAIN rows, UNK 0
otherwise; every split the model sees (TRAIN, earlystop, temperature, blend, dev, mlb_dev) gets the same index column
via id_inputs; fit.json records the vocabulary (n_vocab, ids sha, counts, cutoff, per-split UNK share); load checks
the archived vocabulary against that hash.  The reference arm runs the unchanged embedded-residual path.
P43 batter-ID design (pitcher_id_design p43_batter_id_v1, tentative EXP-P43-001 DRAFT): the same family with batter
arms; the `batter` column comes from the hash-pinned sidecar named by config.batter_sidecar (a sibling directory of
the export, verified in verify_export and attached to every loaded split by with_ids); vocabularies are built per id
column from TRAIN rows only; fit.json records {id_columns, vocabs, index_source, shuffle_seed}.

Workload arms append their scalars after the 52 stripped context columns, i.e.
before the 7 sharing columns training_arrays strips (run_ml_direct_workload
WorkloadContext): routing never enters the network.  Current physics is zero in
every exported token (checked).  The GPU is bound by CUDA_VISIBLE_DEVICES to one
allowed UUID BEFORE torch is imported; no CPU or other-GPU fallback.  Outputs are
written to a temporary directory and renamed; an existing task directory is never
overwritten, and a failed attempt is preserved under failed/.

  python scripts/run_cuda_direct.py --config ../../configs/DRAFT-CUDA-DIRECT-20261004.json \
      --export <export dir> --export-manifest-sha256 <sha> --output <family dir> \
      --gpu-uuid GPU-... task --arm control --seed 0 [--smoke]
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np

PROTOCOL = 'cuda_direct_v1'
SELF = 'scripts/run_cuda_direct.py'
MODEL_SOURCES = ['pitchmdp/matrix_models.py', 'pitchmdp/direct_residual_model.py', 'pitchmdp/direct_periodic_model.py',
                 'pitchmdp/direct_combo_model.py', 'pitchmdp/direct_combo_registry.py',
                 'pitchmdp/direct_dropout_model.py', 'pitchmdp/direct_dropout_registry.py',
                 'pitchmdp/direct_workload_dropout_model.py', 'pitchmdp/direct_workload_dropout_registry.py',
                 'pitchmdp/direct_tabm_model.py', 'pitchmdp/direct_tabm_factory.py', 'pitchmdp/direct_tabm_registry.py',
                 'pitchmdp/direct_periodic_residual_model.py', 'pitchmdp/direct_embed_residual_factory.py',
                 'pitchmdp/direct_embed_residual_registry.py',
                 'pitchmdp/direct_ema_model.py', 'pitchmdp/direct_ema_registry.py',
                 'pitchmdp/direct_tabm_residual_model.py', 'pitchmdp/direct_tabm_residual_factory.py',
                 'pitchmdp/direct_tabm_residual_registry.py',
                 'pitchmdp/direct_ple_model.py', 'pitchmdp/direct_ple_factory.py', 'pitchmdp/direct_ple_registry.py',
                 'pitchmdp/direct_robust_clip_model.py', 'pitchmdp/direct_robust_clip_factory.py',
                 'pitchmdp/direct_robust_clip_registry.py', 'pitchmdp/direct_embed_type_registry.py',
                 'pitchmdp/direct_pitcher_id_model.py', 'pitchmdp/direct_pitcher_id_registry.py',
                 'pitchmdp/direct_pitcher_batter_id_model.py', 'pitchmdp/batter_sidecar.py',
                 'pitchmdp/direct_frontier_model.py', 'pitchmdp/direct_frontier_registry.py']
ARRAYS = ('tokens', 'valid', 'context', 'workload', 'keys', 'y', 'game_pk', 'pitcher', 'date')
SPLITS = ('train', 'earlystop', 'temperature', 'blend', 'dev', 'mlb_dev')
TRAINING = {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': 0.0005,
            'loss_weighting': 'uniform', 'hpo_trials': 0}
BOUNDS = (.5, 2.5)
NO_PHYSICS_LEVEL = -2
N_PHYSICAL = 8
N_OUTCOME = 11  # current-token outcome channels (last 11; MatrixHistoryStore.gather keeps them zero)
SMOKE = {'train': 8192, 'earlystop': 2048, 'temperature': 256, 'predict': 256, 'epochs': 2}
MODELS = ('flatten_mlp', 'residual_mlp', 'periodic_flatten_mlp')
BASE_GROUPED_FEATURES = ['six_batter_style_rates', 'six_style_reliabilities', 'five_soft_memberships']
SHARING_TAIL = 7


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json_atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with open(tmp, 'w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def own_hashes():
    return {rel: sha256_file(PROJECT / rel) for rel in (SELF, *MODEL_SOURCES)}


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def require_executable(config, smoke):
    """Real (non-smoke) fits need a registered, enabled config; DRAFT allows smoke only."""
    if smoke:
        from pitchmdp.direct_pitcher_id_registry import is_p47
        if is_p47(config):
            raise ValueError('P47 legacy --smoke reads DEV payloads; use the TRAIN/earlystop-only P47 probe')
        return
    if config.get('status') != 'REGISTERED' or config.get('execution', {}).get('enabled') is not True:
        raise ValueError('Real CUDA fits require status REGISTERED and execution.enabled true; DRAFT allows smoke only')


def expected_identity(config, manifest_sha256, arm_name, seed, smoke, device, sources=None):
    """What a finished task's identity must equal (sources optional only for callers that pin them separately)."""
    expected = {'protocol': PROTOCOL, 'experiment_id': config['experiment_id'], 'arm': arm_spec(config, arm_name),
                'seed': seed, 'smoke': smoke, 'config_sha256': config_digest(config),
                'export_manifest_sha256': manifest_sha256, 'device': device}
    if sources is not None:
        expected['source_sha256'] = sources
    return expected


# ------------------------------------------------------------------ config / export

def uses_workload_dropout(config):
    return any('workload_dropout' in arm for arm in config['arms'])


def uses_tabm(config):
    return any('tabm' in arm for arm in config['arms'])


def uses_embed_residual(config):
    return any('embed_residual' in arm for arm in config['arms'])


def uses_ema(config):
    return any('ema' in arm for arm in config['arms'])


def uses_tabm_residual(config):
    return any('tabm_residual' in arm for arm in config['arms'])


PLE_FLAGS = ('ple_residual', 'ple_tabm_residual')  # == direct_ple_registry.FLAG_KEYS (tested)


def is_ple_arm(arm):
    return any(flag in arm for flag in PLE_FLAGS)


def uses_ple(config):
    """The P35 family predicate; checked before uses_tabm_residual everywhere (its arm 4 keeps the P34 flag)."""
    return any(is_ple_arm(arm) for arm in config['arms'])


ROBUST_CLIP_FLAG = 'robust_clip_residual'  # == direct_robust_clip_registry.FLAG (tested)


def is_robust_clip_arm(arm):
    return ROBUST_CLIP_FLAG in arm


def uses_robust_clip(config):
    """The P36 DRAFT family predicate; checked first everywhere (its periodic_residual_counts keeps embed_residual)."""
    return any(is_robust_clip_arm(arm) for arm in config['arms'])


EMBED_TYPE_MARKER = 'embed_type_family'  # == direct_embed_type_registry.MARKER (tested)


def uses_embed_type(config):
    """The P37 DRAFT family predicate (top-level marker); checked first everywhere (its arms keep embed_residual)."""
    return EMBED_TYPE_MARKER in config


PITCHER_ID_MARKER = 'pitcher_id_family'  # == direct_pitcher_id_registry.MARKER (tested)
PITCHER_ID_FLAG = 'pitcher_id'  # == direct_pitcher_id_registry.FLAG (tested)


def uses_pitcher_id(config):
    """The P38 DRAFT family predicate (top-level marker); checked before every other family."""
    return PITCHER_ID_MARKER in config


def is_pitcher_id_arm(arm):
    return PITCHER_ID_FLAG in arm


def uses_epoch_cap(config):
    return uses_pitcher_id(config) or uses_embed_type(config) or uses_robust_clip(config) or uses_ple(config) or uses_tabm_residual(config)


def training_budget(config):
    """EXP-P15-001 budget; only the embed-type, robust-clip, PLE and TabM-residual families have their epoch cap."""
    if uses_pitcher_id(config):
        from pitchmdp.direct_pitcher_id_registry import TRAINING as PID_CAPPED  # torch-free
        return PID_CAPPED
    if uses_embed_type(config):
        from pitchmdp.direct_embed_type_registry import TRAINING as ET_CAPPED  # torch-free
        return ET_CAPPED
    if uses_robust_clip(config):
        from pitchmdp.direct_robust_clip_registry import TRAINING as RC_CAPPED  # torch-free
        return RC_CAPPED
    if uses_ple(config):
        from pitchmdp.direct_ple_registry import TRAINING as PLE_CAPPED  # torch-free
        return PLE_CAPPED
    if uses_tabm_residual(config):
        from pitchmdp.direct_tabm_residual_registry import TRAINING as CAPPED  # torch-free
        return CAPPED
    return TRAINING


def config_check(config):
    if config.get('protocol') != PROTOCOL:
        raise ValueError('Config is not the cuda_direct_v1 protocol')
    if any(is_pitcher_id_arm(arm) for arm in config['arms']) and not uses_pitcher_id(config):
        raise ValueError('Pitcher-ID arms belong to the pitcher-ID marker family only')
    if uses_pitcher_id(config):
        from pitchmdp.direct_pitcher_id_registry import check_family  # torch-free; 2 arms, seeds 44-48, cap 60, GPU3
        check_family(config)
    elif uses_embed_type(config):
        from pitchmdp.direct_embed_type_registry import check_family  # torch-free; exact P32 arms, seeds 39-43, cap 60
        check_family(config)
    elif uses_robust_clip(config):
        from pitchmdp.direct_robust_clip_registry import check_family  # torch-free; exact arms, seeds 34-38, pin
        check_family(config)
    elif uses_ple(config):
        from pitchmdp.direct_ple_registry import check_family  # torch-free; exact arms, seeds 29-33, cap 60
        check_family(config)
    elif uses_tabm_residual(config):
        from pitchmdp.direct_tabm_residual_registry import check_family  # torch-free; exact arms, seeds 24-28, cap 60
        check_family(config)
    elif uses_ema(config):
        from pitchmdp.direct_ema_registry import check_family  # torch-free; exact arms + seeds 19-23
        check_family(config)
    elif uses_tabm(config):
        from pitchmdp.direct_tabm_registry import check_family  # torch-free; exact arms + seeds 13-15
        check_family(config)
    elif uses_embed_residual(config):
        from pitchmdp.direct_embed_residual_registry import check_family  # torch-free; exact arms + seeds 16-18
        check_family(config)
    elif uses_workload_dropout(config):
        from pitchmdp.direct_workload_dropout_registry import check_family  # torch-free; exact arms + fresh seeds
        check_family(config)
    elif config.get('seeds') != [0, 1, 2] or 'repeat_tag' in config:
        raise ValueError('Config is not the cuda_direct_v1 protocol with seeds 0-2')
    if config['training'] != training_budget(config):
        raise ValueError('Training budget differs from EXP-P15-001')
    if list(config['calibration']['bounds']) != list(BOUNDS):
        raise ValueError('Temperature bounds differ from JointDelivery.calibrate')
    device = config['device']
    if device['kind'] != 'cuda' or device['cpu_fallback'] is not False or len(device['allowed_gpu_uuids']) > 2 \
            or device['max_parallel'] > 2:
        raise ValueError('Explicit CUDA on at most two allowed GPUs, no fallback')
    columns = config['export']['workload_columns']
    names = set()
    for arm in config['arms']:
        if is_pitcher_id_arm(arm):
            from pitchmdp.direct_pitcher_id_registry import check_arm  # torch-free
            check_arm(arm)
            if arm['name'] in names or any(c not in columns for c in arm['workload']):
                raise ValueError('Duplicated pitcher-ID arm or unexported workload: ' + arm['name'])
            names.add(arm['name'])
            continue
        if is_robust_clip_arm(arm):
            from pitchmdp.direct_robust_clip_registry import check_arm  # torch-free
            check_arm(arm)
            if arm['name'] in names or any(c not in columns for c in arm['workload']):
                raise ValueError('Duplicated robust-clip arm or unexported workload: ' + arm['name'])
            names.add(arm['name'])
            continue
        if is_ple_arm(arm):
            from pitchmdp.direct_ple_registry import check_arm  # torch-free
            check_arm(arm)
            if arm['name'] in names or any(c not in columns for c in arm['workload']):
                raise ValueError('Duplicated PLE arm or unexported workload: ' + arm['name'])
            names.add(arm['name'])
            continue
        if 'tabm_residual' in arm:
            from pitchmdp.direct_tabm_residual_registry import check_arm  # torch-free
            check_arm(arm)
            if arm['name'] in names or any(c not in columns for c in arm['workload']):
                raise ValueError('Duplicated TabM-residual arm or unexported workload: ' + arm['name'])
            names.add(arm['name'])
            continue
        if 'combo' in arm:
            combo_arm_check(config, arm)
            if arm['name'] in names or any(c not in columns for c in arm['workload']):
                raise ValueError('Duplicated combination arm or unexported workload: ' + arm['name'])
            names.add(arm['name'])
            continue
        if 'tabm' in arm:
            from pitchmdp.direct_tabm_registry import check_arm  # torch-free
            check_arm(arm)
            if arm['name'] in names or any(c not in columns for c in arm['workload']):
                raise ValueError('Duplicated TabM-mini arm or unexported workload: ' + arm['name'])
            names.add(arm['name'])
            continue
        if 'ema' in arm:
            from pitchmdp.direct_ema_registry import check_arm  # torch-free
            check_arm(arm)
            if arm['name'] in names:
                raise ValueError('Duplicated EMA arm: ' + arm['name'])
            names.add(arm['name'])
            continue
        if 'embed_residual' in arm:
            from pitchmdp.direct_embed_residual_registry import check_arm  # torch-free
            check_arm(arm)
            if arm['name'] in names or any(c not in columns for c in arm['workload']):
                raise ValueError('Duplicated embedded-residual arm or unexported workload: ' + arm['name'])
            names.add(arm['name'])
            continue
        if 'workload_dropout' in arm:
            from pitchmdp.direct_workload_dropout_registry import check_arm  # torch-free
            check_arm(arm)
            if arm['name'] in names or any(c not in columns for c in arm['workload']):
                raise ValueError('Duplicated workload-dropout arm or unexported workload: ' + arm['name'])
            names.add(arm['name'])
            continue
        if 'dropout_head' in arm:
            from pitchmdp.direct_dropout_registry import check_arm  # torch-free
            check_arm(arm)
            if arm['name'] in names:
                raise ValueError('Duplicated arm: ' + arm['name'])
            names.add(arm['name'])
            continue
        if 'frontier' in arm:
            frontier_arm_check(arm)
            if arm['name'] in names:
                raise ValueError('Duplicated arm: ' + arm['name'])
            names.add(arm['name'])
            continue
        if arm['model'] not in MODELS or arm['width'] != 128 or arm['name'] in names:
            raise ValueError('Unknown or duplicated arm: ' + arm['name'])
        if any(c not in columns for c in arm['workload']) or len(set(arm['workload'])) != len(arm['workload']):
            raise ValueError('Workload arm uses an unexported column')
        if arm['model'] != 'flatten_mlp' and arm['workload']:
            raise ValueError('Architecture arms keep the control inputs (one change per arm)')
        names.add(arm['name'])
    if uses_pitcher_id(config):
        pass  # the two-arm P38 family compares against its reference arm (registry), not a flatten control
    elif 'control' not in names or [a for a in config['arms'] if a['name'] == 'control'][0]['workload']:
        raise ValueError('A no-workload flatten control is required')
    if any('dropout_head' in arm for arm in config['arms']):
        from pitchmdp.direct_dropout_registry import check_family  # torch-free
        check_family(config)
    if int(config['export']['last_year']) != 2025:
        raise ValueError('No 2026 rows: last year is 2025')
    if any(arm.get('frontier') or arm.get('combo') for arm in config['arms']):
        if expected_context_names(config['export']['context_layout'], config['export']['shapes']['n_context']) \
                != config['export']['context_names']:
            raise ValueError('Config context names differ from the names its expected layout derives')
    return config


def frontier_arm_check(arm):
    """A frontier arm is exactly its fixed registry entry (+ name, priority, empty workload)."""
    from pitchmdp.direct_frontier_registry import ARMS  # torch-free
    rest = {k: v for k, v in arm.items() if k not in ('name', 'priority', 'frontier', 'workload')}
    if arm['frontier'] is not True or arm['name'] not in ARMS or rest != ARMS[arm['name']]:
        raise ValueError('Frontier arm differs from the fixed registry: ' + str(arm.get('name')))
    if arm.get('workload') != [] or arm.get('priority') not in (1, 2):
        raise ValueError('Frontier arms keep the control inputs (no workload) and a priority')


def combo_arm_check(config, arm):
    """A combination arm is exactly its registry entry; data arms need the config-owned subset rule."""
    from pitchmdp.direct_combo_registry import check_arm, check_subset  # torch-free
    subset = config.get('train_subset')
    if any(a.get('train_fraction') for a in config['arms']) or subset is not None:
        check_subset(subset or {})
    if config['export']['shapes']['n_context'] != 52:
        raise ValueError('Combination arms need the 52-column direct context')
    check_arm(arm, config['export']['context_layout']['pitcher_profile_columns'], subset or {})


def needs_context_names(arm):
    return (arm.get('frontier') is True or arm.get('combo') is True) and arm['model'] == 'masked_flatten_mlp'


def train_subset(config, arm, train):
    """(TRAIN arrays, provenance): unchanged unless a data arm; never applied to another split."""
    if 'train_fraction' not in arm:
        return train, None
    if arm.get('combo') is not True:
        raise ValueError('Only registered combination data arms subsample TRAIN')
    from pitchmdp.direct_combo_registry import subset_split
    return subset_split(train, config['train_subset']['salt'], arm['train_fraction'])


def expected_context_names(layout, n_context):
    """52 ordered names from the GROUPED export layout, only through source equivalences.

    SequenceContext.report() lists 11 named game-state features then three grouped
    blocks; Archetypes.report() names the 6 style and 6 reliability columns and
    n_clusters (memberships, LAST in numeric_features); SharingContext appends the
    pitcher profile (clusters['columns']) and log1p(TRAIN count)/10.
    """
    from pitchmdp.direct_frontier_registry import (BASE_CONTEXT_NAMES, BATTER_RELIABILITY, BATTER_STYLE,
                                                   GAME_STATE, direct_context_names)
    profile = list(layout['pitcher_profile_columns'])
    base = len(BASE_CONTEXT_NAMES)
    groups = {'base': [0, base], 'pitcher_profile': [base, base + len(profile)],
              'pitcher_support': [base + len(profile), base + len(profile) + 1]}
    if layout['groups'] != groups or groups['pitcher_support'][1] != n_context:
        raise ValueError('Context group boundaries differ from 0:28 / 28:51 / 51:52')
    if layout['base_features'] != [*GAME_STATE, *BASE_GROUPED_FEATURES]:
        raise ValueError('Base context features differ from SequenceContext.report')
    if layout['style_columns'] != list(BATTER_STYLE) or layout['reliability_columns'] != list(BATTER_RELIABILITY) \
            or layout['n_clusters'] != 5 or layout['stripped_tail'] != SHARING_TAIL:
        raise ValueError('Batter archetype columns differ from Archetypes(n_clusters=5)')
    return direct_context_names(profile)


def manifest_context_layout(manifest):
    """The same comparable fields, read from the export manifest's recorded lineage (metadata only)."""
    try:
        layout = manifest['lineage']['context_layout']
        report = layout['base_report']
        return {'groups': layout['groups'], 'base_features': report['features'],
                'style_columns': report['archetypes']['style_columns'],
                'reliability_columns': report['archetypes']['reliability_columns'],
                'n_clusters': report['archetypes']['n_clusters'],
                'pitcher_profile_columns': layout['pitcher_profile_columns'], 'stripped_tail': layout['stripped_tail']}
    except (KeyError, TypeError) as error:
        raise ValueError('Export manifest has no recorded context layout') from error


def verified_context_names(config, manifest):
    """Config-expected names, only if the export manifest records exactly the expected layout."""
    expected = config['export']['context_layout']
    if manifest_context_layout(manifest) != expected:
        raise ValueError('Export context layout differs from the config expected layout')
    names = expected_context_names(expected, manifest['shapes']['n_context'])
    if names != config['export']['context_names']:
        raise ValueError('Export context names differ from the config')
    return names


def arm_spec(config, name):
    for arm in config['arms']:
        if arm['name'] == name:
            return arm
    raise ValueError('Unregistered arm: ' + name)


def verify_export(export, config, manifest_sha256, hash_arrays=True):
    """Fail closed unless the manifest is the pinned one and every array matches it."""
    export = Path(export)
    if not manifest_sha256 or sha256_file(export / 'manifest.json') != manifest_sha256:
        raise ValueError('Export manifest differs from the pinned SHA256')
    if config['export'].get('manifest_sha256', manifest_sha256) != manifest_sha256:
        raise ValueError('Export manifest differs from the config pin')
    manifest = read_json(export / 'manifest.json')
    expected = config['export']
    if manifest.get('format') != 'cuda_direct_export_v1' or set(manifest['splits']) != set(expected['splits']):
        raise ValueError('Export format or split list differs')
    if manifest['workload_columns'] != expected['workload_columns'] or manifest['shapes'] != expected['shapes']:
        raise ValueError('Export layout differs from the config')
    for split, record in manifest['splits'].items():
        if int(record['max_year']) > int(expected['last_year']):
            raise ValueError(f'{split}: rows after {expected["last_year"]} are forbidden')
        if set(record['files']) != set(ARRAYS):
            raise ValueError(f'{split}: missing exported arrays')
        if hash_arrays:
            for name, meta in record['files'].items():
                if sha256_file(export / meta['path']) != meta['sha256']:
                    raise ValueError(f'{split}/{name}: array hash differs from the manifest')
    if uses_batter_sidecar(config):
        batter_sidecar(export, config, manifest, hash_arrays)
    return manifest


def uses_batter_sidecar(config):
    from pitchmdp.direct_pitcher_id_registry import is_batter_family  # torch-free
    return uses_pitcher_id(config) and is_batter_family(config)


def batter_sidecar(export, config, manifest, hash_arrays=True):
    """(directory, verified record) of the P43 batter sidecar: sibling `dirname` of the export, sidecar.json sha pinned."""
    from pitchmdp import batter_sidecar as bs  # numpy only
    from pitchmdp.direct_pitcher_id_registry import check_sidecar_pin
    pin = check_sidecar_pin(config)
    directory = Path(export).absolute().parent / pin['dirname']  # sibling of the export path as given (no symlink resolution)
    return directory, bs.verify(directory, manifest, pin['sidecar_sha256'], hash_arrays)


def with_ids(config, export, manifest, split, arrays, limit=None):
    """P43: the split's arrays plus `batter` from the verified sidecar (same row order / limit); others unchanged."""
    if not uses_batter_sidecar(config):
        return arrays
    from pitchmdp import batter_sidecar as bs
    directory, record = batter_sidecar(export, config, manifest, hash_arrays=False)
    batter = bs.load(directory, record, split, limit)
    if len(batter) != len(arrays['y']):
        raise ValueError(f'{split}: batter sidecar misaligned with the export split')
    return {**arrays, 'batter': batter}


def load_split(export, manifest, split, limit=None):
    record = manifest['splits'][split]
    arrays = {}
    for name, meta in record['files'].items():
        value = np.load(Path(export) / meta['path'], mmap_mode='r', allow_pickle=False)
        if list(value.shape) != meta['shape'] or str(value.dtype) != meta['dtype']:
            raise ValueError(f'{split}/{name}: shape or dtype differs from the manifest')
        arrays[name] = value[:limit] if limit else value
    n = len(arrays['y'])
    if any(len(v) != n for v in arrays.values()) or n != (min(record['n'], limit) if limit else record['n']):
        raise ValueError(f'{split}: misaligned arrays')
    if hashlib.sha256(np.ascontiguousarray(arrays['keys'], dtype=np.int64).tobytes()).hexdigest() != record['rows_sha256'] \
            and not limit:
        raise ValueError(f'{split}: ordered key hash differs from the frozen sample')
    if (np.asarray(arrays['date']) // 10000).max(initial=0) > 2025:
        raise ValueError(f'{split}: 2026 row')
    return arrays


def arm_inputs(arrays, arm, columns):
    """(tokens, valid, context) for the arm; workload after the stripped base context."""
    tokens = np.array(arrays['tokens'], dtype=np.float32)  # writable copies of the read-only mmaps
    if np.any(tokens[:, -1, :N_PHYSICAL] != 0):
        raise ValueError('Current-pitch physics leaked into the direct head input')
    if np.any(tokens[:, -1, -N_OUTCOME:] != 0):
        raise ValueError('Current-pitch outcome leaked into the direct head input')
    context = np.array(arrays['context'], dtype=np.float32)
    if arm['workload']:
        keep = [columns.index(c) for c in arm['workload']]
        context = np.concatenate([context, np.asarray(arrays['workload'], dtype=np.float32)[:, keep]], axis=1)
    return tokens, np.array(arrays['valid'], dtype=bool), context


# ------------------------------------------------------------------ model / calibration

def build_model(arm, seed, device, context_names=None, vocab_ids=None):
    if is_pitcher_id_arm(arm) and 'id_columns' in arm[PITCHER_ID_FLAG]:  # P43 arms
        from pitchmdp import direct_pitcher_batter_id_model as pbm
        if context_names or vocab_ids is None:
            raise ValueError('Batter-ID arms take no context names and need the TRAIN vocabularies')
        builder = pbm.build_pitcher_batter_id if len(arm[PITCHER_ID_FLAG]['id_columns']) == 2 else pbm.build_batter_id
        return builder(arm, seed, device, vocab_ids)
    if is_pitcher_id_arm(arm):
        from pitchmdp.direct_pitcher_id_model import build_pitcher_id
        if context_names or vocab_ids is None:
            raise ValueError('Pitcher-ID arms take no context names and need the TRAIN vocabulary')
        return build_pitcher_id(arm, seed, device, vocab_ids)
    if vocab_ids is not None:
        raise ValueError('Only pitcher-ID arms take a vocabulary')
    if is_robust_clip_arm(arm):
        from pitchmdp.direct_robust_clip_factory import build_robust_clip
        if context_names:
            raise ValueError('Robust-clip arms take no context names')
        return build_robust_clip(arm, seed, device)
    if is_ple_arm(arm):
        from pitchmdp.direct_ple_factory import build_ple
        if context_names:
            raise ValueError('PLE arms take no context names')
        return build_ple(arm, seed, device)
    if arm.get('tabm_residual') is True:
        from pitchmdp.direct_tabm_residual_factory import build_tabm_residual
        if context_names:
            raise ValueError('TabM-residual arms take no context names')
        return build_tabm_residual(arm, seed, device)
    if arm.get('ema') is True:
        from pitchmdp.direct_ema_model import build_ema
        if context_names:
            raise ValueError('EMA arms take no context names')
        return build_ema(arm['name'], seed, device)
    if arm.get('tabm') is True:
        from pitchmdp.direct_tabm_factory import build_tabm
        if context_names:
            raise ValueError('TabM-mini arms take no context names')
        return build_tabm(arm, seed, device)
    if arm.get('embed_residual') is True:
        from pitchmdp.direct_embed_residual_factory import build_embed_residual
        if context_names:
            raise ValueError('Embedded-residual arms take no context names')
        return build_embed_residual(arm, seed, device)
    if arm.get('workload_dropout') is True:
        from pitchmdp.direct_workload_dropout_model import build_workload_dropout
        if context_names:
            raise ValueError('Workload-dropout arms take no context names')
        return build_workload_dropout(arm, seed, device)
    if arm.get('dropout_head') is True:
        from pitchmdp.direct_dropout_model import build_dropout
        if context_names:
            raise ValueError('Dropout arms take no context names')
        return build_dropout(arm, seed, device)
    if arm.get('combo') is True:
        from pitchmdp.direct_combo_model import build_combo
        if needs_context_names(arm) != bool(context_names):
            raise ValueError('Pitcher-information arms (only) need the verified export context names')
        return build_combo(arm, seed, device, context_names)
    if arm.get('frontier') is True:
        from pitchmdp.direct_frontier_model import build_frontier
        if needs_context_names(arm) != bool(context_names):
            raise ValueError('Information arms (only) need the verified export context names')
        return build_frontier(arm['name'], seed, device, context_names)
    if arm['model'] == 'flatten_mlp':
        from pitchmdp.matrix_models import MatrixModel
        return MatrixModel('flatten_mlp', seed=seed, width=arm['width'], device=device)
    if arm['model'] == 'residual_mlp':
        from pitchmdp.direct_residual_model import DirectResidualModel
        return DirectResidualModel(seed, device=device)
    from pitchmdp.direct_periodic_model import DirectPeriodicModel  # imports pandas via matrix_sharing
    return DirectPeriodicModel(seed, arm['context_continuous'], device=device)


def recorded_ple_bins(path):
    """The bins hash the runner recorded for this task (fit.json next to model.pt); required for PLE loads."""
    record = Path(path).parent / 'fit.json'
    if not record.is_file():
        raise ValueError('PLE load needs the run-recorded bins hash (fit.json next to the archive)')
    return read_json(record).get('ple_bins_sha256')


def recorded_rc_stats(path):
    """The statistics hash the runner recorded for this task (fit.json next to model.pt); required for robust-clip loads."""
    record = Path(path).parent / 'fit.json'
    if not record.is_file():
        raise ValueError('Robust-clip load needs the run-recorded statistics hash (fit.json next to the archive)')
    return read_json(record).get('rc_stats_sha256')


def recorded_vocab(path):
    """The vocabulary hash the runner recorded for this task (fit.json next to model.pt); required for pitcher-ID loads."""
    record = Path(path).parent / 'fit.json'
    if not record.is_file():
        raise ValueError('Pitcher-ID load needs the run-recorded vocabulary hash (fit.json next to the archive)')
    block = read_json(record).get(PITCHER_ID_FLAG) or {}
    if 'vocabs' in block:  # P43: {column: ids_sha256}
        return {c: v.get('ids_sha256') for c, v in block['vocabs'].items()}
    return block.get('ids_sha256')


def load_model(arm, path, device, context_names=None, ple_bins_sha256=None, rc_stats_sha256=None, ids_sha256=None):
    if is_pitcher_id_arm(arm) and 'id_columns' in arm[PITCHER_ID_FLAG]:  # P43 arms
        from pitchmdp import direct_pitcher_batter_id_model as pbm
        hashes = ids_sha256 or recorded_vocab(path)
        if len(arm[PITCHER_ID_FLAG]['id_columns']) == 2:
            return pbm.load_pitcher_batter_id(arm, path, device, hashes)
        return pbm.load_batter_id(arm, path, device, hashes)
    if is_pitcher_id_arm(arm):
        from pitchmdp.direct_pitcher_id_model import load_pitcher_id
        return load_pitcher_id(arm, path, device, ids_sha256 or recorded_vocab(path))
    if is_robust_clip_arm(arm):
        from pitchmdp.direct_robust_clip_factory import load_robust_clip
        return load_robust_clip(arm, path, device, rc_stats_sha256 or recorded_rc_stats(path))
    if is_ple_arm(arm):
        from pitchmdp.direct_ple_factory import load_ple
        return load_ple(arm, path, device, ple_bins_sha256 or recorded_ple_bins(path))
    if arm.get('tabm_residual') is True:
        from pitchmdp.direct_tabm_residual_factory import load_tabm_residual
        return load_tabm_residual(arm, path, device)
    if arm.get('ema') is True:
        from pitchmdp.direct_ema_model import load_ema
        return load_ema(arm['name'], path, device)
    if arm.get('tabm') is True:
        from pitchmdp.direct_tabm_factory import load_tabm
        return load_tabm(arm, path, device)
    if arm.get('embed_residual') is True:
        from pitchmdp.direct_embed_residual_factory import load_embed_residual
        return load_embed_residual(arm, path, device)
    if arm.get('workload_dropout') is True:
        from pitchmdp.direct_workload_dropout_model import load_workload_dropout
        return load_workload_dropout(arm, path, device)
    if arm.get('dropout_head') is True:
        from pitchmdp.direct_dropout_model import load_dropout
        return load_dropout(arm, path, device)
    if arm.get('combo') is True:
        from pitchmdp.direct_combo_model import load_combo
        model = load_combo(arm, path, device)
        if needs_context_names(arm) and model.context_names != list(context_names or []):
            raise ValueError('Archived context names differ from the verified export names')
        return model
    if arm.get('frontier') is True:
        from pitchmdp.direct_frontier_model import load_frontier
        model = load_frontier(arm['name'], path, device)
        if needs_context_names(arm) and model.context_names != list(context_names or []):
            raise ValueError('Archived context names differ from the verified export names')
        return model
    if arm['model'] == 'flatten_mlp':
        from pitchmdp.matrix_models import MatrixModel
        return MatrixModel.load(path, device)
    if arm['model'] == 'residual_mlp':
        from pitchmdp.direct_residual_model import DirectResidualModel
        return DirectResidualModel.load(path, device)
    from pitchmdp.direct_periodic_model import DirectPeriodicModel
    return DirectPeriodicModel.load(path, device)


def g0_log_probs(logits):
    """SharingPredictor('G0-global').logits: log of clipped global softmax."""
    from scipy.special import softmax
    return np.log(np.clip(softmax(np.asarray(logits).astype(float), axis=1), 1e-30, 1))


def fit_temperature(log_probs, labels):
    """JointDelivery.calibrate with one all-zero-physics draw."""
    from scipy.optimize import minimize_scalar
    from scipy.special import softmax
    logits, labels = np.asarray(log_probs)[:, None, :], np.asarray(labels, dtype=np.int64)

    def loss(temperature):
        p = softmax(logits / temperature, axis=-1).mean(1)
        return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    fitted = minimize_scalar(loss, bounds=BOUNDS, method='bounded')
    return float(fitted.x), float(fitted.fun)


def probabilities(log_probs, temperature):
    """matrix_benchmark.integrated_probabilities for one draw: (calibrated, raw)."""
    from scipy.special import softmax
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Expected positive finite temperature')
    result = tuple(softmax(np.asarray(log_probs, dtype=np.float64) / t, axis=-1) for t in (temperature, 1.))
    for p in result:
        if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1., atol=1e-8):
            raise ValueError('Direct prediction lost probability mass')
    return result


def epoch_record(report, cap, patience):
    """Epoch cap passed to fit and where the fit stopped, so cap hits are visible per fit.

    The frozen loops break as soon as run - best reaches patience, so a stop before the cap has
    run - best == patience and a run to the cap has run - best <= patience."""
    run, best = int(report['epochs_run']), int(report['best_epoch'])
    if not 1 <= best <= run <= cap or run - best > patience or (run < cap and run - best != patience):
        raise ValueError('Fit report epochs are inconsistent with the epoch cap and patience')
    return {'cap': int(cap), 'patience': int(patience), 'epochs_run': run, 'best_epoch': best,
            'stopped_by': 'patience' if run - best >= patience else 'cap'}


def model_device_type(model):
    return next(model.net.parameters()).device.type


# ------------------------------------------------------------------ device binding

def bind_gpu(uuid, allowed):
    """Bind this process to one allowed GPU before torch is imported; refuse anything else."""
    if uuid not in allowed:
        raise ValueError('GPU UUID is not in the allowed list')
    if 'torch' in sys.modules:
        raise RuntimeError('torch was imported before the GPU was bound')
    current = os.environ.get('CUDA_VISIBLE_DEVICES')
    if current not in (None, '', uuid):
        raise ValueError('CUDA_VISIBLE_DEVICES already points elsewhere: ' + current)
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid  # process-local only
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('CUDA unavailable or not exactly one visible device; no fallback')
    props = torch.cuda.get_device_properties(0)
    seen = getattr(props, 'uuid', None)
    if seen is not None and str(seen).lower() not in uuid.lower():
        raise RuntimeError(f'Visible device UUID {seen} differs from the bound {uuid}')
    return runtime_report(torch, uuid, props)


def runtime_report(torch, uuid=None, props=None):
    report = {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
              'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'bound_uuid': uuid,
              'hostname': platform.node(), 'pid': os.getpid(), 'pgid': os.getpgid(0)}
    try:
        import scipy
        report['scipy'] = scipy.__version__
    except ImportError:
        report['scipy'] = None
    if props is not None:
        report.update(device_name=props.name, device_uuid=str(getattr(props, 'uuid', 'unavailable')),
                      device_total_memory=int(props.total_memory), device_capability=[props.major, props.minor])
    return report


# ------------------------------------------------------------------ task

def task_dir(output, arm, seed, smoke):
    return Path(output) / (arm + ('-smoke' if smoke else '')) / str(seed)


def run_task(config, export, manifest_sha256, output, arm_name, seed, device, runtime, smoke=False):
    """Fit, calibrate and predict one (arm, seed); atomic final directory, failures preserved."""
    require_executable(config, smoke)
    arm = arm_spec(config, arm_name)
    if seed not in config['seeds']:
        raise ValueError('Unregistered seed')
    dest = task_dir(output, arm_name, seed, smoke)
    if dest.exists():
        raise FileExistsError(f'{dest} exists; artifacts are never overwritten')
    dest.parent.mkdir(parents=True, exist_ok=True)
    work = dest.parent / f'.{seed}.tmp-{os.getpid()}'
    work.mkdir()
    started = time.perf_counter()
    try:
        identity = expected_identity(config, manifest_sha256, arm_name, seed, smoke, device, own_hashes())
        write_json_atomic(work / 'identity.json', identity)
        manifest = verify_export(export, config, manifest_sha256)
        columns = config['export']['workload_columns']
        lim = (lambda key: SMOKE[key]) if smoke else (lambda key: None)
        load = lambda split, n: with_ids(config, export, manifest, split, load_split(export, manifest, split, n), n)  # noqa: E731
        train, subset = train_subset(config, arm, load('train', lim('train')))
        early = load('earlystop', lim('earlystop'))
        spec = config['training']
        names = verified_context_names(config, manifest) if needs_context_names(arm) else None
        vocab = pitcher_vocab(arm, manifest, train)
        unk_share = {}

        def inputs(split, arrays):  # the arm's inputs; pitcher-ID arms append the TRAIN-vocabulary index column
            if vocab is None:
                return arm_inputs(arrays, arm, columns)
            if 'per_column' in vocab:  # P43 arms: one index column per id column, in id_columns order
                from pitchmdp.direct_pitcher_batter_id_model import append_indices
                index, shares = p43_indices(arrays, vocab, split)
                unk_share[split] = shares
                return append_indices(arm_inputs(arrays, arm, columns), index)
            from pitchmdp.direct_pitcher_id_model import id_inputs
            arm_arrays, unk_share[split] = id_inputs(arm_inputs(arrays, arm, columns), arrays['pitcher'], vocab['ids'],
                                                     split, vocab['index_source'], vocab['shuffle_seed'])
            return arm_arrays
        model = build_model(arm, seed, device, names, None if vocab is None else vocab_ids_of(vocab))
        train_inputs = inputs('train', train)
        context_width = int(train_inputs[2].shape[1])
        epochs = SMOKE['epochs'] if smoke else spec['epochs']
        model.fit(train_inputs, np.array(train['y']), inputs('earlystop', early),
                  np.array(early['y']), epochs=epochs, patience=spec['patience'],
                  batch_size=spec['batch_size'], learning_rate=spec['learning_rate'])
        if model_device_type(model) != device.split(':')[0]:
            raise RuntimeError('Model is not on the requested device')
        fit_seconds = time.perf_counter() - started
        model.save(work / 'model.pt')
        train_rows = hashlib.sha256(np.ascontiguousarray(train['keys'], dtype=np.int64).tobytes()).hexdigest()
        early_rows = hashlib.sha256(np.ascontiguousarray(early['keys'], dtype=np.int64).tobytes()).hexdigest()
        bin_rows = ple_bin_rows(train_inputs) if is_ple_arm(arm) else None
        stats_rows = ple_bin_rows(train_inputs) if is_robust_clip_arm(arm) else None  # same row rule as PLE bins
        del train, train_inputs
        # calibration on the May 16-31 Cpanel rows only
        temp = load('temperature', lim('temperature'))
        temperature, temp_loss = fit_temperature(g0_log_probs(model.logits(inputs('temperature', temp))), temp['y'])
        calibration = {**model.report, 'delivery_temperature': temperature,
                       'calibration_integrated_log_loss': temp_loss, 'delivery_calibration_rows': int(len(temp['y']))}
        write_json_atomic(work / 'calibration.json', calibration)
        archives = {'predictions.npz': ('blend', 'dev'), 'mlb_predictions.npz': ('mlb_dev',)}
        rows = {}
        for archive, splits in archives.items():
            values = {}
            for split in splits:
                part = load(split, lim('predict'))
                calibrated, raw = probabilities(g0_log_probs(model.logits(inputs(split, part))), temperature)
                values.update({split: calibrated, split + '_raw': raw,
                               split + '_delivery_level': np.full(len(calibrated), NO_PHYSICS_LEVEL, dtype=np.int64),
                               split + '_keys': np.asarray(part['keys'], dtype=np.int64),
                               split + '_y': np.asarray(part['y'], dtype=np.int64),
                               split + '_game_pk': np.asarray(part['game_pk'], dtype=np.int64),
                               split + '_pitcher': np.asarray(part['pitcher'], dtype=np.int64)})
                rows[split] = len(calibrated)
            np.savez_compressed(work / archive, **values)
        if own_hashes() != identity['source_sha256']:
            raise RuntimeError('Sources changed during the task')
        fit_report = {'report': model.report, 'device': model.device, 'fit_seconds': fit_seconds,
                      'seconds_total': time.perf_counter() - started, 'context_width': context_width,
                      'parameter_count': int(sum(p.numel() for p in model.net.parameters())),
                      'train_rows_sha256': train_rows, 'earlystop_rows_sha256': early_rows,
                      'predicted_rows': rows, 'runtime': runtime, 'train_subset': subset, 'context_names_verified': names is not None, 'peak_device_bytes': peak_device_bytes(device),
                      'finished_utc': datetime.now(timezone.utc).isoformat(), 'dev_used_for_fit_or_selection': False}
        if uses_epoch_cap(config):
            fit_report['epoch_cap'] = epoch_record(model.report, epochs, spec['patience'])
        if vocab is not None and 'per_column' in vocab:
            fit_report[PITCHER_ID_FLAG] = p43_fit_record(model, vocab, unk_share)
            print('PITCHER_ID_VOCAB', arm_name, seed, json.dumps(fit_report[PITCHER_ID_FLAG]), flush=True)
        elif vocab is not None:
            from pitchmdp.direct_pitcher_id_model import vocab_sha256
            from pitchmdp.direct_pitcher_id_registry import vocab_record
            if vocab_sha256(model.vocab_ids) != vocab['ids_sha256']:
                raise RuntimeError('Model vocabulary differs from the TRAIN vocabulary record')
            fit_report[PITCHER_ID_FLAG] = {**vocab_record(vocab), 'unk_share': unk_share,
                                           'split_dates': vocab['split_dates']}
            if vocab['index_source'] != 'identity' or model.id_dropout:  # P40 controls only (P38 record unchanged)
                fit_report[PITCHER_ID_FLAG].update(index_source=vocab['index_source'],
                                                   shuffle_seed=vocab['shuffle_seed'], id_dropout=model.id_dropout)
            print('PITCHER_ID_VOCAB', arm_name, seed, json.dumps({k: v for k, v in fit_report[PITCHER_ID_FLAG].items()
                                                                  if k != 'split_dates'}), flush=True)
        if is_ple_arm(arm):
            from pitchmdp.direct_ple_registry import effective_record, fit_record
            fit_report.update(fit_record(model.report['network'], config['export']['shapes']['length'],
                                         verified_context_names(config, manifest), arm['workload']))
            fit_report.update(effective_record(model.report['network'], fit_report['parameter_count']),
                              ple_bin_rows=bin_rows)
            print('PLE_BIN_ROWS', arm_name, seed, json.dumps(bin_rows), flush=True)
        if is_robust_clip_arm(arm):
            from pitchmdp.direct_robust_clip_registry import check_module, check_stats_rows, fit_record as rc_record
            fit_report.update(rc_record(model.report, config['export']['shapes']['length']),
                              rc_stats_rows=check_stats_rows(stats_rows, config['export']['shapes']['length']),
                              rc_module_sha256=check_module())
            if fit_report['rc_module_sha256'] != identity['source_sha256'].get('pitchmdp/direct_robust_clip_model.py'):
                raise RuntimeError('Robust-clip module differs from the task identity source map')
            print('RC_STATS_ROWS', arm_name, seed, fit_report['rc_stats_sha256'], json.dumps(stats_rows), flush=True)
        write_json_atomic(work / 'fit.json', fit_report)
        names = ['identity.json', 'model.pt', 'calibration.json', 'predictions.npz', 'mlb_predictions.npz', 'fit.json']
        write_json_atomic(work / 'state.json', {'identity': identity,
                                                'artifact_hashes': {n: sha256_file(work / n) for n in names}})
        if dest.exists():
            raise FileExistsError(f'{dest} appeared during the task')
        os.rename(work, dest)
        print('CUDA_DIRECT_TASK_COMPLETE', arm_name, seed, device, f'{time.perf_counter() - started:.1f}s', flush=True)
        return dest
    except BaseException as error:
        failed = Path(output) / 'failed' / f'{arm_name}{"-smoke" if smoke else ""}-{seed}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{os.getpid()}'
        failed.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_json_atomic(work / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc(),
                                                    'seconds': time.perf_counter() - started})
        finally:
            os.rename(work, failed)
        raise


def pitcher_vocab(arm, manifest, train):
    """TRAIN-only vocabulary of a pitcher-ID arm (None for every other arm): manifest split dates pinned, TRAIN rows
    dated on/before the cutoff, labels never read."""
    if not is_pitcher_id_arm(arm):
        return None
    from pitchmdp.direct_pitcher_id_registry import build_vocab, check_split_dates, id_spec
    spec = arm[PITCHER_ID_FLAG]
    if 'id_columns' in spec:  # P43: one TRAIN-only vocabulary per id column (same rule as P38)
        from pitchmdp.direct_pitcher_id_registry import id_columns
        index_source, shuffle_seed, _ = id_spec(arm)
        split_dates = check_split_dates(manifest, spec['last_train_date'])
        per_column = {c: build_vocab(np.asarray(train[c]), np.asarray(train['date']), spec['last_train_date'],
                                     spec['min_train_pitches']) for c in id_columns(arm)}
        return {'per_column': per_column, 'columns': id_columns(arm), 'split_dates': split_dates,
                'index_source': index_source, 'shuffle_seed': shuffle_seed}
    index_source, shuffle_seed, _ = id_spec(arm)
    split_dates = check_split_dates(manifest, spec['last_train_date'])
    vocab = build_vocab(np.asarray(train['pitcher']), np.asarray(train['date']), spec['last_train_date'],
                        spec['min_train_pitches'])
    return {**vocab, 'split_dates': split_dates, 'index_source': index_source, 'shuffle_seed': shuffle_seed}


def vocab_ids_of(vocab):
    """Vocabulary ids for build_model: P38/P40 one array; P43 a list in id_columns order (one per column)."""
    if 'per_column' not in vocab:
        return vocab['ids']
    ids = [vocab['per_column'][c]['ids'] for c in vocab['columns']]
    return ids if len(ids) > 1 else ids[0]


def p43_indices(arrays, vocab, split):
    """Index arrays (id_columns order) and per-column UNK shares; row shuffle permutes the split's own column with the
    fixed label-free seed before the lookup (subsets elsewhere always use the TRUE column)."""
    from pitchmdp.direct_pitcher_id_model import shuffled_pitcher  # generic fixed permutation of a split column
    from pitchmdp.direct_pitcher_id_registry import UNK, map_ids
    index, shares = [], {}
    for c in vocab['columns']:
        column = np.asarray(arrays[c], dtype=np.int64)
        if vocab['index_source'] == 'row_shuffle':
            column = shuffled_pitcher(column, split, vocab['shuffle_seed'])
        elif vocab['index_source'] != 'identity':
            raise ValueError('Unknown id index source: ' + str(vocab['index_source']))
        one = map_ids(column, vocab['per_column'][c]['ids'])
        index.append(one)
        shares[c] = float((one == UNK).mean()) if len(one) else 0.
    return index, shares


def p43_fit_record(model, vocab, unk_share):
    """fit.json id block of a P43 arm: per column the TRAIN vocabulary record + per-split UNK share."""
    from pitchmdp.direct_pitcher_id_model import vocab_sha256
    from pitchmdp.direct_pitcher_id_registry import vocab_record
    archived = model.vocab_ids if isinstance(model.vocab_ids, list) else [model.vocab_ids]
    vocabs = {}
    for c, ids in zip(vocab['columns'], archived):
        if vocab_sha256(ids) != vocab['per_column'][c]['ids_sha256']:
            raise RuntimeError(f'Model {c} vocabulary differs from the TRAIN vocabulary record')
        vocabs[c] = {**vocab_record(vocab['per_column'][c]), 'unk_share': {s: v[c] for s, v in unk_share.items()}}
    record = {'id_columns': list(vocab['columns']), 'vocabs': vocabs, 'index_source': vocab['index_source'],
              'shuffle_seed': vocab['shuffle_seed'], 'split_dates': vocab['split_dates']}
    if model.id_dropout:
        record['id_dropout'] = model.id_dropout
    if getattr(model, 'freeze_unk', False):
        record['freeze_unk'] = True
        record['unk_rows_exact_zero'] = {c: bool((t[0].detach() == 0).all().item())
                                         for c, t in zip(vocab['columns'], model.net.id_tables)}
        if not all(record['unk_rows_exact_zero'].values()):
            raise RuntimeError('Frozen UNK rows moved away from zero')
    return record


def ple_bin_rows(inputs):
    """Rows the PLE bins were fitted on (module review N-b / W5): all TRAIN-argument rows for context scalars and,
    per past position (oldest first), the rows where that position is valid for physics scalars."""
    valid = np.asarray(inputs[1], dtype=bool)
    return {'train_rows': int(len(valid)), 'past_valid_rows': [int(n) for n in valid[:, :-1].sum(axis=0)]}


def peak_device_bytes(device):
    if not device.startswith('cuda'):
        return None
    import torch
    return int(torch.cuda.max_memory_allocated())


def verify_task(dest, expected):
    """A finished task is the expected one (config/export/arm/seed/smoke/device[/sources]) with intact artifacts.

    Used by the queue (is_complete) and the Mac scorer; self-consistent hashes alone are not enough.
    """
    state = read_json(Path(dest) / 'state.json')
    names = ['identity.json', 'model.pt', 'calibration.json', 'predictions.npz', 'mlb_predictions.npz', 'fit.json']
    if sorted(state['artifact_hashes']) != sorted(names):
        raise ValueError(f'{dest}: artifact list differs')
    for name, digest in state['artifact_hashes'].items():
        if sha256_file(Path(dest) / name) != digest:
            raise ValueError(f'{dest}/{name}: artifact hash differs')
    if read_json(Path(dest) / 'identity.json') != state['identity']:
        raise ValueError(f'{dest}: state identity differs from identity.json')
    for key, value in expected.items():
        if state['identity'].get(key) != value:
            raise ValueError(f'{dest}: identity {key} differs from the expected task')
    return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--export-manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('task')
    t.add_argument('--arm', required=True)
    t.add_argument('--seed', type=int, required=True)
    t.add_argument('--smoke', action='store_true', help='row/epoch-limited wiring check; output must end -smoke')
    sub.add_parser('check', help='bind GPU, verify export and print the runtime report; no fit')
    args = p.parse_args(argv)
    config = config_check(read_json(args.config))
    if args.command == 'task':
        require_executable(config, args.smoke)
    runtime = bind_gpu(args.gpu_uuid, config['device']['allowed_gpu_uuids'])
    output = args.output.resolve()
    if getattr(args, 'smoke', False) != output.name.endswith('-smoke'):
        raise ValueError('Smoke runs write only to an output ending -smoke, and only smoke runs may')
    if args.command == 'check':
        verify_export(args.export, config, args.export_manifest_sha256)
        print('CUDA_DIRECT_CHECK_OK', json.dumps(runtime), flush=True)
        return
    run_task(config, args.export, args.export_manifest_sha256, output, args.arm, args.seed, 'cuda', runtime, args.smoke)


if __name__ == '__main__':
    main()
