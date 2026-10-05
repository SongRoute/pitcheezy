"""DRAFT portable scorer for a COMPLETE CUDA direct-head family (newsrv side, numpy only).

Scores the base 5-arm family (DRAFT-CUDA-DIRECT-20261004), the frontier 9-arm
family (DRAFT-CUDA-FRONTIER-RUN-20261004) or the data/combination 15-arm family
(DRAFT-CUDA-DATA-COMBO-RUN-20261004) where the tasks were written, so results do
not wait for the Mac.  A config with comparison.factorial_contrasts also gets those
fixed candidate-minus-reference pairs (same paired whole-game bootstrap helper,
same-seed and ensemble), validated before any prediction is read; no ranking or
verdict.  Families without the key are scored exactly as before.  Likewise
comparison.regularization_contrasts (architecture x dropout family) must equal
direct_dropout_registry.CONTRASTS (dropout arm minus its same-architecture no-dropout
arm) and is checked before any read.  comparison.workload_dropout_contrasts (periodic
workload x head-dropout family) must equal direct_workload_dropout_registry.CONTRASTS
(12 one-factor pairs; every one of the 50 fits must exist first) and is checked
before any read; no ranking, winner or interaction verdict.  comparison.tabm_contrasts
(TabM-mini family) must equal direct_tabm_registry.CONTRASTS (3 one-factor pairs on
seeds 13-15; all 15 fits must exist first), likewise checked before any read.
comparison.embed_residual_contrasts (embedded-input residual family) must equal
direct_embed_residual_registry.CONTRASTS (6 fixed pairs on seeds 16-18; all 18 fits
must exist first), likewise checked before any read.
comparison.ema_contrasts (EMA weight-averaging family) must equal
direct_ema_registry.CONTRASTS (3 fixed pairs on seeds 19-23; all 20 fits must exist
first), likewise checked before any read.
comparison.tabm_residual_contrasts (TabM-mini x periodic-residual family) must equal
direct_tabm_residual_registry.CONTRASTS (7 pre-registered pairs on seeds 24-28; all 25
fits must exist first), likewise checked before any read; every fit must also record the
epoch cap 60 it was given and where it stopped (cap hits listed per fit).
comparison.ple_contrasts (PLE-embedding family, EXP-P35-001 DRAFT) must equal
direct_ple_registry.CONTRASTS (7 pre-registered pairs on seeds 29-33; all 25 fits must exist
first), likewise checked before any read and tested BEFORE the TabM-residual family (its arm
tabm_residual_counts keeps the P34 flag); every fit records the epoch cap as P34, every PLE
fit its bins record and effective parameter count, and all PLE fits of the family must carry ONE bins hash (bins are
TRAIN-only); the degenerate-column count and the observed same-GPU pairs per contrast are
reported, the two embedding_type rows first.
comparison.robust_clip_contrasts (robust-clip family, tentative EXP-P36-001 DRAFT) must equal
direct_robust_clip_registry.CONTRASTS (2 pre-registered pairs on seeds 34-38, primary first; all 20
fits must exist first), likewise checked before any read and tested FIRST (its arm
periodic_residual_counts keeps the embed_residual flag); every fit records the epoch cap as P34/P35,
every robust-clip fit its statistics record (rc_stats_sha256 re-derived from the fit report, rule
counts, statistics rows, module sha == the registry pin), and all robust-clip fits must carry ONE
statistics hash and one TRAIN row hash (statistics are TRAIN-only, so every seed and GPU sees the same
statistics); the secondary row is labelled not compute-matched.  Descriptive only.
comparison.embed_type_contrasts (embedding-map control, tentative EXP-P37-001 DRAFT) must equal
direct_embed_type_registry.CONTRASTS (1 pre-registered pair on seeds 39-43; all 15 fits must exist first),
likewise checked before any read and tested FIRST (both embedded arms keep the embed_residual flag); every fit
records the epoch cap as P34-P36 and a cost record (parameters == the registered count, fit seconds, optimizer
updates, seconds per update, epochs, peak device bytes) from fit.json; the pair is labelled parameter-near-matched,
not compute-matched, and cost never adjusts the NLL contrast.
comparison.pitcher_id_contrasts (pitcher-ID family, tentative EXP-P38-001 DRAFT, two arms, no flatten control) must
equal direct_pitcher_id_registry.CONTRASTS and is tested FIRST (both arms keep embed_residual); all 10 fits must exist;
every candidate fit carries one TRAIN vocabulary record whose ids hash the scorer RE-DERIVES from the export TRAIN
`pitcher`/`date` arrays (hash-checked); parameters == reference + (n_vocab + 1) x 8 + 128 x 8; the contrast is reported
on all rows and on the in-vocabulary / UNK subsets (rows mapped with that vocabulary), every seed and the ensemble,
labelled not parameter- or compute-matched; no verdict.  Data arms must record the config's subset rule over the
export's own TRAIN rows (fit.json train_subset); every other arm must record none.  No raw data, pandas, sklearn, scipy, torch, G0 or Savant.

Scope: UNBLENDED calibrated direct-head probabilities (predictions.npz 'dev',
mlb_predictions.npz 'mlb_dev') of every arm vs the same-family CUDA control, BEFORE
the June blend / G0 mix.  Not the G15 mix, not a policy.  Descriptive repeated-DEV
only: no decision, ranking or winner (arms are listed in config order).

Fail closed, before any probability is read, unless
  * the config is the pinned file, passes config_check and is REGISTERED/enabled;
  * the export manifest is the pinned one and the dev/mlb_dev keys, y and game_pk
    arrays match their manifest hashes;
  * EVERY config arm x seed is a finished, non-smoke CUDA task whose identity equals
    this config/export/arm/seed and the EXPECTED task source map, with intact
    artifacts and one shared CUDA runtime signature.
The expected task sources are the current runner's own_hashes() by default, or an
explicit --source-manifest file (e.g. the first batch, built on older code without
the frontier modules); an artifact's self-declared sources are never trusted.
Every member's keys, labels and game ids must equal the export rows in order.

Metrics: matrix_metrics.prediction_metrics of the 3-seed probability-mean ensemble;
paired_game_comparison (arm minus control, whole-game bootstrap, 10,000 draws, seed
20260924) for the ensemble and for each seed vs the same-seed control.

  python scripts/score_cuda_direct.py --config <cfg> --config-sha256 <sha> --export <dir> \\
      --export-manifest-sha256 <sha> --family <family dir> --destination <new dir> \\
      [--source-manifest <task sources json>]
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np

import run_cuda_direct as rc  # noqa: E402  numpy only at import
from pitchmdp import direct_dropout_registry as dropout_registry  # noqa: E402  torch-free
from pitchmdp import direct_workload_dropout_registry as workload_dropout_registry  # noqa: E402  torch-free
from pitchmdp import direct_tabm_registry as tabm_registry  # noqa: E402  torch-free
from pitchmdp import direct_embed_residual_registry as embed_residual_registry  # noqa: E402  torch-free
from pitchmdp import direct_ema_registry as ema_registry  # noqa: E402  torch-free
from pitchmdp import direct_tabm_residual_registry as tabm_residual_registry  # noqa: E402  torch-free
from pitchmdp import direct_ple_registry as ple_registry  # noqa: E402  torch-free
from pitchmdp import direct_robust_clip_registry as robust_clip_registry  # noqa: E402  torch-free
from pitchmdp import direct_embed_type_registry as embed_type_registry  # noqa: E402  torch-free
from pitchmdp import direct_pitcher_id_registry as pid_registry  # noqa: E402  torch-free
from pitchmdp import batter_sidecar as bs  # noqa: E402  numpy only (P43 batter ids)
from pitchmdp.matrix_metrics import paired_game_comparison, pitch_losses, prediction_metrics  # noqa: E402

FORMAT = 'cuda_direct_score_v1'
SOURCE_FORMAT = 'cuda_direct_task_sources_v1'
SPLITS = {'dev': 'predictions.npz', 'mlb_dev': 'mlb_predictions.npz'}
EXPORT_ARRAYS = ('keys', 'y', 'game_pk')
DRAWS, BOOTSTRAP_SEED = 10000, 20260924
SCORER_SOURCES = ('scripts/score_cuda_direct.py', 'scripts/run_cuda_direct.py', 'pitchmdp/matrix_metrics.py',
                  'pitchmdp/direct_dropout_registry.py', 'pitchmdp/direct_combo_registry.py',
                  'pitchmdp/direct_frontier_registry.py', 'pitchmdp/direct_workload_dropout_registry.py',
                  'pitchmdp/direct_tabm_registry.py', 'pitchmdp/direct_embed_residual_registry.py',
                  'pitchmdp/direct_ema_registry.py', 'pitchmdp/direct_tabm_residual_registry.py',
                  'pitchmdp/direct_ple_registry.py', 'pitchmdp/direct_robust_clip_registry.py',
                  'pitchmdp/direct_embed_type_registry.py', 'pitchmdp/direct_pitcher_id_registry.py',
                  'pitchmdp/batter_sidecar.py')
RUNTIME_SIGNATURE = ('torch', 'torch_cuda', 'cudnn', 'numpy', 'scipy', 'device_name', 'device_capability',
                     'device_total_memory')
SCOPE = ('unblended calibrated direct-head probabilities vs the same-family CUDA control, before June blend / G0 mix; '
         'not G15 mix, not policy; descriptive repeated-DEV; no decision, ranking or winner')
HEX64 = re.compile(r'[0-9a-f]{64}')


def scorer_sources():
    return {rel: rc.sha256_file(PROJECT / rel) for rel in SCORER_SOURCES}


def task_sources(path=None):
    """Expected task source map: current runner by default, else an explicit pinned file (exact map)."""
    if path is None:
        return rc.own_hashes(), {'kind': 'current_runner_own_hashes'}
    data = rc.read_json(path)
    sources = data.get('source_sha256') if data.get('format') == SOURCE_FORMAT else None
    if not isinstance(sources, dict) or rc.SELF not in sources or rc.MODEL_SOURCES[0] not in sources:
        raise ValueError('Source manifest must be ' + SOURCE_FORMAT + ' with the runner and matrix_models hashes')
    if set(sources) - {rc.SELF, *rc.MODEL_SOURCES} or not all(isinstance(v, str) and HEX64.fullmatch(v)
                                                              for v in sources.values()):
        raise ValueError('Source manifest names unknown files or malformed hashes')
    return dict(sources), {'kind': 'source_manifest', 'path': str(Path(path).resolve()),
                           'sha256': rc.sha256_file(path), 'label': data.get('label')}


def export_rows(export, config, manifest_sha256):
    """dev/mlb_dev keys, labels and game ids from the pinned export (hash-checked files only)."""
    manifest = rc.verify_export(export, config, manifest_sha256, hash_arrays=False)
    rows, hashes = {}, {'train/rows_sha256': manifest['splits']['train']['rows_sha256']}
    for split in SPLITS:
        record = manifest['splits'][split]
        rows[split] = {}
        for name in EXPORT_ARRAYS:
            meta = record['files'][name]
            if rc.sha256_file(Path(export) / meta['path']) != meta['sha256']:
                raise ValueError(f'{split}/{name}: export array hash differs from the manifest')
            rows[split][name] = np.load(Path(export) / meta['path'], allow_pickle=False)
            hashes[f'{split}/{name}'] = meta['sha256']
        r = rows[split]
        if not (len(r['keys']) == len(r['y']) == len(r['game_pk']) == record['n']) \
                or not np.array_equal(r['keys'][:, 0], r['game_pk']):
            raise ValueError(f'{split}: export keys, labels and game ids are misaligned')
    return rows, hashes


def check_train_subset(config, arm, seed, fit, train_rows_sha256):
    """Data arms: the recorded subset is the config rule over the export TRAIN rows; others: full TRAIN."""
    spec, record = rc.arm_spec(config, arm), fit.get('train_subset')
    if 'train_fraction' not in spec:
        if record is not None or fit.get('train_rows_sha256') != train_rows_sha256:
            raise ValueError(f'{arm}/{seed}: full-TRAIN arm did not fit on the export TRAIN rows')
        return None
    expected = {'method': 'nested_key_hash', 'salt': config['train_subset']['salt'], 'fraction': spec['train_fraction'],
                'source_rows_sha256': train_rows_sha256}
    if not isinstance(record, dict) or any(record.get(k) != v for k, v in expected.items()) \
            or record.get('selected_rows_sha256') != fit.get('train_rows_sha256') \
            or not 0 < int(record.get('n_selected', 0)) < int(record.get('n_source', 0)):
        raise ValueError(f'{arm}/{seed}: TRAIN subset provenance differs from the config rule')
    return {k: record[k] for k in ('fraction', 'n_source', 'n_selected', 'selected_rows_sha256', 'selected_mask_sha256')}


def check_epoch_cap(config, arm, seed, fit):
    """TabM-residual fits: the recorded cap is the config cap and agrees with the fit report (no read of predictions)."""
    record, report, spec = fit.get('epoch_cap'), fit.get('report') or {}, config['training']
    if not isinstance(record, dict) or record.get('cap') != spec['epochs'] or record.get('patience') != spec['patience'] \
            or rc.epoch_record(report, spec['epochs'], spec['patience']) != record:
        raise ValueError(f'{arm}/{seed}: epoch cap record differs from the config cap or the fit report')
    return record


def check_ple_record(config, arm, seed, fit):
    """PLE fits: the top-level bins record equals the one re-derived from the fit's own network report."""
    spec, network = rc.arm_spec(config, arm), (fit.get('report') or {}).get('network') or {}
    try:
        expected = ple_registry.fit_record(network, config['export']['shapes']['length'],
                                           config['export']['context_names'], spec['workload'])
        effective = ple_registry.effective_record(network, fit.get('parameter_count'))
    except ValueError as error:
        raise ValueError(f'{arm}/{seed}: PLE bins record malformed ({error})') from error
    if {k: fit.get(k) for k in (*expected, *effective)} != {**expected, **effective}:
        raise ValueError(f'{arm}/{seed}: PLE bins record differs from the fit network report')
    return expected


def check_one_bins_hash(tasks):
    """All PLE fits of one family carry one bins hash (TRAIN-only bins; review N12); the shared record."""
    records = [t['ple'] for seeds in tasks.values() for t in seeds.values() if 'ple' in t]
    if not records or len({r['ple_bins_sha256'] for r in records}) != 1 \
            or len({json.dumps(r, sort_keys=True) for r in records}) != 1:
        raise ValueError('PLE fits of one family must carry one bins hash and one bins record')
    first = records[0]
    return {'ple_bins_sha256': first['ple_bins_sha256'], 'n_fits': len(records),
            'n_bins_total': sum(first['ple_n_bins']), 'degenerate_count': first['ple_degenerate_count'],
            'degenerate_columns': first['ple_degenerate_columns']}


def check_rc_record(config, arm, seed, fit):
    """Robust-clip fits: the top-level statistics record equals the one re-derived from the fit report; module pin."""
    length = config['export']['shapes']['length']
    try:
        expected = robust_clip_registry.fit_record(fit.get('report') or {}, length)
        rows = robust_clip_registry.check_stats_rows(fit.get('rc_stats_rows'), length)
    except ValueError as error:
        raise ValueError(f'{arm}/{seed}: robust-clip statistics record malformed ({error})') from error
    if {k: fit.get(k) for k in expected} != expected:
        raise ValueError(f'{arm}/{seed}: robust-clip statistics record differs from the fit report')
    if fit.get('rc_module_sha256') != robust_clip_registry.MODULE_SHA256:
        raise ValueError(f'{arm}/{seed}: robust-clip module sha differs from the registry pin')
    return {**expected, 'rc_stats_rows': rows, 'train_rows_sha256': fit.get('train_rows_sha256'),
            'rc_module_sha256': fit['rc_module_sha256']}


def check_one_stats_hash(tasks):
    """All robust-clip fits carry one statistics hash, one TRAIN row hash and one record (TRAIN-only statistics)."""
    records = [t['robust_clip'] for seeds in tasks.values() for t in seeds.values() if 'robust_clip' in t]
    if not records or len({r['rc_stats_sha256'] for r in records}) != 1 \
            or len({json.dumps(r, sort_keys=True) for r in records}) != 1:
        raise ValueError('Robust-clip fits of one family must carry one statistics hash and one statistics record')
    return {**records[0], 'n_fits': len(records)}


def same_gpu_pairs(tasks, contrasts, seeds):
    """Observed (candidate, reference, seed) pairs that ran on the same GPU, per contrast (P34 F2; descriptive)."""
    return {c['name']: sum(tasks[c['candidate']][s]['device_uuid'] == tasks[c['reference']][s]['device_uuid']
                           for s in seeds) for c in contrasts}


def verify_family(family, config, manifest_sha256, sources, train_rows_sha256=None):
    """Every arm x seed finished with the expected identity; nothing is read from predictions here."""
    family = Path(family)
    if family.name.endswith('-smoke'):
        raise ValueError('Smoke families are never scored')
    arms = [a['name'] for a in config['arms']]
    missing = [f'{a}/{s}' for a in arms for s in config['seeds']
               if not (rc.task_dir(family, a, s, False) / 'state.json').is_file()]
    if missing:
        raise ValueError('Incomplete family; nothing scored. Missing: ' + ', '.join(missing))
    tasks, signature = {}, None
    for arm in arms:
        tasks[arm] = {}
        for seed in config['seeds']:
            dest = rc.task_dir(family, arm, seed, False)
            state = rc.verify_task(dest, rc.expected_identity(config, manifest_sha256, arm, seed, False, 'cuda', sources))
            fit = rc.read_json(dest / 'fit.json')
            runtime = fit.get('runtime') or {}
            seen = {k: runtime.get(k) for k in RUNTIME_SIGNATURE}
            if fit.get('device') != 'cuda' or seen['torch_cuda'] is None or seen['device_name'] is None:
                raise ValueError(f'{arm}/{seed}: not a CUDA fit')
            if signature is None:
                signature = seen
            elif seen != signature:
                raise ValueError(f'{arm}/{seed}: CUDA runtime differs within the family')
            subset = check_train_subset(config, arm, seed, fit, train_rows_sha256) \
                if train_rows_sha256 is not None else None
            tasks[arm][seed] = {'state_sha256': rc.sha256_file(dest / 'state.json'), 'train_subset': subset,
                                'device_uuid': runtime.get('device_uuid'),
                                'delivery_temperature': rc.read_json(dest / 'calibration.json')['delivery_temperature']}
            if rc.uses_epoch_cap(config):
                tasks[arm][seed]['epoch_cap'] = check_epoch_cap(config, arm, seed, fit)
            if rc.uses_ple(config) and rc.is_ple_arm(rc.arm_spec(config, arm)):
                tasks[arm][seed]['ple'] = check_ple_record(config, arm, seed, fit)
                tasks[arm][seed]['ple_parameters'] = {'allocated': fit['parameter_count'],
                                                      'effective': fit['ple_effective_parameter_count']}
            if rc.uses_robust_clip(config) and rc.is_robust_clip_arm(rc.arm_spec(config, arm)):
                tasks[arm][seed]['robust_clip'] = check_rc_record(config, arm, seed, fit)
            if rc.uses_pitcher_id(config):
                tasks[arm][seed]['pitcher_id'] = pid_registry.check_fit(config, arm, fit)
            if rc.uses_embed_type(config):
                tasks[arm][seed]['cost'] = embed_type_registry.check_parameters(
                    arm, embed_type_registry.cost_record(fit))
    return tasks, signature


def cost_summary(tasks):
    """Per-arm cost across seeds (descriptive; from fit.json only) and per-seed candidate/reference ratios."""
    arms = {}
    for arm, seeds in tasks.items():
        values = {k: [seeds[s]['cost'][k] for s in seeds] for k in ('fit_seconds', 'optimizer_updates',
                                                                    'seconds_per_update', 'epochs_run', 'train_seconds',
                                                                    'train_seconds_per_epoch', 'train_seconds_per_update')}
        arms[arm] = {'parameter_count': embed_type_registry.PARAMETERS[arm],
                     **{f'{k}_median': float(np.median(v)) for k, v in values.items()},
                     **{f'{k}_min': float(np.min(v)) for k, v in values.items()},
                     **{f'{k}_max': float(np.max(v)) for k, v in values.items()}}
    ratios = {}
    for c in embed_type_registry.CONTRASTS:
        cand, ref = tasks[c['candidate']], tasks[c['reference']]
        ratios[c['name']] = {str(s): {k: cand[s]['cost'][k] / ref[s]['cost'][k]
                                      for k in ('fit_seconds', 'optimizer_updates', 'seconds_per_update',
                                                'train_seconds', 'train_seconds_per_epoch', 'train_seconds_per_update')}
                             for s in cand}
    return {'arms': arms, 'candidate_over_reference': ratios,
            'note': 'wall-clock depends on the GPU, co-running fit and early stopping; descriptive only. fit_seconds = fit '
                    'wall incl. export load/hash; train_seconds = training loop only (report.seconds)'}


def member(family, arm, seed, split, rows):
    """Calibrated probabilities of one member after exact key/label/game order checks."""
    with np.load(rc.task_dir(family, arm, seed, False) / SPLITS[split], allow_pickle=False) as data:
        for name, ours in (('keys', '_keys'), ('y', '_y'), ('game_pk', '_game_pk')):
            if not np.array_equal(data[split + ours], rows[name]):
                raise ValueError(f'{arm}/{seed}/{split}: {name} differ from the export rows')
        if not (data[split + '_delivery_level'] == rc.NO_PHYSICS_LEVEL).all():
            raise ValueError(f'{arm}/{seed}/{split}: not the no-physics direct head')
        probabilities = np.asarray(data[split], dtype=np.float64)
    pitch_losses(rows['y'], probabilities)  # shape / mass / range contract
    return probabilities


def summary(metrics):
    return {k: metrics[k] for k in ('n', 'log_loss', 'brier_multiclass', 'accuracy', 'top_label_ece10')}


def compare(y, games, candidate, control):
    return paired_game_comparison(y, candidate, control, games, draws=DRAWS, seed=BOOTSTRAP_SEED)


FACTOR_FREE = ('name', 'priority', 'combo', 'family', 'model', 'workload', 'context_continuous')


def check_contrasts(config):
    """Config-owned factorial contrasts (or None): distinct registered arms differing in exactly one factor.

    Validated before any prediction is read.  A contrast pairs a joint arm with its
    same-architecture no-workload arm (isolates workload) or its same-workload flatten
    arm (isolates architecture); data / pitcher-information arms never enter."""
    contrasts = config.get('comparison', {}).get('factorial_contrasts')
    if contrasts is None:
        return None
    if not isinstance(contrasts, list) or not contrasts:
        raise ValueError('factorial_contrasts must be a nonempty list')
    names, pairs = set(), set()
    for c in contrasts:
        if not isinstance(c, dict) or set(c) != {'name', 'candidate', 'reference', 'isolates'}:
            raise ValueError('Contrast needs exactly name / candidate / reference / isolates')
        cand, ref = rc.arm_spec(config, c['candidate']), rc.arm_spec(config, c['reference'])
        if c['candidate'] == c['reference'] or c['name'] in names or (c['candidate'], c['reference']) in pairs:
            raise ValueError('Contrast arms must be distinct and contrasts unique: ' + str(c['name']))
        rest = [{k: v for k, v in a.items() if k not in FACTOR_FREE} for a in (cand, ref)]
        if rest[0] != rest[1] or any('train_fraction' in a or 'drop_context' in a for a in (cand, ref)):
            raise ValueError(c['name'] + ': contrast arms differ beyond architecture / workload')
        same_model, same_workload = cand['model'] == ref['model'], cand['workload'] == ref['workload']
        isolates = 'workload' if same_model and not same_workload else \
            'architecture' if same_workload and not same_model and ref['model'] == 'flatten_mlp' else None
        if isolates is None or c['isolates'] != isolates:
            raise ValueError(c['name'] + ': contrast must change exactly the declared factor')
        names.add(c['name'])
        pairs.add((c['candidate'], c['reference']))
    return contrasts


def paired_block(y, games, candidate, reference, seeds):
    return {'ensemble': compare(y, games, np.mean(candidate, axis=0), np.mean(reference, axis=0)),
            'seeds': {str(s): compare(y, games, p, q) for s, p, q in zip(seeds, candidate, reference)}}


def score_family(config, family, rows, seeds, contrasts=None, regularization=None, workload_dropout=None, tabm=None,
                 embed_residual=None, ema=None, tabm_residual=None, ple=None, robust_clip=None, embed_type=None):
    arms = [a['name'] for a in config['arms']]
    control_arm = config.get('comparison', {}).get('control_arm', 'control')
    if control_arm != 'control' or 'control' not in arms:
        raise ValueError('The same-family CUDA control is the only comparator')
    out = {split: {} for split in SPLITS}
    factorial = {split: {} for split in SPLITS} if contrasts else None
    paired = {split: {} for split in SPLITS} if regularization else None
    joint = {split: {} for split in SPLITS} if workload_dropout else None
    head = {split: {} for split in SPLITS} if tabm else None
    embedded = {split: {} for split in SPLITS} if embed_residual else None
    averaged = {split: {} for split in SPLITS} if ema else None
    residual_head = {split: {} for split in SPLITS} if tabm_residual else None
    embedding_type = {split: {} for split in SPLITS} if ple else None
    preprocessing = {split: {} for split in SPLITS} if robust_clip else None
    embedding_map = {split: {} for split in SPLITS} if embed_type else None
    for split, r in rows.items():
        y, games = r['y'], r['game_pk']
        load = lambda arm: [member(family, arm, s, split, r) for s in seeds]  # noqa: E731  (bounded memory: reload)
        control = load(control_arm)
        out[split][control_arm] = {'role': 'control', 'ensemble': prediction_metrics(y, np.mean(control, axis=0)),
                                   'seeds': {str(s): summary(prediction_metrics(y, p)) for s, p in zip(seeds, control)}}
        for arm in arms:
            if arm == control_arm:
                continue
            candidate = load(arm)
            out[split][arm] = {
                'role': 'arm', 'ensemble': prediction_metrics(y, np.mean(candidate, axis=0)),
                'paired_vs_control': paired_block(y, games, candidate, control, seeds),
                'seeds': {str(s): summary(prediction_metrics(y, p)) for s, p in zip(seeds, candidate)}}
        for c in contrasts or ():
            factorial[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                           'isolates': c['isolates'],
                                           **paired_block(y, games, load(c['candidate']), load(c['reference']), seeds)}
        for c in regularization or ():
            paired[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                        'architecture': c['architecture'], 'isolates': c['isolates'],
                                        **paired_block(y, games, load(c['candidate']), load(c['reference']), seeds)}
        for c in workload_dropout or ():
            joint[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                       'isolates': c['isolates'], 'held': c['held'],
                                       **paired_block(y, games, load(c['candidate']), load(c['reference']), seeds)}
        for c in tabm or ():
            head[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                      'isolates': c['isolates'], 'held': c['held'],
                                      **paired_block(y, games, load(c['candidate']), load(c['reference']), seeds)}
        for c in embed_residual or ():
            embedded[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                          'isolates': c['isolates'], 'held': c['held'],
                                          **paired_block(y, games, load(c['candidate']), load(c['reference']), seeds)}
        for c in ema or ():
            averaged[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                          'isolates': c['isolates'], 'held': c['held'],
                                          **paired_block(y, games, load(c['candidate']), load(c['reference']), seeds)}
        for c in tabm_residual or ():
            residual_head[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                               'isolates': c['isolates'], 'held': c['held'],
                                               **paired_block(y, games, load(c['candidate']), load(c['reference']),
                                                              seeds)}
        for c in ple or ():
            embedding_type[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                                'isolates': c['isolates'], 'held': c['held'],
                                                **paired_block(y, games, load(c['candidate']), load(c['reference']),
                                                               seeds)}
        for c in robust_clip or ():
            preprocessing[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
                                               'isolates': c['isolates'], 'role': c['role'],
                                               'compute_matched': c['compute_matched'], 'held': c['held'],
                                               **paired_block(y, games, load(c['candidate']), load(c['reference']),
                                                              seeds)}
        for c in embed_type or ():
            embedding_map[split][c['name']] = {k: c[k] for k in ('candidate', 'reference', 'isolates', 'role',
                                                                  'parameter_matched', 'parameter_near_matched',
                                                                  'compute_matched', 'held')}
            embedding_map[split][c['name']].update(paired_block(y, games, load(c['candidate']),
                                                                load(c['reference']), seeds))
    if embed_type:
        return (out, factorial, paired, joint, head, embedded, averaged, residual_head, embedding_type, preprocessing,
                embedding_map)
    if robust_clip:
        return out, factorial, paired, joint, head, embedded, averaged, residual_head, embedding_type, preprocessing
    if ple:
        return out, factorial, paired, joint, head, embedded, averaged, residual_head, embedding_type
    if tabm_residual:
        return out, factorial, paired, joint, head, embedded, averaged, residual_head
    if ema:
        return out, factorial, paired, joint, head, embedded, averaged
    if embed_residual:
        return out, factorial, paired, joint, head, embedded
    if tabm:
        return out, factorial, paired, joint, head
    if workload_dropout:
        return out, factorial, paired, joint
    return out, factorial, paired


def korean_table(result):
    def ci(block, key):
        b = block[key]
        lo, hi = b['ci95'] or (float('nan'), float('nan'))
        return f"{b['delta']:+.5f} [{lo:+.5f}, {hi:+.5f}]"
    lines = [f"# {result['family']} 직접 헤드 기술 비교 (보정 확률, 혼합 전)", '',
             f'- 범위: 시드 {len(result.get("seed_block", {}).get("seeds", [0, 1, 2]))}개 평균 보정 확률 vs 같은 가족 CUDA 대조군. 6월 블렌드·G0 혼합·G15·정책 아님.',
             '- 음수 = 대조군보다 손실이 낮음. 경기 단위 부트스트랩 10,000회 95% 구간. 결정·순위·승자 없음(설정 순서).',
             f"- config {result['provenance']['config']['file_sha256'][:12]} · export "
             f"{result['provenance']['export']['manifest_sha256'][:12]} · 과제 소스 {result['provenance']['task_sources']['kind']}",
             '']
    for split, arms in result['splits'].items():
        control = arms['control']['ensemble']
        lines += [f"## {split} (n={control['n']}, 대조군 NLL {control['log_loss']:.5f}, Brier {control['brier_multiclass']:.5f})",
                  '', '| 팔 | ΔNLL [95%] | ΔBrier [95%] | 시드별 ΔNLL |', '|---|---|---|---|']
        for arm, block in arms.items():
            if block['role'] == 'control':
                continue
            paired = block['paired_vs_control']
            seeds = ' / '.join(f"{v['nll']['delta']:+.4f}" for v in paired['seeds'].values())
            lines.append(f"| {arm} | {ci(paired['ensemble'], 'nll')} | {ci(paired['ensemble'], 'brier')} | {seeds} |")
        lines.append('')
    for split, contrasts in (result.get('regularization_contrasts') or {}).items():
        lines += [f'## {split} 헤드 드롭아웃 추가 대비 (추가한 모델 − 기존 구조, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  'ResNet 기준 구조에는 블록 드롭아웃 0.1이 이미 있습니다. 이 대비는 출력 헤드에 0.1을 더한 효과입니다.', '',
                  '| 대비 | 구조 | ΔNLL [95%] | ΔBrier [95%] |', '|---|---|---|---|']
        for name, block in contrasts.items():
            lines.append(f"| {name} | {block['architecture']} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} |")
        lines.append('')
    for split, contrasts in (result.get('workload_dropout_contrasts') or {}).items():
        lines += [f'## {split} 작업량×헤드 드롭아웃 대비 (한 요인만 바꾼 짝, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  '| 대비 | 분리 요인 | 고정 | ΔNLL [95%] | ΔBrier [95%] |', '|---|---|---|---|---|']
        for name, block in contrasts.items():
            held = ', '.join(f'{k}={v}' for k, v in block['held'].items())
            lines.append(f"| {name} | {block['isolates']} | {held} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} |")
        lines.append('')
    for split, contrasts in (result.get('tabm_contrasts') or {}).items():
        lines += [f'## {split} TabM-mini 헤드·헤드 드롭아웃 대비 (한 요인만 바꾼 짝, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  'ensemble_head 행은 TabM-mini 1회 학습(구성원 8)과 단일 신경망 1회 학습의 비교입니다. 연산량은 같지 않습니다. '
                  'head_dropout 행은 단일 신경망끼리의 비교입니다.', '',
                  f"시드 {', '.join(map(str, result['seed_block']['seeds']))}는 새 모델 난수일 뿐 새 데이터가 아닙니다"
                  '(같은 DEV·MLB DEV 반복 노출). counts 작업량은 P27 결과를 읽은 뒤 고른 사후 선택입니다.', '',
                  '| 대비 | 분리 요인 | 고정 | ΔNLL [95%] | ΔBrier [95%] |', '|---|---|---|---|---|']
        for name, block in contrasts.items():
            held = ', '.join(f'{k}={v}' for k, v in block['held'].items())
            lines.append(f"| {name} | {block['isolates']} | {held} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} |")
        lines.append('')
    for split, contrasts in (result.get('embed_residual_contrasts') or {}).items():
        lines += [f'## {split} 임베딩·잔차 구조 대비 (후보 − 기준, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  'backbone_bundle 행은 구조와 드롭아웃(0 → 블록 0.1)을 함께 바꿉니다. backbone_structure 행은 드롭아웃 비율 0.1을 '
                  '맞췄지만 위치(헤드 vs 블록)는 다릅니다. 같은 시드라 backbone_bundle = backbone_structure + head_dropout, '
                  'input = embedding_type + embedding이 정확히 성립해 여섯 대비 중 독립은 넷입니다.', '',
                  f"시드 {', '.join(map(str, result['seed_block']['seeds']))}는 새 모델 난수일 뿐 새 데이터가 아닙니다"
                  '(같은 DEV·MLB DEV 반복 노출). counts 작업량과 이 가족은 P24~P31 결과가 나온 뒤 설계한 사후 탐색입니다.', '',
                  '| 대비 | 분리 요인 | 고정 | ΔNLL [95%] | ΔBrier [95%] |', '|---|---|---|---|---|']
        for name, block in contrasts.items():
            held = ', '.join(f'{k}={v}' for k, v in block['held'].items())
            lines.append(f"| {name} | {block['isolates']} | {held} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} |")
        lines.append('')
    for split, contrasts in (result.get('ema_contrasts') or {}).items():
        lines += [f'## {split} 가중치 평균(EMA) 대비 (후보 − 기준, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  'EMA 팔은 같은 시드에서 원래 가중치의 학습 경로가 기준 팔과 같고(CPU 테스트, CUDA는 ema_trajectory_check 기록), '
                  '평가·멈춤·저장하는 가중치가 다릅니다. 그래서 멈추는 epoch와 update 수가 달라질 수 있습니다. '
                  'dropout_vs_control은 P25·P29 대비의 재현 확인용입니다.', '',
                  f"시드 {', '.join(map(str, result['seed_block']['seeds']))}는 새 모델 난수일 뿐 새 데이터가 아닙니다"
                  '(같은 DEV·MLB DEV 반복 노출). 이 가족은 P24~P32 결과를 읽은 뒤 설계한 탐색입니다.', '',
                  '| 대비 | 분리 요인 | 고정 | ΔNLL [95%] | ΔBrier [95%] |', '|---|---|---|---|---|']
        for name, block in contrasts.items():
            held = ', '.join(f'{k}={v}' for k, v in block['held'].items())
            lines.append(f"| {name} | {block['isolates']} | {held} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} |")
        lines.append('')
    for split, contrasts in (result.get('tabm_residual_contrasts') or {}).items():
        lines += [f'## {split} TabM-mini × 잔차 구조 대비 (후보 − 기준, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  'ensemble_head 행은 잔차 구조에서 TabM-mini 1회 학습(구성원 8)과 단일 신경망 1회 학습의 비교입니다. 연산량은 같지 않습니다. '
                  'backbone_bundle_d10 행은 TabM-mini에서 드롭아웃 0.1을 양쪽에 두고 구조·구성원 헤드 입력 폭·드롭아웃 위치(헤드 → 블록)를 함께 바꿉니다. '
                  'backbone_bundle 행은 TabM-mini에서 구조와 드롭아웃(헤드 0 → 블록 0.1)을 함께 바꿉니다. 두 행 모두 한 요인 비교가 아닙니다. '
                  '*_vs_control 행은 여러 요인이 함께 바뀐 묶음 비교입니다.', '',
                  f"시드 {', '.join(map(str, result['seed_block']['seeds']))}는 새 모델 난수일 뿐 새 데이터가 아닙니다"
                  '(같은 DEV·MLB DEV 반복 노출). 이 가족은 P30~P33 결과를 읽은 뒤 설계한 사후 탐색입니다. '
                  '모든 팔의 epoch 상한은 60이라 P31·P32 숫자와 비교하지 않습니다.', '',
                  '| 대비 | 분리 요인 | 고정 | ΔNLL [95%] | ΔBrier [95%] |', '|---|---|---|---|---|']
        for name, block in contrasts.items():
            held = ', '.join(f'{k}={v}' for k, v in block['held'].items())
            lines.append(f"| {name} | {block['isolates']} | {held} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} |")
        lines.append('')
    ple = result.get('ple_contrasts')
    if ple:
        lines += ['## PLE 임베딩 대비 — 주요 행 (ΔNLL [95%], dev / mlb_dev 나란히, 판정 없음)', '',
                  '| 대비 | dev ΔNLL [95%] | mlb_dev ΔNLL [95%] |', '|---|---|---|']
        for name in result['seed_block']['headline']:
            lines.append(f"| {name} | {ci(ple['dev'][name]['ensemble'], 'nll')} | "
                         f"{ci(ple['mlb_dev'][name]['ensemble'], 'nll')} |")
        bins = result['ple_bins']
        lines += ['', f"PLE 구간: 모든 PLE fit {bins['n_fits']}개가 같은 구간 해시 {bins['ple_bins_sha256'][:12]} "
                      f"(TRAIN만 사용). 값이 하나뿐인 열(T_j = 0) {bins['degenerate_count']}개"
                      + (f": {', '.join(bins['degenerate_columns'])}" if bins['degenerate_columns'] else '') + '.', '']
    for split, contrasts in (ple or {}).items():
        pairs = result['seed_block']['same_gpu_pairs']
        lines += [f'## {split} PLE 임베딩 대비 (후보 − 기준, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  'embedding_type 행은 주기 임베딩(PLR-lite)을 PLE로 바꾼 묶음(함수·파라미터 수·TRAIN 구간·활성화)입니다. '
                  'ensemble_head_ple 행은 PLE에서 TabM-mini 1회 학습(구성원 8)과 단일 신경망 1회 학습의 비교로 연산량이 같지 않습니다. '
                  'bundle 행은 여러 요인이 함께 바뀐 대조군 비교입니다.', '',
                  f"시드 {', '.join(map(str, result['seed_block']['seeds']))}는 새 모델 난수일 뿐 새 데이터가 아닙니다"
                  '(같은 DEV·MLB DEV 반복 노출). 이 가족은 P24~P34 결과를 읽은 뒤 설계한 사후 탐색입니다. '
                  '모든 팔의 epoch 상한은 60입니다.', '',
                  '| 대비 | 분리 요인 | 고정 | ΔNLL [95%] | ΔBrier [95%] | 같은 GPU 짝 |', '|---|---|---|---|---|---|']
        for name, block in contrasts.items():
            held = ', '.join(f'{k}={v}' for k, v in block['held'].items())
            lines.append(f"| {name} | {block['isolates']} | {held} | {ci(block['ensemble'], 'nll')} | "
                         f"{ci(block['ensemble'], 'brier')} | {pairs[name]}/{len(result['seed_block']['seeds'])} |")
        lines.append('')
    robust = result.get('robust_clip_contrasts')
    for split, contrasts in (robust or {}).items():
        pairs, stats = result['seed_block']['same_gpu_pairs'], result['robust_clip_stats']
        lines += [f'## {split} 강건 스케일 + 부드러운 자르기 대비 (후보 − 기준, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  'input_preprocessing 행(주요)은 같은 잔차 모델에서 입력 처리만 바꾼 비교로, 같은 시드면 초기 파라미터가 같습니다. '
                  'numeric_input_treatment 행(보조)은 주기 임베딩 팔과의 비교로 파라미터 수가 달라 연산량이 같지 않으며 고르는 데 쓰지 않습니다.', '',
                  f"시드 {', '.join(map(str, result['seed_block']['seeds']))}는 새 모델 난수일 뿐 새 데이터가 아닙니다"
                  '(같은 DEV·MLB DEV 반복 노출). 이 가족은 P24~P34 결과를 읽은 뒤 설계한 사후 탐색입니다. '
                  '모든 팔의 epoch 상한은 60입니다.', '',
                  '| 대비 | 역할 | 분리 요인 | 연산량 같음 | ΔNLL [95%] | ΔBrier [95%] | 같은 GPU 짝 |',
                  '|---|---|---|---|---|---|---|']
        for name, block in contrasts.items():
            lines.append(f"| {name} | {block['role']} | {block['isolates']} | {block['compute_matched']} | "
                         f"{ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} | "
                         f"{pairs[name]}/{len(result['seed_block']['seeds'])} |")
        lines += ['', f"통계: 강건 자르기 fit {stats['n_fits']}개 모두 같은 통계 해시 {stats['rc_stats_sha256'][:12]} "
                      f"(TRAIN {stats['rc_stats_rows']['train_rows']}행만 사용), 규칙 수 {stats['rc_rule_counts']}.", '']
    mapped = result.get('embed_type_contrasts')
    for split, contrasts in (mapped or {}).items():
        pairs = result['seed_block']['same_gpu_pairs']
        lines += [f'## {split} 임베딩 함수 대비 (periodic − linear, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  '같은 잔차 모델·같은 임베딩 열·같은 임베딩 폭 8에서 사인/코사인(periodic) 대신 열별 1차식(linear)만 바꾼 비교입니다. '
                  '파라미터 수는 거의 같지만(+264) 연산량·시간은 같지 않습니다. 시간은 아래 표로만 보고하고 NLL 비교를 보정하지 않습니다.', '',
                  f"시드 {', '.join(map(str, result['seed_block']['seeds']))}는 새 모델 난수일 뿐 새 데이터가 아닙니다"
                  '(같은 DEV·MLB DEV 반복 노출, P32에서 같은 대비를 3시드로 한 번 봄). 모든 팔의 epoch 상한은 60입니다.', '',
                  '| 대비 | 분리 요인 | 파라미터 거의 같음 | 연산량 같음 | ΔNLL [95%] | ΔBrier [95%] | 같은 GPU 짝 |',
                  '|---|---|---|---|---|---|---|']
        for name, block in contrasts.items():
            lines.append(f"| {name} | {block['isolates']} | {block['parameter_near_matched']} | "
                         f"{block['compute_matched']} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} | "
                         f"{pairs[name]}/{len(result['seed_block']['seeds'])} |")
        lines.append('')
    if mapped:
        lines += ['## 비용 (fit.json, 시드 중앙값 [최소, 최대])', '',
                  '| 팔 | 파라미터 | fit 초 | 학습 루프 초 | 학습 epoch당 초 | 갱신 수 | 갱신당 초 | epoch |',
                  '|---|---|---|---|---|---|---|---|']
        for arm, c in result['cost']['arms'].items():
            cell = lambda k, f='{:.1f}': f"{f.format(c[k + '_median'])} [{f.format(c[k + '_min'])}, {f.format(c[k + '_max'])}]"  # noqa: E731
            lines.append(f"| {arm} | {c['parameter_count']} | {cell('fit_seconds')} | {cell('train_seconds')} | "
                         f"{cell('train_seconds_per_epoch', '{:.2f}')} | {cell('optimizer_updates', '{:.0f}')} | "
                         f"{cell('seconds_per_update', '{:.4f}')} | {cell('epochs_run', '{:.0f}')} |")
        lines.append('')
    if result.get('tabm_residual_contrasts') or ple or robust or mapped:
        caps = {arm: {seed: task['epoch_cap'] for seed, task in seeds.items()}
                for arm, seeds in result['provenance']['tasks'].items()}
        lines += ['## epoch 상한(60) 도달 여부 (fit별)', '', '| 팔 | 시드별 epochs_run / best_epoch / 멈춘 이유 |', '|---|---|']
        for arm, seeds in caps.items():
            cells = ' · '.join(f"{s}: {r['epochs_run']}/{r['best_epoch']}/{r['stopped_by']}" for s, r in seeds.items())
            lines.append(f'| {arm} | {cells} |')
        lines.append('')
    for split, contrasts in (result.get('factorial_contrasts') or {}).items():
        lines += [f'## {split} 요인 대비 (후보 − 기준, 같은 시드·앙상블 짝 비교, 판정 없음)', '',
                  '| 대비 | 분리 요인 | ΔNLL [95%] | ΔBrier [95%] |', '|---|---|---|---|']
        for name, block in contrasts.items():
            lines.append(f"| {name} | {block['isolates']} | {ci(block['ensemble'], 'nll')} | {ci(block['ensemble'], 'brier')} |")
        lines.append('')
    return '\n'.join(lines)


def score(config_path, config_sha256, export, manifest_sha256, family, destination, source_manifest=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'{destination} exists; scores are immutable')
    if rc.sha256_file(config_path) != config_sha256:
        raise ValueError('Config differs from the pinned SHA256')
    config = rc.config_check(rc.read_json(config_path))
    if config.get('status') != 'REGISTERED' or config.get('execution', {}).get('enabled') is not True:
        raise ValueError('Only a REGISTERED, enabled family has real tasks to score')
    contrasts = check_contrasts(config)
    regularization = dropout_registry.check_contrasts(config)
    workload_dropout = workload_dropout_registry.check_contrasts(config)
    tabm = tabm_registry.check_contrasts(config)
    embed_residual = embed_residual_registry.check_contrasts(config)
    ema = ema_registry.check_contrasts(config)
    tabm_residual = tabm_residual_registry.check_contrasts(config)
    ple = ple_registry.check_contrasts(config)
    robust_clip = robust_clip_registry.check_contrasts(config)
    embed_type = embed_type_registry.check_contrasts(config)
    pitcher_id = pid_registry.check_contrasts(config)
    others = (tabm_residual, ema, embed_residual, tabm, workload_dropout, contrasts, regularization)
    if rc.uses_pitcher_id(config):  # before every family: both arms keep the embed_residual flag
        if pitcher_id is None or any(o is not None for o in (embed_type, robust_clip, ple, *others)):
            raise ValueError('The pitcher-ID family is scored only with its fixed contrast')
    elif pitcher_id is not None:
        raise ValueError('pitcher_id_contrasts belong to the pitcher-ID family only')
    elif rc.uses_embed_type(config):  # first: both embedded arms keep the embed_residual flag
        if embed_type is None or robust_clip is not None or ple is not None or any(o is not None for o in others):
            raise ValueError('The embed-type family is scored only with its fixed contrast')
    elif embed_type is not None:
        raise ValueError('embed_type_contrasts belong to the embed-type family only')
    elif rc.uses_robust_clip(config):  # first: its periodic_residual_counts keeps the embed_residual flag
        if robust_clip is None or ple is not None or any(o is not None for o in others):
            raise ValueError('The robust-clip family is scored only with its fixed contrasts')
    elif robust_clip is not None:
        raise ValueError('robust_clip_contrasts belong to the robust-clip family only')
    elif rc.uses_ple(config):  # first: the PLE family's arm tabm_residual_counts keeps the P34 flag
        if ple is None or any(o is not None for o in others):
            raise ValueError('The PLE family is scored only with its fixed contrasts')
    elif ple is not None:
        raise ValueError('ple_contrasts belong to the PLE family only')
    elif rc.uses_tabm_residual(config):
        if tabm_residual is None or ema is not None or embed_residual is not None or tabm is not None \
                or workload_dropout is not None or contrasts is not None or regularization is not None:
            raise ValueError('The TabM-residual family is scored only with its fixed contrasts')
    elif tabm_residual is not None:
        raise ValueError('tabm_residual_contrasts belong to the TabM-residual family only')
    elif rc.uses_ema(config):
        if ema is None or embed_residual is not None or tabm is not None or workload_dropout is not None \
                or contrasts is not None or regularization is not None:
            raise ValueError('The EMA family is scored only with its fixed contrasts')
    elif ema is not None:
        raise ValueError('ema_contrasts belong to the EMA family only')
    elif rc.uses_tabm(config):
        if tabm is None or workload_dropout is not None or contrasts is not None or regularization is not None \
                or embed_residual is not None:
            raise ValueError('The TabM-mini family is scored only with its fixed contrasts')
    elif rc.uses_embed_residual(config):
        if embed_residual is None or tabm is not None or workload_dropout is not None or contrasts is not None \
                or regularization is not None:
            raise ValueError('The embedded-residual family is scored only with its fixed contrasts')
    elif rc.uses_workload_dropout(config) and workload_dropout is None:
        raise ValueError('The workload-dropout family is scored only with its fixed contrasts')
    sources, source_record = task_sources(source_manifest)
    rows, export_hashes = export_rows(export, config, manifest_sha256)
    tasks, signature = verify_family(family, config, manifest_sha256, sources,
                                     export_hashes['train/rows_sha256'] if config.get('train_subset') else None)
    if pitcher_id:
        return score_pitcher_id(config, config_path, config_sha256, export, manifest_sha256, family, destination,
                                rows, export_hashes, tasks, signature, sources, source_record)
    ple_bins = check_one_bins_hash(tasks) if ple else None
    rc_stats = check_one_stats_hash(tasks) if robust_clip else None
    splits, factorial, paired, *joint = score_family(config, family, rows, config['seeds'], contrasts, regularization,
                                                     workload_dropout, tabm, embed_residual, ema, tabm_residual, ple,
                                                     robust_clip, embed_type)
    result = {'format': FORMAT, 'family': config['experiment_id'], 'scope': SCOPE, 'decision': None, 'ranking': None,
              'arm_order': 'config order', 'splits': splits,
              **({'factorial_contrasts': factorial} if contrasts else {}),
              **({'regularization_contrasts': paired} if regularization else {}),
              **({'workload_dropout_contrasts': joint[0], 'seed_block': {'seeds': config['seeds'],
                  'repeat_tag': config.get('repeat_tag'),
                  'note': 'fresh model RNG seeds on the same export and exposed DEV rows: not new data, '
                          'not an independent confirmatory test'}} if workload_dropout else {}),
              **({'tabm_contrasts': joint[1], 'seed_block': {'seeds': config['seeds'], 'repeat_tag': None,
                  'note': 'fresh model RNG seeds on the same export and exposed DEV rows: not new data, '
                          'not an independent confirmatory test; workload counts chosen after reading P27 '
                          '(post-selection); one TabM-mini fit (k=8) vs one single-network fit is not '
                          'compute-matched'}} if tabm else {}),
              **({'embed_residual_contrasts': joint[2], 'seed_block': {'seeds': config['seeds'], 'repeat_tag': None,
                  'note': 'fresh model RNG seeds on the same export and exposed DEV rows: not new data, '
                          'not an independent confirmatory test; family designed after reading P24-P31 '
                          '(post-selection); backbone_bundle changes structure and dropout together, '
                          'backbone_structure still differs in dropout position; input = embedding_type + '
                          'embedding; backbone_bundle = backbone_structure + head_dropout'}} if embed_residual else {}),
              **({'ema_contrasts': joint[3], 'seed_block': {'seeds': config['seeds'], 'repeat_tag': None,
                  'note': 'fresh model RNG seeds on the same export and exposed DEV rows: not new data, '
                          'not an independent confirmatory test; family designed after reading P24-P32; '
                          'EMA arms share the raw trajectory of their base arm per seed (CPU tests; CUDA: '
                          'ema_trajectory_check), the evaluated, stopped and saved weights differ, so epochs '
                          'run and optimizer updates can differ (EMA usually stops later)'}} if ema else {}),
              **({'tabm_residual_contrasts': joint[4], 'seed_block': {'seeds': config['seeds'], 'repeat_tag': None,
                  'epoch_cap': config['training']['epochs'],
                  'note': 'fresh model RNG seeds on the same export and exposed DEV rows: not new data, '
                          'not an independent confirmatory test; family designed after reading P30-P33 '
                          '(post-selection); epoch cap 60 for every arm, so numbers are not compared with '
                          'P31/P32; one TabM-mini fit (k=8) vs one single-network fit is not compute-matched; '
                          'backbone_bundle_d10 holds dropout .1 but changes backbone, member-head input width '
                          '(256 -> 128) and dropout placement (head -> blocks) together; '
                          'backbone_bundle changes backbone and dropout (head 0 -> blocks .1) together; '
                          'neither is one-factor; per-fit cap hits in provenance.tasks.*.epoch_cap'}} if tabm_residual else {}),
              **({'ple_contrasts': joint[5], 'ple_bins': ple_bins, 'seed_block': {
                  'seeds': config['seeds'], 'repeat_tag': None, 'epoch_cap': config['training']['epochs'],
                  'headline': list(ple_registry.HEADLINE),
                  'same_gpu_pairs': same_gpu_pairs(tasks, ple, config['seeds']),
                  'note': 'fresh model RNG seeds on the same export and exposed DEV rows: not new data, '
                          'not an independent confirmatory test; family designed after reading P24-P34 '
                          '(post-selection); epoch cap 60 for every arm; embedding_type rows change the whole '
                          'embedding bundle (map, parameter count, TRAIN-fitted bins, no activation) together; '
                          'one TabM-mini fit (k=8) vs one single-network fit is not compute-matched; bundle rows '
                          'change several factors; same_gpu_pairs is observed, not scheduled; per-fit cap hits in '
                          'provenance.tasks.*.epoch_cap'}} if ple else {}),
              **({'robust_clip_contrasts': joint[6], 'robust_clip_stats': rc_stats, 'seed_block': {
                  'seeds': config['seeds'], 'repeat_tag': None, 'epoch_cap': config['training']['epochs'],
                  'primary': robust_clip_registry.PRIMARY, 'secondary': robust_clip_registry.SECONDARY,
                  'same_gpu_pairs': same_gpu_pairs(tasks, robust_clip, config['seeds']),
                  'note': 'fresh model RNG seeds on the same export and exposed DEV rows: not new data, '
                          'not an independent confirmatory test; family designed after reading P24-P34 '
                          '(post-selection); epoch cap 60 for every arm; the primary row changes only the input '
                          'preprocessing (same seed, identical initial parameters); the secondary row compares two '
                          'numeric-input treatments and is not compute-matched (the periodic arm adds embedding '
                          'parameters); same_gpu_pairs is observed, not scheduled; per-fit cap hits in '
                          'provenance.tasks.*.epoch_cap'}} if robust_clip else {}),
              **({'embed_type_contrasts': joint[7], 'cost': cost_summary(tasks), 'seed_block': {
                  'seeds': config['seeds'], 'repeat_tag': None, 'epoch_cap': config['training']['epochs'],
                  'primary': embed_type_registry.PRIMARY, 'parameters': dict(embed_type_registry.PARAMETERS),
                  'same_gpu_pairs': same_gpu_pairs(tasks, embed_type, config['seeds']),
                  'note': 'fresh model RNG seeds on the same export and exposed DEV rows: not new data, '
                          'not an independent confirmatory test; family designed after reading P24-P36 '
                          '(post-selection); the same pair ran once in EXP-P32-001 (seeds 16-18, cap 30), so this '
                          'is a repeat on the same rows, not independent evidence; epoch cap 60 for every arm; the '
                          'pair changes only the embedding map (same embedded columns, width 8, backbone input '
                          'width, backbone, dropout, head) but is parameter-near-matched (+264 parameters) and NOT '
                          'compute-matched (periodic map does more work per row; early stopping can change epochs '
                          'and updates); cost is reported per fit and never adjusts the NLL contrast; '
                          'same_gpu_pairs is observed, not scheduled'}} if embed_type else {}),
              'limits': ['Cpanel DEV and mlb_dev already exposed: repeated-DEV, descriptive only',
                         'Bootstrap conditions on fixed predictions (no refit, calibration or selection uncertainty)',
                         'Prediction only; no 2026 rows; no policy or service claim'],
              'provenance': {
                  'config': {'path': str(Path(config_path).resolve()), 'file_sha256': config_sha256,
                             'digest': rc.config_digest(config), 'experiment_id': config['experiment_id']},
                  'export': {'path': str(Path(export).resolve()), 'manifest_sha256': manifest_sha256,
                             'scored_arrays_sha256': export_hashes},
                  'task_sources': {**source_record, 'source_sha256': sources},
                  'task_runtime_signature': signature, 'tasks': {a: {str(s): v for s, v in t.items()}
                                                                 for a, t in tasks.items()},
                  'scorer_sources_sha256': scorer_sources(),
                  'scorer_runtime': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                                     'hostname': platform.node()},
                  'bootstrap': {'draws': DRAWS, 'seed': BOOTSTRAP_SEED, 'function': 'matrix_metrics.paired_game_comparison'},
                  'created_utc': datetime.now(timezone.utc).isoformat()}}
    if scorer_sources() != result['provenance']['scorer_sources_sha256']:
        raise RuntimeError('Scorer sources changed during scoring')
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.with_name(f'.{destination.name}.tmp-{os.getpid()}')
    work.mkdir()
    rc.write_json_atomic(work / 'score.json', result)
    (work / 'table_ko.md').write_text(korean_table(result) + '\n')
    (work / 'SHA256SUMS').write_text(''.join(f'{rc.sha256_file(work / n)}  {n}\n' for n in ('score.json', 'table_ko.md')))
    for name in ('score.json', 'table_ko.md', 'SHA256SUMS'):
        os.chmod(work / name, 0o444)
    if destination.exists():
        raise FileExistsError(f'{destination} appeared during scoring')
    os.rename(work, destination)
    return result


def export_pitchers(export, config, manifest_sha256):
    """Hash-checked `pitcher` (all splits) and TRAIN `date` arrays from the pinned export (no labels read here)."""
    manifest = rc.verify_export(export, config, manifest_sha256, hash_arrays=False)
    out, hashes = {}, {}
    for split, names in (('train', ('pitcher', 'date')), *((s, ('pitcher',)) for s in SPLITS)):
        for name in names:
            meta = manifest['splits'][split]['files'][name]
            if rc.sha256_file(Path(export) / meta['path']) != meta['sha256']:
                raise ValueError(f'{split}/{name}: export array hash differs from the manifest')
            out[f'{split}/{name}'] = np.load(Path(export) / meta['path'], allow_pickle=False)
            hashes[f'{split}/{name}'] = meta['sha256']
    return manifest, out, hashes


P40_NOTE = ('fresh model RNG seeds 60-64 on the same export and exposed DEV rows: not new data, not an independent '
            'confirmatory test; designed after reading the P38 score (post-selection). Primary contrast = candidate vs '
            'row-shuffled-id control (same network, parameters and compute; pitcher column permuted across rows per '
            'split by a fixed permutation): isolates identity information from the id path. The row-shuffle control '
            'matches the id path, not a generic wider backbone; ID dropout 0.2 is fixed, not tuned. Vocabulary is '
            'TRAIN-only and re-derived here; subsets use TRAIN-vocabulary membership of the TRUE pitcher; all seeds, '
            'subsets and contrasts reported, no verdict')


P43_NOTE = ('fresh model RNG seeds 70-74 on the same export and exposed DEV rows: not new data, not an independent '
            'confirmatory test; designed after reading P25 and P38 (post-selection). Primary contrast = batter id vs '
            'row-shuffled batter id (same network, parameters and compute; batter column permuted across rows per '
            'split by a fixed permutation): isolates batter identity information from the id path. Batter ids come '
            'from a sha-pinned sidecar joined by the export keys (ids only). Vocabularies are TRAIN-only per column '
            'and re-derived here; subsets use TRAIN-vocabulary membership of the TRUE batter; all seeds, subsets and '
            'contrasts reported, no verdict')


def p43_parameters(arms, vocab):
    """Allocated parameters per P43 arm: base + one id path per column the arm reads."""
    return {arm: pid_registry.BASE_PARAMETERS + sum(vocab[c]['id_parameters'] for c in (
        pid_registry.BASELINES[arm].get(pid_registry.FLAG, {}).get('id_columns', []))) for arm in arms}


def score_pitcher_id(config, config_path, config_sha256, export, manifest_sha256, family, destination, rows,
                     export_hashes, tasks, signature, sources, source_record):
    """Two-arm pitcher-ID family: candidate - reference on all / in-vocabulary / UNK rows, every seed + ensemble."""
    p43 = pid_registry.is_p43(config)
    manifest, arrays, pitcher_hashes = export_pitchers(export, config, manifest_sha256)
    pid_registry.check_split_dates(manifest)
    if p43:  # one TRAIN vocabulary per id column; subsets by the TRUE batter's TRAIN-vocabulary membership
        vocab = pid_registry.check_column_vocabs({a: {s: t['pitcher_id'] for s, t in v.items()} for a, v in tasks.items()})
        directory, record = rc.batter_sidecar(export, config, manifest, hash_arrays=True)
        for split in ('train', *SPLITS):
            arrays[split + '/batter'] = np.asarray(bs.load(directory, record, split))
            pitcher_hashes[split + '/batter'] = record['splits'][split]['sha256']
        pitcher_hashes['batter_sidecar.json'] = config[pid_registry.BATTER_SIDECAR_KEY]['sidecar_sha256']
        rederived_all = {c: pid_registry.build_vocab(arrays['train/' + c], arrays['train/date'],
                                                     pid_registry.LAST_TRAIN_DATE) for c in ('pitcher', 'batter')}
        for c, one in rederived_all.items():
            if {k: one[k] for k in pid_registry.VOCAB_KEYS} != {k: vocab[c][k] for k in pid_registry.VOCAB_KEYS}:
                raise ValueError(f'Recorded {c} vocabulary differs from the one re-derived from the export TRAIN rows')
        subset_column, rederived = pid_registry.P43_SUBSET_COLUMN, rederived_all[pid_registry.P43_SUBSET_COLUMN]
    else:
        vocab = pid_registry.check_one_vocab({a: {s: t['pitcher_id'] for s, t in v.items()} for a, v in tasks.items()})
        rederived = pid_registry.build_vocab(arrays['train/pitcher'], arrays['train/date'], pid_registry.LAST_TRAIN_DATE)
        if {k: rederived[k] for k in pid_registry.VOCAB_KEYS} != {k: vocab[k] for k in pid_registry.VOCAB_KEYS}:
            raise ValueError('Recorded vocabulary differs from the one re-derived from the export TRAIN rows')
        subset_column = 'pitcher'
    seeds, design = config['seeds'], pid_registry.design(config)
    arms = [a['name'] for a in config['arms']]
    roles = {pid_registry.REFERENCE: 'reference', pid_registry.CANDIDATE: 'candidate'}
    splits, contrasts, subsets = {}, {}, {}
    for split, r in rows.items():
        index = pid_registry.map_ids(arrays[split + '/' + subset_column], rederived['ids'])
        if len(index) != len(r['y']):
            raise ValueError(f'{split}: export {subset_column} ids misaligned with the scored rows')
        probs = {arm: [member(family, arm, s, split, r) for s in seeds] for arm in arms}
        splits[split] = {arm: {'role': roles.get(arm, 'control'),
                               'ensemble': prediction_metrics(r['y'], np.mean(p, axis=0)),
                               'seeds': {str(s): summary(prediction_metrics(r['y'], q)) for s, q in zip(seeds, p)}}
                         for arm, p in probs.items()}
        contrasts[split] = {c['name']: {} for c in design['contrasts']}
        subsets[split] = {}
        for subset, mask in (('all', np.ones(len(index), bool)), ('in_vocab', index != pid_registry.UNK),
                             ('unk', index == pid_registry.UNK)):
            subsets[split][subset] = {'rows': int(mask.sum()), 'games': int(len(np.unique(r['game_pk'][mask])))}
            if subset != 'all' and mask.any():
                subsets[split][subset]['arms'] = {
                    arm: summary(prediction_metrics(r['y'][mask], np.mean(p, axis=0)[mask])) for arm, p in probs.items()}
            for c in design['contrasts']:
                if not mask.any():
                    contrasts[split][c['name']][subset] = None
                    continue
                cand, ref = probs[c['candidate']], probs[c['reference']]
                contrasts[split][c['name']][subset] = {
                    **{k: c[k] for k in ('candidate', 'reference', 'isolates', 'role', 'parameter_matched',
                                         'compute_matched', 'information_free_capacity_control')},
                    **paired_block(r['y'][mask], r['game_pk'][mask], [p[mask] for p in cand], [p[mask] for p in ref],
                                   seeds)}
        del probs
    result = {'format': FORMAT, 'family': config['experiment_id'], 'scope': SCOPE, 'decision': None, 'ranking': None,
              'arm_order': 'config order', 'splits': splits, 'pitcher_id_contrasts': contrasts,
              'pitcher_id_rows': subsets, 'vocabulary': {k: v for k, v in vocab.items()},
              'cost': {arm: {str(s): t['pitcher_id']['cost'] for s, t in v.items()} for arm, v in tasks.items()},
              'seed_block': {'seeds': seeds, 'repeat_tag': None, 'epoch_cap': config['training']['epochs'],
                             'primary': design['primary'],
                             'parameters': p43_parameters(arms, vocab) if p43 else {
                                 arm: pid_registry.BASE_PARAMETERS + (
                                     0 if arm == pid_registry.REFERENCE else vocab['id_parameters']) for arm in arms},
                             **({'subset_column': subset_column} if p43 else {}),
                             'same_gpu_pairs': same_gpu_pairs(tasks, design['contrasts'], seeds),
                             'note': P43_NOTE if p43 else P40_NOTE if config.get(pid_registry.DESIGN_KEY) else
                                     'fresh model RNG seeds on the same export and exposed DEV rows: not new data, not '
                                     'an independent confirmatory test; family designed after reading P24-P36 '
                                     '(post-selection); two arms only (no flatten control); the candidate adds pitcher '
                                     'identity AND id parameters together; NOT parameter-matched, NOT compute-matched; '
                                     'a capacity-matched information-free control is not included in this bounded '
                                     'study, which limits causal attribution; vocabulary is TRAIN-only and re-derived '
                                     'here from the export; all seeds and subsets reported, no verdict'},
              'limits': ['Cpanel DEV and mlb_dev already exposed: repeated-DEV, descriptive only',
                         'Bootstrap conditions on fixed predictions (no refit, calibration or selection uncertainty)',
                         'In-vocabulary / UNK subsets are defined by the TRAIN vocabulary, not chosen on DEV',
                         'Prediction only; no 2026 rows; no policy or service claim'],
              'provenance': {
                  'config': {'path': str(Path(config_path).resolve()), 'file_sha256': config_sha256,
                             'digest': rc.config_digest(config), 'experiment_id': config['experiment_id']},
                  'export': {'path': str(Path(export).resolve()), 'manifest_sha256': manifest_sha256,
                             'scored_arrays_sha256': {**export_hashes, **pitcher_hashes}},
                  'task_sources': {**source_record, 'source_sha256': sources},
                  'task_runtime_signature': signature,
                  'tasks': {a: {str(s): {k: v for k, v in t.items() if k != 'pitcher_id'} for s, t in seeds_.items()}
                            for a, seeds_ in tasks.items()},
                  'scorer_sources_sha256': scorer_sources(),
                  'scorer_runtime': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                                     'hostname': platform.node()},
                  'bootstrap': {'draws': DRAWS, 'seed': BOOTSTRAP_SEED, 'function': 'matrix_metrics.paired_game_comparison'},
                  'created_utc': datetime.now(timezone.utc).isoformat()}}
    if p43:
        for one in result['vocabulary'].values():
            one.pop('n_fits', None)
    else:
        result['vocabulary'].pop('n_fits', None)
    if scorer_sources() != result['provenance']['scorer_sources_sha256']:
        raise RuntimeError('Scorer sources changed during scoring')
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.with_name(f'.{destination.name}.tmp-{os.getpid()}')
    work.mkdir()
    rc.write_json_atomic(work / 'score.json', result)
    (work / 'table_ko.md').write_text(pitcher_id_table(result) + '\n')
    (work / 'SHA256SUMS').write_text(''.join(f'{rc.sha256_file(work / n)}  {n}\n' for n in ('score.json', 'table_ko.md')))
    for name in ('score.json', 'table_ko.md', 'SHA256SUMS'):
        os.chmod(work / name, 0o444)
    if destination.exists():
        raise FileExistsError(f'{destination} appeared during scoring')
    os.rename(work, destination)
    return result


def pitcher_id_table(result):
    """Korean descriptive table: every split, subset and seed; no winner column."""
    v, block = result['vocabulary'], result['seed_block']
    if block.get('subset_column') == 'batter':
        lines = [f"# {result['family']} 타자 ID 임베딩 (후보 − 기준, 반복 DEV 서술, 판정 없음)", '']
        for column, one in v.items():
            lines.append(f"{column} TRAIN 어휘 {one['n_vocab']}명(TRAIN {one['min_count']}구 이상, 전체 {one['train_pitchers']}명, "
                         f"UNK TRAIN 행 {one['train_unk_rows']}/{one['train_rows']}), 추가 파라미터 {one['id_parameters']}.")
        lines += ['주 대비는 타자 ID − 행 섞기 대조군(같은 망·파라미터·연산, 타자 열만 행끼리 섞음)입니다. 행 묶음(in_vocab/unk)은 실제 '
                  '타자의 TRAIN 어휘 포함 여부입니다. 대조군은 ID 경로와 크기를 맞춘 것이고 일반 백본 확장과 같다는 뜻은 아닙니다.', '']
        return '\n'.join(lines + pitcher_id_table_body(result))
    lines = [f"# {result['family']} 투수 ID 임베딩 (후보 − 기준, 반복 DEV 서술, 판정 없음)", '',
             f"TRAIN 어휘 {v['n_vocab']}명(TRAIN {v['min_count']}구 이상, 전체 TRAIN 투수 {v['train_pitchers']}명, "
             f"UNK TRAIN 행 {v['train_unk_rows']}/{v['train_rows']}), TRAIN 마지막 날짜 {v['train_max_date']} ≤ "
             f"{v['last_train_date']}. 추가 파라미터 {v['id_parameters']}(파라미터·연산량 같지 않음). "
             + ('정보 없이 크기만 맞춘 대조군은 이번 2팔 연구에 포함하지 않았으므로 차이를 ID 정보 때문이라고 단정할 수 없습니다.'
              if len(result['seed_block']['parameters']) == 2 else
              '주 대비는 후보 − 행 섞기 대조군(같은 망·파라미터·연산, 투수 열만 행끼리 섞음)입니다. 대조군은 ID 경로와 크기를 맞춘 것이고 '
              '일반 백본 확장과 같다는 뜻은 아닙니다.'), '']
    return '\n'.join(lines + pitcher_id_table_body(result))


def pitcher_id_table_body(result):
    """Contrast and cost rows shared by the pitcher-ID designs (P38 / P40 / P43)."""
    def ci(block, key):
        b = block[key]
        lo, hi = b['ci95'] or (float('nan'), float('nan'))
        return f"{b['delta']:+.5f} [{lo:+.5f}, {hi:+.5f}]"
    block, lines = result['seed_block'], []
    for split, contrasts in result['pitcher_id_contrasts'].items():
        for name, by_subset in contrasts.items():
            head = f'## {split}' if len(contrasts) == 1 else f'## {split} — {name}' + (' (주 대비)' if name == block['primary'] else '')
            lines += [head, '', '| 행 묶음 | 행 수 | ΔNLL 앙상블 [95%] | ΔBrier 앙상블 [95%] | 시드별 ΔNLL |', '|---|---|---|---|---|']
            for subset, one in by_subset.items():
                rows = result['pitcher_id_rows'][split][subset]['rows']
                if one is None:
                    lines.append(f'| {subset} | {rows} | 행 없음 | 행 없음 | - |')
                    continue
                per_seed = ', '.join(f"{s}:{one['seeds'][s]['nll']['delta']:+.5f}" for s in one['seeds'])
                lines.append(f"| {subset} | {rows} | {ci(one['ensemble'], 'nll')} | {ci(one['ensemble'], 'brier')} | {per_seed} |")
            lines.append('')
    lines += ['## 비용 (fit.json, 시드별 학습 루프 초 / epoch)', '', '| 팔 | 시드: 초/epoch, 파라미터 |', '|---|---|']
    for arm, seeds in result['cost'].items():
        lines.append(f"| {arm} | " + ', '.join(f"{s}: {c['train_seconds_per_epoch']:.2f}, {c['parameter_count']}"
                                               for s, c in seeds.items()) + ' |')
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--export-manifest-sha256', required=True)
    p.add_argument('--family', type=Path, required=True, help='runner --output directory (arm/seed/...)')
    p.add_argument('--destination', type=Path, required=True, help='new directory; must not exist')
    p.add_argument('--source-manifest', type=Path,
                   help=f'{SOURCE_FORMAT} JSON of the EXACT task source map (default: current runner own_hashes)')
    args = p.parse_args(argv)
    score(args.config, args.config_sha256, args.export, args.export_manifest_sha256, args.family, args.destination,
          args.source_manifest)
    print('CUDA_DIRECT_SCORED', args.destination, flush=True)


if __name__ == '__main__':
    main()
