"""EXP-P48-001 Stage B (Mac CPU): CUDA adoption-seed heads mixed with the frozen G0-global vs the frozen G15 mix.

Primary comparison of the adoption-seed confirmation family (pre-registration
docs/PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md; registry
pitchmdp/direct_adoption_confirm_registry.py).
Every parameter below is read from config comparison.stage_b.rules, which must equal the registry's STAGE_B_RULES.

  1. Family: the config passes config_check (marker adoption_confirm) and is REGISTERED/enabled; all 15 Stage-A tasks
     have fit.json and state.json and pass score_cuda_direct.verify_family (identity = this config / pinned export /
     arm / seed / CUDA / expected task sources, intact artifacts, one CUDA runtime, epoch-cap records).  The CUDA
     archives (predictions.npz blend/dev, mlb_predictions.npz mlb_dev, each with _raw/_delivery_level/_keys/_y/
     _game_pk/_pitcher) must carry exactly the Mac baseline rows (EXP-P4-001 baseline blend/dev, EXP-P11-001 baseline
     mlb_dev): the same keys, labels, game and pitcher ids (score_ml_matrix.assert_aligned semantics) and the
     no-physics marker.  A different row order is aligned by the 3-column pitch key with a recorded permutation; a
     missing, extra or repeated key fails closed.  Only metadata is read in this step.
  2. Frozen inputs, each anchored to a pinned chain before use (rules.inputs.anchors, uniform): the G15 config, the
     G0 bundle it pins and the recorded EXP-P16-001 results against rules.reference; the frequency baselines and the
     G0-global Cpanel member states against the G0 bundle listing; the flatten direct-head member states against the
     G15 listing; the whole-MLB G0-global member states AND archives against the EXP-P11-001 analysis manifest, whose
     sha256 the G0 bundle pins (bundle -> manifest sha -> manifest inputs -> member state -> artifact hashes).  A file
     listed in more than one pinned map must agree with every one; a missing link fails closed.  Every member npz is
     also checked against the artifact_hashes of its adjacent *_state.json; member model hashes against the frozen
     member records.
  3. Reference G15 = run_ml_direct_head.mixed(flatten, G0, 1/2) -> score_ml_matrix.summarize_cell (June blend on the
     Cpanel blend rows); whole MLB blended exactly as run_ml_direct_head_confirm.score_arrays (EXP-P11-001 frequency
     baseline, Cpanel June weights, ensemble and per seed).  Reproduction gate: the recomputed G15 Cpanel-after-blend
     and whole-MLB-blended NLL, the G0-global values of the same rows and the five per-seed G15 - G0 deltas of both
     splits must equal results/EXP-P16-001.json within the registered tolerance, else abort before any candidate
     probability is read (the aggregate NLL alone is invariant to swapping members between seeds).
  4. Candidates: each CUDA arm x {MIX_FIXED (1/2, the G15 rule), MIX_FIT (per-seed grid weight on the June blend rows
     only; ensemble weight chosen the same way on the seed means)}, then the June frequency blend as usual.
  5. candidate - G15 by matrix_metrics.paired_game_comparison on (i) Cpanel DEV after the blend and (ii) whole-MLB
     DEV blended; N = matrix_metrics.prediction_decision (final stage).  Exactly one confirmatory test: the
     pre-specified primary (tabm_residual_counts MIX_FIT) is confirmed by intersection-union, i.e. a
     predictive_improvement on both splits, each judged with its own unadjusted one-sided p at alpha .05.  The three
     other candidate rows are secondary / exploratory: per-split Holm across those three as a descriptive adjustment,
     and a both-split outcome is labelled exploratory, never confirmed.  Control rows and the unblended
     head-vs-G15-head rows are descriptive (no N).  This is a pipeline comparison (CUDA heads with the epoch cap 60
     vs the G15 MPS flatten heads with the cap 30); control MIX_FIXED measures that difference directly.  No
     12-group robustness (R): no such helper in score_ml_matrix or matrix_metrics.
  6. <destination>/results.json, SUMMARY.md and SHA256SUMS (read-only) with every input sha256, the config sha256,
     git commit, seconds and peak RSS.  The destination must not exist and must not be inside a repository results/
     directory; nothing is written anywhere else.  Steps 2-6 hold the Mac's common heavy-job lock
     (<runs-root>/.heavy.lock, non-blocking, as run_ml_matrix.heavy_lock): a held lock fails fast.

  python scripts/score_cuda_mix_vs_g15.py --config ../../configs/EXP-P48-001.json --family <runner output dir> \\
      --g15 ../../configs/G15-MIX-RESEARCH-FROZEN-v1.json \\
      --runs-root "/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924" --destination <new dir> \\
      [--source-manifest <task sources json>] [--g0-bundle <path>] [--recorded-results <path>]
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np

import run_cuda_direct as rc  # noqa: E402  numpy only at import
import score_cuda_direct as sc  # noqa: E402  numpy only at import (family verification, task sources)
from pitchmdp import direct_adoption_confirm_registry as reg  # noqa: E402  torch-free
from pitchmdp.matrix_metrics import (holm_adjust, paired_game_comparison, pitch_losses,  # noqa: E402
                                     prediction_decision, prediction_metrics, validate_probabilities)

FORMAT = 'cuda_mix_vs_g15_v1'
PASS = 'predictive_improvement'
SOURCES = ('scripts/score_cuda_mix_vs_g15.py', 'scripts/score_cuda_direct.py', 'scripts/run_cuda_direct.py',
           'scripts/score_ml_matrix.py', 'scripts/run_ml_direct_head.py', 'scripts/run_sequence_calibration.py',
           'pitchmdp/matrix_metrics.py', 'pitchmdp/direct_adoption_confirm_registry.py')
CUDA_ARCHIVES = {'predictions.npz': ('blend', 'dev'), 'mlb_predictions.npz': ('mlb_dev',)}
SUFFIXES = ('', '_raw', '_delivery_level', '_keys', '_y', '_game_pk', '_pitcher')
META = ('_y', '_game_pk', '_pitcher')
CPANEL, MLB = 'cpanel_dev_after_blend', 'whole_mlb_dev_blended'
R_NOTE = ('not computed: no 12-group robustness (R) helper exists in score_ml_matrix or matrix_metrics '
          '(pitchmdp.matrix_group_metrics has 12 group guardrails that need dev_metadata.parquet columns; not wired '
          'in this pre-registration)')


def helpers():
    """The exact EXP-P16-001 scoring helpers (lazy: their modules import torch / pandas)."""
    from score_ml_matrix import assert_aligned, summarize_cell
    import run_ml_direct_head as base
    return assert_aligned, summarize_cell, base.mixed, base.paired


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def npz(path, names=None):
    with np.load(path, allow_pickle=False) as data:
        names = list(data.files) if names is None else list(names)
        missing = [n for n in names if n not in data.files]
        if missing:
            raise ValueError(f'{path}: missing arrays {missing}')
        return {n: data[n].copy() for n in names}


def get_path(value, path):
    for key in path:
        value = value[key]
    return value


# ------------------------------------------------------------------ rules and pins

def stage_b_rules(config):
    """Registered config + Stage-B rules (== registry STAGE_B_RULES); the helper bootstrap defaults must match them."""
    rules = reg.check_stage_b(config)
    if rules['splits'] != [CPANEL, MLB] or list(rules['mix_rules']) != [reg.MIX_FIXED, reg.MIX_FIT]:
        raise ValueError('Stage-B splits or mix rules differ from the scorer')
    boot = inspect.signature(paired_game_comparison).parameters
    if (boot['draws'].default, boot['seed'].default) != (rules['bootstrap']['draws'], rules['bootstrap']['seed']):
        raise ValueError('paired_game_comparison defaults differ from the registered bootstrap')
    if rules['alpha'] != .05:  # prediction_decision judges `p <= .05` (fixed inside matrix_metrics)
        raise ValueError('Stage-B alpha differs from the matrix_metrics decision threshold')
    primary, secondary = test_rows(rules)
    if primary in secondary or len(secondary) != 3:
        raise ValueError('Stage B needs one primary and three secondary candidate rows')
    return rules


def test_rows(rules):
    """(primary row, secondary rows) of the candidate tests, from the config rules (never from module constants)."""
    primary = row_name(rules['primary']['arm'], rules['primary']['mix_rule'])
    candidates = [row_name(arm, rule) for arm in rules['candidate_arms'] for rule in rules['mix_rules']]
    if primary not in candidates:
        raise ValueError('The primary candidate is not a candidate test')
    return primary, [name for name in candidates if name != primary]


def check_destination(destination):
    """New directory outside every repository results/ directory (this checkout's and the main checkout's)."""
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f'{destination} exists; Stage-B results are immutable')
    resolved = destination.parent.resolve() / destination.name
    for forbidden in results_dirs():
        if resolved == forbidden or resolved.is_relative_to(forbidden):
            raise ValueError(f'Stage-B output must not be written into a repository results/ directory: {forbidden}')
    return resolved


def results_dirs():
    dirs = {(REPO / 'results').resolve()}
    try:
        common = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--git-common-dir'], capture_output=True,
                                text=True, check=True).stdout.strip()
        common = Path(common) if Path(common).is_absolute() else REPO / common
        dirs.add((common.resolve().parent / 'results').resolve())
    except (OSError, subprocess.CalledProcessError):
        pass
    return dirs


def relative(path, root_name):
    """Path below the frozen runs root (posix), or None for a path outside it."""
    parts = Path(path).parts
    return Path(*parts[parts.index(root_name) + 1:]).as_posix() if root_name in parts else None


def listing(files, root_name):
    """{path relative to the frozen runs root: sha256} of a frozen bundle's `files` entries under that root."""
    out = {}
    for entry in files.values():
        rel = relative(entry['path'], root_name)
        if rel is not None:
            out[rel] = entry['sha256']
    return out


def paths_map(hashes, root_name):
    """{path relative to the frozen runs root: sha256} of a manifest `inputs` map ({absolute path: sha256})."""
    return {rel: digest for path, digest in hashes.items() if (rel := relative(path, root_name)) is not None}


def pinned(path, expected, what, inputs):
    digest = rc.sha256_file(path)
    if digest != expected:
        raise ValueError(f'{what} differs from its pin: {path}')
    inputs[str(Path(path).resolve())] = digest
    return read_json(path)


def references(rules, g15_path, g0_bundle=None, recorded_results=None, inputs=None):
    """G15 config, the G0 bundle it pins and the recorded EXP-P16-001 results, all sha-checked."""
    ref = rules['reference']
    g15 = pinned(g15_path, ref['config_sha256'], 'G15 config', inputs)
    if g15['definition']['arm'] != ref['arm'] \
            or g15['definition']['direct_weight'] != rules['mix_rules'][reg.MIX_FIXED]['weight'] \
            or g15['g0_component']['bundle_sha256'] != ref['g0_bundle_sha256']:
        raise ValueError('G15 config does not define the registered reference')
    g0 = pinned(g0_bundle or REPO / ref['g0_bundle'], ref['g0_bundle_sha256'], 'G0 bundle', inputs)
    recorded = pinned(recorded_results or REPO / ref['recorded_results'], ref['recorded_results_sha256'],
                      'Recorded EXP-P16-001 results', inputs)
    if g15['files']['results_p16']['sha256'] != ref['recorded_results_sha256']:
        raise ValueError('G15 config lists other EXP-P16-001 results')
    return g15, g0, recorded


# ------------------------------------------------------------------ frozen members

def pinned_maps(rules, runs_root, g15, g0_bundle, inputs):
    """{source: {path below the runs root: sha256}} of the pinned chains: the G0 bundle listing, the G15 listing and
    the EXP-P11-001 analysis manifest `inputs`, whose own sha256 the G0 bundle pins (fail closed on any broken link)."""
    spec = rules['inputs']
    root_name, chain = spec['runs_root_name'], spec['p11_manifest']
    maps = {'g0_bundle': listing(g0_bundle['files'], root_name), 'g15': listing(g15['files'], root_name)}
    entry = g0_bundle['files'].get(chain['g0_bundle_key'])
    if not isinstance(entry, dict) or relative(entry.get('path', ''), root_name) != chain['path']:
        raise ValueError('G0 bundle does not pin the EXP-P11-001 analysis manifest at the registered path')
    path = Path(runs_root) / chain['path']
    digest = rc.sha256_file(path)
    if digest != entry['sha256']:
        raise ValueError('EXP-P11-001 analysis manifest differs from the G0 bundle pin')
    inputs[str(path.resolve())] = digest
    maps['p11_manifest'] = paths_map(read_json(path)['inputs'], root_name)
    if set(spec['anchors']) != {'cpanel_baseline', 'mlb_baseline', 'g0_cpanel', 'g0_mlb', 'flatten'} \
            or any(a['source'] not in maps or not set(a['files']) <= {'state', 'archive'}
                   for a in spec['anchors'].values()):
        raise ValueError('Stage-B anchors must name a pinned chain for every frozen input')
    return maps


def anchored(maps, anchor, kind, rel, digest, record):
    """`rel` (sha `digest`) must be listed in its anchor chain when `kind` is anchored there, and must agree with every
    pinned map that lists it; the chain is recorded for the result (uniform for every frozen input)."""
    source = anchor['source']
    if kind in anchor['files'] and rel not in maps[source]:
        raise ValueError(f'{rel}: not reachable from the pinned {source} chain')
    listed = sorted(name for name, one in maps.items() if rel in one)
    for name in listed:
        if maps[name][rel] != digest:
            raise ValueError(f'{rel}: differs from the pinned {name} record')
    record[rel] = {'sha256': digest, 'listed_in': listed,
                   'anchor': source if kind in anchor['files'] else f'{source} -> state artifact_hashes'}


def frozen_member(root, rel, state_name, maps, anchor, identity, model_sha256, inputs, record):
    """Archive path of one frozen member after its anchored state, artifact and identity checks (nothing loaded)."""
    dest = Path(root) / rel
    state_path = dest / state_name
    key = f'{rel}/{state_name}'
    digest = rc.sha256_file(state_path)
    anchored(maps, anchor, 'state', key, digest, record)
    state = read_json(state_path)
    for name, expected in state['artifact_hashes'].items():
        path = (dest / name).resolve()
        if not path.is_relative_to(dest.resolve()) or not path.is_file() or rc.sha256_file(path) != expected:
            raise ValueError(f'{key}: artifact identity changed: {name}')
    for field, value in identity.items():
        if state['identity'].get(field) != value:
            raise ValueError(f'{key}: identity {field} differs from the frozen member')
    seen = {state['identity'].get('model_sha256'), *(state.get('dependencies') or {}).values()}
    if model_sha256 not in seen:
        raise ValueError(f'{key}: model hash differs from the frozen member record')
    archive = 'mlb_predictions.npz' if state_name.startswith('mlb_') else 'predictions.npz'
    if archive not in state['artifact_hashes']:
        raise ValueError(f'{key}: state does not cover {archive}')
    for name, expected in state['artifact_hashes'].items():  # the archive per its anchor, the rest for agreement
        anchored(maps, anchor if name == archive else {'source': anchor['source'], 'files': []}, 'archive',
                 f'{rel}/{name}', expected, record)
    inputs[str(state_path.resolve())] = digest
    inputs[str((dest / archive).resolve())] = state['artifact_hashes'][archive]
    return dest / archive


def load_frozen(rules, runs_root, g15, g0_bundle, inputs, record=None):
    """Cpanel/whole-MLB baselines, G0-global members and flatten direct heads (seeds 0-4), each anchored to a pinned
    chain (rules.inputs.anchors) before any array is read."""
    spec, runs_root = rules['inputs'], Path(runs_root)
    if runs_root.name != spec['runs_root_name']:
        raise ValueError('runs root must be the frozen ' + spec['runs_root_name'])
    record = {} if record is None else record
    maps, anchors = pinned_maps(rules, runs_root, g15, g0_bundle, inputs), spec['anchors']
    baselines = {}
    for name in ('cpanel_baseline', 'mlb_baseline'):
        path = runs_root / spec[name]
        digest = rc.sha256_file(path)
        anchored(maps, anchors[name], 'archive', spec[name], digest, record)
        inputs[str(path.resolve())] = digest
        baselines[name] = path
    seeds = [str(s) for s in reg.SEEDS]
    whole = lambda prefix: [prefix, *(prefix + s for s in ('_keys', *META))]  # noqa: E731  calibrated + metadata
    archives = {}
    for kind, state_name, names in (('g0_cpanel', 'prediction_state.json', None),
                                    ('g0_mlb', 'prediction_state.json', whole('dev')),
                                    ('flatten', 'prediction_state.json', None),
                                    ('flatten_mlb', 'mlb_prediction_state.json', whole('mlb_dev'))):
        spec_key = 'flatten' if kind == 'flatten_mlb' else kind
        archives[kind] = []
        for s in seeds:
            if spec_key == 'flatten':
                expected = {'seed': int(s), 'smoke': False, 'model_sha256': g15['members'][s]['model_sha256']}
                model = expected['model_sha256']
            else:
                expected, model = {'seed': int(s), 'cell': reg.G0_CELL}, g0_bundle['members'][s]['model_sha256']
            archives[kind].append((frozen_member(runs_root, spec[spec_key][s], state_name, maps, anchors[spec_key],
                                                 expected, model, inputs, record), names))
    # every chain above passed before any frozen array is read
    cpanel = npz(baselines['cpanel_baseline'], [p + s for p in ('blend', 'dev') for s in ('', '_raw', '_keys', *META)])
    mlb = npz(baselines['mlb_baseline'], ['mlb_dev', 'mlb_dev_keys', *('mlb_dev' + s for s in META)])
    loaded = {kind: [npz(path, names) for path, names in items] for kind, items in archives.items()}
    return cpanel, mlb, loaded['g0_cpanel'], loaded['g0_mlb'], loaded['flatten'], loaded['flatten_mlb']


def check_mlb_rows(mlb, members, prefix, what):
    """Whole-MLB rows: unique (n, 3) keys whose first column is the game, same labels / games / pitchers in order."""
    keys = mlb['mlb_dev_keys']
    if keys.ndim != 2 or keys.shape[1] != 3 or len(np.unique(keys, axis=0)) != len(keys) \
            or not np.array_equal(keys[:, 0], mlb['mlb_dev_game_pk']):
        raise ValueError('Repeated or invalid whole-MLB baseline keys')
    validate_probabilities(mlb['mlb_dev_y'], mlb['mlb_dev'])
    for i, member in enumerate(members):
        for suffix in ('_keys', *META):
            if not np.array_equal(member[prefix + suffix], mlb['mlb_dev' + suffix]):
                raise ValueError(f'{what} {i}: whole-MLB {suffix[1:]} differ from the baseline rows')
        validate_probabilities(mlb['mlb_dev_y'], member[prefix])


# ------------------------------------------------------------------ CUDA family

def key_permutation(source, target):
    """None if the (n, 3) keys are already in target order, else idx with source[idx] == target (fail closed)."""
    source, target = np.asarray(source), np.asarray(target)
    for keys in (source, target):
        if keys.ndim != 2 or keys.shape[1] != 3 or len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError('Pitch keys must be unique (n, 3) rows')
    if np.array_equal(source, target):
        return None
    if len(source) != len(target):
        raise ValueError(f'CUDA archive has {len(source)} rows, the baseline {len(target)}')
    index = {tuple(row): i for i, row in enumerate(source.tolist())}
    order = np.array([index.get(tuple(row), -1) for row in target.tolist()], dtype=np.int64)
    if (order < 0).any():
        raise ValueError(f'{int((order < 0).sum())} baseline pitch keys are missing from the CUDA archive')
    return order


def complete_family(family, config):
    family = Path(family)
    missing = [f'{a}/{s}/{n}' for a in reg.ARM_ORDER for s in config['seeds'] for n in ('fit.json', 'state.json')
               if not (rc.task_dir(family, a, s, False) / n).is_file()]
    if missing:
        raise ValueError('Incomplete family; nothing scored. Missing: ' + ', '.join(missing))
    if [a['name'] for a in config['arms']] != list(reg.ARM_ORDER) \
            or len(reg.ARM_ORDER) * len(config['seeds']) != reg.FITS:
        raise ValueError('Family is not the registered 15-task family')
    return family


def cuda_alignment(family, config, cpanel, mlb):
    """Row permutation (None = same order) of every CUDA archive split onto the Mac baseline rows (metadata only)."""
    plan = {}
    for arm in reg.ARM_ORDER:
        for seed in config['seeds']:
            dest = rc.task_dir(family, arm, seed, False)
            for archive, splits in CUDA_ARCHIVES.items():
                with np.load(dest / archive, allow_pickle=False) as data:
                    for split in splits:
                        missing = [split + s for s in SUFFIXES if split + s not in data.files]
                        if missing:
                            raise ValueError(f'{arm}/{seed}/{archive}: missing arrays {missing}')
                        reference = mlb if split == 'mlb_dev' else cpanel
                        order = key_permutation(data[split + '_keys'], reference[split + '_keys'])
                        for suffix in META:
                            values = data[split + suffix] if order is None else data[split + suffix][order]
                            if not np.array_equal(values, reference[split + suffix]):
                                raise ValueError(f'{arm}/{seed}/{split}: {suffix[1:]} differ from the baseline rows')
                        if not (data[split + '_delivery_level'] == rc.NO_PHYSICS_LEVEL).all():
                            raise ValueError(f'{arm}/{seed}/{split}: not the no-physics direct head')
                        plan[f'{arm}/{seed}/{split}'] = order
    return plan


def cuda_members(family, arm, seeds, plan, cpanel, assert_aligned):
    """Calibrated (+ raw for Cpanel) probabilities of one arm in baseline row order, with baseline metadata."""
    panel, whole = [], []
    for seed in seeds:
        dest = rc.task_dir(family, arm, seed, False)
        member = {f'{part}{suffix}': cpanel[f'{part}{suffix}'] for part in ('blend', 'dev')
                  for suffix in ('_keys', *META)}
        with np.load(dest / 'predictions.npz', allow_pickle=False) as data:
            for part in ('blend', 'dev'):
                order = plan[f'{arm}/{seed}/{part}']
                for suffix in ('', '_raw'):
                    values = np.asarray(data[part + suffix], dtype=np.float64)
                    member[part + suffix] = values if order is None else values[order]
        assert_aligned(member, cpanel)
        panel.append(member)
        with np.load(dest / 'mlb_predictions.npz', allow_pickle=False) as data:
            order = plan[f'{arm}/{seed}/mlb_dev']
            values = np.asarray(data['mlb_dev'], dtype=np.float64)
            whole.append(values if order is None else values[order])
    return panel, whole


# ------------------------------------------------------------------ mixing

def fit_mix_weight(y, head, g0, grid):
    """Grid weight minimizing the pitch-weighted NLL of w*head + (1-w)*g0 on the supplied rows only (the June blend
    split); ties go to the smallest weight (first minimum of the ascending grid)."""
    grid = [float(w) for w in grid]
    if not grid or grid != sorted(set(grid)) or not all(0 <= w <= 1 for w in grid):
        raise ValueError('Mix grid must be strictly ascending weights in [0, 1]')
    head, g0 = np.asarray(head, dtype=np.float64), np.asarray(g0, dtype=np.float64)
    if head.shape != g0.shape:
        raise ValueError('Head and G0 probabilities must align')
    nll = [float(pitch_losses(y, w * head + (1 - w) * g0)[:, 0].mean()) for w in grid]
    best = int(np.argmin(nll))
    return {'weight': grid[best], 'grid': grid, 'blend_nll_by_weight': nll, 'rows': int(len(y))}


def ensemble_mix(head, g0, weight):
    """Seed-mean head and seed-mean G0 mixed with one weight (MIX_FIT ensemble); metadata of the first G0 member."""
    member = dict(g0[0])
    for part in ('blend', 'dev'):
        for suffix in ('', '_raw'):
            member[part + suffix] = weight * np.mean([h[part + suffix] for h in head], axis=0) + \
                (1 - weight) * np.mean([g[part + suffix] for g in g0], axis=0)
    return member


def blend_mlb(report, ensemble_mix_mlb, seed_mix_mlb, frequency):
    """Whole-MLB June blend with the Cpanel weights (ensemble weight on the ensemble, per-seed weights per seed)."""
    w = report['selection']['model_weight']
    seeds = [s['blend_selection']['model_weight'] * m + (1 - s['blend_selection']['model_weight']) * frequency
             for s, m in zip(report['seeds'], seed_mix_mlb)]
    return w * ensemble_mix_mlb + (1 - w) * frequency, seeds


def seed_deltas(y, candidate_seeds, reference_seeds):
    """Per-seed NLL differences exactly as run_ml_direct_head.paired computes `seed_deltas`."""
    return [float((pitch_losses(y, a) - pitch_losses(y, b))[:, 0].mean())
            for a, b in zip(candidate_seeds, reference_seeds)]


def g15_reference(flatten, g0, cpanel, flatten_mlb, g0_mlb, frequency, y_mlb, weight, tools):
    """G15 exactly as EXP-P16-001 (run_ml_direct_head_confirm.score_arrays, MIX arm) and the values the reproduction
    gate compares: G15 / G0-global NLL on both splits and the per-seed G15 - G0 deltas of both splits."""
    assert_aligned, summarize_cell, mixed, _ = tools
    for member in (*flatten, *g0):
        assert_aligned(member, cpanel)
    report, predicted = summarize_cell(mixed(flatten, g0, weight), cpanel)
    rep_g0, p_g0 = summarize_cell(g0, cpanel)
    b = [m['dev'] for m in g0_mlb]
    a = [weight * d['mlb_dev'] + (1 - weight) * g for d, g in zip(flatten_mlb, b)]
    blended, seeds = blend_mlb(report, np.mean(a, 0), a, frequency)
    g0_blended, g0_seeds = blend_mlb(rep_g0, np.mean(b, 0), b, frequency)
    y = cpanel['dev_y']
    recomputed = {
        'g15_cpanel_dev_after_blend_nll': report['primary']['log_loss'],
        'g15_whole_mlb_dev_blended_nll': prediction_metrics(y_mlb, blended)['log_loss'],
        'g0_cpanel_dev_after_blend_nll': rep_g0['primary']['log_loss'],
        'g0_whole_mlb_dev_blended_nll': prediction_metrics(y_mlb, g0_blended)['log_loss'],
        'g15_minus_g0_cpanel_dev_after_blend_seed_deltas': seed_deltas(y, predicted['seed_primary'],
                                                                       p_g0['seed_primary']),
        'g15_minus_g0_whole_mlb_dev_blended_seed_deltas': seed_deltas(y_mlb, seeds, g0_seeds)}
    return {'report': report, 'cpanel': predicted, 'mlb_blended': blended, 'mlb_seed_blended': seeds,
            'recomputed': recomputed}


def reproduction_gate(reference, recorded, gate):
    """Recomputed G15 / G0 NLL and per-seed deltas vs the recorded EXP-P16-001 values; raises (nothing scored) on any
    difference beyond the tolerance or any missing value."""
    recomputed = reference['recomputed']
    if set(recomputed) != set(gate['fields']) | set(gate['seed_delta_fields']) \
            or set(gate['fields']) & set(gate['seed_delta_fields']):
        raise ValueError('Reproduction gate fields differ from the registered list')
    rows = {}
    for name, path in gate['fields'].items():
        value = float(get_path(recorded, path))
        rows[name] = {'recorded': value, 'recomputed': recomputed[name], 'abs_diff': abs(recomputed[name] - value),
                      'path': path}
    for name, path in gate['seed_delta_fields'].items():
        values, mine = [float(v) for v in get_path(recorded, path)], recomputed[name]
        diff = max((abs(x - v) for x, v in zip(mine, values)), default=float('inf')) \
            if len(values) == len(mine) == len(reg.SEEDS) else float('inf')
        rows[name] = {'recorded': values, 'recomputed': mine, 'abs_diff': diff, 'path': path}
    failed = [n for n, r in rows.items() if not r['abs_diff'] <= gate['tolerance']]
    if failed:
        raise ValueError('Reproduction gate failed (nothing scored): ' + ', '.join(
            f"{n} recorded {rows[n]['recorded']!r} recomputed {rows[n]['recomputed']!r}" for n in failed))
    return {'passed': True, 'tolerance': gate['tolerance'], 'fields': rows}


def candidate(head, g0, cpanel, head_mlb, g0_mlb_dev, frequency, rule, rules, tools):
    """Cpanel (June blend) and whole-MLB (blended) predictions of one CUDA arm under one mix rule."""
    _, summarize_cell, mixed, _ = tools
    if rule == reg.MIX_FIXED:
        w = rules['mix_rules'][reg.MIX_FIXED]['weight']
        report, predicted = summarize_cell(mixed(head, g0, w), cpanel)
        seed_mix = [w * h + (1 - w) * g for h, g in zip(head_mlb, g0_mlb_dev)]
        weights = {'seeds': [w] * len(head), 'ensemble': w, 'selection': None}
        ensemble = np.mean(seed_mix, 0)
    elif rule == reg.MIX_FIT:
        spec = rules['mix_rules'][reg.MIX_FIT]
        if spec['split'] != 'blend' or spec['probabilities'] != 'calibrated':
            raise ValueError('MIX_FIT weights are chosen on the calibrated June blend rows only')
        y = cpanel['blend_y']
        seed_fit = [fit_mix_weight(y, h['blend'], g['blend'], spec['grid']) for h, g in zip(head, g0)]
        ens_fit = fit_mix_weight(y, np.mean([h['blend'] for h in head], 0), np.mean([g['blend'] for g in g0], 0),
                                 spec['grid'])
        seed_report, seed_predicted = summarize_cell(
            [mixed([h], [g], f['weight'])[0] for h, g, f in zip(head, g0, seed_fit)], cpanel)
        ens_report, ens_predicted = summarize_cell([ensemble_mix(head, g0, ens_fit['weight'])], cpanel)
        report = {**{k: ens_report[k] for k in ('selection', 'raw_ensemble', 'calibrated_ensemble', 'primary')},
                  'seeds': seed_report['seeds']}
        predicted = {**{k: ens_predicted[k] for k in ('primary', 'calibrated', 'raw')},
                     'seed_primary': seed_predicted['seed_primary']}
        seed_mix = [f['weight'] * h + (1 - f['weight']) * g for h, g, f in zip(head_mlb, g0_mlb_dev, seed_fit)]
        ensemble = ens_fit['weight'] * np.mean(head_mlb, 0) + (1 - ens_fit['weight']) * np.mean(g0_mlb_dev, 0)
        weights = {'seeds': [f['weight'] for f in seed_fit], 'ensemble': ens_fit['weight'],
                   'selection': {'seeds': seed_fit, 'ensemble': ens_fit}}
    else:
        raise ValueError('Unknown mix rule: ' + str(rule))
    blended, seeds = blend_mlb(report, ensemble, seed_mix, frequency)
    return {'report': report, 'cpanel': predicted, 'mlb_blended': blended, 'mlb_seed_blended': seeds,
            'weights': weights}


def decide(rows, primary, secondary, n_rule):
    """Final-stage N of one split.  The single prespecified primary is judged with its own unadjusted one-sided p
    (its confirmation is the intersection-union of both splits, outside this function); the three secondary rows
    with Holm across those three (a descriptive adjustment; their combined outcome is exploratory).  Every other row
    (control, descriptive) gets no N."""
    def judge(name, p, adjustment):
        rows[name]['N'] = prediction_decision(rows[name]['paired'], p, rows[name]['seed_deltas'],
                                              minimum_improvement=-n_rule['delta_nll_max'],
                                              brier_margin=n_rule['brier_ci95_upper_max'], stage=n_rule['stage'])
        rows[name].update(p_decision=p, p_adjustment=adjustment)
    judge(primary, rows[primary]['paired']['nll']['p_less'], 'none: single prespecified primary test')
    adjusted = holm_adjust([rows[name]['paired']['nll']['p_less'] for name in secondary])
    for name, p in zip(secondary, adjusted):
        judge(name, p, f'holm across the {len(secondary)} secondary rows of this split (descriptive)')
    for row in rows.values():
        for key in ('N', 'p_decision', 'p_adjustment'):
            row.setdefault(key, None)
    return rows


def row_name(arm, rule):
    return f'{arm}/{rule}'


def score_arrays(heads, flatten, g0, cpanel, heads_mlb, flatten_mlb, g0_mlb, mlb, rules, recorded, tools=None):
    """Pure Stage-B scoring on aligned arrays (synthetic-testable): reproduction gate first, then candidates.

    heads / heads_mlb are callables arm -> five aligned members (Cpanel dicts / whole-MLB calibrated arrays), so no
    candidate probability is touched before the gate passes; the other inputs are the five frozen members."""
    tools = tools or helpers()
    _, _, _, paired = tools
    weight, n_rule = rules['mix_rules'][reg.MIX_FIXED]['weight'], rules['N']
    check_mlb_rows(mlb, flatten_mlb, 'mlb_dev', 'flatten member')
    check_mlb_rows(mlb, g0_mlb, 'dev', 'G0 whole-MLB member')
    frequency, y_mlb, games_mlb = mlb['mlb_dev'], mlb['mlb_dev_y'], mlb['mlb_dev_game_pk']
    y, games = cpanel['dev_y'], cpanel['dev_game_pk']
    reference = g15_reference(flatten, g0, cpanel, flatten_mlb, g0_mlb, frequency, y_mlb, weight, tools)
    gate = reproduction_gate(reference, recorded, rules['reproduction_gate'])
    primary, secondary = test_rows(rules)
    g0_mlb_dev = [m['dev'] for m in g0_mlb]
    flatten_dev, flatten_whole = [m['dev'] for m in flatten], [m['mlb_dev'] for m in flatten_mlb]
    rows = {CPANEL: {}, MLB: {}}
    heads_vs = {'cpanel_dev': {}, 'whole_mlb_dev': {}}
    weights = {}
    arms = [*rules['descriptive_arms'], *rules['candidate_arms']]
    for arm in reg.ARM_ORDER:
        if arm not in arms:
            raise ValueError(arm + ': not a registered Stage-B arm')
        head, head_mlb = heads(arm), heads_mlb(arm)
        if len(head) != len(reg.SEEDS) or len(head_mlb) != len(reg.SEEDS):
            raise ValueError(arm + ': Stage B needs all five seeds')
        check_mlb_rows(mlb, [{'mlb_dev': p, **{'mlb_dev' + s: mlb['mlb_dev' + s] for s in ('_keys', *META)}}
                             for p in head_mlb], 'mlb_dev', arm)
        heads_vs['cpanel_dev'][arm] = paired(y, games, np.mean([h['dev'] for h in head], 0), np.mean(flatten_dev, 0),
                                             [h['dev'] for h in head], flatten_dev)
        heads_vs['whole_mlb_dev'][arm] = paired(y_mlb, games_mlb, np.mean(head_mlb, 0), np.mean(flatten_whole, 0),
                                                head_mlb, flatten_whole)
        weights[arm] = {}
        for rule in rules['mix_rules']:
            got = candidate(head, g0, cpanel, head_mlb, g0_mlb_dev, frequency, rule, rules, tools)
            name = row_name(arm, rule)
            role = 'primary' if name == primary else 'secondary_exploratory' if name in secondary else (
                'pipeline_difference_descriptive' if rule == reg.MIX_FIXED else 'descriptive')
            rows[CPANEL][name] = {'arm': arm, 'mix_rule': rule, 'role': role,
                                  **paired(y, games, got['cpanel']['primary'], reference['cpanel']['primary'],
                                           got['cpanel']['seed_primary'], reference['cpanel']['seed_primary']),
                                  'june_weight': {'ensemble': got['report']['selection']['model_weight'],
                                                  'seeds': [s['blend_selection']['model_weight']
                                                            for s in got['report']['seeds']]}}
            rows[MLB][name] = {'arm': arm, 'mix_rule': rule, 'role': role,
                               **paired(y_mlb, games_mlb, got['mlb_blended'], reference['mlb_blended'],
                                        got['mlb_seed_blended'], reference['mlb_seed_blended'])}
            weights[arm][rule] = got['weights']
            del got
        del head, head_mlb
    for split in (CPANEL, MLB):
        decide(rows[split], primary, secondary, n_rule)
    status = {split: rows[split][primary]['N']['status'] for split in (CPANEL, MLB)}
    exploratory = {}
    for name in secondary:
        one = {split: rows[split][name]['N']['status'] for split in (CPANEL, MLB)}
        exploratory[name] = {**one, 'status': 'exploratory',
                             'both_splits_predictive_improvement': all(s == PASS for s in one.values()),
                             'p_adjustment': rows[CPANEL][name]['p_adjustment']}
    sanity = row_name('control', reg.MIX_FIXED)
    return {
        'reproduction_gate': gate,
        'reference': {'arm': rules['reference']['arm'],
                      CPANEL: {'n': int(len(y)), 'games': int(len(np.unique(games))), 'report': reference['report']},
                      MLB: {'n': int(len(y_mlb)), 'games': int(len(np.unique(games_mlb))),
                            'metrics': prediction_metrics(y_mlb, reference['mlb_blended']),
                            'frequency': prediction_metrics(y_mlb, frequency)}},
        CPANEL: {'n': int(len(y)), 'games': int(len(np.unique(games))), 'rows': rows[CPANEL]},
        MLB: {'n': int(len(y_mlb)), 'games': int(len(np.unique(games_mlb))), 'rows': rows[MLB]},
        'multiplicity': {'primary': primary, 'secondary': secondary, 'rules': rules['multiplicity'],
                         'note': 'one confirmatory test: the prespecified primary, judged on each split with its own '
                                 'unadjusted one-sided p (alpha .05) and confirmed only by intersection-union of both '
                                 'splits; the three secondary rows get Holm across those three per split as a '
                                 'descriptive adjustment and are exploratory; control rows and the unblended head rows '
                                 'have no N'},
        'confirmation': {'primary': primary,
                         'primary_status': 'confirmed' if all(s == PASS for s in status.values()) else 'not_confirmed',
                         'splits': status, 'rule': rules['confirmation'],
                         'test': 'intersection-union over the two splits; each split one-sided at alpha .05 with its '
                                 'own unadjusted p; no Holm (single prespecified test)',
                         'note': 'confirmed = predictive_improvement on both splits (final stage, 4/5 seeds); any '
                                 'adoption is a separate Song decision and changes nothing by itself'},
        'secondary_exploratory': {
            'rows': exploratory,
            'note': 'secondary candidate rows: per-split Holm across the three as a descriptive adjustment; a '
                    'both-split outcome is exploratory, never a confirmation'},
        'pipeline_difference': {
            'row': sanity,
            'note': 'control at MIX_FIXED is the same flatten architecture and the same 1/2 mix as G15, so it measures '
                    'the pipeline difference directly: CUDA, epoch cap 60, seeds 0-4 of this family vs the frozen MPS '
                    'G15 flatten heads fitted with the cap 30 (configs/EXP-P16-001.yaml); descriptive',
            **{split: {'nll': rows[split][sanity]['paired']['nll'], 'brier': rows[split][sanity]['paired']['brier']}
               for split in (CPANEL, MLB)}},
        'mix_weights': weights,
        'descriptive_unblended_head_vs_g15_head': {
            **heads_vs, 'scope': 'calibrated five-seed head (no G0 mix, no June blend) minus the G15 flatten head '
                                 '(EXP-P15-001/P16-001 MPS); descriptive'},
        'robustness_R': {'status': 'not_computed', 'reason': R_NOTE},
    }


# ------------------------------------------------------------------ CLI scoring

@contextmanager
def heavy_lock(runs_root):
    """The Mac's single heavy-job queue (same file as run_ml_matrix.heavy_lock: <runs root>/.heavy.lock)."""
    path = Path(runs_root) / '.heavy.lock'
    with path.open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another heavy job holds {path}; nothing scored') from error
        try:
            yield path
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def git_state():
    try:
        head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain'], capture_output=True, text=True,
                               check=True).stdout
        return {'commit': head, 'dirty': bool(dirty.strip())}
    except (OSError, subprocess.CalledProcessError) as error:
        return {'error': repr(error)}


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'ru_maxrss': int(value), 'unit': 'bytes' if sys.platform == 'darwin' else 'kilobytes'}


def own_sources():
    return {rel: rc.sha256_file(PROJECT / rel) for rel in SOURCES}


def score(config_path, family, g15_path, runs_root, destination, source_manifest=None, g0_bundle=None,
          recorded_results=None):
    started = time.perf_counter()
    destination = check_destination(destination)
    sources = own_sources()
    inputs = {}
    config_sha256 = rc.sha256_file(config_path)
    config = rc.config_check(read_json(config_path))
    if not rc.uses_adoption_confirm(config):
        raise ValueError('Stage B scores only the adoption-confirm family')
    if config.get('status') != 'REGISTERED' or config.get('execution', {}).get('enabled') is not True:
        raise ValueError('Only a REGISTERED, enabled family has real tasks to score')
    rules = stage_b_rules(config)
    manifest_sha256 = config['export'].get('manifest_sha256')
    if not manifest_sha256:
        raise ValueError('The config must pin the export manifest sha256')
    g15, g0_bundle, recorded = references(rules, g15_path, g0_bundle, recorded_results, inputs)
    family = complete_family(family, config)
    task_sources, source_record = sc.task_sources(source_manifest)
    tasks, signature = sc.verify_family(family, config, manifest_sha256, task_sources)
    with heavy_lock(runs_root):  # steps 2-6
        anchors = {}
        cpanel, mlb, g0, g0_mlb, flatten, flatten_mlb = load_frozen(rules, runs_root, g15, g0_bundle, inputs,
                                                                    anchors)
        plan = cuda_alignment(family, config, cpanel, mlb)
        tools = helpers()
        cache = {}

        def load_arm(arm):
            if arm not in cache:
                cache.clear()  # bounded memory: one CUDA arm at a time
                cache[arm] = cuda_members(family, arm, config['seeds'], plan, cpanel, tools[0])
            return cache[arm]

        result = score_arrays(lambda arm: load_arm(arm)[0], flatten, g0, cpanel, lambda arm: load_arm(arm)[1],
                              flatten_mlb, g0_mlb, mlb, rules, recorded, tools)
        cache.clear()
        for arm in reg.ARM_ORDER:
            for seed in config['seeds']:
                dest = rc.task_dir(family, arm, seed, False)
                inputs[str((dest / 'state.json').resolve())] = tasks[arm][seed]['state_sha256']
                hashes = rc.read_json(dest / 'state.json')['artifact_hashes']
                for archive in CUDA_ARCHIVES:
                    inputs[str((dest / archive).resolve())] = hashes[archive]
        permuted = sorted(k for k, v in plan.items() if v is not None)
        if own_sources() != sources:
            raise RuntimeError('Stage-B scorer sources changed during scoring')
        result = {
            'format': FORMAT, 'experiment_id': config['experiment_id'], 'stage': 'B',
            'scope': ('each CUDA head mixed with the frozen G0-global member of the same seed, June frequency blend, '
                      'minus the frozen G15 mix on Cpanel DEV after the blend and on whole-MLB DEV blended; repeated '
                      'exposed DEV, prediction only'),
            'primary_candidate': rules['primary'], 'adoption_decision': None, 'policy_effect': None,
            **result,
            'alignment': {'permuted': permuted, 'same_order': len(plan) - len(permuted), 'checked': len(plan),
                          'rule': '3-column pitch key permutation onto the Mac baseline rows; missing keys fail '
                                  'closed'},
            'limits': ['Cpanel DEV and whole-MLB DEV were repeatedly exposed (G15 numbers read before, P34/P35 read '
                       'before this family was designed): a repeated-DEV result, not an independent confirmation',
                       'Seeds 0-4 are adoption RNG seeds reused across families (0-2 P24/P25/P27/P29 arms, 3-4 '
                       'EXP-P30-001 arms); not new data',
                       'Bootstrap conditions on fixed predictions (no refit, calibration or selection uncertainty)',
                       'MIX_FIT weights and June blend weights are both chosen on the June (blend) rows; that '
                       'selection is not part of the bootstrap',
                       'Pipeline comparison: the CUDA heads train with the epoch cap 60 on CUDA, the G15 flatten heads '
                       'were fitted with the cap 30 on MPS (configs/EXP-P16-001.yaml); the control MIX_FIXED row '
                       '(same architecture, same 1/2 mix) measures that pipeline difference directly',
                       'A not-confirmed primary documents the ceiling of this direct-head pipeline on the exposed 2025 '
                       'DEV, not of direct heads in general',
                       'Only the prespecified primary can be confirmed; secondary rows are exploratory',
                       'No 12-group robustness (R) in this family; no 2026 rows; no policy or service claim'],
            'provenance': {
                'config': {'path': str(Path(config_path).resolve()), 'sha256': config_sha256,
                           'digest': rc.config_digest(config)},
                'inputs_sha256': inputs,
                'frozen_anchors': {'chains': {name: spec['source'] for name, spec in
                                              rules['inputs']['anchors'].items()},
                                   'files': anchors},
                'family': {'path': str(Path(family).resolve()),
                           'task_sources': {**source_record, 'source_sha256': task_sources},
                           'task_runtime_signature': signature,
                           'tasks': {a: {str(s): v for s, v in t.items()} for a, t in tasks.items()}},
                'scorer_sources_sha256': sources, 'git': git_state(),
                'runtime': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                            'hostname': platform.node()},
                'seconds': time.perf_counter() - started, 'peak_rss': peak_rss(),
                'created_utc': datetime.now(timezone.utc).isoformat()},
        }
        write(destination, result)
    return result


def summary_markdown(result):
    """Korean summary: the one confirmatory verdict, every row of both splits, exploratory rows, weights, gate and
    inputs."""
    def ci(block):
        lo, hi = block['ci95'] or (float('nan'), float('nan'))
        return f"{block['delta']:+.5f} [{lo:+.5f}, {hi:+.5f}]"
    conf, pipe = result['confirmation'], result['pipeline_difference']
    lines = [f"# {result['experiment_id']} Stage B — 채택 시드 헤드 × 동결 G0-global 혼합 vs 동결 G15", '',
             f"- 확인 검정은 하나뿐: 미리 정한 주 후보 {conf['primary']} → **{conf['primary_status']}** "
             '(두 분할 모두 각자의 단측 검정 α .05로 predictive_improvement여야 confirmed, 교집합-합집합 규칙, Holm 없음).',
             '- 나머지 후보 행은 보조·탐색 결과입니다(분할마다 세 행 Holm 보정은 참고용). 보조 행은 "confirmed"가 될 수 '
             '없습니다. 채택 여부는 Song 결정이며 이 결과만으로 G15·G0·정책 식별자·서비스는 바뀌지 않습니다.',
             '- 음수 = G15보다 손실이 낮음. 경기 단위 부트스트랩 95% 구간. 반복 노출된 2025 DEV, 예측만.',
             '- 파이프라인 비교: CUDA 헤드는 epoch 상한 60, G15의 flatten 헤드는 MPS에서 상한 30으로 학습했습니다. '
             f"control/MIX_FIXED 행(같은 구조·같은 반반 혼합)이 그 차이를 직접 잽니다: Cpanel ΔNLL "
             f"{ci(pipe[CPANEL]['nll'])}, 전체 MLB ΔNLL {ci(pipe[MLB]['nll'])}.",
             f"- 재현 게이트: 통과(허용 오차 {result['reproduction_gate']['tolerance']}, 시드별 G15−G0 차이 포함).", '']
    for split in (CPANEL, MLB):
        block = result[split]
        lines += [f"## {split} (n={block['n']}, 경기 {block['games']})", '',
                  '| 행 | 역할 | ΔNLL [95%] | ΔBrier [95%] | 시드별 ΔNLL | N (판정 p, 보정) |', '|---|---|---|---|---|---|']
        for name, row in block['rows'].items():
            seeds = ' / '.join(f'{d:+.4f}' for d in row['seed_deltas'])
            n = '-' if row['N'] is None else f"{row['N']['status']} (p={row['p_decision']}, {row['p_adjustment']})"
            lines.append(f"| {name} | {row['role']} | {ci(row['paired']['nll'])} | {ci(row['paired']['brier'])} | "
                         f'{seeds} | {n} |')
        lines.append('')
    lines += ['## 확인 판정 (주 후보 하나, 두 분할 모두 통과해야 confirmed)', '', '| 행 | Cpanel | 전체 MLB | 판정 |',
              '|---|---|---|---|', f"| {conf['primary']} | {conf['splits'][CPANEL]} | {conf['splits'][MLB]} | "
                                  f"{conf['primary_status']} |", '',
              '## 보조 행 (탐색, confirmed 아님)', '', '| 행 | Cpanel | 전체 MLB | 두 분할 모두 통과 | 표시 |',
              '|---|---|---|---|---|']
    for name, row in result['secondary_exploratory']['rows'].items():
        lines.append(f"| {name} | {row[CPANEL]} | {row[MLB]} | {row['both_splits_predictive_improvement']} | "
                     f"{row['status']} |")
    lines += ['', '## MIX_FIT 가중치 (6월 blend 행에서만 선택)', '', '| 팔 | 시드별 w | 앙상블 w |', '|---|---|---|']
    for arm, rules in result['mix_weights'].items():
        w = rules[reg.MIX_FIT]
        lines.append(f"| {arm} | {' / '.join(f'{x:.2f}' for x in w['seeds'])} | {w['ensemble']:.2f} |")
    lines += ['', '## 혼합 전 헤드 − G15 헤드 (서술)', '', '| 팔 | Cpanel ΔNLL [95%] | 전체 MLB ΔNLL [95%] |', '|---|---|---|']
    for arm in result['descriptive_unblended_head_vs_g15_head']['cpanel_dev']:
        heads = result['descriptive_unblended_head_vs_g15_head']
        lines.append(f"| {arm} | {ci(heads['cpanel_dev'][arm]['paired']['nll'])} | "
                     f"{ci(heads['whole_mlb_dev'][arm]['paired']['nll'])} |")
    gate = result['reproduction_gate']['fields']
    lines += ['', '## 재현 게이트 (results/EXP-P16-001.json)', '', '| 항목 | 기록 | 재계산 | 최대 차이 |', '|---|---|---|---|']
    lines += [f"| {k} | {v['recorded']!r} | {v['recomputed']!r} | {v['abs_diff']:.3e} |" for k, v in gate.items()]
    prov = result['provenance']
    chains = ', '.join(f'{k}: {v}' for k, v in prov['frozen_anchors']['chains'].items())
    lines += ['', '## 출처', '', f"- config sha256 {prov['config']['sha256']}", f"- git {prov['git']}",
              f"- 동결 입력 고정 사슬: {chains} (EXP-P11-001 분석 manifest는 G0 bundle이 sha로 고정)",
              f"- 정렬: 순열 {len(result['alignment']['permuted'])}개 / 확인 {result['alignment']['checked']}개",
              f"- 소요 {prov['seconds']:.1f}초, 최대 RSS {prov['peak_rss']['ru_maxrss']} {prov['peak_rss']['unit']}",
              f"- R(12그룹 강건성): {result['robustness_R']['reason']}", '', '### 입력 sha256', '',
              '| 파일 | sha256 |', '|---|---|']
    lines += [f'| {path} | {digest} |' for path, digest in sorted(prov['inputs_sha256'].items())]
    return '\n'.join(lines) + '\n'


def write(destination, result):
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.with_name(f'.{destination.name}.tmp-{os.getpid()}')
    work.mkdir()
    rc.write_json_atomic(work / 'results.json', result)
    (work / 'SUMMARY.md').write_text(summary_markdown(result))
    (work / 'SHA256SUMS').write_text(''.join(f'{rc.sha256_file(work / n)}  {n}\n'
                                             for n in ('results.json', 'SUMMARY.md')))
    for name in ('results.json', 'SUMMARY.md', 'SHA256SUMS'):
        os.chmod(work / name, 0o444)
    if destination.exists():
        raise FileExistsError(f'{destination} appeared during scoring')
    os.rename(work, destination)
    return destination


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True, help='configs/EXP-P48-001.json (REGISTERED)')
    p.add_argument('--family', type=Path, required=True, help='pulled runner --output directory (arm/seed/...)')
    p.add_argument('--g15', type=Path, required=True, help='configs/G15-MIX-RESEARCH-FROZEN-v1.json')
    p.add_argument('--runs-root', type=Path, required=True, help='frozen ML-MATRIX-20260924 runs root (read only)')
    p.add_argument('--destination', type=Path, required=True, help='new directory outside any repository results/')
    p.add_argument('--source-manifest', type=Path, help=f'{sc.SOURCE_FORMAT} JSON of the exact task source map')
    p.add_argument('--g0-bundle', type=Path, help='default: <repo>/configs/G0-RESEARCH-FROZEN-v1.json (sha pinned)')
    p.add_argument('--recorded-results', type=Path, help='default: <repo>/results/EXP-P16-001.json (sha pinned)')
    args = p.parse_args(argv)
    result = score(args.config, args.family, args.g15, args.runs_root, args.destination, args.source_manifest,
                   args.g0_bundle, args.recorded_results)
    print('CUDA_MIX_VS_G15_SCORED', args.destination, result['confirmation']['primary'],
          result['confirmation']['primary_status'], flush=True)


if __name__ == '__main__':
    main()
