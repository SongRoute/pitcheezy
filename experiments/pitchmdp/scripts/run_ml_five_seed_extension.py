"""F1 five-seed extension worker: matched profiles, third-parent freeze, prepare, masked fit/predict.

Arm ``c1`` (full G0 seeds 3/4) runs through the unchanged frozen C1 runner
``run_ml_confirmation.py``; this module only adds a restricted-loader
entry point for the mandatory C1 profile and a heavy-locked wrapper for the
frozen C1 scorer. Arm ``f1ext`` fits two masked G0 members (seeds 3/4) whose
full counterparts are the C1 seed-3/4 fits, pinned through an immutable
third-parent manifest kept in the F1-extension output, never inside C1.
Every command re-verifies bundle pins (sources incl. both frozen scorers,
contracts, configs, repo commit), both original parents and the C1 parent.
The authoritative wall ledger is owned by ``supervise_ml_five_seed_extension.py``.
No DEV score is computed here; scoring is ``score_ml_five_seed_extension.py``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import stat
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np
import pandas as pd

from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.matrix_bridge import (BATTER_START, BATTER_STOP, CONTEXT_WIDTH, ROUTING_WIDTH, batter_train_volume,
                                    check_context_layout, fit_masked, masked_g0_predictor, masked_training_arrays,
                                    network_signature)
from pitchmdp.matrix_confirmation import validate_config as c1_validate_config
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_group_metrics import bootstrap_difference
from pitchmdp.matrix_metrics import paired_game_comparison
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_policy_artifacts import is_appledouble
from pitchmdp.matrix_sharing import SharingPredictor, training_arrays
from pitchmdp.model import outcome_labels
from pitchmdp.matrix_five_seed_extension import (BUDGET, EXPECTED_SAMPLES, G_EXPERIMENT, F1_EXPERIMENT, NEW_SEEDS,
    PARENT_CELL, PROFILE, REUSED_SEEDS, SEEDS, check_same_seed_pair, check_signature_matches_report,
    extension_member_identity, project_costs, third_parent_identity, validate_bundle, validate_c1_config_shape,
    validate_extension_config)
import run_ml_bridge as bridge
import run_ml_confirmation as c1
from run_ml_benchmark import read_json, dump, identity as base_identity, validate_native_runtime
from run_ml_matrix import PROTOCOL_DIR, assert_hashes, artifact_hashes, check_location, heavy_lock
from run_ml_sharing import identity as sharing_identity, verify as verify_sharing, load_data as sharing_load_data
from run_sequence_pilot import arrays
from score_ml_matrix import archive, assert_aligned, summarize_cell

PARTS = ('train', 'earlystop', 'temperature', 'blend', 'dev')
PROFILE_PARTS = ('train', 'earlystop', 'temperature')
FULL_PROBE_ROWS, FULL_PROBE_ATOL = 64, 1e-6
SOURCES = list(dict.fromkeys([*c1.SOURCES, 'scripts/score_ml_confirmation.py', *bridge.SOURCES,
                              'pitchmdp/matrix_policy_artifacts.py', 'pitchmdp/matrix_five_seed_extension.py',
                              'scripts/run_ml_five_seed_extension.py', 'scripts/score_ml_five_seed_extension.py',
                              'scripts/supervise_ml_five_seed_extension.py']))
CONTRACTS = {'runner': 'docs/contracts/ML-G0-F1-FIVE-SEED-RUNNER-v1.md',
             'draft': 'docs/contracts/ML-G0-F1-CONFIRMATION-DRAFT-v1.md',
             'c1_runner': 'docs/contracts/ML-C1-G-RUNNER-v1.md',
             'bridge': 'docs/contracts/ML-BATTER-BRIDGE-v1.md',
             'bridge_runner': 'docs/contracts/ML-BRIDGE-RUNNER-v1.md'}
COMMANDS = ('print-pins', 'c1-profile-restricted', 'c1-fit', 'profile', 'freeze-third-parent', 'prepare', 'fit', 'predict',
            'c1-score', 'status')


# ---------------------------------------------------------------- bundle, pins, identity

def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def source_hashes():
    return {name: hash_file(PROJECT / name) for name in SOURCES}


def contract_hashes():
    return {name: hash_file(resolve_path(path)) for name, path in CONTRACTS.items()}


def repo_head():
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if head.returncode != 0:
        raise RuntimeError('Cannot resolve the checked-out commit')
    return head.stdout.strip()


def load_bundle(path, *, real=True):
    return validate_bundle(read_json(path), real=real)


def verify_bundle_pins(bundle):
    """Every source (both frozen scorers included), contract, config and the checkout commit."""
    if set(bundle['sources']) != set(SOURCES):
        raise ValueError('Bundle source pins must cover exactly the five-seed source family')
    current = source_hashes()
    for name, digest in bundle['sources'].items():
        if current[name] != digest:
            raise ValueError('Pinned source changed: ' + name)
    for name, record in bundle['contracts'].items():
        if record['path'] != CONTRACTS[name] or hash_file(resolve_path(record['path'])) != record['sha256']:
            raise ValueError('Pinned contract changed: ' + name)
    for name, record in bundle['configs'].items():
        if hash_file(resolve_path(record['path'])) != record['sha256']:
            raise ValueError('Pinned configuration file changed: ' + name)
    if repo_head() != bundle['repo_commit']:
        raise ValueError('Checked-out commit differs from the registered repo_commit')
    return canonical_hash(bundle)


def load_configs(bundle):
    """Both new configs, validated by their own schemas and cross-bound to the bundle."""
    c1_config = validate_c1_config_shape(c1_validate_config(read_json(resolve_path(bundle['configs']['c1']['path']))))
    ext = validate_extension_config(read_json(resolve_path(bundle['configs']['f1ext']['path'])), real=True)
    arms = bundle['arms']
    declared = ext['parent_c1']
    if (declared['experiment_id'] != c1_config['experiment_id']
            or c1_config['experiment_id'] != arms['c1']['experiment_id']):
        raise ValueError('Declared C1 experiment identity differs between bundle, C1 config and F1 extension')
    if Path(declared['run']).resolve() != Path(arms['c1']['output']).resolve():
        raise ValueError('Declared C1 output differs from the bundle arm output')
    if declared['config_sha256'] != canonical_hash(c1_config):
        raise ValueError('Declared C1 config identity differs from the pinned C1 config')
    if resolve_path(declared['config_file']) != resolve_path(bundle['configs']['c1']['path']):
        raise ValueError('Declared C1 config file differs from the bundle pin')
    if ext['experiment_id'] != arms['f1ext']['experiment_id']:
        raise ValueError('F1 extension identity differs from the bundle arm')
    g = ext['parent_g']
    if (Path(g['run']).resolve() != Path(c1_config['parent_run']).resolve()
            or g['preparation_sha256'] != c1_config['parent_preparation_sha256']
            or g['analysis_manifest_sha256'] != c1_config['parent_analysis_sha256']):
        raise ValueError('C1 and F1 extension must pin the same frozen G parent')
    f1 = ext['parent_f1']
    if (resolve_path(f1['config_file']) != resolve_path(bundle['configs']['f1_parent']['path'])
            or f1['config_file_sha256'] != bundle['configs']['f1_parent']['sha256']):
        raise ValueError('F1 parent config pin differs between bundle and F1 extension')
    for config, arm in ((c1_config, 'c1'), (ext, 'f1ext')):
        registered = config['registration'].get('output')
        if registered is not None and Path(registered).resolve() != Path(arms[arm]['output']).resolve():
            raise ValueError(f'{arm} registration.output differs from the bundle arm output')
    return c1_config, ext


def identity(ext, local_path, bundle):
    return {**base_identity(ext, local_path), 'source_hashes': source_hashes(), 'contract_hashes': contract_hashes(),
            'bundle_sha256': canonical_hash(bundle), 'repo_commit': bundle['repo_commit']}


def arm_output(bundle, local, arm):
    output = Path(bundle['arms'][arm]['output']).resolve()
    check_location(local, output)
    return output


def ledger_location(bundle, local):
    """Coordination sibling under the protocol root, disjoint from both arm outputs and both parents."""
    root = Path(local['artifact_root']).resolve() / 'runs' / PROTOCOL_DIR
    ledger = Path(bundle['ledger_dir']).resolve()
    others = [Path(bundle['arms'][arm]['output']).resolve() for arm in bundle['arms']]
    if not ledger.is_relative_to(root) or ledger == root or any(
            ledger == o or ledger.is_relative_to(o) or o.is_relative_to(ledger) for o in others):
        raise ValueError('Ledger directory must be a coordination sibling under the protocol root, outside both arm outputs')
    return ledger


def profile_dir(bundle, local, arm):
    return ledger_location(bundle, local) / 'profiles' / arm


def verify(output, expected):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != expected:
        raise ValueError('F1 extension source/config/contract/bundle/environment changed from preparation')
    assert_hashes(output, prep['artifact_hashes'])
    for name, digest in prep['external_hashes'].items():
        if hash_file(Path(name)) != digest:
            raise ValueError('Frozen dependency changed: ' + name)
    return prep


def _fresh(directory, allowed=()):
    directory = Path(directory)
    if directory.exists():
        leftovers = [p for p in directory.rglob('*') if p.is_file() and not is_appledouble(p)
                     and str(p.relative_to(directory)) not in allowed]
        if leftovers:
            raise ValueError(f'Preserve existing content in {directory}; register a fresh attempt: {leftovers[:3]}')


def _freeze(path):
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


# ---------------------------------------------------------------- parents

def verify_g_parent(ext, local_path, local):
    g = ext['parent_g']
    parent = Path(g['run']).resolve()
    check_location(local, parent)
    shared_config = read_json(parent / 'registered_config.json')
    if shared_config.get('experiment_id') != G_EXPERIMENT or hash_file(parent / 'registered_config.json') != g['registered_config_sha256']:
        raise ValueError('G parent must be the registered EXP-P4-001 run')
    shared = verify_sharing(parent, sharing_identity(shared_config, local_path))
    analysis = parent / 'analysis' / 'panel'
    manifest = read_json(analysis / 'manifest.json')
    checks = {parent / 'preparation.json': g['preparation_sha256'], analysis / 'manifest.json': g['analysis_manifest_sha256'],
              analysis / 'results.json': g['analysis_results_sha256'], analysis / 'predictions.npz': manifest['predictions_sha256'],
              parent / 'baseline_predictions.npz': g['baseline_predictions_sha256'],
              parent / 'dev_metadata.parquet': g['dev_metadata_sha256']}
    if manifest['results_sha256'] != g['analysis_results_sha256']:
        raise ValueError('G analysis manifest disagrees with the pinned results digest')
    for path, digest in checks.items():
        if hash_file(path) != digest:
            raise ValueError('Frozen G artifact identity differs: ' + str(path))
    for path, digest in manifest['inputs'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('G analysis input changed: ' + path)
    member_hashes, reuse = bridge._verify_full_members(parent, shared)
    for name, digest in member_hashes.items():
        if name in manifest['inputs'] and manifest['inputs'][name] != digest:
            raise ValueError('G analysis manifest and preserved member disagree: ' + name)
    for name in PARTS:
        if shared['samples'][name]['n'] != EXPECTED_SAMPLES[name]:
            raise ValueError('Frozen G sample size differs from registration: ' + name)
    if shared['samples']['dev']['games'] != EXPECTED_SAMPLES['dev_games']:
        raise ValueError('Frozen Cpanel DEV game count differs from registration')
    external = {**manifest['inputs'], **member_hashes, **{str(p): d for p, d in checks.items()},
                str(parent / 'registered_config.json'): g['registered_config_sha256']}
    return {'run': parent, 'prep': shared, 'analysis': analysis, 'manifest': manifest, 'reuse': reuse, 'external': external}


def verify_f1_parent(ext, local_path, local, baseline, g):
    """Old three-seed masked members: identities, hashes, alignment, exact summary reconstruction, pairing."""
    f = ext['parent_f1']
    old = Path(f['run']).resolve()
    check_location(local, old)
    old_config = bridge.config_check(read_json(old / 'registered_config.json'))
    if (old_config['experiment_id'] != F1_EXPERIMENT or hash_file(old / 'registered_config.json') != f['registered_config_sha256']
            or hash_file(resolve_path(f['config_file'])) != f['config_file_sha256']
            or canonical_hash(read_json(resolve_path(f['config_file']))) != canonical_hash(old_config)):
        raise ValueError('F1 parent must be the registered EXP-P9-001-v2 run with its pinned config')
    old_prep = bridge.verify(old, bridge.identity(old_config, local_path))
    if hash_file(old / 'preparation.json') != f['preparation_sha256'] or Path(old_prep['parent_run']).resolve() != g['run']:
        raise ValueError('F1 parent preparation identity or G parent differs')
    analysis = old / 'analysis' / 'bridge'
    manifest = read_json(analysis / 'manifest.json')
    checks = {analysis / 'manifest.json': f['analysis_manifest_sha256'], analysis / 'results.json': f['analysis_results_sha256'],
              analysis / 'predictions.npz': manifest['predictions_sha256'],
              old / 'batter_train_volume.json': f['batter_train_volume_sha256'],
              old / 'baseline_predictions.npz': ext['parent_g']['baseline_predictions_sha256']}
    if manifest['results_sha256'] != f['analysis_results_sha256']:
        raise ValueError('F1 analysis manifest disagrees with the pinned results digest')
    for path, digest in checks.items():
        if hash_file(path) != digest:
            raise ValueError('Frozen F1 artifact identity differs: ' + str(path))
    for path, digest in manifest['inputs'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('F1 analysis input changed: ' + path)
    external = {**manifest['inputs'], **{str(p): d for p, d in checks.items()},
                str(old / 'preparation.json'): f['preparation_sha256'],
                str(old / 'registered_config.json'): f['registered_config_sha256']}
    members, reuse, pairing = [], [], []
    for seed in REUSED_SEEDS:
        dest = bridge.member_dir(old, seed)
        fitted = read_json(dest / 'fit_state.json')
        if fitted['identity'] != bridge.member_identity(old_prep, seed):
            raise ValueError('Preserved masked fit identity differs')
        assert_hashes(dest, fitted['artifact_hashes'])
        state = read_json(dest / 'prediction_state.json')
        if state['identity'] != bridge.member_identity(old_prep, seed) or state['fit_state_sha256'] != hash_file(dest / 'fit_state.json'):
            raise ValueError('Preserved masked prediction identity differs')
        assert_hashes(dest, state['artifact_hashes'])
        for name in ('fit_state.json', 'prediction_state.json', *fitted['artifact_hashes'], *state['artifact_hashes']):
            external[str(dest / name)] = hash_file(dest / name)
        member = archive(dest / 'predictions.npz')
        assert_aligned(member, baseline)
        members.append(member)
        full_fit = read_json(bridge.full_member_paths(g['run'], seed)['fit_dir'] / 'fit.json')
        masked_fit = read_json(dest / 'fit.json')
        pairing.append(check_same_seed_pair(full_fit, masked_fit, seed=seed))
        if masked_fit['network_signature']['network'] != full_fit['report']['network']:
            raise ValueError('Preserved masked network signature differs from G0')
        reuse.append({'seed': seed, 'member_dir': str(dest), 'source_run': str(old),
                      'delivery_temperature': read_json(dest / 'calibration.json')['delivery_temperature'],
                      'device': masked_fit['report']['device'], 'logical_fit_seconds': masked_fit['seconds_total'],
                      'logical_prediction_seconds': read_json(dest / 'prediction_runtime.json')['seconds']})
        if old_prep['full_reuse']['members'][seed]['fit_dir'] != str(bridge.full_member_paths(g['run'], seed)['fit_dir']):
            raise ValueError('Preserved F1 full-arm reference differs from the G parent')
    stored = archive(analysis / 'predictions.npz')
    frozen = read_json(analysis / 'results.json')
    for name in ('keys', 'y', 'game_pk', 'pitcher'):
        if not np.array_equal(stored[name], baseline['dev_' + name]):
            raise ValueError('Preserved F1 analysis metadata differs: ' + name)
    report, values = summarize_cell(members, baseline)
    for kind in ('primary', 'calibrated', 'raw', 'seed_primary'):
        if not np.array_equal(stored['masked_' + kind], values[kind]):
            raise ValueError('Preserved masked three-seed reconstruction differs: ' + kind)
    stored_report = frozen['reports']['masked']
    if (_normalized(report['selection']) != _normalized(stored_report['selection'])
            or [_normalized(s['blend_selection']) for s in report['seeds']] != [_normalized(s['blend_selection']) for s in stored_report['seeds']]):
        raise ValueError('Preserved masked June weights differ')
    for kind in ('primary', 'calibrated_ensemble', 'raw_ensemble'):
        for metric in ('log_loss', 'brier_multiclass'):
            if report[kind][metric] != stored_report[kind][metric]:
                raise ValueError('Preserved masked metric reconstruction differs: ' + kind)
    return {'run': old, 'prep': old_prep, 'analysis': analysis, 'external': external, 'reuse': reuse,
            'report': report, 'pairing': pairing, 'three_seed_result': frozen['primary']}


def verify_reused_references(ext, local_path, local):
    """Reconstruct both immutable three-seed arms before any new production fit.

    These are the already published parent predictions; no new member or
    five-seed quality result is opened by this audit.
    """
    g = verify_g_parent(ext, local_path, local)
    baseline = archive(g['run'] / 'baseline_predictions.npz')
    assert_aligned(baseline, baseline)
    stored = archive(g['analysis'] / 'predictions.npz')
    frozen = read_json(g['analysis'] / 'results.json')
    members = [archive(Path(item['member_dir']) / 'predictions.npz') for item in g['reuse']]
    full_report, full = bridge.verify_full_reconstruction(members, baseline, stored, frozen['reports'][PARENT_CELL])
    old = verify_f1_parent(ext, local_path, local, baseline, g)
    old_arrays = archive(old['analysis'] / 'predictions.npz')
    old_result = read_json(old['analysis'] / 'results.json')
    for kind in ('primary', 'calibrated', 'raw', 'seed_primary'):
        if not np.array_equal(old_arrays['full_' + kind], full[kind]):
            raise ValueError('Preserved F1 full-arm three-seed reconstruction differs: ' + kind)
    if _normalized(old_result['reports']['full']) != _normalized(full_report):
        raise ValueError('Preserved F1 full-arm three-seed summary differs from G0')
    return g, old


def verify_c1_parent(ext, c1_config, local_path, local):
    """Live verification of the C1 output as the F1 extension's third parent (seeds 3/4)."""
    declared = ext['parent_c1']
    c1_out = Path(declared['run']).resolve()
    check_location(local, c1_out)
    expected = c1.identity(c1_config, local_path)
    if declared['source_hashes'] is not None and expected['source_hashes'] != declared['source_hashes']:
        raise ValueError('Frozen C1 source hashes differ from the declared third parent')
    prep = c1.verify(c1_out, expected)
    registered = read_json(c1_out / 'registered_config.json')
    if canonical_hash(registered) != declared['config_sha256'] or registered['experiment_id'] != declared['experiment_id']:
        raise ValueError('C1 registered config differs from the declared third parent')
    if Path(prep['parent_run']).resolve() != Path(ext['parent_g']['run']).resolve() or prep['parent_preparation_sha256'] != ext['parent_g']['preparation_sha256']:
        raise ValueError('C1 third parent extends a different G parent')
    if prep['cells'] != [PARENT_CELL] or prep['primary_comparisons'] or prep['selection_status'] != 'baseline_stability_only':
        raise ValueError('C1 third parent must be the G0-global baseline-stability registration')
    resolved, fits, members = {}, {}, {}
    for seed in NEW_SEEDS:
        resolved[seed] = c1.resolve_member(c1_config, c1_out, prep, PARENT_CELL, seed)
        fit_dir = c1_out / 'fits' / f'seed{seed}' / 'global'
        fits[str(seed)] = {name: hash_file(fit_dir / name) for name in ('state.json', 'model.pt', 'fit.json')}
        folder = Path(resolved[seed]['directory'])
        members[str(seed)] = {name: hash_file(folder / name) for name in
                              ('prediction_state.json', 'predictions.npz', 'calibration.json', 'prediction_runtime.json')}
    body = {'c1_experiment_id': declared['experiment_id'], 'c1_run': str(c1_out), 'c1_config_sha256': canonical_hash(registered),
            'c1_registered_config_file_sha256': hash_file(c1_out / 'registered_config.json'),
            'c1_source_hashes': prep['identity']['source_hashes'], 'c1_preparation_sha256': hash_file(c1_out / 'preparation.json'),
            'c1_identity': {k: v for k, v in prep['identity'].items() if k != 'source_hashes'},
            'fits': fits, 'members': members}
    return {'run': c1_out, 'prep': prep, 'resolved': resolved, 'body': body}


def _pairing_new_seed(g, c1_out, seed):
    """C1 seed-3/4 fits must use the frozen TRAIN/early rows and the G0 network on the G device."""
    fit = read_json(c1_out / 'fits' / f'seed{seed}' / 'global' / 'fit.json')
    samples = g['prep']['samples']
    if fit['train_rows_sha256'] != samples['train']['rows_sha256'] or fit['earlystop_rows_sha256'] != samples['earlystop']['rows_sha256']:
        raise ValueError(f'C1 seed {seed} fit rows differ from the frozen G TRAIN/early-stop rows')
    reference = read_json(bridge.full_member_paths(g['run'], 0)['fit_dir'] / 'fit.json')['report']
    if fit['report']['network'] != reference['network'] or fit['report']['parameter_count'] != reference['parameter_count']:
        raise ValueError(f'C1 seed {seed} network differs from the frozen G0 network')
    if fit['report']['device'] != reference['device']:
        raise ValueError(f'C1 seed {seed} device {fit["report"]["device"]} differs from frozen G0 {reference["device"]}')
    return fit


# ---------------------------------------------------------------- print-pins (read-only)

def print_pins(bundle_path, local_path):
    bundle = load_bundle(bundle_path, real=False)
    c1_path = resolve_path(bundle['configs']['c1']['path'])
    ext_path = resolve_path(bundle['configs']['f1ext']['path'])
    c1_config = read_json(c1_path)
    pins = {'repo_commit': repo_head(), 'sources': source_hashes(), 'contracts': contract_hashes(),
            'configs': {name: hash_file(resolve_path(record['path'])) for name, record in bundle['configs'].items()},
            'parent_c1': {'config_sha256': canonical_hash(c1_config),
                          'source_hashes': {name: hash_file(PROJECT / name) for name in c1.SOURCES}},
            'f1ext_config_sha256': canonical_hash(read_json(ext_path)),
            'local_config_sha256': hash_file(local_path), 'note': 'Read-only; paste into the frozen configs before commit D'}
    print(json.dumps(pins, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- matched profiles (pre-prepare)

def _scoring_probe(n_dev, n_blend, games):
    """Time the frozen scoring functions on shape-matched random arrays; no real predictions or labels."""
    rng = np.random.default_rng(0)
    def part(n, g, offset):
        game_pk = np.sort(rng.integers(0, g, n)).astype(np.int64) + offset
        y = rng.integers(0, 10, n).astype(np.int64)
        keys = np.column_stack([game_pk, np.arange(n) // 4 + 1, np.arange(n) % 4 + 1]).astype(np.int64)
        return {'keys': keys, 'y': y, 'game_pk': game_pk, 'pitcher': rng.integers(1, 40, n).astype(np.int64)}
    baseline = {}
    for name, n, g, offset in (('blend', n_blend, 110, 5000), ('dev', n_dev, games, 9000)):
        for key, value in part(n, g, offset).items():
            baseline[name + '_' + key] = value
        p = rng.dirichlet(np.ones(10), n)
        baseline[name], baseline[name + '_raw'] = p, p
    def member():
        values = {k: v for k, v in baseline.items() if k not in ('blend', 'dev', 'blend_raw', 'dev_raw')}
        for name in ('blend', 'dev'):
            values[name] = rng.dirichlet(np.ones(10), len(baseline[name + '_y']))
            values[name + '_raw'] = rng.dirichlet(np.ones(10), len(baseline[name + '_y']))
        return values
    members = [member() for _ in range(len(SEEDS))]
    before = time.perf_counter()
    _, full = summarize_cell(members, baseline)
    _, masked = summarize_cell([member() for _ in range(len(SEEDS))], baseline)
    y, game = baseline['dev_y'], baseline['dev_game_pk']
    paired_game_comparison(y, full['primary'], masked['primary'], game, draws=10000, seed=20260924)
    from pitchmdp.matrix_metrics import pitch_losses
    delta = pitch_losses(y, full['primary']) - pitch_losses(y, masked['primary'])
    one = time.perf_counter()
    bootstrap_difference(delta, game, upper_alpha=.05 / 24, draws=100000, seed=20260924)
    bound = time.perf_counter() - one
    total = time.perf_counter() - before + 11 * bound + 2 * (one - before) / 3
    return {'scoring_probe_seconds': float(total), 'single_bound_seconds': float(bound),
            'shape': {'dev': n_dev, 'blend': n_blend, 'games': games, 'members': len(SEEDS)},
            'note': 'Frozen summarize_cell/paired bootstrap/24-bound R on random arrays of the registered shape; no data'}


def profile(bundle, ext, c1_config, local_path, local, arm, expected):
    """Matched 65,536-row profile for one arm; models discarded; output in the coordination sibling."""
    dest = profile_dir(bundle, local, arm)
    if (dest / 'state.json').exists():
        state = read_json(dest / 'state.json')
        if state['identity'] != expected or state['arm'] != arm:
            raise ValueError('Existing matched profile belongs to another registration')
        assert_hashes(dest, state['artifact_hashes'])
        print('FIVE_SEED_PROFILE_COMPLETE', arm, flush=True)
        return
    _fresh(dest)
    started = time.perf_counter()
    g, old = verify_reused_references(ext, local_path, local)
    parents = {'g_preparation_sha256': ext['parent_g']['preparation_sha256'],
               'f1_preparation_sha256': ext['parent_f1']['preparation_sha256'],
               'both_three_seed_references_reconstructed': True}
    del old
    verify_seconds = time.perf_counter() - started
    shared = g['prep']
    view = {'features': shared['features'], 'clusters': shared['clusters'],
            'samples': {name: shared['samples'][name] for name in PROFILE_PARTS}}
    before = time.perf_counter()
    store, context, parts, aux = sharing_load_data(local, g['run'], view)
    load_seconds = time.perf_counter() - before
    for name in PROFILE_PARTS:
        if ordered_key_hash(parts[name]) != shared['samples'][name]['rows_sha256']:
            raise ValueError('Profile sample differs from frozen G rows: ' + name)
    train = parts['train'].iloc[:PROFILE['train_rows']]
    early = parts['earlystop'].iloc[:PROFILE['earlystop_rows']]
    query = parts['temperature'].iloc[:PROFILE['temperature_rows']]
    train_arrays, early_arrays = arrays(store, context, train.index.to_numpy()), arrays(store, context, early.index.to_numpy())
    model = MatrixModel(ext['kind'], seed=NEW_SEEDS[0], width=ext['width'])
    before = time.perf_counter()
    if arm == 'full':
        model.fit(training_arrays(train_arrays), outcome_labels(train), training_arrays(early_arrays), outcome_labels(early),
                  epochs=PROFILE['epochs'], patience=PROFILE['epochs'], batch_size=BUDGET['batch_size'],
                  learning_rate=BUDGET['learning_rate'])
        predictor = SharingPredictor(PARENT_CELL, model, shared['clusters'])
    else:
        fit_masked(model, train_arrays, outcome_labels(train), early_arrays, outcome_labels(early),
                   epochs=PROFILE['epochs'], patience=PROFILE['epochs'], batch_size=BUDGET['batch_size'],
                   learning_rate=BUDGET['learning_rate'])
        predictor = masked_g0_predictor(model, shared['clusters'])
    fit_seconds = time.perf_counter() - before
    before = time.perf_counter()
    aux['delivery'].calibrate(predictor, store, context, query.index.to_numpy(), outcome_labels(query))
    calibration_seconds = time.perf_counter() - before
    before = time.perf_counter()
    p, _, _ = predict_streamed(predictor, aux['delivery'], store, context, query.index.to_numpy())
    inference_seconds = time.perf_counter() - before
    samples = {name: shared['samples'][name]['n'] for name in PARTS}
    probe = _scoring_probe(samples['dev'], samples['blend'], shared['samples']['dev']['games'])
    wall = time.perf_counter() - started
    if wall > bundle['caps']['profile']:
        raise TimeoutError('Matched profile exceeded the registered 600-second cap')
    if source_hashes() != expected['source_hashes']:
        raise ValueError('Sources changed during the matched profile')
    measured = {'command_overhead_seconds': verify_seconds, 'load_seconds': load_seconds, 'fit_seconds': fit_seconds,
                'calibration_seconds': calibration_seconds, 'inference_seconds': inference_seconds,
                'scoring_probe_seconds': probe['scoring_probe_seconds'], 'wall_seconds': wall}
    result = {'arm': arm, **PROFILE, 'draws': ext['draws'], 'device': model.device, 'measured': measured,
              'projection': project_costs(measured, samples, arm='c1' if arm == 'full' else 'f1ext'),
              'scoring_probe': probe, 'maximum_mass_error': float(np.abs(p.sum(1) - 1).max()),
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'parents': parents,
              'parts_loaded': list(PROFILE_PARTS), 'blend_rows_selected': False, 'dev_rows_selected': False,
              'dev_scores_read': False, 'profile_model_discarded': True,
              'profiled_utc': datetime.now(timezone.utc).isoformat()}
    del model, predictor
    dest.mkdir(parents=True, exist_ok=True)
    dump(dest / 'profile.json', result)
    dump(dest / 'state.json', {'identity': expected, 'arm': arm, 'artifact_hashes': artifact_hashes(dest, ['profile.json'])})
    print('FIVE_SEED_PROFILE_COMPLETE', arm, result['projection'], flush=True)


def load_profile(bundle, local, arm, expected):
    dest = profile_dir(bundle, local, arm)
    if not (dest / 'state.json').is_file():
        raise ValueError(f'Matched {arm} profile must complete before this stage')
    state = read_json(dest / 'state.json')
    if state['identity'] != expected or state['arm'] != arm:
        raise ValueError(f'Matched {arm} profile belongs to another registration')
    assert_hashes(dest, state['artifact_hashes'])
    result = read_json(dest / 'profile.json')
    if result['projection']['member_gate'] is not True or result['projection']['arm_gate'] is not True:
        raise ValueError(f'Matched {arm} profile projection exceeds the 7200-second member limit or arm share')
    return result, {'state_sha256': hash_file(dest / 'state.json'), 'profile_sha256': hash_file(dest / 'profile.json'),
                    'directory': str(dest)}


# ---------------------------------------------------------------- restricted mandatory C1 profile

def c1_profile_restricted(bundle, c1_config, local_path, local):
    """Unchanged frozen C1 profile logic, fed by a TRAIN/early/May-only loader; evidence in the sibling."""
    c1_out = arm_output(bundle, local, 'c1')
    prep = c1.verify(c1_out, c1.identity(c1_config, local_path))

    def restricted(local_, output_, prep_):
        parent = Path(prep_['parent_run'])
        gprep = read_json(parent / 'preparation.json')
        view = {'features': gprep['features'], 'clusters': gprep['clusters'],
                'samples': {name: gprep['samples'][name] for name in PROFILE_PARTS}}
        return sharing_load_data(local_, parent, view)

    original = c1.load_data
    c1.load_data = restricted
    try:
        c1.profile(c1_config, local, c1_out, prep)
    finally:
        c1.load_data = original
    evidence = ledger_location(bundle, local) / 'profiles' / 'c1_mandatory_loader.json'
    evidence.parent.mkdir(parents=True, exist_ok=True)
    dump(evidence, {'c1_output': str(c1_out), 'parts_loaded': list(PROFILE_PARTS), 'blend_rows_selected': False,
                    'dev_rows_selected': False,
                    'loader': 'run_ml_sharing.load_data on the G parent with TRAIN/early-stop/May samples only; profile logic unchanged',
                    'c1_profile_sha256': hash_file(c1_out / 'profile' / 'profile.json'),
                    'c1_profile_state_sha256': hash_file(c1_out / 'profile' / 'state.json'),
                    'c1_preparation_sha256': hash_file(c1_out / 'preparation.json'),
                    'source_hashes': source_hashes(), 'recorded_utc': datetime.now(timezone.utc).isoformat()})


def c1_fit(bundle, ext, c1_config, local_path, local, seed):
    """Charge the pre-fit reuse audit with the full fit under the supervisor wall cap."""
    if seed not in NEW_SEEDS:
        raise ValueError('Only seeds 3/4 are new full members')
    expected = identity(ext, local_path, bundle)
    for arm in ('full', 'masked'):
        load_profile(bundle, local, arm, expected)
    verify_reused_references(ext, local_path, local)
    output = arm_output(bundle, local, 'c1')
    prep = c1.verify(output, c1.identity(c1_config, local_path))
    c1.fit(c1_config, local, output, prep, seed)


# ---------------------------------------------------------------- third parent

def freeze_third_parent(bundle, ext, c1_config, local_path, local, output, expected):
    """Materialize the immutable C1 seed-3/4 manifest in the F1-extension output before prepare."""
    manifest_path = output / 'third_parent' / 'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        third_parent_identity(manifest, ext['parent_c1'])
        live = verify_c1_parent(ext, c1_config, local_path, local)
        for name in ('fits', 'members', 'c1_preparation_sha256', 'c1_source_hashes', 'c1_config_sha256'):
            if manifest[name] != live['body'][name]:
                raise ValueError('Frozen third-parent manifest no longer matches the live C1 output: ' + name)
        print('FIVE_SEED_THIRD_PARENT_FROZEN', manifest_path, flush=True)
        return
    _fresh(output)
    g = verify_g_parent(ext, local_path, local)
    live = verify_c1_parent(ext, c1_config, local_path, local)
    for seed in NEW_SEEDS:
        _pairing_new_seed(g, live['run'], seed)
    manifest = {**live['body'], 'frozen_utc': datetime.now(timezone.utc).isoformat(), 'registration_identity': expected,
                'bundle_sha256': canonical_hash(bundle), 'scope': 'C1 seed-3/4 full G0 fits/predictions pinned for the F1 extension; C1 output untouched'}
    third_parent_identity(manifest, ext['parent_c1'])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dump(manifest_path, manifest)
    _freeze(manifest_path)
    print('FIVE_SEED_THIRD_PARENT_FROZEN', manifest_path, flush=True)


# ---------------------------------------------------------------- prepare

def prepare(bundle, ext, c1_config, local_path, local, output, expected):
    if (output / 'preparation.json').exists():
        verify(output, expected)
        print('FIVE_SEED_PREPARED', output, flush=True)
        return
    _fresh(output, allowed=('third_parent/manifest.json',))
    manifest_path = output / 'third_parent' / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Freeze the C1 third-parent manifest before F1-extension prepare')
    started = time.perf_counter()
    manifest = read_json(manifest_path)
    third_parent_identity(manifest, ext['parent_c1'])
    g = verify_g_parent(ext, local_path, local)
    for other in (Path(ext['parent_f1']['run']).resolve(), Path(ext['parent_c1']['run']).resolve(), g['run']):
        if output == other or output.is_relative_to(other) or other.is_relative_to(output):
            raise ValueError('F1 extension output must be a distinct sibling of every parent')
    baseline = archive(g['run'] / 'baseline_predictions.npz')
    assert_aligned(baseline, baseline)
    stored = archive(g['analysis'] / 'predictions.npz')
    frozen = read_json(g['analysis'] / 'results.json')
    full_members = [archive(Path(item['member_dir']) / 'predictions.npz') for item in g['reuse']]
    full_report, _ = bridge.verify_full_reconstruction(full_members, baseline, stored, frozen['reports'][PARENT_CELL])
    old = verify_f1_parent(ext, local_path, local, baseline, g)
    live = verify_c1_parent(ext, c1_config, local_path, local)
    for name in ('fits', 'members', 'c1_preparation_sha256', 'c1_source_hashes', 'c1_config_sha256'):
        if manifest[name] != live['body'][name]:
            raise ValueError('Third-parent manifest differs from the live C1 output: ' + name)
    full_reuse = [dict(item, source_run=str(g['run'])) for item in g['reuse']]
    pairing = list(old['pairing'])
    for seed in NEW_SEEDS:
        fit = _pairing_new_seed(g, live['run'], seed)
        member = archive(Path(live['resolved'][seed]['predictions_path']))
        assert_aligned(member, baseline)
        folder = Path(live['resolved'][seed]['directory'])
        full_reuse.append({'seed': seed, 'fit_dir': str(live['run'] / 'fits' / f'seed{seed}' / 'global'),
                           'member_dir': str(folder), 'source_run': str(live['run']),
                           'delivery_temperature': read_json(folder / 'calibration.json')['delivery_temperature'],
                           'logical_fit_seconds': fit['seconds_total'], 'device': fit['report']['device'],
                           'logical_prediction_seconds': read_json(folder / 'prediction_runtime.json')['seconds']})
        pairing.append({'seed': seed, 'full_only': True, 'device': fit['report']['device'], 'network': fit['report']['network'],
                        'parameter_count': fit['report']['parameter_count'], 'train_rows_sha256': fit['train_rows_sha256'],
                        'earlystop_rows_sha256': fit['earlystop_rows_sha256'], 'masked_pending': True})
    if [m['seed'] for m in full_reuse] != list(SEEDS):
        raise ValueError('Full-arm members must be ordered seeds 0-4')
    external = {**g['external'], **old['external'], str(manifest_path): hash_file(manifest_path)}
    for seed in NEW_SEEDS:
        external.update(live['resolved'][seed]['hashes'])
    external[str(live['run'] / 'preparation.json')] = manifest['c1_preparation_sha256']
    profile_result, profile_pins = load_profile(bundle, local, 'masked', expected)
    external.update({str(Path(profile_pins['directory']) / 'state.json'): profile_pins['state_sha256'],
                     str(Path(profile_pins['directory']) / 'profile.json'): profile_pins['profile_sha256']})
    for name, digest in external.items():
        if hash_file(Path(name)) != digest:
            raise ValueError('Dependency changed before F1-extension preparation: ' + name)
    records = {name: g['prep']['samples'][name] for name in PARTS}
    output.mkdir(parents=True, exist_ok=True)
    for rel in SOURCES:
        target = output / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, target)
    for name, rel in CONTRACTS.items():
        target = output / 'contracts' / Path(rel).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolve_path(rel), target)
    dump(output / 'registered_config.json', ext)
    dump(output / 'declared_c1_config.json', c1_config)
    dump(output / 'bundle.json', bundle)
    dump(output / 'sharing_preparation.json', g['prep'])
    copies = [('parent_preparation.json', 'parent_preparation.json'), ('aux.pkl', 'aux.pkl'),
              ('baseline_predictions.npz', 'baseline_predictions.npz'), ('dev_metadata.parquet', 'dev_metadata.parquet'),
              *[(records[name]['path'], records[name]['path']) for name in PARTS]]
    for src, dst in copies:
        shutil.copyfile(g['run'] / src, output / dst)
        if hash_file(output / dst) != hash_file(g['run'] / src):
            raise ValueError('Copied frozen G artifact differs: ' + src)
    shutil.copyfile(old['run'] / 'batter_train_volume.json', output / 'batter_train_volume.json')
    if hash_file(output / 'batter_train_volume.json') != ext['parent_f1']['batter_train_volume_sha256']:
        raise ValueError('Copied batter volume differs from the pinned F1 parent')
    view = {'features': g['prep']['features'], 'clusters': g['prep']['clusters'], 'samples': records}
    store, context, parts, aux = sharing_load_data(local, output, view)
    for name in PARTS:
        if ordered_key_hash(parts[name]) != records[name]['rows_sha256']:
            raise ValueError('Rebuilt sample differs from frozen G: ' + name)
    layout = check_context_layout(aux['context'], context, parts['train'].iloc[:16])
    metadata = pd.read_parquet(output / 'dev_metadata.parquet')
    if (not np.array_equal(metadata[KEY].to_numpy(np.int64), parts['dev'][KEY].to_numpy(np.int64))
            or not np.array_equal(metadata.batter.to_numpy(np.int64), parts['dev'].batter.to_numpy(np.int64))):
        raise ValueError('Frozen DEV metadata and batter identities are unpaired')
    volume = batter_train_volume(parts['train'].batter.to_numpy(), quantile=ext['batter_volume']['quantile'])
    stored_volume = read_json(output / 'batter_train_volume.json')
    if volume['counts'] != stored_volume['counts'] or volume['q25'] != stored_volume['q25']:
        raise ValueError('Recomputed TRAIN batter volume differs from the pinned F1 parent')
    del store, context, parts, aux
    if source_hashes() != expected['source_hashes']:
        raise ValueError('Sources changed during F1-extension preparation')
    files = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and not is_appledouble(p)]
    prep = {'identity': expected, 'bundle_sha256': canonical_hash(bundle), 'parent_run': str(g['run']),
            'parent_f1_run': str(old['run']), 'parent_c1_run': str(live['run']), 'external_hashes': external,
            'samples': records, 'features': g['prep']['features'], 'clusters': g['prep']['clusters'], 'panel': g['prep']['panel'],
            'full_reuse': {'cell': PARENT_CELL, 'members': full_reuse, 'june_selection': full_report['selection'],
                           'june_seed_selection': [s['blend_selection'] for s in full_report['seeds']],
                           'reconstruction': 'exact three-seed raw/calibrated/primary/seed_primary and June weights'},
            'masked_reuse': {'members': old['reuse'], 'june_selection': old['report']['selection'],
                             'june_seed_selection': [s['blend_selection'] for s in old['report']['seeds']],
                             'reconstruction': 'exact three-seed masked arrays, June weights and metrics'},
            'three_seed_result': old['three_seed_result'],
            'third_parent': {'manifest_path': str(manifest_path), 'manifest_sha256': hash_file(manifest_path), 'manifest': manifest},
            'pairing': pairing, 'matched_profile': {'arm': 'masked', **profile_pins, 'projection': profile_result['projection']},
            'context_layout': layout,
            'mask': {'channels': [BATTER_START, BATTER_STOP], 'width': CONTEXT_WIDTH, 'routing_width': ROUTING_WIDTH,
                     'scope': 'training, early stopping, calibration and inference'},
            'refit': {'normalizer': False, 'batter_statistics': False, 'pitcher_representation': False,
                      'frequency_baseline': False, 'delivery': False,
                      'masked_new_fits': ['network seeds 3/4', 'May delivery temperature', 'June seed/ensemble blend']},
            'batter_volume': {k: v for k, v in volume.items() if k != 'counts'},
            'artifact_hashes': artifact_hashes(output, files), 'seconds': time.perf_counter() - started,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'prepared_utc': datetime.now(timezone.utc).isoformat(),
            'masked_dev_scores_computed': False, 'full_scores_read': 'frozen public G0/F1 analyses only, for exact reconstruction'}
    dump(output / 'preparation.json', prep)
    print('FIVE_SEED_PREPARED', output, flush=True)


# ---------------------------------------------------------------- fit / predict

def load_data(local, output, prep, parts=PARTS):
    view = {'features': prep['features'], 'clusters': prep['clusters'],
            'samples': {name: prep['samples'][name] for name in parts}}
    return sharing_load_data(local, output, view)


def member_dir(output, seed):
    if seed not in NEW_SEEDS:
        raise ValueError('Only seeds 3/4 are new masked members')
    return output / 'members' / 'masked' / f'seed{seed}'


def _c1_counterpart(ext, c1_config, local_path, local, prep, seed):
    live = verify_c1_parent(ext, c1_config, local_path, local)
    third = prep['third_parent']['manifest']
    for name in ('fits', 'members', 'c1_preparation_sha256', 'c1_source_hashes', 'c1_config_sha256'):
        if third[name] != live['body'][name]:
            raise ValueError('C1 third parent changed after F1-extension preparation: ' + name)
    fit_dir = live['run'] / 'fits' / f'seed{seed}' / 'global'
    return live, fit_dir, third['fits'][str(seed)]['state.json']


def fit(bundle, ext, c1_config, local_path, local, output, prep, seed):
    dest = member_dir(output, seed)
    statepath = dest / 'fit_state.json'
    live, c1_fit_dir, c1_state_sha = _c1_counterpart(ext, c1_config, local_path, local, prep, seed)
    membership = extension_member_identity(prep, seed, c1_fit_state_sha256=c1_state_sha)
    if statepath.exists():
        state = read_json(statepath)
        if state['identity'] != membership:
            raise ValueError('Completed masked fit identity changed')
        assert_hashes(dest, state['artifact_hashes'])
        print('FIVE_SEED_FIT_COMPLETE', seed, flush=True)
        return
    _fresh(dest)
    profile_result, pins = load_profile(bundle, local, 'masked', prep['identity'])
    if pins['state_sha256'] != prep['matched_profile']['state_sha256']:
        raise ValueError('Matched masked profile differs from the one pinned at preparation')
    c1_fit = read_json(c1_fit_dir / 'fit.json')
    device = MatrixModel().device
    if c1_fit['report']['device'] != device:
        raise ValueError(f'Device {device} differs from the C1 seed {seed} device {c1_fit["report"]["device"]}')
    parent_signature = network_signature(MatrixModel.load(c1_fit_dir / 'model.pt', device='cpu'))
    check_signature_matches_report(parent_signature, c1_fit['report'])
    start = time.perf_counter()
    store, context, parts, aux = load_data(local, output, prep, parts=('train', 'earlystop'))
    train, early = parts['train'], parts['earlystop']
    rows = {'train_rows_sha256': ordered_key_hash(train), 'earlystop_rows_sha256': ordered_key_hash(early)}
    if rows['train_rows_sha256'] != c1_fit['train_rows_sha256'] or rows['earlystop_rows_sha256'] != c1_fit['earlystop_rows_sha256']:
        raise ValueError(f'Masked seed {seed} TRAIN/early rows differ from the C1 counterpart')
    train_arrays, early_arrays = arrays(store, context, train.index.to_numpy()), arrays(store, context, early.index.to_numpy())
    bridge.check_input_shape(masked_training_arrays(early_arrays), parent_signature['network'])
    dest.mkdir(parents=True, exist_ok=True)
    model = fit_masked(MatrixModel(ext['kind'], seed=seed, width=ext['width']), train_arrays, outcome_labels(train),
                       early_arrays, outcome_labels(early), **BUDGET, checkpoint=dest / 'best_training.pt')
    signature = network_signature(model)
    if signature != parent_signature or model.device != device:
        raise ValueError('Masked network shape/parameter count/device differs from the C1 counterpart')
    mine = {'report': model.report, **rows}
    pairing = check_same_seed_pair(c1_fit, mine, seed=seed)
    model.save(dest / 'model.pt')
    dump(dest / 'fit.json', {'report': model.report, 'seconds_total': time.perf_counter() - start, **rows,
         'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'mask': [BATTER_START, BATTER_STOP],
         'network_signature': signature, 'pairing': pairing,
         'c1_counterpart': {'fit_dir': str(c1_fit_dir), 'state_sha256': c1_state_sha,
                            'model_sha256': prep['third_parent']['manifest']['fits'][str(seed)]['model.pt']},
         'checkpoint_note': 'best_training.pt is the best early-stop state written at fit end'})
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during masked fit')
    dump(statepath, {'identity': membership, 'artifact_hashes': artifact_hashes(dest, ['model.pt', 'best_training.pt', 'fit.json']),
                     'finished_utc': datetime.now(timezone.utc).isoformat()})
    print('FIVE_SEED_FIT_COMPLETE', seed, flush=True)


def full_path_probe(item, clusters, store, context, aux, blend):
    """Recompute the first C1 blend chunk through the shared G0 wrapper; guards the numerical path."""
    model = MatrixModel.load(Path(item['fit_dir']) / 'model.pt', device=item['device'])
    predictor = SharingPredictor(PARENT_CELL, model, clusters)
    predictor.delivery_temperature = item['delivery_temperature']
    rows = blend.index.to_numpy()[:FULL_PROBE_ROWS]
    probability, _, _ = predict_streamed(predictor, aux['delivery'], store, context, rows)
    with np.load(Path(item['member_dir']) / 'predictions.npz', allow_pickle=False) as frozen:
        preserved = frozen['blend'][:FULL_PROBE_ROWS]
        if not np.array_equal(frozen['blend_keys'][:FULL_PROBE_ROWS], blend[KEY].to_numpy(np.int64)[:FULL_PROBE_ROWS]):
            raise ValueError('Full-path probe rows differ from the C1 blend keys')
    difference = float(np.abs(probability - preserved).max())
    if not difference <= FULL_PROBE_ATOL:
        raise ValueError(f'Shared G0 numerical path no longer reproduces the C1 predictions ({difference:.3g})')
    return {'rows': len(rows), 'maximum_absolute_difference': difference, 'tolerance': FULL_PROBE_ATOL}


def predict(bundle, ext, c1_config, local_path, local, output, prep, seed):
    dest = member_dir(output, seed)
    live, c1_fit_dir, c1_state_sha = _c1_counterpart(ext, c1_config, local_path, local, prep, seed)
    membership = extension_member_identity(prep, seed, c1_fit_state_sha256=c1_state_sha)
    fitted = read_json(dest / 'fit_state.json')
    if fitted['identity'] != membership:
        raise ValueError('Masked fit identity changed')
    assert_hashes(dest, fitted['artifact_hashes'])
    statepath = dest / 'prediction_state.json'
    if statepath.exists():
        state = read_json(statepath)
        if state['identity'] != membership or state['fit_state_sha256'] != hash_file(dest / 'fit_state.json'):
            raise ValueError('Masked prediction differs from the fitted checkpoint')
        assert_hashes(dest, state['artifact_hashes'])
        print('FIVE_SEED_PREDICT_COMPLETE', seed, flush=True)
        return
    if any((dest / name).exists() for name in ('predictions.npz', 'calibration.json', 'prediction_runtime.json')):
        raise ValueError('Uncommitted masked predictions require failure review')
    item = prep['full_reuse']['members'][seed]
    if item['seed'] != seed or item['source_run'] != str(live['run']):
        raise ValueError('Full-arm counterpart order changed')
    device = MatrixModel().device
    if item['device'] != device:
        raise ValueError(f'Device {device} differs from the C1 seed {seed} device {item["device"]}')
    start = time.perf_counter()
    store, context, parts, aux = load_data(local, output, prep, parts=('temperature', 'blend', 'dev'))
    inner = MatrixModel.load(dest / 'model.pt', device=device)
    if inner.kind != ext['kind'] or inner.seed != seed:
        raise ValueError('Masked checkpoint architecture/seed changed')
    probe = full_path_probe(item, prep['clusters'], store, context, aux, parts['blend'])
    predictor = masked_g0_predictor(inner, prep['clusters'])
    temperature = parts['temperature']
    aux['delivery'].calibrate(predictor, store, context, temperature.index.to_numpy(), outcome_labels(temperature))
    dump(dest / 'calibration.json', predictor.report)
    baseline = archive(output / 'baseline_predictions.npz')
    values, counts = {}, {}
    for split in ('blend', 'dev'):
        part = parts[split]
        p, raw, levels = predict_streamed(predictor, aux['delivery'], store, context, part.index.to_numpy())
        values.update({split: p, split + '_raw': raw, split + '_delivery_level': levels,
                       split + '_keys': part[KEY].to_numpy(np.int64), split + '_y': outcome_labels(part),
                       split + '_game_pk': part.game_pk.to_numpy(np.int64), split + '_pitcher': part.pitcher.to_numpy(np.int64)})
        unique, n = np.unique(levels, return_counts=True)
        counts[split] = {str(int(k)): int(v) for k, v in zip(unique, n)}
    assert_aligned(values, baseline)
    counterpart = archive(Path(item['member_dir']) / 'predictions.npz')
    assert_aligned(counterpart, values)  # same June/DEV keys and labels as the C1 counterpart
    np.savez_compressed(dest / 'predictions.npz', **values)
    dump(dest / 'prediction_runtime.json', {'seconds': time.perf_counter() - start, 'device': device,
         'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'delivery_tier_counts': counts,
         'full_path_probe': probe, 'dev_scored': False})
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during masked prediction')
    dump(statepath, {'identity': membership, 'fit_state_sha256': hash_file(dest / 'fit_state.json'),
                     'artifact_hashes': artifact_hashes(dest, ['predictions.npz', 'prediction_runtime.json', 'calibration.json']),
                     'finished_utc': datetime.now(timezone.utc).isoformat()})
    print('FIVE_SEED_PREDICT_COMPLETE', seed, flush=True)


# ---------------------------------------------------------------- complete-family gate and C1 score wrapper

def verify_ten_predictions(bundle, ext, c1_config, local_path, local):
    """All ten ordered member predictions verified: full 0-4 via the C1 resolver, masked 0-2 old, 3-4 new."""
    ext_output = arm_output(bundle, local, 'f1ext')
    expected = identity(ext, local_path, bundle)
    prep = verify(ext_output, expected)
    live = verify_c1_parent(ext, c1_config, local_path, local)
    for name in ('fits', 'members', 'c1_preparation_sha256'):
        if prep['third_parent']['manifest'][name] != live['body'][name]:
            raise ValueError('C1 third parent changed after preparation: ' + name)
    full = {seed: c1.resolve_member(c1_config, live['run'], live['prep'], PARENT_CELL, seed) for seed in SEEDS}
    hashes = {}
    for seed in SEEDS:
        hashes.update(full[seed]['hashes'])
    old_run = Path(prep['parent_f1_run'])
    old_prep = read_json(old_run / 'preparation.json')
    if hash_file(old_run / 'preparation.json') != ext['parent_f1']['preparation_sha256']:
        raise ValueError('F1 parent preparation changed')
    masked = {}
    for seed in REUSED_SEEDS:
        dest = bridge.member_dir(old_run, seed)
        fitted, state = read_json(dest / 'fit_state.json'), read_json(dest / 'prediction_state.json')
        if (fitted['identity'] != bridge.member_identity(old_prep, seed) or state['identity'] != bridge.member_identity(old_prep, seed)
                or state['fit_state_sha256'] != hash_file(dest / 'fit_state.json')):
            raise ValueError('Preserved masked member identity differs')
        assert_hashes(dest, fitted['artifact_hashes'])
        assert_hashes(dest, state['artifact_hashes'])
        masked[seed] = dest
    missing = [seed for seed in NEW_SEEDS if not (member_dir(ext_output, seed) / 'prediction_state.json').is_file()]
    if missing:
        raise ValueError('Masked family incomplete; no score may open: seeds ' + str(missing))
    for seed in NEW_SEEDS:
        dest = member_dir(ext_output, seed)
        membership = extension_member_identity(prep, seed, c1_fit_state_sha256=prep['third_parent']['manifest']['fits'][str(seed)]['state.json'])
        fitted, state = read_json(dest / 'fit_state.json'), read_json(dest / 'prediction_state.json')
        if fitted['identity'] != membership or state['identity'] != membership or state['fit_state_sha256'] != hash_file(dest / 'fit_state.json'):
            raise ValueError(f'Masked seed {seed} identity differs')
        assert_hashes(dest, fitted['artifact_hashes'])
        assert_hashes(dest, state['artifact_hashes'])
        masked[seed] = dest
    for seed, dest in masked.items():
        for name in ('fit_state.json', 'prediction_state.json', 'fit.json', 'model.pt', 'predictions.npz', 'calibration.json', 'prediction_runtime.json'):
            hashes[str(dest / name)] = hash_file(dest / name)
    return {'prep': prep, 'c1': live, 'full': full, 'masked': masked, 'hashes': hashes, 'ext_output': ext_output}


def c1_score(bundle, ext, c1_config, local_path, local):
    """Frozen C1 scorer under the shared heavy lock, only after the ten-prediction gate."""
    import score_ml_confirmation
    verify_ten_predictions(bundle, ext, c1_config, local_path, local)
    score_ml_confirmation.score(c1_config, local_path, arm_output(bundle, local, 'c1'))


def status(bundle, ext, c1_config, local_path, local):
    c1_out, ext_out = arm_output(bundle, local, 'c1'), arm_output(bundle, local, 'f1ext')
    print('c1_prepared', (c1_out / 'preparation.json').is_file(), 'c1_profile', (c1_out / 'profile' / 'state.json').is_file())
    for arm in ('full', 'masked'):
        print('matched_profile', arm, (profile_dir(bundle, local, arm) / 'state.json').is_file())
    for seed in NEW_SEEDS:
        print('c1', seed, 'fitted', (c1_out / 'fits' / f'seed{seed}' / 'global' / 'state.json').is_file(),
              'predicted', (c1_out / 'members' / PARENT_CELL / f'seed{seed}' / 'prediction_state.json').is_file())
    print('third_parent_frozen', (ext_out / 'third_parent' / 'manifest.json').is_file(), 'f1ext_prepared', (ext_out / 'preparation.json').is_file())
    for seed in NEW_SEEDS:
        folder = member_dir(ext_out, seed)
        print('masked', seed, 'fitted', (folder / 'fit_state.json').is_file(), 'predicted', (folder / 'prediction_state.json').is_file())
    print('c1_scored', (c1_out / 'analysis' / 'manifest.json').is_file(), 'f1ext_scored', (ext_out / 'analysis' / 'five_seed' / 'manifest.json').is_file())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path, required=True, help='configs/ML-G0-F1-FIVE-SEED-v1.json (frozen copy, absolute path)')
    parser.add_argument('--local-config', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in COMMANDS:
        command = sub.add_parser(name)
        if name == 'profile':
            command.add_argument('--arm', choices=('full', 'masked'), required=True)
        if name in ('fit', 'predict', 'c1-fit'):
            command.add_argument('--seed', type=int, choices=NEW_SEEDS, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == 'print-pins':
        print_pins(args.bundle, args.local_config)
        return
    bundle = load_bundle(args.bundle, real=True)
    verify_bundle_pins(bundle)
    c1_config, ext = load_configs(bundle)
    local = read_json(args.local_config)
    validate_native_runtime()
    root = Path(local['artifact_root']).resolve()
    expected = identity(ext, args.local_config, bundle)
    if args.command == 'status':
        status(bundle, ext, c1_config, args.local_config, local)
        return
    with heavy_lock(root):
        if args.command == 'c1-profile-restricted':
            c1_profile_restricted(bundle, c1_config, args.local_config, local)
        elif args.command == 'c1-fit':
            c1_fit(bundle, ext, c1_config, args.local_config, local, args.seed)
        elif args.command == 'profile':
            profile(bundle, ext, c1_config, args.local_config, local, args.arm, expected)
        elif args.command == 'c1-score':
            c1_score(bundle, ext, c1_config, args.local_config, local)
        else:
            output = arm_output(bundle, local, 'f1ext')
            if args.command == 'freeze-third-parent':
                freeze_third_parent(bundle, ext, c1_config, args.local_config, local, output, expected)
            elif args.command == 'prepare':
                prepare(bundle, ext, c1_config, args.local_config, local, output, expected)
            else:
                prep = verify(output, expected)
                if args.command == 'fit':
                    fit(bundle, ext, c1_config, args.local_config, local, output, prep, args.seed)
                else:
                    predict(bundle, ext, c1_config, args.local_config, local, output, prep, args.seed)


if __name__ == '__main__':
    main()
