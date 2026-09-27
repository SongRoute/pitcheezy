"""Additive G0-only whole-MLB five-seed inference adapter (prepare / profile / predict).

Predicts the five frozen ``G0-global`` members (G ``EXP-P4-001`` seeds 0-2,
C1 ``EXP-P10-001`` seeds 3-4) on the T4-frozen eligible whole-MLB DEV
population (``EXP-P7-003`` keys/labels/metadata) into a NEW sibling output.
No fit, no May recalibration, no June reweighting: each member's checkpoint
and May delivery temperature are reused unchanged through the same
``SharingPredictor`` + 400-draw ``predict_streamed`` path as the sealed Cpanel
predictions, and every parent file is read-only.  Unlike
``run_ml_transfer.py`` this adapter never writes inside a parent member.

Every command re-verifies the three parents (identity, hashes, dependencies,
native environment) and the adapter's own source closure.  The profile is a
measured, timed gate on a registered number of real eligible rows with the
seed-0 member; predictions require a passing profile and a passing internal
ledger gate.  Failures preserve their outputs; nothing retries silently.
No DEV score is computed here; the scorer is a separate registration.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np
import pandas as pd

from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_g0_whole import (CELL, C1_EXPERIMENT, DEV_FIELDS, FIXED, G_EXPERIMENT, SEEDS,
    SOURCE_ARM, T4_EXPERIMENT, check_eligible_population, compare_predictions, cpanel_positions, frozen_weights,
    launch_gate, ledger_totals, member_identity, project_costs, require_within_tolerance, validate_config,
    validate_member_archive, validate_native_environment)
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_policy_artifacts import is_appledouble
from pitchmdp.matrix_sharing import SharingPredictor
from pitchmdp.model import outcome_labels
import run_ml_confirmation as c1
import run_ml_transfer as transfer
from run_ml_benchmark import dump, identity as base_identity, read_json, validate_native_runtime
from run_ml_matrix import artifact_hashes, assert_hashes, check_location, heavy_lock
from run_ml_sharing import (config_check as g_config_check, identity as sharing_identity,
                            load_data as sharing_load_data, verify as verify_sharing)
from score_ml_matrix import archive

SOURCES = list(dict.fromkeys([*c1.SOURCES, 'scripts/score_ml_confirmation.py', *transfer.SOURCES,
                              'pitchmdp/matrix_policy_artifacts.py', 'pitchmdp/matrix_g0_whole.py',
                              'scripts/run_ml_g0_whole.py']))
PARTS = ('mlb_dev', 'dev')
FIXED_DRAWS = FIXED['draws']
COPIES = {'parent_g_preparation.json': ('g', 'preparation.json'),
          'parent_g_registered_config.json': ('g', 'registered_config.json'),
          'dev_keys.parquet': ('g', 'mlb_dev_keys.parquet'),
          'dev_metadata.parquet': ('g', 'mlb_dev_metadata.parquet'),
          'cpanel_dev_keys.parquet': ('g', 'dev_keys.parquet'),
          'baseline_predictions.npz': ('g', 'baseline_predictions.npz'),
          'parent_c1_preparation.json': ('c1', 'preparation.json'),
          'parent_c1_registered_config.json': ('c1', 'registered_config.json'),
          'parent_c1_analysis_results.json': ('c1', 'analysis/results.json'),
          'parent_c1_analysis_manifest.json': ('c1', 'analysis/manifest.json'),
          'parent_t4_preparation.json': ('t4', 'preparation.json'),
          'parent_t4_registered_config.json': ('t4', 'registered_config.json')}
MUTABLE = ('ledger.jsonl',)


# ---------------------------------------------------------------- identity, verification, ledger

def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def source_hashes():
    return {name: hash_file(PROJECT / name) for name in SOURCES}


def identity(config, local_path):
    return {**base_identity(config, local_path), 'source_hashes': source_hashes()}


def stage_caps(config):
    budget = config['budget']
    return {'prepare': budget['single_member_wall_limit_seconds'], 'profile': budget['profile_wall_limit_seconds'],
            'predict': budget['single_member_wall_limit_seconds']}


def verify(output, expected):
    prep = read_json(output / 'preparation.json')
    if prep['identity'] != expected:
        raise ValueError('G0 whole-MLB config, source or native environment changed from preparation')
    assert_hashes(output, prep['artifact_hashes'])
    for name, digest in prep['external_hashes'].items():
        if hash_file(Path(name)) != digest:
            raise ValueError('Frozen parent dependency changed: ' + name)
    return prep


def _fresh(directory, allowed=()):
    directory = Path(directory)
    if directory.exists():
        leftovers = [p for p in directory.rglob('*') if p.is_file() and not is_appledouble(p)
                     and str(p.relative_to(directory)) not in allowed]
        if leftovers:
            raise ValueError(f'Preserve existing content in {directory}; register a fresh attempt: {leftovers[:3]}')


def _files(output):
    return [str(p.relative_to(output)) for p in output.rglob('*')
            if p.is_file() and not is_appledouble(p) and str(p.relative_to(output)) not in MUTABLE]


def ledger_entries(output):
    path = output / 'ledger.jsonl'
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_ledger(output, entry):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


class Active:
    def __init__(self, identifier, stage, seed):
        self.id, self.stage, self.seed, self.clock = identifier, stage, seed, time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.clock


@contextmanager
def ledger_stage(output, config, stage, seed=None):
    """Internal cross-check ledger: every command start/end, including failures. Never a retry."""
    caps = stage_caps(config)
    if stage not in caps:
        raise ValueError('Unregistered ledger stage')
    active = Active(uuid.uuid4().hex, stage, seed)
    _append_ledger(output, {'event': 'start', 'id': active.id, 'stage': stage, 'seed': seed, 'cap_seconds': caps[stage],
                            'unix': time.time(), 'utc': datetime.now(timezone.utc).isoformat()})
    try:
        yield active
    except BaseException as error:
        _append_ledger(output, {'event': 'end', 'id': active.id, 'status': 'failed', 'seconds': active.elapsed(),
                                'error': f'{type(error).__name__}: {error}', 'utc': datetime.now(timezone.utc).isoformat()})
        raise
    _append_ledger(output, {'event': 'end', 'id': active.id, 'status': 'completed', 'seconds': active.elapsed(),
                            'utc': datetime.now(timezone.utc).isoformat()})


def ledger_gate(output, config, active, projected, stage):
    totals = ledger_totals(ledger_entries(output), stage_caps(config), exclude=active.id)
    totals['charged_seconds'] += active.elapsed()
    return {**launch_gate(totals, projected=projected, budget_seconds=config['budget']['batch_wall_budget_seconds'],
                          stage=stage), 'ledger': totals}


def _stage_seconds(output, config, stage):
    """Charged wall of one stage: ended attempts plus reserved caps of unresolved starts."""
    entries = [e for e in ledger_entries(output) if e.get('event') == 'end' or e.get('stage') == stage]
    return ledger_totals(entries, stage_caps(config))['charged_seconds']


# ---------------------------------------------------------------- parents (read-only)

def _distinct(local, parent, output):
    check_location(local, parent)
    if parent == output or parent.is_relative_to(output) or output.is_relative_to(parent):
        raise ValueError('Whole-MLB output must be a distinct sibling of every parent')


def verify_g_parent(config, local_path, local, output, expected):
    g = config['parent_g']
    run = Path(g['run']).resolve()
    _distinct(local, run, output)
    registered = read_json(run / 'registered_config.json')
    if registered.get('experiment_id') != G_EXPERIMENT or hash_file(run / 'registered_config.json') != g['registered_config_sha256']:
        raise ValueError('G parent must be the pinned EXP-P4-001 run')
    shared = verify_sharing(run, sharing_identity(g_config_check(registered), local_path))
    validate_native_environment(shared['identity'], expected)
    analysis = run / 'analysis' / 'panel'
    manifest = read_json(analysis / 'manifest.json')
    checks = {run / 'preparation.json': g['preparation_sha256'], analysis / 'manifest.json': g['analysis_manifest_sha256'],
              analysis / 'results.json': manifest['results_sha256'], analysis / 'predictions.npz': manifest['predictions_sha256'],
              run / 'mlb_dev_keys.parquet': g['mlb_dev_keys_sha256'], run / 'mlb_dev_metadata.parquet': g['mlb_dev_metadata_sha256'],
              run / 'baseline_predictions.npz': g['baseline_predictions_sha256']}
    for path, digest in checks.items():
        if hash_file(path) != digest:
            raise ValueError('Frozen G artifact identity differs: ' + str(path))
    for path, digest in manifest['inputs'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('G analysis input changed: ' + path)
    samples, expected_samples = shared['samples'], config['expected_samples']
    if (samples['mlb_dev']['n'], samples['mlb_dev']['games']) != (expected_samples['dev'], expected_samples['dev_games']):
        raise ValueError('G eligible whole-MLB DEV sample differs from registration')
    if (samples['dev']['n'], samples['dev']['games']) != (expected_samples['cpanel_dev'], expected_samples['cpanel_games']):
        raise ValueError('G Cpanel DEV sample differs from registration')
    if (samples['temperature']['n'], samples['blend']['n']) != (expected_samples['temperature'], expected_samples['blend']):
        raise ValueError('G May/June samples differ from registration')
    if shared['units']['global'] != {'mode': 'global', 'train_n': samples['train']['n'], 'earlystop_n': samples['earlystop']['n']}:
        raise ValueError('G global unit differs from the frozen TRAIN/early-stop population')
    external = {**shared['external_hashes'], **{str(run / n): d for n, d in shared['artifact_hashes'].items()},
                **manifest['inputs'], **{str(p): d for p, d in checks.items()},
                str(run / 'registered_config.json'): g['registered_config_sha256']}
    return {'run': run, 'prep': shared, 'config': registered, 'analysis': analysis, 'manifest': manifest, 'external': external}


def verify_c1_parent(config, local_path, local, output, g):
    declared = config['parent_c1']
    run = Path(declared['run']).resolve()
    _distinct(local, run, output)
    c1_config = c1.validate_config(read_json(resolve_path(declared['config_file'])))
    registered = read_json(run / 'registered_config.json')
    if (registered.get('experiment_id') != C1_EXPERIMENT or canonical_hash(registered) != canonical_hash(c1_config)
            or hash_file(run / 'registered_config.json') != declared['registered_config_sha256']):
        raise ValueError('C1 parent must be the pinned EXP-P10-001 run with its pinned config file')
    prep = c1.verify(run, c1.identity(c1_config, local_path))
    if hash_file(run / 'preparation.json') != declared['preparation_sha256']:
        raise ValueError('C1 preparation identity differs from the pin')
    if Path(prep['parent_run']).resolve() != g['run'] or prep['parent_preparation_sha256'] != config['parent_g']['preparation_sha256']:
        raise ValueError('C1 parent extends a different G parent')
    if prep['cells'] != [CELL] or prep['primary_comparisons'] or prep['selection_status'] != 'baseline_stability_only':
        raise ValueError('C1 parent must be the G0-global baseline-stability registration')
    analysis = run / 'analysis'
    manifest = read_json(analysis / 'manifest.json')
    checks = {analysis / 'manifest.json': declared['analysis_manifest_sha256'], analysis / 'results.json': declared['analysis_results_sha256'],
              analysis / 'predictions.npz': manifest['predictions_sha256']}
    if manifest['results_sha256'] != declared['analysis_results_sha256']:
        raise ValueError('C1 analysis manifest disagrees with the pinned results digest')
    for path, digest in checks.items():
        if hash_file(path) != digest:
            raise ValueError('Frozen C1 analysis identity differs: ' + str(path))
    for path, digest in manifest['inputs'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('C1 analysis input changed: ' + path)
    results = read_json(analysis / 'results.json')
    if results.get('cells') != [CELL] or results.get('seeds') != list(SEEDS) or results.get('selection_status') != 'baseline_stability_only':
        raise ValueError('C1 analysis is not the five-seed G0-global baseline stability result')
    resolved = {seed: c1.resolve_member(c1_config, run, prep, CELL, seed) for seed in SEEDS}
    external = {**{str(p): d for p, d in checks.items()}, **manifest['inputs'],
                str(run / 'registered_config.json'): declared['registered_config_sha256'],
                str(run / 'preparation.json'): declared['preparation_sha256']}
    for item in resolved.values():
        external.update(item['hashes'])
    return {'run': run, 'prep': prep, 'config': c1_config, 'analysis': analysis, 'results': results,
            'resolved': resolved, 'external': external}


def verify_t4_parent(config, local_path, local, output, g):
    declared = config['parent_t4']
    run = Path(declared['run']).resolve()
    _distinct(local, run, output)
    registered = read_json(run / 'registered_config.json')
    if registered.get('experiment_id') != T4_EXPERIMENT or hash_file(run / 'registered_config.json') != declared['registered_config_sha256']:
        raise ValueError('T4 parent must be the pinned EXP-P7-003 run')
    t4_config = transfer.config_check(registered)
    prep = transfer.verify(run, transfer.identity(t4_config, local_path))
    checks = {run / 'preparation.json': declared['preparation_sha256'], run / 'dev_metadata.parquet': declared['dev_metadata_sha256'],
              run / 'baseline_predictions.npz': declared['baseline_predictions_sha256']}
    for path, digest in checks.items():
        if hash_file(path) != digest:
            raise ValueError('Frozen T4 artifact identity differs: ' + str(path))
    if Path(prep['parent_run']).resolve() != g['run']:
        raise ValueError('T4 parent evaluated a different G parent')
    mine, theirs = g['prep']['samples']['mlb_dev'], prep['samples']['dev']
    if (theirs['n'], theirs['games'], theirs['rows_sha256']) != (mine['n'], mine['games'], mine['rows_sha256']):
        raise ValueError('T4 frozen eligible whole-MLB rows differ from the G mlb_dev sample')
    if prep['coverage'] != g['prep']['coverage']['mlb_dev']:
        raise ValueError('T4 eligibility coverage differs from G')
    # T4 copied the G metadata/baseline byte-for-byte; the frozen eligibility is that exact file pair.
    if declared['dev_metadata_sha256'] != config['parent_g']['mlb_dev_metadata_sha256']:
        raise ValueError('T4 dev_metadata.parquet is not byte-identical to the G mlb_dev_metadata.parquet')
    if declared['baseline_predictions_sha256'] != config['parent_g']['baseline_predictions_sha256']:
        raise ValueError('T4 baseline_predictions.npz is not byte-identical to the G baseline archive')
    external = {**{str(p): d for p, d in checks.items()}, **{str(run / n): d for n, d in prep['artifact_hashes'].items()},
                str(run / 'registered_config.json'): declared['registered_config_sha256']}
    return {'run': run, 'prep': prep, 'config': t4_config, 'external': external}


def _member_records(config, g, c1_parent):
    """One immutable record per ordered seed: checkpoint, fit report, May temperature, sealed predictions."""
    members, networks, devices = {}, set(), set()
    for seed in SEEDS:
        item = c1_parent['resolved'][seed]
        expected_root = g['run'] if SOURCE_ARM[seed] == 'g' else c1_parent['run']
        if Path(item['source_run']).resolve() != expected_root or item['reused'] != (SOURCE_ARM[seed] == 'g'):
            raise ValueError(f'Seed {seed} member must come from the registered {SOURCE_ARM[seed]} arm')
        folder = Path(item['directory'])
        state = read_json(item['state_path'])
        if len(state['dependencies']) != 1:
            raise ValueError('G0-global member must depend on exactly one global checkpoint')
        (model_path, model_sha256), = state['dependencies'].items()
        fit_dir = Path(model_path).parent
        fit = read_json(fit_dir / 'fit.json')
        report = fit['report']
        network = report['network']
        if (report['kind'], report['seed'], network['n_context'], network['width']) != (config['kind'], seed, config['context_width'], config['width']):
            raise ValueError(f'Seed {seed} checkpoint report differs from the frozen G0 network contract')
        if (fit['train_rows_sha256'], fit['earlystop_rows_sha256']) != (g['prep']['samples']['train']['rows_sha256'], g['prep']['samples']['earlystop']['rows_sha256']):
            raise ValueError(f'Seed {seed} fit rows differ from the frozen G TRAIN/early-stop rows')
        calibration = read_json(folder / 'calibration.json')
        if calibration['cell'] != CELL or calibration['delivery_calibration_rows'] != config['expected_samples']['temperature']:
            raise ValueError(f'Seed {seed} May calibration is not the frozen 2,603-row G0 temperature')
        networks.add(canonical_hash(network)); devices.add(report['device'])
        members[str(seed)] = {'seed': seed, 'source_arm': SOURCE_ARM[seed], 'source_run': str(expected_root),
            'directory': str(folder), 'state_path': item['state_path'], 'prediction_state_sha256': item['state_sha256'],
            'predictions_sha256': item['predictions_sha256'], 'calibration_sha256': hash_file(folder / 'calibration.json'),
            'model_path': model_path, 'model_sha256': model_sha256, 'fit_dir': str(fit_dir),
            'fit_state_sha256': hash_file(fit_dir / 'state.json'), 'fit_sha256': hash_file(fit_dir / 'fit.json'),
            'delivery_temperature': calibration['delivery_temperature'], 'device': report['device'],
            'network': network, 'parameter_count': report['parameter_count'],
            'logical_fit_seconds': fit['seconds_total'],
            'cpanel_prediction_seconds': read_json(folder / 'prediction_runtime.json')['seconds']}
    if len(networks) != 1 or len(devices) != 1:
        raise ValueError('Five members must share one network signature and one fitted device')
    return members


# ---------------------------------------------------------------- prepare

def prepare(config, local_path, local, output, expected, active):
    if (output / 'preparation.json').exists():
        verify(output, expected)
        print('G0_WHOLE_PREPARED', output, flush=True)
        return
    _fresh(output, allowed=MUTABLE)
    started = time.perf_counter()
    g = verify_g_parent(config, local_path, local, output, expected)
    c1_parent = verify_c1_parent(config, local_path, local, output, g)
    t4 = verify_t4_parent(config, local_path, local, output, g)
    members = _member_records(config, g, c1_parent)
    calibration = frozen_weights(c1_parent['results']['reports'][CELL],
                                 {seed: record['delivery_temperature'] for seed, record in members.items()})
    # Population: ordered eligible keys/labels/metadata and the exact Cpanel overlap, from keys and labels only.
    saved = archive(g['run'] / 'baseline_predictions.npz')
    baseline = {'dev_' + name: saved['mlb_dev_' + name] for name in DEV_FIELDS}
    metadata = pd.read_parquet(g['run'] / 'mlb_dev_metadata.parquet')
    keys = pd.read_parquet(g['run'] / 'mlb_dev_keys.parquet')
    cpanel_keys = pd.read_parquet(g['run'] / 'dev_keys.parquet')
    if (ordered_key_hash(keys) != g['prep']['samples']['mlb_dev']['rows_sha256']
            or ordered_key_hash(cpanel_keys) != g['prep']['samples']['dev']['rows_sha256']
            or not np.array_equal(keys[KEY].to_numpy(np.int64), baseline['dev_keys'])):
        raise ValueError('Frozen eligible key files disagree with the G samples/baseline archive')
    population = check_eligible_population(baseline, metadata[KEY].to_numpy(np.int64), metadata.game_pk.to_numpy(np.int64),
                                           metadata.pitcher.to_numpy(np.int64), config['expected_samples'])
    overlap = cpanel_positions(baseline['dev_keys'], metadata.in_cpanel.to_numpy(bool), cpanel_keys[KEY].to_numpy(np.int64),
                               config['expected_samples'])
    overlap = {k: v for k, v in overlap.items() if k != 'positions'}
    external = {**g['external'], **c1_parent['external'], **t4['external']}
    output.mkdir(parents=True, exist_ok=True)
    for rel in SOURCES:
        target = output / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, target)
    roots = {'g': g['run'], 'c1': c1_parent['run'], 't4': t4['run']}
    for target, (arm, rel) in COPIES.items():
        shutil.copyfile(roots[arm] / rel, output / target)
    dump(output / 'registered_config.json', config)
    dump(output / 'frozen_calibration.json', calibration)
    report = {'identity': expected, 'experiment_id': config['experiment_id'], 'scope': 'Cmlb eligible whole-MLB DEV',
        'parents': {'g': {'run': str(g['run']), **{k: v for k, v in config['parent_g'].items() if k != 'run'}},
                    'c1': {'run': str(c1_parent['run']), **{k: v for k, v in config['parent_c1'].items() if k != 'run'}},
                    't4': {'run': str(t4['run']), **{k: v for k, v in config['parent_t4'].items() if k != 'run'}}},
        'samples': {'dev': g['prep']['samples']['mlb_dev'], 'cpanel_dev': g['prep']['samples']['dev']},
        'coverage': g['prep']['coverage']['mlb_dev'], 'population': population,
        'cpanel_overlap': {**overlap, 'note': 'Cpanel rows are the ordered in_cpanel subset; complement excludes them'},
        'features': g['prep']['features'], 'clusters': g['prep']['clusters'], 'panel': g['prep']['panel'],
        'members': members, 'frozen_calibration': calibration, 'budget': config['budget'],
        'equivalence_probe': config['equivalence_probe'], 'new_fits': 0, 'reused_fits': len(SEEDS),
        'external_hashes': external, 'artifact_hashes': artifact_hashes(output, _files(output)),
        'seconds': time.perf_counter() - started, 'command_seconds': active.elapsed(),
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'prepared_utc': datetime.now(timezone.utc).isoformat(),
        'scorer': 'not implemented here; scientific scoring rules are a separate registration',
        'dev_scores_read': False}
    if source_hashes() != expected['source_hashes']:
        raise ValueError('Sources changed during whole-MLB preparation')
    dump(output / 'preparation.json', report)
    print('G0_WHOLE_PREPARED', population, overlap, flush=True)


# ---------------------------------------------------------------- data, predictor, probe

def load_data(local, prep, parts=PARTS):
    """Read-only G loader restricted to the eligible whole-MLB and Cpanel DEV key sets."""
    g_run = Path(prep['parents']['g']['run'])
    gprep = read_json(g_run / 'preparation.json')
    if hash_file(g_run / 'preparation.json') != prep['parents']['g']['preparation_sha256']:
        raise ValueError('G preparation changed before loading')
    view = {'features': gprep['features'], 'clusters': gprep['clusters'],
            'samples': {name: gprep['samples'][name] for name in parts}}
    store, context, loaded, aux = sharing_load_data(local, g_run, view)
    if aux['delivery'].draws != FIXED_DRAWS:
        raise ValueError('Frozen delivery draw budget differs from the registered 400 draws')
    return store, context, loaded, aux


def load_member(config, member, clusters):
    """Frozen checkpoint + frozen May temperature through the unchanged G0 wrapper."""
    path = Path(member['model_path'])
    if hash_file(path) != member['model_sha256']:
        raise ValueError('Frozen checkpoint changed: ' + str(path))
    model = MatrixModel.load(path, device=None if config['device'] == 'auto' else config['device'])
    if (model.kind, model.seed, model.width, model.n_classes) != (config['kind'], member['seed'], config['width'], 10):
        raise ValueError('Loaded checkpoint identity differs from the member record')
    if model.report['network'] != member['network'] or model.report['parameter_count'] != member['parameter_count']:
        raise ValueError('Loaded checkpoint network differs from its fit report')
    if model.device != member['device']:
        raise ValueError(f'Device {model.device} differs from the frozen member device {member["device"]}')
    predictor = SharingPredictor(CELL, model, clusters, individual_tau=config['individual_tau'], cluster_tau=config['cluster_tau'])
    predictor.delivery_temperature = float(member['delivery_temperature'])
    predictor.report = {**predictor.report, 'delivery_temperature': predictor.delivery_temperature,
                        'delivery_calibration_rows': config['expected_samples']['temperature'],
                        'calibration_reused_unchanged': True}
    return predictor, model.device


def probe_cpanel(predictor, aux, store, context, dev_part, member, rows, atol):
    """Recompute the first Cpanel DEV rows and compare with the sealed member predictions."""
    selected = dev_part.iloc[:rows]
    with np.load(Path(member['directory']) / 'predictions.npz', allow_pickle=False) as sealed:
        if not np.array_equal(sealed['dev_keys'][:rows], selected[KEY].to_numpy(np.int64)):
            raise ValueError('Probe rows differ from the sealed Cpanel keys')
        reference = {'dev': sealed['dev'][:rows].copy(), 'dev_raw': sealed['dev_raw'][:rows].copy()}
    calibrated, raw, _ = predict_streamed(predictor, aux['delivery'], store, context, selected.index.to_numpy())
    reports = [require_within_tolerance(compare_predictions(calibrated, reference['dev'], atol=atol, name='probe_calibrated')),
               require_within_tolerance(compare_predictions(raw, reference['dev_raw'], atol=atol, name='probe_raw'))]
    return {'rows': int(len(selected)), 'seed': member['seed'], 'reports': reports}


def _check_parts(prep, parts):
    for name, sample in (('mlb_dev', prep['samples']['dev']), ('dev', prep['samples']['cpanel_dev'])):
        if len(parts[name]) != sample['n'] or ordered_key_hash(parts[name]) != sample['rows_sha256']:
            raise ValueError('Loaded rows differ from the frozen sample: ' + name)


# ---------------------------------------------------------------- profile (timed gate)

def profile(config, local, output, prep, active):
    dest = output / 'profile'
    if (dest / 'state.json').exists():
        state = read_json(dest / 'state.json')
        if state['preparation_sha256'] != hash_file(output / 'preparation.json'):
            raise ValueError('Profile belongs to another preparation')
        assert_hashes(dest, state['artifact_hashes'])
        print('G0_WHOLE_PROFILE_COMPLETE', flush=True)
        return
    _fresh(dest)
    budget, probe_config = config['budget'], config['equivalence_probe']
    overhead = active.elapsed()
    started = time.perf_counter()
    store, context, parts, aux = load_data(local, prep)
    _check_parts(prep, parts)
    load_seconds = time.perf_counter() - started
    member = prep['members']['0']
    predictor, device = load_member(config, member, prep['clusters'])
    before = time.perf_counter()
    probe = probe_cpanel(predictor, aux, store, context, parts['dev'], member, probe_config['rows'], probe_config['atol'])
    probe_seconds = time.perf_counter() - before
    rows = parts['mlb_dev'].iloc[:budget['profile_rows']]
    before = time.perf_counter()
    p, _, levels = predict_streamed(predictor, aux['delivery'], store, context, rows.index.to_numpy())
    inference_seconds = time.perf_counter() - before
    wall = time.perf_counter() - started + overhead
    if wall > budget['profile_wall_limit_seconds']:
        raise TimeoutError(f'Whole-MLB profile wall {wall:.1f}s exceeded the registered {budget["profile_wall_limit_seconds"]}s cap')
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during the whole-MLB profile')
    measured = {'rows': int(len(rows)), 'command_overhead_seconds': overhead, 'load_seconds': load_seconds,
                'probe_seconds': probe_seconds, 'inference_seconds': inference_seconds, 'wall_seconds': wall}
    projection = project_costs(measured, n_dev=prep['samples']['dev']['n'], probe_rows=probe_config['rows'], budget=budget,
                               prepare_seconds=_stage_seconds(output, config, 'prepare'))
    t4 = Path(prep['parents']['t4']['run'])
    t4_config = read_json(output / 'parent_t4_registered_config.json')
    reference = {cell: [read_json(t4 / 'members' / cell / f'seed{seed}' / 'prediction_runtime.json')['seconds'] for seed in (0, 1, 2)]
                 for cell in (t4_config['candidate'], t4_config['control'])}
    unique, counts = np.unique(levels, return_counts=True)
    result = {'seed': member['seed'], 'device': device, 'draws': config['draws'], 'measured': measured, 'projection': projection,
              'equivalence_probe': probe, 'delivery_tier_counts': {str(int(k)): int(v) for k, v in zip(unique, counts)},
              'maximum_mass_error': float(np.abs(p.sum(1) - 1).max()),
              'parent_t4_whole_mlb_member_seconds': {**reference, 'note': 'G2/G3 cells, planning reference only'},
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'profile_predictions_discarded': True, 'dev_scores_read': False,
              'profiled_utc': datetime.now(timezone.utc).isoformat()}
    dest.mkdir(parents=True, exist_ok=True)
    dump(dest / 'profile.json', result)
    dump(dest / 'state.json', {'preparation_sha256': hash_file(output / 'preparation.json'),
                               'artifact_hashes': artifact_hashes(dest, ['profile.json'])})
    print('G0_WHOLE_PROFILE_COMPLETE', projection, flush=True)


def require_profile(output):
    state_path = output / 'profile' / 'state.json'
    if not state_path.is_file():
        raise ValueError('Registered whole-MLB profile must complete before any prediction')
    state = read_json(state_path)
    if state['preparation_sha256'] != hash_file(output / 'preparation.json'):
        raise ValueError('Matching whole-MLB profile required')
    assert_hashes(output / 'profile', state['artifact_hashes'])
    projection = read_json(output / 'profile' / 'profile.json')['projection']
    if projection['member_gate'] is not True or projection['batch_gate'] is not True:
        raise ValueError('Whole-MLB projected cost exceeds the registered member limit or batch budget')
    return projection


# ---------------------------------------------------------------- predict

def member_dir(output, seed):
    if seed not in SEEDS:
        raise ValueError('Unregistered whole-MLB member seed')
    return output / 'members' / CELL / f'seed{seed}'


def verify_member(output, prep, seed):
    folder = member_dir(output, seed)
    state = read_json(folder / 'prediction_state.json')
    if state['identity'] != member_identity(prep, seed):
        raise ValueError(f'Whole-MLB member seed {seed} identity differs')
    if set(state['artifact_hashes']) != {'predictions.npz', 'prediction_runtime.json', 'source_prediction_state.json', 'source_calibration.json'}:
        raise ValueError('Whole-MLB member artifact family differs')
    assert_hashes(folder, state['artifact_hashes'])
    for path, digest in state['dependencies'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('Whole-MLB member dependency changed: ' + path)
    return state


def predict(config, local, output, prep, seed, active):
    folder = member_dir(output, seed)
    state_path = folder / 'prediction_state.json'
    if state_path.exists():
        verify_member(output, prep, seed)
        print('G0_WHOLE_PREDICT_COMPLETE', CELL, seed, flush=True)
        return
    _fresh(folder)
    projection = require_profile(output)
    gate = ledger_gate(output, config, active, projection['member_seconds'], 'predict')
    budget, probe_config = config['budget'], config['equivalence_probe']
    member = prep['members'][str(seed)]
    started = time.perf_counter()
    store, context, parts, aux = load_data(local, prep)
    _check_parts(prep, parts)
    predictor, device = load_member(config, member, prep['clusters'])
    probe = probe_cpanel(predictor, aux, store, context, parts['dev'], member, probe_config['rows'], probe_config['atol'])
    part = parts['mlb_dev']
    calibrated, raw, levels = predict_streamed(predictor, aux['delivery'], store, context, part.index.to_numpy())
    values = {'dev': calibrated, 'dev_raw': raw, 'dev_delivery_level': levels, 'dev_keys': part[KEY].to_numpy(np.int64),
              'dev_y': outcome_labels(part), 'dev_game_pk': part.game_pk.to_numpy(np.int64),
              'dev_pitcher': part.pitcher.to_numpy(np.int64)}
    saved = archive(output / 'baseline_predictions.npz')
    baseline = {'dev_' + name: saved['mlb_dev_' + name] for name in DEV_FIELDS}
    validate_member_archive(values, baseline)
    metadata = pd.read_parquet(output / 'dev_metadata.parquet')
    cpanel_keys = pd.read_parquet(output / 'cpanel_dev_keys.parquet')[KEY].to_numpy(np.int64)
    overlap = cpanel_positions(values['dev_keys'], metadata.in_cpanel.to_numpy(bool), cpanel_keys, config['expected_samples'])
    with np.load(Path(member['directory']) / 'predictions.npz', allow_pickle=False) as sealed:
        if not np.array_equal(sealed['dev_keys'], cpanel_keys) or not np.array_equal(sealed['dev_y'], values['dev_y'][overlap['positions']]):
            raise ValueError('Sealed Cpanel keys/labels differ from the whole-MLB overlap rows')
        alignment = [compare_predictions(values['dev'][overlap['positions']], sealed['dev'], atol=probe_config['atol'], name='cpanel_subset_calibrated'),
                     compare_predictions(values['dev_raw'][overlap['positions']], sealed['dev_raw'], atol=probe_config['atol'], name='cpanel_subset_raw')]
    unique, counts = np.unique(levels, return_counts=True)
    elapsed = time.perf_counter() - started
    command_seconds = active.elapsed()
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'predictions.npz', **values)
    dump(folder / 'prediction_runtime.json', {'seconds': elapsed, 'command_seconds': command_seconds, 'device': device,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'delivery_tier_counts': {str(int(k)): int(v) for k, v in zip(unique, counts)},
        'equivalence_probe': probe, 'cpanel_subset_alignment': alignment,
        'cpanel_overlap': {k: v for k, v in overlap.items() if k != 'positions'}, 'ledger_gate': gate,
        'within_member_limit': command_seconds <= budget['single_member_wall_limit_seconds'],
        'projected_member_seconds': projection['member_seconds'], 'dev_scored': False})
    # Evidence is on disk before any refusal; a refusal leaves no prediction_state.json and no retry.
    for report in alignment:
        require_within_tolerance(report)
    if command_seconds > budget['single_member_wall_limit_seconds']:
        raise TimeoutError(f'Member seed {seed} wall {command_seconds:.1f}s exceeded the registered '
                           f'{budget["single_member_wall_limit_seconds"]}s limit; output preserved for review')
    if source_hashes() != prep['identity']['source_hashes']:
        raise ValueError('Sources changed during whole-MLB prediction')
    shutil.copyfile(Path(member['state_path']), folder / 'source_prediction_state.json')
    shutil.copyfile(Path(member['directory']) / 'calibration.json', folder / 'source_calibration.json')
    dependencies = {member['model_path']: member['model_sha256'], member['state_path']: member['prediction_state_sha256'],
                    str(Path(member['directory']) / 'calibration.json'): member['calibration_sha256'],
                    str(Path(member['directory']) / 'predictions.npz'): member['predictions_sha256'],
                    str(output / 'preparation.json'): hash_file(output / 'preparation.json')}
    dump(state_path, {'identity': member_identity(prep, seed), 'dependencies': dependencies,
        'artifact_hashes': artifact_hashes(folder, ['predictions.npz', 'prediction_runtime.json',
                                                    'source_prediction_state.json', 'source_calibration.json']),
        'finished_utc': datetime.now(timezone.utc).isoformat()})
    print('G0_WHOLE_PREDICT_COMPLETE', CELL, seed, flush=True)


# ---------------------------------------------------------------- status / main

def status(output, expected):
    prepared = (output / 'preparation.json').is_file()
    print('prepared', prepared, 'profile', (output / 'profile' / 'state.json').is_file())
    if prepared:
        verify(output, expected)
    for seed in SEEDS:
        print(CELL, seed, SOURCE_ARM[seed], 'predicted' if (member_dir(output, seed) / 'prediction_state.json').is_file() else 'pending')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'profile', 'status'):
        sub.add_parser(name)
    predict_parser = sub.add_parser('predict')
    predict_parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    config = validate_config(read_json(args.config), real=True)
    local = read_json(args.local_config)
    output = args.output.resolve()
    root = check_location(local, output)
    validate_native_runtime()
    expected = identity(config, args.local_config)
    if args.command == 'status':
        status(output, expected)
        return
    with heavy_lock(root):
        stage = 'predict' if args.command == 'predict' else args.command
        with ledger_stage(output, config, stage, getattr(args, 'seed', None)) as active:
            if args.command == 'prepare':
                prepare(config, args.local_config, local, output, expected, active)
            else:
                prep = verify(output, expected)
                if args.command == 'profile':
                    profile(config, local, output, prep, active)
                else:
                    predict(config, local, output, prep, args.seed, active)


if __name__ == '__main__':
    main()
