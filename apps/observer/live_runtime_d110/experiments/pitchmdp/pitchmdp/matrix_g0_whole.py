"""Pure helpers for the additive G0-only whole-MLB five-seed inference adapter.

The adapter predicts the frozen five ``G0-global`` members (G ``EXP-P4-001``
seeds 0-2, C1 ``EXP-P10-001`` seeds 3-4) on the T4-frozen eligible whole-MLB
DEV population without any new fit, May recalibration or June reweighting.
This module holds only schema validation, identity, cost projection, ledger
arithmetic and array alignment rules; nothing here reads real data, opens a
checkpoint or computes a DEV score.  Scoring is a separate registration.
"""
from __future__ import annotations

import math

import numpy as np

from .matrix_data import canonical_hash

PROTOCOL = 'ml_g0_whole_mlb_v1'
CELL = 'G0-global'
SEEDS = (0, 1, 2, 3, 4)
SOURCE_ARM = {0: 'g', 1: 'g', 2: 'g', 3: 'c1', 4: 'c1'}
G_EXPERIMENT, C1_EXPERIMENT, T4_EXPERIMENT = 'EXP-P4-001', 'EXP-P10-001', 'EXP-P7-003'
DEV_FIELDS = ('keys', 'y', 'game_pk', 'pitcher')
ARCHIVE_FIELDS = ('dev', 'dev_raw', 'dev_delivery_level', 'dev_keys', 'dev_y', 'dev_game_pk', 'dev_pitcher')
FIXED = {'protocol': PROTOCOL, 'scope': 'Cmlb', 'cell': CELL, 'seeds': list(SEEDS),
         'source_arms': {str(seed): arm for seed, arm in SOURCE_ARM.items()},
         'draws': 400, 'kind': 'flatten_mlp', 'width': 128, 'device': 'auto', 'history_length': 5,
         'context_width': 52, 'routing_width': 7, 'individual_tau': 1000, 'cluster_tau': 10000,
         'calibration': {'may_temperature': 'reuse each frozen member calibration.json unchanged',
                         'june_weights': 'reuse C1 analysis G0-global ensemble and per-seed blend weights unchanged',
                         'refit': False},
         'expected_samples': {'dev': 311721, 'dev_games': 1161, 'cpanel_dev': 12334, 'cpanel_games': 328,
                              'temperature': 2603, 'blend': 4821},
         'new_fits': 0, 'independent_confirmation': None, 'held_out_confirmation': False}
G_PINS = ('run', 'preparation_sha256', 'registered_config_sha256', 'analysis_manifest_sha256',
          'mlb_dev_keys_sha256', 'mlb_dev_metadata_sha256', 'baseline_predictions_sha256')
C1_PINS = ('run', 'config_file', 'preparation_sha256', 'registered_config_sha256',
           'analysis_manifest_sha256', 'analysis_results_sha256')
T4_PINS = ('run', 'preparation_sha256', 'registered_config_sha256', 'dev_metadata_sha256',
           'baseline_predictions_sha256')
BUDGET_KEYS = ('profile_rows', 'profile_wall_limit_seconds', 'single_member_wall_limit_seconds',
               'batch_wall_budget_seconds')
PROBE_KEYS = ('rows', 'atol')
STAGES = ('prepare', 'profile', 'predict')
HEX = set('0123456789abcdef')


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def pin(value, *, real, name):
    """Draft configs may carry null pins; a real command refuses them."""
    if value is None:
        if real:
            raise ValueError(f'Null identity pin is not allowed in a real command: {name}; freeze the registration first')
        return
    if not _sha(value):
        raise ValueError(f'Identity pin must be a lowercase hex SHA256: {name}')


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# ---------------------------------------------------------------- configuration

def validate_config(config, *, real=True):
    required = {*FIXED, 'experiment_id', 'parent_g', 'parent_c1', 'parent_t4', 'budget',
                'equivalence_probe', 'registration'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Invalid G0 whole-MLB configuration schema')
    for name, value in FIXED.items():
        if config[name] != value:
            raise ValueError('G0 whole-MLB setting differs from the frozen protocol: ' + name)
    if not isinstance(config['experiment_id'], str) or not config['experiment_id'].strip():
        raise ValueError('Registered experiment identity required')
    parents = (('parent_g', G_EXPERIMENT, G_PINS), ('parent_c1', C1_EXPERIMENT, C1_PINS),
               ('parent_t4', T4_EXPERIMENT, T4_PINS))
    runs = set()
    for name, experiment, pins in parents:
        record = config[name]
        if not isinstance(record, dict) or set(record) != {'experiment_id', *pins}:
            raise ValueError(f'{name} requires exactly: experiment_id, ' + ', '.join(pins))
        if record['experiment_id'] != experiment:
            raise ValueError(f'{name} must be {experiment}')
        for key in pins:
            if key.endswith('sha256'):
                pin(record[key], real=real, name=f'{name}.{key}')
            elif not isinstance(record[key], str) or not record[key]:
                raise ValueError(f'{name}.{key} must be a nonempty path')
        runs.add(record['run'])
    if len(runs) != 3:
        raise ValueError('The three parent runs must be distinct')
    budget = config['budget']
    if not isinstance(budget, dict) or set(budget) != set(BUDGET_KEYS):
        raise ValueError('Budget requires exactly: ' + ', '.join(BUDGET_KEYS))
    for key in BUDGET_KEYS:
        value = budget[key]
        if value is None:
            if real:
                raise ValueError(f'Budget {key} must be frozen before a real command')
            continue
        if not _positive_int(value):
            raise ValueError(f'Budget {key} must be a positive integer')
    if budget['profile_rows'] is not None and budget['profile_rows'] > FIXED['expected_samples']['dev']:
        raise ValueError('Profile rows cannot exceed the eligible whole-MLB DEV size')
    if (budget['profile_wall_limit_seconds'] is not None and budget['single_member_wall_limit_seconds'] is not None
            and budget['profile_wall_limit_seconds'] > budget['single_member_wall_limit_seconds']):
        raise ValueError('A member wall limit never authorizes a longer profile')
    probe = config['equivalence_probe']
    if not isinstance(probe, dict) or set(probe) != set(PROBE_KEYS):
        raise ValueError('Equivalence probe requires exactly: rows, atol')
    if not _positive_int(probe['rows']) or probe['rows'] > FIXED['expected_samples']['cpanel_dev']:
        raise ValueError('Equivalence probe rows must be a positive integer within the Cpanel DEV size')
    if not _finite(probe['atol']) or not 0 < probe['atol'] <= 1e-3:
        raise ValueError('Equivalence probe tolerance must be finite in (0, 1e-3]')
    registration = config['registration']
    if not isinstance(registration, dict) or not isinstance(registration.get('status'), str):
        raise ValueError('Registration must be an object with a status string')
    if not isinstance(registration.get('output'), str) or not registration['output']:
        raise ValueError('Registration must name the registered output path')
    if real and 'DRAFT' in registration['status'].upper():
        raise ValueError('A DRAFT registration status cannot execute a real command')
    return config


def output_guard(output, config):
    """The requested output must be the registered output and disjoint from all three parents.

    Runs before any write, including the internal ledger: an output equal to,
    inside, or containing a sealed parent would otherwise mutate that parent
    before the runner refuses.  Pure path arithmetic; nothing is touched.
    """
    from pathlib import Path
    output = Path(output).resolve()
    registered = Path(config['registration']['output']).resolve()
    if output != registered:
        raise ValueError(f'Output {output} differs from the registered output {registered}')
    for name in ('parent_g', 'parent_c1', 'parent_t4'):
        parent = Path(config[name]['run']).resolve()
        if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
            raise ValueError(f'Output {output} is not disjoint from the sealed {name} run {parent}')
    return output


def validate_native_environment(parent_identity, expected):
    """Parent and adapter must share python, libraries, local config and native paths."""
    skip = {'config_sha256', 'source_hashes'}
    theirs = {k: v for k, v in parent_identity.items() if k not in skip}
    mine = {k: v for k, v in expected.items() if k not in skip}
    if theirs != mine:
        differing = sorted(k for k in set(theirs) | set(mine) if theirs.get(k) != mine.get(k))
        raise ValueError('Native/local environment differs from the frozen parent: ' + ', '.join(differing))
    return True


# ---------------------------------------------------------------- identity

def member_identity(prep, seed):
    if seed not in SEEDS:
        raise ValueError('Unregistered whole-MLB member seed')
    member = prep['members'][str(seed)]
    return {'preparation_sha256': canonical_hash(prep), 'cell': CELL, 'seed': seed,
            'source_arm': SOURCE_ARM[seed], 'source_run': member['source_run'],
            'model_sha256': member['model_sha256'],
            'source_prediction_state_sha256': member['prediction_state_sha256'],
            'delivery_temperature': member['delivery_temperature'], 'draws': FIXED['draws']}


# ---------------------------------------------------------------- frozen calibration

def frozen_weights(c1_report, member_temperatures):
    """C1 five-seed June ensemble/per-seed weights and May temperatures, validated, never refitted."""
    selection = c1_report.get('selection', {})
    seeds = c1_report.get('seeds', [])
    if selection.get('calibration_objective') != 'log_loss' or len(seeds) != len(SEEDS):
        raise ValueError('C1 G0-global report must carry a log-loss June selection and five ordered seed reports')
    ensemble = selection.get('model_weight')
    per_seed = [row.get('blend_selection', {}).get('model_weight') for row in seeds]
    for name, value in (('ensemble', ensemble), *((f'seed{seed}', w) for seed, w in zip(SEEDS, per_seed))):
        if not _finite(value) or not 0 <= value <= 1:
            raise ValueError(f'Frozen June model weight must be finite in [0,1]: {name}')
    if set(member_temperatures) != {str(seed) for seed in SEEDS}:
        raise ValueError('Five ordered May temperatures required')
    for seed, temperature in member_temperatures.items():
        if not _finite(temperature) or not .5 <= temperature <= 2.5:
            raise ValueError(f'Frozen May temperature outside the calibration bounds: seed {seed}')
    return {'june_ensemble_model_weight': float(ensemble),
            'june_seed_model_weights': {str(seed): float(w) for seed, w in zip(SEEDS, per_seed)},
            'may_delivery_temperatures': {str(seed): float(member_temperatures[str(seed)]) for seed in SEEDS},
            'source': 'C1 analysis reports G0-global selection/seeds and each member calibration.json',
            'refit_on_whole_mlb': False}


# ---------------------------------------------------------------- population alignment

def check_eligible_population(baseline, metadata_keys, metadata_game, metadata_pitcher, expected):
    """Ordered eligible whole-MLB keys/labels/metadata must agree exactly with each other."""
    keys, y, games, pitchers = (np.asarray(baseline['dev_' + name]) for name in DEV_FIELDS)
    if keys.ndim != 2 or keys.shape[1] != 3 or not np.issubdtype(keys.dtype, np.integer):
        raise ValueError('Eligible keys must be an integer [n,3] array')
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError('Repeated eligible pitch keys')
    if y.shape != (len(keys),) or games.shape != (len(keys),) or pitchers.shape != (len(keys),):
        raise ValueError('Eligible labels and metadata must align with keys')
    if not np.issubdtype(y.dtype, np.integer) or ((y < 0) | (y >= 10)).any():
        raise ValueError('Eligible labels outside the ten-class contract')
    if not np.array_equal(keys[:, 0], games):
        raise ValueError('Pitch key and game metadata disagree')
    if (not np.array_equal(np.asarray(metadata_keys), keys) or not np.array_equal(np.asarray(metadata_game), games)
            or not np.array_equal(np.asarray(metadata_pitcher), pitchers)):
        raise ValueError('Grouping metadata keys/games/pitchers differ from the eligible population')
    n_games = len(np.unique(games))
    if len(keys) != expected['dev'] or n_games != expected['dev_games']:
        raise ValueError(f'Eligible population {len(keys)}/{n_games} differs from registration '
                         f'{expected["dev"]}/{expected["dev_games"]}')
    return {'n': int(len(keys)), 'games': int(n_games)}


def cpanel_positions(mlb_keys, in_cpanel, cpanel_keys, expected):
    """Ordered positions of the sealed Cpanel DEV rows inside the eligible whole-MLB population."""
    mask = np.asarray(in_cpanel, dtype=bool)
    mlb_keys, cpanel_keys = np.asarray(mlb_keys), np.asarray(cpanel_keys)
    if mask.shape != (len(mlb_keys),):
        raise ValueError('in_cpanel mask must align with the eligible keys')
    positions = np.flatnonzero(mask)
    if len(positions) != len(cpanel_keys) or not np.array_equal(mlb_keys[positions], cpanel_keys):
        raise ValueError('Cpanel DEV rows are not the ordered in_cpanel subset of the eligible whole-MLB population')
    games = len(np.unique(cpanel_keys[:, 0]))
    if len(positions) != expected['cpanel_dev'] or games != expected['cpanel_games']:
        raise ValueError(f'Cpanel overlap {len(positions)}/{games} differs from registration')
    complement = np.flatnonzero(~mask)
    return {'positions': positions, 'overlap_pitches': int(len(positions)), 'overlap_games': int(games),
            'complement_pitches': int(len(complement)),
            'complement_games': int(len(np.unique(mlb_keys[complement, 0]))) if len(complement) else 0}


def compare_predictions(mine, sealed, *, atol, name):
    mine, sealed = np.asarray(mine, dtype=np.float64), np.asarray(sealed, dtype=np.float64)
    if mine.shape != sealed.shape or mine.ndim != 2 or mine.shape[1] != 10:
        raise ValueError(f'{name}: prediction shapes differ from the sealed reference')
    if not np.isfinite(mine).all() or not np.isfinite(sealed).all():
        raise ValueError(f'{name}: nonfinite probabilities')
    difference = float(np.abs(mine - sealed).max()) if len(mine) else 0.
    return {'name': name, 'rows': int(len(mine)), 'maximum_absolute_difference': difference,
            'tolerance': float(atol), 'within_tolerance': bool(difference <= atol)}


def require_within_tolerance(report):
    if report['within_tolerance'] is not True:
        raise ValueError(f'{report["name"]}: whole-MLB path no longer reproduces the sealed Cpanel predictions '
                         f'({report["maximum_absolute_difference"]:.3g} > {report["tolerance"]:.3g})')
    return report


def compare_levels(mine, sealed, *, name):
    """Delivery tiers are discrete routing decisions: exact equality, no tolerance."""
    mine, sealed = np.asarray(mine), np.asarray(sealed)
    if mine.shape != sealed.shape or mine.ndim != 1 or not np.issubdtype(mine.dtype, np.integer) or not np.issubdtype(sealed.dtype, np.integer):
        raise ValueError(f'{name}: delivery tier shapes/dtypes differ from the sealed reference')
    mismatches = int((mine != sealed).sum())
    return {'name': name, 'rows': int(len(mine)), 'mismatches': mismatches, 'equal': mismatches == 0}


def require_equal_levels(report):
    if report['equal'] is not True:
        raise ValueError(f'{report["name"]}: {report["mismatches"]} delivery tiers differ from the sealed Cpanel member')
    return report


def cpanel_subset_alignment(values, sealed, positions, *, atol):
    """Whole-MLB rows at the Cpanel positions versus the sealed member: probabilities within tolerance, tiers exact."""
    positions = np.asarray(positions)
    if not np.array_equal(values['dev_keys'][positions], sealed['dev_keys']) or not np.array_equal(values['dev_y'][positions], sealed['dev_y']):
        raise ValueError('Sealed Cpanel keys/labels differ from the whole-MLB overlap rows')
    return [compare_predictions(values['dev'][positions], sealed['dev'], atol=atol, name='cpanel_subset_calibrated'),
            compare_predictions(values['dev_raw'][positions], sealed['dev_raw'], atol=atol, name='cpanel_subset_raw'),
            compare_levels(values['dev_delivery_level'][positions], sealed['dev_delivery_level'], name='cpanel_subset_delivery_level')]


def require_alignment(reports):
    for report in reports:
        (require_equal_levels if 'equal' in report else require_within_tolerance)(report)
    return reports


def validate_member_archive(values, baseline):
    """Whole-MLB member archive: exact keys/labels pairing and valid probability mass."""
    if set(values) != set(ARCHIVE_FIELDS):
        raise ValueError('Member archive fields differ from the registered contract')
    for name in DEV_FIELDS:
        if not np.array_equal(values['dev_' + name], baseline['dev_' + name]):
            raise ValueError('Whole-MLB member unpaired: ' + name)
    y = np.asarray(baseline['dev_y'])
    for name in ('dev', 'dev_raw'):
        p = np.asarray(values[name], dtype=np.float64)
        if p.shape != (len(y), 10) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError(f'{name}: invalid probability array')
        if not np.allclose(p.sum(1), 1., atol=1e-6, rtol=0):
            raise ValueError(f'{name}: probability mass differs from one')
    levels = np.asarray(values['dev_delivery_level'])
    if levels.shape != (len(y),) or not np.issubdtype(levels.dtype, np.integer) or ((levels < -1) | (levels > 3)).any():
        raise ValueError('Delivery tier levels outside the frozen contract')
    return True


# ---------------------------------------------------------------- cost projection and ledger

def project_costs(measured, *, n_dev, probe_rows, budget, prepare_seconds):
    """Linear per-row extrapolation of one measured member; a planning estimate, never a bound."""
    for name in ('command_overhead_seconds', 'load_seconds', 'model_load_seconds', 'inference_seconds', 'wall_seconds'):
        if not _finite(measured.get(name)) or measured[name] < 0:
            raise ValueError('Finite nonnegative profile measurement required: ' + name)
    rows = measured.get('rows')
    if not _positive_int(rows) or not _positive_int(n_dev) or not _positive_int(probe_rows):
        raise ValueError('Positive profile, population and probe row counts required')
    if not _finite(prepare_seconds) or prepare_seconds < 0:
        raise ValueError('Finite nonnegative prepare wall required')
    per_row = measured['inference_seconds'] / rows
    # Every predict command pays verification, data load and checkpoint load before its first row.
    overhead = measured['command_overhead_seconds'] + measured['load_seconds'] + measured['model_load_seconds']
    member = overhead + per_row * (n_dev + probe_rows)
    family = prepare_seconds + measured['wall_seconds'] + len(SEEDS) * member
    return {'seconds_per_row': per_row, 'member_seconds': member, 'five_member_seconds': len(SEEDS) * member,
            'family_seconds': family,
            'member_gate': member <= budget['single_member_wall_limit_seconds'],
            'batch_gate': family <= budget['batch_wall_budget_seconds'],
            'note': ('Per-row 400-draw inference plus measured verification/load overhead; profile and prepare '
                     'walls are charged to the family; the caller enforces timeouts.')}


def ledger_totals(entries, caps, *, exclude=None):
    """Ended commands charge their wall; a start without an end reserves its stage cap."""
    ended = {e['id']: e for e in entries if e.get('event') == 'end'}
    starts = [e for e in entries if e.get('event') == 'start' and e['id'] != exclude]
    spent = reserved = 0.
    unresolved = []
    for entry in starts:
        if entry['id'] in ended:
            seconds = ended[entry['id']]['seconds']
            if not _finite(seconds) or seconds < 0:
                raise ValueError('Ledger wall must be finite and nonnegative')
            spent += seconds
        else:
            if entry['stage'] not in caps:
                raise ValueError('Unregistered ledger stage: ' + str(entry['stage']))
            reserved += caps[entry['stage']]
            unresolved.append(entry['id'])
    return {'spent_seconds': float(spent), 'reserved_seconds': float(reserved),
            'charged_seconds': float(spent + reserved), 'unresolved': unresolved}


def launch_gate(totals, *, projected, budget_seconds, stage):
    """Refuse a new command when charged wall plus its projection exceeds the frozen batch budget."""
    if totals['unresolved']:
        raise RuntimeError('Unresolved ledger start reserves its cap; review it before launching: '
                           + ', '.join(totals['unresolved']))
    if not _finite(projected) or projected < 0:
        raise RuntimeError('Finite nonnegative projection required')
    remaining = budget_seconds - totals['charged_seconds']
    if projected > remaining:
        raise RuntimeError(f'{stage}: projected {projected:.1f}s exceeds the remaining batch budget {remaining:.1f}s')
    return {'remaining_seconds': float(remaining), 'projected_seconds': float(projected)}
