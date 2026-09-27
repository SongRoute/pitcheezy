"""Pure helpers for the G0/F1 five-seed execution bundle (COOP-003).

Two arms share four new G0 fits: the frozen C1 runner extends ``G0-global``
to seeds 3/4 (arm ``c1``), and an additive F1 extension fits two more masked
G0 models (arm ``f1ext``).  This module holds only decision rules, schema
validation, wall-ledger accounting and pairing checks; nothing here reads
real data, deserializes checkpoints, or touches the frozen G/C1/F1 modules.
Scientific constants are frozen by docs/contracts/ML-G0-F1-CONFIRMATION-DRAFT-v1.md.
"""
from __future__ import annotations

import math

import numpy as np

from .matrix_bridge import BATTER_START, BATTER_STOP, CONTEXT_WIDTH, ROUTING_WIDTH
from .matrix_data import canonical_hash

SEEDS = (0, 1, 2, 3, 4)
REUSED_SEEDS = (0, 1, 2)
NEW_SEEDS = (3, 4)
ARMS = ('c1', 'f1ext')
BUNDLE_PROTOCOL = 'ml_g0_f1_five_seed_bundle_v1'
EXTENSION_PROTOCOL = 'ml_bridge_five_seed_v1'
C1_PROTOCOL = 'ml_confirmation_g_v1'
G_EXPERIMENT, F1_EXPERIMENT = 'EXP-P4-001', 'EXP-P9-001-v2'
PARENT_CELL = 'G0-global'
STAGES = ('prepare', 'profile', 'fit', 'predict', 'freeze', 'score')
# Frozen caps: a 7200 member limit never authorizes a 7200-second profile.
STAGE_CAPS = {'prepare': 7200, 'profile': 600, 'fit': 7200, 'predict': 7200, 'freeze': 7200, 'score': 7200}
MEMBER_LIMIT_SECONDS = 7200
SHARE_SECONDS = 7200
COMBINED_SECONDS = 14400
BUDGET = {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': .0005}
PROFILE = {'train_rows': 65536, 'epochs': 2, 'earlystop_rows': 2048, 'temperature_rows': 64}
EXPECTED_SAMPLES = {'train': 1252824, 'earlystop': 16000, 'temperature': 2603,
                    'blend': 4821, 'dev': 12334, 'dev_games': 328}
BOOTSTRAP = {'draws': 10000, 'seed': 20260924, 'unit': 'whole game',
             'estimand': 'pitch-weighted paired mean loss'}
DECISION = {'delta_nll_max': -.003, 'nll_ci95_upper_max': 0, 'one_sided_p_max': .05,
            'brier_ci95_upper_max': .001, 'required_negative_seeds': 4, 'seeds': 5}
ROBUSTNESS = {'groups': 12, 'metrics': 2, 'comparisons': 1, 'family_size': 24, 'family_alpha': .05,
              'draws': 100000, 'seed': 20260924, 'minimum_games': 30, 'minimum_pitches': 500,
              'nll_margin': .01, 'brier_margin': .002}
BATTER_VOLUME = {'quantile': .25, 'method': 'linear', 'boundary': 'low_inclusive',
                 'scope': 'D100 TRAIN positive batter pitch counts'}
STATUSES = ('development_stability_pass', 'worse_or_guardrail_failure', 'inconclusive')
STAGE_TEXT = 'five-seed development stability, exposed Cpanel'
MULTIPLICITY = ('one primary NLL hypothesis; no within-family adjustment; second-look stability diagnostic '
                'on the same hypothesis and games after a passing three-seed look; no fresh alpha guarantee')
EXTENSION_FIXED = {'protocol': EXTENSION_PROTOCOL, 'scope': 'Cpanel', 'seeds': list(SEEDS),
                   'reused_seeds': list(REUSED_SEEDS), 'new_seeds': list(NEW_SEEDS), 'draws': 400,
                   'kind': 'flatten_mlp', 'width': 128, 'device': 'auto', 'budget': BUDGET,
                   'mask': {'start': BATTER_START, 'stop': BATTER_STOP}, 'context_width': CONTEXT_WIDTH,
                   'routing_width': ROUTING_WIDTH, 'expected_samples': EXPECTED_SAMPLES, 'profile': PROFILE,
                   'bootstrap': BOOTSTRAP, 'decision': DECISION, 'robustness': ROBUSTNESS,
                   'batter_volume': BATTER_VOLUME, 'status_enum': list(STATUSES), 'stage': STAGE_TEXT,
                   'held_out_confirmation': False, 'independent_confirmation': None,
                   'family_wall_budget_seconds': SHARE_SECONDS, 'member_wall_limit_seconds': MEMBER_LIMIT_SECONDS}
G_PINS = ('run', 'preparation_sha256', 'registered_config_sha256', 'analysis_manifest_sha256',
          'analysis_results_sha256', 'baseline_predictions_sha256', 'dev_metadata_sha256')
F1_PINS = ('run', 'preparation_sha256', 'registered_config_sha256', 'analysis_manifest_sha256',
           'analysis_results_sha256', 'batter_train_volume_sha256', 'config_file', 'config_file_sha256')
C1_DECLARED = ('experiment_id', 'run', 'config_file', 'config_sha256', 'source_hashes', 'third_parent_manifest')
HEX = set('0123456789abcdef')


def _sha(value, length=64):
    return isinstance(value, str) and len(value) == length and set(value) <= HEX


def pin(value, length=64, *, real, name='pin'):
    """Draft configs may carry null pins; a real command refuses them."""
    if value is None:
        if real:
            raise ValueError(f'Null identity pin is not allowed in a real command: {name}; freeze the registration first')
        return
    if not _sha(value, length):
        raise ValueError(f'Identity pin must be a lowercase hex digest: {name}')


def _seconds(value, upper=None):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and value > 0 and (upper is None or value <= upper))


# ---------------------------------------------------------------- decision rule

def seed_deltas(labels_losses, full_seed_primary, masked_seed_primary, pitch_losses):
    """Per-seed pitch-weighted NLL difference (full - masked) for five ordered paired seeds."""
    if len(full_seed_primary) != len(SEEDS) or len(masked_seed_primary) != len(SEEDS):
        raise ValueError('Five seed-paired primary predictions required')
    return [float((pitch_losses(labels_losses, a) - pitch_losses(labels_losses, b))[:, 0].mean())
            for a, b in zip(full_seed_primary, masked_seed_primary)]


def five_seed_decision(comparison, deltas, rule=DECISION):
    """Frozen five-seed rule: development stability, never promotion or confirmation."""
    if len(deltas) != rule['seeds'] or not np.isfinite(np.asarray(deltas, float)).all():
        raise ValueError('Five finite paired seed differences required')
    if comparison.get('status') != 'measured' or comparison['nll']['p_less'] is None:
        return {'status': 'unmeasured', 'reason': 'paired inference unavailable', 'stage': STAGE_TEXT}
    nll, brier = comparison['nll'], comparison['brier']
    criteria = {'practical_improvement': nll['delta'] <= rule['delta_nll_max'],
                'paired_ci_below_zero': nll['ci95'][1] < rule['nll_ci95_upper_max'],
                'one_sided_p': nll['p_less'] <= rule['one_sided_p_max'],
                'brier_noninferior': brier['ci95'][1] <= rule['brier_ci95_upper_max'],
                'seed_direction_stable': sum(x < 0 for x in deltas) >= rule['required_negative_seeds']}
    if all(criteria.values()):
        status = STATUSES[0]
    elif nll['ci95'][0] > 0 or brier['ci95'][0] > rule['brier_ci95_upper_max']:
        status = STATUSES[1]
    else:
        status = STATUSES[2]
    return {'status': status, 'criteria': criteria, 'p_one_sided': nll['p_less'],
            'negative_seeds': int(sum(x < 0 for x in deltas)), 'required_negative_seeds': rule['required_negative_seeds'],
            'multiplicity': MULTIPLICITY, 'seed_deltas': [float(x) for x in deltas],
            'reverse_point_estimate': bool(nll['delta'] > 0),
            'reverse_note': ('Masked point estimate is better; no automatic promotion of the masked arm.'
                             if nll['delta'] > 0 else None),
            'stage': STAGE_TEXT, 'held_out_confirmation': False, 'independent_confirmation': None,
            'policy_effect': None}


# ---------------------------------------------------------------- configuration schemas

def validate_extension_config(config, *, real=True):
    required = {*EXTENSION_FIXED, 'experiment_id', 'parent_g', 'parent_f1', 'parent_c1', 'registration'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Invalid F1 five-seed extension configuration schema')
    for name, value in EXTENSION_FIXED.items():
        if config[name] != value:
            raise ValueError('F1 extension setting differs from the frozen protocol: ' + name)
    if not isinstance(config['experiment_id'], str) or not config['experiment_id'].strip():
        raise ValueError('Registered F1 extension experiment identity required')
    g, f1, c1 = config['parent_g'], config['parent_f1'], config['parent_c1']
    if set(g) != {'experiment_id', *G_PINS} or g['experiment_id'] != G_EXPERIMENT or not g['run']:
        raise ValueError('Frozen G parent pins required: ' + ', '.join(G_PINS))
    if set(f1) != {'experiment_id', *F1_PINS} or f1['experiment_id'] != F1_EXPERIMENT or not f1['run'] or not f1['config_file']:
        raise ValueError('Frozen F1 parent pins required: ' + ', '.join(F1_PINS))
    for record, names in ((g, G_PINS), (f1, F1_PINS)):
        for name in names:
            if name.endswith('sha256'):
                pin(record[name], real=real, name=record['experiment_id'] + '.' + name)
    if set(c1) != set(C1_DECLARED) or not c1['experiment_id'] or not c1['run'] or not c1['third_parent_manifest']:
        raise ValueError('Declared C1 third parent requires: ' + ', '.join(C1_DECLARED))
    if c1['run'] in (g['run'], f1['run']):
        raise ValueError('C1 third parent must be a distinct run')
    pin(c1['config_sha256'], real=real, name='parent_c1.config_sha256')
    if c1['source_hashes'] is None:
        if real:
            raise ValueError('Declared C1 source hashes must be frozen before a real command')
    elif not isinstance(c1['source_hashes'], dict) or not c1['source_hashes']:
        raise ValueError('Declared C1 source hashes must be a nonempty mapping')
    else:
        for name, digest in c1['source_hashes'].items():
            pin(digest, real=real, name='parent_c1.source_hashes.' + name)
    if not isinstance(config['registration'], dict):
        raise ValueError('Registration must be an object')
    return config


def validate_c1_config_shape(config):
    """Shape checks the bundle needs from the C1 config; frozen validate_config runs in the C1 runner."""
    if (config.get('protocol') != C1_PROTOCOL or config.get('cells') != [PARENT_CELL]
            or config.get('primary_comparisons') != [] or config.get('selection_status') != 'baseline_stability_only'
            or config.get('seeds') != list(SEEDS) or config.get('device') != 'auto'):
        raise ValueError('C1 arm must be the G0-global baseline_stability_only five-seed registration')
    registration = config.get('registration', {})
    if registration.get('batch_wall_budget_seconds') != SHARE_SECONDS:
        raise ValueError('C1 batch_wall_budget_seconds must equal the frozen 7200-second C1 share')
    if registration.get('single_member_wall_limit_seconds') != MEMBER_LIMIT_SECONDS:
        raise ValueError('C1 member limit must be 7200 seconds')
    return config


def validate_bundle(bundle, *, real=True):
    required = {'protocol', 'bundle_id', 'repo_commit', 'contracts', 'configs', 'sources', 'arms',
                'combined_seconds', 'ledger_dir', 'caps', 'termination_grace_seconds', 'sequence'}
    if not isinstance(bundle, dict) or set(bundle) != required:
        raise ValueError('Invalid five-seed bundle schema')
    if bundle['protocol'] != BUNDLE_PROTOCOL or not isinstance(bundle['bundle_id'], str) or not bundle['bundle_id']:
        raise ValueError('Unregistered bundle protocol/identity')
    pin(bundle['repo_commit'], 40, real=real, name='repo_commit')
    for group in ('contracts', 'configs'):
        records = bundle[group]
        if not isinstance(records, dict) or not records:
            raise ValueError(f'Bundle {group} pins required')
        for name, record in records.items():
            if set(record) != {'path', 'sha256'} or not record['path']:
                raise ValueError(f'{group}.{name} needs path and sha256')
            pin(record['sha256'], real=real, name=f'{group}.{name}')
    if set(bundle['contracts']) != {'runner', 'draft', 'c1_runner', 'bridge', 'bridge_runner'}:
        raise ValueError('Bundle must pin the runner, draft, C1, bridge and bridge-runner contracts')
    if set(bundle['configs']) != {'c1', 'f1ext', 'g_parent', 'f1_parent'}:
        raise ValueError('Bundle must pin both new configs and both parent configs')
    if not isinstance(bundle['sources'], dict) or not bundle['sources']:
        raise ValueError('Bundle source pins required')
    for name, digest in bundle['sources'].items():
        pin(digest, real=real, name='sources.' + name)
    arms = bundle['arms']
    if set(arms) != set(ARMS):
        raise ValueError('Bundle arms must be exactly c1 and f1ext')
    for arm, record in arms.items():
        if set(record) != {'experiment_id', 'output', 'share_seconds'} or not record['experiment_id'] or not record['output']:
            raise ValueError('Arm needs experiment_id, output and share_seconds: ' + arm)
        if record['share_seconds'] != SHARE_SECONDS:
            raise ValueError('Each arm share is frozen at 7200 seconds; no transfer between arms')
    if arms['c1']['output'] == arms['f1ext']['output']:
        raise ValueError('Arm outputs must be distinct fresh siblings')
    if bundle['combined_seconds'] != COMBINED_SECONDS:
        raise ValueError('Combined ceiling is frozen at 14400 seconds')
    if not isinstance(bundle['ledger_dir'], str) or not bundle['ledger_dir']:
        raise ValueError('Authoritative external ledger directory required')
    caps = bundle['caps']
    if set(caps) != set(STAGES) or any(caps[s] != STAGE_CAPS[s] for s in STAGES):
        raise ValueError('Stage caps are frozen: profile 600, other commands 7200')
    grace = bundle['termination_grace_seconds']
    if not _seconds(grace) or grace >= min(caps.values()):
        raise ValueError('Termination grace must be positive and inside every cap')
    if bundle['sequence'] != [list(step) for step in SEQUENCE]:
        raise ValueError('Frozen execution sequence differs')
    return bundle


# ---------------------------------------------------------------- execution sequence

# (command, arm, stage, seed). Preconditions are enforced by the supervisor.
SEQUENCE = (
    ('c1-prepare', 'c1', 'prepare', None),
    ('c1-profile', 'c1', 'profile', None),
    ('profile-full', 'c1', 'profile', None),
    ('profile-masked', 'f1ext', 'profile', None),
    ('c1-fit', 'c1', 'fit', 3), ('c1-fit', 'c1', 'fit', 4),
    ('c1-predict', 'c1', 'predict', 3), ('c1-predict', 'c1', 'predict', 4),
    ('freeze-third-parent', 'f1ext', 'freeze', None),
    ('f1-prepare', 'f1ext', 'prepare', None),
    ('f1-fit', 'f1ext', 'fit', 3), ('f1-fit', 'f1ext', 'fit', 4),
    ('f1-predict', 'f1ext', 'predict', 3), ('f1-predict', 'f1ext', 'predict', 4),
    ('c1-score', 'c1', 'score', None),
    ('f1-score', 'f1ext', 'score', None),
)
COMMANDS = tuple(dict.fromkeys(step[0] for step in SEQUENCE))


def sequence_step(command, seed=None):
    for step in SEQUENCE:
        if step[0] == command and step[3] == seed:
            return step
    raise ValueError(f'Unregistered bundle step {command} seed={seed}')


# ---------------------------------------------------------------- wall ledger accounting

def ledger_totals(jobs, arm):
    """Ended jobs charge their wall; unresolved starts reserve their full cap."""
    mine = [j for j in jobs if j['arm'] == arm]
    ended = math.fsum(j['elapsed_seconds'] for j in mine if j['ended'])
    reserved = math.fsum(j['cap_seconds'] for j in mine if not j['ended'])
    return {'ended_seconds': ended, 'reserved_seconds': reserved, 'charged_seconds': ended + reserved,
            'unresolved_jobs': [j['job_id'] for j in mine if not j['ended']],
            'jobs': len(mine)}


def member_fit_seconds(jobs, arm, seed):
    """Every fit attempt of a member counts toward its 7200-second fit+predict limit."""
    return math.fsum(j['elapsed_seconds'] if j['ended'] else j['cap_seconds']
                     for j in jobs if j['arm'] == arm and j['stage'] == 'fit' and j['seed'] == seed)


def effective_cap(stage, arm_remaining, *, member_remaining=None, grace):
    """Stage cap limited by the arm's remaining share (and the member's remaining limit)."""
    cap = min(STAGE_CAPS[stage], arm_remaining)
    if member_remaining is not None:
        cap = min(cap, member_remaining)
    if stage == 'profile' and cap < STAGE_CAPS['profile']:
        raise RuntimeError(f'Profile requires its full {STAGE_CAPS["profile"]}-second cap inside the remaining share; remaining {arm_remaining:.1f}s')
    if cap <= grace:
        raise RuntimeError(f'Remaining budget {cap:.1f}s for {stage} does not exceed the termination grace {grace}s; fail closed')
    return float(cap)


def launch_gate(totals, share, stage, *, projected_command, remaining_obligation):
    """Refuse when spent + reserved + the remaining projected obligation would exceed the arm share.

    ``remaining_obligation`` already includes ``projected_command`` (this
    command's projection) and every not-yet-completed stage of the arm.
    """
    if totals['unresolved_jobs']:
        raise RuntimeError('Unresolved job reserves its cap; resolve it before launching: ' + ', '.join(totals['unresolved_jobs']))
    if not isinstance(projected_command, (int, float)) or not math.isfinite(projected_command) or projected_command < 0:
        raise RuntimeError('Finite nonnegative command projection required')
    if remaining_obligation < projected_command:
        raise RuntimeError('Remaining obligation must include this command')
    remaining = share - totals['charged_seconds']
    if projected_command > remaining:
        raise RuntimeError(f'{stage}: projected {projected_command:.1f}s exceeds remaining arm share {remaining:.1f}s')
    if remaining_obligation > remaining:
        raise RuntimeError(f'{stage}: remaining projected obligation {remaining_obligation:.1f}s exceeds remaining arm share {remaining:.1f}s; fail closed')
    return {'remaining_share_seconds': remaining, 'projected_command_seconds': projected_command,
            'remaining_obligation_seconds': remaining_obligation}


def arm_obligation(projection, completed_stages, arm):
    """Projected seconds still owed by an arm, from its matched profile projection.

    ``projection`` keys: prepare_seconds, fit_command_seconds,
    predict_command_seconds, score_seconds. ``completed_stages`` is a set of
    (stage, seed) already completed (ended successfully) for the arm.
    """
    owed = []
    if arm == 'f1ext' and ('prepare', None) not in completed_stages:
        owed.append(projection['prepare_seconds'])
    if arm == 'f1ext' and ('freeze', None) not in completed_stages:
        owed.append(projection['prepare_seconds'])  # verification-only; bounded by a prepare-sized load/verify
    for seed in NEW_SEEDS:
        if ('fit', seed) not in completed_stages:
            owed.append(projection['fit_command_seconds'])
        if ('predict', seed) not in completed_stages:
            owed.append(projection['predict_command_seconds'])
    if ('score', None) not in completed_stages:
        owed.append(projection['score_seconds'])
    return float(math.fsum(owed))


def project_costs(measured, samples, *, arm):
    """Linear projection from a matched 65,536-row profile; a planning estimate, never a bound.

    Covers verify/load overhead, 30-epoch fit on TRAIN+early, full May
    temperature, June+DEV 400-draw predictions and a shape-matched scoring probe.
    """
    rows = PROFILE['train_rows'] + PROFILE['earlystop_rows']
    per_epoch = measured['fit_seconds'] / PROFILE['epochs']
    fit = per_epoch * (samples['train'] + samples['earlystop']) / rows * BUDGET['epochs']
    calibrate = measured['calibration_seconds'] / PROFILE['temperature_rows'] * samples['temperature']
    per_row = measured['inference_seconds'] / PROFILE['temperature_rows']
    probe_rows = 64 if arm == 'f1ext' else 0  # masked predict re-runs the 64-row full-path probe
    predict = per_row * (samples['blend'] + samples['dev'] + probe_rows)
    overhead = measured['command_overhead_seconds'] + measured['load_seconds']
    fit_command = overhead + fit
    predict_command = overhead + calibrate + predict
    prepare_seconds = overhead + measured['load_seconds']  # prepare loads every part once more
    score_seconds = measured['command_overhead_seconds'] + measured['scoring_probe_seconds']
    member = fit_command + predict_command
    arm_total = arm_obligation({'prepare_seconds': prepare_seconds, 'fit_command_seconds': fit_command,
                                'predict_command_seconds': predict_command, 'score_seconds': score_seconds}, set(), arm)
    return {'prepare_seconds': prepare_seconds, 'fit_command_seconds': fit_command,
            'predict_command_seconds': predict_command, 'member_seconds': member,
            'score_seconds': score_seconds, 'two_member_seconds': 2 * member,
            'arm_total_seconds': arm_total, 'member_gate': member <= MEMBER_LIMIT_SECONDS,
            'arm_gate': arm_total <= SHARE_SECONDS,
            'note': 'Linear 30-epoch/all-row extrapolation plus measured overhead; the supervisor ledger enforces walls.'}


# ---------------------------------------------------------------- pairing and identity checks

def check_same_seed_pair(full_fit, masked_fit, *, seed):
    """Full (C1/G) and masked fit reports must share seed, device, network and ordered row hashes."""
    a, b = full_fit['report'], masked_fit['report']
    if a.get('seed') != seed or b.get('seed') != seed:
        raise ValueError(f'Paired fit seed differs from {seed}')
    if a.get('device') != b.get('device'):
        raise ValueError(f'Paired seed {seed} devices differ: {a.get("device")} vs {b.get("device")}')
    if a.get('network') != b.get('network') or a.get('parameter_count') != b.get('parameter_count'):
        raise ValueError(f'Paired seed {seed} network signature differs')
    for name in ('train_rows_sha256', 'earlystop_rows_sha256'):
        if full_fit.get(name) != masked_fit.get(name) or not _sha(full_fit.get(name)):
            raise ValueError(f'Paired seed {seed} ordered {name} differs')
    return {'seed': seed, 'device': a['device'], 'network': a['network'], 'parameter_count': a['parameter_count'],
            'train_rows_sha256': full_fit['train_rows_sha256'], 'earlystop_rows_sha256': full_fit['earlystop_rows_sha256']}


def check_signature_matches_report(signature, report):
    """A model.pt-derived network_signature must agree with the stored fit report."""
    if dict(signature['network']) != dict(report['network']) or signature['parameter_count'] != report['parameter_count']:
        raise ValueError('Checkpoint network signature differs from its fit report')
    return True


def check_full_arrays_equal(mine, c1, *, cell=PARENT_CELL):
    """F1-extension full-arm five-seed arrays must be byte-equal to the C1 analysis arrays."""
    for kind in ('primary', 'calibrated', 'raw', 'seed_primary'):
        theirs = c1.get(f'{cell}_{kind}')
        ours = mine[kind]
        if theirs is None or theirs.shape != ours.shape or theirs.dtype != ours.dtype or not np.array_equal(theirs, ours):
            raise ValueError('Full-arm five-seed array differs from the C1 analysis: ' + kind)
        if ours.tobytes() != theirs.tobytes():
            raise ValueError('Full-arm five-seed array bytes differ from the C1 analysis: ' + kind)
    return True


def third_parent_identity(manifest, declared):
    """The materialized C1 manifest must match the declared third parent."""
    for name in ('experiment_id', 'run'):
        if manifest.get('c1_' + name) != declared[name]:
            raise ValueError('Third-parent manifest differs from the declared C1 parent: ' + name)
    if manifest.get('c1_config_sha256') != declared['config_sha256']:
        raise ValueError('Third-parent C1 config identity differs from the declaration')
    if declared['source_hashes'] is not None and manifest.get('c1_source_hashes') != declared['source_hashes']:
        raise ValueError('Third-parent C1 source hashes differ from the declaration')
    required = {'c1_experiment_id', 'c1_run', 'c1_config_sha256', 'c1_source_hashes', 'c1_preparation_sha256',
                'c1_identity', 'fits', 'members', 'frozen_utc'}
    if not required <= set(manifest):
        raise ValueError('Third-parent manifest incomplete')
    for seed in NEW_SEEDS:
        fit = manifest['fits'].get(str(seed), {})
        member = manifest['members'].get(str(seed), {})
        if set(fit) != {'state.json', 'model.pt', 'fit.json'} or set(member) != {'prediction_state.json', 'predictions.npz',
                                                                                 'calibration.json', 'prediction_runtime.json'}:
            raise ValueError(f'Third-parent manifest lacks the seed {seed} fit/prediction hash family')
        for digest in (*fit.values(), *member.values()):
            pin(digest, real=True, name='third_parent')
    pin(manifest['c1_preparation_sha256'], real=True, name='c1_preparation_sha256')
    return True


def extension_member_identity(prep, seed, *, c1_fit_state_sha256):
    if seed not in NEW_SEEDS:
        raise ValueError('Only seeds 3/4 are new masked members')
    return {'preparation_sha256': canonical_hash(prep), 'arm': 'masked', 'seed': seed,
            'mask': [BATTER_START, BATTER_STOP], 'train_rows_sha256': prep['samples']['train']['rows_sha256'],
            'c1_fit_state_sha256': c1_fit_state_sha256}
