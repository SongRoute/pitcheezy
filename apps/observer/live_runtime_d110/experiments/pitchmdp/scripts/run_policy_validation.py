"""<=2025 policy validation runner (ML-POLICY-VAL-v1, COOP-018, D93).

Stages: census (S0), materialize-bc (S1), style-snapshot (S1b), bind-probe (S2), profile (S3),
tau-select (S3b), v5-denominators (S4), v2-world (S5), dr-evaluate (S6/V4). The CLI refuses unless
configs/ML-POLICY-MATERIALIZATION-v1.json is registered and execution-enabled, every decision has
its registered value, the source commit is HEAD on a clean tree, the local config matches its
pin, the stage's review gate is passed (M-3) and every field the stage reads is registered. Stage
outputs of earlier stages reach later ones only through an append-only addendum chain of SHA pins
(M-2). Every stage holds the shared heavy lock, loads its data inside a fresh stage directory
under the registered root, records its start/identity, seals a manifest with its measured cost
on success and preserves a failure record otherwise (no silent retry, no scope shrinking). D-11
(measure first): label-blind stages carry only a hang guard; candidate rollouts carry registered
RowBudgets. No row dated 2026 or later can enter. Stage logic is in pure functions over
already-loaded inputs so it can be checked without real data.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import resource
import signal
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
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_policy import context_key, safe_rows
from pitchmdp.matrix_policy_artifacts import is_appledouble
from pitchmdp.policy_artifacts import (IntegrityError, _require, export_train_bc, load_hand_registry,
                                       load_style_snapshot, load_support_table, load_train_bc, normalize_hand,
                                       save_hand_registry, save_style_snapshot, save_support_table)
from pitchmdp import policy_estimator as est
from pitchmdp import policy_identity as pid
from pitchmdp import policy_requests as preq
from pitchmdp import policy_runtime as prt
from pitchmdp import policy_semisynthetic as pss
from pitchmdp import policy_tau as ptau
from pitchmdp.rollout_policy import BudgetExceeded, PAState, PastPitch, RowBudget

PROTOCOL = 'ml_policy_materialization_v1'
STAGES = {'census': 'S0_census', 'materialize-bc': 'S1_materialize', 'style-snapshot': 'S1b_style_snapshot',
          'bind-probe': 'S2_bind_probe', 'profile': 'S3_profile', 'tau-select': 'S3b_tau_select',
          'v5-denominators': 'S4_V5_denominators', 'v2-world': 'S5_V2_V3', 'dr-evaluate': 'S6_V4'}
COMMANDS = tuple(STAGES)
EARLY = ('census', 'materialize-bc', 'style-snapshot', 'bind-probe')  # M-3: open after the code review gate
LAST_DATE = pd.Timestamp('2025-12-31')
DECISION_VALUES = {  # D93: the user's decisions (2026-09-29); nothing else can run
    'D-1': 'BC_P', 'D-2': 'all_pa_bounds_plus_predecision_start_population', 'D-3': 'split_cause_refusal_sticky',
    'D-4': 'continue_primary_natural_course_secondary', 'D-5': 'e0_censored_affine_bounds',
    'D-6': 'structural-end-v1', 'D-7': 'window_start_snapshot_pinned', 'D-8': 'train_single_hand_registry',
    'D-9': 'two_stage_profile_then_overlap_noise', 'D-10': 'absorb_off_mask', 'D-11': 'measure_first_hang_guard'}
ADDENDUM_KEYS = {'parent_sha256', 'stage', 'registered_inputs', 'expected_identity_sha256', 'note'}
SEED_ROLES = ('evaluation', 'bootstrap', 'v2', 'selection_salt', 'profile_salt')
P, S = 'le2025_validation_plan', 'le2025_validation_plan.stages'
REQUIRED = {  # config fields a stage reads; null = unregistered = refuse before any data is touched
    '*': (f'{P}.local_config.sha256', f'{P}.output_root', 'pa_time_rules.R3_codes.no_pitch_descriptions', 'seeds.base'),
    'census': (f'{S}.S0_census.hang_guard_seconds', f'{S}.S0_census.outcome_adjacent_splits',
               'pa_time_rules.R7_mid_pa_change.thresholds_before_S0.switch_to_secondary_primary_if_unknown_change_share_above',
               'pa_time_rules.R7_mid_pa_change.thresholds_before_S0.light_version_if_below'),
    'materialize-bc': (f'{S}.S1_materialize.hang_guard_seconds', 'train_bc_plan.bc_p_only_pitcher_rule',
                       f'{S}.S1_materialize.gates.bc_e_rows',
                       f'{S}.S1_materialize.gates.bc_e_actions', f'{S}.S1_materialize.gates.token_vocabulary',
                       f'{S}.S1_materialize.gates.train_rows', 'train_bc_plan.bc_parameters.prior_strength',
                       'train_bc_plan.bc_parameters.minimum_action_count', 'subgroups.volume_quantiles'),
    'style-snapshot': (f'{S}.S1b_style_snapshot.hang_guard_seconds', f'{S}.S1b_style_snapshot.as_of_exclusive.dev',
                       f'{S}.S1b_style_snapshot.as_of_exclusive.temperature', f'{S}.S1b_style_snapshot.as_of_exclusive.blend'),
    'bind-probe': (f'{S}.S2_bind_probe.hang_guard_seconds', f'{S}.S2_bind_probe.rows', f'{S}.S2_bind_probe.atol_primary',
                   f'{S}.S2_bind_probe.atol_frequency_raw'),
    'profile': (f'{S}.S3_profile.starts', f'{S}.S3_profile.candidates', f'{S}.S3_profile.row_budget',
                f'{S}.S3_profile.selection_row_budget.rows', f'{S}.S3_profile.selection_row_budget.decisions',
                f'{S}.S3b_tau_select.tau_grid', f'{S}.S3b_tau_select.samples_minimum_for_noise_rule',
                f'{S}.S6_V4.dr_q_source', 'seeds.planning_main'),
    'tau-select': (f'{S}.S3b_tau_select.n_games', f'{S}.S3b_tau_select.tau_grid', f'{S}.S3b_tau_select.row_budget',
                   f'{S}.S3b_tau_select.thresholds.pa_ess_ratio_min', f'{S}.S3b_tau_select.thresholds.game_ess_min',
                   f'{S}.S3b_tau_select.thresholds.ess_ratio_candidate_reference_min',
                   f'{S}.S3b_tau_select.thresholds.noise_ratio_q90_max', f'{S}.S3b_tau_select.thresholds.safety_multiplier',
                   f'{S}.S3b_tau_select.determinism_check_pas', f'{S}.S3b_tau_select.samples_minimum_for_noise_rule',
                   f'{S}.S2_bind_probe.atol_primary', f'{S}.S6_V4.dr_q_source', f'{S}.S6_V4.row_budget',
                   f'{S}.S6_V4.planned_decisions', 'ess_gate.thresholds.game', 'seeds.planning_main'),
    'v5-denominators': (f'{S}.S4_V5_denominators.hang_guard_seconds',),
    'v2-world': (f'{S}.S5_V2_V3.n_games', f'{S}.S5_V2_V3.logs_per_start', f'{S}.S5_V2_V3.truth_rollouts',
                 f'{S}.S5_V2_V3.cap', f'{S}.S5_V2_V3.row_budget_per_run', f'{S}.S5_V2_V3.tempered_alpha_grid',
                 f'{S}.S5_V2_V3.tolerance', f'{S}.S6_V4.dr_q_source', f'{P}.bootstrap.draws', 'seeds.planning_v2'),
    'dr-evaluate': (f'{S}.S6_V4.n_games', f'{S}.S6_V4.row_budget', f'{S}.S6_V4.dr_q_source',
                    f'{S}.S6_V4.d7_diagnostic_starts', f'{P}.bootstrap.draws',
                    f'{P}.bootstrap.invalid_share_max', f'{P}.bootstrap.minimum.games', f'{P}.bootstrap.minimum.pa_starts',
                    'ess_gate.thresholds.pa', 'ess_gate.thresholds.game', 'sensitivity.same_ledger', 'seeds.planning_main'),
}
PREREQUISITES = {  # registered inputs (sealed stage outputs) a stage needs, checked before any data load
    'census': (), 'materialize-bc': ('census',), 'style-snapshot': ('census',),
    'bind-probe': ('census', 'bc', 'materialize'),
    'v5-denominators': ('census', 'bc', 'support', 'hands', 'materialize', 'bind_probe'),
    'profile': ('census', 'bc', 'support', 'hands', 'materialize', 'bind_probe', 'bind_identity', 'style_temperature'),
    'tau-select': ('census', 'bc', 'support', 'hands', 'materialize', 'bind_probe', 'bind_identity', 'style_blend',
                   'profile'),
    'v2-world': ('census', 'bc', 'support', 'hands', 'materialize', 'bind_probe', 'bind_identity', 'style_dev',
                 'profile', 'tau_freeze', 'v5'),
    'dr-evaluate': ('census', 'bc', 'support', 'hands', 'materialize', 'bind_probe', 'bind_identity', 'style_dev',
                    'profile', 'tau_freeze', 'v5', 'v2'),
}
INPUT_COMMAND = {  # the stage whose sealed manifest must list each registered input
    'census': 'census', 'bc': 'materialize-bc', 'support': 'materialize-bc', 'hands': 'materialize-bc',
    'materialize': 'materialize-bc', 'style_dev': 'style-snapshot', 'style_temperature': 'style-snapshot',
    'style_blend': 'style-snapshot', 'bind_probe': 'bind-probe', 'bind_identity': 'bind-probe', 'profile': 'profile',
    'tau_freeze': 'tau-select', 'v5': 'v5-denominators', 'v2': 'v2-world'}
DR_Q_RULE = 'evaluation_seed_with_cost_fallback'  # M-7: fixed rule, applied mechanically from the S3 cost record
SENSITIVITIES = ('flags_to_bounds', 'r5-events-v1', 'post_pitch_scores')
CODE_PATHS = ('experiments', 'src', 'scripts')  # a registered run needs these unchanged since the source commit
SOURCES = ('experiments/pitchmdp/scripts/run_policy_validation.py', 'experiments/pitchmdp/pitchmdp/policy_requests.py',
           'experiments/pitchmdp/pitchmdp/policy_estimator.py', 'experiments/pitchmdp/pitchmdp/policy_runtime.py',
           'experiments/pitchmdp/pitchmdp/policy_semisynthetic.py', 'experiments/pitchmdp/pitchmdp/policy_tau.py',
           'experiments/pitchmdp/pitchmdp/policy_identity.py', 'experiments/pitchmdp/pitchmdp/policy_artifacts.py')


class HangGuardExceeded(BaseException):
    """D-11 safety timeout: the stage is failed and preserved; never a budget or a truncation. A
    BaseException, so no ``except Exception`` in the runtime turns it into a FAILED_RUNTIME decision
    or a malformed-request row: the request in flight is simply not recorded and the ledger gets
    the abort row."""


# ---------------------------------------------------------------- registration

def _field(block, dotted):
    for part in dotted.split('.'):
        block = block.get(part) if isinstance(block, dict) else None
    return block


def registration(config, command=None):
    """Refuse unless the registration is final, execution-enabled, decided and (for ``command``)
    complete for that stage and past its review gate. Runs before any data is touched."""
    _require(config.get('protocol') == PROTOCOL, 'not an ML-POLICY materialization/validation config')
    _require(config.get('registered') is True and config.get('status') == 'REGISTERED'
             and (config.get('execution') or {}).get('enabled') is True
             and (config.get('execution') or {}).get('real_data_enabled') is True,
             'ML-POLICY-VAL-v1 is not registered and execution-enabled; refusing to run')
    decisions = config.get('decisions') or {}
    for key, value in DECISION_VALUES.items():
        _require(decisions.get(key) == value, f'registered decision required: {key} = {value}')
    if command is None:
        return decisions
    _require(command in COMMANDS, f'unknown command {command!r}')
    gates = config.get('review_gates') or {}
    _require((gates.get('code_review') or {}).get('status') == 'PASS', 'code review gate not passed (M-3)')
    if command not in EARLY:
        _require((gates.get('independent_review') or {}).get('status') == 'PASS',
                 'independent review gate not passed: S3 and later stay closed (M-3)')
    for dotted in REQUIRED['*'] + REQUIRED[command]:
        _require(_field(config, dotted) is not None, f'registered field required: {dotted}')
    stages = config['le2025_validation_plan']['stages']
    plan = config['le2025_validation_plan']
    positive = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    if command in ('profile', 'tau-select', 'v2-world', 'dr-evaluate'):
        _require(stages['S6_V4']['dr_q_source'] == DR_Q_RULE, f'S6 dr_q_source must be the registered M-7 rule {DR_Q_RULE}')
    if command in ('v2-world', 'dr-evaluate'):
        _require(type(plan['bootstrap']['draws']) is int and plan['bootstrap']['draws'] >= 1, 'bootstrap.draws: positive int')
    if command == 'dr-evaluate':
        _require(all(type(plan['bootstrap']['minimum'][k]) is int and plan['bootstrap']['minimum'][k] >= 1
                     for k in ('games', 'pa_starts')), 'bootstrap.minimum: positive ints')
        _require(all(positive(config['ess_gate']['thresholds'][k]) for k in ('pa', 'game')), 'ESS gate thresholds > 0')
        _require(set(config['sensitivity']['same_ledger']) <= set(SENSITIVITIES), 'unregistered same-ledger sensitivity')
        _require(type(stages['S6_V4']['d7_diagnostic_starts']) is int and stages['S6_V4']['d7_diagnostic_starts'] >= 1,
                 'S6 d7_diagnostic_starts: positive int')
    if command == 'tau-select':
        _require(positive(stages['S6_V4']['planned_decisions']) and positive(stages['S6_V4']['row_budget']),
                 'M-7 cost rule needs S6 planned_decisions and row_budget')
    if command == 'profile':  # the D-9a selection must fit the budget S3b actually runs under
        _require(stages['S3_profile']['selection_row_budget']['rows'] == stages['S3b_tau_select'].get('row_budget'),
                 'S3 selection row budget must equal the S3b row budget')
    if command == 'tau-select':  # D-9 critic 5: tau is chosen against the same game gate S6/2026 use
        _require(stages['S3b_tau_select']['thresholds']['game_ess_min'] == config['ess_gate']['thresholds']['game'],
                 'S3b game ESS threshold must equal the registered ESS gate')
    if command == 'v2-world':
        s5 = stages['S5_V2_V3']
        _require(s5['logs_per_start'] >= 2 and s5['truth_rollouts'] >= 2, 'V2 needs at least two logs and two truth rollouts per start')
        hazard = s5.get('declared_hazard')
        _require(hazard is None or (isinstance(hazard, dict) and all(0 <= float(v) <= 1 for v in hazard.values())
                                    and type(s5.get('hazard_replicates')) is int and s5['hazard_replicates'] >= 1),
                 'a declared hazard needs {action: probability} and a positive replicate count')
    if command == 'dr-evaluate':
        share = config['le2025_validation_plan']['bootstrap']['invalid_share_max']
        _require(isinstance(share, (int, float)) and not isinstance(share, bool) and 0 <= share <= 1,
                 'bootstrap.invalid_share_max must be a registered share in [0, 1]')
    return decisions


def load_registration(config_path, addendum_paths=()):
    """Config + append-only addenda (M-2): each addendum names its parent's file sha256 and may only
    add registered-input pins (and at most once the expected identity); nothing is overwritten."""
    raw = Path(config_path).read_bytes()
    config, chain = json.loads(raw), [{'path': str(config_path), 'sha256': hash_file_bytes(raw)}]
    inputs, expected = dict(config.get('registered_inputs') or {}), config['identity_registration'].get(
        'expected_identity_sha256')
    for path in addendum_paths:
        data = Path(path).read_bytes()
        addendum = json.loads(data)
        _require(isinstance(addendum, dict) and set(addendum) <= ADDENDUM_KEYS and addendum.get('parent_sha256')
                 == chain[-1]['sha256'], f'addendum is not the next link of the chain: {path}')
        for name, entry in (addendum.get('registered_inputs') or {}).items():
            _require(name not in inputs and isinstance(entry, dict) and set(entry) == {'path', 'file_sha256'},
                     f'addendum may only add new registered inputs: {name}')
            inputs[name] = entry
        if addendum.get('expected_identity_sha256') is not None:
            _require(expected is None, 'the expected identity is pinned at most once')
            expected = addendum['expected_identity_sha256']
        chain.append({'path': str(path), 'sha256': hash_file_bytes(data), 'stage': addendum.get('stage')})
    return {'config': config, 'config_sha256': chain[0]['sha256'], 'chain': chain, 'inputs': inputs,
            'expected_identity_sha256': expected}


def hash_file_bytes(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def committed(path):
    """True when ``path`` is inside the repository and its bytes equal the blob at HEAD."""
    path = Path(path).resolve()
    if not path.is_relative_to(REPO):
        return False
    try:
        blob = subprocess.check_output(['git', 'show', f'HEAD:{path.relative_to(REPO).as_posix()}'], cwd=REPO,
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    return blob == path.read_bytes()


def git_state(source=None, registration_files=()):
    """HEAD, uncommitted changes under the code paths, code paths changed since ``source``, and the
    registration files (config, addenda) that are not committed at HEAD byte for byte."""
    run = lambda *args: subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()
    state = {'commit': run('rev-parse', 'HEAD'), 'code_dirty': bool(run('status', '--porcelain', '--', *CODE_PATHS)),
             'source_is_ancestor': None, 'code_changed_since_source': None,
             'registration_uncommitted': [str(f) for f in registration_files if not committed(f)]}
    if source:
        state['source_is_ancestor'] = subprocess.call(['git', 'merge-base', '--is-ancestor', source, 'HEAD'], cwd=REPO,
                                                      stderr=subprocess.DEVNULL) == 0
        state['code_changed_since_source'] = [line for line in run('diff', '--name-only', source, 'HEAD', '--',
                                                                   *CODE_PATHS).splitlines() if line]
    return state


def enforce_source(config, state=None, registration_files=()):
    """C15/C29: the code at HEAD is the registered source commit's code (HEAD may add registration
    commits on top), with no uncommitted change under the code paths; the registration config and
    every addendum are committed at HEAD byte for byte; the member loader matches its pin. The
    locally retargeted ``runs`` symlink is outside the code paths."""
    source = config['le2025_validation_plan'].get('source_commit')
    _require(isinstance(source, str) and source, 'registered source commit required')
    state = git_state(source, registration_files) if state is None else state
    _require(state['source_is_ancestor'] is True and not state['code_changed_since_source'],
             'code at HEAD differs from the registered source commit')
    _require(not state['code_dirty'], 'uncommitted changes under the code paths; refusing to run')
    _require(not state['registration_uncommitted'],
             f'registration files not committed at HEAD: {state["registration_uncommitted"]}')
    loader = config['identity_registration']['member_loader']
    _require(hash_file(REPO / loader['file']) == loader['sha256'], 'member loader source differs from its pin')
    return state


def role_seed(config, role):
    """M-11: non-overlapping role seeds from the registered base (SeedSequence spawn key per role)."""
    _require(role in SEED_ROLES, f'unknown seed role {role}')
    base = config['seeds']['base']
    return int(np.random.SeedSequence([int(base), SEED_ROLES.index(role)]).generate_state(1)[0])


def environment():
    return {'python': sys.executable, 'python_version': platform.python_version(), **pid.environment()}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False, default=str) + '\n')


def _raise_hang(signum, frame):
    prt.defer_or_raise(HangGuardExceeded('stage hang guard exceeded; preserve partial stage (not citable)'))


@contextmanager
def stage(output, command, identity, hang_guard_seconds=None):
    """Fresh stage directory: started.json, then manifest.json (with measured cost) on success or
    failure-*.json (kept). A registered hang guard (SIGALRM) fails the stage; it is not a budget."""
    output = Path(output)
    _require(not output.exists(), f'stage output exists; preserve it and register a new attempt: {output}')
    output.mkdir(parents=True)
    started, cpu = time.perf_counter(), time.process_time()
    dump(output / 'started.json', {'command': command, 'started_utc': datetime.now(timezone.utc).isoformat(), **identity})
    if hang_guard_seconds is not None:
        previous = signal.signal(signal.SIGALRM, _raise_hang)
        signal.setitimer(signal.ITIMER_REAL, float(hang_guard_seconds))
    try:
        yield output
    except BaseException as error:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        dump(output / f'failure-{stamp}.json', {'command': command, 'error_type': type(error).__name__,
             'error': str(error), 'seconds': time.perf_counter() - started, 'preserve_partial_artifacts': True,
             'partial_ledger_prefix': True, 'citable': False})
        raise
    finally:
        if hang_guard_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    names = sorted(str(p.relative_to(output)) for p in output.rglob('*')  # exFAT AppleDouble sidecars are not artifacts
                   if p.is_file() and not is_appledouble(p))
    dump(output / 'manifest.json', {'command': command, 'cost': {
        'wall_seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu,
        'peak_rss_raw': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'peak_rss_unit': 'platform ru_maxrss'},
        'artifact_sha256': {name: hash_file(output / name) for name in names}})


class Deadline:
    """Optional wall-clock guard for a stage loop; exceeding it aborts the ledger (HALTED)."""
    def __init__(self, seconds=None):
        _require(seconds is None or (isinstance(seconds, (int, float)) and seconds > 0), 'positive stage guard required')
        self.seconds, self.start = (None if seconds is None else float(seconds)), time.monotonic()

    def check(self):
        if self.seconds is not None and time.monotonic() - self.start > self.seconds:
            raise BudgetExceeded(f'stage wall guard {self.seconds}s exceeded; preserve partial stage')

    def elapsed(self):
        return time.monotonic() - self.start


def guard_dates(frame):
    dates = pd.to_datetime(frame.game_date)
    _require(len(frame) and dates.max() <= LAST_DATE, 'rows dated after 2025 cannot enter <=2025 validation')
    return frame


# ---------------------------------------------------------------- S0 census

def run_census(frame, vocabulary, no_pitch, outcome_adjacent_splits=('train',)):
    return {'census': preq.census(guard_dates(frame), vocabulary, no_pitch=frozenset(no_pitch),
                                  outcome_adjacent_splits=tuple(outcome_adjacent_splits)), 'label_blind': True}


# ---------------------------------------------------------------- S1 TRAIN BC, hands and support table

def bc_rows(frame, rule, train_keys=None, keys_record=None, no_pitch=preq.AUTOMATIC):
    """TRAIN rows for the BC: BC_P (D-1: no post-decision condition) or BC_E (pinned eligible keys)."""
    if rule == 'BC_P':
        return frame.loc[preq.bc_population_mask(frame, no_pitch)]
    _require(rule == 'BC_E' and train_keys is not None and keys_record is not None, 'BC_E needs pinned TRAIN keys')
    positions = pd.MultiIndex.from_frame(frame[KEY]).get_indexer(pd.MultiIndex.from_frame(train_keys[KEY]))
    _require(len(train_keys) == keys_record['n'] and (positions >= 0).all(), 'pinned BC_E keys absent from the frame')
    selected = frame.iloc[positions]
    _require(ordered_key_hash(selected) == keys_record['rows_sha256'], 'BC_E ordered key identity changed')
    return selected.sort_index()


def logging_mass_report(bc, table):
    """TRAIN-cell distribution of pi_b_hat(M(H)|H): D89's rho_ref == 1 condition, checked not assumed."""
    masses, weights = [], []
    for (pitcher, balls, strikes, side, previous), counter in bc.cells.items():
        history = () if previous == '<START>' else (PastPitch(previous, (0.,) * 8, 'unknown', 0, 0),)
        state = PAState(balls, strikes, pitcher, side, history)
        mask = table.get((pitcher, side))
        p = bc.probabilities(state)
        masses.append(0. if mask is None else float(p[mask & bc.support(state)].sum()))
        weights.append(sum(counter.values()))
    masses, weights = np.asarray(masses), np.asarray(weights, dtype=float)
    order = np.argsort(masses)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    quantile = {str(q): float(masses[order][np.searchsorted(cumulative, q)]) for q in (.01, .05, .25, .5, .75, .95)}
    return {'cells': int(len(masses)), 'decision_weighted_quantiles': quantile,
            'decision_share_mass_equal_one': float(weights[masses >= 1 - 1e-12].sum() / weights.sum()),
            'cell_share_mass_equal_one': float((masses >= 1 - 1e-12).mean()),
            'decision_share_mass_zero': float(weights[masses == 0].sum() / weights.sum())}


def s1_gates(frame, rows_p, rows_e, art_p, art_e, gates, prep_vocabulary):
    """D-1 fail-closed S1 gates registered before the BC exists (FAILED_INTEGRITY on any miss)."""
    keys_p = set(map(tuple, rows_p[KEY].to_numpy().tolist()))
    keys_e = set(map(tuple, rows_e[KEY].to_numpy().tolist()))
    train_rows = int(frame.split.eq('train').sum())
    checks = {'train_rows_equal_registered': train_rows == gates['train_rows'],
              'token_vocabulary_equals_pinned_preparation': list(gates['token_vocabulary']) == list(prep_vocabulary),
              'bc_e_rows': len(rows_e) == gates['bc_e_rows'],
              'bc_e_actions': list(art_e.vocabulary) == list(gates['bc_e_actions']),
              'bc_e_keys_inside_bc_p': keys_e <= keys_p,
              'bc_p_rows_between_bc_e_and_train': gates['bc_e_rows'] <= len(rows_p) <= train_rows,
              'bc_p_vocabulary_equals_token_vocabulary': list(art_p.vocabulary) == list(gates['token_vocabulary']),
              'bc_p_batter_side_known': bool(rows_p.stand.map(normalize_hand).isin(['L', 'R']).all())}
    cells_e, cells_p = art_e.bc.cells, art_p.bc.cells
    checks['bc_p_cell_counts_dominate_bc_e'] = all(cells_p.get(key, {}).get(a, 0) >= n
                                                   for key, counter in cells_e.items() for a, n in counter.items())
    _require(all(checks.values()), f'S1 gate failed: {[k for k, ok in checks.items() if not ok]}')
    differing = sum(1 for key, counter in cells_p.items() if dict(counter) != dict(cells_e.get(key, {})))
    return {'checks': checks, 'cells_bc_p': len(cells_p), 'cells_bc_e': len(cells_e), 'cells_differing': differing,
            'rows_bc_p_minus_bc_e': len(rows_p) - len(rows_e)}


def _code_counts(rows):
    return {str(k): int(v) for k, v in rows.pitch_type.astype(str).value_counts().sort_index().items()}


def change_decision_report(train_rows, bc, table, no_pitch):
    """D-4 critic 5 (TRAIN, label-blind): decisions right after a pitcher change inside a PA, the
    share whose BC cell is empty (pi_b_hat falls back to the pitcher frequency) and their pi_b_hat(M|H)."""
    rows = train_rows.reset_index(drop=True)
    same_pa = rows.game_pk.eq(rows.game_pk.shift()) & rows.at_bat_number.eq(rows.at_bat_number.shift())
    changed = np.flatnonzero((same_pa & rows.pitcher.ne(rows.pitcher.shift())).to_numpy())
    empty, masses = 0, []
    for i in changed:
        row, previous = rows.iloc[i], rows.iloc[i - 1]
        label = preq.logged_label(previous.pitch_type, previous.description, no_pitch)
        action = '<UNKNOWN>' if label in prt.SENTINELS else label
        pitcher = str(int(row.pitcher))
        if pitcher not in bc.pitchers or not (0 <= row.balls <= 3 and 0 <= row.strikes <= 2):
            continue
        state = PAState(int(row.balls), int(row.strikes), pitcher, str(row.stand), (PastPitch(action, (0.,) * 8, 'unknown', 0, 0),))
        empty += bc._key(state) not in bc.cells
        mask = table.get((pitcher, str(row.stand)))
        if mask is not None and bc.support(state).any():
            masses.append(float(bc.probabilities(state)[mask & bc.support(state)].sum()))
    return {'decisions_after_change': int(len(changed)), 'empty_bc_cells': int(empty),
            'pi_b_hat_mass_on_mask_quantiles': ({str(q): float(np.quantile(masses, q)) for q in (.05, .25, .5)}
                                                if masses else None)}


def run_materialize(frame, store, output, *, provenance, bind_inputs, train_keys, keys_record, gates, bc_parameters,
                    no_pitch, volume_quantiles, prep_vocabulary, bc_p_only_rule):
    """Export BC_P (primary) and BC_E (reproduction gate/descriptive only), the TRAIN hand
    registry, the intervention support table over single-hand pitchers and the TRAIN reports.

    ``bind_inputs(bc_artifact, context_rows)`` is ``policy_identity.bind_policy_inputs`` with pins.
    """
    guard_dates(frame)
    _require(isinstance(frame.index, pd.RangeIndex) and frame.index.equals(store.frame.index), 'store/frame index')
    rows = {'BC_P': bc_rows(frame, 'BC_P', no_pitch=no_pitch), 'BC_E': bc_rows(frame, 'BC_E', train_keys, keys_record)}
    rule = {'BC_P': 'split==train & logged_label not a sentinel (registered no-pitch descriptions) & legal count',
            'BC_E': 'pinned eligible D100 TRAIN keys (p4_train_keys)'}
    artifacts = {}
    for name, part in rows.items():
        ids = {**provenance['source_ids'], 'population_rule_sha256': canonical_hash(rule[name]),
               'ordered_row_keys_sha256': ordered_key_hash(part)}
        artifacts[name] = export_train_bc(store, part, output / f'bc_{name}.json', **{**provenance, 'source_ids': ids},
                                          **bc_parameters)
    primary = artifacts['BC_P']
    gate = s1_gates(frame, rows['BC_P'], rows['BC_E'], primary, artifacts['BC_E'], gates, prep_vocabulary)
    train = frame.loc[frame.split.eq('train')]
    hands = pid.hand_registry(train, primary.bc.pitchers)
    _, hands_file_sha = save_hand_registry(primary, hands, int(len(train)), output / 'hands.json')
    templates = pid.support_templates(safe_rows(train).assign(pitch_type=train.pitch_type.to_numpy()), hands)
    inputs, inputs_identity = bind_inputs(primary, templates[list(safe_rows(train).columns)])
    support = pid.support_rows(inputs, templates)
    _, support_file_sha = save_support_table(primary, support, output / 'support_primary.json')
    table, support_sha = load_support_table(output / 'support_primary.json', support_file_sha, primary)
    rare = [a for a in ('UN', 'PO', 'FA', 'EP') if a in primary.vocabulary]
    only_p = sorted(set(primary.bc.pitchers) - set(artifacts['BC_E'].bc.pitchers))
    clustered = set((inputs.context_encoder.clusters or {}).get('pitcher_cluster', {}))
    keys = {(str(int(r['pitcher'])), str(r['stand'])): context_key(r) for r in templates.to_dict('records')}
    pitcher_levels = {level for level, columns in enumerate(inputs.delivery.TIERS) if 'pitcher' in columns}
    league_only = 0
    for pitcher, side, mask in support:  # D-1 critic 5(b): M of BC-P-only pitchers from league/type pools only
        if pitcher in only_p:
            state = PAState(0, 0, pitcher, side, (), keys[(pitcher, side)])
            for action in (a for a, ok in zip(primary.vocabulary, mask) if ok):
                inputs.pool(state, action, count_access=False)
                row = inputs.query(state, action)
                signature = tuple(row[k] for k in ('pitcher', 'pitch_type', 'p_throws', 'stand', 'balls', 'strikes'))
                league_only += inputs.pool_cache[signature][1] not in pitcher_levels
    report = {'bc': {name: {**art.identity(), 'train_rows': art.provenance['train_rows'], 'rule': rule[name],
                            'vocabulary': list(art.vocabulary)} for name, art in artifacts.items()},
              'bc_roles': {'BC_P': 'registered pi_b_hat and reference base', 'BC_E': 'S1 reproduction gate and '
                           'descriptive comparison only; never used for estimation, selection or sensitivity'},
              's1_gates': gate,
              'bc_p_only_pitchers': {'count': len(only_p), 'ids': only_p,
                                     'without_g0_cluster': sum(p not in clustered for p in only_p),
                                     'support_actions_from_league_tiers_only': int(league_only), 'rule': bc_p_only_rule},
              'bc_p_minus_bc_e_codes': _code_counts(rows['BC_P'].loc[~rows['BC_P'].index.isin(rows['BC_E'].index)]),
              'hands': {'file_sha256': hands_file_sha, 'ambiguous': sum(h == 'AMBIGUOUS' for h in hands.values()),
                        'single': sum(h != 'AMBIGUOUS' for h in hands.values())},
              'support_table': {'file_sha256': support_file_sha, 'content_sha256': support_sha, 'rows': len(support),
                                'rare_codes_in_support': {a: sum(bool(m[primary.vocabulary.index(a)]) for *_, m in support)
                                                          for a in rare}},
              'change_decisions': change_decision_report(train, primary.bc, table, no_pitch),
              'inputs_identity': inputs_identity,
              'logging_mass_on_mask': logging_mass_report(primary.bc, table),
              'volume_edges': preq.volume_edges(train, volume_quantiles), 'volume_quantiles': list(volume_quantiles)}
    dump(output / 'materialize.json', report)
    return report


# ---------------------------------------------------------------- S1b style snapshots (D-7)

def run_style_snapshots(frame, output, as_of, provenance):
    """Pinned window-start snapshots plus the label-blind D-7 diagnostics for each window: share of
    rows whose frozen prior differs from the rolling one, |delta| quantiles per column, rows mapped
    to the league row. (The G0 prediction shift needs a bind and is not computed here.)"""
    out = {}
    for split, day in sorted(as_of.items()):
        snapshot = preq.style_snapshot(frame, day)
        source = {**provenance, 'rows_before_as_of': int((pd.to_datetime(frame.game_date) < day).sum())}
        payload, file_sha = save_style_snapshot(snapshot, day, source, output / f'style_{split}.json')
        window = frame.loc[frame.split.eq(split) & (pd.to_datetime(frame.game_date) >= day)]
        diagnostics = None
        if len(window):
            applied, unknown = preq.apply_style_snapshot(window, snapshot, day)
            columns = list(preq.HISTORY_COLUMNS)
            delta = np.abs(applied[columns].to_numpy(np.float64) - window[columns].to_numpy(np.float64))
            diagnostics = {'rows': int(len(window)), 'rows_differing_from_rolling': float((delta > 0).any(axis=1).mean()),
                           'abs_delta_quantiles': {c: {str(q): float(np.quantile(delta[:, j], q)) for q in (.5, .9, .99)}
                                                   for j, c in enumerate(columns)}, **unknown,
                           'rolling_check_on_as_of': preq.snapshot_rolling_mismatches(window, snapshot, day)}
        out[split] = {'file_sha256': file_sha, 'as_of_exclusive': day, 'batters': len(snapshot) - 1,
                      'content_sha256': canonical_hash(payload), 'diagnostics': diagnostics}
    dump(output / 'style_snapshots.json', out)
    return out


# ---------------------------------------------------------------- S2 bind + connection probe

def probe_positions(sealed_keys, levels, frame, count):
    """First ``count`` rows in sealed order whose evaluation used an action-specific pool (level >= 0)."""
    levels = np.asarray(levels)
    _require(all(np.array_equal(levels[0], other) for other in levels[1:]), 'member delivery levels disagree')
    chosen = np.flatnonzero(levels[0] >= 0)[:count]
    _require(len(chosen) == count, 'not enough non-fallback probe rows')
    keys = pd.DataFrame(np.asarray(sealed_keys)[chosen], columns=KEY)
    positions = pd.MultiIndex.from_frame(frame[KEY]).get_indexer(pd.MultiIndex.from_frame(keys))
    _require((positions >= 0).all(), 'probe keys absent from the frame')
    return chosen, positions


def blocks_containing(frame, positions):
    """The PA blocks that contain each position (by the frame's actual PA boundaries)."""
    return preq.pa_blocks(frame, positions)


def run_bind_probe(store, components, sealed, *, count, atol_primary, atol_frequency, no_pitch):
    """Policy path vs sealed evaluation path on registered probe rows (no labels read); a failed
    comparison fails the stage (the report is kept)."""
    chosen, positions = probe_positions(sealed['keys'], sealed['levels'], store.frame, count)
    requests = {}
    for _, block in blocks_containing(store.frame, positions):
        built, problem, index = preq.pa_requests(store, block, 'probe', no_pitch)
        offsets = [int(p - block[0]) for p in positions if block[0] <= p <= block[-1]]
        _require(len(built) > max(offsets), f'probe PA unsubmittable at or before a probe row: {problem}')
        requests.update(zip(block.tolist(), built))
    states = [requests[int(p)].state for p in positions]
    actions = [requests[int(p)].logged_action for p in positions]
    primary = pid.integrated_predictions(components.g0, states, actions)
    frequency_rows = pd.DataFrame([components.inputs.query(s, a) for s, a in zip(states, actions)])
    raw = components.g0.baseline.baseline.predict(frequency_rows)
    dates = pd.to_datetime(store.frame.game_date.iloc[positions]).dt.strftime('%Y-%m-%d')
    report = {'rows': count, 'sealed_indices': chosen.tolist(),
              'probe_dates': {str(k): int(v) for k, v in dates.value_counts().sort_index().items()},
              'primary': pid.compare_probe(primary, sealed['primary'][chosen], atol_primary),
              'frequency_raw': pid.compare_probe(raw, sealed['frequency_raw'][chosen], atol_frequency),
              'verify': components.verify()}
    report['pass'] = report['primary']['pass'] and report['frequency_raw']['pass']
    return report


# ---------------------------------------------------------------- runtime passes

def _counts(values):
    return {str(k): int(v) for k, v in pd.Series(list(values), dtype=object).value_counts().items()}


def submit_pas(runtime, store, blocks, deadline, *, no_pitch, facts=None, outcome=None):
    """Submit every request of every PA (refusals stay in the ledger). Returns {pa_id: facts} with
    the structural problem, E0 membership (first row, pre-decision) and the optional PA end."""
    facts = {} if facts is None else facts
    try:
        for pa_id, positions in blocks:
            deadline.check()
            requests, problem, index = preq.pa_requests(store, positions, runtime.sha256, no_pitch)
            info = {'game': int(store.frame.game_pk.iloc[positions[0]]), 'problem': problem, 'problem_index': index,
                    'positions': positions, 'submitted': len(requests), **preq.pa_manifest(store.frame, positions)}
            if requests:
                first = requests[0]
                inside, reason = runtime.start_population(first.state, first.pitcher_hand)
            else:
                inside, reason = False, f'unsubmittable:{problem}'
            info.update(in_population=inside, start_reason=reason)
            for request in requests:
                runtime.submit(request)
            if outcome is not None:
                info.update(outcome(positions))
            facts[pa_id] = info
    except (BudgetExceeded, HangGuardExceeded) as error:
        runtime.abort(f'{type(error).__name__}: {error}')
        raise
    return facts


def seal_counts(facts, runtime, expected_pas):
    """Seal condition (D-11/M-10): every PA the pinned census lists for the selection was handled
    once, and every submitted request has exactly one ledger decision row (a fresh stage ledger)."""
    _require(len(facts) == expected_pas, 'handled PAs differ from the pinned census count')
    submitted = sum(info['submitted'] for info in facts.values())
    _require(submitted == len(runtime.ledger.decisions()), 'ledger decision rows differ from the submitted requests')
    return {'pas': len(facts), 'requests': submitted}


def run_v5(runtime, store, blocks, deadline, *, no_pitch, census_split=None):
    """Logging-law/reference denominators on one split: statuses, codes and pi_b_hat(M|H)."""
    facts = submit_pas(runtime, store, blocks, deadline, no_pitch=no_pitch)
    rows = runtime.ledger.decisions()
    sealed = seal_counts(facts, runtime, len(blocks) if census_split is None else census_split['pas'])
    masses = np.array([r['result']['logging_mass_on_mask'] for r in rows if r['status'] in prt.EVALUATED])
    outside = sum(r['status'] == prt.OUTSIDE_POLICY_SUPPORT for r in rows)
    return {'pas_in_split': len(blocks), 'pas_handled': len(facts), 'sealed_counts': sealed,
            'structural_problems': _counts(f['problem'] for f in facts.values() if f['problem'] is not None),
            'start_population': {'inside': sum(f['in_population'] for f in facts.values()),
                                 'outside_reasons': _counts(f['start_reason'] for f in facts.values()
                                                            if not f['in_population'])},
            'summary': runtime.summary(),
            'logged_outside_mask_share_of_evaluated': float(outside / len(masses)) if len(masses) else None,
            'logging_mass_on_mask_quantiles': ({str(q): float(np.quantile(masses, q)) for q in (.01, .05, .25, .5)}
                                              if len(masses) else None),
            'seconds': deadline.elapsed(), 'population_value': None}


def outcome_function(store, components, games, rule, score_source='next_row'):
    return lambda positions: preq.pa_outcome(store.frame, positions, components.defense_we, games, rule=rule,
                                             score_source=score_source)


def variant_facts(facts, variant, store, components, games):
    """M-12 registered sensitivities from the same ledger (only the PA-end facts change)."""
    _require(variant in ('flags_to_bounds', 'r5-events-v1', 'post_pitch_scores'), f'unregistered sensitivity {variant}')
    out = {}
    for pa_id, info in facts.items():
        info = dict(info)
        if variant == 'flags_to_bounds' and info.get('flags'):
            info.update(reward=None, kind=preq.TERMINAL_VALUE_MISSING, reason='flagged')
        elif variant != 'flags_to_bounds':
            rule, score = ('r5-events-v1', 'next_row') if variant == 'r5-events-v1' else ('structural-end-v1', 'post_pitch')
            info.update(preq.pa_outcome(store.frame, info['positions'], components.defense_we, games, rule=rule,
                                        score_source=score))
        out[pa_id] = info
    return out


def paired_identity_run(runtime, store, blocks, facts, rows, *, no_pitch, kwargs):
    """M-10 / D89 V4(e): the same requests through an independent cand=ref runtime; every complete
    PA's paired delta must be exactly 0 and the complete set must equal the primary run's."""
    pair_facts = submit_pas(runtime, store, blocks, Deadline(), no_pitch=no_pitch)
    for pa_id, info in pair_facts.items():  # the PA end is the same observed fact
        info.update({k: facts[pa_id].get(k) for k in ('reward', 'reason', 'kind', 'end', 'end_kind', 'flags')})
    _, pair_rows = est.estimate(runtime.ledger.decisions(), pair_facts, **{**kwargs, 'ess_gate': None})
    complete = {r['pa_id'] for r in pair_rows if r['status'] == est.COMPLETE}
    deltas = [abs(r['delta']) for r in pair_rows if r['status'] == est.COMPLETE]
    record = {'pas': len(complete), 'max_abs_delta': float(max(deltas, default=0.)),
              'complete_sets_equal': bool(complete == {r['pa_id'] for r in rows if r['status'] == est.COMPLETE}),
              'runtime_sha256': runtime.sha256}
    record['pass'] = bool(record['complete_sets_equal'] and record['max_abs_delta'] == 0.)
    _require(record['pass'], f'M-10 cand=ref paired identity run failed: {record}')
    return record


def run_dr(runtime, store, components, blocks, deadline, *, no_pitch, bootstrap, ess_gate, sensitivities, strata=None,
           expected_pas=None, pair=None):
    """Candidate runtime over the selected PAs, PA-end facts from the verified WE, the DR estimator
    (primary), the registered same-ledger sensitivities and (``pair``) the M-10 paired run."""
    games = preq.game_table(store.frame, {int(store.frame.game_pk.iloc[p[0]]) for _, p in blocks})
    facts = submit_pas(runtime, store, blocks, deadline, no_pitch=no_pitch,
                       outcome=outcome_function(store, components, games, 'structural-end-v1'))
    if strata is not None:
        for pa_id, info in facts.items():
            info['strata'] = strata(pa_id, info)
    kwargs = dict(draws=bootstrap['draws'], seed=bootstrap['seed'], invalid_share_max=bootstrap['invalid_share_max'],
                  minimum=bootstrap['minimum'], ess_gate=ess_gate)
    sealed = seal_counts(facts, runtime, len(blocks) if expected_pas is None else expected_pas)
    result, rows = est.estimate(runtime.ledger.decisions(), facts, **kwargs)
    result['sealed_counts'] = sealed
    result['sensitivity'] = {}
    for variant in sensitivities:
        other, _ = est.estimate(runtime.ledger.decisions(), variant_facts(facts, variant, store, components, games),
                                **kwargs)
        result['sensitivity'][variant] = {'layers': other['layers'], 'status': other['status'],
                                          'label': 'registered sensitivity; descriptive'}
    runtime.verify_components()
    result['paired_identity_run'] = (None if pair is None else
                                     paired_identity_run(pair(), store, blocks, facts, rows, no_pitch=no_pitch, kwargs=kwargs))
    result.update(pas_in_selection=len(blocks), reward_flags=int(sum(bool(r['flags']) for r in rows
                                                                     if r['status'] == est.COMPLETE)),
                  ledger=runtime.summary(), seconds=deadline.elapsed(),
                  interpretation='exposed_development only; not independent confirmation, not a causal effect')
    return result, rows


# ---------------------------------------------------------------- selection helpers

def straddling_games(frame):
    """Games with rows in more than one split (e.g. suspended in June, resumed in July)."""
    splits = frame.groupby('game_pk').split.nunique()
    return sorted(int(g) for g in splits.index[splits > 1])


def select_games(frame, split, n_games, salt):
    """M-1: complete games of ``split`` by salted hash order, allocated to months in proportion to
    their game counts (largest remainder); every PA of a chosen game is kept. Games that straddle a
    split boundary are never chosen (date rule only; reported by the stage)."""
    straddle = set(straddling_games(frame))
    part = frame.loc[frame.split.eq(split) & ~frame.game_pk.isin(straddle), ['game_pk', 'game_date']]
    part = part.drop_duplicates('game_pk')
    part = part.assign(month=pd.to_datetime(part.game_date).dt.strftime('%Y-%m'))
    _require(0 < n_games <= len(part), 'registered game count outside the split')
    counts = part.month.value_counts().sort_index()
    quota = counts / counts.sum() * n_games
    alloc = np.floor(quota).astype(int)
    for month in (quota - alloc).sort_values(ascending=False, kind='stable').index[:n_games - alloc.sum()]:
        alloc[month] += 1
    chosen = []
    for month, k in alloc.items():
        games = part.loc[part.month.eq(month), 'game_pk'].astype(int).tolist()
        chosen += sorted(games, key=lambda g: canonical_hash([salt, g]))[:k]
    return sorted(chosen)


def game_blocks(frame, games, split):
    positions = np.flatnonzero((frame.game_pk.isin(games) & frame.split.eq(split)).to_numpy())
    return preq.pa_blocks(frame, positions)


def census_pas(census_split, games):
    counts = census_split['pas_by_game']
    _require(all(str(g) in counts for g in games), 'selected game missing from the pinned census')
    return sum(counts[str(g)] for g in games)


def strata_function(frame, volume_edges, bc_p_only=()):
    """M-13 descriptive strata per PA: D87 role, month, TRAIN volume bin, extra innings, and the
    D-1 critic 5 stratum of pitchers known only through BC-P (no eligible TRAIN pitch)."""
    bc_p_only = {str(p) for p in bc_p_only}
    roles = preq.pitcher_roles(frame)
    train_counts = frame.loc[frame.split.eq('train')].groupby('pitcher').size()
    first = {}
    for pa_id, positions in preq.pa_blocks(frame):
        first[pa_id] = positions[0]

    def strata(pa_id, info):
        row = frame.iloc[first[pa_id]]
        volume = int(np.searchsorted(volume_edges, train_counts.get(row.pitcher, 0), side='right'))
        return {'role': roles.iloc[first[pa_id]], 'month': str(pd.Timestamp(row.game_date))[:7],
                'volume_bin': volume, 'extra_innings': bool(int(row.inning) >= 10),
                'bc_p_only_pitcher': str(int(row.pitcher)) in bc_p_only}
    return strata


# ---------------------------------------------------------------- CLI (real data; gated)

def pinned_bytes(path, sha256):
    raw = Path(path).read_bytes()
    _require(hash_file_bytes(raw) == sha256, f'pinned file changed: {path}')
    return raw


def pinned_npz(path, sha256, names):
    """Hash the bytes once, then read only the named arrays from those same bytes."""
    with np.load(io.BytesIO(pinned_bytes(path, sha256)), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in names}


def pinned_parquet(path, sha256):
    return pd.read_parquet(io.BytesIO(pinned_bytes(path, sha256)))


def registered_path(reg, name):
    """A registered input must be a file listed with the same sha256 in the manifest.json of a
    sealed stage directory (the manifest exists only for a successful stage; M-2/M-10)."""
    entry = reg['inputs'].get(name)
    _require(isinstance(entry, dict) and entry.get('file_sha256'), f'registered input required: {name}')
    path, sha = Path(entry['path']), entry['file_sha256']
    manifest = path.parent / 'manifest.json'
    _require(manifest.exists(), f'registered input {name} is not inside a sealed stage directory')
    sealed = json.loads(manifest.read_bytes())
    _require(sealed.get('artifact_sha256', {}).get(path.name) == sha,
             f'registered input {name} differs from its sealed stage manifest')
    _require(name in INPUT_COMMAND and sealed.get('command') == INPUT_COMMAND[name],
             f'registered input {name} does not come from a sealed {INPUT_COMMAND.get(name)} stage')
    return path, sha


def registered_json(reg, name):
    return json.loads(pinned_bytes(*registered_path(reg, name)))


def load_inputs(config, local, *, store=True):
    """Pinned G0 bundle, the verified processed regular-season frame (same loader and order as G0:
    game date, then the pitch key; the history store refuses any other order) and, unless ``store`` is False (census), the normalizer and history store."""
    from run_ml_benchmark import regular_frame
    reg = config['identity_registration']
    bundle_path = REPO / reg['g0_bundle']['path']
    bundle_sha = reg['g0_bundle']['file_sha256']
    bundle = pid.pinned_json(bundle_path, bundle_sha)
    files = bundle['files']
    paths = {role: Path(entry['path']) for role, entry in files.items()}
    paths.update({role: (REPO / value if not Path(value).is_absolute() else Path(value))
                  for role, value in reg['we_paths'].items()})
    parent = pid.pinned_json(paths['p4_parent_preparation'], files['p4_parent_preparation']['sha256'])
    frame = regular_frame(local, parent).sort_values(['game_date', *KEY], kind='stable').reset_index(drop=True)
    frame = guard_dates(frame)
    prep = pid.pinned_json(paths['p4_preparation'], files['p4_preparation']['sha256'])
    out = {'bundle_path': bundle_path, 'bundle_sha': bundle_sha, 'files': files, 'paths': paths, 'parent': parent,
           'prep': prep, 'frame': frame, 'store': None}
    if store:
        from pitchmdp.matrix_features import MatrixHistoryStore
        aux = pid.pinned_pickle(paths['p4_auxiliary'], files['p4_auxiliary']['sha256'])
        out['store'] = MatrixHistoryStore.from_frame(frame, normalizer=aux['normalizer'], history_length=5,
                                                     type_vocabulary=prep['features']['tokens']['type_vocabulary'])
        from run_ml_g0_whole import load_member  # the pinned G0 loader (policy identity records its source)
        out['member_loader'] = load_member
    return out


def pa_contexts(store, blocks, snapshot=None, as_of=None):
    """Safe context rows of the PAs; with a snapshot, their style priors are the frozen values."""
    rows = store.frame.iloc[np.concatenate([positions for _, positions in blocks])]
    contexts = safe_rows(rows)
    if snapshot is None:
        return contexts, None
    return preq.apply_style_snapshot(contexts, snapshot, as_of)


def sealed_probe_arrays(inputs):
    files, paths = inputs['files'], inputs['paths']
    predictions = pinned_npz(paths['p11_predictions'], files['p11_predictions']['sha256'], ('keys', 'primary'))
    frequency = pinned_npz(paths['p11_frequency'], files['p11_frequency']['sha256'], ('mlb_dev_keys', 'mlb_dev_raw'))
    manifest = pid.pinned_json(paths['p11_manifest'], files['p11_manifest']['sha256'])
    levels = []
    for seed in range(5):
        name = str(paths['p11_preparation'].parent / 'members' / 'G0-global' / f'seed{seed}' / 'predictions.npz')
        member = pinned_npz(name, manifest['inputs'][name], ('dev_keys', 'dev_delivery_level'))
        _require(np.array_equal(member['dev_keys'], predictions['keys']), 'member archive keys differ')
        levels.append(member['dev_delivery_level'])
    _require(np.array_equal(frequency['mlb_dev_keys'], predictions['keys']), 'frequency archive keys differ')
    return {'keys': predictions['keys'], 'primary': predictions['primary'], 'frequency_raw': frequency['mlb_dev_raw'],
            'levels': levels}


def check_output(output, plan):
    """C17: a stage directory lives directly under the registered root and never inside another stage."""
    root = Path(plan['output_root']).resolve()
    output = Path(output).resolve()
    _require(output.parent == root, f'stage output must be a new directory directly under {root}')
    return root


SPLIT_OF = {'materialize-bc': 'train', 'profile': 'temperature', 'tau-select': 'blend', 'v5-denominators': 'dev',
            'v2-world': 'dev', 'dr-evaluate': 'dev'}


def dr_q_decision(config, profile, setting):
    """M-7, applied mechanically at S3b from the sealed S3 cost record: the DR q-hat is re-estimated
    with the evaluation seed unless the measured evaluator-inclusive rows per decision of the chosen
    setting, times the registered S6 decisions, exceed the S6 row budget (then planning-Q reuse)."""
    s6 = config['le2025_validation_plan']['stages']['S6_V4']
    entry = next(p for p in profile['profiles'] if (p['samples'], p['pitch_cap']) == (setting['samples'],
                                                                                        setting['pitch_cap']))
    per_decision = (entry['planning_rows'] + entry['evaluation_rows']) / entry['decisions']
    projected = per_decision * s6['planned_decisions']
    source = 'planning_reuse' if projected > s6['row_budget'] else 'evaluation_seed'
    return {'source': source, 'evaluator_inclusive_rows_per_decision': per_decision, 'projected_s6_rows': projected,
            's6_row_budget': s6['row_budget'], 'rule': DR_Q_RULE}


def prerequisites(command, reg):
    """Everything a stage needs from earlier stages, checked before any data load: sealed pins from
    the right stage, the R3(b) census code gate for the stage's split, the S2 pass, the D-9a
    selection (registered noise-rule minimum), a selected tau frozen under the current gate, a
    sealed S4 record and the V2 acceptance of the S6 identity."""
    config = reg['config']
    stages = config['le2025_validation_plan']['stages']
    for name in PREREQUISITES[command]:
        registered_path(reg, name)
    out = {}
    if command == 'census':
        return out
    census = registered_json(reg, 'census')['census']
    out['census'] = census
    split = SPLIT_OF.get(command)
    if split is not None:
        _require(not census[split]['codes_outside_vocabulary'], f'{split} codes outside the vocabulary (R3b): re-register')
    if 'bind_probe' in PREREQUISITES[command]:
        _require(registered_json(reg, 'bind_probe').get('pass') is True, 'S2 probe record must pass')
    if 'bind_identity' in PREREQUISITES[command]:
        out['components_sha256'] = registered_json(reg, 'bind_identity')['sha256']
    if 'profile' in PREREQUISITES[command]:
        out['profile'] = registered_json(reg, 'profile')
        out['setting'] = out['profile'].get('selection')
        _require(out['setting'] is not None, 'registered S3 profile selection (D-9a) required')
        minimum = stages['S3b_tau_select']['samples_minimum_for_noise_rule']
        _require(command != 'tau-select' or out['setting']['samples'] >= minimum,
                 f'the D-9 noise rule needs samples >= {minimum} (paired-difference s.e.)')
    if 'tau_freeze' in PREREQUISITES[command]:
        freeze = out['freeze'] = registered_json(reg, 'tau_freeze')
        _require(freeze['status'] == 'SELECTED' and freeze['final_identity_sha256'],
                 f'no registered tau: {freeze["status"]}')
        gate = {'thresholds': stages['S3b_tau_select']['thresholds'], 'tau_grid': stages['S3b_tau_select']['tau_grid'],
                'ess_gate_game': config['ess_gate']['thresholds']['game']}
        _require(freeze['gate'] == json.loads(json.dumps(gate)), 'tau was frozen under another registered gate (D-9 critic 5)')
    if 'v5' in PREREQUISITES[command]:
        _require(registered_json(reg, 'v5').get('sealed_counts'), 'a sealed S4 record is required')
    if 'v2' in PREREQUISITES[command]:
        v2 = registered_json(reg, 'v2')
        _require(v2.get('accept') is True, 'V2 acceptance is an S6 prerequisite')
        main = [r for r in v2['runs'] if r['check'] == 'V2' and r['planning_seed'] == config['seeds']['planning_main']]
        _require(main and all(r['world']['candidate_identity_sha256'] == out['freeze']['final_identity_sha256']
                              for r in main), 'V2 acceptance does not certify the S6 policy identity')
    return out


def dispatch(command, reg, local, output, load):
    """Run one registered stage. Registration, gates and every prerequisite are checked before the
    data is touched; ``load(store: bool)`` is called only inside the heavy lock and the fresh stage
    directory (C11)."""
    config = reg['config']
    decisions = registration(config, command)
    plan, ident = config['le2025_validation_plan'], config['identity_registration']
    stages, spec = plan['stages'], plan['stages'][STAGES[command]]
    from run_ml_matrix import check_location, heavy_lock
    root = check_location(local, output)
    check_output(output, plan)
    no_pitch = frozenset(config['pa_time_rules']['R3_codes']['no_pitch_descriptions'])
    identity = {'config_sha256': reg['config_sha256'], 'registration_chain': reg['chain'], 'git': git_state(plan.get('source_commit')),
                'environment': environment(), 'decisions': decisions, 'g0_bundle_file_sha256': ident['g0_bundle']['file_sha256'],
                'runner_sources': {rel: hash_file(REPO / rel) for rel in SOURCES}}
    with heavy_lock(root), stage(output, command, identity, spec.get('hang_guard_seconds')) as out:
        pre = prerequisites(command, reg)
        inputs = load(command != 'census')
        frame, store = inputs['frame'], inputs['store']
        vocabulary = inputs['prep']['features']['tokens']['type_vocabulary']
        straddle = straddling_games(frame)

        def bind(blocks, bc_artifact, snapshot=None, as_of=None):
            contexts, style = pa_contexts(store, blocks, snapshot, as_of)
            components = pid.bind_components(inputs['bundle_path'], inputs['bundle_sha'], inputs['paths'],
                                             bc_artifact=bc_artifact, context_rows=contexts,
                                             member_loader=inputs['member_loader'], classes=ident['classes'],
                                             we_contract_sha256=ident['we_contract_sha256'])
            return components, style

        def snapshot_for(split, blocks):
            path, sha = registered_path(reg, f'style_{split}')
            snapshot, as_of, content = load_style_snapshot(path, sha)
            _require(as_of == stages['S1b_style_snapshot']['as_of_exclusive'][split], 'style as-of differs from registration')
            first = pd.to_datetime(frame.game_date.iloc[np.concatenate([p for _, p in blocks])]).min()
            _require(first >= pd.Timestamp(as_of), 'evaluation blocks start before the style as-of date')
            day = frame.loc[frame.split.eq(split)]
            check = preq.snapshot_rolling_mismatches(day, snapshot, as_of)
            _require(check['mismatches'] == 0, f'snapshot differs from rolling priors on the as-of date: {check}')
            return snapshot, as_of, {'style_snapshot_sha256': content, 'as_of_exclusive': as_of, 'rolling_check': check}

        def candidate(blocks, split, *, tau, samples, pitch_cap, seed, budget, evaluation_seed=None, expected=None,
                      tag=''):
            bc_path, bc_sha = registered_path(reg, 'bc')
            support_path, support_sha = registered_path(reg, 'support')
            snapshot, as_of, provenance = snapshot_for(split, blocks)
            components, style = bind(blocks, load_train_bc(bc_path, bc_sha), snapshot, as_of)
            _require(components.sha256 == pre['components_sha256'],
                     'bound components differ from the S2-certified identity (M-10)')
            provenance = {**provenance, 'style_report': style}
            runtime = prt.build_runtime(bc_path, bc_sha, support_path, support_sha, out / f'ledger-{split}-{seed}{tag}.jsonl',
                                        components=components, budget=RowBudget(int(budget), seed_count=5), tau=tau,
                                        samples=samples, pitch_cap=pitch_cap, seed=seed, evaluation_seed=evaluation_seed,
                                        expected_identity_sha256=expected, hand_registry=registered_path(reg, 'hands'),
                                        provenance=provenance)
            return runtime, components, provenance

        if command == 'census':
            dump(out / 'census.json', run_census(frame, vocabulary, no_pitch, spec['outcome_adjacent_splits']))
        elif command == 'materialize-bc':
            dataset = inputs['parent']['dataset_identity']
            provenance = {'source_ids': {**{s['file']: s['sha256'] for s in dataset['sources']},
                                         'processed_cache': dataset['processed_sha256']},
                          'config_sha256': reg['config_sha256'], 'code_commit': identity['git']['commit'],
                          'data_version': f"processed {dataset['processed_sha256'][:12]}; regular R; assign_fold(2025)"}
            files, paths = inputs['files'], inputs['paths']
            run_materialize(frame, store, out, provenance=provenance,
                            bind_inputs=lambda art, rows: pid.bind_policy_inputs(
                                inputs['bundle_path'], inputs['bundle_sha'], paths, bc_artifact=art, context_rows=rows,
                                aux_classes=ident['classes']['aux']),
                            train_keys=pinned_parquet(paths['p4_train_keys'], files['p4_train_keys']['sha256']),
                            keys_record=inputs['prep']['samples']['train'], gates=spec['gates'],
                            bc_parameters=config['train_bc_plan']['bc_parameters'], no_pitch=no_pitch,
                            volume_quantiles=config['subgroups']['volume_quantiles'], prep_vocabulary=vocabulary,
                            bc_p_only_rule=config['train_bc_plan']['bc_p_only_pitcher_rule'])
        elif command == 'style-snapshot':
            dataset = inputs['parent']['dataset_identity']
            run_style_snapshots(frame, out, spec['as_of_exclusive'], {'processed_sha256': dataset['processed_sha256'],
                                                                     'code_commit': identity['git']['commit']})
        elif command == 'bind-probe':
            sealed = sealed_probe_arrays(inputs)
            _, positions = probe_positions(sealed['keys'], sealed['levels'], frame, spec['rows'])
            blocks = blocks_containing(frame, positions)
            bc_path, bc_sha = registered_path(reg, 'bc')
            components, _ = bind(blocks, load_train_bc(bc_path, bc_sha))  # S2 keeps rolling priors (D-7)
            dump(out / 'identity.json', {'sha256': components.sha256, 'identity': components.identity})
            report = run_bind_probe(store, components, sealed, count=spec['rows'], atol_primary=spec['atol_primary'],
                                    atol_frequency=spec['atol_frequency_raw'], no_pitch=no_pitch)
            dump(out / 'probe.json', report)
            _require(report['pass'], 'S2 connection probe failed; the stage is not sealed')
        elif command == 'v5-denominators':
            bc_path, bc_sha = registered_path(reg, 'bc')
            support_path, support_sha = registered_path(reg, 'support')
            runtime = prt.build_runtime(bc_path, bc_sha, support_path, support_sha, out / 'ledger.jsonl',
                                        hand_registry=registered_path(reg, 'hands'))
            blocks = preq.pa_blocks(frame, np.flatnonzero(frame.split.eq('dev').to_numpy()))
            dump(out / 'v5.json', run_v5(runtime, store, blocks, Deadline(), no_pitch=no_pitch,
                                         census_split=pre['census']['dev']))
        elif command == 'profile':
            blocks = preq.pa_blocks(frame, np.flatnonzero(frame.split.eq('temperature').to_numpy()))
            salt = role_seed(config, 'profile_salt')
            blocks = sorted(blocks, key=lambda b: canonical_hash([salt, b[0]]))[:spec['starts']]
            profiles = []
            for setting in spec['candidates']:  # D-9a: measure every registered setting on the same starts
                runtime, _, provenance = candidate(
                    blocks, 'temperature', tau=stages['S3b_tau_select']['tau_grid'][0], samples=setting['samples'],
                    pitch_cap=setting['pitch_cap'], seed=config['seeds']['planning_main'], budget=spec['row_budget'],
                    evaluation_seed=role_seed(config, 'evaluation'), tag=f"-s{setting['samples']}c{setting['pitch_cap']}")
                deadline = Deadline()
                submit_pas(runtime, store, blocks, deadline, no_pitch=no_pitch)
                runtime.verify_components()
                planning = runtime.improvement.diagnostics['search_conditional_rows']
                profiles.append({'samples': setting['samples'], 'pitch_cap': setting['pitch_cap'],
                                 'decisions': sum(r['status'] in prt.EVALUATED for r in runtime.ledger.decisions()),
                                 'requests': len(runtime.ledger.decisions()), 'planning_rows': planning,
                                 'evaluation_rows': runtime.evaluator.diagnostics['search_conditional_rows'],
                                 'conditional_rows': planning, 'seconds': deadline.elapsed()})
            minimum = stages['S3b_tau_select']['samples_minimum_for_noise_rule']
            dump(out / 'profile.json', {'pas': len(blocks), 'profiles': profiles,
                                        'selection': ptau.select_search_settings(profiles, spec['selection_row_budget'],
                                                                                 min_samples=minimum),
                                        'selection_rows': 'planning search rows (S3b runs no evaluator)',
                                        'tau_placeholder_not_selected': stages['S3b_tau_select']['tau_grid'][0],
                                        'quality_values_read': False, 'provenance': provenance})
        elif command == 'tau-select':
            setting, seed = pre['setting'], config['seeds']['planning_main']
            games = select_games(frame, 'blend', spec['n_games'], role_seed(config, 'selection_salt'))
            blocks = game_blocks(frame, games, 'blend')
            runtime, components, provenance = candidate(blocks, 'blend', tau=spec['tau_grid'][0], samples=setting['samples'],
                                                        pitch_cap=setting['pitch_cap'], seed=seed, budget=spec['row_budget'])
            facts = submit_pas(runtime, store, blocks, Deadline(), no_pitch=no_pitch)
            sealed = seal_counts(facts, runtime, census_pas(pre['census']['blend'], games))
            for pa_id, positions in blocks:  # only the presence of a terminal marker is read
                facts[pa_id]['terminal_marker'] = bool(pd.notna(frame.events.iloc[positions[-1]]))
            table = ptau.tau_table(runtime.ledger.decisions(), facts, spec['tau_grid'], spec['thresholds'])
            runtime.verify_components()
            final, determinism = None, None
            dr_q = dr_q_decision(config, pre['profile'], setting)
            evaluation = role_seed(config, 'evaluation') if dr_q['source'] == 'evaluation_seed' else None
            if table['selected_tau'] is not None:
                bc_art = load_train_bc(*registered_path(reg, 'bc'))
                _, support_identity = load_support_table(*registered_path(reg, 'support'), bc_art)
                _, hands_identity = load_hand_registry(*registered_path(reg, 'hands'), bc_art)
                final = prt.candidate_identity(components, support_identity, hands_identity,
                                               {'tau': table['selected_tau'], 'samples': setting['samples'],
                                                'pitch_cap': setting['pitch_cap'], 'seed': seed}, evaluation)['sha256']
                determinism = determinism_check(runtime, candidate, blocks, facts, store, no_pitch,
                                                tau=table['selected_tau'], setting=setting, seed=seed,
                                                budget=spec['row_budget'], pas=spec['determinism_check_pas'],
                                                atol=stages['S2_bind_probe']['atol_primary'])
            dump(out / 'tau_freeze.json', {**table, 'games': games, 'straddling_games_excluded': straddle,
                 'sealed_counts': sealed, 'ledger_head_sha256': runtime.summary()['ledger_head_sha256'],
                 'config_sha256': reg['config_sha256'], 'dr_q': dr_q, 'setting': setting,
                 'gate': {'thresholds': spec['thresholds'], 'tau_grid': spec['tau_grid'],
                          'ess_gate_game': config['ess_gate']['thresholds']['game']},
                 'tau_code_sha256': hash_file(REPO / 'experiments/pitchmdp/pitchmdp/policy_tau.py'),
                 'final_identity_sha256': final, 'determinism_check': determinism, 'outcomes_read': False,
                 'provenance': provenance})
        elif command == 'v2-world':
            freeze = pre['freeze']
            evaluation = role_seed(config, 'evaluation') if freeze['dr_q']['source'] == 'evaluation_seed' else None
            dump(out / 'v2.json', run_v2_stage(reg, spec, frame, store, candidate, freeze['selected_tau'], pre['setting'],
                                               no_pitch, evaluation))
        else:  # dr-evaluate (S6/V4)
            freeze, setting = pre['freeze'], pre['setting']
            expected = freeze['final_identity_sha256']
            evaluation = role_seed(config, 'evaluation') if freeze['dr_q']['source'] == 'evaluation_seed' else None
            _require(reg['expected_identity_sha256'] in (None, expected), 'registered identity differs from the tau record')
            games = select_games(frame, 'dev', spec['n_games'], role_seed(config, 'selection_salt'))
            blocks = game_blocks(frame, games, 'dev')
            runtime, components, provenance = candidate(
                blocks, 'dev', tau=freeze['selected_tau'], samples=setting['samples'], pitch_cap=setting['pitch_cap'],
                seed=config['seeds']['planning_main'], budget=spec['row_budget'], evaluation_seed=evaluation,
                expected=expected)
            diagnostic = style_shift_diagnostic(runtime, components, lambda sub: bind(sub, load_train_bc(
                *registered_path(reg, 'bc')))[0], store, blocks, spec['d7_diagnostic_starts'], no_pitch)
            pair = lambda: prt.build_reference_pair_runtime(
                *registered_path(reg, 'bc'), *registered_path(reg, 'support'), out / 'ledger-dev-paired-identity.jsonl',
                hand_registry=registered_path(reg, 'hands'), provenance=runtime.pins['provenance'])
            boot = plan['bootstrap']
            materialized = registered_json(reg, 'materialize')
            result, rows = run_dr(runtime, store, components, blocks, Deadline(), no_pitch=no_pitch,
                                  bootstrap={**boot, 'seed': role_seed(config, 'bootstrap')},
                                  ess_gate=config['ess_gate']['thresholds'],
                                  sensitivities=config['sensitivity']['same_ledger'],
                                  strata=strata_function(frame, materialized['volume_edges'],
                                                         materialized['bc_p_only_pitchers']['ids']),
                                  expected_pas=census_pas(pre['census']['dev'], games), pair=pair)
            dump(out / 'dr.json', {**result, 'games': games, 'straddling_games_excluded': straddle,
                                   'identity_sha256': expected, 'dr_q': freeze['dr_q'], 'provenance': provenance,
                                   'd7_label_blind_diagnostic': diagnostic})
            table = pd.DataFrame([{**{k: v for k, v in r.items() if k != 'strata'},
                                   **{f'stratum_{k}': v for k, v in r['strata'].items()}} for r in rows])
            table.to_parquet(out / 'pa_values.parquet', index=False)


def style_shift_diagnostic(runtime, components, bind_rolling, store, blocks, count, no_pitch):
    """D-7 label-blind diagnostics before any reward is read, on the first ``count`` E0 PA starts in
    selection order: G0 10-class TV and KL(snapshot || rolling) over each start's supported actions,
    and the batter soft-membership TV, frozen vs rolling style priors. No outcome, WE or next row."""
    starts = []
    for pa_id, positions in blocks:
        requests, _, _ = preq.pa_requests(store, positions, runtime.sha256, no_pitch)
        if requests and runtime.start_population(requests[0].state, requests[0].pitcher_hand)[0]:
            starts.append(((pa_id, positions), requests[0]))
        if len(starts) == count:
            break
    if not starts:
        return {'starts': 0, 'labels_read': False}
    rolling = bind_rolling([block for block, _ in starts])
    tv, kl = [], []
    for _, request in starts:
        actions = [a for a, ok in zip(runtime.bc.actions, runtime.reference.support(request.state)) if ok]
        states = [request.state] * len(actions)
        snap = np.asarray(pid.integrated_predictions(components.g0, states, actions), dtype=np.float64)
        roll = np.asarray(pid.integrated_predictions(rolling.g0, states, actions), dtype=np.float64)
        tv += list(.5 * np.abs(snap - roll).sum(axis=1))
        kl += list((snap * np.log(np.clip(snap, 1e-300, None) / np.clip(roll, 1e-300, None))).sum(axis=1))
    quantiles = lambda v: {str(q): float(np.quantile(v, q)) for q in (.5, .9, .99)}
    archetypes = getattr(getattr(components.inputs.context_encoder, 'base', None), 'archetypes', None)
    membership = None
    if archetypes is not None:
        keys = [request.state.context_key for _, request in starts]
        frozen = pd.DataFrame([components.inputs.rows[k] for k in keys])
        moving = pd.DataFrame([rolling.inputs.rows[k] for k in keys])
        membership = quantiles(.5 * np.abs(archetypes.transform(frozen) - archetypes.transform(moving)).sum(axis=1))
    return {'starts': len(starts), 'g0_tv': quantiles(tv), 'g0_kl_snapshot_rolling': quantiles(kl),
            'membership_tv': membership, 'labels_read': False,
            'note': 'membership_tv None when the bound context encoder has no archetypes (synthetic encoders)'}


def determinism_check(runtime, candidate, blocks, facts, store, no_pitch, *, tau, setting, seed, budget, pas, atol):
    """D-9 note 5: a runtime rebuilt with the selected tau reproduces the recorded planning Q of the
    first registered number of E0 PAs (fresh ledger; no outcome read)."""
    chosen = [(pa_id, positions) for pa_id, positions in blocks if facts[pa_id]['in_population']][:pas]
    rebuilt, _, _ = candidate(chosen, 'blend', tau=tau, samples=setting['samples'], pitch_cap=setting['pitch_cap'],
                              seed=seed, budget=budget, tag='-determinism')
    submit_pas(rebuilt, store, chosen, Deadline(), no_pitch=no_pitch)
    first = {r['request_id']: r for r in runtime.ledger.decisions()}
    worst = 0.
    for row in rebuilt.ledger.decisions():
        a, b = row['result'], first[row['request_id']]['result']
        if a is None or b is None:
            _require(a is None and b is None, 'determinism check: statuses differ')
            continue
        diffs = [abs(x - y) for x, y in zip(a['q_planning'], b['q_planning']) if x is not None and y is not None]
        worst = max([worst, *diffs])
    _require(worst <= atol, f'determinism check failed: max |dQ| {worst}')
    return {'pas': len(chosen), 'max_abs_q_difference': worst, 'atol': atol}


def run_v2_stage(reg, spec, frame, store, candidate, tau, setting, no_pitch, evaluation):
    """S5: V2 (generating law = pi_b_hat) and V3 (frequency, tempered BC) from the E0 PA starts of
    M-1 selected DEV games, repeated for the registered planning seeds (M-11), with the S6 q-hat
    configuration (evaluation seed when registered); one fresh ledger and one RowBudget per
    (seed, law) run. Acceptance needs every V2 seed to accept."""
    config = reg['config']
    games = select_games(frame, 'dev', spec['n_games'], role_seed(config, 'selection_salt'))
    blocks = game_blocks(frame, games, 'dev')
    laws = [('V2', 'pi_b_hat', lambda bc: (lambda s: (bc.actions, bc.probabilities(s)))),
            ('V3', 'frequency', pss.frequency_law)]
    laws += [('V3', f'tempered_alpha_{a}', lambda bc, a=a: pss.tempered_law(bc, a)) for a in spec['tempered_alpha_grid']]
    report, accepts = {'games': games, 'runs': []}, []
    for seed in config['seeds']['planning_v2']:
        for check, law_name, make in laws:
            runtime, components, _ = candidate(blocks, 'dev', tau=tau, samples=setting['samples'],
                                               pitch_cap=setting['pitch_cap'], seed=seed,
                                               budget=spec['row_budget_per_run'], evaluation_seed=evaluation,
                                               tag=f'-{law_name}')
            starts = []
            for _, positions in blocks:
                requests, _, _ = preq.pa_requests(store, positions, runtime.sha256, no_pitch)
                if requests and runtime.start_population(requests[0].state, requests[0].pitcher_hand)[0]:
                    starts.append((requests[0].state, requests[0].pitcher_hand))
            result = pss.run_world(runtime, components, starts, law=make(runtime.bc), law_identity=law_name,
                                   logs_per_start=spec['logs_per_start'], cap=spec['cap'],
                                   truth_rollouts=spec['truth_rollouts'], seed=role_seed(config, 'v2'),
                                   draws=config['le2025_validation_plan']['bootstrap']['draws'],
                                   tolerance=spec['tolerance'] if check == 'V2' else None)
            report['runs'].append({'check': check, 'law': law_name, 'planning_seed': seed, **result})
            if check == 'V2':
                accepts.append(result['accept'])
    report['accept'] = None if any(a is None for a in accepts) else all(accepts)
    hazard = spec.get('declared_hazard')
    if hazard:  # D-5 critic 5: coverage of the L1 bounds and endpoint CIs under a declared hazard (not acceptance)
        checks = []
        for r in range(spec['hazard_replicates']):
            runtime, components, _ = candidate(blocks, 'dev', tau=tau, samples=setting['samples'],
                                               pitch_cap=setting['pitch_cap'], seed=config['seeds']['planning_main'],
                                               budget=spec['row_budget_per_run'], evaluation_seed=evaluation,
                                               tag=f'-hazard{r}')
            starts = []
            for _, positions in blocks:
                requests, _, _ = preq.pa_requests(store, positions, runtime.sha256, no_pitch)
                if requests and runtime.start_population(requests[0].state, requests[0].pitcher_hand)[0]:
                    starts.append((requests[0].state, requests[0].pitcher_hand))
            seed_r = int(np.random.SeedSequence([role_seed(config, 'v2'), 2, r]).generate_state(1)[0])
            result = pss.run_world(runtime, components, starts, law=laws[0][2](runtime.bc), law_identity='pi_b_hat',
                                   logs_per_start=spec['logs_per_start'], cap=spec['cap'],
                                   truth_rollouts=spec['truth_rollouts'], seed=seed_r,
                                   draws=config['le2025_validation_plan']['bootstrap']['draws'],
                                   censor_hazard=lambda state, action: float(hazard.get(action, 0.)))
            checks.append({k: result[k] for k in ('truth_delta_mean', 'l1_delta_bounds', 'truth_inside_l1_bounds',
                                                  'truth_inside_endpoint_ci', 'ends')})
        report['hazard_check'] = {'declared_hazard': hazard, 'replicates': len(checks), 'runs': checks,
                                  'share_inside_l1_bounds': float(np.mean([c['truth_inside_l1_bounds'] for c in checks])),
                                  'share_inside_endpoint_ci': float(np.mean([c['truth_inside_endpoint_ci'] for c in checks])),
                                  'label': 'D-5 diagnostic; not part of acceptance'}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--addendum', action='append', default=[], type=Path)
    parser.add_argument('--local-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('command', choices=COMMANDS)
    args = parser.parse_args()
    reg = load_registration(args.config, args.addendum)
    registration(reg['config'], args.command)  # refuse before any data is touched
    enforce_source(reg['config'], registration_files=[args.config, *args.addendum])
    raw = args.local_config.read_bytes()
    _require(hash_file_bytes(raw) == reg['config']['le2025_validation_plan']['local_config']['sha256'],
             'local config differs from its registered pin')
    local = json.loads(raw)
    dispatch(args.command, reg, local, args.output.resolve(), lambda store: load_inputs(reg['config'], local, store=store))


if __name__ == '__main__':
    main()
