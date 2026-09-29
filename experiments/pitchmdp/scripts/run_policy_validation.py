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
from pitchmdp.matrix_policy import safe_rows
from pitchmdp.policy_artifacts import (IntegrityError, _require, export_train_bc, load_hand_registry,
                                       load_style_snapshot, load_support_table, load_train_bc, save_hand_registry,
                                       save_style_snapshot, save_support_table)
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
REQUIRED = {  # fields a stage reads; null = unregistered = refuse before any data is touched
    'profile': ('S3_profile.starts', 'S3_profile.probe_setting.samples', 'S3_profile.probe_setting.pitch_cap',
                'S3_profile.row_budget', 'S3b_tau_select.tau_grid'),
    'tau-select': ('S3b_tau_select.n_games', 'S3b_tau_select.tau_grid', 'S3b_tau_select.thresholds.pa_ess_ratio_min',
                   'S3b_tau_select.thresholds.game_ess_min', 'S3b_tau_select.thresholds.ess_ratio_candidate_reference_min',
                   'S3b_tau_select.thresholds.noise_ratio_q90_max', 'S3b_tau_select.thresholds.safety_multiplier',
                   'S3b_tau_select.row_budget'),
    'v2-world': ('S5_V2_V3.starts', 'S5_V2_V3.logs_per_start', 'S5_V2_V3.truth_rollouts', 'S5_V2_V3.cap',
                 'S5_V2_V3.row_budget', 'S5_V2_V3.tempered_alpha_grid'),
    'dr-evaluate': ('S6_V4.n_games', 'S6_V4.row_budget', 'S6_V4.dr_q_source'),
}
SOURCES = ('experiments/pitchmdp/scripts/run_policy_validation.py', 'experiments/pitchmdp/pitchmdp/policy_requests.py',
           'experiments/pitchmdp/pitchmdp/policy_estimator.py', 'experiments/pitchmdp/pitchmdp/policy_runtime.py',
           'experiments/pitchmdp/pitchmdp/policy_semisynthetic.py', 'experiments/pitchmdp/pitchmdp/policy_tau.py',
           'experiments/pitchmdp/pitchmdp/policy_identity.py', 'experiments/pitchmdp/pitchmdp/policy_artifacts.py')


class HangGuardExceeded(RuntimeError):
    """D-11 safety timeout: the stage is failed and preserved; never a budget or a truncation."""


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
    plan = config['le2025_validation_plan']
    _require(isinstance((plan.get('local_config') or {}).get('sha256'), str), 'local config pin required')
    for dotted in REQUIRED.get(command, ()):
        _require(_field(plan['stages'], dotted) is not None, f'registered field required: stages.{dotted}')
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


def git_state():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()
    return {'commit': head, 'dirty': bool(dirty)}


def enforce_source(config, state=None):
    """C15: the registered source commit is HEAD on a clean tree; the member loader matches its pin."""
    state = git_state() if state is None else state
    source = config['le2025_validation_plan'].get('source_commit')
    _require(isinstance(source, str) and state['commit'] == source, 'HEAD differs from the registered source commit')
    _require(not state['dirty'], 'working tree is dirty; refusing to run')
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
    raise HangGuardExceeded('stage hang guard exceeded; preserve partial stage (not citable)')


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
    names = sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())
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

def run_census(frame, vocabulary, no_pitch):
    return {'census': preq.census(guard_dates(frame), vocabulary, no_pitch=frozenset(no_pitch)), 'label_blind': True}


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


def s1_gates(frame, rows_p, rows_e, art_p, art_e, gates):
    """D-1 fail-closed S1 gates registered before the BC exists (FAILED_INTEGRITY on any miss)."""
    keys_p = set(map(tuple, rows_p[KEY].to_numpy().tolist()))
    keys_e = set(map(tuple, rows_e[KEY].to_numpy().tolist()))
    train_rows = int(frame.split.eq('train').sum())
    checks = {'bc_e_rows': len(rows_e) == gates['bc_e_rows'],
              'bc_e_actions': list(art_e.vocabulary) == list(gates['bc_e_actions']),
              'bc_e_keys_inside_bc_p': keys_e <= keys_p,
              'bc_p_rows_between_bc_e_and_train': gates['bc_e_rows'] <= len(rows_p) <= train_rows,
              'bc_p_vocabulary_equals_token_vocabulary': list(art_p.vocabulary) == list(gates['token_vocabulary'])}
    cells_e, cells_p = art_e.bc.cells, art_p.bc.cells
    checks['bc_p_cell_counts_dominate_bc_e'] = all(cells_p.get(key, {}).get(a, 0) >= n
                                                   for key, counter in cells_e.items() for a, n in counter.items())
    _require(all(checks.values()), f'S1 gate failed: {[k for k, ok in checks.items() if not ok]}')
    differing = sum(1 for key, counter in cells_p.items() if dict(counter) != dict(cells_e.get(key, {})))
    return {'checks': checks, 'cells_bc_p': len(cells_p), 'cells_bc_e': len(cells_e), 'cells_differing': differing,
            'rows_bc_p_minus_bc_e': len(rows_p) - len(rows_e)}


def run_materialize(frame, store, output, *, provenance, bind_inputs, train_keys, keys_record, gates, bc_parameters,
                    no_pitch, volume_quantiles):
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
    gate = s1_gates(frame, rows['BC_P'], rows['BC_E'], primary, artifacts['BC_E'], gates)
    train = frame.loc[frame.split.eq('train')]
    hands = pid.hand_registry(train, primary.bc.pitchers)
    _, hands_file_sha = save_hand_registry(primary, hands, int(len(train)), output / 'hands.json')
    templates = pid.support_templates(safe_rows(train).assign(pitch_type=train.pitch_type.to_numpy()), hands)
    inputs, inputs_identity = bind_inputs(primary, templates[list(safe_rows(train).columns)])
    support = pid.support_rows(inputs, templates)
    _, support_file_sha = save_support_table(primary, support, output / 'support_primary.json')
    table, support_sha = load_support_table(output / 'support_primary.json', support_file_sha, primary)
    rare = [a for a in ('UN', 'PO', 'FA', 'EP') if a in primary.vocabulary]
    report = {'bc': {name: {**art.identity(), 'train_rows': art.provenance['train_rows'], 'rule': rule[name],
                            'vocabulary': list(art.vocabulary)} for name, art in artifacts.items()},
              'bc_roles': {'BC_P': 'registered pi_b_hat and reference base', 'BC_E': 'S1 reproduction gate and '
                           'descriptive comparison only; never used for estimation, selection or sensitivity'},
              's1_gates': gate,
              'hands': {'file_sha256': hands_file_sha, 'ambiguous': sum(h == 'AMBIGUOUS' for h in hands.values()),
                        'single': sum(h != 'AMBIGUOUS' for h in hands.values())},
              'support_table': {'file_sha256': support_file_sha, 'content_sha256': support_sha, 'rows': len(support),
                                'rare_codes_in_support': {a: sum(bool(m[primary.vocabulary.index(a)]) for *_, m in support)
                                                          for a in rare}},
              'inputs_identity': inputs_identity,
              'logging_mass_on_mask': logging_mass_report(primary.bc, table),
              'volume_edges': preq.volume_edges(train, volume_quantiles), 'volume_quantiles': list(volume_quantiles)}
    dump(output / 'materialize.json', report)
    return report


# ---------------------------------------------------------------- S1b style snapshots (D-7)

def run_style_snapshots(frame, output, as_of, provenance):
    out = {}
    for split, day in sorted(as_of.items()):
        snapshot = preq.style_snapshot(frame, day)
        source = {**provenance, 'rows_before_as_of': int((pd.to_datetime(frame.game_date) < day).sum())}
        payload, file_sha = save_style_snapshot(snapshot, day, source, output / f'style_{split}.json')
        out[split] = {'file_sha256': file_sha, 'as_of_exclusive': day, 'batters': len(snapshot) - 1,
                      'content_sha256': canonical_hash(payload)}
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


def run_bind_probe(store, components, sealed, *, count, atol_primary, atol_frequency):
    """Policy path vs sealed evaluation path on registered probe rows (no labels read); a failed
    comparison fails the stage (the report is kept)."""
    chosen, positions = probe_positions(sealed['keys'], sealed['levels'], store.frame, count)
    requests = {}
    for _, block in blocks_containing(store.frame, positions):
        built, problem, index = preq.pa_requests(store, block, 'probe')
        _require(problem is None or index > int(np.max(np.searchsorted(block, positions))),
                 f'probe PA unsubmittable before a probe row: {problem}')
        requests.update(zip(block.tolist(), built))
    states = [requests[int(p)].state for p in positions]
    actions = [requests[int(p)].logged_action for p in positions]
    primary = pid.integrated_predictions(components.g0, states, actions)
    frequency_rows = pd.DataFrame([components.inputs.query(s, a) for s, a in zip(states, actions)])
    raw = components.g0.baseline.baseline.predict(frequency_rows)
    dates = pd.to_datetime(store.frame.game_date.iloc[positions]).dt.strftime('%Y-%m-%d')
    report = {'rows': count, 'sealed_indices': chosen.tolist(), 'probe_dates': dict(dates.value_counts().sort_index()),
              'primary': pid.compare_probe(primary, sealed['primary'][chosen], atol_primary),
              'frequency_raw': pid.compare_probe(raw, sealed['frequency_raw'][chosen], atol_frequency),
              'verify': components.verify()}
    report['pass'] = report['primary']['pass'] and report['frequency_raw']['pass']
    return report


# ---------------------------------------------------------------- runtime passes

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
    """Seal condition (D-11): every PA of the selection was handled once and every submitted request
    has exactly one ledger decision row (a fresh stage ledger)."""
    _require(len(facts) == expected_pas, 'handled PAs differ from the expected count')
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
    problems = pd.Series([f['problem'] for f in facts.values() if f['problem'] is not None], dtype=object)
    return {'pas_in_split': len(blocks), 'pas_handled': len(facts), 'sealed_counts': sealed,
            'structural_problems': {str(k): int(v) for k, v in problems.value_counts().items()},
            'start_population': {'inside': sum(f['in_population'] for f in facts.values()),
                                 'outside_reasons': dict(pd.Series([f['start_reason'] for f in facts.values()
                                                                    if not f['in_population']], dtype=object)
                                                         .value_counts().astype(int))},
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


def run_dr(runtime, store, components, blocks, deadline, *, no_pitch, bootstrap, ess_gate, sensitivities, strata=None):
    """Candidate runtime over the selected PAs, PA-end facts from the verified WE, then the DR estimator
    (primary) and the registered same-ledger sensitivities."""
    games = preq.game_table(store.frame)
    facts = submit_pas(runtime, store, blocks, deadline, no_pitch=no_pitch,
                       outcome=outcome_function(store, components, games, 'structural-end-v1'))
    if strata is not None:
        for pa_id, info in facts.items():
            info['strata'] = strata(pa_id, info)
    kwargs = dict(draws=bootstrap['draws'], seed=bootstrap['seed'], invalid_share_max=bootstrap['invalid_share_max'],
                  minimum=bootstrap['minimum'], ess_gate=ess_gate)
    sealed = seal_counts(facts, runtime, len(blocks))
    result, rows = est.estimate(runtime.ledger.decisions(), facts, **kwargs)
    result['sealed_counts'] = sealed
    result['sensitivity'] = {}
    for variant in sensitivities:
        other, _ = est.estimate(runtime.ledger.decisions(), variant_facts(facts, variant, store, components, games),
                                **kwargs)
        result['sensitivity'][variant] = {'layers': other['layers'], 'status': other['status'],
                                          'label': 'registered sensitivity; descriptive'}
    runtime.verify_components()
    result.update(pas_in_selection=len(blocks), reward_flags=int(sum(bool(r['flags']) for r in rows
                                                                     if r['status'] == est.COMPLETE)),
                  ledger=runtime.summary(), seconds=deadline.elapsed(),
                  interpretation='exposed_development only; not independent confirmation, not a causal effect')
    return result, rows


# ---------------------------------------------------------------- selection helpers

def select_games(frame, split, n_games, salt):
    """M-1: complete games of ``split`` by salted hash order, allocated to months in proportion to
    their game counts (largest remainder); every PA of a chosen game is kept."""
    part = frame.loc[frame.split.eq(split), ['game_pk', 'game_date']].drop_duplicates('game_pk')
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


def game_blocks(frame, games):
    positions = np.flatnonzero(frame.game_pk.isin(games).to_numpy())
    return preq.pa_blocks(frame, positions)


def strata_function(frame, volume_edges):
    """M-13 descriptive strata per PA: D87 role, month, TRAIN volume bin, extra innings."""
    roles = preq.pitcher_roles(frame)
    train_counts = frame.loc[frame.split.eq('train')].groupby('pitcher').size()
    first = {}
    for pa_id, positions in preq.pa_blocks(frame):
        first[pa_id] = positions[0]

    def strata(pa_id, info):
        row = frame.iloc[first[pa_id]]
        volume = int(np.searchsorted(volume_edges, train_counts.get(row.pitcher, 0), side='right'))
        return {'role': roles.iloc[first[pa_id]], 'month': str(pd.Timestamp(row.game_date))[:7],
                'volume_bin': volume, 'extra_innings': bool(int(row.inning) >= 10)}
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


def registered_json(reg, name):
    entry = reg['inputs'].get(name)
    _require(isinstance(entry, dict) and entry.get('file_sha256'), f'registered input required: {name}')
    return json.loads(pinned_bytes(entry['path'], entry['file_sha256']))


def registered_path(reg, name):
    entry = reg['inputs'].get(name)
    _require(isinstance(entry, dict) and entry.get('file_sha256'), f'registered input required: {name}')
    return Path(entry['path']), entry['file_sha256']


def load_inputs(config, local, *, store=True):
    """Pinned G0 bundle, the verified processed regular-season frame (same loader as G0, sorted by
    the pitch key) and, unless ``store`` is False (census), the normalizer and history store."""
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
    frame = regular_frame(local, parent).sort_values(KEY, kind='stable').reset_index(drop=True)
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


def dispatch(command, reg, local, output, load):
    """Run one registered stage. ``load(store: bool)`` returns the pinned inputs; it is called only
    inside the heavy lock and the fresh stage directory (C11)."""
    config = reg['config']
    decisions = registration(config, command)
    plan, ident = config['le2025_validation_plan'], config['identity_registration']
    stages, spec = plan['stages'], plan['stages'][STAGES[command]]
    from run_ml_matrix import check_location, heavy_lock
    root = check_location(local, output)
    check_output(output, plan)
    no_pitch = frozenset(config['pa_time_rules']['R3_codes']['no_pitch_descriptions'])
    identity = {'config_sha256': reg['config_sha256'], 'registration_chain': reg['chain'], 'git': git_state(),
                'environment': environment(), 'decisions': decisions, 'g0_bundle_file_sha256': ident['g0_bundle']['file_sha256'],
                'runner_sources': {rel: hash_file(REPO / rel) for rel in SOURCES}}
    with heavy_lock(root), stage(output, command, identity, spec.get('hang_guard_seconds')) as out:
        inputs = load(command != 'census')
        frame, store = inputs['frame'], inputs['store']
        vocabulary = inputs['prep']['features']['tokens']['type_vocabulary']

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
            runtime = prt.build_runtime(bc_path, bc_sha, support_path, support_sha, out / f'ledger-{split}-{seed}{tag}.jsonl',
                                        components=components, budget=RowBudget(int(budget), seed_count=5), tau=tau,
                                        samples=samples, pitch_cap=pitch_cap, seed=seed, evaluation_seed=evaluation_seed,
                                        expected_identity_sha256=expected, hand_registry=registered_path(reg, 'hands'),
                                        provenance={**provenance, 'style_report': style})
            return runtime, components, provenance

        if command == 'census':
            dump(out / 'census.json', run_census(frame, vocabulary, no_pitch))
        elif command == 'materialize-bc':
            dataset = inputs['parent']['dataset_identity']
            provenance = {'source_ids': {**{s['file']: s['sha256'] for s in dataset['sources']},
                                         'processed_cache': dataset['processed_sha256']},
                          'config_sha256': reg['config_sha256'], 'code_commit': identity['git']['commit'],
                          'data_version': f"processed {dataset['processed_sha256'][:12]}; regular R; assign_fold(2025)"}
            files, paths = inputs['files'], inputs['paths']
            plan_bc = config['train_bc_plan']
            run_materialize(frame, store, out, provenance=provenance,
                            bind_inputs=lambda art, rows: pid.bind_policy_inputs(
                                inputs['bundle_path'], inputs['bundle_sha'], paths, bc_artifact=art, context_rows=rows,
                                aux_classes=ident['classes']['aux']),
                            train_keys=pinned_parquet(paths['p4_train_keys'], files['p4_train_keys']['sha256']),
                            keys_record=inputs['prep']['samples']['train'], gates=spec['gates'],
                            bc_parameters=plan_bc['bc_parameters'], no_pitch=no_pitch,
                            volume_quantiles=config['subgroups']['volume_quantiles'])
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
                                    atol_frequency=spec['atol_frequency_raw'])
            dump(out / 'probe.json', report)
            _require(report['pass'], 'S2 connection probe failed; the stage is not sealed')
        else:
            _require(registered_json(reg, 'bind_probe')['pass'] is True, 'S2 probe record must pass')
            census = registered_json(reg, 'census')['census']
            if command == 'v5-denominators':
                bc_path, bc_sha = registered_path(reg, 'bc')
                support_path, support_sha = registered_path(reg, 'support')
                runtime = prt.build_runtime(bc_path, bc_sha, support_path, support_sha, out / 'ledger.jsonl',
                                            hand_registry=registered_path(reg, 'hands'))
                blocks = preq.pa_blocks(frame, np.flatnonzero(frame.split.eq('dev').to_numpy()))
                dump(out / 'v5.json', run_v5(runtime, store, blocks, Deadline(), no_pitch=no_pitch,
                                             census_split=census['dev']))
            elif command == 'profile':
                blocks = preq.pa_blocks(frame, np.flatnonzero(frame.split.eq('temperature').to_numpy()))
                salt = role_seed(config, 'profile_salt')
                blocks = sorted(blocks, key=lambda b: canonical_hash([salt, b[0]]))[:spec['starts']]
                setting = spec['probe_setting']
                runtime, _, provenance = candidate(blocks, 'temperature', tau=stages['S3b_tau_select']['tau_grid'][0],
                                                   samples=setting['samples'], pitch_cap=setting['pitch_cap'],
                                                   seed=config['seeds']['planning_main'], budget=spec['row_budget'])
                deadline = Deadline()
                submit_pas(runtime, store, blocks, deadline, no_pitch=no_pitch)
                runtime.verify_components()
                profile = {'pas': len(blocks), 'decisions': runtime.summary()['requests'], 'seconds': deadline.elapsed(),
                           'conditional_rows': runtime.improvement.simulator.budget.conditional_rows,
                           'samples': setting['samples'], 'pitch_cap': setting['pitch_cap'],
                           'tau_placeholder_not_selected': stages['S3b_tau_select']['tau_grid'][0],
                           'quality_values_read': False, 'provenance': provenance}
                if spec.get('candidates') and spec.get('selection_row_budget', {}).get('rows'):
                    profile['selection'] = ptau.select_search_settings(profile, spec['candidates'],
                                                                       spec['selection_row_budget'])
                dump(out / 'profile.json', profile)
            else:
                profile = registered_json(reg, 'profile')
                setting = profile.get('selection')
                _require(setting is not None, 'registered S3 profile selection (D-9a) required')
                if command == 'tau-select':
                    _require(not census['blend']['codes_outside_vocabulary'], 'June codes outside the vocabulary (R3b)')
                    games = select_games(frame, 'blend', spec['n_games'], role_seed(config, 'selection_salt'))
                    blocks = game_blocks(frame, games)
                    runtime, components, provenance = candidate(
                        blocks, 'blend', tau=spec['tau_grid'][0], samples=setting['samples'],
                        pitch_cap=setting['pitch_cap'], seed=config['seeds']['planning_main'], budget=spec['row_budget'])
                    facts = submit_pas(runtime, store, blocks, Deadline(), no_pitch=no_pitch)
                    for pa_id, positions in blocks:
                        facts[pa_id]['terminal_marker'] = bool(pd.notna(frame.events.iloc[positions[-1]]))
                    table = ptau.tau_table(runtime.ledger.decisions(), facts, spec['tau_grid'], spec['thresholds'])
                    runtime.verify_components()
                    final = None
                    if table['selected_tau'] is not None:
                        _, support_identity = load_support_table(*registered_path(reg, 'support'),
                                                                 load_train_bc(*registered_path(reg, 'bc')))
                        _, hands_identity = load_hand_registry(*registered_path(reg, 'hands'),
                                                               load_train_bc(*registered_path(reg, 'bc')))
                        s6 = stages['S6_V4']
                        evaluation = role_seed(config, 'evaluation') if s6['dr_q_source'] == 'evaluation_seed' else None
                        final = prt.candidate_identity(components, support_identity, hands_identity,
                                                       {'tau': table['selected_tau'], 'samples': setting['samples'],
                                                        'pitch_cap': setting['pitch_cap'],
                                                        'seed': config['seeds']['planning_main']}, evaluation)['sha256']
                    dump(out / 'tau_freeze.json', {**table, 'games': games, 'ledger_head_sha256':
                         runtime.summary()['ledger_head_sha256'], 'config_sha256': reg['config_sha256'],
                         'tau_code_sha256': hash_file(REPO / 'experiments/pitchmdp/pitchmdp/policy_tau.py'),
                         'final_identity_sha256': final, 'outcomes_read': False, 'provenance': provenance})
                else:
                    freeze = registered_json(reg, 'tau_freeze')
                    _require(freeze['status'] == 'SELECTED', f'no registered tau: {freeze["status"]}')
                    tau = freeze['selected_tau']
                    if command == 'v2-world':
                        dump(out / 'v2.json', run_v2_stage(reg, spec, frame, store, candidate, tau, setting, no_pitch))
                    else:  # dr-evaluate (S6/V4)
                        _require(registered_json(reg, 'v2').get('accept') is True, 'V2 acceptance is an S6 prerequisite')
                        games = select_games(frame, 'dev', spec['n_games'], role_seed(config, 'selection_salt'))
                        blocks = game_blocks(frame, games)
                        evaluation = role_seed(config, 'evaluation') if spec['dr_q_source'] == 'evaluation_seed' else None
                        runtime, components, _ = candidate(
                            blocks, 'dev', tau=tau, samples=setting['samples'], pitch_cap=setting['pitch_cap'],
                            seed=config['seeds']['planning_main'], budget=spec['row_budget'], evaluation_seed=evaluation,
                            expected=reg['expected_identity_sha256'])
                        materialized = registered_json(reg, 'materialize')
                        boot = plan['bootstrap']
                        result, rows = run_dr(runtime, store, components, blocks, Deadline(), no_pitch=no_pitch,
                                              bootstrap={**boot, 'seed': role_seed(config, 'bootstrap')},
                                              ess_gate=config['ess_gate']['thresholds'],
                                              sensitivities=config['sensitivity']['same_ledger'],
                                              strata=strata_function(frame, materialized['volume_edges']))
                        dump(out / 'dr.json', {**result, 'games': games})
                        pd.DataFrame([{k: v for k, v in r.items() if k != 'strata'} for r in rows]).to_parquet(
                            out / 'pa_values.parquet', index=False)


def run_v2_stage(reg, spec, frame, store, candidate, tau, setting, no_pitch):
    """S5: V2 (generating law = pi_b_hat) and V3 (frequency, tempered BC) over registered DEV starts,
    repeated for the registered planning seeds (M-11); one fresh ledger per (seed, law). Acceptance
    needs every V2 seed to accept."""
    config = reg['config']
    blocks = preq.pa_blocks(frame, np.flatnonzero(frame.split.eq('dev').to_numpy()))
    salt = role_seed(config, 'selection_salt')
    blocks = sorted(blocks, key=lambda b: canonical_hash([salt, 'v2', b[0]]))
    laws = [('V2', 'pi_b_hat', lambda bc: (lambda s: (bc.actions, bc.probabilities(s)))),
            ('V3', 'frequency', pss.frequency_law)]
    laws += [('V3', f'tempered_alpha_{a}', lambda bc, a=a: pss.tempered_law(bc, a)) for a in spec['tempered_alpha_grid']]
    report, accepts = {'runs': []}, []
    for seed in config['seeds']['planning_v2']:
        for check, law_name, make in laws:
            runtime, components, _ = candidate(blocks, 'dev', tau=tau, samples=setting['samples'],
                                               pitch_cap=setting['pitch_cap'], seed=seed, budget=spec['row_budget'],
                                               tag=f'-{law_name}')
            starts = []
            for _, positions in blocks:
                requests, _, _ = preq.pa_requests(store, positions, runtime.sha256, no_pitch)
                if requests and runtime.start_population(requests[0].state, requests[0].pitcher_hand)[0]:
                    starts.append((requests[0].state, requests[0].pitcher_hand))
                if len(starts) == spec['starts']:
                    break
            result = pss.run_world(runtime, components, starts, law=make(runtime.bc), law_identity=law_name,
                                   logs_per_start=spec['logs_per_start'], cap=spec['cap'],
                                   truth_rollouts=spec['truth_rollouts'], seed=role_seed(config, 'v2'),
                                   draws=config['le2025_validation_plan']['bootstrap']['draws'],
                                   budget=RowBudget(int(spec['row_budget']), seed_count=5),
                                   tolerance=spec.get('tolerance') if check == 'V2' else None)
            report['runs'].append({'check': check, 'law': law_name, 'planning_seed': seed, **result})
            if check == 'V2':
                accepts.append(result['accept'])
    report['accept'] = None if any(a is None for a in accepts) else all(accepts)
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
    enforce_source(reg['config'])
    raw = args.local_config.read_bytes()
    _require(hash_file_bytes(raw) == reg['config']['le2025_validation_plan']['local_config']['sha256'],
             'local config differs from its registered pin')
    local = json.loads(raw)
    dispatch(args.command, reg, local, args.output.resolve(), lambda store: load_inputs(reg['config'], local, store=store))


if __name__ == '__main__':
    main()
