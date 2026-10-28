"""EXP-OPE-RES-02 rehearsal population (design 6, decision 1 option 2) and the reward-free ledger stage A2b-L
(design 9.2).

Selection (exact procedure): (1) S6 list = ML-POLICY ``select_games(frame, 'dev', 150, 2021242497)`` (the registered
rule, imported unchanged from run_policy_validation; game_pk, game_date and split only) must equal the game_pk set
of the S6-dr-a5 ledger (read through the allowlisted ledger reader; dr.json is never opened); (2) ``filtered`` = the
frame without every row of the S6 games (all splits kept); (3) new list = ``select_games(filtered, 'dev', 150,
759396736)``, which drops split-straddling games itself and allocates the month quotas over DEV minus S6 minus the
straddlers by largest remainder; (4) the two list sha256 and the per-month quota table are pinned at G-REG and
recomputed here (they must match).

Ledger stage: the ML-POLICY dr-evaluate runtime assembly re-implemented outside ``dispatch`` (its ``bind``,
``snapshot_for`` and ``candidate`` closures, runner :1211, :1219, :1230; ``prerequisites`` :1119 is a module-level
function and is called unchanged), the reward-free submission ``submit_pas(..., outcome=None)`` (runner :642, as
``run_v5`` :678-680), the M-10 cand=ref paired identity run (``paired_identity_run`` :736-750, PA-end facts bound
internally, only pass/fail and counts written), the identity checks (header ``pins.candidate.sha256`` = the sealed
candidate, runtime sha256 recorded, bound components = the D127 bind identity, the vectorized B0 recipe reproduces
the ledger logging vectors within 1e-12) and the driver-equivalence comparator. No DR estimate, center, reward or
pa_values file is written.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import ope_registry_a2b as rb
from . import policy_estimator as est
from .matrix_policy_artifacts import is_appledouble
from .policy_artifacts import IntegrityError, _require

N_GAMES = 150  # S6 and the new DEV sample (decision 1 option 2)
DEV_WINDOW = ('2025-07-01', '2025-09-30')
ALL_STAGES_MAX_DATE = '2025-12-31'
FORBIDDEN_OUTPUTS = ('dr.json', 'pa_values.parquet', 'centers.sealed.json')
FORBIDDEN_KEYS = ('reward', 'delta', 'delta_bounds', 'center', 'pa_values', 'candidate_bounds', 'reference_bounds')
ALLOWED_FIELDS = ('kind', 'request_id', 'runtime_sha256', 'pa_id', 'decision_index', 'status', 'pitcher', 'pitcher_hand',
                  'batter_side', 'balls', 'strikes', 'history_actions', 'logged_action', 'context_key')
ALLOWED_RESULT = ('mask', 'logging', 'reference', 'candidate', 'logged_index', 'rho_reference', 'rho_candidate')


def game_list_sha256(games):
    """sha256 of the canonical JSON of the sorted integer game list (the G-REG pin form)."""
    return hashlib.sha256(json.dumps(sorted(int(g) for g in games), separators=(',', ':')).encode()).hexdigest()


def ledger_games(decisions):
    """The game_pk set of a ledger from its decision rows' PA ids (game:at_bat)."""
    return sorted({int(str(row['pa_id']).split(':')[0]) for row in decisions})


def month_quota_table(frame, split, n_games, chosen, straddling):
    """The registered allocation (largest remainder over months) recomputed for the record, and the chosen games
    per month (they must agree)."""
    part = frame.loc[frame.split.eq(split) & ~frame.game_pk.isin(set(straddling)), ['game_pk', 'game_date']]
    part = part.drop_duplicates('game_pk').assign(month=lambda f: pd.to_datetime(f.game_date).dt.strftime('%Y-%m'))
    counts = part.month.value_counts().sort_index()
    quota = counts / counts.sum() * n_games
    alloc = np.floor(quota).astype(int)
    for month in (quota - alloc).sort_values(ascending=False, kind='stable').index[:n_games - alloc.sum()]:
        alloc[month] += 1
    month_of = dict(zip(part.game_pk.astype(int), part.month))
    chosen_by = pd.Series([month_of[int(g)] for g in chosen], dtype=object).value_counts()
    table = {m: {'eligible_games': int(counts[m]), 'quota': int(alloc[m]), 'chosen': int(chosen_by.get(m, 0))}
             for m in counts.index}
    _require(all(v['quota'] == v['chosen'] for v in table.values()), 'chosen games per month differ from the quotas')
    return table


def selection(frame, select_games, straddling_games, *, s6_ledger_games, n_games=N_GAMES, expected=None):
    """The A2b selection (design 6 steps 1-4). ``select_games`` / ``straddling_games``: the registered ML-POLICY
    functions; ``s6_ledger_games``: the S6-dr-a5 ledger's game set; ``expected``: the G-REG pins (selection_s6_game_
    list_sha256, a2b_new_game_list_sha256, month_quota_table) or None at G-REG itself. Reward-free."""
    _require(set(frame.columns) >= {'game_pk', 'game_date', 'split'}, 'selection reads game_pk, game_date and split')
    view = frame[['game_pk', 'game_date', 'split']]
    s6 = sorted(int(g) for g in select_games(view, 'dev', n_games, rb.S6_SELECTION_SALT_RECOMPUTED))
    _require(s6 == sorted(int(g) for g in s6_ledger_games),
             'the recomputed S6 list differs from the S6-dr-a5 ledger games (deterministic integrity)')
    filtered = view.loc[~view.game_pk.isin(s6)]
    new = sorted(int(g) for g in select_games(filtered, 'dev', n_games, rb.A2B_SELECTION_SALT))
    _require(len(new) == n_games and not set(new) & set(s6), 'the new list must be n games disjoint from S6')
    straddle = straddling_games(filtered)
    _require(not set(new) & set(straddle), 'a split-straddling game was chosen')
    record = {'s6_games': s6, 'new_games': new, 'selection_s6_game_list_sha256': game_list_sha256(s6),
              'a2b_new_game_list_sha256': game_list_sha256(new),
              'month_quota_table': month_quota_table(filtered, 'dev', n_games, new, straddle),
              'straddling_games_excluded': [int(g) for g in straddle],
              'salts': {'s6_selection_salt_recomputed': rb.S6_SELECTION_SALT_RECOMPUTED,
                        'a2b_selection_salt': rb.A2B_SELECTION_SALT},
              'reads': ['game_pk', 'game_date', 'split'], 'rewards_read': False}
    if expected is not None:
        for key in ('selection_s6_game_list_sha256', 'a2b_new_game_list_sha256'):
            _require(record[key] == expected[key], f'{key} differs from its G-REG pin (deterministic integrity)')
        _require({m: v['quota'] for m, v in record['month_quota_table'].items()} ==
                 {m: (v['quota'] if isinstance(v, dict) else v) for m, v in expected['month_quota_table'].items()},
                 'the month quota table differs from its G-REG pin (deterministic integrity)')
    return record


def guard_rehearsal_dates(frame, blocks):
    """Every selected row is a DEV row of 2025-07..09 (never 2026, never after 2025-12-31)."""
    rows = frame.iloc[np.concatenate([p for _, p in blocks])]
    dates = pd.to_datetime(rows.game_date).dt.normalize()
    _require(len(rows) and rows.split.eq('dev').all(), 'a non-DEV row reached the rehearsal selection')
    _require(not dates.dt.year.ge(2026).any() and dates.max() <= pd.Timestamp(ALL_STAGES_MAX_DATE),
             'a 2026 row or a row after 2025-12-31 reached the rehearsal (stop)')
    _require(dates.min() >= pd.Timestamp(DEV_WINDOW[0]) and dates.max() <= pd.Timestamp(DEV_WINDOW[1]),
             'a rehearsal row lies outside DEV 2025-07..09')
    return {'first': str(dates.min().date()), 'last': str(dates.max().date()), 'rows': int(len(rows))}


# ---------------------------------------------------------------- the runtime assembly (dispatch closures, re-implemented)
def snapshot_for(rpv, policy, frame, split, blocks):
    """``dispatch.snapshot_for``: the registered window-start style snapshot of ``split``, its as-of date equal to
    the registration, the blocks starting on or after it, and the rolling priors equal to the snapshot on the as-of
    day (D-7)."""
    from .policy_artifacts import load_style_snapshot
    from . import policy_requests as preq
    stages = policy['config']['le2025_validation_plan']['stages']
    path, sha = rpv.registered_path(policy, f'style_{split}')
    snapshot, as_of, content = load_style_snapshot(path, sha)
    _require(as_of == stages['S1b_style_snapshot']['as_of_exclusive'][split], 'style as-of differs from registration')
    first = pd.to_datetime(frame.game_date.iloc[np.concatenate([p for _, p in blocks])]).min()
    _require(first >= pd.Timestamp(as_of), 'evaluation blocks start before the style as-of date')
    day = frame.loc[frame.split.eq(split)]
    check = preq.snapshot_rolling_mismatches(day, snapshot, as_of)
    _require(check['mismatches'] == 0, f'snapshot differs from rolling priors on the as-of date: {check}')
    return snapshot, as_of, {'style_snapshot_sha256': content, 'as_of_exclusive': as_of, 'rolling_check': check}


def bind(rpv, policy, inputs, blocks, bc_artifact, snapshot=None, as_of=None, rows=None):
    """``dispatch.bind``: the candidate components bound to the blocks' context rows (style priors frozen)."""
    from . import policy_identity as pid
    ident = policy['config']['identity_registration']
    contexts, style = rpv.pa_contexts(inputs['store'] if rows is None else rows, blocks, snapshot, as_of)
    components = pid.bind_components(inputs['bundle_path'], inputs['bundle_sha'], inputs['paths'],
                                     bc_artifact=bc_artifact, context_rows=contexts, member_loader=inputs['member_loader'],
                                     classes=ident['classes'], we_contract_sha256=ident['we_contract_sha256'])
    return components, style


def candidate(rpv, policy, inputs, blocks, out, *, pre, ledger_name='ledger-dev-0.jsonl', style=None):
    """``dispatch.candidate`` with the dr-evaluate (S6_V4) settings: the frozen tau, the registered setting
    (samples, pitch cap), planning seed ``seeds.planning_main``, the S6_V4 row budget, the evaluation seed when the
    frozen dr_q source says so, the expected (sealed) identity and runtime v3 positivity records. Returns (runtime,
    components, provenance)."""
    from . import policy_runtime as prt
    from .policy_artifacts import load_train_bc
    from .rollout_policy import RowBudget
    config = policy['config']
    spec = config['le2025_validation_plan']['stages']['S6_V4']
    _require(spec.get('estimator') is not None,
             'the S6_V4 stage must carry its L1-R estimator block (runtime v3 positivity records, as S6-dr-a5)')
    freeze, setting = pre['freeze'], pre['setting']
    expected = freeze['final_identity_sha256']
    _require(policy['expected_identity_sha256'] in (None, expected), 'registered identity differs from the tau record')
    evaluation = rpv.role_seed(config, 'evaluation') if freeze['dr_q']['source'] == 'evaluation_seed' else None
    bc_path, bc_sha = rpv.registered_path(policy, 'bc')
    support_path, support_sha = rpv.registered_path(policy, 'support')
    snapshot, as_of, provenance = snapshot_for(rpv, policy, inputs['frame'], 'dev', blocks)
    components, own = bind(rpv, policy, inputs, blocks, load_train_bc(bc_path, bc_sha), snapshot, as_of)
    _require(components.sha256 == pre['components_sha256'], 'bound components differ from the S2-certified identity (M-10)')
    provenance = {**provenance, 'style_report': own if style is None else style}
    runtime = prt.build_runtime(bc_path, bc_sha, support_path, support_sha, Path(out) / ledger_name,
                                components=components, budget=RowBudget(int(spec['row_budget']), seed_count=5),
                                tau=freeze['selected_tau'], samples=setting['samples'], pitch_cap=setting['pitch_cap'],
                                seed=config['seeds']['planning_main'], evaluation_seed=evaluation,
                                expected_identity_sha256=expected, hand_registry=rpv.registered_path(policy, 'hands'),
                                provenance=provenance, positivity_record=True)
    return runtime, components, provenance


def pair_runtime(rpv, policy, out, provenance):
    """The M-10 cand=ref runtime of the dr-evaluate stage (its own ledger file in the stage directory)."""
    from . import policy_runtime as prt
    return prt.build_reference_pair_runtime(*rpv.registered_path(policy, 'bc'), *rpv.registered_path(policy, 'support'),
                                            Path(out) / 'ledger-dev-paired-identity.jsonl',
                                            hand_registry=rpv.registered_path(policy, 'hands'), provenance=provenance,
                                            positivity_record=True)


# ---------------------------------------------------------------- submission, M-10 and identity checks
def submit(rpv, runtime, store, blocks, deadline, *, no_pitch, expected_pas):
    """The reward-free submission (outcome=None) and the seal counts (every PA handled once, one ledger decision row
    per submitted request)."""
    facts = rpv.submit_pas(runtime, store, blocks, deadline, no_pitch=no_pitch, outcome=None)
    sealed = rpv.seal_counts(facts, runtime, expected_pas)
    for info in facts.values():
        _require(not set(info) & {'reward', 'reason', 'kind', 'end_kind'}, 'the ledger stage bound a PA end (reward)')
    return facts, sealed


def primary_statuses(decisions, facts):
    """The primary run's PA classification (COMPLETE set) with the PA-end facts bound internally (no value)."""
    rows = []
    for pa_id, info, pa_rows in est.by_pa(iter(decisions), facts):
        status = est.classify(sorted(pa_rows, key=lambda d: d['decision_index']), info)[0]
        rows.append({'pa_id': pa_id, 'status': status})
    return rows


def m10_check(rpv, pair, store, blocks, facts, decisions, *, no_pitch, kwargs, outcome):
    """M-10 inside A2b-L: the PA-end facts are bound internally from the frozen C0 WE (``outcome``), the primary
    COMPLETE set from the classification only, the cand=ref run through ``paired_identity_run`` (every complete PA's
    delta exactly 0, complete sets equal). Returns pass/fail and counts only."""
    ended = {pa: {**info, **outcome(info['positions'])} for pa, info in facts.items()}
    rows = primary_statuses(decisions, ended)
    record = rpv.paired_identity_run(pair, store, blocks, ended, rows, no_pitch=no_pitch, kwargs=kwargs)
    return {'check': 'M-10', 'pass': bool(record['pass']), 'complete_pas': int(record['pas']),
            'complete_sets_equal': bool(record['complete_sets_equal']),
            'max_abs_delta_is_zero': bool(record['max_abs_delta'] == 0.0), 'runtime_sha256': record['runtime_sha256']}


def ledger_header(path):
    with Path(path).open('rb') as stream:
        return json.loads(stream.readline())


def identity_checks(header, runtime, components, pre, decisions, vector_bc, *, candidate_sha256):
    """Header pins.candidate.sha256 = the sealed candidate identity; runtime sha256 recorded and carried by every
    row; bound components = the D127 bind identity; the vectorized B0 recipe reproduces every recorded logging
    vector within 1e-12 (A3 R1b analogue)."""
    from .behavior_weights import header_identities, r1b_logging
    runtime_sha, candidate_pin = header_identities(header)
    r1b = r1b_logging(decisions, vector_bc)
    checks = {'header_candidate_identity': candidate_pin == candidate_sha256,
              'header_runtime_sha256': runtime_sha == runtime.sha256,
              'rows_carry_runtime_sha256': all(d.get('runtime_sha256') == runtime.sha256 for d in decisions),
              'components_equal_bind_identity': components.sha256 == pre['components_sha256'],
              'vectorized_b0_reproduces_logging': r1b['rows'] > 0 and r1b['max_abs_diff'] <= rb.INTEGRITY_TOL}
    _require(all(checks.values()), f'A2b-L identity checks failed (deterministic integrity): '
                                   f'{[k for k, v in checks.items() if not v]}')
    return {'checks': checks, 'runtime_sha256': runtime.sha256, 'candidate_sha256': candidate_pin,
            'vectorized_b0_rows': r1b['rows'], 'vectorized_b0_max_abs_diff': r1b['max_abs_diff']}


def allowed_view(row):
    out = {k: row.get(k) for k in ALLOWED_FIELDS}
    result = row.get('result')
    out['result'] = None if not isinstance(result, dict) else {k: result.get(k) for k in ALLOWED_RESULT}
    return out


def compare_allowed_fields(reference_rows, new_rows):
    """Driver equivalence (design 9.2): the A3-allowlisted fields of the re-run decisions equal the reference
    ledger's bit for bit, matched by (pa_id, decision_index). Returns (pass, compared, mismatched keys)."""
    ref = {(r['pa_id'], r['decision_index']): allowed_view(r) for r in reference_rows}
    new = {(r['pa_id'], r['decision_index']): allowed_view(r) for r in new_rows}
    _require(set(new) <= set(ref), 'the re-run produced decisions the reference ledger does not hold')
    mismatched = sorted(f'{k[0]}#{k[1]}' for k in new if json.dumps(new[k], sort_keys=True) !=
                        json.dumps(ref[k], sort_keys=True))
    return {'pass': not mismatched and bool(new), 'compared': len(new), 'mismatched': mismatched}


def reward_free_facts(facts):
    """The facts written by A2b-L: the PA's game, structural problem, E0 membership and positions (no end)."""
    out = {}
    for pa_id, info in facts.items():
        _require(not set(info) & {'reward', 'reason', 'kind', 'end_kind'}, 'a PA end reached the ledger facts')
        out[pa_id] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}
    return out


def e0_count(facts):
    return int(sum(bool(info['in_population']) for info in facts.values()))


def check_no_reward_outputs(directory):
    """Every file the ledger stage wrote is neither a DR / center / pa_values file nor a JSON carrying a reward,
    delta or center key (deterministic integrity). AppleDouble sidecars (``._*`` with the AppleDouble magic, written
    by macOS on an exFAT output root such as T7) are metadata, never read as JSON (G-CODE MAJOR-2; the A2 rule,
    ``matrix_policy_artifacts.is_appledouble`` as ``run_ope_a2`` uses it)."""
    directory = Path(directory)
    for path in sorted(p for p in directory.rglob('*') if p.is_file() and not is_appledouble(p)):
        _require(path.name not in FORBIDDEN_OUTPUTS, f'the ledger stage wrote a forbidden file: {path.name}')
        if path.suffix == '.json':
            def walk(node):
                if isinstance(node, dict):
                    bad = set(node) & set(FORBIDDEN_KEYS)
                    _require(not bad, f'the ledger stage wrote a reward / center key {sorted(bad)} in {path.name}')
                    for v in node.values():
                        walk(v)
                elif isinstance(node, list):
                    for v in node:
                        walk(v)
            walk(json.loads(path.read_text()))
    return True


def require_finite(value, what):
    _require(isinstance(value, (int, float)) and math.isfinite(value), f'{what} must be finite')
    return value


def describe(error):
    return {'class': 'deterministic_integrity' if isinstance(error, IntegrityError) else 'error', 'error': str(error)}
