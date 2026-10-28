"""EXP-OPE-RES-02 (track A2b, A2 OPE follow-up) registry: fixed values, rules, config check, digest, gates.

Design docs/pm/research/track-a/A2b-ope-followup-design.md v2.3 and its registration draft (D192 decided values:
decision 1 option 2 NEW_DEV_150, 2 option 3 B2 diagnostic only, 3 option 1 N' 55 + 10 (+ 5 descriptive), 4 option 1
five B0 candidates, 5 option 1 X = 0.30). Nothing here reads data. The EXP-OPE-RES-01 registry
(``ope_redesign_registry``) is imported unchanged for the inherited rules (cell outcomes, nets, minimiser, run
window, the strict public allowlist machinery and its nine public types); every A2b value lives here.

- seeds: base 20261028, nine roles int(SeedSequence([20261028, r]).generate_state(1)[0]) checked against the
  registered values at import; S-2' world seeds SeedSequence([20261028, 5, k]) for k = 1..76 checked against the
  registered list and disjoint from every EXP-OPE-RES-01 world seed, role value and from each other;
- rows: the 25 phase-1 rows (12 B0 = the A2 pairs, 12 B2, the selection-free lambda = 100 row), the five adoption
  candidates in their total order and the label precedence (label 4 is kept although unreachable under D192 2 3);
- rules: condition (ii) per bootstrap, the gate, the TAB reuse rule, the exact-value cell rule, the integrity
  comparison (absolute 1e-12 for floats, equality for counts, strings and the selected lambda / M incl. inf),
  hang guards, the run window and the S-2' cost gate;
- the typed public allowlist (A2 items 1-9 on the 25 rows + items 10-14), the config check and the digest.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from . import ope_redesign_registry as a2
from .policy_artifacts import _require

EXPERIMENT_ID = 'EXP-OPE-RES-02'
PROTOCOL = 'exp_ope_res_02_v1'  # the registered config's protocol (the draft carries its draft protocol name)
DRAFT, REGISTERED = 'DRAFT_UNREGISTERED', 'REGISTERED'
REPO = Path(__file__).resolve().parents[3]
CONFIG_PATH = 'configs/EXP-OPE-RES-02.json'
BASE_COMMIT = '11eeb6e2bcb10206d1f7eccfb37118184920bc58'
A2_EXPERIMENT_ID = a2.EXPERIMENT_ID

# ---------------------------------------------------------------- fixed values (D186 4-1, D192; never changed here)
B = a2.B  # 1.26e-5 exactly
X = 0.30  # condition (ii): H_point(tuple) <= (1 - X) H_point(baseline) under each bootstrap
PRECISION_SE_MAX = B / 2
INTEGRITY_TOL = 1e-12  # ABSOLUTE difference for floats (G-INT-A2 form); counts, strings, bools and lambda / M: equality
DRAWS = a2.DRAWS
INF = math.inf
LAMBDA_GRID, M_GRID = a2.LAMBDA_GRID, a2.M_GRID
BOOTS = ('game', 'pitcher')
assert B == 1.26e-05 and a2.X == X and a2.INTEGRITY_TOL == INTEGRITY_TOL

# ---------------------------------------------------------------- rows (decision 2 option 3, decision 4 option 1)
R1, R2 = a2.R1, a2.R2
DENOMINATORS = ('B0', 'B2')


def row_id(r1, r2, denominator):
    _require(r1 in R1 and r2 in R2 and denominator in DENOMINATORS, f'unknown row {r1} x {r2} x {denominator}')
    return f'{r1}x{r2}x{denominator}'


BASELINE = 'R1_0xR2_0xB0'  # the S6-dr-a5 configuration; denominator of every half-width ratio
CANDIDATES = ('R1_axR2_0xB0', 'R1_bxR2_0xB0', 'R1_0xR2_1xB0', 'R1_axR2_1xB0', 'R1_bxR2_1xB0')  # = total order
TOTAL_ORDER = CANDIDATES  # denominator B0 < B2, estimator R2_0 < R2_1, reward R1_0 < R1_a < R1_b (judgment value)
TOTAL_ORDER_APPENDED_OPTIONS_1_2 = tuple(f'{p}xB2' for p in ('R1_0xR2_0', 'R1_axR2_0', 'R1_bxR2_0', 'R1_0xR2_1',
                                                             'R1_axR2_1', 'R1_bxR2_1'))  # history: not selected (D192)
B0_ROWS = tuple(f'{p}xB0' for p in a2.PAIRS)  # the 12 EXP-OPE-RES-01 pairs, A2 order
B2_ROWS = tuple(f'{p}xB2' for p in a2.PAIRS)
LAMBDA_FIXED_ROW = 'R1_bxR2_1xB0_lambda100'  # selection-free row (review NOTE N1): lambda fixed at 100
LAMBDA_FIXED_VALUE = 100.0
ROWS = (*B0_ROWS, *B2_ROWS, LAMBDA_FIXED_ROW)  # the 25 phase-1 rows
S6_MODE_ROWS = B0_ROWS  # G-INT-EVAL computes these 12 rows only on S6
S6_REFUSED_ROWS = (*B2_ROWS, LAMBDA_FIXED_ROW)  # the 13 rows never computed or written on S6
DESCRIPTIVE_ROWS = tuple(r for r in ROWS if r != BASELINE and r not in CANDIDATES)  # 6 B0 + 12 B2 + 1 fixed = 19
assert len(ROWS) == 25 and len(S6_REFUSED_ROWS) == 13 and len(DESCRIPTIVE_ROWS) == 19
assert set(CANDIDATES) <= set(B0_ROWS) and BASELINE == f'{a2.BASELINE}xB0'


def a2_pair(row):
    """The EXP-OPE-RES-01 pair name of a B0 row (the A2 record's key), else None."""
    return row[:-3] if row in B0_ROWS else None


def split_row(row):
    _require(row in ROWS, f'unknown A2b row {row!r}')
    if row == LAMBDA_FIXED_ROW:
        return 'R1_b', 'R2_1', 'B0'
    r1, r2, den = row.split('x')
    return r1, r2, den


def denominator_of(row):
    return split_row(row)[2]


# ---------------------------------------------------------------- labels (design 8.2, precedence)
LABELS = ('FAILED_INTEGRITY', 'NOT_EVALUABLE', 'ADOPT_FOR_2027_PROTOCOL:<tuple>', 'DENOMINATOR_NOT_CERTIFIED:<B2 tuples>',
          'BIAS_NOT_CERTIFIED:<tuples>', 'NO_PRECISION_GAIN')
FAILING_FAMILIES = ('R1_SURROGATE', 'TAB', 'COARSE_NET', 'REAL_SHIFT', 'DENOMINATOR')
FLAGS = ('SE_CALIBRATION_WARNING', 'B2_PRECISION_PASS_DIAGNOSTIC', 'r3_stage_failed_integrity')
R3_STATUS_DECIDED = 'DIAGNOSTIC_ONLY:DECIDED'  # D192 decision 2 option 3, pinned at G-REG
R3_STATUSES = ('DIAGNOSTIC_ONLY:DECIDED', 'ADOPTION_ELIGIBLE', 'ADOPTION_ELIGIBLE_TAB_ONLY',
               'DIAGNOSTIC_ONLY:STAND_IN_OUTSIDE_BAND', 'DIAGNOSTIC_ONLY:SWAP_FAILS_C1',
               'DIAGNOSTIC_ONLY:S5X_INFEASIBLE')  # the rule of every status is implemented; only DECIDED is pinnable
ADOPTION_ELIGIBLE_STATUSES = ('ADOPTION_ELIGIBLE', 'ADOPTION_ELIGIBLE_TAB_ONLY')
REHEARSAL_POPULATION = 'NEW_DEV_150'
REHEARSAL_EXPOSURE = 'OPE_FIRST_USE_DEV'


# ---------------------------------------------------------------- seeds (design 9.4)
A2B_BASE = 20261028
ROLES = {'game_bootstrap': 0, 'pitcher_bootstrap': 1, 'placebo': 2, 'tab_world': 3, 's5x_layers': 4,
         's5x_world_seed': 5, 'selection_salt': 6, 'd2_field': 7, 'smoke': 8}
ROLE_VALUES = {'game_bootstrap': 3287529750, 'pitcher_bootstrap': 1876336626, 'placebo': 1525748996,
               'tab_world': 1290308336, 's5x_layers': 840755410, 's5x_world_seed': 1332778283,
               'selection_salt': 759396736, 'd2_field': 24791393, 'smoke': 2536502601}
A2B_SELECTION_SALT = ROLE_VALUES['selection_salt']  # a2b_selection_salt
S6_SELECTION_SALT_RECOMPUTED = 2021242497  # ML-POLICY role_seed(config, 'selection_salt') = SeedSequence([20260929, 3])
ML_POLICY_SEED_BASE = a2.S6_SEED_BASE  # 20260929
WORLD_SEEDS_DERIVED = (  # world_seed(k) = int(SeedSequence([20261028, 5, k]).generate_state(1)[0]), k = 1..76
    3487338189, 1593615457, 3335725428, 3924253116, 2575176357, 4085600646, 2947982827, 2237763523, 1456350973,
    3812701017, 64860427, 719234839, 1615741356, 3086553071, 2046929984, 649090949, 3785133292, 791636507, 270495971,
    2744465018, 3658351151, 2084708118, 2375561717, 2449341978, 1738250986, 3658014187, 915798980, 3437220775,
    1307080277, 1742457729, 1032322144, 2571737169, 342936305, 4140413200, 2814925956, 2195681454, 3495018751,
    561253861, 1640386709, 2397719618, 4153440541, 178519677, 2777550592, 1722611652, 1674992030, 1948350570,
    819331974, 4255992751, 3452317085, 2752262582, 2649237001, 433146583, 1814282960, 553449660, 3819602381,
    398913650, 2230639562, 2301468866, 3758820880, 2369320160, 1213243620, 957157742, 4199708379, 3834255112,
    3203032671, 1437180462, 1457372184, 1097716565, 3743794318, 490312879, 3447519373, 3956802691, 2897520368,
    1094654693, 1946956706, 2219760289)
A2_WORLD_SEEDS = tuple(a2.world_seed(r['k']) for r in a2.S5X_RUNS)  # role_seed(v2) XOR k, k = 1..15


def derive_seed(base, *key):
    return int(np.random.SeedSequence([int(base), *(int(k) for k in key)]).generate_state(1)[0])


def role_seed(role):
    _require(role in ROLES, f'unknown A2b seed role {role!r}')
    return ROLE_VALUES[role]


def role_sequence(role, *key):
    """SeedSequence([20261028, role_index, *key]) (EXP-OPE-RES-01 substream layout with the A2b base)."""
    _require(role in ROLES, f'unknown A2b seed role {role!r}')
    _require(all(type(k) is int and k >= 0 for k in key), 'substream keys are non-negative ints')
    return np.random.SeedSequence([A2B_BASE, ROLES[role], *key])


def world_seed(run_index):
    """S-2' run k: int(SeedSequence([20261028, 5, k]).generate_state(1)[0]) (k = 1..76 derived; 71..76 belonged
    to the non-selected S5-X-R3 and are never a registered run)."""
    _require(type(run_index) is int and 1 <= run_index <= len(WORLD_SEEDS_DERIVED), 'A2b world-seed index is 1..76')
    return WORLD_SEEDS_DERIVED[run_index - 1]


def verify_seeds():
    """Deterministic integrity of every A2b seed value (run at import; numpy SeedSequence)."""
    for role, index in ROLES.items():
        _require(derive_seed(A2B_BASE, index) == ROLE_VALUES[role], f'A2b seed role {role} does not derive')
    _require(derive_seed(ML_POLICY_SEED_BASE, 3) == S6_SELECTION_SALT_RECOMPUTED, 'S6 selection salt does not derive')
    _require(all(derive_seed(A2B_BASE, 5, k) == v for k, v in enumerate(WORLD_SEEDS_DERIVED, start=1)),
             'A2b world seeds do not derive from SeedSequence([20261028, 5, k])')
    others = {*A2_WORLD_SEEDS, a2.V2_ROLE_SEED, *a2.ROLE_VALUES.values(), a2.GAME_BOOTSTRAP_SEED, *ROLE_VALUES.values()}
    _require(len(set(WORLD_SEEDS_DERIVED)) == len(WORLD_SEEDS_DERIVED), 'A2b world seeds collide with each other')
    _require(not set(WORLD_SEEDS_DERIVED) & others, 'an A2b world seed collides with an A2 / role seed')
    _require(len(set(ROLE_VALUES.values())) == len(ROLE_VALUES)
             and not set(ROLE_VALUES.values()) & {*a2.ROLE_VALUES.values(), a2.GAME_BOOTSTRAP_SEED, a2.V2_ROLE_SEED},
             'A2b role seeds collide')
    return True


verify_seeds()

# bootstrap and descriptive seeds by evaluator mode (design 9.4; G-INT-A2 and G-INT-EVAL always use the A2 values)
MODE_S6 = 'S6_GINT_EVAL'
MODE_REHEARSAL = 'REHEARSAL'
EVALUATOR_SEEDS = {
    MODE_S6: {'game': a2.GAME_BOOTSTRAP_SEED, 'pitcher': a2.ROLE_VALUES['pitcher_bootstrap'],
              'placebo': ('EXP-OPE-RES-01', 'placebo'), 'pigeonhole': a2.ROLE_VALUES['pigeonhole']},
    MODE_REHEARSAL: {'game': ROLE_VALUES['game_bootstrap'], 'pitcher': ROLE_VALUES['pitcher_bootstrap'],
                     'placebo': ('EXP-OPE-RES-02', 'placebo'), 'pigeonhole': None},  # no A2b pigeonhole role: NOT_RUN
}


def placebo_stream(owner):
    """The placebo substream factory k -> SeedSequence of the registered owner (A2: [20261027, 1, k]; A2b:
    [20261028, 2, k])."""
    if owner == ('EXP-OPE-RES-01', 'placebo'):
        return lambda k: a2.role_sequence('placebo', k)
    _require(owner == ('EXP-OPE-RES-02', 'placebo'), f'unknown placebo stream {owner!r}')
    return lambda k: role_sequence('placebo', k)


# ---------------------------------------------------------------- S-2' (decision 3 option 1, D192)
S2P_CELLS = ('2*zeta_p', '2*kappa_p', 'descriptive_kappa_prime')
N_PRIME = {'2*zeta_p': 10, '2*kappa_p': 55, 'descriptive_kappa_prime': 5}
RUN_INDEX = {'2*zeta_p': tuple(range(1, 11)), '2*kappa_p': tuple(range(11, 66)),
             'descriptive_kappa_prime': tuple(range(66, 71))}
S2P_CELL_LEVELS = {'2*zeta_p': ('0', '2*zeta_p'), '2*kappa_p': ('2*kappa_p', '0'),
                   'descriptive_kappa_prime': ("2*kappa_p'", '0')}  # (kappa, zeta)
S2P_DECISION_CELLS = ('2*zeta_p', '2*kappa_p')  # the descriptive kappa_p' cell is never a decision input
S2P_COMPONENT_CELLS = {'R1_a': ('2*zeta_p',), 'R1_b': ('2*kappa_p', '2*zeta_p')}  # R1_0: none
KAPPA_PRIME_GROUPS = ('FB', 'FC', 'BR', 'OS')
GINT_S2EQ_RUNS = {'2*zeta_p': (5, 11, 12), '2*kappa_p': (4, 9, 10)}  # recorded EXP-OPE-RES-01 dose-family runs
S2P_WORLD = {'starts_recorded': 147, 'cap': a2.S5X_CAP, 'planning_seed': 0, 'setting': 'X1', 'law': 'V2',
             'repetitions': a2.S5X_REPETITIONS, 'logs_per_start': a2.S5X_LOGS_PER_START,
             'truth_rollouts': a2.S5X_TRUTH_ROLLOUTS}
S2P_PATH = 'a2_order_with_log_phase'  # G-CODE judgment (truth path order): the exact form and the new runs share it
assert all(len(RUN_INDEX[c]) == N_PRIME[c] for c in S2P_CELLS)
assert sorted(k for c in S2P_CELLS for k in RUN_INDEX[c]) == list(range(1, 71))


def registered_run(run_index):
    """(cell, k) of a registered S-2' run index (1..70)."""
    for cell, ks in RUN_INDEX.items():
        if run_index in ks:
            return cell
    _require(False, f'run index {run_index!r} is not a registered S-2\' run (1..70; 71..76 not selected)')


def s2p_cells_for(tuples):
    """The S-2' cells the gate opens for the gated tuples (R1_a: 2 zeta_p; R1_b: 2 kappa_p, 2 zeta_p and the
    descriptive kappa_p' cell)."""
    cells = set()
    for t in tuples:
        r1 = split_row(t)[0]
        cells |= set(S2P_COMPONENT_CELLS.get(r1, ()))
        if r1 == 'R1_b':
            cells.add('descriptive_kappa_prime')
    return tuple(c for c in S2P_CELLS if c in cells)


# ---------------------------------------------------------------- TAB reuse rule (design 5.1 / 5.7; judgment value)
TAB_DATASET_PAS = a2.TAB_DATASET_PAS  # 10,131
TAB_REUSE_REL = 0.20
TAB_ACCEPTANCE = a2.TAB_ACCEPTANCE  # T-A1..T-A4, R = 2,000 each
ETA_P = 0.16405664825560962  # sealed A2-S1 eta_p (reused, not re-calibrated)


def tab_reuse(e0_count):
    """Reuse the A2 T-A1..T-A4 results iff the rehearsal E0 count is within +-20% of 10,131."""
    _require(type(e0_count) is int and e0_count > 0, 'the new E0 count is a positive integer')
    reuse = abs(e0_count - TAB_DATASET_PAS) <= TAB_REUSE_REL * TAB_DATASET_PAS
    return {'e0_count': e0_count, 'reference': TAB_DATASET_PAS, 'tolerance_share': TAB_REUSE_REL, 'reuse': bool(reuse),
            'rerun_dataset_pas': None if reuse else e0_count}


# ---------------------------------------------------------------- cell rules
CELL_OUTCOMES = (*a2.CELL_OUTCOMES, 'EXACT_PASS', 'EXACT_FAIL')
PASSING = ('PASS', 'EXACT_PASS')
cell_outcome = a2.cell_outcome  # the EXP-OPE-RES-01 rule (precision first, then U <= bound, L > bound)
net_outcome = a2.net_outcome


def exact_cell_outcome(beta, *, beta_0=None):
    """The exact-value cell rule (new, design 5.1 (b)): a cell decided by exact dynamic programming (every
    replication equal to its own denominator's baseline, or a pure denominator swap) has no Monte-Carlo error and
    is EXACT_PASS iff |beta| <= bound (B, or |beta_0| + B in a relative cell), else EXACT_FAIL."""
    _require(beta is not None and math.isfinite(beta), 'an exact cell needs a finite exact bias')
    bound = B if beta_0 is None else abs(float(beta_0)) + B
    return {'outcome': 'EXACT_PASS' if abs(beta) <= bound else 'EXACT_FAIL', 'b_hat': float(beta), 'se': 0.0,
            'bound': bound, 'rule': 'absolute' if beta_0 is None else 'relative', 'exact': True}


def condition_ii(h_tuple, h_baseline):
    """(ii) per bootstrap: H(tuple) <= 0.70 H(baseline), evaluated directly; a missing or zero half-width is not a
    pass (INVALID / INCONCLUSIVE). Inputs: {boot: half-width or None}."""
    valid = all(h_tuple.get(b) is not None and h_baseline.get(b) is not None
                and math.isfinite(h_tuple[b]) and math.isfinite(h_baseline[b]) for b in BOOTS)
    degenerate = valid and any(h_tuple[b] == 0 or h_baseline[b] == 0 for b in BOOTS)
    per = {b: (bool(h_tuple[b] <= (1 - X) * h_baseline[b]) if valid else None) for b in BOOTS}
    ratio = {b: (h_tuple[b] / h_baseline[b] if valid and h_baseline[b] > 0 else None) for b in BOOTS}
    passed = valid and not degenerate and all(per.values())
    return {'valid': bool(valid), 'degenerate': bool(degenerate), 'per_bootstrap': per, 'ratio': ratio,
            'pass': bool(passed),
            'outcome': 'INVALID' if not valid else 'INCONCLUSIVE' if degenerate else 'PASS' if passed else 'FAIL'}


# ---------------------------------------------------------------- integrity comparison (G-INT-EVAL, G-INT-S2EQ)
def integrity_equal(value, expected):
    """One leaf: floats by ABSOLUTE difference <= 1e-12 (both finite); inf / -inf only by equality (review 004
    note 4: A2's ``_same`` would reject inf - inf); ints, bools, strings, None exactly (no int / float mix-up)."""
    if isinstance(expected, bool) or isinstance(value, bool):
        return type(value) is type(expected) and value == expected
    if isinstance(expected, float) or isinstance(value, float):
        if not (isinstance(value, (int, float)) and isinstance(expected, (int, float))):
            return False
        a, b = float(value), float(expected)
        if math.isfinite(a) and math.isfinite(b):
            return abs(a - b) <= INTEGRITY_TOL
        return a == b  # inf by equality; nan never equal
    return type(value) is type(expected) and value == expected


def leaves(node, path=()):
    """(path tuple, leaf) of a JSON-like document (lists are indexed)."""
    if isinstance(node, dict):
        for key in sorted(node):
            yield from leaves(node[key], (*path, str(key)))
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            yield from leaves(value, (*path, f'[{i}]'))
    else:
        yield path, node


def compare_documents(got, expected):
    """Every leaf of ``expected`` (the record) against ``got``: (compared, not_comparable, mismatched, max |delta|).
    A leaf missing from ``got`` is a mismatch; leaves of ``got`` absent from the record are not comparable."""
    got_leaves = dict(leaves(got))
    exp_leaves = dict(leaves(expected))
    compared, mismatched, max_delta = [], [], 0.0
    for path, value in exp_leaves.items():
        key = '.'.join(path)
        compared.append(key)
        if path not in got_leaves:
            mismatched.append(key)
            continue
        mine = got_leaves[path]
        if not integrity_equal(mine, value):
            mismatched.append(key)
        if isinstance(mine, (int, float)) and isinstance(value, (int, float)) and not isinstance(mine, bool) \
                and not isinstance(value, bool) and math.isfinite(float(mine)) and math.isfinite(float(value)):
            max_delta = max(max_delta, abs(float(mine) - float(value)))
    not_comparable = sorted('.'.join(p) for p in set(got_leaves) - set(exp_leaves))
    return sorted(compared), not_comparable, sorted(mismatched), max_delta


# ---------------------------------------------------------------- stages, guards, window, cost gates
STAGES = ('A2b-SM', 'A2b-S0', 'A2b-L', 'A2b-S3', 'A2b-G', 'A2b-S2a', 'A2b-S1', 'A2b-S4', 'A2b-UB')
CONFIG_STAGES = ('A2b-SM', 'A2b-S0', 'A2b-L', 'A2b-S3', 'A2b-G', 'A2b-S2a', 'A2b-S1', 'A2b-S2b', 'A2b-S4', 'A2b-UB')
NOT_APPLICABLE_STAGES = ('A2b-S2b',)  # decision 2 option 1 not selected (D192)
HANG_GUARDS = {'A2b-SM': 7200, 'A2b-S0': 7200, 'A2b-L': 14400, 'A2b-S3': 3600, 'A2b-G': 600, 'A2b-S2a_per_run': 1800,
               'A2b-S1': 7200, 'A2b-S2b_per_run': 1800, 'A2b-S4': 3600, 'A2b-UB': 3600}
GUARD_OF = {'A2b-SM': 'A2b-SM', 'A2b-S0': 'A2b-S0', 'A2b-L': 'A2b-L', 'A2b-S3': 'A2b-S3', 'A2b-G': 'A2b-G',
            'A2b-S2a': 'A2b-S2a_per_run', 'A2b-S1': 'A2b-S1', 'A2b-S4': 'A2b-S4', 'A2b-UB': 'A2b-UB'}
UPSTREAM = {'A2b-SM': (), 'A2b-S0': ('A2b-SM',), 'A2b-L': ('A2b-S0',), 'A2b-S3': ('A2b-L',), 'A2b-G': ('A2b-S3',),
            'A2b-S2a': ('A2b-G',), 'A2b-S1': ('A2b-G',), 'A2b-S4': ('A2b-G',), 'A2b-UB': ('A2b-S4',)}
RSS_CAP_GIB, FREE_MEMORY_GIB = 8, 4
KST, PROTECTION_WINDOW_KST = a2.KST, a2.PROTECTION_WINDOW_KST
stage_window_ok = a2.stage_window_ok  # whole guard strictly before 06:30 KST and never inside 06:30-13:30 KST
S5X_RUN_SECONDS = 433.9  # the registered projection unit (A2-S2 6,508.7 s / 15 runs, rounded as registered)
S2P_PROJECTION_SECONDS = {cell: N_PRIME[cell] * S5X_RUN_SECONDS for cell in S2P_CELLS}  # 4,339 / 23,864.5 / 2,169.5


def guard_seconds(stage_id):
    _require(stage_id in GUARD_OF, f'unknown A2b stage {stage_id!r}')
    return HANG_GUARDS[GUARD_OF[stage_id]]


def window_ok(start, stage_id):
    """The run window of one command of ``stage_id`` (an S-2' run uses the per-run guard)."""
    return stage_window_ok(start, guard_seconds(stage_id))


def s2p_cost_gate(cell, first_run_wall_seconds):
    """Cost gate (design 9.6): if the first run's wall time x the cell's N' exceeds 2 x the cell projection
    (N' x 433.9 s), stop before the next run (decision needed); N' never changes."""
    _require(cell in S2P_CELLS, f'unknown S-2\' cell {cell!r}')
    _require(isinstance(first_run_wall_seconds, (int, float)) and first_run_wall_seconds > 0, 'positive wall time')
    projected = first_run_wall_seconds * N_PRIME[cell]
    limit = 2 * S2P_PROJECTION_SECONDS[cell]
    return {'cell': cell, 'first_run_wall_seconds': float(first_run_wall_seconds), 'n_prime': N_PRIME[cell],
            'projected_seconds': projected, 'registered_projection_seconds': S2P_PROJECTION_SECONDS[cell],
            'limit_seconds': limit, 'continue': bool(projected <= limit),
            'action': None if projected <= limit else 'stop before the next run (decision needed); N\' unchanged'}


def smoke_gate(projected_seconds, stage_id):
    """G-SMOKE: a stage whose smoke-projected cost exceeds its hang guard is not started (decision needed)."""
    return bool(isinstance(projected_seconds, (int, float)) and math.isfinite(projected_seconds)
                and projected_seconds <= guard_seconds(stage_id))


# ---------------------------------------------------------------- strict typed public allowlist (design 9.5)
PUBLIC_TYPES = (*a2.PUBLIC_TYPES, 'flag_or_cost', 'integrity_record')  # A2 items 1-9 types + item 12 + items 13-14
W_, P_, RF_, CS_, WV_, G_, T_, V_, F_ = a2.PUBLIC_TYPES
FC_, IR_ = 'flag_or_cost', 'integrity_record'
Ident = a2.Ident
I_ROW = Ident('row', ROWS)
I_BOOT, I_POLICY, I_LEVEL = a2.I_BOOT, a2.I_POLICY, a2.I_LEVEL
I_QS = Ident('quantile', ('q50', 'q90', 'q99', 'max'))
BOOL, HEX = a2.BOOL, a2.HEX


def _row_schema():
    """A2's schema with the A2 pair identifier replaced by the 25 A2b rows; the B0-only R5 sections keep their
    A2 shape; the header constraints name EXP-OPE-RES-02 / A2b-S3 and the A2b r3 status."""
    out = []
    for pattern, kind, constraint in a2.PUBLIC_SCHEMA:
        if pattern == ('experiment',):
            out.append((pattern, kind, frozenset({EXPERIMENT_ID})))
        elif pattern == ('stage',):
            out.append((pattern, kind, frozenset({'A2b-S3'})))
        elif pattern == ('r3_status',):
            out.append((pattern, kind, frozenset({R3_STATUS_DECIDED})))
        elif pattern == ('S5', a2.I_PAIR, a2.I_BOOT):
            out.append((('S5', I_ROW, I_BOOT), kind, frozenset({'PASS', 'FAIL', 'INCONCLUSIVE', 'INVALID', 'BASELINE'})))
        else:
            out.append((tuple(I_ROW if seg is a2.I_PAIR else seg for seg in pattern), kind, constraint))
    return out


B2_SCHEMA = (  # item 10 (reward-free B2 denominator diagnostics) and item 11 (B2 R5 g)
    (('B2', 'validity', 'pass'), F_, BOOL), (('B2', 'validity', 'decisions_checked'), CS_, None),
    (('B2', 'validity', 'zero_mass_with_positive_pi'), CS_, None), (('B2', 'validity', 'nonfinite_rho'), CS_, None),
    (('B2', 'validity', 'support_differs_from_b0'), CS_, None),
    (('B2', 'E5', 'pass'), F_, BOOL), *((('B2', 'E5', 'checks', f'{p}_{lvl}'), F_, BOOL)
                                         for p in ('candidate', 'reference') for lvl in ('pa', 'game')),
    *((('B2', 'E5', 'thresholds', lvl), CS_, None) for lvl in ('pa', 'game')),
    (('B2', 'ess', I_POLICY, I_LEVEL), RF_, None), (('B2', 'ess_ratio_vs_b0', I_POLICY, I_LEVEL), RF_, None),
    (('B2', 'rho_quantiles', I_POLICY, I_QS), RF_, None), (('B2', 'w_T_quantiles', I_POLICY, I_QS), RF_, None),
    (('B2', 'rho_gt_large_share', I_POLICY), CS_, None), (('B2', 'w_T_gt_large_share', I_POLICY), CS_, None),
    (('B2', 'large_weight_threshold'), RF_, None),
    (('B2', 'kl_b2_vs_b0', 'mean'), RF_, None), (('B2', 'kl_b2_vs_b0', 'finite_only_mean'), RF_, None),
    (('B2', 'kl_b2_vs_b0', 'decisions'), CS_, None), (('B2', 'kl_b2_vs_b0', 'infinite_decisions'), CS_, None),
    *((('B2', 'weight_shift_D', key), RF_, None) for key in ('candidate', 'reference', 'paired')),
    (('B2', 'weight_shift_D', 'paired_below_mei'), F_, BOOL),
    *((('B2', 'R5_g', I_POLICY, key), G_, None) for key in ('mean_rho_logged', 'mean_pa_end_weight',
                                                            'mean_rho_logged_ci95', 'mean_pa_end_weight_ci95')),
    (('B2', 'R5_g', I_POLICY, 'mean_rho_logged_ci_excludes_1'), F_, BOOL),
    (('B2', 'R5_g', I_POLICY, 'mean_pa_end_weight_ci_excludes_1'), F_, BOOL),
    (('B2', 'R5_b', a2.I_VARIANT, I_POLICY, a2.I_SUBSET, I_LEVEL), RF_, None),
    (('B2', 'R5_c', 'weights', I_POLICY, a2.I_QSTAT), RF_, None),
    (('B2', 'sn_normalizer_full_sample_valid'), F_, BOOL),
    (('flags', 'B2_PRECISION_PASS_DIAGNOSTIC'), F_, BOOL),
    (('counts', 'rows'), CS_, None),
)
PUBLIC_SCHEMA = (*_row_schema(), *B2_SCHEMA)


def validate_public(document):
    """Strict allowlist of the A2b-S3 public file (everything else belongs in centers.sealed.json)."""
    for path, value in a2._public_leaves(document):
        if value == {}:
            _require(any(a2._prefix_match(p, path) for p, _, _ in PUBLIC_SCHEMA),
                     f'public key outside the typed allowlist: {".".join(path)}')
            continue
        found = [(kind, c) for p, kind, c in PUBLIC_SCHEMA if a2._schema_match(p, path)]
        _require(found, f'public key outside the typed allowlist: {".".join(path)}')
        kind, constraint = found[0]
        _require(a2.public_value_ok(kind, constraint, value, path[-1]),
                 f'public value outside its type {kind!r}: {".".join(path)}')
    return True


INTEGRITY_RECORD_KEYS = {  # items 13-14: pass/fail, maximum |delta|, tolerance, key lists / run indices only
    'gint_eval': ('experiment', 'check', 'result', 'max_abs_delta', 'tolerance', 'compared_keys', 'not_comparable_keys',
                  'mismatched_keys', 'rows', 'refused_rows', 'reference'),
    'gint_s2eq': ('experiment', 'check', 'result', 'max_abs_delta', 'tolerance', 'run_indices', 'cells', 'paths',
                  'compared_keys', 'mismatched_keys', 'auxiliary_statistical_form', 'reference'),
}
_TOLERANCE = {'float_absolute': INTEGRITY_TOL, 'counts_strings_selected_lambda_M': 'equality (inf by equality)'}


def validate_integrity_record(document, kind):
    """Items 13 and 14: the typed integrity_record (no center, shift, b_hat or other value)."""
    _require(kind in INTEGRITY_RECORD_KEYS, f'unknown integrity record {kind!r}')
    _require(isinstance(document, dict) and set(document) == set(INTEGRITY_RECORD_KEYS[kind]),
             f'{kind}: keys outside the integrity_record type')
    _require(document['experiment'] == EXPERIMENT_ID and document['result'] in ('PASS', 'FAIL'),
             f'{kind}: experiment / result')
    _require(document['tolerance'] == _TOLERANCE, f'{kind}: tolerance must be the registered absolute 1e-12 form')
    m = document['max_abs_delta']
    _require(m is None or (isinstance(m, float) and math.isfinite(m) and m >= 0), f'{kind}: max |delta|')
    for key in ('compared_keys', 'mismatched_keys') + (('not_comparable_keys',) if kind == 'gint_eval' else ()):
        _require(isinstance(document[key], list) and all(isinstance(v, str) for v in document[key]),
                 f'{kind}: {key} must be a list of key names')
    _require(isinstance(document['reference'], dict) and set(document['reference']) <= {'path', 'sha256', 'keys'}
             and all(isinstance(v, (str, list)) for v in document['reference'].values()), f'{kind}: reference')
    if kind == 'gint_eval':
        _require(document['check'] == 'G-INT-EVAL' and document['rows'] == list(S6_MODE_ROWS)
                 and document['refused_rows'] == list(S6_REFUSED_ROWS), 'gint_eval: rows')
    else:
        _require(document['check'] == 'G-INT-S2EQ', 'gint_s2eq: check')
        _require(all(type(k) is int for k in document['run_indices'])
                 and all(c in S2P_DECISION_CELLS for c in document['cells']), 'gint_s2eq: runs / cells')
        _require(isinstance(document['paths'], dict) and set(document['paths']) == {'exact_form', 'statistical_form',
                                                                                    'new_seed_runs'}
                 and all(isinstance(v, str) for v in document['paths'].values()), 'gint_s2eq: paths of each form')
        aux = document['auxiliary_statistical_form']
        _require(isinstance(aux, dict) and set(aux) == {'result', 'detection_limit_B', 'decision_input'}
                 and aux['result'] in ('PASS', 'FAIL') and aux['decision_input'] is False, 'gint_s2eq: auxiliary form')
        for k, comps in aux['detection_limit_B'].items():
            _require(re.fullmatch(r'k[0-9]+', k) is not None and set(comps) <= {'R1_a', 'R1_b'} and all(
                v is None or (isinstance(v, float) and math.isfinite(v) and v >= 0) for v in comps.values()),
                'gint_s2eq: detection limits in B units')
    return True


RUN_PUBLIC_KEYS = ('experiment', 'stage', 'cell', 'k', 'wall_seconds', 'peak_rss_bytes', 'integrity', 'sealed_sha256')


def validate_run_public(document):
    """Item 12: an S-2' run's public line (wall time, RSS, integrity pass/fail and the seal; no b_hat or SE)."""
    _require(isinstance(document, dict) and set(document) == set(RUN_PUBLIC_KEYS), 'S-2\' run public keys')
    _require(document['experiment'] == EXPERIMENT_ID and document['stage'] == 'A2b-S2a' and document['cell'] in S2P_CELLS
             and type(document['k']) is int and document['integrity'] in ('PASS', 'FAIL'), 'S-2\' run public values')
    _require(isinstance(document['wall_seconds'], float) and document['wall_seconds'] >= 0
             and type(document['peak_rss_bytes']) is int and isinstance(document['sealed_sha256'], str)
             and re.fullmatch(r'[0-9a-f]{64}', document['sealed_sha256']) is not None, 'S-2\' run public types')
    return True


# ---------------------------------------------------------------- config digest
NULLED_PATHS = (*a2.NULLED_PATHS[:6], '$.registration.codex_reconfirmation_reference',
                '$.registration.code_review_reference', '$.registration.code_review_phase2_reference',
                '$.pre_run_gates[*].status', '$.stage_output_seals')
assert NULLED_PATHS == ('$.status', '$.registered', '$.preregistration_complete', '$.execution',
                        '$.registration.registered_kst', '$.registration.review_reference',
                        '$.registration.codex_reconfirmation_reference', '$.registration.code_review_reference',
                        '$.registration.code_review_phase2_reference', '$.pre_run_gates[*].status',
                        '$.stage_output_seals')


def canonical_digest_text(config):
    obj = copy.deepcopy(config)
    for path in NULLED_PATHS:
        a2._set_null(obj, path)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def config_digest(config):
    """sha256 of the UTF-8 bytes of json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) plus one trailing
    newline, obj = the config with exactly NULLED_PATHS set to null (the A2 rule plus the two A2b null paths)."""
    return hashlib.sha256(canonical_digest_text(config).encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- protected sources and new files
PROTECTED_SOURCES = dict(a2.PROTECTED_SOURCES)  # D122 closure: byte-identical to the base commit
NEW_FILES = (  # registration_input_pins.new_files_sha256 scope (design 9.1 items 1-7 and the new tests)
    'experiments/pitchmdp/pitchmdp/ope_registry_a2b.py',
    'experiments/pitchmdp/pitchmdp/ope_sealed_a2b.py',
    'experiments/pitchmdp/pitchmdp/ope_decision_a2b.py',
    'experiments/pitchmdp/pitchmdp/ope_denominator_a2b.py',
    'experiments/pitchmdp/pitchmdp/ledger_a2b.py',
    'experiments/pitchmdp/pitchmdp/ope_synthetic_a2b.py',
    'experiments/pitchmdp/scripts/run_ope_a2b.py',
    'experiments/pitchmdp/tests/ope_a2b_fixtures.py',
    'experiments/pitchmdp/tests/test_ope_a2b_registry.py',
    'experiments/pitchmdp/tests/test_ope_a2b_sealed.py',
    'experiments/pitchmdp/tests/test_ope_a2b_synthetic.py',
    'experiments/pitchmdp/tests/test_ope_a2b_decision.py',
    'experiments/pitchmdp/tests/test_ope_a2b_denominator.py',
    'experiments/pitchmdp/tests/isolated_ope_a2b_denominator_lightgbm.py',
    'experiments/pitchmdp/tests/test_ope_a2b_ledger.py',
    'experiments/pitchmdp/tests/test_ope_a2b_runner.py',
)
file_sha256 = a2.file_sha256


def verify_protected_sources(root=REPO):
    for rel, sha in PROTECTED_SOURCES.items():
        _require(file_sha256(Path(root) / rel) == sha, f'protected source changed (D122 closure): {rel}')
    return dict(PROTECTED_SOURCES)


# ---------------------------------------------------------------- config check
_get = a2._get
_HEX64, _HEX40 = re.compile(r'[0-9a-f]{64}'), re.compile(r'[0-9a-f]{40}')
CANDIDATE_POLICY_SHA256 = 'c13cc98994ae9eb08a8f939a583ebe712312594ac25c05314a9ca790bd46f73c'
DRAFT_PIN_KEYS = ('rehearsal_population', 'rehearsal_population_allowed', 'r3_status', 'r3_status_allowed',
                  's2_prime_runs', 's6_dr_a5_ledger_path', 's6_dr_a5_ledger_sha256', 's6_dr_a5_manifest_sha256',
                  's6_runtime_sha256_header', 'a2_s0_outputs_manifest_sha256', 'a2_s2_outputs_manifest_sha256',
                  'a2_digest_manifest_sha256', 'a2_execution_source_commit', 'a3_b2_artifacts_sha256',
                  'a3_freeze_primary_manifest_sha256', 'a3_reader_commit', 'processed_frame_aligned',
                  'x2_world_predictor_identity', 'execution_source_commit', 'new_files_sha256', 'new_files_sha256_scope',
                  'digest_manifest_path', 'smoke_train_subset', 'smoke_tab_replications_per_cell', 'rule',
                  'r3_status_rule', 'selection', 'ml_policy_d127_chain', 'runtime_setting', 'dev_style_snapshot',
                  'g0_bundle_sha256', 'b2_inference_environment')
ADDED_PIN_KEYS = (  # implementation needs (reported for G-REG; the draft has no slot for them)
    'output_root',  # fresh stage directories live directly under it (disjoint from every input)
    'a2_digest_manifest_path',  # locates the sealed A2 stage outputs (A2-S0 s0.json, A2-S1 tab.json, A2-S3 public)
    'a2_results_file',  # {"path", "sha256"}: the committed results/EXP-OPE-RES-01.json (G-INT-A2/EVAL/S2EQ records)
    'a3_freeze_primary_manifest_path',  # the A3 04-freeze-primary manifest.json (B2 artifacts relative to its root)
    'a3_s6_weights_record',  # {"path", "sha256"}: A3 09-s6-weights weights.json (G-INT-R3 reference)
    'policy_registration_chain',  # [{"path", "sha256"}]: ML-POLICY config first, then addenda 1-8 d127
    'processed_frame_identity',  # {"path", "sha256"}: the original processed frame (identity source)
    'train_raw_files_sha256',  # the three approved raw files (TRAIN kappa_p' inputs, S6 / DEV R1_b quality join)
    'a3_weight_settings',  # the A3 s6-weights settings G-INT-R3 reproduces (large, mei, draws, seed, E5)
)
PIN_KEYS = (*DRAFT_PIN_KEYS, *ADDED_PIN_KEYS)
CONFIG_EXPECTATIONS = (  # (json path, registered value)
    ('$.id', EXPERIMENT_ID), ('$.protocol', PROTOCOL), ('$.base_commit', BASE_COMMIT),
    ('$.seeds.a2b_base.value', A2B_BASE),
    ('$.seeds.world_seeds_derived.k_1_to_76', list(WORLD_SEEDS_DERIVED)),
    ('$.seeds.run_index_map_decided_plan.S2_prime_2zeta', [1, 10]),
    ('$.seeds.run_index_map_decided_plan.S2_prime_2kappa', [11, 65]),
    ('$.seeds.run_index_map_decided_plan.descriptive_kappa_prime', [66, 70]),
    ('$.seeds.s6_selection_salt_recomputed.value', S6_SELECTION_SALT_RECOMPUTED),
    ('$.seeds.planning_seeds_S5X', [0, 1, 2]),
    ('$.seeds.bootstrap_seed_by_option.G-INT-A2_always.game', EVALUATOR_SEEDS[MODE_S6]['game']),
    ('$.seeds.bootstrap_seed_by_option.G-INT-A2_always.pitcher', EVALUATOR_SEEDS[MODE_S6]['pitcher']),
    ('$.seeds.bootstrap_seed_by_option.rehearsal_options_2_3.game', EVALUATOR_SEEDS[MODE_REHEARSAL]['game']),
    ('$.seeds.bootstrap_seed_by_option.rehearsal_options_2_3.pitcher', EVALUATOR_SEEDS[MODE_REHEARSAL]['pitcher']),
    ('$.arms.tuples.adoption_candidates', list(CANDIDATES)), ('$.arms.tuples.count_phase_1_rows', len(ROWS)),
    ('$.arms.R4_bootstrap.game.draws', DRAWS), ('$.arms.R4_bootstrap.pitcher.draws', DRAWS),
    ('$.decision_rules.condition_ii_half_width.X.value', X),
    ('$.decision_rules.selection.total_order.value', list(TOTAL_ORDER)),
    ('$.synthetic_bias_check.acceptance.B.value', B),
    ('$.synthetic_bias_check.S2_prime.N_prime_decided.zeta_2x', N_PRIME['2*zeta_p']),
    ('$.synthetic_bias_check.S2_prime.N_prime_decided.kappa_2x', N_PRIME['2*kappa_p']),
    ('$.synthetic_bias_check.S2_prime.N_prime_decided.descriptive_kappa_prime', N_PRIME['descriptive_kappa_prime']),
    ('$.synthetic_bias_check.reuse_from_EXP_OPE_RES_01.eta_p', ETA_P),
    ('$.cost_budget.rss_cap_gib.value', RSS_CAP_GIB), ('$.cost_budget.free_memory_guard_gib.value', FREE_MEMORY_GIB),
    ('$.registration.config_digest.nulled_json_paths', list(NULLED_PATHS)),
    ('$.governing_rules.not_modified_paths', list(PROTECTED_SOURCES)),
    ('$.identities.candidate_policy_sha256', CANDIDATE_POLICY_SHA256),
)


SELF_DIGEST_KEYS = ('config_sha256', 'config_digest', 'config_digest_sha256', 'registered_config_sha256', 'digest')
CONFIG_DIGEST_BLOCK_KEYS = ('in_config_digest_field', 'hash_target', 'nulled_json_paths', 'change_from_EXP_OPE_RES_01',
                            'digest_manifest', 'other_fields', 'why')


def _no_self_digest(config):
    """The config never carries its own digest (A2 rule). Unlike EXP-OPE-RES-01, the A2b config legitimately quotes
    OTHER configs' sha256 under 'config_sha256' (a3_record, identities.a3_run, ml_policy_d127_chain), so the
    forbidden names are refused only where this config's own digest would sit: the top level, $.registration and
    $.registration.config_digest (whose keys are the registered description only)."""
    for node, where in ((config, '$'), (config.get('registration') or {}, '$.registration')):
        bad = [k for k, v in node.items() if k in SELF_DIGEST_KEYS and not (k == 'config_digest' and isinstance(v, dict))]
        _require(not bad, f'in-config digest field forbidden: {where}.{bad[0] if bad else ""}')
    block = _get(config, '$.registration.config_digest')
    _require(isinstance(block, dict) and set(block) <= set(CONFIG_DIGEST_BLOCK_KEYS),
             'the config_digest block must describe the rule only (no digest value)')


def _expect(config, path, value):
    got = _get(config, path)
    same = (got == value and type(got) is type(value)) if not isinstance(value, float) else (
        isinstance(got, (int, float)) and not isinstance(got, bool) and float(got) == value)
    _require(same, f'config deviates from the A2b registry at {path}: {got!r} != {value!r}')


def config_check(config):
    """Reject any deviation from the A2b registry. Returns the config status."""
    _require(isinstance(config, dict), 'config must be a JSON object')
    for path, value in CONFIG_EXPECTATIONS:
        _expect(config, path, value)
    status = config.get('status')
    _require(status in (DRAFT, REGISTERED), f'unknown status {status!r}')
    flags = (config.get('registered'), config.get('preregistration_complete'), (config.get('execution') or {}).get('enabled'))
    _require(all(type(f) is bool for f in flags), 'registered/preregistration_complete/execution.enabled must be booleans')
    _require(flags == ((False,) * 3 if status == DRAFT else (True,) * 3),
             'a draft is unregistered and execution-disabled; REGISTERED needs all three flags true')
    roles = _get(config, '$.seeds.roles')
    _require(set(roles) == set(ROLES), 'A2b seed roles deviate from the registry')
    for role, entry in roles.items():
        _require(entry.get('index') == ROLES[role] and entry.get('value') == ROLE_VALUES[role],
                 f'A2b seed role {role} deviates')
    _require([row.get('label') for row in _get(config, '$.decision_rules.labels_in_precedence')] == list(LABELS),
             'labels or their precedence deviate from the A2b registry')
    _require([s.get('id') for s in _get(config, '$.stages')] == list(CONFIG_STAGES), 'stage list deviates')
    guards = _get(config, '$.cost_budget.hang_guards_seconds')
    _require({k: v.get('value') for k, v in guards.items()} == HANG_GUARDS, 'hang guards deviate from the registry')
    _require(set(_get(config, '$.registration_input_pins')) == set(PIN_KEYS),
             'registration_input_pins keys deviate from the A2b registry')
    _require(set(_get(config, '$.registration.config_digest')) >= {'in_config_digest_field', 'hash_target',
                                                                   'nulled_json_paths'},
             'config_digest block incomplete')
    _no_self_digest(config)
    seals = _get(config, '$.stage_output_seals')
    _require(all(v is None for k, v in seals.items() if k != 'rule'),
             'stage output seals live in the external manifest; they stay null in the config')
    if status == REGISTERED:
        check_pins(config)
    return status


def _hex64(v):
    return isinstance(v, str) and _HEX64.fullmatch(v) is not None


def _absolute(v):
    return isinstance(v, str) and Path(v).is_absolute()


def _file_pin(v):
    return isinstance(v, dict) and set(v) == {'path', 'sha256'} and isinstance(v['path'], str) and _hex64(v['sha256'])


def check_pins(config):
    """Every registration-time pin filled and well formed, and the D192 single values (review 004 note 2: the
    registered code asserts each pinned leaf equals the decided value)."""
    pins = _get(config, '$.registration_input_pins')
    _require(set(pins) == set(PIN_KEYS), 'registration_input_pins keys deviate from the A2b registry')
    _require(pins['rehearsal_population'] == REHEARSAL_POPULATION, 'pin rehearsal_population must be NEW_DEV_150 (D192)')
    _require(pins['r3_status'] == R3_STATUS_DECIDED, 'pin r3_status must be DIAGNOSTIC_ONLY:DECIDED (D192 2 3)')
    _require(pins['s2_prime_runs'] == {'zeta_2x': 10, 'kappa_2x': 55, 'descriptive_kappa_prime': 5},
             'pin s2_prime_runs must be the D192 N\' (10, 55, 5)')
    _require(_absolute(pins['s6_dr_a5_ledger_path']), 'pin s6_dr_a5_ledger_path')
    for key in ('s6_dr_a5_ledger_sha256', 's6_dr_a5_manifest_sha256', 's6_runtime_sha256_header',
                'a2_s0_outputs_manifest_sha256', 'a2_s2_outputs_manifest_sha256', 'a2_digest_manifest_sha256',
                'a3_freeze_primary_manifest_sha256', 'g0_bundle_sha256'):
        _require(_hex64(pins[key]), f'pin {key}')
    for key in ('a2_execution_source_commit', 'a3_reader_commit', 'execution_source_commit'):
        _require(isinstance(pins[key], str) and _HEX40.fullmatch(pins[key]) is not None, f'pin {key} (40-hex)')
    b2 = pins['a3_b2_artifacts_sha256']
    _require(isinstance(b2, dict) and set(b2) == {'02-fit/b2/model.txt', '02-fit/b2/b2.json',
                                                 '03-calibrate/calibrators/B2.json'} and all(map(_hex64, b2.values())),
             'pin a3_b2_artifacts_sha256 (the three frozen B2 files)')
    aligned = pins['processed_frame_aligned']
    _require(isinstance(aligned, dict) and set(aligned) == set(a2.ALIGNED_PIN_KEYS) and _absolute(aligned['path'])
             and _absolute(aligned['identity_manifest_path'])
             and all(_hex64(aligned[k]) for k in ('sha256', 'identity_manifest_sha256', 'source_sha256')),
             'pin processed_frame_aligned = the EXP-OPE-RES-01 form (path, sha256, identity_manifest_path, '
             'identity_manifest_sha256, source_sha256)')
    raw = pins['train_raw_files_sha256']
    _require(isinstance(raw, dict) and set(raw) == set(a2.RAW_ALLOWLIST) and all(map(_hex64, raw.values())),
             'pin train_raw_files_sha256 (the three approved raw files)')
    _require(_file_pin(pins['processed_frame_identity']) and _absolute(pins['processed_frame_identity']['path'])
             and pins['processed_frame_identity']['sha256'] == aligned['source_sha256'],
             'pin processed_frame_identity (its sha256 = processed_frame_aligned.source_sha256)')
    x2 = pins['x2_world_predictor_identity']
    _require(isinstance(x2, dict) and x2.get('kind') == 'g0_single_member' and type(x2.get('seed')) is int,
             'pin x2_world_predictor_identity')
    files = pins['new_files_sha256']
    _require(isinstance(files, dict) and set(files) == set(NEW_FILES) and all(map(_hex64, files.values())),
             'pin new_files_sha256 must cover exactly the new A2b source and test files')
    for key in ('digest_manifest_path', 'output_root', 'a2_digest_manifest_path', 'a3_freeze_primary_manifest_path'):
        _require(_absolute(pins[key]), f'pin {key} (absolute)')
    for key in ('digest_manifest_path', 'output_root'):
        _require(not Path(pins[key]).resolve().is_relative_to((REPO / 'results').resolve()),
                 f'{key} must be outside the repository results/')
        _require(not Path(pins[key]).resolve().is_relative_to(REPO.resolve()), f'{key} must be outside the repository')
    for key in ('a2_results_file', 'a3_s6_weights_record'):
        _require(_file_pin(pins[key]), f'pin {key} = {{"path", "sha256"}}')
    settings = pins['a3_weight_settings']
    _require(isinstance(settings, dict) and set(settings) == {'large_weight_threshold', 'mei', 'bootstrap_draws',
                                                            'bootstrap_seed', 'E5_thresholds'}
             and type(settings['bootstrap_draws']) is int and type(settings['bootstrap_seed']) is int
             and settings['E5_thresholds'] == {'pa': 100, 'game': 30},
             'pin a3_weight_settings (the A3 s6-weights large threshold, MEI, bootstrap draws and seed, E5)')
    chain = pins['policy_registration_chain']
    _require(isinstance(chain, list) and len(chain) == 9 and all(_file_pin(c) for c in chain),
             'pin policy_registration_chain = the ML-POLICY config and its addenda 1-8 d127')
    d127 = pins['ml_policy_d127_chain']
    _require(isinstance(d127, dict) and _hex64(d127.get('config_sha256'))
             and isinstance(d127.get('addenda_1_to_8_d127_sha256'), list) and len(d127['addenda_1_to_8_d127_sha256']) == 8
             and all(map(_hex64, d127['addenda_1_to_8_d127_sha256'])) and d127.get('member_loader_pin') is not None,
             'pin ml_policy_d127_chain')
    _require([c['sha256'] for c in chain] == [d127['config_sha256'], *d127['addenda_1_to_8_d127_sha256']],
             'policy_registration_chain differs from ml_policy_d127_chain')
    selection = pins['selection']
    _require(isinstance(selection, dict) and _hex64(selection.get('selection_s6_game_list_sha256'))
             and _hex64(selection.get('a2b_new_game_list_sha256')) and isinstance(selection.get('month_quota_table'), dict),
             'pin selection (two list sha256 and the month quota table)')
    setting = pins['runtime_setting']
    _require(isinstance(setting, dict) and setting.get('row_budget') == 9000000 and setting.get('planning_seed') == 0
             and setting.get('evaluation_role') == 'evaluation' and all(
                 setting.get(k) is not None for k in ('selected_tau', 'samples', 'pitch_cap')), 'pin runtime_setting')
    style = pins['dev_style_snapshot']
    _require(isinstance(style, dict) and _absolute(style.get('path')) and _hex64(style.get('sha256'))
             and style.get('as_of_exclusive') == '2025-07-01', 'pin dev_style_snapshot')
    env = pins['b2_inference_environment']
    _require(isinstance(env, dict) and set(env) == {'lightgbm', 'pyproject_sha256'} and env['lightgbm'] == '4.6.0'
             and _hex64(env['pyproject_sha256']), 'pin b2_inference_environment (lightgbm 4.6.0, pyproject sha256)')
    subset = pins['smoke_train_subset']
    _require(isinstance(subset, dict) and set(subset) == {'game_dates'} and len(subset['game_dates']) == 2
             and a2.TRAIN_WINDOW[0] <= subset['game_dates'][0] <= subset['game_dates'][1] <= a2.TRAIN_WINDOW[1],
             'pin smoke_train_subset inside TRAIN')
    reps = pins['smoke_tab_replications_per_cell']
    _require(type(reps) is int and 1 <= reps <= a2.TAB_DESCRIPTIVE_R, 'pin smoke_tab_replications_per_cell')
    _require(_hex64(_get(config, '$.identities.exp_ope_res_01.results_file_sha256'))
             and pins['a2_results_file']['sha256'] == _get(config, '$.identities.exp_ope_res_01.results_file_sha256'),
             'pin a2_results_file must equal identities.exp_ope_res_01.results_file_sha256')
    return pins


GATES_PASSED = ('G-REV', 'G-CODE', 'G-REG')
REQUIRED_RECEIPTS = ('G-CODE', 'G-REG')
RECEIPT_VERDICTS = a2.RECEIPT_VERDICTS


def require_executable(config):
    """Fail closed: every real-data command refuses unless REGISTERED, enabled, every pin filled, the registration
    references recorded (incl. the Codex re-confirmation) and G-REV / G-CODE / G-REG passed."""
    status = config_check(config)
    _require(status == REGISTERED, f'{EXPERIMENT_ID} is not REGISTERED and execution-enabled; refusing to run (fail closed)')
    check_pins(config)
    registration = config['registration']
    for key in ('registered_kst', 'review_reference', 'codex_reconfirmation_reference', 'code_review_reference'):
        _require(isinstance(registration.get(key), str) and registration[key], f'registration.{key} is not recorded')
    gates = {g.get('id'): g.get('status') for g in config['pre_run_gates']}
    for gate in GATES_PASSED:
        _require(gates.get(gate) in RECEIPT_VERDICTS, f'pre-run gate {gate} has not passed ({gates.get(gate)!r})')
    return config


def load_config(path):
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    config_check(config)
    return config, hashlib.sha256(raw).hexdigest()
