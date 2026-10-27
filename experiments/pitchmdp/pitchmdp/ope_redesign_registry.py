"""EXP-OPE-RES-01 (track A2, OPE scale redesign) registry: fixed values, config check, digest, gates.

Nothing here reads data. ``config_check`` compares the parsed configs/EXP-OPE-RES-01.json with the
values fixed below (design docs/pm/research/track-a/A2-ope-redesign-design.md and its registration
draft) and raises ``IntegrityError`` on any deviation. ``require_executable`` is the fail-closed
gate of every real-data stage (smoke included): status REGISTERED, registered = true,
preregistration_complete = true, execution enabled and every registration-time input pin filled.
``config_digest`` is the external digest (canonical JSON with the self-referential paths nulled);
the config itself never carries a digest field. ``stage_window_ok`` is the run-window rule.
"""
from __future__ import annotations

import copy
from datetime import datetime, time as dtime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .policy_artifacts import _require

EXPERIMENT_ID = 'EXP-OPE-RES-01'
PROTOCOL = 'exp_ope_res_01_v1'
DRAFT, REGISTERED = 'DRAFT_UNREGISTERED', 'REGISTERED'
REPO = Path(__file__).resolve().parents[3]
CONFIG_PATH = 'configs/EXP-OPE-RES-01.json'
BASE_COMMIT = '11eeb6e2bcb10206d1f7eccfb37118184920bc58'

# ---------------------------------------------------------------- arms, pairs and their order
R1 = ('R1_0', 'R1_a', 'R1_b')
R2 = ('R2_0', 'R2_1', 'R2_2', 'R2_3')
BASELINE = 'R1_0xR2_0'
TOTAL_ORDER = ('R1_axR2_0', 'R1_bxR2_0', 'R1_0xR2_3', 'R1_axR2_3', 'R1_bxR2_3', 'R1_0xR2_2', 'R1_axR2_2',
               'R1_bxR2_2', 'R1_0xR2_1', 'R1_axR2_1', 'R1_bxR2_1')  # design 5.2; the baseline is not a candidate
PAIRS = (BASELINE, *TOTAL_ORDER)  # the 12 original-denominator pairs


def pair_id(r1, r2):
    _require(r1 in R1 and r2 in R2, f'unknown arm {r1} x {r2}')
    return f'{r1}x{r2}'


def split_pair(pair):
    r1, r2 = pair.split('x')
    _require(pair in PAIRS and r1 in R1 and r2 in R2, f'unknown pair {pair!r}')
    return r1, r2


INF = math.inf
LAMBDA_GRID = (INF, 1000.0, 100.0, 30.0, 10.0, 3.0)  # R2_1 DR-shrinkage (inf = current DR)
M_GRID = (INF, 20.0, 10.0, 5.0, 3.0, 2.0)  # R2_2 Switch-DR (inf = current DR)
M_SMOOTH_GRID = (5, 20, 80)  # R1_b hierarchical Dirichlet pseudo-counts

# ---------------------------------------------------------------- R1 surrogate rewards
TERMINALS9 = ('walk', 'strikeout', 'out', 'single', 'double', 'triple', 'home_run', 'hbp', 'double_play')
BIP_CLASSES = ('out', 'single', 'double', 'triple', 'home_run', 'double_play')
HIT_CLASSES = ('single', 'double', 'triple', 'home_run')
EV_EDGES = (60, 70, 80, 85, 90, 95, 98, 101, 104, 107, 110)  # mph, 12 bins
LA_EDGES = (-20, -10, 0, 8, 16, 24, 32, 40, 50)  # degrees, 10 bins
SPRAY_ORIGIN = (125.42, 198.27)  # phi = atan((hc_x - 125.42) / (198.27 - hc_y)), community convention
SPRAY_LIMIT_DEG = 15.0  # pull > +15, oppo < -15, center otherwise, plus missing
N_EV, N_LA, N_SPRAY = len(EV_EDGES) + 1, len(LA_EDGES) + 1, 4
N_X = N_EV * N_LA * N_SPRAY  # 480 quality cells
N_CELLS = N_X * 2  # x 2 double-play eligibility = 960
TRAIN_WINDOW = ('2023-05-15', '2025-04-30')
ALL_STAGES_MAX_DATE = '2025-12-31'
RAW_ALLOWLIST = ('statcast_2023.parquet', 'statcast_2024.parquet', 'statcast_2025.parquet')
TRAIN_EVENT_MAP = {  # TRAIN fit mapping of hit_into_play events (design R1-b); others -> excluded
    'field_out': 'out', 'force_out': 'out', 'fielders_choice_out': 'out', 'sac_fly': 'out', 'sac_bunt': 'out',
    'other_out': 'out', 'grounded_into_double_play': 'double_play', 'double_play': 'double_play',
    'sac_fly_double_play': 'double_play', 'single': 'single', 'double': 'double', 'triple': 'triple',
    'home_run': 'home_run', 'field_error': 'single', 'fielders_choice': 'single', 'catcher_interf': 'single',
    'triple_play': 'double_play'}
TRAIN_EVENT_JUDGMENT = ('field_error', 'fielders_choice', 'catcher_interf', 'triple_play')
PITCH_GROUPS = {'FB': ('FF', 'SI'), 'FC': ('FC',), 'BR': ('SL', 'ST', 'CU', 'KC', 'SV', 'CS'),
                'OS': ('CH', 'FS', 'FO', 'SC'), 'other': ('KN', 'EP', 'FA', 'PO', 'UN')}
S5X_GROUPS = ('FB', 'FC', 'BR', 'OS', 'other')
TAB_GROUPS = ('FB', 'FC', 'BR', 'OS')


def pitch_group(code):
    for group, codes in PITCH_GROUPS.items():
        if code in codes:
            return group
    return None


# ---------------------------------------------------------------- thresholds (fixed before results)
B = 1.26e-05  # exactly the D186 4-1 text; never the unrounded derivation
X = 0.30
PRECISION_SE_MAX = B / 2  # SE <= B/2 before any bound comparison
Z_BOUND = 1.96  # U = |b| + 1.96 SE, L = |b| - 1.96 SE
Z_NET = 2.2414  # S1/S4 97.5% interval (Bonferroni over the two nets)
NET_GAP_MAX = 0.01
NET_SE_MAX = 0.005
TIE_WINDOW = 0.02  # x H_star(baseline)
TIE_REL_TOL = 1e-12  # |v - v_min| <= 1e-12 * max(1, |v_min|)
INTEGRITY_TOL = 1e-12
MDE_MULTIPLIER = 2.801585  # z_0.975 + z_0.80
MEI = 0.001
B_V = 0.002734729542731599
PLACEBOS = 100
PLACEBO_KL = 0.0074
PLACEBO_FLAG = 1.25
REPLICATION_2027_RATIO = 0.85
CI_LEVELS = (0.025, 0.975)

# ---------------------------------------------------------------- fixed rules (pure functions)
def select_minimizer(grid, objectives):
    """Registered minimizer rule (design 5.2): over the finite objective values, v_min = their global
    minimum; tie set = {finite v : |v - v_min| <= 1e-12 * max(1, |v_min|)} (exact equality stays a
    tie, also at v_min = 0); inside the tie set the LARGEST grid value wins (inf counts as largest:
    larger m = stronger pooling, larger lambda / M = toward the current DR). None if all nonfinite."""
    grid, values = list(grid), [float(v) for v in objectives]
    _require(len(grid) == len(values) and grid, 'one objective per grid value')
    finite = [i for i, v in enumerate(values) if math.isfinite(v)]
    if not finite:
        return None
    v_min = min(values[i] for i in finite)
    ties = [i for i in finite if abs(values[i] - v_min) <= TIE_REL_TOL * max(1.0, abs(v_min))]
    return grid[max(ties, key=lambda i: (float(grid[i]), -i))]


def cluster_se_index(values, inverse, n_clusters):
    """``cluster_se`` with integer cluster indices 0..n_clusters-1 (same formula; empty clusters skipped)."""
    v = np.asarray(values, dtype=np.float64)
    if not len(v) or not np.isfinite(v).all():
        return math.nan
    n = np.bincount(inverse, minlength=n_clusters).astype(np.float64)
    t = np.bincount(inverse, weights=v, minlength=n_clusters)
    keep = n > 0
    t, n = t[keep], n[keep]
    g = len(t)
    if g < 2:
        return math.nan
    mean = t.sum() / n.sum()
    return float(math.sqrt(g / (g - 1) * float(np.sum((t - n * mean) ** 2)) / n.sum() ** 2))


def cluster_se(values, clusters):
    """Pinned closed-form cluster SE (no seed): over the clusters g = 1..G holding at least one value,
    T_g = cluster sum, n_g = cluster count, N = sum n_g, mean = sum T_g / N;
    SE_cl^2 = (G / (G - 1)) * sum_g (T_g - n_g * mean)^2 / N^2. NaN when G < 2 or a value is nonfinite."""
    v = np.asarray(values, dtype=np.float64)
    _require(v.ndim == 1 and len(v) == len(clusters), 'one cluster per value')
    if not len(v) or not np.isfinite(v).all():
        return math.nan
    _, inverse = np.unique(np.asarray([str(c) for c in clusters]), return_inverse=True)
    t = np.bincount(inverse, weights=v)
    n = np.bincount(inverse).astype(np.float64)
    g = len(t)
    if g < 2:
        return math.nan
    mean = t.sum() / n.sum()
    return float(math.sqrt(g / (g - 1) * float(np.sum((t - n * mean) ** 2)) / n.sum() ** 2))


def cell_outcome(b_hat, se, *, beta_0=None, valid=True):
    """Acceptance cell outcome, in the registered order: invalid -> INVALID; (1) SE > B/2 ->
    INCONCLUSIVE (never PASS); (2) U = |b| + 1.96 SE <= bound -> PASS; (3) L = |b| - 1.96 SE > bound ->
    FAIL; (4) otherwise INCONCLUSIVE. bound = B, or |beta_0| + B for a V3 (relative) cell."""
    if not valid or b_hat is None or se is None or not (math.isfinite(b_hat) and math.isfinite(se)) or se < 0:
        return {'outcome': 'INVALID', 'b_hat': b_hat, 'se': se, 'U': None, 'L': None, 'bound': None}
    bound = B if beta_0 is None else abs(float(beta_0)) + B
    upper, lower = abs(b_hat) + Z_BOUND * se, abs(b_hat) - Z_BOUND * se
    if se > PRECISION_SE_MAX:
        outcome = 'INCONCLUSIVE'
    elif upper <= bound:
        outcome = 'PASS'
    elif lower > bound:
        outcome = 'FAIL'
    else:
        outcome = 'INCONCLUSIVE'
    return {'outcome': outcome, 'b_hat': float(b_hat), 'se': float(se), 'U': upper, 'L': lower, 'bound': bound,
            'precision_met': bool(se <= PRECISION_SE_MAX), 'rule': 'absolute' if beta_0 is None else 'relative'}


def net_outcome(gaps, ses, *, valid=True):
    """S1/S4 coarse net over independent runs: pooled gap = mean of the run gaps, SE = sqrt(sum SE_s^2) / n;
    PASS iff the 97.5% interval (z = 2.2414) contains 0, |gap| <= 0.01 and SE <= 0.005."""
    gaps, ses = [float(g) for g in gaps], [float(s) for s in ses]
    if not valid or not gaps or len(gaps) != len(ses) or not all(map(math.isfinite, gaps + ses)):
        return {'outcome': 'INVALID', 'gap': None, 'se': None, 'interval': None}
    gap = sum(gaps) / len(gaps)
    se = math.sqrt(sum(s * s for s in ses)) / len(ses)
    interval = [gap - Z_NET * se, gap + Z_NET * se]
    passed = interval[0] <= 0 <= interval[1] and abs(gap) <= NET_GAP_MAX and se <= NET_SE_MAX
    return {'outcome': 'PASS' if passed else 'FAIL', 'gap': gap, 'se': se, 'interval': interval,
            'interval_contains_zero': bool(interval[0] <= 0 <= interval[1]), 'abs_gap_ok': bool(abs(gap) <= NET_GAP_MAX),
            'se_ok': bool(se <= NET_SE_MAX), 'runs': len(gaps)}


def select_pair(h_star, passing, baseline_h_star):
    """Design 5.2: among the pairs passing (i) and (ii), H*_min = the smallest H*; the window is
    H*(p) - H*_min <= 0.02 x H*(baseline) (fixed reference, no chaining); the first pair of the total
    order inside the window wins. None when no pair passes."""
    candidates = [p for p in TOTAL_ORDER if p in passing]
    if not candidates:
        return None
    _require(all(math.isfinite(h_star[p]) for p in candidates) and math.isfinite(baseline_h_star),
             'selection needs finite half-widths')
    h_min = min(h_star[p] for p in candidates)
    window = [p for p in candidates if h_star[p] - h_min <= TIE_WINDOW * baseline_h_star]
    return window[0]


# ---------------------------------------------------------------- bootstraps and validity
GAME_BOOTSTRAP_SEED = 3060201743  # SeedSequence([20260929, 1]): the registered S6 bootstrap role
DRAWS = 10000
MIN_GAMES, MIN_PA_STARTS, INVALID_SHARE_MAX = 30, 50, 0.05
MIN_PITCHERS = 30

# ---------------------------------------------------------------- seeds
S6_SEED_BASE = 20260929
A2_BASE = 20261027
ROLES = {'pitcher_bootstrap': 0, 'placebo': 1, 'tab_world': 2, 's5x_world': 3, 'pigeonhole': 4, 'smoke': 5}
ROLE_VALUES = {'pitcher_bootstrap': 3624346705, 'placebo': 883926100, 'tab_world': 4073359316,
               's5x_world': 2396189424, 'pigeonhole': 3724654704, 'smoke': 4209337365}
V2_ROLE_SEED = 4115122098  # int(SeedSequence([20260929, 2]).generate_state(1)[0]) (run_policy_validation.role_seed v2)
PLANNING_SEEDS_S5X = (0, 1, 2)
TAB_FIXED_COMPONENTS = {'situations': 0, 'archetypes': 1, 'archetype_perturbations': 2, 'eta_epsilon': 3,
                        'planning_noise': 4, 'q_hat_noise': 5}


def derive_seed(base, index):
    return int(np.random.SeedSequence([int(base), int(index)]).generate_state(1)[0])


def role_seed(role):
    _require(role in ROLES, f'unknown A2 seed role {role!r}')
    return derive_seed(A2_BASE, ROLES[role])


def role_sequence(role, *key):
    """SeedSequence([a2_base, role_index, *key]) (draft seeds.substreams.rule)."""
    _require(role in ROLES, f'unknown A2 seed role {role!r}')
    _require(all(type(k) is int and k >= 0 for k in key), 'substream keys are non-negative ints')
    return np.random.SeedSequence([A2_BASE, ROLES[role], *key])


def world_seed(run_index):
    """S5-X run k: role_seed(config, 'v2') XOR k (independent Monte-Carlo stream per run)."""
    _require(type(run_index) is int and 1 <= run_index <= 15, 'S5-X run index is 1..15')
    return V2_ROLE_SEED ^ run_index


# ---------------------------------------------------------------- synthetic matrix
S5X_RUNS = (  # design 4.6, in this order; acceptance runs first (k = run index)
    {'k': 1, 'setting': 'X0', 'law': 'V2', 'planning_seed': 0, 'kappa': '0', 'zeta': '0', 'role': 'S1'},
    {'k': 2, 'setting': 'X0', 'law': 'V2', 'planning_seed': 1, 'kappa': '0', 'zeta': '0', 'role': 'S1'},
    {'k': 3, 'setting': 'X0', 'law': 'V2', 'planning_seed': 2, 'kappa': '0', 'zeta': '0', 'role': 'S1'},
    {'k': 4, 'setting': 'X1', 'law': 'V2', 'planning_seed': 0, 'kappa': '2*kappa_p', 'zeta': '0', 'role': 'S2'},
    {'k': 5, 'setting': 'X1', 'law': 'V2', 'planning_seed': 0, 'kappa': '0', 'zeta': '2*zeta_p', 'role': 'S2'},
    {'k': 6, 'setting': 'X2', 'law': 'V2', 'planning_seed': 0, 'kappa': '0', 'zeta': '0', 'role': 'S4'},
    {'k': 7, 'setting': 'X2', 'law': 'V2', 'planning_seed': 1, 'kappa': '0', 'zeta': '0', 'role': 'S4'},
    {'k': 8, 'setting': 'X2', 'law': 'V2', 'planning_seed': 2, 'kappa': '0', 'zeta': '0', 'role': 'S4'},
    {'k': 9, 'setting': 'X1', 'law': 'V2', 'planning_seed': 0, 'kappa': 'kappa_p', 'zeta': '0', 'role': 'descriptive'},
    {'k': 10, 'setting': 'X1', 'law': 'V2', 'planning_seed': 0, 'kappa': '5*kappa_p', 'zeta': '0', 'role': 'descriptive'},
    {'k': 11, 'setting': 'X1', 'law': 'V2', 'planning_seed': 0, 'kappa': '0', 'zeta': 'zeta_p', 'role': 'descriptive'},
    {'k': 12, 'setting': 'X1', 'law': 'V2', 'planning_seed': 0, 'kappa': '0', 'zeta': '5*zeta_p', 'role': 'descriptive'},
    {'k': 13, 'setting': 'X0', 'law': 'V3', 'planning_seed': 0, 'kappa': '0', 'zeta': '0', 'role': 'descriptive'},
    {'k': 14, 'setting': 'X0', 'law': 'V3', 'planning_seed': 1, 'kappa': '0', 'zeta': '0', 'role': 'descriptive'},
    {'k': 15, 'setting': 'X0', 'law': 'V3', 'planning_seed': 2, 'kappa': '0', 'zeta': '0', 'role': 'descriptive'},
)
S5X_REPETITIONS = 6
S5X_LOGS_PER_START, S5X_TRUTH_ROLLOUTS, S5X_CAP = 6, 12, 12
S5X_STARTS_RECORDED = 147  # S5 registration (unverified count; the run uses every registered E0 start)
S5X_V3_ALPHA = 0.5
TAB_LAWS = ('V2', 'V3')
TAB_ETAS = ('0', 'eta_p/2', 'eta_p', '2*eta_p')
TAB_KAPPA_ZETA = (('0', '0'), ('kappa_p', '0'), ('2*kappa_p', '0'), ('5*kappa_p', '0'), ('0', 'zeta_p'),
                  ('0', '2*zeta_p'), ('0', '5*zeta_p'))
TAB_ACCEPTANCE = (
    {'id': 'T-A1', 'law': 'V2', 'eta': 'eta_p', 'kappa': '0', 'zeta': '0', 'R': 2000, 'rule': 'absolute'},
    {'id': 'T-A2', 'law': 'V3', 'eta': 'eta_p', 'kappa': '0', 'zeta': '0', 'R': 2000, 'rule': 'relative'},
    {'id': 'T-A3', 'law': 'V2', 'eta': 'eta_p', 'kappa': '2*kappa_p', 'zeta': '0', 'R': 2000, 'rule': 'absolute'},
    {'id': 'T-A4', 'law': 'V2', 'eta': 'eta_p', 'kappa': '0', 'zeta': '2*zeta_p', 'R': 2000, 'rule': 'absolute'},
)
TAB_DESCRIPTIVE_R = 300
TAB_DATASET_PAS = 10131
TAB_PSEUDO_GAMES = 150
TAB_SITUATIONS, TAB_ARCHETYPES = 64, 8
TAB_ECE_RECORDED = 0.0087  # G0 top-label ECE10 (experiment-proposals-2026-10-26.md:35)
TAB_ECE_FACTOR = 2.0
TAB_PLANNING_SIGMA = 0.01025  # 0.0145 / sqrt(2) as registered
TAB_TARGET_KL = 0.0074


def tab_cells():
    """All 57 cells in registered order (cell_index = position): T-A1..T-A4, the other 52 axis
    combinations in the order law -> eta -> (kappa, zeta), then the era-drift cell."""
    acceptance = [dict(c, index=i, role='acceptance', drift=False) for i, c in enumerate(TAB_ACCEPTANCE)]
    taken = {(c['law'], c['eta'], c['kappa'], c['zeta']) for c in TAB_ACCEPTANCE}
    cells = list(acceptance)
    for law in TAB_LAWS:
        for eta in TAB_ETAS:
            for kappa, zeta in TAB_KAPPA_ZETA:
                if (law, eta, kappa, zeta) in taken:
                    continue
                cells.append({'id': f'T-D{len(cells) - 3:02d}', 'law': law, 'eta': eta, 'kappa': kappa, 'zeta': zeta,
                              'R': TAB_DESCRIPTIVE_R, 'rule': 'descriptive', 'index': len(cells), 'role': 'descriptive',
                              'drift': False})
    cells.append({'id': 'T-DRIFT', 'law': 'V2', 'eta': 'eta_p', 'kappa': '0', 'zeta': '0', 'R': TAB_DESCRIPTIVE_R,
                  'rule': 'descriptive', 'index': len(cells), 'role': 'descriptive', 'drift': True})
    return cells


def scale(label, unit):
    """'0', 'kappa_p', '2*kappa_p', 'eta_p/2' ... -> number, given the calibrated unit value."""
    if label == '0':
        return 0.0
    match = re.fullmatch(r'(?:(\d+)\*)?[a-z]+_p(?:/(\d+))?', label)
    _require(match is not None, f'unknown registered level {label!r}')
    factor = float(match.group(1) or 1) / float(match.group(2) or 1)
    _require(isinstance(unit, (int, float)) and math.isfinite(unit) and unit >= 0, f'{label}: calibrated unit required')
    return factor * float(unit)


# ---------------------------------------------------------------- stages, guards and the run window
STAGES = ('A2-SM', 'A2-S0', 'A2-S1', 'A2-S2', 'A2-S3', 'A2-S4')
HANG_GUARDS = {'A2-SM': 7200, 'A2-S0': 7200, 'A2-S1': 7200, 'A2-S2': 9667, 'A2-S3': 7616, 'A2-S4': 3600}
UPSTREAM = {'A2-SM': (), 'A2-S0': ('A2-SM',), 'A2-S1': ('A2-S0',), 'A2-S2': ('A2-S0',),
            'A2-S3': ('A2-S0',), 'A2-S4': ('A2-S1', 'A2-S2', 'A2-S3')}
RSS_CAP_GIB, FREE_MEMORY_GIB = 8, 4
KST = timezone(timedelta(hours=9))
PROTECTION_WINDOW_KST = (dtime(6, 30), dtime(13, 30))


def stage_window_ok(start, guard_seconds):
    """A stage may start only if [start, start + hang guard] avoids every 06:30-13:30 KST protection
    window (D186 2-2; every day is treated as a game day, the conservative reading). ``start`` is an
    aware datetime."""
    _require(isinstance(start, datetime) and start.tzinfo is not None, 'aware start time required')
    _require(isinstance(guard_seconds, (int, float)) and guard_seconds > 0, 'positive hang guard required')
    begin = start.astimezone(KST)
    end = begin + timedelta(seconds=float(guard_seconds))
    day = begin.date() - timedelta(days=1)
    while datetime.combine(day, PROTECTION_WINDOW_KST[0], KST) <= end:
        lo = datetime.combine(day, PROTECTION_WINDOW_KST[0], KST)
        hi = datetime.combine(day, PROTECTION_WINDOW_KST[1], KST)
        if begin < hi and lo < end:
            return False
        day += timedelta(days=1)
    return True


# ---------------------------------------------------------------- blinding
PUBLIC_TYPES = ('width_or_se', 'pass_fail_only', 'reward_free_weight_or_policy_diagnostic', 'count_or_share',
                'width_se_or_variance (no center)', 'reward_free_weight_statistic_with_ci (signed values and CI endpoints allowed)',
                'train_only_signed_diagnostic (allowed)', 'variance', 'flag_or_hash')
R3_STATUSES = ('ENABLED_DIAGNOSTIC_ONLY', 'NOT_RUN:R3')
LABELS = ('FAILED_INTEGRITY', 'NOT_EVALUABLE', 'ADOPT_FOR_2027_PROTOCOL:<pair>', 'BIAS_NOT_CERTIFIED:<pairs>',
          'NO_PRECISION_GAIN')

# ---------------------------------------------------------------- config digest and protected sources
NULLED_PATHS = ('$.status', '$.registered', '$.preregistration_complete', '$.execution',
                '$.registration.registered_kst', '$.registration.review_reference',
                '$.registration.code_review_reference', '$.pre_run_gates[*].status', '$.stage_output_seals')
PROTECTED_SOURCES = {  # design 6: never modified (policy identity closure, D122); sha256 at base 11eeb6e2
    'experiments/pitchmdp/pitchmdp/policy_estimator.py': '7b32e0efb36036f4a949c6f6079b21df61b58db716ce88080d968bb15e2614be',
    'experiments/pitchmdp/pitchmdp/policy_runtime.py': '8bf42e69c693133cc55a382e5013cd0b3fae6b6c2e9b511f159b1d8ffe5513b2',
    'experiments/pitchmdp/pitchmdp/policy_semisynthetic.py': '4d5ee6f63627c7f1e6614a5e0212e99b2b778758871466426c9432be8e86f42f',
    'experiments/pitchmdp/pitchmdp/rollout_policy.py': 'b2c5843694d0e73170b2cdde755f373e34b3b923a47e3251262f67a20bb87d50',
    'experiments/pitchmdp/pitchmdp/policy_requests.py': '31af23452643a8d9c1ff2df0c95512fbf51dfb927ded1480856b4a623dc5bca8',
    'experiments/pitchmdp/scripts/run_policy_validation.py': '0c1a9a24c8728193f6e2e284d009e663ebb6386c058b6e0e9361a7b30c585fa5',
}
NEW_FILES = (  # registration_input_pins.new_files_sha256 scope (the config itself excluded)
    'experiments/pitchmdp/pitchmdp/ope_redesign_registry.py',
    'experiments/pitchmdp/pitchmdp/ope_surrogate.py',
    'experiments/pitchmdp/pitchmdp/ope_variants.py',
    'experiments/pitchmdp/pitchmdp/ope_bootstrap.py',
    'experiments/pitchmdp/pitchmdp/ope_diagnostics.py',
    'experiments/pitchmdp/pitchmdp/ope_decision.py',
    'experiments/pitchmdp/pitchmdp/ope_sealed.py',
    'experiments/pitchmdp/pitchmdp/ope_synthetic_a2.py',
    'experiments/pitchmdp/scripts/run_ope_a2.py',
    'experiments/pitchmdp/tests/ope_a2_fixtures.py',
    'experiments/pitchmdp/tests/test_ope_a2_registry.py',
    'experiments/pitchmdp/tests/test_ope_a2_estimators.py',
    'experiments/pitchmdp/tests/test_ope_a2_synthetic.py',
    'experiments/pitchmdp/tests/test_ope_a2_runner.py',
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protected_sources(root=REPO):
    """The sealed estimator/runtime/world files are byte-identical to the base commit (run-time check)."""
    for rel, sha in PROTECTED_SOURCES.items():
        _require(file_sha256(Path(root) / rel) == sha, f'protected source changed (D122 closure): {rel}')
    return dict(PROTECTED_SOURCES)


def _set_null(obj, path):
    parts = path[2:].split('.') if path != '$' else []
    _require(path.startswith('$.') and parts, f'bad nulled path {path}')
    node = obj
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if part.endswith('[*]'):
            key = part[:-3]
            _require(isinstance(node, dict) and isinstance(node.get(key), list), f'nulled path missing: {path}')
            rest = '.'.join(parts[i + 1:])
            _require(rest and '[' not in rest and '.' not in rest, f'unsupported nulled path {path}')
            for item in node[key]:
                _require(isinstance(item, dict) and rest in item, f'nulled path missing: {path}')
                item[rest] = None
            return
        _require(isinstance(node, dict) and part in node, f'nulled path missing: {path}')
        if last:
            node[part] = None
        else:
            node = node[part]


def canonical_digest_text(config):
    obj = copy.deepcopy(config)
    for path in NULLED_PATHS:
        _set_null(obj, path)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def config_digest(config):
    """sha256 of the UTF-8 bytes of json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) plus
    exactly one trailing newline, obj = the config with exactly NULLED_PATHS set to null."""
    return hashlib.sha256(canonical_digest_text(config).encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- config check
def _get(config, path):
    node = config
    for part in path[2:].split('.'):
        _require(isinstance(node, dict) and part in node, f'config field missing: {path}')
        node = node[part]
    return node


def _expect(config, path, value):
    got = _get(config, path)
    same = got == value and type(got) is type(value) if not isinstance(value, float) else (
        isinstance(got, (int, float)) and not isinstance(got, bool) and float(got) == value)
    _require(same, f'config deviates from the registry at {path}: {got!r} != {value!r}')


def _grid(values):
    return [('inf' if v == INF else (int(v) if float(v).is_integer() else v)) for v in values]


_HEX64 = re.compile(r'[0-9a-f]{64}')
_HEX40 = re.compile(r'[0-9a-f]{40}')
IMPLEMENTATION = {  # implementation choices fixed in the config (reviewed at G-CODE; see run_ope_a2 report)
    'numeric_rules': {'bias_bound_B': B, 'X': X, 'precision_se_max_share_of_B': 0.5, 'z_bound': Z_BOUND,
                      'z_nets_s1_s4': Z_NET, 'net_gap_max': NET_GAP_MAX, 'net_gap_se_max': NET_SE_MAX,
                      'tie_window_share_of_baseline_h_star': TIE_WINDOW, 'minimizer_tie_relative_tolerance': TIE_REL_TOL,
                      'integrity_tolerance': INTEGRITY_TOL, 'mde_multiplier': MDE_MULTIPLIER,
                      'placebo_flag_ratio': PLACEBO_FLAG, 'replication_2027_ratio': REPLICATION_2027_RATIO},
    'r1': {'root_level': 'maximum likelihood P(e | dp)', 'dp0_double_play_mapping': 'out',
           'spray_boundaries': 'pull if phi > +15, oppo if phi < -15, center otherwise, missing if hc_x, hc_y or stand missing or phi nonfinite',
           'pull_sign': 'phi negated for stand R, kept for stand L',
           'cv_log_loss': 'total held-out negative log-likelihood over both game_pk-parity folds / held-out rows',
           'missing_ev_or_la': 'R1_a fallback', 'source_of_launch_columns': 'processed frame if present, else approved raw join',
           'raw_join_columns': ['hc_x', 'hc_y', 'bb_type', 'launch_speed', 'launch_angle']},
    'calibration': {'se_g': 'sample SD (ddof 1) / sqrt(n_g) of the cross-fitted residuals',
                    'cross_fit': 'game_pk parity two folds, both directions pooled',
                    'world_residual': 'TRAIN empirical (two-strike, dp, x) distribution of the group',
                    'bisection_relative_tolerance': 1e-10, 'bisection_max_upper': 1000.0,
                    'zeta_tilt': 'logit of the scoring-outcome mass of the C0 out distribution plus zeta * u_prime; relative masses within scoring and non-scoring outcomes kept'},
    'tab_world': {'archetype_perturbation_log_sd': 0.1, 'archetype_min_train_pitches': 500,
                  'archetype_sampling': 'uniform without replacement among TRAIN pitchers with >= 500 four-group pitches',
                  'situation_sampling': 'uniform without replacement among TRAIN PA-start rows',
                  'logging_law': 'pi_b(a | count, arch) proportional to P_league(a | count) * share_arch(a) / share_league(a)',
                  'start_distribution': 'uniform over 64 situations x 8 archetypes, iid per PA',
                  'pseudo_game': 'PA index modulo 150', 'pa_cap': 64,
                  'planner_model': '10-class softmax(log P_W10 + eta * epsilon), class-expected terminal values',
                  'eta_p': 'bisection so that the exact top-label ECE10 (weights: pi_b visitation x pi_b) equals 2 x 0.0087',
                  'tau': 'bisection so that the pi_b-visitation-weighted mean KL(pi_c || pi_b) equals 0.0074',
                  'synthetic_train': 'multinomial counts over (fold, dp, x, e), fold iid with probability 1/2',
                  'drift': 'world class law clip(P_hat + (P_hat_second_half - P_hat_first_half), 0) renormalized; synthetic TRAIN from the undrifted law'},
    's5x': {'gamma_in_play_probability': 'exact 400-draw integral (policy_identity.integrated_predictions)',
            'layer_uniforms_per_pitch': ['quality x', 'class e', 'advancement'],
            'in_play_event': 'the G0 10-class draw lands on an in-play class (same uniform as the S5 world)',
            'x2_world_predictor': 'loader must be named by registration_input_pins.x2_world_predictor_identity'},
    'bootstrap': {'se': 'SD of the valid replicates (ddof 1)', 'percentile': 'numpy.quantile linear',
                  'cluster_order': 'sorted by str (as policy_estimator.game_bootstrap)',
                  'pitcher_cluster': 'first-row pitcher of the PA (facts start_pitcher, else first ledger row)',
                  'pitcher_minimum_pa_starts': MIN_PA_STARTS, 'pigeonhole': 'sealed (descriptive), never public'},
    'placebo': {'count_buckets': ['ahead (strikes > balls)', 'even', 'behind (balls > strikes)'],
                'pair': BASELINE, 'ratio_bootstraps': ['game', 'pitcher'], 'flag_if_any_ratio_above': PLACEBO_FLAG},
    'window': {'treat_every_day_as_game_day': True, 'protection_window_kst': ['06:30', '13:30']},
    'smoke': {'eta': 0.0, 'kappa': 0.0, 'zeta': 0.0, 'fixed_m': 20, 'fixed_lambda': 30.0, 'fixed_M': 5.0},
}


def config_check(config):
    """Reject any deviation from the registry. Returns the config status."""
    _require(isinstance(config, dict), 'config must be a JSON object')
    _expect(config, '$.id', EXPERIMENT_ID)
    _expect(config, '$.protocol', PROTOCOL)
    status = config.get('status')
    _require(status in (DRAFT, REGISTERED), f'unknown status {status!r}')
    flags = (config.get('registered'), config.get('preregistration_complete'), (config.get('execution') or {}).get('enabled'))
    _require(all(type(f) is bool for f in flags), 'registered/preregistration_complete/execution.enabled must be booleans')
    if status == DRAFT:
        _require(flags == (False, False, False), 'a draft is unregistered and execution-disabled')
    else:
        _require(flags == (True, True, True), 'REGISTERED needs registered, preregistration_complete and execution enabled')
    _expect(config, '$.base_commit', BASE_COMMIT)
    _expect(config, '$.synthetic_bias_check.acceptance.bias_bound_B.value', B)
    _expect(config, '$.decision_rules.condition_ii_half_width.X.value', X)
    _expect(config, '$.mde_from_records.method.mei.value', MEI)
    _expect(config, '$.mde_from_records.method.b_v.value', B_V)
    _expect(config, '$.arms.R2_estimator.R2_1_dr_shrinkage.lambda_grid.value', _grid(LAMBDA_GRID))
    _expect(config, '$.arms.R2_estimator.R2_2_switch_dr.M_grid.value', _grid(M_GRID))
    _expect(config, '$.arms.R1_reward.R1_b_bip_quality_expected.shrinkage.m_grid.value', list(M_SMOOTH_GRID))
    _expect(config, '$.arms.R1_reward.R1_b_bip_quality_expected.keys.ev_mph_edges.value', list(EV_EDGES))
    _expect(config, '$.arms.R1_reward.R1_b_bip_quality_expected.keys.la_deg_edges.value', list(LA_EDGES))
    _expect(config, '$.arms.R1_reward.R1_b_bip_quality_expected.keys.cells', N_CELLS)
    _expect(config, '$.arms.R1_reward.R1_b_bip_quality_expected.train_event_mapping.out',
            [k for k, v in TRAIN_EVENT_MAP.items() if v == 'out'])
    _expect(config, '$.arms.pairs.ids_in_total_order',
            ['R1_0xR2_0 (baseline, not a candidate)', *TOTAL_ORDER])
    _expect(config, '$.decision_rules.selection.total_order.value', list(TOTAL_ORDER))
    _expect(config, '$.decision_rules.validity_gates.inherited.minimum_games', MIN_GAMES)
    _expect(config, '$.decision_rules.validity_gates.inherited.minimum_pa_starts', MIN_PA_STARTS)
    _expect(config, '$.decision_rules.validity_gates.inherited.invalid_replicate_share_max', INVALID_SHARE_MAX)
    _expect(config, '$.decision_rules.validity_gates.new.minimum_pitchers_pitcher_bootstrap.value', MIN_PITCHERS)
    _require([row.get('label') for row in _get(config, '$.decision_rules.labels_in_precedence')] == list(LABELS),
             'labels or their precedence deviate from the registry')
    _expect(config, '$.arms.R4_bootstrap.game.draws', DRAWS)
    _expect(config, '$.arms.R4_bootstrap.game.seed', GAME_BOOTSTRAP_SEED)
    _expect(config, '$.arms.R4_bootstrap.pitcher.draws', DRAWS)
    _expect(config, '$.arms.R4_bootstrap.pitcher.seed', ROLE_VALUES['pitcher_bootstrap'])
    _expect(config, '$.arms.R5_diagnostics_before_any_delta.h_placebo_se_check.placebos', PLACEBOS)
    _expect(config, '$.arms.R5_diagnostics_before_any_delta.h_placebo_se_check.flag_threshold.value', PLACEBO_FLAG)
    _expect(config, '$.seeds.game_bootstrap.value', GAME_BOOTSTRAP_SEED)
    _expect(config, '$.seeds.a2_base.value', A2_BASE)
    _expect(config, '$.seeds.planning_seeds_S5X', list(PLANNING_SEEDS_S5X))
    roles = _get(config, '$.seeds.roles')
    _require(set(roles) == set(ROLES), 'seed roles deviate from the registry')
    for role, entry in roles.items():
        _require(entry == {'index': ROLES[role], 'value': ROLE_VALUES[role]}, f'seed role {role} deviates')
        _require(derive_seed(A2_BASE, ROLES[role]) == ROLE_VALUES[role], f'seed role {role} does not derive')
    _require(derive_seed(S6_SEED_BASE, 1) == GAME_BOOTSTRAP_SEED and derive_seed(S6_SEED_BASE, 2) == V2_ROLE_SEED,
             'registered S6 role seeds do not derive')
    streams = _get(config, '$.synthetic_bias_check.synthetic_matrix.S5X_independent_streams')
    _require(streams.get('world_seed_values_derived') == [world_seed(r['k']) for r in S5X_RUNS],
             'S5-X world seeds deviate from role_seed(v2) XOR k')
    runs = _get(config, '$.synthetic_bias_check.synthetic_matrix.S5X_runs_in_order')
    _require([r.get('run_index') for r in runs] == [[1, 2, 3], [4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14, 15]]
             and [r.get('planning_seeds') for r in runs] == [[0, 1, 2], [0], [0, 1, 2], [0], [0, 1, 2]],
             'S5-X run order deviates from the registry')
    _expect(config, '$.synthetic_bias_check.synthetic_matrix.S5X_total_runs', len(S5X_RUNS))
    _expect(config, '$.synthetic_bias_check.S5X_world.repetitions.value', S5X_REPETITIONS)
    _expect(config, '$.synthetic_bias_check.S5X_world.pitch_groups.value', {k: list(v) for k, v in PITCH_GROUPS.items()})
    _expect(config, '$.synthetic_bias_check.TAB_world.actions.value', list(TAB_GROUPS))
    _expect(config, '$.synthetic_bias_check.TAB_world.planning_noise.sigma', TAB_PLANNING_SIGMA)
    _expect(config, '$.synthetic_bias_check.TAB_world.replications.dataset_pas', TAB_DATASET_PAS)
    _expect(config, '$.synthetic_bias_check.TAB_world.replications.pseudo_games', TAB_PSEUDO_GAMES)
    accepted = _get(config, '$.synthetic_bias_check.synthetic_matrix.TAB_acceptance_cells')
    _require([{k: c.get(k) for k in ('id', 'law', 'eta', 'kappa', 'zeta', 'R')} for c in accepted]
             == [{k: c[k] for k in ('id', 'law', 'eta', 'kappa', 'zeta', 'R')} for c in TAB_ACCEPTANCE],
             'TAB acceptance cells deviate from the registry')
    _expect(config, '$.synthetic_bias_check.synthetic_matrix.TAB_descriptive_cells.R', TAB_DESCRIPTIVE_R)
    _expect(config, '$.synthetic_bias_check.synthetic_matrix.TAB_axes.cells', 56)
    _expect(config, '$.synthetic_bias_check.synthetic_matrix.TAB_total_replications.value',
            sum(c['R'] for c in tab_cells()))
    _require([s.get('id') for s in _get(config, '$.stages')] == list(STAGES), 'stage list deviates')
    guards = _get(config, '$.cost_budget.hang_guards_seconds')
    _require({k: v.get('value') for k, v in guards.items()} == HANG_GUARDS, 'hang guards deviate from the registry')
    _expect(config, '$.cost_budget.rss_cap_gib.value', RSS_CAP_GIB)
    _expect(config, '$.cost_budget.free_memory_guard_gib.value', FREE_MEMORY_GIB)
    _require([e.get('public_type') for e in _get(config, '$.blinding.public_before_decision_allowlist')]
             == list(PUBLIC_TYPES), 'public allowlist types deviate from the registry')
    _expect(config, '$.registration.config_digest.nulled_json_paths', list(NULLED_PATHS))
    _require(set(_get(config, '$.registration.config_digest')) == {
        'in_config_digest_field', 'hash_target', 'nulled_json_paths', 'other_fields', 'digest_manifest', 'why'},
        'the config must not carry a digest value')
    _expect(config, '$.governing_rules.not_modified_paths', list(PROTECTED_SOURCES))
    _expect(config, '$.registration_input_pins.r3_status_allowed', list(R3_STATUSES))
    _expect(config, '$.implementation', IMPLEMENTATION)
    _no_digest_fields(config)
    seals = _get(config, '$.stage_output_seals')
    _require(all(v is None for k, v in seals.items() if k not in ('A2-S0_contents', 'rule')),
             'stage output seals live in the external manifest; they stay null in the config')
    if status == REGISTERED:
        check_pins(config)
    return status


def _no_digest_fields(node, path='$'):
    if isinstance(node, dict):
        for key, value in node.items():
            _require(key not in ('config_sha256', 'config_digest_sha256', 'registered_config_sha256'),
                     f'in-config digest field forbidden: {path}.{key}')
            _no_digest_fields(value, f'{path}.{key}')
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _no_digest_fields(value, f'{path}[{i}]')


def _outside_results(path_text, name):
    _require(isinstance(path_text, str) and Path(path_text).is_absolute(), f'{name}: absolute path required')
    path = Path(path_text).resolve()
    _require(not path.is_relative_to((REPO / 'results').resolve()), f'{name} must be outside the repository results/')
    return path


def check_pins(config):
    """Every registration-time input pin filled and well formed (needed before any registered stage)."""
    pins = _get(config, '$.registration_input_pins')
    hex64 = lambda v: isinstance(v, str) and _HEX64.fullmatch(v) is not None
    _require(isinstance(pins.get('s6_dr_a5_ledger_path'), str) and Path(pins['s6_dr_a5_ledger_path']).is_absolute(),
             'pin s6_dr_a5_ledger_path')
    for key in ('s6_dr_a5_ledger_sha256', 's6_dr_a5_manifest_sha256'):
        _require(hex64(pins.get(key)), f'pin {key}')
    raw = pins.get('train_raw_files_sha256')
    _require(isinstance(raw, dict) and set(raw) == set(RAW_ALLOWLIST) and all(hex64(v) for v in raw.values()),
             'pin train_raw_files_sha256 (the three approved raw files)')
    _require(pins.get('processed_frame_identity') is not None, 'pin processed_frame_identity')
    _require(pins.get('x2_world_predictor_identity') is not None, 'pin x2_world_predictor_identity')
    _require(isinstance(pins.get('execution_source_commit'), str) and _HEX40.fullmatch(pins['execution_source_commit']),
             'pin execution_source_commit (40-hex)')
    files = pins.get('new_files_sha256')
    _require(isinstance(files, dict) and set(files) == set(NEW_FILES) and all(hex64(v) for v in files.values()),
             'pin new_files_sha256 must cover exactly the new source and test files')
    _outside_results(pins.get('digest_manifest_path'), 'digest_manifest_path')
    _outside_results(pins.get('output_root'), 'output_root')
    r3 = pins.get('r3_status')
    _require(r3 in R3_STATUSES, 'pin r3_status')
    a3 = (pins.get('a3_behavior_policy_sha256'), pins.get('a3_provenance_manifest_sha256'))
    if r3 == 'NOT_RUN:R3':
        _require(a3 == ('NOT_APPLICABLE', 'NOT_APPLICABLE'), 'NOT_RUN:R3 needs the literal NOT_APPLICABLE a3 pins')
    else:
        _require(all(hex64(v) for v in a3), 'ENABLED_DIAGNOSTIC_ONLY needs the A3 artifact and provenance sha256')
    subset = pins.get('smoke_train_subset')
    _require(isinstance(subset, dict) and set(subset) == {'game_dates'} and len(subset['game_dates']) == 2,
             'pin smoke_train_subset = {"game_dates": [first, last]}')
    first, last = subset['game_dates']
    _require(TRAIN_WINDOW[0] <= first <= last <= TRAIN_WINDOW[1], 'the smoke TRAIN subset must lie inside TRAIN')
    reps = pins.get('smoke_tab_replications_per_cell')
    _require(type(reps) is int and 1 <= reps <= TAB_DESCRIPTIVE_R, 'pin smoke_tab_replications_per_cell')
    return pins


def require_executable(config):
    """Fail closed: every real-data stage (smoke included) refuses unless registered and enabled."""
    status = config_check(config)
    _require(status == REGISTERED and config.get('registered') is True and config.get('preregistration_complete') is True
             and (config.get('execution') or {}).get('enabled') is True,
             'EXP-OPE-RES-01 is not REGISTERED and execution-enabled; refusing to run (fail closed)')
    check_pins(config)
    return config


def load_config(path=REPO / CONFIG_PATH):
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    config_check(config)
    return config, hashlib.sha256(raw).hexdigest()
