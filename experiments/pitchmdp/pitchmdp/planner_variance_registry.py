"""EXP-STAB-02 (track A1, planner variance removal) registered values and fail-closed rules.

The design (docs/pm/research/track-a/A1-planner-variance-design.md) and its registration draft
(A1-planner-variance-registration-draft.json, design review PASS at round 3) fix every value pinned
here: arms, seeds, the decision set, thresholds, row budgets, pass caps and the run window.
``IMPLEMENTATION_CHOICES`` lists the points the design leaves open and this code fixes.

``config_check`` compares a config with these pins and raises ``RegistrationError`` naming every
deviation; it accepts the draft (status DRAFT_UNREGISTERED, execute false).  ``registration_gate``
additionally requires status REGISTERED, execute true and every registration field: every
real-data stage, the smoke included, calls it first and refuses otherwise.  ``verdict`` is the
exhaustive verdict partition; H2 is not one of its inputs.  Nothing here reads data.
"""
from __future__ import annotations

from datetime import datetime, time as clock_time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

EXPERIMENT_ID = 'EXP-STAB-02'
PROTOCOL = 'planner_variance_rao_blackwell_v1'
CONFIG_PATH = 'configs/EXP-STAB-02.yaml'
STATUS_DRAFT, STATUS_REGISTERED = 'DRAFT_UNREGISTERED', 'REGISTERED'
REGISTERED_POLICY_IDENTITY = 'c13cc98994ae9eb08a8f939a583ebe712312594ac25c05314a9ca790bd46f73c'

# Registered planner path: byte-identical to 6ca4d3da (registration draft code.unchanged_sha256).
UNCHANGED_SHA256 = {
    'experiments/pitchmdp/pitchmdp/rollout_policy.py': 'b2c5843694d0e73170b2cdde755f373e34b3b923a47e3251262f67a20bb87d50',
    'experiments/pitchmdp/pitchmdp/planner.py': '46a51faf797cf74dc193cd0cace9fe5f54e5282c00ff56a9b91895cd74419a58',
    'experiments/pitchmdp/pitchmdp/policy_runtime.py': '8bf42e69c693133cc55a382e5013cd0b3fae6b6c2e9b511f159b1d8ffe5513b2',
    'experiments/pitchmdp/pitchmdp/policy_identity.py': 'c1dfec8294116089d8a545edb9c098c5cac35ff6845eeb666cfb9a4ea816a81a',
    'experiments/pitchmdp/pitchmdp/policy_artifacts.py': '76b8fd5ef0ea2b7cf8ae8c7e071821c73b206aad8f09b6ca4f5ea7335d199dd0',
    'experiments/pitchmdp/pitchmdp/policy_requests.py': '31af23452643a8d9c1ff2df0c95512fbf51dfb927ded1480856b4a623dc5bca8',
    'experiments/pitchmdp/pitchmdp/matrix_policy.py': '421f3d13f187a27505485790cbe470cd98f38b4af61ccba36a738ee9e2e6793c',
    'experiments/pitchmdp/pitchmdp/matrix_sharing.py': '30d9b0344bed8f0d8afd9aeb736e5f81e5fb0e98229813423bd2c17669f406b7',
    'experiments/pitchmdp/pitchmdp/matrix_data.py': '439912ef3adb52bfdbe6916b9fde464dd9fc9ef019e1066b2c32e4571b52d29b',
    'experiments/pitchmdp/pitchmdp/data.py': '5fb3a385985aecc9db1a1145b2a2ab0d6d9dd34e7faa8c44ca50f7eb79810608',
    'experiments/pitchmdp/pitchmdp/game.py': '0e365c03c821e249f019b241dced8ac8e004c6ea9f95227c408cb2387389efc6',
    'experiments/pitchmdp/scripts/run_policy_validation.py': '0c1a9a24c8728193f6e2e284d009e663ebb6386c058b6e0e9361a7b30c585fa5',
    'experiments/pitchmdp/scripts/diagnose_policy_seed_ensemble.py': '390945280f56a74e8c225199a357f1178a401642f29eb8cbafe738fbcdff2485',
    'experiments/pitchmdp/scripts/diagnose_policy_predictor_swap.py': 'b4d06fe071f4a0fdefe49f2e44e2fdd136abd9bff92f8f42508772ffb5a484ae',
}
# New code of this experiment; its sha256 is pinned in the config's registration block at registration.
NEW_CODE = ('experiments/pitchmdp/pitchmdp/planner_variance.py',
            'experiments/pitchmdp/pitchmdp/planner_variance_registry.py',
            'experiments/pitchmdp/scripts/diagnose_planner_variance.py')

TAU, SAMPLES, PITCH_CAP, MAIN_SEED = 0.1, 6, 12, 0
PAIR = (0, 5)
REFERENCE_SEED = 100
STAB01_SEEDS = {'A': [0, 1, 2, 3, 4], 'B': [5, 6, 7, 8, 9]}
FIRST_PITCH_CHUNK = 6  # RB first-pitch batches of six samples: chunk 0-5 has the A0-MC6 depth-0 composition

ARMS = {
    'A0-MC6': {'estimator': 'RolloutImprovement', 'samples': 6, 'pitch_cap': 12, 'seeds': [0, 5],
               'seed_0_source': 'runtime', 'games': 'registered', 'pass': 'M', 'mandatory': True,
               'row_budget': {'0': 8000000, '5': 8000000}},
    'RB6': {'estimator': 'RaoBlackwellFirstStep', 'samples': 6, 'pitch_cap': 12, 'seeds': [0, 5],
            'games': 'registered', 'pass': 'M', 'mandatory': True,
            'row_budget': {'0': 18000000, '5': 18000000}},
    'REF-RB6': {'estimator': 'RaoBlackwellFirstStep', 'samples': 6, 'pitch_cap': 12, 'seeds': [100],
                'games': 'registered', 'pass': 'M', 'mandatory': True, 'row_budget': {'100': 18000000}},
    'K5-STAB01': {'estimator': 'none (EXP-STAB-01 per-decision columns joined by request_id)', 'seeds': [],
                  'games': 'registered', 'pass': 'M', 'mandatory': True, 'row_budget': {}},
    'RB12': {'estimator': 'RaoBlackwellFirstStep', 'samples': 12, 'pitch_cap': 12, 'seeds': [0, 5],
             'games': 'registered', 'pass': 'C', 'mandatory': False,
             'row_budget': {'0': 36000000, '5': 36000000},
             'run_condition': 'smoke r_b <= 2.0 and RB6 not STABLE; enters the verdict'},
    'BF48': {'estimator': 'RolloutImprovement', 'samples': 48, 'pitch_cap': 12, 'seeds': [0, 5],
             'games': 'BF48 subsample', 'pass': 'D', 'mandatory': False,
             'row_budget': {'0': 12000000, '5': 12000000},
             'run_condition': 'RB6 not (STABLE and COST_OK) and the smoke projection fits the BF48 cap; '
                              'descriptive only'},
    'FH-DP': {'estimator': 'FrozenHistoryDP', 'seeds': [], 'games': 'FH-DP subsample', 'pass': 'D',
              'mandatory': False, 'row_budget': {},
              'run_condition': 'RB6 not (STABLE and COST_OK) and the 1-game smoke extrapolates 10 games '
                               'to <= 7,200 s; descriptive only'},
}
EXCLUDED_ARMS = ('BF24', 'member-level Q averaging', 'second-pitch Rao-Blackwellization')

GAMES = {'split': 'dev', 'exclude': ['S5_V2_V3', 'S6_V4'], 'n_games': 150, 'salt': 20261020,
         'games_sha256': 'cadb081234e1f02293753b3d744f16c738a64a6eb3075f1a9706f3981edb84d1', 'smoke_games': 3}
SUBSAMPLES = {'salt': 20261027, 'BF48': 50, 'FH-DP': 10, 'fhdp_smoke_games': 1}
CONFIRMATION = {'n_games': 50, 'salt': 20261028, 'exclude': ['S5_V2_V3', 'S6_V4', 'EXP-STAB-01'],
                'arms': ['A0-MC6', '<passing RB arm>'], 'started_only_after': 'STABLE_WITHIN_COST and Song confirmation'}
STAB01 = {'config': 'configs/EXP-STAB-01.yaml',
          'config_sha256': '4b0de55c2916bc6097f3f8f6a9b008192d068bd5bc7c31fbb9496f1b6629982f',
          'decisions_npz': '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/EXP-STAB-01/decisions.npz',
          'decisions_npz_sha256': '8a6d5724bc8d40dcef53c4d743f58e07fda0efefca63625d7f69d28a723a6969',
          'decisions_compared': 37559}
STAB01_COLUMNS = ('request_id', 'game_pk', 'K1_tv', 'K1_top1_same', 'K3_tv', 'K3_top1_same', 'K5_tv', 'K5_top1_same',
                  'K1_max_abs_q_difference', 'K3_max_abs_q_difference', 'K5_max_abs_q_difference')
POLICY_REGISTRATION = {
    'config': 'configs/ML-POLICY-MATERIALIZATION-v1.json',
    'config_sha256': '869b2611c027e36573cd4d9a3e766f02bfda9c76a3485cb5e1870f68c7954469',
    'addenda': [f'configs/ML-POLICY-MATERIALIZATION-v1.addendum-{k}-d127.json' for k in range(1, 9)]}
TAU_FREEZE = {'selected_tau': 0.1, 'setting': {'samples': 6, 'pitch_cap': 12}, 'planning_main_seed': 0}
G0_BUNDLE = {'path': 'configs/G0-RESEARCH-FROZEN-v1.json',
             'file_sha256': 'c839dbf710c02ca5a1baafcc168e6ca3eaa6d80e269a99f71f08d7d537ec8062'}
COMPONENTS_SHA256 = '5683fb61f7c039cfad8c21e5e1dc48231ec487a7c0a4aab1e72ff3b8d022d074'
DATA = {'processed_sha256': '9f65d1dbf28374d6879dbf0c19c5c98834a2e1e95a6218c83e213c955e8ede27',
        'style_snapshot_sha256': '943e2a10af514c0a7481da0c4ffe41d6785bea56ceea5a4aab132ed1e1db727b',
        'sources': [{'file': 'statcast_2023.parquet', 'bytes': 95124285,
                     'sha256': '46c354beae196ce2d501fe989c491725f2ecc47601f58d4b8b0234b982984730'},
                    {'file': 'statcast_2024.parquet', 'bytes': 103060977,
                     'sha256': '3cf05edddd22a2eac3d55be11cc56e00fe56f7fcf75823a8ecc8fa89e7e9a274'},
                    {'file': 'statcast_2025.parquet', 'bytes': 104922105,
                     'sha256': '4c785756d8901f8c46c70891954a4c54b8ad1c6909db67f4a000221691dc0249'}],
        'dev_dates': ['2025-07-01', '2025-10-01']}  # [first, end) of the selected games

BOOTSTRAP = {'draws': 10000, 'seed': 20260924, 'unit': 'whole game', 'method': 'ratio of per-game sums'}
GAP_BINS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05]  # bin k = [GAP_BINS[k], GAP_BINS[k+1]), the last one open
TV_QUANTILES = [0.5, 0.9, 0.99, 1.0]
TV_SHARE_ABOVE = 0.05
NOISE_QUANTILES = [0.5, 0.9, 0.99]

THRESHOLDS = {'top1_lower_at_least': 0.95, 'mean_tv_upper_at_most': 0.02, 'r_rows_at_most': 4.0,
              'smoke_r_b_at_most': 4.0, 'rb12_smoke_r_b_at_most': 2.0}
INTEGRITY = {'replay_decisions': 200, 'replay_atol': 0.0, 'stab01_k1_match_min': 0.999,
             'reconstruction_mismatch_share_max': 0.001, 'reconstruction_margin_max': 1e-05,
             'first_pitch_flips_max': 0, 'law_sum_atol': 1e-12, 'reference_atol': 1e-12}
COST = {'unit_U_seconds': 1258.65, 'rss_cap_bytes': 8589934592, 'rss_sample_every_pa_blocks': 10,
        'memory_pressure_free_min_percent': 50}
PASSES = {
    'smoke': {'output': 'smoke', 'cap_seconds': 3600, 'mode': 'runtime',
              'on_stop': 'NO_FULL_RUN'},
    'M': {'output': 'mandatory', 'cap_seconds': 21600, 'mode': 'runtime', 'on_stop': 'EXPERIMENT_ABORTED'},
    'C': {'output': 'rb12', 'cap_rule': {'factor': 1.2, 'units_per_r_b': 4}, 'mode': 'replay',
          'on_stop': 'RB12_ABORTED'},
    'D': {'output': 'descriptive', 'mode': 'replay', 'on_stop': 'ARM_DISCARDED',
          'BF48_cap_rule': {'factor': 1.2, 'units': 16, 'games': [50, 150]}, 'FH-DP_cap_seconds': 7200},
    'confirmation': {'output': 'confirmation', 'mode': 'runtime', 'on_stop': 'CONFIRMATION_ABORTED',
                     'cap_rule': {'factor': 1.2, 'constant_units': 2, 'units_per_r_b': {'RB6': 2, 'RB12': 4},
                                  'games': [50, 150]}},
}
PASS_ORDER = ('smoke', 'M', 'C', 'D')
RUN_WINDOW = {'utc_offset_hours': 9, 'deadline_kst': '06:00', 'no_start_kst': ['06:00', '13:30'],
              'protection_window_kst': ['06:30', '13:30'],
              'game_days': 'not known to the code: every day is treated as a game day'}
REPLAY_DECISIONS = INTEGRITY['replay_decisions']

# Every operative config field and its registered value (dotted path -> value).
PINNED_FIELDS = {
    'protocol': PROTOCOL, 'experiment_id': EXPERIMENT_ID,
    'settings': {'tau': TAU, 'samples': SAMPLES, 'pitch_cap': PITCH_CAP, 'seed': MAIN_SEED},
    'tau_freeze': TAU_FREEZE,
    'policy_registration.config': POLICY_REGISTRATION['config'],
    'policy_registration.config_sha256': POLICY_REGISTRATION['config_sha256'],
    'policy_registration.addenda': POLICY_REGISTRATION['addenda'],
    'g0_bundle': G0_BUNDLE,
    'registered_policy_identity': REGISTERED_POLICY_IDENTITY,
    'components_sha256': COMPONENTS_SHA256,
    'data': DATA,
    'games': GAMES, 'subsamples': SUBSAMPLES,
    'confirmation_slice.n_games': CONFIRMATION['n_games'], 'confirmation_slice.salt': CONFIRMATION['salt'],
    'confirmation_slice.exclude': CONFIRMATION['exclude'],
    'stab01': STAB01,
    'arms': ARMS,
    'seeds.pair': list(PAIR), 'seeds.reference': REFERENCE_SEED, 'seeds.stab01_sets': STAB01_SEEDS,
    'first_pitch_chunk': FIRST_PITCH_CHUNK,
    'bootstrap': BOOTSTRAP,
    'metrics.gap_bins': GAP_BINS, 'metrics.tv_quantiles': TV_QUANTILES, 'metrics.tv_share_above': TV_SHARE_ABOVE,
    'metrics.noise_quantiles': NOISE_QUANTILES,
    'decision_rules': THRESHOLDS,
    'integrity': INTEGRITY,
    'cost': COST,
    'passes': PASSES,
    'run_window': RUN_WINDOW,
}
REGISTRATION_FIELDS = ('registered_kst', 'registered_commit', 'design_review_reference', 'code_review_reference',
                       'code_sha256')

# Points the design leaves to the implementation (simplest deterministic option, pinned here).
IMPLEMENTATION_CHOICES = [
    'Config path configs/EXP-STAB-02.yaml with a JSON body (STAB-01 convention); this registry module is a fifth '
    'new file next to the four the design lists; its sha256 is pinned with the runner and module at registration '
    '(config registration.code_sha256) and verified by the runner before any data is read.',
    'RB continuation: one rollouts() call (the registered function) over every non-terminal child of every first '
    'pitch, children ordered action, sample, outcome; continuation uniforms U[i,1:,:] of the parent sample; a cap of '
    '1 truncates every child to the cutoff. Children with outcome probability 0 are still rolled out (weight 0).',
    'RB12 first pitch: chunks of six samples (0-5, then 6-11), action-major inside a chunk; RB6 is one chunk.',
    'Reconstruction mismatch margin: for a mismatching decision the RB search is re-run once with a tracing twin of '
    'JointSimulator.step on a scratch budget (it must reproduce the RB values exactly); a path whose selected first '
    'outcome is terminal has margin inf (a first-pitch flip); otherwise its margin is the smallest |u - CDF| over '
    'the outcome draws of its selected continuation. Per mismatching action: max over the mismatching samples when '
    'A0 per-sample values exist (pass M seed 5, confirmation seed 5), else min over the six samples; the decision '
    'margin is the max over its mismatching actions. Stop if any decision margin > 1e-5 (inf counts as a '
    'first-pitch flip) or, at the end of the pass, mismatching decisions exceed 0.1% of the compared decisions.',
    'A0 reproduction (pass M): K1_tv and K1_top1_same from diagnose_policy_seed_ensemble.compare() must equal the '
    'STAB-01 columns exactly; the run stops as soon as mismatches exceed floor(0.001 x the STAB-01 decision count).',
    'r_b (smoke) = mean over seeds 0 and 5 of rows(RB6, seed)/rows(A0, seed) on the three smoke games, the same '
    'definition as r_rows; A0 seed-0 rows are the runtime planner rows of those decisions only.',
    'Smoke projections scale by games: BF48 pass-D time = smoke BF48 seconds x 50/3; FH-DP 10-game time = '
    'FH-DP seconds on its 1 smoke game x 10.',
    'Deadline: the code does not know WS game days, so it treats every day as a game day: a pass may start only '
    'outside 06:00-13:30 KST and only if start + cap <= the next 06:00 KST; its deadline is min(next 06:00 KST, '
    'start + cap). A pass that does not fit is deferred: receipt DEFERRED, lock released, exit code 4.',
    'Pass clocks start when the pass starts (after the data load of the process); the deadline and the cap are '
    'checked before every PA block, peak RSS (ru_maxrss) every 10 PA blocks.',
    'Heavy lock held by another job: nothing starts, no pass folder is written, a LOCK_HELD receipt is appended and '
    'the process exits 3 (the pass is not marked aborted, so a later window can start it).',
    'Passes C and D re-create the pass-M decisions without the runtime planner (requests rebuilt from the store, '
    'kept by the pass-M request_id sequence, masked reference recomputed and compared with the stored mask and '
    'reference, components.check_context/check_support per request); the runtime seed-0 and A0 seed-5 Q come from '
    'the sha-verified pass-M arrays (no planner search is repeated).',
    'Exit codes: 0 done or nothing left, 2 refused (gate, config or path), 3 stopped (aborted, lock held, earlier '
    'output failed verification), 4 deferred to the next window.',
    'Variances (VRF, RB residual, mc_se) use ddof=1 over the samples; the efficiency interval bootstraps the VRF '
    'numerator/denominator sums and the per-seed row sums on the same game draws.',
    'Truncation share: A0 = truncated rollouts / rollouts; RB = probability-weighted truncated children / '
    'first-pitch samples (the expected A0 share).',
    'FH-DP: no registered row budget; its RowBudget limit is sys.maxsize and the 7,200 s cap bounds it; the '
    'predictor is called once per decision for all nodes (root 400 draws, other nodes indices 0,10,...,390); '
    'unreachable counts reuse the root law and BC rows (they never feed reachable values).',
    'FH-DP comparison: for RB pairs (0,5), (0,100), (5,100) the per-decision difference '
    'agreement(DP, RB_s) - agreement(RB_t, RB_s) and the same for TV are reported with intervals (descriptive).',
    'Smoke outputs contain no Q vector, law, TV, top-1, KL or reconstruction value (checked by key before writing).',
    'The final verdict is written once to <attempt>/verdict.json when it becomes final (smoke cost stop, end of '
    'pass M without RB12, end or stop of pass C), recomputed from the recorded pass statuses; '
    'results/EXP-STAB-02.json is not written by the runner (the PM copies a completed, verified result).',
    'A pass folder without results (crash or interrupt) is closed as an abort with the pass effect on the next start; '
    'a process in which any stop rule fired (pass C or a pass-D arm aborted while later passes ran) exits 3.',
    'Confirmation retries under the same profile go to <attempt>/confirmation/try-NN; a completed try (CONFIRMED or '
    'NOT_CONFIRMED) is final. The 50-game list is recorded by --confirmation-select (no planning call) and its sha256 '
    'must equal the pin file passed to --confirmation; Song\'s go is the PM gate before writing that pin.',
    'Peak RSS is ru_maxrss (bytes on macOS), conservative against the 8 GiB sampled-RSS cap; RSS per arm is not '
    'separable in one process, so the smoke records the peak of the whole smoke pass.',
    'One data scope per process: smoke (3 smoke games + the FH-DP smoke game), registered (the 150 games; smoke, M, C, D) '
    'or confirmation (50 games); the processed sha, statcast 2023-2025 sources, style snapshot, G0 bundle, tau freeze '
    'and the S2 bind identity 5683fb61 are checked before any planning call in every scope.',
]


class RegistrationError(ValueError):
    """The config deviates from the registered values or is not executable (fail closed)."""


def _lookup(config, dotted):
    node = config
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


_MISSING = object()


def _same(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            _same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            _same(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return math.isfinite(actual) and float(actual) == expected
    return type(actual) is type(expected) and actual == expected


def config_check(config):
    """Every operative field equals its registered value; status/execute/registration consistent.

    Accepts the draft. Raises ``RegistrationError`` listing every deviation."""
    if not isinstance(config, dict):
        raise RegistrationError('config must be a JSON object')
    problems = [f'{path} differs from the registered value' if _lookup(config, path) is not _MISSING
                else f'{path} missing' for path, value in PINNED_FIELDS.items()
                if not _same(_lookup(config, path), value)]
    seeds = config.get('seeds') if isinstance(config.get('seeds'), dict) else {}
    pair, reference = seeds.get('pair'), seeds.get('reference')
    stab = seeds.get('stab01_sets') if isinstance(seeds.get('stab01_sets'), dict) else {}
    used = set(stab.get('A', [])) | set(stab.get('B', []))
    if not (isinstance(pair, list) and len(pair) == 2 and len(set(pair)) == 2):
        problems.append('seeds.pair must be two distinct seeds')
    if reference in used or (isinstance(pair, list) and reference in pair):
        problems.append('seeds.reference must be disjoint from the EXP-STAB-01 seeds {0..9} and the pair')
    status, execute = config.get('status'), config.get('execute')
    registration = config.get('registration') if isinstance(config.get('registration'), dict) else None
    if status not in (STATUS_DRAFT, STATUS_REGISTERED):
        problems.append('status must be DRAFT_UNREGISTERED or REGISTERED')
    if type(execute) is not bool:
        problems.append('execute must be a boolean')
    if registration is None or set(registration) < set(REGISTRATION_FIELDS):
        problems.append('registration block with ' + ', '.join(REGISTRATION_FIELDS) + ' required')
    elif status == STATUS_DRAFT:
        if execute is not False:
            problems.append('a DRAFT_UNREGISTERED config cannot enable execution')
        if any(registration.get(k) is not None for k in REGISTRATION_FIELDS):
            problems.append('registration fields stay null until registration')
    elif status == STATUS_REGISTERED and execute is not True:
        problems.append('a REGISTERED config must set execute true')
    if problems:
        raise RegistrationError('; '.join(problems))
    return config


def registration_gate(config):
    """Fail closed: only a REGISTERED, executable config with every registration field may touch real data."""
    config_check(config)
    if config['status'] != STATUS_REGISTERED or config['execute'] is not True:
        raise RegistrationError(f'{EXPERIMENT_ID} is {config["status"]} with execute={config["execute"]}: '
                                'no real-data stage (the smoke included) runs before registration')
    registration = config['registration']
    missing = [k for k in REGISTRATION_FIELDS if not registration.get(k)]
    code = registration.get('code_sha256')
    if not missing and (not isinstance(code, dict) or set(code) != set(NEW_CODE)
                        or not all(isinstance(v, str) and len(v) == 64 for v in code.values())):
        missing.append('code_sha256 for ' + ', '.join(NEW_CODE))
    if missing:
        raise RegistrationError('registration fields missing: ' + ', '.join(missing))
    return config


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_unchanged_sources(repo):
    """The registered planner path is byte-identical to the pinned sources (A0 path unchanged)."""
    changed = [rel for rel, sha in UNCHANGED_SHA256.items()
               if not (Path(repo) / rel).is_file() or file_sha256(Path(repo) / rel) != sha]
    if changed:
        raise RegistrationError('registered planner path changed: ' + ', '.join(changed))
    return dict(UNCHANGED_SHA256)


def verify_registered_code(config, repo):
    """The new code is the reviewed code pinned at registration."""
    pins = config['registration']['code_sha256']
    changed = [rel for rel in NEW_CODE if file_sha256(Path(repo) / rel) != pins[rel]]
    if changed:
        raise RegistrationError('code differs from its registration pin: ' + ', '.join(changed))
    return dict(pins)


def pins_digest():
    return hashlib.sha256(json.dumps(PINNED_FIELDS, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# ---------------------------------------------------------------- decision rules

def stable(top1_ci95, mean_tv_ci95):
    """STABLE_X: top-1 agreement 95% lower bound >= 0.95 and mean TV 95% upper bound <= 0.02."""
    return (bool(top1_ci95[0] >= THRESHOLDS['top1_lower_at_least'])
            and bool(mean_tv_ci95[1] <= THRESHOLDS['mean_tv_upper_at_most']))


def cost_ok(r_rows):
    return bool(r_rows <= THRESHOLDS['r_rows_at_most'])


def rb12_required(smoke_r_b, rb6_stable):
    """RB12 (pass C) runs only if the smoke r_b <= 2.0 and RB6 is not STABLE."""
    return bool(smoke_r_b <= THRESHOLDS['rb12_smoke_r_b_at_most']) and not rb6_stable


def descriptive_allowed(rb6_stable, rb6_cost_ok, projected_seconds, cap_seconds):
    """BF48 / FH-DP (pass D) run only if RB6 is not (STABLE and COST_OK) and the smoke projection fits."""
    return not (rb6_stable and rb6_cost_ok) and bool(projected_seconds <= cap_seconds)


VERDICT_ROWS = {
    0: 'smoke r_b > 4.0 -> COST_EXCEEDED_AT_SMOKE (no full run)',
    1: 'RB6 STABLE and COST_OK -> STABLE_WITHIN_COST:RB6 (RB12 not run)',
    2: 'RB6 STABLE and not COST_OK -> STABLE_OVER_COST:RB6 (RB12 not run)',
    3: 'RB6 not STABLE, smoke r_b > 2.0 (RB12 not run) -> NOT_STABLE',
    4: 'RB6 not STABLE, RB12 run, RB12 STABLE and COST_OK -> STABLE_WITHIN_COST:RB12',
    5: 'RB6 not STABLE, RB12 run, RB12 STABLE and not COST_OK -> STABLE_OVER_COST:RB12',
    6: 'RB6 not STABLE, RB12 run, RB12 not STABLE -> NOT_STABLE',
    7: 'RB6 not STABLE, RB12 condition met but pass C aborted -> NOT_STABLE with flag RB12_ABORTED',
    8: 'pass M aborted -> no verdict label (aborted.json, nothing citable)',
}


def verdict(*, smoke_r_b, pass_m, rb6=None, rb12=None):
    """Registered verdict partition (design section 6), precedence (0) > (1) > (2) > (3).

    ``pass_m``: 'completed' | 'aborted' | 'not_run'. ``rb6``: {'stable', 'cost_ok'} after a completed pass M.
    ``rb12``: None (not required), {'state': 'pending'}, {'state': 'aborted'} or
    {'state': 'completed', 'stable', 'cost_ok'}. Returns {'row', 'label', 'flags', 'final'}; a combination the
    registered run conditions cannot produce raises ``ValueError``. H2 is not an input."""
    if not (isinstance(smoke_r_b, (int, float)) and math.isfinite(smoke_r_b) and smoke_r_b >= 0):
        raise ValueError('smoke r_b must be a finite nonnegative number')
    if smoke_r_b > THRESHOLDS['smoke_r_b_at_most']:
        if pass_m != 'not_run' or rb6 is not None or rb12 is not None:
            raise ValueError('no pass runs after COST_EXCEEDED_AT_SMOKE')
        return {'row': 0, 'label': 'COST_EXCEEDED_AT_SMOKE', 'flags': [], 'final': True}
    if pass_m == 'aborted':
        if rb6 is not None or rb12 is not None:
            raise ValueError('an aborted pass M leaves no RB6 status and no later pass')
        return {'row': 8, 'label': None, 'flags': ['PASS_M_ABORTED'], 'final': True}
    if pass_m != 'completed' or not isinstance(rb6, dict) or set(rb6) != {'stable', 'cost_ok'}:
        raise ValueError('a verdict needs a completed pass M with the RB6 status')
    required = rb12_required(smoke_r_b, rb6['stable'])
    if not required:
        if rb12 is not None:
            raise ValueError('RB12 runs only if smoke r_b <= 2.0 and RB6 is not STABLE')
        if rb6['stable']:
            return ({'row': 1, 'label': 'STABLE_WITHIN_COST:RB6', 'flags': [], 'final': True} if rb6['cost_ok']
                    else {'row': 2, 'label': 'STABLE_OVER_COST:RB6', 'flags': [], 'final': True})
        return {'row': 3, 'label': 'NOT_STABLE', 'flags': [], 'final': True}
    state = (rb12 or {}).get('state')
    if state == 'pending' and set(rb12) == {'state'}:
        return {'row': None, 'label': None, 'flags': ['PENDING_PASS_C'], 'final': False}
    if state == 'aborted' and set(rb12) == {'state'}:
        return {'row': 7, 'label': 'NOT_STABLE', 'flags': ['RB12_ABORTED'], 'final': True}
    if state != 'completed' or set(rb12) != {'state', 'stable', 'cost_ok'}:
        raise ValueError('RB12 is required: its state must be pending, aborted or completed with its status')
    if rb12['stable']:
        return ({'row': 4, 'label': 'STABLE_WITHIN_COST:RB12', 'flags': [], 'final': True} if rb12['cost_ok']
                else {'row': 5, 'label': 'STABLE_OVER_COST:RB12', 'flags': [], 'final': True})
    return {'row': 6, 'label': 'NOT_STABLE', 'flags': [], 'final': True}


def confirmation_verdict(state, *, top1_ci95=None, mean_tv_ci95=None, r_rows=None):
    """CONFIRMED iff the passing arm meets all three thresholds on the slice; an abort is CONFIRMATION_ABORTED."""
    if state == 'aborted':
        return 'CONFIRMATION_ABORTED'
    if state != 'completed':
        raise ValueError('confirmation state must be completed or aborted')
    return 'CONFIRMED' if stable(top1_ci95, mean_tv_ci95) and cost_ok(r_rows) else 'NOT_CONFIRMED'


# ---------------------------------------------------------------- caps and the run window

KST = timezone(timedelta(hours=RUN_WINDOW['utc_offset_hours']))
NO_START = (clock_time(6, 0), clock_time(13, 30))
DEADLINE_KST = clock_time(6, 0)


def pass_cap_seconds(name, *, r_b=None, arm=None):
    """Registered wall cap of a pass (or of one pass-D arm) in seconds."""
    unit = COST['unit_U_seconds']
    if name in ('smoke', 'M'):
        return float(PASSES[name]['cap_seconds'])
    if name == 'C':
        rule = PASSES['C']['cap_rule']
        return rule['factor'] * rule['units_per_r_b'] * _r_b(r_b) * unit
    if name == 'D':
        if arm == 'BF48':
            rule = PASSES['D']['BF48_cap_rule']
            return rule['factor'] * rule['units'] * unit * rule['games'][0] / rule['games'][1]
        if arm == 'FH-DP':
            return float(PASSES['D']['FH-DP_cap_seconds'])
        raise ValueError('pass D cap needs arm BF48 or FH-DP')
    if name == 'confirmation':
        rule = PASSES['confirmation']['cap_rule']
        if arm not in rule['units_per_r_b']:
            raise ValueError('the confirmation cap needs the passing arm RB6 or RB12')
        units = rule['constant_units'] + rule['units_per_r_b'][arm] * _r_b(r_b)
        return rule['factor'] * units * unit * rule['games'][0] / rule['games'][1]
    raise ValueError(f'unknown pass {name}')


def _r_b(r_b):
    if not (isinstance(r_b, (int, float)) and math.isfinite(r_b) and r_b > 0):
        raise ValueError('a measured smoke r_b is required')
    return float(r_b)


def next_deadline_utc(now_utc):
    """The first 06:00 KST strictly after ``now_utc``."""
    local = now_utc.astimezone(KST)
    target = local.replace(hour=DEADLINE_KST.hour, minute=DEADLINE_KST.minute, second=0, microsecond=0)
    if target <= local:
        target += timedelta(days=1)
    return target.astimezone(timezone.utc)


def start_decision(now_utc, cap_seconds):
    """('START', deadline_utc) if a pass with this cap may start now, else ('DEFER', reason)."""
    if now_utc.tzinfo is None:
        raise ValueError('aware UTC time required')
    local = now_utc.astimezone(KST).time()
    if NO_START[0] <= local < NO_START[1]:
        return 'DEFER', 'inside 06:00-13:30 KST: the run window opens at 13:30 KST'
    window_end = next_deadline_utc(now_utc)
    end = now_utc + timedelta(seconds=float(cap_seconds))
    if end > window_end:
        return 'DEFER', f'cap {cap_seconds:.0f} s ends after 06:00 KST ({window_end.isoformat()})'
    return 'START', min(window_end, end)


# ---------------------------------------------------------------- smoke record hygiene

FORBIDDEN_SMOKE_KEY_PARTS = ('tv', 'top1', 'agreement', 'kl', 'law', 'verdict', 'stable', 'vrf', 'efficiency',
                             'gap', 'bin', 'reconstruct', 'k1', 'k3', 'k5', 'h2', 'gain', 'q_values', 'q_vector',
                             'sample_values', 'max_abs')


def smoke_metric_keys(record, prefix=''):
    """Keys of a smoke record that look like a comparison metric or a Q/law value (must be empty)."""
    found = []
    if isinstance(record, dict):
        for key, value in record.items():
            name = str(key).lower()
            path = f'{prefix}.{key}' if prefix else str(key)
            if any(part in name for part in FORBIDDEN_SMOKE_KEY_PARTS):
                found.append(path)
            found += smoke_metric_keys(value, path)
    elif isinstance(record, list):
        for i, value in enumerate(record):
            found += smoke_metric_keys(value, f'{prefix}[{i}]')
    return found


def utc_now():
    return datetime.now(timezone.utc)
