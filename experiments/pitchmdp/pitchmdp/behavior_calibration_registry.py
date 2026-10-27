"""EXP-BC-CAL-01 (track A3) registered values and the fail-closed config check.

The design (docs/pm/research/track-a/A3-behavior-policy-design.md) and its registration draft fix
every value pinned here: arms B0/B1/B2 plus the descriptive cross-fit arms B0-XF/B1-XF (B3 and
B2-XF excluded by D186 5-1), seeds, splits and data roles including the A3-own batter-style
snapshot dates, the verdict thresholds, the cost caps and the recorder/publisher protection
window. ``IMPLEMENTATION_CHOICES`` lists the values the design leaves open and this code fixes
(they are pinned the same way so a reviewer sees one complete contract).

``config_check`` compares a config with these pins and raises ``RegistrationError`` naming every
deviation. It does not decide whether a config may run: ``registration_gate`` (in the runner)
additionally requires status REGISTERED, execution enabled and the registration pins. Nothing
here reads data.
"""
from __future__ import annotations

import math

EXPERIMENT_ID = 'EXP-BC-CAL-01'
ACTION_CODES = ('CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'PO', 'SC', 'SI', 'SL', 'ST',
                'SV', 'UN')
BASELINE = 'B0'
ADOPTION_CANDIDATES = ('B1', 'B2')
XF_ARMS = ('B0-XF', 'B1-XF')
DESCRIPTIVE_ARM_KEYS = ('B0-T', 'B1_raw_B2_raw', 'B0-XF', 'B1-XF', 'off_repertoire_mass')
EXCLUDED_ARMS = ('B3', 'B2-XF')
CANDIDATE_KEYS = ('B0', 'B1', 'B2', 'descriptive_arms_not_adoptable')

SEEDS = {'bootstrap': 20260924, 'bootstrap_draws': 10000, 'lightgbm': 20261043,
         'fold_salts': {'train_oof': 'EXP-BC-CAL-01/te', 'dev_xf': 'EXP-BC-CAL-01/xf'},
         'calibration_cv': 20260924}
FOLDS = 5

SPLIT_DATES = {'train': ['2023-05-15', '2025-04-30'], 'earlystop': ['2025-05-01', '2025-05-15'],
               'temperature': ['2025-05-16', '2025-05-31'], 'blend': ['2025-06-01', '2025-06-30'],
               'dev': ['2025-07-01', '2025-09-30']}
USED_SPLITS = ('train', 'earlystop', 'temperature', 'dev')  # blend (June) and 2023-03-30..05-14 are not read
LAST_DATE = '2025-09-30'  # the loader refuses any later date and any 2026 row
STYLE_AS_OF = {'earlystop': '2025-05-01', 'temperature': '2025-05-16', 'dev': '2025-07-01'}
STYLE_INPUT_FILTER = {'start': '2023-05-15', 'splits': ['train', 'earlystop', 'temperature'],
                      'end': '2025-05-31'}

B0_PARAMETERS = {'prior_strength': 20.0, 'minimum_action_count': 1}
B1_LEVELS = ('league', 'pitcher', 'batter_side', 'count', 'previous_type', 'base_out', 'times_through_order',
             'batter_cluster')
B1_B0_SETTING = {'pitcher': 0, 'batter_side': 'inf', 'count': 'inf', 'previous_type': 20, 'base_out': 'inf',
                 'times_through_order': 'inf', 'batter_cluster': 'inf'}
B1_ALPHA_GRID = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 'inf']  # 0 only for the pitcher level
B1_SWEEPS = 2

B2_PARAMS = {'objective': 'multiclass', 'num_class': 18, 'learning_rate': 0.05, 'num_leaves': 15, 'max_bin': 63,
             'min_data_in_leaf': 100, 'lambda_l2': 1.0, 'feature_fraction': 1.0, 'bagging_fraction': 1.0,
             'deterministic': True, 'force_col_wise': True, 'num_threads': 2, 'seed': 20261043, 'verbosity': -1}
B2_TRAIN_ROUNDS = 300
B2_CATEGORICAL = ['count_id', 'prev_type_1', 'prev_type_2', 'prev_type_3', 'prev_type_4', 'prev_type_5',
                  'prev_outcome_1', 'prev_outcome_2', 'prev_outcome_3', 'prev_outcome_4', 'prev_outcome_5']
STYLE_NAMES = ('contact', 'swing', 'walk', 'strikeout', 'isolated_power', 'groundball')
B2_FEATURES = ([f'b0_p_{code}' for code in ACTION_CODES] + ['pitcher_train_log_volume']
               + ['balls', 'strikes', 'count_id', 'pa_pitch_index']
               + [f'prev_type_{k}' for k in range(1, 6)] + [f'prev_outcome_{k}' for k in range(1, 6)]
               + ['outs', 'first_base', 'second_base', 'third_base', 'inning', 'defensive_score_lead',
                  'defender_is_home']
               + ['batter_left', 'pitcher_left', 'same_hand']
               + [f'style_rate_{n}' for n in STYLE_NAMES] + [f'style_reliability_{n}' for n in STYLE_NAMES]
               + [f'membership_{k}' for k in range(5)]
               + ['prior_pitches_game', 'times_through_order', 'is_game_starter'])
assert len(B2_FEATURES) == 63 and len(set(B2_FEATURES)) == 63

CALIBRATION = {'temperature_bounds': [0.5, 2.5], 'vector_l2': 0.001, 'vector_margin': 0.0005, 'cv_folds': 5}

THRESHOLDS = {'E1_nll_margin': -0.005, 'E2_ece_margin': 0.002, 'E3_p1_margin': 0.0005, 'holm_alpha': 0.05,
              'holm_family': ['B1', 'B2'], 'tie_margin': 0.001, 'tie_comparator': 'strict <', 'tie_simpler': 'B1',
              'P1_threshold': 0.001, 'P2_threshold': 0.01, 'E5_ess_min': {'pa': 100, 'game': 30},
              'large_weight': 10, 'mei': 0.001, 'ece_bins': 10,
              'sc3_bins': [0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0]}
SC1_GROUPS = {'fastball': ['FF', 'SI', 'FC', 'FA'], 'breaking': ['SL', 'ST', 'CU', 'KC', 'SV', 'CS'],
              'offspeed': ['CH', 'FS', 'FO', 'SC'], 'other': ['KN', 'EP', 'PO', 'UN']}

CAPS = {'stage_hang_guard_seconds': 7200, 'total_wall_cap_seconds': 14400, 'b2_fit_soft_cap_seconds': 3600,
        'b2_soft_cap_action': 'warn_and_continue', 'start_free_share_min': 0.4, 'abort_rss_bytes': 10737418240}
assert CAPS['abort_rss_bytes'] == 10 * 1024 ** 3
PROTECTION_WINDOW_KST = {'start': '06:30', 'end': '13:30', 'timezone': 'Asia/Seoul'}

S6 = {'expected_runtime_sha256': 'c13cc98994ae9eb08a8f939a583ebe712312594ac25c05314a9ca790bd46f73c',
      'r1b_atol': 1e-12, 'r2_rtol': 1e-9,
      'expected_ess': {'candidate': {'pa': 9120.218896471748, 'game': 144.2828223791589},
                       'reference': {'pa': 9725.694335061527, 'game': 144.16008032451504}},
      'allowed_header_fields': ['runtime_sha256', 'pins'],
      'allowed_fields': ['kind', 'request_id', 'runtime_sha256', 'pa_id', 'decision_index', 'status', 'pitcher',
                         'pitcher_hand', 'batter_side', 'balls', 'strikes', 'history_actions', 'logged_action',
                         'context_key', 'result.mask', 'result.logging', 'result.reference', 'result.candidate',
                         'result.logged_index', 'result.rho_reference', 'result.rho_candidate'],
      'never_opened': ['pa_values.parquet', 'dr.json']}
B0_ARTIFACT = {'file_sha256': 'b4f274eb39089553f813ae4d4419de5ddbb6b4edb381b0fe84ef5c9b49168e11',
               'support_table_sha256': '152cdab41e87faba57f0f46a402ec7796c4e131d41d6ff4a758a53616c6342d4',
               'hands_sha256': '0be0326e9d35a951b19749ada641097d240d0db6b7d8f6aadf90fa5c5cfd18ad'}
DATA_PINS = {'processed_sha256': '9f65d1dbf28374d6879dbf0c19c5c98834a2e1e95a6218c83e213c955e8ede27',
             'g0_bundle_sha256': 'c839dbf710c02ca5a1baafcc168e6ca3eaa6d80e269a99f71f08d7d537ec8062',
             'raw_sha256': {'statcast_2023.parquet': '46c354beae196ce2d501fe989c491725f2ecc47601f58d4b8b0234b982984730',
                            'statcast_2024.parquet': '3cf05edddd22a2eac3d55be11cc56e00fe56f7fcf75823a8ecc8fa89e7e9a274',
                            'statcast_2025.parquet': '4c785756d8901f8c46c70891954a4c54b8ad1c6909db67f4a000221691dc0249'}}

STAGES = ('smoke', 'census', 'fit', 'calibrate', 'freeze-primary', 's6-integrity', 'xf-fit', 'freeze-xf',
          'score-dev', 's6-weights', 'decide')
RUN_LABELS = ('ADOPT_FOR_2027_REGISTRATION:B1', 'ADOPT_FOR_2027_REGISTRATION:B2', 'KEEP_B0', 'FAILED_INTEGRITY',
              'NOT_EVALUABLE', 'ABORTED')
CANDIDATE_STATUSES = ('ELIGIBLE', 'INELIGIBLE', 'CALIBRATION_FAILED', 'NOT_EVALUABLE')
REGISTERED_LABELS = ['ADOPT_FOR_2027_REGISTRATION:<candidate>', 'KEEP_B0', 'FAILED_INTEGRITY', 'NOT_EVALUABLE']

SMOKE = {'train_rows': 50000, 'state_checks': 2000, 'b2_fits': 2, 'b2_rounds': 300,
         'rows': 'first train_rows TRAIN BC_P rows in frame order (game_date, game_pk, at_bat_number, pitch_number)'}

# Values the design leaves open; fixed here by this implementation (reported, pinned like the rest).
IMPLEMENTATION_CHOICES = {
    'seeds.calibration_cv': 'cv_folds (ported from experiments/ws_seq/calibrate_logits.py:324) needs a seed; '
                            'the registered bootstrap seed 20260924 is reused',
    'calibration.vector_a_lower_bound': 'a_c > 0 is enforced with L-BFGS-B bounds a_c >= 1e-6 (b_c free)',
    'calibration.optimizer': 'temperature: scipy minimize_scalar bounded [0.5, 2.5], default xatol; vector: '
                             'scipy L-BFGS-B with the analytic gradient, default options, init a = 1/T, b = 0',
    'calibration.failure': 'failure = optimizer not successful or non-finite objective/parameters',
    'b1.tie': 'ties on the earlystop S_in NLL are exact float64 equality',
    'b1.unseen_cell': 'a level whose key has no TRAIN count keeps the parent distribution (any alpha, incl. 0)',
    'b2.round_selection': 'masked/renormalized earlystop S_in NLL per round from a LightGBM feval (no extra params)',
    'b2.matrix_dtype': 'feature matrix float32 for training and prediction',
    'is_game_starter': 'pitcher equals the first pitcher of (game_pk, inning_topbot) in frame order',
    'batter_cluster': 'argmax of the five frozen-G0 soft memberships; unknown (5) when mean reliability is 0',
    'b1_xf_calibration': 'B1-XF applies the frozen primary B1 calibrator (no fold-specific calibration)',
    'style_train_rows': 'TRAIN rolling style values are hashed in the freeze manifest, not stored',
    'protection_window': 'applied every day (game days are not detected): never start inside 06:30-13:30 KST, '
                         'abort and preserve when 06:30 KST is reached',
    'quantiles': 'numpy linear quantiles for rho / W_T / CI summaries',
    'smoke': 'first 50,000 TRAIN BC_P rows; B2 fitted twice with the registered params and 300 rounds',
    'xf_artifacts': 'freeze-xf pins the XF out-of-fold prediction arrays (npy) and their row keys',
    'config_sha256': 'canonical sha256 of the config with code.config_sha256 nulled and '
                     'registration.addenda_records removed',
}


class RegistrationError(ValueError):
    """The config deviates from the registered EXP-BC-CAL-01 contract (fail closed)."""


def alpha_value(value):
    """Registered alpha token -> float ('inf' -> math.inf)."""
    if value == 'inf':
        return math.inf
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RegistrationError(f'invalid alpha {value!r}')
    return float(value)


def _field(config, dotted):
    node = config
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


_MISSING = object()

PINS = {
    'id': EXPERIMENT_ID,
    'action_space.codes': list(ACTION_CODES),
    'candidates.B0.parameters': B0_PARAMETERS,
    'data.b0_artifact.file_sha256': B0_ARTIFACT['file_sha256'],
    'data.b0_artifact.support_table_sha256': B0_ARTIFACT['support_table_sha256'],
    'data.b0_artifact.hands_sha256': B0_ARTIFACT['hands_sha256'],
    'data.b0_artifact.refit': False,
    'data.pins': DATA_PINS,
    'candidates.B1.levels': list(B1_LEVELS),
    'candidates.B1.b0_setting_values': B1_B0_SETTING,
    'candidates.B1.alpha_grid.value': B1_ALPHA_GRID,
    'candidates.B1.search.sweeps': B1_SWEEPS,
    'candidates.B1.search.tie': 'larger_alpha',
    'candidates.B2.params': B2_PARAMS,
    'candidates.B2.categorical_features': B2_CATEGORICAL,
    'candidates.B2.rounds.train_rounds': B2_TRAIN_ROUNDS,
    'candidates.B2.features': B2_FEATURES,
    'calibration.methods.temperature.bounds': CALIBRATION['temperature_bounds'],
    'calibration.methods.vector_scaling.l2_penalty.value': CALIBRATION['vector_l2'],
    'calibration.selection.margin': CALIBRATION['vector_margin'],
    'calibration.selection.cv': {'k': CALIBRATION['cv_folds'], 'seed': SEEDS['calibration_cv'],
                                 'assignment': 'cv_folds'},
    'calibration.vector_a_lower_bound': 1e-6,
    'seeds.bootstrap': SEEDS['bootstrap'],
    'seeds.bootstrap_draws': SEEDS['bootstrap_draws'],
    'seeds.lightgbm': SEEDS['lightgbm'],
    'seeds.fold_salts': SEEDS['fold_salts'],
    'seeds.calibration_cv': SEEDS['calibration_cv'],
    'cross_fitting.folds.k': FOLDS,
    'splits.train.dates': SPLIT_DATES['train'],
    'splits.earlystop.dates': SPLIT_DATES['earlystop'],
    'splits.temperature.dates': SPLIT_DATES['temperature'],
    'splits.blend.dates': SPLIT_DATES['blend'],
    'splits.dev.dates': SPLIT_DATES['dev'],
    'splits.style_as_of': STYLE_AS_OF,
    'splits.style_input_filter': STYLE_INPUT_FILTER,
    'splits.last_date': LAST_DATE,
    'positivity.P1.threshold': THRESHOLDS['P1_threshold'],
    'positivity.P2.threshold': THRESHOLDS['P2_threshold'],
    'self_consistency.SC3_small_probability_ratio.bins': THRESHOLDS['sc3_bins'],
    'self_consistency.SC1_groups': SC1_GROUPS,
    'metrics.ece_bins': THRESHOLDS['ece_bins'],
    'verdict_rule.eligibility.E1_nll_superiority.margin': THRESHOLDS['E1_nll_margin'],
    'verdict_rule.eligibility.E1_nll_superiority.holm_family': THRESHOLDS['holm_family'],
    'verdict_rule.eligibility.E1_nll_superiority.holm_alpha': THRESHOLDS['holm_alpha'],
    'verdict_rule.eligibility.E2_ece_noninferiority.margin': THRESHOLDS['E2_ece_margin'],
    'verdict_rule.eligibility.E3_positivity_noninferiority.margin': THRESHOLDS['E3_p1_margin'],
    'verdict_rule.eligibility.E5_thresholds': THRESHOLDS['E5_ess_min'],
    'verdict_rule.selection.tie_margin': THRESHOLDS['tie_margin'],
    'verdict_rule.selection.comparator': THRESHOLDS['tie_comparator'],
    'verdict_rule.selection.simpler': THRESHOLDS['tie_simpler'],
    'verdict_rule.labels': REGISTERED_LABELS,
    'ope_weight_diagnostics.integrity_gates.R2.expected': S6['expected_ess'],
    'ope_weight_diagnostics.integrity_gates.R1b_atol': S6['r1b_atol'],
    'ope_weight_diagnostics.integrity_gates.R2_rtol': S6['r2_rtol'],
    'ope_weight_diagnostics.large_weight_threshold.value': THRESHOLDS['large_weight'],
    'ope_weight_diagnostics.mei': THRESHOLDS['mei'],
    'data.s6_ledger.runtime_sha256.expected_value': S6['expected_runtime_sha256'],
    'data.s6_ledger.allowed_header_fields': S6['allowed_header_fields'],
    'data.s6_ledger.allowed_fields': S6['allowed_fields'],
    'cost_budget.hang_guard_seconds_per_stage.value': CAPS['stage_hang_guard_seconds'],
    'cost_budget.total_wall_cap_mandatory_seconds.value': CAPS['total_wall_cap_seconds'],
    'cost_budget.b2_per_fit_soft_cap_seconds.value': CAPS['b2_fit_soft_cap_seconds'],
    'cost_budget.b2_per_fit_soft_cap_seconds.mode': CAPS['b2_soft_cap_action'],
    'cost_budget.memory.start_requires_free_share_min': CAPS['start_free_share_min'],
    'cost_budget.memory.abort_rss_bytes': CAPS['abort_rss_bytes'],
    'cost_budget.protection_window_kst': PROTECTION_WINDOW_KST,
    'runner.stage_order': list(STAGES),
    'runner.smoke': SMOKE,
    'runner.run_labels': list(RUN_LABELS),
}


def _same(actual, expected):
    """Exact structural equality; numbers compare by value but bool never equals a number."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return actual == expected
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            _same(actual[k], expected[k]) for k in expected)
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            _same(a, e) for a, e in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def config_check(config):
    """Raise ``RegistrationError`` unless every registered value of EXP-BC-CAL-01 is unchanged.

    Checks the pinned fields, the arm partition (B0/B1/B2 + descriptive B0-T, raw, B0-XF, B1-XF only;
    B3 and B2-XF excluded and never an arm) and the protection-window / cap types. Status and
    execution flags are NOT checked here (``registration_gate`` does that)."""
    errors = []
    if not isinstance(config, dict):
        raise RegistrationError('config must be a JSON object')
    for dotted, expected in PINS.items():
        actual = _field(config, dotted)
        if actual is _MISSING:
            errors.append(f'missing registered field {dotted}')
        elif not _same(actual, expected):
            errors.append(f'{dotted} differs from the registered value')
    candidates = config.get('candidates')
    if isinstance(candidates, dict):
        if set(candidates) != set(CANDIDATE_KEYS):
            errors.append(f'candidate arms must be exactly {list(CANDIDATE_KEYS)}')
        descriptive = candidates.get('descriptive_arms_not_adoptable')
        if not isinstance(descriptive, dict) or set(descriptive) != set(DESCRIPTIVE_ARM_KEYS):
            errors.append(f'descriptive arms must be exactly {list(DESCRIPTIVE_ARM_KEYS)}')
        for name in EXCLUDED_ARMS:
            if name in candidates or (isinstance(descriptive, dict) and name in descriptive):
                errors.append(f'{name} is excluded (D186 5-1) and must not be an arm')
    excluded = config.get('excluded_not_run')
    if not isinstance(excluded, dict) or set(excluded) != set(EXCLUDED_ARMS):
        errors.append(f'excluded_not_run must list exactly {list(EXCLUDED_ARMS)}')
    family = _field(config, 'verdict_rule.eligibility.E1_nll_superiority.holm_family')
    if family is not _MISSING and isinstance(family, list) and any(name in EXCLUDED_ARMS for name in family):
        errors.append('the Holm family must not contain an excluded arm')
    if errors:
        raise RegistrationError('; '.join(errors))
    return config


def b0_alphas():
    """The B0 special case of the B1 chain as floats per level (pitcher .. batter_cluster)."""
    return {level: alpha_value(B1_B0_SETTING[level]) for level in B1_LEVELS[1:]}


def alpha_grid(level):
    """Registered grid for one level: 0 only at the pitcher level; 'inf' skips the level."""
    values = [alpha_value(v) for v in B1_ALPHA_GRID]
    return values if level == 'pitcher' else [v for v in values if v != 0]
