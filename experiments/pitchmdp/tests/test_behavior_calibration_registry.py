"""EXP-BC-CAL-01 registry: the materialized config matches every pin; deviations are rejected; the
registration-pin schema is immutable code. Status-neutral: these tests run as the preflight after
registration too, so they never assume the committed config's status (explicit DRAFT / REGISTERED
fixtures come from behavior_calibration_synthetic.as_draft / as_registered)."""
import copy
import math

import pytest

from behavior_calibration_synthetic import as_draft, as_registered, load_config
from pitchmdp import behavior_calibration_registry as reg


@pytest.fixture(scope='module')
def config():
    return load_config()


def test_committed_config_is_consistent_and_passes_the_registered_contract(config):
    assert reg.registration_state(config) in ('DRAFT', 'REGISTERED')
    assert reg.config_check(config) is config
    assert set(config['candidates']) == {'B0', 'B1', 'B2', 'descriptive_arms_not_adoptable'}
    assert set(config['excluded_not_run']) == {'B3', 'B2-XF'}
    assert config['registration']['fields_pinned_at_registration'] == reg.DRAFT_PIN_FIELDS


def test_registration_state_of_explicit_fixtures(config):
    assert reg.registration_state(as_draft(config)) == 'DRAFT'
    assert reg.registration_state(as_registered(config)) == 'REGISTERED'
    for mutate in (lambda c: c.update(execute=False), lambda c: c.update(status='DRAFT_UNREGISTERED'),
                   lambda c: c['code'].update(commit=None),
                   lambda c: c['data']['s6_ledger'].update(candidate_sha256='0' * 64)):
        changed = as_registered(config)
        mutate(changed)
        with pytest.raises(reg.RegistrationError):
            reg.registration_state(changed)
    half = as_draft(config)
    half.update(registered=True)
    with pytest.raises(reg.RegistrationError, match='inconsistent'):
        reg.registration_state(half)


def test_registration_pin_schema_is_immutable_and_format_checked(config):
    good = as_registered(config)
    assert reg.registration_pin_errors(good) == []
    assert {d for d, _ in reg.REGISTRATION_PIN_SCHEMA} >= set(reg.DRAFT_PIN_FIELDS) | {
        'registration.registered_kst', 'registration.design_review_reference', 'registration.code_review_reference'}
    wrong = {'kst_datetime': '2026-10-28T05:00:00+00:00', 'text': '   ', 'sha1': 'g' * 40, 'sha256': 'A' * 64,
             'expected_candidate': '0' * 64}
    for dotted, kind in reg.REGISTRATION_PIN_SCHEMA:
        missing = copy.deepcopy(good)
        _set(missing, dotted, None)
        assert any(dotted in e and 'missing' in e for e in reg.registration_pin_errors(missing)), dotted
        malformed = copy.deepcopy(good)
        _set(malformed, dotted, wrong[kind])
        assert any(dotted in e and kind in e for e in reg.registration_pin_errors(malformed)), dotted
    naive = copy.deepcopy(good)
    naive['registration']['registered_kst'] = '2026-10-28T14:00:00'
    assert reg.registration_pin_errors(naive)
    # the config's own list cannot weaken the schema: emptying it deviates from the contract and the
    # immutable schema still reports every null pin
    emptied = copy.deepcopy(good)
    emptied['registration']['fields_pinned_at_registration'] = []
    with pytest.raises(reg.RegistrationError, match='fields_pinned_at_registration'):
        reg.config_check(emptied)
    emptied['data']['s6_ledger'].update(ledger_sha256=None, manifest_sha256=None)
    assert len(reg.registration_pin_errors(emptied)) == 2


def _set(config, dotted, value):
    node = config
    *parents, last = dotted.split('.')
    for part in parents:
        node = node[part]
    node[last] = value


DEVIATIONS = [
    ('verdict_rule.eligibility.E1_nll_superiority.margin', -0.004),
    ('verdict_rule.eligibility.E1_nll_superiority.holm_family', ['B1']),
    ('verdict_rule.eligibility.E1_nll_superiority.holm_family', ['B1', 'B2', 'B3']),
    ('verdict_rule.eligibility.E1_nll_superiority.holm_alpha', 0.1),
    ('verdict_rule.eligibility.E2_ece_noninferiority.margin', 0.003),
    ('verdict_rule.eligibility.E3_positivity_noninferiority.margin', 0.001),
    ('verdict_rule.eligibility.E5_thresholds', {'pa': 50, 'game': 30}),
    ('verdict_rule.selection.tie_margin', 0.002),
    ('verdict_rule.selection.comparator', '<='),
    ('verdict_rule.selection.simpler', 'B2'),
    ('cost_budget.total_wall_cap_mandatory_seconds.value', 20000),
    ('cost_budget.hang_guard_seconds_per_stage.value', 9000),
    ('cost_budget.b2_per_fit_soft_cap_seconds.value', 1800),
    ('cost_budget.b2_per_fit_soft_cap_seconds.mode', 'abort'),
    ('cost_budget.memory.abort_rss_bytes', 11 * 1024 ** 3),
    ('cost_budget.memory.start_requires_free_share_min', 0.3),
    ('cost_budget.protection_window_kst', {'start': '07:00', 'end': '13:30', 'timezone': 'Asia/Seoul'}),
    ('seeds.bootstrap', 1),
    ('seeds.lightgbm', 7),
    ('seeds.bootstrap_draws', 2000),
    ('seeds.fold_salts', {'train_oof': 'x', 'dev_xf': 'EXP-BC-CAL-01/xf'}),
    ('splits.style_as_of', {'earlystop': '2025-05-01', 'temperature': '2025-05-16', 'dev': '2025-06-01'}),
    ('splits.style_input_filter', {'start': '2023-03-30', 'splits': ['train', 'earlystop', 'temperature'],
                                   'end': '2025-05-31'}),
    ('splits.dev.dates', ['2025-06-01', '2025-09-30']),
    ('splits.train.dates', ['2023-03-30', '2025-04-30']),
    ('candidates.B2.params', {**reg.B2_PARAMS, 'num_leaves': 31}),
    ('candidates.B2.rounds.train_rounds', 500),
    ('candidates.B1.alpha_grid.value', [0, 1, 5, 'inf']),
    ('candidates.B1.b0_setting_values', {**reg.B1_B0_SETTING, 'previous_type': 10}),
    ('candidates.B1.search.sweeps', 3),
    ('candidates.B0.parameters', {'prior_strength': 10.0, 'minimum_action_count': 1}),
    ('calibration.methods.temperature.bounds', [0.25, 4.0]),
    ('calibration.selection.margin', 0.0),
    ('positivity.P1.threshold', 0.01),
    ('data.s6_ledger.expected_candidate_sha256', '0' * 64),
    ('ope_weight_diagnostics.integrity_gates.R2.expected',
     {'candidate': {'pa': 1.0, 'game': 1.0}, 'reference': {'pa': 1.0, 'game': 1.0}}),
    ('ope_weight_diagnostics.integrity_gates.R1b_atol', 1e-9),
    ('data.b0_artifact.file_sha256', 'f' * 64),
    ('data.b0_artifact.refit', True),
    ('runner.stage_order', list(reg.STAGES[1:])),
    ('action_space.codes', list(reg.ACTION_CODES[:-1])),
]


@pytest.mark.parametrize('dotted,value', DEVIATIONS, ids=[d for d, _ in DEVIATIONS])
def test_config_check_rejects_every_deviation(config, dotted, value):
    changed = copy.deepcopy(config)
    _set(changed, dotted, value)
    with pytest.raises(reg.RegistrationError, match=dotted.replace('.', r'\.')):
        reg.config_check(changed)


def test_config_check_rejects_excluded_and_extra_arms(config):
    for mutate, message in (
            (lambda c: c['candidates'].__setitem__('B3', {'mandatory': True}), 'B3 is excluded'),
            (lambda c: c['candidates']['descriptive_arms_not_adoptable'].__setitem__('B2-XF', 'x'), 'B2-XF is excluded'),
            (lambda c: c['excluded_not_run'].pop('B3'), 'excluded_not_run'),
            (lambda c: c['candidates'].pop('B2'), 'candidate arms'),
            (lambda c: c['candidates']['descriptive_arms_not_adoptable'].pop('B1-XF'), 'descriptive arms')):
        changed = copy.deepcopy(config)
        mutate(changed)
        with pytest.raises(reg.RegistrationError, match=message):
            reg.config_check(changed)


def test_missing_field_and_type_confusion_are_rejected(config):
    changed = copy.deepcopy(config)
    del changed['seeds']['calibration_cv']
    with pytest.raises(reg.RegistrationError, match='missing registered field seeds.calibration_cv'):
        reg.config_check(changed)
    changed = copy.deepcopy(config)
    changed['data']['b0_artifact']['refit'] = 0  # bool must stay bool
    with pytest.raises(reg.RegistrationError):
        reg.config_check(changed)


def test_registered_constants():
    assert reg.CAPS['abort_rss_bytes'] == 10_737_418_240 and reg.CAPS['total_wall_cap_seconds'] == 14_400
    assert reg.CAPS['b2_fit_soft_cap_seconds'] == 3_600 and reg.CAPS['b2_soft_cap_action'] == 'warn_and_continue'
    assert reg.THRESHOLDS['E1_nll_margin'] == -0.005 and reg.THRESHOLDS['holm_family'] == ['B1', 'B2']
    assert reg.THRESHOLDS['tie_margin'] == 0.001 and reg.THRESHOLDS['tie_comparator'] == 'strict <'
    assert reg.STYLE_AS_OF == {'earlystop': '2025-05-01', 'temperature': '2025-05-16', 'dev': '2025-07-01'}
    assert reg.S6['expected_candidate_sha256'].startswith('c13cc98994ae')
    assert len(reg.B2_FEATURES) == 63 and set(reg.B2_CATEGORICAL) <= set(reg.B2_FEATURES)
    assert reg.STAGES[0] == 'smoke' and reg.STAGES[-1] == 'decide'
    assert 'B3' not in reg.ADOPTION_CANDIDATES and reg.XF_ARMS == ('B0-XF', 'B1-XF')


def test_alpha_grid_and_b0_special_case():
    assert reg.alpha_grid('pitcher')[0] == 0 and reg.alpha_grid('pitcher')[-1] == math.inf
    assert 0 not in reg.alpha_grid('count') and len(reg.alpha_grid('count')) == 11
    assert reg.b0_alphas() == {'pitcher': 0.0, 'batter_side': math.inf, 'count': math.inf, 'previous_type': 20.0,
                               'base_out': math.inf, 'times_through_order': math.inf, 'batter_cluster': math.inf}
    with pytest.raises(reg.RegistrationError):
        reg.alpha_value(-1)
