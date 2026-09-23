import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from observer_app.event_analysis import (analyze_event, failed_event, frozen_observed_value,
    recommendation_sha256, unavailable_replacement, value_point)


IDENTITY = {'model_version': 'observer-zone-v1', 'model_sha256': 'synthetic-model',
            'value_spec_version': 'defense-we-pa-v1',
            'baseline_policy_id': 'observer-repertoire-kernel-v1'}
INTENTS = json.loads((Path(__file__).parents[4] / 'docs/contracts/examples/c0-v1.json').read_text())
INTENT = next(x['estimate'] for x in INTENTS['examples'] if x['case'] == 'normal_complete_chain')


def fixture(*, reference=.5, plan=.49, execution=.47, observed=.4, intent=INTENT, revision=1):
    recommendation = {'id': 'rec-1', 'baseline_value': reference}
    linkage = {'session_id': 's1', 'pitch_id': '990001:12:3', 'recommendation_id': 'rec-1',
               'recommendation_created_at': '2026-09-23T12:01:59Z',
               'recommendation_sha256': recommendation_sha256(recommendation),
               'event_input_revision': revision}
    points = {key: None if value is None else value_point(value, identity=IDENTITY,
              initial_defender='home', source='synthetic frozen evaluator')
              for key, value in [('reference', reference), ('plan', plan),
                                 ('execution', execution), ('observed', observed)]}
    return dict(linkage=linkage, identity=IDENTITY, initial_defender='home', values=points,
                evidence={'development_only': True,
                          'plan_action': {'pitch_type': 'FF', 'zone_id': 'middle_middle', 'source': 'synthetic_fixture'},
                          'execution_action': {'pitch_type': 'FF', 'zone_id': 'high_middle', 'source': 'recorded_delivery'}},
                provenance={'received_at': '2026-09-23T12:02:01Z',
                            'generated_at': '2026-09-23T12:02:02Z'},
                intent_estimate=intent, release_frame_time=8.,
                clip_pitch_id='990001:12:3', stored_recommendation=recommendation)


def test_full_telescope_sign_unit_and_shares():
    result = analyze_event(**fixture())
    assert result['status'] == 'complete'
    assert result['values']['total_pp'] == pytest.approx(-10)
    parts = result['components']
    assert [parts[k]['value_pp'] for k in ('strategy_contrast_pp', 'execution_contrast_pp', 'outcome_residual_pp')] == pytest.approx([-1, -2, -7])
    assert sum(parts[k]['value_pp'] for k in ('strategy_contrast_pp', 'execution_contrast_pp', 'outcome_residual_pp')) == pytest.approx(result['values']['total_pp'])
    assert result['shares']['denominator_pp'] == pytest.approx(10)
    assert parts['outcome_residual_pp']['causal_batter_skill'] is False
    assert result['interactions']['value_pp'] is None
    assert parts['unallocated_residual_pp'] == 0


def test_no_intent_retains_total_without_player_credit():
    result = analyze_event(**fixture(plan=None, execution=None, intent=None))
    assert result['status'] == 'partial' and result['reason'] == 'missing_intent'
    assert result['values']['total_pp'] == pytest.approx(-10)
    assert result['components']['unallocated_residual_pp'] == pytest.approx(-10)
    for key in ('strategy_contrast_pp', 'execution_contrast_pp', 'outcome_residual_pp'):
        assert result['components'][key]['value_pp'] is None
        assert result['components'][key]['abs_share'] is None
    assert result['shares']['denominator_pp'] is None


def test_unavailable_and_failed_states_have_no_numeric_credit():
    args = fixture(reference=None, plan=None, execution=None, observed=None, intent=None)
    unavailable = analyze_event(**args)
    assert unavailable['status'] == 'unavailable'
    assert unavailable['values']['total_pp'] is None
    failed = failed_event(linkage=args['linkage'], identity=args['identity'],
        initial_defender='home', evidence=args['evidence'], provenance=args['provenance'],
        error_code='frozen_evaluator_error', stored_recommendation=args['stored_recommendation'])
    assert failed['status'] == 'failed' and failed['reason'] == 'frozen_evaluator_error'
    assert failed['shares']['denominator_pp'] is None


def test_near_zero_denominator_has_null_shares():
    result = analyze_event(**fixture(reference=.5, plan=.5, execution=.5, observed=.5))
    assert result['status'] == 'complete'
    assert result['shares'] == {'basis': 'sum_abs_assigned_components',
                                'denominator_pp': None, 'stable': False}


@pytest.mark.parametrize('change', ['pitch', 'clip', 'late', 'wrong_model', 'wrong_defender', 'actual_as_intent', 'wrong_recommendation', 'plan_from_actual_zone', 'plan_from_actual_type'])
def test_wrong_linkage_frame_or_value_identity_rejected(change):
    args = fixture()
    if change == 'pitch':
        args['intent_estimate'] = {**INTENT, 'pitch_id': '990001:12:4'}
    elif change == 'clip':
        args['clip_pitch_id'] = '990001:12:4'
    elif change == 'late':
        args['release_frame_time'] = 7.4
    elif change == 'wrong_model':
        args['values']['observed']['identity'] = {**IDENTITY, 'model_sha256': 'wrong'}
    elif change == 'wrong_defender':
        args['values']['observed']['initial_defender'] = 'away'
    elif change == 'actual_as_intent':
        args['evidence']['actual_is_intent'] = True
    elif change == 'plan_from_actual_zone':
        args['evidence']['plan_action']['zone_id'] = 'high_middle'
    elif change == 'plan_from_actual_type':
        args['evidence']['plan_action']['source'] = 'recorded_delivery'
    else:
        args['stored_recommendation']['baseline_value'] = .9
    with pytest.raises(ValueError):
        analyze_event(**args)


def test_correction_increments_analysis_revision_and_preserves_recommendation():
    original = analyze_event(**fixture(plan=None, execution=None, intent=None))
    corrected = analyze_event(**fixture(plan=None, execution=None, intent=None, revision=2))
    assert original['analysis_id'] == corrected['analysis_id']
    assert original['revision'] == 1 and corrected['revision'] == 2
    assert original['linkage']['recommendation_sha256'] == corrected['linkage']['recommendation_sha256']


def test_frozen_observed_adapter_checks_loaded_manifest_and_initial_defender(tmp_path):
    from pitchmdp.game import GameState
    manifest = tmp_path / 'bundle_manifest.json'
    manifest.write_text('{}')
    import hashlib
    ident = {**IDENTITY, 'model_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    class WE:
        def predict_defense(self, state, defender_is_home):
            assert state.half == 'Top' and defender_is_home is False
            return .42
    engine = SimpleNamespace(bundle=tmp_path, we=WE())
    initial = GameState(2, 'Bot', 2, 0, 1, 2)
    post = GameState(3, 'Top', 0, 0, 1, 2)
    point = frozen_observed_value(engine=engine, initial_state=initial, post_state=post, identity=ident)
    assert point['value'] == .42 and point['initial_defender'] == 'away'
    with pytest.raises(ValueError, match='model_sha256'):
        frozen_observed_value(engine=engine, initial_state=initial, post_state=post, identity=IDENTITY)


def test_replacement_boundary_requires_contemporary_eligibility_and_inning_evaluator():
    assert unavailable_replacement()['status'] == 'unavailable'
    with pytest.raises(ValueError, match='postdate'):
        unavailable_replacement(keep_pitcher_id=1, substitute_pitcher_ids=[2],
            decision_at='2025-07-05T12:00:00Z', roster_as_of='2025-07-05T13:00:00Z', candidates=[])
    result = unavailable_replacement(keep_pitcher_id=1, substitute_pitcher_ids=[123],
        decision_at='2025-07-05T12:00:00Z',
        roster_as_of='2025-07-05T11:00:00Z', candidates=[{'pitcher_id': 123, 'eligible': True}],
        workload_evidence={'source': 'contemporaneous'}, inning_evaluator=None)
    assert result['reason'] == 'missing_inning_end_evaluator' and result['value_pp'] is None
