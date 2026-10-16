"""Print reproducible C1 input/result pairs using only synthetic C0 evidence.

Run with the observer interpreter; stdout is JSON, with no data/model access.
These constants exercise arithmetic and are not measured win probabilities.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps/observer/backend'))
sys.path.insert(0, str(ROOT / 'scripts'))

from a_contract_examples import build_examples as intent_examples  # noqa: E402
from observer_app.event_analysis import (analyze_event, failed_event,  # noqa: E402
    recommendation_sha256, value_point)


def build_examples():
    intents = {item['case']: item['estimate'] for item in intent_examples()['examples']}
    identity = dict(model_version='synthetic-c1-v1', model_sha256='synthetic-no-model-loaded',
                    value_spec_version='defense-we-pa-v1', baseline_policy_id='synthetic-fixed-policy')
    recommendation = {'id': 'synthetic-rec-1', 'baseline_value': .5}
    base = dict(
        linkage=dict(session_id='synthetic-session', pitch_id='990001:12:3',
                     recommendation_id=recommendation['id'],
                     recommendation_created_at='2026-09-23T12:01:59Z',
                     recommendation_sha256=recommendation_sha256(recommendation), event_input_revision=1),
        identity=identity, initial_defender='home', stored_recommendation=recommendation,
        evidence=dict(development_only=True, use_for_performance_evaluation=False,
                      actual_source='synthetic_contract_example', references=[],
                      plan_action=dict(pitch_type='FF', zone_id='middle_middle', source='synthetic_fixture'),
                      execution_action=dict(pitch_type='FF', zone_id='high_middle', source='recorded_delivery')),
        provenance=dict(received_at='2026-09-23T12:02:01Z', generated_at='2026-09-23T12:02:02Z'),
        release_frame_time=8., clip_pitch_id='990001:12:3')
    examples = []
    cases = [
        ('complete', (.5, .49, .47, .4), 'normal_complete_chain'),
        ('cancelling_contrasts', (.5, .6, .4, .5), 'normal_complete_chain'),
        ('zero_contrasts', (.5, .5, .5, .5), 'normal_complete_chain'),
        ('missing_intent', (.5, None, None, .4), None),
        ('partial_coordinates', (.5, None, None, .4), 'partial_coordinates'),
        ('missing_execution', (.5, .49, None, .4), 'normal_complete_chain'),
        ('missing_observed', (.5, .49, .47, None), 'normal_complete_chain'),
        ('failed', (None, None, None, None), None),
    ]
    for case, numbers, intent_case in cases:
        inputs = copy.deepcopy(base)
        inputs['values'] = {key: None if number is None else value_point(
            number, identity=identity, initial_defender='home', source='synthetic_constant_not_model_output')
            for key, number in zip(('reference', 'plan', 'execution', 'observed'), numbers)}
        inputs['intent_estimate'] = copy.deepcopy(intents.get(intent_case))
        if case == 'failed':
            inputs = {key: inputs[key] for key in ('linkage', 'identity', 'initial_defender',
                                                  'evidence', 'provenance', 'stored_recommendation')}
            inputs['error_code'] = 'synthetic_calculation_error'
            result = failed_event(**inputs)
        else:
            result = analyze_event(**inputs)
        examples.append(dict(case=case, function='failed_event' if case == 'failed' else 'analyze_event',
                             inputs=inputs, result=result))
    return dict(development_only=True, use_for_training=False, use_for_performance_evaluation=False,
                source='synthetic_contract_example', examples=examples)


if __name__ == '__main__':
    print(json.dumps(build_examples(), ensure_ascii=False, indent=2, allow_nan=False))
