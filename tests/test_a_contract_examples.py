"""C0 examples exercise the real intent validator and fixture consumer boundaries."""
from datetime import datetime
import json
from pathlib import Path

import pytest

from scripts.a_contract_examples import OUTPUT, build_examples
from observer_app.intent import validate_intent_estimate


def cases():
    return {item['case']: item for item in build_examples()['examples']}


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def test_checked_in_examples_are_deterministic_and_development_only():
    expected = json.loads(OUTPUT.read_text())
    assert expected == build_examples()
    assert len(expected['examples']) == len(cases()) == 9
    for item in expected['examples']:
        assert item['development_only'] is True
        assert item['use_for_training'] is False
        assert item['use_for_performance_evaluation'] is False
        assert item['source'] == 'synthetic_contract_example'
        assert item['estimate']['provenance']['review_status'] == 'unreviewed'
        assert item['estimate']['claims']['independent_ground_truth'] is False


def test_actual_contract_accepts_valid_examples_and_rejects_missing_hop():
    for item in cases().values():
        if item['expected']['intent_contract'] == 'valid':
            assert validate_intent_estimate(item['estimate']) is item['estimate']
        else:
            with pytest.raises(ValueError, match='frames reached'):
                validate_intent_estimate(item['estimate'])


def test_full_partial_and_abstaining_coordinates_are_distinct():
    sample = cases()
    full = sample['normal_complete_chain']['estimate']
    partial = sample['partial_coordinates']['estimate']
    assert list(full['points']) == ['image_pixels', 'annotated_image_zone', 'plate_feet', 'zone9']
    assert full['points']['zone9']['zone_id'] == 'middle_middle'
    assert full['transform_chain'][1]['error_status'] == 'unmeasured'
    assert list(partial['points']) == ['image_pixels', 'annotated_image_zone']
    assert partial['blocked_by'] == 'no_plate_plane_calibration'
    for case in ('missing_intent', 'low_quality_unavailable'):
        estimate = sample[case]['estimate']
        assert estimate['status'] == 'unavailable'
        assert estimate['points'] == {} and estimate['transform_chain'] == []


def test_wrong_link_is_a_consumer_rejection_despite_valid_intent_contract():
    item = cases()['wrong_pitch_link']
    validate_intent_estimate(item['estimate'])
    assert item['estimate']['pitch_id'] != item['replay_pitch_id']
    assert item['expected']['consumer'] == 'reject'


def test_late_evidence_is_pre_release_but_received_after_recommendation():
    item = cases()['late_arrival']
    assert item['estimate']['evidence']['frame_time'] < item['release_frame_time']
    assert timestamp(item['received_at']) > timestamp(item['recommendation_created_at'])
    assert item['expected']['consumer'] == 'analysis_only'


def test_correction_is_a_new_version_with_prior_estimate_unchanged():
    sample = cases()
    first, second = sample['correction_original'], sample['correction_new_version']
    assert first['replay_pitch_id'] == second['replay_pitch_id']
    assert first['analysis_revision'] == 1 and second['analysis_revision'] == 2
    assert first['estimate']['points']['image_pixels'] != second['estimate']['points']['image_pixels']
    assert first['estimate']['method']['version'] != second['estimate']['method']['version']
    assert timestamp(second['received_at']) > timestamp(first['received_at'])
    assert second['expected']['consumer'] == 'analysis_only'
