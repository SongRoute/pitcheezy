"""IntentEstimate 계약 검사. 합성 좌표만 쓰고 영상·외부 정답은 쓰지 않는다.

담당자의 영상 모듈이 통과해야 하는 적합성 검사이기도 하다:
산출물을 validate_intent_estimate 에 넣어 통과하면 저장할 수 있다.
"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.intent import (FRAME_CHAIN, NullIntentSource, build_chain, intent_estimate,
                                 project, setup_actual_difference, validate_intent_estimate,
                                 validate_plate_calibration, validate_reviewed_label, zone9_of)
from observer_app.intent_store import IntentLabels

DIMENSIONS = {'width': 100, 'height': 80}
CORNERS = [{'x': 0, 'y': 0}, {'x': 100, 'y': 0}, {'x': 100, 'y': 80}, {'x': 0, 'y': 80}]
BOUNDS = {'top': 3.5, 'bottom': 1.5}
# 단위정사각형 -> 피트. 실제 구장 보정이 아니라 검사용 합성 행렬이다.
PLATE = {'version': 'synthetic-test-v0', 'method': 'synthetic matrix for contract tests only',
         'x_convention': 'statcast_plate_x_catcher_view', 'rms_error_feet': None,
         'fit_evidence': {'note': 'not fit on any real footage'},
         'matrix': [[1.66, 0, -.83], [0, -2., 3.5], [0, 0, 1]]}


def estimate(**overrides):
    base = dict(pitch_id='p1', clip_id='seven_strikeouts', clip_sha256='a'*64,
                evidence={'frame_index': 12, 'frame_time': 7.4},
                method={'kind': 'video_module', 'version': '1.0.0'},
                provenance={'label_source': 'video_module', 'review_status': 'unreviewed'},
                pixel_point={'x': 50, 'y': 40}, image_dimensions=DIMENSIONS,
                uncertainty={'value': 10., 'units': 'pixels', 'basis': 'stated by the producing module'})
    return intent_estimate(**{**base, **overrides})


def review(**overrides):
    base = {'schema_version': 1, 'label_type': 'reviewed_setup_label', 'source_estimate_id': 'x',
            'decision': 'accepted', 'reviewer_id': 'song', 'reviewed_at': '2026-09-22T10:00:00+09:00',
            'evidence': {'frame_index': 12, 'frame_time': 7.4},
            'uncertainty': {'value': 8., 'units': 'pixels', 'basis': 'reviewer judgement'},
            'frame': None, 'point': None, 'notes': ''}
    return {**base, **overrides}


# --- 좌표계: 없는 홉은 만들지 않는다 ---

def test_chain_stops_at_image_plane_when_no_plate_calibration_exists():
    chain, deepest, blocked = build_chain(calibration_corners=CORNERS)
    assert deepest == 'annotated_image_zone'
    assert blocked == 'no_plate_plane_calibration'
    assert [record['target_frame'] for record in chain] == ['annotated_image_zone']
    assert chain[0]['error'] is None and chain[0]['error_status'] == 'unmeasured'


def test_chain_without_any_calibration_stays_in_pixels():
    chain, deepest, blocked = build_chain()
    assert (chain, deepest, blocked) == ([], 'image_pixels', 'no_image_plane_calibration')


def test_full_chain_records_one_versioned_transform_per_hop():
    chain, deepest, blocked = build_chain(calibration_corners=CORNERS, plate_calibration=PLATE, zone_bounds=BOUNDS)
    assert (deepest, blocked) == ('zone9', None)
    assert [record['source_frame'] for record in chain] == list(FRAME_CHAIN[:-1])
    assert all(record['version'] for record in chain)


def test_projection_reaches_only_the_frames_the_chain_supports():
    result = project({'x': 50, 'y': 40}, DIMENSIONS, calibration_corners=CORNERS)
    assert tuple(result['points']) == ('image_pixels', 'annotated_image_zone')
    assert 'plate_feet' not in result['points']
    assert result['points']['annotated_image_zone']['x'] == pytest.approx(.5)


def test_projection_through_declared_plate_calibration_reaches_nine_zones():
    result = project({'x': 50, 'y': 40}, DIMENSIONS, calibration_corners=CORNERS,
                     plate_calibration=PLATE, zone_bounds=BOUNDS)
    assert result['points']['plate_feet'] == {'x': pytest.approx(0.), 'z': pytest.approx(2.5),
                                              'x_convention': 'statcast_plate_x_catcher_view'}
    assert result['points']['zone9']['zone_id'] == 'middle_middle'


def test_plate_calibration_must_declare_the_left_right_convention():
    with pytest.raises(ValueError, match='x_convention'):
        validate_plate_calibration({**PLATE, 'x_convention': 'image_left_to_right'})


def test_plate_calibration_accepts_unmeasured_error_but_marks_it():
    chain, _, _ = build_chain(calibration_corners=CORNERS, plate_calibration=PLATE)
    assert chain[1]['error'] is None and chain[1]['error_status'] == 'unmeasured'
    measured, _, _ = build_chain(calibration_corners=CORNERS, plate_calibration={**PLATE, 'rms_error_feet': .12})
    assert measured[1]['error'] == .12 and measured[1]['error_status'] == 'measured'


def test_zone_quantization_reports_distance_to_the_nearest_boundary():
    centre = zone9_of(0., 2.5, BOUNDS)
    assert centre['boundary_margin_feet'] == pytest.approx(.2767, abs=1e-3)
    edge = zone9_of(.27, 2.5, BOUNDS)
    assert edge['zone_id'] == 'middle_middle' and edge['boundary_margin_feet'] < .01
    assert zone9_of(2., 2.5, BOUNDS)['outside_zone'] is True


# --- IntentEstimate: 통과하지 못하면 저장되지 않는다 ---

def test_estimate_without_plate_calibration_carries_no_nine_zone_intent():
    result = estimate(calibration_corners=CORNERS)
    assert result['deepest_frame'] == 'annotated_image_zone'
    assert result['blocked_by'] == 'no_plate_plane_calibration'
    assert 'zone9' not in result['points']
    assert result['claims']['physical_plate_coordinates'] is False


def test_estimate_is_always_a_proxy_and_never_an_observation():
    result = estimate(calibration_corners=CORNERS)
    assert result['is_intent_proxy'] is True
    assert result['claims']['catcher_intent_verified'] is False
    with pytest.raises(ValueError, match='never an observation'):
        validate_intent_estimate({**result, 'status': 'observed'})
    with pytest.raises(ValueError, match='is_intent_proxy'):
        validate_intent_estimate({**result, 'is_intent_proxy': False})


def test_estimate_cannot_claim_a_review_it_did_not_receive():
    result = estimate(calibration_corners=CORNERS)
    with pytest.raises(ValueError, match='separate record'):
        validate_intent_estimate({**result, 'provenance': {'label_source': 'video_module',
                                                           'review_status': 'reviewed'}})


def test_estimate_cannot_claim_plate_coordinates_it_did_not_reach():
    result = estimate(calibration_corners=CORNERS)
    with pytest.raises(ValueError, match='physical_plate_coordinates'):
        validate_intent_estimate({**result, 'claims': {**result['claims'], 'physical_plate_coordinates': True}})


def test_estimate_cannot_skip_a_coordinate_hop():
    result = estimate(calibration_corners=CORNERS, plate_calibration=PLATE, zone_bounds=BOUNDS)
    gapped = {**result, 'transform_chain': result['transform_chain'][:1]+result['transform_chain'][2:]}
    with pytest.raises(ValueError, match='one record per hop'):
        validate_intent_estimate(gapped)
    with pytest.raises(ValueError, match='frames reached'):
        validate_intent_estimate({**result, 'points': {key: value for key, value in result['points'].items()
                                                       if key != 'annotated_image_zone'}})


def test_estimate_cannot_report_accuracy_before_it_is_measured():
    result = estimate(calibration_corners=CORNERS)
    with pytest.raises(ValueError, match='accuracy_estimate'):
        validate_intent_estimate({**result, 'claims': {**result['claims'], 'accuracy_estimate': .9}})


def test_abstention_is_recorded_as_abstention_with_a_reason():
    result = NullIntentSource().estimate(pitch_id='p1', clip_id='seven_strikeouts', clip_sha256='a'*64)
    assert result['status'] == 'unavailable'
    assert result['unavailable_reason'] == 'video_module_not_connected'
    assert result['points'] == {} and result['transform_chain'] == []
    with pytest.raises(ValueError, match='no points'):
        validate_intent_estimate({**result, 'points': {'image_pixels': {'x': 1., 'y': 1.}}})


def test_estimate_must_name_the_frame_it_was_read_from():
    with pytest.raises(ValueError, match='frame_index'):
        estimate(calibration_corners=CORNERS, evidence={'frame_time': 7.4, 'frame_index': -1})


# --- 사람 검토: 자동 승격 없음 ---

def test_reviewed_label_requires_a_named_reviewer_and_an_explicit_decision():
    assert validate_reviewed_label(review())['decision'] == 'accepted'
    with pytest.raises(ValueError, match='reviewer_id'):
        validate_reviewed_label(review(reviewer_id=''))
    with pytest.raises(ValueError, match='decision'):
        validate_reviewed_label(review(decision='probably_fine'))
    with pytest.raises(ValueError, match='reviewed_at'):
        validate_reviewed_label(review(reviewed_at='2026-09-22T10:00:00'))


def test_reviewed_label_records_uncertainty_and_may_state_it_is_unmeasured():
    assert validate_reviewed_label(review(uncertainty={'value': None, 'units': None,
                                                       'basis': '미측정'}))['uncertainty']['value'] is None
    with pytest.raises(ValueError, match='uncertainty'):
        validate_reviewed_label(review(uncertainty={'value': 5.}))


def test_only_a_corrected_label_carries_its_own_coordinates():
    corrected = validate_reviewed_label(review(decision='corrected', frame='image_pixels',
                                               point={'x': 51., 'y': 39.}))
    assert corrected['point'] == {'x': 51., 'y': 39.}
    with pytest.raises(ValueError, match='Only a corrected label'):
        validate_reviewed_label(review(decision='accepted', frame='image_pixels', point={'x': 1., 'y': 1.}))
    with pytest.raises(ValueError, match='corrected label requires'):
        validate_reviewed_label(review(decision='corrected', frame='image_pixels', point={'x': 1.}))


def test_store_keeps_estimates_and_reviews_apart_with_no_promotion_path(tmp_path):
    labels = IntentLabels(tmp_path)
    assert not hasattr(labels, 'promote')
    saved = labels.save_estimate(estimate(calibration_corners=CORNERS))
    assert saved['review_status'] == 'unreviewed' and saved['reviews'] == []
    assert labels.reviewed_labels()['labels'] == []
    assert saved['estimate']['provenance']['review_status'] == 'unreviewed'


def test_store_marks_reviewed_only_after_a_human_decision(tmp_path):
    labels = IntentLabels(tmp_path)
    saved = labels.save_estimate(estimate(calibration_corners=CORNERS))
    reviewed = labels.review(saved['id'], review(decision='corrected', frame='image_pixels',
                                                 point={'x': 51., 'y': 39.}))
    assert reviewed['review_status'] == 'reviewed'
    assert reviewed['reviewer_ids'] == ['song']
    assert reviewed['estimate'] == saved['estimate']
    assert labels.reviewed_labels()['labels'][0]['id'] == saved['id']
    assert reviewed['reviews'][0]['source_estimate_id'] == saved['id']


def test_rejected_review_does_not_produce_a_reviewed_label(tmp_path):
    labels = IntentLabels(tmp_path)
    saved = labels.save_estimate(estimate(calibration_corners=CORNERS))
    record = labels.review(saved['id'], review(decision='rejected'))
    assert record['review_status'] == 'unreviewed'
    assert labels.reviewed_labels()['labels'] == []


def test_two_reviewers_on_one_estimate_are_both_kept(tmp_path):
    labels = IntentLabels(tmp_path)
    saved = labels.save_estimate(estimate(calibration_corners=CORNERS))
    labels.review(saved['id'], review(reviewer_id='song'))
    record = labels.review(saved['id'], review(reviewer_id='reviewer_b', decision='corrected',
                                               frame='image_pixels', point={'x': 55., 'y': 41.}))
    assert record['reviewer_ids'] == ['reviewer_b', 'song']
    assert len(record['reviews']) == 2


def test_review_of_an_unknown_estimate_is_refused(tmp_path):
    with pytest.raises(KeyError):
        IntentLabels(tmp_path).review('missing', review())


def test_stored_estimates_and_reviews_are_immutable(tmp_path):
    import sqlite3
    labels = IntentLabels(tmp_path)
    saved = labels.save_estimate(estimate(calibration_corners=CORNERS))
    with labels.connect() as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute('UPDATE estimates SET payload=? WHERE id=?', ('{}', saved['id']))


def test_sample_composition_counts_only_what_is_stored(tmp_path):
    labels = IntentLabels(tmp_path)
    first = labels.save_estimate(estimate(calibration_corners=CORNERS))
    labels.save_estimate(estimate(pitch_id='p2', calibration_corners=CORNERS))
    labels.review(first['id'], review())
    summary = labels.sample_composition()
    assert summary['estimates'] == 2
    assert summary['reviewed_labels'] == 1 and summary['unreviewed_estimates'] == 1
    assert summary['review_decisions'] == {'accepted': 1}
    assert summary['reviewers'] == ['song']
    assert summary['estimates_by_clip'] == {'seven_strikeouts': 2}
    assert summary['estimates_with_zone9'] == 0


# --- 미트-공 차이는 제구 오차가 아니다 ---

def test_difference_across_different_frames_is_refused():
    result = setup_actual_difference({'x': .5, 'y': .5}, 'annotated_image_zone', {'x': .1, 'z': 2.4}, 'plate_feet')
    assert result['status'] == 'unavailable' and result['reason'] == 'frame_mismatch'
    assert result['distance'] is None


def test_difference_in_one_frame_is_a_spatial_difference_not_command_error():
    result = setup_actual_difference({'x': 0., 'z': 2.5}, 'plate_feet', {'x': .3, 'z': 2.1}, 'plate_feet')
    assert result['status'] == 'measured'
    assert result['difference'] == {'x': pytest.approx(.3), 'z': pytest.approx(-.4)}
    assert result['distance'] == pytest.approx(.5)
    assert result['is_command_error'] is False
    assert 'setup_is_a_proxy_not_the_pitcher_intent' in result['named_unmeasured_components']
    assert 'glove_movement_between_setup_frame_and_release' in result['named_unmeasured_components']
