"""Synthetic image-plane tracking checks; no footage or external ground truth used."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.video_lab import validate_annotation, image_to_zone, track_frames, resolve_annotation_clip


def annotation():
    return {'schema_version': 1, 'clip_id': 'seven_strikeouts', 'clip_sha256': 'a'*64,
            'source_url': 'https://example.com/source', 'pitch_id': None,
            'seed_time': 0., 'end_time': .2, 'release_time': .3,
            'roi': {'x': 20, 'y': 20, 'w': 12, 'h': 10},
            'image_dimensions': {'width': 100, 'height': 80}, 'calibration_corners': None,
            'label_source': 'assistant_visual_estimate', 'review_status': 'unreviewed',
            'annotation_version': 1, 'annotated_at': '2026-09-21T13:00:00+00:00'}


def frames():
    patch = np.random.default_rng(14).integers(0, 256, (10, 12), dtype=np.uint8)
    output = []
    for i in range(3):
        image = np.zeros((80, 100), dtype=np.uint8)
        image[20+i:30+i, 20+3*i:32+3*i] = patch
        output.append((i*.1, image))
    return output


def test_translated_textured_template_tracks_pixels_only_without_calibration():
    result = track_frames(frames(), annotation(), search_radius=8)
    assert result['status'] == 'tracked'
    assert result['pixel_target'] == {'x': 32., 'y': 27.}
    assert result['normalized_target'] is None
    assert result['processed_frames'] == 3 and result['matched_frames'] == 2
    assert all(entry['confidence'] >= .99 for entry in result['trace'])
    assert result['label_source'] == 'assistant_visual_estimate' and result['review_status'] == 'unreviewed'
    assert result['claims']['accuracy_estimate'] is None
    assert not result['claims']['independent_ground_truth']
    assert not result['claims']['physical_plate_coordinates']
    assert result['pixel_stability']['max_displacement_from_seed'] == pytest.approx(np.hypot(6, 2))


def test_calibration_maps_only_annotated_image_plane_without_feet_or_mirroring():
    request = annotation()
    request['calibration_corners'] = [{'x': 0, 'y': 0}, {'x': 100, 'y': 0}, {'x': 100, 'y': 80}, {'x': 0, 'y': 80}]
    result = track_frames(frames(), request, search_radius=8)
    target = result['normalized_target']
    assert target['x'] == pytest.approx(.32)
    assert target['y'] == pytest.approx(27/80)
    assert target['coordinate_frame'] == 'annotated_image_zone'
    assert target['x_orientation'] == 'image_left_to_right_no_hand_mirroring'
    assert target['units'] == 'unit_square'


def test_occlusion_abstains_at_first_failed_frame_and_does_not_recover():
    sequence = frames()
    sequence[1] = (.1, np.zeros((80, 100), dtype=np.uint8))
    def supplied():
        yield sequence[0]
        yield sequence[1]
        raise AssertionError('Must not consume frames after the first failure')
    result = track_frames(supplied(), annotation())
    assert result['status'] == 'abstained'
    assert result['abstain_reason'] == 'low_match_confidence_or_occlusion'
    assert result['processed_frames'] == 2
    assert result['pixel_target'] is None and result['normalized_target'] is None


def test_shot_cut_abstains_even_before_template_search():
    sequence = frames()
    sequence[1] = (.1, np.full((80, 100), 255, dtype=np.uint8))
    result = track_frames(sequence, annotation())
    assert result['abstain_reason'] == 'shot_cut_or_high_frame_difference'
    assert result['trace'][-1]['confidence'] is None


def test_uniform_seed_does_not_produce_artificial_perfect_match():
    result = track_frames([(0., np.zeros((80, 100), dtype=np.uint8))], annotation())
    assert result['abstain_reason'] == 'insufficient_seed_texture'
    assert result['accepted_frames'] == 0


@pytest.mark.parametrize('changes', [
    {'end_time': .3}, {'end_time': .4}, {'seed_time': -.1}, {'seed_time': .2},
    {'seed_time': 0., 'end_time': 4., 'release_time': 5.}, {'release_time': float('nan')},
    {'roi': {'x': -1, 'y': 0, 'w': 12, 'h': 10}},
    {'roi': {'x': 95, 'y': 0, 'w': 12, 'h': 10}},
    {'review_status': 'assistant_reviewed'}, {'label_source': 'automatic_ground_truth'},
    {'annotated_at': '2026-09-21T13:00:00'}, {'annotation_version': 0},
])
def test_annotation_rejects_temporal_bounds_and_unsubstantiated_labels(changes):
    with pytest.raises(ValueError):
        validate_annotation(annotation() | changes)


@pytest.mark.parametrize('corners', [
    [(0, 0), (10, 0), (20, 0), (30, 0)],
    [(0, 0), (100, 80), (100, 0), (0, 80)],
    [(0, 0), (0, 80), (100, 80), (100, 0)],
    [(-1, 0), (100, 0), (100, 80), (0, 80)],
])
def test_degenerate_crossed_reversed_or_outside_calibration_is_rejected(corners):
    request = annotation()
    request['calibration_corners'] = [{'x': x, 'y': y} for x, y in corners]
    with pytest.raises(ValueError):
        validate_annotation(request)


def test_perspective_corner_correspondences_and_no_clamping():
    corners = [{'x': 20, 'y': 10}, {'x': 80, 'y': 15}, {'x': 90, 'y': 70}, {'x': 10, 'y': 65}]
    for point, expected in zip(corners, [(0,0), (1,0), (1,1), (0,1)]):
        projected = image_to_zone(point, corners, annotation()['image_dimensions'])
        assert projected['x'] == pytest.approx(expected[0], abs=1e-10)
        assert projected['y'] == pytest.approx(expected[1], abs=1e-10)
    outside = image_to_zone({'x': 0, 'y': 0}, corners, annotation()['image_dimensions'])
    assert not outside['inside']
    assert outside['x'] < 0 or outside['y'] < 0


def test_future_frame_or_nonmonotonic_time_is_rejected():
    with pytest.raises(ValueError, match='pre-release'):
        track_frames([(0., frames()[0][1]), (.3, frames()[1][1])], annotation())
    with pytest.raises(ValueError, match='increase'):
        track_frames([(0., frames()[0][1]), (0., frames()[1][1])], annotation())


def test_browser_annotation_has_no_server_path_and_unknown_id_is_rejected():
    clean = validate_annotation(annotation())
    assert 'clip_path' not in clean
    with pytest.raises(ValueError, match='Unknown approved clip'):
        resolve_annotation_clip(clean | {'clip_id': 'unapproved'})
