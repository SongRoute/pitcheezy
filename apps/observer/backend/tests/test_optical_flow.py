"""Bounded synthetic gates for the frozen seed-only flow candidate."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

RESEARCH = Path(__file__).resolve().parents[2] / 'research'
sys.path.insert(0, str(RESEARCH))
from optical_flow_tracker import compare_endpoint, track_flow_frames

SPEC = json.loads((RESEARCH / 'optical_flow_spec.json').read_text())


def annotation():
    return {'schema_version': 1, 'clip_id': 'synthetic', 'clip_sha256': '0' * 64,
            'source_url': 'https://example.org/synthetic', 'pitch_id': None,
            'seed_time': 0.0, 'end_time': 0.2, 'release_time': 0.3,
            'roi': {'x': 24, 'y': 20, 'w': 48, 'h': 40},
            'image_dimensions': {'width': 128, 'height': 96},
            'label_source': 'assistant_visual_estimate', 'review_status': 'unreviewed',
            'annotation_version': 1, 'annotated_at': '2026-09-21T00:00:00Z'}


def textured_frame():
    rng = np.random.default_rng(42)
    image = np.zeros((96, 128), dtype=np.uint8)
    image[20:60, 24:72] = rng.integers(40, 216, (40, 48), dtype=np.uint8)
    return image


def shifted(image, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (128, 96))


def test_small_translation_reaches_complete_pixel_endpoint():
    first = textured_frame()
    result = track_flow_frames([(0.0, first), (0.1, shifted(first, 2, 1)),
                                (0.2, shifted(first, 4, 2))], annotation(), SPEC,
                               expected_frame_count=3)
    assert result['status'] == 'tracked'
    assert result['complete_window']
    assert result['pixel_target']['x'] == pytest.approx(52, abs=1)
    assert result['pixel_target']['y'] == pytest.approx(41, abs=1)
    assert result['tracker']['redetection'] is False


def test_flat_seed_abstains():
    flat = np.zeros((96, 128), dtype=np.uint8)
    result = track_flow_frames([(0.0, flat), (0.1, flat)], annotation(), SPEC,
                               expected_frame_count=2)
    assert result['status'] == 'abstained'
    assert result['abstain_reason'] == 'insufficient_seed_features'
    assert result['pixel_target'] is None


def test_shot_cut_abstains():
    first = textured_frame()
    cut = np.full_like(first, 255)
    result = track_flow_frames([(0.0, first), (0.1, cut)], annotation(), SPEC,
                               expected_frame_count=2)
    assert result['status'] == 'abstained'
    assert result['abstain_reason'] == 'shot_cut_or_high_frame_difference'


def test_partial_window_cannot_claim_endpoint():
    first = textured_frame()
    result = track_flow_frames([(0.0, first), (0.1, shifted(first, 2, 1))],
                               annotation(), SPEC, expected_frame_count=3)
    assert result['status'] == 'abstained'
    assert result['abstain_reason'] == 'incomplete_requested_window'
    assert result['pixel_target'] is None
    assert compare_endpoint(result, {'x': 50, 'y': 40}, 10)['available'] is False


def test_post_release_frame_rejected():
    frame = textured_frame()
    with pytest.raises(ValueError, match='pre-release'):
        track_flow_frames([(0.0, frame), (0.3, frame)], annotation(), SPEC,
                          expected_frame_count=2)
