"""Generate deterministic, development-only IntentEstimate contract examples.

The wrapper describes consumer scenarios. ``estimate`` alone is the existing
IntentEstimate v1 contract; no wrapper field is passed to its validator.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps/observer/backend'))

from observer_app.intent import intent_estimate, validate_intent_estimate  # noqa: E402

OUTPUT = ROOT / 'docs/contracts/examples/c0-v1.json'
PITCH_ID = '990001:12:3'
OTHER_PITCH_ID = '990001:12:4'
RECOMMENDED_AT = '2026-09-23T12:01:59.700Z'
RELEASE_AT = '2026-09-23T12:02:00Z'
RECEIVED_AT = '2026-09-23T12:01:59.600Z'
CORNERS = [{'x': 0, 'y': 0}, {'x': 100, 'y': 0},
           {'x': 100, 'y': 80}, {'x': 0, 'y': 80}]
PLATE = {'version': 'synthetic-fixture-v1',
         'method': 'synthetic matrix for development fixtures only',
         'x_convention': 'statcast_plate_x_catcher_view',
         'rms_error_feet': None,
         'fit_evidence': {'synthetic': True, 'note': 'not fit to real footage'},
         'matrix': [[1.66, 0, -.83], [0, -2.0, 3.5], [0, 0, 1]]}


def estimate(*, pitch_id=PITCH_ID, point=None, corners=CORNERS, plate=PLATE,
             bounds=None, unavailable_reason=None, method_version='synthetic-fixture-v1'):
    """Use the production builder, with explicitly synthetic inputs."""
    return intent_estimate(
        pitch_id=pitch_id, clip_id='synthetic-c0-clip', clip_sha256='0' * 64,
        evidence={'frame_index': 12, 'frame_time': 7.4},
        method={'kind': 'synthetic_contract_fixture', 'version': method_version},
        provenance={'label_source': 'video_module', 'review_status': 'unreviewed'},
        pixel_point=point, image_dimensions={'width': 100, 'height': 80},
        unavailable_reason=unavailable_reason, calibration_corners=corners,
        plate_calibration=plate, zone_bounds=bounds,
        uncertainty={'value': None, 'units': 'pixels',
                     'basis': 'synthetic example; accuracy unmeasured'})


def envelope(case, raw, *, expected_contract='valid', expected_consumer='accept',
             reason=None, pitch_id=PITCH_ID, received_at=RECEIVED_AT,
             recommendation_at=RECOMMENDED_AT, revision=1):
    return {'case': case, 'development_only': True, 'use_for_training': False,
            'use_for_performance_evaluation': False,
            'source': 'synthetic_contract_example',
            'replay_pitch_id': pitch_id, 'recommendation_created_at': recommendation_at,
            'release_at': RELEASE_AT, 'release_frame_time': 8.0,
            'received_at': received_at,
            'analysis_revision': revision,
            'expected': {'intent_contract': expected_contract,
                         'consumer': expected_consumer, 'reason': reason},
            'estimate': raw}


def build_examples():
    full = estimate(point={'x': 50, 'y': 40}, bounds={'top': 3.5, 'bottom': 1.5})
    partial = estimate(point={'x': 50, 'y': 40}, plate=None)
    missing = estimate(unavailable_reason='no_media')
    low_quality = estimate(unavailable_reason='low_quality_or_occlusion')
    bad_chain = copy.deepcopy(full)
    del bad_chain['points']['annotated_image_zone']
    wrong_link = estimate(pitch_id=OTHER_PITCH_ID, point={'x': 50, 'y': 40},
                          bounds={'top': 3.5, 'bottom': 1.5})
    corrected = estimate(point={'x': 58, 'y': 40}, bounds={'top': 3.5, 'bottom': 1.5},
                         method_version='synthetic-fixture-v2')
    examples = [
        envelope('normal_complete_chain', full),
        envelope('missing_intent', missing, expected_consumer='abstain', reason='no_media'),
        envelope('partial_coordinates', partial, expected_consumer='partial',
                 reason='no_plate_plane_calibration'),
        envelope('coordinate_chain_mismatch', bad_chain, expected_contract='reject',
                 expected_consumer='reject', reason='points omit an intermediate frame'),
        envelope('wrong_pitch_link', wrong_link, expected_consumer='reject',
                 reason='estimate pitch_id differs from replay_pitch_id'),
        envelope('low_quality_unavailable', low_quality, expected_consumer='abstain',
                 reason='low_quality_or_occlusion'),
        envelope('late_arrival', full, received_at='2026-09-23T12:02:30Z',
                 expected_consumer='analysis_only',
                 reason='received after recommendation; preserve prior recommendation'),
        envelope('correction_original', full, revision=1, received_at=RECEIVED_AT),
        envelope('correction_new_version', corrected, revision=2,
                 received_at='2026-09-23T12:03:00Z', expected_consumer='analysis_only',
                 reason='new immutable analysis version; preserve prior recommendation'),
    ]
    for item in examples:
        try:
            validate_intent_estimate(item['estimate'])
            actual = 'valid'
        except ValueError:
            actual = 'reject'
        if actual != item['expected']['intent_contract']:
            raise AssertionError(f"{item['case']}: expected {item['expected']['intent_contract']}, got {actual}")
    return {'description': 'Synthetic C0 development examples; no real video, labels, or measured calibration.',
            'pitch_id_convention': 'game_pk:at_bat_number:pitch_number',
            'examples': examples}


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_examples(), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
