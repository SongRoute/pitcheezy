"""IntentEstimate 계약: 영상 모듈이 내놓는 값과 그것을 소비하는 쪽의 경계.

영상 모듈은 별도 담당자가 만든다. 이 파일은 그 산출물이 꽂힐 자리의 타입·검사이며,
docs/intent-spec.md 의 코드화다. 변경 = 계약 테스트 + 스펙 이력 + decisions.md 한 줄.

좌표는 네 프레임을 건너뛰지 않고 순서대로만 지난다:
    image_pixels -> annotated_image_zone -> plate_feet -> zone9
각 홉은 버전과 오차를 가진 transform 이고, 없는 홉은 만들어내지 않는다. 홉이 없으면
도달한 곳까지만 내고 왜 멈췄는지 이름을 남긴다.

지키는 것 세 가지:
  1. 포수 셋업은 의도의 대리값이다. is_intent_proxy 는 항상 True 이고 끌 수 없다.
  2. 미검토 추정은 검토 라벨이 될 수 없다. IntentEstimate 는 review_status='reviewed' 를
     받지 않는다. 검토 라벨은 사람의 결정이 실린 별도 타입이다.
  3. 미트-공 차이는 제구 오차가 아니다. setup_actual_difference 는 이름 붙은
     미측정 성분들을 함께 내고 is_command_error=False 를 끌 수 없다.
"""
from __future__ import annotations

from datetime import datetime
import math

import numpy as np

from .domain import ZONES
from .video_lab import _number, image_to_zone, validate_corners

SCHEMA_VERSION = 1
FRAME_CHAIN = ('image_pixels', 'annotated_image_zone', 'plate_feet', 'zone9')
FRAME_UNITS = {'image_pixels': 'pixels', 'annotated_image_zone': 'unit_square',
               'plate_feet': 'feet', 'zone9': 'zone_id'}
FRAME_KEYS = {'image_pixels': ('x', 'y'), 'annotated_image_zone': ('x', 'y'), 'plate_feet': ('x', 'z')}
# domain.py 의 9구역 경계와 같은 값. 여기서 바꾸면 두 곳이 갈린다.
PLATE_HALF_WIDTH_FEET = .83
X_CONVENTION = 'statcast_plate_x_catcher_view'
IMAGE_ZONE_VERSION = 'annotated_quad_homography_v1'
ZONE9_VERSION = 'domain_zone9_v1'
LABEL_SOURCES = ('video_module', 'assistant_visual_estimate', 'human_manual_annotation')
DECISIONS = ('accepted', 'corrected', 'rejected')
# 셋업과 실제 도달 위치의 차이에 섞여 있는, 아직 아무도 재지 않은 성분들.
UNMEASURED_DIFFERENCE_COMPONENTS = (
    'setup_is_a_proxy_not_the_pitcher_intent',
    'glove_detection_error',
    'coordinate_transform_error',
    'glove_movement_between_setup_frame_and_release',
    'pitch_movement_and_measurement_error')


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty string')
    return value


def _timestamp(value, label):
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f'{label} must be an ISO timestamp with timezone') from None
    if parsed.tzinfo is None:
        raise ValueError(f'{label} must include timezone')
    return value


def transform(source, target, *, method, version, error, error_units, evidence):
    """한 홉의 기록. error=None 은 '미측정'이며 0 이나 추정치로 대신하지 않는다."""
    if (source, target) not in set(zip(FRAME_CHAIN, FRAME_CHAIN[1:])):
        raise ValueError(f'{source}->{target} is not an adjacent coordinate hop')
    if error is not None:
        error = _number(error, 'transform error')
        if error < 0:
            raise ValueError('transform error must be non-negative')
    return {'source_frame': source, 'target_frame': target, 'method': _text(method, 'method'),
            'version': _text(version, 'version'), 'error': error,
            'error_units': None if error is None else _text(error_units, 'error_units'),
            'error_status': 'unmeasured' if error is None else 'measured', 'evidence': evidence}


def validate_plate_calibration(raw):
    """annotated_image_zone -> plate_feet. 지금 저장소에 이 보정을 맞춘 기록은 없다.

    담당자가 채워야 하는 홉이고, 채우기 전에는 9구역 의도가 나오지 않는다.
    """
    required = {'version', 'method', 'x_convention', 'matrix', 'rms_error_feet', 'fit_evidence'}
    if not isinstance(raw, dict) or required.difference(raw):
        raise ValueError('Missing plate calibration fields: '+','.join(sorted(required.difference(raw if isinstance(raw, dict) else {}))))
    if raw['x_convention'] != X_CONVENTION:
        raise ValueError(f'Plate calibration must declare x_convention={X_CONVENTION!r}; '
                         'left/right mirroring is not inferred from the image')
    matrix = raw['matrix']
    if (not isinstance(matrix, list) or len(matrix) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in matrix)):
        raise ValueError('Plate calibration matrix must be3x3')
    values = np.array([[_number(cell, 'matrix cell') for cell in row] for row in matrix])
    if np.linalg.cond(values) > 1e12:
        raise ValueError('Plate calibration matrix is numerically unstable')
    error = raw['rms_error_feet']
    if error is not None:
        error = _number(error, 'rms_error_feet')
        if error < 0:
            raise ValueError('rms_error_feet must be non-negative')
    if not isinstance(raw['fit_evidence'], dict) or not raw['fit_evidence']:
        raise ValueError('Plate calibration requires fit_evidence describing what it was fit on')
    return {'version': _text(raw['version'], 'plate calibration version'),
            'method': _text(raw['method'], 'plate calibration method'),
            'x_convention': X_CONVENTION, 'matrix': values.tolist(),
            'rms_error_feet': error, 'fit_evidence': raw['fit_evidence']}


def validate_zone_bounds(raw):
    if not isinstance(raw, dict) or not {'top', 'bottom'}.issubset(raw):
        raise ValueError('zone_bounds requires top and bottom in feet')
    top, bottom = _number(raw['top'], 'zone top'), _number(raw['bottom'], 'zone bottom')
    if not bottom < top:
        raise ValueError('zone_bounds requires bottom < top')
    return {'top': top, 'bottom': bottom}


def zone9_of(x, z, bounds):
    """plate_feet -> zone9. 양자화이므로 오차 대신 경계까지의 여유를 남긴다."""
    bounds = validate_zone_bounds(bounds)
    top, bottom, width = bounds['top'], bounds['bottom'], 2*PLATE_HALF_WIDTH_FEET
    if abs(x) > PLATE_HALF_WIDTH_FEET or not bottom <= z <= top:
        return {'zone_id': None, 'label': '존 바깥', 'column': None, 'row': None,
                'boundary_margin_feet': None, 'outside_zone': True}
    column = min(2, int((x+PLATE_HALF_WIDTH_FEET)/width*3))
    row = min(2, int((z-bottom)/(top-bottom)*3))
    zone = ZONES[row*3+column]
    column_edges = [-PLATE_HALF_WIDTH_FEET+index*width/3 for index in range(4)]
    row_edges = [bottom+index*(top-bottom)/3 for index in range(4)]
    margin = min(min(abs(x-edge) for edge in column_edges), min(abs(z-edge) for edge in row_edges))
    return {'zone_id': zone['id'], 'label': zone['label'], 'column': zone['column'], 'row': zone['row'],
            'boundary_margin_feet': float(margin), 'outside_zone': False}


def build_chain(*, calibration_corners=None, plate_calibration=None, zone_bounds=None):
    """있는 홉까지만 잇고, 어디서 왜 멈췄는지를 함께 낸다."""
    chain = []
    if calibration_corners is None:
        return chain, 'image_pixels', 'no_image_plane_calibration'
    chain.append(transform('image_pixels', 'annotated_image_zone',
                           method='four annotated image corners mapped to the unit square',
                           version=IMAGE_ZONE_VERSION, error=None, error_units=None,
                           evidence={'calibration_corners': calibration_corners}))
    if plate_calibration is None:
        return chain, 'annotated_image_zone', 'no_plate_plane_calibration'
    plate = validate_plate_calibration(plate_calibration)
    chain.append(transform('annotated_image_zone', 'plate_feet', method=plate['method'],
                           version=plate['version'], error=plate['rms_error_feet'],
                           error_units='feet', evidence=plate['fit_evidence']))
    if zone_bounds is None:
        return chain, 'plate_feet', 'no_batter_zone_bounds'
    chain.append(transform('plate_feet', 'zone9', method='fixed3x3 quantization of the batter strike zone',
                           version=ZONE9_VERSION, error=None, error_units=None,
                           evidence={'zone_bounds': validate_zone_bounds(zone_bounds),
                                     'note': 'quantization; see boundary_margin_feet'}))
    return chain, 'zone9', None


def project(pixel_point, image_dimensions, *, calibration_corners=None,
            plate_calibration=None, zone_bounds=None):
    """픽셀 한 점을 갈 수 있는 데까지만 옮긴다."""
    chain, deepest, blocked = build_chain(calibration_corners=calibration_corners,
                                          plate_calibration=plate_calibration, zone_bounds=zone_bounds)
    x, y = _number(pixel_point['x'], 'pixel x'), _number(pixel_point['y'], 'pixel y')
    points = {'image_pixels': {'x': x, 'y': y}}
    if deepest != 'image_pixels':
        projected = image_to_zone({'x': x, 'y': y}, calibration_corners, image_dimensions)
        points['annotated_image_zone'] = {'x': projected['x'], 'y': projected['y'],
                                          'inside_annotated_quad': projected['inside']}
    if deepest in ('plate_feet', 'zone9'):
        matrix = np.array(validate_plate_calibration(plate_calibration)['matrix'])
        unit = points['annotated_image_zone']
        result = matrix @ np.array([unit['x'], unit['y'], 1.])
        if abs(result[2]) < 1e-10:
            raise ValueError('Plate calibration cannot project this point')
        points['plate_feet'] = {'x': float(result[0]/result[2]), 'z': float(result[1]/result[2]),
                                'x_convention': X_CONVENTION}
    if deepest == 'zone9':
        points['zone9'] = zone9_of(points['plate_feet']['x'], points['plate_feet']['z'], zone_bounds)
    return {'deepest_frame': deepest, 'blocked_by': blocked, 'points': points, 'transform_chain': chain}


def intent_estimate(*, pitch_id, clip_id, clip_sha256, evidence, method, provenance,
                    pixel_point=None, image_dimensions=None, unavailable_reason=None,
                    calibration_corners=None, plate_calibration=None, zone_bounds=None,
                    uncertainty=None):
    """영상 모듈 쪽에서 IntentEstimate 를 만드는 생성자. 결과는 항상 검사를 통과한다."""
    if pixel_point is None:
        estimate = {'status': 'unavailable', 'unavailable_reason': _text(unavailable_reason or '', 'unavailable_reason'),
                    'deepest_frame': None, 'blocked_by': None, 'points': {}, 'transform_chain': []}
    else:
        projected = project(pixel_point, image_dimensions, calibration_corners=calibration_corners,
                            plate_calibration=plate_calibration, zone_bounds=zone_bounds)
        estimate = {'status': 'estimated', 'unavailable_reason': None, **projected}
    estimate.update({'schema_version': SCHEMA_VERSION, 'pitch_id': pitch_id, 'clip_id': clip_id,
                     'clip_sha256': clip_sha256, 'method': method, 'evidence': evidence,
                     'provenance': provenance, 'uncertainty': uncertainty, 'is_intent_proxy': True,
                     'claims': {'catcher_intent_verified': False, 'independent_ground_truth': False,
                                'physical_plate_coordinates': estimate['deepest_frame'] in ('plate_feet', 'zone9'),
                                'accuracy_estimate': None}})
    return validate_intent_estimate(estimate)


def validate_intent_estimate(raw):
    """담당자 모듈의 산출물을 받아들이는 유일한 문. 여기를 통과하지 못하면 저장하지 않는다."""
    required = {'schema_version', 'pitch_id', 'clip_id', 'clip_sha256', 'status', 'unavailable_reason',
                'method', 'evidence', 'provenance', 'claims', 'deepest_frame', 'blocked_by',
                'points', 'transform_chain', 'uncertainty', 'is_intent_proxy'}
    if not isinstance(raw, dict) or required.difference(raw):
        raise ValueError('Missing IntentEstimate fields: '+','.join(sorted(required.difference(raw if isinstance(raw, dict) else {}))))
    if raw['schema_version'] != SCHEMA_VERSION or isinstance(raw['schema_version'], bool):
        raise ValueError('Unsupported IntentEstimate schema')
    if raw['status'] == 'observed':
        raise ValueError('A setup estimate is never an observation; status must be estimated or unavailable')
    if raw['status'] not in ('estimated', 'unavailable'):
        raise ValueError('status must be estimated or unavailable')
    if raw['is_intent_proxy'] is not True:
        raise ValueError('Catcher setup is a proxy for intent; is_intent_proxy cannot be disabled')
    _text(raw['clip_id'], 'clip_id')
    _text(raw['clip_sha256'], 'clip_sha256')
    if raw['pitch_id'] is not None:
        _text(raw['pitch_id'], 'pitch_id')

    method = raw['method']
    if not isinstance(method, dict) or not {'kind', 'version'}.issubset(method):
        raise ValueError('method requires kind and version')
    _text(method['kind'], 'method kind')
    _text(method['version'], 'method version')

    evidence = raw['evidence']
    if not isinstance(evidence, dict) or not {'frame_index', 'frame_time'}.issubset(evidence):
        raise ValueError('evidence requires the frame_index and frame_time it was read from')
    if type(evidence['frame_index']) is not int or evidence['frame_index'] < 0:
        raise ValueError('evidence frame_index must be a non-negative integer')
    _number(evidence['frame_time'], 'evidence frame_time')

    provenance = raw['provenance']
    if not isinstance(provenance, dict) or not {'label_source', 'review_status'}.issubset(provenance):
        raise ValueError('provenance requires label_source and review_status')
    if provenance['label_source'] not in LABEL_SOURCES:
        raise ValueError('label_source must be one of: '+','.join(LABEL_SOURCES))
    if provenance['review_status'] != 'unreviewed':
        raise ValueError('An IntentEstimate is always unreviewed; a reviewed label is a separate '
                         'record created by an explicit human decision (validate_reviewed_label)')

    claims = raw['claims']
    if not isinstance(claims, dict):
        raise ValueError('claims must be a dict')
    if claims.get('catcher_intent_verified') is not False:
        raise ValueError('catcher_intent_verified must stay False')
    if claims.get('independent_ground_truth') is not False:
        raise ValueError('independent_ground_truth must stay False')
    if claims.get('accuracy_estimate') is not None:
        raise ValueError('accuracy_estimate must stay None until it is measured on a reviewed sample')

    points, chain, deepest = raw['points'], raw['transform_chain'], raw['deepest_frame']
    if raw['status'] == 'unavailable':
        if points or chain or deepest is not None:
            raise ValueError('An unavailable estimate carries no points, chain, or frame')
        _text(raw['unavailable_reason'] or '', 'unavailable_reason')
        return raw
    if raw['unavailable_reason'] is not None:
        raise ValueError('An estimated result has no unavailable_reason')
    if deepest not in FRAME_CHAIN:
        raise ValueError('deepest_frame must be one of: '+','.join(FRAME_CHAIN))
    reached = FRAME_CHAIN[:FRAME_CHAIN.index(deepest)+1]
    if tuple(points) != reached:
        raise ValueError('points must contain exactly the frames reached, in order: '+','.join(reached))
    if len(chain) != len(reached)-1:
        raise ValueError('transform_chain must hold one record per hop actually made')
    for record, (source, target) in zip(chain, zip(reached, reached[1:])):
        if record.get('source_frame') != source or record.get('target_frame') != target:
            raise ValueError(f'transform_chain must run {"->".join(reached)} without gaps')
        _text(record.get('version') or '', 'transform version')
        if record.get('error_status') not in ('measured', 'unmeasured'):
            raise ValueError('every transform declares error_status measured or unmeasured')
    for frame in reached:
        if frame == 'zone9':
            continue
        if not {key: None for key in FRAME_KEYS[frame]}.keys() <= points[frame].keys():
            raise ValueError(f'{frame} point requires '+','.join(FRAME_KEYS[frame]))
    if ('zone9' in points) != (deepest == 'zone9'):
        raise ValueError('A9-zone intent exists only when the plate calibration and zone bounds exist')
    if claims.get('physical_plate_coordinates') is not (deepest in ('plate_feet', 'zone9')):
        raise ValueError('physical_plate_coordinates must match the frame actually reached')
    if deepest != FRAME_CHAIN[-1] and not raw['blocked_by']:
        raise ValueError('A chain that stops short must name what blocked it')
    return raw


def validate_reviewed_label(raw):
    """사람이 검토해 확정한 라벨. 추정에서 자동으로 만들어지는 경로는 없다.

    reviewer_id 와 decision 은 기본값이 없다. 사람이 고르지 않으면 이 레코드는 존재할 수 없다.
    """
    required = {'schema_version', 'label_type', 'source_estimate_id', 'decision', 'reviewer_id',
                'reviewed_at', 'evidence', 'uncertainty', 'frame', 'point', 'notes'}
    if not isinstance(raw, dict) or required.difference(raw):
        raise ValueError('Missing reviewed label fields: '+','.join(sorted(required.difference(raw if isinstance(raw, dict) else {}))))
    if raw['schema_version'] != SCHEMA_VERSION or isinstance(raw['schema_version'], bool):
        raise ValueError('Unsupported reviewed label schema')
    if raw['label_type'] != 'reviewed_setup_label':
        raise ValueError('label_type must be reviewed_setup_label')
    if raw['decision'] not in DECISIONS:
        raise ValueError('decision must be one of: '+','.join(DECISIONS))
    _text(raw['source_estimate_id'], 'source_estimate_id')
    _text(raw['reviewer_id'], 'reviewer_id')
    _timestamp(raw['reviewed_at'], 'reviewed_at')
    evidence = raw['evidence']
    if not isinstance(evidence, dict) or type(evidence.get('frame_index')) is not int or evidence['frame_index'] < 0:
        raise ValueError('A reviewed label records the frame_index the reviewer looked at')
    _number(evidence.get('frame_time'), 'evidence frame_time')
    uncertainty = raw['uncertainty']
    if not isinstance(uncertainty, dict) or not {'value', 'units', 'basis'}.issubset(uncertainty):
        raise ValueError('uncertainty requires value, units, and basis; value may be null when unmeasured')
    if uncertainty['value'] is not None:
        if _number(uncertainty['value'], 'uncertainty value') < 0:
            raise ValueError('uncertainty value must be non-negative')
        _text(uncertainty['units'], 'uncertainty units')
    _text(uncertainty['basis'], 'uncertainty basis')
    if raw['decision'] == 'corrected':
        if raw['frame'] not in FRAME_KEYS:
            raise ValueError('A corrected label states the frame its coordinates are in')
        point = raw['point']
        if not isinstance(point, dict) or not {key: None for key in FRAME_KEYS[raw['frame']]}.keys() <= point.keys():
            raise ValueError('A corrected label requires '+','.join(FRAME_KEYS[raw['frame']])+' in its frame')
        for key in FRAME_KEYS[raw['frame']]:
            _number(point[key], 'corrected '+key)
    elif raw['point'] is not None or raw['frame'] is not None:
        raise ValueError('Only a corrected label carries its own coordinates')
    return raw


def setup_actual_difference(setup_point, setup_frame, actual_point, actual_frame):
    """셋업과 실제 도달 위치의 공간적 차이. 제구 오차가 아니며 그렇게 부를 수 없다."""
    base = {'is_command_error': False, 'named_unmeasured_components': list(UNMEASURED_DIFFERENCE_COMPONENTS),
            'interpretation': '포수 셋업 추정과 실제 도달 위치의 공간적 차이입니다. '
                              '아래 성분들이 아직 분리되지 않았으므로 투수의 제구 오차가 아닙니다.'}
    if setup_frame != actual_frame:
        return {**base, 'status': 'unavailable', 'reason': 'frame_mismatch',
                'detail': f'{setup_frame} 와 {actual_frame} 는 다른 좌표계라 빼지 않습니다.',
                'frame': None, 'difference': None, 'distance': None, 'units': None}
    if setup_frame not in FRAME_KEYS:
        return {**base, 'status': 'unavailable', 'reason': 'unsupported_frame',
                'detail': f'{setup_frame} 에서는 거리를 정의하지 않습니다.',
                'frame': setup_frame, 'difference': None, 'distance': None, 'units': None}
    keys = FRAME_KEYS[setup_frame]
    difference = {key: _number(actual_point[key], 'actual '+key)-_number(setup_point[key], 'setup '+key) for key in keys}
    return {**base, 'status': 'measured', 'reason': None, 'detail': None, 'frame': setup_frame,
            'difference': difference, 'distance': float(math.hypot(*difference.values())),
            'units': FRAME_UNITS[setup_frame]}


class NullIntentSource:
    """영상 모듈이 붙기 전의 자리. 성공을 흉내내지 않고 이유를 붙여 기권한다."""

    def estimate(self, *, pitch_id, clip_id, clip_sha256, **_):
        return intent_estimate(pitch_id=pitch_id, clip_id=clip_id, clip_sha256=clip_sha256,
                               evidence={'frame_index': 0, 'frame_time': 0.},
                               method={'kind': 'null_intent_source', 'version': '0'},
                               provenance={'label_source': 'video_module', 'review_status': 'unreviewed'},
                               unavailable_reason='video_module_not_connected')
