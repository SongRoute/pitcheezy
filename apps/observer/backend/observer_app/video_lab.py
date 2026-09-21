"""Manual-seed pre-release image tracker; an experimental baseline, never glove/intent ground truth.

The fixed initial ROI is matched locally. A homography maps annotated image-plane
corners only; it does not recover plate coordinates, feet, or physical depth.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import math
from pathlib import Path
import re

import numpy as np

from .settings import ARTIFACT_ROOT

LAB_ROOT = ARTIFACT_ROOT/'runs/observer-improvement-v2'
ALLOWED_MEDIA = (LAB_ROOT/'media/seven_strikeouts.mp4',)
MAX_WINDOW_SECONDS = 3.
MAX_FRAMES = 600


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite')
    return float(value)


def validate_corners(corners, dimensions):
    if not isinstance(corners, list) or len(corners) != 4:
        raise ValueError('Calibration requires four image corners ordered TL,TR,BR,BL')
    if any(not isinstance(point, dict) or not {'x', 'y'}.issubset(point) for point in corners):
        raise ValueError('Each calibration corner requires numeric x and y')
    points = np.array([[_number(point['x'], 'corner x'), _number(point['y'], 'corner y')] for point in corners])
    width, height = dimensions['width'], dimensions['height']
    if np.any(points < 0) or np.any(points[:, 0] > width) or np.any(points[:, 1] > height):
        raise ValueError('Calibration corners must lie inside the image')
    edges = np.roll(points, -1, axis=0)-points
    turns = edges[:, 0]*np.roll(edges[:, 1], -1)-edges[:, 1]*np.roll(edges[:, 0], -1)
    area = .5*abs(np.dot(points[:, 0], np.roll(points[:, 1], -1))-np.dot(points[:, 1], np.roll(points[:, 0], -1)))
    if np.any(turns <= 1e-6) or np.any(np.linalg.norm(edges, axis=1) < 2) or area < 16:
        raise ValueError('Calibration quadrilateral is degenerate, crossed, or incorrectly ordered')
    if points[[0, 1], 1].mean() >= points[[2, 3], 1].mean() or points[[0, 3], 0].mean() >= points[[1, 2], 0].mean():
        raise ValueError('Expected top then bottom and left then right image corners')
    return points


def validate_annotation(raw):
    required = {'schema_version', 'clip_id', 'clip_sha256', 'source_url', 'pitch_id',
                'seed_time', 'end_time', 'release_time', 'roi', 'image_dimensions', 'label_source',
                'review_status', 'annotation_version', 'annotated_at'}
    if not isinstance(raw, dict) or required.difference(raw):
        raise ValueError('Missing video annotation fields: '+','.join(sorted(required.difference(raw if isinstance(raw, dict) else {}))))
    if raw['schema_version'] != 1 or isinstance(raw['schema_version'], bool):
        raise ValueError('Unsupported video annotation schema')
    if not isinstance(raw['clip_id'], str) or not raw['clip_id'].strip():
        raise ValueError('A clip ID is required')
    if 'clip_path' in raw and (not isinstance(raw['clip_path'], str) or not raw['clip_path']):
        raise ValueError('A local clip path is required')
    if not isinstance(raw['source_url'], str) or not re.match(r'^https?://[^/\s]+', raw['source_url']):
        raise ValueError('An HTTP(S) source attribution URL is required; it will not be fetched')
    if not isinstance(raw['clip_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', raw['clip_sha256']):
        raise ValueError('Expected lowercase clip SHA256')
    if raw['pitch_id'] is not None and (not isinstance(raw['pitch_id'], str) or not raw['pitch_id']):
        raise ValueError('pitch_id must be null or a nonempty ID')
    if raw['label_source'] not in ('user_manual', 'assistant_visual_estimate'):
        raise ValueError('Label source must explicitly identify user or assistant annotation')
    if raw['review_status'] not in ('unreviewed', 'user_reviewed'):
        raise ValueError('Review status requires explicit independent user review or unreviewed')
    if type(raw['annotation_version']) is not int or raw['annotation_version'] < 1:
        raise ValueError('Positive annotation_version required')
    try:
        timestamp = datetime.fromisoformat(raw['annotated_at'].replace('Z', '+00:00'))
    except (ValueError, TypeError, AttributeError):
        raise ValueError('annotated_at must be an ISO timestamp with timezone') from None
    if timestamp.tzinfo is None:
        raise ValueError('annotated_at must include timezone')
    seed, end, release = [_number(raw[name], name) for name in ('seed_time', 'end_time', 'release_time')]
    if not 0 <= seed < end < release or end-seed > MAX_WINDOW_SECONDS:
        raise ValueError('Require 0 <= seed_time < end_time < release_time and a window no longer than3seconds')
    dimensions = raw['image_dimensions']
    if not isinstance(dimensions, dict) or any(type(dimensions.get(k)) is not int or dimensions[k] <= 0 for k in ('width', 'height')):
        raise ValueError('Positive integer image width/height required')
    roi = raw['roi']
    if not isinstance(roi, dict) or not {'x', 'y', 'w', 'h'}.issubset(roi):
        raise ValueError('ROI requires x,y,w,h')
    x, y, w, h = [_number(roi[key], 'ROI '+key) for key in ('x', 'y', 'w', 'h')]
    if x < 0 or y < 0 or w < 4 or h < 4 or x+w > dimensions['width'] or y+h > dimensions['height']:
        raise ValueError('ROI must be at least4×4 and entirely inside image bounds')
    corners = raw.get('calibration_corners')
    if corners is not None:
        validate_corners(corners, dimensions)
    # Preserve only the published contract fields, including optional calibration.
    result = {key: raw[key] for key in required}
    if 'clip_path' in raw:
        result['clip_path'] = raw['clip_path']
    result['calibration_corners'] = corners
    return result


def image_to_zone(point, corners, dimensions):
    """Project an image point onto the manually annotated quadrilateral's unit square."""
    points = validate_corners(corners, dimensions)
    target = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    matrix, values = [], []
    for (x, y), (u, v) in zip(points, target):
        matrix.extend([[x,y,1,0,0,0,-u*x,-u*y], [0,0,0,x,y,1,-v*x,-v*y]])
        values.extend([u, v])
    matrix = np.asarray(matrix)
    if np.linalg.cond(matrix) > 1e12:
        raise ValueError('Calibration homography is numerically unstable')
    h = np.r_[np.linalg.solve(matrix, np.asarray(values)), 1.].reshape(3, 3)
    projection = h @ np.array([_number(point['x'], 'point x'), _number(point['y'], 'point y'), 1.])
    if abs(projection[2]) < 1e-10:
        raise ValueError('Point cannot be projected onto the image-zone plane')
    x, y = projection[:2]/projection[2]
    return {'x': float(x), 'y': float(y), 'inside': bool(0 <= x <= 1 and 0 <= y <= 1),
            'coordinate_frame': 'annotated_image_zone', 'units': 'unit_square',
            'x_orientation': 'image_left_to_right_no_hand_mirroring', 'y_orientation': 'image_top_to_bottom'}


def allowed_clip(path):
    path = Path(path).resolve()
    if path not in {item.resolve() for item in ALLOWED_MEDIA} or not path.is_file():
        raise ValueError('Clip must be an existing explicitly approved local v2 media asset')
    return path


def resolve_annotation_clip(annotation):
    registered = {path.stem: path for path in ALLOWED_MEDIA}
    if annotation['clip_id'] not in registered:
        raise ValueError('Unknown approved clip ID')
    path = allowed_clip(registered[annotation['clip_id']])
    if annotation.get('clip_path') is not None and Path(annotation['clip_path']).resolve() != path:
        raise ValueError('Research clip_path differs from registered clip ID')
    return path


def track_frames(frames, raw_annotation, *, search_radius=40, confidence_threshold=.7,
                 scene_difference_threshold=.20):
    """Track ``(seconds, uint8 BGR/grayscale image)`` frames; stop at first uncertainty.

    No template updates or expanding search after failure. The initial ROI is a
    supplied label, not a detector result or independently verified glove location.
    """
    import cv2

    annotation = validate_annotation(raw_annotation)
    if type(search_radius) is not int or not 1 <= search_radius <= 128:
        raise ValueError('Search radius must be1..128pixels')
    if not .7 <= confidence_threshold <= 1 or not 0 < scene_difference_threshold < 1:
        raise ValueError('Confidence must be at least0.7 and scene threshold between0and1')
    width, height = annotation['image_dimensions']['width'], annotation['image_dimensions']['height']
    roi = annotation['roi']
    x, y = math.floor(roi['x']), math.floor(roi['y'])
    w, h = math.ceil(roi['x']+roi['w'])-x, math.ceil(roi['y']+roi['h'])-y
    trace, centers = [], []
    previous, template, reason, last_time = None, None, None, None
    for index, (timestamp, frame) in enumerate(frames):
        timestamp = _number(timestamp, 'frame timestamp')
        if index >= MAX_FRAMES:
            reason = 'frame_limit'
            break
        if timestamp < annotation['seed_time'] or timestamp > annotation['end_time'] or timestamp >= annotation['release_time']:
            raise ValueError('Every supplied frame must be inside the strictly pre-release interval')
        if last_time is not None and timestamp <= last_time:
            raise ValueError('Frame timestamps must increase strictly')
        last_time = timestamp
        entry = {'time': timestamp, 'accepted': False, 'confidence': None, 'x': None, 'y': None,
                 'width': w, 'height': h, 'reason': None, 'frame_difference': None}
        if (frame.dtype != np.uint8 or frame.shape[:2] != (height, width) or frame.ndim not in (2, 3)
                or (frame.ndim == 3 and frame.shape[2] not in (3, 4))):
            reason = entry['reason'] = 'frame_dimensions_or_format_changed'
            trace.append(entry)
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if template is None:
            template = gray[y:y+h, x:x+w].copy()
            if float(template.std()) < 8:
                reason = entry['reason'] = 'insufficient_seed_texture'
                trace.append(entry)
                break
            entry.update(accepted=True, confidence=1., x=x+w/2, y=y+h/2, reason='supplied_seed')
        else:
            difference = float(np.abs(gray.astype(np.float32)-previous.astype(np.float32)).mean()/255.)
            entry['frame_difference'] = difference
            if difference >= scene_difference_threshold:
                reason = entry['reason'] = 'shot_cut_or_high_frame_difference'
                trace.append(entry)
                break
            left, top = max(0, x-search_radius), max(0, y-search_radius)
            right, bottom = min(width, x+w+search_radius), min(height, y+h+search_radius)
            matches = cv2.matchTemplate(gray[top:bottom, left:right], template, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, location = cv2.minMaxLoc(matches)
            entry['confidence'] = float(confidence) if math.isfinite(confidence) else None
            if not math.isfinite(confidence) or confidence < confidence_threshold:
                reason = entry['reason'] = 'low_match_confidence_or_occlusion'
                trace.append(entry)
                break
            x, y = left+location[0], top+location[1]
            entry.update(accepted=True, x=x+w/2, y=y+h/2)
        trace.append(entry)
        centers.append((entry['x'], entry['y']))
        previous = gray
    if reason is None and len(centers) < 2:
        reason = 'insufficient_tracking_frames'
    status = 'abstained' if reason else 'tracked'
    points = np.asarray(centers, dtype=float)
    final = {'x': centers[-1][0], 'y': centers[-1][1]} if centers and status == 'tracked' else None
    normalized = image_to_zone(final, annotation['calibration_corners'], annotation['image_dimensions']) if final and annotation['calibration_corners'] else None
    return {'schema_version': 1, 'status': status, 'abstain_reason': reason,
            'clip_id': annotation['clip_id'], 'pitch_id': annotation['pitch_id'],
            'label_source': annotation['label_source'], 'review_status': annotation['review_status'],
            'annotation_version': annotation['annotation_version'], 'annotated_at': annotation['annotated_at'],
            'coordinate_frame': 'image_pixels', 'pixel_target': final, 'normalized_target': normalized,
            'trace': trace, 'processed_frames': len(trace), 'accepted_frames': len(centers),
            'matched_frames': max(0, len(centers)-1),
            'pixel_stability': {'center_std_x': float(points[:, 0].std()) if len(points) else None,
                                'center_std_y': float(points[:, 1].std()) if len(points) else None,
                                'max_displacement_from_seed': float(np.linalg.norm(points-points[0], axis=1).max()) if len(points) else None},
            'tracker': {'kind': 'fixed_manual_template_local_search', 'search_radius_pixels': search_radius,
                        'confidence_threshold': confidence_threshold, 'scene_difference_threshold': scene_difference_threshold},
            'claims': {'independent_ground_truth': False, 'accuracy_estimate': None,
                       'physical_plate_coordinates': False, 'catcher_intent_verified': False},
            'limitations': ['Manual/assistant seed labels are not independent tracking ground truth.',
                           'Template match confidence is not a calibrated probability of a glove detection.',
                           'Pixel motion/stability is descriptive and is not tracking accuracy.',
                           'Calibration maps an annotated image plane only, not feet, depth, or verified catcher-view orientation.']}


def track_video(path, annotation, **tracker_options):
    import cv2

    annotation = validate_annotation(annotation)
    path = allowed_clip(path)
    if path != resolve_annotation_clip(annotation):
        raise ValueError('Requested video differs from annotated registered clip')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != annotation['clip_sha256']:
        raise ValueError('Video identity differs from annotated clip SHA256')
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Approved video could not be decoded')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError('Video requires a valid constant frame rate and frame count')
        dimensions = {'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        if dimensions != annotation['image_dimensions']:
            raise ValueError('Annotation dimensions differ from decoded video')
        if annotation['release_time'] >= count/fps:
            raise ValueError('Annotated release must occur within clip duration')
        first = math.ceil(annotation['seed_time']*fps)
        last = math.floor(annotation['end_time']*fps)
        if last < first or last >= count or last-first+1 > MAX_FRAMES:
            raise ValueError('Invalid or excessive frame interval')
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        def frames():
            for index in range(first, last+1):
                success, image = capture.read()
                if not success:
                    raise ValueError('Video decoding stopped before requested pre-release interval completed')
                yield index/fps, image
        result = track_frames(frames(), annotation, **tracker_options)
        result['video'] = {'fps': fps, 'frame_count': count, 'image_dimensions': dimensions,
                           'requested_first_frame': first, 'requested_last_frame': last,
                           'timestamp_basis': 'constant_frame_rate_frame_index_divided_by_fps'}
        result['opencv_version'] = cv2.__version__
        return result
    finally:
        capture.release()
