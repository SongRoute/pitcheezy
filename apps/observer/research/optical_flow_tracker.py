"""One conservative seed-only sparse-flow alternative; image motion, not glove detection."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'apps/observer/backend'))
from observer_app.video_lab import validate_annotation, resolve_annotation_clip, MAX_FRAMES


def track_flow_frames(frames, raw_annotation, spec, *, expected_frame_count):
    import cv2

    annotation = validate_annotation(raw_annotation)
    if type(expected_frame_count) is not int or not 2 <= expected_frame_count <= MAX_FRAMES:
        raise ValueError('Expected complete bounded frame count required')
    width, height = annotation['image_dimensions']['width'], annotation['image_dimensions']['height']
    roi = annotation['roi']
    x, y = math.floor(roi['x']), math.floor(roi['y'])
    w, h = math.ceil(roi['x']+roi['w'])-x, math.ceil(roi['y']+roi['h'])-y
    center = np.array([x+w/2, y+h/2], dtype=float)
    points = None
    previous, previous_time, seed_count = None, None, 0
    trace, centers, reason = [], [], None
    shi, lk = spec['shi_tomasi'], spec['lk']
    lk_args = {'winSize': (lk['window_size'], lk['window_size']), 'maxLevel': lk['pyramid_levels'],
               'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, lk['iterations'], lk['epsilon']),
               'minEigThreshold': lk['minimum_eigenvalue']}
    for index, (timestamp, frame) in enumerate(frames):
        if index >= expected_frame_count:
            raise ValueError('Supplied frames exceed frozen requested interval')
        if not np.isfinite(timestamp) or not annotation['seed_time'] <= timestamp <= annotation['end_time'] or timestamp >= annotation['release_time']:
            raise ValueError('Optical flow accepts strictly pre-release frames only')
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError('Frame timestamps must strictly increase')
        previous_time = timestamp
        entry = {'time': float(timestamp), 'accepted': False, 'confidence': None,
                 'confidence_kind': 'point_retention_fraction_not_probability', 'x': None, 'y': None,
                 'width': w, 'height': h, 'reason': None, 'frame_difference': None,
                 'points_before': len(points) if points is not None else 0, 'points_retained': 0,
                 'median_forward_backward_error_px': None, 'median_photometric_error_gray': None,
                 'median_inlier_motion_residual_px': None}
        if (frame.dtype != np.uint8 or frame.shape[:2] != (height, width) or frame.ndim not in (2, 3)
                or (frame.ndim == 3 and frame.shape[2] not in (3, 4))):
            reason = entry['reason'] = 'frame_dimensions_or_format_changed'
            trace.append(entry)
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if previous is None:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            points = cv2.goodFeaturesToTrack(gray, maxCorners=shi['max_corners'], qualityLevel=shi['quality_level'],
                minDistance=shi['min_distance'], mask=mask, blockSize=shi['block_size'])
            seed_count = 0 if points is None else len(points)
            if seed_count < spec['minimum_seed_points']:
                reason = entry['reason'] = 'insufficient_seed_features'
                entry['points_retained'] = seed_count
                trace.append(entry)
                break
            identifiers = np.arange(seed_count)
            entry.update(accepted=True, confidence=1., x=float(center[0]), y=float(center[1]),
                         points_retained=seed_count, reason='supplied_roi_seed_features')
        else:
            difference = float(np.abs(gray.astype(np.float32)-previous.astype(np.float32)).mean()/255.)
            entry['frame_difference'] = difference
            if difference >= spec['scene_difference_threshold']:
                reason = entry['reason'] = 'shot_cut_or_high_frame_difference'
                trace.append(entry)
                break
            forward, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **lk_args)
            if forward is None or forward_status is None or forward_error is None or not np.isfinite(forward).all():
                reason = entry['reason'] = 'forward_flow_unavailable'
                trace.append(entry)
                break
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, forward, None, **lk_args)
            if backward is None or backward_status is None:
                reason = entry['reason'] = 'backward_flow_unavailable'
                trace.append(entry)
                break
            old = points.reshape(-1, 2)
            new = forward.reshape(-1, 2)
            fb = np.linalg.norm(backward.reshape(-1, 2)-old, axis=1)
            photometric = forward_error.ravel()
            valid = (forward_status.ravel().astype(bool) & backward_status.ravel().astype(bool) &
                     np.isfinite(fb) & np.isfinite(photometric) &
                     (fb <= spec['maximum_forward_backward_error_px']) &
                     (photometric <= spec['maximum_photometric_error_gray']) &
                     (new[:, 0] >= 0) & (new[:, 0] < width) & (new[:, 1] >= 0) & (new[:, 1] < height))
            count_before = len(old)
            if valid.any():
                motion = new-old
                median = np.median(motion[valid], axis=0)
                residual = np.linalg.norm(motion-median, axis=1)
                valid &= residual <= spec['median_motion_residual_threshold_px']
            retained = int(valid.sum())
            fraction = retained/count_before
            entry.update(points_retained=retained, confidence=fraction)
            if retained:
                median = np.median((new-old)[valid], axis=0)
                entry.update(median_forward_backward_error_px=float(np.median(fb[valid])),
                             median_photometric_error_gray=float(np.median(photometric[valid])),
                             median_inlier_motion_residual_px=float(np.median(np.linalg.norm((new-old)[valid]-median, axis=1))))
            if (retained < spec['minimum_retained_points'] or fraction < spec['minimum_previous_point_fraction']
                    or retained/seed_count < spec['minimum_seed_point_fraction']):
                reason = entry['reason'] = 'insufficient_consistent_tracks_or_occlusion'
                trace.append(entry)
                break
            if np.linalg.norm(median) > spec['maximum_step_px']:
                reason = entry['reason'] = 'excessive_per_frame_motion'
                trace.append(entry)
                break
            center += median
            if not 0 <= center[0] < width or not 0 <= center[1] < height:
                reason = entry['reason'] = 'translated_roi_center_outside_image'
                trace.append(entry)
                break
            # Propagate only surviving original seed identities: no re-detection.
            points = forward[valid].reshape(-1, 1, 2)
            identifiers = identifiers[valid]
            entry.update(accepted=True, x=float(center[0]), y=float(center[1]))
        entry['seed_point_ids'] = identifiers.tolist()
        entry['tracked_points'] = points.reshape(-1, 2).astype(float).tolist()
        trace.append(entry)
        centers.append(center.copy())
        previous = gray
    if reason is None and len(trace) != expected_frame_count:
        reason = 'incomplete_requested_window'
    if reason is None and len(centers) < 2:
        reason = 'insufficient_tracking_frames'
    complete = reason is None and len(trace) == expected_frame_count
    accepted = np.asarray(centers, dtype=float)
    return {'schema_version': 1, 'status': 'tracked' if complete else 'abstained', 'complete_window': complete,
            'abstain_reason': reason, 'clip_id': annotation['clip_id'], 'pitch_id': annotation['pitch_id'],
            'label_source': annotation['label_source'], 'review_status': annotation['review_status'],
            'annotation_version': annotation['annotation_version'], 'annotated_at': annotation['annotated_at'],
            'coordinate_frame': 'image_pixels', 'normalized_target': None,
            'pixel_target': {'x': float(center[0]), 'y': float(center[1])} if complete else None,
            'trace': trace, 'processed_frames': len(trace), 'expected_frames': expected_frame_count,
            'accepted_frames': len(centers), 'matched_frames': max(0, len(centers)-1), 'seed_points': seed_count,
            'pixel_stability': {'center_std_x': float(accepted[:, 0].std()) if len(accepted) else None,
                                'center_std_y': float(accepted[:, 1].std()) if len(accepted) else None,
                                'max_displacement_from_seed': float(np.linalg.norm(accepted-accepted[0], axis=1).max()) if len(accepted) else None},
            'tracker': {'kind': spec['candidate'], 'configuration': spec, 'redetection': False},
            'claims': {'independent_ground_truth': False, 'accuracy_estimate': None,
                       'physical_plate_coordinates': False, 'catcher_intent_verified': False},
            'limitations': ['Only motion of seed-ROI features is tracked; surviving features need not belong to a glove.',
                           'Median translation transports the seed ROI center; changing glove shape may move its visual center differently.',
                           'Quality gates are engineering consistency checks, not calibrated correctness probabilities.',
                           'Same four reused montage windows and assistant-unreviewed endpoint estimates are not an independent test set.']}


def track_flow_video(raw_annotation, spec):
    import cv2

    annotation = validate_annotation(raw_annotation)
    path = resolve_annotation_clip(annotation)
    if hashlib.sha256(path.read_bytes()).hexdigest() != annotation['clip_sha256']:
        raise ValueError('Approved media SHA differs from frozen annotation')
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Approved media cannot be decoded')
        fps, count = float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError('Valid constant-rate video metadata required')
        dimensions = {'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        if dimensions != annotation['image_dimensions'] or annotation['release_time'] >= count/fps:
            raise ValueError('Annotation dimensions/release boundary differ from video')
        first, last = math.ceil(annotation['seed_time']*fps), math.floor(annotation['end_time']*fps)
        if not 0 <= first < last < count or last-first+1 > MAX_FRAMES:
            raise ValueError('Invalid bounded pre-release frame interval')
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        def frames():
            for index in range(first, last+1):
                ok, image = capture.read()
                if not ok:
                    raise ValueError('Decode failed inside requested frame window')
                yield index/fps, image
        result = track_flow_frames(frames(), annotation, spec, expected_frame_count=last-first+1)
        result['video'] = {'fps': fps, 'frame_count': count, 'requested_first_frame': first, 'requested_last_frame': last,
                           'timestamp_basis': 'constant_frame_rate_frame_index_divided_by_fps'}
        result['opencv_version'] = cv2.__version__
        return result
    finally:
        capture.release()


def compare_endpoint(result, reference, uncertainty_pixels):
    if not result['complete_window'] or result['pixel_target'] is None:
        return {'available': False, 'distance_pixels': None, 'within_reference_uncertainty': None,
                'reference_uncertainty_pixels': uncertainty_pixels, 'reason': 'abstained_before_full_window_end'}
    distance = math.hypot(result['pixel_target']['x']-reference['x'], result['pixel_target']['y']-reference['y'])
    return {'available': True, 'distance_pixels': distance,
            'within_reference_uncertainty': distance <= uncertainty_pixels,
            'reference_uncertainty_pixels': uncertainty_pixels,
            'scope': 'Distance to an assistant-unreviewed approximate endpoint, not physical CV accuracy'}
