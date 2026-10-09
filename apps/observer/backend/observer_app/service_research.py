"""Isolated, opt-in service research adapters. No production route or policy is changed.

The caller must supply an immutable saved feed version and a pre-pitch input built
from that version. A replay file's order is never treated as its field arrival time.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math

from .intent import setup_actual_difference, validate_intent_estimate
from .live_store import _canonical
from .domain import PITCH_LABELS

# ``LiveSnapshotStore`` provider order -> abstention reason; only 'ordered' is fresh, anything else abstains.
FRESHNESS = {'ordered': None, 'stale': 'stale_provider_stamp', 'unorderable': 'unorderable_provider_stamp',
             'same_stamp_conflict': 'same_stamp_conflict_provider_stamp'}


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('A timezone-aware decision timestamp is required')
    return parsed.astimezone(timezone.utc)


def capture_input(entry, pre_pitch_input, *, game_pk, decision_at, trusted_clock=False, disconnected=False):
    """Copy the exact input and store-version identity before an inference call.

    ``trusted_clock`` means the supplied first/version receipt times came from a
    trusted local clock; replay file mtime and provider timestamps do not qualify.
    ``disconnected`` means the feed's last poll failed and ``entry`` is only the last snapshot served: an
    otherwise fresh receipt abstains with ``feed_disconnected_last_snapshot_only``.
    A memory-only ``LiveFeed`` entry (store down) carries no committed identity and raises ``ValueError``.
    """
    decision = _utc(decision_at)
    if entry.get('game_pk') != int(game_pk):
        raise ValueError('Feed version belongs to another game')
    raw = _canonical(entry['feed'])
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry.get('sha'):
        raise ValueError('Feed bytes differ from the committed store version')
    first, version_at = entry['fetched'], entry['version_fetched']
    if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (first, version_at)):
        raise ValueError('Missing or invalid observer receipt time')
    if first > version_at or version_at > decision.timestamp():
        raise ValueError('Recommendation cannot see a future correction')
    if type(entry.get('seq')) is not int or type(entry.get('version')) is not int:
        raise ValueError('Committed version identity is required')
    copied = json.loads(_bytes(pre_pitch_input))
    # A newer game event selects a slot whatever its provider stamp; a verified local receipt clock says nothing
    # about provider revision order, so only an 'ordered' store version counts as fresh prediction input.
    order = entry.get('provider_order')
    freshness = FRESHNESS.get(order, 'provider_order_unknown') if isinstance(order, str) else 'provider_order_unknown'
    if disconnected is not False:  # anything but an explicit False is treated as disconnected
        freshness = freshness or 'feed_disconnected_last_snapshot_only'
    return {'provider_order': order, 'freshness_reason': freshness, 'feed_disconnected': disconnected is not False,
            'schema': 'service-input-receipt-v1', 'game_pk': int(game_pk),
            'store_seq': entry['seq'], 'slot_version': entry['version'], 'feed_sha256': digest,
            'input': copied, 'input_sha256': hashlib.sha256(_bytes(copied)).hexdigest(),
            'first_received_at': datetime.fromtimestamp(first, timezone.utc).isoformat(),
            'version_received_at': datetime.fromtimestamp(version_at, timezone.utc).isoformat(),
            'provider_generated_at': (entry['feed'].get('metaData') or {}).get('timeStamp'),
            'decision_at': decision.isoformat(), 'source': entry['source'],
            'feed_receipt_status': 'verified_local_receipt' if entry['source'] == 'live_poll' and trusted_clock else 'unverified',
            'input_provenance': 'unverified',
            'first_to_decision_ms': round(1000*(decision.timestamp()-first), 3),
            'version_to_decision_ms': round(1000*(decision.timestamp()-version_at), 3)}


def conditional_candidates(receipt, vectors, actions, predict, *, model_sha256, encoder_sha256,
                           support_counts, minimum_support=30, input_evidence=None):
    """Evaluate given F1 vectors; never select, rank, or claim intervention effects.

    A caller must pin the model/encoder and prove the 199-vector construction. The
    provider returns ten outcome probabilities per candidate in ``actions`` order.
    """
    import numpy as np
    x = np.asarray(vectors, dtype=np.float64)
    if x.shape != (len(actions), 199) or not np.isfinite(x).all():
        raise ValueError('Expected finite F1 199-feature vectors per candidate')
    if not actions or any(type(action) is not str or action not in PITCH_LABELS for action in actions):
        raise ValueError('Known pitch-type actions required')
    if len(set(actions)) != len(actions) or len(support_counts) != len(actions):
        raise ValueError('Candidate identities/support must align')
    if type(minimum_support) is not int or minimum_support <= 0:
        raise ValueError('Positive support threshold required')
    if not all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
               for v in (model_sha256, encoder_sha256)):
        raise ValueError('Pinned model and encoder SHA-256 required')
    if hashlib.sha256(_bytes(receipt['input'])).hexdigest() != receipt['input_sha256']:
        raise ValueError('Recommendation-time input changed after capture')
    vector_sha = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    action_sha = hashlib.sha256(_bytes(list(actions))).hexdigest()
    expected = {'input_sha256': receipt['input_sha256'], 'vectors_sha256': vector_sha,
                'actions_sha256': action_sha, 'model_sha256': model_sha256,
                'encoder_sha256': encoder_sha256}
    if not input_evidence or any(input_evidence.get(key) != value for key, value in expected.items()):
        return {'status': 'unsupported', 'reason': 'input_lineage_unverified', 'candidates': []}
    if input_evidence.get('kind') not in ('synthetic_fixture', 'audited_offline_book96'):
        return {'status': 'unsupported', 'reason': 'input_lineage_unverified', 'candidates': []}
    if (stale := receipt.get('freshness_reason', 'provider_order_unknown')):  # abstain: not fresh input
        return {'status': 'unsupported', 'reason': stale, 'candidates': []}
    probabilities = np.asarray(predict(x), dtype=np.float64)
    if probabilities.shape != (len(actions), 10) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError('Predictor must return normalized ten-class probabilities')
    rows = []
    for action, count, distribution in zip(actions, support_counts, probabilities):
        if type(count) is not int or count < 0:
            raise ValueError('Support count must be a nonnegative integer')
        rows.append({'pitch_type': action, 'status': 'supported' if count >= minimum_support else 'unsupported',
                     'reason': None if count >= minimum_support else 'low_train_support',
                     'train_support_n': count, 'probabilities': distribution.tolist() if count >= minimum_support else None})
    return {'status': 'conditional_fixture' if input_evidence['kind'] == 'synthetic_fixture' else 'conditional_offline_estimates',
            'input_evidence_kind': input_evidence['kind'], 'input_sha256': receipt['input_sha256'],
            'vectors_sha256': vector_sha, 'actions_sha256': action_sha,
            'feed_sha256': receipt['feed_sha256'], 'store_seq': receipt['store_seq'],
            'model_sha256': model_sha256, 'encoder_sha256': encoder_sha256,
            'outcome_order': ['ball', 'strike', 'foul', 'out', 'single', 'double', 'triple',
                              'home_run', 'hbp', 'double_play'], 'candidates': rows,
            'causal_effect': False, 'policy_adopted': False, 'live_input_verified': False}


def reviewed_setup_vs_delivery(estimate, actual_point, actual_frame):
    """Compare a CV setup proxy with realized delivery only in a shared frame."""
    valid = validate_intent_estimate(estimate)
    if valid['status'] != 'estimated':
        return {'status': 'unavailable', 'reason': 'setup_unavailable', 'is_intended_target': False}
    frame = valid['deepest_frame']
    return setup_actual_difference(valid['points'][frame], frame, actual_point, actual_frame) | {
        'setup_source': 'cv_unreviewed_proxy', 'is_intended_target': False,
        'actual_source': 'realized_delivery'}


def explain_conditional(row, outcome, *, receipt):
    """One short numerical display sentence for a supported conditional estimate."""
    if row['status'] != 'supported' or row['probabilities'] is None:
        return {'status': 'unsupported', 'text': '이 구종은 학습 표본이 부족해 확률을 표시하지 않습니다.'}
    order = ('ball', 'strike', 'foul', 'out', 'single', 'double', 'triple', 'home_run', 'hbp', 'double_play')
    index = order.index(outcome)
    value = row['probabilities'][index]
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid probability')
    return {'status': 'example',
            'text': f"{row['pitch_type']} 조건에서 {outcome} 모델 추정 {value*100:.1f}%. "
                    f"투구 1건당 결과 확률이며 학습 지지 표본은 {row['train_support_n']}건입니다. "
                    '실제 빈도나 선택 효과는 아닙니다.',
            'probability': value, 'event_unit': 'per_pitch', 'support_n': row['train_support_n'],
            'causal_effect': False}
