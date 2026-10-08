"""Synthetic opt-in service checks; no real feed, fit, or recommendation policy."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.intent import intent_estimate
from observer_app.live_armb import state_key
from observer_app.live_feed import parse_state
from observer_app.live_store import LiveSnapshotStore
from observer_app.service_research import (capture_input, conditional_candidates,
                                           explain_conditional, reviewed_setup_vs_delivery)


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def test_recommendation_receipt_keeps_exact_version_and_real_arrival(tmp_path):
    store = LiveSnapshotStore(tmp_path/'research.sqlite3')
    base = {'metaData': {'eventWall': 100., 'timeStamp': '20261008_000100'}, 'state': {'balls': 1}}
    _, first = store.ingest(7, base, fetched=110., source='live_poll')
    captured = capture_input(first, {'balls': 1}, game_pk=7, decision_at=iso(112.), trusted_clock=True)
    corrected = copy.deepcopy(base)
    corrected['state']['balls'] = 2
    _, second = store.ingest(7, corrected, fetched=120., source='live_poll')
    assert (first['seq'], first['version'], second['seq'], second['version']) == (1, 1, 2, 2)
    assert captured['feed_sha256'] == first['sha'] != second['sha']
    assert captured['feed_receipt_status'] == 'verified_local_receipt' and captured['input_provenance'] == 'unverified'
    assert captured['version_to_decision_ms'] == 2000
    assert store.restore(7, 130., 60)[0]['seq'] == 2
    with pytest.raises(ValueError, match='future correction'):
        capture_input(second, {'balls': 2}, game_pk=7, decision_at=iso(112.), trusted_clock=True)
    with pytest.raises(ValueError, match='timezone-aware'):
        capture_input(first, {'balls': 1}, game_pk=7, decision_at='1970-01-01T00:01:52', trusted_clock=True)
    with pytest.raises(ValueError, match='another game'):
        capture_input(first, {'balls': 1}, game_pk=8, decision_at=iso(112.), trusted_clock=True)


def test_recorded_feed_parser_to_decision_snapshot_is_explicitly_unverified(tmp_path):
    path = Path(__file__).parent/'fixtures/live/849845_20260929_195940.json'
    feed = json.loads(path.read_text())
    feed['metaData']['eventWall'] = 100.
    store = LiveSnapshotStore(tmp_path/'recorded.sqlite3')
    _, entry = store.ingest(849845, feed, fetched=110., source='recorded:test_fixture')
    parsed = parse_state(entry['feed'], None)
    payload = {'state_key': state_key(entry['feed']), 'parsed': parsed}
    receipt = capture_input(entry, payload, game_pk=849845, decision_at=iso(112.), trusted_clock=False)
    assert receipt['feed_sha256'] == entry['sha'] and receipt['input']['state_key'] == payload['state_key']
    assert receipt['first_to_decision_ms'] == 2000 and receipt['feed_receipt_status'] == 'unverified'
    assert receipt['input_provenance'] == 'unverified'


def test_replay_is_unverified_and_probability_adapter_guards_support(tmp_path):
    store = LiveSnapshotStore(tmp_path/'replay.sqlite3')
    _, entry = store.ingest(7, {'metaData': {'eventWall': 100.}}, fetched=110., source='recorded:file.json')
    replay = capture_input(entry, {'balls': 1}, game_pk=7, decision_at=iso(112.), trusted_clock=True)
    vectors = np.zeros((2, 199))
    calls = []
    def predict(x):
        calls.append(len(x))
        return np.full((len(x), 10), .1)
    kwargs = dict(model_sha256='a'*64, encoder_sha256='b'*64, support_counts=[40, 2])
    assert conditional_candidates(replay, vectors, ['FF', 'SL'], predict, **kwargs)['reason'] == 'input_lineage_unverified'
    assert not calls
    _, live = store.ingest(8, {'metaData': {'eventWall': 100.}}, fetched=110., source='live_poll')
    receipt = capture_input(live, {'balls': 1}, game_pk=8, decision_at=iso(112.), trusted_clock=True)
    evidence = {'kind': 'synthetic_fixture', 'input_sha256': receipt['input_sha256'],
                'vectors_sha256': hashlib.sha256(np.ascontiguousarray(vectors.astype(np.float64)).tobytes()).hexdigest(),
                'actions_sha256': hashlib.sha256(json.dumps(['FF', 'SL'], separators=(',', ':')).encode()).hexdigest(),
                'model_sha256': kwargs['model_sha256'], 'encoder_sha256': kwargs['encoder_sha256']}
    result = conditional_candidates(receipt, vectors, ['FF', 'SL'], predict,
                                    input_evidence=evidence, **kwargs)
    assert result['status'] == 'conditional_fixture' and calls == [2]
    assert result['candidates'][0]['probabilities'] == [.1]*10
    assert result['candidates'][1]['probabilities'] is None
    assert result['causal_effect'] is False and result['policy_adopted'] is False
    text = explain_conditional(result['candidates'][0], 'strike', receipt=receipt)['text']
    assert '10.0%' in text and '40건' in text and '모델 추정' in text
    receipt['input']['balls'] = 2
    with pytest.raises(ValueError, match='input changed'):
        conditional_candidates(receipt, vectors, ['FF', 'SL'], predict,
                               input_evidence=evidence, **kwargs)
    receipt['input']['balls'] = 1
    wrong = {**evidence, 'vectors_sha256': '0'*64}
    assert conditional_candidates(receipt, vectors, ['FF', 'SL'], predict,
                                  input_evidence=wrong, **kwargs)['reason'] == 'input_lineage_unverified'


def test_cv_proxy_never_becomes_intended_target_or_command_error():
    estimate = intent_estimate(pitch_id='p1', clip_id='synthetic', clip_sha256='a'*64,
                               evidence={'frame_index': 1, 'frame_time': 1.},
                               method={'kind': 'synthetic', 'version': '1'},
                               provenance={'label_source': 'video_module', 'review_status': 'unreviewed'},
                               pixel_point={'x': 50, 'y': 40}, image_dimensions={'width': 100, 'height': 80})
    mismatched = reviewed_setup_vs_delivery(estimate, {'x': 0., 'z': 2.}, 'plate_feet')
    assert mismatched['reason'] == 'frame_mismatch' and mismatched['distance'] is None
    same = reviewed_setup_vs_delivery(estimate, {'x': 55., 'y': 42.}, 'image_pixels')
    assert same['is_intended_target'] is False and same['is_command_error'] is False
