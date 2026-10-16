"""Generated receipt-only check; no captured body or model asset access."""
import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest
from scripts import fullseason_captured_replay as replay


def test_one_member_and_exact_regular_file(tmp_path):
    body = b'{"ok":true}'
    blob = gzip.compress(body)
    path = tmp_path / 'body.gz'
    path.write_bytes(blob)
    assert replay.inflate_one(replay.exact_file(path, 100), 100) == body
    with pytest.raises(ValueError): replay.inflate_one(blob + gzip.compress(body), 100)
    with pytest.raises(ValueError): replay.inflate_one(blob, len(body) - 1)
    alias = tmp_path / 'alias.gz'; alias.symlink_to(path)
    with pytest.raises(ValueError): replay.exact_file(alias, 100)
    for game, date in ((849810, '2026-10-11'), (849809, '2026-10-12')):
        wrong = {'gameData': {'game': {'pk': game}, 'datetime': {'officialDate': date}}}
        with pytest.raises(ValueError, match='game/officialDate'):
            replay.feed_identity(json.dumps(wrong).encode())


def test_receipt_order_selection_and_unknown_abstention(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, 'MAX_FILES', 2)
    capture = tmp_path / 'capture'; capture.mkdir()
    files = []
    for seq in (1, 2):
        body = json.dumps({'receipt': seq, 'gameData': {'game': {'pk': 849809},
                          'datetime': {'officialDate': '2026-10-11'}}}).encode()
        name = f'849809_{seq:06d}.json.gz'
        (capture / name).write_bytes(gzip.compress(body))
        files.append({'file': name, 'seq': seq, 'version': 1,
                      'sha256': hashlib.sha256(body).hexdigest(), 'fetched': float(seq),
                      'first_fetched': float(seq), 'event_wall': float(seq), 'source': 'generated'})
    manifest = capture / 'MANIFEST.json'
    manifest.write_text(json.dumps({'game_pk': 849809, 'status': 'OK', 'files': files}))
    allow = tmp_path / 'allow.json'
    allow.write_text(json.dumps({'manifest': {'path': str(manifest), 'sha256': replay.sha(manifest)},
                                 'root': str(capture), 'files': files}))
    config = tmp_path / 'adapter.json'
    config.write_text(json.dumps({'schema': 'ws-fullseason-shadow-v1', 'seeds': [0, 1, 2],
                                  'leaf_order': ['a'] * 7, 'threads': 1}))
    normalizer = tmp_path / 'bridge.py'; normalizer.write_text('generated')
    out = tmp_path / 'out'; out.mkdir()
    reg = {'bounds': {'wall_seconds': 300}, 'allowlist': {'path': str(allow), 'sha256': replay.sha(allow)},
           'manifest': {'path': str(manifest), 'sha256': replay.sha(manifest)}, 'capture_root': str(capture),
           'adapter_config': {'path': str(config), 'sha256': replay.sha(config)},
           'leaf_order': ['a'] * 7, 'normalizer_source': str(normalizer),
           'normalizer_sha256': replay.sha(normalizer), 'profile_bundle': {'directory': str(tmp_path),
           'manifest_sha256': '0' * 64}, 'output': str(out), 'source_commit': 'generated',
           'capture_compressed_bytes_reported': sum((capture / f['file']).stat().st_size for f in files),
           'capture_inflated_bytes_reported': sum(len(json.dumps({'receipt': f['seq'], 'gameData':
              {'game': {'pk': 849809}, 'datetime': {'officialDate': '2026-10-11'}}}).encode()) for f in files)}
    monkeypatch.setattr(replay, 'gated', lambda *_: reg)

    class Runtime:
        bundle_sha256 = '1' * 64
        def __init__(self, *_): pass
        def predict(self, snapshot):
            return {'status': snapshot['status'], 'reason': snapshot['reason'],
                    'call_sha256': snapshot['snapshot_sha256'], 'probabilities': None}
    monkeypatch.setattr(replay, 'Runtime', Runtime)

    from pitcheezy.seq_shadow.fullseason_stream import CONTEXT
    query = dict(zip(CONTEXT, (849809, 1, 2, '2026-10-11', 10, 20, 'FF', 1, 0, 0,
                               1, 'Top', 0, 0, None, None, None, 'R', 'R')))
    def step(feed, **kw):
        event = {'event_id': 'pitch-1', 'key': (849809, 1, 1), 'revision': feed['receipt'],
                 'row': {'status': 'unknown', 'reason': 'unverified', 'accepted': None}}
        candidate = {'decision_id': 'candidate-1', 'query': query,
                     'coverage': {'complete': True, 'before_key': [849809, 1, 2],
                                  'source_sha256': kw['source_sha256']},
                     'pitch_type_basis': 'hypothetical_candidate'}
        return {'events': [event], 'candidate': candidate, 'abstain_reason': None}
    monkeypatch.setattr(replay.captured_bridge, 'step', step)
    regpath = tmp_path / 'reg'; regpath.write_text('{}')
    grantpath = tmp_path / 'grant'; grantpath.write_text('{}')
    with (tmp_path / 'lock').open('w') as lock:
        monkeypatch.setenv('PITCHEEZY_FULLSEASON_LOCK_FD', str(lock.fileno()))
        try: replay.run(regpath, grantpath)
        finally: replay.signal.alarm(0)
    receipt = json.loads((out / 'receipt.pending.json').read_text())
    assert receipt['counts']['bodies'] == 2
    assert receipt['counts']['event_new'] == 1
    assert receipt['counts']['unchanged_observation'] == 1
    assert receipt['counts']['decision_abstain'] == 1
    assert receipt['counts']['repeated_candidate_key'] == 1
    assert receipt['immutable_restart_checks'] == 1
    assert receipt['prediction_ready'] is False
    assert receipt['raw_probabilities_written'] is False
