"""Generated native replay custody/immutability check; no real payload or model."""
import copy
import gzip
import hashlib
import json

import pytest

from scripts import fullseason_capture_native_replay as native
from test_captured_bridge import feed


def test_generated_native_replay_strict_unknown_and_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(native.replay, 'MAX_FILES', 2)
    monkeypatch.setattr(native.replay, 'MAX_QUERIES', 2)
    first = feed(); first['gameData']['game']['pk'] = native.replay.GAME
    play = first['liveData']['plays']['currentPlay']
    play['playEvents'][0]['details']['type'] = {'code': 'FF'}
    second = copy.deepcopy(first)
    play2 = second['liveData']['plays']['currentPlay']
    event2 = copy.deepcopy(play2['playEvents'][0])
    event2.update(playId='p2', pitchNumber=2)
    event2['details'].update(type={'code': 'FF'}, call={'code': 'C'})
    event2['count'].update(balls=1, strikes=1)
    play2['playEvents'].append(event2)
    play2['count'].update(balls=1, strikes=1)
    capture = tmp_path / 'capture'; capture.mkdir()
    files = []
    for seq, body in enumerate((first, second), 1):
        payload = json.dumps(body).encode()
        name = f'{native.replay.GAME}_{seq:06d}.json.gz'
        (capture / name).write_bytes(gzip.compress(payload))
        files.append({'file': name, 'seq': seq, 'version': 1,
                      'sha256': hashlib.sha256(payload).hexdigest(), 'fetched': float(seq)})
    manifest = capture / 'MANIFEST.json'
    manifest.write_text(json.dumps({'game_pk': native.replay.GAME, 'status': 'OK', 'files': files}))
    allow = tmp_path / 'allow.json'
    allow.write_text(json.dumps({'manifest': {'path': str(manifest), 'sha256': native.file_sha(manifest)},
                                 'root': str(capture), 'files': files}))
    config = tmp_path / 'adapter.json'
    config.write_text(json.dumps({'schema': 'ws-fullseason-shadow-v1', 'seeds': [0, 1, 2],
                                  'leaf_order': ['a'] * 7, 'threads': 1}))
    out = tmp_path / 'out'; out.mkdir()
    reg = {'bounds': {'wall_seconds': 300}, 'allowlist': {'path': str(allow), 'sha256': native.file_sha(allow)},
           'manifest': {'path': str(manifest), 'sha256': native.file_sha(manifest)},
           'capture_root': str(capture), 'adapter_config': {'path': str(config), 'sha256': native.file_sha(config)},
           'leaf_order': ['a'] * 7, 'profile_bundle': {'directory': str(tmp_path),
           'manifest_sha256': '0' * 64}, 'normalizer_sha256': 'a' * 64,
           'output': str(out), 'source_commit': 'generated',
           'capture_compressed_bytes_reported': sum((capture / f['file']).stat().st_size for f in files),
           'capture_inflated_bytes_reported': sum(len(json.dumps(b).encode()) for b in (first, second))}
    monkeypatch.setattr(native, 'gated', lambda *_: reg)

    class Runtime:
        bundle_sha256 = 'b' * 64
        def __init__(self, *_): pass
        def predict(self, snapshot):
            return {'status': 'scored', 'stream_evidence_kind': native.captured_native_state.NATIVE_KIND,
                    'prospective_claim': False, 'call_sha256': snapshot['snapshot_sha256'],
                    'probabilities': [1 / 7] * 7}
    monkeypatch.setattr(native, 'Runtime', Runtime)
    regpath = tmp_path / 'registration.json'; regpath.write_text('{}')
    grantpath = tmp_path / 'grant.json'; grantpath.write_text('{}')
    with (tmp_path / 'lock').open('w') as lock:
        monkeypatch.setenv('PITCHEEZY_FULLSEASON_LOCK_FD', str(lock.fileno()))
        try: native.run(regpath, grantpath)
        finally: native.signal.alarm(0)
    receipt = json.loads((out / 'receipt.pending.json').read_text())
    assert receipt['counts']['bodies'] == 2
    assert receipt['counts']['strict_new'] == 2
    assert receipt['counts']['strict_unchanged_observation'] == 1
    assert receipt['counts']['strict_decision_abstain'] == 2
    assert receipt['counts']['native_decision_native_ready'] == 2
    assert receipt['counts']['experimental_experimental_conditional_score'] == 2
    assert receipt['context']['selected_known_tokens_sum'] == 3
    assert receipt['context']['prior_unknown_relevant_sum'] == 0
    assert receipt['context']['previous_pitch_unknown'] == 0
    assert receipt['immutable_restart_checks'] == 2
    assert receipt['strict_accepted_rows'] == 0
    assert receipt['corrected_reference_reads'] == 0
    assert receipt['prediction_ready'] is False
    assert receipt['raw_probabilities_written'] is False


def test_native_row_custody_rejects_promotion():
    row = {'key': (native.replay.GAME, 1, 1), 'input_contract': native.CONTRACT,
           'accepted': False, 'statcast_training_equivalent': False,
           'receipt_seq': 1, 'source_sha256': 'a' * 64}
    assert native._native_map([row], receipt_seq=1, source_sha256='a' * 64) == {(native.replay.GAME, 1, 1): row}
    for field, value in (('accepted', True), ('statcast_training_equivalent', True),
                         ('receipt_seq', 2), ('source_sha256', 'b' * 64)):
        altered = dict(row, **{field: value})
        with pytest.raises(ValueError, match='custody or acceptance'):
            native._native_map([altered], receipt_seq=1, source_sha256='a' * 64)


def test_composite_normalizer_provenance_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(native, 'ROOT', tmp_path)
    name = 'src/pitcheezy/seq_shadow/captured_native_state.py'
    path = tmp_path / name; path.parent.mkdir(parents=True)
    path.write_text('generated native source')
    digest = native.file_sha(path)
    sources = {name: digest}
    reg = {'input_contract': native.CONTRACT, 'oracle_absence': native.ORACLE_ABSENCE,
           'normalizer_sha_semantics': native.NORMALIZER_SHA_SEMANTICS,
           'normalizer_sha256': native.object_sha(sources), 'native_source_pins': sources,
           'source_pins': {str(path): digest}, 'data_role': '2026 EXPOSED DEVELOPMENT'}
    monkeypatch.setattr(native.replay, 'gated', lambda *_: reg)
    assert native.gated(tmp_path/'registration', tmp_path/'grant') is reg
    reg['normalizer_source'] = str(path)
    with pytest.raises(ValueError, match='source/oracle contract'):
        native.gated(tmp_path/'registration', tmp_path/'grant')
    reg.pop('normalizer_source')
    reg['normalizer_sha_semantics'] = 'file_sha256'
    with pytest.raises(ValueError, match='source/oracle contract'):
        native.gated(tmp_path/'registration', tmp_path/'grant')
