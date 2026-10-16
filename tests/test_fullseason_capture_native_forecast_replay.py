"""Generated forecast replay through one verified body; no actual feed or model."""
import gzip
import hashlib
import json
import sqlite3

from pitcheezy.seq_shadow import fullseason
from pitcheezy.seq_shadow.ledger import sha
from scripts import fullseason_capture_native_replay as native
from test_captured_bridge import feed


def test_generated_forecast_runner_persists_and_reopens_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(native.replay, 'MAX_FILES', 1)
    monkeypatch.setattr(native.replay, 'MAX_QUERIES', 1)
    body = feed()
    body['gameData']['game']['pk'] = native.replay.GAME
    body['liveData']['plays']['currentPlay']['playEvents'][0]['details']['type'] = {'code': 'FF'}
    payload = json.dumps(body).encode()
    capture = tmp_path / 'capture'
    capture.mkdir()
    name = f'{native.replay.GAME}_000001.json.gz'
    (capture / name).write_bytes(gzip.compress(payload))
    files = [{'file': name, 'seq': 1, 'version': 1,
              'sha256': hashlib.sha256(payload).hexdigest(), 'fetched': 1.0}]
    manifest = capture / 'MANIFEST.json'
    manifest.write_text(json.dumps({'game_pk': native.replay.GAME, 'status': 'OK', 'files': files}))
    allow = tmp_path / 'allow.json'
    allow.write_text(json.dumps({'manifest': {'path': str(manifest), 'sha256': native.file_sha(manifest)},
                                 'root': str(capture), 'files': files}))
    cfg = {'schema': 'ws-fullseason-shadow-v1', 'seeds': [0, 1, 2],
           'leaf_order': list(fullseason.LEAVES), 'threads': 1,
           'state': {seed: {'sha256': char * 64}
                     for seed, char in (('0', 'c'), ('1', 'd'), ('2', 'e'))},
           'encoder': {'sha256': 'f' * 64}, 'calibrator': {'sha256': '1' * 64},
           'recipe': {'sha256': '2' * 64}}
    config = tmp_path / 'adapter.json'
    config.write_text(json.dumps(cfg))
    output = tmp_path / 'out'
    output.mkdir()
    reg = {'bounds': {'wall_seconds': 300}, 'forecast_persistence': 'append_only_native_forecast_v1',
           'allowlist': {'path': str(allow), 'sha256': native.file_sha(allow)},
           'manifest': {'path': str(manifest), 'sha256': native.file_sha(manifest)},
           'capture_root': str(capture),
           'adapter_config': {'path': str(config), 'sha256': native.file_sha(config)},
           'leaf_order': list(fullseason.LEAVES),
           'profile_bundle': {'directory': str(tmp_path), 'manifest_sha256': 'b' * 64},
           'normalizer_sha256': 'a' * 64, 'output': str(output),
           'source_commit': 'generated',
           'capture_compressed_bytes_reported': (capture / name).stat().st_size,
           'capture_inflated_bytes_reported': len(payload)}
    monkeypatch.setattr(native, 'gated', lambda *_: reg)

    class Runtime:
        calls = 0
        def __init__(self, config, _, manifest_sha):
            self.identity = {'adapter_config_sha256': sha(config),
                             'profile_manifest_sha256': manifest_sha}
            self.bundle_sha256 = sha(self.identity)
        def predict(self, snapshot):
            Runtime.calls += 1
            return {'status': 'scored', 'stream_evidence_kind': native.captured_native_state.NATIVE_KIND,
                    'prospective_claim': False,
                    'call_sha256': sha({'bundle': self.bundle_sha256,
                                        'snapshot': snapshot['snapshot_sha256']}),
                    'probabilities': [1 / 7] * 7}
    monkeypatch.setattr(native, 'Runtime', Runtime)
    regpath = tmp_path / 'registration.json'
    grantpath = tmp_path / 'grant.json'
    regpath.write_text('{}')
    grantpath.write_text('{}')
    with (tmp_path / 'lock').open('w') as lock:
        monkeypatch.setenv('PITCHEEZY_FULLSEASON_LOCK_FD', str(lock.fileno()))
        try:
            native.run(regpath, grantpath, persist_forecasts=True)
        finally:
            native.signal.alarm(0)
    receipt = json.loads((output / 'receipt.pending.json').read_text())
    assert receipt['forecast_intents'] == receipt['forecast_terminal_results'] == 1
    assert receipt['selected_keys_sha256'] and receipt['forecast_results_sha256']
    assert receipt['immutable_restart_checks'] == 1
    assert receipt['strict_accepted_rows'] == 0 and not receipt['prediction_ready']
    assert Runtime.calls == receipt['score_compute_latency']['count']
    with sqlite3.connect(output / 'native_stream.sqlite3') as db:
        assert db.execute('SELECT COUNT(*) FROM native_forecast_intents').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM native_forecast_results').fetchone()[0] == 1
        result = json.loads(db.execute('SELECT body FROM native_forecast_results').fetchone()[0])
        assert len(result['probabilities']) == 7
