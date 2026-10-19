"""Generated v1 bytes through the owned upload worker, the pinned PR42 localhost gateway, and the pinned public viewer.

Candidate pins (source, public build, gateway runtime, smoke) come from test_public_viewer_e2e.load_candidate:
``OBSERVER_E2E_CANDIDATE`` JSON or its DEFAULT_CANDIDATE. All pins are checked before the gateway starts.
"""
import hashlib
import json
import os
import select
import subprocess
import time

import pytest

from observer_app.publisher import HttpSink, LocalDirSink, MirrorSink, Publisher, load_config
from test_public_viewer_e2e import (PinMismatch, WEB, assert_receipt_build, localhost_certificate,
    pinned_candidate, smoke_env, synthetic_publication, verify_gateway)
from test_publisher_replay import LIVE_GAME

TOKEN = 'synthetic-upload-token-never-persisted-0000000000'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publisher_upload_gateway_and_viewer_bytes(tmp_path, monkeypatch):
    pinned = pinned_candidate(tmp_path)
    try:
        _, adapter = verify_gateway(pinned['candidate'])
    except PinMismatch as error:
        pytest.fail(f'REFUSED candidate pin: {error}')
    output = synthetic_publication(tmp_path, monkeypatch)
    private_key, certificate = localhost_certificate(tmp_path)

    env = {key: value for key, value in os.environ.items() if not key.startswith(('PITCHEEZY_', 'TEST_'))}
    env.update(TEST_TLS_KEY=str(private_key), TEST_TLS_CERT=str(certificate), TEST_MODE='ok')
    gateway = subprocess.Popen(['node', str(adapter)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    calls = None
    try:
        assert select.select([gateway.stdout], [], [], 10)[0]
        port = json.loads(gateway.stdout.readline())['port']
        assert 0 < port < 65536
        monkeypatch.setenv('SSL_CERT_FILE', str(certificate))
        monkeypatch.setenv('PITCHEEZY_REHEARSAL_UPLOAD_TOKEN', TOKEN)
        config = load_config(env={}, out_dir=str(tmp_path/'mirror'), state_path=str(tmp_path/'mirror.state.json'))
        config['upload'] |= {'enabled': True, 'base_url': f'https://localhost:{port}/_publish',
            'token_env': 'PITCHEEZY_REHEARSAL_UPLOAD_TOKEN', 'timeout_s': 5,
            'operation_deadline_s': 5, 'shutdown_grace_s': 0, 'retries': 0, 'backoff_s': 0}
        remote = HttpSink(config['upload'])
        sink = MirrorSink(LocalDirSink(tmp_path/'mirror'), remote, background=True)
        owner = Publisher(None, sink, config=config, state_path=tmp_path/'mirror.state.json')
        paths = sorted(str(path.relative_to(output)) for path in (output/'v1').rglob('*.json'))
        assert len(paths) == 7
        try:
            for path in paths:
                sink.write(path, (output/path).read_bytes(), final=path.startswith('v1/game/'))
            deadline = time.monotonic()+25
            while time.monotonic() < deadline:
                with owner._state_lock:
                    if not sink.pending:
                        break
                    assert all(row.get('phase') not in ('blocked', 'rejected')
                        for row in owner.state['upload_attempts'].values())
                time.sleep(.01)
            assert not sink.pending, 'upload worker did not settle'
        finally:
            owner.close()
        assert owner._lock is None and sink.worker is not None
        assert sink.worker.thread is None or not sink.worker.thread.is_alive()
        for path in paths:
            head = remote.head(path)
            assert not head['deleted'] and head['sha256'] == sha(output/path)

        receipt = tmp_path/'gateway-browser-receipt.json'
        browser = subprocess.run(['node', str(pinned['smoke']), str(pinned['build']), str(output),
            str(LIVE_GAME), str(certificate), str(private_key), str(receipt), str(port)],
            capture_output=True, text=True, timeout=90, cwd=WEB, env=smoke_env(pinned))
        assert browser.returncode == 0, browser.stdout + '\n' + browser.stderr
        result = json.loads(receipt.read_text())
        assert result['status'] == 'PASS_SYNTHETIC_LOCAL_HTTPS' and result['ownedServerClosed']
        assert result['transport'] == 'pinned_local_gateway_via_front'
        assert all(row['gatewayStored'] for row in result['requests'] if row['path'].startswith('/v1/') and row['status'] == 200)
        assert result['runnerHashes'][f'v1/game/{LIVE_GAME}.json'] == sha(output/f'v1/game/{LIVE_GAME}.json')
        assert_receipt_build(result, pinned)
    finally:
        if gateway.poll() is None:
            gateway.stdin.close()
            try:
                gateway.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gateway.terminate(); gateway.wait(timeout=2)
        assert gateway.returncode == 0, gateway.stderr.read()
        calls = json.loads(gateway.stdout.readline())['requests']
        assert len(calls) <= 80
        (tmp_path/'gateway-requests-aggregate.json').write_text(json.dumps({
            'total': len(calls), 'put': sum(row['method'] == 'PUT' for row in calls),
            'head': sum(row['method'] == 'HEAD' for row in calls),
            'get': sum(row['method'] == 'GET' for row in calls),
            'all_gets_cookie_present': all(row['cookie_present'] for row in calls if row['method'] == 'GET'),
            'owned_gateway_exit': gateway.returncode}, sort_keys=True)+'\n')
    assert calls is not None and sum(row['method'] == 'PUT' for row in calls) == 7
    assert any(row['method'] == 'GET' and row['path'] == f'/v1/game/{LIVE_GAME}.json' for row in calls)
    assert all(row['cookie_present'] for row in calls if row['method'] == 'GET')
