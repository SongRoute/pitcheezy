"""Generated-only publisher output consumed by the pinned public viewer on owned localhost TLS."""
import hashlib
import json
from pathlib import Path
import subprocess

from test_publisher_replay import CONTRACT, LIVE_GAME, LIMITS, capture_db, publisher_replay

REPO = Path(__file__).resolve().parents[4]
WEB = REPO/'apps/observer/web'
BUILD = Path('/Users/song/Projects/pitcheezy-pm-jobs/20261016-parallel-execution-001/service/browser-review/dist-public')


def test_public_viewer_consumes_synthetic_publisher_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '1')
    monkeypatch.delenv('PITCHEEZY_OBSERVER_REC_LEDGER', raising=False)
    job = tmp_path/'job'
    capture = capture_db(job/'input')
    capture_sha = hashlib.sha256(capture.read_bytes()).hexdigest()
    run = job/'run-001'
    run.mkdir()
    rows, info = publisher_replay.capture_rows(capture, LIVE_GAME, 1 << 26)
    schemas = {path.name.split('.')[0]: json.loads(path.read_text()) for path in (CONTRACT/'schema').glob('*.schema.json')}
    try:
        summary = publisher_replay.run(rows=rows, info=info, run_dir=run, game_pk=LIVE_GAME,
            commit='56b4347baa9aef4d8c09e23ecd87bee6a20a823f', limits=LIMITS,
            policy_factory=None, schemas=schemas, log=lambda _: None)
    finally:
        info['db'].close()
    assert summary['status'] == 'COMPLETED' and not summary['incomplete']
    assert summary['schema_validation']['validated'] and not summary['schema_validation']['problems']
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == capture_sha
    output = run/'out'
    assert sorted(str(path.relative_to(output)) for path in (output/'v1').rglob('*') if path.is_file()) == [
        'v1/dates.json', f'v1/game/{LIVE_GAME}.json', 'v1/games/2026-10-04.json',
        'v1/games/2026-10-05.json', 'v1/games/2026-10-06.json', 'v1/health.json', 'v1/manifest.json']
    game = json.loads((output/f'v1/game/{LIVE_GAME}.json').read_text())
    assert game['game']['game_pk'] == LIVE_GAME and game['publish']['contract'] == 'v1'

    private_key, certificate = tmp_path/'key.pem', tmp_path/'cert.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
        '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost',
        '-keyout', str(private_key), '-out', str(certificate)], capture_output=True, check=True, timeout=20)
    receipt = tmp_path/'browser-receipt.json'
    process = subprocess.run(['node', str(WEB/'tests/public-viewer-e2e-smoke.mjs'), str(BUILD), str(output),
        str(LIVE_GAME), str(certificate), str(private_key), str(receipt)],
        capture_output=True, text=True, timeout=90, cwd=WEB)
    assert process.returncode == 0, process.stdout + '\n' + process.stderr
    result = json.loads(receipt.read_text())
    assert result['status'] == 'PASS_SYNTHETIC_LOCAL_HTTPS' and result['ownedServerClosed']
    assert [case['kind'] for case in result['scenarios']] == [
        'browser_consumed_runner_output', 'browser_consumed_runner_output', 'synthetic_error_override']
    assert result['runnerHashes'][f'v1/game/{LIVE_GAME}.json'] == hashlib.sha256(
        (output/f'v1/game/{LIVE_GAME}.json').read_bytes()).hexdigest()
