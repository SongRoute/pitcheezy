"""Generated-only publisher output consumed by an explicitly pinned public viewer build on owned localhost TLS.

The tested candidate (source commit, public build members, PR42 gateway runtime, browser smoke) is an explicit
input: ``OBSERVER_E2E_CANDIDATE=<path to JSON>`` overrides ``DEFAULT_CANDIDATE`` below, which is the
2026-10-19 launch candidate (dccde71 rebuild; byte-identical to partner pack v4.3.1 / 6282c63). Every pin is
checked before any browser or gateway process starts, and a mismatch refuses the run instead of testing
something else. Final candidate run (post 10/21 UI) = new JSON or new defaults; see
docs/pm/service/final-candidate-e2e-receipt.md.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from test_publisher_replay import CONTRACT, LIVE_GAME, LIMITS, capture_db, publisher_replay

REPO = Path(__file__).resolve().parents[4]
WEB = REPO/'apps/observer/web'
CANDIDATE_ENV = 'OBSERVER_E2E_CANDIDATE'
CANDIDATE_SCHEMA = 'pitcheezy-public-e2e-candidate-v1'
# Files that determine the public build bytes; the source pin requires these to equal the pinned commit.
WEB_BUILD_INPUTS = ('index.html', 'public', 'src', 'package.json', 'package-lock.json', 'tsconfig.json',
                    'vite.config.ts', '.env.public')
PR42_RUNTIME = '/Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/compat-v4.3/runtime'
DEFAULT_CANDIDATE = {
    'schema': CANDIDATE_SCHEMA,
    'label': 'ws-launch-2026 candidate 2026-10-19 (dccde71 build:public == pack v4.3.1 public-build)',
    'source_commit': 'dccde711748d40565a78c50356cf8cb5c4f43d3e',
    'public_build': {
        'dir': 'apps/observer/web/dist-public',  # repo-relative or absolute; `npm run build:public`
        'members': {
            'index.html': {'bytes': 659, 'sha256': '9c354a586b990873942cd40c3d4168f2a88d50225de9b7a65c7f799b1728513d'},
            'favicon.svg': {'bytes': 249, 'sha256': 'bbfd72e5d077bd545b2cf568e5267c6637598d5253746f5b395e0d5f3fb144d9'},
            'assets/index-BLGE1MgF.js': {'bytes': 287672,
                'sha256': '50ead108da5d8ada2a003c1ae2538d15ae43cf6b471676e5f4b64bef9575fcb6'},
            'assets/index-CUX1xjau.css': {'bytes': 103215,
                'sha256': '90bb1d702395964f4eae9993407c85424f772bba729e412253fb9fd2d1e7cd18'},
        },
    },
    'gateway': {
        'commit': '48ec406589982547e298ff2ee4bc2d64ed0216c0',  # PR42 runtime pin carried by 6282c63 / pack v4.3.1
        'runtime_dir': PR42_RUNTIME,
        'files': {
            'worker.mjs': 'ba6bb83bbbbd8ba4727feb6b870fbe85c0c8c5c9bc65ef837aa2703235f3f985',
            'gateway.mjs': '3fd0217beae30ae23ca548c309e2535b66e80024728b10ddb249eec08f2dd2b6',
            'notice.mjs': 'd94b5c22ef66b38b3bb824e028fcaa92075856014bbc7e2e28bd3a85bdb64247',
            'access.mjs': '67c3865be7754a9cf172075db67f60a8bc2cd076203799b0bf64a8fe0ef1219a',
            'telemetry.mjs': 'c90d0f23dbf80eb965d343bef8fb75129cf3ce7ee9e64fbe6f683228ad0f646a',
        },
        'adapter': {'name': 'local-https-gateway.mjs',
                    'sha256': '470c3cf7d760b83011af6f3177ba0421b8d978a420c87a0cdf93c17eff546864'},
    },
    # Browser smoke run by both E2E tests; its internal asset table must name the same build (checked below).
    'smoke': {'path': 'apps/observer/web/tests/public-viewer-e2e-smoke.mjs'},
}


class PinMismatch(AssertionError):
    """An explicit candidate input disagrees with the bytes on disk: refuse, never fall back."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO/path


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise PinMismatch(f'pinned file missing, not regular, or a symlink: {path}')
    return path


def load_candidate(env=None):
    """Explicit candidate JSON from OBSERVER_E2E_CANDIDATE, else DEFAULT_CANDIDATE; returns (candidate, origin)."""
    env = os.environ if env is None else env
    if env.get(CANDIDATE_ENV):
        path = _regular(Path(env[CANDIDATE_ENV]))
        raw = path.read_bytes()
        candidate, origin = json.loads(raw), {'source': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    else:
        candidate, origin = DEFAULT_CANDIDATE, {'source': 'DEFAULT_CANDIDATE'}
    if candidate.get('schema') != CANDIDATE_SCHEMA:
        raise PinMismatch(f'candidate schema must be {CANDIDATE_SCHEMA}')
    for key in ('source_commit', 'public_build', 'gateway', 'smoke'):
        if key not in candidate:
            raise PinMismatch(f'candidate missing {key}')
    return candidate, origin


def verify_source(candidate, repo=REPO):
    """Web build inputs in this checkout (working tree included) must equal the pinned source commit."""
    commit = candidate['source_commit']
    if not (isinstance(commit, str) and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)):
        raise PinMismatch('source_commit must be a full 40-hex commit')
    if subprocess.run(['git', '-C', str(repo), 'cat-file', '-e', f'{commit}^{{commit}}'],
                      capture_output=True).returncode != 0:
        raise PinMismatch(f'source pin {commit} not present in {repo}')
    specs = [f'apps/observer/web/{name}' for name in WEB_BUILD_INPUTS]
    diff = subprocess.run(['git', '-C', str(repo), 'diff', '--name-only', commit, '--', *specs],
                          capture_output=True, text=True, check=True).stdout.split()
    if diff:
        raise PinMismatch(f'source pin mismatch: web build inputs differ from {commit[:12]}: {diff[:5]}')
    return commit


def verify_build(candidate):
    """Exactly the pinned members, regular files, matching byte length and SHA-256."""
    build = candidate['public_build']
    root, members = _path(build['dir']), build['members']
    if not members or 'index.html' not in members:
        raise PinMismatch('public_build.members must include index.html')
    if not root.is_dir() or root.is_symlink():
        raise PinMismatch(f'public build dir missing (run `npm run build:public`?): {root}')
    on_disk = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink())
    if on_disk != sorted(members):
        raise PinMismatch(f'public build members mismatch: on disk {on_disk}, pinned {sorted(members)}')
    for name, pin in members.items():
        path = _regular(root/name)
        if path.stat().st_size != pin['bytes'] or sha(path) != pin['sha256']:
            raise PinMismatch(f'public build pin mismatch: {name}')
    return root, {name: pin['sha256'] for name, pin in members.items()}


def verify_gateway(candidate):
    """Exact PR42 runtime file hashes plus the owned loopback adapter; returns (runtime dir, adapter path)."""
    gateway = candidate['gateway']
    runtime = _path(gateway['runtime_dir'])
    if not gateway.get('commit') or not gateway.get('files'):
        raise PinMismatch('gateway pin needs commit and files')
    for name, digest in gateway['files'].items():
        if sha(_regular(runtime/name)) != digest:
            raise PinMismatch(f'gateway pin mismatch: {name}')
    adapter = _regular(runtime/gateway['adapter']['name'])
    if sha(adapter) != gateway['adapter']['sha256']:
        raise PinMismatch('gateway adapter pin mismatch')
    return runtime, adapter


def verify_smoke(candidate, build_hashes):
    """The browser smoke hard-codes its own asset table; refuse unless it names exactly the candidate build."""
    smoke = _regular(_path(candidate['smoke']['path']))
    if candidate['smoke'].get('sha256') and sha(smoke) != candidate['smoke']['sha256']:
        raise PinMismatch('smoke script pin mismatch')
    text = smoke.read_text()
    missing = sorted(name for name, digest in build_hashes.items() if digest not in text)
    if missing:
        raise PinMismatch(f'smoke {smoke.name} pins a different public build (missing {missing}); '
                          'the smoke must accept the candidate pins before this candidate can be tested')
    return smoke


def pinned_candidate():
    """All pins checked up front; any mismatch fails the test before a browser or gateway starts."""
    try:
        candidate, origin = load_candidate()
        source = verify_source(candidate)
        build, build_hashes = verify_build(candidate)
        smoke = verify_smoke(candidate, build_hashes)
    except PinMismatch as error:
        pytest.fail(f'REFUSED candidate pin: {error}')
    return {'candidate': candidate, 'origin': origin, 'source': source, 'build': build,
            'build_hashes': build_hashes, 'smoke': smoke}


def assert_receipt_build(result, pinned):
    """The smoke's own receipt must report the candidate's asset hashes (served bytes == pinned bytes)."""
    served = {key.lstrip('/'): digest for key, digest in result['buildHashes'].items()}
    assert served == pinned['build_hashes'], (served, pinned['build_hashes'])


def synthetic_publication(tmp_path, monkeypatch):
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
    return output


def localhost_certificate(tmp_path):

    private_key, certificate = tmp_path/'key.pem', tmp_path/'cert.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
        '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost',
        '-keyout', str(private_key), '-out', str(certificate)], capture_output=True, check=True, timeout=20)
    return private_key, certificate


def test_candidate_pins_refuse_mismatch(tmp_path):
    """Synthetic controls: a valid pin set passes; tampered build, source, gateway or smoke inputs refuse."""
    build = tmp_path/'build'
    (build/'assets').mkdir(parents=True)
    members = {}
    for name in ('index.html', 'favicon.svg', 'assets/index-x.js', 'assets/index-x.css'):
        data = f'synthetic {name}'.encode()
        (build/name).write_bytes(data)
        members[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    runtime = tmp_path/'runtime'
    runtime.mkdir()
    files = {}
    for name in ('gateway.mjs', 'local-https-gateway.mjs'):
        (runtime/name).write_text(f'// synthetic {name}\n')
        files[name] = sha(runtime/name)
    smoke = tmp_path/'smoke.mjs'
    smoke.write_text(''.join(f"'{pin['sha256']}'\n" for pin in members.values()))
    candidate = {'schema': CANDIDATE_SCHEMA, 'source_commit': DEFAULT_CANDIDATE['source_commit'],
        'public_build': {'dir': str(build), 'members': members},
        'gateway': {'commit': '0'*40, 'runtime_dir': str(runtime), 'files': {'gateway.mjs': files['gateway.mjs']},
                    'adapter': {'name': 'local-https-gateway.mjs', 'sha256': files['local-https-gateway.mjs']}},
        'smoke': {'path': str(smoke)}}
    manifest = tmp_path/'candidate.json'
    manifest.write_text(json.dumps(candidate))
    loaded, origin = load_candidate({CANDIDATE_ENV: str(manifest)})
    assert loaded == candidate and origin['sha256'] == sha(manifest)
    _, hashes = verify_build(loaded)
    verify_gateway(loaded)
    verify_smoke(loaded, hashes)

    (build/'index.html').write_text('changed')
    with pytest.raises(PinMismatch, match='public build pin mismatch'):
        verify_build(loaded)
    (build/'index.html').write_bytes(b'synthetic index.html')
    (build/'extra.js').write_text('x')
    with pytest.raises(PinMismatch, match='members mismatch'):
        verify_build(loaded)
    (build/'extra.js').unlink()
    (build/'assets/index-x.js').unlink()
    (build/'assets/index-x.js').symlink_to(build/'index.html')
    with pytest.raises(PinMismatch, match='symlink'):
        verify_build(loaded)
    (build/'assets/index-x.js').unlink()
    (build/'assets/index-x.js').write_bytes(b'synthetic assets/index-x.js')
    verify_build(loaded)

    (runtime/'gateway.mjs').write_text('// tampered\n')
    with pytest.raises(PinMismatch, match='gateway pin mismatch'):
        verify_gateway(loaded)
    smoke.write_text("'0000'\n")
    with pytest.raises(PinMismatch, match='pins a different public build'):
        verify_smoke(loaded, hashes)
    with pytest.raises(PinMismatch, match='source pin'):
        verify_source({**loaded, 'source_commit': 'f'*40})
    # 56b4347 predates the public static build inputs: the web sources differ, so the pin refuses.
    with pytest.raises(PinMismatch, match='source pin mismatch'):
        verify_source({**loaded, 'source_commit': '56b4347baa9aef4d8c09e23ecd87bee6a20a823f'})
    manifest.write_text(json.dumps({**candidate, 'schema': 'other'}))
    with pytest.raises(PinMismatch, match='schema'):
        load_candidate({CANDIDATE_ENV: str(manifest)})


def test_public_viewer_consumes_synthetic_publisher_bytes(tmp_path, monkeypatch):
    pinned = pinned_candidate()
    output = synthetic_publication(tmp_path, monkeypatch)
    private_key, certificate = localhost_certificate(tmp_path)
    receipt = tmp_path/'browser-receipt.json'
    process = subprocess.run(['node', str(pinned['smoke']), str(pinned['build']), str(output),
        str(LIVE_GAME), str(certificate), str(private_key), str(receipt)],
        capture_output=True, text=True, timeout=90, cwd=WEB)
    assert process.returncode == 0, process.stdout + '\n' + process.stderr
    result = json.loads(receipt.read_text())
    assert result['status'] == 'PASS_SYNTHETIC_LOCAL_HTTPS' and result['ownedServerClosed']
    assert [case['kind'] for case in result['scenarios']] == [
        'browser_consumed_runner_output', 'browser_consumed_runner_output', 'synthetic_error_override']
    assert result['runnerHashes'][f'v1/game/{LIVE_GAME}.json'] == hashlib.sha256(
        (output/f'v1/game/{LIVE_GAME}.json').read_bytes()).hexdigest()
    assert_receipt_build(result, pinned)
