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
# Files that determine the public build bytes; the source pin requires these to equal the pinned commit, with no
# untracked or ignored extras. `vite --mode public` also reads .env, .env.local and .env.public.local from this folder
# (absent at the pinned commit, so any copy on disk refuses) and lets shell VITE_* variables win (refused below).
WEB_BUILD_INPUTS = ('index.html', 'public', 'src', 'package.json', 'package-lock.json', 'tsconfig.json',
                    'vite.config.ts', '.env', '.env.local', '.env.public', '.env.public.local')
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
    # Browser smoke run by both E2E tests; it receives this candidate via PUBLIC_E2E_CANDIDATE (no built-in table).
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
        # A custom candidate names its own smoke script, so that script must be pinned too.
        digest = candidate.get('smoke', {}).get('sha256') if isinstance(candidate.get('smoke'), dict) else None
        if not (isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)):
            raise PinMismatch(f'{CANDIDATE_ENV} candidate must pin smoke.sha256 (64 hex)')
    else:
        candidate, origin = DEFAULT_CANDIDATE, {'source': 'DEFAULT_CANDIDATE'}
    if candidate.get('schema') != CANDIDATE_SCHEMA:
        raise PinMismatch(f'candidate schema must be {CANDIDATE_SCHEMA}')
    for key in ('source_commit', 'public_build', 'gateway', 'smoke'):
        if key not in candidate:
            raise PinMismatch(f'candidate missing {key}')
    return candidate, origin


def verify_source(candidate, repo=REPO, env=None):
    """Web build inputs in this checkout (working tree included, no untracked or ignored extras, no shell VITE_*
    overrides) must equal the pinned source commit."""
    env = os.environ if env is None else env
    overrides = sorted(key for key in env if key.startswith('VITE_'))
    if overrides:
        raise PinMismatch(f'shell VITE_* variables would override the pinned build env: {overrides[:5]}')
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
    extra = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all',
                            '--ignored=matching', '--', *specs],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    if extra:
        raise PinMismatch(f'source pin mismatch: untracked or ignored web build inputs: {extra[:5]}')
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


def verify_smoke(candidate):
    """Smoke script path and SHA pin (required for OBSERVER_E2E_CANDIDATE, optional for DEFAULT_CANDIDATE);
    the smoke itself re-checks the build from the candidate JSON."""
    smoke = _regular(_path(candidate['smoke']['path']))
    if candidate['smoke'].get('sha256') and sha(smoke) != candidate['smoke']['sha256']:
        raise PinMismatch('smoke script pin mismatch')
    return smoke


def pinned_candidate(tmp_path):
    """All pins checked up front; any mismatch fails the test before a browser or gateway starts.
    The exact candidate JSON handed to the smoke (PUBLIC_E2E_CANDIDATE) is written under tmp_path."""
    try:
        candidate, origin = load_candidate()
        source = verify_source(candidate)
        build, build_hashes = verify_build(candidate)
        smoke = verify_smoke(candidate)
    except PinMismatch as error:
        pytest.fail(f'REFUSED candidate pin: {error}')
    candidate_json = tmp_path/'candidate.json'
    # `origin` (candidate JSON path + sha, or DEFAULT_CANDIDATE) and the smoke sha travel to the smoke receipt.
    origin = {**origin, 'smoke_sha256': sha(smoke)}
    candidate_json.write_text(json.dumps({**candidate, 'origin': origin}, sort_keys=True, indent=1) + '\n')
    return {'candidate': candidate, 'origin': origin, 'source': source, 'build': build,
            'build_hashes': build_hashes, 'smoke': smoke, 'json': candidate_json}


def smoke_env(pinned):
    return {**os.environ, 'PUBLIC_E2E_CANDIDATE': str(pinned['json'])}


def assert_receipt_build(result, pinned):
    """The smoke's own receipt must report the candidate's asset hashes (served bytes == pinned bytes)."""
    served = {key.lstrip('/'): digest for key, digest in result['buildHashes'].items()}
    assert served == pinned['build_hashes'], (served, pinned['build_hashes'])
    assert result['candidateOrigin'] == pinned['origin'], (result.get('candidateOrigin'), pinned['origin'])
    assert result['candidateSha256'] == sha(pinned['json'])


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
    smoke.write_text('// synthetic smoke\n')
    candidate = {'schema': CANDIDATE_SCHEMA, 'source_commit': DEFAULT_CANDIDATE['source_commit'],
        'public_build': {'dir': str(build), 'members': members},
        'gateway': {'commit': '0'*40, 'runtime_dir': str(runtime), 'files': {'gateway.mjs': files['gateway.mjs']},
                    'adapter': {'name': 'local-https-gateway.mjs', 'sha256': files['local-https-gateway.mjs']}},
        'smoke': {'path': str(smoke), 'sha256': sha(smoke)}}
    manifest = tmp_path/'candidate.json'
    manifest.write_text(json.dumps(candidate))
    loaded, origin = load_candidate({CANDIDATE_ENV: str(manifest)})
    assert loaded == candidate and origin['sha256'] == sha(manifest)
    _, hashes = verify_build(loaded)
    verify_gateway(loaded)
    verify_smoke(loaded)
    assert hashes == {name: pin['sha256'] for name, pin in members.items()}

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
    with pytest.raises(PinMismatch, match='smoke script pin mismatch'):
        verify_smoke({**loaded, 'smoke': {'path': str(smoke), 'sha256': '0'*64}})
    with pytest.raises(PinMismatch, match='source pin'):
        verify_source({**loaded, 'source_commit': 'f'*40}, env={})
    # 56b4347 predates the public static build inputs: the web sources differ, so the pin refuses.
    with pytest.raises(PinMismatch, match='source pin mismatch'):
        verify_source({**loaded, 'source_commit': '56b4347baa9aef4d8c09e23ecd87bee6a20a823f'}, env={})
    manifest.write_text(json.dumps({**candidate, 'schema': 'other'}))
    with pytest.raises(PinMismatch, match='schema'):
        load_candidate({CANDIDATE_ENV: str(manifest)})
    # A custom candidate must pin its smoke script: missing or malformed smoke.sha256 refuses.
    for smoke_pin in ({'path': str(smoke)}, {'path': str(smoke), 'sha256': 'abc'}):
        manifest.write_text(json.dumps({**candidate, 'smoke': smoke_pin}))
        with pytest.raises(PinMismatch, match='must pin smoke.sha256'):
            load_candidate({CANDIDATE_ENV: str(manifest)})
    assert load_candidate({})[1] == {'source': 'DEFAULT_CANDIDATE'}


def test_source_pin_refuses_untracked_env_and_shell_overrides(tmp_path):
    """Synthetic git checkout: clean inputs pass; an untracked source file, a vite .env*(.local) file (tracked-ignored
    or not) or a shell VITE_* variable refuses."""
    repo = tmp_path/'repo'
    web = repo/'apps/observer/web'
    (web/'src').mkdir(parents=True)
    (web/'src/main.ts').write_text('export {};\n')
    (web/'.env.public').write_text('VITE_PUBLISH_BASE=.\n')
    (repo/'.gitignore').write_text('.env\n')
    git = ['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', '-c', 'commit.gpgsign=false']
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run([*git, 'add', '-A'], check=True)
    subprocess.run([*git, 'commit', '-qm', 'pin'], check=True)
    commit = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    candidate = {'source_commit': commit}
    assert verify_source(candidate, repo=repo, env={}) == commit
    (repo/'unrelated.txt').write_text('outside the web build inputs\n')
    assert verify_source(candidate, repo=repo, env={'PATH': '/usr/bin'}) == commit

    for extra in ('src/new.ts', '.env.local', '.env.public.local', '.env'):
        (web/extra).write_text('VITE_PUBLISH_BASE=https://elsewhere.invalid\n')
        with pytest.raises(PinMismatch, match='untracked or ignored web build inputs'):
            verify_source(candidate, repo=repo, env={})
        (web/extra).unlink()
    verify_source(candidate, repo=repo, env={})

    (web/'src/main.ts').write_text('export const changed = 1;\n')
    with pytest.raises(PinMismatch, match='web build inputs differ'):
        verify_source(candidate, repo=repo, env={})
    subprocess.run([*git, 'checkout', '-q', '--', 'apps/observer/web/src/main.ts'], check=True)
    with pytest.raises(PinMismatch, match='shell VITE_'):
        verify_source(candidate, repo=repo, env={'VITE_PUBLISH_BASE': 'https://elsewhere.invalid'})


def test_smoke_refuses_without_or_with_wrong_candidate(tmp_path):
    """The browser smoke has no built-in asset table: no candidate JSON, or a pin that disagrees, exits before any server."""
    build = tmp_path/'build'
    build.mkdir()
    (build/'index.html').write_bytes(b'synthetic index')
    smoke = WEB/'tests/public-viewer-e2e-smoke.mjs'
    argv = ['node', str(smoke), str(build), str(tmp_path/'out'), str(LIVE_GAME), 'cert', 'key', str(tmp_path/'r.json')]
    env = {key: value for key, value in os.environ.items() if key != 'PUBLIC_E2E_CANDIDATE'}
    missing = subprocess.run(argv, capture_output=True, text=True, timeout=30, cwd=WEB, env=env)
    assert missing.returncode != 0 and 'PUBLIC_E2E_CANDIDATE' in missing.stderr
    wrong = tmp_path/'candidate.json'
    wrong.write_text(json.dumps({'schema': CANDIDATE_SCHEMA, 'public_build': {'members': {
        'index.html': {'bytes': 15, 'sha256': '0'*64}}}}))
    mismatch = subprocess.run(argv, capture_output=True, text=True, timeout=30, cwd=WEB,
                              env={**env, 'PUBLIC_E2E_CANDIDATE': str(wrong)})
    assert mismatch.returncode != 0 and 'public build index.html' in mismatch.stderr
    assert not (tmp_path/'r.json').exists()


def test_public_viewer_consumes_synthetic_publisher_bytes(tmp_path, monkeypatch):
    pinned = pinned_candidate(tmp_path)
    output = synthetic_publication(tmp_path, monkeypatch)
    private_key, certificate = localhost_certificate(tmp_path)
    receipt = tmp_path/'browser-receipt.json'
    process = subprocess.run(['node', str(pinned['smoke']), str(pinned['build']), str(output),
        str(LIVE_GAME), str(certificate), str(private_key), str(receipt)],
        capture_output=True, text=True, timeout=90, cwd=WEB, env=smoke_env(pinned))
    assert process.returncode == 0, process.stdout + '\n' + process.stderr
    result = json.loads(receipt.read_text())
    assert result['status'] == 'PASS_SYNTHETIC_LOCAL_HTTPS' and result['ownedServerClosed']
    assert [case['kind'] for case in result['scenarios']] == [
        'browser_consumed_runner_output', 'browser_consumed_runner_output', 'synthetic_error_override']
    assert result['runnerHashes'][f'v1/game/{LIVE_GAME}.json'] == hashlib.sha256(
        (output/f'v1/game/{LIVE_GAME}.json').read_bytes()).hexdigest()
    assert_receipt_build(result, pinned)
