#!/usr/bin/env python3
"""Finite dedicated-key HTTPS rehearsal. Default is offline validation; execution requires a bound grant.
No real feed/model, credential logging, host toggles, automatic retry or uncertainty reconciliation.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'apps/observer/backend'))
from observer_app.publisher import HttpSink, LocalDirSink, MirrorSink, Publisher, UploadError, load_config

SOURCE_ALLOWLIST = ('apps/observer/backend/observer_app/publisher.py',
    'apps/observer/backend/observer_app/upload_worker.py', 'apps/observer/publish_config.json',
    'apps/observer/scripts/host_rehearsal.py')
PUBLISHER_BASE = '1f0260fa0d2da40b229507d5ca15f45d7c13593d'
GATEWAY = '87168d7bc42f11f76886e475f919caff7674b60a'
MAX_CALLS, WALL_SECONDS, OP_SECONDS, MAX_BODY = 18, 110, 5, 2048  # supervisor: 120s including reap


class Stop(RuntimeError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path, limit):
    if path.stat().st_size > limit:raise Stop('pinned_file_too_large')
    with path.open('rb') as handle:data = handle.read(limit+1)
    if len(data) > limit:raise Stop('pinned_file_too_large')
    return digest(data)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def read_small(path):
    with open(path, 'rb') as handle:
        raw = handle.read(65537)
    if len(raw) > 65536:
        raise Stop('input_too_large')
    return json.loads(raw)


def validate(plan, grant=None):
    if (plan.get('schema') != 'pitcheezy-dedicated-https-rehearsal-v1'
            or plan.get('publisher_base') != PUBLISHER_BASE or plan.get('gateway_commit') != GATEWAY
            or not re.fullmatch(r'[a-z0-9-]{8,64}', plan.get('run_id', ''))
            or not re.fullmatch(r'[1-9][0-9]{0,11}', str(plan.get('game_key', '')))):
        raise Stop('plan_invalid')
    urls = [urlsplit(plan.get(name, '')) for name in ('upload_base', 'public_origin')]
    for url in urls:
        if (url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment
                or any(ord(c) <= 32 or ord(c) >= 127 for c in url.geturl())):
            raise Stop('target_invalid')
        url.port
    if urls[0].path != '/_publish' or urls[1].path not in ('', '/') or urls[0].netloc != urls[1].netloc:
        raise Stop('target_not_same_origin')
    if plan.get('scope') not in ('local_synthetic', 'approved_https_synthetic'):
        raise Stop('scope_invalid')
    if plan['scope'] == 'local_synthetic' and any(u.hostname not in ('localhost', '127.0.0.1') for u in urls):
        raise Stop('local_target_invalid')
    if plan['scope'] != 'local_synthetic' and (os.environ.get('SSL_CERT_FILE') or os.environ.get('SSL_CERT_DIR')):
        raise Stop('custom_tls_trust_not_allowed')
    for name in ('token_env', 'cookie_env'):
        if not re.fullmatch(r'PITCHEEZY_REHEARSAL_[A-Z_]{1,48}', plan.get(name, '')):
            raise Stop('credential_name_invalid')
    if plan['token_env'] == plan['cookie_env']:
        raise Stop('credential_name_invalid')
    pins = plan.get('source_sha256', {})
    if set(pins) != set(SOURCE_ALLOWLIST):
        raise Stop('source_pin_set_invalid')
    for relative in SOURCE_ALLOWLIST:
        if file_digest(ROOT/relative, 1024*1024) != pins[relative]:
            raise Stop('source_pin_mismatch')
    assets = plan.get('public_assets', {})
    if (set(assets) != {'index.html', 'favicon.svg'} | {p for p in assets if re.fullmatch(r'assets/index-[A-Za-z0-9_-]+\.(js|css)', p)}
            or not 4 <= len(assets) <= 8):
        raise Stop('build_pin_set_invalid')
    build = Path(plan['public_build']).resolve()
    for relative, sha in assets.items():
        path = build/relative
        if path.is_symlink() or not path.resolve().is_relative_to(build) or file_digest(path, 8*1024*1024) != sha:
            raise Stop('build_pin_mismatch')
    if grant is not None:
        required = ('execute_synthetic_only', 'exclusive_writer', 'dedicated_target', 'read_upload_enabled', 'cookie_access_allowed')
        if (grant.get('plan_sha256') != digest(canonical(plan)) or not all(grant.get(k) is True for k in required)
                or not isinstance(grant.get('authority'), str) or not grant['authority'].strip()
                or type(grant.get('expires_epoch')) not in (int, float)
                or not time.time() < grant['expires_epoch'] <= time.time()+3600):
            raise Stop('grant_missing_or_invalid')
    return {'plan_sha256': digest(canonical(plan)), 'target_sha256': digest(canonical([u.geturl() for u in urls])),
        'source_bound': True, 'build_bound': True,
        'token_present': bool(os.environ.get(plan['token_env'])), 'access_cookie_present': bool(os.environ.get(plan['cookie_env']))}


def public_child():
    """One bounded GET, no redirects/proxy, fixed metadata output only; parent kills/reaps on DNS/time limit."""
    connection = None
    try:
        request = json.loads(sys.stdin.buffer.read(8193))
        url = urlsplit(request['url'])
        cookie = os.environ[request['cookie_env']]
        if not cookie or len(cookie) > 4096 or any(ord(c) < 32 or ord(c) > 126 for c in cookie):
            raise Stop('cookie_invalid')
        connection = http.client.HTTPSConnection(url.hostname, url.port, timeout=OP_SECONDS)
        connection.request('GET', url.path, headers={'Cookie': cookie, 'Cache-Control': 'no-cache'})
        response = connection.getresponse()
        body = response.read(MAX_BODY+1)
        if len(body) > MAX_BODY:
            raise Stop('public_body_too_large')
        etag = response.getheader('ETag', '')
        if etag and re.fullmatch(r'"[a-f0-9]{32}"', etag) is None:
            raise Stop('public_etag_invalid')
        print(json.dumps({'status': response.status, 'etag': etag, 'sha256': digest(body), 'bytes': len(body)}))
    except BaseException:
        print(json.dumps({'error': 'public_probe_failed'}))
    finally:
        if connection is not None:
            connection.close()


def execute(plan, grant, output):
    readiness = validate(plan, grant)
    if plan.get('execute_enabled') is not True:
        raise Stop('execution_disabled')
    if not readiness['token_present'] or not readiness['access_cookie_present']:
        raise Stop('credentials_missing')
    output = Path(output)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)  # never reuse another run's state/output
    receipt = readiness | {'schema': 'pitcheezy-rehearsal-receipt-v1', 'run_id': plan['run_id'],
        'result': 'STOPPED', 'game_key': str(plan['game_key']), 'requests': [], 'remote_toggle_changes': 0, 'remote_cleanup': 'not_started',
        'last_confirmed_generation': None,
        'deployment_binding': 'operator_attested_not_account_verified', 'credentials_recorded': False}
    deadline = time.monotonic()+WALL_SECONDS
    config = load_config(env={}, out_dir=str(output/'out'), state_path=str(output/'publisher.state.json'))
    config['upload'] |= {'enabled': True, 'base_url': plan['upload_base'], 'token_env': plan['token_env'],
        'timeout_s': OP_SECONDS, 'operation_deadline_s': OP_SECONDS, 'shutdown_grace_s': 0}
    sink = MirrorSink(LocalDirSink(output/'out'), HttpSink(config['upload']), background=True)
    owner = Publisher(None, sink, config=config, state_path=output/'publisher.state.json')
    worker = sink.worker
    request = worker.request
    path = f"v1/game/{plan['game_key']}.json"
    own_etag, fenced, fatal = None, False, False

    def reserve(method):
        if time.time() >= grant['expires_epoch']:raise Stop('grant_expired')
        if len(receipt['requests']) >= MAX_CALLS or time.monotonic() >= deadline:
            raise Stop('budget_exhausted')
        receipt['requests'].append({'method': method, 'result': 'started'})
        return receipt['requests'][-1]

    def bounded(method, relative, **options):
        nonlocal own_etag, fatal
        row = reserve(method)
        worker.deadline = max(1, min(OP_SECONDS, deadline-time.monotonic()))
        try:
            if fenced and method in ('PUT', 'DELETE') and options.get('expected') != own_etag:
                raise UploadError('rehearsal_condition_changed', outcome='blocked')
            result = request(method, relative, **options)
            if fenced and method == 'HEAD' and result['etag'] != own_etag:
                raise UploadError('rehearsal_generation_changed', outcome='blocked')
            if fenced and method in ('PUT', 'DELETE'):
                own_etag = result['etag']  # ownership comes from this mutation receipt, never a fresh HEAD
                receipt['last_confirmed_generation'] = dict(result)
            row['result'] = 'confirmed'
            return result
        except UploadError as error:
            row['result'] = str(error)
            fatal = True
            raise
    worker.request = bounded

    def settled():
        while time.monotonic() < deadline:
            if fatal:
                raise Stop('request_unconfirmed')
            with owner._state_lock:
                if path not in sink.pending:
                    return
                attempt = owner.state['upload_attempts'].get(path, {})
                if attempt.get('phase') in ('blocked', 'rejected', 'retry'):
                    raise Stop('mutation_unconfirmed')
            time.sleep(.01)
        raise Stop('budget_exhausted')

    def public(expected, etag):
        row = reserve('GET')
        descriptor = {'url': plan['public_origin'].rstrip('/')+'/'+path, 'cookie_env': plan['cookie_env']}
        try:
            probe = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_public'], input=canonical(descriptor),
                capture_output=True, timeout=max(.1, min(OP_SECONDS, deadline-time.monotonic())), check=True)
            result = json.loads(probe.stdout)
        except (subprocess.SubprocessError, ValueError):
            raise Stop('public_probe_failed') from None
        row['result'] = 'confirmed' if (result.get('status') == 200 and result.get('sha256') == digest(expected)
            and result.get('etag') == etag) else 'mismatch'
        if row['result'] != 'confirmed':
            raise Stop('public_receipt_mismatch')

    try:
        if bounded('HEAD', path)['etag'] is not None:
            raise Stop('existing_remote_object')  # tombstones count as existing; never adopt them
        body = canonical({'schema': 'pitcheezy-upload-rehearsal-v1', 'synthetic': True, 'run_id': plan['run_id'], 'sequence': 1})
        assert len(body) <= MAX_BODY
        fenced = True
        sink.write(path, body);settled()
        fenced = False
        first = bounded('HEAD', path)
        if first['etag'] != own_etag:raise Stop('ownership_changed')
        if first['sha256'] != digest(body) or first['deleted']:
            raise Stop('head_receipt_mismatch')
        public(body, own_etag)
        fenced = True
        sink.delete(path);settled()
        fenced = False
        tombstone = bounded('HEAD', path)
        if tombstone['etag'] != own_etag:raise Stop('ownership_changed')
        if not tombstone['deleted'] or own_etag == first['etag']:
            raise Stop('tombstone_mismatch')
        fenced = True
        sink.write(path, body);settled()
        fenced = False
        restored = bounded('HEAD', path)
        if restored['etag'] != own_etag:raise Stop('ownership_changed')
        if restored['deleted'] or restored['sha256'] != digest(body) or own_etag in (first['etag'], tombstone['etag']):
            raise Stop('restore_mismatch')
        public(body, own_etag)
        # An explicit stale precondition is the only expected 412. It never authorizes a new mutation/rebase.
        try:
            bounded('DELETE', path, expected=first['etag'])
        except UploadError as error:
            if str(error) != 'upload_conflict':raise
            fatal = False  # only this deliberate stale-condition test expects 412
        else:
            raise Stop('stale_condition_accepted')
        after = bounded('HEAD', path)
        if after != restored:
            raise Stop('stale_request_changed_remote')
        previous = own_etag
        fenced = True
        sink.delete(path);settled()  # durable attempt; exact receipt-owned generation only
        fenced = False
        final = bounded('HEAD', path)
        if not final['deleted'] or final['etag'] != own_etag or final['etag'] == previous:
            raise Stop('cleanup_unconfirmed')
        receipt.update(result='PASS_SYNTHETIC_HTTPS', remote_cleanup='confirmed_tombstone_retained')
    except (Stop, UploadError) as error:
        receipt['stop_code'] = str(error)
        # No finally DELETE: outcome/ownership may be unknown. Preserve state and require read-only reconciliation.
        receipt['remote_cleanup'] = 'not_attempted_after_stop'
    finally:
        try:
            owner.close()
            receipt['owned_worker_stopped'] = worker.process is None and not worker.thread.is_alive()
            receipt['output_lock_released'] = owner._lock is None
        except BaseException:
            receipt.update(result='STOPPED', owned_worker_stopped=False, output_lock_released=False)
        receipt['pending_paths'] = len(owner.state['upload_pending'])
        receipt['elapsed_s'] = round(WALL_SECONDS-max(0, deadline-time.monotonic()), 3)
        (output/'receipt.json').write_bytes(canonical(receipt)+b'\n')
    return receipt


def inject_credentials(directory, plan):
    directory = Path(directory)
    if directory.is_symlink() or directory.stat().st_mode & 0o077:
        raise Stop('credential_directory_permissions')
    for name, key, limit in (('upload-token', 'token_env', 128), ('access-cookie', 'cookie_env', 4096)):
        path = directory/name
        if path.is_symlink() or path.stat().st_mode & 0o077 or path.stat().st_uid != os.getuid():
            raise Stop('credential_file_permissions')
        with path.open('rb') as handle:data = handle.read(limit+2).strip()
        if not data or len(data) > limit:
            raise Stop('credential_file_invalid')
        os.environ[plan[key]] = data.decode('ascii')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--grant')
    parser.add_argument('--credential-dir')
    parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--owned-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        plan = read_small(args.plan)
        missing = [key for key in ('upload_base', 'public_origin', 'game_key') if not plan.get(key)]
        if not args.execute and missing:
            print(json.dumps({'result':'NOT_READY', 'execute_enabled':plan.get('execute_enabled') is True,
                'missing':missing+['bound_grant', 'exclusive_writer_confirmation', 'issued_token_and_access_cookie'],
                'pins_checked':False, 'network_requests':0}))
            return 2
        if args.credential_dir:
            validate(plan)  # exact allowlisted source/target names before reading the explicit credential files
            inject_credentials(args.credential_dir, plan)
        if args.execute:
            if not args.grant or not args.output:raise Stop('execution_inputs_missing')
            if not args.owned_child:
                validate(plan, read_small(args.grant))
                if plan.get('execute_enabled') is not True:raise Stop('execution_disabled')
                command = [sys.executable, str(Path(__file__).resolve()), '--execute', '--owned-child',
                    '--plan', args.plan, '--grant', args.grant, '--output', args.output]
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                try:
                    stdout, _ = child.communicate(timeout=114)
                    if len(stdout) > 8192:raise Stop('child_receipt_invalid')
                    result = json.loads(stdout)
                    print(json.dumps(result))
                    return child.returncode
                except subprocess.TimeoutExpired:
                    # The new session contains only this run's coordinator/transport/public-probe descendants.
                    try:os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL);child.wait(timeout=2)
                    print(json.dumps({'result':'STOPPED','reason':'wall_deadline','remote_cleanup':'unknown_preserve_state'}))
                    return 2
                except (Stop, OSError, ValueError):
                    print(json.dumps({'result':'STOPPED','reason':'child_receipt_unavailable','remote_cleanup':'unknown_preserve_state'}))
                    return 2
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL);child.wait(timeout=2)
            receipt = execute(plan, read_small(args.grant), args.output)
            print(json.dumps({'result': receipt['result'], 'requests': len(receipt['requests']), 'remote_cleanup': receipt['remote_cleanup']}))
            return 0 if receipt['result'] == 'PASS_SYNTHETIC_HTTPS' else 2
        print(json.dumps(validate(plan)))
        return 0
    except (Stop, UploadError, OSError, ValueError, KeyError, TypeError):
        print(json.dumps({'result': 'REFUSED', 'reason': 'preflight_or_execution_invalid'}))
        return 2


if __name__ == '__main__':
    if sys.argv[1:] == ['_public']:
        public_child()
    else:
        raise SystemExit(main())
