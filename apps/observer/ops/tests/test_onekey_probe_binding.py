"""ONEKEY-PROBE-01: the deferred probe of the v7 candidate wrapper is bound to the main run's game_key/scope/expiry.

Derived from Codex RED dfaa1a1a (tests/coordination/test_onekey_probe_identity.py). Synthetic only: client check,
plan/grant, HEAD transport and the runner child are stubs; the actual cmd_run/cmd_probe/assess/finish decide.
No network, no process, no gateway. This is a local consistency refusal; it says nothing about server evidence."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import time
import types
from datetime import datetime, timezone

import pytest

SOURCE = Path(__file__).resolve().parents[1]/'onekey'/'onekey_wrapper_v7_candidate.py'
TOKEN = 'cx04-fake-bearer-only-0123456789abcdefghijklm'
COOKIE = 'session=fake-cookie-only'
START = 1893456000.0  # synthetic 2030-01-01


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


@pytest.fixture
def h(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('onekey_v7_candidate', SOURCE)
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    clock = [START]
    monkeypatch.setattr(time, 'time', lambda: clock[0])
    monkeypatch.setattr(time, 'sleep', lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(w, 'now_utc', lambda: iso(clock[0]))
    monkeypatch.setenv(w.TOKEN_ENV, TOKEN)   # harness mode reads credentials from the environment
    monkeypatch.setenv(w.COOKIE_ENV, COOKIE)
    heads = []
    state = {'revoked': False}

    def head(pre, relative, token, cafile):
        status = 401 if state['revoked'] else 403 if relative != 'v1/game/123456.json' or clock[0] >= START + 1200 else 200
        heads.append((relative, status))
        return status, {'content_type': 'application/json', 'cf_mitigated_present': False, 'cf_ray_present': False}

    def popen(argv, **_kw):
        out = Path(argv[argv.index('--output') + 1])
        out.mkdir()
        (out/'receipt.json').write_text(json.dumps({
            'result': 'PASS_SYNTHETIC_HTTPS', 'elapsed_s': 0.1, 'remote_cleanup': 'confirmed_tombstone_retained',
            'owned_worker_stopped': True, 'pending_paths': 0,
            'requests': [{'method': 'PUT', 'result': 'confirmed'}] * 17}))
        return types.SimpleNamespace(returncode=0, communicate=lambda **_k: ('{"result": "PASS_SYNTHETIC_HTTPS"}', ''),
                                     kill=lambda: None)

    monkeypatch.setattr(w, 'head', head)
    monkeypatch.setattr(w, 'check_client', lambda pre: {'commit': '0'*40})
    monkeypatch.setattr(subprocess, 'Popen', popen)
    return types.SimpleNamespace(w=w, tmp=tmp_path, clock=clock, heads=heads, state=state, out=tmp_path/'out')


def pre_dict():
    return {
        'run_id': 'synthetic-run', 'game_key': '123456',
        'teammate_confirmation': {'scope_mode': 'scoped', 'publish_key_scope': 'v1/game/123456.json',
                                  'publish_expires_at': iso(START + 1200), 'reference_private': 'synthetic'},
        'test_window': {'start_utc': iso(START - 60), 'end_utc': iso(START + 1800)},
        'target': {'upload_base': 'https://synthetic.invalid/_publish', 'public_origin': 'https://synthetic.invalid'},
        'client': {'commit': '0'*40, 'worktree': '/nonexistent', 'python': 'never-executed'},
        'probes': {'out_of_scope_head': True, 'expired_head': True, 'revoked_head': True},
    }


def main_run(h, monkeypatch):
    pre = pre_dict()
    plan = dict(pre['target'], game_key=pre['game_key'])
    monkeypatch.setattr(h.w, 'build_plan_grant', lambda *_a: (plan, {'plan_sha256': h.w.sha256(h.w.canonical(plan))}))
    monkeypatch.setattr('sys.stdin', io.StringIO('SKIP\n'))
    rc = h.w.cmd_run(types.SimpleNamespace(out=str(h.out), ledger=str(h.tmp/'ledger.json'), dry_run=False),
                     pre, True, None)
    log = json.loads((h.out/'wrapper-log.json').read_text())
    # expiry is 20 min away (> 15 min wait cap) and the operator typed SKIP: both remain for deferred probes.
    assert (rc, log['missing_evidence']) == (4, ['expired', 'revoked'])
    return pre, log


def forbid_token_prompt(h, monkeypatch):
    def never(*_a, **_k):
        raise AssertionError('read_token must not be reached when the identity binding refuses')
    monkeypatch.setattr(h.w, 'read_token', never)


def probe(h, pre, kind):
    try:
        return h.w.cmd_probe(types.SimpleNamespace(out=str(h.out), kind=kind), pre, True, None), None
    except h.w.Refuse as error:
        return 2, str(error)


def test_main_run_binds_identity_without_exposing_it(h, monkeypatch):
    _pre, log = main_run(h, monkeypatch)
    assert len(log['identity_sha256']) == 64
    text = (h.out/'wrapper-log.json').read_text() + (h.out/'PUBLIC-SUMMARY.md').read_text()
    assert 'v1/game/123456.json' not in text and '"game_key"' not in text and '"123456"' not in text


def test_same_identity_completes(h, monkeypatch):
    pre, _ = main_run(h, monkeypatch)
    h.clock[0] = START + 1210
    assert probe(h, pre, 'expired') == (4, None)
    h.state['revoked'] = True
    assert probe(h, pre, 'revoked') == (0, None)
    log = json.loads((h.out/'wrapper-log.json').read_text())
    assert log['completion'] == 'COMPLETE' and len(h.heads) == 3


@pytest.mark.parametrize('change', ['game_key_scope', 'expiry_earlier', 'expiry_later', 'scope_only', 'scope_mode'])
def test_changed_identity_is_refused_before_any_head(h, monkeypatch, change):
    pre, _ = main_run(h, monkeypatch)
    before_log = (h.out/'wrapper-log.json').read_bytes()
    tc = pre['teammate_confirmation']
    if change == 'game_key_scope':
        pre['game_key'], tc['publish_key_scope'] = '123457', 'v1/game/123457.json'
        h.clock[0] = START + 1210
    elif change == 'scope_only':
        tc['publish_key_scope'] = 'v1/game/123457.json'
        h.clock[0] = START + 1210
    elif change == 'scope_mode':
        tc['scope_mode'] = 'hostwide'
        h.clock[0] = START + 1210
    elif change == 'expiry_earlier':
        tc['publish_expires_at'] = iso(START + 600)
        h.clock[0] = START + 610
    else:
        tc['publish_expires_at'] = iso(START + 1500)
        h.clock[0] = START + 1510
    heads = len(h.heads)
    forbid_token_prompt(h, monkeypatch)
    for kind in ('expired', 'revoked'):
        rc, refusal = probe(h, pre, kind)
        assert rc == 2 and refusal.startswith('run_identity_mismatch')
    assert len(h.heads) == heads, 'no deferred HEAD and no post slot may be used'
    assert (h.out/'wrapper-log.json').read_bytes() == before_log


def test_log_without_binding_is_refused(h, monkeypatch):
    pre, log = main_run(h, monkeypatch)
    log.pop('identity_sha256')
    (h.out/'wrapper-log.json').write_text(json.dumps(log))
    h.clock[0] = START + 1210
    heads = len(h.heads)
    forbid_token_prompt(h, monkeypatch)
    rc, refusal = probe(h, pre, 'expired')
    assert rc == 2 and refusal.startswith('run_identity_missing') and len(h.heads) == heads


def test_candidate_is_inert_until_registered():
    spec = importlib.util.spec_from_file_location('onekey_v7_inert', SOURCE)
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    assert w.REGISTRATION.name == 'REGISTRATION-v7.candidate.json' and not w.REGISTRATION.exists()
