"""Public publisher: writes, with no viewer request, the bodies a viewer would have fetched, as static files.

    python -m observer_app.publisher --out <dir> [--once] [--date YYYY-MM-DD ...]

Each cycle the publisher asks the SAME builders the ``/api/service/**`` routes call (``service_api.ServiceBodies``)
for the dates, each date's game list and each game's full body, and hands them to a sink:

    v1/manifest.json   contract version, code commit, published_at, publish interval, fixed delay
    v1/health.json     last publish time and sequence (written last: a cycle is complete when it changes),
                       the games and dates that could not be built this cycle
    v1/dates.json      GET /api/service/dates
    v1/games/<YYYY-MM-DD>.json   GET /api/service/games?date=
    v1/game/<game_pk>.json       GET /api/service/games/{pk}   (the full body, never ``after``)
    v1/setup/<game_pk>.json      GET /api/watch/{pk}/setup, finished games whose estimates pass validation only

What is shown is decided by those builders alone (display gate, unconfirmed pitches, the recommendation ledger
written before the body is returned): nothing here selects or changes a field. A body the route would refuse
(423 demo embargo, 503 feed) is not written. Every file gets one added top-level key and nothing else changes:

    "publish": {"contract": "v1", "sequence": <int>, "published_at": "<UTC ISO>"[, "final": <bool>]}

``sequence`` is one counter per publisher, raised once per cycle; a file carries the sequence of the cycle that
last wrote it. ``final`` exists on game files: once a game is written with ``final: true`` it is not rewritten
unless its content changes (a correction), which gets the then-current, higher sequence.

Calling the game builder is what drives the upstream poll (``LiveFeed.snapshot`` fetches when its interval has
passed), so the publisher is the "someone is watching". Demo sessions, the operator API and replay clocks are
never asked for; a game a demo session takes is withdrawn (its game and setup files are deleted) in the next
cycle and published again when the demo ends. ``MirrorSink`` optionally sends local files through ``HttpSink``;
local files remain the recovery source. One publisher per output folder: a second one refuses to start. The live store and the recommendation ledger are
required: without either the publisher does not start, and it stops publishing if the ledger is switched off.
The setup file is the one body that is NOT the route's: its ``source`` keeps only ``commit`` and ``sha256``.
"""
from __future__ import annotations

import argparse
import fcntl
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import ipaddress
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import threading
import uuid
from urllib.parse import urlsplit

LOGGER = logging.getLogger(__name__)
APP = Path(__file__).resolve().parents[2]
CONFIG_PATH = APP/'publish_config.json'
KST = timezone(timedelta(hours=9))
FILES = {'manifest': 'manifest.json', 'health': 'health.json', 'dates': 'dates.json', 'games': 'games/{date}.json',
         'game': 'game/{game_pk}.json', 'setup': 'setup/{game_pk}.json'}
# ``display_state.feed`` describes the connection at the moment a body was built (last good poll, receipt time,
# measured lag, outage): it moves with the clock and with every poll. Only these keys say something about the
# game's content (the admitted order of the served snapshot) or are configuration.
STATE_FEED_CONTENT = ('provider_order', 'lag_threshold_s', 'thresholds_provisional')
# The publish block is the last key of every file this module writes and is far shorter than this: the start-up
# scan reads no more than this much of the end of a file (a bound of the format, not a tunable).
PUBLISH_TAIL_BYTES = 1024
PUBLISH_TAIL = re.compile(rb'"publish"\s*:\s*(\{[^{}]*\})\s*\}\s*$')


def layout(contract):
    """Published path of each file kind. The contract version IS the prefix (``v1/...``): one value, never two."""
    return {kind: f'{contract}/{path}' for kind, path in FILES.items()}
ENV = {'out_dir': ('PITCHEEZY_PUBLISH_OUT_DIR', str), 'state_path': ('PITCHEEZY_PUBLISH_STATE', str),
       'publish_interval_s': ('PITCHEEZY_PUBLISH_INTERVAL_S', float), 'fixed_delay_s': ('PITCHEEZY_PUBLISH_DELAY_S', float)}
PROVISIONAL = ('publish_interval_s', 'settled_recheck_s', 'final_recheck_for_s', 'final_slow_recheck_s', 'live_days_back', 'live_days_ahead')


def load_config(path=None, env=None, **overrides):
    """``publish_config.json`` <- environment <- explicit overrides (None = not given)."""
    config = json.loads(Path(path or CONFIG_PATH).read_text())
    env = os.environ if env is None else env
    for key, (name, kind) in ENV.items():
        if env.get(name) not in (None, ''):
            config[key] = kind(env[name])
    config |= {key: value for key, value in overrides.items() if value is not None}
    interval, delay, recheck = config['publish_interval_s'], config['fixed_delay_s'], config['settled_recheck_s']
    if not isinstance(config['contract_version'], str) or re.fullmatch(r'v[1-9]\d*', config['contract_version']) is None:
        raise ValueError('contract_version must look like v1, v2, ... (it is also the path prefix)')
    for key in ('final_recheck_for_s', 'final_slow_recheck_s'):
        if config.get(key) is not None and not (isinstance(config[key], (int, float)) and config[key] >= 0):
            raise ValueError(f'{key} must be null or a number of seconds')
    if not (isinstance(interval, (int, float)) and 0 < interval <= 3600):  # also refuses NaN
        raise ValueError('publish_interval_s must be between 0 (exclusive) and 3600 seconds')
    if not (isinstance(delay, (int, float)) and 0 <= delay <= 3600):
        raise ValueError('fixed_delay_s must be between 0 and 3600 seconds')
    if recheck is not None and not (isinstance(recheck, (int, float)) and recheck >= 0):
        raise ValueError('settled_recheck_s must be null or a number of seconds')
    for key in ('live_days_back', 'live_days_ahead'):
        if type(config[key]) is not int or not 0 <= config[key] <= 7:
            raise ValueError(f'{key} must be an integer between 0 and 7')
    return config


class Sink:
    """Where published files go. ``path`` is always a relative POSIX path such as ``v1/game/123.json``.

    ``write`` must replace the whole file at once (a reader sees the old or the new content, never a part).
    ``final`` says the content is not expected to change again (an uploader may cache it longer). An uploader to a
    public host implements these three calls; the publisher's build logic does not change.

    Temporary names: a sink must never let a half-written or temporary object be readable under the public
    prefix. ``LocalDirSink`` keeps its temporary files in a sibling folder outside the output directory, so the
    output holds published files only. An uploader that mirrors a local folder must still skip dot-files and
    anything that is not one of the published names (``publisher.FILES``), and one that writes remotely must
    upload under a private name and switch, or rely on the host's atomic object replace."""

    def write(self, path, data, *, final=False):
        raise NotImplementedError

    lock_path = None  # a local file naming this destination, for the one-publisher lock (None: the state file's)

    def read(self, path):
        """The bytes last written at ``path``, or None if there are none or this sink cannot read back."""
        return None

    def read_tail(self, path, size):
        """At most the last ``size`` bytes at ``path`` (None as ``read``). The publisher's start-up scan only
        ever needs the end of a file, so a sink that can should not fetch the rest."""
        data = self.read(path)
        return None if data is None else data[-size:]

    def delete(self, path):
        """Remove ``path`` if it is there (never an error when it is not): a withdrawn file must not stay readable."""
        raise NotImplementedError

    def paths(self):
        """Every published path this sink holds (empty if it cannot list). Read at startup so the sequence
        continues above every file already out, whatever happened to the publisher's own bookkeeping."""
        return []


class UnsafeOutput(OSError):
    """The output folder is not a plain tree of our own: a symbolic link sits inside it."""


class LocalDirSink(Sink):
    """Files under one local directory, each replaced atomically: the content is written completely to a
    temporary file in ``<parent>/.<name>.tmp/`` (a sibling of the output directory, so on the same filesystem and
    never inside what gets uploaded) and then renamed onto the published name.

    The root is resolved ONCE, here: the lock, the temporary folder and every read, write and delete use that
    one resolved folder, so repointing a link later cannot move the writes away from the folder that is locked.
    Nothing under the root may be a symbolic link (checked now and again on every operation): a link there
    would make a write or a forced delete land outside the output folder."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.staging = self.root.parent/f'.{self.root.name}.tmp'
        self.lock_path = self.root.parent/f'.{self.root.name}.lock'
        if self.staging.is_symlink():
            raise UnsafeOutput(f'{self.staging} is a symbolic link; refusing to publish')
        for folder, folders, files in os.walk(self.root, followlinks=False):
            for name in (*folders, *files):
                if os.path.islink(os.path.join(folder, name)):
                    raise UnsafeOutput(f'{os.path.join(folder, name)} is a symbolic link inside the output folder; refusing to publish')

    def _target(self, path):
        parts = PurePosixPath(path).parts
        if not parts or PurePosixPath(path).is_absolute() or any(part in ('..', '.') or part.startswith('.') for part in parts):
            raise ValueError(f'not a relative publish path: {path}')
        target = self.root
        for part in parts:  # no component below the root may be a link: the file stays inside the root
            target = target/part
            if target.is_symlink():
                raise UnsafeOutput(f'{path}: a symbolic link inside the output folder; nothing is read, written or deleted through it')
        return target

    def write(self, path, data, *, final=False):
        target = self._target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        temporary = self.staging/f"{'_'.join(PurePosixPath(path).parts)}.{os.getpid()}.tmp"
        try:
            with open(temporary, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def cleanup(self):
        """Remove temporary files a crashed publisher left behind: plain files directly in the temporary folder
        whose name has this class's own pattern (``<name>.<pid>.tmp``). Nothing else there is touched. Called by
        the publisher once it holds the lock."""
        if self.staging.is_symlink() or not self.staging.is_dir():
            return
        for path in self.staging.iterdir():
            if re.fullmatch(r'.+\.\d+\.tmp', path.name) and path.is_file() and not path.is_symlink():
                try:
                    path.unlink()
                except OSError:
                    LOGGER.warning('leftover temporary file %s not removed', path.name)

    def read(self, path):
        target = self._target(path)
        return target.read_bytes() if target.is_file() else None

    def read_tail(self, path, size):
        target = self._target(path)
        if not target.is_file():
            return None
        with open(target, 'rb') as handle:
            handle.seek(max(0, target.stat().st_size-size))
            return handle.read(size)

    def delete(self, path):
        self._target(path).unlink(missing_ok=True)

    def paths(self):
        found = []
        for folder, folders, files in os.walk(self.root, followlinks=False):
            folders[:] = [name for name in folders if not name.startswith('.') and not os.path.islink(os.path.join(folder, name))]
            found += [(Path(folder)/name).relative_to(self.root).as_posix() for name in files
                      if name.endswith('.json') and not name.startswith('.') and not os.path.islink(os.path.join(folder, name))]
        return sorted(found)


def public_path(path):
    """Only the publisher's names, never directory scans of arbitrary JSON or temporary files."""
    return isinstance(path, str) and re.fullmatch(
        r'v[1-9]\d*/(manifest\.json|health\.json|dates\.json|games/\d{4}-\d{2}-\d{2}\.json|(?:game|setup)/\d+\.json)', path) is not None


UPLOAD_MAX_BYTES = 5 * 1024 * 1024  # Partner gateway's fixed request limit.
_ETAG = re.compile(r'"[a-f0-9]{32}"')
_SHA256 = re.compile(r'[a-f0-9]{64}')


def upload_path(path):
    """The partner's v1 whitelist, separately from the local publisher's versioned layout."""
    if not isinstance(path, str):
        return False
    if re.fullmatch(r'v1/(manifest|health|dates)\.json|v1/(game|setup)/[1-9][0-9]{0,11}\.json', path):
        return True
    match = re.fullmatch(r'v1/games/(\d{4}-\d{2}-\d{2})\.json', path)
    if match:
        try:
            return datetime.fromisoformat(match[1]).strftime('%Y-%m-%d') == match[1]
        except ValueError:
            pass
    return False


class UploadError(OSError):
    """Fixed codes only; no credentials, URLs, server bodies or chained transport errors."""

    def __init__(self, code, *, outcome='rejected', retry_after=0):
        super().__init__(code)
        self.outcome, self.retry_after = outcome, retry_after


class HttpSink(Sink):
    """PUT/DELETE transport. No GET, redirects, proxy environment or response-body logging."""

    def __init__(self, config, *, dry_run=False):
        self.dry_run, self.config = dry_run, dict(config)
        try:
            self.url = urlsplit(config['base_url'])
            host = self.url.hostname
            loopback = host == 'localhost'
            try:
                loopback = loopback or ipaddress.ip_address(host).is_loopback
            except ValueError:
                pass
            if (not host or self.url.username is not None or self.url.password is not None
                    or self.url.query or self.url.fragment or self.url.scheme not in ('http', 'https')
                    or (self.url.scheme == 'http' and not loopback)):
                raise ValueError
            self.url.port  # validate port without echoing it in an exception
            self.prefix = self.url.path.rstrip('/')
            if any(ord(c) <= 32 or ord(c) >= 127 for c in config['base_url']):
                raise ValueError
            self.timeout = config['timeout_s']
            self.retries = config['retries']
            self.backoff = config['backoff_s']
            if (type(self.timeout) not in (int, float) or not 0 < self.timeout <= 60
                    or type(self.retries) is not int or not 0 <= self.retries <= 5
                    or type(self.backoff) not in (int, float) or not 0 <= self.backoff <= 10):
                raise ValueError
            self.headers = config['headers']
            for kind in ('live', 'final'):
                headers = self.headers[kind]
                if (set(headers) != {'Content-Type', 'Cache-Control'} or headers['Content-Type'] != 'application/json'
                        or not isinstance(headers['Cache-Control'], str)
                        or any(ord(c) < 32 or ord(c) > 126 for c in headers['Cache-Control'])):
                    raise ValueError
            # A dry run deliberately does not load credentials.
            if not dry_run:
                env_name, file_name = config.get('token_env'), config.get('token_file')
                if bool(env_name) == bool(file_name):
                    raise ValueError
                if file_name:
                    token_path = Path(file_name).expanduser()
                    if not token_path.is_absolute():
                        raise ValueError
                    token_path = token_path.resolve()
                    # Reject files inside ANY checkout (including another worktree or a symlink into one).
                    if any((parent/'.git').exists() for parent in token_path.parents):
                        raise ValueError
                    token = token_path.read_text().strip()
                else:
                    token = os.environ[env_name]
                if not token or re.fullmatch(r'[A-Za-z0-9_-]{43,128}', token) is None:
                    raise ValueError
                self._authorization = f'Bearer {token}'
        except (KeyError, TypeError, ValueError, OSError):
            raise UploadError('upload_config_invalid') from None

    @property
    def destination(self):
        # Bind attempts to an endpoint without putting the URL or credentials in the state file.
        identity = (self.url.scheme, self.url.hostname, self.url.port or (443 if self.url.scheme == 'https' else 80), self.prefix)
        return hashlib.sha256(repr(identity).encode()).hexdigest()

    def _request(self, method, path, data=None, *, expected=None, final=False):
        if not upload_path(path):
            raise UploadError('upload_path_invalid')
        if self.dry_run:
            LOGGER.info('dry-run-upload %s %s (%s bytes, %s)', method, path, len(data or b''), 'final' if final else 'live')
            return None
        connection = None
        try:
            factory = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
            connection = factory(self.url.hostname, self.url.port, timeout=self.timeout)
            headers = self.headers['final' if final else 'live'] | {'Authorization': self._authorization}
            if method != 'HEAD':
                if expected is not None and (not isinstance(expected, str) or _ETAG.fullmatch(expected) is None):
                    raise UploadError('upload_precondition_invalid')
                if method == 'DELETE' and expected is None:
                    raise UploadError('upload_precondition_invalid')
                headers |= {'If-None-Match': '*'} if expected is None else {'If-Match': expected}
                if method == 'PUT':
                    headers['X-Content-SHA256'] = hashlib.sha256(data).hexdigest()
            connection.request(method, f'{self.prefix}/{path}', body=data, headers=headers)
            response = connection.getresponse()
            if method == 'HEAD' and response.status == 404:
                return {'etag': None, 'sha256': None, 'deleted': False}
            if response.status != 200:
                if response.status == 412:
                    raise UploadError('upload_conflict', outcome='blocked')
                if response.status == 429:
                    raw = response.getheader('Retry-After', '')
                    delay = float(raw) if raw.isdigit() else 60.0
                    raise UploadError('upload_rate_limited', outcome='retry', retry_after=max(1.0, delay))
                # A timeout/5xx/unexpected 2xx does not establish that a mutation failed at the storage layer.
                rejected = response.status in (400, 401, 403, 404, 405, 411, 413, 415, 422, 428)
                raise UploadError('upload_rejected' if rejected else 'upload_unknown', outcome='rejected' if rejected else 'blocked')
            etag = response.getheader('ETag', '')
            if not isinstance(etag, str) or _ETAG.fullmatch(etag) is None:
                raise UploadError('upload_receipt_invalid', outcome='blocked')
            if method == 'HEAD':
                deleted, sha = response.getheader('X-Publish-Deleted'), response.getheader('X-Content-SHA256', '')
                if deleted not in ('true', 'false') or not isinstance(sha, str) or _SHA256.fullmatch(sha) is None:
                    raise UploadError('upload_receipt_invalid', outcome='blocked')
                if deleted == 'true' and sha != hashlib.sha256(b'').hexdigest():
                    raise UploadError('upload_receipt_invalid', outcome='blocked')
                return {'etag': etag, 'sha256': sha, 'deleted': deleted == 'true'}
            raw = response.read(4097)
            receipt = json.loads(raw) if len(raw) <= 4096 else None
            sha = hashlib.sha256(data if method == 'PUT' else b'').hexdigest()
            if (not isinstance(receipt, dict) or receipt.get('stored') is not True
                    or receipt.get('deleted') is not (method == 'DELETE') or receipt.get('sha256') != sha
                    or receipt.get('schema_validated') is not False or etag == expected):
                raise UploadError('upload_receipt_invalid', outcome='blocked')
            return {'etag': etag, 'sha256': sha, 'deleted': method == 'DELETE'}
        except UploadError:
            raise
        except (OSError, http.client.HTTPException, ValueError, RecursionError, UnicodeError):
            raise UploadError('upload_unknown', outcome='blocked') from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass

    def head(self, path):
        return self._request('HEAD', path)

    def write(self, path, data, *, expected=None, final=False):
        if not isinstance(data, bytes) or len(data) > UPLOAD_MAX_BYTES:
            raise UploadError('upload_body_invalid')
        try:
            if not isinstance(json.loads(data.decode('utf-8')), dict):
                raise ValueError
        except (ValueError, UnicodeError, RecursionError):
            raise UploadError('upload_body_invalid') from None
        return self._request('PUT', path, data, expected=expected, final=final)

    def delete(self, path, *, expected):
        return self._request('DELETE', path, expected=expected)

    def check(self):
        """Authenticated metadata read only. A missing manifest is a valid diagnostic, never proof of writes."""
        return self.head('v1/manifest.json')


def _upload_pending(state):
    pending = state.setdefault('upload_pending', {})
    if not isinstance(pending, dict) or any(not upload_path(p) or a not in ('put', 'delete') for p, a in pending.items()):
        raise UploadError('upload_state_invalid')
    if 'upload_protocol' not in state:
        # Old clients may have sent unconditioned requests whose completion is unknown. Do not infer safety.
        state.update(upload_protocol=1, upload_revision=0, upload_intents={}, upload_attempts={
            path: {'phase': 'blocked', 'reason': 'legacy_unknown'} for path in pending})
    if state['upload_protocol'] != 1 or type(state.get('upload_revision')) is not int or state['upload_revision'] < 0:
        raise UploadError('upload_state_invalid')
    for key in ('upload_intents', 'upload_attempts'):
        if not isinstance(state.get(key), dict) or any(not upload_path(p) or not isinstance(v, dict) for p, v in state[key].items()):
            raise UploadError('upload_state_invalid')
    for intent in state['upload_intents'].values():
        if (intent.get('action') not in ('put', 'delete') or type(intent.get('revision')) is not int
                or not 0 < intent['revision'] <= state['upload_revision']
                or not isinstance(intent.get('sha256'), str) or _SHA256.fullmatch(intent['sha256']) is None):
            raise UploadError('upload_state_invalid')
    for attempt in state['upload_attempts'].values():
        if attempt.get('phase') not in ('inflight', 'blocked', 'rejected', 'retry'):
            raise UploadError('upload_state_invalid')
        minimal = attempt['phase'] == 'blocked' and attempt.get('reason') in ('legacy_unknown', 'local_intent_mismatch', 'upload_body_invalid')
        if not minimal and (attempt.get('action') not in ('put', 'delete')
                or type(attempt.get('revision')) is not int or not 0 < attempt['revision'] <= state['upload_revision']
                or not isinstance(attempt.get('sha256'), str) or _SHA256.fullmatch(attempt['sha256']) is None
                or 'expected' not in attempt or (attempt['expected'] is not None and
                    (not isinstance(attempt['expected'], str) or _ETAG.fullmatch(attempt['expected']) is None))
                or (attempt['phase'] == 'retry' and (type(attempt.get('not_before')) not in (int, float)
                    or not 0 <= attempt['not_before'] < float('inf')))):
            raise UploadError('upload_state_invalid')
        # Interrupted attempts are never retried/rebased, even if HEAD would show identical bytes.
        if attempt['phase'] == 'inflight':
            attempt.update(phase='blocked', reason='interrupted_unknown')
    return pending


def _queue_upload(state, path, action, sha):
    if not upload_path(path):
        raise UploadError('upload_path_invalid')
    pending = state['upload_pending']
    old = state['upload_intents'].get(path)
    if old is None or (old['action'], old['sha256']) != (action, sha):
        state['upload_revision'] += 1
        state['upload_intents'][path] = {'action': action, 'sha256': sha, 'revision': state['upload_revision']}
    pending[path] = action


class MirrorSink(Sink):
    """Local state first; durable latest intents and immutable CAS attempts, with fail-closed uncertainty."""

    def __init__(self, local, remote, *, background=False):
        self.local, self.remote = local, remote
        self.lock_path = local.lock_path
        self.pending, self.attempted = {}, set()
        self.retry_after = None  # scheduling cursor only; durable intent remains in state
        self.save = lambda: None
        self.background, self.worker, self.closing = background, None, False
        self.lock = threading.RLock()

    def bind(self, state, save, lock=None):
        self.lock = lock if lock is not None else self.lock
        self.pending, self.save, self.state = _upload_pending(state), save, state
        destination = state.setdefault('upload_destination', self.remote.destination)
        if destination != self.remote.destination:
            raise UploadError('upload_destination_changed')
        for path in self.paths():
            if upload_path(path) and not path.endswith('/health.json') and path not in self.pending:
                try:
                    data = self._read_upload(path)
                except UploadError:
                    self.pending[path] = 'put'
                    state['upload_attempts'][path] = {'phase': 'blocked', 'reason': 'upload_body_invalid'}
                    continue
                if data is not None:
                    _queue_upload(state, path, 'put', hashlib.sha256(data).hexdigest())
        self.save()
        if self.background and not self.remote.dry_run:
            from .upload_worker import UploadWorker
            self.worker = UploadWorker(self._transfer, self.remote.config)

    def close(self):
        with self.lock:
            self.closing = True
        if self.worker is not None:
            self.worker.close()

    def cleanup(self):
        if self.worker is None:  # startup only; an active worker owns its immutable staging snapshot
            self.local.cleanup()

    def paths(self):
        return self.local.paths()

    def read(self, path):
        return self.local.read(path)

    def read_tail(self, path, size):
        return self.local.read_tail(path, size)

    def _read_upload(self, path):
        target = self.local._target(path)
        if not target.is_file():
            return None
        with target.open('rb') as handle:
            data = handle.read(UPLOAD_MAX_BYTES+1)
        if len(data) > UPLOAD_MAX_BYTES:
            raise UploadError('upload_body_invalid')
        return data

    def _queue(self, path, action, data=None):
        sha = hashlib.sha256(data if action == 'put' else b'').hexdigest()
        _queue_upload(self.state, path, action, sha)
        self.save()

    def _eligible(self, path):
        if self.remote.dry_run:
            return True
        intent = self.state['upload_intents'].get(path)
        if self.state.get('upload_recovery_blocked') or intent is None:
            return False
        attempt = self.state['upload_attempts'].get(path)
        return not (attempt and (attempt['phase'] in ('blocked', 'inflight') or
            (attempt['phase'] == 'rejected' and attempt.get('revision') == intent['revision']) or
            (attempt['phase'] == 'retry' and time.time() < attempt.get('not_before', float('inf')))))

    def _send(self, path, data=None, *, final=False):
        self.attempted.add(path)
        if not self._eligible(path):
            return
        if self.worker is not None:
            return self.worker.submit(path)  # metadata only, bounded and coalesced before insertion
        else:
            self._transfer(path, data, final=final)

    def _transfer(self, path, data=None, *, final=False):
        with self.lock:
            action = self.pending.get(path)
            if action is None:
                return
            intent = self.state['upload_intents'].get(path)
            attempts = self.state['upload_attempts']
            attempt = attempts.get(path)
            if self.remote.dry_run:
                self.remote._request(action.upper(), path, data, final=final)
                return
            if not self._eligible(path):
                return
            intent = dict(intent)
            if action == 'delete':
                self.local.delete(path)  # queued older work may now see a delete whose local unlink failed
            if self.worker:
                final = self.state.get('files', {}).get(path, {}).get('final') is True
            if action == 'put' and self.worker is None:
                data = self._read_upload(path) if data is None else data
                if data is None or hashlib.sha256(data).hexdigest() != intent['sha256']:
                    attempts[path] = {'phase': 'blocked', 'reason': 'local_intent_mismatch'}
                    self.save()
                    return
        try:
            meta = self.worker.request('HEAD', path) if self.worker else self.remote.head(path)
        except UploadError:
            LOGGER.warning('upload_failed')  # HEAD sent no mutation; later metadata retry remains safe.
            return
        snapshot = None
        with self.lock:
            # Includes DELETE+404: an old HEAD must never erase a newer local PUT intent.
            if self.state['upload_intents'].get(path) != intent or self.pending.get(path) != action:
                return
            if action == 'delete' and meta['etag'] is None:
                self.pending.pop(path, None)
                self.state['upload_intents'].pop(path, None)
                attempts.pop(path, None)
                self.save()
                return
            if self.worker and action == 'put':
                target = self.local._target(path)
                if not target.is_file():
                    return
                if target.stat().st_size > UPLOAD_MAX_BYTES:
                    attempts[path] = {'phase': 'blocked', 'reason': 'upload_body_invalid'}
                    self.save()
                    return
                self.local.staging.mkdir(parents=True, exist_ok=True)
                snapshot = self.local.staging/f'upload_{uuid.uuid4().hex}.{os.getpid()}.tmp'
                os.link(target, snapshot, follow_symlinks=False)  # our writer only replaces inodes atomically
            attempt = attempts[path] = intent | {'expected': meta['etag'], 'phase': 'inflight'}
            try:
                self.save()  # immutable body/condition durably prepared before spawning the mutating child
            except BaseException:
                if snapshot is not None: snapshot.unlink(missing_ok=True)
                raise
        try:
            if self.worker:
                self.worker.request(action.upper(), path, expected=attempt['expected'],
                    snapshot=str(snapshot) if snapshot else None, sha256=intent['sha256'], final=final)
            elif action == 'delete':
                self.remote.delete(path, expected=attempt['expected'])
            else:
                self.remote.write(path, data, expected=attempt['expected'], final=final)
        except UploadError as error:
            with self.lock:
                attempt.update(phase=error.outcome, reason=str(error))
                if error.outcome == 'retry':
                    attempt['not_before'] = time.time()+error.retry_after
                self.save()
            LOGGER.warning('upload_failed')
        else:
            with self.lock:
                if self.state['upload_intents'].get(path) == intent:
                    self.pending.pop(path, None)
                    self.state['upload_intents'].pop(path, None)
                attempts.pop(path, None)
                self.save()
                if self.worker is not None and path in self.pending:
                    self.worker.submit(path)  # a confirmed old ACK unlocks the queued newer intent
        finally:
            if snapshot is not None and (self.worker is None or self.worker.process is None):
                snapshot.unlink(missing_ok=True)  # an unreaped child keeps its snapshot and output lock

    def write(self, path, data, *, final=False):
        with self.lock:
            if self.closing:
                raise UploadError('upload_closed')
            self._queue(path, 'put', data)
            self.local.write(path, data, final=final)
            self._send(path, data, final=final)

    def delete(self, path):
        with self.lock:
            if self.closing:
                raise UploadError('upload_closed')
            self._queue(path, 'delete')
            self.local.delete(path)
            self._send(path)

    def retry(self, health):
        with self.lock:
            self._retry(health)

    def _retry(self, health):
        paths = sorted(path for path in set(self.pending)-self.attempted-{health} if self._eligible(path))
        if self.worker is not None and self.retry_after is not None:
            paths = [p for p in paths if p > self.retry_after]+[p for p in paths if p <= self.retry_after]
        for path in paths:
            if self.worker is not None:
                if not self._send(path):
                    break
                self.retry_after = path  # repeated HEAD failures cannot monopolize the next cycle's mailbox
                continue
            try:
                if self.pending[path] == 'delete':
                    self.local.delete(path)
                    self._send(path)
                else:
                    data = self._read_upload(path)
                    if data is None:
                        continue  # write-ahead intent without local bytes: preserve, never silently call it sent
                    final = json.loads(data).get('publish', {}).get('final') is True
                    self._send(path, data, final=final)
            except (OSError, ValueError, RecursionError):
                LOGGER.warning('upload_failed')


def code_commit(repo=None, env=None):
    """(commit sha or None, dirty flag or None) of the code that publishes. Never guessed: None when unknown."""
    env = os.environ if env is None else env
    if env.get('PITCHEEZY_PUBLISH_COMMIT'):
        return env['PITCHEEZY_PUBLISH_COMMIT'], None
    repo = repo or APP.parents[1]
    try:
        head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10, check=True)
        dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None, None
    return head.stdout.strip() or None, bool(dirty.stdout.strip())


def _iso(epoch):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(epoch))


def _digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def game_digest(body):
    """What makes a settled game file new: its content, not the moment it was built. Left out are the fields that
    move with the clock or with the connection while the game itself is unchanged: the whole top-level ``feed``
    (``as_of``, ``ok``, ``stale_s``) and, inside ``display_state.feed``, everything but ``STATE_FEED_CONTENT``
    (``ok``, ``delayed``, ``lag_s``, ``lag_max_s``, ``lag_measured``, ``last_ok_at``, ``failing_since``,
    ``failing_for_s``, ``received_at``). Pitches, plate appearances, receipts, the display state itself, the
    recommendation and its ledger record all count."""
    stable = {key: value for key, value in body.items() if key != 'feed'}
    state = stable.get('display_state')
    if isinstance(state, dict) and isinstance(state.get('feed'), dict):
        stable['display_state'] = state | {'feed': {key: state['feed'].get(key) for key in STATE_FEED_CONTENT}}
    return _digest(stable)


def is_final(body):
    """True when a game body is finished and settled: the builder calls the game final and no pitch of it is
    still unconfirmed (a body with a pitch in 확인 중 keeps being rewritten until the store admits it)."""
    state, receipts = body.get('display_state'), body.get('receipts')
    return bool(body['game'].get('status') == 'final' and (body.get('current') or {}).get('phase') == 'final'
                and (state is None or state.get('state') == 'final')
                and (receipts is None or receipts.get('unconfirmed') == 0))


class NotPublic(RuntimeError):
    """A body that is not the plain public view reached the publisher (it is never written)."""


def check_public(body):
    """Last line of defence, not the gate: the builders are only ever asked for the non-demo view, and this
    refuses to write anything that says otherwise."""
    source = body.get('source') or {}
    if source.get('kind') not in ('archive', 'live') or source.get('session_id') is not None or 'publish' in body:
        raise NotPublic(f"refusing to publish a body with source {source!r}")
    return body


UNCONFIRMED_KEPT = ('call_label', 'result_label')  # all an unadmitted pitch may say (both read 확인 중)


def check_live(body):
    """A body from the live feed may leave only in the store-and-ledger form (also a last line of defence: the
    builders decide, this refuses). Without a store the builders attach no receipts and mask no unconfirmed pitch,
    and show a recommendation unrecorded: neither is ever published."""
    if (body.get('source') or {}).get('kind') != 'live':
        return body
    if not isinstance(body.get('receipts'), dict):
        raise NotPublic('a live body without pitch receipts (no live store): unconfirmed pitches would not be masked')
    for pitch in body.get('pitches') or []:
        if (pitch.get('receipt') or {}).get('status') != 'received' and any(
                value is not None for key, value in (pitch.get('actual') or {}).items() if key not in UNCONFIRMED_KEPT):
            raise NotPublic('an unconfirmed pitch carries observed details')
    rec = (body.get('next') or {}).get('rec') or {}
    if rec.get('candidates') and not ((rec.get('ledger') or {}).get('status') == 'recorded'
                                      and (body.get('display_state') or {}).get('recommendation_shown') is True):
        raise NotPublic('a recommendation that is not recorded in the ledger, or that the display state does not show')
    return body


def public_setup(view):
    """The setup body as published: the route's body with its ``source`` cut down to what a viewer can use to
    name the estimates (the video module's commit and the content hash that was validated). Where the file
    lives, its repository, branch, video and selection notes are internal and free text: none of them leaves."""
    source = view.get('source') or {}
    commit, sha = source.get('commit'), source.get('sha256')
    return view | {'source': {
        'commit': commit if isinstance(commit, str) and re.fullmatch(r'[0-9a-fA-F]{4,64}', commit) else None,
        'sha256': sha if isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{64}', sha) else None}}


class AlreadyRunning(RuntimeError):
    """Another publisher holds this output's lock."""


class Unsafe(RuntimeError):
    """The publisher may not run as configured: it would publish without the live store or the ledger."""


class Publisher:
    """One publish cycle at a time. ``bodies`` is ``service_api.ServiceBodies`` of the app whose routes would
    otherwise answer; ``setups`` is its ``SetupEstimates``; ``dates`` (KST ``YYYY-MM-DD``) replaces the live
    window. ``state_path`` keeps the sequence and what was written across restarts."""

    def __init__(self, bodies, sink, *, config, setups=None, state_path=None, wall=time.time, commit=(None, None),
                 dates=None, safe=None):
        self.bodies, self.sink, self.config, self.setups, self.wall = bodies, sink, config, setups, wall
        self._state_lock = threading.RLock()
        self._closing = False
        self.commit, self.dates, self.safe = commit, None if dates is None else sorted(set(dates)), safe
        for date in self.dates or []:
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}', date) is None:
                raise ValueError('date must be YYYY-MM-DD')
        self.layout = layout(config['contract_version'])
        self.state_path = None if state_path is None else Path(state_path)
        self._lock = self._acquire()
        try:
            if hasattr(self.sink, 'cleanup'):
                self.sink.cleanup()  # only now: the lock says no other publisher is in the middle of a write
            self.state = self._load_state()
            if isinstance(self.sink, MirrorSink):
                self.sink.bind(self.state, self._save_state, self._state_lock)
            elif 'upload_pending' in self.state:
                _upload_pending(self.state)  # disabled upload must not discard a corrupt outbox
        except BaseException:
            self.close()
            raise

    def _acquire(self):
        """Exclusive, non-blocking lock for this object's life, on the file the sink names for its destination
        (``LocalDirSink``: beside the resolved output folder, so the same folder is one lock whatever state file
        or spelling a second process uses), else on ``<state file>.lock``. Two publishers over one output would
        hand out the same sequence twice and overwrite each other's files with older ones. The kernel drops the
        lock when the holder exits or is killed."""
        target = getattr(self.sink, 'lock_path', None) or (None if self.state_path is None else f'{self.state_path}.lock')
        if target is None:
            return None
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = open(target, 'a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise AlreadyRunning(f'another publisher is already running on this output (lock {target} is held)') from None
        return handle

    def close(self):
        """Stop/reap owned upload work before releasing the exclusive output lock."""
        with self._state_lock:
            self._closing = True  # wait for any local cycle, then reject subsequent local writes
        if isinstance(self.sink, MirrorSink):
            self.sink.close()  # worker needs the state lock to commit its final outcome
        if getattr(self, '_lock', None) is not None:
            self._lock.close()
            self._lock = None

    # ---- bookkeeping --------------------------------------------------------------------------------
    def _load_state(self):
        state = {'sequence': 0, 'files': {}, 'checked': {}, 'embargoed': [], 'pending': {}, 'game_checks': {}}
        if self.state_path is not None and self.state_path.is_file():
            try:
                saved = json.loads(self.state_path.read_text())
                if not (isinstance(saved, dict) and type(saved.get('sequence')) is int and isinstance(saved.get('files'), dict)
                        and isinstance(saved.get('checked'), dict) and isinstance(saved.get('embargoed', []), list)
                        and isinstance(saved.get('pending', {}), dict) and isinstance(saved.get('game_checks', {}), dict)):
                    raise ValueError('not a publisher state')
                state |= saved
            except (ValueError, OSError, RecursionError):  # unreadable bookkeeping: start as if it were lost (everything is rebuilt)
                LOGGER.warning('state file %s unreadable; continuing from the published files', self.state_path, exc_info=True)
                state['upload_recovery_blocked'] = True
        if isinstance(self.sink, MirrorSink) and (self.state_path is None or not self.state_path.is_file()) and self.sink.paths():
            state['upload_recovery_blocked'] = True  # missing custody cannot erase an older in-flight remote attempt
        # The sequence never goes back, whatever happened to the state file: it continues above EVERY file that
        # is out. Health alone is not enough: a cycle that failed before health may have written files above it.
        state['sequence'] = max(state['sequence'], self._published_sequence())
        return state

    def _published_sequence(self):
        """The highest sequence any of OUR files carries. Only paths of this contract's layout are looked at, only
        the end of each file is read, only a publish block of this contract counts, and nothing a file contains
        can stop the start-up: whatever cannot be read as such a block is skipped."""
        prefix = re.escape(self.config['contract_version'])
        ours = re.compile(prefix+r'/(manifest\.json|health\.json|dates\.json|games/\d{4}-\d{2}-\d{2}\.json|game/\d+\.json|setup/\d+\.json)')
        highest = 0
        try:
            paths = [path for path in self.sink.paths() if isinstance(path, str) and ours.fullmatch(path)]
        except Exception:  # noqa: BLE001
            LOGGER.warning('published files cannot be listed; the sequence continues from the state and health files only', exc_info=True)
            paths = []
        for path in {*paths, self.layout['health']}:
            try:
                tail = self.sink.read_tail(path, PUBLISH_TAIL_BYTES)
                match = PUBLISH_TAIL.search(tail) if tail else None
                block = json.loads(match[1]) if match else {}
                number = block.get('sequence')
                if block.get('contract') == self.config['contract_version'] and type(number) is int and number > 0:
                    highest = max(highest, number)
            except Exception:  # noqa: BLE001 - incl. undecodable bytes and RecursionError: a file never blocks start-up
                LOGGER.warning('published file %s unreadable; ignored for the sequence', path)
        return highest

    def _save_state(self):
        with self._state_lock:
            self._save_state_locked()

    def _save_state_locked(self):
        """Durably: the state is on disk (file and folder synced) before the caller goes on. The cycle calls this
        right after taking a sequence number, so no file can ever carry a number the state does not know."""
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f'.{self.state_path.name}.{os.getpid()}.tmp')
        with open(temporary, 'w') as handle:
            handle.write(json.dumps(self.state, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.state_path)
        folder = os.open(self.state_path.parent, os.O_RDONLY)
        try:
            os.fsync(folder)
        finally:
            os.close(folder)

    def _due(self, path, now):
        """A settled file is rebuilt when it was never tried or ``settled_recheck_s`` has passed since."""
        checked, every = self.state['checked'].get(path), self.config['settled_recheck_s']
        return checked is None or (every is not None and now-checked >= every)

    def live_dates(self, now):
        if self.dates is not None:
            return list(self.dates)
        today = datetime.fromtimestamp(now, KST).date()
        return [(today+timedelta(days=offset)).isoformat()
                for offset in range(-self.config['live_days_back'], self.config['live_days_ahead']+1)]

    def _queue_local_upload(self, path, action, data=None):
        # Keep the existing remote's intents durable while only local publishing is enabled.
        if not isinstance(self.sink, MirrorSink) and 'upload_pending' in self.state:
            _queue_upload(self.state, path, action, hashlib.sha256(data if action == 'put' else b'').hexdigest())
            self._save_state()

    def _put(self, path, body, *, always, final=None, digest=None):
        with self._state_lock:
            if self._closing:
                raise Unsafe('publisher_closed')
            return self._put_locked(path, body, always=always, final=final, digest=digest)

    def _put_locked(self, path, body, *, always, final=None, digest=None):
        """Write ``body`` + the ``publish`` key at ``path``. ``always`` False: only if the content (``digest``,
        default the whole body) or its final mark differs from what is published there."""
        digest = _digest(body) if digest is None else digest
        known = self.state['files'].get(path)
        self.state['checked'][path] = self._now
        if not always and known and known['digest'] == digest and known.get('final') == final:
            return False
        if 'publish' in body:
            raise NotPublic(f'{path}: the body already has a publish key')
        publish = {'contract': self.config['contract_version'], 'sequence': self._sequence, 'published_at': _iso(self._now)}
        if final is not None:
            publish['final'] = final
        data = json.dumps(body | {'publish': publish}, ensure_ascii=False, separators=(',', ':')).encode()
        self._queue_local_upload(path, 'put', data)
        self.sink.write(path, data, final=bool(final))
        self.state['files'][path] = {'digest': digest, 'sequence': self._sequence, 'final': final}
        self._written.append(path)
        return True

    def _drop(self, path, *, force=False):
        with self._state_lock:
            if self._closing:
                raise Unsafe('publisher_closed')
            return self._drop_locked(path, force=force)

    def _drop_locked(self, path, *, force=False):
        """Delete a published file. ``force``: whether or not the state knows it (the state file may have been
        lost, or a crash may have come between the write and the save): what must not be public is deleted."""
        if not (force or path in self.state['files']):
            return True
        try:
            self._queue_local_upload(path, 'delete')
            self.sink.delete(path)
        except OSError:  # the file may still be out: the caller reports it; it is tried again when next due
            LOGGER.error('%s could not be deleted', path, exc_info=True)
            if path in self.state['files']:
                # The state keeps the file (a later drop still finds it), but not its content: the DELETE intent
                # is already in the outbox and may still run, so the next write of an identical body must be a
                # real PUT (which replaces that intent) rather than the unchanged-content skip.
                self.state['files'][path]['digest'] = None
                self._save_state()
            return False
        self.state['files'].pop(path, None)
        return True

    def _withdraw(self, game_pk):
        """A demo session holds this game: nothing of it stays public. Its game and setup files are deleted,
        registered in the state or not, and it is tried again in every cycle, so it returns when the demo ends."""
        gone = True
        for kind in ('game', 'setup'):
            path = self.layout[kind].format(game_pk=game_pk)
            gone = self._drop(path, force=True) and gone
            for timer in (path, f'{path}#built', f'{path}#archive'):
                self.state['checked'].pop(timer, None)
        self.state['pending'].pop(str(game_pk), None)
        if not gone:
            self._fail(game_pk, None, 'withdraw_failed')  # loud: a file that must not be public may still be out

    def _embargoed(self):
        return {int(game_pk) for game_pk in self.bodies.embargoed()}

    def _forget_lists(self):
        prefix = self.layout['games'].split('{')[0]
        for path in [path for path in self.state['checked'] if path.startswith(prefix)]:
            del self.state['checked'][path]

    def _fail(self, game_pk, status, code):
        """Health names the game and a fixed code only: never a path, an exception text or a file name."""
        item = {'game_pk': game_pk, 'status': status, 'code': code}
        if item not in self._unavailable:
            self._unavailable.append(item)

    # ---- one cycle ----------------------------------------------------------------------------------
    def cycle(self):
        with self._state_lock:
            if self._closing:
                raise Unsafe('publisher_closed')
            return self._cycle()

    def _cycle(self):
        """Build and write everything that is due. Returns the health body of this cycle."""
        reason = self.safe() if self.safe is not None else None
        if reason:
            raise Unsafe(reason)  # before a sequence is taken or a file is touched
        self._now, self._written, self._unavailable, self._feed_failing, self._game_attempts = self.wall(), [], [], [], {}
        if isinstance(self.sink, MirrorSink):
            self.sink.attempted.clear()
        self._sequence = self.state['sequence'] = self.state['sequence']+1
        self._save_state()  # the number is durably taken before any file carries it: a crash never reuses it
        now, unavailable_dates = self._now, []
        embargoed = self._embargoed()
        if sorted(embargoed) != self.state['embargoed']:
            # a demo started or ended: every game list is rebuilt now (its card is hidden or shown again), not at
            # the next settled recheck
            self._forget_lists()
            self.state['embargoed'] = sorted(embargoed)
        for game_pk in embargoed:
            self._withdraw(game_pk)
        dates_body = check_public(self.bodies.dates())
        live_dates = self.live_dates(now)
        fast, listed = self._lists(dates_body, live_dates, unavailable_dates)
        archived = {int(game_pk) for game_pk in self.bodies.archived()}
        prefix = self.layout['game'].split('{')[0]
        known = {int(PurePosixPath(path).stem) for path in self.state['files'] if path.startswith(prefix)}
        for game_pk in sorted((listed | known)-embargoed):
            self._game(game_pk, fast=game_pk in fast, archived=game_pk in archived)
        self._put(self.layout['dates'], dates_body, always=True)
        commit, dirty = self.commit
        self._put(self.layout['manifest'], {
            'schema': 'pitcheezy-publish-manifest-v1', 'contract': self.config['contract_version'],
            'commit': commit, 'commit_dirty': dirty, 'published_at': _iso(now),
            'publish_interval_s': self.config['publish_interval_s'], 'fixed_delay_s': self.config['fixed_delay_s'],
            'provisional': list(PROVISIONAL), 'live_dates': live_dates, 'files': self.layout}, always=True)
        if isinstance(self.sink, MirrorSink):
            self.sink.retry(self.layout['health'])
        # The last embargo check, and the last step before health. ACCEPTED WINDOW (PM decision 2026-10-11): the
        # demo clock lives in another process and there is deliberately no lock shared with it, so a demo that
        # begins after this line is reflected by the NEXT cycle (at most one publish interval later). The embargo
        # hides an already finished game from a demo audience; it does not guard unreleased information.
        late = self._embargoed()-embargoed
        if late:
            # a demo took a game while this cycle ran: its files go now, and every list (some were written before
            # the demo began, with its score) is built again
            for game_pk in late:
                self._withdraw(game_pk)
            self._forget_lists()
            unavailable_dates.clear()
            self._lists(dates_body, live_dates, unavailable_dates)
            self.state['embargoed'] = sorted(embargoed | late)
        for game_pk, code in sorted(self.state['pending'].items()):  # still unresolved, asked this cycle or not
            self._fail(int(game_pk), None, code)
        if isinstance(self.sink, MirrorSink) and not self.sink.remote.dry_run:
            for path in sorted(self.sink.pending):
                match = re.fullmatch(re.escape(self.config['contract_version'])+r'/(?:game|setup)/(\d+)\.json', path)
                if match:
                    self._fail(int(match[1]), None, 'upload_failed')
        # Only a real game-body check advances freshness; settled games may skip many cycles.
        self._now = max(now, self.wall())
        # Setup and remote-upload faults are reported in unavailable_games, but neither says the
        # game feed/body check failed. In particular, a good final game may lose only its setup file.
        failed_games = {item['game_pk'] for item in self._unavailable
                        if item['code'] not in ('setup_failed', 'upload_failed')}
        game_checks = {}
        for game_pk in sorted(listed | known | embargoed | late):
            entry = self.state['files'].get(self.layout['game'].format(game_pk=game_pk))
            withdrawn = game_pk in embargoed | late or bool(entry and entry.get('digest') is None)
            if entry is None and not withdrawn:
                continue  # no verified file: the client retains the full-file fetch path
            key = str(game_pk)
            if withdrawn or game_pk in self._game_attempts:
                game_checks[key] = {'checked_at': _iso(self._game_attempts.get(game_pk, self._now)),
                    'sequence_checked': self._sequence, 'file_sequence': None if withdrawn or entry is None else entry['sequence'],
                    'feed_ok': entry is not None and not withdrawn and game_pk not in self._feed_failing and game_pk not in failed_games}
            elif entry is not None:
                previous = self.state['game_checks'].get(key)
                if isinstance(previous, dict) and previous.get('file_sequence') == entry['sequence']:
                    game_checks[key] = previous
        self.state['game_checks'] = game_checks
        health = {'schema': 'pitcheezy-publish-health-v1', 'sequence': self._sequence, 'published_at': _iso(self._now),
                  'publish_interval_s': self.config['publish_interval_s'], 'files_written': len(self._written)+1,
                  'unavailable_games': self._unavailable, 'unavailable_dates': sorted(set(unavailable_dates)),
                  'feed_failing_games': sorted(set(self._feed_failing)), 'game_checks': game_checks}
        if isinstance(self.sink, MirrorSink):
            health['upload'] = {'code': 'upload_failed' if self.sink.pending and not self.sink.remote.dry_run else None,
                                'pending': len(self.sink.pending), 'dry_run': self.sink.remote.dry_run}
        self._put(self.layout['health'], health, always=True)
        self._save_state()
        return health

    def _lists(self, dates_body, live_dates, failed):
        """Write the game list of every date that is due. Returns (games to rebuild every cycle, games listed).
        A list never shows a game a demo holds: it is built again if a demo began meanwhile, and a list that
        cannot be built or written while any demo is on is deleted rather than left with its old scores."""
        now, fast, listed = self._now, set(), set()
        for date in sorted(set(dates_body['dates']) | set(live_dates)):
            path, live = self.layout['games'].format(date=date), date in live_dates
            if not live and not self._due(path, now):
                continue
            self.state['checked'][path] = now
            body = None
            for _ in range(2):
                try:
                    body = check_public(self.bodies.games(date))
                except Exception:  # noqa: BLE001 - one date never stops the cycle
                    LOGGER.warning('game list of %s not built', date, exc_info=True)
                    body = None
                    break
                held = self._embargoed()
                if not any(int(card['game_pk']) in held and not card.get('embargoed') for card in body['games']):
                    break
                body = None  # a demo began while the list was being built: once more
            if body is not None:
                try:
                    self._put(path, body, always=live)
                except OSError:
                    LOGGER.warning('game list of %s not written', date, exc_info=True)
                    body = None
            if body is None:
                failed.append(date)
                if self._embargoed():
                    self._drop(path, force=True)  # no list is better than one that still shows an embargoed score
                continue  # otherwise the last published list stays
            for card in body['games']:
                if card.get('embargoed'):
                    continue
                listed.add(int(card['game_pk']))
                if live:
                    fast.add(int(card['game_pk']))
        return fast, listed

    def _game(self, game_pk, *, fast, archived):
        """One game: decide whether it is due, build it, write it, then its setup file. Nothing that goes wrong
        here stops the cycle: the game's last file stays and health carries a fixed code for it."""
        now, path = self._now, self.layout['game'].format(game_pk=game_pk)
        entry = self.state['files'].get(path) or {}
        settled = entry.get('final') is True
        from_feed = settled and entry.get('kind') == 'live'
        # The local final archive of a game so far published from the live feed has arrived: taken at once (a
        # local read, no upstream poll). Until it has actually been read as the archive it is tried again at the
        # settled pace and health keeps saying ``archive_unread``.
        arrived = from_feed and archived and self._due(f'{path}#archive', now)
        if arrived:
            self.state['checked'][f'{path}#archive'] = now
        else:
            period = self.config.get('final_recheck_for_s')
            if from_feed and period is not None and now-entry['final_at'] >= period:
                # Long final: upstream is asked only at the slow pace (a late official correction is still found);
                # the setup file is still looked for at the settled pace (a local read).
                slow = self.config.get('final_slow_recheck_s')
                built = self.state['checked'].get(f'{path}#built', entry['final_at'])
                # A kept entry without a digest (a withdrawal's delete failed) is rebuilt whatever the pace: its
                # content is unknown and a DELETE may still be queued, so only a real PUT restores it.
                if entry.get('digest') is not None and (slow is None or now-built < slow):
                    if self._due(path, now):
                        self.state['checked'][path] = now
                        self._setup(game_pk)
                    return
            elif (settled or not fast) and not self._due(path, now):
                return
        self.state['checked'][path] = self.state['checked'][f'{path}#built'] = now
        self._game_attempts[game_pk] = self.wall()
        from .service_api import Refused
        try:
            body = check_live(check_public(self.bodies.game(game_pk)))  # polls upstream; the ledger is written in here
        except Refused as refused:
            if refused.status == 423:  # a demo took the game since the cycle began: not a fault
                self._withdraw(game_pk)
            else:
                self._fail(game_pk, refused.status, 'refused')
            return
        except NotPublic:
            LOGGER.error('game %s: the body is not the public view; not written', game_pk, exc_info=True)
            return self._fail(game_pk, None, 'not_public')
        except Exception:  # noqa: BLE001
            LOGGER.warning('game %s not built; the published file is kept', game_pk, exc_info=True)
            if arrived:
                self.state['pending'][str(game_pk)] = 'archive_unread'
                return self._fail(game_pk, None, 'archive_unread')
            return self._fail(game_pk, None, 'build_failed')
        if arrived and body['source']['kind'] != 'archive':
            self.state['pending'][str(game_pk)] = 'archive_unread'  # a stored feed that is not a final one
        elif body['source']['kind'] == 'archive':
            self.state['pending'].pop(str(game_pk), None)
        if (body.get('feed') or {}).get('ok') is False:
            # the body is the last snapshot: upstream did not answer. Said here, because a final file that does
            # not change (the digest leaves ``feed`` out) would otherwise hide it.
            self._feed_failing.append(game_pk)
        final = is_final(body)
        if game_pk in self._embargoed():  # a demo began after the builder's own check
            return self._withdraw(game_pk)
        try:
            self._put(path, body, always=not final, final=final, digest=game_digest(body))
        except OSError:
            LOGGER.warning('game %s not written; the published file is kept', game_pk, exc_info=True)
            return self._fail(game_pk, None, 'write_failed')
        written = self.state['files'][path]
        written['kind'] = body['source']['kind']
        written['final_at'] = (entry.get('final_at') if settled else now) if final else None
        if game_pk in self._embargoed():  # ...or while the file was being written
            return self._withdraw(game_pk)
        if final:
            self._setup(game_pk)
        elif not self._drop(self.layout['setup'].format(game_pk=game_pk)):
            self._fail(game_pk, None, 'setup_failed')

    def _setup(self, game_pk):
        """A finished game's catcher-setup file: only what ``SetupEstimates.view`` accepts (sha256 pin and every
        line validated), as ``public_setup`` cuts it. A game without estimates has no file, one that stops
        validating loses its file, and one whose estimates cannot be read at all loses it too (code in health)."""
        if self.setups is None:
            return
        from .setup_estimates import Unreadable
        path = self.layout['setup'].format(game_pk=game_pk)
        try:
            view = self.setups.view(game_pk)
        except KeyError:
            view = None
        except Unreadable:  # there, but not readable as estimates: a fault somebody must see
            LOGGER.warning('setup estimates of %s could not be read; not published', game_pk, exc_info=True)
            self._drop(path, force=True)
            return self._fail(game_pk, None, 'setup_failed')
        except ValueError:  # readable, refused by validation: simply not published
            LOGGER.warning('setup estimates of %s refused by validation; not published', game_pk, exc_info=True)
            view = None
        except Exception:  # noqa: BLE001 - anything else the reader did not name
            LOGGER.warning('setup estimates of %s could not be read; not published', game_pk, exc_info=True)
            self._drop(path, force=True)
            return self._fail(game_pk, None, 'setup_failed')
        if view is None:
            return None if self._drop(path) else self._fail(game_pk, None, 'setup_failed')
        try:
            self._put(path, public_setup(view), always=False, final=True)
        except OSError:
            LOGGER.warning('setup file of %s not written', game_pk, exc_info=True)
            self._fail(game_pk, None, 'setup_failed')

    def run(self, *, once=False, sleep=time.sleep, clock=time.monotonic):
        """Cycles forever at ``publish_interval_s`` (start to start). A failed cycle is logged and the next one
        runs: the health file then simply stops moving, which is what viewers and monitors read."""
        while True:
            started = clock()
            try:
                self.cycle()
            except Exception:  # noqa: BLE001
                if once:
                    raise
                LOGGER.exception('publish cycle failed')
            if once:
                return
            sleep(max(0.0, self.config['publish_interval_s']-(clock()-started)))


def unsafe_reason(app):
    """Why this app must not be published from, or None. The publisher needs the live store (pitch admission:
    without it no pitch is masked as 확인 중) and the recommendation ledger (without it a recommendation is shown
    unrecorded). The server tolerates both being off; a public file does not."""
    from . import live_ledger
    feed = app.state.live_parts()
    if not hasattr(feed, 'store'):
        error = getattr(feed, 'store_error', None)
        return ('the live store is not on: set PITCHEEZY_OBSERVER_LIVE_STORE to the store database'
                + (f' (it was set but could not be opened: {error})' if error else '')
                + '. Without it nothing is published.')
    if not live_ledger.enabled():
        return 'the recommendation ledger is off (PITCHEEZY_OBSERVER_REC_LEDGER=0). Without it nothing is published.'
    return None


def build(config, *, sink=None, dates=None, commit=None, wall=None, **app_options):
    """A ``Publisher`` over a private app (never served: no port, no worker) built like the server's. The app's
    fan route delay is the publisher's fixed delay. ``app_options`` go to ``main.create_app``."""
    from .main import create_app
    if not config.get('out_dir') and sink is None:
        raise ValueError('an output directory is required (--out, PITCHEEZY_PUBLISH_OUT_DIR or publish_config.json out_dir)')
    app_options.setdefault('service', object())  # the legacy replay service and its database are not opened
    app = create_app(start_worker=False, service_delay_s=config['fixed_delay_s'], **app_options)
    safe = None if app_options.get('live_feed') is False else (lambda: unsafe_reason(app))  # archive only: no live path
    reason = safe() if safe else None
    if reason:
        raise Unsafe(reason)  # before the output folder, the lock or the state file is touched
    state_path = config.get('state_path') or (config.get('out_dir') and f"{str(config['out_dir']).rstrip('/')}.state.json")
    if sink is None:
        sink = LocalDirSink(config['out_dir'])
        if config.get('upload', {}).get('enabled') or config.get('dry_run_upload'):
            sink = MirrorSink(sink, HttpSink(config['upload'], dry_run=config.get('dry_run_upload', False)),
                              background=config['upload'].get('background', True))
    return app, Publisher(app.state.service_bodies, sink, config=config,
                          setups=app.state.setups, state_path=state_path, dates=dates, safe=safe,
                          commit=code_commit() if commit is None else commit, **({} if wall is None else {'wall': wall}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', help='output directory (or PITCHEEZY_PUBLISH_OUT_DIR)')
    parser.add_argument('--once', action='store_true', help='one cycle, then exit')
    parser.add_argument('--date', action='append', help='KST date whose games are rebuilt every cycle (repeatable); default: the window in publish_config.json')
    parser.add_argument('--config', help='publish_config.json to use')
    upload_mode = parser.add_mutually_exclusive_group()
    upload_mode.add_argument('--check-remote', action='store_true', help='authenticated HEAD of v1/manifest.json only (no app, write or local output)')
    upload_mode.add_argument('--dry-run-upload', action='store_true', help='log upload operations without sending or loading a token; local publishing still runs')
    parser.add_argument('--interval', type=float, help='publish interval in seconds')
    parser.add_argument('--delay', type=float, help='fixed delay in seconds')
    parser.add_argument('--feed-dir', help='finished-game feeds (default PITCHEEZY_FEED_DIR)')
    parser.add_argument('--watch-dir', help='watch datasets (default PITCHEEZY_WATCH_DIR)')
    parser.add_argument('--intent-dir', help='setup estimates (default PITCHEEZY_INTENT_DIR)')
    parser.add_argument('--demo-dir', help='demo clock folder (default PITCHEEZY_DEMO_DIR)')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    if args.check_remote:
        try:
            HttpSink(load_config(args.config)['upload']).check()
        except UploadError as error:
            raise SystemExit(f'remote check failed: {error}') from None
        print('remote metadata check passed: authenticated HEAD only; writes not tested')
        return 0
    config, publisher = _start(args)
    LOGGER.info('publishing to %s every %s s (fixed delay %s s)', config['out_dir'], config['publish_interval_s'], config['fixed_delay_s'])
    try:
        publisher.run(once=args.once)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.close()
    return 0


def _start(args):
    config = load_config(args.config, out_dir=args.out, publish_interval_s=args.interval, fixed_delay_s=args.delay,
                         dry_run_upload=args.dry_run_upload)
    if not config.get('out_dir'):
        raise SystemExit('no output folder: give --out <dir>, or set PITCHEEZY_PUBLISH_OUT_DIR or out_dir in publish_config.json')
    try:
        app, publisher = build(config, dates=args.date, feed_dir=args.feed_dir, watch_dir=args.watch_dir,
                               intent_dir=args.intent_dir, demo_dir=args.demo_dir)
    except (AlreadyRunning, Unsafe, UnsafeOutput, UploadError) as refused:
        raise SystemExit(str(refused)) from None
    try:
        feed = app.state.live_parts()
        if config['fixed_delay_s'] > getattr(feed, 'max_delay_s', config['fixed_delay_s']):
            raise SystemExit(f"fixed_delay_s {config['fixed_delay_s']} exceeds the live feed's max_delay_s {feed.max_delay_s}")
        if os.environ.get('PITCHEEZY_OBSERVER_LIVE_ARMB', '1') != '0':
            app.state.live_policy.start()  # warm the frozen policy in the background, as the server does at startup
    except BaseException:
        publisher.close()
        raise
    return config, publisher


if __name__ == '__main__':
    sys.exit(main())
