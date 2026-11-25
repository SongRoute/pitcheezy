"""Server-owned shared demo clock (pitcheezy-service-v2 §4-6).

One session = one game + one mapping (``reveal_at_s`` per pitch index) + a clock. Every viewer of the
session computes the same cutoff from the persisted anchor, so reload/reconnect/another device shows the
same state. Storage is a separate directory (``PITCHEEZY_DEMO_DIR``): one JSON file per session and the
mapping files; nothing here writes live snapshots, the observer DB or research data.

Clock: ``progress_s = anchor.progress_s + (now - anchor.server_epoch) * rate`` while running, frozen
otherwise, never below the anchor (a wall clock stepping back does not un-reveal) and capped at the
segment's last row. The server does not control any video player.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
import uuid
from pathlib import Path

MAPPING_SCHEMA = 'pitcheezy-demo-mapping-v1'
SESSION_SCHEMA = 'pitcheezy-demo-session-v1'
KINDS = ('video_verified', 'game_timestamps_unsynced', 'test_fixture', 'pa_end_observed')
# pa_end_observed: only PA-ENDING pitches carry a time, each one a person's observation of the PA result on the video; a
# whole PA is revealed at that time. Pitches inside a PA have no time (none is interpolated); never 'verified'.
PA_END_METHOD = 'human_observation_pa_end_result_visible'
ACTIVE = ('ready', 'running', 'paused')
RATE_MIN, RATE_MAX = 0.25, 4.0
DEFAULT_DIR = Path.home()/'.pitcheezy-demo'
# video_verified provenance: a local file (bytes hashed, video.sha256) or a remote stream a person watched
# (video.source remote_observed: no bytes, so no content hash; every row carries its own human review).
REMOTE_SOURCE = 'remote_observed'
REMOTE_METHOD = 'human_observation_result_visible'
YOUTUBE_ID = re.compile(r'[A-Za-z0-9_-]{11}')
SHA256 = re.compile(r'[0-9a-f]{64}')


class DemoError(Exception):
    def __init__(self, status, detail, payload=None):
        super().__init__(detail)
        self.status, self.detail, self.payload = status, detail, payload


def _iso(epoch):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(epoch))


def validate_mapping(mapping, keys=None):
    """Checks a mapping document; ``keys`` = dataset pitch keys by index (verifies every row's key)."""
    if mapping.get('schema') != MAPPING_SCHEMA or mapping.get('kind') not in KINDS:
        raise ValueError('mapping schema/kind')
    if mapping['kind'] != 'video_verified' and mapping.get('verified'):
        raise ValueError('only video_verified mappings may be verified')
    remote = (mapping.get('video') or {}).get('source') == REMOTE_SOURCE
    if mapping['kind'] == 'video_verified' and not remote and not (mapping.get('verified') and (mapping.get('video') or {}).get('sha256')):
        raise ValueError('video_verified needs verified=true and a video sha256')
    if remote:
        _validate_remote(mapping)
    if mapping['kind'] == 'pa_end_observed':
        _validate_pa_end(mapping)
    rows = mapping.get('rows') or []
    if not rows:
        raise ValueError('mapping has no rows')
    for before, after in zip(rows, rows[1:]):
        if not after['index'] > before['index'] or after['reveal_at_s'] < before['reveal_at_s']:
            raise ValueError('mapping rows must have increasing index and non-decreasing reveal_at_s')
    for row in rows:
        if type(row['index']) is not int or row['index'] < 0 or not isinstance(row['reveal_at_s'], (int, float)):
            raise ValueError('mapping row types')
        if keys is not None and (row['index'] >= len(keys) or keys[row['index']] != row['pitch_key']):
            raise ValueError(f"mapping row {row['index']} pitch_key differs from the game record")
    return mapping


def _validate_remote(mapping):
    """A remote_observed video is verified only by a person's per-row observation of the result-visible frame."""
    video, check = mapping['video'], mapping.get('verification') or {}
    if mapping['kind'] != 'video_verified' or mapping.get('verified') is not True:
        raise ValueError('remote_observed video belongs to a verified video_verified mapping')
    if video.get('sha256') is not None:
        raise ValueError('remote_observed video has no content sha256 (bytes were never read)')
    vid = video.get('video_id')
    if video.get('platform') != 'youtube' or not isinstance(vid, str) or not YOUTUBE_ID.fullmatch(vid) \
            or video.get('url') != f'https://www.youtube.com/watch?v={vid}':
        raise ValueError('remote_observed video needs a canonical YouTube url and video_id')
    duration = video.get('observed_duration_s')
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('remote_observed video needs a finite observed_duration_s')
    if check.get('method') != REMOTE_METHOD or not str(check.get('by') or '').strip() or not str(check.get('observed_on') or '').strip() \
            or not SHA256.fullmatch(str(check.get('csv_sha256') or '')):
        raise ValueError('remote_observed verification needs method, by, observed_on and csv_sha256')
    rows = mapping.get('rows') or []
    for before, after in zip(rows, rows[1:]):
        if after.get('index') != before.get('index', -2)+1 or not after.get('reveal_at_s', -1) > before.get('reveal_at_s', math.inf):
            raise ValueError('remote_observed rows must be consecutive pitches with increasing result-visible times')
    for row in rows:
        at, review = row.get('reveal_at_s'), row.get('review') or {}
        if not isinstance(at, (int, float)) or isinstance(at, bool) or not math.isfinite(at) or not 0 <= at <= duration:
            raise ValueError(f"remote_observed row {row.get('index')} time outside 0..observed_duration_s")
        if not str(review.get('by') or '').strip() or not str(review.get('evidence') or '').strip():
            raise ValueError(f"remote_observed row {row.get('index')} lacks a per-row review (by, evidence)")


def _validate_pa_end(mapping):
    if mapping.get('verified') is not False:
        raise ValueError('pa_end_observed mappings are never verified (pitch-level sync is unknown)')
    if (mapping.get('verification') or {}).get('method') != PA_END_METHOD:
        raise ValueError('pa_end_observed needs verification.method human_observation_pa_end_result_visible')
    rows, seg = mapping.get('rows') or [], mapping.get('segment') or {}
    first = seg.get('first_index')
    if type(first) is not int or first < 0 or not rows or first > rows[0].get('index', -1):
        raise ValueError('pa_end_observed needs segment.first_index (first pitch of the first PA) <= first row index')
    for before, after in zip(rows, rows[1:]):
        if not after.get('reveal_at_s', -1) > before.get('reveal_at_s', math.inf):
            raise ValueError('pa_end_observed rows need strictly increasing observed times')
    for row in rows:
        at, review = row.get('reveal_at_s'), row.get('review') or {}
        if not isinstance(at, (int, float)) or isinstance(at, bool) or not math.isfinite(at) or at < 0:
            raise ValueError(f"pa_end_observed row {row.get('index')} needs a finite observed time")
        if not str(review.get('by') or '').strip() or not str(review.get('evidence') or '').strip():
            raise ValueError(f"pa_end_observed row {row.get('index')} lacks a per-row review (by, evidence)")
    lead = mapping.get('lead_s', 5.0)
    if not isinstance(lead, (int, float)) or isinstance(lead, bool) or not math.isfinite(lead) or not 0 <= lead <= rows[0]['reveal_at_s']:
        raise ValueError('pa_end_observed lead_s must be within 0..first observed time')


class DemoClock:
    def __init__(self, directory=None, *, wall=time.time):
        self.directory = Path(directory or os.environ.get('PITCHEEZY_DEMO_DIR', DEFAULT_DIR))
        self.wall = wall
        self._lock = threading.RLock()

    # storage ------------------------------------------------------------------------------------
    def _sessions_dir(self):
        path = self.directory/'sessions'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def token(self):
        """Operator capability; created once (0600) inside the demo dir, never served to viewers."""
        path = self.directory/'operator.token'
        with self._lock:
            if not path.is_file():
                self.directory.mkdir(parents=True, exist_ok=True)
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'w') as handle:
                    handle.write(secrets.token_urlsafe(24))
            return path.read_text().strip()

    def check_token(self, supplied):
        if not supplied or not secrets.compare_digest(str(supplied), self.token()):
            raise DemoError(401, '운영자 인증이 필요합니다.')

    def mappings(self):
        out = []
        folder = self.directory/'mappings'
        for path in sorted(folder.glob('*.json')) if folder.is_dir() else []:
            try:
                out.append(self.mapping(path.stem))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
        return out

    def mapping(self, mapping_id):
        if not isinstance(mapping_id, str) or not mapping_id.replace('-', '').replace('_', '').isalnum():
            raise DemoError(400, 'mapping_id 형식이 올바르지 않습니다.')
        path = self.directory/'mappings'/f'{mapping_id}.json'
        if not path.is_file():
            raise DemoError(404, '대응표를 찾을 수 없습니다.')
        raw = path.read_bytes()
        mapping = validate_mapping(json.loads(raw))
        if mapping.get('mapping_id') != mapping_id:
            raise ValueError('mapping_id differs from file name')
        return mapping | {'source_sha256': hashlib.sha256(raw).hexdigest()}

    def _path(self, session_id):
        if not isinstance(session_id, str) or len(session_id) != 32 or not all(c in '0123456789abcdef' for c in session_id):
            raise DemoError(404, '시연 세션을 찾을 수 없습니다.')
        return self._sessions_dir()/f'{session_id}.json'

    def load(self, session_id):
        path = self._path(session_id)
        if not path.is_file():
            raise DemoError(404, '시연 세션을 찾을 수 없습니다.')
        return json.loads(path.read_text())

    def _save(self, session):
        path = self._path(session['session_id'])
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(session, ensure_ascii=False))
        os.replace(tmp, path)  # atomic: readers see the old or the new state, never half

    def sessions(self):
        out = []
        folder = self.directory/'sessions'
        for path in sorted(folder.glob('*.json')) if folder.is_dir() else []:
            try:
                out.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def active_games(self):
        return {s['game_pk'] for s in self.sessions() if s.get('status') in ACTIVE}

    # clock --------------------------------------------------------------------------------------
    def progress(self, session, now=None):
        now = self.wall() if now is None else now
        anchor = session['anchor']
        value = anchor['progress_s']
        if session['status'] == 'running':
            value += max(0.0, now-anchor['server_epoch'])*session['rate']
        return min(value, session['rows'][-1]['reveal_at_s'])

    def cutoff(self, session, now=None):
        """Global pitch index of the last revealed pitch (-1 = none). Pitches before the segment's
        start row are always revealed (pre-segment state)."""
        progress = self.progress(session, now)
        revealed = [row['index'] for row in session['rows'] if row['reveal_at_s'] <= progress and row['index'] >= session['start_index']]
        base = session.get('pre_index', session['start_index'])-1  # pa_end_observed: state before the segment's first PA
        return max([base, *revealed])

    def _row(self, session, index):
        row = next((r for r in session['rows'] if r['index'] == index), None)
        if row is None:
            raise DemoError(400, '대응표에 없는 투구입니다.')
        return row

    def _start_progress(self, session):
        """Progress at creation/reset: ``lead_s`` before the start row's reveal (start row not yet revealed)."""
        return self._row(session, session['start_index'])['reveal_at_s']-session.get('lead_s', 5.0)

    def create(self, game_pk, mapping, keys, *, start_index=None, op_id=None, lead_s=5.0):
        validate_mapping(mapping, keys)
        if mapping['game_pk'] != game_pk:
            raise DemoError(400, '대응표의 경기가 다릅니다.')
        with self._lock:
            if op_id:
                for existing in self.sessions():
                    if existing.get('create_op_id') == op_id:
                        return existing
            rows = [{'index': r['index'], 'pitch_key': r['pitch_key'], 'reveal_at_s': float(r['reveal_at_s'])} for r in mapping['rows']]
            start = rows[0]['index'] if start_index is None else start_index
            now = self.wall()
            session = {'schema': SESSION_SCHEMA, 'session_id': uuid.uuid4().hex, 'game_pk': game_pk,
                       'mapping': {'mapping_id': mapping['mapping_id'], 'kind': mapping['kind'], 'verified': bool(mapping.get('verified')),
                                   'video_sha256': (mapping.get('video') or {}).get('sha256'), 'source_sha256': mapping['source_sha256']}
                       | ({'video_source': REMOTE_SOURCE, 'video_url': mapping['video']['url']}
                          if (mapping.get('video') or {}).get('source') == REMOTE_SOURCE else {}),
                       'rows': rows, 'start_index': start, 'lead_s': float(mapping.get('lead_s', lead_s) if mapping['kind'] == 'pa_end_observed' else lead_s), 'status': 'ready', 'rate': 1.0,
                       'version': 1, 'anchor': {'server_epoch': now, 'progress_s': 0.0}, 'ops': [],
                       'create_op_id': op_id, 'created_at': _iso(now), 'updated_at': _iso(now)}
            self._row(session, start)
            if mapping['kind'] == 'pa_end_observed':
                if start != rows[0]['index']:
                    raise DemoError(400, '타석 끝 관찰 대응표는 첫 관찰 행에서만 시작합니다.')
                session['pre_index'] = mapping['segment']['first_index']
            session['anchor']['progress_s'] = self._start_progress(session)
            self._save(session)
            return session

    def control(self, session_id, action, *, expected_version=None, op_id=None, index=None, rate=None):
        with self._lock:
            session = self.load(session_id)
            if op_id and op_id in session['ops']:
                return session  # idempotent replay of an applied command
            if expected_version is not None and expected_version != session['version']:
                raise DemoError(409, '다른 조작이 먼저 반영되었습니다. 새 상태를 확인해 주세요.', session)
            if session['status'] == 'ended':
                raise DemoError(410, '끝난 시연 세션입니다.')
            now = self.wall()
            current = self.progress(session, now)
            status = session['status']
            if action == 'start' and status == 'ready' or action == 'resume' and status == 'paused':
                session['status'], session['anchor'] = 'running', {'server_epoch': now, 'progress_s': current}
            elif action == 'pause' and status == 'running':
                session['status'], session['anchor'] = 'paused', {'server_epoch': now, 'progress_s': current}
            elif action == 'seek':
                if type(index) is not int:
                    raise DemoError(400, 'index가 필요합니다.')
                if index < session['start_index']:
                    raise DemoError(400, '시연 구간 시작 전으로는 이동할 수 없습니다. 새 세션을 만들어 주세요.')
                at = self._row(session, index)['reveal_at_s']
                session['anchor'] = {'server_epoch': now, 'progress_s': at}
                if status == 'ready':
                    session['status'] = 'paused'
            elif action == 'reset':
                session['status'] = 'ready'
                session['anchor'] = {'server_epoch': now, 'progress_s': self._start_progress(session)}
                session['rate'] = 1.0
            elif action == 'rate':
                if not isinstance(rate, (int, float)) or not RATE_MIN <= rate <= RATE_MAX:
                    raise DemoError(400, f'속도는 {RATE_MIN}~{RATE_MAX} 사이여야 합니다.')
                session['anchor'], session['rate'] = {'server_epoch': now, 'progress_s': current}, float(rate)
            elif action == 'end':
                session['status'] = 'ended'
                session['anchor'] = {'server_epoch': now, 'progress_s': current}
            elif action in ('start', 'resume', 'pause'):
                raise DemoError(409, f'지금 상태({status})에서는 할 수 없는 조작입니다.', session)
            else:
                raise DemoError(400, '알 수 없는 조작입니다.')
            session['version'] += 1
            session['updated_at'] = _iso(now)
            if op_id:
                session['ops'] = (session['ops']+[op_id])[-50:]
            self._save(session)
            return session

    def public(self, session, keys, now=None):
        now = self.wall() if now is None else now
        index = self.cutoff(session, now)
        return {'session_id': session['session_id'], 'game_pk': session['game_pk'], 'status': session['status'],
                'rate': session['rate'], 'server_time': _iso(now), 'version': session['version'],
                'cutoff': {'index': index, 'pitch_key': keys[index] if index >= 0 else None}}

    def operator_view(self, session, keys, now=None):
        now = self.wall() if now is None else now
        progress = self.progress(session, now)
        index = self.cutoff(session, now)
        upcoming = next((r for r in session['rows'] if r['index'] > index and r['index'] >= session['start_index']), None)
        rows = session['rows']
        return self.public(session, keys, now) | {
            'mapping': session['mapping'], 'start_index': session['start_index'], 'progress_s': round(progress, 3),
            'video_time_s': round(progress, 3), 'anchor': session['anchor'],
            'next_reveal': None if upcoming is None else {
                'index': upcoming['index'], 'pitch_key': upcoming['pitch_key'], 'at_video_s': upcoming['reveal_at_s'],
                'in_s': None if session['status'] != 'running' else round((upcoming['reveal_at_s']-progress)/session['rate'], 2)},
            'segment': {'first_index': rows[0]['index'], 'last_index': rows[-1]['index']},
            'created_at': session['created_at'], 'updated_at': session['updated_at'],
            'viewer_url_path': f"/watch/{session['game_pk']}?demo={session['session_id']}"}
