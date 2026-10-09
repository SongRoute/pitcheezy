"""Loopback-only SYNTHETIC whole-game walk-through: first -> last pitch with reveal, plus a read-only receipt log.

Nothing here is live, real or a model result. At start-up a scripted provider (the SVC-WS-02 nine-inning plan:
disconnect, store outage, crash restart, corrections, a same-stamp conflict) is polled once through the real
``LiveFeed.snapshot`` -> ``PersistentLiveFeed`` -> ``LiveSnapshotStore`` path into a temporary SQLite file, and
the served entry after every pitch is kept as a frame. Each viewer then steps through those frames: before a
reveal only the frame before the pitch is used, so the response cannot carry that pitch or any later one.

No predictor is connected, so the fan view never shows a probability. ``predict`` exists only so tests can
prove the freshness gate (``capture_input`` -> ``conditional_candidates``) abstains before calling it. The
plan's 129 keys are a fixture-only denominator, never one for an unknown real game.

    python -m observer_app.synthetic_game [--port 0]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import urllib.error
import uuid

import numpy as np

from .live_store import LiveSnapshotStore, PersistentLiveFeed, _canonical
from .service_research import capture_input, conditional_candidates

WEB = Path(__file__).resolve().parents[2]/'synthetic-game'
GAME, T0 = 31, 1_000_000.
CALLS = {'B': '볼', 'C': '루킹 스트라이크', 'S': '헛스윙', 'X': '인플레이'}
CHECKING = {'stale_provider_stamp': '제공처 시각이 앞선 자료보다 이르기 때문에 최신 자료로 확정하지 않았습니다.',
            'unorderable_provider_stamp': '제공처 시각이 없거나 형식이 맞지 않아 순서를 확정할 수 없습니다.',
            'same_stamp_conflict_provider_stamp': '같은 제공처 시각에 내용이 다른 자료가 있어 확정할 수 없습니다.',
            'provider_order_unknown': '자료 순서를 확인할 수 없습니다.'}
LIMITS = ['합성 경기 · 실제 경기나 실시간 중계가 아닙니다.',
          '예측 모델을 연결하지 않아 확률이나 추천을 표시하지 않습니다.',
          '수신 시각은 관측기 로컬 시계 기준 첫 *기록* 시각이며 절대 정확도는 검증되지 않았습니다.']


def plan():
    """Synthetic truth: (at_bat_number, pitch_number, play_id) per pitch. Nine innings, three batters a half,
    no bottom ninth, 1-4 pitches per at-bat."""
    keys, ab = [], 0
    for inning in range(1, 10):
        for half in ('top', 'bottom')[:1 if inning == 9 else 2]:
            for _ in range(3):
                ab += 1
                keys += [(ab, n, f'g{ab:02d}p{n}') for n in range(1, 2+ab % 4)]
    return keys


PLAN = plan()
STEP = {key[2]: i for i, key in enumerate(PLAN)}
LAST = len(PLAN)-1
GAP, OUTAGE, CRASH = range(60, 65), range(80, 85), range(100, 105)
STALE_STEP, UNORDERED_STEP, CONFLICT_STEP, DUP_STEPS = 40, 45, 30, {5, 15, 35, 50, 70, 90, 110, 120}
REVISIONS = [(20, 3, PLAN[18][2], 'C'), (61, 1, PLAN[59][2], 'C'), (63, 1, PLAN[59][2], 'S'),
             (81, 2, PLAN[79][2], 'C'), (81, 3, PLAN[79][2], 'S'), (101, 1, PLAN[99][2], 'C'), (103, 1, PLAN[99][2], 'S')]


def stamp(step, sub=0):
    return (datetime(2026, 10, 9, tzinfo=timezone.utc)+timedelta(seconds=10*step+sub)).strftime('%Y%m%d_%H%M%S')


def provider(step, sub=0, *, ts=None, conflict=False):
    """The scripted provider's cumulative feed after pitch ``step`` (same script as the SVC-WS-02 test)."""
    calls = {pid: call for at, s, pid, call in REVISIONS if (at, s) <= (step, sub)}
    plays = {}
    for i, (ab, n, pid) in enumerate(PLAN[:step+1]):
        if (ab, n) == (11, 2) and step < STEP['g11p3']:
            pid = 'g11p3'  # provider first sent g11p3 as pitch 2, then renumbered
        if (ab, n) == (5, 1) and step == i:
            pid = None  # first sighting before the provider assigned a playId
        events = plays.setdefault(ab, [])
        if ab == 20 and n == 1:
            events.append({'isPitch': False, 'type': 'action', 'details': {'eventType': 'pitching_substitution'}})
        if ab == 30 and n == 2:
            events.append({'isPitch': False, 'type': 'action', 'details': {'eventType': 'offensive_substitution'}})
        call = 'X' if conflict and i == step else calls.get(pid, 'B')
        events.append({'isPitch': True, 'playId': pid, 'pitchNumber': n, 'details': {'call': {'code': call}},
                       'startTime': stamp(i)})
    meta = {'eventWall': T0+10*step, 'timeStamp': stamp(step, sub) if ts is None else ts}
    if ts is False:
        del meta['timeStamp']
    return {'metaData': meta, 'liveData': {'plays': {'allPlays': [
        {'about': {'atBatIndex': ab-1, 'inning': (ab-1)//6+1, 'halfInning': 'top' if (ab-1) % 6 < 3 else 'bottom'},
         'matchup': {'batter': 600+ab if ab == 30 and len(events) > 2 else 100+ab, 'pitcher': 1 if ab < 20 else 2},
         'playEvents': events} for ab, events in plays.items()]}}}


class _Store(LiveSnapshotStore):
    """Scripted store outage; counts ingest outcomes so duplicates stay separate from ledger events."""
    def __init__(self, path, script):
        super().__init__(path)
        self.script = script

    def ingest(self, game_pk, feed, **kwargs):
        if self.script.down:
            raise sqlite3.OperationalError('scripted store outage')
        outcome, entry = super().ingest(game_pk, feed, **kwargs)
        self.script.outcomes.append((kwargs['source'], outcome))
        return outcome, entry


class Script:
    """Runs the scripted game once and keeps one frame per pitch step."""

    def __init__(self, directory):
        self.path, self.down, self.outcomes, self.frames = Path(directory)/'synthetic-game.sqlite3', False, [], []
        now, serving = [T0], [None]

        def transport(url, timeout):
            if isinstance(serving[0], Exception):
                raise serving[0]
            return serving[0]

        def start():
            return PersistentLiveFeed(transport, store=_Store(self.path, self), delay_s=0, clock=lambda: now[0],
                                      wall=lambda: now[0], sleep=lambda _: None)

        def poll(feed, at):
            serving[0], now[0] = feed, at
            live.fetched_at.clear()
            return live.snapshot(GAME)[0]

        live = start()
        for step in range(LAST+1):
            at = T0+10*step+1
            self.down = step in OUTAGE or step in CRASH
            if step == CRASH.stop:  # process crash during the outage: memory is gone, a new process restores
                live = start()
            if step in GAP:
                served = poll(urllib.error.URLError('scripted disconnect'), at)
            else:
                ts = stamp(step-3) if step == STALE_STEP else False if step == UNORDERED_STEP else None
                served = poll(provider(step, ts=ts), at)
                if step in DUP_STEPS:
                    served = poll(provider(step), at+.5)
                for s in sorted({s for st, s, _, _ in REVISIONS if st == step}):
                    served = poll(provider(step, s), at+s/10)
                if step == CONFLICT_STEP:
                    served = poll(provider(step, conflict=True), at+.2)  # same provider stamp, different payload
            ledger = live.store.pitch_events(GAME)
            store_ok = live.store_error is None
            committed = served['feed'] if served and store_ok else committed if step else None
            self.frames.append({'entry': dict(served) if served else None, 'disconnected': GAME in live.failing,
                                'store_ok': store_ok, 'committed_feed': committed, 'as_of': ledger['history']['as_of'],
                                'history': ledger['history']})
        self.store = live.store


def _situation(feed):
    """Latest received play: inning/half/batter/pitcher and the count after its received pitches."""
    plays = feed['liveData']['plays']['allPlays']
    play = plays[-1]
    calls = [e['details']['call']['code'] for e in play['playEvents'] if e.get('isPitch')]
    return {'inning': play['about']['inning'], 'half': '초' if play['about']['halfInning'] == 'top' else '말',
            'batter': play['matchup']['batter'], 'pitcher': play['matchup']['pitcher'],
            'count': {'balls': min(3, calls.count('B')), 'strikes': min(2, sum(c in 'CS' for c in calls))}}


def _pitch_event(feed, key):
    """The playEvent recorded for ``key`` in this frame's feed, or None if it had not arrived."""
    ab, n, _ = key
    for play in feed['liveData']['plays']['allPlays'] if feed else []:
        if play['about']['atBatIndex'] == ab-1:
            for event in play['playEvents']:
                if event.get('isPitch') and event.get('pitchNumber') == n:
                    return event
    return None


def _received(feed, key):
    """The call recorded for ``key`` in this frame's feed, or None if it had not arrived."""
    event = _pitch_event(feed, key)
    return event['details']['call']['code'] if event else None


def _sha(value):
    return value if isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value) else None


def _shown(frame, key, rows):
    """What the fan view may say about ``key`` as of ``frame``: (result, status, reason). A body the ledger
    quarantined at the selected seq is never shown as an observed call, and a memory-only call that differs from
    the last committed body is not durable. A quarantine row refuses the display only if its hash is the displayed
    pitch body's (the ledger's ``sha256(_canonical(playEvent))``): a stale arrival's row also names the selected
    seq but not its body. An unusable hash on either side fails closed as unconfirmed."""
    entry = frame['entry']
    event = _pitch_event(entry['feed'] if entry else None, key)
    call = event['details']['call']['code'] if event else None
    if not call:
        return None, 'not_received', None
    at_seq = [r for r in rows if r['outcome'] == 'quarantine' and r['store_seq'] == entry.get('seq')]
    if at_seq:
        try:
            shown = hashlib.sha256(_canonical(event)).hexdigest()
        except (TypeError, ValueError):
            shown = None
        if shown is None or any(_sha(r['sha']) is None for r in at_seq):
            return '확인 중', 'unconfirmed', f"{entry.get('provider_order')}/body_hash_unusable"
        refused = [r for r in at_seq if r['sha'] == shown]
        if refused:
            return '확인 중', 'unconfirmed', f"{entry.get('provider_order')}/{refused[-1]['reason']}"
    if not frame['store_ok'] and call != _received(frame.get('committed_feed'), key):
        return f'{CALLS[call]} · 저장 전', 'unsaved', 'uncommitted_memory_entry'
    return CALLS[call], 'received', None


def comparison(entry, decision_wall, predict=None):
    """Plain fan state for the pre-pitch comparison plus the gate's machine reason. Never a probability."""
    if entry is None:
        return {'state': 'checking', 'label': '자료 확인 중', 'reason': '아직 받은 경기 자료가 없습니다.',
                'gate': {'status': 'unsupported', 'reason': 'no_snapshot'}}
    try:
        receipt = capture_input(entry, {'game_pk': GAME}, game_pk=GAME, trusted_clock=True,
                                decision_at=datetime.fromtimestamp(decision_wall, timezone.utc).isoformat())
    except (KeyError, ValueError) as exc:  # a memory-only entry (store down) has no committed identity
        return {'state': 'unsupported', 'label': '지원 없음',
                'reason': '저장소에 확정되지 않은 임시 자료라 비교하지 않습니다.',
                'gate': {'status': 'unsupported', 'reason': 'uncommitted_memory_entry', 'error': str(exc)}}
    gate = {'status': 'unsupported', 'reason': receipt['freshness_reason'] or 'no_predictor_connected',
            'receipt': {k: receipt[k] for k in ('store_seq', 'slot_version', 'feed_sha256', 'provider_order',
                                               'provider_generated_at', 'first_received_at', 'version_received_at',
                                               'feed_receipt_status')}}
    if predict is not None:
        vectors = np.zeros((1, 199))
        evidence = {'kind': 'synthetic_fixture', 'input_sha256': receipt['input_sha256'],
                    'vectors_sha256': hashlib.sha256(vectors.tobytes()).hexdigest(),
                    'actions_sha256': hashlib.sha256(json.dumps(['FF']).encode()).hexdigest(),
                    'model_sha256': 'a'*64, 'encoder_sha256': 'b'*64}
        result = conditional_candidates(receipt, vectors, ['FF'], predict, model_sha256='a'*64, encoder_sha256='b'*64,
                                        support_counts=[40], input_evidence=evidence)
        gate.update(status=result['status'], reason=result.get('reason'))
    if receipt['freshness_reason']:
        return {'state': 'checking', 'label': '자료 확인 중', 'reason': CHECKING[receipt['freshness_reason']], 'gate': gate}
    return {'state': 'waiting', 'label': '예측 대기', 'reason': '합성 경기에는 예측 모델을 연결하지 않았습니다.', 'gate': gate}


class Game:
    def __init__(self, script, predict=None):
        self.script, self.predict = script, predict

    def _pitch_log(self, frame, upto, shown_from):
        """Ledger rows committed by ``frame`` for the first ``upto`` plan pitches (all already revealed). Frames
        ``i..shown_from-1`` were already shown for pitch ``i``; a differing earlier display is marked, not hidden."""
        rows = [r for r in self.script.store.pitch_events(GAME)['events'] if r['id'] <= (frame['as_of']['pitch_event_id'] or 0)]
        log = []
        for i, key in enumerate(PLAN[:upto]):
            mine = [r for r in rows if (r['at_bat_number'], r['pitch_number']) == key[:2] or r['provider_event_id'] == key[2]]
            accepted = [r for r in mine if r['outcome'] != 'quarantine' and (r['at_bat_number'], r['pitch_number']) == key[:2]]
            result, status, reason = _shown(frame, key, mine)
            corrected = any(r['outcome'] == 'correction' for r in accepted)
            earlier = [_shown(self.script.frames[j], key, mine)[0] for j in range(i, shown_from)]
            changed = [s for s in earlier if s is not None and s != result]
            log.append({'game_pitch': i+1, 'inning': (key[0]-1)//6+1, 'result': result or '아직 받지 못함',
                        'status': status, 'corrected': corrected,
                        'changed_from': changed[-1] if changed and not corrected else None,
                        'diagnostics': {'key': {'game_pk': GAME, 'at_bat_number': key[0], 'pitch_number': key[1]},
                                        'display_reason': reason,
                                        'provider_event_ids': sorted({r['provider_event_id'] for r in accepted if r['provider_event_id']}),
                                        'first_recorded_receipt': accepted[0]['first_received'] if accepted else None,
                                        'receipt_clock': accepted[0]['receipt_clock'] if accepted else 'unknown',
                                        'trail': [{k: r[k] for k in ('id', 'outcome', 'reason', 'revision', 'provider_ts',
                                                                     'received', 'receipt_clock', 'store_seq')} for r in mine]}})
        return log

    def state(self, cursor, revealed):
        if not 0 <= cursor <= LAST:
            raise ValueError('Cursor outside synthetic game')
        frame = self.script.frames[cursor] if revealed else self.script.frames[cursor-1] if cursor else \
            {'entry': None, 'disconnected': False, 'store_ok': True, 'as_of': {'store_seq': None, 'pitch_event_id': None},
             'history': {'status': 'UNKNOWN', 'reasons': ['intermediate_revisions_unlogged'], 'as_of': None}}
        entry = frame['entry']
        view = {'schema': 'synthetic-game-view-v1', 'synthetic': True, 'live': False, 'game_label': '합성 경기',
                'game_pitch': cursor+1, 'reveal': revealed, 'situation': _situation(entry['feed']) if entry else None,
                'connection': '연결이 끊겨 마지막으로 받은 자료를 보여 줍니다.' if frame['disconnected'] else None,
                'comparison': None, 'actual': None,
                'history': self._pitch_log(frame, cursor+int(revealed), cursor if revealed else max(cursor-1, 0)),
                'complete': False, 'partition': None, 'limitations': LIMITS,
                'diagnostics': {'selected': None if not entry else {
                                    'store_seq': entry.get('seq'), 'slot_version': entry.get('version'),
                                    'provider_order': entry.get('provider_order'), 'restored': bool(entry.get('restored')),
                                    'provider_generated_at': entry['feed']['metaData'].get('timeStamp')},
                                'store_ok': frame['store_ok'], 'history': frame['history'],
                                'clock_note': '관측기 로컬 시계 · 첫 기록 수신 시각(상한) · 절대 정확도 미검증'}}
        if not revealed:
            view['comparison'] = comparison(entry, T0+10*cursor+.5, self.predict)
        else:
            row = view['history'][cursor]
            view['actual'] = {'game_pitch': cursor+1, 'result': None if row['status'] == 'not_received' else row['result'],
                              'status': row['status'], 'reason': row['diagnostics']['display_reason'],
                              'note': {'not_received': '이 시점에는 이 투구 자료를 받지 못했습니다.',
                                       'unconfirmed': '받은 자료가 저장소 검증을 통과하지 못해 결과를 확정하지 않았습니다.',
                                       'unsaved': '저장소에 기록되기 전의 임시 자료입니다.'}.get(row['status'])}
            if cursor == LAST:
                view['complete'], view['partition'] = True, self.partition()
        return view

    def partition(self):
        """Known synthetic final boundary only: plan keys split into accepted / quarantined-only / missing, extra
        observed keys apart; ingest/ledger event counts kept separate."""
        events = self.script.store.pitch_events(GAME)['events']
        key = lambda e: (e['at_bat_number'], e['pitch_number'], e['provider_event_id'])
        accepted = {key(e) for e in events if e['outcome'] != 'quarantine'}
        quarantined = {key(e) for e in events if e['outcome'] == 'quarantine'}
        truth = set(PLAN)
        parts = {'accepted': truth & accepted, 'quarantined_only': (truth & quarantined)-accepted,
                 'missing': truth-accepted-quarantined, 'extra': accepted-truth}
        return {'scope': 'synthetic_fixture_only', 'expected_keys': len(truth),
                **{name: sorted(map(list, keys)) for name, keys in parts.items()},
                'counts': {name: len(keys) for name, keys in parts.items()},
                'events': {'live_duplicates': sum(o == ('live_poll', 'duplicate') for o in self.script.outcomes),
                           'corrections': sum(e['outcome'] == 'correction' for e in events),
                           'quarantines': sum(e['outcome'] == 'quarantine' for e in events), 'rows': len(events)},
                'history': self.script.store.pitch_events(GAME)['history'],
                'note': '합성 계획 기준 키 집계입니다. 실제 경기의 완전성이나 모든 중간 정정·최초 도착을 뜻하지 않습니다.'}


def make_handler(game):
    sessions, guard = {}, threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _session(self):
            cookies = self.headers.get('Cookie', '').split(';')
            existing = next((c.strip()[10:] for c in cookies if c.strip().startswith('synthgame=')), None)
            with guard:
                if existing not in sessions:
                    existing = uuid.uuid4().hex
                    sessions[existing] = [0, False]
            return existing

        def _json(self, obj, cookie=None, status=HTTPStatus.OK):
            data = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            if cookie:
                self.send_header('Set-Cookie', f'synthgame={cookie}; HttpOnly; SameSite=Strict; Path=/')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ('/api/state', '/api/log'):
                session = self._session()
                with guard:
                    cursor, revealed = sessions[session]
                view = game.state(cursor, revealed)
                if self.path == '/api/log':
                    view = {k: view[k] for k in ('schema', 'synthetic', 'live', 'game_pitch', 'reveal', 'history',
                                                 'diagnostics', 'partition')}
                self._json(view, cookie=session)
                return
            routes = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
            path = routes.get(self.path)
            if not path:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = (WEB/path).read_bytes()
            media = 'text/html' if path.endswith('.html') else 'text/javascript' if path.endswith('.js') else 'text/css'
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', media+'; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if self.path not in ('/api/reveal', '/api/next', '/api/restart'):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            session = self._session()
            with guard:
                cursor, revealed = sessions[session]
                if self.path == '/api/reveal':
                    if revealed:
                        self._json({'detail': '이미 공개했습니다.'}, status=HTTPStatus.CONFLICT)
                        return
                    revealed = True
                elif self.path == '/api/next':
                    if not revealed or cursor == LAST:
                        self._json({'detail': '다음 투구를 열 수 없습니다.'}, status=HTTPStatus.CONFLICT)
                        return
                    cursor, revealed = cursor+1, False
                else:
                    cursor, revealed = 0, False
                sessions[session] = [cursor, revealed]
            self._json(game.state(cursor, revealed), cookie=session)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--port', type=int, default=0, help='127.0.0.1 port; 0 picks a free one')
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix='pz-synthetic-game-') as directory:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(Game(Script(directory))))
        print(f'SYNTHETIC game (not live) http://127.0.0.1:{server.server_port}/', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
