"""Service game API v2 (/api/service/**) and the demo operator API; see CONTRACT pitcheezy-service-v2.

Namespaces: ``archive`` (stored final feeds + watch datasets, real live feed) and ``demo`` (``?demo=<id>``,
one game cut at the server-owned demo clock). While a demo session is active its game is embargoed on
every archive/raw path (423, or redacted in lists) so a demo client cannot read ahead by accident.
"""
from __future__ import annotations

from datetime import date as Date, timedelta
import gzip
import json
import os
import re
import time
from pathlib import Path

from fastapi import Header, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import game_history, player_metadata
from .demo_clock import DemoError

DEFAULT_FEED_DIR = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/feeds')
EMBARGO_PATH = re.compile(r'^/api/(?:watch|replay|live)/(\d+)(?:/|$)')
NO_STORE = {'Cache-Control': 'no-store'}
INTERNAL = ('situation_after', 'gap_after', 'line_before', 'line_after')  # server-only: gap_after would reveal that something happens later
LIVE_STALE_S = 30


class CreateDemo(BaseModel):
    model_config = ConfigDict(extra='forbid')
    game_pk: int = Field(strict=True, gt=0)
    mapping_id: str
    start_index: int | None = Field(default=None, strict=True, ge=0)
    op_id: str | None = Field(default=None, max_length=80)


class ControlDemo(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: str
    expected_version: int | None = Field(default=None, strict=True, ge=1)
    op_id: str | None = Field(default=None, max_length=80)
    index: int | None = Field(default=None, strict=True, ge=0)
    rate: float | None = None


def _read_feed(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else open(path)) as handle:
        return json.load(handle)


class GameArchive:
    """Completed games: ``<feed_dir>/<pk>_final.json.gz`` + optional watch dataset + CV setup rows."""

    def __init__(self, feed_dir=None, watch=None, setups=None, snapshot_dir=None):
        self.feed_dir = Path(feed_dir or os.environ.get('PITCHEEZY_FEED_DIR', DEFAULT_FEED_DIR))
        self.watch, self.setups, self._cache = watch, setups, {}
        self.snapshot_dir = snapshot_dir or os.environ.get('PITCHEEZY_PLAYER_SNAPSHOT_DIR') or None

    def game_pks(self):
        if not self.feed_dir.is_dir():
            return []
        return sorted(int(p.name.split('_')[0]) for p in self.feed_dir.glob('*_final.json*') if p.name.split('_')[0].isdigit())

    def record(self, game_pk):
        path = next(iter(sorted(self.feed_dir.glob(f'{int(game_pk)}_final.json*'))), None) if self.feed_dir.is_dir() else None
        if path is None:
            raise KeyError(game_pk)
        stamp = path.stat().st_mtime_ns
        cached = self._cache.get(game_pk)
        if cached is None or cached[0] != stamp:
            watch = setup = None
            if self.watch is not None:
                try:
                    watch = self.watch.load(int(game_pk))
                except KeyError:
                    watch = None
            if self.setups is not None:
                try:
                    setup = self.setups.view(int(game_pk))
                except (KeyError, ValueError):
                    setup = None
            record = game_history.build(_read_feed(path), watch, setup)
            record['pregame'] = player_metadata.load_pregame(self.snapshot_dir, int(game_pk), record['official_date'])
            cached = self._cache[game_pk] = (stamp, record)
        return cached[1]

    def cards(self):
        out = []
        for pk in self.game_pks():
            try:
                out.append(self.record(pk))
            except (KeyError, ValueError, OSError, json.JSONDecodeError):
                continue
        return out


def keys_of(record):
    return [p['key'] for p in record['pitches']]


def _pregame_view(record, people):
    """Pre-game stats of the people shown in this view only (pitchers: ERA, batters: AVG/OPS); None without a cache."""
    cache = record.get('pregame')
    if cache is None:
        return None
    players = {}
    for role, person in people:
        entry = cache['players'].get(str((person or {}).get('id')))
        if entry is None:
            continue
        stats = entry['pitching'] if role == 'pitcher' else entry['batting']
        keys = ('era',) if role == 'pitcher' else ('avg', 'ops')
        out = players.setdefault(str(person['id']), {'as_of': entry['as_of'], 'source': entry['source']})
        out |= {k: (stats or {}).get(k) for k in keys}
    return {'basis': cache['date_basis'], 'as_of': cache['as_of'], 'players': players}


def cut_view(record, cutoff, source, *, feed=None, live_current=None, next_rec=None, after=None):
    """The game as visible at pitch index ``cutoff`` (inclusive). Nothing past it is read into the output
    except the pre-pitch view of pitch ``cutoff+1`` (situation, batter, pitcher, stored pre-pitch rec)."""
    pitches_all, pas_all = record['pitches'], record['pas']
    last = len(pitches_all)-1
    cutoff = min(cutoff, last)
    # A finished game is shown as final at its last pitch only if nothing state-changing follows that pitch; in a demo
    # a trailing game-ending event (CS/pickoff 3rd out, walk-off on a wild pitch...) has no released time, so it is held.
    over = (cutoff == last and record['final'] is not None and live_current is None
            and (source['kind'] != 'demo' or not pitches_all[last].get('gap_after', False)))
    nxt = pitches_all[cutoff+1] if cutoff+1 <= last else None
    lastp = pitches_all[cutoff] if cutoff >= 0 else None
    # A runner/score/out change or a pitcher/batter substitution between the last released pitch and the next one is
    # not released yet (its own time is not mapped): hold the after-last-pitch state and withhold the next pitch's view.
    held = not over and live_current is None and lastp is not None and lastp.get('gap_after', False)
    visible = pitches_all[:cutoff+1]
    boundary_ab = lastp['at_bat_number'] if held else nxt['at_bat_number'] if nxt else float('inf')
    whole = over or live_current is not None or (nxt is None and not held)
    pas = []
    for pa in pas_all:
        if not whole and pa['at_bat_number'] > boundary_ab:
            continue
        done = pa['complete'] and (whole or pa['at_bat_number'] < boundary_ab or
                                   (pa.get('ends_on_pitch') and pa['last_index'] is not None and pa['last_index'] <= cutoff))
        item = {k: v for k, v in pa.items() if k != 'ends_on_pitch'}
        item['pitch_count'] = sum(1 for p in visible if p['pa_key'] == pa['pa_key'])
        item['last_index'] = max((p['index'] for p in visible if p['pa_key'] == pa['pa_key']), default=None)
        if not done:
            item |= {'complete': False, 'event': None, 'event_label': None, 'result_text': None, 'play_text': None}
            # who is batting/pitching NOW in an unfinished PA: the last released pitch's people (the feed's matchup is
            # the end-of-PA one and would announce a later reliever / pinch hitter); a PA with no released pitch yet
            # takes the next pitch's pre-pitch people, or none while a between-pitch event is held.
            seen = [p for p in visible if p['pa_key'] == pa['pa_key']]
            people = seen[-1] if seen else (nxt if nxt is not None and nxt['pa_key'] == pa['pa_key'] and not held else None)
            item |= {'batter': people['batter'] if people else None, 'pitcher': people['pitcher'] if people else None}
        pas.append(item)
    # linescore_before: the server's as-of linescore BEFORE this visible pitch (same pre-pitch moment as situation_before), so a
    # selected past pitch never shows later runs/hits/errors. Built only from events before the pitch; the rest of INTERNAL stays hidden.
    visible = [{k: v for k, v in p.items() if k not in INTERNAL} | {'linescore_before': p.get('line_before')} for p in visible]
    for pitch in visible:  # terminal fields of a PA still in progress at the cutoff stay hidden
        if not next((pa for pa in pas if pa['pa_key'] == pitch['pa_key']), {}).get('complete'):
            pitch['actual'] = pitch['actual'] | {'event': None, 'event_label': None, 'play_text': None,
                                                 'result_label': pitch['actual']['call_label']}
    card = dict(record['card'])
    if live_current is not None:
        current, situation = live_current, live_current.get('situation')
    elif over:  # after the final event, not before the last pitch
        situation = dict(lastp.get('situation_after') or lastp['situation_before']) | {'away_score': record['final']['away'], 'home_score': record['final']['home']}
        current = {'phase': 'final', 'situation': situation, 'batter': lastp['batter'], 'pitcher': lastp['pitcher'], 'pa_key': lastp['pa_key']}
    elif held:
        situation = dict(lastp['situation_after'])
        pa_done = next((pa for pa in pas_all if pa['pa_key'] == lastp['pa_key']), {}).get('last_index') == lastp['index']
        current = {'phase': 'between_innings' if situation['outs'] >= 3 else 'resolving', 'situation': situation,
                   'batter': None if pa_done else lastp['batter'], 'pitcher': lastp['pitcher'],
                   'pa_key': None if pa_done else lastp['pa_key']}
    elif nxt is not None:
        situation = nxt['situation_before']
        current = {'phase': 'pre_pitch', 'situation': situation, 'batter': nxt['batter'], 'pitcher': nxt['pitcher'], 'pa_key': nxt['pa_key']}
    else:
        situation, current = None, {'phase': 'resolving', 'situation': None, 'batter': None, 'pitcher': None, 'pa_key': None}
    if source['kind'] == 'demo':
        card['status'], card['live'] = ('final', False) if over else ('live', True)
        card['status_text'] = 'Final' if over else 'In Progress'
    if situation:
        card |= {'away_score': situation.get('away_score'), 'home_score': situation.get('home_score')}
        if not over:
            ahead = nxt and live_current is None and not held
            card |= {'inning': nxt['inning'] if ahead else situation.get('inning'),
                     'half': nxt['half'] if ahead else situation.get('half')}
    if over:
        card |= {'away_score': record['final']['away'], 'home_score': record['final']['home'], 'inning': None, 'half': None}
    if next_rec is None and nxt is not None and not over and live_current is None and not held:
        next_rec = {'pitch_key': None, 'rec': nxt['rec']}
    if live_current is not None or over:
        linescore = record.get('line_now')
    elif held:
        linescore = lastp.get('line_after')
    elif nxt is not None:
        linescore = nxt.get('line_before')
    else:
        linescore = None
    people = [(role, holder.get(role)) for holder in [current, *visible, *pas] for role in ('pitcher', 'batter')
              if holder.get(role)]
    pregame = _pregame_view(record, people)
    if after is not None:
        visible = [p for p in visible if p['index'] > after]
    return {'schema': 'pitcheezy-service-game-v2', 'source': source, 'game': card,
            'feed': feed or {'as_of': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'ok': True, 'stale_s': None},
            'cutoff': {'index': cutoff, 'pitch_key': pitches_all[cutoff]['key'] if cutoff >= 0 else None},
            'zone_bounds': record['zone_bounds'], 'pitches': visible, 'pas': pas, 'current': current,
            'next': None if over else next_rec, 'linescore': linescore, 'pregame': pregame}


def list_card(record, cutoff, source):
    return cut_view(record, cutoff, source)['game']


def register(app, *, archive, clock, schedule=None, live_feed=None, live_rec=None):
    """Adds /api/service/** and the embargo guard to ``app``. ``schedule(date)`` and ``live_feed(pk)``
    reach the real live feed (``None`` = archive only); ``live_rec(feed)`` returns the next-pitch rec."""

    def ok(payload, status=200):
        return JSONResponse(payload, status_code=status, headers=NO_STORE)

    def fail(status, detail):
        return JSONResponse({'detail': detail}, status_code=status, headers=NO_STORE)

    @app.middleware('http')
    async def demo_embargo(request: Request, call_next):
        match = EMBARGO_PATH.match(request.url.path)
        if match and int(match[1]) in clock.active_games():
            return fail(423, '이 경기는 지금 다른 화면에서 진행 중입니다.')
        return await call_next(request)

    def demo_session(sid):
        session = clock.load(sid)
        record = archive.record(session['game_pk'])
        return session, record, clock.cutoff(session)

    def guarded(operation):
        try:
            return operation()
        except DemoError as error:
            payload = {'detail': error.detail}
            if error.payload is not None:
                payload['session'] = error.payload.get('session_id') and clock.operator_view(error.payload, keys_of(archive.record(error.payload['game_pk'])))
            return JSONResponse(payload, status_code=error.status, headers=NO_STORE)
        except KeyError:
            return fail(404, '이 경기의 기록이 없습니다.')
        except ValueError:
            return fail(503, '기록 형식을 확인할 수 없습니다.')

    schedule_cache = {}

    def live_cards(day):
        """Real schedule entries whose ORIGINAL start falls on KST date ``day`` (two US dates queried)."""
        if schedule is None:
            return [], True
        cached = schedule_cache.get(day)
        if cached and time.time()-cached[0] < 20:
            return cached[1], True
        cards, healthy = [], True
        d = Date.fromisoformat(day)
        for us_day in ((d-timedelta(days=1)).isoformat(), day):
            try:
                rows = schedule(us_day)
            except Exception:
                healthy = False
                continue
            for row in rows:
                start_kst, date_kst = game_history.kst_fields(row.get('start'))
                if date_kst != day:
                    continue
                code = row.get('double_header') or 'N'
                cards.append({'game_pk': row['game_pk'], 'start': row.get('start'), 'start_kst': start_kst, 'date_kst': date_kst,
                              'official_date': row.get('official_date'),
                              'status': game_history.status_kind(row.get('abstract'), row.get('status')),
                              'status_text': row.get('status'), 'live': bool(row.get('live')),
                              'away_team': row.get('away_team'), 'home_team': row.get('home_team'),
                              'away_abbr': None, 'home_abbr': None,
                              'away_score': row.get('away_score'), 'home_score': row.get('home_score'),
                              'inning': row.get('inning'), 'half': row.get('half'),
                              'game_number': row.get('game_number') or 1, 'double_header': code in ('Y', 'S'),
                              'double_header_code': code, 'embargoed': False, 'has_history': True})
        if healthy:
            schedule_cache[day] = (time.time(), cards)
        return cards, healthy

    def redact(card):
        return card | {'away_score': None, 'home_score': None, 'inning': None, 'half': None, 'status': 'other',
                       'status_text': None, 'live': False, 'embargoed': True}

    @app.get('/api/service/dates')
    def service_dates(demo: str | None = Query(None)):
        def run():
            if demo is not None:
                session, record, _ = demo_session(demo)
                return ok({'schema': 'pitcheezy-service-dates-v2', 'source': {'kind': 'demo', 'session_id': demo},
                           'dates': [record['card']['date_kst']]})
            dates = sorted({r['card']['date_kst'] for r in archive.cards() if r['card']['date_kst']}, reverse=True)
            return ok({'schema': 'pitcheezy-service-dates-v2', 'source': {'kind': 'archive', 'session_id': None}, 'dates': dates})
        return guarded(run)

    @app.get('/api/service/games')
    def service_games(date: str = Query(...), demo: str | None = Query(None)):
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', date) is None:
            return fail(400, 'date는 YYYY-MM-DD 형식이어야 합니다.')

        def run():
            if demo is not None:
                session, record, cutoff = demo_session(demo)
                source = {'kind': 'demo', 'session_id': demo}
                games = [list_card(record, cutoff, source)] if record['card']['date_kst'] == date else []
                return ok({'schema': 'pitcheezy-service-games-v2', 'source': source, 'date': date, 'games': games, 'live_ok': True})
            embargoed = clock.active_games()
            games = {}
            for record in archive.cards():
                if record['card']['date_kst'] != date:
                    continue
                card = list_card(record, len(record['pitches'])-1, {'kind': 'archive', 'session_id': None})
                games[card['game_pk']] = card
            live, healthy = live_cards(date)
            for card in live:  # a stored final record wins over the schedule row of the same game
                games.setdefault(card['game_pk'], card)
            out = [redact(c) if c['game_pk'] in embargoed else c for c in games.values()]
            out.sort(key=lambda c: (c['start'] or '', c['game_number'], c['game_pk']))
            return ok({'schema': 'pitcheezy-service-games-v2', 'source': {'kind': 'archive', 'session_id': None},
                       'date': date, 'games': out, 'live_ok': healthy})
        return guarded(run)

    @app.get('/api/service/games/{game_pk}')
    def service_game(game_pk: int, demo: str | None = Query(None), after: int | None = Query(None, ge=-1)):
        def run():
            if demo is not None:
                session, record, cutoff = demo_session(demo)
                if session['game_pk'] != game_pk:
                    return fail(404, '이 시연에 없는 경기입니다.')
                return ok(cut_view(record, cutoff, {'kind': 'demo', 'session_id': demo}, after=after))
            if game_pk in clock.active_games():
                return fail(423, '이 경기는 지금 다른 화면에서 진행 중입니다.')
            try:
                record = archive.record(game_pk)
                if record['final'] is not None:
                    return ok(cut_view(record, len(record['pitches'])-1, {'kind': 'archive', 'session_id': None}, after=after))
            except KeyError:
                pass
            if live_feed is None:
                raise KeyError(game_pk)
            return ok(live_view(game_pk, after))
        return guarded(run)

    def live_view(game_pk, after):
        from .live_feed import LiveFeedError, parse_state
        try:
            feed, meta = live_feed(game_pk)
        except LiveFeedError:
            return fail(503, 'MLB 경기 정보를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.')
        record = game_history.build(feed)
        parsed = parse_state(feed, None)
        phase = {'ready': 'pre_pitch', 'resolving': 'resolving', 'between_innings': 'between_innings'}.get(parsed['status'])
        abstract = (feed['gameData'].get('status') or {}).get('abstractGameState')
        if phase is None:
            phase = 'final' if abstract == 'Final' else 'pregame'
        current = {'phase': phase, 'situation': parsed.get('situation'), 'batter': parsed.get('batter'),
                   'pitcher': parsed.get('pitcher'), 'pa_key': None}
        for role in ('batter', 'pitcher'):
            if isinstance(current[role], dict):
                current[role] = current[role] | {'jersey': game_history.jersey(feed, current[role].get('id'))}
        next_rec = None
        if phase == 'pre_pitch':
            rec = live_rec(feed, parsed) if live_rec else None
            next_rec = {'pitch_key': None, 'rec': rec or game_history.absent_rec('이 실행에서는 실시간 추천을 제공하지 않습니다.')}
        view = cut_view(record, len(record['pitches'])-1, {'kind': 'live', 'session_id': None},
                        feed=meta, live_current=None if abstract == 'Final' else current, next_rec=next_rec, after=after)
        if abstract == 'Final':
            view['game'] |= {'status': 'final', 'live': False}
        return view

    @app.get('/api/service/demo/{sid}')
    def demo_public(sid: str):
        def run():
            session, record, _ = demo_session(sid)
            return ok(clock.public(session, keys_of(record)) | {'date_kst': record['card']['date_kst']})
        return guarded(run)

    # operator ---------------------------------------------------------------------------------
    def operator(token, operation):
        def run():
            clock.check_token(token)
            return operation()
        return guarded(run)

    @app.get('/api/service/operator/catalog')
    def operator_catalog(x_pitcheezy_operator: str | None = Header(None)):
        def run():
            games = [{'game_pk': r['card']['game_pk'], 'date_kst': r['card']['date_kst'], 'away_team': r['card']['away_team'],
                      'home_team': r['card']['home_team'], 'pitches': len(r['pitches'])} for r in archive.cards()]
            mappings = [{'mapping_id': m['mapping_id'], 'game_pk': m['game_pk'], 'kind': m['kind'], 'verified': bool(m.get('verified')),
                         'video_sha256': (m.get('video') or {}).get('sha256'), 'source_sha256': m['source_sha256'],
                         'rows': len(m['rows']), 'first_index': m['rows'][0]['index'], 'last_index': m['rows'][-1]['index']}
                        for m in clock.mappings()]
            return ok({'games': games, 'mappings': mappings})
        return operator(x_pitcheezy_operator, run)

    def op_view(session):
        return clock.operator_view(session, keys_of(archive.record(session['game_pk'])))

    @app.post('/api/service/operator/sessions')
    def operator_create(body: CreateDemo, x_pitcheezy_operator: str | None = Header(None)):
        def run():
            record = archive.record(body.game_pk)
            if record['final'] is None:
                raise DemoError(400, '종료된 경기만 시연할 수 있습니다.')
            session = clock.create(body.game_pk, clock.mapping(body.mapping_id), keys_of(record),
                                   start_index=body.start_index, op_id=body.op_id)
            return ok(op_view(session), 201)
        return operator(x_pitcheezy_operator, run)

    @app.get('/api/service/operator/sessions')
    def operator_sessions(x_pitcheezy_operator: str | None = Header(None)):
        return operator(x_pitcheezy_operator, lambda: ok({'sessions': [op_view(s) for s in clock.sessions()]}))

    @app.get('/api/service/operator/sessions/{sid}')
    def operator_session(sid: str, x_pitcheezy_operator: str | None = Header(None)):
        return operator(x_pitcheezy_operator, lambda: ok(op_view(clock.load(sid))))

    @app.post('/api/service/operator/sessions/{sid}/control')
    def operator_control(sid: str, body: ControlDemo, x_pitcheezy_operator: str | None = Header(None)):
        return operator(x_pitcheezy_operator, lambda: ok(op_view(clock.control(
            sid, body.action, expected_version=body.expected_version, op_id=body.op_id, index=body.index, rate=body.rate))))

