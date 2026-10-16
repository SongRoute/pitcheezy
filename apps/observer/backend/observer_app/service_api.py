"""Service game API v2 (/api/service/**) and the demo operator API; see CONTRACT pitcheezy-service-v2.

Namespaces: ``archive`` (stored final feeds + watch datasets, real live feed) and ``demo`` (``?demo=<id>``,
one game cut at the server-owned demo clock). While a demo session is active its game is embargoed on
every archive/raw path (423, or redacted in lists) so a demo client cannot read ahead by accident.
"""
from __future__ import annotations

from datetime import date as Date, timedelta
import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path

from fastapi import Header, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import display_state, features as feature_flags, game_history, player_metadata, rec_pairing
from .demo_clock import DemoError

DEFAULT_FEED_DIR = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/feeds')
EMBARGO_PATH = re.compile(r'^/api/(?:watch|replay|live)/(\d+)(?:/|$)')
NO_STORE = {'Cache-Control': 'no-store'}
INTERNAL = ('situation_after', 'gap_after', 'line_before', 'line_after', 'context_after')  # server-only: gap_after would reveal that something happens later


class Refused(Exception):
    """A service body that is not served right now (423 embargo, 503 feed): the route answers ``status`` with
    ``detail``; the publisher (``publisher.py``) writes no file for it."""

    def __init__(self, status, detail):
        super().__init__(detail)
        self.status, self.detail = status, detail


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
    for pitch in visible:
        if pitch.get('context_before') is not None:
            pitch['context_before'] = {k: v for k, v in pitch['context_before'].items() if k not in ('lineups', 'boxscore')}
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
    if live_current is not None or over:
        current_context = record.get('context_now')
        context_people = record.get('context_people_now') or {}
    elif held:
        current_context = lastp.get('context_after')
        context_people = lastp
    elif nxt is not None:
        current_context = nxt.get('context_before')
        context_people = nxt
    else:
        current_context, context_people = None, {}
    if current_context is not None:
        current_context = dict(current_context)
        for role in ('batter', 'pitcher'):
            if (current.get(role) or {}).get('id') != (context_people.get(role) or {}).get('id'):
                current_context[role] = dict.fromkeys(current_context[role])
                if role == 'batter':
                    current_context['next_batters'] = []
    current = current | {'context': current_context}
    people = [(role, holder.get(role)) for holder in [current, *visible, *pas] for role in ('pitcher', 'batter')
              if holder.get(role)]
    for team in (current_context or {}).get('boxscore', {}).values():
        people.extend((role, person) for role in ('batter', 'pitcher') for person in team[role+'s'])
    pregame = _pregame_view(record, people)
    if after is not None:
        visible = [p for p in visible if p['index'] > after]
    return {'schema': 'pitcheezy-service-game-v2', 'source': source, 'game': card,
            'feed': feed or {'as_of': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'ok': True, 'stale_s': None},
            'cutoff': {'index': cutoff, 'pitch_key': pitches_all[cutoff]['key'] if cutoff >= 0 else None},
            'zone_bounds': record['zone_bounds'], 'pitches': visible, 'pas': pas, 'current': current,
            'next': None if over else next_rec, 'linescore': linescore, 'pregame': pregame}


UNCONFIRMED_LABEL = '확인 중'


def _apply_receipts(pitches, receipts, pas=()):
    """Attach each pitch's SVC-WS-03 receipt. A body that is not admitted is never shown as an observed pitch (D1):
    its type, speed, location and result are blanked and the row says 확인 중. A pitch this store has no admission
    for at all fails closed the same way. A held pitch (``display_state.HELD``) was drawn from its last accepted body
    (``held_feed``) and stays as it is; if that body is not attached it is blanked too. Returns the summary the
    client compares between polls (the public contract is unchanged: a held pitch is told by its ``reason``)."""
    unconfirmed = corrected = 0
    open_pas = set()
    for pitch in pitches:
        item = receipts.get(display_state.RECEIPT_ALL) or receipts.get((pitch['at_bat_number'], pitch['pitch_number']))
        receipt = display_state.public_receipt(item) if item else {'status': 'unconfirmed', 'reason': 'incomplete_identity', 'corrected': False}
        if receipt['status'] == 'received' and receipt['reason'] == display_state.HELD and not ((item or {}).get('held') or {}).get('body'):
            receipt = {'status': 'unconfirmed', 'reason': 'held_body_missing', 'corrected': False}  # never the refused body
        pitch['receipt'] = receipt
        corrected += receipt['corrected']
        if receipt['status'] != 'received':
            unconfirmed += 1
            open_pas.add(pitch['pa_key'])
            pitch['actual'] = dict.fromkeys(pitch['actual']) | {'result_label': UNCONFIRMED_LABEL, 'call_label': UNCONFIRMED_LABEL}
    for pa in pas:  # a plate appearance holding an unadmitted pitch does not announce a result either
        if pa['pa_key'] in open_pas and pa.get('complete'):
            pa |= {'event': None, 'event_label': None, 'result_text': UNCONFIRMED_LABEL, 'play_text': None}
    return {'version': display_state.receipts_version(receipts), 'unconfirmed': unconfirmed, 'corrected': corrected}


def _finished_half(feed):
    """``{'inning', 'half'}`` of the half-inning that just ended, from the feed's linescore, or None. A fact about
    what is over: which half comes next is not stated (after the last out of a game the feed stays Live a while)."""
    line = (feed.get('liveData') or {}).get('linescore') or {}
    inning, state = line.get('currentInning'), line.get('inningState')
    half = {'Middle': 'Top', 'End': 'Bot', 'Top': 'Top', 'Bottom': 'Bot'}.get(state)
    return {'inning': inning, 'half': half} if type(inning) is int and half else None


def _line_score(feed):
    teams = ((feed.get('liveData') or {}).get('linescore') or {}).get('teams') or {}
    runs = {side: (teams.get(side) or {}).get('runs') for side in ('away', 'home')}
    return {f'{side}_score': value for side, value in runs.items() if type(value) is int}


def _pair_recommendations(view, recs):
    """Each past pitch of a live game gets the recommendation recorded before it, or a plain absent one. ``recs``
    is ``live_pairing``'s ``{pitch key: rec}`` or None. A pairing is settled when its pitch first appears, except:
    the pitch's own receipt changes (already in ``receipts.version``), a passing condition withholds pairings
    (``rec_pairing.TRANSIENT``: the ledger could not be read just now), or a provider correction changes whether
    a decision's inputs match the pitch (``rec_pairing.REVISABLE``). The last two are folded into
    ``receipts.version``: it is the value an open screen already compares to take the full history again."""
    recs = recs or {}
    for pitch in view['pitches']:
        pitch['rec'] = recs.get(pitch['key']) or rec_pairing.absent('no_store')
    if view['receipts'] is not None:
        view['receipts']['paired'] = sum(p['rec']['provenance'] != 'absent' for p in view['pitches'])
        watched = (*rec_pairing.TRANSIENT, *rec_pairing.REVISABLE)
        marks = sorted((p['key'], p['rec']['pairing']['rule']) for p in view['pitches'] if p['rec']['pairing']['rule'] in watched)
        if marks:
            view['receipts']['version'] = hashlib.sha256(repr((view['receipts']['version'], marks)).encode()).hexdigest()[:12]


def list_card(record, cutoff, source):
    return cut_view(record, cutoff, source)['game']


def register(app, *, archive, clock, schedule=None, live_feed=None, live_rec=None, live_display=None, live_pairing=None,
             features=None, live_extras=None, delayed_card=None, wall=None, service_delay_s=0, live_win_expectancy=None):
    """Adds /api/service/** and the embargo guard to ``app``. ``schedule(date)`` and ``live_feed(pk)``
    reach the real live feed (``None`` = archive only); ``live_feed`` returns ``(feed, meta)`` or
    ``(feed, meta, entry)`` with the store-backed buffer entry; ``live_rec(feed, parsed, entry)`` returns the
    next-pitch rec (and records it in the append-only ledger before it is returned, when a store is on).
    ``live_display(feed, parsed, entry)`` replaces ``live_rec`` on the real fan screen: it returns
    ``(rec or None, display_state, pitch receipts or None)`` so the payload states what is shown and why.
    ``live_pairing(feed, pitches, receipts, entry)`` returns ``{pitch key: rec}``: the recommendation recorded
    before each PAST pitch of a live game (``rec_pairing``); without it every past live pitch says none is recorded.

    Viewer delay (``features()['viewer_delay']``, default ``features.features``; PROVISIONAL settings): with
    ``?delay=<seconds>`` the live game is served as it stood for a viewer whose broadcast runs that far behind:
    ``live_feed(pk, delay)`` returns that viewer's snapshot and everything in the payload derives from it. Its
    ``meta['viewer']`` (delay used, buffer limit, buffering) is echoed as ``viewer``; a ``None`` snapshot is the
    calm buffering answer. Without the parameter (or with the feature off) ``live_feed(pk)`` is called as before.
    ``service_delay_s`` is the one fixed delay every viewer gets (the publisher's; ``main.create_app``): the game
    list is then held back the same way whether or not the viewer feature is on, as ``live_feed`` already holds the
    game. Only lead information (a live score, inning, status words; a too-recent Final) is withheld from a card."""
    features = features or feature_flags.features

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
        except Refused as refused:
            return fail(refused.status, refused.detail)
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
    now = wall or (lambda: time.time())  # the schedule cache's clock; a replay passes its simulated one

    def live_cards(day):
        """Real schedule entries whose ORIGINAL start falls on KST date ``day`` (two US dates queried)."""
        if schedule is None:
            return [], True
        cached = schedule_cache.get(day)
        if cached and now()-cached[0] < 20:
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
                              **{key: row.get(key) for key in ('game_type', 'series_description', 'series_game_number',
                                                              'games_in_series', 'probable_pitchers', 'venue')},
                              'away_abbr': None, 'home_abbr': None,
                              'away_score': row.get('away_score'), 'home_score': row.get('home_score'),
                              'inning': row.get('inning'), 'half': row.get('half'),
                              'game_number': row.get('game_number') or 1, 'double_header': code in ('Y', 'S'),
                              'double_header_code': code, 'embargoed': False, 'has_history': True})
        if healthy:
            schedule_cache[day] = (now(), cards)
        return cards, healthy

    def unspoiled(card):
        """A live game's list card for a delayed viewer: the schedule row is the ballpark's present, so its score
        and inning would run ahead of that viewer's broadcast. Final and scheduled games are unchanged."""
        if card.get('status') != 'live' and not card.get('live'):
            return card
        return card | {'away_score': None, 'home_score': None, 'inning': None, 'half': None, 'status_text': None, 'score_hidden': True}

    def flags():
        return feature_flags.public(features())

    @app.get('/api/service/features')
    def service_features():
        """The owner switches the screen obeys (booleans and default seconds only), asked once before the first load."""
        return ok({'schema': 'pitcheezy-service-features-v1', 'features': flags()})

    def redact(card):
        return card | {'away_score': None, 'home_score': None, 'inning': None, 'half': None, 'status': 'other',
                       'status_text': None, 'live': False, 'embargoed': True}

    # The three bodies a viewer without a demo session is served. The routes below and the public publisher
    # (``publisher.py``) both call these, so a published file is the body a request would have been answered with.
    def dates_body():
        dates = sorted({r['card']['date_kst'] for r in archive.cards() if r['card']['date_kst']}, reverse=True)
        return {'schema': 'pitcheezy-service-dates-v2', 'source': {'kind': 'archive', 'session_id': None}, 'dates': dates}

    def games_body(date, delay=None):
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
        seconds = service_delay_s or feature_flags.delay_param(delay, features())  # as ``main.service_live_feed``
        if seconds:
            out = [delayed_card(c, seconds) if delayed_card else unspoiled(c) for c in out]
        out.sort(key=lambda c: (c['start'] or '', c['game_number'], c['game_pk']))
        return {'schema': 'pitcheezy-service-games-v2', 'source': {'kind': 'archive', 'session_id': None},
                'date': date, 'games': out, 'live_ok': healthy, 'features': flags()}

    def game_body(game_pk, after=None, delay=None):
        """Raises ``Refused`` (423 embargo, 503 feed) or ``KeyError`` (no record) instead of returning a body."""
        if game_pk in clock.active_games():
            raise Refused(423, '이 경기는 지금 다른 화면에서 진행 중입니다.')
        try:
            record = archive.record(game_pk)
            if record['final'] is not None:
                return cut_view(record, len(record['pitches'])-1, {'kind': 'archive', 'session_id': None}, after=after) | {'features': flags()}
        except KeyError:
            pass
        if live_feed is None:
            raise KeyError(game_pk)
        return live_view(game_pk, after, delay) | {'features': flags()}

    @app.get('/api/service/dates')
    def service_dates(demo: str | None = Query(None)):
        def run():
            if demo is not None:
                session, record, _ = demo_session(demo)
                return ok({'schema': 'pitcheezy-service-dates-v2', 'source': {'kind': 'demo', 'session_id': demo, 'session_status': session['status']},
                           'dates': [record['card']['date_kst']]})
            return ok(dates_body())
        return guarded(run)

    @app.get('/api/service/games')
    def service_games(date: str = Query(...), demo: str | None = Query(None), delay: str | None = Query(None)):
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', date) is None:
            return fail(400, 'date는 YYYY-MM-DD 형식이어야 합니다.')

        def run():
            if demo is not None:
                session, record, cutoff = demo_session(demo)
                source = {'kind': 'demo', 'session_id': demo, 'session_status': session['status']}
                games = [list_card(record, cutoff, source)] if record['card']['date_kst'] == date else []
                return ok({'schema': 'pitcheezy-service-games-v2', 'source': source, 'date': date, 'games': games, 'live_ok': True,
                           'features': flags()})
            return ok(games_body(date, delay))
        return guarded(run)

    @app.get('/api/service/games/{game_pk}')
    def service_game(game_pk: int, demo: str | None = Query(None), after: int | None = Query(None, ge=-1),
                     delay: str | None = Query(None)):
        def run():
            view = answer()
            if isinstance(view, JSONResponse):
                return view
            return ok(view | {'features': flags()})

        def answer():
            if demo is not None:
                session, record, cutoff = demo_session(demo)
                if session['game_pk'] != game_pk:
                    return fail(404, '이 시연에 없는 경기입니다.')
                return cut_view(record, cutoff, {'kind': 'demo', 'session_id': demo, 'session_status': session['status']}, after=after)
            return game_body(game_pk, after, delay)
        return guarded(run)

    def buffering_view(static, meta, viewer):
        """A delayed viewer whose moment the buffer does not reach back to yet: who plays and when, and one calm
        state. No score, inning, pitch or recommendation, and never a newer state in their place."""
        card = game_history.game_card(static)
        if card['status'] != 'scheduled':  # whatever the ballpark's present is (even a game just over), not said here
            card |= {'status': 'live', 'live': True, 'status_text': None}
        card |= {'score_hidden': True}
        return {'schema': 'pitcheezy-service-game-v2', 'source': {'kind': 'live', 'session_id': None}, 'game': card,
                'feed': meta, 'cutoff': {'index': -1, 'pitch_key': None}, 'zone_bounds': dict(game_history.DEFAULT_ZONE_BOUNDS),
                'pitches': [], 'pas': [], 'next': None, 'linescore': None, 'pregame': None, 'receipts': None,
                'current': {'phase': None, 'situation': None, 'batter': None, 'pitcher': None, 'pa_key': None, 'context': None},
                'display_state': meta.pop('display_state'), 'viewer': viewer}

    def live_view(game_pk, after, delay=None):
        from .live_feed import LiveFeedError, parse_state
        switch = features()['viewer_delay']
        delay = feature_flags.delay_param(delay, features())
        try:
            feed, meta, *rest = live_feed(game_pk, delay) if delay else live_feed(game_pk)
        except LiveFeedError:
            raise Refused(503, 'MLB 경기 정보를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.') from None
        meta = dict(meta)
        viewer = {'delay_s': 0.0, 'max_delay_s': None, 'buffering': False, 'wait_s': 0.0} | (meta.pop('viewer', None) or {})
        viewer = {'enabled': switch['enabled'], 'default_s': switch['default_s'], 'provisional': True} | viewer
        static = meta.pop('static', None)
        if feed is None:
            return buffering_view(static, meta, viewer)
        record = game_history.build(feed)
        record['pregame'] = player_metadata.load_pregame(archive.snapshot_dir, game_pk, record['official_date'])
        parsed = parse_state(feed, None)
        phase = {'ready': 'pre_pitch', 'resolving': 'resolving', 'between_innings': 'between_innings'}.get(parsed['status'])
        abstract = (feed['gameData'].get('status') or {}).get('abstractGameState')
        if phase is None:
            phase = 'final' if abstract == 'Final' else 'pregame'
        # The plate appearance the screen is in: the pending pitch's while one is awaited or resolving (a new batter's
        # PA has no pitch yet, so the previous batter's pitches are not shown under his name); none between innings.
        pending = display_state.pending_key(feed) if phase in ('pre_pitch', 'resolving') else None
        current = {'phase': phase, 'situation': parsed.get('situation'), 'batter': parsed.get('batter'),
                   'pitcher': parsed.get('pitcher'), 'pa_key': pending.rsplit(':', 1)[0] if pending else None}
        for role in ('batter', 'pitcher'):
            if isinstance(current[role], dict):
                current[role] = current[role] | {'jersey': game_history.jersey(feed, current[role].get('id'))}
        if phase == 'between_innings':  # nobody is batting: no situation, batter or pitcher; only the half that ended
            current['finished_half'] = _finished_half(feed)
        next_rec = state = receipts = None
        entry = rest[0] if rest else None
        if live_display is not None:
            rec, state, receipts = live_display(feed, parsed, entry)
            shown_feed = display_state.held_feed(feed, receipts)
            if shown_feed is not feed:  # a held pitch is drawn from its last accepted body, never the refused one
                record = game_history.build(shown_feed) | {'pregame': record['pregame']}
        else:  # no display hook (archive-only or an older caller): the state is still explicit
            rec = live_rec(feed, parsed, entry) if live_rec and phase == 'pre_pitch' else None
            facts = {'ok': meta.get('ok', True), 'delayed': False, 'lag_s': None, 'lag_measured': False,
                     'lag_threshold_s': display_state.config()['feed_lag_delayed_s'], 'thresholds_provisional': True,
                     'last_ok_at': None, 'failing_since': None, 'failing_for_s': None, 'received_at': meta.get('as_of'),
                     'provider_order': (entry or {}).get('provider_order'), 'store': 'off', 'store_ok': True, 'committed': None}
            ready = bool(rec and rec.get('status') == 'ready' and rec.get('candidates'))
            state = display_state.decide(
                gated=display_state.gate(parsed['status'], abstract, facts, None, feed), facts=facts,
                view=None if rec is None else {'status': 'ready', 'recommendation': {'candidates': rec['candidates']},
                                               'ledger': rec.get('ledger')} if ready else
                {'status': 'computing' if rec.get('status') == 'missing' else 'unsupported', 'reason': rec.get('reason')},
                recommender=live_rec is not None, pitch_key=None, held=ready)
            if not state['recommendation_shown']:
                rec = None if not ready else game_history.absent_rec(state['message'])
        if phase == 'pre_pitch':
            if state['pitch_key']:  # the pending pitch's plate appearance, so the screen can number the next pitch
                current['pa_key'] = state['pitch_key'].rsplit(':', 1)[0]
            next_rec = {'pitch_key': state['pitch_key'],
                        'rec': rec or game_history.absent_rec(state['message'] or '이 실행에서는 실시간 추천을 제공하지 않습니다.')}
        view = cut_view(record, len(record['pitches'])-1, {'kind': 'live', 'session_id': None},
                        feed=meta, live_current=None if abstract == 'Final' else current, next_rec=next_rec)
        if abstract == 'Final':
            view['game'] |= {'status': 'final', 'live': False}
        elif phase == 'between_innings':  # ``situation`` is None then, so the card takes the score from the feed's line
            view['game'] |= _line_score(feed)
        view['display_state'] = state
        view['viewer'] = viewer
        view['receipts'] = None
        if receipts is not None:
            view['receipts'] = _apply_receipts(view['pitches'], receipts, view['pas'])
            if view['receipts']['unconfirmed']:
                # An unadmitted result must not reappear as an incremented hit/K in the statistics.
                view['current']['context'] = None
                first = min(p['index'] for p in view['pitches'] if p['receipt']['status'] != 'received')
                for pitch in view['pitches']:
                    if pitch['index'] >= first:
                        pitch['context_before'] = None
        if (live_win_expectancy is not None and phase == 'pre_pitch' and state['state'] == 'recommended'
                and state['recommendation_shown'] and view['receipts'] is not None
                and not view['receipts']['unconfirmed'] and (state.get('ledger') or {}).get('status') == 'recorded'):
            view['win_expectancy'] = live_win_expectancy(feed, state['pitch_key'], entry)
        _pair_recommendations(view, live_pairing and live_pairing(feed, view['pitches'], receipts, entry))
        view = live_extras(view, feed, parsed, receipts, entry) if live_extras else view  # service_extras: optional, switched
        if after is not None:  # a pitch whose record is unconfirmed, held or corrected is re-sent whatever the cursor
            view['pitches'] = [p for p in view['pitches'] if p['index'] > after or
                               (p.get('receipt') and (p['receipt']['status'] != 'received' or p['receipt']['corrected']
                                                      or p['receipt']['reason'] == display_state.HELD))]
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

    return ServiceBodies(dates_body, games_body, game_body, clock.active_games, lambda: set(archive.game_pks()))


class ServiceBodies:
    """The non-demo body builders of one registered app (``register`` returns it; ``app.state.service_bodies``)."""

    def __init__(self, dates, games, game, embargoed=lambda: set(), archived=lambda: set()):
        self.dates, self.games, self.game = dates, games, game
        self.embargoed = embargoed  # game pks a demo session holds right now: ``game`` refuses them with 423
        self.archived = archived  # game pks with a stored final feed: ``game`` reads those locally, no upstream poll
