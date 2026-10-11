"""Local Observer HTTP application and supervised no-media worker lifecycle."""
from collections import OrderedDict
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from .dataset import DemoDataset
from .domain import ZONES
from .media_registry import MediaRegistry
from .service import ObserverService, ServiceError
from .settings import CONFIG, REPO, RUN, WEB, database_path
from .store import Store
from .video_annotations import VideoAnnotations
from .live_armb import LiveArmB
from . import display_state, live_ledger, rec_pairing
from .setup_estimates import SetupEstimates
from .watch_along import BADGE as WATCH_BADGE, WatchAlong
from . import service_api
from .demo_clock import DemoClock

LOGGER = logging.getLogger(__name__)


class CreateSession(BaseModel):
    model_config = ConfigDict(extra='forbid')
    game_id: int = Field(strict=True, gt=0)
    pa_id: int = Field(strict=True, gt=0)


class Advance(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(strict=True, ge=0)


class ManualIntent(Advance):
    zone_id: str


class ResolveInningDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(strict=True, gt=0)
    context: dict


STORAGE = Path('/Volumes/T7 Shield')
LIVE_WAIT_S = 1.0  # a request waits this long for a just-arrived state's recommendation (about 0.15 s to compute)
ERROR_PAGE = ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
              '<title>Pitcheezy</title><body style="font-family:system-ui;margin:0;padding:32px 16px;background:#f5f3eb;color:#1d2a26">'
              '<h1 style="font-size:22px">{title}</h1><p style="line-height:1.6">{body}</p><p><a href="/">처음으로</a></p></body></html>')


class UnavailableRecommender:
    ready = False


def _timed_replay(game_pk, **options):
    """(transport, outline) of a completed game's real-pace replay, from the stored final feed."""
    scripts = str(REPO/'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import demo_precompute as demo
    path = demo.DEMO_ROOT/'feeds'/f'{int(game_pk)}_final.json.gz'
    if not path.is_file():
        raise KeyError(game_pk)
    final = demo.read_feed(path)
    return (demo.TimedReplay(final, **options) if options else None), demo.replay_outline(final)


def create_app(service=None, *, start_worker=None, live_feed=None, live_policy=None, watch_dir=None,
               timed_replay=_timed_replay, feed_dir=None, demo_dir=None, intent_dir=None, demo_wall=None,
               service_delay_s=0, service_wall=None):
    """``service_delay_s``: the one delay every viewer of ``/api/service/games/{pk}`` gets (seconds behind the
    game's own event times). The routes keep 0; the public publisher passes its configured fixed delay.
    ``service_wall``: the clock of the service routes' 20 s schedule cache (None = the machine's; an offline
    replay passes its simulated clock so the game list ages with the replayed game, not with the run)."""
    injected = service is not None
    if start_worker is None:
        start_worker = not injected

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service
        app.state.startup_error = None
        app.state.worker = None
        worker_log = None
        try:
            if app.state.service is None and os.environ.get('PITCHEEZY_OBSERVER_LEGACY', '1') == '0':
                # Watch-along/live service only: the legacy replay (and its T7 database writes) stays off.
                app.state.startup_error = '예전 과거 기록 관전 화면은 이 실행에서 꺼져 있습니다. /watch 를 이용해 주세요.'
            elif app.state.service is None:
                try:
                    dataset = DemoDataset(RUN/'dataset.json')
                    store = Store(database_path())
                    try:
                        from .recommender import Recommender
                        recommender = Recommender()
                    except Exception:
                        LOGGER.exception('Observer model unavailable; historical replay remains available')
                        app.state.startup_error = '모델을 준비하지 못해 추천 없이 기록을 재생합니다.'
                        recommender = UnavailableRecommender()
                    app.state.service = ObserverService(dataset, recommender, store)
                except Exception:
                    LOGGER.exception('Observer startup failed')
                    app.state.startup_error = '데이터 또는 모델을 준비하지 못했습니다. 실행 환경을 확인해 주세요.'
            if not injected and armb_enabled():
                armb.start()  # warm the frozen ARM-B policy in the background; the server is usable meanwhile
            if start_worker and app.state.service is not None:
                worker_log = (app.state.service.store.path.parent/'worker.log').open('a')
                backend = Path(__file__).resolve().parents[1]
                env = dict(os.environ)
                env['PYTHONPATH'] = str(backend)+os.pathsep+env.get('PYTHONPATH', '')
                app.state.worker = subprocess.Popen(
                    [sys.executable, '-m', 'observer_app.worker', '--database', str(app.state.service.store.path)],
                    cwd=backend, env=env, stdout=worker_log, stderr=subprocess.STDOUT)
            yield
        finally:
            worker = app.state.worker
            if worker is not None and worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
            if worker_log is not None:
                worker_log.close()

    app = FastAPI(title='Pitcheezy Observer', lifespan=lifespan)
    app.state.service = service
    app.state.startup_error = None
    app.state.worker = None
    media_registry = MediaRegistry(RUN)
    demo_clock = DemoClock(demo_dir, **({} if demo_wall is None else {'wall': demo_wall}))
    app.state.demo_clock = demo_clock
    video_annotations = VideoAnnotations(RUN)

    def active():
        current = app.state.service
        if current is None:
            raise ServiceError(503, app.state.startup_error or '관찰 서비스를 준비 중입니다.')
        return current

    @app.middleware('http')
    async def access_log(request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith('/api/'):
            LOGGER.info('request', extra={'fields': {'method': request.method, 'path': request.url.path,
                                                     'status': response.status_code,
                                                     'ms': round((time.perf_counter()-started)*1000, 1)}})
        return response

    @app.exception_handler(ServiceError)
    async def service_error(_request, exc):
        return JSONResponse(status_code=exc.status, content={'detail': exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, _exc):
        return JSONResponse(status_code=400, content={'detail': '요청 형식을 확인해 주세요.'})

    @app.exception_handler(Exception)
    async def unexpected_error(_request, exc):
        LOGGER.exception('Observer request failed', exc_info=exc)
        return JSONResponse(status_code=503, content={'detail': '요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.'})

    @app.get('/api/health')
    def health():
        current = app.state.service
        ready = current is not None and getattr(getattr(current, 'recommender', None), 'ready', False)
        policy = armb.status()
        demo = 'ok' if watch.available() and policy['state'] != 'unavailable' else 'degraded'
        legacy = 'ok' if ready else ('degraded' if current is not None else 'unavailable')
        if os.environ.get('PITCHEEZY_OBSERVER_LEGACY', '1') == '0':
            legacy = demo  # this run serves the watch-along/live product only
        body = {'status': legacy, 'model_ready': ready,
                'dataset_ready': current is not None, 'mode': 'historical_replay', 'cv_mode': 'unavailable:no_media',
                'demo': {'status': demo, 'storage_mounted': STORAGE.is_mount(), 'watch_dir_ready': watch.available(),
                         'watch_games': len(watch.games()), 'live_policy': policy}}
        if os.environ.get('PITCHEEZY_OBSERVER_LIVE_STORE'):
            feed = live['feed']
            body['live_store'] = {'configured': True, 'started': hasattr(feed, 'store'),
                                  'error': getattr(feed, 'store_error', None), 'games_buffered': sorted(getattr(feed, 'buffers', {})),
                                  'source': getattr(feed, 'source', None),
                                  'poll_interval_s': getattr(feed, 'config', {}).get('poll_interval_s'),
                                  'replay_position': getattr(getattr(feed, 'transport', None), 'position', lambda: None)(),
                                  'recommendation_ledger': hasattr(feed, 'store') and live_ledger.enabled()}
        return body

    @app.get('/api/catalog')
    def catalog():
        return active().catalog()

    @app.get('/api/inning-decision-games')
    def list_inning_decision_games():
        return active().list_inning_decision_games()

    @app.get('/api/inning-decisions')
    def list_inning_decisions(game_id: str = Query(...)):
        if re.fullmatch(r'[0-9]+', game_id) is None:
            raise ServiceError(400, 'game_id는 양의 정수여야 합니다.')
        try:
            parsed_game_id = int(game_id)
        except ValueError:
            raise ServiceError(400, 'game_id는 양의 정수여야 합니다.') from None
        if parsed_game_id < 1:
            raise ServiceError(400, 'game_id는 양의 정수여야 합니다.')
        return active().list_inning_decisions(parsed_game_id)

    @app.post('/api/inning-decisions/{decision_id}/resolve')
    def resolve_inning_decision(decision_id: str, body: ResolveInningDecision):
        return active().resolve_inning_decision(decision_id, body.revision, body.context)

    @app.post('/api/sessions')
    def create(body: CreateSession):
        return active().create(body.game_id, body.pa_id)

    @app.get('/api/sessions/{session_id}')
    def view(session_id: str):
        return active().get(session_id)

    @app.post('/api/sessions/{session_id}/advance')
    def advance(session_id: str, body: Advance):
        return active().advance(session_id, body.revision)

    @app.post('/api/sessions/{session_id}/manual-intent')
    def manual(session_id: str, body: ManualIntent):
        return active().manual_intent(session_id, body.revision, body.zone_id)

    @app.get('/api/zones')
    def zones():
        return {'zones': ZONES, 'coordinate_frame': 'catcher_view'}

    @app.get('/api/video-lab/catalog')
    def video_catalog():
        try:
            return media_registry.catalog()
        except (KeyError, ValueError) as exc:
            raise ServiceError(503, str(exc)) from None

    @app.get('/api/video-lab/clips/{clip_id}')
    def video_clip(clip_id: str):
        try:
            return FileResponse(media_registry.path(clip_id), media_type='video/mp4')
        except KeyError:
            raise ServiceError(404, '등록된 영상을 찾을 수 없습니다.') from None
        except ValueError as exc:
            raise ServiceError(503, str(exc)) from None

    def lab_call(operation, *args):
        try:
            return operation(*args)
        except KeyError:
            raise ServiceError(404, '저장된 영상 라벨을 찾을 수 없습니다.') from None
        except (ValueError, TypeError) as exc:
            raise ServiceError(400, str(exc)) from None
        except ModuleNotFoundError:
            raise ServiceError(503, '영상 추적 패키지를 준비하지 못했습니다. 저장한 라벨은 유지됩니다.') from None

    @app.get('/api/video-lab/annotations')
    def list_video_annotations():
        return lab_call(video_annotations.list)

    @app.post('/api/video-lab/annotations')
    def save_video_annotation(body: dict):
        return lab_call(video_annotations.save, body)

    @app.get('/api/video-lab/annotations/{annotation_id}')
    def get_video_annotation(annotation_id: str):
        return lab_call(video_annotations.get, annotation_id)

    @app.post('/api/video-lab/annotations/{annotation_id}/track')
    def track_video_annotation(annotation_id: str):
        return lab_call(video_annotations.track, annotation_id)

    live = {'feed': live_feed}
    armb = live_policy if live_policy is not None else LiveArmB()
    armb_enabled = lambda: os.environ.get('PITCHEEZY_OBSERVER_LIVE_ARMB', '1') != '0'
    app.state.live_policy = armb

    live_lock = threading.Lock()  # sync routes run in a thread pool; the one live feed is built once

    def live_parts():
        from .live_feed import LIVE_CONFIG, LiveFeed, RecordedPaceReplay, ReplayTransport, http_json, replay_config
        with live_lock:  # first requests arrive together: two feeds would poll and replay twice
            if live['feed'] is None:
                replay = os.environ.get('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR')
                delay = os.environ.get('PITCHEEZY_OBSERVER_LIVE_DELAY_S')
                # Rehearsal speed: a replay hands out one recorded body per poll, so only a replay may change the pace.
                # A real upstream never reads the variable: a stray or invalid value must not break live polling.
                pace = replay_config(os.environ.get('PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S')) if replay else None
                if replay and os.environ.get('PITCHEEZY_OBSERVER_LIVE_REPLAY_PACE') == 'recorded':
                    # delayed-viewing rehearsal: bodies at their recorded receipt times on today's clock (never upstream)
                    gap = os.environ.get('PITCHEEZY_OBSERVER_LIVE_REPLAY_MAX_GAP_S')
                    player = RecordedPaceReplay(replay, max_gap_s=float(gap) if gap else None)
                elif replay:
                    player = ReplayTransport(replay)
                transport, options = ((player, {'delay_s': float(delay or 0), 'config': pace}) if replay else
                                      (http_json, {'record_dir': REPO/LIVE_CONFIG['record_dir'],
                                                   'delay_s': None if delay is None else float(delay)}))
                if (store := os.environ.get('PITCHEEZY_OBSERVER_LIVE_STORE')):  # opt-in restart-safe buffer (live_store.py)
                    import sqlite3
                    from .live_store import LiveSnapshotStore, PersistentLiveFeed
                    try:  # a replayed recording is labelled as such: its receipts are not the game's arrival times
                        live['feed'] = PersistentLiveFeed(transport, store=LiveSnapshotStore(store),
                                                          source='replay_poll' if replay else 'live_poll', **options)
                    except (OSError, sqlite3.Error) as exc:  # unopenable store: serve live from memory, health says why
                        LOGGER.warning('live store %s unavailable; serving from memory only', store, exc_info=True)
                        live['feed'] = LiveFeed(transport, **options)
                        live['feed'].store_error = f'open: {type(exc).__name__}'
                else:
                    live['feed'] = LiveFeed(transport, **options)
        return live['feed']

    def live_call(operation):
        from .live_feed import LIVE_CONFIG, LiveFeedError
        try:
            return {'demo_status': LIVE_CONFIG['demo_status'], 'badge': WATCH_BADGE} | operation()
        except LiveFeedError:
            LOGGER.warning('live feed unavailable', exc_info=True)
            raise ServiceError(503, 'MLB 경기 정보를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.') from None

    @app.get('/api/live/games')
    def live_games(date: str = Query(...)):
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', date) is None:
            raise ServiceError(400, 'date는 YYYY-MM-DD 형식이어야 합니다.')
        feed = live_parts()
        hidden = demo_clock.active_games()
        return live_call(lambda: {'date': date, 'games': [g for g in feed.schedule(date) if g['game_pk'] not in hidden]})

    def armb_view(snapshot, parsed, feed=None, entry=None, record_new=True):
        """Frozen ARM-B pre-pitch recommendation for the served (delayed) state, from the worker cache.
        With ``feed`` (the ``LiveFeed`` that served buffer ``entry``) a recommendation passes the append-only
        ledger before it is returned for display (``live_ledger.admit``); without a store that is a no-op.
        ``record_new=False`` (disconnected feed) computes and records nothing: only a recommendation already
        cached and already in the ledger comes back. ``code`` is the machine reason ``display_state`` reads."""
        if parsed['status'] != 'ready':
            return None
        if not armb_enabled():
            return {'status': 'unavailable', 'code': 'recommender_off', 'reason': '이 실행에서는 지연 중계 추천을 껐습니다.'}
        cached = armb.lookup(snapshot, first=True, wait_s=LIVE_WAIT_S) if record_new else armb.lookup(snapshot, queue=False)
        policy = armb.status()
        if cached is None:
            if not record_new:
                return None
            if policy['state'] == 'unavailable':
                return {'status': 'unavailable', 'code': 'recommender_unavailable',
                        'reason': '추천 모델을 준비하지 못했습니다(저장장치 연결을 확인해 주세요).'}
            return {'status': 'computing', 'code': 'computing', 'reason': '추천을 계산하는 중입니다(보통 10초 안쪽).'}
        if 'error' in cached:
            return {'status': 'unavailable', 'code': 'compute_failed', 'reason': '이 상황의 추천을 계산하지 못했습니다.'}
        result = cached['result']
        if result is None:
            return None
        pitch = result['pitch']
        view = {'status': pitch['pre']['status'], 'reason': pitch['pre']['reason'], 'key': result['key'],
                'recommendation': pitch['pre']['recommendation'], 'pa_pitches': result['pa_pitches'],
                'previous': result['previous'], 'home_we_now': result['home_we_now'],
                'policy_identity': result['policy_identity'][:8], 'seconds': result['seconds'],
                'tally': armb.tally(result['key']), 'policy_status': pitch.get('status')}
        return view if feed is None else live_ledger.admit(feed, entry, view, game_pk=snapshot['gameData']['game']['pk'],
                                                           snapshot=snapshot, policy_identity=result['policy_identity'],
                                                           record_new=record_new, inputs=rec_pairing.decision_inputs(parsed))

    shown_memory = OrderedDict()  # runs without a ledger: (gamePk, pitch key) -> what was displayed, oldest first
    receipt_memory = {}  # gamePk -> {pitch key: body sha} last admitted, for when the pitch ledger cannot be read
    memory_lock = threading.Lock()  # sync routes run in a thread pool

    def _displayed_before(feed, game_pk, key, entry):
        """Earlier displays of the pending pitch ``key`` (D1: a later, different one is a visible correction), or
        None if they cannot be read. From the append-only ledger when one is on (restart-safe), else from this
        process's memory. Only displays made from a snapshot no newer than the served ``entry`` count: a viewer on
        a longer delay is never told about a decision made for a later state of the same pitch."""
        if key is None:
            return []
        if hasattr(feed, 'store') and live_ledger.enabled():
            import sqlite3
            try:
                rows = feed.store.recommendations_for_pitch(game_pk, key)
            except (sqlite3.Error, OSError, ValueError):
                LOGGER.warning('recommendation ledger unreadable; earlier displays of %s unknown', key, exc_info=True)
                return None
            seq = (entry or {}).get('seq')
            return [{'summary': display_state.summary(row['displayed'].get('recommendation')), 'decided': row['decided'],
                     'candidates': (row['displayed'].get('recommendation') or {}).get('candidates'),
                     'decision_id': row['decision_id']} for row in rows
                    if seq is None or row['snapshot']['store_seq'] <= seq]
        event = (entry or {}).get('event')
        with memory_lock:
            return [item for item in shown_memory.get((game_pk, key), []) if event is None or item['event'] <= event]

    def _pitch_receipts(feed, game_pk, snapshot, entry, committed):
        """SVC-WS-03 admission of every pitch body in the served snapshot (None without a store), as of that
        snapshot: ledger rows written for a later store version are not consulted, so a viewer on a longer delay
        is judged against what was known when their snapshot was the newest."""
        if not hasattr(feed, 'store'):
            return None
        import sqlite3
        seq = (entry or {}).get('seq')
        try:
            rows = [row for row in feed.store.pitch_events(game_pk)['events'] if seq is None or row['store_seq'] <= seq]
            receipts = display_state.pitch_receipts(snapshot, rows, committed=committed)
        except (sqlite3.Error, OSError, ValueError):
            LOGGER.warning('pitch ledger unreadable; only bodies admitted earlier stay confirmed', exc_info=True)
            with memory_lock:
                return display_state.fallback_receipts(snapshot, dict(receipt_memory.get(game_pk, {})))
        if seq is not None:
            with memory_lock:  # per served version, so another viewer's delay never lends this one an admission
                receipt_memory[game_pk] = {key: item['sha'] for key, item in receipts.items() if item['status'] == 'received'}
        return receipts

    def live_display(game_pk, snapshot, parsed, entry, feed):
        """(recommendation view as it may be displayed, the single display state, pitch receipts or None) for the
        served snapshot. The gate (final/pregame/disconnected/waiting/확인 중) is decided from store facts before
        the recommender is asked for a view, so a recommendation on an unconfirmed input is never recorded in the
        ledger or returned (the legacy route's head start may still compute it into the worker cache)."""
        game_pk = int(game_pk)
        facts = display_state.feed_facts(feed, game_pk, entry, display_state.config(), now=getattr(feed, 'wall', time.time)())
        receipts = _pitch_receipts(feed, game_pk, snapshot, entry, bool(facts['committed']))
        abstract = (snapshot['gameData'].get('status') or {}).get('abstractGameState')
        gated = display_state.gate(parsed['status'], abstract, facts, receipts, snapshot)
        ready = parsed['status'] == 'ready'
        view = None
        if gated is None:
            view = armb_view(snapshot, parsed, feed, entry)
        elif gated[0] == 'disconnected' and gated[2] is None and ready:
            view = armb_view(snapshot, parsed, feed, entry, record_new=False)
        key = ((view or {}).get('key') or display_state.pending_key(snapshot)) if ready else None
        history = _displayed_before(feed, game_pk, key, entry)
        now_shown = display_state.summary((view or {}).get('recommendation'))
        ledgered = hasattr(feed, 'store') and live_ledger.enabled()
        held = ((view or {}).get('ledger') or {}).get('status') == 'recorded' if ledgered else \
            bool(history) and history[-1]['summary'] == now_shown
        state = display_state.decide(gated=gated, view=view, facts=facts, history=history, pitch_key=key, held=held)
        if state['recommendation_shown']:
            if not ledgered:
                with memory_lock:
                    seen = shown_memory.setdefault((game_pk, key), [])
                    shown_memory.move_to_end((game_pk, key))
                    if not seen or seen[-1]['summary'] != now_shown:
                        seen.append({'summary': now_shown, 'decided': getattr(feed, 'wall', time.time)(),
                                     'event': (entry or {}).get('event', 0), 'candidates': view['recommendation'].get('candidates')})
                    while len(shown_memory) > 64:
                        shown_memory.popitem(last=False)
        elif view:  # the state says nothing is shown: no candidate leaves, not even the at-bat's recomputed ones
            hidden = {'recommendation': None, 'pa_pitches': [], 'previous': None}
            # ...and the reason is the state's sentence, never a policy status code or a ledger note that would
            # contradict the state the screen shows (a disconnect before the first ledger write is "연결 끊김")
            view = view | {key: value for key, value in hidden.items() if key in view} | {'reason': state['message']} | (
                {'status': 'unavailable'} if view.get('recommendation') else {})
        return view, state, receipts

    def _confirmed_only(parsed, receipts, at_bat, snapshot):
        """``/api/live/{pk}/state``: a pitch body that is not admitted is not listed as an observed pitch (D1).
        ``at_bat`` is the open at-bat's number, whose pitches ``pitch_sequence`` lists from ``currentPlay``.
        With a store on, a listed pitch needs an admission of its own: one the pitch ledger has no row for (a
        ``currentPlay`` that is not the last ``allPlays`` entry) or holds with another body is blanked like a
        refused one, and ``display_state.gate`` withholds the recommendation (``prior_pitch_unconfirmed``).
        Without a store (``receipts`` None) nothing is checked."""
        if receipts is None:
            return parsed
        blank = lambda item, keep: {k: (v if k in keep else None) for k, v in item.items()} | {'receipt': 'unconfirmed'}
        tainted = display_state.RECEIPT_ALL in receipts
        unlisted = display_state.unlisted_current(snapshot, receipts)
        admitted = lambda ab, number: (not tainted and (ab, number) not in unlisted
                                       and (receipts.get((ab, number)) or {}).get('status') == 'received')
        out = parsed | {'pitch_sequence': [item if admitted(at_bat, item.get('pitch_number')) else blank(item, ('pitch_number',))
                                           for item in parsed.get('pitch_sequence') or []]}
        last = parsed.get('last_pitch')
        if last:
            _, last_ab, number = (int(part) for part in last['key'].split(':'))
            if not admitted(last_ab, number):
                out['last_pitch'] = blank(last, ('key', 'inning', 'half'))
        return out

    replays = OrderedDict()  # (gamePk, start, t0, speed) -> (LiveFeed, transport): one shared clock per link

    def replay_parts(game_pk, start, t0, speed, lag):
        from .live_feed import LIVE_CONFIG, LiveFeed
        now = time.time()
        t0 = now if t0 is None else t0
        if not now-12*3600 <= t0 <= now+5:
            raise ServiceError(400, '다시 보기 시작 시각이 올바르지 않습니다. 처음부터 다시 시작해 주세요.')
        lag = LIVE_CONFIG.get('replay_lag_s', 0.0) if lag is None else lag
        key = (game_pk, start, round(t0, 3), speed, lag)
        if key not in replays:
            try:
                transport, _ = timed_replay(game_pk, start=start, t0=key[2], speed=speed, lag_s=lag)
            except KeyError:
                raise ServiceError(404, '이 경기는 다시 보기 자료가 없습니다.') from None
            except IndexError:
                raise ServiceError(404, '해당 투구를 찾을 수 없습니다.') from None
            config = LIVE_CONFIG | {'poll_interval_s': max(1.0, LIVE_CONFIG['poll_interval_s']/speed)}
            replays[key] = (LiveFeed(transport, config=config, delay_s=0), transport)
            while len(replays) > 6:
                replays.popitem(last=False)
        replays.move_to_end(key)
        return replays[key]

    @app.get('/api/replay/{game_pk}/outline')
    def replay_outline(game_pk: int):
        try:
            return {'badge': WATCH_BADGE} | timed_replay(game_pk)[1]
        except KeyError:
            raise ServiceError(404, '이 경기는 다시 보기 자료가 없습니다.') from None

    @app.get('/api/live/{game_pk}/state')
    def live_state(game_pk: int, delay: float | None = Query(None, ge=0, le=600), replay: int | None = Query(None, ge=0),
                   t0: float | None = Query(None), speed: float = Query(1.0, ge=0.25, le=30),
                   lag: float | None = Query(None, ge=0, le=60)):
        from .live_feed import parse_state
        if game_pk < 1:
            raise ServiceError(400, 'gamePk는 양의 정수여야 합니다.')
        feed, transport = (live_parts(), None) if replay is None else replay_parts(game_pk, replay, t0, speed, lag)

        def state():
            entry, wait = feed.snapshot(game_pk, delay)
            newest = feed.latest(game_pk)
            if armb_enabled() and newest is not None and parse_state(newest, None)['status'] == 'ready':
                armb.lookup(newest)  # head start: compute the raw state before the delay serves it
            used = feed.delay_s if delay is None else min(delay, feed.max_delay_s)
            timing = {'delay_s': used, 'replay': None if transport is None else
                      {'start': replay, 't0': transport.t0, 'speed': speed, 'lag_s': transport.lag_s} | transport.position()}
            if entry is None:
                facts = display_state.feed_facts(feed, game_pk, None, display_state.config(), now=getattr(feed, 'wall', time.time)())
                return timing | {'status': 'buffering', 'reason': f'내 중계 시점까지 기다리는 중입니다({wait}초).',
                                 'recommendation': None, 'sync': {'max_delay_s': feed.max_delay_s},
                                 'display_state': display_state.decide(gated=('waiting', 'buffering', None), view=None, facts=facts)}
            snapshot = entry['feed']
            parsed = parse_state(snapshot, None)
            parsed.pop('pre_pitch')
            sync = {'max_delay_s': feed.max_delay_s, 'event_age_s': round(time.time()-entry['event'], 1),
                    'feed_lag_s': round(entry['fetched']-entry['event'], 1), 'feed_ok': game_pk not in feed.failing}
            if entry.get('requested') is not None:  # how long the fetch that first brought this state took
                sync['fetch_s'] = round(entry['fetched']-entry['requested'], 3)
            if hasattr(feed, 'sync_extra'):
                sync |= feed.sync_extra(game_pk, entry)
            view, shown, receipts = live_display(game_pk, snapshot, parsed, entry, feed)
            play = snapshot['liveData']['plays'].get('currentPlay') or {}
            index = (play.get('about') or {}).get('atBatIndex')
            parsed = _confirmed_only(parsed, receipts, index+1 if type(index) is int else None, snapshot)
            return parsed | timing | {'sync': sync, 'recommendation': view, 'display_state': shown}
        return live_call(state)

    @app.get('/api/live/{game_pk}/log')
    def live_log(game_pk: int, after: int = Query(0, ge=0), delay: float | None = Query(None, ge=0, le=600),
                 limit: int = Query(100, ge=1, le=500)):
        """Replay cursor over the opt-in live store: committed snapshot versions no newer than the delay."""
        feed = live_parts()
        if not hasattr(feed, 'store'):
            raise ServiceError(404, '이 실행에서는 라이브 저장소를 켜지 않았습니다.')
        used = feed.delay_s if delay is None else min(delay, feed.max_delay_s)
        return {'game_pk': game_pk, 'delay_s': used, 'store_ok': feed.store_error is None} | feed.store.read_after(
            game_pk, after, until_event=feed.wall()-used, limit=limit)

    @app.get('/api/live/{game_pk}/tap')
    def live_tap(game_pk: int, delay: float = Query(..., ge=0, le=600)):
        """Fine sync: the viewer taps as a pitch lands on their broadcast; answers their measured delay."""
        from .live_feed import tap_delay
        feed = live_parts()
        newest = feed.latest(game_pk)
        measured = None if newest is None else tap_delay(newest, time.time(), delay)
        if measured is None:
            return {'status': 'unclear', 'delay_s': delay,
                    'reason': '어느 공인지 가리지 못했어요. ± 버튼으로 대략 맞춘 뒤, 공이 미트에 들어오는 순간 다시 눌러 주세요.'}
        return {'status': 'ok', 'delay_s': min(measured, feed.max_delay_s)}

    watch = WatchAlong(watch_dir)

    def watch_call(operation):
        try:
            return operation()
        except KeyError:
            raise ServiceError(404, '이 경기의 관전 자료가 없습니다. 먼저 사전 계산을 실행해 주세요.') from None
        except IndexError:
            raise ServiceError(404, '해당 투구를 찾을 수 없습니다.') from None
        except ValueError:
            raise ServiceError(503, '관전 자료 형식을 확인할 수 없습니다.') from None

    @app.get('/api/watch/games')
    def watch_games():
        hidden = demo_clock.active_games()
        return {'games': [g for g in watch.games() if g['game_pk'] not in hidden]}

    @app.get('/api/watch/{game_pk}')
    def watch_timeline(game_pk: int):
        return watch_call(lambda: watch.timeline(game_pk))

    @app.get('/api/watch/{game_pk}/reveal/{index}')
    def watch_reveal(game_pk: int, index: int):
        return watch_call(lambda: watch.reveal(game_pk, index))

    @app.get('/api/watch/{game_pk}/summary')
    def watch_summary(game_pk: int, through: int = Query(..., ge=-1), top: int = Query(5, ge=0, le=20)):
        return watch_call(lambda: watch.summary(game_pk, through, top))

    setups = SetupEstimates(intent_dir)

    def service_live_feed(game_pk):
        feed = live_parts()
        entry, _ = feed.snapshot(game_pk, service_delay_s)
        if entry is None:
            from .live_feed import LiveFeedError
            raise LiveFeedError('no snapshot yet')
        # ``stale_s`` is the age of the last upstream answer, not of the shown state's first receipt: a game with
        # no new event for a while (inning break, mound visit, review) is still fresh while polls keep answering.
        answered = getattr(feed, 'polled_ok', {}).get(game_pk, entry['fetched'])
        now = feed.wall() if hasattr(feed, 'wall') else time.time()
        return entry['feed'], {'as_of': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(entry['fetched'])),
                               'ok': game_pk not in feed.failing, 'stale_s': round(max(0.0, now-answered), 1)}, entry

    def service_live_display(snapshot, parsed, entry=None):
        """(next-pitch rec in the service shape or None, display state, pitch receipts) for the fan screen."""
        view, state, receipts = live_display(snapshot['gameData']['game']['pk'], snapshot, parsed, entry, live_parts())
        return (service_rec(view, state) if parsed.get('status') == 'ready' else None), state, receipts

    def service_rec(view, state):
        candidates = ((view or {}).get('recommendation') or {}).get('candidates') or []
        if not view or view.get('status') != 'ready' or not candidates:
            return {'provenance': 'absent', 'status': 'missing' if state['state'] == 'waiting' else 'unsupported',
                    'reason': state['message'], 'recorded_at': None, 'policy': None, 'candidates': []}
        ledger = view.get('ledger')  # present only when the append-only ledger recorded it before this display
        return {'provenance': 'captured_live', 'status': 'ready', 'reason': None, 'ledger': ledger,
                'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ledger['decided'] if ledger else None)),
                'policy': {'name': 'ARM-B', 'identity': view.get('policy_identity')},
                'candidates': [{'rank': c.get('rank', i+1), 'pitch_type': c.get('pitch_type'), 'pitch_label': c.get('pitch_label'),
                                'zone_id': c.get('zone_id'), 'zone_label': c.get('zone_label'), 'target': c.get('target'),
                                'detail': {'probability': (c.get('detail') or {}).get('probability')}}
                               for i, c in enumerate(candidates)]}

    ledger_rows = rec_pairing.LedgerCache()

    def service_live_pairing(snapshot, pitches, receipts, entry=None):
        """``{pitch key: rec}`` for the PAST pitches of the fan screen: the recommendation the ledger recorded
        before each pitch, or an absent one naming the rule that refused it (``rec_pairing`` has the rules).
        Read-only and fail closed: without a store or a ledger, or when one cannot be read, nothing is paired."""
        feed, game_pk = live_parts(), int(snapshot['gameData']['game']['pk'])
        nothing = lambda rule: {pitch['key']: rec_pairing.absent(rule) for pitch in pitches}
        if receipts is None or not hasattr(feed, 'store') or not live_ledger.enabled():
            return nothing('no_store')
        try:
            rows = ledger_rows.rows(feed.store, game_pk)
            return rec_pairing.pair(pitches, receipts, rows, (entry or {}).get('seq'),
                                    lambda row: ledger_rows.inputs(feed.store, game_pk, row))
        except Exception:  # noqa: BLE001 - a display-only read: any failure shows no pairing, never an error page
            LOGGER.warning('recommendation ledger unreadable; past pitches shown without a recorded recommendation', exc_info=True)
            return nothing('ledger_unreadable')

    app.state.setups = setups
    app.state.live_parts = live_parts
    app.state.service_bodies = service_api.register(app, archive=service_api.GameArchive(feed_dir, watch, setups), clock=demo_clock,
                         schedule=None if live_feed is False else (lambda day: live_parts().schedule(day)),
                         live_feed=None if live_feed is False else service_live_feed,
                         live_display=service_live_display, live_pairing=service_live_pairing, wall=service_wall)

    @app.get('/api/watch/{game_pk}/setup')
    def watch_setup(game_pk: int):
        try:
            return setups.view(game_pk)
        except KeyError:
            raise ServiceError(404, '이 경기의 포수 셋업 자료가 없습니다.') from None
        except ValueError:
            raise ServiceError(503, '포수 셋업 자료를 확인할 수 없습니다.') from None

    @app.get('/api/runtime')
    def runtime():
        current, worker = app.state.service, app.state.worker
        return {'mode': 'historical_replay', 'model_version': CONFIG['model_version'],
                'scientific_runtime': getattr(current.recommender, 'runtime_kind', None) if current else None,
                'dataset_identity': current.dataset.identity if current else None,
                'worker_status': 'running' if worker is not None and worker.poll() is None else 'stopped',
                'cv_status': 'unavailable:no_media', 'startup_error': app.state.startup_error}

    @app.get('/{path:path}', include_in_schema=False)
    def frontend(path: str, request: Request):
        if path == 'api' or path.startswith('api/'):
            raise ServiceError(404, 'API 경로를 찾을 수 없습니다.')
        if path == '' and os.environ.get('PITCHEEZY_OBSERVER_LEGACY', '1') == '0':
            return RedirectResponse('/watch')
        target = (WEB/path).resolve()
        if not target.is_relative_to(WEB.resolve()):
            raise ServiceError(404, '파일을 찾을 수 없습니다.')
        if target.is_file():
            return FileResponse(target)
        if (WEB/'index.html').is_file():
            return FileResponse(WEB/'index.html')
        return HTMLResponse(ERROR_PAGE.format(title='화면을 준비하지 못했어요',
                                              body='웹 화면을 아직 빌드하지 않았습니다. <code>sh apps/observer/serve.sh</code>로 다시 시작해 주세요.'),
                            status_code=503)

    return app


app = create_app()
