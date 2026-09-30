"""Local Observer HTTP application and supervised no-media worker lifecycle."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
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
from .watch_along import BADGE as WATCH_BADGE, WatchAlong

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
ERROR_PAGE = ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
              '<title>Pitcheezy</title><body style="font-family:system-ui;margin:0;padding:32px 16px;background:#f5f3eb;color:#1d2a26">'
              '<h1 style="font-size:22px">{title}</h1><p style="line-height:1.6">{body}</p><p><a href="/">처음으로</a></p></body></html>')


class UnavailableRecommender:
    ready = False


def create_app(service=None, *, start_worker=None, live_feed=None, live_policy=None, watch_dir=None):
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
        return {'status': legacy, 'model_ready': ready,
                'dataset_ready': current is not None, 'mode': 'historical_replay', 'cv_mode': 'unavailable:no_media',
                'demo': {'status': demo, 'storage_mounted': STORAGE.is_mount(), 'watch_dir_ready': watch.available(),
                         'watch_games': len(watch.games()), 'live_policy': policy}}

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

    def live_parts():
        from .live_feed import LIVE_CONFIG, LiveFeed, ReplayTransport, http_json
        if live['feed'] is None:
            replay = os.environ.get('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR')
            delay = os.environ.get('PITCHEEZY_OBSERVER_LIVE_DELAY_S')
            live['feed'] = (LiveFeed(ReplayTransport(replay), delay_s=float(delay or 0)) if replay else
                            LiveFeed(http_json, record_dir=REPO/LIVE_CONFIG['record_dir'],
                                     delay_s=None if delay is None else float(delay)))
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
        return live_call(lambda: {'date': date, 'games': feed.schedule(date)})

    def armb_view(snapshot, parsed):
        """Frozen ARM-B pre-pitch recommendation for the served (delayed) state, from the worker cache."""
        if parsed['status'] != 'ready':
            return None
        if not armb_enabled():
            return {'status': 'unavailable', 'reason': '이 실행에서는 지연 중계 추천을 껐습니다.'}
        cached = armb.lookup(snapshot, first=True)
        policy = armb.status()
        if cached is None:
            if policy['state'] == 'unavailable':
                return {'status': 'unavailable', 'reason': '추천 모델을 준비하지 못했습니다(저장장치 연결을 확인해 주세요).'}
            return {'status': 'computing', 'reason': '추천을 계산하는 중입니다(보통 10초 안쪽).'}
        if 'error' in cached:
            return {'status': 'unavailable', 'reason': '이 상황의 추천을 계산하지 못했습니다.'}
        result = cached['result']
        if result is None:
            return None
        pitch = result['pitch']
        return {'status': pitch['pre']['status'], 'reason': pitch['pre']['reason'], 'key': result['key'],
                'recommendation': pitch['pre']['recommendation'], 'pa_pitches': result['pa_pitches'],
                'previous': result['previous'], 'home_we_now': result['home_we_now'],
                'policy_identity': result['policy_identity'][:8], 'seconds': result['seconds']}

    @app.get('/api/live/{game_pk}/state')
    def live_state(game_pk: int):
        from .live_feed import parse_state
        if game_pk < 1:
            raise ServiceError(400, 'gamePk는 양의 정수여야 합니다.')
        feed = live_parts()

        def state():
            snapshot, wait = feed.snapshot(game_pk)
            newest = feed.latest(game_pk)
            if armb_enabled() and newest is not None and parse_state(newest, None)['status'] == 'ready':
                armb.lookup(newest)  # head start: compute the raw state before the delay serves it
            if snapshot is None:
                return {'status': 'buffering', 'reason': f'지연 중계 버퍼를 채우는 중입니다({wait}초).',
                        'delay_s': feed.delay_s, 'recommendation': None}
            parsed = parse_state(snapshot, None)
            parsed.pop('pre_pitch')
            return parsed | {'delay_s': feed.delay_s, 'recommendation': armb_view(snapshot, parsed)}
        return live_call(state)

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
        return {'games': watch.games()}

    @app.get('/api/watch/{game_pk}')
    def watch_timeline(game_pk: int):
        return watch_call(lambda: watch.timeline(game_pk))

    @app.get('/api/watch/{game_pk}/reveal/{index}')
    def watch_reveal(game_pk: int, index: int):
        return watch_call(lambda: watch.reveal(game_pk, index))

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
