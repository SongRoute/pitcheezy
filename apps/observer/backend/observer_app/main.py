"""Local Observer HTTP application and supervised no-media worker lifecycle."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import subprocess
import sys

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .dataset import DemoDataset
from .domain import ZONES
from .media_registry import MediaRegistry
from .service import ObserverService, ServiceError
from .settings import CONFIG, RUN, WEB, database_path
from .store import Store
from .video_annotations import VideoAnnotations

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


class UnavailableRecommender:
    ready = False


def create_app(service=None, *, start_worker=None):
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
            if app.state.service is None:
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
        ready = current is not None and current.recommender.ready
        return {'status': 'ok' if ready else ('degraded' if current is not None else 'unavailable'), 'model_ready': ready,
                'dataset_ready': current is not None, 'mode': 'historical_replay', 'cv_mode': 'unavailable:no_media'}

    @app.get('/api/catalog')
    def catalog():
        return active().catalog()

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
        target = (WEB/path).resolve()
        if not target.is_relative_to(WEB.resolve()):
            raise ServiceError(404, '파일을 찾을 수 없습니다.')
        if target.is_file():
            return FileResponse(target)
        if (WEB/'index.html').is_file():
            return FileResponse(WEB/'index.html')
        raise ServiceError(503, '브라우저 화면을 아직 빌드하지 않았습니다.')

    return app


app = create_app()
