from pathlib import Path
import json
import os

REPO = Path(__file__).resolve().parents[4]
APP = REPO/'apps/observer'
ARTIFACT_ROOT = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp')
RUN = Path(os.environ.get('PITCHEEZY_OBSERVER_RUN', str(ARTIFACT_ROOT/'runs/observer-mvp-v1')))
BUNDLE = ARTIFACT_ROOT/'runs/minimal-pitch-service-v1'
CONFIG = json.loads((APP/'config.json').read_text())
WEB = APP/'web/dist'


def require_storage():
    if not Path('/Volumes/T7 Shield').is_mount() or not RUN.resolve().is_relative_to(ARTIFACT_ROOT/'runs'):
        raise RuntimeError('마운트된 T7 Shield의 승인된 저장 경로가 필요합니다.')
    RUN.mkdir(parents=True, exist_ok=True)


def database_path():
    require_storage()
    return RUN/'observer.sqlite3'
