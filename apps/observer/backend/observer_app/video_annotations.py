"""Immutable local video annotations and versioned, explicitly requested tracking.

Separate from pre-pitch recommendations: neither labels nor results are inputs
to the recommendation engine. Slow CV runs outside SQLite write transactions.
"""
from contextlib import closing
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import threading

from .media_registry import MediaRegistry
from .video_lab import validate_annotation, track_video


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def identity(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


class VideoAnnotations:
    def __init__(self, run):
        self.run = Path(run)
        self.registry = MediaRegistry(run)
        self.path = self.run/'video_annotations.sqlite3'
        self.lock = threading.Lock()

    def connect(self):
        self.run.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA foreign_keys=ON')
        db.executescript('''
          CREATE TABLE IF NOT EXISTS annotations(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY, annotation_id TEXT NOT NULL
            REFERENCES annotations(id), payload TEXT NOT NULL);
          CREATE TRIGGER IF NOT EXISTS immutable_annotations BEFORE UPDATE ON annotations
            BEGIN SELECT RAISE(ABORT,'immutable video annotation'); END;
          CREATE TRIGGER IF NOT EXISTS immutable_results BEFORE UPDATE ON results
            BEGIN SELECT RAISE(ABORT,'immutable tracking result'); END;
        ''')
        return db

    def _validated(self, raw):
        if len(encoded(raw)) > 16384:
            raise ValueError('영상 라벨은 16KB 이하의 좌표 정보만 저장할 수 있습니다.')
        if 'clip_path' in raw:
            raise ValueError('로컬 파일 경로 대신 등록된 영상 ID를 사용하세요.')
        annotation = validate_annotation(raw)
        clip = next((item for item in self.registry.catalog()['clips'] if item['id'] == annotation['clip_id']), None)
        if clip is None or not clip['usable_for_tracking']:
            raise ValueError('추적 가능한 등록 영상을 선택하세요.')
        if clip['clip_sha256'] != annotation['clip_sha256'] or clip['source_url'] != annotation['source_url']:
            raise ValueError('라벨의 영상 출처 또는 파일 식별자가 등록 내용과 다릅니다.')
        if annotation['image_dimensions'] != {'width': clip['width'], 'height': clip['height']}:
            raise ValueError('라벨 좌표는 원본 영상 크기를 사용해야 합니다.')
        if annotation['release_time'] >= clip['duration_seconds']:
            raise ValueError('투구 구간은 영상 재생 시간 안에 있어야 합니다.')
        return annotation

    @staticmethod
    def _run_identity(annotation_id):
        source = Path(__file__).with_name('video_lab.py')
        versions = {}
        for package in ('opencv-python-headless', 'numpy'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return identity({'annotation_id': annotation_id, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'dependency_versions': versions,
                         'algorithm': 'fixed_manual_template_local_search', 'search_radius': 40,
                         'confidence_threshold': .7, 'scene_difference_threshold': .20})

    def save(self, raw):
        annotation = self._validated(raw)
        key = identity(annotation)
        with closing(self.connect()) as db, db:
            db.execute('INSERT OR IGNORE INTO annotations VALUES (?,?)', (key, encoded(annotation)))
        return self.get(key)

    def get(self, key):
        with closing(self.connect()) as db:
            row = db.execute('SELECT payload FROM annotations WHERE id=?', (key,)).fetchone()
            if row is None:
                raise KeyError('저장된 영상 라벨을 찾을 수 없습니다.')
            result = db.execute('SELECT payload FROM results WHERE id=?', (self._run_identity(key),)).fetchone()
        return {'id': key, 'annotation': json.loads(row['payload']),
                'result': json.loads(result['payload']) if result else None}

    def list(self):
        with closing(self.connect()) as db:
            keys = [row['id'] for row in db.execute('SELECT id FROM annotations ORDER BY rowid DESC LIMIT 200')]
        return {'annotations': [self.get(key) for key in keys]}

    def track(self, key):
        with self.lock:
            saved = self.get(key)
            annotation = self._validated(saved['annotation'])
            if saved['result'] is not None:
                return saved
            run_id = self._run_identity(key)
            result = track_video(self.registry.path(annotation['clip_id']), annotation)
            if run_id != self._run_identity(key):
                raise RuntimeError('추적 중 코드가 변경되어 결과를 저장하지 않았습니다.')
            result['run_id'] = run_id
            result['annotation_id'] = key
            with closing(self.connect()) as db, db:
                db.execute('INSERT OR IGNORE INTO results VALUES (?,?,?)', (run_id, key, encoded(result)))
            return self.get(key)
