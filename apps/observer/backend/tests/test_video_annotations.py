import copy
import sqlite3

import pytest

from observer_app import video_annotations as module


@pytest.fixture
def lab(tmp_path, monkeypatch):
    store = module.VideoAnnotations(tmp_path)
    clip = {'id': 'clip', 'usable_for_tracking': True, 'clip_sha256': 'a'*64,
            'source_url': 'https://example.org/clip', 'width': 1280, 'height': 720, 'duration_seconds': 5}
    monkeypatch.setattr(store.registry, 'catalog', lambda: {'clips': [clip]})
    monkeypatch.setattr(store.registry, 'path', lambda _key: tmp_path/'clip.mp4')
    annotation = {'schema_version': 1, 'clip_id': 'clip', 'clip_sha256': 'a'*64,
                  'source_url': clip['source_url'], 'pitch_id': None, 'seed_time': 1., 'end_time': 1.5,
                  'release_time': 2., 'roi': {'x': 10, 'y': 10, 'w': 20, 'h': 20},
                  'image_dimensions': {'width': 1280, 'height': 720}, 'label_source': 'assistant_visual_estimate',
                  'review_status': 'unreviewed', 'annotation_version': 1, 'annotated_at': '2026-09-21T13:34:00Z'}
    return store, annotation


def test_immutable_versions_and_restart(lab):
    store, annotation = lab
    first = store.save(annotation)
    assert store.save(annotation) == first
    changed = copy.deepcopy(annotation)
    changed['roi']['x'] += 1
    changed['annotation_version'] = 2
    second = store.save(changed)
    assert second['id'] != first['id']
    with sqlite3.connect(store.path) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute('UPDATE annotations SET payload=? WHERE id=?', ('{}', first['id']))
    restored = module.VideoAnnotations(store.run)
    assert restored.get(first['id']) == first
    assert len(restored.list()['annotations']) == 2


@pytest.mark.parametrize('field,value', [('clip_sha256', 'b'*64), ('source_url', 'https://wrong.example/'),
    ('image_dimensions', {'width': 640, 'height': 360}), ('release_time', 9), ('end_time', 2)])
def test_registry_identity_and_timing_before_save(lab, field, value):
    store, annotation = lab
    annotation[field] = value
    with pytest.raises(ValueError):
        store.save(annotation)
    assert store.list()['annotations'] == []


def test_tracking_is_explicit_cached_and_runs_outside_write_transaction(lab, monkeypatch):
    store, annotation = lab
    calls = []
    def track(_path, supplied):
        with sqlite3.connect(store.path, timeout=.1) as db:
            db.execute('BEGIN IMMEDIATE')
        calls.append(supplied)
        return {'status': 'abstained', 'pixel_target': None, 'abstain_reason': 'occlusion'}
    monkeypatch.setattr(module, 'track_video', track)
    saved = store.save(annotation)
    assert calls == [] and saved['result'] is None
    result = store.track(saved['id'])
    assert result['result']['status'] == 'abstained'
    assert store.track(saved['id']) == result
    assert len(calls) == 1
    assert store.get(saved['id'])['annotation']['review_status'] == 'unreviewed'


def test_failed_tracking_preserves_saved_label(lab, monkeypatch):
    store, annotation = lab
    saved = store.save(annotation)
    def fail(*_args):
        raise ValueError('decoder unavailable')
    monkeypatch.setattr(module, 'track_video', fail)
    with pytest.raises(ValueError):
        store.track(saved['id'])
    assert store.get(saved['id']) == saved
