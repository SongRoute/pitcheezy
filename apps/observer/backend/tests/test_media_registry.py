import hashlib
import json
import pytest
from observer_app.media_registry import MediaRegistry


def test_registry_only_serves_manifest_media_with_matching_hash(tmp_path):
    media = tmp_path/'media'
    media.mkdir()
    path = media/'clip.mp4'
    path.write_bytes(b'synthetic video bytes for registry-only test')
    entry = {'key': 'clip', 'local_file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path/'media_sources.json').write_text(json.dumps({'clips': [entry]}))
    registry = MediaRegistry(tmp_path)
    assert registry.path('clip') == path
    with pytest.raises(KeyError):
        registry.path('../dataset.json')
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='해시'):
        registry.path('clip')


def test_registry_does_not_accept_path_outside_media(tmp_path):
    path = tmp_path/'outside.mp4'
    path.write_bytes(b'bytes')
    (tmp_path/'media_sources.json').write_text(json.dumps({'clips': [{'key': 'outside', 'local_file': str(path), 'sha256': 'none'}]}))
    with pytest.raises(ValueError, match='경로'):
        MediaRegistry(tmp_path).path('outside')


def test_no_media_manifest_is_an_empty_honest_catalog(tmp_path):
    assert MediaRegistry(tmp_path).catalog()['clips'] == []
