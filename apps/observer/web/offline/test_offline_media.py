"""Saved video copy for the offline bundle: --media-dir checks and byte-range serving (synthetic bytes; no network)."""
import hashlib
import http.client
import os
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from serve_offline import SUMS, load_media, load_snapshot, make_handler
from test_offline_bundle import write_bundle

HERE = Path(__file__).resolve().parent
NAME = 'clip_1280x720.mp4'
BODY = bytes(range(256)) * 40  # 10240 bytes, every offset distinguishable enough for slicing checks


def write_media(directory, files=None, sums=None):
    directory.mkdir(parents=True, exist_ok=True)
    files = {NAME: BODY} if files is None else files
    for name, body in files.items():
        (directory / name).write_bytes(body)
    if sums is None:
        sums = ''.join(f'{hashlib.sha256(b).hexdigest()}  {n}\n' for n, b in files.items())
    (directory / SUMS).write_text(sums)
    return directory


def start(handler):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def media_server(tmp_path):
    dist, snap, _ = write_bundle(tmp_path)
    manifest, loaded = load_snapshot(snap)
    server = start(make_handler(dist, manifest, loaded, load_media(write_media(tmp_path / 'media'))))
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def request(port, path, method='GET', **headers):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    conn.request(method, path, headers=headers)
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r.status, r.headers, body


URL = f'/offline-media/{NAME}'


def test_full_get_and_head(media_server):
    status, headers, body = request(media_server, URL)
    assert (status, body) == (200, BODY)
    assert headers['Content-Type'] == 'video/mp4' and headers['Accept-Ranges'] == 'bytes'
    assert headers['Content-Length'] == str(len(BODY)) and 'Content-Range' not in headers
    status, headers, body = request(media_server, URL, 'HEAD')
    assert (status, body, headers['Content-Length'], headers['Content-Type']) == (200, b'', str(len(BODY)), 'video/mp4')


@pytest.mark.parametrize('spec, start, end', [
    ('bytes=0-1', 0, 1),                      # the first probe Safari sends
    ('bytes=0-', 0, len(BODY) - 1),           # open range (Chromium's first request)
    ('bytes=100-', 100, len(BODY) - 1),
    ('bytes=100-199', 100, 199),
    (f'bytes={len(BODY) - 1}-{len(BODY) - 1}', len(BODY) - 1, len(BODY) - 1),  # last byte
    ('bytes=-1', len(BODY) - 1, len(BODY) - 1),  # suffix: last byte
    ('bytes=-500', len(BODY) - 500, len(BODY) - 1),
    (f'bytes=-{len(BODY) + 50}', 0, len(BODY) - 1),  # suffix longer than the file: whole file
    (f'bytes=10-{len(BODY) + 999}', 10, len(BODY) - 1),  # end past the file is clipped
])
def test_single_range_is_206(media_server, spec, start, end):
    status, headers, body = request(media_server, URL, Range=spec)
    assert status == 206, spec
    assert body == BODY[start:end + 1]
    assert headers['Content-Range'] == f'bytes {start}-{end}/{len(BODY)}'
    assert headers['Content-Length'] == str(end - start + 1) and headers['Content-Type'] == 'video/mp4'


@pytest.mark.parametrize('spec', [f'bytes={len(BODY)}-', f'bytes={len(BODY) + 5}-{len(BODY) + 9}', 'bytes=-0'])
def test_unsatisfiable_range_is_416(media_server, spec):
    status, headers, body = request(media_server, URL, Range=spec)
    assert (status, body) == (416, b''), spec
    assert headers['Content-Range'] == f'bytes */{len(BODY)}'


@pytest.mark.parametrize('spec', ['bytes=0-1,5-9', 'bytes=5-2', 'items=0-1', 'bytes=x-1', 'bytes=-'])
def test_multi_or_bad_range_sends_full_file(media_server, spec):
    status, headers, body = request(media_server, URL, Range=spec)
    assert (status, body) == (200, BODY), spec
    assert 'Content-Range' not in headers


def test_head_and_stale_if_range_ignore_range(media_server):
    status, headers, body = request(media_server, URL, 'HEAD', Range='bytes=0-1')
    assert (status, body, headers['Content-Length']) == (200, b'', str(len(BODY)))
    status, headers, _ = request(media_server, URL)
    etag = headers['ETag']
    assert etag == f'"{hashlib.sha256(BODY).hexdigest()}"'
    assert request(media_server, URL, Range='bytes=0-1', **{'If-Range': etag})[0] == 206
    status, _, body = request(media_server, URL, Range='bytes=0-1', **{'If-Range': '"other"'})
    assert (status, body) == (200, BODY)


@pytest.mark.parametrize('path', ['/offline-media/other.mp4', '/offline-media/../secret.txt',
                                  '/offline-media/%2e%2e/secret.txt', f'/offline-media/{SUMS}',
                                  '/offline-media/', f'/offline-media/sub/{NAME}', f'/offline-media/{NAME}.part'])
def test_unknown_or_traversal_names_are_404(media_server, path):
    assert request(media_server, path)[0] == 404, path


def test_media_banner_and_pages(media_server):
    status, _, page = request(media_server, '/showcase?media=local&scene=1')
    assert status == 200 and '맥북 사본(주소 끝 ?media=local)'.encode() in page


def test_without_media_dir_nothing_is_served(tmp_path):
    dist, snap, _ = write_bundle(tmp_path)
    write_media(tmp_path / 'media')
    manifest, loaded = load_snapshot(snap)
    server = start(make_handler(dist, manifest, loaded))
    try:
        port = server.server_address[1]
        assert request(port, URL)[0] == 404
        _, _, page = request(port, '/showcase')
        assert '영상은 인터넷 필요'.encode() in page and b'media=local' not in page
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('case, match', [
    ('no_sums', 'missing'),
    ('changed', 'sha256 mismatch'),
    ('listed_missing', 'listed file missing'),
    ('unlisted_mp4', 'unlisted video'),
    ('empty', 'lists no file'),
    ('bad_line', 'line 1'),
    ('traversal_name', 'line 1'),
    ('duplicate', 'line 2'),
])
def test_media_dir_refusals(tmp_path, case, match):
    d = tmp_path / 'media'
    sha = hashlib.sha256(BODY).hexdigest()
    if case == 'no_sums':
        d.mkdir()
        (d / NAME).write_bytes(BODY)
    elif case == 'changed':
        write_media(d)
        (d / NAME).write_bytes(BODY[:-1] + b'\x00')
    elif case == 'listed_missing':
        write_media(d, files={}, sums=f'{sha}  {NAME}\n')
    elif case == 'unlisted_mp4':
        write_media(d)
        (d / 'extra.MP4').write_bytes(b'x')
    elif case == 'empty':
        write_media(d, files={}, sums='\n')
    elif case == 'bad_line':
        write_media(d, sums=f'{sha} {NAME}\n')  # one space: not the sha256sum format
    elif case == 'traversal_name':
        write_media(d, sums=f'{sha}  ../{NAME}\n')
    elif case == 'duplicate':
        write_media(d, sums=f'{sha}  {NAME}\n{sha}  {NAME}\n')
    with pytest.raises(SystemExit, match=match):
        load_media(d)


def test_symlinked_media_is_refused(tmp_path):
    d = write_media(tmp_path / 'media', files={})
    (tmp_path / NAME).write_bytes(BODY)
    os.symlink(tmp_path / NAME, d / NAME)
    (d / SUMS).write_text(f'{hashlib.sha256(BODY).hexdigest()}  {NAME}\n')
    with pytest.raises(SystemExit, match='listed file missing'):
        load_media(d)


def test_cli_refuses_to_start_on_changed_media(tmp_path):
    dist, snap, _ = write_bundle(tmp_path)
    d = write_media(tmp_path / 'media')
    (d / NAME).write_bytes(b'changed')
    r = subprocess.run([sys.executable, str(HERE / 'serve_offline.py'), '--dist', str(dist), '--snapshot', str(snap),
                        '--media-dir', str(d), '--port', '0'], capture_output=True, text=True, timeout=20)
    assert r.returncode != 0 and 'sha256 mismatch' in r.stderr and 'http://' not in r.stdout
