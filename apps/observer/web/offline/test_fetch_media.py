"""fetch_media.py against a local 127.0.0.1 server only (tiny synthetic file; never the MLB URL)."""
import hashlib
import http.client
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fetch_media import add_sum, fetch, main
from serve_offline import SUMS, load_media

NAME = 'asset_1280x720_59_4000K.mp4'
BODY = b'\x00\x00\x00\x18ftypmp42' + bytes(range(256)) * 3


class Origin(BaseHTTPRequestHandler):
    requests = []
    short = False  # announce the full length but send half, then close

    def do_GET(self):
        Origin.requests.append(self.path)
        if self.path != f'/media/{NAME}':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Content-Length', str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY[:len(BODY) // 2] if Origin.short else BODY)

    def log_message(self, *args):
        pass


@pytest.fixture
def origin():
    Origin.requests, Origin.short = [], False
    server = ThreadingHTTPServer(('127.0.0.1', 0), Origin)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_address[1]}/media/{NAME}'
    server.shutdown()
    server.server_close()


def test_fetch_then_serve_check_accepts_it(origin, tmp_path):
    path, size, sha = fetch(origin, tmp_path / 'media')
    add_sum(tmp_path / 'media', path.name, sha)
    assert (path.read_bytes(), size, sha) == (BODY, len(BODY), hashlib.sha256(BODY).hexdigest())
    assert (tmp_path / 'media' / SUMS).read_text() == f'{sha}  {NAME}\n'
    assert not (tmp_path / 'media' / f'{NAME}.part').exists()
    assert load_media(tmp_path / 'media')[NAME][1:] == (len(BODY), sha)


def test_existing_file_is_never_overwritten(origin, tmp_path):
    (tmp_path / NAME).write_bytes(b'mine')
    with pytest.raises(SystemExit, match='not overwritten'):
        fetch(origin, tmp_path)
    assert (tmp_path / NAME).read_bytes() == b'mine' and Origin.requests == []


def test_leftover_part_is_not_overwritten(origin, tmp_path):
    (tmp_path / f'{NAME}.part').write_bytes(b'half')
    with pytest.raises(SystemExit, match='unfinished download'):
        fetch(origin, tmp_path)
    assert (tmp_path / f'{NAME}.part').read_bytes() == b'half' and Origin.requests == []


def test_short_download_leaves_no_file(origin, tmp_path):
    Origin.short = True
    with pytest.raises((SystemExit, http.client.IncompleteRead)):
        fetch(origin, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_http_error_leaves_no_file(origin, tmp_path):
    with pytest.raises(urllib.error.HTTPError):
        fetch(origin.replace(NAME, 'other.mp4'), tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_sum_line_is_not_duplicated(tmp_path):
    add_sum(tmp_path, NAME, '0' * 64)
    with pytest.raises(SystemExit, match='already lists'):
        add_sum(tmp_path, NAME, '1' * 64)


def scenes_file(tmp_path, url):
    path = tmp_path / 'scenes.json'
    path.write_text(json.dumps({'gamePk': 1, 'mediaUrl': url, 'scenes': []}))
    return path


def test_dry_run_prints_only(origin, tmp_path, capsys):
    main(['--scenes', str(scenes_file(tmp_path, origin)), '--target', str(tmp_path / 'media'), '--dry-run'])
    out = capsys.readouterr().out
    assert origin in out and str(tmp_path / 'media' / NAME) in out
    assert not (tmp_path / 'media').exists() and Origin.requests == []


def test_cli_downloads_and_prints_size_and_sha(origin, tmp_path, capsys):
    main(['--scenes', str(scenes_file(tmp_path, origin)), '--target', str(tmp_path / 'media')])
    out = capsys.readouterr().out
    assert f'bytes  {len(BODY)}' in out and hashlib.sha256(BODY).hexdigest() in out
    assert Origin.requests == [f'/media/{NAME}']


def test_non_mp4_url_is_refused(tmp_path):
    with pytest.raises(SystemExit, match='plain .mp4'):
        fetch('http://127.0.0.1:9/media/clip.mov', tmp_path)  # refused before any connection
