"""Serve the showcase build + a frozen API snapshot on this machine only (MacBook fallback, stdlib only).

    python3 serve_offline.py            # then open http://127.0.0.1:8790/showcase

Every API answer is a file from capture_snapshot.py (sha256 checked at start): byte-identical to the preview, except
/api/watch/games, which is the preview list filtered to the snapshot game (marked in manifest.json). Anything else
under /api gets an explicit error instead of made-up data; live games are never offered. The MLB video still comes
from the internet. A top line on every non-embedded page says this is a saved copy.

    python3 serve_offline.py --media-dir media   # also serve a saved copy: open /showcase?media=local

--media-dir serves only the files listed in DIR/MEDIA-SHA256SUMS (fetch_media.py writes it; every hash is checked at
start, and an unlisted .mp4 stops the server) at /offline-media/<name>. GET honours one byte range (RFC 9110 §14):
206 + Content-Range, 416 + `bytes */size` when unsatisfiable. A multi-range or malformed Range header is ignored and
answered 200 with the full file; HEAD ignores Range (§14.2: range handling is defined for GET only).
"""
import argparse
import hashlib
import html
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_snapshot import snapshot_file  # noqa: E402

HERE = Path(__file__).resolve().parent
MISSING = '오프라인 예비 묶음에는 이 자료가 없어요. 시연 장면 {scenes}개만 저장돼 있어요.'
MISSING_ALL = '오프라인 예비 묶음에는 이 자료가 없어요. 경기 {game_pk}의 공 {decisions}개만 저장돼 있어요.'
NO_LIVE = '오프라인 예비 묶음이라 실시간 경기 정보는 볼 수 없어요.'
SUMS = 'MEDIA-SHA256SUMS'
MEDIA_PREFIX = '/offline-media/'
CHUNK = 1 << 20
SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\.mp4')
SINGLE_RANGE = re.compile(r'bytes=(\d*)-(\d*)', re.IGNORECASE)


def load_snapshot(directory):
    """Manifest + verified bytes per API path; any missing or altered file stops the server."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    answers = {}
    for entry in manifest['files']:
        if snapshot_file(entry['path']) != entry['file']:  # before reading: a manifest cannot point outside api/
            raise SystemExit(f'unexpected file name for {entry["path"]}')
        if not (directory / entry['file']).is_file():
            raise SystemExit(f'snapshot file missing: {entry["file"]}')
        body = (directory / entry['file']).read_bytes()
        if hashlib.sha256(body).hexdigest() != entry['sha256'] or len(body) != entry['bytes']:
            raise SystemExit(f'snapshot file changed: {entry["file"]}')
        answers[entry['path']] = body
    if all_decisions(manifest):  # a bundle that claims every pitch must really hold every reveal and summary
        game, n, top = manifest['game_pk'], manifest.get('decisions'), manifest.get('top', 5)
        if type(n) is not int or n <= 0:  # checked before range(); bool is an int subclass
            raise SystemExit(f'all-decisions snapshot has no valid decision count: {n!r}')
        saved = json.loads(answers.get(f'/api/watch/{game}', b'{}')).get('decisions')
        if not isinstance(saved, list) or len(saved) != n:  # the banner count must be the watch page's pitch count
            raise SystemExit(f'all-decisions snapshot: manifest says {n} decisions, saved timeline does not')
        need = [f'/api/watch/{game}/reveal/{i}' for i in range(n)]
        need += [f'/api/watch/{game}/summary?through={i}&top={top}' for i in range(n)]
        lacking = [p for p in need if p not in answers]
        if lacking:
            raise SystemExit(f'all-decisions snapshot incomplete: {len(lacking)} of {len(need)} missing')
    return manifest, answers


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_media(directory):
    """{name: (path, size, sha256)} for the files listed in MEDIA-SHA256SUMS; any doubt stops the server."""
    directory = Path(directory)
    sums = directory / SUMS
    if not sums.is_file():
        raise SystemExit(f'media: {sums} missing (fetch_media.py writes it)')
    media = {}
    for n, line in enumerate(sums.read_text().splitlines(), 1):
        if not line.strip():
            continue
        m = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', line)  # sha256sum / shasum -a 256 format
        if not m or not SAFE_NAME.fullmatch(m[2]) or m[2] in media:
            raise SystemExit(f'media: {SUMS} line {n} is not "<sha256>  <name>.mp4" with a new plain file name')
        path = directory / m[2]
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f'media: listed file missing: {m[2]}')
        if file_sha256(path) != m[1]:
            raise SystemExit(f'media: file changed (sha256 mismatch): {m[2]}')
        media[m[2]] = (path, path.stat().st_size, m[1])
    if not media:
        raise SystemExit(f'media: {SUMS} lists no file')
    extra = sorted(p.name for p in directory.iterdir() if p.suffix.lower() == '.mp4' and p.name not in media)
    if extra:
        raise SystemExit(f'media: unlisted video in {directory}: {", ".join(extra)}')
    return media


def byte_range(header, size):
    """(start, end) inclusive for one satisfiable range, 'unsatisfiable', or None to send the full file."""
    m = SINGLE_RANGE.fullmatch(header.strip().replace(' ', ''))
    if not m or (not m[1] and not m[2]):
        return None  # other unit, several ranges or bad syntax: ignored (RFC 9110 §14.2 allows it)
    if not m[1]:  # suffix: the last n bytes
        n = int(m[2])
        return (size - min(n, size), size - 1) if n > 0 and size > 0 else 'unsatisfiable'
    start = int(m[1])
    if m[2] and int(m[2]) < start:
        return None  # invalid range-spec: ignored
    end = int(m[2]) if m[2] else size - 1
    return (start, min(end, size - 1)) if start < size else 'unsatisfiable'


def api_key(path, query):
    """Canonical snapshot key; summary query order/extra params must not matter."""
    if path.endswith('/summary'):
        q = parse_qs(query)
        if set(q) - {'through', 'top'} or not q.get('through') or any(len(v) > 1 for v in q.values()):
            return None  # a repeated param is ambiguous: no saved answer
        return f'{path}?through={q["through"][0]}&top={q.get("top", ["5"])[0]}'
    return path if not query else None


def all_decisions(manifest):
    """v1/v2 manifests have no coverage field: they hold only the scene paths."""
    return manifest.get('coverage') == 'all_decisions'


def missing_text(manifest):
    if all_decisions(manifest):
        return MISSING_ALL.format(game_pk=manifest['game_pk'], decisions=manifest['decisions'])
    return MISSING.format(scenes=manifest['scenes'])


def banner(manifest, media=False):
    scope = f'공 {manifest["decisions"]}개 전체' if all_decisions(manifest) else '시연 장면만'
    video = '영상은 인터넷 또는 맥북 사본(주소 끝 ?media=local)' if media else '영상은 인터넷 필요'
    text = (f'오프라인 예비 묶음 · 맥미니 응답을 {manifest["captured_kst"]}에 저장한 사본 · '
            f'경기 {manifest["game_pk"]} {scope} · {video}')
    return ('<div id="pz-offline-note" role="note" style="background:#fff4d6;color:#4a3a00;font:13px/1.4 system-ui,'
            'sans-serif;padding:6px 10px;border-bottom:1px solid #e6cf8a">' + html.escape(text) + '</div>')


def make_handler(dist, manifest, answers, media=None):
    dist = Path(dist).resolve()
    index = (dist / 'index.html').read_bytes()
    note = banner(manifest, bool(media)).encode()
    missing = missing_text(manifest)

    class Handler(BaseHTTPRequestHandler):
        server_version = 'pitcheezy-offline'

        def send(self, status, body, ctype, cache='no-store'):
            self.send_response(status)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', cache)
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def error_json(self, status, detail):
            self.send(status, json.dumps({'detail': detail}, ensure_ascii=False).encode(), 'application/json')

        def do_HEAD(self):
            self.do_GET()

        def send_media(self, name):
            path, size, sha = media[name]
            etag = f'"{sha}"'
            span = None
            # If-Range other than this file's strong ETag means "send it all" (§13.1.5); dates are not matched.
            if self.command == 'GET' and self.headers.get('Range') and self.headers.get('If-Range', etag) == etag:
                span = byte_range(self.headers['Range'], size)
            if span == 'unsatisfiable':
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.send_header('Accept-Ranges', 'bytes')
                return self.end_headers()
            start, end = span or (0, size - 1)
            self.send_response(206 if span else 200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('ETag', etag)
            self.send_header('Cache-Control', 'no-cache')
            if span:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command == 'HEAD':
                return
            left = end - start + 1
            try:
                with open(path, 'rb') as f:
                    f.seek(start)
                    while left > 0 and (chunk := f.read(min(CHUNK, left))):
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the player dropped a range it no longer needs (normal while seeking)

        def do_GET(self):
            parts = urlsplit(self.path)
            path = parts.path
            if media and path.startswith(MEDIA_PREFIX):
                name = path[len(MEDIA_PREFIX):]
                return self.send_media(name) if name in media else self.error_json(404, '저장된 영상이 아니에요.')
            if path.startswith('/api/'):
                if path.startswith('/api/live/'):
                    return self.error_json(503, NO_LIVE)
                key = api_key(path, parts.query)
                body = answers.get(key) if key else None
                return self.send(200, body, 'application/json') if body is not None else self.error_json(404, missing)
            if path != '/' and '.' in path.rsplit('/', 1)[-1]:
                target = (dist / path.lstrip('/')).resolve()
                if dist not in target.parents or not target.is_file():
                    return self.error_json(404, '파일이 없어요.')
                ctype = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
                cache = 'public, max-age=31536000, immutable' if path.startswith('/assets/') else 'no-cache'
                return self.send(200, target.read_bytes(), ctype, cache)
            # SPA route: same index as the preview; the saved-copy line is left out of the embedded watch frame.
            embedded = parse_qs(parts.query).get('embed') == ['1']
            body = index if embedded else index.replace(b'<body>', b'<body>' + note, 1)
            return self.send(200, body, 'text/html; charset=utf-8', 'no-cache')

        def log_message(self, fmt, *args):
            sys.stderr.write('%s %s\n' % (self.log_date_time_string(), fmt % args))

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default=str(HERE.parent / 'dist'))
    parser.add_argument('--snapshot', default=str(HERE / 'snapshot'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8790)
    parser.add_argument('--media-dir', help=f'serve the saved video copy listed in DIR/{SUMS} (/showcase?media=local)')
    args = parser.parse_args()
    manifest, answers = load_snapshot(args.snapshot)
    media = load_media(args.media_dir) if args.media_dir else None
    if b'<body>' not in (Path(args.dist) / 'index.html').read_bytes():
        raise SystemExit('index.html has no <body> tag')
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(args.dist, manifest, answers, media))
    except OSError as e:
        raise SystemExit(f'{args.host}:{args.port} 포트를 열지 못했어요({e.strerror}). 다른 번호로: --port {args.port + 1}')
    mitt = 'no setup' if manifest.get('setup_rows') is None else f'setup rows {manifest["setup_rows"]}'
    port = server.server_address[1]  # differs from --port only for --port 0 (tests)
    print(f'http://{args.host}:{port}/showcase  ({len(answers)} saved answers, {mitt}, {manifest["captured_kst"]})',
          flush=True)
    for name, (_, size, sha) in (media or {}).items():
        print(f'http://{args.host}:{port}/showcase?media=local  (saved video {name}, {size} bytes, sha256 {sha})',
              flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
