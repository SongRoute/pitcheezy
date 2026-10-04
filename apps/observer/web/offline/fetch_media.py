"""Save the showcase's official MLB video once, for serve_offline.py --media-dir (stdlib only).

    python3 fetch_media.py --dry-run          # prints the URL and the target file; no network
    python3 fetch_media.py                    # downloads into ./media (about 436 MB), then writes MEDIA-SHA256SUMS

The URL is showcase-scenes.json `mediaUrl`, so the copy is the same file the page plays (same scene timestamps).
Downloads to <name>.part and renames it only when complete; an existing file is never overwritten. The copy is kept
privately for the demo and is not committed (the media folder is gitignored).
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from serve_offline import CHUNK, SAFE_NAME, SUMS  # noqa: E402


def media_name(url):
    name = urlsplit(url).path.rsplit('/', 1)[-1]
    if not SAFE_NAME.fullmatch(name):
        raise SystemExit(f'not a plain .mp4 file name: {name!r}')
    return name


def fetch(url, target_dir, timeout=60):
    """Download url into target_dir/<basename>; returns (path, size, sha256)."""
    target_dir = Path(target_dir)
    name = media_name(url)
    final, part = target_dir / name, target_dir / f'{name}.part'
    if final.exists():
        raise SystemExit(f'already there, not overwritten: {final}')
    target_dir.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    request = urllib.request.Request(url, headers={'User-Agent': 'pitcheezy-offline-fetch'})
    try:
        out = open(part, 'xb')  # a leftover .part from an earlier try is not overwritten either
    except FileExistsError:
        raise SystemExit(f'unfinished download left at {part}: check it and remove it by hand, then retry') from None
    try:
        with out, urllib.request.urlopen(request, timeout=timeout) as r:
            if r.status != 200:
                raise SystemExit(f'HTTP {r.status} for {url}')
            expected = r.headers.get('Content-Length')
            while chunk := r.read(CHUNK):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            if expected is not None and int(expected) != size:
                raise SystemExit(f'incomplete download: {size} of {expected} bytes')
            out.flush()
            os.fsync(out.fileno())
        if final.exists():
            raise SystemExit(f'appeared while downloading, not overwritten: {final}')
        os.rename(part, final)
    except BaseException:
        part.unlink(missing_ok=True)  # only this run's own partial file
        raise
    return final, size, digest.hexdigest()


def add_sum(target_dir, name, sha):
    """Append `<sha256>  <name>` (sha256sum format) to MEDIA-SHA256SUMS via a temporary file + rename."""
    sums = Path(target_dir) / SUMS
    lines = sums.read_text().splitlines() if sums.exists() else []
    if any(line.split()[-1:] == [name] for line in lines):
        raise SystemExit(f'{SUMS} already lists {name}')
    tmp = sums.with_name(SUMS + '.tmp')
    tmp.write_text(''.join(f'{line}\n' for line in lines + [f'{sha}  {name}']))
    os.replace(tmp, sums)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', default=str(HERE.parent / 'src' / 'showcase-scenes.json'))
    parser.add_argument('--target', default=str(HERE / 'media'))
    parser.add_argument('--dry-run', action='store_true', help='print the URL and target file only')
    args = parser.parse_args(argv)
    url = json.loads(Path(args.scenes).read_text())['mediaUrl']
    target = Path(args.target) / media_name(url)
    print(f'url    {url}\ntarget {target}', flush=True)
    if args.dry_run:
        return
    path, size, sha = fetch(url, args.target)
    add_sum(args.target, path.name, sha)
    print(f'saved  {path}\nbytes  {size}\nsha256 {sha}\n'
          f'next   python3 serve_offline.py --media-dir {args.target}  →  /showcase?media=local', flush=True)


if __name__ == '__main__':
    main()
