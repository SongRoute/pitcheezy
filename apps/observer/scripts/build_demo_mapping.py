"""Writes a demo-clock mapping (pitcheezy-demo-mapping-v1) into ``$PITCHEEZY_DEMO_DIR/mappings``.

Three modes, never mixed:
  --from-feed   pacing fallback from the final feed's pitch times (kind game_timestamps_unsynced, verified
                false). Feed times are NOT video offsets; use only when no verified video mapping exists.
  --from-csv    an alignment table: ``index,pitch_key,video_s`` or the MEDIA template
                (``pitch_id,video_offset_result_visible_s,verified_by``). Kind video_verified needs --video (sha256
                computed here), --verified-by and, for the MEDIA template, verified_by on every row; else unsynced.
                With --video-url (no local file) the CSV is a person's observation of a remote official stream:
                kind video_verified, video.source remote_observed, no content sha256 (bytes never read). Every row
                must be filled (status OBSERVED_RESULT_VISIBLE, verified_by, evidence, offset within
                --observed-duration-s, strictly increasing); anything missing refuses the import (no downgrade).
  --template    writes a blank human-review CSV (canonical keys, people, counts, feed call) for --first/--last-ab;
                every offset blank and status PENDING_NOT_OBSERVED, so it cannot be imported until filled.

    python apps/observer/scripts/build_demo_mapping.py --game 849843 --first-ab 4 --last-ab 8 --from-feed --id g849843-ab4-8-pace
"""
import argparse
import gzip
import hashlib
import csv
import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backend'))
from observer_app import game_history  # noqa: E402
from observer_app.demo_clock import DEFAULT_DIR, REMOTE_METHOD, REMOTE_SOURCE, YOUTUBE_ID, validate_mapping  # noqa: E402
from observer_app.service_api import DEFAULT_FEED_DIR  # noqa: E402


OBSERVED = 'OBSERVED_RESULT_VISIBLE'
PENDING = 'PENDING_NOT_OBSERVED'
OFFSET_COLUMNS = ('video_s', 'video_offset_result_visible_s', 'video_seconds_result_confirmed')  # aliases, result-visible time
RESULT_RULE = ('first frame where this pitch\'s result is visible: catch + umpire call (ball/called strike), swing-and-miss, '
               'foul or ball in play; for the PA-ending pitch the walk/strikeout/out confirmation. Not the release.')
TEMPLATE_COLUMNS = ('index', 'pitch_key', 'inning', 'half', 'at_bat_number', 'pitch_number', 'batter', 'pitcher', 'count_before',
                    'outs_before', 'feed_call', 'pa_result', 'video_offset_result_visible_s', 'status', 'verified_by', 'evidence',
                    'result_rule')


def _wall(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def youtube_id(url):
    """Video id of a YouTube watch/short link, else None. Purely textual: nothing is fetched."""
    parsed = urlparse((url or '').strip())
    host = (parsed.hostname or '').lower()
    if parsed.scheme != 'https':
        return None
    if host in ('www.youtube.com', 'youtube.com', 'm.youtube.com') and parsed.path == '/watch':
        vid = (parse_qs(parsed.query).get('v') or [''])[0]
    elif host == 'youtu.be':
        vid = parsed.path.lstrip('/')
    else:
        return None
    return vid if YOUTUBE_ID.fullmatch(vid) else None


def _offset(row):
    return next((row[c] for c in OFFSET_COLUMNS if (row.get(c) or '').strip()), None)


def _template(record, chosen, out):
    rows = []
    for p in chosen:
        before, actual = p['situation_before'], p['actual']
        rows.append({'index': p['index'], 'pitch_key': p['key'], 'inning': p['inning'], 'half': p['half'],
                     'at_bat_number': p['at_bat_number'], 'pitch_number': p['pitch_number'], 'batter': p['batter']['name'],
                     'pitcher': p['pitcher']['name'], 'count_before': f"{before['balls']}-{before['strikes']}",
                     'outs_before': before['outs'], 'feed_call': actual.get('description') or '',
                     'pa_result': actual.get('event_label') or '' if actual.get('event') else '',
                     'video_offset_result_visible_s': '', 'status': PENDING, 'verified_by': '', 'evidence': '',
                     'result_rule': RESULT_RULE})
    with open(out, 'x', newline='') as handle:  # never overwrite an existing (possibly filled) sheet
        writer = csv.DictWriter(handle, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _remote(parser, args, table, keys):
    """Rows + mapping fields for a person-observed remote stream; any gap refuses the whole import."""
    vid = youtube_id(args.video_url)
    if not vid:
        parser.error('--video-url must be an https YouTube watch or youtu.be link with an 11-character video id')
    try:
        duration = float(args.observed_duration_s)
        observed_on = date.fromisoformat(args.observed_on or '')
    except (TypeError, ValueError):
        parser.error('--video-url needs --observed-duration-s (seconds) and --observed-on YYYY-MM-DD from the actual viewing')
    if not math.isfinite(duration) or duration <= 0 or observed_on > date.today():
        parser.error('observed duration must be finite and positive; observed date cannot be in the future')
    if not (args.verified_by or '').strip() or not (args.observed_channel or '').strip():
        parser.error('--video-url needs --verified-by and --observed-channel (channel name as seen on the page)')
    rows = []
    for n, r in enumerate(table, start=2):
        key, offset = (r.get('pitch_key') or r.get('pitch_id') or '').strip(), _offset(r)
        if key not in keys:
            parser.error(f'line {n}: unknown pitch key {key!r} (service keys look like game:at_bat:pitch, e.g. {keys[0]})')
        if (r.get('index') or '').strip() and r['index'].strip() != str(keys.index(key)):
            parser.error(f'line {n}: index {r["index"]} differs from the game record index {keys.index(key)} of {key}')
        if (r.get('game_pk') or '').strip() and r['game_pk'].strip() != str(args.game):
            parser.error(f'line {n}: game_pk {r["game_pk"]} is not --game {args.game}')
        if (r.get('source_url') or '').strip() and youtube_id(r['source_url']) != vid:
            parser.error(f'line {n}: source_url is a different video than --video-url')
        status, by, evidence = (r.get('status') or '').strip(), (r.get('verified_by') or '').strip(), (r.get('evidence') or '').strip()
        if offset is None or status != OBSERVED or not by or not evidence:
            parser.error(f'line {n} ({key}): remote import needs a result-visible offset, status {OBSERVED}, verified_by and '
                         f'evidence on every row (got status {status or "blank"})')
        try:
            at = float(offset)
        except ValueError:
            parser.error(f'line {n} ({key}): offset {offset!r} is not a number')
        if not math.isfinite(at) or not 0 <= at <= duration:
            parser.error(f'line {n} ({key}): offset {offset} outside 0..{duration} (observed duration)')
        if rows and not (keys.index(key) == rows[-1]['index']+1 and at > rows[-1]['reveal_at_s']):
            parser.error(f'line {n} ({key}): rows must be consecutive pitches in game order with increasing offsets')
        rows.append({'index': keys.index(key), 'pitch_key': key, 'reveal_at_s': at, 'review': {'by': by, 'evidence': evidence}})
    if not rows:
        parser.error('CSV has no rows')
    video = {'source': REMOTE_SOURCE, 'platform': 'youtube', 'video_id': vid, 'url': f'https://www.youtube.com/watch?v={vid}',
             'observed_channel': args.observed_channel.strip(), 'observed_duration_s': duration, 'sha256': None,
             'content_note': 'stream watched, not downloaded: content bytes unknown, no content hash'}
    verification = {'method': REMOTE_METHOD, 'semantic': 'reveal_at_s = first frame where the pitch result is visible',
                    'by': args.verified_by.strip(), 'observed_on': observed_on.isoformat(), 'rows_reviewed': len(rows),
                    'csv': Path(args.from_csv).name, 'csv_sha256': _sha256(args.from_csv),
                    'evidence_file': Path(args.evidence_file).name if args.evidence_file else None,
                    'evidence_file_sha256': _sha256(args.evidence_file) if args.evidence_file else None}
    return rows, {'kind': 'video_verified', 'verified': True, 'video': video, 'verification': verification}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--game', type=int, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--first-ab', type=int)
    parser.add_argument('--last-ab', type=int)
    parser.add_argument('--from-feed', action='store_true')
    parser.add_argument('--from-csv')
    parser.add_argument('--template', help='write a blank review CSV here and stop')
    parser.add_argument('--video')
    parser.add_argument('--video-url', help='remote official stream a person watched (no download)')
    parser.add_argument('--observed-duration-s', help='player duration seen during that viewing')
    parser.add_argument('--observed-on', help='date of that viewing, YYYY-MM-DD')
    parser.add_argument('--observed-channel', help='channel name as shown on the video page')
    parser.add_argument('--evidence-file', help='optional notes/screenshot bundle; only its sha256 is recorded')
    parser.add_argument('--verified-by')
    parser.add_argument('--feed-dir', default=os.environ.get('PITCHEEZY_FEED_DIR', str(DEFAULT_FEED_DIR)))
    parser.add_argument('--demo-dir', default=os.environ.get('PITCHEEZY_DEMO_DIR', str(DEFAULT_DIR)))
    args = parser.parse_args(argv)
    if sum((args.from_feed, bool(args.from_csv), bool(args.template))) != 1:
        parser.error('choose exactly one of --from-feed / --from-csv / --template')
    if args.video and args.video_url:
        parser.error('--video (local file) and --video-url (remote stream) are separate provenances; give one')
    if args.video_url and not args.from_csv:
        parser.error('--video-url only applies to --from-csv')
    feed_path = Path(args.feed_dir)/f'{args.game}_final.json.gz'
    with gzip.open(feed_path, 'rt') as handle:
        feed = json.load(handle)
    record = game_history.build(feed)
    keys = [p['key'] for p in record['pitches']]
    if args.template:
        chosen = [p for p in record['pitches'] if (args.first_ab is None or p['at_bat_number'] >= args.first_ab)
                  and (args.last_ab is None or p['at_bat_number'] <= args.last_ab)]
        rows = _template(record, chosen, args.template)
        print(json.dumps({'template': args.template, 'rows': len(rows), 'first': rows[0]['pitch_key'], 'last': rows[-1]['pitch_key'],
                          'status': PENDING}, ensure_ascii=False))
        return
    if args.from_feed:
        times = {}
        for play in feed['liveData']['plays']['allPlays']:
            ab = play['about']['atBatIndex']+1
            for event in play.get('playEvents') or []:
                if event.get('isPitch') and event.get('endTime'):
                    times[f"{args.game}:{ab}:{event['pitchNumber']}"] = _wall(event['endTime'])
        chosen = [p for p in record['pitches'] if (args.first_ab is None or p['at_bat_number'] >= args.first_ab)
                  and (args.last_ab is None or p['at_bat_number'] <= args.last_ab)]
        origin = times[chosen[0]['key']]
        rows = [{'index': p['index'], 'pitch_key': p['key'], 'reveal_at_s': round(times[p['key']]-origin+5.0, 3)} for p in chosen]
        mapping = {'kind': 'game_timestamps_unsynced', 'verified': False, 'video': None,
                   'verification': {'note': 'feed pitch endTime pacing only; not aligned to any video'}}
    else:
        with open(args.from_csv, newline='') as handle:
            table = list(csv.DictReader(handle))
        if args.video_url:
            rows, mapping = _remote(parser, args, table, keys)
        else:
            rows, mapping = _local(parser, args, table, keys)
    mapping = {'schema': 'pitcheezy-demo-mapping-v1', 'mapping_id': args.id, 'game_pk': args.game} | mapping | {'rows': rows}
    try:
        validate_mapping(mapping, keys)
    except ValueError as error:
        parser.error(f'mapping refused: {error}')
    out = Path(args.demo_dir)/'mappings'/f'{args.id}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        parser.error(f'{out} exists; choose a new --id (mappings are immutable)')
    out.write_text(json.dumps(mapping, ensure_ascii=False, indent=1))
    print(json.dumps({'path': str(out), 'kind': mapping['kind'], 'verified': mapping['verified'], 'rows': len(rows),
                      'video_source': (mapping['video'] or {}).get('source', 'local_file' if mapping['video'] else None),
                      'first': {k: rows[0][k] for k in ('index', 'pitch_key', 'reveal_at_s')},
                      'last': {k: rows[-1][k] for k in ('index', 'pitch_key', 'reveal_at_s')}}, ensure_ascii=False))


def _local(parser, args, table, keys):
    """Original path: own columns or the MEDIA template; verified only with a hashed local --video file."""
    rows, unverified_rows = [], 0
    for r in table:  # own columns (index,pitch_key,video_s) or MEDIA template (pitch_id,video_offset_result_visible_s,verified_by)
        key = r.get('pitch_key') or r.get('pitch_id')
        offset = _offset(r)
        if not key or offset in (None, ''):
            parser.error(f'row without pitch key or video offset: {r}')
        if key not in keys:
            parser.error(f'unknown pitch key {key}')
        rows.append({'index': keys.index(key), 'pitch_key': key, 'reveal_at_s': float(offset)})
        unverified_rows += ('pitch_id' in r or 'verified_by' in r) and not (r.get('verified_by') or '').strip()
    video = {'file_name': Path(args.video).name, 'sha256': _sha256(args.video)} if args.video else None
    verified = bool(video and args.verified_by and not unverified_rows)
    mapping = {'kind': 'video_verified' if verified else 'game_timestamps_unsynced', 'verified': verified, 'video': video,
               'verification': {'by': args.verified_by, 'csv': Path(args.from_csv).name} if verified else
               {'note': 'CSV without verified video; treated as unsynced pacing'}}
    return rows, mapping


if __name__ == '__main__':
    main()
