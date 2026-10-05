"""Writes a demo-clock mapping (pitcheezy-demo-mapping-v1) into ``$PITCHEEZY_DEMO_DIR/mappings``.

Two sources, never mixed:
  --from-feed   pacing fallback from the final feed's pitch times (kind game_timestamps_unsynced, verified
                false). Feed times are NOT video offsets; use only when no verified video mapping exists.
  --from-csv    an alignment table: ``index,pitch_key,video_s`` or the MEDIA template
                (``pitch_id,video_offset_result_visible_s,verified_by``). Kind video_verified needs --video (sha256
                computed here), --verified-by and, for the MEDIA template, verified_by on every row; else unsynced.

    python apps/observer/scripts/build_demo_mapping.py --game 849843 --first-ab 4 --last-ab 8 --from-feed --id g849843-ab4-8-pace
"""
import argparse
import gzip
import hashlib
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backend'))
from observer_app import game_history  # noqa: E402
from observer_app.demo_clock import DEFAULT_DIR, validate_mapping  # noqa: E402
from observer_app.service_api import DEFAULT_FEED_DIR  # noqa: E402


def _wall(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--game', type=int, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--first-ab', type=int)
    parser.add_argument('--last-ab', type=int)
    parser.add_argument('--from-feed', action='store_true')
    parser.add_argument('--from-csv')
    parser.add_argument('--video')
    parser.add_argument('--verified-by')
    parser.add_argument('--feed-dir', default=os.environ.get('PITCHEEZY_FEED_DIR', str(DEFAULT_FEED_DIR)))
    parser.add_argument('--demo-dir', default=os.environ.get('PITCHEEZY_DEMO_DIR', str(DEFAULT_DIR)))
    args = parser.parse_args(argv)
    if args.from_feed == bool(args.from_csv):
        parser.error('choose exactly one of --from-feed / --from-csv')
    feed_path = Path(args.feed_dir)/f'{args.game}_final.json.gz'
    with gzip.open(feed_path, 'rt') as handle:
        feed = json.load(handle)
    record = game_history.build(feed)
    keys = [p['key'] for p in record['pitches']]
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
        with open(args.from_csv) as handle:
            table = list(csv.DictReader(handle))
        rows, unverified_rows = [], 0
        for r in table:  # own columns (index,pitch_key,video_s) or MEDIA template (pitch_id,video_offset_result_visible_s,verified_by)
            key = r.get('pitch_key') or r.get('pitch_id')
            offset = r.get('video_s') or r.get('video_offset_result_visible_s')
            if not key or offset in (None, ''):
                parser.error(f'row without pitch key or video offset: {r}')
            if key not in keys:
                parser.error(f'unknown pitch key {key}')
            rows.append({'index': keys.index(key), 'pitch_key': key, 'reveal_at_s': float(offset)})
            unverified_rows += 'pitch_id' in r and not (r.get('verified_by') or '').strip()
        video = None
        if args.video:
            digest = hashlib.sha256()
            with open(args.video, 'rb') as handle:
                for block in iter(lambda: handle.read(1 << 20), b''):
                    digest.update(block)
            video = {'file_name': Path(args.video).name, 'sha256': digest.hexdigest()}
        verified = bool(video and args.verified_by and not unverified_rows)
        mapping = {'kind': 'video_verified' if verified else 'game_timestamps_unsynced', 'verified': verified, 'video': video,
                   'verification': {'by': args.verified_by, 'csv': Path(args.from_csv).name} if verified else
                   {'note': 'CSV without verified video; treated as unsynced pacing'}}
    mapping = {'schema': 'pitcheezy-demo-mapping-v1', 'mapping_id': args.id, 'game_pk': args.game} | mapping | {'rows': rows}
    validate_mapping(mapping, keys)
    out = Path(args.demo_dir)/'mappings'/f'{args.id}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        parser.error(f'{out} exists; choose a new --id (mappings are immutable)')
    out.write_text(json.dumps(mapping, ensure_ascii=False, indent=1))
    print(json.dumps({'path': str(out), 'kind': mapping['kind'], 'verified': mapping['verified'], 'rows': len(rows),
                      'first': rows[0], 'last': rows[-1]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
