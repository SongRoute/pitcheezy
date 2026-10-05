"""Writes a PA-level demo mapping (kind ``pa_end_observed``) from a review CSV in which a person observed only PA-ENDING
result times on a remote official video (e.g. observed-anchors-020: 3 of 16 rows).

Only rows with status OBSERVED_RESULT_VISIBLE, a numeric offset, verified_by and evidence are used, and each must be the
last pitch of its PA in the game record. The observed PAs must be consecutive, starting with the PA that begins at the
segment's first pitch. Rows without an observation are left out (never interpolated). The result is verified=false: the
clock reveals each whole PA at the person's observed PA-end time; pitch-level sync stays unknown.

  python apps/observer/scripts/build_pa_end_mapping.py --game 813031 --id g813031-ab1-3-pa-end \
      --from-csv .../observed-anchors-020/mapping-813031-AB1-3-PARTIAL-3of16-OBSERVED.csv \
      --video-url 'https://www.youtube.com/watch?v=45Rqrp4aZM4' --lead-s 30 \
      --feed-dir .../preview-data-overlay-014/overlay/feeds --demo-dir <PITCHEEZY_DEMO_DIR>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backend'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from observer_app import game_history  # noqa: E402
from observer_app.demo_clock import DEFAULT_DIR, PA_END_METHOD, validate_mapping  # noqa: E402
from observer_app.service_api import DEFAULT_FEED_DIR  # noqa: E402
from build_demo_mapping import youtube_id  # noqa: E402

OBSERVED = 'OBSERVED_RESULT_VISIBLE'


def build(record, table, *, mapping_id, game_pk, video_url, csv_sha256, lead_s):
    pitches, pas = record['pitches'], record['pas']
    keys = [p['key'] for p in pitches]
    pa_of = {p['index']: p['pa_key'] for p in pitches}
    last_of = {pa['pa_key']: pa['last_index'] for pa in pas if pa.get('complete')}
    rows = []
    for line in table:
        if (line.get('status') or '').strip() != OBSERVED:
            continue
        index, key = int(line['index']), line['pitch_key'].strip()
        if index >= len(keys) or keys[index] != key:
            raise ValueError(f'row {index}: pitch_key {key} differs from the game record')
        if last_of.get(pa_of[index]) != index:
            raise ValueError(f'row {index} ({key}) is not the last pitch of its PA; only PA-end observations are allowed')
        at = float(line['video_offset_result_visible_s'])
        by, evidence = (line.get('verified_by') or '').strip(), (line.get('evidence') or '').strip()
        if not math.isfinite(at) or at < 0 or not by or not evidence:
            raise ValueError(f'row {index}: needs a finite offset, verified_by and evidence')
        rows.append({'index': index, 'pitch_key': key, 'reveal_at_s': at, 'review': {'by': by, 'evidence': evidence}})
    if not rows:
        raise ValueError('no observed rows')
    rows.sort(key=lambda r: r['index'])
    # consecutive PAs from the segment start: the first observed PA's first pitch, then each next PA directly after
    order = [pa['pa_key'] for pa in pas]
    first_pa = pa_of[rows[0]['index']]
    first_index = next(pa['first_index'] for pa in pas if pa['pa_key'] == first_pa)
    for before, after in zip(rows, rows[1:]):
        if order.index(pa_of[after['index']]) != order.index(pa_of[before['index']]) + 1:
            raise ValueError(f"rows {before['index']} and {after['index']} are not consecutive PAs (a PA end in between is unobserved)")
    vid = youtube_id(video_url)
    mapping = {'schema': 'pitcheezy-demo-mapping-v1', 'mapping_id': mapping_id, 'game_pk': game_pk, 'kind': 'pa_end_observed',
               'verified': False, 'lead_s': lead_s,
               'segment': {'first_index': first_index, 'last_index': rows[-1]['index'], 'pas': [pa_of[r['index']] for r in rows]},
               'video': {'source': 'remote_observed_pa_end', 'platform': 'youtube', 'video_id': vid,
                         'url': f'https://www.youtube.com/watch?v={vid}', 'sha256': None,
                         'content_note': 'stream watched by a person; bytes never read, so no content hash'},
               'verification': {'method': PA_END_METHOD, 'csv_sha256': csv_sha256, 'rows_observed': len(rows),
                                'semantics': 'each row = first video time the PA-ending result is visible; '
                                             'pitches inside a PA carry no time and are revealed together at the PA end'},
               'rows': rows}
    return validate_mapping(mapping, keys)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--game', type=int, required=True)
    ap.add_argument('--id', required=True)
    ap.add_argument('--from-csv', required=True)
    ap.add_argument('--video-url', required=True)
    ap.add_argument('--lead-s', type=float, default=30.0, help='clock starts this many seconds before the first PA-end time')
    ap.add_argument('--feed-dir', default=os.environ.get('PITCHEEZY_FEED_DIR', str(DEFAULT_FEED_DIR)))
    ap.add_argument('--demo-dir', default=os.environ.get('PITCHEEZY_DEMO_DIR', str(DEFAULT_DIR)))
    args = ap.parse_args(argv)
    feed_path = Path(args.feed_dir)/f'{args.game}_final.json.gz'
    record = game_history.build(json.load(gzip.open(feed_path)))
    raw = Path(args.from_csv).read_bytes()
    table = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    try:
        mapping = build(record, table, mapping_id=args.id, game_pk=args.game, video_url=args.video_url,
                        csv_sha256=hashlib.sha256(raw).hexdigest(), lead_s=args.lead_s)
    except ValueError as error:
        ap.error(f'mapping refused: {error}')
    out = Path(args.demo_dir)/'mappings'/f'{args.id}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        ap.error(f'{out} exists; choose a new --id (mappings are immutable)')
    out.write_text(json.dumps(mapping, ensure_ascii=False, indent=1))
    print(json.dumps({'path': str(out), 'kind': mapping['kind'], 'verified': False, 'rows': [(r['index'], r['pitch_key'], r['reveal_at_s']) for r in mapping['rows']],
                      'segment_first_index': mapping['segment']['first_index'], 'lead_s': mapping['lead_s']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
