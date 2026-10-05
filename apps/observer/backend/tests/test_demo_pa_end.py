"""pa_end_observed demo mappings (3/16 observed PA-end anchors for 813031): whole PAs are revealed at the observed PA-end
times, nothing is interpolated, the mapping is never 'verified', and the state before the first PA is restored."""
import copy
import csv
import gzip
import json
import sys
from pathlib import Path

import pytest

from observer_app import game_history, service_api
from observer_app.demo_clock import DemoClock, DemoError, validate_mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import build_pa_end_mapping as builder  # noqa: E402

JOB = Path('/Users/song/Projects/pitcheezy-pm-jobs/20261005-ui-live-redesign')
FEED = JOB/'preview-data-overlay-014/overlay/feeds/813031_final.json.gz'
CSV = JOB/'observed-anchors-020/mapping-813031-AB1-3-PARTIAL-3of16-OBSERVED.csv'
pytestmark = pytest.mark.skipif(not (FEED.is_file() and CSV.is_file()), reason='813031 overlay feed / observed CSV absent')
URL = 'https://www.youtube.com/watch?v=45Rqrp4aZM4'


class Wall:
    def __init__(self):
        self.t = 1_900_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture(scope='module')
def record():
    return game_history.build(json.load(gzip.open(FEED)))


def table():
    return list(csv.DictReader(CSV.read_text(encoding='utf-8-sig').splitlines()))


def make(record, rows=None, lead=30.0):
    return builder.build(record, rows if rows is not None else table(), mapping_id='m', game_pk=813031, video_url=URL,
                         csv_sha256='0'*64, lead_s=lead)


def test_mapping_uses_only_the_three_observed_pa_ends(record):
    m = make(record)
    assert [(r['index'], r['pitch_key'], r['reveal_at_s']) for r in m['rows']] == [(5, '813031:1:6', 448.0), (9, '813031:2:4', 538.0), (15, '813031:3:6', 673.0)]
    assert m['verified'] is False and m['kind'] == 'pa_end_observed' and m['segment']['first_index'] == 0
    assert m['video']['sha256'] is None and all(r['review']['by'] == 'Song' for r in m['rows'])


def test_clock_reveals_whole_pas_at_observed_times_and_restores_pre_ab1(record, tmp_path):
    wall = Wall()
    clock = DemoClock(tmp_path, wall=wall)
    m = make(record) | {'source_sha256': 'x'}
    keys = [p['key'] for p in record['pitches']]
    s = clock.create(813031, m, keys)
    assert clock.progress(s) == 418.0 and clock.cutoff(s) == -1  # before AB1: no pitch shown
    view = service_api.cut_view(record, clock.cutoff(s), {'kind': 'demo', 'session_id': 'x'})
    assert view['pitches'] == [] and (view['game']['away_score'], view['game']['home_score']) == (0, 0)
    s = clock.control(s['session_id'], 'start')
    for elapsed, cutoff in ((29.9, -1), (30.0, 5), (119.9, 5), (120.0, 9), (254.9, 9), (255.0, 15), (400.0, 15)):
        wall.t = 1_900_000_000.0 + elapsed
        assert clock.cutoff(clock.load(s['session_id'])) == cutoff, elapsed
    # whole AB1 (6 pitches) appears together at 448 s; AB2 pitches never appear before 538 s
    wall.t = 1_900_000_000.0 + 30.0
    v = service_api.cut_view(record, clock.cutoff(clock.load(s['session_id'])), {'kind': 'demo', 'session_id': 'x'})
    assert [p['key'] for p in v['pitches']] == keys[:6] and v['pas'][0]['complete'] is True
    s = clock.control(s['session_id'], 'seek', index=9)
    assert clock.cutoff(s) == 9
    with pytest.raises(DemoError):
        clock.control(s['session_id'], 'seek', index=7)  # pitches inside a PA have no time
    s = clock.control(s['session_id'], 'reset')
    assert clock.cutoff(s) == -1 and s['status'] == 'ready'


def test_refuses_mid_pa_rows_gaps_verified_and_bad_lead(record):
    rows = table()
    mid = copy.deepcopy(rows)
    for r in mid:
        if r['index'] == '4':
            r.update(status='OBSERVED_RESULT_VISIBLE', video_offset_result_visible_s='440', verified_by='X', evidence='e')
    with pytest.raises(ValueError, match='not the last pitch'):
        make(record, mid)
    gap = [r for r in rows if r['index'] != '9']
    with pytest.raises(ValueError, match='not consecutive'):
        make(record, gap)
    with pytest.raises(ValueError, match='lead_s'):
        make(record, lead=500.0)
    m = make(record)
    with pytest.raises(ValueError):
        validate_mapping(m | {'verified': True})
    with pytest.raises(ValueError):
        validate_mapping(m | {'rows': [m['rows'][0] | {'review': {}}] + m['rows'][1:]})
    pending = [r for r in rows]
    for r in pending:
        r['status'] = 'PENDING_NOT_OBSERVED'
    with pytest.raises(ValueError, match='no observed rows'):
        make(record, pending)
