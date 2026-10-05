"""Selected-pitch box score (ui-calm-polish-1006, D2): every visible pitch carries `linescore_before`, the server's as-of
linescore just before that pitch, so a pinned past pitch never shows later runs/hits/errors. Read-only overlay-014 feeds;
skipped when absent."""
import copy
import gzip
import json
from pathlib import Path

import pytest

from observer_app import game_history, service_api

FEEDS = Path('/Users/song/Projects/pitcheezy-pm-jobs/20261005-ui-live-redesign/preview-data-overlay-014/overlay/feeds')
DEMO = {'kind': 'demo', 'session_id': 'x'}
ARCHIVE = {'kind': 'archive', 'session_id': None}
pytestmark = pytest.mark.skipif(not FEEDS.is_dir(), reason='overlay feeds not present')


def feed(pk):
    return json.load(gzip.open(FEEDS/f'{pk}_final.json.gz'))


@pytest.mark.parametrize('pk', [849843, 813031])
def test_each_pitch_line_matches_its_pre_pitch_score_and_internals_stay_hidden(pk):
    rec = game_history.build(feed(pk))
    view = service_api.cut_view(rec, len(rec['pitches']) - 1, ARCHIVE)
    for p in view['pitches']:
        assert not set(service_api.INTERNAL) & set(p), p['index']
        ls, sit = p['linescore_before'], p['situation_before']
        if ls is None:  # first pitch of the game: nothing has happened yet
            assert p['index'] == 0
            continue
        assert (ls['teams']['away']['runs'], ls['teams']['home']['runs']) == (sit['away_score'], sit['home_score']), p['index']
        cur = ls['current']
        assert (cur['inning'], cur['half']) == (p['inning'], p['half']), p['index']


def test_849843_first_inning_pitch_does_not_show_later_scoring():
    # 1회초 3구 (index 2): SD scores 2 in 1회말 and 8 in all; none of that may appear in the selected pitch's box score
    rec = game_history.build(feed(849843))
    view = service_api.cut_view(rec, len(rec['pitches']) - 1, ARCHIVE)
    final, p = view['linescore'], next(x for x in view['pitches'] if x['index'] == 2)
    assert final['teams']['home']['runs'] == 8 and final['innings'][0]['home'] == 2
    ls = p['linescore_before']
    assert ls['teams']['home'] == {'runs': 0, 'hits': 0, 'errors': 0, 'bb_hbp': 0}
    assert ls['innings'][0] == {'inning': 1, 'away': 0, 'home': None} and all(r['home'] is None for r in ls['innings'])


def test_813031_removing_future_plays_does_not_change_selected_lines():
    f = feed(813031)
    full = game_history.build(f)
    plays = f['liveData']['plays']['allPlays']
    for idx in (1, 6, 20, 41, 121, 201):
        ab = full['pitches'][idx]['at_bat_number']
        cut = copy.deepcopy(f)
        cut['liveData']['plays']['allPlays'] = plays[:ab]  # through the selected pitch's PA; every later play dropped
        trimmed = game_history.build(cut)
        a = service_api.cut_view(full, idx, DEMO)['pitches']
        b = service_api.cut_view(trimmed, idx, DEMO)['pitches']
        assert [p['linescore_before'] for p in a] == [p['linescore_before'] for p in b], idx
