"""Integration review (rev3, review-integration): two-way player, revealed-only pregame stats, and as-of linescore on all
16 preview games including 813031 (read-only overlay-014 feeds + pregame-data-032 cache). Skipped when those dirs are absent."""
import copy
import gzip
import json
from pathlib import Path

import pytest

from observer_app import game_history, player_metadata, service_api

JOB = Path('/Users/song/Projects/pitcheezy-pm-jobs/20261005-ui-live-redesign')
FEEDS, CACHE = JOB/'preview-data-overlay-014/overlay/feeds', JOB/'pregame-data-032/cache'
DEMO = {'kind': 'demo', 'session_id': 'x'}
OHTANI = '660271'
pytestmark = pytest.mark.skipif(not (FEEDS.is_dir() and CACHE.is_dir()), reason='overlay/cache not present')


def feed(pk):
    return json.load(gzip.open(FEEDS/f'{pk}_final.json.gz'))


def record(pk, f=None):
    f = f or feed(pk)
    rec = game_history.build(f)
    rec['pregame'] = player_metadata.load_pregame(CACHE, pk, rec['official_date'])
    return rec


def shown_ids(view, roles=('batter', 'pitcher')):
    """Everyone the view itself shows: pitch/PA/current holders, and (contract of 0fe66af9, 2026-10-13 UI connection)
    the as-of ``current.context`` lineups and box score. Those lists hold only the starting lineup and people an
    already revealed event brought in (substitutions); ``test_context_people_are_starters_or_revealed`` checks that
    against the feed, and ``test_813031_future_plays_removed_do_not_change_view`` checks no later play changes them."""
    ids = set()
    for holder in [view['current'], *view['pitches'], *view['pas']]:
        for role in roles:
            if (holder or {}).get(role):
                ids.add(str(holder[role]['id']))
    context = view['current'].get('context') or {}
    for team in (context.get('boxscore') or {}).values():
        ids |= {str(p['id']) for role in roles for p in team[role+'s']}
    if 'batter' in roles:
        ids |= {str(p['id']) for team in (context.get('lineups') or {}).values() for p in team}
    return ids


def all_games():
    return sorted(int(p.name.split('_')[0]) for p in FEEDS.iterdir() if p.name[0].isdigit() and p.name.endswith('_final.json.gz'))


def test_two_way_ohtani_gets_era_as_pitcher_and_avg_ops_once_he_bats():
    rec = record(813031)
    pitches = rec['pitches']
    first_pitching = next(i for i, p in enumerate(pitches) if str(p['pitcher']['id']) == OHTANI)
    first_batting = next(i for i, p in enumerate(pitches) if str(p['batter']['id']) == OHTANI)
    view = service_api.cut_view(rec, first_pitching, DEMO)
    o = view['pregame']['players'][OHTANI]
    assert o['era'] == '2.87'
    # batting stats exactly when the view shows him as a batter. Since the 2026-10-13 contract he is shown as one
    # before his first PA because he is in the as-of starting lineup (current.context.lineups).
    assert first_batting > first_pitching + 1
    assert OHTANI in {str(p['id']) for team in view['current']['context']['lineups'].values() for p in team}
    assert (('avg' in o) and ('ops' in o)) == (OHTANI in shown_ids(view, roles=('batter',)))
    later = service_api.cut_view(rec, first_batting, DEMO)['pregame']['players'][OHTANI]
    assert (later['era'], later['avg'], later['ops']) == ('2.87', '.282', '1.014')  # both roles kept, no first-role winner
    assert later['as_of'] == '2025-10-16' and view['pregame']['basis'] == 'regular_season_through_previous_day'


@pytest.mark.parametrize('pk', [849843, 813031])
def test_pregame_only_for_revealed_people_at_every_cutoff(pk):
    rec = record(pk)
    for cutoff in range(-1, len(rec['pitches'])):
        view = service_api.cut_view(rec, cutoff, DEMO)
        assert set(view['pregame']['players']) <= shown_ids(view), (pk, cutoff)


@pytest.mark.parametrize('pk', [849843, 813031])
def test_context_people_are_starters_or_revealed(pk):
    """The lineup/box-score people that widen ``shown_ids`` are never a future substitute: each is a starter
    (battingOrder N00 in the feed) or a player some event at or before the next pitch already names."""
    f = feed(pk)
    rec = record(pk, f)
    starters = {str((p.get('person') or {}).get('id')) for side in ('away', 'home')
                for p in f['liveData']['boxscore']['teams'][side]['players'].values()
                if str(p.get('battingOrder', '')).endswith('00')}
    plays = f['liveData']['plays']['allPlays']
    for cutoff in range(-1, len(rec['pitches'])):
        view = service_api.cut_view(rec, cutoff, DEMO)
        context = view['current'].get('context') or {}
        people = {str(p['id']) for team in (context.get('boxscore') or {}).values()
                  for role in ('batters', 'pitchers') for p in team[role]}
        people |= {str(p['id']) for team in (context.get('lineups') or {}).values() for p in team}
        upto = rec['pitches'][cutoff+1]['at_bat_number'] if cutoff+1 < len(rec['pitches']) else len(plays)
        named = set()
        for play in plays[:upto]:
            named |= {str(play['matchup'][r]['id']) for r in ('batter', 'pitcher')}
            named |= {str(e['player']['id']) for e in play.get('playEvents', []) if (e.get('player') or {}).get('id')}
        assert people <= starters | named, (pk, cutoff, people - starters - named)


def test_sixteen_games_present():
    assert len(all_games()) == 16 and 813031 in all_games()


@pytest.mark.parametrize('pk', all_games() if FEEDS.is_dir() else [])
def test_linescore_every_cutoff_matches_score_and_final(pk):
    f = feed(pk)
    rec = game_history.build(f)
    for cutoff in range(-1, len(rec['pitches'])):
        view = service_api.cut_view(rec, cutoff, DEMO)
        ls, sit = view['linescore'], view['current']['situation']
        if ls is None:
            continue
        checked = True
        assert (ls['teams']['away']['runs'], ls['teams']['home']['runs']) == (sit['away_score'], sit['home_score']), (pk, cutoff)
    final = service_api.cut_view(rec, len(rec['pitches'])-1, {'kind': 'archive', 'session_id': None})['linescore']
    line = f['liveData']['linescore']['teams']
    for side in ('away', 'home'):
        assert (final['teams'][side]['runs'], final['teams'][side]['hits'], final['teams'][side]['errors']) == \
            (line[side]['runs'], line[side]['hits'], line[side]['errors']), (pk, side)


def test_813031_future_plays_removed_do_not_change_view():
    f = feed(813031)
    full = record(813031, f)
    plays = f['liveData']['plays']['allPlays']
    for cutoff in (0, 5, 15, 40, 120, 200):
        nxt = full['pitches'][cutoff+1]
        ab = nxt['at_bat_number']
        cut = copy.deepcopy(f)
        cut['liveData']['plays']['allPlays'] = plays[:ab]  # keep through the next pitch's PA, drop everything later
        trimmed = record(813031, cut)
        a, b = service_api.cut_view(full, cutoff, DEMO), service_api.cut_view(trimmed, cutoff, DEMO)
        assert a['linescore'] == b['linescore'] and a['current'] == b['current'] and a['pregame'] == b['pregame'], cutoff
