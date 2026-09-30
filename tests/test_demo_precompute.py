"""Watch-along demo precompute: GUMBO -> Statcast mapping and display pieces on a synthetic game."""
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'apps/observer/backend')]
import demo_precompute as demo  # noqa: E402


def pitch(index, code, balls, strikes, pitch_type='FF', **breaks):
    return {'index': index, 'isPitch': True, 'type': 'pitch', 'pitchNumber': index + 1,
            'details': {'call': {'code': code}, 'type': {'code': pitch_type, 'description': pitch_type}},
            'count': {'balls': balls, 'strikes': strikes, 'outs': 0},
            'pitchData': {'startSpeed': 95.0, 'extension': 6.5, 'coordinates': {'pX': .1, 'pZ': 2.5},
                          'breaks': {'spinRate': 2300, 'spinDirection': 210, 'breakHorizontal': breaks.get('h', 12.0),
                                     'breakVerticalInduced': breaks.get('v', 18.0)}}}


def action(index, event_type, balls=0, strikes=0, player=None):
    return {'index': index, 'isPitch': False, 'type': 'action', 'player': player and {'id': player},
            'details': {'eventType': event_type}, 'count': {'balls': balls, 'strikes': strikes, 'outs': 0}}


def run(index, runner, start, end, out=False):
    return {'movement': {'start': start, 'end': end, 'isOut': out}, 'details': {'playIndex': index, 'runner': {'id': runner}}}


def play(ab, events, runners, result, pitcher=10, batter=20):
    return {'about': {'atBatIndex': ab, 'inning': 1, 'isTopInning': True, 'isComplete': True},
            'matchup': {'pitcher': {'id': pitcher}, 'pitchHand': {'code': 'R'}, 'batter': {'id': batter}, 'batSide': {'code': 'L'}},
            'playEvents': events, 'runners': runners, 'result': {'eventType': result, 'description': result}}


FEED = {'gameData': {'game': {'pk': 1, 'type': 'F'}, 'datetime': {'officialDate': '2026-09-29'},
                     'players': {'ID10': {'pitchHand': {'code': 'R'}}, 'ID11': {'pitchHand': {'code': 'L'}}}},
        'liveData': {'plays': {'allPlays': [
            # automatic ball (no_pitch) is numbered like Statcast; single puts the batter on first
            play(0, [pitch(0, 'B', 1, 0), {'index': 1, 'isPitch': False, 'type': 'no_pitch',
                                           'details': {'call': {'code': 'VP'}}, 'count': {'balls': 2, 'strikes': 0}},
                     pitch(2, 'D', 2, 0)], [run(2, 20, None, '1B')], 'single'),
            # pinch runner 22 replaces 20 without a movement, steals 2B; mid-PA pitching change 10 -> 11; HR
            play(1, [action(0, 'offensive_substitution', player=22), pitch(1, 'C', 0, 1),
                     action(2, 'stolen_base_2b', 0, 1), action(3, 'pitching_substitution', 0, 1, player=11),
                     pitch(4, 'E', 0, 1)],
                 [run(2, 22, '1B', '2B'), run(4, 22, '2B', 'score'), run(4, 21, None, 'score')], 'home_run',
                 pitcher=11, batter=21),
            # batter doubles and takes third on the error (segments listed per runner)
            play(2, [pitch(0, 'D', 0, 0)], [run(0, 23, None, '2B'), run(0, 23, '2B', '3B')], 'double', pitcher=11, batter=23),
            play(3, [pitch(0, 'X', 0, 0)], [run(0, 24, None, None, out=True)], 'field_out', pitcher=11, batter=24)]}}}


def test_gumbo_rows_follow_statcast_conventions():
    rows = demo.statcast_rows(FEED).set_index(['at_bat_number', 'pitch_number'])
    assert list(rows.loc[1].description) == ['ball', 'automatic_ball', 'hit_into_play']
    assert list(rows.loc[1].balls) == [0, 1, 2] and rows.loc[(1, 3), 'events'] == 'single'
    assert np.isnan(rows.loc[(1, 2), 'release_speed']) and demo.pd.isna(rows.loc[(1, 2), 'pitch_type'])
    first, second = rows.loc[(2, 1)], rows.loc[(2, 2)]
    assert (first.pitcher, first.p_throws, second.pitcher, second.p_throws) == (10, 'R', 11, 'R')
    assert first.bases == 1 and second.bases == 2  # pinch runner is tracked by base, steal before pitch 2
    assert (first.post_away_score, second.away_score, second.post_away_score) == (0, 0, 2)
    assert rows.loc[(3, 1), 'bases'] == 0 and rows.loc[(4, 1), 'bases'] == 4  # chained segments end on third
    assert rows.loc[(4, 1), 'outs_when_up'] == 0 and rows.loc[(4, 1), 'events'] == 'field_out'
    assert rows.loc[(1, 1), 'pfx_x'] == pytest.approx(-1.0) and rows.loc[(1, 1), 'pfx_z'] == pytest.approx(1.5)
    assert rows.loc[(1, 1), 'effective_speed'] == 95.0  # release-speed proxy (documented)
    assert np.isnan(demo.statcast_rows(FEED, effective_speed='missing').effective_speed).all()
    assert demo._statcast_event('caught_stealing_2b') == 'truncated_pa'


def test_location_proxy_uses_the_observer_support_rule():
    bounds = {'bottom': 1.6, 'top': 3.4}
    rng = np.random.default_rng(0)
    near = np.broadcast_to(np.array([-.55, 1.9]) + rng.normal(0, .15, (400, 2)), (4, 3, 400, 2))
    zone = demo.location_proxy(near, 1, 2, bounds, sigma=.45, minimum_draws=20, minimum_mass=.01)
    assert zone['zone_id'] == 'low_left' and zone['kernel_ess'] >= 20
    far = np.full((4, 3, 400, 2), 9.)
    assert demo.location_proxy(far, 0, 0, bounds, sigma=.45, minimum_draws=20, minimum_mass=.01) is None


def test_recommendation_orders_types_first_and_keeps_numbers_secondary():
    out = demo.recommendation({'candidate': [.2, 0., .5, .3], 'reference': [.25, 0., .45, .3]}, ['CH', 'CU', 'FF', 'SL'],
                              lambda t: None if t == 'SL' else {'zone_id': 'middle_middle', 'zone_label': 'x',
                                                                'target': {'x': 0, 'z': 2.5}, 'kernel_mass': .3, 'kernel_ess': 99.})
    assert [c['pitch_type'] for c in out['candidates']] == ['FF', 'SL', 'CH']
    assert out['candidates'][1]['zone_id'] is None and out['candidates'][0]['detail']['reference_probability'] == .45
    assert 'CU' not in out['candidate_law']


def test_writes_only_inside_the_demo_directory(tmp_path):
    assert demo._check_output_root(tmp_path) == tmp_path.resolve()
    with pytest.raises(RuntimeError):
        demo._check_output_root('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/ML-POLICY-VAL-v1/x')


STATE = ['pitcher', 'p_throws', 'batter', 'stand', 'inning', 'inning_topbot', 'balls', 'strikes', 'outs_when_up',
         'bases', 'on_1b', 'on_2b', 'on_3b', 'home_score', 'away_score', 'at_bat_number', 'pitch_number']


def live(plays, **linescore):
    import copy
    feed = copy.deepcopy(FEED)
    feed['gameData']['status'] = {'abstractGameState': 'Live'}
    feed['gameData']['players'].update({'ID23': {'batSide': {'code': 'S'}}})
    feed['liveData']['plays']['allPlays'] = copy.deepcopy(plays)
    feed['liveData']['linescore'] = linescore
    return feed


def test_live_placeholder_row_has_the_completed_games_pre_pitch_state():
    import copy
    final = demo.statcast_rows(FEED).set_index(['at_bat_number', 'pitch_number'], drop=False)
    # mid-PA: right before the home run pitch (after the steal and the pitching change)
    second = copy.deepcopy(FEED['liveData']['plays']['allPlays'][1])
    second['playEvents'] = second['playEvents'][:4]
    second['runners'] = [r for r in second['runners'] if r['details']['playIndex'] < 4]
    second['about']['isComplete'] = False
    rows = demo.live_rows(live([FEED['liveData']['plays']['allPlays'][0], second]))
    got = rows.iloc[-1]
    assert got.placeholder and not rows.placeholder.iloc[:-1].any() and demo.pd.isna(got.pitch_type)
    assert got[STATE].astype(str).tolist() == final.loc[(2, 2), STATE].astype(str).tolist()
    # between batters: the due-up batter/pitcher come from the linescore; a switch hitter bats opposite
    rows = demo.live_rows(live(FEED['liveData']['plays']['allPlays'][:2], outs=0, inningState='Top', currentInning=1,
                               isTopInning=True, offense={'batter': {'id': 23}}, defense={'pitcher': {'id': 11}}))
    got = rows.iloc[-1]
    same = [c for c in STATE if c not in ('stand', 'p_throws')]  # the synthetic FEED's matchup hand is always R
    assert got[same].astype(str).tolist() == final.loc[(3, 1), same].astype(str).tolist()
    assert (got.p_throws, got.stand) == ('L', 'R')  # pitcher 11 throws L in gameData.players
    assert demo.live_rows(live(FEED['liveData']['plays']['allPlays'][:2], outs=3, offense={}, defense={})) is None
    not_live = live(FEED['liveData']['plays']['allPlays'][:2])
    not_live['gameData']['status'] = {'abstractGameState': 'Final'}
    assert demo.live_rows(not_live) is None


def test_sync_precomputes_only_new_completed_postseason_games(tmp_path):
    (tmp_path / 'watch').mkdir()
    (tmp_path / 'watch' / '2.json').write_text('{}')
    game = lambda pk, kind, state, detail: {'gamePk': pk, 'gameType': kind,
                                            'status': {'abstractGameState': state, 'detailedState': detail}}
    schedule = {'dates': [{'games': [game(1, 'F', 'Final', 'Final'), game(2, 'F', 'Final', 'Final'),
                                     game(3, 'F', 'Live', 'In Progress'), game(4, 'R', 'Final', 'Final'),
                                     game(5, 'D', 'Final', 'Postponed'), game(6, 'D', 'Final', 'Game Over'),
                                     game(7, 'L', 'Final', 'Final')]}]}
    urls = []
    fetch = lambda url: urls.append(url) or schedule

    def run(pk):
        if pk == 7:
            raise RuntimeError('feed unavailable')
        return {'coverage': {'ready_share_of_pitches': .5}}
    report = demo.sync('2026-09-29', '2026-10-01', tmp_path, {}, fetch=fetch, run=run)
    assert 'startDate=2026-09-29&endDate=2026-10-01&gameType=F,D,L,W' in urls[0]
    assert [d['game_pk'] for d in report['done']] == [1, 6] and report['present'] == [2]
    assert list(report['failed']) == ['7']


def test_rehearsal_snapshots_replay_every_pitch_with_its_pre_pitch_state():
    import copy
    final = copy.deepcopy(FEED)
    final['gameData']['status'] = {'abstractGameState': 'Final'}
    rows = demo.statcast_rows(final)
    pitched = rows.loc[rows.release_speed.notna()].reset_index(drop=True)  # the automatic ball is not a pitch event
    snaps = list(demo.replay_snapshots(final))
    assert len(snaps) == len(pitched)
    for (n, feed), (_, want) in zip(snaps, pitched.iterrows()):
        got = demo.live_rows(feed).iloc[-1]
        assert got[STATE].astype(str).tolist() == want[STATE].astype(str).tolist()
        line = feed['liveData']['linescore']
        assert (line['outs'], line['teams']['away']['runs']) == (want.outs_when_up, want.away_score)


WATCH = demo.DEMO_ROOT / 'watch/849843.json'


def _rss_mb():
    import os
    import subprocess
    return int(subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True).stdout) / 1024


@pytest.mark.skipif(not WATCH.exists(), reason='needs T7 (frozen policy and the 849843 watch-along dataset)')
def test_live_policy_reuses_one_binding_and_equals_the_precomputed_game():
    """Delayed-live recommendations over 50 rehearsal states of 849843 are bit-identical to the watch-along
    precompute (pending pitch, the PA's earlier pitches, WE), bind once, and do not grow the process."""
    import gc
    import json
    decisions = {f"849843:{d['at_bat_number']}:{d['pitch_number']}": d for d in json.loads(WATCH.read_text())['decisions']}
    final = demo.read_feed(demo.DEMO_ROOT / 'feeds/849843_final.json.gz')
    policy = demo.LivePolicy()
    rss, compared = [], 0
    for n, feed in demo.replay_snapshots(final, start=30, count=50):
        result = policy(feed)
        for pitch in [*result['pa_pitches'], result['pitch']]:
            assert pitch['pre'] == decisions[pitch['key']]['pre'], (n, pitch['key'])
            compared += 1
        assert result['home_we_now'] == decisions[result['key']]['we']['home_before']
        assert result['previous']['actual'] == decisions[result['previous']['key']]['actual']
        gc.collect()
        rss.append(_rss_mb())
    assert compared >= 50 and policy.binds == 1
    assert (rss[-1] - rss[4]) / (len(rss) - 5) < 0.5, rss
    # A bound row that changed before the pending pitch (e.g. a runner moved) is never reused: rebind.
    frame, _ = demo.demo_frame(demo.live_rows(feed), policy.snapshot, policy.as_of)
    changed = frame.copy()
    changed.loc[changed.index[-1], 'outs_when_up'] = (int(frame.outs_when_up.iloc[-1]) + 1) % 3
    assert policy._extend(frame) and not policy._extend(changed)
