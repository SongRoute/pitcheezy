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
