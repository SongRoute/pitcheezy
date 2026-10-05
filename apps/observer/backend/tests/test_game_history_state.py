"""Raw-feed situation reconstruction (game_history.build without a watch dataset), on synthetic GUMBO plays.

Each pitch's ``situation_before`` must hold every earlier event of the game (non-pitch runner moves, outs,
runs, substitutions) and nothing from the pitch itself or later.
"""
from observer_app import game_history

PK = 900201


def pitch(n, call, balls, strikes, outs):
    return {'type': 'pitch', 'isPitch': True, 'pitchNumber': n,
            'details': {'call': {'code': call}, 'description': call, 'type': {'code': 'FF'}},
            'count': {'balls': balls, 'strikes': strikes, 'outs': outs}, 'pitchData': {'coordinates': {'pX': 0.0, 'pZ': 2.5}}}


def action(kind, balls, strikes, outs, **extra):
    return {'type': 'action', 'isPitch': False, 'details': {'eventType': kind, 'description': kind} | extra.pop('details', {}),
            'count': {'balls': balls, 'strikes': strikes, 'outs': outs}} | extra


def move(rid, index, start, end, out=False, kind='x'):
    return {'movement': {'originBase': start, 'start': start, 'end': end, 'isOut': out},
            'details': {'runner': {'id': rid}, 'playIndex': index, 'eventType': kind}, 'credits': []}


def pa(ab, inning, top, batter, pitcher, events, runners, *, event='field_out', away=0, home=0, outs=0, complete=True):
    return {'about': {'atBatIndex': ab-1, 'inning': inning, 'isTopInning': top, 'isComplete': complete},
            'matchup': {'batter': {'id': batter, 'fullName': f'B{batter}'}, 'pitcher': {'id': pitcher, 'fullName': f'P{pitcher}'},
                        'batSide': {'code': 'R'}, 'pitchHand': {'code': 'R'}},
            'result': {'eventType': event, 'event': event, 'description': f'pa {ab}', 'awayScore': away, 'homeScore': home},
            'count': {'outs': outs}, 'playEvents': events, 'runners': runners}


def feed_of(*plays, players=None):
    return {'gameData': {'game': {'pk': PK, 'type': 'R', 'doubleHeader': 'N', 'gameNumber': 1},
                         'datetime': {'dateTime': '2026-09-30T02:00:00Z', 'officialDate': '2026-09-29'},
                         'status': {'abstractGameState': 'Live', 'detailedState': 'In Progress'},
                         'teams': {'away': {'name': 'A', 'abbreviation': 'A'}, 'home': {'name': 'H', 'abbreviation': 'H'}},
                         'players': players or {}},
            'liveData': {'plays': {'allPlays': list(plays)}, 'linescore': {}}}


def build(*plays, players=None):
    record = game_history.build(feed_of(*plays, players=players))
    return {p['key']: p for p in record['pitches']}, record


def sit(pitches, ab, n):
    s = pitches[f'{PK}:{ab}:{n}']['situation_before']
    return (s['outs'], s['balls'], s['strikes'], s['bases'], s['away_score'], s['home_score'])


def single_to_first(ab, batter, outs=0, away=0, home=0):
    return pa(ab, 1, True, batter, 90, [pitch(1, 'X', 0, 0, outs)], [move(batter, 0, None, '1B', kind='single')],
              event='single', away=away, home=home, outs=outs)


def test_stolen_base_between_pitches_reaches_only_later_pitches():
    pitches, _ = build(
        single_to_first(1, 11),
        pa(2, 1, True, 12, 90, [pitch(1, 'B', 1, 0, 0), action('stolen_base_2b', 1, 0, 0), pitch(2, 'S', 1, 1, 0),
                                pitch(3, 'X', 1, 1, 1)],
           [move(11, 1, '1B', '2B', kind='stolen_base_2b'), move(12, 3, None, None, out=True)], outs=1))
    assert sit(pitches, 2, 1) == (0, 0, 0, 1, 0, 0)  # runner still on 1st: the steal is later
    assert sit(pitches, 2, 2) == (0, 1, 0, 2, 0, 0)  # steal applied before pitch 2, count carried from pitch 1
    assert pitches[f'{PK}:2:2']['situation_before']['runners'] == {'first': False, 'second': True, 'third': False}
    assert sit(pitches, 2, 3) == (0, 1, 1, 2, 0, 0)


def test_pickoff_and_caught_stealing_add_outs_and_clear_bases():
    pitches, _ = build(
        single_to_first(1, 11),
        pa(2, 1, True, 12, 90, [pitch(1, 'B', 1, 0, 0), {'type': 'pickoff', 'isPitch': False, 'details': {'description': 'Pickoff Attempt 1B'},
                                                         'count': {'balls': 1, 'strikes': 0, 'outs': 0}},
                                action('pickoff_1b', 1, 0, 1), pitch(2, 'X', 1, 0, 1)],
           [move(11, 2, '1B', None, out=True, kind='pickoff_1b'), move(12, 3, None, '1B', kind='single')], event='single', outs=1),
        pa(3, 1, True, 13, 90, [pitch(1, 'B', 1, 0, 1), action('caught_stealing_2b', 1, 0, 2), pitch(2, 'B', 2, 0, 2),
                                pitch(3, 'X', 2, 0, 3)],
           [move(12, 1, '1B', None, out=True, kind='caught_stealing_2b'), move(13, 3, None, None, out=True)], outs=3))
    assert sit(pitches, 2, 1) == (0, 0, 0, 1, 0, 0)
    assert sit(pitches, 2, 2) == (1, 1, 0, 0, 0, 0)  # picked off: one out, bases empty, count kept
    assert sit(pitches, 3, 1) == (1, 0, 0, 1, 0, 0)  # new PA: count reset, batter of PA 2 on 1st
    assert sit(pitches, 3, 2) == (2, 1, 0, 0, 0, 0)  # caught stealing


def test_inning_transition_resets_and_scores_carry():
    pitches, record = build(
        single_to_first(1, 11),
        pa(2, 1, True, 12, 90, [pitch(1, 'B', 1, 0, 0), action('wild_pitch', 1, 0, 0, details={'awayScore': 0, 'homeScore': 0}),
                                pitch(2, 'B', 2, 0, 0), action('caught_stealing_3b', 2, 0, 3)],
           [move(11, 1, '1B', '2B', kind='wild_pitch'), move(11, 3, '2B', None, out=True, kind='caught_stealing_3b')],
           complete=False, outs=3),
        # the same batter resumes in the next half-inning (feed starts a new play there)
        pa(3, 1, False, 31, 80, [pitch(1, 'B', 1, 0, 0), action('wild_pitch', 1, 0, 0), pitch(2, 'X', 1, 0, 1)],
           [move(32, 1, '3B', 'score', kind='wild_pitch'), move(31, 2, None, None, out=True)], away=0, home=1, outs=1))
    assert sit(pitches, 2, 2) == (0, 1, 0, 2, 0, 0)
    assert pitches[f'{PK}:3:1']['half'] == 'Bot'
    assert sit(pitches, 3, 1) == (0, 0, 0, 0, 0, 0)  # bottom half: no outs, nobody on, count fresh
    assert sit(pitches, 3, 2)[-1] == 1  # run scored on the wild pitch before pitch 2, not before pitch 1
    assert sit(pitches, 3, 1)[-1] == 0
    s = pitches[f'{PK}:3:1']['situation_before']
    assert (s['inning'], s['half']) == (1, 'Bot')
    assert record['pas'][1]['complete'] is False


def test_multi_segment_moves_in_one_event_do_not_overwrite():
    # runner on 1st, runner on 3rd; batter reaches on an error and takes 2nd on the throw, lead runner 1st -> 3rd
    pitches, _ = build(
        single_to_first(1, 11),
        pa(2, 1, True, 12, 90, [pitch(1, 'X', 0, 0, 0)],
           [move(12, 0, None, '1B', kind='single'), move(11, 0, '1B', '2B', kind='single'), move(11, 0, '2B', '3B', kind='error')], event='single'),
        pa(3, 1, True, 13, 90, [pitch(1, 'X', 0, 0, 0)],
           [move(13, 0, None, '1B', kind='field_error'), move(13, 0, '1B', '2B', kind='error'), move(11, 0, '3B', 'score', kind='error'),
            move(12, 0, '1B', '2B', kind='error'), move(12, 0, '2B', '3B', kind='error')], event='field_error', away=1),
        pa(4, 1, True, 14, 90, [pitch(1, 'B', 1, 0, 0)], [], complete=False))
    assert sit(pitches, 3, 1)[3] == 5  # 1st + 3rd
    assert sit(pitches, 4, 1)[3] == 6 and sit(pitches, 4, 1)[4] == 1  # 2nd + 3rd, one run


def test_pinch_runner_then_steal():
    pitches, _ = build(
        single_to_first(1, 11),
        pa(2, 1, True, 12, 90, [action('offensive_substitution', 0, 0, 0, position={'abbreviation': 'PR'}, player={'id': 15},
                                       replacedPlayer={'id': 11}),
                                pitch(1, 'B', 1, 0, 0), action('stolen_base_2b', 1, 0, 0), pitch(2, 'B', 2, 0, 0)],
           [move(15, 2, '1B', '2B', kind='stolen_base_2b')], complete=False))
    assert sit(pitches, 2, 1)[3] == 1
    assert sit(pitches, 2, 2)[3] == 2  # the pinch runner (not the replaced runner) moved; 1st is empty


def test_mid_pa_pitching_change_keeps_earlier_pitcher():
    players = {'ID90': {'fullName': 'Starter', 'pitchHand': {'code': 'L'}}}
    pitches, _ = build(pa(1, 1, True, 11, 91, [pitch(1, 'B', 1, 0, 0),
                                               action('pitching_substitution', 1, 0, 0, player={'id': 91}, replacedPlayer={'id': 90}),
                                               pitch(2, 'X', 1, 0, 1)],
                          [move(11, 2, None, None, out=True)], outs=1), players=players)
    assert pitches[f'{PK}:1:1']['pitcher'] == {'id': 90, 'name': 'Starter', 'hand': 'L'}
    assert pitches[f'{PK}:1:2']['pitcher']['id'] == 91


def test_watch_situation_is_not_used_over_the_feed():
    plays = [single_to_first(1, 11),
             pa(2, 1, True, 12, 90, [pitch(1, 'B', 1, 0, 0), action('stolen_base_2b', 1, 0, 0), pitch(2, 'B', 2, 0, 0)],
                [move(11, 1, '1B', '2B')], complete=False)]
    wrong = {'decisions': [{'pa_id': f'{PK}:2', 'pitch_number': 2, 'pre': {'status': 'unsupported', 'reason': 'r'},
                            'situation': {'inning': 1, 'half': 'Top', 'outs': 2, 'balls': 0, 'strikes': 0, 'bases': 7,
                                          'home_score': 9, 'away_score': 9}}]}
    p = {x['key']: x for x in game_history.build(feed_of(*plays), wrong)['pitches']}[f'{PK}:2:2']
    assert (p['situation_before']['outs'], p['situation_before']['bases'], p['situation_before']['balls']) == (0, 2, 1)
    assert p['rec']['status'] == 'unsupported'  # the stored pre-pitch rec is still attached
