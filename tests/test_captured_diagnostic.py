"""Generated field evidence and chronology only; no real capture or model reads."""
import copy

import pandas as pd

from pitcheezy.seq_shadow.captured_diagnostic import inspect_body, summarize
from experiments.ws_fullseason.data import closure
from test_captured_bridge import feed


def observed(body):
    return inspect_body(body, game_pk=99, game_date='2026-10-11')


def pitch(number, play_id, call, balls, strikes, outs=0):
    return {'isPitch': True, 'pitchNumber': number, 'playId': play_id,
            'details': {'type': {'code': 'FF'}, 'call': {'code': call},
                        'isInPlay': call == 'X'},
            'count': {'balls': balls, 'strikes': strikes, 'outs': outs}}


def test_nonbip_count_hypothesis_and_frozen_parent_label_mapping():
    body = feed()
    play = body['liveData']['plays']['currentPlay']
    play['playEvents'] = [pitch(1, 'p1', 'B', 1, 0), pitch(2, 'p2', 'S', 1, 1),
                          pitch(3, 'p3', 'L', 1, 2)]
    rows = observed(body)
    assert [r['pre_count'] for r in rows] == [(0, 0), (1, 0), (1, 1)]
    assert [r['outcome10_hypothesis'] for r in rows] == [0, 1, 2]
    assert all(r['native_nonbip_candidate'] and not r['accepted'] for r in rows)
    assert all('statcast_description' in r['proof_missing'] for r in rows)
    parent = closure().safe.outcome_labels(pd.DataFrame({
        'description': [r['description_hypothesis'] for r in rows],
        'events': ['', '', ''], 'strikes': [r['pre_count'][1] for r in rows]}))
    assert parent.tolist() == [r['outcome10_hypothesis'] for r in rows]


def test_automatic_count_action_and_missing_count_do_not_guess():
    body = feed()
    play = body['liveData']['plays']['currentPlay']
    play['playEvents'] = [pitch(1, 'p1', 'B', 1, 0),
                          {'isPitch': False, 'count': {'balls': 2, 'strikes': 0, 'outs': 0}},
                          pitch(2, 'p2', 'C', 2, 1)]
    assert observed(body)[1]['pre_count'] == (2, 0)
    del play['playEvents'][1]['count']
    assert observed(body)[1]['reason'] == 'before_count_unverified'
    del play['playEvents'][0]['count']
    assert observed(body)[1]['reason'] == 'before_count_unverified'


def test_bip_scoring_third_out_and_substitution_stay_unaccepted():
    body = feed()
    play = body['liveData']['plays']['currentPlay']
    play['playEvents'] = [pitch(1, 'p1', 'X', 0, 0, outs=3)]
    play['result']['homeScore'] = 1
    rows = observed(body)
    assert rows[0]['reason'] == 'bip_event_and_bbtype_unverified'
    assert rows[0]['pre_count'] == (0, 0)
    assert not rows[0]['accepted'] and 'before_outs_score' in rows[0]['proof_missing']
    play['playEvents'] = [pitch(1, 'p1', 'H', 0, 0),
                          {'isPitch': False, 'details': {'eventType': 'pitching_substitution'},
                           'count': {'balls': 0, 'strikes': 0, 'outs': 0}},
                          pitch(2, 'p2', 'B', 1, 0)]
    play['matchup']['pitcher']['id'] = 11  # final play matchup cannot label pitch 1.
    rows = observed(body)
    assert rows[0]['outcome10_hypothesis'] == 8
    assert all(not r['accepted'] and 'pitch_level_identity' in r['proof_missing'] for r in rows)


def test_first_seen_and_latest_distinct_correction_without_row_export():
    first = observed(feed())
    corrected_body = copy.deepcopy(feed())
    event = corrected_body['liveData']['plays']['currentPlay']['playEvents'][0]
    event['details']['call']['code'] = 'S'
    event['count'] = {'balls': 0, 'strikes': 1, 'outs': 0}
    latest = observed(corrected_body)
    summary = summarize([(1, first), (2, latest)])
    assert summary['first_seen']['pitches'] == summary['latest_seen']['pitches'] == 1
    assert summary['changed_keys'] == 1
    assert not any('key' in v for v in summary.values() if isinstance(v, dict))


def test_gap_or_malformed_call_never_recovers_to_an_accepted_row():
    body = feed()
    play = body['liveData']['plays']['currentPlay']
    play['playEvents'] = [pitch(2, 'p2', 'B', 1, 0), pitch(3, 'p3', 'C', 1, 1)]
    rows = observed(body)
    assert [r['reason'] for r in rows] == ['physical_key_or_prefix_unverified'] * 2
    play['playEvents'] = [pitch(1, 'p1', 'B', 1, 0)]
    play['playEvents'][0]['details']['call']['code'] = ['malformed']
    assert observed(body)[0]['reason'] == 'call_mapping_unavailable'
    assert not observed(body)[0]['accepted']
