"""Generated GUMBO contracts; no captured body or model artifact is read."""
import copy

from pitcheezy.seq_shadow.captured_bridge import step
from pitcheezy.seq_shadow.fullseason_stream import StreamState


SHA = 'a' * 64


def feed():
    pitch = {'isPitch': True, 'pitchNumber': 1, 'playId': 'p1',
             'details': {'isInPlay': False, 'isOut': False, 'call': {'code': 'B'}},
             'count': {'balls': 1, 'strikes': 0, 'outs': 0}}
    play = {'about': {'atBatIndex': 0, 'isComplete': False, 'inning': 1, 'isTopInning': True},
            'count': {'balls': 1, 'strikes': 0, 'outs': 0},
            'matchup': {'pitcher': {'id': 10}, 'batter': {'id': 20},
                        'batSide': {'code': 'L'}, 'pitchHand': {'code': 'R'}},
            'result': {'homeScore': 0, 'awayScore': 0}, 'playEvents': [pitch]}
    return {'gameData': {'game': {'pk': 99}, 'datetime': {'officialDate': '2026-10-11'},
                         'status': {'abstractGameState': 'Live'}},
            'liveData': {'plays': {'allPlays': [play], 'currentPlay': play},
                         'linescore': {'currentInning': 1, 'isTopInning': True,
                                       'outs': 0, 'teams': {'home': {'runs': 0}, 'away': {'runs': 0}},
                                       'offense': {'first': {'id': 30}}}}}


def run(body, seq=1):
    return step(body, game_pk=99, game_date='2026-10-11', receipt_seq=seq,
                source_sha256=SHA, candidate_pitch_type='FF')


def test_open_next_pitch_is_hypothetical_and_observed_prior_stays_unknown():
    result = run(feed())
    assert result['abstain_reason'] is None
    assert result['events'] == [{'event_id': 'p1', 'key': (99, 1, 1), 'revision': 1,
                                  'row': {'status': 'unknown', 'reason': 'statcast_acceptance_unverified',
                                          'accepted': None}}]
    candidate = result['candidate']
    assert candidate['query']['pitch_number'] == 2
    assert candidate['query']['balls'] == 1
    assert candidate['query']['on_1b'] == 30
    assert candidate['query']['pitch_type'] == 'FF'
    assert candidate['pitch_type_basis'] == 'hypothetical_candidate'
    assert candidate['coverage'] == {'complete': True, 'before_key': [99, 1, 2], 'source_sha256': SHA}
    assert run(feed(), 2)['events'][0]['revision'] == 2


def test_missing_or_conflicting_poststate_abstains_without_beforestate_guess():
    base = feed()
    cases = [
        ('count_sources_disagree', lambda b: b['liveData']['plays']['currentPlay']['count'].update(balls=2)),
        ('last_pitch_ended_pa', lambda b: b['liveData']['plays']['currentPlay']['playEvents'][0]['details'].update(isInPlay=True)),
        ('pa_ended_or_completion_unknown', lambda b: b['liveData']['plays']['currentPlay']['about'].update(isComplete=True)),
        ('next_prestate_unverified', lambda b: b['liveData']['linescore'].update(currentInning=2)),
        ('runner_identity_unverified', lambda b: b['liveData']['linescore']['offense'].update(first={})),
        ('nonpitch_action_in_current_pa', lambda b: b['liveData']['plays']['currentPlay']['playEvents'].append({'isPitch': False, 'count': {'balls': 2}})),
        ('stored_prefix_unkeyed_or_conflicting', lambda b: b['liveData']['plays']['currentPlay']['playEvents'][0].update(pitchNumber=2)),
    ]
    for reason, mutate in cases:
        body = copy.deepcopy(base)
        mutate(body)
        result = run(body)
        assert result['candidate'] is None and result['abstain_reason'] == reason


def test_identity_and_missing_provider_key_fail_closed():
    body = feed()
    body['gameData']['datetime']['officialDate'] = '2026-10-12'
    assert run(body)['abstain_reason'] == 'official_date_unverified'
    body = feed()
    del body['liveData']['plays']['currentPlay']['playEvents'][0]['playId']
    result = run(body)
    assert result['abstain_reason'] == 'stored_prefix_unkeyed_or_conflicting'
    assert result['events'] == []
    body = feed()
    duplicate = copy.deepcopy(body['liveData']['plays']['currentPlay']['playEvents'][0])
    duplicate['pitchNumber'] = 2
    body['liveData']['plays']['currentPlay']['playEvents'].append(duplicate)
    assert run(body)['abstain_reason'] == 'provider_identity_conflict'
    assert run(body)['events'] == []


def test_unknown_receipt_then_verified_correction_preserves_first_decision(tmp_path):
    receipt = run(feed())
    stream = StreamState(tmp_path/'stream.sqlite', game_pk=99, game_date='2026-10-11',
                         normalizer_sha256=SHA, evidence_kind='captured')
    event = receipt['events'][0]
    first = stream.ingest(**event, received_ns=100)
    assert first['status'] == 'new' and first['first_received_ns'] == 100
    candidate = receipt['candidate']
    before = stream.decide(**candidate, received_ns=101)
    assert before['status'] == 'abstain' and before['reason'] == 'prior_quality_unknown'
    stream.ingest(event['event_id'], event['key'], 2,
                  {'status': 'accepted', 'reason': 'independently_verified',
                   'accepted': {'pitcher': 10, 'batter': 20, 'pitch_type': 'FF',
                                'outcome10': 0, 'balls': 0, 'strikes': 0}}, received_ns=102)
    assert stream.decide(**candidate, received_ns=103) == before
    later = dict(candidate, decision_id=candidate['decision_id']+':later')
    after = stream.decide(**later, received_ns=104)
    assert after['status'] == 'ready' and after['tokens']['valid'][0][0]
    stream.close()
