"""Generated-only native token contract, separate from strict accepted rows."""
import copy

import pytest

from pitcheezy.seq_shadow.captured_native import CONTRACT, TOKEN_FIELDS, inspect_tokens
from pitcheezy.seq_shadow.fullseason_stream import StreamState
from test_captured_bridge import feed

SHA = 'a' * 64


def tokens(body):
    return inspect_tokens(body, game_pk=99, game_date='2026-10-11',
                          receipt_seq=1, source_sha256=SHA)


def test_six_token_fields_are_source_native_and_not_strict_accepted(tmp_path):
    body = feed()
    body['liveData']['plays']['currentPlay']['playEvents'][0]['details']['type'] = {'code': 'FF'}
    row = tokens(body)[0]
    assert row['input_contract'] == CONTRACT
    assert row['status'] == 'native_candidate' and not row['accepted']
    assert tuple(row['native_token']) == TOKEN_FIELDS
    assert row['native_token'] == {'pitcher': 10, 'batter': 20, 'pitch_type': 'FF',
                                   'outcome10': 0, 'balls': 0, 'strikes': 0}
    assert not row['statcast_training_equivalent']
    strict = StreamState(tmp_path/'strict.sqlite', game_pk=99, game_date='2026-10-11',
                         normalizer_sha256=SHA, evidence_kind='captured')
    with pytest.raises(ValueError, match='exact normalized row schema'):
        strict.ingest('p1', row['key'], 1, row, received_ns=1)
    strict.close()


def test_missing_type_or_outcome_or_matchup_stays_native_unknown():
    body = feed()
    assert tokens(body)[0]['reason'] == 'direct_type_unavailable'
    pitch = body['liveData']['plays']['currentPlay']['playEvents'][0]
    pitch['details']['type'] = {'code': 'FF'}
    pitch['details']['call']['code'] = ['B']
    assert tokens(body)[0]['reason'] == 'source_outcome_unverified'
    pitch['details']['call']['code'] = 'B'
    body['liveData']['plays']['currentPlay']['playEvents'].insert(0, {'isPitch': False})
    assert tokens(body)[0]['reason'] == 'pitch_time_matchup_unverified'


def test_bip_token_uses_event_result_without_requiring_bb_type():
    body = feed()
    play = body['liveData']['plays']['currentPlay']
    play['about']['isComplete'] = True
    play['result']['eventType'] = 'single'
    pitch = play['playEvents'][0]
    pitch['details'].update(type={'code': 'FF'}, call={'code': 'X'}, isInPlay=True)
    row = tokens(body)[0]
    assert row['status'] == 'native_candidate'
    assert row['native_token']['outcome10'] == 4
    assert row['statcast_training_equivalent'] is False
    play['result']['eventType'] = 'field_error'
    assert tokens(body)[0]['reason'] == 'source_outcome_unverified'


def test_later_nonpitch_action_reclassifies_new_receipt_not_prior_row():
    body = feed()
    body['liveData']['plays']['currentPlay']['playEvents'][0]['details']['type'] = {'code': 'FF'}
    first = tokens(body)[0]
    later = copy.deepcopy(body)
    later['liveData']['plays']['currentPlay']['playEvents'].append(
        {'isPitch': False, 'details': {'eventType': 'pitching_substitution'}})
    later['liveData']['plays']['currentPlay']['matchup']['pitcher']['id'] = 11
    second = tokens(later)[0]
    assert first['status'] == 'native_candidate' and first['native_token']['pitcher'] == 10
    assert second['status'] == 'native_unknown' and second['native_token'] is None
    assert first['native_token']['pitcher'] == 10


def test_bad_receipt_binding_refuses_without_defaults():
    with pytest.raises(ValueError, match='receipt identity'):
        inspect_tokens(feed(), game_pk=99, game_date='2026-10-11', receipt_seq=0, source_sha256=SHA)
