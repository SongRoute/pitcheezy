"""Synthetic cached-WE identity checks; no recorded feeds, models or inference."""
import copy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.live_armb import LiveArmB, state_key, _we_input_sha256
from observer_app.live_store import _canonical


PITCH = '1:1:1'


def snapshot():
    return {'gameData': {'game': {'pk': 1}}, 'liveData': {
        'plays': {'allPlays': [], 'currentPlay': {'about': {'atBatIndex': 0, 'isComplete': False},
                                               'count': {'balls': 0, 'strikes': 0}}},
        'linescore': {'currentInning': 1, 'inningState': 'Top', 'outs': 0, 'teams': {'home': {'runs': 0}}}}}


def result(value=.6):
    return {'key': PITCH, 'pitch': {'key': PITCH, 'actual': None, 'pre': {'status': 'ready'}},
            'pa_pitches': [], 'home_we_now': value, 'policy_identity': 'a' * 64}


def cached(feed, value=.6):
    def forbidden():
        raise AssertionError('cache-only read loaded the policy')
    armb = LiveArmB(forbidden)
    armb.cache[state_key(feed)] = {'result': result(value), 'snapshot_sha256': hashlib.sha256(_canonical(feed)).hexdigest(),
                                  'we_input_sha256': _we_input_sha256(feed)}
    return armb


@pytest.mark.parametrize('value', [0, .6, 1])
def test_exact_snapshot_and_pending_identity_are_required_without_loading(value):
    feed = snapshot()
    armb = cached(feed, value)
    sha = hashlib.sha256(_canonical(feed)).hexdigest()
    view = armb.win_expectancy(feed, pending_pitch_key=PITCH, snapshot_revision=sha)
    assert view['status'] == 'available' and view['home_probability'] == value
    assert view['perspective'] == 'home' and view['basis'] == 'count_independent_game_state'
    assert armb._thread is None and armb.policy is None and not armb.pending
    assert armb.win_expectancy(feed, pending_pitch_key='1:1:2')['status'] == 'unavailable'
    assert armb.win_expectancy(feed, pending_pitch_key=PITCH, snapshot_revision='b' * 64)['status'] == 'unavailable'
    corrected = copy.deepcopy(feed)
    corrected['liveData']['linescore']['teams']['home']['runs'] = 1
    assert state_key(feed) == state_key(corrected)  # the recommendation key alone is insufficient for WE
    rejected = armb.win_expectancy(corrected, pending_pitch_key=PITCH)
    assert rejected['status'] == 'unavailable' and rejected['home_probability'] is None


@pytest.mark.parametrize('value', [None, True, -.01, 1.01, float('nan'), float('inf'), '.6'])
def test_invalid_probabilities_are_never_public(value):
    feed = snapshot()
    assert cached(feed, value).win_expectancy(feed, pending_pitch_key=PITCH)['home_probability'] is None


def test_only_poll_timestamp_can_change_and_raw_receipt_must_still_match():
    feed = snapshot() | {'metaData': {'timeStamp': 'old', 'eventWall': 100}}
    armb = cached(feed)
    newer = copy.deepcopy(feed)
    newer['metaData']['timeStamp'] = 'new'
    raw_sha = hashlib.sha256(_canonical(newer)).hexdigest()
    assert armb.win_expectancy(newer, pending_pitch_key=PITCH, snapshot_revision=raw_sha)['status'] == 'available'
    old_sha = hashlib.sha256(_canonical(feed)).hexdigest()
    assert armb.win_expectancy(newer, pending_pitch_key=PITCH, snapshot_revision=old_sha)['status'] == 'unavailable'
    newer['metaData']['eventWall'] = 101
    assert armb.win_expectancy(newer, pending_pitch_key=PITCH)['status'] == 'unavailable'
    assert feed['metaData']['timeStamp'] == 'old'  # fingerprint never mutates the feed


@pytest.mark.parametrize('change', ['key', 'actual', 'unsupported', 'identity', 'missing_hash'])
def test_malformed_or_old_cached_results_fail_closed(change):
    feed = snapshot()
    armb = cached(feed)
    entry = armb.cache[state_key(feed)]
    if change == 'key':
        entry['result']['pitch']['key'] = '1:2:1'
    elif change == 'actual':
        entry['result']['pitch']['actual'] = {'pitch_type': 'FF'}
    elif change == 'unsupported':
        entry['result']['pitch']['pre']['status'] = 'unsupported'
    elif change == 'identity':
        entry['result']['policy_identity'] = ''
    else:
        del entry['snapshot_sha256']
    assert armb.win_expectancy(feed, pending_pitch_key=PITCH)['status'] == 'unavailable'


@pytest.mark.parametrize('mutate', [False, True])
def test_worker_records_only_the_snapshot_it_actually_computed(mutate):
    feed = snapshot()
    def policy(body):
        if mutate:
            body['revision'] = 2
        return result()
    armb = LiveArmB(lambda: policy)
    armb.lookup(feed)
    with armb._cond:
        assert armb._cond.wait_for(lambda: armb.computed == 1, timeout=2)
    view = armb.win_expectancy(feed, pending_pitch_key=PITCH)
    assert view['status'] == ('unavailable' if mutate else 'available')
