"""The watch-backed stand-in policy (``observer_app.watch_policy``) on the SYNTHETIC game of
``tests/test_demo_precompute.py`` (no T7, model or port): it answers each pending pitch with the stored ``pre`` in the
shape ``demo_precompute.LivePolicy`` returns, fails closed on a missing or mismatched pitch, and is switched on only
by ``PITCHEEZY_OBSERVER_LIVE_WATCH_POLICY``."""
import copy
import json
from pathlib import Path
import sys
import time

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parents[2]
sys.path[:0] = [str(BACKEND), str(REPO/'scripts'), str(REPO/'tests')]
import demo_precompute as dp  # noqa: E402
from observer_app import live_armb, watch_policy  # noqa: E402
from observer_app.watch_policy import WatchPolicy, WatchUnusable  # noqa: E402
from test_demo_precompute import FEED  # noqa: E402

IDENTITY = 'c'*64
GAME = FEED['gameData']['game']['pk']
# demo_precompute.LivePolicy.__call__'s result and pitch entries (main.armb_view reads exactly these)
RESULT_KEYS = {'key', 'pitch', 'pa_pitches', 'previous', 'home_we_now', 'policy_identity', 'seconds'}
PITCH_KEYS = {'key', 'pitch_number', 'status', 'pre', 'actual'}
PREVIOUS_KEYS = {'key', 'actual', 'inning', 'half', 'home_we_before'}


def final():
    game = copy.deepcopy(FEED)
    game['gameData']['status'] = {'abstractGameState': 'Final'}
    return game


def snapshots():
    return [feed for _, feed in dp.replay_snapshots(final())]


def candidates(top):
    return {'candidates': [{'rank': 1, 'pitch_type': top, 'pitch_label': top, 'zone_id': 'low_middle', 'zone_label': '낮은 중앙',
                            'target': {'x': 0.0, 'z': 1.9}, 'detail': {'probability': .5, 'reference_probability': .4,
                                                                       'kernel_mass': .3, 'kernel_ess': 50.0}}],
            'candidate_law': {top: .5}, 'reference_law': {top: .4}}


def dataset(tmp_path, *, refuse=(), drop=(), mid_pa=(), name='watch.json'):
    """A watch dataset with one decision per row of the synthetic game (``statcast_rows`` of the final feed, the
    precompute's rows). ``refuse``: keys stored as
    UNKNOWN_PITCHER refusals; ``mid_pa``: keys stored as a sticky MID_PA refusal whose root is UNKNOWN_PITCHER."""
    decisions = []
    for _, row in dp.statcast_rows(final()).iterrows():  # every row, as the precompute (the automatic ball too)
        key = dp._key(row)
        if key in drop:
            continue
        if key in refuse or key in mid_pa:
            status = 'UNSUPPORTED_MID_PA' if key in mid_pa else 'UNSUPPORTED_UNKNOWN_PITCHER'
            pre = {'status': 'unsupported', 'reason': dp.STATUS_TEXT['UNSUPPORTED_UNKNOWN_PITCHER'], 'recommendation': None}
            root = 'UNSUPPORTED_UNKNOWN_PITCHER'
        else:
            status, root, pre = 'SUPPORTED', 'SUPPORTED', {'status': 'ready', 'reason': None, 'recommendation': candidates(f'T{len(decisions)}')}
        decisions.append({'index': len(decisions), 'pa_id': f"{GAME}:{int(row['at_bat_number'])}",
                          'at_bat_number': int(row['at_bat_number']), 'pitch_number': int(row['pitch_number']),
                          'situation': {'inning': int(row['inning']), 'half': row['inning_topbot'], 'outs': int(row['outs_when_up']),
                                        'balls': int(row['balls']), 'strikes': int(row['strikes']), 'bases': int(row['bases']),
                                        'home_score': int(row['home_score']), 'away_score': int(row['away_score'])},
                          'pitcher': {'id': int(row['pitcher']), 'name': None, 'hand': row['p_throws']},
                          'batter': {'id': int(row['batter']), 'name': None, 'side': row['stand']},
                          'status': status, 'root_status': root, 'path': 'ledger', 'pre': pre, 'actual': None,
                          'we': {'home_before': .5 + len(decisions)/100, 'home_after': None, 'home_delta': None, 'batting_delta': None}})
    path = tmp_path/name
    path.write_text(json.dumps({'schema': watch_policy.SCHEMA, 'game': {'game_pk': GAME}, 'policy': {'identity_sha256': IDENTITY},
                                'decisions': decisions}))
    return path, {f"{GAME}:{d['at_bat_number']}:{d['pitch_number']}": d for d in decisions}


def test_answers_every_pitch_with_the_stored_pre_in_the_live_policy_shape(tmp_path):
    path, stored = dataset(tmp_path)
    policy = WatchPolicy(path)
    feeds = snapshots()
    assert len(stored) > len(feeds) >= 5  # the automatic ball has a decision but is never a pending pitch
    for feed in feeds:
        result = policy(feed)
        assert set(result) == RESULT_KEYS and result['policy_identity'] == IDENTITY
        assert result['seconds'] == {'bind': 0.0, 'total': 0.0}
        rows = dp.live_rows(feed).to_dict('records')
        assert result['key'] == dp._key(rows[-1])
        for pitch in [*result['pa_pitches'], result['pitch']]:
            assert set(pitch) == PITCH_KEYS
            assert pitch['pre'] == stored[pitch['key']]['pre'] and pitch['status'] == 'SUPPORTED'
        assert result['pitch']['actual'] is None  # the pending pitch is never revealed
        assert [p['key'] for p in result['pa_pitches']] == [dp._key(r) for r in rows[:-1]
                                                            if r['at_bat_number'] == rows[-1]['at_bat_number']]
        for pitch, row in zip(result['pa_pitches'], [r for r in rows if r['at_bat_number'] == rows[-1]['at_bat_number']]):
            assert pitch['actual'] == dp.actual_view(row, policy.bounds)  # the live policy's own view of the thrown pitch
        assert result['home_we_now'] == stored[result['key']]['we']['home_before']
        if len(rows) > 1:
            assert set(result['previous']) == PREVIOUS_KEYS
            assert result['previous']['home_we_before'] == stored[result['previous']['key']]['we']['home_before']
            assert result['previous']['actual'] == dp.actual_view(rows[-2], policy.bounds)
        else:
            assert result['previous'] is None
    assert policy.counters == {'SUPPORTED': len(feeds)} and policy.binds == 0


def test_a_missing_or_mismatched_pitch_is_refused_never_another_pitchs(tmp_path):
    feeds = snapshots()
    keys = [dp._key(dp.live_rows(feed).iloc[-1]) for feed in feeds]
    path, stored = dataset(tmp_path, drop={keys[2]})
    policy = WatchPolicy(path)
    missing = policy(feeds[2])
    # the shape of LivePolicy's refusal (pre_view of a refused status): unsupported, a sentence, no recommendation
    assert missing['pitch']['pre'] == {'status': 'unsupported', 'reason': watch_policy.NO_ROW_REASON, 'recommendation': None}
    assert missing['pitch']['status'] == watch_policy.NO_ROW_STATUS and missing['home_we_now'] is None
    assert set(missing) == RESULT_KEYS and set(missing['pitch']) == PITCH_KEYS
    # a stored decision whose situation is not this pitch's (another count) is never shown under its name
    path, _ = dataset(tmp_path, name='whole.json')
    body = json.loads(path.read_text())
    at = [f"{GAME}:{d['at_bat_number']}:{d['pitch_number']}" for d in body['decisions']].index(keys[3])
    body['decisions'][at]['situation']['balls'] += 1
    path.write_text(json.dumps(body))
    differs = WatchPolicy(path)(feeds[3])
    assert differs['pitch']['status'] == watch_policy.DIFFERS_STATUS and differs['pitch']['pre']['recommendation'] is None
    assert differs['home_we_now'] is None


def test_refusals_keep_their_status_and_a_mid_at_bat_pitcher_change_says_mid_pa(tmp_path):
    feeds = snapshots()
    rows = [dp.live_rows(feed).iloc[-1] for feed in feeds]
    keys = [dp._key(row) for row in rows]
    # the synthetic game's second at-bat changes pitcher 10 -> 11 between its two pitches
    first, second = [k for k, row in zip(keys, rows) if int(row['at_bat_number']) == 2]
    assert int(rows[keys.index(first)]['pitcher']) != int(rows[keys.index(second)]['pitcher'])
    path, _ = dataset(tmp_path, refuse={first, keys[0]}, mid_pa={second})
    policy = WatchPolicy(path)
    refused = policy(feeds[0])['pitch']
    assert refused['status'] == 'UNSUPPORTED_UNKNOWN_PITCHER' and refused['pre']['recommendation'] is None
    result = policy(feeds[keys.index(second)])
    assert result['pa_pitches'][0]['status'] == 'UNSUPPORTED_UNKNOWN_PITCHER'
    # the refusal belonged to the previous pitcher: not "unknown pitcher" about the new one
    assert result['pitch']['status'] == 'UNSUPPORTED_MID_PA'
    assert result['pitch']['pre']['reason'] == dp.STATUS_TEXT['UNSUPPORTED_MID_PA']


def test_another_game_or_no_pending_pitch_is_none_as_the_live_policy(tmp_path):
    path, _ = dataset(tmp_path)
    policy = WatchPolicy(path)
    other = copy.deepcopy(snapshots()[0])
    other['gameData']['game']['pk'] = GAME + 1
    assert policy(other) is None
    over = copy.deepcopy(snapshots()[0])
    over['gameData']['status']['abstractGameState'] = 'Final'
    assert dp.live_rows(over) is None and policy(over) is None


@pytest.mark.parametrize('change', ['schema', 'identity', 'index', 'duplicate', 'pre'])
def test_an_unusable_dataset_is_refused(tmp_path, change):
    path, _ = dataset(tmp_path)
    body = json.loads(path.read_text())
    if change == 'schema':
        body['schema'] = 'other'
    elif change == 'identity':
        body['policy']['identity_sha256'] = 'abc'
    elif change == 'index':
        body['decisions'][1]['index'] = 5
    elif change == 'duplicate':
        body['decisions'].append({**body['decisions'][0], 'index': len(body['decisions'])})
    else:
        body['decisions'][0]['pre']['recommendation'] = None  # ready with no candidates
    path.write_text(json.dumps(body))
    with pytest.raises(WatchUnusable):
        WatchPolicy(path)


def test_the_env_switch_picks_the_watch_policy_and_is_off_by_default(tmp_path, monkeypatch):
    path, _ = dataset(tmp_path)
    monkeypatch.setenv(watch_policy.ENV, str(path))
    assert isinstance(live_armb._default_factory(), WatchPolicy)
    # through the worker: a cached result, and an unusable file makes the policy 'unavailable' (never a crash)
    armb = live_armb.LiveArmB()
    feed = snapshots()[1]
    for _ in range(100):  # the worker loads the policy first; until then a lookup only queues
        result = armb.lookup(feed, wait_s=0.1)
        if result is not None:
            break
        time.sleep(0.05)
    assert result['result']['pitch']['pre']['status'] == 'ready' and armb.status()['state'] == 'ready'
    monkeypatch.setenv(watch_policy.ENV, str(tmp_path/'missing.json'))
    broken = live_armb.LiveArmB()
    assert broken.lookup(feed, wait_s=2.0) is None
    for _ in range(100):
        if broken.status()['state'] == 'unavailable':
            break
        time.sleep(0.05)
    assert broken.status()['state'] == 'unavailable' and 'WatchUnusable' in broken.status()['reason']
    monkeypatch.delenv(watch_policy.ENV)
    called = []
    monkeypatch.setattr(sys.modules.setdefault('demo_precompute', dp), 'LivePolicy', lambda **kw: called.append(kw) or 'live')
    assert live_armb._default_factory() == 'live' and called  # unset: the live policy, as before
