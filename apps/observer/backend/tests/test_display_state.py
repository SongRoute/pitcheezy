"""The real fan screen's single display state per pending pitch slot, driven by the live store and the
recommendation ledger: every state, the SVC-WS-03 admission rule on the real path, D1 (no silent flip: a
recommendation that changes after it was displayed is a visible correction) and ledger failure.
Synthetic feeds and a fake policy only; no network, no model."""
import copy
from pathlib import Path
import sqlite3
import sys
import time
import urllib.error

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app import display_state, synthetic_game
from observer_app.live_armb import LiveArmB
from observer_app.live_feed import LiveFeed
from observer_app.live_store import LiveSnapshotStore, PersistentLiveFeed, _stale_stamp
from observer_app.main import create_app
from test_service_demo_api import feed as final_feed

GAME = 900200
SERVICE, LIVE = f'/api/service/games/{GAME}', f'/api/live/{GAME}/state'
CAND = {'SL': {'rank': 1, 'pitch_type': 'SL', 'pitch_label': '슬라이더', 'zone_id': 'low_away', 'zone_label': '바깥쪽 낮게'},
        'FF': {'rank': 1, 'pitch_type': 'FF', 'pitch_label': '포심', 'zone_id': 'high_in', 'zone_label': '몸쪽 높게'}}


def snap(wall, ts, *, call='B', pid='g5p1', second=False, half='live'):
    """An in-progress game: four finished at-bats and an open fifth with one pitch (two with ``second``)."""
    out = final_feed(GAME, '2026-10-05T10:00:00Z')
    out['gameData']['status'] = {'abstractGameState': 'Live', 'detailedState': 'In Progress'}
    plays = out['liveData']['plays']['allPlays']
    for play in plays:
        for event in play['playEvents']:
            event['playId'] = f"g{play['about']['atBatIndex']+1}p{event['pitchNumber']}"
    current = plays[-1]
    current['about']['isComplete'], current['result'] = False, {}
    current['playEvents'] = current['playEvents'][:2 if second else 1]
    first = current['playEvents'][0]
    first['playId'] = pid
    first['details'] = first['details'] | {'call': {'code': call}}
    first['count'] = {'balls': int(call == 'B'), 'strikes': int(call != 'B'), 'outs': 0}
    if second:
        current['playEvents'][1]['details'] = {'call': {'code': 'B'}, 'description': 'Ball', 'type': {'code': 'SL'}}
        current['playEvents'][1].pop('hitData', None)
    balls = sum(e['details']['call']['code'] == 'B' for e in current['playEvents'])
    current['count'] = {'balls': balls, 'strikes': len(current['playEvents'])-balls, 'outs': 0}
    out['liveData']['linescore'] |= {'currentInning': 1, 'isTopInning': False, 'outs': 0, 'inningState': 'Bottom',
                                     'offense': {'batter': {'id': 32}}, 'defense': {'pitcher': {'id': 41}}}
    out['liveData']['plays']['currentPlay'] = current
    out['metaData'] = {'eventWall': wall} | ({} if ts is None else {'timeStamp': ts})
    if half == 'between':
        current['about']['isComplete'] = True
        out['liveData']['linescore'] |= {'outs': 3, 'inningState': 'End'}
    elif half == 'final':
        out['gameData']['status'] = {'abstractGameState': 'Final', 'detailedState': 'Final'}
    elif half == 'preview':
        out['gameData']['status'] = {'abstractGameState': 'Preview', 'detailedState': 'Warmup'}
    return out


class Policy:
    """Stands in for ``demo_precompute.LivePolicy``: the top type follows the open at-bat's first call, so a
    provider correction of that pitch changes the recommendation. ``status`` forces a policy refusal."""
    def __init__(self):
        self.calls, self.status, self.top, self.reason = [], None, None, '정책 문구'

    def __call__(self, snapshot):
        key = display_state.pending_key(snapshot)
        self.calls.append(key)
        call = snapshot['liveData']['plays']['currentPlay']['playEvents'][0]['details']['call']['code']
        ready = self.status is None
        pre = {'status': 'ready' if ready else 'unsupported', 'reason': None if ready else self.reason,
               'recommendation': {'candidates': [CAND[self.top or ('SL' if call == 'B' else 'FF')]]} if ready else None}
        return {'key': key, 'pitch': {'key': key, 'pre': pre, 'status': self.status or 'SUPPORTED'}, 'pa_pitches': [],
                'previous': None, 'home_we_now': .5, 'policy_identity': 'a6dffaea'+'0'*56, 'seconds': {'bind': 0.0}}


class Upstream:
    def __init__(self, body):
        self.body = body

    def __call__(self, url, timeout):
        if isinstance(self.body, Exception):
            raise self.body
        return copy.deepcopy(self.body)


class Run:
    """One served game: an injected clock, a mutable upstream, a store-backed (or plain) feed and a fake policy."""
    def __init__(self, tmp_path, *, store=True, source='live_poll', first=None):
        self.tmp, self.now, self.policy = tmp_path, [1000.0], Policy()
        self.upstream = Upstream(first or snap(995.0, '20261005_100000'))
        self.store, self.source = store, source
        self.start()

    def start(self):
        """(Re)start the process: a new feed object over the same store file, a new policy worker."""
        options = dict(delay_s=0, clock=lambda: self.now[0], wall=lambda: self.now[0], sleep=lambda _: None)
        self.feed = (PersistentLiveFeed(self.upstream, store=LiveSnapshotStore(self.tmp/'live.sqlite3'), source=self.source, **options)
                     if self.store else LiveFeed(self.upstream, **options))
        self.armb = LiveArmB(lambda: self.policy)
        self.app = create_app(object(), start_worker=False, live_feed=self.feed, live_policy=self.armb, watch_dir=self.tmp,
                              feed_dir=self.tmp, demo_dir=self.tmp/'demo', intent_dir=self.tmp)
        self.client = TestClient(self.app)

    def poll(self, body=None, *, after=12.0, path=SERVICE, settle=True):
        """Advance past the poll interval, optionally change what upstream answers, and ask the screen's route."""
        if body is not None:
            self.upstream.body = body
        self.now[0] += after
        answer = self.client.get(path)
        if settle:  # give the fake policy worker its (instant) turn, then read the settled view
            deadline = time.time()+5
            while time.time() < deadline and (self.armb.pending or self.armb.running is not None or self.armb.loading):
                time.sleep(.01)
            answer = self.client.get(path)
        assert answer.status_code == 200, answer.text
        return answer.json()


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


def shown(view):
    return [c['pitch_type'] for c in view['next']['rec']['candidates']]


# ---------- recommended ----------
def test_a_ledger_committed_recommendation_is_the_recommended_state(run):
    view = run.poll()
    state = view['display_state']
    assert (state['schema'], state['state'], state['reason']) == ('pitcheezy-display-state-v1', 'recommended', 'ledger_recorded')
    assert state['recommendation_shown'] is True and state['correction'] is None and state['pitch_key'] == f'{GAME}:5:2'
    assert state['ledger']['status'] == 'recorded' and state['ledger']['outcome'] in ('first', 'duplicate')
    assert state['feed']['ok'] is True and state['feed']['provider_order'] == 'ordered' and state['feed']['thresholds_provisional'] is True
    assert shown(view) == ['SL'] and view['next']['pitch_key'] == f'{GAME}:5:2'
    assert view['current']['pa_key'] == view['pitches'][-1]['pa_key'] == f'{GAME}:5'
    assert len(run.feed.store.served_recommendations(GAME)) == 1  # committed before it was returned
    assert {p['receipt']['status'] for p in view['pitches']} == {'received'} and view['receipts']['unconfirmed'] == 0


def test_without_a_store_the_recommendation_is_shown_and_says_it_is_unrecorded(tmp_path):
    run = Run(tmp_path, store=False)
    view = run.poll()
    state = view['display_state']
    assert (state['state'], state['reason'], state['ledger']['status']) == ('recommended', 'shown_unrecorded_no_ledger', 'off')
    assert shown(view) == ['SL'] and view['receipts'] is None and 'receipt' not in view['pitches'][0]


def test_the_legacy_live_route_carries_the_same_state(run):
    state = run.poll(path=LIVE)['display_state']
    assert (state['state'], state['reason'], state['pitch_key']) == ('recommended', 'ledger_recorded', f'{GAME}:5:2')
    waiting = run.client.get(LIVE, params={'delay': 100}).json()  # nothing old enough for this viewer's delay yet
    assert waiting['status'] == 'buffering' and (waiting['display_state']['state'], waiting['display_state']['reason']) == ('waiting', 'buffering')


# ---------- unconfirmed (SVC-WS-03 on the real path) ----------
@pytest.mark.parametrize('ts, order, reason', [
    ('20261005_095900', 'stale', 'stale_provider_stamp'),  # a newer event under an older provider stamp
    (None, 'unorderable', 'unorderable_provider_stamp')])
def test_a_snapshot_whose_provider_order_is_not_established_is_unconfirmed_and_never_reaches_the_policy(run, ts, order, reason):
    run.poll()
    calls, rows = list(run.policy.calls), len(run.feed.store.served_recommendations(GAME))
    view = run.poll(snap(1010.0, ts, second=True))
    state = view['display_state']
    assert (state['state'], state['reason'], state['feed']['provider_order']) == ('unconfirmed', reason, order)
    assert state['label'] == '확인 중' and state['message'] == display_state.RECHECK_GAME and state['recommendation_shown'] is False
    assert shown(view) == [] and view['next']['rec']['reason'] == state['message']
    assert run.policy.calls == calls and len(run.feed.store.served_recommendations(GAME)) == rows  # abstained before predict
    assert run.poll(path=LIVE)['recommendation'] is None


def test_same_stamp_with_different_content_is_unconfirmed(run):
    run.poll()
    state = run.poll(snap(1010.0, '20261005_100000', second=True))['display_state']
    assert (state['state'], state['reason']) == ('unconfirmed', 'same_stamp_conflict_provider_stamp')


def test_a_quarantined_pitch_of_the_open_at_bat_is_shown_as_unconfirmed_not_as_an_observed_pitch(run):
    assert run.poll()['display_state']['state'] == 'recommended'
    # a later, ordered revision renames the open at-bat's pitch: its key is already held by another provider id
    view = run.poll(snap(995.0, '20261005_100100', pid='renamed'))
    state = view['display_state']
    assert (state['state'], state['reason'], state['detail']) == ('unconfirmed', 'prior_pitch_unconfirmed', '5:1/ambiguous_remap')
    assert state['feed']['provider_order'] == 'ordered' and shown(view) == []
    last = view['pitches'][-1]
    assert last['receipt'] == {'status': 'unconfirmed', 'reason': 'ambiguous_remap', 'corrected': False}
    assert last['actual']['result_label'] == '확인 중' and last['actual']['pitch_type'] is None and last['actual']['x'] is None
    assert view['receipts']['unconfirmed'] == 1
    assert state['correction']['kind'] == 'withdrawn'  # the earlier recommendation for this pitch is said to be gone


def test_a_pitch_without_a_complete_identity_is_unconfirmed(tmp_path):
    body = snap(995.0, '20261005_100000')
    del body['liveData']['plays']['currentPlay']['playEvents'][0]['playId']
    view = Run(tmp_path, first=body).poll()
    assert (view['display_state']['state'], view['display_state']['detail']) == ('unconfirmed', '5:1/incomplete_identity')
    assert view['pitches'][-1]['receipt']['status'] == 'unconfirmed' and view['pitches'][0]['receipt']['status'] == 'received'


def test_an_unreadable_pitch_ledger_keeps_only_bodies_admitted_before(run, monkeypatch):
    run.poll()

    def broken(game_pk):
        raise sqlite3.OperationalError('disk I/O error')
    monkeypatch.setattr(run.feed.store, 'pitch_events', broken)
    same = run.poll()
    assert same['display_state']['state'] == 'recommended' and same['receipts']['unconfirmed'] == 0
    newer = run.poll(snap(1030.0, '20261005_100100', second=True))  # a pitch never seen admitted
    assert (newer['display_state']['state'], newer['display_state']['reason']) == ('unconfirmed', 'receipts_unreadable')
    assert newer['pitches'][-1]['receipt']['reason'] == 'receipts_unreadable' and shown(newer) == []


def test_the_synthetic_walk_through_and_the_real_screen_share_one_admission_rule():
    assert synthetic_game.admit_body is display_state.admit_body
    row = {'outcome': 'new', 'at_bat_number': 1, 'pitch_number': 1, 'sha': display_state.body_sha({'a': 1}), 'reason': None}
    assert display_state.admit_body({'a': 1}, (1, 1), [row]) == ('received', None)
    assert display_state.admit_body({'a': 2}, (1, 1), [row]) == ('unconfirmed', 'not_accepted')
    refused = row | {'outcome': 'quarantine', 'sha': display_state.body_sha({'a': 2}), 'reason': 'stale_revision'}
    assert display_state.admit_body({'a': 2}, (1, 1), [row, refused]) == ('unconfirmed', 'stale_revision')
    newer = row | {'outcome': 'correction', 'sha': display_state.body_sha({'a': 3})}
    assert display_state.admit_body({'a': 1}, (1, 1), [row, newer]) == ('unconfirmed', 'not_accepted')  # superseded
    assert display_state.admit_body({'a': {1, 2}}, (1, 1), [row]) == ('unconfirmed', 'body_hash_unusable')


# ---------- D1: no silent flip ----------
def test_a_recommendation_that_changes_after_display_is_a_visible_correction(run):
    assert shown(run.poll()) == ['SL']
    # the provider corrects the open at-bat's first pitch (ball -> called strike) under a later stamp
    view = run.poll(snap(995.0, '20261005_100100', call='C'))
    state = view['display_state']
    assert shown(view) == ['FF'] and (state['state'], state['reason']) == ('recommended', 'ledger_recorded')
    assert state['correction']['kind'] == 'replaced' and state['correction']['message'] == '상황이나 기록이 바뀌어 추천이 달라졌어요.'
    assert state['correction']['previous'] == {'pitch_type': 'SL', 'pitch_label': '슬라이더', 'zone_id': 'low_away', 'zone_label': '바깥쪽 낮게'}
    assert state['correction']['previous_decided'] == '1970-01-01T00:16:52Z'
    assert view['pitches'][-1]['receipt'] == {'status': 'received', 'reason': None, 'corrected': True}
    assert [r['outcome'] for r in run.feed.store.served_recommendations(GAME)] == ['first', 'first']  # appended, not rewritten
    assert run.poll()['display_state']['correction']['kind'] == 'replaced'  # it stays for as long as the slot does
    # ... and the provider takes the correction back: the first recommendation returns, again as a correction
    back = run.poll(snap(995.0, '20261005_100200'))
    assert shown(back) == ['SL'] and back['display_state']['correction']['previous']['pitch_type'] == 'FF'
    assert len(run.feed.store.served_recommendations(GAME)) == 2


def test_the_correction_is_still_reported_after_a_restart(run):
    assert shown(run.poll()) == ['SL']
    run.start()  # process restart: memory is gone, the ledger is not
    view = run.poll(snap(995.0, '20261005_100100', call='C'))
    assert shown(view) == ['FF'] and view['display_state']['correction']['previous']['pitch_type'] == 'SL'


def test_a_recommendation_withdrawn_after_display_says_so(run):
    assert shown(run.poll()) == ['SL']
    run.policy.status = 'UNSUPPORTED_MID_PA'
    view = run.poll(snap(995.0, '20261005_100100', call='C'))
    state = view['display_state']
    assert (state['state'], state['reason'], shown(view)) == ('unsupported', 'at_bat_rejected', [])
    assert state['correction']['kind'] == 'withdrawn' and state['correction']['previous']['pitch_type'] == 'SL'


def test_a_different_content_for_the_same_decision_never_replaces_the_displayed_one(run):
    assert shown(run.poll()) == ['SL']
    run.policy.top = 'FF'
    run.armb.cache.clear()  # e.g. recomputed after a restart with a different result for the very same inputs
    view = run.poll()
    state = view['display_state']
    assert shown(view) == ['SL'] and (state['state'], state['reason'], state['correction']) == ('recommended', 'ledger_conflict_first_kept', None)
    assert [r['outcome'] for r in run.feed.store.served_recommendations(GAME)] == ['first', 'conflict']


def test_without_a_ledger_the_process_still_remembers_what_it_displayed(tmp_path):
    run = Run(tmp_path, store=False)
    assert shown(run.poll()) == ['SL']
    view = run.poll(snap(995.0, '20261005_100100', call='C'))
    assert shown(view) == ['FF'] and view['display_state']['correction']['previous']['pitch_type'] == 'SL'


def test_a_new_pitch_slot_starts_without_a_correction(run):
    run.poll()
    view = run.poll(snap(1010.0, '20261005_100100', second=True))
    assert view['display_state']['pitch_key'] == f'{GAME}:5:3' and view['display_state']['correction'] is None


# ---------- ledger failure ----------
def test_a_recommendation_that_cannot_be_committed_is_not_shown_and_the_state_says_why(run, monkeypatch):
    def broken(**_):
        raise sqlite3.OperationalError('disk I/O error')
    monkeypatch.setattr(run.feed.store, 'append_recommendation', broken)
    view = run.poll()
    state = view['display_state']
    assert (state['state'], state['reason'], state['recommendation_shown']) == ('ledger_unavailable', 'OperationalError', False)
    assert state['ledger'] == {'status': 'withheld', 'outcome': None, 'why': 'OperationalError', 'store_ok': True}
    assert shown(view) == [] and view['current']['situation'] is not None and len(view['pitches']) == 9  # the game goes on
    assert '표시하지 않아요' in state['message']
    live = run.poll(path=LIVE)
    assert live['recommendation']['recommendation'] is None and live['display_state']['state'] == 'ledger_unavailable'
    monkeypatch.undo()
    again = run.poll()
    assert again['display_state']['state'] == 'recommended' and shown(again) == ['SL']


def test_a_snapshot_the_store_could_not_save_shows_no_new_pitch_as_observed_and_no_recommendation(run, monkeypatch):
    run.poll()

    def down(*args, **kwargs):
        raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(run.feed.store, 'ingest', down)
    view = run.poll(snap(1010.0, '20261005_100100', second=True))
    state = view['display_state']
    assert state['state'] == 'unconfirmed' and state['detail'] == '5:2/uncommitted_memory_entry' and shown(view) == []
    assert state['ledger']['store_ok'] is False
    assert view['pitches'][-1]['receipt']['reason'] == 'uncommitted_memory_entry' and view['pitches'][-2]['receipt']['status'] == 'received'
    monkeypatch.undo()
    assert run.poll()['display_state']['state'] == 'recommended'


# ---------- unsupported ----------
@pytest.mark.parametrize('status, reason, words', [
    ('UNSUPPORTED_UNKNOWN_PITCHER', 'unknown_pitcher', '이 투수는'),
    ('UNSUPPORTED_EMPTY_SUPPORT', 'no_supported_pitch_types', '구종 기록이 부족'),
    ('UNSUPPORTED_MID_PA', 'at_bat_rejected', '이 타석은'),
    ('SOMETHING_NEW', 'policy_refused', '정책 문구')])
def test_an_unsupported_pitch_says_which_kind_in_one_sentence(run, status, reason, words):
    run.policy.status = status
    view = run.poll()
    state = view['display_state']
    assert (state['state'], state['reason'], state['detail']) == ('unsupported', reason, status)
    assert words in state['message'] and state['recommendation_shown'] is False and shown(view) == []
    run.policy.reason = 'UNSUPPORTED_SOMETHING'  # a refusal the copy table does not know: no code reaches the fan
    if reason == 'policy_refused':
        run.armb.cache.clear()
        assert run.poll()['display_state']['message'] == '이 상황은 추천 대상이 아니에요.'
    assert run.feed.store.served_recommendations(GAME) == []


# ---------- feed delayed / disconnected ----------
def test_an_event_that_a_good_poll_missed_by_more_than_the_provisional_threshold_is_feed_delayed(tmp_path, monkeypatch):
    run = Run(tmp_path, first=snap(900.0, '20261005_100000'))
    first = run.poll()['display_state']  # first poll (wall 1012): nobody was polling before, so nothing is measured
    assert first['state'] == 'recommended' and first['feed']['lag_s'] is None and first['feed']['lag_measured'] is False
    # the next poll (wall 1024) brings an event of wall 950 that the good poll at 1012 did not have: >= 62 s late
    view = run.poll(snap(950.0, '20261005_100100', second=True))
    state = view['display_state']
    assert (state['state'], state['reason'], state['detail']) == ('feed_delayed', 'feed_lag_above_threshold', 'ledger_recorded')
    assert state['feed'] | {'last_ok_at': None, 'received_at': None} == {
        'ok': True, 'delayed': True, 'lag_s': 62.0, 'lag_max_s': 74.0, 'lag_measured': True, 'lag_threshold_s': 45.0,
        'thresholds_provisional': True, 'last_ok_at': None, 'failing_since': None, 'failing_for_s': None, 'received_at': None,
        'provider_order': 'ordered'}
    assert '62초 넘게' in state['message'] and state['recommendation_shown'] is True and shown(view) == ['SL']
    monkeypatch.setenv('PITCHEEZY_OBSERVER_FEED_LAG_DELAYED_S', '100')  # the threshold is configuration
    assert run.poll()['display_state']['state'] == 'recommended'


def test_time_nobody_polled_is_never_reported_as_feed_lag(tmp_path):
    state = Run(tmp_path, first=snap(700.0, '20261005_100000')).poll()['display_state']  # first viewer 312 s after the event
    assert state['state'] == 'recommended' and state['feed']['delayed'] is False and state['feed']['lag_s'] is None


def test_a_replayed_recording_never_claims_a_measured_lag(tmp_path):
    state = Run(tmp_path, source='replay_poll', first=snap(100.0, '20261005_100000')).poll()['display_state']
    assert state['state'] == 'recommended' and state['feed']['lag_s'] is None and state['feed']['lag_measured'] is False


def test_poll_failures_are_disconnected_and_keep_only_what_was_already_displayed(run):
    assert shown(run.poll()) == ['SL']
    calls = list(run.policy.calls)
    view = run.poll(urllib.error.URLError('down'))
    state = view['display_state']
    assert (state['state'], state['reason']) == ('disconnected', 'poll_failed_last_snapshot_only')
    assert state['feed']['ok'] is False and state['feed']['failing_since'] == '1970-01-01T00:17:04Z' and state['feed']['last_ok_at'] == '1970-01-01T00:16:52Z'
    assert state['recommendation_shown'] is True and shown(view) == ['SL'] and state['ledger']['outcome'] == 'held'
    assert view['feed']['ok'] is False and run.policy.calls == calls
    later = run.poll()
    assert later['display_state']['feed']['failing_for_s'] == 12.0 and later['display_state']['feed']['failing_since'] == state['feed']['failing_since']
    assert len(run.feed.store.served_recommendations(GAME)) == 1  # nothing recorded while disconnected
    back = run.poll(snap(995.0, '20261005_100000'))  # reconnect: no reload, the next poll is simply fine again
    assert back['display_state']['state'] == 'recommended' and back['display_state']['feed']['failing_since'] is None


def test_disconnected_before_anything_was_recommended_shows_no_recommendation(run):
    run.feed.snapshot(GAME)  # a first good poll that no viewer asked a recommendation for
    view = run.poll(urllib.error.URLError('down'), settle=False)
    state = view['display_state']
    assert state['state'] == 'disconnected' and state['recommendation_shown'] is False and shown(view) == []
    time.sleep(.1)  # a wrongly queued state would be computed by now
    assert run.policy.calls == [] and not run.armb.pending and run.armb._thread is None
    assert run.feed.store.served_recommendations(GAME) == []


# ---------- waiting / pregame / final / unavailable ----------
def test_states_without_a_pending_pitch(tmp_path, monkeypatch):
    for half, expect in (('between', ('waiting', 'between_innings')), ('final', ('final', 'game_over')), ('preview', ('pregame', 'scheduled'))):
        folder = tmp_path/half
        folder.mkdir()
        view = Run(folder, first=snap(995.0, '20261005_100000', half=half)).poll(settle=False)
        state = view['display_state']
        assert (state['state'], state['reason']) == expect and state['pitch_key'] is None and state['recommendation_shown'] is False
        assert view['next'] is None or view['next']['rec']['candidates'] == []
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '0')
    off = Run(tmp_path).poll(settle=False)['display_state']
    assert (off['state'], off['reason']) == ('unavailable', 'recommender_off')


def test_decide_covers_every_remaining_branch():
    facts = display_state.feed_facts(object(), GAME, None, display_state.config(), now=0.0)
    decide = lambda **kw: display_state.decide(facts=facts, **({'gated': None, 'view': None} | kw))
    pick = lambda state: (state['state'], state['reason'])
    assert pick(decide(view={'status': 'computing'})) == ('waiting', 'computing')
    assert pick(decide(view={'status': 'unavailable', 'code': 'compute_failed'})) == ('unavailable', 'compute_failed')
    assert pick(decide(view={'status': 'unavailable'})) == ('unavailable', 'recommender_unavailable')
    assert pick(decide()) == ('waiting', 'no_pending_pitch')
    assert pick(decide(recommender=False)) == ('unavailable', 'recommender_off')
    assert pick(decide(gated=('waiting', 'resolving', None))) == ('waiting', 'resolving')
    for key, (label, message) in display_state.COPY.items():  # every state has calm fan copy; no machine codes in it
        assert key[0] in display_state.STATES and label and all(word not in (message or '') for word in ('ledger', 'provider', '_'))
    gate = lambda status, abstract='Live': display_state.gate(status, abstract, facts | {'ok': True}, None, {})
    assert gate('not_live', 'Suspended')[:2] == ('pregame', 'not_live')
    assert gate('unsupported_game_type')[:2] == ('unsupported', 'game_type_out_of_scope')
    assert display_state.config({'display': {'feed_lag_delayed_s': 60}})['feed_lag_delayed_s'] == 60
    assert display_state.config({})['provisional'] is True


# ---------- incremental polling ----------
def test_a_pitch_whose_record_changed_is_resent_whatever_the_cursor(run):
    first = run.poll()
    tail = run.client.get(SERVICE, params={'after': 8}).json()
    assert tail['pitches'] == [] and tail['receipts']['version'] == first['receipts']['version']
    run.poll(snap(995.0, '20261005_100100', call='C'))
    tail = run.client.get(SERVICE, params={'after': 8}).json()
    assert [(p['index'], p['receipt']['corrected']) for p in tail['pitches']] == [(8, True)]
    assert tail['receipts']['version'] != first['receipts']['version'] and tail['receipts']['corrected'] == 1


# ---------- independent review regressions ----------
class EventPolicy(Policy):
    """The top type follows the number of play events: a pickoff changes the decision, not the pending pitch."""
    def __call__(self, snapshot):
        out = super().__call__(snapshot)
        count = len(snapshot['liveData']['plays']['currentPlay']['playEvents'])
        if out['pitch']['pre']['recommendation']:
            out['pitch']['pre']['recommendation'] = {'candidates': [CAND['SL' if count == 1 else 'FF']]}
        out['pa_pitches'] = [{'key': f'{GAME}:5:1', 'pre': {'status': 'ready', 'recommendation': {'candidates': [CAND['FF']]}},
                              'actual': {'pitch_type': 'FF'}}]
        return out


def with_pickoff(wall, ts):
    body = snap(wall, ts)
    play = body['liveData']['plays']['currentPlay']
    play['playEvents'].append({'isPitch': False, 'type': 'pickoff', 'index': 9, 'details': {'description': 'Pickoff Attempt 1B'},
                               'count': play['playEvents'][0]['count']})
    return body


def brief(view):
    state = view['display_state']
    return state['state'], state['recommendation_shown'], state['correction'] and state['correction']['kind']


def test_a_disconnect_never_brings_back_a_recommendation_whose_inputs_are_unconfirmed(run):
    assert shown(run.poll()) == ['SL']
    assert run.poll(snap(995.0, '20261005_100100', pid='renamed'))['display_state']['state'] == 'unconfirmed'
    view = run.poll(urllib.error.URLError('down'))
    state = view['display_state']
    assert (state['state'], state['detail'], state['recommendation_shown']) == ('disconnected', 'unconfirmed:prior_pitch_unconfirmed', False)
    assert shown(view) == [] and view['pitches'][-1]['receipt']['status'] == 'unconfirmed'
    assert run.poll(path=LIVE)['recommendation'] is None


@pytest.mark.parametrize('store', [True, False])
def test_a_viewer_on_a_longer_delay_is_not_told_about_a_later_decision_for_the_same_pitch(tmp_path, store):
    run = Run(tmp_path, store=store)
    run.policy = EventPolicy()
    run.start()
    first = run.poll(path=LIVE+'?delay=0')
    assert brief(first) == ('recommended', True, None)
    newer = run.poll(with_pickoff(1015.0, '20261005_100100'), path=LIVE+'?delay=0')  # same pending pitch, new decision
    assert brief(newer) == ('recommended', True, 'replaced') and newer['display_state']['correction']['previous']['pitch_type'] == 'SL'
    delayed = run.client.get(LIVE, params={'delay': 20}).json()  # still on the first slot: SL is all this viewer saw
    assert brief(delayed) == ('recommended', True, None)
    assert delayed['recommendation']['recommendation']['candidates'][0]['pitch_type'] == 'SL'


def test_a_viewer_on_a_longer_delay_is_judged_against_the_ledger_as_of_their_snapshot(run):
    assert brief(run.poll(path=LIVE+'?delay=0')) == ('recommended', True, None)
    newer = snap(1015.0, '20261005_100100', second=True)
    newer['liveData']['plays']['currentPlay']['playEvents'][0]['pitchData'] = {'breaks': {'spinRate': 2400}}  # enriched later
    run.poll(newer, path=LIVE+'?delay=0')
    delayed = run.client.get(LIVE, params={'delay': 20}).json()
    assert brief(delayed) == ('recommended', True, None) and delayed['pitch_sequence'][0]['pitch_type'] == 'FF'


def test_when_earlier_displays_cannot_be_read_nothing_is_shown_rather_than_flipped_silently(run, monkeypatch):
    assert shown(run.poll()) == ['SL']

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError('disk I/O error')
    monkeypatch.setattr(run.feed.store, 'recommendations_for_pitch', broken)
    view = run.poll(snap(995.0, '20261005_100100', call='C'))
    state = view['display_state']
    assert (state['state'], state['reason'], state['recommendation_shown']) == ('ledger_unavailable', 'display_history_unreadable', False)
    assert shown(view) == [] and '표시하지 않아요' in state['message']
    monkeypatch.undo()
    assert brief(run.poll()) == ('recommended', True, 'replaced')


def test_a_configured_store_that_did_not_open_shows_no_unrecorded_recommendation(tmp_path):
    run = Run(tmp_path, store=False)
    run.feed.store_error = 'open: OperationalError'  # what main.live_parts leaves when the store cannot be opened
    view = run.poll()
    state = view['display_state']
    assert (state['state'], state['reason'], state['ledger']['status']) == ('ledger_unavailable', 'store_open_failed', 'store_unavailable')
    assert shown(view) == [] and state['ledger']['store_ok'] is False


def test_the_legacy_route_lists_no_unadmitted_pitch_and_no_candidate_when_nothing_is_shown(tmp_path):
    run = Run(tmp_path)
    run.policy = EventPolicy()
    run.start()
    run.poll(path=LIVE)
    run.policy.status = 'UNSUPPORTED_MID_PA'  # (the route's head start may compute the newest state before it is admitted)
    live = run.poll(snap(995.0, '20261005_100100', pid='renamed', call='C'), path=LIVE)
    assert live['display_state']['state'] == 'unconfirmed' and live['recommendation'] is None
    assert live['pitch_sequence'] == [{key: None for key in live['pitch_sequence'][0]} | {'pitch_number': 1, 'receipt': 'unconfirmed'}]
    refused = run.poll(snap(995.0, '20261005_100200', call='C'), path=LIVE)  # an admitted correction, but the policy refuses
    assert refused['display_state']['state'] == 'unsupported' and refused['pitch_sequence'][0]['pitch_type'] == 'FF'
    assert refused['recommendation']['recommendation'] is None and refused['recommendation']['pa_pitches'] == []


def test_a_pitch_without_a_pitch_number_never_borrows_another_pitch_admission(tmp_path):
    body = snap(995.0, '20261005_100000', second=True)
    events = body['liveData']['plays']['currentPlay']['playEvents']
    del events[0]['pitchNumber']
    events[1]['pitchNumber'] = 1
    view = Run(tmp_path, first=body).poll()
    assert view['display_state']['state'] == 'unconfirmed'
    assert [(p['receipt']['status'], p['actual']['pitch_type']) for p in view['pitches'][-2:]] == [('unconfirmed', None)]*2
    assert {p['receipt']['status'] for p in view['pitches'][:-2]} == {'received'}


def test_a_withdrawal_after_a_ledger_failure_is_reported(run, monkeypatch):
    assert shown(run.poll()) == ['SL']

    def broken(**_):
        raise sqlite3.OperationalError('disk I/O error')
    monkeypatch.setattr(run.feed.store, 'append_recommendation', broken)
    view = run.poll(snap(995.0, '20261005_100100', call='C'))
    assert brief(view) == ('ledger_unavailable', False, 'withdrawn') and shown(view) == []


def test_with_the_ledger_off_and_the_store_on_displays_are_remembered_in_memory(tmp_path, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_REC_LEDGER', '0')
    run = Run(tmp_path)
    first = run.poll()
    assert first['display_state']['reason'] == 'shown_unrecorded_no_ledger' and first['display_state']['ledger']['status'] == 'not_applicable'
    assert brief(run.poll(snap(995.0, '20261005_100100', call='C'))) == ('recommended', True, 'replaced')
    assert brief(run.poll(urllib.error.URLError('down'))) == ('disconnected', True, 'replaced')
    assert run.feed.store.served_recommendations(GAME) == []


def test_an_unconfirmed_pitch_before_the_cursor_is_resent_and_its_return_changes_the_version(run):
    first = run.poll()
    run.poll(snap(995.0, '20261005_100100', pid='renamed'))
    tail = run.client.get(SERVICE, params={'after': 8}).json()
    assert [(p['index'], p['receipt']['status']) for p in tail['pitches']] == [(8, 'unconfirmed')]
    assert tail['receipts']['version'] != first['receipts']['version']
    run.poll(snap(995.0, '20261005_100200'))  # the provider goes back to the admitted body
    tail = run.client.get(SERVICE, params={'after': 8}).json()
    assert tail['pitches'] == [] and tail['receipts']['version'] == first['receipts']['version']  # the client refetches on the change


def test_a_final_live_game_keeps_its_receipts(tmp_path):
    view = Run(tmp_path, first=snap(995.0, '20261005_100000', half='final')).poll(settle=False)
    assert view['display_state']['state'] == 'final' and {p['receipt']['status'] for p in view['pitches']} == {'received'}


def test_a_plate_appearance_holding_an_unadmitted_pitch_announces_no_result(tmp_path):
    body = snap(995.0, '20261005_100000')
    del body['liveData']['plays']['allPlays'][3]['playEvents'][1]['playId']  # the home-run pitch of at-bat 4
    view = Run(tmp_path, first=body).poll()
    done = view['pas'][3]
    assert (done['complete'], done['event'], done['result_text'], done['play_text']) == (True, None, '확인 중', None)
    assert view['pas'][2]['event'] == 'field_out' and view['pitches'][7]['actual']['result_label'] == '확인 중'


def test_a_game_over_body_behind_an_older_stamp_ends_the_game_and_keeps_every_pitch_admitted(run):
    """849809 regression: the Game Over body carried an older provider stamp than the last in-progress body and
    enriched every pitch's payload. Dropped as stale, it left the screen 'between innings' for good and, because its
    per-pitch corrections were still accepted, every past pitch 확인 중 with its recorded recommendation gone."""
    run.poll()
    thrown = run.poll(snap(995.0, '20261005_100100', second=True))  # the recommended pitch 5:2 is thrown
    assert thrown['pitches'][-1]['rec']['provenance'] == 'captured_live' and thrown['receipts']['unconfirmed'] == 0
    over = snap(995.0, '20261005_100059', second=True, half='final')
    for play in over['liveData']['plays']['allPlays'][:-1]:  # the finished at-bats come back with fuller pitch data
        for event in play['playEvents']:
            event['pitchData'] = dict(event.get('pitchData') or {}) | {'breaks': {'breakAngle': 7.2}}
    assert _stale_stamp('20261005_100059', '20261005_100100', False)  # the stamp alone would drop it
    view = run.poll(over)
    state = view['display_state']
    assert (state['state'], state['reason'], state['feed']['provider_order']) == ('final', 'game_over', 'stale')
    assert view['receipts']['unconfirmed'] == 0 and {p['receipt']['status'] for p in view['pitches']} == {'received'}
    assert view['pitches'][-1]['rec']['provenance'] == 'captured_live' and view['receipts']['paired'] == 1
    newest = run.feed.store.restore(GAME, run.now[0], 120)[-1]
    assert newest['version'] == 3 and newest['feed']['gameData']['status']['abstractGameState'] == 'Final'  # version 2 was the thrown pitch
