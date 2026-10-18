"""Synthetic tests for apps/observer/ops/ws_alert.py. Fake files only; the only notifier is FakeNotifier."""
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ws_alert as wa  # noqa: E402

GIB = wa.GIB
T0 = datetime(2026, 10, 24, 0, 30, tzinfo=timezone.utc).timestamp()  # 09:30 KST, game started 00:08Z
PK = 776001


def iso(t):
    return wa._iso(t)


class World:
    """A fake publisher output folder + state file + ops status + profile log, and a clock."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.out = tmp / 'pub'
        (self.out / 'v1' / 'games').mkdir(parents=True)
        (self.out / 'v1' / 'game').mkdir(parents=True)
        self.state_path = tmp / 'pub.state.json'
        self.ops_path = tmp / 'ops.json'
        self.log = tmp / 'RESULTS.log'
        self.memory, self.disk = 8 * GIB, 100 * GIB
        self.fake = wa.FakeNotifier()
        self.cfg = wa.load_config(None, out_dir=str(self.out), alert_dir=str(tmp / 'alert'), env={})
        self.cfg['paths']['profile_log'] = str(self.log)
        self.cfg['paths']['disk_paths'] = [str(tmp)]
        self.cfg['paths']['ops_status'] = None  # never the real ~/Projects/pitcheezy-pm-jobs/ws-ops/status.json
        self.raw = []
        self.journal = wa.JsonlNotifier(tmp / 'alert' / 'alerts.jsonl')

    def games(self, *entries, date='2026-10-24'):
        (self.out / 'v1' / 'games' / f'{date}.json').write_text(json.dumps({'date': date, 'games': list(entries)}))

    def game(self, pk=PK, status='live', start=T0 - 1320):
        return {'game_pk': pk, 'status': status, 'live': status == 'live', 'start': iso(start)}

    def health(self, now, *, published=None, checks=None, failing=(), unavailable=(), upload=None):
        body = {'schema': 'pitcheezy-publish-health-v1', 'sequence': 10, 'published_at': iso(published or now),
                'publish_interval_s': 10, 'files_written': 1, 'unavailable_games': list(unavailable),
                'unavailable_dates': [], 'feed_failing_games': list(failing),
                'game_checks': checks if checks is not None else {
                    str(PK): {'checked_at': iso(now), 'sequence_checked': 10, 'file_sequence': 5, 'feed_ok': True}}}
        if upload is not None:
            body['upload'] = upload
        (self.out / 'v1' / 'health.json').write_text(json.dumps(body))

    def game_file(self, pk=PK, *, delayed=False, lag=3.0, status='live', text='In Progress'):
        (self.out / 'v1' / 'game' / f'{pk}.json').write_text(json.dumps(
            {'game': {'game_pk': pk, 'status': status, 'status_text': text},
             'display_state': {'feed': {'delayed': delayed, 'lag_s': lag, 'lag_threshold_s': 45}}}))

    def state(self, attempts=None, **extra):
        self.state_path.write_text(json.dumps({'sequence': 10, 'upload_pending': {}, 'upload_attempts': attempts or {},
                                               **extra}))

    def collect(self, cfg, now):
        return wa.collect(cfg, now, memory=lambda: self.memory, disk=lambda paths: {p: self.disk for p in paths})

    def run(self, now):
        """Events without the window open/close notices (tested on their own via self.raw)."""
        self.raw = wa.run_once(self.cfg, now=now, senders=[self.fake], journal=self.journal, collect_fn=self.collect)
        return [e for e in self.raw if e['kind'] != 'notice']

    def healthy(self, now):
        self.games(self.game())
        self.health(now)
        self.game_file()
        self.state()

    def keys(self, events, kind='fire'):
        return {e['key'] for e in events if e['kind'] == kind}


@pytest.fixture
def w(tmp_path):
    return World(tmp_path)


def test_healthy_live_game_is_quiet(w):
    w.healthy(T0)
    assert w.run(T0) == []
    assert [m['title'].split()[1] for m in w.fake.sent] == ['감시']  # only the one window-open notice
    w.health(T0 + 15)
    assert w.run(T0 + 15) == [] and len(w.fake.sent) == 1


def test_no_game_is_quiet_even_with_a_dead_publisher(w):
    w.games(w.game(status='final'))
    w.health(T0, published=T0 - 7200, failing=[PK])
    w.state()
    assert w.run(T0) == []
    assert w.fake.sent == []


def test_health_stalled_then_publisher_down(w):
    w.healthy(T0)
    w.health(T0, published=T0 - 45, checks={str(PK): {'checked_at': iso(T0), 'sequence_checked': 10,
                                                       'file_sequence': 5, 'feed_ok': True}})
    assert w.keys(w.run(T0)) == {'health_stalled'}
    w.health(T0, published=T0 - 130, checks={str(PK): {'checked_at': iso(T0 + 15), 'sequence_checked': 10,
                                                        'file_sequence': 5, 'feed_ok': True}})
    assert w.keys(w.run(T0 + 15)) == {'publisher_down'}
    assert '긴급' in w.fake.sent[-1]['title'] and w.fake.sent[-1]['critical']


def test_publisher_missing_before_first_pitch(w):
    # scheduled game 20 min away, publisher never started: no health.json at all
    w.games(w.game(status='scheduled', start=T0 + 1200))
    events = w.run(T0)
    assert w.keys(events) == {'publisher_down'}
    assert 'health.json 없음' in w.fake.sent[0]['body']


def test_manual_schedule_opens_a_window_without_publisher_files(w):
    w.cfg['schedule'] = [{'game_pk': PK, 'start': iso(T0 + 600)}]
    assert w.keys(w.run(T0)) == {'publisher_down'}


def test_feed_down_needs_60s_and_recovers(w):
    w.healthy(T0)
    bad = {str(PK): {'checked_at': iso(T0), 'sequence_checked': 10, 'file_sequence': 5, 'feed_ok': False}}
    w.health(T0, checks=bad)
    assert w.run(T0) == []  # held
    bad[str(PK)]['checked_at'] = iso(T0 + 61)
    w.health(T0 + 61, checks=bad)
    assert w.keys(w.run(T0 + 61)) == {f'feed_down:{PK}'}
    w.health(T0 + 75)
    events = w.run(T0 + 75)
    assert w.keys(events, 'recover') == {f'feed_down:{PK}'}
    assert w.fake.sent[-1]['title'].startswith('Pitcheezy 복구')


def test_feed_failing_games_list_also_counts(w):
    w.healthy(T0)
    w.health(T0, failing=[PK])
    w.run(T0)
    w.health(T0 + 61, failing=[PK])
    assert f'feed_down:{PK}' in w.keys(w.run(T0 + 61))


def test_feed_delayed(w):
    w.healthy(T0)
    w.game_file(delayed=True, lag=52.0)
    assert w.run(T0) == []
    w.health(T0 + 21)
    events = w.run(T0 + 21)
    assert w.keys(events) == {f'feed_delayed:{PK}'}
    assert '52.0초' in w.fake.sent[-1]['body']


def test_game_check_stalled(w):
    w.healthy(T0)
    w.health(T0, checks={str(PK): {'checked_at': iso(T0 - 40), 'sequence_checked': 10, 'file_sequence': 5,
                                   'feed_ok': True}})
    assert w.keys(w.run(T0)) == {f'game_check_stalled:{PK}'}


def test_file_stuck_by_time_and_by_ops_upstream(w):
    w.healthy(T0)
    assert w.run(T0) == []
    w.health(T0 + 290)
    assert w.run(T0 + 290) == []  # file_sequence 5 for 290 s: under 300
    w.health(T0 + 310)
    assert w.keys(w.run(T0 + 310)) == {f'file_stuck:{PK}'}
    # file_sequence moves -> clears
    w.health(T0 + 320, checks={str(PK): {'checked_at': iso(T0 + 320), 'sequence_checked': 12, 'file_sequence': 12,
                                         'feed_ok': True}})
    assert w.keys(w.run(T0 + 320), 'recover') == {f'file_stuck:{PK}'}


def test_file_stuck_sooner_when_ops_reports_upstream_change(w, tmp_path):
    w.healthy(T0)
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    w.ops_path.write_text(json.dumps({'updated_at': iso(T0), 'games': [{'game_pk': PK, 'start': iso(T0 - 1320),
                                                                          'upstream_changed_at': iso(T0 + 5)}]}))
    w.run(T0)
    w.ops_path.write_text(json.dumps({'updated_at': iso(T0 + 70), 'games': [{'game_pk': PK,
                                                                               'upstream_changed_at': iso(T0 + 5)}]}))
    w.health(T0 + 70)
    assert w.keys(w.run(T0 + 70)) == {f'file_stuck:{PK}'}


def test_game_unavailable_code(w):
    w.healthy(T0)
    w.health(T0, unavailable=[{'game_pk': PK, 'code': 'build_failed', 'status': 503}])
    w.run(T0)
    w.health(T0 + 31, unavailable=[{'game_pk': PK, 'code': 'build_failed', 'status': 503}])
    assert w.keys(w.run(T0 + 31)) == {f'game_unavailable:{PK}'}


def test_background_upload_steady_state_is_quiet(w):
    # B1: real background upload keeps 2-3 pending and upload.code='upload_failed' every cycle; that is normal.
    w.healthy(T0)
    for i in range(12):
        t = T0 + 10 * i
        w.health(t, upload={'pending': 2 + i % 2, 'code': 'upload_failed', 'dry_run': False},
                 unavailable=[{'game_pk': PK, 'code': 'upload_failed', 'status': 502}],
                 checks={str(PK): {'checked_at': iso(t), 'sequence_checked': 10 + i, 'file_sequence': 10 + i,
                                   'feed_ok': True}})
        body = json.loads((w.out / 'v1' / 'health.json').read_text())
        body['sequence'] = 10 + i
        (w.out / 'v1' / 'health.json').write_text(json.dumps(body))
        assert w.run(t) == [], i


def _pending_cycle(w, t, seq, pending, dry=False):
    w.health(t, upload={'pending': pending, 'code': 'upload_failed', 'dry_run': dry},
             checks={str(PK): {'checked_at': iso(t), 'sequence_checked': seq, 'file_sequence': seq, 'feed_ok': True}})
    body = json.loads((w.out / 'v1' / 'health.json').read_text())
    body['sequence'] = seq
    (w.out / 'v1' / 'health.json').write_text(json.dumps(body))
    return w.run(t)


def test_upload_pending_above_floor_and_not_decreasing(w):
    w.healthy(T0)
    for i, pending in enumerate((6, 8, 8)):
        assert _pending_cycle(w, T0 + 10 * i, 20 + i, pending) == []
    events = _pending_cycle(w, T0 + 30, 23, 11)
    assert w.keys(events) == {'upload_pending'}
    assert '6 → 8 → 8 → 11' in w.fake.sent[-1]['body']
    assert w.keys(_pending_cycle(w, T0 + 40, 24, 7), 'recover') == {'upload_pending'}  # draining again


def test_upload_rules_skipped_in_dry_run_upload(w):
    w.healthy(T0)
    w.state({'v1/game/1.json': {'phase': 'blocked', 'reason': 'upload_conflict'}})
    for i in range(6):
        assert _pending_cycle(w, T0 + 10 * i, 30 + i, 50 + 10 * i, dry=True) == []


def test_upload_oldest_intent_stalled(w):
    w.healthy(T0)
    w.state(upload_pending={'v1/game/1.json': 'put', 'v1/manifest.json': 'put'},
            upload_intents={'v1/game/1.json': {'action': 'put', 'revision': 7, 'sha256': 'a' * 64},
                            'v1/manifest.json': {'action': 'put', 'revision': 9, 'sha256': 'b' * 64}})
    assert w.run(T0) == []
    w.health(T0 + 50)
    assert w.run(T0 + 50) == []
    w.health(T0 + 61)
    assert w.keys(w.run(T0 + 61)) == {'upload_stalled'}
    # the old intent went out; only newer ones are queued -> recovers
    w.state(upload_pending={'v1/manifest.json': 'put'},
            upload_intents={'v1/manifest.json': {'action': 'put', 'revision': 12, 'sha256': 'b' * 64}})
    w.health(T0 + 70)
    assert w.keys(w.run(T0 + 70), 'recover') == {'upload_stalled'}


def test_upload_412_unknown_blocked_and_rejected(w):
    w.healthy(T0)
    w.state({'v1/game/1.json': {'phase': 'blocked', 'reason': 'upload_conflict'},
             'v1/game/2.json': {'phase': 'blocked', 'reason': 'interrupted_unknown'},
             'v1/game/3.json': {'phase': 'blocked', 'reason': 'upload_receipt_invalid'},
             'v1/setup/1.json': {'phase': 'rejected', 'reason': 'upload_rejected', 'sha256': 'a' * 64}})
    events = w.run(T0)
    assert w.keys(events) == {'upload_conflict', 'upload_unknown', 'upload_blocked', 'upload_rejected'}
    assert all(e['severity'] == 'critical' for e in events)
    body = w.fake.sent[-1]['body']
    assert '412' in body and '결과 모름' in body and '거절' in body
    assert len(w.fake.sent) == 1  # batched into one message


def test_upload_blocked_is_announced_outside_game_too(w):
    w.games(w.game(status='final'))
    w.state({'v1/game/1.json': {'phase': 'blocked', 'reason': 'upload_conflict'}})
    assert w.keys(w.run(T0)) == {'upload_conflict'}


def test_memory_and_disk(w):
    w.healthy(T0)
    w.memory, w.disk = 1.5 * GIB, 11 * GIB
    events = w.run(T0)
    assert w.keys(events) == {'disk_low'}  # memory held 60 s
    w.health(T0 + 61)
    assert w.keys(w.run(T0 + 61)) == {'memory_low'}


def test_profile_refresh_deadline(w):
    evening = datetime(2026, 10, 24, 12, 25, tzinfo=timezone.utc).timestamp()  # 21:25 KST, no game
    before = datetime(2026, 10, 24, 12, 10, tzinfo=timezone.utc).timestamp()   # 21:10 KST
    w.log.write_text('2026-10-23T12:01:00Z ok target=2026-10-23 cutoff=2026-10-22 version=x log=/p/a.log\n')
    assert w.run(before) == []
    events = w.run(evening)
    assert w.keys(events) == {'profile_refresh:2026-10-24'}
    assert 'log=' not in w.fake.sent[-1]['body']
    w.log.write_text(w.log.read_text() + '2026-10-24T12:26:00Z ok target=2026-10-24 cutoff=2026-10-23 log=/p/b\n')
    assert w.keys(w.run(evening + 120), 'recover') == {'profile_refresh:2026-10-24'}


def test_profile_failed_line(w):
    evening = datetime(2026, 10, 24, 12, 25, tzinfo=timezone.utc).timestamp()
    w.log.write_text('2026-10-24T12:16:38Z failed refresh TIMEOUT (guard wall cap) receipt=/Users/x/a.log log=/x\n')
    assert w.keys(w.run(evening)) == {'profile_refresh:2026-10-24'}
    body = w.fake.sent[-1]['body']
    assert 'failed refresh TIMEOUT' in body and '/' not in body.split('ok 아님:')[1]


def test_dedup_renotify_and_cooldown(w):
    w.healthy(T0)
    w.health(T0, published=T0 - 45)
    assert w.keys(w.run(T0)) == {'health_stalled'}
    # still stalled: no repeat inside renotify (warn 1800 s)
    for dt in (15, 30, 600):
        w.health(T0 + dt, published=T0 - 45 + 0)  # published_at frozen -> stays stalled, also publisher_down later
    assert w.keys(w.run(T0 + 15)) == set()
    # recover
    w.health(T0 + 20)
    assert w.keys(w.run(T0 + 20), 'recover') == {'health_stalled'}
    # flap inside cooldown (300 s): logged as suppressed, not sent
    sent = len(w.fake.sent)
    w.health(T0 + 100, published=T0 + 60)
    events = w.run(T0 + 100)
    assert w.keys(events) == set() and w.keys(events, 'suppressed') == {'health_stalled'}
    assert len(w.fake.sent) == sent
    # still stalled after the cooldown: now it is sent
    w.health(T0 + 330, published=T0 + 290)
    assert 'health_stalled' in w.keys(w.run(T0 + 330))


def test_renotify_while_it_lasts(w):
    w.healthy(T0)
    w.state({'v1/game/1.json': {'phase': 'blocked', 'reason': 'upload_conflict'}})
    assert w.keys(w.run(T0)) == {'upload_conflict'}
    moving = lambda t, seq: {str(PK): {'checked_at': iso(t), 'sequence_checked': seq, 'file_sequence': seq,  # noqa: E731
                                       'feed_ok': True}}
    w.health(T0 + 600, checks=moving(T0 + 600, 60))
    assert w.run(T0 + 600) == []
    w.health(T0 + 901, checks=moving(T0 + 901, 90))
    events = w.run(T0 + 901)
    assert w.keys(events, 'renotify') == {'upload_conflict'}
    assert '[계속]' in w.fake.sent[-1]['body']


def test_game_end_closes_silently(w):
    w.healthy(T0)
    w.health(T0, published=T0 - 45)
    w.run(T0)
    n = len(w.fake.sent)
    w.games(w.game(status='final'))
    w.game_file(status='final')
    events = w.run(T0 + 30)
    assert [e['kind'] for e in events] == ['closed']
    assert len(w.fake.sent) == n + 1 and '복구' not in w.fake.sent[-1]['body']  # only the watch_end notice
    assert '감시 끝' in w.fake.sent[-1]['body']
    lines = [json.loads(x) for x in (w.tmp / 'alert' / 'alerts.jsonl').read_text().splitlines()]
    assert [x['kind'] for x in lines].count('closed') == 1


def ops_status(now, state='running', **extra):
    """A body shaped like ws_operate.py writes it (docs/pm/service/ws-ops-status-contract.md)."""
    body = {'schema': 'pitcheezy-ws-ops-status-v1', 'written_at': iso(now), 'written_epoch': now, 'state': state,
            'state_reason': None, 'kst_date': datetime.fromtimestamp(now, wa.KST).date().isoformat(), 'run_dir': '/x',
            'games': [{'game_pk': PK, 'label': 'G1', 'start_utc': iso(T0 - 1320), 'final_seen_at': None}],
            'timeline': None, 'supervisor': {'pid': 1}, 'publisher': {'pid': 2, 'upload_enabled': False},
            'health': None, 'game_checks': {}, 'feed_ok': True, 'upload': {'enabled': False}, 'resources': {},
            'restarts': {'count': 0, 'max': 3, 'last_exit': None}, 'protected': {'ok': True}, 'alerts': [],
            'guard': None}
    body.update(extra)
    return body


def test_ops_contract_healthy_running_is_quiet_and_opens_window(w):
    # no publisher games file at all: the ops games list (start_utc) alone opens the window
    w.health(T0)
    w.game_file()
    w.state()
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    w.ops_path.write_text(json.dumps(ops_status(T0)))
    assert w.run(T0) == []
    assert [e['key'] for e in w.raw] == [f'watch_start:{PK}']


def test_ops_contract_states(w):
    w.healthy(T0)
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    w.ops_path.write_text(json.dumps(ops_status(T0, 'backoff', restarts={'count': 1, 'last_exit': {
        'why': 'exit_code_3'}})))
    assert w.keys(w.run(T0)) == {'ops_state:backoff', 'publisher_down'}
    assert 'exit_code_3' in w.fake.sent[-1]['body']
    w.ops_path.write_text(json.dumps(ops_status(T0 + 15, 'failed', state_reason='guard:disk_low')))
    w.health(T0 + 15)
    events = w.run(T0 + 15)
    assert w.keys(events) == {'ops_state:failed'} and w.keys(events, 'recover') == {'ops_state:backoff'}
    w.ops_path.write_text(json.dumps(ops_status(T0 + 30, 'refused', state_reason='supervisor_lock_held')))
    w.health(T0 + 30)
    assert 'ops_state:refused' in w.keys(w.run(T0 + 30))


def test_ops_contract_stale_heartbeat_and_alert_strings(w):
    w.healthy(T0)
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    w.ops_path.write_text(json.dumps(ops_status(T0 - 91, alerts=['memory_low', 'health_stale', 'feed_failing',
                                                                 'upload_blocked', 'protected_process_changed'])))
    events = w.run(T0)
    assert w.keys(events) == {'ops_status_stale', 'ops_problem:protected_process_changed'}
    assert [e['severity'] for e in events if e['key'] == 'ops_problem:protected_process_changed'] == ['critical']


def test_ops_waiting_allows_300s_and_is_watched_before_the_window(w):
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    later = T0 + 6 * 3600  # 15:30 KST, the game list says nothing is in window
    w.ops_path.write_text(json.dumps(ops_status(later - 290, 'waiting', games=[])))
    assert w.run(later) == []
    w.ops_path.write_text(json.dumps(ops_status(later - 310, 'waiting', games=[])))
    assert w.keys(w.run(later + 1)) == {'ops_status_stale'}


def test_ops_old_day_and_stopped_final_are_ignored(w):
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    old = ops_status(T0 - 3 * 86400, 'failed', state_reason='guard:disk_low', out_dir=str(w.tmp / 'gone'))
    w.ops_path.write_text(json.dumps(old))
    assert w.run(T0) == []
    w.ops_path.write_text(json.dumps(ops_status(T0 - 3600, 'stopped', state_reason='final_settled', games=[
        {'game_pk': PK, 'start_utc': iso(T0 - 1320), 'final_seen_at': iso(T0 - 3700)}])))
    assert w.run(T0 + 1) == []


def ops_v11(now, tmp, state='running', **extra):
    """v1.1 sample (additive fields the launcher adds, coordinator message 2026-10-18)."""
    body = ops_status(now, state, updated_at=iso(now), publisher={'running': state == 'running', 'pid': 2},
                      problems=[], out_dir=str(tmp / 'day-out'), state_path=str(tmp / 'day-out.publisher.state.json'),
                      games=[{'game_pk': PK, 'start': iso(T0 - 1320), 'status': 'live', 'window_start': None,
                              'window_end': None, 'upstream_changed_at': None}])
    body.pop('written_epoch')
    body.pop('written_at')
    body.update(extra)
    return body


def test_ops_v11_follows_out_dir_and_state_path(w):
    day = World(w.tmp / 'day')  # a second publisher folder: the one the launcher really runs today
    day.out = w.tmp / 'day-out'
    (day.out / 'v1' / 'games').mkdir(parents=True)
    (day.out / 'v1' / 'game').mkdir(parents=True)
    day.healthy(T0)
    (w.tmp / 'day-out.publisher.state.json').write_text(json.dumps(
        {'upload_attempts': {'v1/game/1.json': {'phase': 'blocked', 'reason': 'upload_conflict'}}}))
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    w.cfg['cli_out'] = False  # as if w.out came from env PITCHEEZY_PUBLISH_OUT_DIR, not --out
    w.ops_path.write_text(json.dumps(ops_v11(T0, w.tmp)))
    # env/config out_dir (w.out) is empty; the watcher reads day-out and its publisher.state.json instead
    events = w.run(T0)
    assert w.keys(events) == {'upload_conflict'}
    # an explicit --out still wins
    cfg = wa.load_config(None, out_dir=str(w.out), alert_dir=str(w.tmp / 'alert2'), env={})
    cfg['paths'].update(ops_status=str(w.ops_path), profile_log=None, disk_paths=[])
    obs = w.collect(cfg, T0)
    assert obs['out_dir'] == str(w.out)


def test_ops_v11_states_problems_and_staleness(w):
    w.healthy(T0)
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    w.ops_path.write_text(json.dumps(ops_v11(T0, w.tmp, out_dir=None, state_path=None)))
    assert w.run(T0) == []
    # refused during the window: urgent; urgent problem passes through; covered key is not repeated
    w.ops_path.write_text(json.dumps(ops_v11(T0 + 15, w.tmp, 'refused', out_dir=None, state_path=None,
        state_reason='prestart_refused', problems=[
            {'key': 'prestart_refused', 'severity': 'urgent', 'message': '시작 전 점검 거부', 'active': True},
            {'key': 'upload_blocked', 'severity': 'urgent', 'message': 'dup', 'active': True}])))
    w.health(T0 + 15)
    events = w.run(T0 + 15)
    assert w.keys(events) == {'ops_state:refused', 'ops_problem:prestart_refused', 'publisher_down'}
    assert all(e['severity'] == 'critical' for e in events)
    # running heartbeat older than 90 s -> stale
    w.ops_path.write_text(json.dumps(ops_v11(T0 - 100, w.tmp, out_dir=None, state_path=None)))
    w.health(T0 + 30)
    assert 'ops_status_stale' in w.keys(w.run(T0 + 30))


def test_ops_v11_postponed_game_is_not_a_window(w):
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    w.ops_path.write_text(json.dumps(ops_v11(T0, w.tmp, 'waiting', out_dir=None, state_path=None, games=[
        {'game_pk': PK, 'start': iso(T0 + 600), 'status': 'postponed'}])))
    assert w.run(T0) == [] and w.raw == []


def test_ops_hard_end_is_a_warning(w):
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    w.ops_path.write_text(json.dumps(ops_status(T0, 'stopped', state_reason='hard_end', games=[])))
    events = w.run(T0)
    assert w.keys(events) == {'ops_state:hard_end'} and events[0]['severity'] == 'warn'


def test_ops_status_tolerates_garbage(w):
    w.healthy(T0)
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    w.ops_path.write_text(json.dumps({'games': 'nope', 'alerts': [1, None, {'no_key': True}, ''], 'publisher': 3,
                                      'restarts': 'x', 'written_epoch': T0, 'state': 7}))
    assert w.run(T0) == []
    w.ops_path.write_text('{not json')
    w.run(T0 + 1)
    w.health(T0 + 31)
    assert w.keys(w.run(T0 + 31)) == {'ops_problem:unreadable'}


def test_file_stuck_not_in_warmup_or_delay(w):
    # B2: Warmup and rain delays are 'live' in the service, but the file may sit still for many minutes.
    w.healthy(T0)
    for text in ('Warmup', 'Delayed: Rain', 'Suspended: Rain'):
        w.game_file(text=text)
        for dt in (0, 400, 800):
            w.health(T0 + dt)
            assert w.run(T0 + dt) == [], (text, dt)
    # play starts: the clock restarted at the last Warmup/Delayed sighting (T0+800), not at the old file change
    w.game_file(text='In Progress')
    w.health(T0 + 900)
    assert w.run(T0 + 900) == []
    w.health(T0 + 1090)
    assert w.run(T0 + 1090) == []
    w.health(T0 + 1110)
    assert w.keys(w.run(T0 + 1110)) == {f'file_stuck:{PK}'}


def test_postponed_game_opens_no_window(w):
    w.games(w.game(status='postponed'))
    assert w.run(T0) == [] and w.raw == []


def test_upload_failed_code_is_not_game_unavailable(w):
    w.healthy(T0)
    for t in (T0, T0 + 31):
        w.health(t, unavailable=[{'game_pk': PK, 'code': 'upload_failed', 'status': 502}])
        assert w.run(t) == []


def test_window_open_and_close_notices(w):
    w.healthy(T0)
    w.run(T0)
    assert [e['key'] for e in w.raw] == [f'watch_start:{PK}'] and '감시 시작' in w.fake.sent[-1]['body']
    w.health(T0 + 15)
    w.run(T0 + 15)
    assert w.raw == []
    w.games(w.game(status='final'))
    w.game_file(status='final', text='Final')
    w.run(T0 + 30)
    assert [e['key'] for e in w.raw] == [f'watch_end:{PK}']


def test_journal_failure_does_not_stop_sends(w, tmp_path):
    w.healthy(T0)
    w.health(T0, published=T0 - 45)
    blocker = tmp_path / 'not-a-dir'
    blocker.write_text('x')
    journal = wa.JsonlNotifier(blocker / 'alerts.jsonl')  # parent is a file: every write fails
    wa.run_once(w.cfg, now=T0, senders=[w.fake], journal=journal, collect_fn=w.collect)
    assert 'health_stalled' not in str([]) and '발행 정체' in w.fake.sent[-1]['body']


def test_missing_out_dir_is_reported(tmp_path):
    cfg = wa.load_config(None, alert_dir=str(tmp_path / 'a'), env={})
    cfg['paths']['profile_log'] = None
    cfg['paths']['disk_paths'] = []
    fake = wa.FakeNotifier()
    events = wa.run_once(cfg, now=T0, senders=[fake], journal=wa.JsonlNotifier(tmp_path / 'a' / 'j.jsonl'),
                         collect_fn=partial(wa.collect, memory=lambda: None, disk=lambda p: {}))
    assert {e['key'] for e in events} == {'watch_config'}


def test_dry_run_writes_and_sends_nothing(w, capsys):
    w.healthy(T0)
    w.health(T0, published=T0 - 200)
    events = wa.run_once(w.cfg, now=T0, senders=[w.fake], dry_run=True, collect_fn=w.collect)
    assert {e['key'] for e in events if e['kind'] == 'fire'} == {'health_stalled', 'publisher_down'}
    assert w.fake.sent == []
    assert not (w.tmp / 'alert').exists()
    assert 'would_send' in capsys.readouterr().out


def test_failing_notifier_is_logged_not_raised(w):
    class Boom:
        name = 'boom'

        def send(self, *a, **k):
            raise OSError('offline')
    w.healthy(T0)
    w.health(T0, published=T0 - 45)
    wa.run_once(w.cfg, now=T0, senders=[Boom(), w.fake], journal=w.journal, collect_fn=w.collect)
    assert len(w.fake.sent) == 1
    kinds = [json.loads(x)['kind'] for x in (w.tmp / 'alert' / 'alerts.jsonl').read_text().splitlines()]
    assert 'notify_error' in kinds and 'notified' in kinds


def test_notifier_selection_defaults_and_env(tmp_path):
    cfg = wa.load_config(None, alert_dir=str(tmp_path), env={})
    names = lambda **k: [s.name for s in wa.build_notifiers(cfg, **k)[0]]  # noqa: E731
    assert names(push=True, env={}) == ['macos']  # phone push off by default
    assert names(push=False, env={'PITCHEEZY_ALERT_PUSH': 'ntfy', 'PITCHEEZY_ALERT_NTFY_TOPIC': 't'}) == ['macos']
    env = {'PITCHEEZY_ALERT_PUSH': 'ntfy,imessage', 'PITCHEEZY_ALERT_NTFY_TOPIC': 't',
           'PITCHEEZY_ALERT_IMESSAGE_TO': 'x', 'PITCHEEZY_ALERT_MACOS': '0'}
    assert names(push=True, env=env) == ['ntfy', 'imessage']
    _, problems = wa.build_notifiers(cfg, push=True, env={'PITCHEEZY_ALERT_PUSH': 'ntfy'})
    assert problems
    assert names(push=True, names='fake', env=env) == ['fake']


def test_ntfy_payload_without_network():
    seen = {}

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(request, timeout):
        seen['url'], seen['body'], seen['auth'] = request.full_url, json.loads(request.data), request.headers
        return Resp()
    wa.NtfyNotifier('topic-xyz', 'https://ntfy.example', token='tok', opener=opener).send('제목', '본문', critical=True)
    assert seen['url'] == 'https://ntfy.example/'
    assert seen['body']['topic'] == 'topic-xyz' and seen['body']['priority'] == 5 and seen['body']['title'] == '제목'
    assert seen['auth']['Authorization'] == 'Bearer tok'


def test_shipped_config_loads_and_matches_runbook_numbers():
    cfg = wa.load_config(None, env={})
    th = cfg['thresholds']
    assert (th['health_stale_s'], th['feed_down_hold_s'], th['feed_lag_s'], th['memory_available_gib'],
            th['disk_free_gib'], th['profile_deadline_kst']) == (30, 60, 45, 2.0, 12.5, '21:20')


def test_file_stuck_ignores_clock_only_live_from_ops(w):
    # The launcher's games[].status 'live' only means the scheduled start passed. Without a publisher word on the
    # game (no game block, no publisher games list) the file is not judged 'stuck', however long it sits.
    w.health(T0)
    (w.out / 'v1' / 'game' / f'{PK}.json').write_text(json.dumps({'game_pk': PK, 'publish': {'sequence': 5}}))
    w.state()
    w.cfg['paths']['ops_status'] = str(w.ops_path)
    for dt in (0, 400, 800):
        w.ops_path.write_text(json.dumps(ops_v11(T0 + dt, w.tmp, out_dir=None, state_path=None)))
        w.health(T0 + dt)
        assert w.run(T0 + dt) == [], dt
    # the publisher's games list says live: now the same unchanged file_sequence is stuck after 300 s
    w.games(w.game())
    for dt in (900, 1210):
        w.ops_path.write_text(json.dumps(ops_v11(T0 + dt, w.tmp, out_dir=None, state_path=None)))
        w.health(T0 + dt)
        events = w.run(T0 + dt)
    assert w.keys(events) == {f'file_stuck:{PK}'}
