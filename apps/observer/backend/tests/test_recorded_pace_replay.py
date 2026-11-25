"""Recorded-pace replay (rehearsal only): bodies at their recorded receipt times on today's clock, event walls
moved by the same shift, so ``LiveFeed``'s viewer delay can be rehearsed on a finished capture."""
import gzip
import json

import pytest

from observer_app.live_feed import LiveFeed, LiveFeedError, RecordedPaceReplay, replay_config

GAME = 777001
URL = f'https://x/api/v1.1/game/{GAME}/feed/live'


def folder(tmp_path, rows, **manifest):
    """rows: (fetched, event_wall) per body; body k carries ``k`` so the test can tell them apart."""
    files = []
    for k, (fetched, event) in enumerate(rows):
        name = f'{GAME}_{k:06d}.json.gz'
        with gzip.open(tmp_path/name, 'wt') as handle:
            json.dump({'gameData': {'game': {'pk': GAME}}, 'metaData': {'timeStamp': '20261010_000000'}, 'k': k}, handle)
        files.append({'file': name, 'fetched': fetched, 'event_wall': event})
    (tmp_path/'MANIFEST.json').write_text(json.dumps({'status': 'OK', 'game_pk': GAME, 'mode': {'bodies_modified': False},
                                                      'files': files} | manifest))
    return tmp_path


def test_bodies_come_at_their_recorded_spacing_with_event_walls_moved_by_the_same_shift(tmp_path):
    now = [5000.0]
    replay = RecordedPaceReplay(folder(tmp_path, [(100.0, 90.0), (120.0, 103.0), (125.0, 121.0)]), wall=lambda: now[0])
    first = replay(URL, 1)
    assert (first['k'], first['metaData']['eventWall']) == (0, 4990.0)  # receipt 5000, lag 10 kept
    now[0] = 5019.9
    assert replay(URL, 1) is first  # not due yet: the unchanged body again, never a skip ahead
    now[0] = 5020.0
    second = replay(URL, 1)
    assert (second['k'], second['metaData']['eventWall']) == (1, 5003.0)  # due 5020, lag 17
    now[0] = 5100.0  # polled late: still one body per request, in order, stamped with its DUE time not the late poll
    third = replay(URL, 1)
    assert (third['k'], third['metaData']['eventWall']) == (2, 5021.0)
    assert replay(URL, 1) is third and replay.position() == {'served': 3, 'bodies': 3, 'duration_s': 25.0, 'elapsed_s': 100.0}


def test_long_gaps_are_shortened_and_later_bodies_keep_their_own_lag(tmp_path):
    now = [0.0]
    replay = RecordedPaceReplay(folder(tmp_path, [(100.0, 100.0), (14500.0, 14490.0), (14520.0, 14500.0)]),
                                max_gap_s=30.0, wall=lambda: now[0])
    replay(URL, 1)
    now[0] = 29.0
    assert replay(URL, 1)['k'] == 0
    now[0] = 30.0
    assert replay(URL, 1)['metaData']['eventWall'] == 20.0  # due 30 (gap cut to 30), lag 10
    now[0] = 50.0
    assert replay(URL, 1)['metaData']['eventWall'] == 30.0  # due 50, lag 20


def test_a_viewer_delay_holds_each_recorded_state_until_its_event_is_that_old(tmp_path):
    now = [1000.0]
    replay = RecordedPaceReplay(folder(tmp_path, [(100.0, 95.0), (120.0, 110.0)]), wall=lambda: now[0])
    feed = LiveFeed(replay, config=replay_config('0'), delay_s=0, clock=lambda: now[0], wall=lambda: now[0])
    seen = {}
    for now[0] in (1000.0, 1010.0, 1020.0, 1024.0, 1025.0, 1039.0, 1040.0):
        for delay in (0, 30):
            entry, _ = feed.snapshot(GAME, delay)
            seen[now[0], delay] = None if entry is None else entry['feed']['k']
    # delay 0: on receipt (1000, 1020). delay 30: 30 s after the EVENT (995+30, 1010+30), not after the receipt.
    assert [seen[t, 0] for t in (1000.0, 1010.0, 1020.0, 1040.0)] == [0, 0, 1, 1]
    assert [seen[t, 30] for t in (1000.0, 1024.0, 1025.0, 1039.0, 1040.0)] == [None, None, 0, 0, 1]


def test_a_modified_or_failed_folder_is_refused(tmp_path):
    for bad in ({'status': 'FAILED_BODY_CHECK'}, {'mode': {'bodies_modified': True}}):
        with pytest.raises(LiveFeedError):
            RecordedPaceReplay(folder(tmp_path, [(1.0, 1.0)], **bad))


def test_the_env_switch_builds_the_paced_player_only_for_a_replay(tmp_path, monkeypatch):
    from observer_app.main import create_app
    from starlette.testclient import TestClient
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', str(folder(tmp_path, [(100.0, 95.0)])))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_PACE', 'recorded')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_MAX_GAP_S', '20')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_STORE', str(tmp_path/'store.sqlite3'))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '0')
    with TestClient(create_app(object())) as client:
        client.get('/api/live/games', params={'date': '2026-10-10'})
        health = client.get('/api/health').json()['live_store']
    assert health['source'] == 'replay_poll' and health['replay_position']['bodies'] == 1


def test_first_requests_arriving_together_build_one_feed(tmp_path, monkeypatch):
    """Sync routes run in a thread pool: two first requests must not each build a feed (two transports would replay
    or poll twice and store one body as two versions)."""
    import threading
    import time
    from observer_app import live_feed as module
    from observer_app.main import create_app
    from starlette.testclient import TestClient
    built = []

    class Slow(module.RecordedPaceReplay):
        def __init__(self, *args, **kwargs):
            built.append(1)
            time.sleep(0.2)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(module, 'RecordedPaceReplay', Slow)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', str(folder(tmp_path, [(100.0, 95.0)])))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_PACE', 'recorded')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '0')
    with TestClient(create_app(object())) as client:
        threads = [threading.Thread(target=client.get, args=('/api/live/games',), kwargs={'params': {'date': '2026-10-10'}}) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert built == [1]
