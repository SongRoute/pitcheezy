"""Service API v2 + shared demo clock on synthetic games (no network, no T7)."""
import gzip
import json

import pytest
from fastapi.testclient import TestClient

from observer_app import game_history
from observer_app.main import create_app

GAME, TWIN, LATE = 900101, 900102, 900103


class Wall:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def play(ab, inning, top, batter, pitcher, outcome, away, home, outs_after):
    """Two pitches: a ball, then the PA-ending pitch."""
    final = {'strikeout': ('S', 'Swinging Strike', None), 'single': ('X', 'In play, no out', {'location': '8', 'trajectory': 'line_drive'}),
             'field_out': ('X', 'In play, out(s)', {'location': '4', 'trajectory': 'ground_ball'}),
             'home_run': ('E', 'In play, run(s)', {'trajectory': 'fly_ball'})}[outcome]
    events = [{'isPitch': True, 'pitchNumber': 1, 'index': 0, 'details': {'call': {'code': 'B'}, 'description': 'Ball', 'type': {'code': 'FF'}},
               'count': {'balls': 1, 'strikes': 0, 'outs': outs_after-(outcome != 'single' and outcome != 'home_run')},
               'pitchData': {'startSpeed': 95.0, 'coordinates': {'pX': 1.2, 'pZ': 2.0}}},
              {'isPitch': True, 'pitchNumber': 2, 'index': 1, 'details': {'call': {'code': final[0]}, 'description': final[1], 'type': {'code': 'SL'}},
               'count': {'balls': 1, 'strikes': 1, 'outs': outs_after}, 'pitchData': {'startSpeed': 86.0, 'coordinates': {'pX': 0.1, 'pZ': 2.5}}}
              | ({'hitData': final[2]} if final[2] else {})]
    credits = [{'position': {'code': '4'}, 'credit': 'f_assist'}, {'position': {'code': '3'}, 'credit': 'f_putout'}] if outcome == 'field_out' else []
    return {'about': {'atBatIndex': ab-1, 'inning': inning, 'isTopInning': top, 'isComplete': True},
            'matchup': {'batter': {'id': batter, 'fullName': f'B{batter}'}, 'pitcher': {'id': pitcher, 'fullName': f'P{pitcher}'},
                        'batSide': {'code': 'R'}, 'pitchHand': {'code': 'R'}},
            'result': {'eventType': outcome, 'event': outcome, 'description': f'text {ab}', 'awayScore': away, 'homeScore': home},
            'count': {'outs': outs_after}, 'playEvents': events,
            'runners': [{'movement': {'originBase': None}, 'details': {'runner': {'id': batter}}, 'credits': credits}]}


def feed(pk, start, *, dh='N', number=1):
    plays = [play(1, 1, True, 11, 21, 'strikeout', 0, 0, 1), play(2, 1, True, 12, 21, 'single', 0, 0, 1),
             play(3, 1, True, 13, 21, 'field_out', 0, 0, 3), play(4, 1, False, 31, 41, 'home_run', 0, 1, 0),
             play(5, 1, False, 32, 41, 'strikeout', 0, 1, 1)]
    return {'gameData': {'game': {'pk': pk, 'type': 'F', 'doubleHeader': dh, 'gameNumber': number},
                         'datetime': {'dateTime': start, 'officialDate': start[:10]},
                         'status': {'abstractGameState': 'Final', 'detailedState': 'Final'},
                         'teams': {'away': {'name': 'Away Club', 'abbreviation': 'AWY'}, 'home': {'name': 'Home Club', 'abbreviation': 'HOM'}},
                         'players': {}},
            'liveData': {'plays': {'allPlays': plays}, 'linescore': {'teams': {'away': {'runs': 0}, 'home': {'runs': 1}}}}}


def watch(pk, *, ready=True):
    decisions = []
    for i, (ab, n) in enumerate((ab, n) for ab in range(1, 6) for n in (1, 2)):
        pre = ({'status': 'ready', 'reason': None, 'recommendation': {'candidates': [
            {'rank': 1, 'pitch_type': 'SL', 'pitch_label': '슬라이더', 'zone_id': 'low_right', 'zone_label': '낮은 오른쪽',
             'target': {'x': 0.5, 'z': 1.9}, 'detail': {'probability': 0.4}}]}}
               if ready and i != 3 else {'status': 'unsupported', 'reason': '지원하지 않는 투수', 'recommendation': None})
        decisions.append({'index': i, 'pa_id': f'{pk}:{ab}', 'at_bat_number': ab, 'pitch_number': n,
                          'situation': {'inning': 1, 'half': 'Top' if ab <= 3 else 'Bot', 'outs': 0, 'balls': n-1, 'strikes': 0, 'bases': 0,
                                        'home_score': 1 if ab == 5 else 0, 'away_score': 0},
                          'pitcher': {}, 'batter': {}, 'status': 'SUPPORTED', 'pre': pre, 'actual': {}, 'we': {}})
    return {'schema': 'pitcheezy-watch-along-v1', 'badge': '', 'game': {'game_pk': pk, 'date': '2026-09-29', 'game_type': 'F', 'away_team': 'Away Club', 'home_team': 'Home Club', 'final': {'away': 0, 'home': 1}}, 'policy': {'name': 'ARM-B', 'identity_sha256': 'abcdef0123'},
            'location': {'zone_bounds': {'bottom': 1.6, 'top': 3.4}}, 'we': {}, 'coverage': {'pitch_decisions': 10, 'ready': 9, 'ready_share_of_pitches': 0.9}, 'generated_utc': '20260930T084853Z',
            'decisions': decisions}


@pytest.fixture
def env(tmp_path):
    feeds, wdir, demo, intent = tmp_path/'feeds', tmp_path/'watch', tmp_path/'demo', tmp_path/'intent'
    for d in (feeds, wdir, demo/'mappings', intent):
        d.mkdir(parents=True)
    for pk, start, dh, number in ((GAME, '2026-09-30T02:00:00Z', 'S', 1), (TWIN, '2026-09-30T06:30:00Z', 'S', 2),
                                  (LATE, '2026-09-29T15:30:00Z', 'N', 1)):  # LATE: 00:30 KST on 9/30
        with gzip.open(feeds/f'{pk}_final.json.gz', 'wt') as handle:
            json.dump(feed(pk, start, dh=dh, number=number), handle)
    (wdir/f'{GAME}.json').write_text(json.dumps(watch(GAME)))
    rows = [{'index': i, 'pitch_key': key, 'reveal_at_s': 10.0*i} for i, key in
            enumerate(f'{GAME}:{ab}:{n}' for ab in range(1, 6) for n in (1, 2))]
    (demo/'mappings'/'fx.json').write_text(json.dumps({'schema': 'pitcheezy-demo-mapping-v1', 'mapping_id': 'fx', 'game_pk': GAME,
                                                       'kind': 'test_fixture', 'verified': False, 'video': None, 'rows': rows}))
    wall = Wall()

    def make():
        return create_app(object(), start_worker=False, watch_dir=wdir, feed_dir=feeds, demo_dir=demo, intent_dir=intent,
                          demo_wall=wall, live_feed=False)
    return make, wall, demo


def token(demo):
    return (demo/'operator.token').read_text().strip()


def start_demo(client, demo, start_index=4):
    client.get('/api/service/dates')  # nothing creates the token before the first operator call
    from observer_app.demo_clock import DemoClock
    DemoClock(demo).token()
    head = {'X-Pitcheezy-Operator': token(demo)}
    made = client.post('/api/service/operator/sessions', json={'game_pk': GAME, 'mapping_id': 'fx', 'start_index': start_index}, headers=head)
    assert made.status_code == 201, made.text
    return made.json(), head


def control(client, head, sid, **body):
    response = client.post(f'/api/service/operator/sessions/{sid}/control', json=body, headers=head)
    return response


def test_kst_dates_doubleheader_and_midnight(env):
    make, _, _ = env
    with TestClient(make()) as client:
        listed = client.get('/api/service/games', params={'date': '2026-09-30'}).json()
        assert [g['game_pk'] for g in listed['games']] == [LATE, GAME, TWIN]
        late, first, second = listed['games']
        assert late['start_kst'] == '2026-09-30T00:30:00+09:00' and late['official_date'] == '2026-09-29'
        assert (first['double_header'], first['game_number'], second['game_number']) == (True, 1, 2)
        assert first['start_kst'] == '2026-09-30T11:00:00+09:00' and first['away_score'] == 0 and first['home_score'] == 1
        assert client.get('/api/service/games', params={'date': '2026-09-29'}).json()['games'] == []
        assert client.get('/api/service/dates').json()['dates'] == ['2026-09-30']
        assert client.get('/api/service/games', params={'date': '9/30'}).status_code == 400


def test_archive_detail_full_history_grounded_results_and_provenance(env):
    make, _, _ = env
    with TestClient(make()) as client:
        view = client.get(f'/api/service/games/{GAME}').json()
        assert view['source']['kind'] == 'archive' and view['next'] is None and view['current']['phase'] == 'final'
        assert len(view['pitches']) == 10 and len(view['pas']) == 5 and all(pa['complete'] for pa in view['pas'])
        assert [pa['result_text'] for pa in view['pas']] == ['헛스윙 삼진', '중견수 방면 1루타', '2루수 땅볼 아웃', '홈런', '헛스윙 삼진']
        assert view['pitches'][0]['rec']['provenance'] == 'asof_replay'
        assert view['pitches'][3]['rec']['status'] == 'unsupported' and view['pitches'][3]['rec']['candidates'] == []
        twin = client.get(f'/api/service/games/{TWIN}').json()  # no stored recs: absent, none invented
        assert {p['rec']['provenance'] for p in twin['pitches']} == {'absent'}
        assert all(p['rec']['candidates'] == [] for p in twin['pitches'])


def test_operator_writes_need_token(env):
    make, _, demo = env
    with TestClient(make()) as client:
        assert client.post('/api/service/operator/sessions', json={'game_pk': GAME, 'mapping_id': 'fx'}).status_code == 401
        assert client.get('/api/service/operator/catalog', headers={'X-Pitcheezy-Operator': 'nope'}).status_code == 401


def test_two_clients_refresh_pause_resume_seek_reset(env):
    make, wall, demo = env
    with TestClient(make()) as a, TestClient(make()) as b:
        session, head = start_demo(a, demo, start_index=4)
        sid = session['session_id']
        # pre-segment state: pitches 0..3 (PA 1-2 complete) visible before start
        for client in (a, b):
            view = client.get(f'/api/service/games/{GAME}', params={'demo': sid}).json()
            assert view['cutoff']['index'] == 3 and [pa['complete'] for pa in view['pas']] == [True, True, False]
            assert view['game']['date_kst'] == '2026-09-30' and view['game']['status'] == 'live'
            assert view['next']['rec']['status'] == 'ready'
        assert control(a, head, sid, action='start').json()['status'] == 'running'
        wall.now += 5.0  # start progress = 40-5 → 40 reveals pitch 4
        assert b.get(f'/api/service/demo/{sid}').json()['cutoff']['index'] == 4
        assert a.get(f'/api/service/games/{GAME}', params={'demo': sid}).json()['cutoff']['index'] == 4
        paused = control(a, head, sid, action='pause', expected_version=2).json()
        wall.now += 100
        assert b.get(f'/api/service/demo/{sid}').json()['cutoff']['index'] == 4
        assert control(a, head, sid, action='pause', expected_version=2).status_code == 409  # stale version
        control(a, head, sid, action='resume', op_id='r1')
        again = control(a, head, sid, action='resume', op_id='r1')  # idempotent replay
        assert again.status_code == 200 and again.json()['version'] == paused['version']+1
        wall.now += 10
        assert b.get(f'/api/service/demo/{sid}').json()['cutoff']['index'] == 5
        control(a, head, sid, action='seek', index=8)
        assert b.get(f'/api/service/games/{GAME}', params={'demo': sid}).json()['cutoff']['index'] == 8
        control(a, head, sid, action='seek', index=5)  # seek back hides later pitches for every client
        view = b.get(f'/api/service/games/{GAME}', params={'demo': sid}).json()
        assert view['cutoff']['index'] == 5 and len(view['pitches']) == 6
        assert control(a, head, sid, action='seek', index=1).status_code == 400  # before segment start
        control(a, head, sid, action='rate', rate=2.0)
        wall.now += 5  # 2x: 10 s of mapping time
        assert a.get(f'/api/service/demo/{sid}').json()['cutoff']['index'] == 6
        reset = control(a, head, sid, action='reset').json()
        assert reset['status'] == 'ready' and reset['rate'] == 1.0 and reset['cutoff']['index'] == 3


def test_reconnect_after_server_restart_keeps_state(env):
    make, wall, demo = env
    with TestClient(make()) as client:
        session, head = start_demo(client, demo)
        control(client, head, session['session_id'], action='start')
        wall.now += 25
        before = client.get(f'/api/service/demo/{session["session_id"]}').json()
    with TestClient(make()) as fresh:  # new process, new device: same persisted clock
        after = fresh.get(f'/api/service/demo/{session["session_id"]}').json()
    assert before['cutoff'] == after['cutoff'] and after['status'] == 'running'


def test_no_future_leak_on_any_surface(env):
    make, wall, demo = env
    with TestClient(make()) as client:
        session, head = start_demo(client, demo, start_index=4)
        sid = session['session_id']
        view = client.get(f'/api/service/games/{GAME}', params={'demo': sid})
        assert view.headers['cache-control'] == 'no-store'
        body = view.json()
        text = json.dumps(body)
        assert max(p['index'] for p in body['pitches']) == 3
        assert all(f'{GAME}:{ab}:' not in text or ab <= 3 for ab in range(4, 6))  # PA 4/5 pitches/keys absent
        assert '홈런' not in text and 'text 4' not in text and body['game']['home_score'] == 0
        assert body['pas'][-1] == body['pas'][-1] | {'complete': False, 'event': None, 'result_text': None}
        assert client.get(f'/api/service/games/{GAME}', params={'demo': sid, 'after': 3}).json()['pitches'] == []
        # list in demo namespace: score at cutoff, no final
        card = client.get('/api/service/games', params={'date': '2026-09-30', 'demo': sid}).json()['games']
        assert [c['game_pk'] for c in card] == [GAME] and card[0]['home_score'] == 0 and card[0]['status'] == 'live'
        # archive / raw paths of the embargoed game
        assert client.get(f'/api/service/games/{GAME}').status_code == 423
        for path in (f'/api/watch/{GAME}', f'/api/watch/{GAME}/reveal/9', f'/api/watch/{GAME}/summary?through=9',
                     f'/api/watch/{GAME}/setup', f'/api/replay/{GAME}/outline', f'/api/live/{GAME}/state'):
            assert client.get(path).status_code == 423, path
        listed = client.get('/api/service/games', params={'date': '2026-09-30'}).json()['games']
        hidden = next(c for c in listed if c['game_pk'] == GAME)
        assert hidden['embargoed'] and hidden['home_score'] is None and hidden['status'] == 'other'
        assert all(g['game_pk'] != GAME for g in client.get('/api/watch/games').json()['games'])
        # other game ids / bogus sessions in demo namespace never fall back to the archive
        assert client.get(f'/api/service/games/{TWIN}', params={'demo': sid}).status_code == 404
        assert client.get(f'/api/service/games/{GAME}', params={'demo': '0'*32}).status_code == 404
        assert client.get(f'/api/service/games/{GAME}', params={'demo': '../x'}).status_code == 404
        # ending lifts the embargo; the frozen demo view stays readable
        control(client, head, sid, action='end')
        assert client.get(f'/api/service/games/{GAME}').status_code == 200
        assert client.get(f'/api/service/games/{GAME}', params={'demo': sid}).json()['cutoff']['index'] == 3


def test_demo_final_only_after_last_reveal(env):
    make, wall, demo = env
    with TestClient(make()) as client:
        session, head = start_demo(client, demo, start_index=0)
        sid = session['session_id']
        control(client, head, sid, action='seek', index=8)
        body = client.get(f'/api/service/games/{GAME}', params={'demo': sid}).json()
        assert body['game']['status'] == 'live' and body['pas'][-1]['complete'] is False and body['next'] is not None
        control(client, head, sid, action='seek', index=9)
        body = client.get(f'/api/service/games/{GAME}', params={'demo': sid}).json()
        assert body['game']['status'] == 'final' and body['game']['home_score'] == 1 and body['next'] is None
        assert body['pas'][-1]['result_text'] == '헛스윙 삼진'


def test_empty_next_rec_is_stated_not_generated(env):
    make, wall, demo = env
    with TestClient(make()) as client:
        session, head = start_demo(client, demo, start_index=2)  # cutoff 1 → next pitch index 2? use seek to 2
        sid = session['session_id']
        control(client, head, sid, action='seek', index=2)
        body = client.get(f'/api/service/games/{GAME}', params={'demo': sid}).json()
        assert body['next']['rec']['status'] == 'unsupported' and body['next']['rec']['candidates'] == []


def test_mapping_must_match_record_keys(env, tmp_path):
    make, _, demo = env
    bad = json.loads((demo/'mappings'/'fx.json').read_text())
    bad['mapping_id'] = 'bad'
    bad['rows'][2]['pitch_key'] = f'{GAME}:9:9'
    (demo/'mappings'/'bad.json').write_text(json.dumps(bad))
    with TestClient(make()) as client:
        from observer_app.demo_clock import DemoClock
        head = {'X-Pitcheezy-Operator': DemoClock(demo).token()}
        response = client.post('/api/service/operator/sessions', json={'game_pk': GAME, 'mapping_id': 'bad'}, headers=head)
        assert response.status_code == 503


def test_result_text_never_invents_fielder():
    bare = play(1, 1, True, 1, 2, 'single', 0, 0, 0)
    bare['playEvents'][-1].pop('hitData')
    assert game_history.result_text(bare) == '1루타'


def test_segment_end_keeps_cutoff_and_embargo(env):
    make, wall, demo = env
    short = json.loads((demo/'mappings'/'fx.json').read_text())
    short['mapping_id'], short['rows'] = 'short', short['rows'][4:8]
    (demo/'mappings'/'short.json').write_text(json.dumps(short))
    with TestClient(make()) as client:
        from observer_app.demo_clock import DemoClock
        head = {'X-Pitcheezy-Operator': DemoClock(demo).token()}
        sid = client.post('/api/service/operator/sessions', json={'game_pk': GAME, 'mapping_id': 'short'}, headers=head).json()['session_id']
        control(client, head, sid, action='start')
        wall.now += 10_000  # far past the segment's last row: the clock stops at row 7, never the game end
        body = client.get(f'/api/service/games/{GAME}', params={'demo': sid}).json()
        assert body['cutoff']['index'] == 7 and body['game']['status'] == 'live' and body['game']['home_score'] == 1
        assert all(p['index'] <= 7 for p in body['pitches']) and 'text 5' not in json.dumps(body)
        assert client.get(f'/api/service/games/{GAME}').status_code == 423
        control(client, head, sid, action='pause')
        assert client.get(f'/api/service/demo/{sid}').json()['cutoff']['index'] == 7


class FakeLive:
    """Stands in for LiveFeed: one snapshot of an in-progress game."""
    failing = {}

    def __init__(self, snapshot):
        self.snap = snapshot

    def snapshot(self, game_pk, delay):
        return {'feed': self.snap, 'event': 0, 'fetched': 1_800_000_000.0}, 0.0

    def latest(self, game_pk):
        return self.snap

    def schedule(self, day):
        return [{'game_pk': 900200, 'start': '2026-10-05T10:00:00Z', 'status': 'In Progress', 'live': True, 'abstract': 'Live',
                 'away_team': 'Away Club', 'home_team': 'Home Club', 'away_score': 0, 'home_score': 1,
                 'game_number': 1, 'double_header': 'N'}] if day == '2026-10-05' else []


def test_live_game_lists_and_serves_only_occurred_pitches(tmp_path):
    snap = feed(900200, '2026-10-05T10:00:00Z')
    snap['gameData']['status'] = {'abstractGameState': 'Live', 'detailedState': 'In Progress'}
    current = snap['liveData']['plays']['allPlays'][-1]
    current['about']['isComplete'] = False
    current['result'] = {}
    current['playEvents'] = current['playEvents'][:1]
    current['count'] = {'balls': 1, 'strikes': 0, 'outs': 0}
    snap['liveData']['linescore'] |= {'currentInning': 1, 'isTopInning': False, 'outs': 0, 'inningState': 'Bottom',
                                      'offense': {'batter': {'id': 32}}, 'defense': {'pitcher': {'id': 41}}}
    snap['liveData']['plays']['currentPlay'] = current
    app = create_app(object(), start_worker=False, watch_dir=tmp_path, feed_dir=tmp_path, demo_dir=tmp_path/'demo',
                     intent_dir=tmp_path, live_feed=FakeLive(snap))
    import os
    os.environ['PITCHEEZY_OBSERVER_LIVE_ARMB'] = '0'
    try:
        with TestClient(app) as client:
            listed = client.get('/api/service/games', params={'date': '2026-10-05'}).json()
            assert listed['games'][0]['live'] and listed['games'][0]['start_kst'] == '2026-10-05T19:00:00+09:00'
            view = client.get('/api/service/games/900200').json()
            assert view['source']['kind'] == 'live' and view['game']['status'] == 'live'
            assert len(view['pitches']) == 9 and view['pas'][-1]['complete'] is False
            assert view['current']['phase'] == 'pre_pitch' and view['current']['situation']['balls'] == 1
            assert view['next']['rec']['provenance'] == 'absent' and view['next']['rec']['candidates'] == []
            assert view['feed']['as_of'] == '2027-01-15T08:00:00Z'
            # the screen is in the open plate appearance: its key is the one its pitches carry
            assert view['current']['pa_key'] == view['pitches'][-1]['pa_key'] == '900200:5'
            # the batter is done and the next one is up with no pitch yet: a new PA, not the previous batter's
            current['about']['isComplete'] = True
            snap['liveData']['linescore']['offense'] = {'batter': {'id': 33}}
            view = client.get('/api/service/games/900200').json()
            assert view['current']['phase'] == 'pre_pitch' and view['current']['situation']['balls'] == 0
            assert view['current']['pa_key'] == '900200:6' and view['pitches'][-1]['pa_key'] == '900200:5'
            snap['liveData']['linescore'] |= {'outs': 3, 'inningState': 'Middle'}  # inning break: no pending PA
            view = client.get('/api/service/games/900200').json()
            assert view['current']['phase'] == 'between_innings' and view['current']['pa_key'] is None
    finally:
        os.environ.pop('PITCHEEZY_OBSERVER_LIVE_ARMB')


def _mapping_script():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2]/'scripts'/'build_demo_mapping.py'
    spec = importlib.util.spec_from_file_location('build_demo_mapping', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mapping_script_feed_pacing_is_never_verified_and_blank_csv_refused(tmp_path):
    script = _mapping_script()
    feeds = tmp_path/'feeds'
    feeds.mkdir()
    snap = feed(GAME, '2026-09-30T02:00:00Z')
    for i, p in enumerate(e for pl in snap['liveData']['plays']['allPlays'] for e in pl['playEvents']):
        p['endTime'] = f'2026-09-30T02:{10+i:02d}:00Z'
    with gzip.open(feeds/f'{GAME}_final.json.gz', 'wt') as handle:
        json.dump(snap, handle)
    script.main(['--game', str(GAME), '--id', 'pace', '--first-ab', '2', '--last-ab', '3', '--from-feed',
                 '--feed-dir', str(feeds), '--demo-dir', str(tmp_path/'demo')])
    made = json.loads((tmp_path/'demo'/'mappings'/'pace.json').read_text())
    assert made['kind'] == 'game_timestamps_unsynced' and made['verified'] is False
    assert [r['index'] for r in made['rows']] == [2, 3, 4, 5] and made['rows'][0]['reveal_at_s'] == 5.0
    blank = tmp_path/'t.csv'
    blank.write_text('pitch_id,video_offset_result_visible_s,verified_by\n900101:2:1,,\n')
    with pytest.raises(SystemExit):
        script.main(['--game', str(GAME), '--id', 'v', '--from-csv', str(blank), '--feed-dir', str(feeds), '--demo-dir', str(tmp_path/'demo')])
    filled = tmp_path/'f.csv'
    filled.write_text('pitch_id,video_offset_result_visible_s,verified_by\n900101:2:1,12.5,\n900101:2:2,30,\n')
    script.main(['--game', str(GAME), '--id', 'v2', '--from-csv', str(filled), '--feed-dir', str(feeds), '--demo-dir', str(tmp_path/'demo')])
    unverified = json.loads((tmp_path/'demo'/'mappings'/'v2.json').read_text())
    assert unverified['verified'] is False and unverified['kind'] == 'game_timestamps_unsynced'
