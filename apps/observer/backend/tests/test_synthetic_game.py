"""SYNTHETIC whole-game API: pre/reveal/next/restart, no future result before reveal, freshness abstention
(B1 same-stamp conflict, stale, unorderable, memory-only), correction and crash restart, final partition.
Temp SQLite and an ephemeral 127.0.0.1 server only; no real feed, fixture, model or port."""
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app import synthetic_game as sg
from test_whole_game_accounting import TRUTH, expected, stamp  # independent truth, never the store

LAST = len(TRUTH)-1


@pytest.fixture(scope='module')
def script(tmp_path_factory):
    return sg.Script(tmp_path_factory.mktemp('synthetic-game'))


def test_scenario_is_the_independent_synthetic_plan():
    assert sg.PLAN == TRUTH and sg.LAST == LAST


def test_pre_reveal_state_never_carries_this_or_a_later_pitch(script):
    game = sg.Game(script)
    for cursor in range(LAST+1):
        view = game.state(cursor, False)
        text = json.dumps(view, ensure_ascii=False)
        assert view['actual'] is None and view['partition'] is None and not view['complete']
        assert len(view['history']) == cursor and view['comparison']['label'] in ('자료 확인 중', '예측 대기', '지원 없음')
        assert not any(TRUTH[k][2] in text for k in range(cursor+1, LAST+1))
        assert not any(stamp(k) in text for k in range(cursor, LAST+1))
        assert 'probabilit' not in text and view['live'] is False and view['synthetic'] is True
        as_of = script.frames[cursor-1]['as_of']['pitch_event_id'] if cursor else 0
        assert all(r['id'] <= as_of for row in view['history'] for r in row['diagnostics']['trail'])


def test_freshness_gate_abstains_before_predict(script):
    calls = []

    def predict(x):  # stub, never a model
        calls.append(len(x))
        return np.full((len(x), 10), .1)

    game = sg.Game(script, predict=predict)
    state = lambda cursor: game.state(cursor, False)['comparison']
    # B1: pitch after the step-30 same-stamp conflict
    c = state(31)
    assert (c['state'], c['label'], c['gate']['status'], c['gate']['reason']) == (
        'checking', '자료 확인 중', 'unsupported', 'same_stamp_conflict_provider_stamp') and calls == []
    assert c['gate']['receipt']['provider_order'] == 'same_stamp_conflict'
    assert state(41)['gate']['reason'] == 'stale_provider_stamp' and state(46)['gate']['reason'] == 'unorderable_provider_stamp'
    assert calls == []
    # real memory-only entries (store outage, then crash window) raise in capture_input: plain unsupported, no predict
    for cursor in [*range(81, 86), *range(101, 106)]:
        c = state(cursor)
        assert (c['state'], c['label'], c['gate']['reason']) == ('unsupported', '지원 없음', 'uncommitted_memory_entry')
    assert calls == []
    # a strictly later valid stamp is ordered again; only the stub runs and the fan state stays without numbers
    c = state(32)
    assert (c['state'], c['label'], c['gate']['status']) == ('waiting', '예측 대기', 'conditional_fixture') and calls == [1]
    assert 'probabilit' not in json.dumps(game.state(32, False))
    assert sg.Game(script).state(32, False)['comparison']['gate']['status'] == 'unsupported'  # default: no predictor


def test_correction_disconnect_and_crash_restart_in_the_log(script):
    game = sg.Game(script)
    before, after = game.state(19, False), game.state(20, True)
    assert before['history'][18]['result'] == '볼' and not before['history'][18]['corrected']
    assert after['history'][18]['result'] == '루킹 스트라이크' and after['history'][18]['corrected']
    trail = [r['outcome'] for r in after['history'][18]['diagnostics']['trail']]
    assert trail == ['new', 'correction']
    gap = game.state(60, True)
    assert gap['actual'] == {'game_pitch': 61, 'result': None, 'status': 'not_received', 'reason': None,
                             'note': '이 시점에는 이 투구 자료를 받지 못했습니다.'}
    assert gap['connection'] and gap['history'][60]['result'] == '아직 받지 못함'
    restarted = game.state(106, False)  # frame 105: new process restored the store, then polled
    assert restarted['comparison']['state'] == 'waiting' and restarted['diagnostics']['store_ok']
    assert [row['result'] for row in restarted['history'][99:105]] == ['헛스윙', '볼', '볼', '볼', '볼', '볼']
    assert restarted['history'][99]['corrected']  # 'C' was lost in the crash; 'S' recorded after restart
    receipts = [row['diagnostics'] for row in restarted['history'][100:105]]
    assert all(d['receipt_clock'] == 'observer_local' and d['first_recorded_receipt'] == sg.T0+10*105+1 for d in receipts)


def test_refused_or_unsaved_body_is_not_shown_as_an_observed_call(script):
    """D1: the selected step-30 body is same_stamp_conflict and quarantined; it is never an unqualified call."""
    game = sg.Game(script)
    reveal = game.state(30, True)
    assert reveal['diagnostics']['selected']['provider_order'] == 'same_stamp_conflict'
    assert (reveal['actual']['result'], reveal['actual']['status']) == ('확인 중', 'unconfirmed')
    assert reveal['actual']['reason'] == 'same_stamp_conflict/same_revision_hash_mismatch'
    assert game.state(31, False)['history'][30]['result'] == '확인 중'
    for later in (game.state(31, True), game.state(LAST, True)):  # reconciled body: change shown, not a correction
        row = later['history'][30]
        assert (row['result'], row['status'], row['corrected'], row['changed_from']) == ('볼', 'received', False, '확인 중')
    corrected = game.state(LAST, True)['history'][18]
    assert corrected['corrected'] and corrected['changed_from'] is None
    unsaved = game.state(81, True)
    assert unsaved['actual']['status'] == 'unsaved' and unsaved['actual']['result'].endswith(' · 저장 전')
    assert game.state(82, False)['comparison']['label'] == '지원 없음'
    assert 'changed_from' in (sg.WEB/'app.js').read_text()


def test_stale_arrival_quarantine_does_not_alias_the_selected_body(tmp_path):
    """Alias: a stale changed arrival's quarantine row names the selected seq, not the selected body."""
    store, key = sg.LiveSnapshotStore(tmp_path/'alias.sqlite3'), sg.PLAN[3]
    selected = sg.provider(3)
    out, entry = store.ingest(sg.GAME, selected, fetched=sg.T0+31, source='live_poll')
    stale = sg.provider(3, ts=sg.stamp(0), conflict=True)  # body X under an older provider stamp
    assert (out, store.ingest(sg.GAME, stale, fetched=sg.T0+32, source='live_poll')[0]) == ('new', 'stale')
    rows = [r for r in store.pitch_events(sg.GAME)['events'] if r['provider_event_id'] == key[2]]
    assert [(r['outcome'], r['reason'], r['store_seq']) for r in rows if r['outcome'] == 'quarantine'] == \
        [('quarantine', 'stale_revision', entry['seq'])]
    frame = {'entry': entry, 'store_ok': True, 'committed_feed': selected}
    assert sg._shown(frame, key, rows) == ('볼', 'received', None)


def test_http_first_to_last_reveal_next_restart_and_final_partition(script):
    server = ThreadingHTTPServer(('127.0.0.1', 0), sg.make_handler(sg.Game(script)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection('127.0.0.1', server.server_port)

    def call(method, path):
        conn.request(method, path, headers={'Cookie': cookie} if cookie else {})
        response = conn.getresponse()
        return response.status, response.getheader('Set-Cookie'), json.loads(response.read())

    try:
        cookie = None
        status, set_cookie, first = call('GET', '/api/state')
        cookie = set_cookie.split(';')[0]
        assert status == 200 and first['game_pitch'] == 1 and first['situation'] is None
        assert first['comparison']['label'] == '자료 확인 중' and first['actual'] is None
        assert call('POST', '/api/next')[0] == 409
        for cursor in range(LAST+1):
            status, _, log = call('GET', '/api/log')
            assert status == 200 and log['game_pitch'] == cursor+1 and log['partition'] is None
            assert not any(TRUTH[k][2] in json.dumps(log) for k in range(cursor+1, LAST+1))
            status, _, view = call('POST', '/api/reveal')
            assert status == 200 and view['reveal'] and view['complete'] == (cursor == LAST)
            if cursor == 0:
                assert view['actual']['result'] == '볼' and view['situation']['inning'] == 1
            if cursor < LAST:
                assert view['partition'] is None
                status, _, view = call('POST', '/api/next')
                assert status == 200 and view['actual'] is None
        assert call('POST', '/api/reveal')[0] == 409 and call('POST', '/api/next')[0] == 409
        assert view['actual']['result'] == '볼' and view['situation']['inning'] == 9 and view['situation']['half'] == '초'
        part = view['partition']
        assert part['scope'] == 'synthetic_fixture_only' and part['expected_keys'] == 129
        exp = expected(LAST)
        assert {name: {tuple(k) for k in part[name]} for name in exp} == exp
        assert part['counts'] == {'accepted': 127, 'quarantined_only': 2, 'missing': 0, 'extra': 1}
        assert part['events'] == {'live_duplicates': 8, 'corrections': 4, 'quarantines': 4, 'rows': 136}
        assert part['history']['status'] == 'UNKNOWN'
        status, _, again = call('POST', '/api/restart')
        assert status == 200 and again['game_pitch'] == 1 and again['actual'] is None and again['partition'] is None
        conn.request('GET', '/')
        response = conn.getresponse()
        assert response.status == 200 and '합성 경기' in response.read().decode()
        conn.request('GET', '/../live_store.py')
        response = conn.getresponse()
        assert response.status == 404
        response.read()
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
