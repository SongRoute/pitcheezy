"""The ledger-backed stand-in policy (``observer_app.ledger_policy``) and the replay's ``--policy ledger`` /
``--keep-moments`` paths, on SYNTHETIC captures and a synthetic live store only (no real ledger, T7, model or port)."""
from contextlib import closing
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parents[2]
sys.path[:0] = [str(BACKEND), str(Path(__file__).resolve().parent), str(REPO/'scripts')]
from observer_app import display_state, ledger_policy, publish_audit  # noqa: E402
from observer_app.live_armb import state_key  # noqa: E402
from observer_app.live_feed import parse_state  # noqa: E402
from observer_app.live_store import LiveSnapshotStore, _canonical  # noqa: E402
import publisher_replay  # noqa: E402
from publish_fixture import BASE, LIVE_GAME  # noqa: E402
from test_display_state import CAND  # noqa: E402
from test_publisher_replay import CONTRACT, LIMITS, capture_db, game_script  # noqa: E402

IDENTITY = 'b'*64


def ready_states():
    """``[(state_sha256, pending key, body)]`` of the synthetic capture's bodies that have a pending pitch, in order."""
    out = []
    for _, body in game_script():
        if parse_state(body, None).get('status') == 'ready':
            out.append((hashlib.sha256(state_key(body).encode()).hexdigest(), display_state.pending_key(body), body))
    return out


def decision_id(state):
    return hashlib.sha256(_canonical([LIVE_GAME, state, IDENTITY])).hexdigest()


def live_store(folder, rows, *, identity=IDENTITY, name='live-copy.sqlite3'):
    """A store with ``first`` ledger rows written the way ``LiveSnapshotStore.append_recommendation`` writes them:
    ``rows`` = ``[(state_sha256, pitch key, candidates)]``. Returns its path (WAL checkpointed, so no -wal beside it)."""
    path = Path(folder)/name
    LiveSnapshotStore(path)  # the schema
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        for number, (state, key, candidates) in enumerate(rows, 1):
            displayed = {'status': 'ready', 'reason': None, 'key': key, 'recommendation': {'candidates': candidates},
                         'policy_identity': identity[:8]}
            record = {'schema': 'served-recommendation-v1', 'decision_id': hashlib.sha256(_canonical([LIVE_GAME, state, identity])).hexdigest(),
                      'game_pk': LIVE_GAME, 'pitch_key': key, 'state_sha256': state, 'policy_identity': identity,
                      'decided': BASE+number, 'decision_clock': 'observer_local', 'feed_ok': True,
                      'snapshot': {'store_seq': number, 'slot_version': 1, 'feed_sha256': '0'*64, 'source': 'live_poll',
                                   'received': BASE+number-.5, 'provider_order': 'ordered'},
                      'displayed': displayed, 'inputs': None, 'target': {'status': 'UNKNOWN', 'reason': 'next_pitch_not_observed_at_decision'}}
            db.execute('INSERT INTO served_recommendations(decision_id, game_pk, pitch_key, outcome, decided, decision_clock, store_seq, sha, body) '
                       'VALUES (?,?,?,?,?,?,?,?,?)', (record['decision_id'], LIVE_GAME, key, 'first', record['decided'], 'observer_local',
                                                     number, hashlib.sha256(_canonical(displayed)).hexdigest(), _canonical(record)))
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    for suffix in ('-wal', '-shm'):  # after the checkpoint and the last close: an empty WAL and the shared-memory index may remain
        beside = Path(str(path)+suffix)
        if beside.exists() and (suffix == '-shm' or beside.stat().st_size == 0):
            beside.unlink()
    return path


def support_file(folder, pitchers, name='support.json'):
    path = Path(folder)/name
    path.write_text(json.dumps({'schema': ledger_policy.SUPPORT_SCHEMA, 'source': 'synthetic test', 'pitchers': pitchers}))
    return path


def test_load_rows_keys_first_rows_by_state_and_refuses_what_cannot_stand_for_one_live_policy(tmp_path):
    (a, key_a, _), (b, key_b, _) = ready_states()[:2]
    path = live_store(tmp_path, [(a, key_a, [CAND['SL']]), (b, key_b, [CAND['FF']])])
    rows = ledger_policy.load_rows(path, LIVE_GAME)
    assert sorted(rows) == sorted([a, b]) and rows[a]['pitch_key'] == key_a and rows[a]['displayed']['recommendation'] == {'candidates': [CAND['SL']]}
    assert rows[b]['policy_identity'] == IDENTITY and rows[b]['decided'] == BASE+2
    assert sorted(p.name for p in tmp_path.iterdir()) == ['live-copy.sqlite3']  # read immutable: nothing appeared beside the copy
    with pytest.raises(ledger_policy.LedgerUnusable, match='no first row'):
        ledger_policy.load_rows(path, LIVE_GAME+1)
    with pytest.raises(ledger_policy.LedgerUnusable, match='missing'):
        ledger_policy.load_rows(tmp_path/'not-there.sqlite3', LIVE_GAME)
    Path(str(path)+'-wal').write_bytes(b'x')
    with pytest.raises(ledger_policy.LedgerUnusable, match='WAL'):
        ledger_policy.load_rows(path, LIVE_GAME)
    Path(str(path)+'-wal').unlink()
    two = live_store(tmp_path/'two', [(a, key_a, [CAND['SL']])])
    with closing(sqlite3.connect(two, isolation_level=None)) as db:  # a second identity for another state
        body = json.loads(db.execute('SELECT body FROM served_recommendations').fetchone()[0])
        other = body | {'state_sha256': b, 'policy_identity': 'c'*64, 'displayed': body['displayed'] | {'policy_identity': 'c'*8, 'key': key_b}, 'decision_id': 'd'*64}
        db.execute('INSERT INTO served_recommendations(decision_id, game_pk, pitch_key, outcome, decided, decision_clock, store_seq, sha, body) '
                   'VALUES (?,?,?,?,?,?,?,?,?)', ('d'*64, LIVE_GAME, key_b, 'first', BASE+9, 'observer_local', 1, 'e'*64, _canonical(other)))
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    with pytest.raises(ledger_policy.LedgerUnusable, match='2 policy identities'):
        ledger_policy.load_rows(two, LIVE_GAME)
    bad = live_store(tmp_path/'bad', [(a, key_a, [])])  # a row without candidates was never displayable
    with pytest.raises(ledger_policy.LedgerUnusable, match='not a displayable'):
        ledger_policy.load_rows(bad, LIVE_GAME)


def test_load_support_accepts_only_the_three_pitcher_level_refusals(tmp_path):
    good = support_file(tmp_path, {'693855': {'status': 'UNSUPPORTED_UNKNOWN_PITCHER', 'name': 'x'}, '1': {'status': 'UNSUPPORTED_EMPTY_SUPPORT'}})
    assert ledger_policy.load_support(good) == {693855: 'UNSUPPORTED_UNKNOWN_PITCHER', 1: 'UNSUPPORTED_EMPTY_SUPPORT'}
    for pitchers in ({'5': {'status': 'SUPPORTED'}}, {'5': {'status': 'UNSUPPORTED_MID_PA'}}, {'x': {'status': 'UNSUPPORTED_UNKNOWN_PITCHER'}}, {'5': 'UNSUPPORTED_UNKNOWN_PITCHER'}):
        with pytest.raises(ledger_policy.LedgerUnusable, match='status must be one of'):
            ledger_policy.load_support(support_file(tmp_path, pitchers))
    (tmp_path/'other.json').write_text('{"schema": "something-else", "pitchers": {}}')
    with pytest.raises(ledger_policy.LedgerUnusable, match='not a pitcheezy-ledger-support-v1'):
        ledger_policy.load_support(tmp_path/'other.json')
    with pytest.raises(ledger_policy.LedgerUnusable, match='unreadable'):
        ledger_policy.load_support(tmp_path/'missing.json')


def test_the_policy_answers_from_the_ledger_the_support_map_or_with_no_recommendation_at_all(tmp_path):
    (a, key_a, body_a), (b, key_b, body_b) = ready_states()[:2]
    rows = ledger_policy.load_rows(live_store(tmp_path, [(a, key_a, [CAND['SL']])]), LIVE_GAME)
    pitcher = ledger_policy.pending_pitcher(body_b)
    assert isinstance(pitcher, int)
    policy = ledger_policy.LedgerPolicy(rows, {pitcher: 'UNSUPPORTED_UNKNOWN_PITCHER'}, game_pk=LIVE_GAME)
    first = policy(body_a)
    assert first['key'] == key_a and first['policy_identity'] == IDENTITY and first['pa_pitches'] == [] and first['previous'] is None
    assert first['home_we_now'] is None and first['pitch']['status'] == ledger_policy.READY_STATUS and first['pitch']['actual'] is None
    assert first['pitch']['pre'] == {'status': 'ready', 'reason': None, 'recommendation': {'candidates': [CAND['SL']]}}
    first['pitch']['pre']['recommendation']['candidates'][0]['rank'] = 9  # a caller's copy, never the stored row
    assert rows[a]['displayed']['recommendation']['candidates'][0]['rank'] == 1
    second = policy(body_b)  # no row: the pitcher the live policy refused
    assert second['key'] == key_b and second['pitch']['status'] == 'UNSUPPORTED_UNKNOWN_PITCHER'
    assert second['pitch']['pre'] == {'status': 'unsupported', 'reason': None, 'recommendation': None}
    bare = ledger_policy.LedgerPolicy(rows, {}, game_pk=LIVE_GAME)(body_b)  # no row, pitcher not in the map
    assert bare['pitch']['status'] == ledger_policy.NO_ROW_STATUS and bare['pitch']['pre']['reason'] == ledger_policy.NO_ROW_REASON
    assert bare['pitch']['pre']['status'] == 'unsupported' and bare['policy_identity'] == IDENTITY
    other = json.loads(json.dumps(body_a))
    other['gameData']['game']['pk'] = LIVE_GAME+1
    assert policy(other) is None
    assert policy.counters == {'ledger': 1, 'unsupported': 1, 'other_game': 1}
    record = policy.record()
    assert record['kind'] == 'ledger' and record['rows'] == 1 and record['rows_used'] == 1 and record['rows_never_asked'] == 0
    assert record['unsupported_pitchers'] == {str(pitcher): 'UNSUPPORTED_UNKNOWN_PITCHER'} and record['live_policy_identity'] == IDENTITY
    # the sentence a fan reads for a no-row state is the policy's own short Korean one (display_state: policy_refused)
    facts = {'delayed': False, 'store': 'on', 'store_ok': True, 'lag_s': None, **{k: None for k in (
        'ok', 'lag_max_s', 'lag_measured', 'lag_threshold_s', 'thresholds_provisional', 'last_ok_at', 'failing_since', 'failing_for_s', 'received_at', 'provider_order')}}
    view = {'status': 'unsupported', 'reason': ledger_policy.NO_ROW_REASON, 'key': key_b, 'recommendation': None, 'policy_status': ledger_policy.NO_ROW_STATUS}
    state = display_state.decide(gated=None, view=view, facts=facts, pitch_key=key_b)
    assert (state['state'], state['reason'], state['detail'], state['message']) == ('unsupported', 'policy_refused', ledger_policy.NO_ROW_STATUS, ledger_policy.NO_ROW_REASON)
    view = {'status': 'unsupported', 'reason': None, 'key': key_b, 'recommendation': None, 'policy_status': 'UNSUPPORTED_UNKNOWN_PITCHER'}
    state = display_state.decide(gated=None, view=view, facts=facts, pitch_key=key_b)
    assert (state['state'], state['reason']) == ('unsupported', 'unknown_pitcher')


def test_a_replay_with_the_ledger_policy_publishes_exactly_the_recorded_candidates_and_keeps_moments(tmp_path, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '1')
    monkeypatch.delenv('PITCHEEZY_OBSERVER_REC_LEDGER', raising=False)
    (a, key_a, body_a), (b, key_b, body_b) = ready_states()[:2]
    path = capture_db(tmp_path/'job'/'input')
    rows = ledger_policy.load_rows(live_store(tmp_path/'job'/'input', [(a, key_a, [CAND['SL']])]), LIVE_GAME)
    support = {ledger_policy.pending_pitcher(body_b): 'UNSUPPORTED_UNKNOWN_PITCHER'}
    schemas = {p.name.split('.')[0]: json.loads(p.read_text()) for p in (CONTRACT/'schema').glob('*.schema.json')}
    capture, info = publisher_replay.capture_rows(path, LIVE_GAME, 1 << 26)
    run = tmp_path/'job'/'run-001'
    run.mkdir()
    try:
        summary = publisher_replay.run(rows=capture, info=info, run_dir=run, game_pk=LIVE_GAME, commit='c'*40, limits=LIMITS,
                                       policy_factory=lambda: ledger_policy.LedgerPolicy(rows, support, game_pk=LIVE_GAME),
                                       schemas=schemas, log=lambda text: None, keep_moments=True, policy_source='ledger')
    finally:
        info['db'].close()
    assert summary['status'] == 'COMPLETED' and summary['invariants']['all_held'] is True, summary['stop']
    assert summary['policy']['source'] == 'ledger' and summary['policy']['ledger']['kind'] == 'ledger'
    answers = summary['policy']['ledger']['answers']
    assert answers.get('ledger', 0) >= 1 and answers.get('unsupported', 0) >= 1 and 'no_row' not in answers
    assert summary['policy']['ledger']['rows_used'] == 1 and summary['policy']['ledger']['live_policy_identity'] == IDENTITY
    assert summary['game_file']['recommendation_by_cycle']['published'] >= 1 and 'unsupported|unknown_pitcher' in summary['game_file']['display_states']
    # the run's own ledger holds the live decision again: same state, same identity, same decision id, same displayed content
    own = LiveSnapshotStore(run/'publisher-store.sqlite3').served_recommendations(LIVE_GAME)
    assert [row['outcome'] for row in own] == ['first'] and own[0]['state_sha256'] == a and own[0]['decision_id'] == decision_id(a)
    assert own[0]['displayed'] == rows[a]['displayed'] and own[0]['policy_identity'] == IDENTITY
    # a past pitch the final file pairs with a recorded recommendation shows exactly the recorded candidates (the synthetic
    # capture pairs none, with this policy as with the tests' fake one: its decision inputs never match a thrown pitch)
    final = json.loads((run/'out'/'v1'/'game'/f'{LIVE_GAME}.json').read_text())
    paired = [p for p in final['pitches'] if (p.get('rec') or {}).get('candidates')]
    assert all(p['rec']['candidates'] == publish_audit.shown_candidates([CAND['SL']]) for p in paired)
    assert all(p['rec']['ledger']['decision_id'] == decision_id(a) and p['rec']['policy']['identity'] == IDENTITY[:8] for p in paired)
    assert isinstance(summary['game_file']['paired_past_recommendations_in_last_file'], int) and len(paired) == summary['game_file']['paired_past_recommendations_in_last_file']
    # moments: one folder per cycle whose game body changed and had a pending pitch; gzip bytes hash to the index's sha256
    lines = [json.loads(line) for line in (run/'moments'/'INDEX.jsonl').read_text().splitlines()]
    assert summary['moments']['kept'] == len(lines) >= 2 and summary['moments']['requested'] is True and summary['moments']['index'] == 'moments/INDEX.jsonl'
    assert all(line['state'] != 'final' and line['pitch_key'] for line in lines)
    assert [line['cycle'] for line in lines] == sorted({line['cycle'] for line in lines})
    cycles = [json.loads(line) for line in (run/'CYCLES.jsonl').read_text().splitlines()]
    assert [row['cycle'] for row in cycles if row['moment']] == [line['cycle'] for line in lines]
    assert all(row['moment'] is None for row in cycles if row['game'] and row['game']['final'])
    kept_bytes = 0
    for line in lines:
        folder = run/line['folder']
        for entry in line['files']:
            raw = gzip.decompress((folder/(entry['path']+'.gz')).read_bytes())
            assert hashlib.sha256(raw).hexdigest() == entry['sha256'] and len(raw) == entry['bytes']
            kept_bytes += len(raw)
        game = json.loads(gzip.decompress((folder/f'v1/game/{LIVE_GAME}.json.gz').read_bytes()))
        assert game['next']['pitch_key'] == line['pitch_key'] and game['display_state']['state'] == line['state']
        if line['recommendation'] == 'published':
            assert line['state'] == 'recommended' and line['pitch_key'] == key_a
            assert game['next']['rec']['candidates'] == publish_audit.shown_candidates([CAND['SL']])
        else:
            assert game['next']['rec']['candidates'] == []
            if line['state'] == 'unsupported':  # the pitcher the map names: refused as live refused him, no row needed
                assert game['display_state']['reason'] == 'unknown_pitcher' and line['pitch_key'] == key_b
    assert kept_bytes == summary['moments']['raw_bytes'] and summary['moments']['gzip_bytes'] > 0
    assert {'recommended', 'unsupported'} <= {line['state'] for line in lines}
    # the live store copy was only read
    assert sorted(p.name for p in (tmp_path/'job'/'input').iterdir()) == ['capture.sqlite3', 'live-copy.sqlite3']


def test_without_keep_moments_nothing_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '1')
    (a, key_a, _), = ready_states()[:1]
    path = capture_db(tmp_path/'job'/'input')
    rows = ledger_policy.load_rows(live_store(tmp_path/'job'/'input', [(a, key_a, [CAND['FF']])]), LIVE_GAME)
    capture, info = publisher_replay.capture_rows(path, LIVE_GAME, 1 << 26)
    run = tmp_path/'job'/'run-001'
    run.mkdir()
    try:
        summary = publisher_replay.run(rows=capture, info=info, run_dir=run, game_pk=LIVE_GAME, commit='c'*40, limits=LIMITS,
                                       policy_factory=lambda: ledger_policy.LedgerPolicy(rows, {}, game_pk=LIVE_GAME), log=lambda text: None)
    finally:
        info['db'].close()
    assert summary['status'] == 'COMPLETED' and summary['moments'] == {'kept': 0, 'raw_bytes': 0, 'gzip_bytes': 0, 'requested': False, 'index': None, 'rule': summary['moments']['rule']}
    assert not (run/'moments').exists() and summary['policy']['source'] == 'armb'  # no name given: a factory is reported as before
    assert summary['policy']['ledger']['answers'].get('no_row', 0) >= 1 and 'unsupported|policy_refused' in summary['game_file']['display_states']


def test_the_command_refuses_unregistered_ledger_inputs_before_writing_anything(tmp_path, capsys):
    job = tmp_path/'job'
    path = capture_db(job/'input')
    (a, key_a, _), = ready_states()[:1]
    ledger = live_store(job/'input', [(a, key_a, [CAND['SL']])])
    support = support_file(job/'input', {'5': {'status': 'UNSUPPORTED_UNKNOWN_PITCHER'}})
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    config, original = job/'replay_config.json', tmp_path/'original'
    original.mkdir()
    config.write_text(json.dumps(LIMITS | {'capture_original_dir': str(original)}))
    base = ['--job-dir', str(job), '--run-name', 'run-001', '--game', str(LIVE_GAME), '--capture-db', str(path), '--capture-sha256', sha(path),
            '--commit', publisher_replay.head_commit(), '--config', str(config), '--config-sha256', sha(config), '--allow-dirty']
    full = ['--ledger-db', str(ledger), '--ledger-sha256', sha(ledger), '--ledger-support', str(support), '--ledger-support-sha256', sha(support)]
    refused = lambda argv, words: publisher_replay.main(argv) == 2 and words in capsys.readouterr().err
    assert refused(base+['--policy', 'ledger'], "needs ['--ledger-db'")
    assert refused(base+['--policy', 'ledger', *full[:6]], "needs ['--ledger-support-sha256']")
    assert refused(base+['--policy', 'off', *full], 'belong to --policy ledger')
    assert refused(base+['--policy', 'ledger', *full[:3], '0'*64, *full[4:]], 'ledger copy does not have the registered sha256')
    elsewhere = live_store(tmp_path/'elsewhere', [(a, key_a, [CAND['SL']])])
    assert refused(base+['--policy', 'ledger', '--ledger-db', str(elsewhere), '--ledger-sha256', sha(elsewhere), *full[4:]], 'ledger copy must be a file inside the job folder')
    bad = support_file(job/'input', {'5': {'status': 'SUPPORTED'}}, name='bad-support.json')
    assert refused(base+['--policy', 'ledger', *full[:4], '--ledger-support', str(bad), '--ledger-support-sha256', sha(bad)], 'ledger input unusable')
    other = live_store(job/'input', [(a, key_a, [CAND['SL']])], name='other-game.sqlite3')
    with closing(sqlite3.connect(other, isolation_level=None)) as db:
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    assert refused(base+['--policy', 'ledger', '--ledger-db', str(other), '--ledger-sha256', sha(other), *full[4:], '--game', str(LIVE_GAME+1)], 'ledger input unusable: no first row')
    assert 'observer_app.main' in sys.modules  # a full ledger run is refused in this process for the usual reason, after every input check passed
    assert refused(base+['--policy', 'ledger', *full], 'before its folders could be pinned')
    assert sorted(p.name for p in job.iterdir()) == ['input', 'replay_config.json']  # nothing was written
