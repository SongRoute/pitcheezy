"""``shadow_expectation``: after a key moment, the pre-pitch Step 5 forecast of the pitch type that WAS thrown, or
nothing. Synthetic ledgers carry the real schema (DDL copied from step5 ``ledger.py`` and ``shadow_runner.py``,
nothing imported from them); the tests on the one reviewed real ledger skip when the pm-jobs folder is absent.
No model, no network; every ledger is opened read-only by the module."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app import shadow_expectation as SE

GAME = 849000
ACTIONS, LEAVES = SE.ACTIONS, SE.LEAF_ORDER
REAL = Path('/Users/song/Projects/pitcheezy-pm-jobs/20261011-step5-shadow-001/run-849831-001/shadow-ledger.sqlite3')
REAL_REPLAY = Path('/Users/song/Projects/pitcheezy-pm-jobs/20261011-rc-replay-849831/replay-849831')

# step5/ledger.py SCHEMA (accepted source, sha 50acf8e8...) + shadow_runner.py SHADOW_SCHEMA, with their triggers.
SCHEMA = '''
CREATE TABLE IF NOT EXISTS poll_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_pk INTEGER,
    store_seq INTEGER,
    feed_sha256 TEXT,
    attempt_key TEXT,
    at_bat_number INTEGER,
    last_pitch_number INTEGER,
    pa_ended INTEGER,
    kind TEXT NOT NULL,
    outcome TEXT,
    reason TEXT,
    sha TEXT NOT NULL UNIQUE,
    body BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS pitch_attempts (
    attempt_key TEXT PRIMARY KEY,
    game_pk INTEGER NOT NULL,
    at_bat_number INTEGER NOT NULL,
    last_pitch_number INTEGER NOT NULL,
    store_seq INTEGER,
    feed_sha256 TEXT,
    outcome TEXT NOT NULL,
    reason TEXT,
    decision_unix REAL,
    sha TEXT NOT NULL,
    body BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS action_cells (
    attempt_key TEXT NOT NULL REFERENCES pitch_attempts(attempt_key),
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    PRIMARY KEY(attempt_key, action));
CREATE TABLE IF NOT EXISTS later_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_key TEXT NOT NULL REFERENCES pitch_attempts(attempt_key),
    feed_sha256 TEXT,
    kind TEXT NOT NULL,
    sha TEXT NOT NULL,
    body BLOB NOT NULL,
    UNIQUE(attempt_key, kind, sha));
CREATE TABLE IF NOT EXISTS physical_reconciliation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_pk INTEGER NOT NULL,
    at_bat_number INTEGER,
    pitch_number INTEGER,
    attempt_key TEXT,
    label TEXT NOT NULL,
    sha TEXT NOT NULL UNIQUE,
    body BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS shadow_run (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    body BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS shadow_forecasts (
    attempt_key TEXT PRIMARY KEY REFERENCES pitch_attempts(attempt_key),
    game_pk INTEGER NOT NULL,
    at_bat_number INTEGER NOT NULL,
    candidate_pitch_number INTEGER NOT NULL,
    written_at_replay_index INTEGER NOT NULL,
    replay_file TEXT,
    store_seq INTEGER,
    feed_sha256 TEXT,
    pitcher INTEGER,
    batter INTEGER,
    context_sha256 TEXT,
    context_json TEXT,
    decision TEXT NOT NULL CHECK (decision IN ('scored', 'abstain')),
    outcome TEXT NOT NULL,
    reason TEXT,
    action_probabilities TEXT NOT NULL,
    profile_type_share TEXT,
    vectors_sha256 TEXT,
    model_kind TEXT,
    model_sha256 TEXT,
    registry_sha256 TEXT,
    encoder_sha256 TEXT,
    profile_sha256 TEXT,
    profile_receipt_sha256 TEXT,
    source_sha256 TEXT NOT NULL,
    attempt_record_sha256 TEXT NOT NULL,
    row_sha256 TEXT NOT NULL,
    compute_ms REAL);
CREATE TABLE IF NOT EXISTS shadow_realized (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_pk INTEGER NOT NULL,
    at_bat_number INTEGER NOT NULL,
    pitch_number INTEGER NOT NULL,
    observed_at_replay_index INTEGER NOT NULL,
    observed_feed_sha256 TEXT,
    play_id TEXT,
    realized_pitch_type TEXT,
    realized_call_code TEXT,
    realized_description TEXT,
    realized_trajectory TEXT,
    realized_leaf TEXT,
    realized_leaf_reason TEXT,
    realized_play_pitcher INTEGER,
    realized_play_batter INTEGER,
    sha TEXT NOT NULL UNIQUE);
'''+''.join(f'''
CREATE TRIGGER IF NOT EXISTS {table}_no_{verb} BEFORE {verb} ON {table}
    BEGIN SELECT RAISE(ABORT, 'append-only ledger'); END;'''
            for table in ('poll_observations', 'pitch_attempts', 'action_cells', 'later_observations',
                          'physical_reconciliation', 'shadow_run', 'shadow_forecasts', 'shadow_realized')
            for verb in ('update', 'delete'))
REPLAY = {'clock': SE.REPLAY_CLOCK, 'source': SE.REPLAY_SOURCE}
# A live run (the ``Ledger`` helper adds ``fan_facing_output: true``: without that grant nothing is ever shown)
LIVE_RUN = {'schema': 'step5-shadow-run-v1', 'mode': 'actual', 'clock': SE.LIVE_CLOCK, 'game_pk': GAME}
ORDERED = ('physical_order_reconciliation', {'label': SE.ORDER_OK})  # the runner's row that the pitch came after the decision
LIVE = {'clock': SE.LIVE_CLOCK, 'source': SE.LIVE_SOURCE}


def _j(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _sha(value):
    return hashlib.sha256(_j(value).encode()).hexdigest()


def seven(whiff=0.12, ground=0.09):
    """A seven-leaf vector with the two shown leaves set and the rest spread so that it sums to 1."""
    rest = (1-whiff-ground)/5
    return [rest, rest, rest, whiff, rest, ground, rest]


class Ledger:
    """A synthetic shadow ledger and the one writer connection that appends to it (the sidecar's role)."""

    def __init__(self, path, *, wal=False, run=None):
        self.path, self.n = Path(path), 0
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=5)
        if wal:
            self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript(SCHEMA)
        if run is not False:
            self.db.execute('INSERT INTO shadow_run(id, body) VALUES (1, ?)', (_j(
                {'fan_facing_output': True} | (run or {'schema': 'step5-shadow-run-v1', 'mode': 'actual', 'clock': SE.REPLAY_CLOCK, 'game_pk': GAME})).encode(),))

    def attempt(self, ab=5, last=2, *, pitcher=111, batter=222, cells=None, outcome='scored', clock=REPLAY,
                decision_unix=1000.0, written=10, state=None, leaf_order=LEAVES, kind=SE.MODEL_KIND, game=GAME,
                forecast_cells=None, compute_ms=5.0, bad_sha=False, receipt_clock=None):
        key = f'{game}:{ab}:after{last}'
        scored = outcome == 'scored'
        cells = {'FF': seven(), 'SL': seven(0.25, 0.05)} if cells is None else cells
        listed = [{'action': a, 'status': 'scored' if scored and a in cells else 'abstain',
                   'reason': None if scored and a in cells else 'unsupported_action:below_count_threshold',
                   'seven_leaf': cells[a] if scored and a in cells else None, 'prior_pitch_count': 100 if a in cells else 0}
                  for a in ACTIONS]
        pre = {'at_bat_number': ab, 'last_pitch_number': last, 'balls': 1, 'strikes': 2, 'outs': 1, 'inning': 3, 'top': True,
               'bases': [True, False, False], 'score': [2, 1], 'pitcher': pitcher, 'batter': batter, 'pa_status': 'open'} | (state or {})
        body = {'attempt_key': key, 'game_pk': game, 'schema': 'step5-pitch-attempt-v2', 'outcome': outcome,
                'reason': None if scored else 'no_valid_prefix_before_close', 'cells': listed,
                'matchup': {'pitcher': pitcher, 'batter': batter}, 'pre_pitch_state': pre,
                'decision': {'unix': decision_unix, 'mono': 1.0, 'clock': clock['clock'], 'anchor_id': 'a'},
                'snapshot': {'source': clock['source'], 'receipt_clock': receipt_clock or clock['clock'], 'store_seq': 7,
                             'feed_sha256': 'f'*64},
                'model': {'kind': kind, 'leaf_order': list(leaf_order), 'sha256': {'parent': 'p'*64}},
                'target': {'candidate_at_bat_number': ab, 'candidate_pitch_number': last+1}}
        sha = _sha(body)
        self.db.execute('BEGIN')
        self.db.execute('INSERT INTO pitch_attempts VALUES (?,?,?,?,?,?,?,?,?,?,?)', (
            key, game, ab, last, 7, 'f'*64, outcome, body['reason'], decision_unix, sha, _j(body).encode()))
        self.db.executemany('INSERT INTO action_cells VALUES (?,?,?,?)', [(key, c['action'], c['status'], c['reason']) for c in listed])
        row = {'attempt_key': key, 'game_pk': game, 'at_bat_number': ab, 'candidate_pitch_number': last+1,
               'written_at_replay_index': written, 'replay_file': 'x.json.gz', 'store_seq': 7, 'feed_sha256': 'f'*64,
               'pitcher': pitcher, 'batter': batter, 'context_sha256': None, 'context_json': None,
               'decision': 'scored' if scored else 'abstain', 'outcome': outcome, 'reason': body['reason'],
               'action_probabilities': _j(forecast_cells if forecast_cells is not None else [
                   {k: c[k] for k in ('action', 'status', 'reason', 'seven_leaf', 'prior_pitch_count')} for c in listed]),
               'profile_type_share': None, 'vectors_sha256': 'v'*64, 'model_kind': kind, 'model_sha256': _j({'parent': 'p'*64}),
               'registry_sha256': 'r'*64, 'encoder_sha256': 'e'*64, 'profile_sha256': 'q'*64, 'profile_receipt_sha256': 'c'*64,
               'source_sha256': 's'*64, 'attempt_record_sha256': 'x'*64 if bad_sha else sha}
        row['row_sha256'], row['compute_ms'] = _sha(row), compute_ms
        self.db.execute(f"INSERT INTO shadow_forecasts({','.join(row)}) VALUES ({','.join('?'*len(row))})", list(row.values()))
        self.db.execute('COMMIT')
        return key

    def realized(self, ab=5, number=3, kind='FF', index=11, game=GAME):
        self.n += 1
        self.db.execute('INSERT INTO shadow_realized(game_pk, at_bat_number, pitch_number, observed_at_replay_index, '
                        'realized_pitch_type, sha) VALUES (?,?,?,?,?,?)', (game, ab, number, index, kind, f'r{self.n}'))

    def poll(self, ab=5, *, matchup=None, observed=None, game=GAME):
        self.n += 1
        body = {'matchup': matchup, 'observed_matchup': observed}
        self.db.execute('INSERT INTO poll_observations(game_pk, at_bat_number, kind, sha, body) VALUES (?,?,?,?,?)',
                        (game, ab, 'first_decision', f'p{self.n}', _j(body).encode()))

    def later(self, key, kind, body=None):
        self.n += 1
        self.db.execute('INSERT INTO later_observations(attempt_key, kind, sha, body) VALUES (?,?,?,?)',
                        (key, kind, f'l{self.n}', _j(body or {}).encode()))

    def physical(self, ab=5, number=3, label='had_attempt', key=None, game=GAME):
        self.n += 1
        self.db.execute('INSERT INTO physical_reconciliation(game_pk, at_bat_number, pitch_number, attempt_key, label, sha, body) '
                        'VALUES (?,?,?,?,?,?,?)', (game, ab, number, key, label, f'y{self.n}', b'{}'))


SITUATION = {'inning': 3, 'half': 'Top', 'outs': 1, 'balls': 1, 'strikes': 2, 'bases': 1, 'home_score': 2, 'away_score': 1,
             'runners': {'first': True, 'second': False, 'third': False}}


def ask(view, ab=5, number=3, kind='FF', **options):
    names = {'pa_event': 'strikeout', 'pitcher_id': 111, 'batter_id': 222, 'situation_before': SITUATION} | options
    return view.explain(GAME, ab, number, kind, **names)


@pytest.fixture
def ledger(tmp_path):
    """The usual case: pitch 5:3 has a scored replay forecast (written at body 10, first shown in body 11)."""
    held = Ledger(tmp_path/'shadow-ledger.sqlite3')
    held.key = held.attempt()
    held.realized()
    yield held
    held.db.close()


def view(ledger, **options):
    return SE.ShadowExpectations(ledger.path, **({'enabled': True, 'allow_replayed': True} | options))


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


# -- flags, shape, wording ------------------------------------------------------------------------------------------
def test_off_by_default_and_replay_needs_its_own_flag(ledger):
    assert SE.ShadowExpectations(ledger.path).explain(
        GAME, 5, 3, 'FF', pa_event='strikeout', pitcher_id=111, batter_id=222, situation_before=SITUATION) == (None, 'disabled')
    assert ask(SE.ShadowExpectations(ledger.path, enabled=True)) == (None, 'replayed_not_allowed')
    assert ask(view(ledger))[1] is None


def test_replayed_shape_text_and_truthful_marking(ledger):
    shown, why = ask(view(ledger))
    assert why is None and shown['kind'] == 'step5_expectation' and shown['recorded'] == 'replayed_pre_pitch_inputs'
    assert shown['pitch_type'] == 'FF' and shown['pitch_label'] == '포심' and shown['basis'] == 'per_pitch'
    assert shown['items'] == [
        {'leaf': 'narrow_whiff', 'label': '헛스윙', 'meaning': '이 공에 헛스윙', 'pct': 12, 'pct_text': '12%'},
        {'leaf': 'ground_ball_in_play', 'label': '땅볼', 'meaning': '이 공을 쳐서 땅볼', 'pct': 9, 'pct_text': '9%'}]
    assert shown['text'] == '던지기 전 정보로 본 예상: 헛스윙 12% · 땅볼 9%'
    assert shown['note'].startswith('2024 모델 · 참고') and '경기 뒤' in shown['note'] and '다시 계산' in shown['note']
    assert shown['lead'] != SE.LEAD_LIVE  # a replay never borrows the live wording
    assert '스윙했을 때' in shown['explain'] and shown['omitted'] == []
    proof = shown['provenance']
    assert proof['attempt_key'] == ledger.key and proof['decision_clock'] == SE.REPLAY_CLOCK and proof['run_mode'] == 'actual'
    assert all(len(proof[k]) == 12 for k in ('row_sha256', 'attempt_record_sha256', 'feed_sha256', 'profile_sha256'))
    assert json.loads(json.dumps(shown, ensure_ascii=False)) == shown


def test_no_text_reads_as_a_recommendation_or_a_pitch_prediction(ledger):
    live = Ledger(ledger.path.with_name('live.sqlite3'), run=LIVE_RUN)
    key = live.attempt(clock=LIVE)
    outputs = [ask(view(ledger))[0], ask(view(live), pitch_first_received=1010.0)[0]]  # replayed; live, order not reconciled
    live.later(key, *ORDERED)
    outputs.append(ask(view(live), pitch_first_received=1010.0)[0])  # live, reconciled
    assert all(outputs) and [o['recorded'] for o in outputs] == [SE.RECORDED_REPLAYED, SE.RECORDED_LOCAL, SE.RECORDED_LIVE]
    for text in [*strings(outputs), SE.EXPLAIN, SE.NOTE, SE.NOTE_REPLAYED, SE.NOTE_LOCAL, SE.LEAD_LIVE, SE.LEAD_REPLAYED, SE.BELOW_ONE]:
        assert not any(word in text for word in SE.FORBIDDEN_WORDS), text
    for shown in outputs:  # no ranking, no other type, nothing about what should have been thrown
        assert not {'candidates', 'rank', 'best', 'types'} & set(shown)
        assert 'SL' not in json.dumps(shown) and '슬라이더' not in json.dumps(shown, ensure_ascii=False)
    live.db.close()


def test_only_the_thrown_type_and_a_fixed_item_order(tmp_path):
    held = Ledger(tmp_path/'l.sqlite3')
    held.attempt(cells={'FF': seven(0.05, 0.20), 'SL': seven(0.29, 0.01)})
    held.realized(kind='SL')
    held.attempt(ab=6, cells={'FF': seven(0.05, 0.20), 'SL': seven(0.29, 0.01)})
    held.realized(ab=6, kind='FF')
    reader = view(held)
    assert [i['pct'] for i in ask(reader, kind='SL')[0]['items']] == [29, 1]
    shown = ask(reader, ab=6, kind='FF')[0]
    assert [(i['leaf'], i['pct']) for i in shown['items']] == [('narrow_whiff', 5), ('ground_ball_in_play', 20)]  # not sorted by value
    assert ask(reader, kind='FF') == (None, 'realized_type_disagrees')  # the ledger saw SL for 5:3
    held.db.close()


# -- key moments ----------------------------------------------------------------------------------------------------
def test_key_moment_list_is_provisional_and_gates_the_answer(ledger):
    for event in ('strikeout', 'strikeout_double_play', 'home_run', 'grounded_into_double_play', 'double_play', 'triple_play'):
        assert SE.is_key_moment(event) and ask(view(ledger), pa_event=event)[1] is None
    for event in (None, '', 'single', 'walk', 'field_out', 'hit_by_pitch', 'Strikeout', 5):
        assert not SE.is_key_moment(event) and ask(view(ledger), pa_event=event) == (None, 'not_key_moment')


def test_for_pitch_reads_a_game_history_row_and_only_its_terminal_event(ledger):
    row = {'key': f'{GAME}:5:3', 'at_bat_number': 5, 'pitch_number': 3, 'pitcher': {'id': 111}, 'batter': {'id': 222},
           'situation_before': SITUATION, 'actual': {'pitch_type': 'FF', 'event': 'strikeout'}}
    reader = view(ledger)
    assert reader.for_pitch(row)['text'].endswith('헛스윙 12% · 땅볼 9%')
    assert reader.for_pitch(row | {'actual': {'pitch_type': 'FF', 'event': None}}) is None  # not the pitch that ended the PA
    assert reader.status()['last_reason'] == 'not_key_moment'
    for broken in ({}, None, {'key': 'x', 'at_bat_number': 5, 'pitch_number': 3}, 'row'):
        assert reader.for_pitch(broken) is None


# -- rounding and the supported range -------------------------------------------------------------------------------
def test_percent_rounds_half_up_and_never_says_zero():
    assert SE.percent(0.125) == (13, '13%') and SE.percent(0.1249) == (12, '12%') and SE.percent(0.005) == (1, '1%')
    assert SE.percent(0.0049) == (0, '1% 미만') and SE.percent(0.0) == (0, '1% 미만') and SE.percent(0.2951) == (30, '30%')


@pytest.mark.parametrize('whiff, ground, texts, omitted, reason', [
    (0.31, 0.09, ['땅볼 9%'], ['narrow_whiff'], None),            # dropped, never clipped to 30
    (0.12, 0.35, ['헛스윙 12%'], ['ground_ball_in_play'], None),
    (0.30, 0.30, None, None, 'above_supported_range'),           # the bin [0.3, 0.4) has no supporting sample
    (0.45, 0.31, None, None, 'above_supported_range'),
    (0.2999, 0.004, ['헛스윙 30%', '땅볼 1% 미만'], [], None)])
def test_values_outside_the_calibrated_range_are_dropped(tmp_path, whiff, ground, texts, omitted, reason):
    held = Ledger(tmp_path/'l.sqlite3')
    held.attempt(cells={'FF': seven(whiff, ground)})
    held.realized()
    shown, why = ask(view(held))
    assert why == reason
    if reason is None:
        assert [f"{i['label']} {i['pct_text']}" for i in shown['items']] == texts and shown['omitted'] == omitted
        assert '30%' not in shown['text'] or whiff < 0.30
    held.db.close()


def test_caps_can_only_tighten(tmp_path):
    held = Ledger(tmp_path/'l.sqlite3')
    held.attempt(cells={'FF': seven(0.25, 0.25)})
    held.realized()
    assert len(ask(view(held))[0]['items']) == 2
    strict = ask(view(held, caps={'ground_ball_in_play': 0.2}))[0]
    assert [i['leaf'] for i in strict['items']] == ['narrow_whiff'] and strict['omitted'] == ['ground_ball_in_play']
    assert view(held, caps={'narrow_whiff': 0.9}).status()['below'] == {'narrow_whiff': 0.30, 'ground_ball_in_play': 0.30}
    held.db.close()


def test_only_leaves_with_calibration_evidence_are_ever_shown():
    assert [leaf['leaf'] for leaf in SE.SHOWN_LEAVES] == ['narrow_whiff', 'ground_ball_in_play']
    assert all(leaf['below'] <= 0.30 and len(leaf['label']) <= 6 for leaf in SE.SHOWN_LEAVES)
    path = Path('/Users/song/Projects/pitcheezy/results/EXP-WS-STEP5-CAL-001.json')
    if not path.is_file():
        pytest.skip('calibration evidence file not on this machine')
    evidence = json.loads(path.read_text())
    assert tuple(evidence['leaf_order']) == SE.LEAF_ORDER
    assert evidence['thresholds']['applies_to_leaves'] == [leaf['leaf'] for leaf in SE.SHOWN_LEAVES]
    design = evidence['designs'][evidence['advisory']['primary_design']]
    assert design['advisory']['label'] == 'CALIBRATION_ACCEPTABLE_FOR_DESCRIPTIVE_DISPLAY'
    for leaf in SE.SHOWN_LEAVES:
        bins = design['leaves'][leaf['leaf']]['reliability']
        above = sum(b['n'] for b in bins if b['lo'] >= leaf['below']-1e-9)
        assert above < 10 and min(b['n'] for b in bins if b['hi'] <= leaf['below']+1e-9) >= 600, (leaf['leaf'], above)


# -- exactly this pitch ---------------------------------------------------------------------------------------------
def test_abstained_unsupported_unknown_and_missing_forecasts_show_nothing(tmp_path):
    held = Ledger(tmp_path/'l.sqlite3')
    held.attempt()
    held.realized()
    held.attempt(ab=6, outcome='conflicted')
    reader = view(held)
    assert ask(reader, ab=6) == (None, 'forecast_abstained')
    assert ask(reader, kind='CU') == (None, 'realized_type_disagrees')  # FF was observed
    held.realized(kind='CU', index=12)                                 # ...then corrected to a type with no scored cell
    assert ask(reader, kind='CU') == (None, 'cell_unsupported')
    for unknown in (None, '', 'XX', 'UN', 'UNKNOWN', 'ff', 7):
        assert ask(reader, kind=unknown) == (None, 'realized_type_unknown')
    assert ask(reader, number=4) == (None, 'no_forecast') and ask(reader, ab=9) == (None, 'no_forecast')
    for ab, number in ((0, 3), (5, 0), (5, 100), ('5', 3), (5, 3.0), (True, 3)):
        assert ask(reader, ab=ab, number=number) == (None, 'bad_pitch_key')
    held.db.close()


def test_runner_numbering_is_the_feed_pitch_number(ledger):
    """The forecast made after pitch k-1 belongs to official pitch k and to no neighbour."""
    reader = view(ledger)
    assert ask(reader, number=3)[0]['provenance']['attempt_key'] == f'{GAME}:5:after2'
    assert ask(reader, number=2) == (None, 'no_forecast') and ask(reader, number=4) == (None, 'no_forecast')


def test_inputs_must_equal_the_pitch_as_thrown(ledger):
    reader = view(ledger)
    assert ask(reader, pitcher_id=999) == (None, 'different_pitcher')
    assert ask(reader, batter_id=999) == (None, 'different_batter')
    assert ask(reader, pitcher_id=None) == (None, 'identity_unverified')
    assert ask(reader, batter_id='222') == (None, 'identity_unverified')
    # an automatic ball between two pitches moves the count, not pitchNumber: the held forecast is not this pitch's
    assert ask(reader, situation_before=SITUATION | {'balls': 2}) == (None, 'different_situation:balls')
    for key, value in (('strikes', 1), ('outs', 2), ('inning', 4), ('half', 'Bot'), ('home_score', 3), ('away_score', 0)):
        assert ask(reader, situation_before=SITUATION | {key: value}) == (None, f'different_situation:{key}')
    moved = {k: v for k, v in SITUATION.items() if k != 'bases'} | {'runners': {'first': False, 'second': True, 'third': False}}
    assert ask(reader, situation_before=moved) == (None, 'different_situation:bases')  # a steal after the decision
    assert ask(reader, situation_before={k: v for k, v in SITUATION.items() if k != 'bases'})[1] is None  # runners alone
    for missing in (None, {}, {k: v for k, v in SITUATION.items() if k != 'balls'}, SITUATION | {'half': None}):
        assert ask(reader, situation_before=missing) == (None, 'situation_unverified')


def test_mid_pa_change_quarantine_and_reconciliation_labels(tmp_path):
    held = Ledger(tmp_path/'l.sqlite3')
    first = held.attempt(ab=5, last=0, pitcher=100)       # made for the pitcher who was then replaced
    held.attempt(ab=5, last=2)
    held.realized()
    reader = view(held)
    assert ask(reader)[1] is None
    held.later(first, 'matchup_changed_quarantine')       # appended later: the whole PA stops answering
    assert ask(reader) == (None, 'matchup_changed:quarantine_in_plate_appearance')

    other = Ledger(tmp_path/'m.sqlite3')
    key = other.attempt()
    other.realized()
    reader = view(other)
    other.poll(matchup={'pitcher': 111, 'batter': 222})
    assert ask(reader)[1] is None
    other.poll(ab=None, observed={'at_bat_number': 6, 'pitcher': 300, 'batter': 222})  # another PA: no effect
    assert ask(reader)[1] is None
    other.poll(ab=None, observed={'at_bat_number': 5, 'pitcher': 111, 'batter': 400})  # seen only by a slot-less poll
    assert ask(reader) == (None, 'matchup_changed:another_player_observed_in_plate_appearance')

    third = Ledger(tmp_path/'n.sqlite3')
    key = third.attempt()
    third.realized()
    reader = view(third)
    third.physical(key=key)
    assert ask(reader)[1] is None
    third.physical(label='attempt_unlinked_matchup_changed', key=key)
    assert ask(reader) == (None, 'unlinked:attempt_unlinked_matchup_changed')

    fourth = Ledger(tmp_path/'o.sqlite3')
    key = fourth.attempt()
    fourth.realized()
    fourth.physical(key=f'{GAME}:5:after9')              # linked, but to another attempt
    assert ask(view(fourth)) == (None, 'unlinked:had_attempt')
    fifth = Ledger(tmp_path/'p.sqlite3')
    key = fifth.attempt()
    fifth.realized()
    fifth.later(key, 'physical_order_reconciliation', {'label': 'provider_time_after_decision_unverified'})
    reader = view(fifth)
    assert ask(reader)[1] is None
    fifth.later(key, 'physical_order_reconciliation', {'label': 'late'})
    assert ask(reader) == (None, 'recorded_late')
    for used in (held, other, third, fourth, fifth):
        used.db.close()


def test_ledger_rows_that_do_not_agree_show_nothing(tmp_path):
    cases = {
        'ledger_inconsistent': {'bad_sha': True},
        'leaf_order_unknown': {'leaf_order': LEAVES[::-1]},
        'model_unknown': {'kind': 'prequential_2026'},
        'cell_invalid': {'cells': {'FF': [0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]}},
        'cell_unsupported': {'forecast_cells': [{'action': 'FF', 'status': 'scored', 'reason': None, 'seven_leaf': seven(0.2, 0.2)}]},
        'forecast_inputs_unknown': {'state': {'bases': None}}}
    for reason, options in cases.items():
        held = Ledger(tmp_path/f'{reason}.sqlite3')
        held.attempt(**options)
        held.realized()
        assert ask(view(held)) == (None, reason)
        held.db.close()
    held = Ledger(tmp_path/'nan.sqlite3')
    held.attempt(cells={'FF': [0.2, 0.2, 0.2, 'x', 0.2, 0.1, 0.1]})
    held.realized()
    assert ask(view(held)) == (None, 'cell_invalid')
    held.db.close()


# -- "recorded before the pitch" ------------------------------------------------------------------------------------
def test_replay_proves_pre_pitch_inputs_only_when_the_ledger_shows_the_order(tmp_path):
    held = Ledger(tmp_path/'l.sqlite3')
    held.attempt(written=10)
    reader = view(held)
    assert ask(reader) == (None, 'written_after_pitch_shown:pitch_not_observed_in_ledger')
    held.realized(index=10)                                # written from the body that already showed the pitch
    assert ask(reader) == (None, 'written_after_pitch_shown')
    held.attempt(ab=6, written=10)
    held.realized(ab=6, index=14)
    held.realized(ab=6, index=12)
    assert ask(reader, ab=6)[0]['recorded'] == 'replayed_pre_pitch_inputs'
    # a first receipt time does not turn a replay into "live": the forecast was computed after the game
    assert ask(reader, ab=6, pitch_first_received=10.0**10)[0]['recorded'] == 'replayed_pre_pitch_inputs'
    held.db.close()


def test_live_needs_a_decision_before_the_store_first_received_the_pitch(tmp_path):
    held = Ledger(tmp_path/'l.sqlite3', run=LIVE_RUN)
    held.later(held.attempt(clock=LIVE, decision_unix=1000.0, compute_ms=200.0), *ORDERED)
    reader = SE.ShadowExpectations(held.path, enabled=True)   # allow_replayed stays off
    shown, why = ask(reader, pitch_first_received=1008.0)
    assert why is None and shown['recorded'] == 'live_before_pitch' and shown['lead'] == '던지기 전 예상'
    assert shown['text'] == '던지기 전 예상: 헛스윙 12% · 땅볼 9%' and shown['note'] == '2024 모델 · 참고'
    assert ask(reader) == (None, 'pitch_first_seen_unknown')
    assert ask(reader, pitch_first_received=float('nan')) == (None, 'pitch_first_seen_unknown')
    assert ask(reader, pitch_first_received=1000.0) == (None, 'recorded_after_pitch_appeared')
    assert ask(reader, pitch_first_received=999.0) == (None, 'recorded_after_pitch_appeared')
    assert ask(reader, pitch_first_received=1001.1) == (None, 'recorded_after_pitch_appeared')  # 1000 + 0.2 + 1.0 lead
    assert ask(reader, pitch_first_received=1001.3)[1] is None
    assert ask(SE.ShadowExpectations(held.path, enabled=True, min_lead_s=5), pitch_first_received=1004.0) == (
        None, 'recorded_after_pitch_appeared')
    held.realized(index=10)                                # the sidecar's own order evidence still applies
    assert ask(reader, pitch_first_received=1008.0) == (None, 'written_after_pitch_shown')
    held.db.close()


def test_unproven_clock_labels_never_count(tmp_path):
    for name, options, run in (
            ('unknown', {'clock': {'clock': 'UNKNOWN', 'source': 'live_poll'}}, LIVE_RUN),
            ('mixed', {'clock': {'clock': SE.LIVE_CLOCK, 'source': SE.REPLAY_SOURCE}}, LIVE_RUN),
            ('receipt', {'clock': LIVE, 'receipt_clock': 'UNKNOWN'}, LIVE_RUN),
            ('replay_run', {'clock': LIVE}, None),          # live labels inside a run declared a post-hoc replay
            ('no_time', {'clock': LIVE, 'decision_unix': None}, LIVE_RUN)):
        held = Ledger(tmp_path/f'{name}.sqlite3', run=run)
        held.later(held.attempt(**options), *ORDERED)
        held.realized()
        shown, why = ask(view(held), pitch_first_received=10.0**10)
        assert shown is None and why in ('provenance_unknown', 'pitch_first_seen_unknown'), (name, why)
        held.db.close()


# -- the file: missing, corrupt, appended to, replaced, bounded -----------------------------------------------------
def test_missing_empty_and_corrupt_ledgers_never_raise(tmp_path):
    path = tmp_path/'later.sqlite3'
    reader = SE.ShadowExpectations(path, enabled=True, allow_replayed=True)
    assert ask(reader) == (None, 'ledger_missing') and not path.exists()
    path.write_bytes(b'')
    assert ask(reader) == (None, 'ledger_not_ready')
    path.write_bytes(b'not a database, just some bytes'*10)
    assert ask(reader) == (None, 'ledger_unreadable:not_sqlite')
    path.unlink()
    sqlite3.connect(path).executescript('CREATE TABLE other(x); INSERT INTO other VALUES (1);')
    assert ask(reader) == (None, 'ledger_not_ready')        # a database without the ledger tables yet
    path.unlink()
    held = Ledger(path)                                     # the sidecar creates it: the same reader now answers
    held.attempt()
    held.realized()
    assert ask(reader)[1] is None
    held.db.close()
    raw = path.read_bytes()
    broken = tmp_path/'broken.sqlite3'
    broken.write_bytes(raw[:4096]+os.urandom(len(raw)-4096))
    shown, why = ask(SE.ShadowExpectations(broken, enabled=True, allow_replayed=True))
    assert shown is None and why.split(':')[0] in ('ledger_unreadable', 'ledger_not_ready', 'no_forecast', 'error')
    assert reader.status()['counts']['ledger_missing'] == 1 and reader.status()['counts']['shown'] == 1
    reader.close()


def test_nothing_is_written_and_no_side_file_appears(tmp_path):
    for wal in (False, True):
        folder = tmp_path/('wal' if wal else 'delete')
        folder.mkdir()
        held = Ledger(folder/'shadow-ledger.sqlite3', wal=wal)
        held.attempt()
        held.realized()
        held.db.close()                                     # no writer attached: a WAL ledger has no side files now
        before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in folder.iterdir()}
        reader = view(held)
        assert ask(reader)[1] is None and ask(reader)[1] is None
        reader.close()
        assert {p.name: (p.stat().st_size, p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                for p in folder.iterdir()} == before, wal


def test_wal_ledger_with_a_writer_attached_is_read_live_and_unrecovered_wal_is_refused(tmp_path):
    held = Ledger(tmp_path/'shadow-ledger.sqlite3', wal=True)
    held.attempt()
    held.realized()
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ['shadow-ledger.sqlite3', 'shadow-ledger.sqlite3-shm', 'shadow-ledger.sqlite3-wal']
    reader = view(held)
    assert ask(reader)[1] is None
    held.attempt(ab=6)                                      # still only in the WAL
    held.realized(ab=6)
    assert ask(reader, ab=6)[1] is None and sorted(p.name for p in tmp_path.iterdir()) == names
    reader.close()
    held.db.close()
    orphan = tmp_path/'orphan'
    orphan.mkdir()
    lone = Ledger(orphan/'l.sqlite3', wal=True)
    lone.attempt()
    lone.db.close()
    (orphan/'l.sqlite3-wal').write_bytes(b'frames a crashed writer left behind')
    assert ask(SE.ShadowExpectations(orphan/'l.sqlite3', enabled=True, allow_replayed=True)) == (None, 'ledger_wal_unrecovered')
    assert sorted(p.name for p in orphan.iterdir()) == ['l.sqlite3', 'l.sqlite3-wal']


def test_the_reader_cannot_write(ledger):
    reader = view(ledger)
    assert ask(reader)[1] is None
    held = next(iter(reader._ledgers.values()))
    for statement in ("INSERT INTO shadow_realized(game_pk, at_bat_number, pitch_number, observed_at_replay_index, sha) VALUES (1,1,1,1,'z')",
                      'CREATE TABLE t(x)', 'PRAGMA wal_checkpoint(TRUNCATE)', 'VACUUM'):
        try:
            held.db.execute(statement).fetchall()
        except sqlite3.Error:
            pass
    assert ledger.db.execute('SELECT COUNT(*) FROM shadow_realized').fetchone()[0] == 1
    assert ledger.db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 't'").fetchone()[0] == 0
    assert ledger.db.execute('PRAGMA journal_mode').fetchone()[0] == 'delete'


def test_rows_appended_by_the_sidecar_are_read_incrementally(ledger):
    reader = view(ledger)
    assert ask(reader, ab=6) == (None, 'no_forecast')
    marks = dict(next(iter(reader._ledgers.values())).marks)
    ledger.attempt(ab=6)
    ledger.realized(ab=6)
    assert ask(reader, ab=6)[1] is None
    held = next(iter(reader._ledgers.values()))
    assert held.marks['forecast'] == marks['forecast']+1 and held.marks['realized'] == marks['realized']+1
    assert held.rows == 4 and len(held.forecasts) == 2      # each row taken in once
    for _ in range(20):
        ask(reader, ab=6)
    assert held.rows == 4


def test_a_replaced_ledger_file_forgets_everything_held(tmp_path):
    path = tmp_path/'l.sqlite3'
    first = Ledger(path)
    first.attempt()
    first.realized()
    reader = view(first)
    assert ask(reader)[0]['items'][0]['pct'] == 12
    first.db.close()
    fresh = Ledger(tmp_path/'new.sqlite3')
    fresh.attempt(cells={'FF': seven(0.20, 0.05)})
    fresh.realized()
    fresh.db.close()
    os.replace(tmp_path/'new.sqlite3', path)
    assert ask(reader)[0]['items'][0]['pct'] == 20
    reader.close()


def test_memory_stays_bounded(tmp_path):
    held = Ledger(tmp_path/'l.sqlite3')
    for ab in range(1, 31):
        held.attempt(ab=ab)
        held.realized(ab=ab)
    reader = view(held, max_cached=4)
    assert all(ask(reader, ab=ab)[1] is None for ab in range(1, 31))
    assert len(next(iter(reader._ledgers.values())).slim) == 4
    small = view(held, max_index_rows=10)
    assert ask(small) == (None, 'ledger_too_large') and ask(small) == (None, 'ledger_too_large')
    assert next(iter(small._ledgers.values())).forecasts == {} or next(iter(small._ledgers.values())).rows <= 11
    huge = Ledger(tmp_path/'huge.sqlite3')
    huge.attempt(cells={'FF': seven()})
    huge.realized()
    assert ask(view(huge, max_body_bytes=100)) == (None, 'ledger_inconsistent')  # an oversized record is not parsed
    held.db.close()
    huge.db.close()


def test_one_ledger_per_game_template(tmp_path):
    for game in (GAME, GAME+1, GAME+2):
        folder = tmp_path/str(game)
        folder.mkdir()
        held = Ledger(folder/'shadow-ledger.sqlite3')
        held.attempt(game=game, cells={'FF': seven(0.10+0.01*(game-GAME), 0.05)})
        held.realized(game=game)
        held.db.close()
    reader = SE.ShadowExpectations(str(tmp_path/'{game_pk}'/'shadow-ledger.sqlite3'), enabled=True, allow_replayed=True, max_ledgers=2)
    names = {'pa_event': 'home_run', 'pitcher_id': 111, 'batter_id': 222, 'situation_before': SITUATION}
    assert [reader.explain(game, 5, 3, 'FF', **names)[0]['items'][0]['pct'] for game in (GAME, GAME+1, GAME+2)] == [10, 11, 12]
    assert len(reader.status()['ledgers']) == 2
    assert reader.explain(GAME+7, 5, 3, 'FF', **names) == (None, 'ledger_missing')
    assert reader.explain(GAME+1, 5, 3, 'FF', **(names | {'pitcher_id': 1})) == (None, 'different_pitcher')
    reader.close()


def test_readers_in_threads_while_a_sidecar_appends(tmp_path):
    held = Ledger(tmp_path/'l.sqlite3')
    held.attempt(ab=1)
    held.realized(ab=1)
    reader = view(held, busy_timeout_s=0.5)
    stop, seen, errors = threading.Event(), [], []

    def read():
        try:
            while not stop.is_set():
                for ab in range(1, 41):
                    shown, why = ask(reader, ab=ab)
                    seen.append(why)
                    if shown is not None and shown['text'] != '던지기 전 정보로 본 예상: 헛스윙 12% · 땅볼 9%':
                        errors.append(shown)
                    if shown is None and why.split(':')[0] not in ('no_forecast', 'ledger_busy', 'written_after_pitch_shown'):
                        errors.append(why)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
    threads = [threading.Thread(target=read) for _ in range(4)]
    for thread in threads:
        thread.start()
    writer = sqlite3.connect(held.path, isolation_level=None, timeout=10)   # another connection, as a sidecar process has
    held.db.close()
    held.db = writer
    for ab in range(2, 41):
        held.attempt(ab=ab)
        held.realized(ab=ab)
        time.sleep(0.002)
    time.sleep(0.05)
    stop.set()
    for thread in threads:
        thread.join(10)
    assert not errors, errors[:3]
    assert all(ask(reader, ab=ab)[1] is None for ab in range(1, 41))
    assert writer.execute('SELECT COUNT(*) FROM shadow_forecasts').fetchone()[0] == 40
    assert sorted(p.name for p in tmp_path.iterdir()) == ['l.sqlite3']
    writer.close()


def test_status_is_operator_readable(ledger):
    reader = view(ledger)
    ask(reader)
    ask(reader, pitcher_id=5)
    state = reader.status()
    assert state['enabled'] and state['allow_replayed'] and state['counts'] == {'shown': 1, 'different_pitcher': 1}
    assert state['last_reason'] == 'different_pitcher' and state['ledgers'][0]['run']['clock'] == SE.REPLAY_CLOCK
    assert set(why.split(':')[0] for why in ('different_pitcher', 'matchup_changed:x')) <= set(SE.REASONS)


# -- the one reviewed real ledger (game 849831; SERVICE_EXPOSED_DEV descriptive checks, not model quality) -----------
def _real_game():
    if not REAL.is_file() or not REAL_REPLAY.is_dir():
        pytest.skip('the reviewed 849831 shadow ledger / replay folder is not on this machine')
    from observer_app import game_history
    last = sorted(REAL_REPLAY.glob('849831_*.json.gz'))[-1]
    with gzip.open(last) as handle:
        return game_history.build(json.load(handle))


def _folder_state(folder):
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir()}


def test_real_ledger_key_moments_849831():
    history = _real_game()
    before, digest = _folder_state(REAL.parent), hashlib.sha256(REAL.read_bytes()).hexdigest()
    last = [p for p in history['pitches'] if SE.is_key_moment(p['actual']['event'])]
    by_event = {}
    for pitch in last:
        by_event[pitch['actual']['event']] = by_event.get(pitch['actual']['event'], 0)+1
    assert by_event == {'strikeout': 28, 'home_run': 1, 'grounded_into_double_play': 1}
    pas = [pa for pa in history['pas'] if SE.is_key_moment(pa['event'])]
    assert len(pas) == len(last) == 30 and all(pa['ends_on_pitch'] for pa in pas)

    assert all(SE.ShadowExpectations(REAL).for_pitch(p) is None for p in last)                 # flag off
    # The run's own body says ``fan_facing_output: false``: nothing of it is ever shown, with or without allow_replayed
    for options in ({}, {'allow_replayed': True}):
        refused = SE.ShadowExpectations(REAL, enabled=True, **options)
        assert all(refused.for_pitch(p) is None for p in last) and refused.status()['counts'].get('shown', 0) == 0
        assert set(refused.status()['counts']) <= {'run_not_granted_fan_facing', 'cell_unsupported', 'matchup_changed'}
        assert refused.status()['counts']['run_not_granted_fan_facing'] == 25
        refused.close()
    # The rest checks the JOIN on real rows, offline (``require_fan_grant=False`` is never used by a serving reader)
    locked = SE.ShadowExpectations(REAL, enabled=True, require_fan_grant=False)                # a replay cannot be "live"
    assert {locked.explain(849831, p['at_bat_number'], p['pitch_number'], p['actual']['pitch_type'], pa_event=p['actual']['event'],
                           pitcher_id=p['pitcher']['id'], batter_id=p['batter']['id'], situation_before=p['situation_before'],
                           pitch_first_received=10.0**10)[1] for p in last} <= {
        'replayed_not_allowed', 'cell_unsupported', 'matchup_changed:quarantine_in_plate_appearance'}
    assert locked.status()['counts'].get('shown', 0) == 0

    reader = SE.ShadowExpectations(REAL, enabled=True, allow_replayed=True, require_fan_grant=False)
    shown, reasons = {}, {}
    for pitch in last:
        result = reader.for_pitch(pitch)
        if result is None:
            reasons[reader.status()['last_reason']] = reasons.get(reader.status()['last_reason'], 0)+1
        else:
            shown[pitch['key']] = result
    assert reasons == {'cell_unsupported': 3, 'matchup_changed:quarantine_in_plate_appearance': 2}
    assert len(shown) == 25 and all(r['recorded'] == 'replayed_pre_pitch_inputs' and not r['omitted'] for r in shown.values())
    assert all(1 <= i['pct'] < 30 for r in shown.values() for i in r['items'])
    assert shown['849831:2:3']['text'] == '던지기 전 정보로 본 예상: 헛스윙 18% · 땅볼 4%'
    assert shown['849831:24:5']['text'] == '던지기 전 정보로 본 예상: 헛스윙 11% · 땅볼 13%'       # the home run
    assert shown['849831:51:3']['text'] == '던지기 전 정보로 본 예상: 헛스윙 16% · 땅볼 11%'       # the double play
    # the displayed numbers are the stored cell of the thrown type, read back independently
    db = sqlite3.connect(f'file:{REAL}?mode=ro', uri=True)
    for key, result in shown.items():
        _, ab, number = key.split(':')
        cells = json.loads(db.execute('SELECT action_probabilities FROM shadow_forecasts WHERE at_bat_number=? AND '
                                      'candidate_pitch_number=?', (int(ab), int(number))).fetchone()[0])
        cell = next(c for c in cells if c['action'] == result['pitch_type'])
        assert [i['pct'] for i in result['items']] == [SE.percent(cell['seven_leaf'][3])[0], SE.percent(cell['seven_leaf'][5])[0]]
    db.close()
    reader.close()
    assert _folder_state(REAL.parent) == before and hashlib.sha256(REAL.read_bytes()).hexdigest() == digest


def test_real_ledger_agrees_with_the_runner_reconciliation_on_every_pitch():
    """Every pitch of the game, asked as if it were a key moment: the module answers only for pitches the runner
    itself linked (``had_attempt``) and never for an unlinked, skipped or abstained one."""
    history = _real_game()
    reader = SE.ShadowExpectations(REAL, enabled=True, allow_replayed=True, require_fan_grant=False)
    db = sqlite3.connect(f'file:{REAL}?mode=ro', uri=True)
    labels = {(ab, n): label for ab, n, label in db.execute('SELECT at_bat_number, pitch_number, label FROM physical_reconciliation')}
    scored = {(ab, n) for ab, n in db.execute("SELECT at_bat_number, candidate_pitch_number FROM shadow_forecasts WHERE decision='scored'")}
    db.close()
    counts = {}
    for pitch in history['pitches']:
        at = (pitch['at_bat_number'], pitch['pitch_number'])
        result, why = reader.explain(849831, *at, pitch['actual']['pitch_type'], pa_event='strikeout',
                                     pitcher_id=pitch['pitcher']['id'], batter_id=pitch['batter']['id'],
                                     situation_before=pitch['situation_before'])
        counts[why] = counts.get(why, 0)+1
        if result is not None:
            assert labels[at] == 'had_attempt' and at in scored
        if labels[at] != 'had_attempt':
            assert result is None
    assert len(history['pitches']) == 257 and counts == {
        None: 211, 'cell_unsupported': 29, 'matchup_changed:quarantine_in_plate_appearance': 11, 'no_forecast': 3,
        'different_situation:balls,strikes': 2, 'forecast_abstained': 1}
    reader.close()


# ---------- review round 2: E2 (what "before the pitch" may claim), E3 (the run's own grant) ----------
def test_live_wording_needs_the_runners_order_reconciliation_not_local_receipt_order_alone(tmp_path):
    held = Ledger(tmp_path/'l.sqlite3', run=LIVE_RUN)
    key = held.attempt(clock=LIVE, decision_unix=1000.0)
    strict, lenient = SE.ShadowExpectations(held.path, enabled=True), SE.ShadowExpectations(held.path, enabled=True, allow_replayed=True)
    # decided 10 s before the pitch's record reached us, but no order reconciliation row yet
    assert ask(strict, pitch_first_received=1010.0) == (None, 'order_not_reconciled')
    weaker, why = ask(lenient, pitch_first_received=1010.0)
    assert why is None and weaker['recorded'] == 'live_before_pitch_record_arrived' and weaker['lead'] == '던지기 전 정보로 본 예상'
    assert weaker['text'] == '던지기 전 정보로 본 예상: 헛스윙 12% · 땅볼 9%' and weaker['note'] == '2024 모델 · 참고 · 이 공의 기록이 도착하기 전에 계산'
    assert '경기 뒤' not in weaker['note'] and weaker['lead'] != SE.LEAD_LIVE
    assert ask(lenient, pitch_first_received=1000.5) == (None, 'recorded_after_pitch_appeared')  # the receipt order still has to hold
    held.later(key, *ORDERED)  # the runner reconciles: the provider's pitch time is after the decision
    for reader in (strict, lenient):
        shown, why = ask(reader, pitch_first_received=1010.0)
        assert why is None and (shown['recorded'], shown['lead'], shown['note']) == ('live_before_pitch', '던지기 전 예상', '2024 모델 · 참고')
    held.later(key, 'physical_order_reconciliation', {'label': 'late'})  # ... and a late label withdraws it for good
    assert ask(strict, pitch_first_received=1010.0) == ask(lenient, pitch_first_received=1010.0) == (None, 'recorded_late')
    held.db.close()


@pytest.mark.parametrize('body', [{'fan_facing_output': False}, {'fan_facing_output': None}, {'fan_facing_output': 'true'},
                                  {'fan_facing_output': 1}, {'grant': {'fan_facing_output': True}, 'fan_facing_output': False}])
def test_a_run_that_does_not_explicitly_allow_fan_facing_output_shows_nothing(tmp_path, body):
    run = {'schema': 'step5-shadow-run-v1', 'mode': 'actual', 'clock': SE.REPLAY_CLOCK, 'game_pk': GAME} | body
    held = Ledger(tmp_path/'l.sqlite3', run=run)
    held.attempt()
    held.realized()
    for options in ({}, {'allow_replayed': True}):
        reader = SE.ShadowExpectations(held.path, enabled=True, **options)
        assert ask(reader) == (None, 'run_not_granted_fan_facing') if options else ask(reader)[0] is None
        assert reader.status()['counts'].get('shown', 0) == 0
        reader.close()
    held.db.close()


def test_a_ledger_without_a_run_record_shows_nothing(tmp_path):
    held = Ledger(tmp_path/'l.sqlite3', run=False)
    held.attempt()
    held.realized()
    assert ask(view(held)) == (None, 'run_not_granted_fan_facing')
    held.db.close()
