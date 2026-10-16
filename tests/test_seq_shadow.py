"""Synthetic tests of the sequence-model shadow (job 20261012-seq-shadow-001).

Fixture replay folders, a fixture profile, randomly initialised N1 weights, fabricated calibrators and a fixture
"Step 5 ledger" made by the pinned Step 5 shadow runner with its synthetic heads. No capture, no replay of a real
game, no real weight is deserialized, no real array is decoded. Registered files are only hashed (one test), and
these JSON parameter files are parsed: the architecture config, the scale-002 encoder, F1-design.json, MODELS.json.
Processes: nothing is signalled; the guard is given fabricated ``vm_stat`` / ``ps`` text.
"""
from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO/'src') not in sys.path:
    sys.path.insert(0, str(REPO/'src'))

import pitcheezy.seq_shadow as SS  # noqa: E402
from pitcheezy.seq_shadow import adapter as AD  # noqa: E402
from pitcheezy.seq_shadow import calibration as CAL  # noqa: E402
from pitcheezy.seq_shadow import ledger as LED  # noqa: E402
from pitcheezy.seq_shadow import model as MOD  # noqa: E402
from pitcheezy.seq_shadow import summary as SUM  # noqa: E402
from pitcheezy.seq_shadow import tokens as TOK  # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONFIG = SS.load_config()
MODS = SS.load_step5_shadow(CONFIG)
S, G, RUN = MODS.S, MODS.G, MODS.RUN
STEP5 = S.load_config()
SRC = S.load(STEP5)
F, P, A = SRC.F, SRC.P, SRC.A
import profile_reader  # noqa: E402  (on sys.path after S.load; pinned bytes)

RUNNER = _by_path('seq_shadow_runner', REPO/'scripts'/'seq_shadow_runner.py')
PARITY = _by_path('seq_shadow_parity', REPO/'scripts'/'seq_shadow_parity.py')
_threads = {k: os.environ.get(k) for k in SS.THREAD_VARIABLES}
CL = _by_path('pinned_calibrate_logits_for_tests', CONFIG['calibration']['code']['path'])  # sets threads to 2
os.environ.update({k: v for k, v in _threads.items() if v is not None})
PROJECTOR = _by_path('pinned_projector_for_tests', CONFIG['tokens']['outcome10_source']['path'])
ARCH = json.loads(Path(CONFIG['model']['architecture']['path']).read_text())
L = ARCH['context']['max_length']
LEAVES = CONFIG['metrics']['leaf_order']

GAME, DAY, CUTOFF = 990001, '2099-01-02', '2099-01-01'
PA, PB, PC, PD = 101, 102, 103, 104  # A: all 17 types; B: absent from the profile; C: five types; D: donor
C_TYPES = ('FF', 'SL', 'CH', 'SI', 'CU')
VM_OK = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               100000.\n'
         'Pages active:                             500000.\nPages inactive:                           400000.\n'
         'Pages speculative:                         50000.\nPages wired down:                         100000.\n')
PS_QUIET = ('    1 Wed Sep 23 22:00:00 2026     /sbin/launchd\n'
            '  777 Tue Oct  6 09:00:00 2026     /usr/bin/some-unrelated-thing --flag\n')
TOKEN_KW = {'length': L, 'types': F.TYPES, 'description': RUN.DESCRIPTION,
            'pitch_number_range': CONFIG['tokens']['pitch_number_range'], 'balls_vocab': ARCH['context']['balls_vocab'],
            'strikes_vocab': ARCH['context']['strikes_vocab']}


def fake_run(vm=VM_OK, ps=PS_QUIET):
    return lambda argv: vm if argv[0].endswith('vm_stat') else ps


def fake_guard(config, **kw):
    return G.Guard(config, run=kw.pop('run', fake_run()), arm_hard_stops=False, **kw)


# ------------------------------------------------------------------ fixture profile (as the Step 5 shadow tests)
def profile_rows():
    rows, start, i = [], date.fromisoformat(CUTOFF)-timedelta(days=60), 0
    for k, (pitcher, types) in enumerate(((PA, F.ACTIONS), (PC, C_TYPES), (PD, F.ACTIONS))):
        for a, action in enumerate(types):
            for j in range(20):
                rows.append(dict(zip(F.DONOR_COLUMNS, (
                    9000000+i//40, 1+(i % 40)//5, 1+i % 5, start+timedelta(days=(7*a+3*j+k) % 60), pitcher, action,
                    80.+a+.25*(j % 7)+k, 1800.+40*a+11.*(j % 5), -1.+.1*a+.03*(j % 4), .5+.05*a+.02*(j % 6),
                    'R', 'LR'[j % 2], (j//2) % 4, (j//8) % 3, (a+j) % 10, (201, 202, 203)[(j+k) % 3],
                    ('ball', 'called_strike', 'swinging_strike', 'foul', 'hit_into_play')[(a+j) % 5],
                    ('ground_ball', 'fly_ball', '')[j % 3]))))
                i += 1
    return rows


def write_profile(directory):
    directory.mkdir(parents=True)
    store = F.ProfileStore(profile_rows(), as_of_cutoff=CUTOFF, source_id='synthetic-seq-shadow-fixture',
                           source_sha256='0'*64, role=profile_reader.ROLE)
    blob = json.dumps(profile_reader.dump_store(store)).encode()
    (directory/'PROFILE.json').write_bytes(blob)
    (directory/'RECEIPT.json').write_text(json.dumps({
        'terminal': 'COMMITTED', 'profile_committed': True, 'scope': profile_reader.SYNTHETIC_SCOPE,
        'attempt_id': 'synthetic-seq-shadow-fixture-001', 'counts': {'counts': {'rows_projected': store.rows}},
        'profile': {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}}))
    return A.load_committed(F, directory)


# ------------------------------------------------------------------ fixture replay
def ev(n, kind, call, balls, strikes, *, in_play=False, out=False, traj=None, speed=95.0):
    e = {'isPitch': True, 'type': 'pitch', 'pitchNumber': n, 'playId': f'synthetic-{n}-{kind}-{call}',
         'startTime': DAY+'T01:00:00Z', 'endTime': DAY+'T01:00:05Z',
         'details': {'description': 'synthetic', 'call': {'code': call}, 'isInPlay': in_play, 'isOut': out},
         'count': {'balls': balls, 'strikes': strikes, 'outs': 0}, 'pitchData': {'startSpeed': speed}}
    if kind:
        e['details']['type'] = {'code': kind}
    if traj:
        e['hitData'] = {'trajectory': traj, 'launchSpeed': 101.5}
    return e


def play(index, pitcher, batter, events, complete, event_type=None, stand='R'):
    counted = [e for e in events if 'count' in e]
    count = counted[-1]['count'] if counted else {'balls': 0, 'strikes': 0, 'outs': 0}
    result = {'homeScore': 0, 'awayScore': 0, 'description': 'synthetic result'}
    if event_type:
        result['eventType'] = event_type
    return {'about': {'atBatIndex': index, 'inning': 1, 'isTopInning': True, 'isComplete': complete},
            'count': dict(count), 'result': result,
            'matchup': {'pitcher': {'id': pitcher}, 'batter': {'id': batter}, 'batSide': {'code': stand},
                        'pitchHand': {'code': 'R'}},
            'playEvents': events}


def scripted(index, pitcher, batter, spec, event_type, traj=None):
    """A complete play from [(type code or None, call code) | 'AUTO'] with ordinary count rules."""
    events, balls, strikes, n = [], 0, 0, 0
    for item in spec:
        if item == 'AUTO':  # automatic ball: a counted event that is not a pitch
            balls += 1
            events.append({'isPitch': False, 'type': 'no_pitch', 'details': {'call': {'code': 'V'}, 'description': 'Automatic Ball'},
                           'count': {'balls': balls, 'strikes': strikes, 'outs': 0}})
            continue
        kind, call = item
        n += 1
        balls += call == 'B'
        strikes += call in ('C', 'S', 'L') or (call == 'F' and strikes < 2)
        in_play = call in ('X', 'D', 'E')
        events.append(ev(n, kind, call, balls, strikes, in_play=in_play, out=call == 'X', traj=traj if in_play else None))
    return play(index, pitcher, batter, events, True, event_type)


def done_plays():
    """Seven complete plate appearances of pitcher A, 35 pitches: one held in play (field_error), one automatic
    ball, one pitch without a type code, a foul bunt at two strikes, one type code outside the 19 frozen types."""
    return [
        scripted(0, PA, 201, [('FF', 'B'), ('SL', 'C'), ('CH', 'F'), ('FF', 'B'), ('SI', 'X')], 'field_out', 'ground_ball'),
        scripted(1, PA, 202, [('FF', 'C'), ('CU', 'B'), ('SL', 'S'), ('FF', 'F'), ('CH', 'D')], 'single', 'line_drive'),
        scripted(2, PA, 203, [('SI', 'B'), ('FF', 'B'), ('SL', 'C'), ('FC', 'F'), ('FF', 'E')], 'field_error', 'ground_ball'),
        scripted(3, PA, 201, [('FF', 'B'), 'AUTO', ('SL', 'C'), ('CH', 'B'), ('FF', 'C'), ('SI', 'S')], 'strikeout'),
        scripted(4, PA, 202, [('FF', 'C'), (None, 'B'), ('SL', 'F'), ('KC', 'B'), ('FF', 'X')], 'grounded_into_double_play', 'ground_ball'),
        scripted(5, PA, 203, [('FF', 'C'), ('SL', 'S'), ('SI', 'L')], 'strikeout'),
        scripted(6, PA, 201, [('XX', 'B'), ('FF', 'C'), ('SL', 'F'), ('CH', 'B'), ('FF', 'F'), ('CU', 'F'), ('SI', 'D')], 'home_run', 'fly_ball'),
    ]


def body(done, current, tick, state='Live', day=DAY):
    plays = {'allPlays': done+([current] if current else [])}
    if current:
        plays['currentPlay'] = current
    return {'gamePk': GAME, 'metaData': {'timeStamp': f'20990102_01{tick//60:02d}{tick % 60:02d}'},
            'gameData': {'game': {'pk': GAME}, 'datetime': {'officialDate': day}, 'status': {'abstractGameState': state}},
            'liveData': {'plays': plays,
                         'linescore': {'currentInning': 1, 'isTopInning': True, 'offense': {'first': {'id': 9}}}}}


def script():
    """[(name, body)] in receipt order (19 bodies): pre-game, an open PA after 35 earlier pitches of the same
    pitcher (truncation), duplicate stamp, revision, untyped transient, PA end, unsupported pitcher, pitcher change
    inside a PA, late correction of an earlier pitch type, a skipped first pitch, hit by pitch, game end."""
    D = done_plays()
    a1, a2 = ev(1, 'FF', 'B', 1, 0), ev(2, 'SL', 'C', 1, 1)
    a2_untyped = ev(2, None, 'C', 1, 1)
    a3 = ev(3, 'FF', 'X', 1, 1, in_play=True, out=True, traj='ground_ball')
    a1_fixed = ev(1, 'SI', 'B', 1, 0)  # later correction of an earlier pitch's type
    b1, b2 = ev(1, 'CH', 'B', 1, 0), ev(2, 'SL', 'S', 1, 1)
    b3 = ev(3, 'KC', 'X', 1, 1, in_play=True, out=True, traj='fly_ball')
    c1, c2, c3 = ev(1, 'FF', 'C', 0, 1), ev(2, 'KC', 'F', 0, 2), ev(3, 'SL', 'H', 0, 2)
    d1 = ev(1, 'FF', 'B', 1, 0)
    pa8 = play(7, PA, 201, [a1, a2, a3], True, 'field_out')
    pa8_fixed = play(7, PA, 201, [a1_fixed, a2, a3], True, 'field_out')
    pa9 = play(8, PC, 202, [b1, b2, b3], True, 'field_out')
    pa10 = play(9, PC, 203, [c1, c2, c3], True, 'hit_by_pitch')
    revised = play(7, PA, 201, [ev(1, 'FF', 'B', 1, 0, speed=96.4)], False)
    E = D+[pa8_fixed, pa9]
    return [
        ('pregame', body([], None, 0, 'Preview')),
        ('pa8_open', body(D, play(7, PA, 201, [], False), 5)),
        ('pa8_p1', body(D, play(7, PA, 201, [a1], False), 10)),
        ('pa8_p1_duplicate', body(D, play(7, PA, 201, [a1], False), 10)),
        ('pa8_p1_revision', body(D, revised, 16)),
        ('pa8_p2_untyped', body(D, play(7, PA, 201, [a1, a2_untyped], False), 20)),
        ('pa8_p2', body(D, play(7, PA, 201, [a1, a2], False), 25)),
        ('pa8_done', body(D, pa8, 30)),
        ('pa9_open_B', body(D+[pa8], play(8, PB, 202, [], False), 35)),
        ('pa9_p1_B', body(D+[pa8], play(8, PB, 202, [b1], False), 40)),
        ('pa9_p1_sub_C', body(D+[pa8], play(8, PC, 202, [b1], False), 45)),
        ('pa9_p2_C', body(D+[pa8], play(8, PC, 202, [b1, b2], False), 50)),
        ('pa9_done_and_correction', body(D+[pa8_fixed], pa9, 55)),
        ('pa10_p1_no_open_body', body(E, play(9, PC, 203, [c1], False), 60)),
        ('pa10_p2', body(E, play(9, PC, 203, [c1, c2], False), 65)),
        ('pa10_done', body(E, pa10, 70)),
        ('pa11_open', body(E+[pa10], play(10, PA, 202, [], False), 75)),
        ('pa11_p1', body(E+[pa10], play(10, PA, 202, [d1], False), 80)),
        ('final', body(E+[pa10], play(10, PA, 202, [d1], False), 85, 'Final')),
    ]


def write_replay(folder, bodies, *, spacing=5.0, fixture=True):
    folder.mkdir(parents=True)
    files = []
    for i, (name, feed) in enumerate(bodies):
        raw = F.canonical(feed)
        file = f'{GAME}_{i+1:06d}_{name}.json.gz'
        (folder/file).write_bytes(gzip.compress(raw, mtime=0))
        t = 4070995200.0+1000+spacing*i
        files.append({'file': file, 'seq': i+1, 'version': 1, 'sha256': hashlib.sha256(raw).hexdigest(),
                      'first_fetched': t, 'fetched': t, 'event_wall': None, 'source': 'live_poll'})
    manifest = {'status': 'OK', 'game_pk': GAME, 'mode': {'bodies_modified': False}, 'files': files}
    if fixture:
        manifest['synthetic_fixture'] = True
    (folder/'MANIFEST.json').write_text(json.dumps(manifest))
    return folder


def rows(db_path, sql, *args):
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
        return db.execute(sql, args).fetchall()


# ------------------------------------------------------------------ fabricated calibrators / weights / config
def fabricate_cal_model(seed):
    """One fitted-model dict in the shape ``calibrate_logits.apply_arms`` takes, with real PAV blocks."""
    rng = np.random.default_rng(100+seed)
    q = rng.dirichlet(np.ones(7)*1.5, size=4000)
    y = np.array([rng.choice(7, p=row) for row in q])
    return {'T': {'parent': 1.0+.05*seed, 'narrow_whiff': .95+.02*seed, 'ground_ball_per_pitch': 1.1},
            'vec': {'w': 1+.1*rng.normal(size=10), 'b': .1*rng.normal(size=10)},
            'pw': {'a': 1.05, 'b': -.1}, 'pg': {'a': .9, 'b': .05},
            'dir': {'W': np.eye(10)+.05*rng.normal(size=(10, 10)), 'b': .05*rng.normal(size=10)},
            'lam': .01, 'blocks': CL.isotonic_fit(q, y), 'clip': 1e-6,
            'info': {'rows': 4000, 'whiff_rows': 10, 'bip_rows': 10, 'vector': {}, 'platt_narrow_whiff': {},
                     'platt_ground_ball_per_pitch': {}, 'dirichlet': {}, 'lambda_cv': {}}}


def write_calibration(directory, selected=None):
    selected = selected or {0: 'C3', 1: 'C2', 2: 'C2'}
    models = {s: fabricate_cal_model(s) for s in selected}
    result = {'leaf_order': LEAVES, 'cal_selected': {str(s): a for s, a in selected.items()},
              'seeds': {str(s): {'cal_selected': selected[s], 'parameters': CL.model_params(models[s])} for s in selected}}
    (directory/'result.json').write_text(json.dumps(result))
    sha = SS.file_sha256(directory/'result.json')
    (directory/'terminal.json').write_text(json.dumps({'success': True, 'result_sha256': sha,
                                                       'cal_selected': result['cal_selected']}))
    return models, {'result': {'path': str(directory/'result.json'), 'sha256': sha},
                    'terminal': {'path': str(directory/'terminal.json'), 'sha256': SS.file_sha256(directory/'terminal.json')}}


def make_config(work, **over):
    """A temp configuration: random-init weights and fabricated calibrators under the registered architecture,
    protected-process list emptied (the tests never look at real processes unless they say so)."""
    import torch
    work.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(CONFIG)
    config['protected'] = {'processes': [], 'command_words': []}
    config['heavy_lock_path'] = str(work/'no-such-heavy-lock')
    config['caps']['min_free_memory_bytes'] = 1 << 30
    config['bootstrap'] |= {'draws': 400, 'chunk': 100}
    T = MOD._train_module(Path(CONFIG['model']['train_seq']['path']),
                          Path(CONFIG['model']['train_seq']['path']).read_bytes())
    models, files = write_calibration(work)
    for s in config['model']['seeds']:
        torch.manual_seed(1000+s)
        net = T.build_model(torch, ARCH, 'N1')
        torch.save(net.state_dict(), work/f'model-state-{s}.pt')
        (work/f'temperatures-{s}.json').write_text(json.dumps({'temperatures': models[s]['T']}))
        config['model']['state'][str(s)] = {'path': str(work/f'model-state-{s}.pt'), 'sha256': SS.file_sha256(work/f'model-state-{s}.pt')}
        config['model']['temperatures'][str(s)] = {'path': str(work/f'temperatures-{s}.json'),
                                                   'sha256': SS.file_sha256(work/f'temperatures-{s}.json')}
    config['calibration'] |= files
    for key, value in over.items():
        config[key] = value
    (work/'config.json').write_text(json.dumps(config))
    return config


def step5_test_config(work):
    config = copy.deepcopy(STEP5)
    config['protected'] = {'processes': [], 'command_words': []}
    config['heavy_lock_path'] = str(work/'no-such-heavy-lock')
    return config


class World:
    """Everything the run tests share: built once per test session."""

    def __init__(self, tmp):
        self.tmp = tmp
        # the hand-computed context counts below were derived under the 'skip' policy; the registered config
        # uses 'unknown_token' (lead decision 2026-10-12), which the equivalence test covers for both policies
        self.config = make_config(tmp/'cfg', tokens=dict(CONFIG['tokens'], missing_type_policy='skip'))
        self.config_path = tmp/'cfg'/'config.json'
        self.config_sha = SS.file_sha256(self.config_path)
        self.profile = write_profile(tmp/'profile')
        self.registry = P.load_registry(STEP5['models']['path'], STEP5['models']['sha256'], STEP5['encoder']['sha256'])
        self.heads = RUN.synthetic_heads(P, self.registry)
        self.seq = RUNNER.load_seq(self.config)
        self.bodies = script()
        self.names = [n for n, _ in self.bodies]
        self.replay = write_replay(tmp/'replay', self.bodies)
        self.step5 = self.step5_run(self.replay, 'step5-main')
        self.rc, self.summary = self.run(self.replay, 'main')
        self.db = tmp/'main'/RUNNER.LEDGER_NAME

    def step5_run(self, replay, name):
        """The fixture 'Step 5 ledger': the pinned Step 5 shadow runner with its synthetic heads on a fixture."""
        cfg = step5_test_config(self.tmp)
        rc, _ = RUN.run_shadow(replay, self.tmp/name, config=cfg, src=SRC, profile=self.profile, heads=self.heads,
                               guard=fake_guard(cfg), mode='synthetic')
        assert rc == 0
        return self.tmp/name/'shadow-ledger.sqlite3'

    def ledger_only(self, name, source=None):
        """A copy of a Step 5 fixture ledger WITHOUT its SUMMARY.json (used with replay folders whose manifest
        differs from the one that ledger was made from)."""
        (self.tmp/name).mkdir()
        shutil.copy(source or self.step5, self.tmp/name/'shadow-ledger.sqlite3')
        return self.tmp/name/'shadow-ledger.sqlite3'

    def run(self, replay, name, *, step5=None, config=None, g=None, seq=None, **kw):
        config = config or self.config
        g = g or fake_guard(config)
        g.preflight()
        ledger = Path(step5 or self.step5)
        summary_path = ledger.parent/'SUMMARY.json'
        return RUNNER.run_seq_shadow(
            replay, self.tmp/name, config=config, config_sha256=self.config_sha, step5_config=STEP5, mods=MODS,
            src=SRC, profile=self.profile, heads=self.heads, guard=g, mode='synthetic',
            seq=seq or self.seq,
            step5_ledger={'path': ledger, 'sha256': SS.file_sha256(ledger),
                          'summary': json.loads(summary_path.read_text()) if summary_path.is_file() else None,
                          'summary_sha256': None}, **kw)

    def forecasts(self, db=None):
        names = ('attempt_key', 'written_at_replay_index', 'replay_file', 'pitcher', 'batter', 'decision', 'reason',
                 'step5_outcome', 'vectors_sha256', 'x_seq_sha256', 'tokens_json', 'tokens_sha256', 'logits_json',
                 'c0_ensemble_json', 'sel_ensemble_json', 'cal_arms_json', 'feed_sha256', 'row_sha256', 'compute_ms',
                 'step5_cells_status_json', 'context_sha256', 'at_bat_number', 'candidate_pitch_number')
        return {r[0]: dict(zip(names, r)) for r in rows(db or self.db, f"SELECT {','.join(names)} FROM seq_forecasts")}


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    return World(tmp_path_factory.mktemp('seq-shadow'))


# =================================================================== 1. token equivalence with the training path
def synthetic_season(seed=0, games=2, plate_appearances=60):
    """Statcast-like rows (one per GUMBO pitch event) and the same content as one GUMBO body per game."""
    rng = np.random.default_rng(seed)
    frame, bodies = [], {}
    kinds = ['FF', 'SL', 'CH', 'SI', 'CU', 'KC', 'FC', 'XX', None]
    kind_p = [.3, .2, .12, .12, .08, .06, .06, .03, .03]
    calls = ['B', 'C', 'S', 'F', 'L', 'X', 'H', 'T']
    call_p = [.34, .16, .1, .2, .04, .12, .02, .02]
    in_play_events = ['single', 'double', 'home_run', 'field_out', 'force_out', 'grounded_into_double_play', 'sac_fly',
                      'field_error', 'fielders_choice', 'catcher_interf', 'triple_play', None]
    for g in range(games):
        game, plays = 7000+g, []
        for index in range(plate_appearances):
            ab = index+1
            if index % 2 == 0:
                pitcher = 11 if ab <= 39 else 13  # pitcher change between plate appearances
            else:
                pitcher = 12
            batter = int(rng.choice([301, 302, 303]))
            events, balls, strikes, n, event_type, over = [], 0, 0, 0, None, False
            if rng.random() < .05:  # a glitch: a counted event that leaves the count outside 0..3
                events.append({'isPitch': False, 'type': 'action', 'details': {'description': 'glitch'},
                               'count': {'balls': 4, 'strikes': 0, 'outs': 0}})
                balls = 4
            while not over:
                if rng.random() < .06 and balls < 3:  # automatic ball: counted, not a pitch
                    balls += 1
                    events.append({'isPitch': False, 'type': 'no_pitch', 'details': {'call': {'code': 'V'}},
                                   'count': {'balls': balls, 'strikes': strikes, 'outs': 0}})
                    continue
                kind = kinds[int(rng.choice(len(kinds), p=kind_p))]
                call = calls[int(rng.choice(len(calls), p=call_p))]
                n += 1
                row = {'game_pk': game, 'at_bat_number': ab, 'pitch_number': n, 'pitcher': pitcher, 'batter': batter,
                       'pitch_type': kind, 'description': RUN.DESCRIPTION[call], 'events': None, 'balls': balls,
                       'strikes': strikes}
                in_play = call == 'X'
                if call == 'B':
                    balls += 1
                    over, event_type = balls >= 4, 'walk'
                elif call in ('C', 'S', 'T', 'L'):
                    strikes += 1
                    over, event_type = strikes >= 3, 'strikeout'
                elif call == 'F':
                    strikes += strikes < 2
                elif call == 'H':
                    over, event_type = True, 'hit_by_pitch'
                else:
                    over, event_type = True, in_play_events[int(rng.integers(len(in_play_events)))]
                over = over or n >= 14
                if over:
                    row['events'] = event_type  # Statcast: events only on the last pitch of the plate appearance
                frame.append(row)
                e = ev(n, kind, call, balls, strikes, in_play=in_play, out=False, traj='ground_ball' if in_play else None)
                events.append(e)
            plays.append(play(index, pitcher, batter, events, True, event_type))
        bodies[game] = {'gamePk': game, 'liveData': {'plays': {'allPlays': plays, 'currentPlay': plays[-1]}}}
    return frame, bodies


def training_tokens(frame, missing_type):
    """Path (i): the accepted-corpus rule, ``sequence_arrays``-equivalent integer arrays, the pinned
    ``train_seq.build_context_index`` and ``train_seq.Data.tokens``. Returns (accepted rows, tokens dict)."""
    import torch
    T = MOD._train_module(Path(CONFIG['model']['train_seq']['path']), Path(CONFIG['model']['train_seq']['path']).read_bytes())
    lookup = {t: i for i, t in enumerate(F.TYPES)}
    accepted = []
    for r in frame:
        label, _ = PROJECTOR.outcome10(r['description'], r['events'], r['strikes'])  # the pinned original
        if label < 0 or not 0 <= r['balls'] <= 3 or not 0 <= r['strikes'] <= 2:
            continue
        if r['pitch_type'] is None and missing_type == 'skip':
            continue
        accepted.append(r | {'outcome10': label, 'type_index': lookup.get(r['pitch_type'], 0)})
    col = lambda k, dt=np.int64: np.array([r[k] for r in accepted], dtype=dt)  # noqa: E731
    seq = {'pitch_type': col('type_index', np.int16), 'outcome10': col('outcome10', np.int16), 'balls': col('balls', np.int16),
           'strikes': col('strikes', np.int16), 'batter': col('batter', np.int32)}
    ctx = T.build_context_index(col('pitcher'), col('game_pk'), col('at_bat_number'), col('pitch_number'), L)
    n = len(accepted)
    corpus = {'y10': col('outcome10'), 'narrow_whiff': np.zeros(n, np.int8), 'ground_ball_per_pitch': np.zeros(n, np.int8)}
    data = T.Data(torch, torch.device('cpu'), np.zeros((n, 1), np.float32), corpus, seq, ctx)
    tok = data.tokens(torch.arange(n))
    return accepted, {k: v.numpy() for k, v in tok.items()}


@pytest.mark.parametrize('missing_type', ['skip', 'unknown_token'])
def test_tokens_equal_the_training_path_for_every_target_pitch(missing_type):
    frame, bodies = synthetic_season()
    accepted, want = training_tokens(frame, missing_type)
    seen = {'short': 0, 'truncated': 0, 'empty': 0, 'same1': 0, 'same0': 0, 'skipped_unmapped_outcome': 0,
            'skipped_count_range': 0, 'skipped_no_type': 0, 'unknown_type_token': 0, 'pitchers': set()}
    for i, r in enumerate(accepted):
        state = {'pitcher': r['pitcher'], 'batter': r['batter'], 'at_bat_number': r['at_bat_number'],
                 'last_pitch_number': r['pitch_number']-1}
        # the FULL game body (later pitches and plays included): the builder itself must take only earlier pitches
        tok, counts = TOK.build(bodies[r['game_pk']], state, missing_type=missing_type, **TOKEN_KW)
        valid = want['valid'][i]
        assert np.array_equal(tok['valid'], valid), (i, r)
        assert np.array_equal(tok['pos'], want['pos'][i]) and tok['pos'].tolist() == list(range(1, L+1))
        for name in ('type', 'out', 'balls', 'strikes', 'same'):
            assert tok[name].dtype == np.int64
            assert np.array_equal(tok[name][valid], want[name][i][valid]), (name, i, r)
            assert not tok[name][~valid].any()
        assert counts['n_context'] == int(valid.sum())
        seen['short'] += 0 < counts['n_context'] < L
        seen['empty'] += counts['n_context'] == 0
        seen['truncated'] += counts['truncated'] > 0
        seen['same1'] += int(tok['same'][valid].sum() > 0)
        seen['same0'] += int((tok['same'][valid] == 0).sum() > 0)
        seen['unknown_type_token'] += int((tok['type'][valid] == 0).sum() > 0)
        seen['pitchers'].add(r['pitcher'])
        for k in ('skipped_unmapped_outcome', 'skipped_count_range', 'skipped_no_type'):
            seen[k] += counts[k] > 0
    # every listed edge case occurred: fewer than L, more than L (truncation), no context, batter changes, held
    # pitches dropped (unmapped in-play events, count outside the range, missing type code), interleaved pitchers,
    # a pitcher change between plate appearances, an unknown type code kept as UNKNOWN
    assert len(accepted) > 300 and seen['pitchers'] == {11, 12, 13}
    for k in ('short', 'truncated', 'empty', 'same1', 'same0', 'skipped_unmapped_outcome', 'skipped_count_range',
              'unknown_type_token'):
        assert seen[k] > 0, k
    assert (seen['skipped_no_type'] > 0) == (missing_type == 'skip')
    descriptions = {(r['description'], r['events']) for r in frame}
    assert {('hit_into_play', 'field_error'), ('hit_into_play', 'catcher_interf'), ('hit_into_play', None)} <= descriptions
    assert any(r['description'] == 'foul_bunt' and r['strikes'] == 2 for r in accepted)  # labelled strike, kept
    assert any(r['pitch_type'] is None for r in frame)


def test_tokens_handle_the_listed_single_cases():
    D = done_plays()
    kw = TOKEN_KW | {'missing_type': 'skip'}
    feed = body(D, play(7, PA, 201, [], False), 5)
    state, missing = F.pre_pitch_state(feed)
    assert not missing and (state['at_bat_number'], state['last_pitch_number'], state['pitcher']) == (8, 0, PA)
    tok, counts = TOK.build(feed, state, **kw)
    assert counts == {'n_context': 32, 'eligible_earlier_pitches': 33, 'skipped_no_type': 1,
                      'skipped_unmapped_outcome': 1, 'skipped_count_range': 0, 'truncated': 1}
    # slot 0 = the most recent earlier pitch: PA 7 pitch 7, SI, in play home run (7), pre-pitch count 2-2, batter 201
    assert [int(tok[k][0]) for k in ('type', 'out', 'balls', 'strikes', 'same', 'pos')] == [F.TYPES.index('SI'), 7, 2, 2, 1, 1]
    # PA 7 pitch 1 has the code XX (outside the 19 types) -> UNKNOWN token 0, kept at slot 6
    assert (int(tok['type'][6]), int(tok['out'][6]), int(tok['balls'][6]), int(tok['strikes'][6])) == (0, 0, 0, 0)
    # PA 6 (three pitches, slots 7..9): the foul bunt at two strikes is a strike (1), pre-pitch count 0-2
    assert [int(tok[k][7]) for k in ('type', 'out', 'balls', 'strikes', 'same')] == [F.TYPES.index('SI'), 1, 0, 2, 0]
    # PA 4 (automatic ball after pitch 1): pitch 2 was thrown at 2-0, not 1-0
    plays = {p['about']['atBatIndex']: p for p in D}
    kept, _ = TOK.earlier_pitches(feed, state, **{k: v for k, v in kw.items() if k != 'length'})
    by_key = {(p[0], p[1]): p for p in kept}
    assert by_key[(4, 2)][4:6] == (2, 0) and by_key[(4, 1)][4:6] == (0, 0)
    assert (3, 5) not in by_key and (5, 2) not in by_key  # held field_error; missing type code
    assert (3, 4) in by_key and (5, 3) in by_key and by_key[(5, 3)][4:6] == (1, 1)  # later pitches shift down, count intact
    assert plays[2]['result']['eventType'] == 'field_error'
    # with the other registered policy the untyped pitch is an UNKNOWN token
    _, other = TOK.build(feed, state, **(kw | {'missing_type': 'unknown_token'}))
    assert (other['skipped_no_type'], other['eligible_earlier_pitches']) == (0, 34)
    with pytest.raises(ValueError):
        TOK.build(feed, state, **(kw | {'missing_type': 'guess'}))
    # another pitcher: no context at all; pitches of the open PA count once (currentPlay duplicates allPlays[-1])
    tok_c, counts_c = TOK.build(feed, state | {'pitcher': PC}, **kw)
    assert counts_c['n_context'] == 0 and not tok_c['valid'].any()
    a1, a2 = ev(1, 'FF', 'B', 1, 0), ev(2, 'SL', 'C', 1, 1)
    feed2 = body([], play(0, PC, 203, [a1, a2], False), 5)
    state2, _ = F.pre_pitch_state(feed2)
    tok2, counts2 = TOK.build(feed2, state2, **kw)
    assert counts2['n_context'] == 2 and tok2['type'][:2].tolist() == [F.TYPES.index('SL'), F.TYPES.index('FF')]
    assert tok2['same'][:2].tolist() == [1, 1] and tok2['balls'][:2].tolist() == [1, 0]
    # an in-play pitch whose play has no eventType yet is not labelled (skipped), never guessed
    x = ev(1, 'FF', 'X', 0, 0, in_play=True, out=True, traj='fly_ball')
    feed3 = body([play(0, PC, 203, [x], True)], play(1, PC, 201, [], False), 5)
    state3, _ = F.pre_pitch_state(feed3)
    assert TOK.build(feed3, state3, **kw)[1]['skipped_unmapped_outcome'] == 1


def test_outcome10_is_the_pinned_projector_function():
    src = CONFIG['tokens']['outcome10_source']
    assert SS.file_sha256(src['path']) == src['sha256']
    theirs = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id: ast.dump(n)
              for n in ast.parse(Path(src['path']).read_text()).body
              if isinstance(n, ast.FunctionDef) or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name))}
    mine = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id: ast.dump(n)
            for n in ast.parse(Path(TOK.__file__).read_text()).body
            if isinstance(n, ast.FunctionDef) or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name))}
    for name in ('outcome10', '_null', 'OUTCOMES', '_BALL', '_STRIKE', '_OUT', '_DP'):
        assert mine[name] == theirs[name], name
    for d in list(RUN.DESCRIPTION.values())+[None, 'x']:
        for e in ('single', 'field_out', 'field_error', 'double_play', None, float('nan')):
            for s in (0, 1, 2):
                assert TOK.outcome10(d, e, s) == PROJECTOR.outcome10(d, e, s)


def test_token_builder_ignores_fields_it_must_not_read():
    feed = body(done_plays(), play(7, PA, 201, [ev(1, 'FF', 'B', 1, 0)], False), 5)
    state, _ = F.pre_pitch_state(feed)
    kw = TOKEN_KW | {'missing_type': 'skip'}
    base = TOK.to_json(*TOK.build(feed, state, **kw))
    other = copy.deepcopy(feed)
    other['liveData']['linescore'] = {'perturbed': True}
    other['metaData']['timeStamp'] = 'perturbed'
    for one in other['liveData']['plays']['allPlays']:
        one['result']['description'] = 'perturbed'
        one['about']['endTime'] = 'perturbed'
        for e in one['playEvents']:
            e['pitchData'] = {'startSpeed': 1.0, 'coordinates': {'pX': 9.9}}
            e['hitData'] = {'trajectory': 'perturbed', 'launchSpeed': 1.0}
            e['details']['description'] = 'perturbed'
            e['startTime'] = e['endTime'] = 'perturbed'
    other['liveData']['plays']['currentPlay'] = other['liveData']['plays']['allPlays'][-1]
    assert TOK.to_json(*TOK.build(other, state, **kw)) == base
    # ... and it does read what it must: a changed earlier call code changes the tokens
    changed = copy.deepcopy(feed)
    changed['liveData']['plays']['allPlays'][6]['playEvents'][5]['details']['call']['code'] = 'S'
    assert TOK.to_json(*TOK.build(changed, state, **kw)) != base


# =================================================================== 2. calibration
def test_calibration_apply_equals_calibrate_logits_apply_arms(tmp_path):
    assert SS.file_sha256(CONFIG['calibration']['code']['path']) == CONFIG['calibration']['code']['sha256']
    rng = np.random.default_rng(5)
    z = {'parent': 2*rng.normal(size=(500, 10)), 'narrow_whiff': 2*rng.normal(size=500),
         'ground_ball_per_pitch': 2*rng.normal(size=500)}
    for seed in (0, 1, 2):
        fitted = fabricate_cal_model(seed)
        reference = CL.apply_arms(fitted, z)
        mine = CAL.model_from_parameters(json.loads(json.dumps(CL.model_params(fitted))), LEAVES)  # JSON round trip
        for k, block in enumerate(mine['blocks']):
            for name in ('left', 'right', 'value', 'weight'):
                assert np.array_equal(block[name], fitted['blocks'][k][name])
        for arm in ('C0', 'C1', 'C2', 'C3'):
            got = CAL.apply_arm(mine, arm, z, CONFIG['calibration']['row_sum_tolerance'])
            assert got.shape == (500, 7) and np.abs(got-reference[arm]).max() <= 1e-12, (seed, arm)
            assert np.abs(got.sum(1)-1).max() < 1e-9 and (got >= 0).all()
    with pytest.raises(ValueError):
        CAL.apply_arm(mine, 'C9', z, 1e-9)


def test_calibrators_are_read_from_the_pinned_result_and_refuse_inconsistency(tmp_path):
    models, files = write_calibration(tmp_path)
    config = copy.deepcopy(CONFIG)
    config['calibration'] |= files
    temps = {s: models[s]['T'] for s in models}
    cal = CAL.load_calibrators(config, [0, 1, 2], temps)
    assert cal.selected == {0: 'C3', 1: 'C2', 2: 'C2'}  # read from the file
    z = {'parent': np.zeros((3, 10)), 'narrow_whiff': np.zeros(3), 'ground_ball_per_pitch': np.zeros(3)}
    assert np.allclose(cal.selected_arm(0, z), CL.apply_arms(models[0], z)['C3'], atol=1e-12, rtol=0)
    assert np.allclose(cal.selected_arm(1, z), CL.apply_arms(models[1], z)['C2'], atol=1e-12, rtol=0)
    with pytest.raises(SS.PinMismatch):  # temperatures.json and the calibrator's C0 must be the same numbers
        CAL.load_calibrators(config, [0, 1, 2], {s: temps[s] | {'parent': 9.0} for s in temps})
    bad = copy.deepcopy(config)
    bad['calibration']['result']['sha256'] = '0'*64
    with pytest.raises(SS.PinMismatch):
        CAL.load_calibrators(bad, [0, 1, 2])
    for change in ({'success': False}, {'result_sha256': '0'*64}, {'cal_selected': {'0': 'C0', '1': 'C2', '2': 'C2'}}):
        terminal = json.loads(Path(files['terminal']['path']).read_text()) | change
        other = tmp_path/'terminal-other.json'
        other.write_text(json.dumps(terminal))
        bad = copy.deepcopy(config)
        bad['calibration']['terminal'] = {'path': str(other), 'sha256': SS.file_sha256(other)}
        with pytest.raises(SS.PinMismatch):
            CAL.load_calibrators(bad, [0, 1, 2])


# =================================================================== 3. model
def test_model_refuses_any_hash_mismatch_before_torch_load(tmp_path, monkeypatch):
    import torch
    config = make_config(tmp_path/'cfg')
    calls = []
    real = torch.load
    monkeypatch.setattr(torch, 'load', lambda *a, **k: calls.append(1) or real(*a, **k))
    for path in (('model', 'state', '1'), ('model', 'temperatures', '2'), ('model', 'architecture'), ('model', 'train_seq'),
                 ('calibration', 'result'), ('calibration', 'terminal'), ('encoder_seq',)):
        bad = copy.deepcopy(config)
        node = bad
        for key in path:
            node = node[key]
        node['sha256'] = 'f'*64
        with pytest.raises(SS.PinMismatch):
            RUNNER.load_seq(bad)
        assert calls == [], path  # nothing was deserialized behind a refused pin
    # changed bytes under an unchanged pin
    changed = make_config(tmp_path/'cfg-changed')
    target = Path(changed['model']['state']['0']['path'])
    target.write_bytes(target.read_bytes()+b'\0')
    with pytest.raises(SS.PinMismatch):
        MOD.load_ensemble(changed)
    assert calls == []
    for key in ('expected_parameters', 'expected_state_tensors'):
        bad = copy.deepcopy(config)
        bad['model'][key] += 1
        with pytest.raises(SS.PinMismatch):
            MOD.load_ensemble(bad)
    assert len(calls) == 2  # each reached the first torch.load and stopped at the count check
    seq = RUNNER.load_seq(make_config(tmp_path/'cfg2'))
    assert len(calls) == 5 and len(seq.ensemble.nets) == 3


def test_model_counts_threads_determinism_and_padding_independence(world):
    import torch
    e = world.seq.ensemble
    assert torch.get_num_threads() == 1 == CONFIG['threads']
    for net in e.nets:
        state = net.state_dict()
        assert (len(state), sum(v.numel() for v in state.values())) == (45, 158860)
        assert not net.training
    assert (CONFIG['model']['expected_state_tensors'], CONFIG['model']['expected_parameters']) == (45, 158860)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(17, 199)).astype(np.float32)
    tok = {k: np.zeros(L, np.int64) for k in TOK.FIELDS} | {'valid': np.zeros(L, bool)}
    tok['pos'][:] = np.arange(1, L+1)
    tok['valid'][:5] = True
    tok['type'][:5], tok['out'][:5], tok['balls'][:5], tok['strikes'][:5], tok['same'][:5] = [7, 15, 1, 14, 3], [0, 1, 2, 3, 9], [0, 1, 2, 3, 0], [0, 1, 2, 0, 1], [1, 0, 1, 0, 1]
    first = world.seq.adapter.forecast(x, tok)
    again = world.seq.adapter.forecast(x, tok)
    assert first['logits'] == again['logits'] and np.array_equal(first['sel_ensemble'], again['sel_ensemble'])
    assert set(first['logits']) == {'0', '1', '2'}
    for z in first['logits'].values():
        assert np.array(z['parent']).shape == (17, 10) and np.array(z['narrow_whiff']).shape == (17,) \
            and np.array(z['ground_ball_per_pitch']).shape == (17,)
    for q in (first['c0_ensemble'], first['sel_ensemble']):
        assert q.shape == (17, 7) and np.abs(q.sum(1)-1).max() < 1e-9 and (q >= 0).all()
    # ensemble = mean over seeds of the per-seed rows; SEL uses each seed's own selected arm
    assert np.allclose(first['c0_ensemble'], np.mean([first['c0_by_seed'][s] for s in (0, 1, 2)], axis=0), atol=1e-15, rtol=0)
    assert np.allclose(first['sel_ensemble'], np.mean([first['sel_by_seed'][s] for s in (0, 1, 2)], axis=0), atol=1e-15, rtol=0)
    assert first['arms'] == {'0': 'C3', '1': 'C2', '2': 'C2'}
    for s in (0, 1, 2):
        z = {h: np.array(first['logits'][str(s)][h]) for h in CAL.HEADS}
        assert np.array_equal(first['sel_by_seed'][s], CAL.apply_arm(world.seq.calibrators.models[s], first['arms'][str(s)], z, 1e-9))
    assert not np.allclose(first['c0_ensemble'], first['sel_ensemble'])
    # logits do not depend on what the padded slots hold
    junk = {k: v.copy() for k, v in tok.items()}
    junk['type'][5:], junk['out'][5:], junk['balls'][5:], junk['strikes'][5:], junk['same'][5:] = 18, 9, 3, 2, 1
    assert world.seq.adapter.forecast(x, junk)['logits'] == first['logits']
    # ... and do depend on the valid ones, and on x
    moved = {k: v.copy() for k, v in tok.items()}
    moved['type'][0] = 8
    assert world.seq.adapter.forecast(x, moved)['logits'] != first['logits']
    # an empty context is finite (the no-context embedding)
    empty = {k: np.zeros(L, np.int64) for k in TOK.FIELDS} | {'valid': np.zeros(L, bool)}
    empty['pos'][:] = np.arange(1, L+1)
    assert np.isfinite(world.seq.adapter.forecast(x, empty)['sel_ensemble']).all()
    # a token outside its vocabulary, a wrong shape and a non-finite feature are refused
    for bad_tok, bad_x in ((tok | {'type': np.where(tok['valid'], 19, 0)}, x), (tok | {'out': tok['out'][:5]}, x),
                           (tok, np.where(np.arange(199) == 3, np.nan, x).astype(np.float32)), (tok, x[:5])):
        with pytest.raises(ValueError):
            world.seq.adapter.forecast(bad_x, bad_tok)


def test_encode_for_seq_is_the_pinned_apply_encoder_with_the_scale_002_statistics():
    enc, sha = AD.load_seq_encoder(CONFIG)
    assert sha == CONFIG['encoder_seq']['sha256'] and len(enc['continuous']) == 36
    f1, _ = F.load_encoder(STEP5['encoder']['path'], STEP5['encoder']['sha256'])
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(17, 91))*3
    raw[rng.random(raw.shape) < .1] = np.nan
    raw[0, :5] = [np.inf, -np.inf, 1e9, -1e9, 0]
    x = AD.encode_for_seq(raw, enc)
    assert x.dtype == np.float32 and x.shape == (17, 199) and np.isfinite(x).all()
    assert np.array_equal(x, F.apply_encoder(raw, enc))
    bare = {k: v for k, v in enc.items() if k != 'z_clip'}  # the bare dict of the file works the same
    assert np.array_equal(x, AD.encode_for_seq(raw, bare))
    assert not np.array_equal(x, F.apply_encoder(raw, f1))  # the F1 encoder is another encoding
    assert np.abs(x[:, :91]).max() <= 8
    bad = copy.deepcopy(CONFIG)
    bad['encoder_seq']['raw_columns'] = 90
    with pytest.raises(SS.PinMismatch):
        AD.load_seq_encoder(bad)


# =================================================================== 4. the run: denominators and leak tests
def test_run_completed_with_expected_rows(world):
    s = world.summary
    assert (world.rc, s['status'], s['integrity_stop']) == (0, 'COMPLETED', None)
    assert s['bodies'] == {'in_replay': 19, 'processed': 19, 'unreadable': 0, 'not_json_object': 0}
    assert s['decide_status_counts'] == {'appended': 10, 'duplicate': 1, 'revision_recorded': 1, 'transient_observed': 1,
                                         'matchup_changed': 1, 'poll_observed': 5}
    d = s['denominators']
    assert (d['pitch_attempts'], d['pitch_attempts_scored'], d['forecast_rows']) == (10, 8, 10)
    assert (d['seq_forecast_rows'], d['seq_scored']) == (10, 8)
    assert d['seq_abstain_by_reason'] == {'unsupported_player:no_prior_pitcher_history': 2}
    assert d['seq_first_decision_written_not_before_its_pitch_was_first_shown'] == 0
    assert d['forecast_written_not_before_its_pitch_was_first_shown'] == 0
    assert (d['physical_pitches_in_final_body'], d['physical_had_attempt'], d['physical_attempt_unlinked_matchup_changed'],
            d['physical_skipped_no_prefix']) == (45, 6, 3, 36)
    assert 'metrics' in s and 'type_share' not in s['metrics'] and 'outcome' not in s['metrics']  # placeholder block removed
    assert 'MUST NOT be read' in s['step5_placeholder_note'] and s['run']['fan_facing_output'] is False
    assert s['run']['torch_num_threads'] == 1 and s['run']['seq_model']['cal_selected'] == {'0': 'C3', '1': 'C2', '2': 'C2'}
    assert s['run']['source']['own'] == SS.own_source_sha256() and s['run']['source']['step5_shadow'] == CONFIG['step5_shadow']['sha256']
    assert s['claims'] == CONFIG['claims'] and 'already exposed' in s['claims']
    assert sorted(p.name for p in (world.tmp/'main').iterdir()) == ['SUMMARY.json', RUNNER.LEDGER_NAME, RUNNER.LEDGER_NAME+'.writer.lock']
    f = world.forecasts()
    assert sorted(k for k, r in f.items() if r['decision'] == 'scored') == sorted(
        f'{GAME}:{k}' for k in ('8:after0', '8:after1', '8:after2', '9:after2', '10:after1', '10:after2', '11:after0', '11:after1'))
    for r in f.values():
        cells = json.loads(r['step5_cells_status_json'])
        assert len(cells) == 17 and all('seven_leaf' not in c for c in cells)
        if r['decision'] == 'scored':
            assert np.array(json.loads(r['sel_ensemble_json'])).shape == (17, 7) and r['compute_ms'] > 0
            assert json.loads(r['cal_arms_json']) == {'0': 'C3', '1': 'C2', '2': 'C2'}
        else:
            assert r['sel_ensemble_json'] is None and r['logits_json'] is None and r['tokens_json'] is None
    # the same attempts, decisions and vector hashes as the Step 5 fixture ledger
    step5 = {r[0]: r[1:] for r in rows(world.step5, 'SELECT attempt_key, decision, vectors_sha256, context_sha256, '
                                                    'written_at_replay_index FROM shadow_forecasts')}
    assert {k: (r['decision'], r['vectors_sha256'], r['context_sha256'], r['written_at_replay_index']) for k, r in f.items()} == step5


def test_context_tokens_of_the_run_and_the_late_correction(world):
    f, index = world.forecasts(), {name: i for i, name in enumerate(world.names)}
    counts = {k.split(':', 1)[1]: json.loads(r['tokens_json'])['counts'] for k, r in f.items() if r['decision'] == 'scored'}
    assert counts['8:after0'] == {'n_context': 32, 'eligible_earlier_pitches': 33, 'skipped_no_type': 1,
                                  'skipped_unmapped_outcome': 1, 'skipped_count_range': 0, 'truncated': 1}
    assert (counts['8:after1']['eligible_earlier_pitches'], counts['8:after2']['eligible_earlier_pitches']) == (34, 35)
    assert counts['9:after2']['n_context'] == 2 and counts['10:after1']['n_context'] == 4 and counts['10:after2']['n_context'] == 5
    assert counts['11:after0']['eligible_earlier_pitches'] == 36
    # (PA 8, pitch 1) was FF when (PA 8, pitch 2/3) were forecast and is SI when PA 11 is forecast: each forecast
    # holds what its own body said
    ff, si = F.TYPES.index('FF'), F.TYPES.index('SI')
    assert json.loads(f[f'{GAME}:8:after1']['tokens_json'])['type'][0] == ff
    assert json.loads(f[f'{GAME}:8:after2']['tokens_json'])['type'][:2] == [F.TYPES.index('SL'), ff]
    assert json.loads(f[f'{GAME}:11:after0']['tokens_json'])['type'][:3] == [ff, F.TYPES.index('SL'), si]
    assert f[f'{GAME}:11:after0']['written_at_replay_index'] == index['pa11_open'] > index['pa9_done_and_correction']
    # the pitcher who came in during PA 9: the play's two earlier pitches are attributed to him (documented limit)
    assert f[f'{GAME}:9:after2']['pitcher'] == PC
    c = world.summary['context']
    assert (c['rows'], c['rows_with_zero_context'], c['rows_truncated']) == (8, 0, 5)
    assert c['n_context'] == {'n': 8, 'min': 2, 'p50': 32, 'max': 32, 'mean': (2+4+5+32*5)/8, 'method': 'nearest rank'}
    assert c['skipped_total_by_reason'] == {'skipped_unmapped_outcome': 5, 'skipped_count_range': 0, 'skipped_no_type': 5}


def test_prefix_replay_gives_byte_identical_rows(world):
    """4a. Replaying only the first k bodies gives the same row_sha256 for every row written at an index < k."""
    full, compared = world.forecasts(), 0
    for k in (2, 3, 7, 9, 12, 14, 17, 18):
        rc, _ = world.run(world.replay, f'prefix-{k}', max_bodies=k)
        assert rc == 0
        part = world.forecasts(world.tmp/f'prefix-{k}'/RUNNER.LEDGER_NAME)
        for key, row in part.items():
            if row['written_at_replay_index'] < k:
                assert row['row_sha256'] == full[key]['row_sha256'], (k, key)
                compared += 1
        assert sum(r['written_at_replay_index'] < k for r in full.values()) == sum(r['written_at_replay_index'] < k for r in part.values())
    assert compared >= 40


def mutate_everything(feed):
    """Change every provider value of a body (types, calls, results, physics, ids of non-pitcher roles, counts)."""
    feed = copy.deepcopy(feed)
    feed['liveData']['linescore'] = {'currentInning': 1, 'isTopInning': True, 'offense': {'second': {'id': 1}, 'third': {'id': 2}}}
    plays = feed['liveData']['plays']
    for one in plays['allPlays']:
        one['result'] |= {'eventType': 'home_run', 'description': 'mutated', 'homeScore': 5, 'awayScore': 3}
        one['matchup']['batter']['id'] = 203
        one['matchup']['batSide']['code'] = 'L'
        for e in one['playEvents']:
            if e.get('isPitch'):
                e['details'] |= {'type': {'code': 'KN'}, 'call': {'code': 'S'}, 'description': 'mutated'}
                e['pitchData'] = {'startSpeed': 55.5, 'coordinates': {'pX': 9.9, 'pZ': 9.9}}
                e['hitData'] = {'trajectory': 'popup', 'launchSpeed': 1.0}
    if plays.get('currentPlay') is not None:
        plays['currentPlay'] = plays['allPlays'][-1]
    return feed


def test_mutating_future_bodies_changes_no_earlier_row(world):
    """4b. Everything after the deciding body is rewritten (the pitch being predicted included): every row written
    before the first rewritten body keeps its row_sha256."""
    full, compared = world.forecasts(), 0
    for k in (2, 6, 7, 12, 14, 17):
        bodies = [(n, b if i < k else mutate_everything(b)) for i, (n, b) in enumerate(world.bodies)]
        replay = write_replay(world.tmp/f'replay-future-{k}', bodies)
        world.run(replay, f'future-{k}', step5=world.ledger_only(f'step5-copy-future-{k}'))
        part = world.forecasts(world.tmp/f'future-{k}'/RUNNER.LEDGER_NAME)
        early = {key: row for key, row in full.items() if row['written_at_replay_index'] < k}
        assert early
        for key, row in early.items():
            assert part[key]['row_sha256'] == row['row_sha256'], (k, key)
            compared += 1
        later = [key for key, row in part.items() if row['written_at_replay_index'] >= k and row['decision'] == 'scored'
                 and key in full and row['row_sha256'] != full[key]['row_sha256']]
        assert later or k >= 17  # the rewrite is visible in what is decided afterwards (the check is sensitive)
    assert compared >= 20


def test_fields_the_forecast_must_not_read_inside_the_deciding_body(world):
    """4c. pitchData, hitData.trajectory, result.description, times and unread linescore fields of EVERY body are
    perturbed: tokens, vectors and both ensembles stay the same (the body hash differs)."""
    def perturb(feed):
        feed = copy.deepcopy(feed)
        feed['liveData']['linescore'] |= {'teams': {'home': {'runs': 9}}, 'balls': 3, 'strikes': 2, 'note': 'perturbed'}
        feed['gameData']['datetime']['dateTime'] = 'perturbed'
        plays = feed['liveData']['plays']
        for one in plays['allPlays']:
            one['result'] |= {'description': 'perturbed', 'event': 'Perturbed', 'rbi': 3}
            one['about']['endTime'] = one['about']['startTime'] = 'perturbed'
            one['playEndTime'] = 'perturbed'
            for e in one['playEvents']:
                if e.get('isPitch'):
                    e['pitchData'] = {'startSpeed': 55.5, 'coordinates': {'pX': 9.9, 'pZ': 9.9}, 'zone': 14}
                    e['hitData'] = {'trajectory': 'perturbed', 'launchSpeed': 1.0, 'totalDistance': 500}
                    e['details']['description'] = 'perturbed'
                    e['startTime'] = e['endTime'] = 'perturbed'
        if plays.get('currentPlay') is not None:
            plays['currentPlay'] = plays['allPlays'][-1]
        return feed
    replay = write_replay(world.tmp/'replay-perturbed', [(n, perturb(b)) for n, b in world.bodies])
    rc, summary = world.run(replay, 'perturbed', step5=world.ledger_only('step5-copy-perturbed'))
    assert rc == 0 and summary['gate_V']['passed'] and summary['gate_V']['keys_common'] == 8
    mine, theirs = world.forecasts(), world.forecasts(world.tmp/'perturbed'/RUNNER.LEDGER_NAME)
    assert set(mine) == set(theirs)
    for key in mine:
        assert mine[key]['feed_sha256'] != theirs[key]['feed_sha256']
        for column in ('tokens_sha256', 'tokens_json', 'vectors_sha256', 'x_seq_sha256', 'context_sha256', 'logits_json',
                       'sel_ensemble_json', 'c0_ensemble_json', 'decision', 'reason'):
            assert mine[key][column] == theirs[key][column], (key, column)
    # the realized side DID change (trajectories are now unplaced), so the metrics differ while the forecasts do not
    assert summary['metrics']['population_B']['n'] < world.summary['metrics']['population_B']['n']


def test_every_first_decision_row_precedes_the_first_body_showing_its_pitch(world):
    """4d."""
    first = {(a, n): i for a, n, i in rows(world.db, 'SELECT at_bat_number, pitch_number, MIN(observed_at_replay_index) '
                                                     'FROM shadow_realized GROUP BY 1, 2')}
    checked = 0
    for ab, number, at, file in rows(world.db, 'SELECT at_bat_number, candidate_pitch_number, written_at_replay_index, '
                                               'replay_file FROM seq_forecasts'):
        assert file is not None  # in this fixture every attempt had a first decision of its own
        if (ab, number) in first:
            assert at < first[(ab, number)], (ab, number)
            checked += 1
    assert checked == 9  # (11, 2) was never thrown
    # the realized side lives in its own table; a forecast row has no realized column
    columns = [r[1] for r in rows(world.db, 'PRAGMA table_info(seq_forecasts)')]
    assert not [c for c in columns if 'realized' in c or 'seven_leaf' in c]


def test_terminalized_attempts_get_an_abstain_row_without_a_forecast(world):
    """An attempt that never had a valid first decision (unknown receipt times) is closed by an observed event:
    one abstain row, no model output, and the order check does not count it."""
    folder = world.tmp/'replay-no-times'
    write_replay(folder, world.bodies)
    manifest = json.loads((folder/'MANIFEST.json').read_text())
    for entry in manifest['files']:
        entry['fetched'] = entry['first_fetched'] = None
    (folder/'MANIFEST.json').write_text(json.dumps(manifest))
    step5 = world.step5_run(folder, 'step5-no-times')
    rc, s = world.run(folder, 'no-times', step5=step5)
    assert rc == 0 and (s['denominators']['seq_scored'], s['denominators']['seq_forecast_rows']) == (0, 10)
    assert set(s['denominators']['seq_abstain_by_reason']) == {
        'timing_unknown:no_valid_prefix_before_close:receipt_or_decision_clock_mark_missing_or_nonfinite'}
    f = world.forecasts(world.tmp/'no-times'/RUNNER.LEDGER_NAME)
    assert all(r['replay_file'] is None and r['decision'] == 'abstain' for r in f.values())
    assert s['denominators']['seq_first_decision_written_not_before_its_pitch_was_first_shown'] == 0
    assert s['gate_V']['keys_common'] == 0 and s['metrics']['population_A']['n'] == 0
    assert s['metrics']['population_A']['models']['N1_SEL']['nll_mean'] is None


def test_ledger_is_append_only_and_a_second_run_refuses(world):
    """4e."""
    copy_db = world.tmp/'immutability-copy.sqlite3'
    shutil.copy(world.db, copy_db)
    db = sqlite3.connect(copy_db)
    for table, word in (('seq_forecasts', 'append-only seq shadow ledger'), ('seq_run', 'append-only seq shadow ledger'),
                        ('shadow_forecasts', 'append-only'), ('shadow_realized', 'append-only'), ('pitch_attempts', 'append-only')):
        for sql in (f'UPDATE {table} SET rowid = rowid', f'DELETE FROM {table}'):
            with pytest.raises(sqlite3.DatabaseError, match=word):
                db.execute(sql)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute('INSERT INTO seq_forecasts SELECT * FROM seq_forecasts LIMIT 1')
    with pytest.raises(sqlite3.IntegrityError):  # a scored row needs its probabilities and the reverse
        db.execute("INSERT INTO seq_forecasts(attempt_key, game_pk, at_bat_number, candidate_pitch_number, written_at_replay_index, "
                   "decision, step5_outcome, step5_cells_status_json, config_sha256, source_sha256_json, row_sha256) "
                   "VALUES ('x', 1, 1, 1, 1, 'scored', 'scored', '[]', 'c', '{}', 'r')")
    db.close()
    before = world.forecasts()
    with pytest.raises(FileExistsError):
        world.run(world.replay, 'main')
    assert world.forecasts() == before
    run = json.loads(rows(world.db, 'SELECT body FROM seq_run')[0][0])
    assert run['schema'] == 'seq-shadow-run-v1' and 'MUST NOT be read' in run['step5_heads']['note']
    # row_sha256 covers every column except the timings
    r = dict(zip([c[1] for c in rows(world.db, 'PRAGMA table_info(seq_forecasts)')], rows(world.db, 'SELECT * FROM seq_forecasts LIMIT 1')[0]))
    assert set(r)-set(LED.HASHED_COLUMNS) == set(LED.TIMING_COLUMNS) | {'row_sha256'}
    assert LED.sha({k: r[k] for k in LED.HASHED_COLUMNS}) == r['row_sha256']


def test_a_cap_reached_mid_replay_leaves_a_partial_ledger_and_no_metrics(world):
    calls = {'n': 0}

    def rss():
        calls['n'] += 1
        return world.config['caps']['rss_bytes']+1 if calls['n'] > 4 else 1
    rc, s = world.run(world.replay, 'capped', g=fake_guard(world.config, rss=rss))
    assert (rc, s['status'], s['bodies']['processed']) == (3, 'STOPPED_BY_CAP', 4)
    assert s['stop'].startswith('rss_cap:') and s['metrics'] is None and s['gate_V'] is None and s['reconciliation'] is None
    full, part = world.forecasts(), world.forecasts(world.tmp/'capped'/RUNNER.LEDGER_NAME)
    assert {k: v['row_sha256'] for k, v in part.items()} == {k: full[k]['row_sha256'] for k in (f'{GAME}:8:after0', f'{GAME}:8:after1')}


# =================================================================== 5. gate V
def test_gate_v_stops_before_metrics_on_one_altered_vector_hash(world):
    assert world.summary['gate_V'] == {
        'keys_this_run': 8, 'keys_step5': 8, 'keys_common': 8, 'keys_only_this_run': 0, 'keys_only_step5': 0,
        'vectors_sha256_mismatch': 0, 'decision_index_mismatch': 0, 'context_sha256_mismatch_report_only': 0,
        'first_mismatching_keys': [], 'passed': True, 'rule': world.summary['gate_V']['rule']}
    for name, sql, field in (('vector', "UPDATE shadow_forecasts SET vectors_sha256 = 'altered' WHERE attempt_key = ?", 'vectors_sha256_mismatch'),
                             ('index', 'UPDATE shadow_forecasts SET written_at_replay_index = 0 WHERE attempt_key = ?', 'decision_index_mismatch')):
        altered = world.ledger_only(f'step5-altered-{name}')
        db = sqlite3.connect(altered)
        db.execute('DROP TRIGGER shadow_forecasts_no_update')  # a fabricated ledger for this test only
        db.execute(sql, (f'{GAME}:10:after2',))
        db.commit()
        db.close()
        rc, s = world.run(world.replay, f'gate-{name}', step5=altered)
        assert (rc, s['status'], s['integrity_stop']) == (4, 'STOPPED_BY_INTEGRITY_GATE', 'gate_V_failed')
        assert s['gate_V'][field] == 1 and s['gate_V']['passed'] is False and s['metrics'] is None
        assert s['gate_V']['first_mismatching_keys'] == [f'{GAME}:10:after2']
        assert s['denominators']['seq_scored'] == 8  # the ledger itself is complete


def test_a_step5_comparator_of_another_replay_or_profile_is_refused(world):
    summary = json.loads((world.step5.parent/'SUMMARY.json').read_text())
    for change, word in (({'replay_manifest_sha256': '0'*64}, 'another_replay_manifest'), ({'game_pk': 1}, 'another_game'),
                         ({'profile': {'profile_sha256': '0'*64}}, 'another_profile')):
        other = world.ledger_only('step5-other-'+word)
        (other.parent/'SUMMARY.json').write_text(json.dumps(summary | {'run': summary['run'] | change}))
        with pytest.raises(G.Refused, match=word):
            world.run(world.replay, 'never-'+word, step5=other)
        assert not (world.tmp/('never-'+word)).exists()
    assert RUNNER.check_step5_run(summary | {'status': 'STOPPED_BY_CAP'}, game=GAME, manifest_sha=summary['run']['replay_manifest_sha256'],
                                  profile_sha=summary['run']['profile']['profile_sha256']) == ['step5_run_not_COMPLETED']


# =================================================================== 6. summary metrics and bootstrap
def hand_metrics(q, y, clip=1e-12, bins=10):
    n = len(y)
    nll = sum(-math.log(min(1., max(q[i][y[i]], clip))) for i in range(n))/n
    brier = sum(sum((q[i][k]-(k == y[i]))**2 for k in range(7)) for i in range(n))/n
    top = [max(range(7), key=lambda k: (q[i][k], -k)) for i in range(n)]
    acc = sum(top[i] == y[i] for i in range(n))/n
    gaps = [0.]*bins
    for i in range(n):
        conf = q[i][top[i]]
        gaps[min(int(conf*bins), bins-1)] += conf-(top[i] == y[i])
    return {'nll_mean': nll, 'brier_mean': brier, 'top1_accuracy': acc, 'ece_top_label_10bin': sum(abs(g) for g in gaps)/n}


def hand_leaf_ece(q, y, leaf, bins=10):
    gaps = [0.]*bins
    for i in range(len(y)):
        gaps[min(int(q[i][leaf]*bins), bins-1)] += q[i][leaf]-(y[i] == leaf)
    return sum(abs(g) for g in gaps)/len(y)


def test_metric_block_equals_a_hand_computation_and_calibrate_logits():
    rng = np.random.default_rng(11)
    q = rng.dirichlet(np.ones(7), size=9)
    q[0] = [1, 0, 0, 0, 0, 0, 0]  # a certain row (bin 9 edge) and a zero-probability realized leaf (clip inside the log)
    q[1] = [0, .5, .5, 0, 0, 0, 0]  # a tie: the first maximum is the top label
    y = np.array([1, 2, 0, 3, 5, 6, 4, 3, 5])
    m = CONFIG['metrics']
    block = SUM.metric_block(q, y, leaves=LEAVES, nll_clip=m['nll_clip'], bins=m['bins'], leaf_ece_leaves=m['leaf_ece_leaves'],
                             calibration=RUN._calibration)
    hand = hand_metrics(q.tolist(), y.tolist())
    for k, v in hand.items():
        assert block[k] == pytest.approx(v, abs=1e-12), k
    assert block['n'] == 9 and block['nll_mean'] > -math.log(1e-12)/9
    assert block['ece_top_label_10bin'] == pytest.approx(CL.ece(q, y), abs=1e-15)  # the calibrate_logits definition
    for name in ('narrow_whiff', 'ground_ball_in_play'):
        assert block['leaf_ece_10bin'][name] == pytest.approx(hand_leaf_ece(q.tolist(), y.tolist(), LEAVES.index(name)), abs=1e-12)
    table = block['calibration_one_vs_rest_pooled']
    assert len(table) == 10 and sum(c['n'] for c in table) == 9*7
    assert table == RUN._calibration([(float(q[r, k]), bool(y[r] == k)) for r in range(9) for k in range(7)], 10)


def test_paired_bootstrap_identical_arms_determinism_and_both_units():
    rng = np.random.default_rng(12)
    n = 60
    a, b = rng.dirichlet(np.ones(7), size=n), rng.dirichlet(np.ones(7)*3, size=n)
    y, at_bats = rng.integers(0, 7, n), np.repeat(np.arange(15), 4)
    m = CONFIG['metrics']
    boot = CONFIG['bootstrap'] | {'draws': 500, 'chunk': 128}
    same = SUM.paired_difference(a, a, y, at_bats, metrics_config=m, bootstrap_config=boot)
    for unit in ('pitch', 'plate_appearance'):
        assert same['bootstrap'][unit]['ci'] == {k: [0.0, 0.0] for k in SUM.METRICS}
    assert same['difference'] == {k: 0.0 for k in SUM.METRICS}
    one = SUM.paired_difference(a, b, y, at_bats, metrics_config=m, bootstrap_config=boot)
    two = SUM.paired_difference(a, b, y, at_bats, metrics_config=m, bootstrap_config=boot)
    assert one == two  # deterministic for the registered seed
    other = SUM.paired_difference(a, b, y, at_bats, metrics_config=m, bootstrap_config=boot | {'seed': 1})
    assert other['bootstrap']['pitch']['ci'] != one['bootstrap']['pitch']['ci']
    ha, hb = hand_metrics(a.tolist(), y.tolist()), hand_metrics(b.tolist(), y.tolist())
    assert one['difference']['nll_mean'] == pytest.approx(ha['nll_mean']-hb['nll_mean'], abs=1e-12)
    assert one['difference']['ece_top_label'] == pytest.approx(ha['ece_top_label_10bin']-hb['ece_top_label_10bin'], abs=1e-12)
    assert (one['bootstrap']['pitch']['units'], one['bootstrap']['plate_appearance']['units']) == (60, 15)
    assert one['bootstrap']['pitch']['ci'] != one['bootstrap']['plate_appearance']['ci']
    for unit in ('pitch', 'plate_appearance'):
        for k in ('nll_mean', 'brier_mean'):
            low, high = one['bootstrap'][unit]['ci'][k]
            assert low < one['difference'][k] < high
    # an independent loop reproduces the pitch-level interval draw for draw (same generator, same picks)
    kw = {'nll_clip': m['nll_clip'], 'bins': m['bins']}
    sa, sb = SUM.pitch_statistics(a, y, **kw), SUM.pitch_statistics(b, y, **kw)
    gen, draws = np.random.default_rng(boot['seed']), []
    for start in range(0, 500, 128):
        for pick in gen.integers(0, n, (min(128, 500-start), n)):
            draws.append([(sa['nll'][pick]-sb['nll'][pick]).mean(),
                          (np.abs(sa['gap'][pick].sum(0)).sum()-np.abs(sb['gap'][pick].sum(0)).sum())/n])
    draws = np.array(draws)
    assert one['bootstrap']['pitch']['ci']['nll_mean'] == pytest.approx([np.quantile(draws[:, 0], .025), np.quantile(draws[:, 0], .975)], abs=1e-12)
    assert one['bootstrap']['pitch']['ci']['ece_top_label'] == pytest.approx([np.quantile(draws[:, 1], .025), np.quantile(draws[:, 1], .975)], abs=1e-12)
    # plate-appearance clusters of unequal size: ratio of sums, pitch-weighted
    clusters = np.array([0]*30+[1]*10+[2]*20)
    pa = SUM.paired_bootstrap(sa, sb, clusters, draws=200, seed=3, chunk=64, quantiles=[.025, .975])
    gen, ref = np.random.default_rng(3), []
    for start in range(0, 200, 64):
        for pick in gen.integers(0, 3, (min(64, 200-start), 3)):
            idx = np.concatenate([np.flatnonzero(clusters == c) for c in pick])
            ref.append((sa['nll'][idx]-sb['nll'][idx]).sum()/len(idx))
    assert pa['ci']['nll_mean'] == pytest.approx([np.quantile(ref, .025), np.quantile(ref, .975)], abs=1e-12)
    empty = SUM.paired_difference(a[:0], b[:0], y[:0], at_bats[:0], metrics_config=m, bootstrap_config=boot)
    assert empty['n'] == 0 and empty['bootstrap'] is None


def test_populations_exclusions_in_order_and_run_metrics_by_hand(world):
    m = world.summary['metrics']
    assert (m['physical_pitches'], m['physical_had_attempt']) == (45, 6)
    assert (m['population_B']['n'], m['population_B']['excluded_in_order']) == (6, {})
    assert (m['population_A']['n'], m['population_A']['excluded_in_order']) == (
        5, {'step5_realized_type_cell_not_scored:unsupported_action:below_count_threshold': 1})  # KC by the five-type pitcher
    assert set(m['population_A']['models']) == {'N1_SEL', 'N1_C0', 'STEP5_2024'} and set(m['population_B']['models']) == {'N1_SEL', 'N1_C0'}
    assert set(m['paired_differences_population_A']) == {'N1_SEL-STEP5_2024', 'N1_C0-STEP5_2024', 'N1_SEL-N1_C0'}
    # by hand, straight from the two ledgers
    f = world.forecasts()
    step5 = dict(rows(world.step5, 'SELECT attempt_key, action_probabilities FROM shadow_forecasts'))
    linked = [(8, 1, 'SI', 'ball'), (8, 2, 'SL', 'other_strike'), (8, 3, 'FF', 'ground_ball_in_play'), (10, 2, 'KC', 'foul'),
              (10, 3, 'SL', 'hit_by_pitch'), (11, 1, 'FF', 'ball')]
    sel, c0, old, y_b, y_a = [], [], [], [], []
    for ab, k, kind, leaf in linked:
        key, at = f'{GAME}:{ab}:after{k-1}', F.ACTIONS.index(kind)
        sel.append(json.loads(f[key]['sel_ensemble_json'])[at])
        c0.append(json.loads(f[key]['c0_ensemble_json'])[at])
        y_b.append(LEAVES.index(leaf))
        cell = json.loads(step5[key])[at]
        if cell['status'] == 'scored':
            old.append((len(sel)-1, cell['seven_leaf']))
            y_a.append(LEAVES.index(leaf))
    for name, q in (('N1_SEL', sel), ('N1_C0', c0)):
        for k, v in hand_metrics(q, y_b).items():
            assert m['population_B']['models'][name][k] == pytest.approx(v, abs=1e-12), (name, k)
    keep = [i for i, _ in old]
    for name, q in (('N1_SEL', [sel[i] for i in keep]), ('N1_C0', [c0[i] for i in keep]), ('STEP5_2024', [p for _, p in old])):
        for k, v in hand_metrics(q, y_a).items():
            assert m['population_A']['models'][name][k] == pytest.approx(v, abs=1e-12), (name, k)
    d = m['paired_differences_population_A']['N1_SEL-STEP5_2024']
    assert d['n'] == 5 and d['difference']['nll_mean'] == pytest.approx(
        hand_metrics([sel[i] for i in keep], y_a)['nll_mean']-hand_metrics([p for _, p in old], y_a)['nll_mean'], abs=1e-12)
    assert (d['bootstrap']['pitch']['units'], d['bootstrap']['plate_appearance']['units']) == (5, 3)
    assert (d['bootstrap']['pitch']['draws'], d['bootstrap']['pitch']['seed']) == (400, CONFIG['bootstrap']['seed'])
    assert (CONFIG['bootstrap']['draws'], CONFIG['bootstrap']['seed']) == (10000, 20261212)
    assert world.summary['metrics']['population_A']['plate_appearances'] == 3
    # exclusion order on a fabricated join
    seq = {'a': {'decision': 'abstain', 'reason': 'r'}, **{k: {'decision': 'scored', 'sel_ensemble_json': json.dumps([[1/7]*7]*17),
                                                              'c0_ensemble_json': json.dumps([[1/7]*7]*17)} for k in 'bcdefg'}}
    cells = [{'action': a, 'status': 'scored', 'reason': None, 'seven_leaf': [1/7]*7} for a in F.ACTIONS]
    weak = [c | ({'status': 'abstain', 'reason': 'unsupported_action:x', 'seven_leaf': None} if c['action'] == 'FF' else {}) for c in cells]
    old = {'e': {'action_probabilities': json.dumps(weak)}, 'g': {'action_probabilities': json.dumps(cells)}}
    physical = [(1, 1, 'a', 'had_attempt'), (1, 2, 'b', 'had_attempt'), (1, 3, 'c', 'had_attempt'), (1, 4, 'd', 'had_attempt'),
                (2, 1, 'e', 'had_attempt'), (2, 2, 'f', 'had_attempt'), (2, 3, 'g', 'had_attempt'), (3, 1, None, 'skipped_no_prefix'),
                (3, 2, 'h', 'attempt_unlinked_matchup_changed')]
    realized = {(1, 2): {'realized_pitch_type': None}, (1, 3): {'realized_pitch_type': 'UN'},
                (1, 4): {'realized_pitch_type': 'FF', 'realized_leaf': None, 'realized_leaf_reason': 'in_play_trajectory_absent'},
                (2, 1): {'realized_pitch_type': 'FF', 'realized_leaf': 'ball'}, (2, 2): {'realized_pitch_type': 'SL', 'realized_leaf': 'foul'},
                (2, 3): {'realized_pitch_type': 'SL', 'realized_leaf': 'foul'}}
    a_rows, b_rows, counts = SUM.populations(physical, realized, seq, old, list(F.ACTIONS), LEAVES)
    assert (counts['physical_pitches'], counts['physical_had_attempt'], len(a_rows), len(b_rows)) == (9, 7, 1, 3)
    shared = {'seq_forecast_not_scored:r': 1, 'realized_pitch_type_absent': 1, 'realized_pitch_type_outside_17_actions': 1,
              'realized_leaf_unplaced:in_play_trajectory_absent': 1}
    assert counts['B']['excluded'] == shared
    assert counts['A']['excluded'] == shared | {'step5_realized_type_cell_not_scored:unsupported_action:x': 1, 'step5_attempt_absent': 1}
    assert list(counts['A']['excluded']) == list(shared)+['step5_realized_type_cell_not_scored:unsupported_action:x', 'step5_attempt_absent']


def test_latency_and_guard_blocks(world):
    lat = world.summary['latency']
    assert (lat['decide_call_all_statuses']['n'], lat['first_decision_scored']['n'], lat['first_decision_abstain']['n'],
            lat['body_load_parse']['n']) == (19, 8, 2, 19)
    for name in ('compute_ms', 'tokens_ms', 'forward_ms', 'calibration_ms', 'features_ms'):
        block = lat['seq'][name]
        assert block['n'] == 8 and 0 < block['min'] <= block['p50'] <= block['p90'] <= block['p99'] <= block['max'], name
    assert lat['seq_model_load_ms'] > 0
    for r in world.forecasts().values():
        if r['decision'] == 'scored':
            t = dict(zip(('c', 't', 'f', 'k'), rows(world.db, 'SELECT compute_ms, tokens_ms, forward_ms, calibration_ms FROM '
                                                              'seq_forecasts WHERE attempt_key = ?', r['attempt_key'])[0]))
            assert t['c'] == pytest.approx(t['t']+t['f']+t['k'], abs=1e-9)
    guard = world.summary['guard']
    assert guard['signals_sent_to_other_processes'] == 0 and guard['heavy_lock'] == 'NOT_TOUCHED_flag_absent'
    assert guard['peak_rss_bytes'] > 0 and guard['elapsed_seconds'] >= 0 and guard['free_memory_bytes'] == 16384*550000


# =================================================================== 7. guard
def test_guard_free_memory_floor_and_unparseable_vm_stat():
    assert G.free_memory_bytes(fake_run()) == 16384*(100000+400000+50000)
    low = VM_OK.replace('400000', '40000')  # (100000+40000+50000)*16384 = 3.1e9 < 6 GiB
    with pytest.raises(G.Refused, match='free_memory_below_floor'):
        G.Guard(CONFIG, run=fake_run(vm=low, ps=PS_QUIET), arm_hard_stops=False).preflight()
    assert CONFIG['caps']['min_free_memory_bytes'] == 6*2**30 and CONFIG['caps']['rss_bytes'] == 2*2**30 and CONFIG['caps']['wall_seconds'] == 1800
    for text in ('', 'Mach 가상 메모리 통계: (페이지 크기 16384 바이트)\n여유 페이지: 100000.\n'):
        with pytest.raises(G.Refused):
            G.Guard(CONFIG, run=fake_run(vm=text), arm_hard_stops=False).preflight()


def test_guard_protected_process_identity_with_the_registered_four():
    registered = CONFIG['protected']['processes']
    assert [p['pid'] for p in registered] == [94628, 94642, 99202, 71753]
    assert all(G.LSTART.fullmatch(p['lstart']) and p['argv_contains'] for p in registered)
    assert CONFIG['protected']['command_words'] == ['capture_one_game.py', 'capture_supervisor.py', 'poll_recorder.py', 'shadow_tee.py']
    step5 = {p['pid']: p for p in STEP5['protected']['processes']}
    assert all(step5[p['pid']] == p for p in registered[:3])  # same identity strings as the Step 5 shadow config
    table = PS_QUIET+''.join(  # ps pads a one-digit day with a second space, as the C locale prints it
        f"{p['pid']:>5} {p['lstart'].replace('Oct 6', 'Oct  6')}     /x/Python -m uvicorn {p['argv_contains']} --log-config l.json\n"
        for p in registered)
    with pytest.raises(G.Refused, match='protected_process_alive'):
        G.Guard(CONFIG, run=fake_run(ps=table), arm_hard_stops=False).preflight()
    receipt = G.Guard(CONFIG, run=fake_run(ps=table), arm_hard_stops=False, allow_concurrent=True).preflight()
    assert [(p['pid'], p['why']) for p in receipt['protected_alive']] == [(p['pid'], 'registered_pid_start_and_argv') for p in registered]
    assert receipt['concurrent_with_protected_allowed'] is True and receipt['signals_sent_to_other_processes'] == 0
    # the same pid with another start time is a reused pid: reported, not protected
    reused = PS_QUIET+'71753 Sun Oct 11 12:00:00 2026     /usr/bin/vim notes.txt\n'
    receipt = G.Guard(CONFIG, run=fake_run(ps=reused), arm_hard_stops=False).preflight()
    assert receipt['protected_alive'] == [] and [p['pid'] for p in receipt['registered_pid_reused']] == [71753]
    for word in ('poll_recorder.py', 'shadow_tee.py'):
        with pytest.raises(G.Refused, match='protected_process_alive'):
            G.Guard(CONFIG, run=fake_run(ps=PS_QUIET+f' 4242 Sun Oct 11 09:00:00 2026     /some/python /x/{word} --game 1\n'),
                    arm_hard_stops=False).preflight()
    with pytest.raises(G.Refused, match='ps_row_not_in_C_locale_shape'):  # the ko_KR shape of lstart is never parsed
        G.Guard(CONFIG, run=fake_run(ps='    1 수  9 23 22:00:00 2026     /sbin/launchd\n'), arm_hard_stops=False).preflight()


def test_guard_subprocesses_run_in_the_C_locale(monkeypatch):
    seen = []

    class Done:
        stdout = VM_OK

    def fake(argv, **kw):
        seen.append((argv, kw))
        return Done() if argv[0].endswith('vm_stat') else type('D', (), {'stdout': PS_QUIET})()
    monkeypatch.setattr(G.subprocess, 'run', fake)
    monkeypatch.setenv('LANG', 'ko_KR.UTF-8')
    G.Guard(CONFIG, arm_hard_stops=False).preflight()
    assert [a[0] for a, _ in seen] == ['/usr/bin/vm_stat', '/bin/ps']
    for _, kw in seen:
        assert kw['env']['LC_ALL'] == 'C' and kw['env']['LANG'] == 'C'


def test_own_source_never_signals_and_writes_no_literal_process_call():
    for name in SS.OWN_SOURCE:
        if not name.endswith('.py'):
            continue
        text = (REPO/name).read_text()
        called = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call):
                called.add(node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', ''))
        assert not called & {'kill', 'killpg', 'send_signal', 'terminate', 'pthread_kill', 'raise_signal', 'Popen', 'system'}, name
        assert not [w for w in ('pkill', 'killall', '/bin/kill', 'subprocess', 'live-849831.sqlite3', '8766', '8771', '8772') if w in text], name


# =================================================================== 8. pins, grant and review binding, CLI
def test_registered_pins_match_the_files_on_disk_hash_only():
    """Hashes only (no file is parsed here): every sha256 of the registered configuration equals the bytes on disk."""
    pins = [(Path(CONFIG['step5_shadow']['dir'])/n, s) for n, s in CONFIG['step5_shadow']['sha256'].items()]
    pins += [(CONFIG['step5_ledger']['path'], CONFIG['step5_ledger']['sha256']),
             (CONFIG['step5_ledger']['summary_path'], CONFIG['step5_ledger']['summary_sha256'])]
    for group in ('state', 'temperatures'):
        pins += [(v['path'], v['sha256']) for v in CONFIG['model'][group].values()]
    pins += [(CONFIG['model'][k]['path'], CONFIG['model'][k]['sha256']) for k in ('architecture', 'train_seq')]
    pins += [(CONFIG['calibration'][k]['path'], CONFIG['calibration'][k]['sha256']) for k in ('result', 'terminal', 'code')]
    pins += [(CONFIG['encoder_seq']['path'], CONFIG['encoder_seq']['sha256']),
             (CONFIG['tokens']['outcome10_source']['path'], CONFIG['tokens']['outcome10_source']['sha256']),
             (Path(CONFIG['parity']['inputs_dir'])/'manifest.json', CONFIG['parity']['manifest_sha256'])]
    pins += [(v['path'], v['sha256']) for v in CONFIG['parity']['margins'].values()]
    for path, sha in pins:
        assert SS.file_sha256(path) == sha, path
    assert CONFIG['runtime_python'] == str(REPO/'.venv'/'bin'/'python')
    assert STEP5['profile_sha256'] == '42f0ce33010447bd07a2092eb018d3da5d88079668eed755914a54655b47e272'  # configuration revision 2 (game 849809)
    assert set(SS.OWN_SOURCE) == {str(p.relative_to(REPO)) for p in (REPO/'src'/'pitcheezy'/'seq_shadow').glob('*.py')} | {
        'scripts/seq_shadow_runner.py', 'scripts/seq_shadow_parity.py', 'configs/seq_shadow.json'}


def test_pinned_step5_shadow_import_refuses_changed_bytes(tmp_path):
    bad = copy.deepcopy(CONFIG)
    bad['step5_shadow']['sha256']['shadow_guard.py'] = '0'*64
    with pytest.raises(SS.PinMismatch, match='shadow_guard.py'):
        SS.load_step5_shadow(bad)
    elsewhere = tmp_path/'copy'
    shutil.copytree(CONFIG['step5_shadow']['dir'], elsewhere, ignore=lambda d, names: [n for n in names if not n.startswith('shadow_')])
    moved = copy.deepcopy(CONFIG)
    moved['step5_shadow']['dir'] = str(elsewhere)
    with pytest.raises(SS.PinMismatch, match='another origin'):  # identical bytes at another path are another origin
        SS.load_step5_shadow(moved)
    assert not list(Path(CONFIG['step5_shadow']['dir']).glob('__pycache__'))
    assert MODS.sha256 == CONFIG['step5_shadow']['sha256']


def good_grant(tmp_path, replay, out, **over):
    review = tmp_path/'REVIEW.json'
    review.write_text(json.dumps({'verdict': 'ACCEPT', 'reviewer': 'seq-shadow-001 independent reviewer', 'source_sha256': SS.own_source_sha256()}
                                 | over.pop('review', {})))
    return {'grant': CONFIG['grant_name'], 'attempt_id': 'seq-shadow-849831-001', 'game_pk': 849831, 'replay_dir': str(replay),
            'output_dir': str(out), 'profile_sha256': STEP5['profile_sha256'],
            'source_sha256': {'own': SS.own_source_sha256(),
                              'step5_shadow': {k: v for k, v in CONFIG['step5_shadow']['sha256'].items() if k.endswith('.py')}},
            'model_sha256': {s: v['sha256'] for s, v in CONFIG['model']['state'].items()},
            'calibrator_sha256': CONFIG['calibration']['result']['sha256'], 'step5_ledger_sha256': CONFIG['step5_ledger']['sha256'],
            'allow_concurrent_with_protected_capture': True, 'acquire_heavy_lock': False,
            'review_path': str(review), 'review_sha256': SS.file_sha256(review)} | over


def test_grant_and_review_binding(tmp_path):
    replay, out = Path('/x/replay'), Path('/x/out')
    kw = dict(replay=replay, out=out, game_pk=849831, config=CONFIG, step5_config=STEP5, step5_shadow_sha256=MODS.sha256,
              executable=CONFIG['runtime_python'])
    assert RUNNER.check_grant(good_grant(tmp_path, replay, out), **kw) == []
    assert CONFIG['grant_name'] == 'SEQ_SHADOW_REPLAY_ONE_GAME'
    assert RUNNER.check_grant(good_grant(tmp_path, replay, out), **(kw | {'executable': '/usr/bin/python3'})) == ['interpreter_is_not_the_registered_runtime']
    assert RUNNER.check_grant(None, **kw) == ['grant_not_an_object']
    grant = good_grant(tmp_path, replay, out)
    for key in ('grant', 'replay_dir', 'output_dir', 'game_pk', 'profile_sha256', 'source_sha256', 'model_sha256',
                'calibrator_sha256', 'step5_ledger_sha256'):
        assert RUNNER.check_grant({k: v for k, v in grant.items() if k != key}, **kw) == [key]  # missing
        wrong = grant | {key: 'other' if not isinstance(grant[key], dict) else {'own': {}}}
        assert RUNNER.check_grant(wrong, **kw) == [key]  # mismatched
    one_byte = copy.deepcopy(grant)
    one_byte['source_sha256']['own']['src/pitcheezy/seq_shadow/tokens.py'] = '0'*64
    assert RUNNER.check_grant(one_byte, **kw) == ['source_sha256']
    one_byte = copy.deepcopy(grant)
    one_byte['source_sha256']['step5_shadow']['shadow_runner.py'] = '0'*64
    assert RUNNER.check_grant(one_byte, **kw) == ['source_sha256']
    assert RUNNER.check_grant(grant | {'acquire_heavy_lock': 'no'}, **kw) == ['acquire_heavy_lock_must_be_true_or_false']
    assert RUNNER.check_grant({k: v for k, v in grant.items() if k != 'allow_concurrent_with_protected_capture'}, **kw) == [
        'allow_concurrent_with_protected_capture_must_be_true_or_false']
    assert RUNNER.check_grant(grant | {'attempt_id': ''}, **kw) == ['attempt_id']
    assert RUNNER.check_grant(grant, **(kw | {'game_pk': None})) == ['game_pk', 'a_real_run_names_its_game_with_--game']
    # the review
    assert RUNNER.check_grant(grant | {'review_sha256': '0'*64}, **kw) == ['review_sha256']
    assert RUNNER.check_grant({k: v for k, v in grant.items() if k != 'review_path'}, **kw) == ['review_path_or_review_sha256']
    assert RUNNER.check_grant(grant | {'review_path': str(tmp_path/'absent.json')}, **kw) == ['review_file_unreadable']
    for review, word in (({'verdict': 'PASS_WITH_NOTES'}, 'review_verdict_is_not_ACCEPT'), ({'verdict': None}, 'review_verdict_is_not_ACCEPT'),
                         ({'reviewer': CONFIG['author']}, 'reviewer_absent_or_the_author'), ({'reviewer': ''}, 'reviewer_absent_or_the_author'),
                         ({'reviewer': None}, 'reviewer_absent_or_the_author'), ({'source_sha256': {}}, 'review_names_other_source_bytes')):
        assert RUNNER.check_grant(good_grant(tmp_path, replay, out, review=review), **kw) == [word]


def cli(*argv):
    return subprocess.run([sys.executable, '-B', str(REPO/'scripts'/'seq_shadow_runner.py'), *argv], capture_output=True, text=True)


def test_cli_refusals_write_nothing(world, tmp_path):
    out = tmp_path/'cli-never'
    synthetic = ('--synthetic', '--profile-dir', str(world.tmp/'profile'), '--config', str(world.config_path),
                 '--step5-ledger', str(world.step5))
    # real mode without a grant; with a grant that is not bound; with a bound-looking grant and a review that is not ACCEPT
    no_grant = cli('--replay-dir', str(world.replay), '--out-dir', str(out), '--game', str(GAME))
    assert (no_grant.returncode, json.loads(no_grant.stdout)['refused']) == (
        2, 'a_real_run_needs_a_grant_and_uses_the_registered_profile_ledger_and_every_body')
    grant = tmp_path/'grant.json'
    grant.write_text(json.dumps({'grant': CONFIG['grant_name']}))
    bad = cli('--replay-dir', str(world.replay), '--out-dir', str(out), '--game', str(GAME), '--grant', str(grant))
    assert bad.returncode == 2 and 'grant_not_bound_to_this_run' in bad.stdout and 'review_path_or_review_sha256' in bad.stdout
    grant.write_text(json.dumps(good_grant(tmp_path, world.replay.resolve(), out.resolve(), game_pk=GAME, review={'verdict': 'REJECT'})))
    rejected = cli('--replay-dir', str(world.replay), '--out-dir', str(out), '--game', str(GAME), '--grant', str(grant))
    assert rejected.returncode == 2 and json.loads(rejected.stdout)['refused'] == 'grant_not_bound_to_this_run:review_verdict_is_not_ACCEPT'
    other_config = cli('--replay-dir', str(world.replay), '--out-dir', str(out), '--game', str(GAME), '--grant', str(grant),
                       '--config', str(world.config_path))
    assert other_config.returncode == 2 and 'a_real_run_uses_the_registered_config_file' in other_config.stdout
    # a real run under another interpreter
    grant.write_text(json.dumps(good_grant(tmp_path, world.replay.resolve(), out.resolve(), game_pk=GAME)))
    assert RUNNER.check_grant(json.loads(grant.read_text()), replay=world.replay.resolve(), out=out.resolve(), game_pk=GAME, config=CONFIG,
                              step5_config=STEP5, step5_shadow_sha256=MODS.sha256, executable='/usr/bin/python3') == [
        'interpreter_is_not_the_registered_runtime']
    if Path('/usr/bin/python3').exists():  # the same grant through the CLI under the system interpreter: rc 2, nothing written
        system = subprocess.run(['/usr/bin/python3', '-B', str(REPO/'scripts'/'seq_shadow_runner.py'), '--replay-dir', str(world.replay),
                                 '--out-dir', str(out), '--game', str(GAME), '--grant', str(grant)], capture_output=True, text=True)
        assert system.returncode == 2, system.stdout+system.stderr
        assert 'interpreter_is_not_the_registered_runtime' in system.stdout or 'import_failed_under_this_interpreter' in system.stdout
    # synthetic mode: forbidden path, existing output, a replay that is not a fixture, a grant, a changed model byte
    inside = tmp_path/'step4-capture-849831'/'x'
    inside.mkdir(parents=True)
    forbidden = cli('--replay-dir', str(inside), '--out-dir', str(out), *synthetic)
    assert forbidden.returncode == 2 and 'path_inside_a_protected_directory' in forbidden.stdout
    exists = cli('--replay-dir', str(world.replay), '--out-dir', str(world.tmp/'main'), *synthetic)
    assert exists.returncode == 2 and 'output_dir_exists' in exists.stdout
    unmarked = write_replay(tmp_path/'replay-unmarked', world.bodies, fixture=False)
    not_fixture = cli('--replay-dir', str(unmarked), '--out-dir', str(out), *synthetic)
    assert not_fixture.returncode == 2 and 'synthetic_mode_needs_a_replay_manifest_marked_synthetic_fixture' in not_fixture.stdout
    with_grant = cli('--replay-dir', str(world.replay), '--out-dir', str(out), '--grant', str(grant), *synthetic)
    assert with_grant.returncode == 2 and 'synthetic_mode_takes' in with_grant.stdout
    broken = copy.deepcopy(world.config)
    broken['model']['state']['2']['sha256'] = '0'*64
    (tmp_path/'broken.json').write_text(json.dumps(broken))
    pin = cli('--replay-dir', str(world.replay), '--out-dir', str(out), '--synthetic', '--profile-dir', str(world.tmp/'profile'),
              '--config', str(tmp_path/'broken.json'), '--step5-ledger', str(world.step5))
    assert pin.returncode == 2 and 'PinMismatch' in pin.stdout
    low = copy.deepcopy(world.config)
    low['caps']['min_free_memory_bytes'] = 1 << 60
    (tmp_path/'low.json').write_text(json.dumps(low))
    memory = cli('--replay-dir', str(world.replay), '--out-dir', str(out), '--synthetic', '--profile-dir', str(world.tmp/'profile'),
                 '--config', str(tmp_path/'low.json'), '--step5-ledger', str(world.step5))
    assert memory.returncode == 2 and 'free_memory_below_floor' in memory.stdout
    assert not out.exists() and not Path(str(out)+'.RSS-STOP.json').exists()
    # actual mode refuses a fixture replay even when everything else is in place (in process: no grant can be made
    # for the registered files by a test)
    with pytest.raises(G.Refused, match='actual_mode_needs_a_capture_to_replay_manifest_that_is_not_a_fixture'):
        RUNNER.run_seq_shadow(world.replay, out, config=world.config, config_sha256=world.config_sha, step5_config=STEP5, mods=MODS,
                              src=SRC, profile=world.profile, heads=world.heads, guard=fake_guard(world.config), mode='actual',
                              seq=world.seq, step5_ledger={'path': world.step5, 'sha256': SS.file_sha256(world.step5), 'summary': None})
    with pytest.raises(G.Refused, match='read_scope_is_not_ACTUAL_S2_ONE_PASS'):
        RUN.check_live_profile(world.profile, STEP5)  # a fixture profile can never be connected to a real replay
    assert not out.exists()


def test_cli_synthetic_end_to_end_equals_the_in_process_run(world, tmp_path):
    out = tmp_path/'cli-run'
    done = cli('--replay-dir', str(world.replay), '--out-dir', str(out), '--synthetic', '--profile-dir', str(world.tmp/'profile'),
               '--config', str(world.config_path), '--step5-ledger', str(world.step5))
    assert done.returncode == 0, done.stdout+done.stderr
    summary = json.loads((out/'SUMMARY.json').read_text())
    assert (summary['status'], summary['denominators']['seq_scored'], summary['gate_V']['passed']) == ('COMPLETED', 8, True)
    assert summary['guard']['signals_sent_to_other_processes'] == 0 and summary['guard']['hard_stops_armed'] is True
    assert summary['guard']['heavy_lock'] == 'NOT_TOUCHED_flag_absent' and summary['guard']['peak_rss_bytes'] < 2*2**30
    assert summary['run']['interpreter'] == sys.executable and summary['run']['torch_num_threads'] == 1
    # a separate process gives the same forecasts bit for bit (timings aside)
    assert {k: v['row_sha256'] for k, v in world.forecasts(out/RUNNER.LEDGER_NAME).items()} == {
        k: v['row_sha256'] for k, v in world.forecasts().items()}
    assert summary['metrics']['population_A']['models'] == world.summary['metrics']['population_A']['models']
    assert summary['metrics']['paired_differences_population_A'] == world.summary['metrics']['paired_differences_population_A']
    second = cli('--replay-dir', str(world.replay), '--out-dir', str(out), '--synthetic', '--profile-dir', str(world.tmp/'profile'),
                 '--config', str(world.config_path), '--step5-ledger', str(world.step5))
    assert second.returncode == 2 and 'output_dir_exists' in second.stdout


def test_parity_selftest_and_refusals(tmp_path):
    out = tmp_path/'parity'
    done = subprocess.run([sys.executable, '-B', str(REPO/'scripts'/'seq_shadow_parity.py'), '--synthetic', '--out', str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout+done.stderr
    result = json.loads((out/'PARITY.json').read_text())
    assert result['verdict'] == 'SELFTEST_PASS' and all(result['checks'].values()) and len(result['checks']) == 7
    again = subprocess.run([sys.executable, '-B', str(REPO/'scripts'/'seq_shadow_parity.py'), '--synthetic', '--out', str(out)],
                           capture_output=True, text=True)
    assert again.returncode == 2 and 'output_dir_exists' in again.stdout
    # comparison arithmetic
    a = {'parent': np.zeros((4, 10)), 'narrow_whiff': np.zeros(4), 'ground_ball_per_pitch': np.zeros(4)}
    b = {k: v.copy() for k, v in a.items()}
    b['parent'][1, 3] = 2e-4
    b['narrow_whiff'][2] = 5e-5
    got = PARITY.compare(a, b, 1e-4)
    assert (got['parent']['max_abs_difference'], got['parent']['fraction_rows_within_tolerance'], got['parent']['within_tolerance']) == (2e-4, .75, False)
    assert (got['narrow_whiff']['max_abs_difference'], got['narrow_whiff']['within_tolerance']) == (5e-5, True)
    assert got['ground_ball_per_pitch']['max_abs_difference'] == 0.0
    with pytest.raises(ValueError):
        PARITY.compare(a, b | {'parent': np.zeros((3, 10))}, 1e-4)
    assert CONFIG['parity']['role'] == 'calibration' and CONFIG['parity']['max_abs_tolerance'] == 1e-3  # registration revision 1 (parity-001 STOP under 1e-4) and CONFIG['parity']['batch'] == 8192
    # a real-mode call with a config that is not the registered file is refused before anything is read
    (tmp_path/'other.json').write_text(json.dumps(CONFIG))
    refused = PARITY.main(['--out', str(tmp_path/'never'), '--config', str(tmp_path/'other.json')])
    assert refused == 2 and not (tmp_path/'never').exists()


def test_registered_token_policy_is_unknown_token_as_in_the_training_corpus():
    # abc_corpus.project kept null pitch_type rows as UNKNOWN (token 0); the registered policy must match it
    assert CONFIG['tokens']['missing_type_policy'] == 'unknown_token'
    assert CONFIG['tokens']['missing_type_policy'] in CONFIG['tokens']['missing_type_policy_values']
