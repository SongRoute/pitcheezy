"""Step 5 shadow feature path == scale-002 training feature path, on SYNTHETIC data only.

Path A (serving shadow): ``step5/features.py`` -> ``ProfileStore`` + ``build_candidates`` (17 candidates x 91 raw).
Path B (training, EXP-WS-SEQ-001 N1 input): ``scale_runner.features(pool, rows, cutoff, M)`` with
``M = scale_runner.closure(scale_runner.load(SCALE-REGISTRATION.json))`` and ``M.safe.apply_encoder``.

Path B runs in a child interpreter (``python -B -I <this file> --path-b <out>``): ``closure`` installs an import
finder that denies torch/lightgbm/duckdb for the whole process, which must not leak into the pytest session.
No real pitch data is read: only source files (sha-pinned) and two encoder statistics JSON files.

Documented residual differences between the two paths (asserted below where reachable):
* pool ``pitch_type`` null: A skips the row for type tables but still counts the batter; B refuses the pool
  (``Missing history grouping key``). The training projection writes the literal 'UNKNOWN' instead, and with
  'UNKNOWN' both paths agree (A given None or 'UNKNOWN' gives the same raw as B given 'UNKNOWN').
* previous pitch: B looks up key (game_pk, at_bat_number, pitch_number-1) among the query rows; A takes the last
  received pitch of the open PA. They agree when PA pitch numbers are contiguous, which ``pre_pitch_state`` requires.
* float summation order: donor order inside a context cell follows pool row order in A and date order in B. With a
  date-sorted pool the order is the same; the shuffled-pool case measures what happens otherwise.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

JOBS = Path('/Users/song/Projects/pitcheezy-pm-jobs')
A_FEATURES = JOBS/'20261010-ws-service-006/step5-2024-runner-lifecycle-f1-author-001/step5/features.py'
B_DIR = JOBS/'20261011-ws-step2-scale-f2-recovery-author-001'
B_RUNNER, B_SAFE, B_REG = B_DIR/'scale_runner.py', B_DIR/'scale_safe.py', B_DIR/'SCALE-REGISTRATION.json'
ENC002 = B_DIR/'attempts/scale-002/encoder.json'
F1_DESIGN = JOBS/'20261008-confirmation-001/scale-preparation/cache100-001/early/F1-design.json'
PINS = {A_FEATURES: '2e9e71228e2946ac88231219927ce7f40cb9c261cb9c72733db1facf6d458d03',
        B_RUNNER: 'e25db72841066e28aa28b25bcd4ed07e7b7c57e199dbf9ab33f947f80ba0a436',
        B_SAFE: '1f2cd2e221bcff53c9f2b1b1ee7ceae0476f02bf5559a21e31a8b520f94b5234',
        ENC002: '3c32aaab1e1ac3c96349f13d5c23e91b63011b58af86b5207c5485382f8759a6',
        F1_DESIGN: '449ceeb0cd3403694e20e9ae8b3b41002a81e2cccad68b8272590e1b84ee90f9'}

TYPES = ('UNKNOWN', 'CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'PO', 'SC', 'SI', 'SL', 'ST',
         'SV', 'UN')
ACTIONS = tuple(t for t in TYPES if t not in ('UNKNOWN', 'UN'))
PHYSICS = ('release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z')
CUTOFF, OFFICIAL, MID = '2026-06-30', '2026-07-01', '2026-05-15'
HAND = {101: 'R', 102: 'R', 103: 'L', 104: 'R', 105: 'R'}       # 105 has no history
STAND = {201: 'L', 202: 'R', 203: 'R', 204: 'L', 299: 'R'}      # 299 has no history
REPERTOIRE = {101: ('FF', 'SL', 'CH', 'FS'), 102: ('FF', 'SL', 'CU'), 103: ('FF', 'SI', 'CH'), 104: ('FF', 'SL')}
BASE = {'FF': (94., 2300., -3., 8.), 'SL': (85., 2500., 2., 0.5), 'CH': (84.5, 1700., -6.5, 3.), 'FS': (86., 1400., -5., 1.5),
        'CU': (78.5, 2700., 4., -6.), 'SI': (92.5, 2100., -7.5, 4.), 'UNKNOWN': (88., 2000., 0., 0.)}
COUNTS = ((0, 0), (1, 1), (3, 2), (0, 2))
DESCRIPTIONS = ('ball', 'called_strike', 'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip',
                'hit_into_play', 'hit_into_play', 'missed_bunt', 'blocked_ball', 'foul_bunt')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Synthetic world, shared by both processes (deterministic; no file is read).
# ---------------------------------------------------------------------------------------------------------------
def synthetic_pool():
    """Date-sorted pitch-level pool, 2026-04-01..2026-06-30, physics in multiples of 0.5 (exact float sums)."""
    rng = np.random.default_rng(20261012)
    start = date(2026, 4, 1)
    days = sorted({start+timedelta(days=int(d)) for d in range(0, 91, 3)} | {date(2026, 5, 15), date(2026, 6, 30)})
    rows = []

    def add(day, pitcher, kind, batter, count, phys, description=None, outcome=None):
        description = DESCRIPTIONS[rng.integers(len(DESCRIPTIONS))] if description is None else description
        bb = (('ground_ball', 'fly_ball', 'line_drive', 'popup')[rng.integers(4)]
              if description == 'hit_into_play' else None)
        rows.append({'game_date': day, 'pitcher': pitcher, 'pitch_type': kind, **dict(zip(PHYSICS, phys)),
                     'p_throws': HAND[pitcher], 'stand': STAND[batter], 'balls': count[0], 'strikes': count[1],
                     'outcome10': int(rng.integers(10)) if outcome is None else outcome, 'batter': batter,
                     'description': description, 'bb_type': bb})

    for _ in range(900):
        pitcher = int(rng.choice(list(REPERTOIRE)))
        kind = str(rng.choice(REPERTOIRE[pitcher]))
        b = BASE[kind]
        phys = [b[0]+.5*rng.integers(-6, 7), b[1]+.5*rng.integers(-400, 401), b[2]+.5*rng.integers(-5, 6),
                b[3]+.5*rng.integers(-5, 6)]
        add(days[rng.integers(len(days))], pitcher, kind, int(rng.choice([201, 202, 203, 204])),
            COUNTS[rng.integers(len(COUNTS))], phys)
    for i in rng.choice(900, 40, replace=False)[:15]:   # one missing physics reading
        rows[i][PHYSICS[i % 4]] = np.nan
    for i in rng.choice(900, 60, replace=False)[:10]:   # NaN-physics rows (no reading at all)
        rows[i].update(dict.fromkeys(PHYSICS, np.nan))
    for i in rng.choice(900, 80, replace=False)[:25]:   # projection literal for an unclassified pitch
        rows[i]['pitch_type'] = 'UNKNOWN'
    # EP: two pitchers, identical physics everywhere -> pooled std 0 -> scale-floor abstention.
    for k in range(12):
        add(days[k], (102, 104)[k % 2], 'EP', 202, (0, 0), (60., 1500., 1., -2.))
    # SC: a 400-pitch constant cluster (102) and one far outlier (104) -> similarity weights underflow to 0.
    for k in range(400):
        add(days[k % 20], 102, 'SC', 202, (0, 0), (80., 2000., 5., 1.))
    add(days[3], 104, 'SC', 202, (0, 0), (95., 2600., -8., 9.))
    # Same-day rows on the mid-pool query date (must be excluded by both paths for the as-of case).
    for k in range(8):
        add(date(2026, 5, 15), 101, 'FF', 201, (1, 1), (95.5, 2350., -2.5, 9.), 'hit_into_play', 4)
    frame = pd.DataFrame(rows).sort_values('game_date', kind='stable').reset_index(drop=True)
    n = frame.groupby('game_date').cumcount()
    frame['game_pk'] = 700000+frame.game_date.map({d: i for i, d in enumerate(days)}).astype(int)
    frame['at_bat_number'] = (n//5+1).astype(int)
    frame['pitch_number'] = (n % 5+1).astype(int)
    frame['game_date'] = pd.to_datetime(frame.game_date)
    return frame


# One scenario = one pre-pitch decision, written in Statcast terms (path B's native vocabulary).
#   prefix: pitch types of the pitches already thrown in the open PA (last one is the previous pitch).
SCENARIOS = {
    'first_pitch': dict(pitcher=101, batter=201, balls=0, strikes=0, outs=0, inning=1, top=True, home=0, away=0,
                        on=(None, None, None), prefix=(), date=OFFICIAL),
    'mid_pa_previous_ff': dict(pitcher=101, batter=201, balls=1, strikes=1, outs=2, inning=7, top=False, home=3, away=5,
                               on=(660001, None, 660003), prefix=('SL', 'FF'), date=OFFICIAL),
    'no_history_pitcher_batter_prev_unknown': dict(pitcher=105, batter=299, balls=0, strikes=2, outs=1, inning=9,
                                                   top=True, home=2, away=2, on=(None, 660002, None),
                                                   prefix=('FF', 'UNKNOWN'), date=OFFICIAL),
    'lefty_full_count_previous_sl': dict(pitcher=103, batter=204, balls=3, strikes=2, outs=1, inning=11, top=False,
                                         home=7, away=1, on=(660001, 660002, 660003),
                                         prefix=('FF', 'SI', 'CH', 'FF', 'SL'), date=OFFICIAL),
    'similarity_underflow_and_scale_floor': dict(pitcher=104, batter=202, balls=0, strikes=0, outs=0, inning=3,
                                                 top=True, home=1, away=0, on=(None, None, None), prefix=(),
                                                 date=OFFICIAL),
    'history_batter_new_pitcher': dict(pitcher=105, batter=203, balls=1, strikes=1, outs=0, inning=5, top=True,
                                       home=0, away=4, on=(None, None, 660003), prefix=('CU', 'CH'), date=OFFICIAL),
    # A FIT-period training row: the pool also holds same-day and later rows, which must not enter.
    'mid_pool_as_of': dict(pitcher=101, batter=201, balls=1, strikes=1, outs=1, inning=4, top=True, home=1, away=1,
                           on=(660001, None, None), prefix=('CH', 'FF'), date=MID),
}
GUMBO_SCENARIO = 'mid_pa_previous_ff'


def query_rows(s, index, context_columns):
    """Path B query frame: 17 hypothetical games, same PA prefix, one candidate current pitch each (unique keys, so
    ``query_context`` derives previous_type/previous_valid exactly as it did for training rows)."""
    rows = []
    for i, action in enumerate(ACTIONS):
        for number, kind in enumerate((*s['prefix'], action), 1):
            rows.append({'game_pk': 900000+100*index+i, 'at_bat_number': 12, 'pitch_number': number,
                         'game_date': pd.Timestamp(s['date']), 'pitcher': s['pitcher'], 'batter': s['batter'],
                         'pitch_type': kind, 'balls': s['balls'], 'strikes': s['strikes'], 'outs_when_up': s['outs'],
                         'inning': s['inning'], 'inning_topbot': 'Top' if s['top'] else 'Bot',
                         'home_score': s['home'], 'away_score': s['away'], 'on_1b': s['on'][0], 'on_2b': s['on'][1],
                         'on_3b': s['on'][2], 'stand': STAND[s['batter']], 'p_throws': HAND[s['pitcher']],
                         'candidate': number == len(s['prefix'])+1})
    frame = pd.DataFrame(rows)
    for c in ('on_1b', 'on_2b', 'on_3b'):
        frame[c] = frame[c].astype(float)
    return frame[list(context_columns)], frame.candidate.to_numpy()


# ---------------------------------------------------------------------------------------------------------------
# Path B child process.
# ---------------------------------------------------------------------------------------------------------------
def path_b_main(out):
    spec = importlib.util.spec_from_file_location('scale_runner', B_RUNNER)
    SR = importlib.util.module_from_spec(spec)
    sys.modules['scale_runner'] = SR
    spec.loader.exec_module(SR)
    c = SR.load(B_REG)
    M = SR.closure(c)           # sha-checks the pinned helpers, installs the deny finder (this process only)
    sf = M.safe
    enc = json.loads(ENC002.read_text())
    base = synthetic_pool()
    pools = {'sorted': base[sf.DONOR], 'shuffled': base.sample(frac=1, random_state=7).reset_index(drop=True)[sf.DONOR]}
    result = {'names': None, 'cases': {}, 'donor_columns': list(sf.DONOR), 'schema': list(sf.SCHEMA),
              'pinned': {k: v[1] for k, v in c['pinned'].items()}, 'history_cutoff': c['history_cutoff']}
    for order, pool in pools.items():
        for index, (name, s) in enumerate(SCENARIOS.items()):
            rows, candidate = query_rows(s, index, sf.CONTEXT)
            joined, _ = SR.features(pool, rows, CUTOFF, M)      # the exact function the training run called
            result['cases'][(order, name)] = {'raw': joined[candidate], 'x': sf.apply_encoder(joined, enc)[candidate]}
    # Field names: the same pieces features() concatenates.
    rows, _ = query_rows(SCENARIOS['first_pitch'], 0, sf.CONTEXT)
    _, _, names = sf.schema_frame(sf.query_context(rows))
    result['names'] = list(names)+list(M.ot.NAMES)
    # Residual difference: a null pitch_type in the pool is refused by the training path.
    null_pool = pools['sorted'].copy()
    null_pool['pitch_type'] = null_pool.pitch_type.astype(object).where(null_pool.pitch_type != 'UNKNOWN', None)
    try:
        SR.features(null_pool, rows, CUTOFF, M)
        result['null_pitch_type'] = 'ACCEPTED'
    except ValueError as error:
        result['null_pitch_type'] = str(error)
    result['denied_loaded'] = sorted(n for n in sys.modules if n.split('.')[0] in SR.DENIED)
    Path(out).write_bytes(pickle.dumps(result))


if __name__ == '__main__':
    assert sys.argv[1] == '--path-b'
    path_b_main(sys.argv[2])
    raise SystemExit(0)


# ---------------------------------------------------------------------------------------------------------------
# Tests (pytest process: path A only; torch/lightgbm stay importable here).
# ---------------------------------------------------------------------------------------------------------------
import pytest  # noqa: E402

EXTERNAL = [*PINS, B_REG]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in EXTERNAL),
                                reason='pitcheezy-pm-jobs sources (outside the repo) are not on this machine')


def _pycache_state():
    out = {}
    for d in {A_FEATURES.parent, B_DIR}:
        cache = d/'__pycache__'
        out[str(cache)] = sorted((f.name, f.stat().st_mtime_ns) for f in cache.iterdir()) if cache.exists() else None
    return out


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    for path, pin in PINS.items():
        assert sha256(path) == pin, f'sha256 pin changed: {path}'
    before = _pycache_state()
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec = importlib.util.spec_from_file_location('_step5_shadow_features', A_FEATURES)
        F = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(F)
    finally:
        sys.dont_write_bytecode = old
    out = tmp_path_factory.mktemp('path_b')/'b.pkl'
    run = subprocess.run([sys.executable, '-B', '-I', str(Path(__file__).resolve()), '--path-b', str(out)],
                         capture_output=True, text=True, timeout=110, cwd=str(out.parent))
    assert run.returncode == 0, 'path B child failed:\n'+run.stdout[-2000:]+run.stderr[-4000:]
    B = pickle.loads(out.read_bytes())
    assert _pycache_state() == before, '__pycache__ written into a pinned source directory'
    assert not any(n in sys.modules for n in ('scale_runner', 'scale_safe', 'first_small_features'))
    assert not any(type(f).__name__ == 'Deny' for f in sys.meta_path), 'deny finder leaked into the pytest process'
    enc002 = json.loads(ENC002.read_text())
    f1, _ = F.load_encoder(F1_DESIGN, PINS[F1_DESIGN])
    return {'F': F, 'B': B, 'enc002': enc002, 'f1': f1, 'pool': synthetic_pool()}


def donor_rows(F, frame, null_unknown=False):
    """DataFrame -> the list-of-dict rows ProfileStore takes (python natives; a missing reading as None)."""
    rows = []
    for r in frame[list(F.DONOR_COLUMNS)].to_dict('records'):
        row = {}
        for k, v in r.items():
            if k == 'game_date':
                v = v.date()
            elif k in PHYSICS:
                v = None if v != v else float(v)
            elif k in ('pitch_type', 'p_throws', 'stand', 'description', 'bb_type'):
                v = None if v is None or v != v else str(v)
            else:
                v = int(v)
            row[k] = v
        if null_unknown and row['pitch_type'] == 'UNKNOWN':
            row['pitch_type'] = None
        rows.append(row)
    return rows


def state_of(s):
    """Hand translation of a Statcast-terms scenario into the Step 5 pre-pitch state."""
    return {'balls': s['balls'], 'strikes': s['strikes'], 'outs': s['outs'], 'inning': s['inning'], 'top': s['top'],
            'score': (s['home'], s['away']), 'bases': tuple(v is not None for v in s['on']),
            'stand': STAND[s['batter']], 'p_throws': HAND[s['pitcher']], 'pitcher': s['pitcher'],
            'batter': s['batter'], 'previous_type': s['prefix'][-1] if s['prefix'] else 'UNKNOWN',
            'previous_valid': bool(s['prefix']), 'official_date': s['date']}


def store_for(F, frame, s, **kw):
    """As-of store for a query date: rows strictly before the query date, cutoff = the day before (or the pool
    cutoff for post-cutoff queries)."""
    if s['date'] > CUTOFF:
        cutoff = CUTOFF
    else:
        cutoff = (date.fromisoformat(s['date'])-timedelta(days=1)).isoformat()
        frame = frame[frame.game_date < pd.Timestamp(s['date'])]
    store = F.ProfileStore(donor_rows(F, frame, **kw), as_of_cutoff=cutoff, source_id='synthetic',
                           source_sha256='0'*64, role='SYNTHETIC_TEST')
    assert store.strictly_prior(s['date'])
    return store


def raw_diff(F, a, b):
    """Columns where two raw matrices differ (NaN == NaN), as 'index:name(max abs diff)'."""
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    cols = np.flatnonzero(~same.all(0))
    return [f'{j}:{F.RAW_NAMES[j]}({np.nanmax(np.abs(a[:, j]-b[:, j])):.3e})' for j in cols]


def test_pins_and_registration(world):
    B = world['B']
    assert B['denied_loaded'] == []
    assert B['history_cutoff'] == '2026-08-02'      # scale-002 registration read, not used as the synthetic cutoff
    assert set(world['F'].DONOR_COLUMNS) == set(B['donor_columns'])
    assert world['enc002']['dimension'] == 199 and len(world['enc002']['continuous']) == 36


def test_raw_names_order(world):
    assert list(world['F'].RAW_NAMES) == world['B']['names']
    assert len(world['B']['names']) == 91
    assert tuple(world['F'].TYPES) == TYPES and tuple(world['F'].ACTIONS) == ACTIONS


@pytest.mark.parametrize('name', list(SCENARIOS))
def test_raw_and_encoded_equal(world, name):
    F, s = world['F'], SCENARIOS[name]
    b = world['B']['cases'][('sorted', name)]
    x, raw = F.build_candidates(state_of(s), store_for(F, world['pool'], s), world['enc002'])
    assert raw.shape == b['raw'].shape == (17, 91) and x.shape == b['x'].shape == (17, 199)
    assert x.dtype == b['x'].dtype == np.float32
    diff = raw_diff(F, raw, b['raw'])
    assert not diff, f'{name}: raw columns differ between Step 5 and scale-002 paths: {diff}'
    assert np.array_equal(F.apply_encoder(raw, world['enc002']), x)
    bad = sorted(set(np.flatnonzero((x != b['x']).any(0)).tolist()))
    assert not bad, f'{name}: encoded float32 columns differ: {bad}'
    assert x.tobytes() == b['x'].tobytes()
    # Negative control: the frozen 2024 F1 encoder gives a different vector from the scale-002 encoder.
    x_f1 = F.apply_encoder(raw, world['f1'])
    assert x_f1.shape == (17, 199) and not np.array_equal(x_f1, b['x'])
    assert (x_f1 != b['x']).any(1).all()


def test_scenarios_exercise_the_intended_cells(world):
    """The synthetic world really contains the cases the equality is claimed for (read from path B's output)."""
    F, cases, pool = world['F'], world['B']['cases'], world['pool']
    n = F.RAW_NAMES.index
    col = lambda name, field, action: cases[('sorted', name)]['raw'][ACTIONS.index(action), n(field)]
    assert col('first_pitch', 'previous_valid', 'FF') == 0 and col('mid_pa_previous_ff', 'previous_FF', 'SL') == 1
    assert col('no_history_pitcher_batter_prev_unknown', 'previous_UNKNOWN', 'FF') == 1
    assert col('mid_pa_previous_ff', 'own_outcome_missing', 'FF') == 0          # own cell in the exact context
    assert col('mid_pa_previous_ff', 'donor_outcome_missing', 'FF') == 0
    assert col('mid_pa_previous_ff', 'similarity_abstention', 'FF') == 0
    assert col('mid_pa_previous_ff', 'own_outcome_missing', 'FS') == 0          # only this pitcher throws FS
    assert col('mid_pa_previous_ff', 'donor_outcome_missing', 'FS') == 1        # cell with no donors
    assert col('mid_pa_previous_ff', 'donor_outcome_missing', 'KN') == 1        # type nobody throws
    assert col('mid_pa_previous_ff', 'batter_prior_log_count', 'FF') > 0        # batter with history
    assert np.isnan(col('mid_pa_previous_ff', 'prior_release_speed', 'KN'))
    none = 'no_history_pitcher_batter_prev_unknown'
    assert col(none, 'pitcher_type_prior_log_count', 'FF') == 0 and col(none, 'batter_prior_log_count', 'FF') == 0
    assert col(none, 'prior_pfx_x_missing', 'FF') == 1 and col(none, 'batter_prior_swing_missing', 'FF') == 1
    assert col('history_batter_new_pitcher', 'batter_prior_log_count', 'FF') > 0
    assert col('history_batter_new_pitcher', 'donor_outcome_missing', 'FF') == 0
    edge = 'similarity_underflow_and_scale_floor'
    assert col(edge, 'similarity_abstention', 'SC') == 1 and col(edge, 'donor_outcome_missing', 'SC') == 0
    assert col(edge, 'similarity_abstention', 'EP') == 1 and col(edge, 'donor_outcome_missing', 'EP') == 0
    assert col(edge, 'similarity_abstention', 'FF') == 0
    # The as-of case differs from the same decision asked after the cutoff (later rows do move the features).
    late = dict(SCENARIOS['mid_pool_as_of'], date=OFFICIAL)
    _, raw_late = F.build_candidates(state_of(late), store_for(F, pool, late), world['enc002'])
    assert raw_diff(F, raw_late, cases[('sorted', 'mid_pool_as_of')]['raw'])
    physics = pool[list(PHYSICS)].isna()
    assert physics.all(axis=1).sum() >= 5 and (physics.any(axis=1) & ~physics.all(axis=1)).sum() >= 5
    assert pool.pitcher.nunique() >= 3 and pool.game_pk.nunique() >= 20
    assert pool.game_date.min() == pd.Timestamp('2026-04-01') and pool.game_date.max() == pd.Timestamp(CUTOFF)
    assert (pool.game_date == pd.Timestamp(MID)).sum() >= 8


@pytest.mark.parametrize('name', list(SCENARIOS))
def test_null_pitch_type_in_pool(world, name):
    """Residual difference 1: A given None where the training pool says 'UNKNOWN' still equals B; B refuses nulls."""
    F, s = world['F'], SCENARIOS[name]
    _, raw = F.build_candidates(state_of(s), store_for(F, world['pool'], s, null_unknown=True), world['enc002'])
    diff = raw_diff(F, raw, world['B']['cases'][('sorted', name)]['raw'])
    assert not diff, f'{name}: null-vs-UNKNOWN pool pitch_type changes raw columns: {diff}'
    assert world['B']['null_pitch_type'] == 'Missing history grouping key'
    assert (world['pool'].pitch_type == 'UNKNOWN').sum() >= 20


@pytest.mark.parametrize('name', list(SCENARIOS))
def test_shuffled_pool_summation_order(world, name):
    """Residual difference 3: with a pool that is not date-sorted, donor order inside a cell differs between the
    paths. Raw must agree to rtol 1e-12; the float32 encoding must still be bit-equal (a flipped float32 would fail
    here with the column list)."""
    F, s = world['F'], SCENARIOS[name]
    shuffled = world['pool'].sample(frac=1, random_state=7).reset_index(drop=True)
    b = world['B']['cases'][('shuffled', name)]
    x, raw = F.build_candidates(state_of(s), store_for(F, shuffled, s), world['enc002'])
    assert np.allclose(raw, b['raw'], rtol=1e-12, atol=0, equal_nan=True), raw_diff(F, raw, b['raw'])
    print('shuffled-pool raw columns not bit-equal:', name, raw_diff(F, raw, b['raw']))
    bad = sorted(set(np.flatnonzero((x != b['x']).any(0)).tolist()))
    assert not bad, f'{name}: float32 flipped by summation order in encoded columns {bad}'
    # Path B itself: shuffled vs sorted pool.
    assert np.allclose(b['raw'], world['B']['cases'][('sorted', name)]['raw'], rtol=1e-12, atol=0, equal_nan=True)


def gumbo_feed(s):
    """Synthetic GUMBO body for the decision ``s`` (fields ``pre_pitch_state`` reads; plus values it must ignore)."""
    balls = strikes = 0
    events = [{'isPitch': False, 'details': {'description': 'Mound visit'}}]
    for number, kind in enumerate(s['prefix'], 1):
        last = number == len(s['prefix'])
        balls, strikes = (s['balls'], s['strikes']) if last else (balls, strikes)
        events.append({'isPitch': True, 'pitchNumber': number, 'count': {'balls': balls, 'strikes': strikes},
                       'details': {'type': {'code': kind}, 'isInPlay': False, 'call': {'code': 'B'}},
                       'pitchData': {'startSpeed': 99.9}})
    offense = {k: {'id': v} for k, v in zip(('first', 'second', 'third'), s['on']) if v is not None}
    return {'gameData': {'status': {'abstractGameState': 'Live'}, 'datetime': {'officialDate': s['date']}},
            'liveData': {'plays': {'currentPlay': {
                'about': {'atBatIndex': 11, 'isComplete': False, 'inning': s['inning'], 'isTopInning': s['top']},
                'count': {'balls': s['balls'], 'strikes': s['strikes'], 'outs': s['outs']},
                'matchup': {'batter': {'id': s['batter']}, 'pitcher': {'id': s['pitcher']},
                            'batSide': {'code': STAND[s['batter']]}, 'pitchHand': {'code': HAND[s['pitcher']]}},
                'playEvents': events, 'result': {'homeScore': s['home'], 'awayScore': s['away']}}},
                'linescore': {'currentInning': s['inning'], 'isTopInning': s['top'], 'offense': offense}}}


@pytest.mark.parametrize('name', [GUMBO_SCENARIO, 'first_pitch', 'lefty_full_count_previous_sl',
                                  'no_history_pitcher_batter_prev_unknown'])
def test_end_to_end_from_gumbo(world, name):
    F, s = world['F'], SCENARIOS[name]
    state, missing = F.pre_pitch_state(gumbo_feed(s))
    assert missing == {}
    assert state['at_bat_number'] == 12 and state['last_pitch_number'] == len(s['prefix'])
    assert state['pa_status'] == 'open' and state['official_date'] == OFFICIAL
    expected = state_of(s)
    assert {k: state[k] for k in expected} == expected
    b = world['B']['cases'][('sorted', name)]
    x, raw = F.build_candidates(state, store_for(F, world['pool'], s), world['enc002'])
    diff = raw_diff(F, raw, b['raw'])
    assert not diff, f'{name}: GUMBO -> raw differs from the training path in columns {diff}'
    assert x.tobytes() == b['x'].tobytes()
    assert not np.array_equal(F.apply_encoder(raw, world['f1']), b['x'])
