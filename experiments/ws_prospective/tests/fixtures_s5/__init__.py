"""Synthetic fixtures for the N1 / Step 5 adapter tests (no real 2026 rows, no network).

The game script, fixture profile and replay writer are copied from the accepted seq-shadow tests
(``tests/test_seq_shadow.py`` @ main: ``profile_rows``, ``write_profile``, ``ev``, ``play``, ``scripted``,
``done_plays``, ``body``, ``script``, ``write_replay``) so the accepted runners and the adapters see the same
bodies. ``write_store`` puts the same canonical bodies into a fake observer store (snapshots table with the real
sha256 of the canonical bytes) for ``store_source.read_snapshot_raw``.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_SOURCES = json.loads((HERE/'model_sources.json').read_text())
GAME, DAY, CUTOFF = 990001, '2099-01-02', '2099-01-01'
PA, PB, PC, PD = 101, 102, 103, 104  # A: all 17 types; B: absent from the profile; C: five types; D: donor
C_TYPES = ('FF', 'SL', 'CH', 'SI', 'CU')
VM_OK = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               100000.\n'
         'Pages active:                             500000.\nPages inactive:                           400000.\n'
         'Pages speculative:                         50000.\nPages wired down:                         100000.\n')
PS_QUIET = ('    1 Wed Sep 23 22:00:00 2026     /sbin/launchd\n'
            '  777 Tue Oct  6 09:00:00 2026     /usr/bin/some-unrelated-thing --flag\n')


def fake_run(argv):
    return VM_OK if argv[0].endswith('vm_stat') else PS_QUIET


def model_sources(scope='SYNTHETIC_OWNED_FIXTURE'):
    out = copy.deepcopy({k: v for k, v in MODEL_SOURCES.items() if not k.startswith('_')})
    for v in out.values():
        v['profile_scope'] = scope
    return out


# ------------------------------------------------------------------ fixture profile
def profile_rows(F):
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


def write_profile(directory, F, profile_reader):
    """Attempt directory with PROFILE.json + COMMITTED RECEIPT.json (synthetic scope, no supervisor). Returns
    the PROFILE.json path and its sha256."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    store = F.ProfileStore(profile_rows(F), as_of_cutoff=CUTOFF, source_id='synthetic-seq-shadow-fixture',
                           source_sha256='0'*64, role=profile_reader.ROLE)
    blob = json.dumps(profile_reader.dump_store(store)).encode()
    (directory/'PROFILE.json').write_bytes(blob)
    (directory/'RECEIPT.json').write_text(json.dumps({
        'terminal': 'COMMITTED', 'profile_committed': True, 'scope': profile_reader.SYNTHETIC_SCOPE,
        'attempt_id': 'synthetic-seq-shadow-fixture-001', 'counts': {'counts': {'rows_projected': store.rows}},
        'profile': {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}}))
    return directory/'PROFILE.json', hashlib.sha256(blob).hexdigest()


# ------------------------------------------------------------------ fixture game
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
    events, balls, strikes, n = [], 0, 0, 0
    for item in spec:
        if item == 'AUTO':
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
    """[(name, body)] in receipt order (19 bodies), identical to the accepted seq-shadow test script."""
    D = done_plays()
    a1, a2 = ev(1, 'FF', 'B', 1, 0), ev(2, 'SL', 'C', 1, 1)
    a2_untyped = ev(2, None, 'C', 1, 1)
    a3 = ev(3, 'FF', 'X', 1, 1, in_play=True, out=True, traj='ground_ball')
    a1_fixed = ev(1, 'SI', 'B', 1, 0)
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


def write_replay(folder, bodies, F, *, spacing=5.0):
    folder = Path(folder)
    folder.mkdir(parents=True)
    files = []
    for i, (name, feed) in enumerate(bodies):
        raw = F.canonical(feed)
        file = f'{GAME}_{i+1:06d}_{name}.json.gz'
        (folder/file).write_bytes(gzip.compress(raw, mtime=0))
        t = 4070995200.0+1000+spacing*i
        files.append({'file': file, 'seq': i+1, 'version': 1, 'sha256': hashlib.sha256(raw).hexdigest(),
                      'first_fetched': t, 'fetched': t, 'event_wall': None, 'source': 'live_poll'})
    manifest = {'status': 'OK', 'game_pk': GAME, 'mode': {'bodies_modified': False}, 'files': files,
                'synthetic_fixture': True}
    (folder/'MANIFEST.json').write_text(json.dumps(manifest))
    return folder


# ------------------------------------------------------------------ fake observer store
SNAPSHOTS_DDL = """
CREATE TABLE IF NOT EXISTS snapshots (seq INTEGER PRIMARY KEY AUTOINCREMENT, game_pk INTEGER NOT NULL,
 event_wall REAL NOT NULL, version INTEGER NOT NULL, feed_ts TEXT, sha TEXT NOT NULL,
 first_fetched REAL NOT NULL, fetched REAL NOT NULL, source TEXT NOT NULL, body BLOB NOT NULL,
 requested REAL);
"""


def write_store(path, bodies, F, *, start=0):
    """Append canonical bodies to a fake store; returns [(seq, sha256)] in order."""
    con = sqlite3.connect(str(path), isolation_level=None)
    try:
        con.executescript(SNAPSHOTS_DDL)
        out = []
        for i, (_, feed) in enumerate(bodies):
            raw = F.canonical(feed)
            sha = hashlib.sha256(raw).hexdigest()
            t = 4070995200.0+1000+5.0*(start+i)
            cur = con.execute("INSERT INTO snapshots (game_pk,event_wall,version,sha,first_fetched,fetched,source,body) "
                              "VALUES (?,?,?,?,?,?,?,?)", (feed.get('gamePk', GAME), t, 1, sha, t, t, 'live_poll',
                                                           gzip.compress(raw, mtime=0)))
            out.append((cur.lastrowid, sha))
        return out
    finally:
        con.close()


def contexts(bodies, refs, profile_sha256):
    """For every body whose feed shows a next-slot pre-pitch state: (index, PrePitchContext) built exactly as the
    RS-01 runner does (store_source.parse_feed -> build_context with that snapshot's seq/sha)."""
    from ws_prospective.context import build_context
    from ws_prospective.store_source import parse_feed
    out = []
    for i, ((_, feed), (seq, sha)) in enumerate(zip(bodies, refs)):
        accepted, state, final = parse_feed(feed, {})
        if state is None or final:
            continue
        out.append((i, build_context(GAME, accepted, len(accepted)+1, state, profile_sha256,
                                     source_seq=seq, source_sha=sha)))
    return out
