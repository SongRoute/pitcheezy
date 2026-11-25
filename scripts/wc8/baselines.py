"""WC8-CONFIRM-001 step 2: label-free baselines for one game, on the same rows as the predictor files.

    python -B scripts/wc8/baselines.py --game <pk> --out <dir>      # -> <dir>/<pk>-baselines.json (never overwrites)

Rows = scripts/demo_precompute.py replay_snapshots of the final feed (the predict stage's own order n); per row the
pre-pitch placeholder row of live_rows (service_key, pitcher, balls, strikes; its type is never read):
  freq   eval_prediction.frozen_frequency (alpha 4, global -> pitcher -> pitcher:balls:strikes) on the FS-001
         fullseason baseline counts (2026 FIT 03-25..09-27; the file EXP-PREQ-001 pinned, sha e7c2f2aa)
  modal  the pitcher's most frequent 9-group class in the same FIT table; tie -> the class with the higher global
         count; pitcher not in the table -> the global modal class. A point prediction (top-1 only).
No label is extracted or written. Same-game earlier pitches are not used (both baselines are FIT tables only).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C  # noqa: E402
import evalfns as E  # noqa: E402


def load_table(spec=C.FREQ_TABLE):
    path = pathlib.Path(spec['path'])
    if C.sha256_file(path) != spec['sha256']:
        raise SystemExit('frequency table differs from its pin')
    t = json.loads(path.read_text())[spec['entry']]
    if tuple(t['classes']) != C.CLASSES or float(t['alpha']) != spec['alpha'] or int(t['rows']) != spec['rows']:
        raise SystemExit('frequency table entry differs from the registration (classes / alpha / rows)')
    return t


def modal_class(table, pitcher):
    """Index of the pitcher's modal class; ties -> higher global count (then lower index); unknown -> global modal."""
    g = np.asarray(table['global'], np.float64)
    order_global = sorted(range(len(g)), key=lambda k: (-g[k], k))
    pc = table['pitcher'].get(str(int(pitcher)))
    if pc is None or float(np.sum(pc)) <= 0:
        return order_global[0], False
    pc = np.asarray(pc, np.float64)
    best = pc.max()
    tied = [k for k in order_global if pc[k] == best]
    return tied[0], True


def baseline_rows(table, states):
    """states: iterable of (n, service_key, pitcher, balls, strikes) -> output rows (pure)."""
    out = []
    for n, key, pitcher, balls, strikes in states:
        q = E.frozen_frequency(table, int(pitcher), int(balls), int(strikes))
        if q.shape != (9,) or not np.isfinite(q).all() or abs(q.sum() - 1) > 1e-9:
            raise SystemExit(f'n {n}: frequency q invalid')
        m, known = modal_class(table, pitcher)
        out.append({'n': int(n), 'service_key': key, 'pitcher': int(pitcher), 'balls': int(balls), 'strikes': int(strikes),
                    'freq_q': [float(v) for v in q], 'modal': int(m), 'pitcher_in_table': bool(known)})
    return out


def replay_states(final):
    """(n, service_key, pitcher, balls, strikes) per replay snapshot, exactly as the predict stage forms service_key."""
    from pitcheezy.svc_p9 import predictor as P9
    P9.activate_snapshot()               # same import order as scripts/svc_r1/predict_stage.run
    import demo_precompute as D
    for n, snap in D.replay_snapshots(final):
        row = D.live_rows(snap).iloc[-1]
        yield n, D._key(row), int(row['pitcher']), int(row['balls']), int(row['strikes'])


def run(pk, out_dir, gate=False):
    commit = C.require_clean()
    game = C.game_entry(pk, allow_demo=gate)
    out = pathlib.Path(out_dir).resolve() / C.baseline_name(game['game_pk'])
    if out.exists():
        raise SystemExit(f'{out} exists')
    table = load_table()
    final = C.read_feed_verified(game)
    rows = baseline_rows(table, replay_states(final))
    del final
    if len(rows) != game['is_pitch'] or [r['n'] for r in rows] != list(range(len(rows))):
        raise SystemExit(f"rows {len(rows)} != is_pitch {game['is_pitch']} or n not 0..N-1")
    doc = {'format': 'wc8-baselines-v1', 'game_pk': game['game_pk'], 'official_date': game['official_date'],
           'final_feed_sha256': game['final_feed_sha256'], 'facts_sha256': C.FACTS_SHA, 'classes': list(C.CLASSES),
           'frequency_table': {k: C.FREQ_TABLE[k] for k in ('path', 'sha256', 'entry', 'alpha', 'rows')},
           'freq_rule': 'eval_prediction.frozen_frequency (alpha 4; global -> pitcher -> pitcher:balls:strikes)',
           'modal_rule': 'pitcher modal 9-group class in the FIT table; tie -> higher global count; unknown pitcher -> global modal',
           'eval_reference_sha256': C.EVAL_REFERENCE['sha256'], 'code_commit': commit, 'labels_read': False, 'rows': rows}
    sha = C.write_new(out, doc)
    return {'out': str(out), 'sha256': sha, 'rows': len(rows), 'pitchers_not_in_table': sum(not r['pitcher_in_table'] for r in rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--game', type=int, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--gate', action='store_true', help='allow 849799 (reproduction gate only)')
    a = ap.parse_args(argv)
    print(json.dumps(run(a.game, a.out, gate=a.gate), indent=1))


if __name__ == '__main__':
    main()
