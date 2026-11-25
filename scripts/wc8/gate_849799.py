"""WC8-CONFIRM-001 reproduction gate on 849799 (no label is read): scripts/wc8 predict path == booth predictions.

    python -B scripts/wc8/gate_849799.py --out <new gate dir>     # each predict / baseline run is a fresh process

  v4    predict.py --gate --game 849799 --model v4  vs  v4 instance predictions.path (sha 9d0dfcc4)
  r     predict.py --gate --game 849799 --model r   vs  R.json predictions.path        (sha 64da605c)
        pass = same row count and, per row, identical n / service_key / w1_key / omit_reason / fallback / flags and
        q both None or max |dq| <= 1e-12.
  bseq  predict.py --gate --game 849799 --model bseq: no booth file exists for B-SEQ alone (it ran only inside the
        C+SEQ+GBT_GM-w mix), so only (n, service_key, w1_key, omit) == the new v4 file and q validity (descriptive).
  base  baselines.py --gate --game 849799: (n, service_key) == the new v4 file (label-free key check of the baseline
        replay), and service_key at-bats == the feed's pitch order (structure only).
Writes <dir>/GATE.json (never overwrites). The WC8 runs should use the same python / env / threads as this gate.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C  # noqa: E402

TOL = 1e-12
ROW_KEYS = ('n', 'service_key', 'w1_key', 'omit_reason', 'fallback', 'flags')


def compare_rows(new, booth, tol=TOL):
    """Pure: problems list and max |dq| between two svc-r1-predictions-v1 row lists."""
    bad, worst = [], 0.0
    if len(new) != len(booth):
        return [f'row count {len(new)} != {len(booth)}'], None
    for a, b in zip(new, booth):
        for k in ROW_KEYS:
            if a.get(k) != b.get(k):
                bad.append(f"n {b.get('n')}: {k} {a.get(k)!r} != {b.get(k)!r}")
        qa, qb = a.get('q'), b.get('q')
        if (qa is None) != (qb is None):
            bad.append(f"n {b.get('n')}: q presence differs")
        elif qa is not None:
            if len(qa) != len(qb):
                bad.append(f"n {b.get('n')}: q length")
                continue
            d = max(abs(x - y) for x, y in zip(qa, qb))
            worst = max(worst, d)
            if not d <= tol:
                bad.append(f"n {b.get('n')}: max |dq| {d:.3e} > {tol}")
    return bad, worst


def booth_file(model):
    inst = json.loads((C.REPO / C.MODEL_SPEC[model]['template'][0]).read_text())
    path, sha = pathlib.Path(inst['predictions']['path']), inst['predictions']['sha256']
    if sha != C.MODEL_SPEC[model]['booth_sha'] or C.sha256_file(path) != sha:
        raise SystemExit(f'{model}: booth predictions file differs from its pin {sha[:8]}')
    return path, sha


def sub(args):
    r = subprocess.run([sys.executable, '-B', *args], cwd=C.REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f'{args[0]} failed: {r.stderr[-2000:]}')
    return r.stdout


def run(out_dir):
    commit = C.require_clean()
    d = pathlib.Path(out_dir).resolve()
    gate_path = d / 'GATE.json'
    if gate_path.exists():
        raise SystemExit(f'{gate_path} exists')
    pk = C.DEMO_GAME
    for m in ('v4', 'r'):
        booth_file(m)                      # pins first, before spending time
    for m in ('v4', 'r', 'bseq'):
        sub([str(C.REPO / 'scripts/wc8/predict.py'), '--gate', '--game', str(pk), '--model', m, '--out', str(d)])
    sub([str(C.REPO / 'scripts/wc8/baselines.py'), '--gate', '--game', str(pk), '--out', str(d)])
    new = {m: json.loads((d / C.pred_name(pk, m)).read_text()) for m in C.MODELS}
    res = {'format': 'wc8-gate-849799-v1', 'code_commit': commit, 'tolerance': TOL, 'models': {}, 'labels_read': False}
    for m in ('v4', 'r'):
        path, sha = booth_file(m)
        bad, worst = compare_rows(new[m]['rows'], json.loads(path.read_text())['rows'])
        res['models'][m] = {'booth': str(path), 'booth_sha256': sha, 'new_sha256': C.sha256_file(d / C.pred_name(pk, m)),
                            'rows': len(new[m]['rows']), 'max_abs_dq': worst, 'problems': bad[:50], 'pass': not bad}
    kv = [(r['n'], r['service_key'], r['w1_key'], r['omit_reason']) for r in new['v4']['rows']]
    kb = [(r['n'], r['service_key'], r['w1_key'], r['omit_reason']) for r in new['bseq']['rows']]
    qok = all(r['q'] is None or (len(r['q']) == 9 and all(math.isfinite(v) and v >= 0 for v in r['q']) and abs(sum(r['q']) - 1) <= 1e-9)
              for r in new['bseq']['rows'])
    res['models']['bseq'] = {'new_sha256': C.sha256_file(d / C.pred_name(pk, 'bseq')), 'keys_omit_equal_v4': kb == kv,
                             'q_valid': qok, 'pass': kb == kv and qok, 'note': 'no booth B-SEQ-only file; descriptive check'}
    base = json.loads((d / C.baseline_name(pk)).read_text())
    final = C.read_feed_verified(C.game_entry(pk, allow_demo=True))
    abs_ = C.ab_order(final)
    del final
    keys = [r['service_key'] for r in new['v4']['rows']]
    res['baselines'] = {'keys_equal_v4': [(r['n'], r['service_key']) for r in base['rows']] == [(k[0], k[1]) for k in kv],
                        'service_key_at_bats_match_feed_order': C.keys_match_order(keys, abs_, pk),
                        'pitchers_not_in_table': sum(not r['pitcher_in_table'] for r in base['rows'])}
    res['pass'] = (res['models']['v4']['pass'] and res['models']['r']['pass'] and res['models']['bseq']['pass']
                   and res['baselines']['keys_equal_v4'] and res['baselines']['service_key_at_bats_match_feed_order'])
    C.write_new(gate_path, res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args(argv)
    r = run(a.out)
    print(json.dumps({'pass': r['pass'], **{m: {k: v for k, v in r['models'][m].items() if k != 'problems'} for m in r['models']},
                      'baselines': r['baselines']}, indent=1))
    if not r['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
