"""WC8-CONFIRM-001 step 4: pin every prediction / baseline file before the one scoring run (label-free).

    python -B scripts/wc8/pin.py --pred-dir <dir>      # -> <dir>/PINS.json (never overwrites)

Requires exactly the 8 games x {v4, bseq, r} predictions + receipts + configs and the 8 baseline files (no other
json in the directory except PINS.json absent). Label-free consistency before pinning: each receipt ok and as-of ok,
receipt sha == file sha, same game / feed sha, rows == registration-facts is_pitch, identical (n, service_key,
w1_key) across the three models and (n, service_key) with the baselines, q valid (9, finite, >= 0, sum 1), and
the service_key at-bat of row n == the at-bat of the n-th isPitch event of the feed (structure only, no pitch type).
Records sha256 of every file, the code commit and the sha of every scripts/wc8 code file (score.py re-checks).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C  # noqa: E402


def validate_game(pk, docs, receipts, base, game):
    """Label-free checks for one game; returns (problems, summary). docs/receipts: {model: dict}."""
    bad = []
    keys = {}
    for m in C.MODELS:
        d, r = docs[m], receipts[m]
        if d.get('format') != 'svc-r1-predictions-v1' or d['game_pk'] != pk or d['final_feed_sha256'] != game['final_feed_sha256']:
            bad.append(f'{pk} {m}: format / game / feed')
        if tuple(d['classes']) != C.CLASSES:
            bad.append(f'{pk} {m}: classes')
        if not (r.get('ok') is True and r.get('as_of_ok') is True and r['trained_through'] < game['official_date']):
            bad.append(f'{pk} {m}: receipt not ok / as-of')
        if r.get('predictor_pin_sha256') != C.MODEL_SPEC[m]['pin'][1] or r.get('labels_read') is not False:
            bad.append(f'{pk} {m}: receipt pin / labels_read')
        if len(d['rows']) != game['is_pitch']:
            bad.append(f"{pk} {m}: rows {len(d['rows'])} != {game['is_pitch']}")
        keys[m] = [(row['n'], row['service_key'], row['w1_key']) for row in d['rows']]
        for row in d['rows']:
            q = row['q']
            if q is None:
                if not row['omit_reason']:
                    bad.append(f"{pk} {m} n {row['n']}: q None without omit_reason")
                continue
            if len(q) != 9 or any((not math.isfinite(v)) or v < 0 for v in q) or abs(sum(q) - 1) > 1e-9:
                bad.append(f"{pk} {m} n {row['n']}: q format")
    if not (keys['v4'] == keys['bseq'] == keys['r']):
        bad.append(f'{pk}: (n, service_key, w1_key) differ across models')
    if base.get('format') != 'wc8-baselines-v1' or base['game_pk'] != pk or base['final_feed_sha256'] != game['final_feed_sha256'] \
            or base['frequency_table']['sha256'] != C.FREQ_TABLE['sha256'] or base.get('labels_read') is not False:
        bad.append(f'{pk} baselines: format / game / feed / table')
    if [(r['n'], r['service_key']) for r in base['rows']] != [(k[0], k[1]) for k in keys['v4']]:
        bad.append(f'{pk}: baseline (n, service_key) differ from the predictions')
    omit = {m: sorted(row['n'] for row in docs[m]['rows'] if row['q'] is None) for m in C.MODELS}
    return bad, {'rows': game['is_pitch'], 'omitted_n': omit, 'omit_identical': omit['v4'] == omit['bseq'] == omit['r']}


def run(pred_dir):
    commit = C.require_clean()
    d = pathlib.Path(pred_dir).resolve()
    out = d / 'PINS.json'
    if out.exists():
        raise SystemExit(f'{out} exists')
    want = C.expected_files()
    have = sorted(p.name for p in d.glob('*.json'))
    if have != want:
        raise SystemExit(f'prediction directory differs from the expected set: missing {sorted(set(want) - set(have))}, '
                         f'extra {sorted(set(have) - set(want))}')
    f = C.facts()
    problems, games = [], {}
    for pk in C.WC8:
        game = C.game_entry(pk, f=f)
        docs = {m: json.loads((d / C.pred_name(pk, m)).read_text()) for m in C.MODELS}
        receipts = {m: json.loads((d / C.receipt_name(pk, m)).read_text()) for m in C.MODELS}
        for m in C.MODELS:
            if receipts[m]['predictions_sha256'] != C.sha256_file(d / C.pred_name(pk, m)):
                problems.append(f'{pk} {m}: predictions sha differs from its receipt')
            if receipts[m]['config_sha256'] != C.sha256_file(d / C.config_name(pk, m)):
                problems.append(f'{pk} {m}: config sha differs from its receipt')
        base = json.loads((d / C.baseline_name(pk)).read_text())
        bad, summary = validate_game(pk, docs, receipts, base, game)
        final = C.read_feed_verified(game)   # at-bat order only (no pitch type / outcome is read)
        if not C.keys_match_order([r['service_key'] for r in docs['v4']['rows']], C.ab_order(final), pk):
            bad.append(f'{pk}: service_key at-bats differ from the feed pitch order')
        del final
        problems += bad
        games[str(pk)] = summary
    if problems:
        raise SystemExit(f'not pinned: {problems[:20]}')
    doc = {'format': 'wc8-pins-v1', 'registration': 'WC8-CONFIRM-001', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
           'code_commit': commit, 'code_sha256': {rel: C.sha256_file(C.REPO / rel) for rel in C.CODE_FILES},
           'eval_reference': C.EVAL_REFERENCE, 'facts_sha256': C.FACTS_SHA, 'frequency_table_sha256': C.FREQ_TABLE['sha256'],
           'files': {name: C.sha256_file(d / name) for name in want}, 'games': games, 'labels_read': False}
    sha = C.write_new(out, doc)
    return {'pins': str(out), 'sha256': sha, 'files': len(want), 'code_commit': commit,
            'omit_identical_all_games': all(g['omit_identical'] for g in games.values())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pred-dir', type=pathlib.Path, required=True)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.pred_dir), indent=1))


if __name__ == '__main__':
    main()
