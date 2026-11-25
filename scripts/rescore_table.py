"""Re-score the 9-model table (31 games, 8,943 pitches of the 2026 postseason, E_PS) from saved probabilities.

No training, no prediction: reads data/eps/{rows.csv, probs.npz} (made by scripts/export_saved_probs.py, or by you:
see docs/REPRODUCE.md) and results/final-mix-weights.json, then computes NLL, top-1 and top-3 for each model, and
the paired game-level bootstrap difference intervals used in figure 4.

Scoring rule (same code path as the original scorers, scripts/svc_p9/eval_prediction.py):
  - every one of the 8,943 rows counts; rows a model did not predict are 1/9 (already filled in probs.npz)
  - floor q' = (1 - 0.001) q + 0.001 / 9, then NLL = mean -log q'[y]; top-k = true class among argsort(-q')[:k]
  - label y = 9-group index of the actual pitch type (src/pitcheezy/interfaces/pitch_types.py)
  - final model = X-GBT, X-GRU-A3-plus, X-GRU-A5-b24 mixed with results/final-mix-weights.json
  - intervals: games resampled with replacement, 2,000 replicates, seed 20261110, percentiles 2.5 / 97.5

    python scripts/rescore_table.py                       # table + intervals -> results/rescore/
    python scripts/rescore_table.py --no-bootstrap        # table only (about a second)
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
from scipy.special import softmax

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts/svc_p9'))
import eval_prediction as E  # noqa: E402  (floor, row_parts, pooled, paired_contrast, label)

# (key in probs.npz or 'TOP1', label used in the README table, date the model was made)
TABLE = [('TOP1', 'LightGBM + GRU 앙상블', '11/21~22'),
         ('B-SEQ', 'GRU', '11/21'),
         ('B-GBT', 'LightGBM', '11/21'),
         ('B-TAK', 'Takamido식 Transformer', '11/21'),
         ('R', 'Transformer + MLP 잔차 앙상블 2', '11/19'),
         ('C', 'Transformer + MLP 잔차 앙상블 1', '11/13'),
         ('B-MLR', 'Logistic Regression', '11/21'),
         ('T_frozen_sept', 'Frequency table (9월 빈도표)', '11/21'),
         ('B-TAB', 'Smoothed frequency table', '11/21')]
# figure 4 / table 2 contrasts on the 31 games (front minus back; negative = front is better)
PAIRS = [('TOP1', 'B-SEQ'), ('R', 'C'), ('R', 'B-MLR'), ('R', 'B-TAK'), ('R', 'B-GBT'), ('R', 'B-SEQ'), ('R', 'TOP1')]
NAME = {k: lab for k, lab, _ in TABLE}


def combine_am_w(Qs, w, T):
    """experiments/lit_base/final2.py:combine for method 'AM-w' with temperature T (copied, not imported: avoids torch)."""
    q = np.tensordot(np.asarray(w), Qs, axes=1)
    lq = np.log(np.clip(q, 1e-15, 1))
    return softmax(lq / T, axis=1) if T != 1.0 else q


def metrics(q, y):
    F = E.floor(q)
    p = E.pooled(E.row_parts(F, y))
    top3 = float(np.mean((np.argsort(-F, 1)[:, :3] == y[:, None]).any(1)))
    return {'nll': p['nll'], 'top1': p['top1'], 'top3': top3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=REPO / 'data/eps', type=pathlib.Path)
    ap.add_argument('--mix', default=REPO / 'results/final-mix-weights.json', type=pathlib.Path)
    ap.add_argument('--out', default=REPO / 'results/rescore', type=pathlib.Path)
    ap.add_argument('--no-bootstrap', action='store_true')
    a = ap.parse_args()

    with open(a.data / 'rows.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    y = np.array([E.label(r['actual']) for r in rows])
    if any(v is None for v in y):
        raise SystemExit('a row has an actual pitch type outside the 9 groups')
    y = y.astype(np.int64)
    games = np.array([int(r['game_pk']) for r in rows])
    z = np.load(a.data / 'probs.npz')
    Q = {k: z[k].astype(np.float64) for k in z.files}
    mix = json.loads(a.mix.read_text())
    Q['TOP1'] = combine_am_w(np.stack([Q[m] for m in mix['members']]), mix['weights'], mix['temperature'])

    out = {'rows': len(rows), 'games': int(len(set(games.tolist()))), 'filled_uniform': int(sum(r['built'] == '0' for r in rows)),
           'floor_eps': E.EPS, 'mix': {k: mix[k] for k in ('members', 'weights', 'temperature')}, 'table': []}
    for key, label, made in TABLE:
        out['table'].append({'key': key, 'model': label, 'made': made, **metrics(Q[key], y)})
    if not a.no_bootstrap:
        out['bootstrap'] = {'unit': 'game_pk', 'replicates': E.REPLICATES, 'seed': E.SEED}
        out['contrasts'] = []
        for fa, fb in PAIRS:
            d = E.paired_contrast(E.floor(Q[fa]), E.floor(Q[fb]), y, games)
            out['contrasts'].append({'front': fa, 'back': fb, 'label': f'{NAME[fa]} - {NAME[fb]} (31경기)',
                                     'nll': d['point']['nll'], 'ci95': d['ci95']['nll']})

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'table1-rescored.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    with open(a.out / 'table1-rescored.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['model', 'nll', 'top1', 'top3', 'made'])
        for t in out['table']:
            w.writerow([t['model'], f"{t['nll']:.5f}", f"{t['top1']:.4f}", f"{t['top3']:.4f}", t['made']])
    print(f"{out['rows']} rows, {out['games']} games, {out['filled_uniform']} rows filled 1/9")
    for t in out['table']:
        print(f"  {t['model']:<34} NLL {t['nll']:.5f}  top1 {t['top1']:.4f}  top3 {t['top3']:.4f}")
    for c in out.get('contrasts', []):
        print(f"  {c['label']:<60} {c['nll']:+.5f} [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}]")
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
