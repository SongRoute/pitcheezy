"""Convert archived run outputs into the small, flat inputs that scripts/rescore_table.py reads.

The archived outputs (not in this repo; MLB-derived, see NOTICE) are:
  --lit-final DIR     EXP-LIT-BASE-001 final/  -> <model>/eps-pred.npz for B-SEQ, B-GBT, B-TAK, B-MLR, B-TAB
  --final-predict DIR EXP-EXPL-1121-FINAL v4 predict/ -> <model>/eps-pred.npz for X-GBT, X-GRU-A3-plus, X-GRU-A5-b24
                      (+ B-SEQ, B-GBT copies, used only as a cross-check against --lit-final)
  --cr DIR            EXP-SVC-P9-003 CR attempt-001: candidate.jsonl (R and C probabilities), baselines.jsonl
                      (actual pitch type, frozen September frequency table), receipt.json (sha256 of both)

Writes to --out (default data/eps, gitignored):
  rows.csv    one line per scored pitch, in scoring order: game_pk, n, key (= game_pk:at_bat_number:pitch_number),
              built (1 = the models produced a prediction for this pitch), actual (Statcast-style pitch type code)
  probs.npz   one (rows x 9) array per model, columns in PITCH_NAMES order (FF SI FC SL ST CU CH FS OT).
              Rows a model did not predict ("built" = 0) are filled with 1/9, exactly as the original scorers did.

Row rule and fill rule are the ones of experiments/lit_base/final2.py:score and run.py:score. Nothing is re-trained.

    python scripts/export_saved_probs.py --lit-final data/archive/lit-final \
        --final-predict data/archive/final-v4-predict --cr data/archive/CR --out data/eps
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib

import numpy as np

LIT = ('B-SEQ', 'B-GBT', 'B-TAK', 'B-MLR', 'B-TAB')
FINAL = ('X-GBT', 'X-GRU-A3-plus', 'X-GRU-A5-b24')
CHECK = ('B-SEQ', 'B-GBT')
U = np.full(9, 1 / 9)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lit-final', required=True, type=pathlib.Path)
    ap.add_argument('--final-predict', required=True, type=pathlib.Path)
    ap.add_argument('--cr', required=True, type=pathlib.Path)
    ap.add_argument('--out', default=pathlib.Path('data/eps'), type=pathlib.Path)
    a = ap.parse_args()

    receipt = json.loads((a.cr / 'receipt.json').read_text())
    for name in ('candidate', 'baselines'):
        if sha(a.cr / f'{name}.jsonl') != receipt['stage_files_sha256'][name]:
            raise SystemExit(f'{name}.jsonl does not match receipt.json')
    cand = [json.loads(line) for line in open(a.cr / 'candidate.jsonl')]
    base = {(b['game_pk'], b['n']): b for b in map(json.loads, open(a.cr / 'baselines.jsonl'))}
    rows = []
    for c in cand:
        b = base[(c['game_pk'], c['n'])]
        if b['service_key'] != c['service_key'] or (c['cand_C'] is None) != (c['cand_R'] is None):
            raise SystemExit('CR row check failed')
        rows.append((c['game_pk'], c['n'], c['service_key'], int(c['cand_R'] is not None), b['actual']))
    present = np.array([r[3] == 1 for r in rows])

    probs = {'R': np.array([c['cand_R'] if c['cand_R'] is not None else U for c in cand], np.float64),
             'C': np.array([c['cand_C'] if c['cand_C'] is not None else U for c in cand], np.float64),
             'T_frozen_sept': np.array([base[(c['game_pk'], c['n'])]['freq'] for c in cand], np.float64)}

    def load(path):
        z = np.load(path, allow_pickle=False)
        pos = {(int(g), int(n)): i for i, (g, n) in enumerate(zip(z['eps_game_pk'], z['eps_n']))}
        order = np.array([pos[(c['game_pk'], c['n'])] for c in cand])
        ok = (len(pos) == len(cand) and np.array_equal(z['eps_built'][order], present)
              and all(str(z['eps_service_key'][i]) == json.dumps(c['service_key']) for i, c in zip(order, cand)))
        if not ok:
            raise SystemExit(f'rows differ: {path}')
        bi = np.cumsum(z['eps_built']) - 1
        q = np.tile(U, (len(cand), 1)); q[present] = z['q_tau'][bi[order][present]]
        return q, sha(path)

    shas = {}
    for m in LIT:
        probs[m], shas[f'lit-final/{m}'] = load(a.lit_final / m / 'eps-pred.npz')
    for m in FINAL:
        probs[m], shas[f'final-predict/{m}'] = load(a.final_predict / m / 'eps-pred.npz')
    for m in CHECK:
        p = a.final_predict / m / 'eps-pred.npz'
        if p.exists():
            q, shas[f'final-predict/{m}'] = load(p)
            print(f'cross-check {m}: max |final-predict - lit-final| = {np.abs(q - probs[m]).max():.3e}')

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / 'rows.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['game_pk', 'n', 'key', 'built', 'actual']); w.writerows(rows)
    np.savez(a.out / 'probs.npz', **probs)
    (a.out / 'export.json').write_text(json.dumps({'rows': len(rows), 'games': len({r[0] for r in rows}),
                                                   'built': int(present.sum()), 'models': sorted(probs),
                                                   'source_sha256': shas, 'cr_receipt_sha256': sha(a.cr / 'receipt.json')}, indent=1) + '\n')
    print(f'wrote {a.out}: {len(rows)} rows, {len({r[0] for r in rows})} games, {int((~present).sum())} filled 1/9, models {sorted(probs)}')


if __name__ == '__main__':
    main()
