"""EXP-EXPL-1121 팔 X-DATA용 과거 시즌(2023~2025 정규시즌) 입력을 R package와 같은 형식으로 만든다.

같은 projection(abc_corpus.project, row_policy = configs/EXP-WS-FULLSEASON-T-OOF3.json, labels10 = scale_safe.outcome_labels)을
시즌 날짜 창만 바꿔 적용하고, state11 = live_inputs.raw11 식 + 9월 package Encoder11(median fill, clip ±8).

동등성 게이트(먼저 실행, 실패하면 팔을 뺀다): 같은 코드로 2026 parquet(d20260930-h2026f)을 projection해서
9월 package와 키·투구 유형 19코드·결과 10·볼·스트라이크·투수·타자·state11(x[:, :11])이 모두 같아야 한다.

usage: python -B scripts/lit_base/build_hist.py --out DIR [--parity-only]
"""
from __future__ import annotations

import os
import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments/ws_campaign/source'))
DRIVE = pathlib.Path(os.path.expandvars('${PITCHEEZY_DATA_DIR}/raw'))
SOURCES = {
    2023: (DRIVE / 'd20260911-s2325/statcast_2023.parquet', '46c354beae196ce2d501fe989c491725f2ecc47601f58d4b8b0234b982984730', '2023-03-30', '2023-10-02'),
    2024: (DRIVE / 'd20260911-s2325/statcast_2024.parquet', '3cf05edddd22a2eac3d55be11cc56e00fe56f7fcf75823a8ecc8fa89e7e9a274', '2024-03-20', '2024-09-30'),
    2025: (DRIVE / 'd20260911-s2325/statcast_2025.parquet', '4c785756d8901f8c46c70891954a4c54b8ad1c6909db67f4a000221691dc0249', '2025-03-18', '2025-09-28'),
    2026: (DRIVE / 'd20260930-h2026f/holdout_2026/statcast_2026.parquet', 'c0c1eda4b515fc14dab79defe577b449e2695d40dd036f5c35ae52e37093ed9b', '2026-03-25', '2026-09-30'),
}
PKG = pathlib.Path(os.path.expandvars('${PITCHEEZY_JOBS}/20261016-oof-completion-003/research/oof3-prepare-attempt-001/inputs'))
TYPES = ('UNKNOWN', 'CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'PO', 'SC', 'SI', 'SL', 'ST', 'SV', 'UN')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def project_season(season):
    import abc_corpus as cp
    import pyarrow.parquet as pq
    import scale_safe as safe
    path, want, start, end = SOURCES[season]
    if sha(path) != want:
        raise SystemExit(f'STOP parquet sha {season}')
    rp = dict(json.loads((REPO / 'configs/EXP-WS-FULLSEASON-T-OOF3.json').read_text())['row_policy'])
    rp['start'], rp['end'] = start, end
    raw = pq.read_table(path, columns=cp.PROJECT).to_pandas()
    g, census = cp.project(raw, {'row_policy': rp}, safe.outcome_labels, safe.head_labels)
    return g, census


def arrays(g):
    e = json.loads((PKG / 'encoder.json').read_text())
    med, mean, scale = (np.asarray(e[k][:11], np.float64) for k in ('median', 'mean', 'scale'))
    top = (g.inning_topbot == 'Top').to_numpy()
    lead = np.where(top, 1.0, -1.0) * (g.home_score.to_numpy(float) - g.away_score.to_numpy(float))
    raw = np.column_stack([g.balls.to_numpy(float) / 3, g.strikes.to_numpy(float) / 2, g.outs_when_up.to_numpy(float) / 2,
                           g.inning.to_numpy(float) / 9, lead / 5, top.astype(float), g.on_1b.notna().to_numpy(float),
                           g.on_2b.notna().to_numpy(float), g.on_3b.notna().to_numpy(float), (g.stand == 'L').to_numpy(float),
                           (g.p_throws == 'L').to_numpy(float)])
    raw = np.where(np.isfinite(raw), raw, med)
    st = np.clip((raw - mean) / scale, -8, 8).astype(np.float32)
    ti = {t: i for i, t in enumerate(TYPES)}
    return dict(keys=g[['game_pk', 'at_bat_number', 'pitch_number']].to_numpy(np.int64), date=g.game_date.dt.strftime('%Y-%m-%d').to_numpy().astype('U10'),
                state11=st, game_pk=g.game_pk.to_numpy(np.int64), at_bat=g.at_bat_number.to_numpy(np.int64), pitch_no=g.pitch_number.to_numpy(np.int64),
                pitcher=g.pitcher.to_numpy(np.int64), batter=g.batter.to_numpy(np.int64), type19=np.array([ti[t] for t in g.pitch_type], np.int64),
                out10=g.outcome10.to_numpy(np.int64), balls=g.balls.to_numpy(np.int64), strikes=g.strikes.to_numpy(np.int64))


def parity():
    g, _ = project_season(2026)
    a = arrays(g)
    c = np.load(PKG / 'corpus-subset.npz'); s = np.load(PKG / 'seq-inputs.npz'); x = np.load(PKG / 'x-prequential.npy', mmap_mode='r')
    pos = {tuple(k): i for i, k in enumerate(a['keys'])}
    idx = np.array([pos.get(tuple(k), -1) for k in c['keys']])
    out = {'package_rows': int(len(idx)), 'missing_in_projection': int((idx < 0).sum()), 'projection_rows': int(len(a['keys']))}
    ok = idx >= 0; j = idx[ok]
    for name, pk in (('pitcher', s['pitcher']), ('batter', s['batter']), ('type19', s['pitch_type']), ('out10', s['outcome10']),
                     ('balls', s['balls']), ('strikes', s['strikes'])):
        out[f'{name}_mismatch'] = int((a[name][j] != np.asarray(pk)[ok]).sum())
    out['state11_maxabs'] = float(np.abs(a['state11'][j] - np.asarray(x[:, :11])[ok]).max())
    out['pass'] = out['missing_in_projection'] == 0 and all(v == 0 for k, v in out.items() if k.endswith('_mismatch')) and out['state11_maxabs'] <= 1e-6
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=pathlib.Path, required=True); ap.add_argument('--parity-only', action='store_true')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    p = parity(); print('parity', json.dumps(p), flush=True)
    rep = {'parity_2026': p, 'labels_read': 'regular-season 2023-2025 training labels only; no 2026 postseason', 'seasons': {}}
    if p['pass'] and not a.parity_only:
        for season in (2023, 2024, 2025):
            g, census = project_season(season); arr = arrays(g)
            f = a.out / f'hist-{season}.npz'; np.savez(f, **arr)
            rep['seasons'][season] = {'rows': int(len(g)), 'dates': [min(arr['date'].tolist()), max(arr['date'].tolist())], 'npz_sha256': sha(f),
                                      'excluded': census['excluded'], 'source_sha256': SOURCES[season][1]}
            print(season, rep['seasons'][season]['rows'], flush=True)
    rep['seconds'] = time.time() - t0; rep['script_sha256'] = sha(__file__)
    (a.out / 'hist.report.json').write_text(json.dumps(rep, indent=1, default=str) + '\n')
    if not p['pass']:
        raise SystemExit('PARITY FAILED: X-DATA arm is dropped')


if __name__ == '__main__':
    main()
