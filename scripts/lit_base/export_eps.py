"""EXP-LIT-BASE-001: E_PS 모델 입력 내보내기 (라벨·모델 없음).

scripts/svc_p9/seed_score.stage_components와 같은 루프(W1 v3, 같은 생략 규칙)로 E_PS 31경기를 다시 재생해, R 성분이
받은 것과 똑같은 입력(state11, raw11, pitcher, batter, 이전 토큰 32칸)을 행마다 저장한다. 모델을 적재하지 않고,
현재 투구의 실제 구종(라벨)을 읽지 않는다(토큰은 같은 경기에서 앞선 투구만).

게이트: 저장한 행 키(game_pk, n, service_key, w1_key)와 built(생략 여부)가 EXP-SVC-P9-003-SEED B2 components.npz와
완전히 같아야 한다(R이 채점된 8,943행·생략 275행과 같은 행 집합).

usage: .venv/bin/python -B scripts/lit_base/export_eps.py --out DIR   (새 프로세스, CPU)
"""
from __future__ import annotations

import os
import argparse
import hashlib
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'scripts/svc_p9'))
import seed_score as S  # noqa: E402  (x3_run -> x1_run sets sys.path: scripts, src, observer backend)

X3, X, E = S.X3, S.X, S.E
SEED_B2 = pathlib.Path(os.path.expandvars('${PITCHEEZY_RUNS}/EXP-SVC-P9-003-SEED/score/B2/components.npz'))
TOKEN_FIELDS = ('valid', 'type', 'out', 'balls', 'strikes', 'same', 'pos')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=pathlib.Path, required=True); a = ap.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from pitcheezy.svc_p9 import predictor as P
    P.activate_snapshot()
    if S.sha256_file(X.FACTS) != X.FACTS_SHA256:
        raise SystemExit('registration-facts receipt differs from the code pin')
    facts_doc = json.loads(X.FACTS.read_text())
    games = E.e_ps_games(X.FACTS); E.refuse_demo_game(games)
    entries = X.inputs(facts_doc, games)
    from pitcheezy.svc_p9 import live_inputs as L
    import demo_precompute as D
    P.check_loaded_modules()
    encoder = L.Encoder11.load(L.encoder_pin(X3.X1_CONFIG))
    t0 = time.perf_counter()
    keys, built, reasons = [], [], []
    st, raw, pit, bat = [], [], [], []
    tok = {k: [] for k in TOKEN_FIELDS}
    Z = L.CONTEXT
    for entry in entries:
        final = X.read_verified(entry)
        conv = L.LiveInputs(entry['game_pk'], final['gameData']['datetime']['officialDate'], encoder, version=X3.W1_VERSION)
        changes = X.batter_changes(final)
        rows_final = D.statcast_rows(final).to_dict('records') if changes else None
        for n, snap in D.replay_snapshots(final):
            row = D.live_rows(snap).iloc[-1]
            x = conv.convert(X.live_shape(snap, row))
            reason = X.batter_change_omit(changes, rows_final, n)
            if reason and not isinstance(x, L.Omit):
                x = L.Omit(reason, x.pitch_key)
            keys.append((entry['game_pk'], n, D._key(row), x.pitch_key))
            if isinstance(x, L.Omit):
                built.append(False); reasons.append(x.reason)
                st.append(np.full(11, np.nan, np.float32)); raw.append(np.full(11, np.nan)); pit.append(0); bat.append(0)
                for k in TOKEN_FIELDS:
                    tok[k].append(np.zeros(Z, bool if k == 'valid' else np.int64))
                continue
            built.append(True); reasons.append('')
            st.append(np.asarray(x.state11, np.float32)); raw.append(np.asarray(x.raw11, np.float64))
            pit.append(int(x.pitcher)); bat.append(int(x.batter))
            for k in TOKEN_FIELDS:
                tok[k].append(np.asarray(x.tokens[k]))
    out = a.out; out.mkdir(parents=True, exist_ok=False)
    arr = dict(game_pk=np.array([k[0] for k in keys], np.int64), n=np.array([k[1] for k in keys], np.int64),
               service_key=np.array([json.dumps(k[2]) for k in keys]), w1_key=np.array([json.dumps(k[3]) for k in keys]),
               built=np.array(built), omit_reason=np.array(reasons), state11=np.stack(st), raw11=np.stack(raw),
               pitcher=np.array(pit, np.int64), batter=np.array(bat, np.int64),
               **{f'tok_{k}': np.stack(v) for k, v in tok.items()})
    # gate: same rows / omitted rows as the R components (B2)
    z = np.load(SEED_B2, allow_pickle=False)
    gate = {f: bool(np.array_equal(arr[f], z[f])) for f in ('game_pk', 'n', 'service_key', 'w1_key', 'built')}
    gate['b2_pitcher_equal_on_built'] = bool(np.array_equal(arr['pitcher'][arr['built']], z['pitcher'][z['built']]))
    gate['state11_encode_raw11_maxabs'] = float(np.abs(encoder.encode(arr['raw11'][arr['built']]) - arr['state11'][arr['built']]).max())
    if not all(v for k, v in gate.items() if k != 'state11_encode_raw11_maxabs') or gate['state11_encode_raw11_maxabs'] > 0:
        raise SystemExit(f'FAILED gate {gate}')
    np.savez(out / 'eps-inputs.npz', **arr)
    report = {'rows': len(keys), 'built': int(sum(built)), 'omitted': int(len(built) - sum(built)), 'games': len(games),
              'gate_vs_seed_B2': gate, 'seed_B2_sha256': sha(SEED_B2), 'labels_read': False, 'models_loaded': 0,
              'seconds': time.perf_counter() - t0, 'commit': X._git()[0], 'dirty': X._git()[1],
              'script_sha256': sha(__file__), 'npz_sha256': sha(out / 'eps-inputs.npz')}
    (out / 'eps-inputs.report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
