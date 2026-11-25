"""EXP-LIT-BASE-001 자료: R과 같은 학습 package(9월 package, full-season fold)와 같은 E_PS 입력.

정보 집합은 R 성분과 같다: state11(카운트·아웃·이닝·점수차·초말·주자 3·타자 좌우·투수 좌우), 투수 ID, 타자 ID,
같은 경기·같은 투수의 엄격히 앞선 투구 최대 32개(구종 19코드, 결과 10, 볼, 스트라이크, 같은 타자 여부).
현재 투구의 물리량·결과·위치는 어느 기준 모델에도 들어가지 않는다(투구 후 정보 차단).

분할(결과 열람 전 고정, configs/EXP-LIT-BASE-001.yaml splits):
  tune  : 학습 = FIT_tr(2026-03-25..08-31), 평가 = DEV(2026-09-01..09-27)   -> 초매개변수·epoch·온도·최고 기준 모델 선정
  final : 학습 = FIT 전체(2026-03-25..09-27, R과 같은 창), 예측 = E_PS 입력(라벨 없음)
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

K9 = 9
CONTEXT = 32
TYPES = ('UNKNOWN', 'CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'PO', 'SC', 'SI', 'SL', 'ST', 'SV', 'UN')
CLASSES = ('FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS', 'OT')
GROUPS = {'FF': ('FF', 'FA'), 'SI': ('SI',), 'FC': ('FC',), 'SL': ('SL', 'SV'), 'ST': ('ST',), 'CU': ('CU', 'KC', 'CS'),
          'CH': ('CH',), 'FS': ('FS', 'FO'), 'OT': ('KN', 'SC', 'EP')}
# 19코드 -> 9묶음(-1 = 묶음 밖: UNKNOWN, PO, UN). experiments/gpu14h_prepitch/train.py:targets와 같은 표.
TYPE9 = np.array([next((i for i, g in enumerate(CLASSES) if t in GROUPS[g]), -1) for t in TYPES], np.int64)


class Stop(RuntimeError):
    pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def need(c, what):
    if not c:
        raise Stop(f'STOP {what}')


# ---------------------------------------------------------------- package (FIT side)
def load_package(cfg, check_sha=True, day_slice=None):
    """cfg['data']['package'] -> dict of numpy arrays for every package row (FS-001 full-season fold: every row is FIT)."""
    d = cfg['data']['package']; root = pathlib.Path(d['root'])
    if not root.is_absolute():
        root = pathlib.Path(__file__).resolve().parents[2] / root
    if check_sha:
        for name, want in d['sha256'].items():
            need(sha256_file(root / name) == want, f'package file sha {name}')
    c = np.load(root / 'corpus-subset.npz', allow_pickle=False)
    s = np.load(root / 'seq-inputs.npz', allow_pickle=False)
    date = np.load(root / 'game-date.npy', allow_pickle=False).astype('datetime64[D]')
    x = np.load(root / 'x-prequential.npy', mmap_mode='r')
    n = len(date)
    need(n == d['rows'] and x.shape == (n, 199), 'package rows')
    need(not c['tuning'].any() and not c['calibration'].any() and (c['fit'] | c['evaluation']).all(), 'package roles')
    keys = c['keys']
    from experiments.ws_fullseason.data import key_digest
    need(key_digest(keys) == d['all_keys_sha256'], 'package keys sha (FS-001 FIT_keys_sha256)')
    w = [np.datetime64(v) for v in cfg['splits']['fit_window']]
    need(date.min() >= w[0] and date.max() <= w[1], 'package dates inside FIT window')
    banned = set(cfg['data']['excluded_game_pks'])
    need(not np.isin(s['game_pk'], list(banned)).any(), 'package has no E_PS / 849799 / 9-29~30 game')
    sel = np.ones(n, bool) if day_slice is None else np.isin(date, np.array(day_slice, 'datetime64[D]'))
    out = dict(keys=keys[sel], date=date[sel], state11=np.ascontiguousarray(x[sel, :11], dtype=np.float32),
               game_pk=s['game_pk'][sel].astype(np.int64), at_bat=s['at_bat_number'][sel].astype(np.int64),
               pitch_no=s['pitch_number'][sel].astype(np.int64), pitcher=s['pitcher'][sel].astype(np.int64),
               batter=s['batter'][sel].astype(np.int64), type19=s['pitch_type'][sel].astype(np.int64),
               out10=s['outcome10'][sel].astype(np.int64), balls=s['balls'][sel].astype(np.int64), strikes=s['strikes'][sel].astype(np.int64))
    out['y'] = TYPE9[out['type19']]
    from experiments.ws_seq.train_seq import build_context_index, check_context
    ctx = build_context_index(out['pitcher'], out['game_pk'], out['at_bat'], out['pitch_no'], CONTEXT)
    check_context(ctx, out['pitcher'], out['game_pk'], out['at_bat'], out['pitch_no'])
    v = ctx >= 0; src = np.maximum(ctx, 0)
    out['tok'] = tokens_from_ctx(out, ctx)
    return out


def tokens_from_ctx(out, ctx):
    v = ctx >= 0; src = np.maximum(ctx, 0); i16 = lambda a: a.astype(np.int16)  # noqa: E731
    return dict(valid=v, type=i16(np.where(v, out['type19'][src], 0)), out=i16(np.where(v, out['out10'][src], 0)),
                balls=i16(np.where(v, out['balls'][src], 0)), strikes=i16(np.where(v, out['strikes'][src], 0)),
                same=i16(np.where(v, (out['batter'][src] == out['batter'][:, None]), 0)))


def resolve(p):
    p = pathlib.Path(p)
    return p if p.is_absolute() else pathlib.Path(__file__).resolve().parents[2] / p


def load_hist(cfg, pk, day_slice=None):
    """Prepend 2023-2025 regular-season rows (scripts/lit_base/build_hist.py, parity-gated) to package rows `pk`
    (before tokens). Context index is rebuilt on the union; game_pk never repeats across seasons (checked)."""
    h = cfg['data']['hist']; root = pathlib.Path(h['root'])
    if not root.is_absolute():
        root = pathlib.Path(__file__).resolve().parents[2] / root
    parts = []
    for season, want in h['sha256'].items():
        f = root / f'hist-{season}.npz'
        need(sha256_file(f) == want, f'hist sha {season}')
        z = np.load(f, allow_pickle=False)
        part = {k: z[k] for k in ('keys', 'date', 'state11', 'game_pk', 'at_bat', 'pitch_no', 'pitcher', 'batter', 'type19', 'out10', 'balls', 'strikes')}
        if day_slice is not None:
            m = np.isin(part['date'], np.array(day_slice)); part = {k: v[m] for k, v in part.items()}
        parts.append(part)
    keys = [k for k in pk if k not in ('tok', 'y', 'is_hist')]
    out = {k: np.concatenate([p[k] if k != 'date' else p[k].astype('datetime64[D]') for p in parts] + [pk[k]]) for k in keys}
    need(len(np.unique(out['keys'], axis=0)) == len(out['keys']), 'hist+package unique keys')
    need(max(np.datetime64(f'{s}-12-31') for s in h['sha256']) < pk['date'].min(), 'hist seasons strictly before 2026')
    out['y'] = TYPE9[out['type19']]
    from experiments.ws_seq.train_seq import build_context_index, check_context
    ctx = build_context_index(out['pitcher'], out['game_pk'], out['at_bat'], out['pitch_no'], CONTEXT)
    check_context(ctx, out['pitcher'], out['game_pk'], out['at_bat'], out['pitch_no'])
    out['tok'] = tokens_from_ctx(out, ctx); out['is_hist'] = np.arange(len(out['y'])) < sum(len(p['keys']) for p in parts)
    return out


def raw_from_state(state11, encoder_path):
    """state11 -> (balls, strikes, stand_L, throws_L) with the pinned Encoder11 (inverse of (raw-mean)/scale)."""
    e = json.loads(pathlib.Path(encoder_path).read_text())
    mean = np.asarray(e['mean'][:11]); scale = np.asarray(e['scale'][:11])
    raw = state11.astype(np.float64) * scale + mean
    balls = np.rint(raw[:, 0] * 3).astype(np.int64); strikes = np.rint(raw[:, 1] * 2).astype(np.int64)
    return balls, strikes, np.rint(raw[:, 9]).astype(np.int64), np.rint(raw[:, 10]).astype(np.int64)


# ---------------------------------------------------------------- E_PS side (inputs only, no labels)
def load_eps(cfg):
    p = pathlib.Path(cfg['data']['eps_inputs']['path'])
    rep = json.loads((p.parent / 'eps-inputs.report.json').read_text())
    need(sha256_file(p) == rep['npz_sha256'], 'eps-inputs npz sha == report')
    want = cfg['data']['eps_inputs'].get('sha256')
    if want:
        need(rep['npz_sha256'] == want, 'eps-inputs sha == registration pin')
    need(rep['labels_read'] is False and rep['rows'] == cfg['data']['eps_inputs']['rows'] and rep['built'] == cfg['data']['eps_inputs']['built'], 'eps-inputs report')
    z = np.load(p, allow_pickle=False)
    b = z['built']
    out = dict(game_pk=z['game_pk'], n=z['n'], service_key=z['service_key'], w1_key=z['w1_key'], built=b,
               state11=np.nan_to_num(z['state11']).astype(np.float32), pitcher=z['pitcher'], batter=z['batter'],
               tok={k: z[f'tok_{k}'] for k in ('valid', 'type', 'out', 'balls', 'strikes', 'same')})
    out['balls'] = np.rint(np.nan_to_num(z['raw11'][:, 0]) * 3).astype(np.int64)
    out['strikes'] = np.rint(np.nan_to_num(z['raw11'][:, 1]) * 2).astype(np.int64)
    return out


# ---------------------------------------------------------------- target encodings (fit rows only)
def _group_counts(keys, y, k=K9):
    u, inv = np.unique(keys, return_inverse=True)
    cnt = np.zeros((len(u), k)); np.add.at(cnt, (inv, y), 1.0)
    return u, cnt


def _lookup(u, table, keys, fallback):
    pos = np.searchsorted(u, keys); pos = np.minimum(pos, max(len(u) - 1, 0))
    hit = (len(u) > 0) & (u[pos] == keys) if len(u) else np.zeros(len(keys), bool)
    out = np.array(fallback, copy=True)
    out[hit] = table[pos[hit]]
    return out, hit


class TargetEncoder:
    """Dirichlet-smoothed hierarchical mixes (gpu14h_prepitch frequency style): global -> pitcher -> pitcher×count,
    pitcher -> pitcher×stand, global -> batter. Fit on training rows with a 9-group label only."""

    def __init__(self, alpha):
        self.alpha = float(alpha)

    def fit(self, pitcher, batter, balls, strikes, stand, y):
        m = y >= 0; a = self.alpha
        p, b, bl, st, sd, yy = pitcher[m], batter[m], balls[m], strikes[m], stand[m], y[m]
        g = np.bincount(yy, minlength=K9).astype(float); self.glob = (g + a / K9) / (g.sum() + a)
        self.pu, self.pc = _group_counts(p, yy)
        self.pcu, self.pcc = _group_counts(p * 100 + bl * 10 + st, yy)
        self.psu, self.psc = _group_counts(p * 10 + sd, yy)
        self.bu, self.bc = _group_counts(b, yy)
        return self

    def transform(self, pitcher, batter, balls, strikes, stand):
        a = self.alpha; n = len(pitcher); G = np.broadcast_to(self.glob, (n, K9))
        cp, hp = _lookup(self.pu, self.pc, pitcher, np.zeros((n, K9)))
        pp = (cp + a * G) / (cp.sum(1, keepdims=True) + a)
        cc, _ = _lookup(self.pcu, self.pcc, pitcher * 100 + balls * 10 + strikes, np.zeros((n, K9)))
        pc = (cc + a * pp) / (cc.sum(1, keepdims=True) + a)
        cs, _ = _lookup(self.psu, self.psc, pitcher * 10 + stand, np.zeros((n, K9)))
        ps = (cs + a * pp) / (cs.sum(1, keepdims=True) + a)
        cb, _ = _lookup(self.bu, self.bc, batter, np.zeros((n, K9)))
        pb = (cb + a * G) / (cb.sum(1, keepdims=True) + a)
        expo = np.log1p(cp.sum(1))
        return dict(p=pp, pc=pc, ps=ps, b=pb, exposure=expo)


def oof_encodings(alpha, folds, game_pk, pitcher, batter, balls, strikes, stand, y):
    """Out-of-fold encodings for training rows (folds by game_pk % folds), so a row never sees its own label."""
    parts = None; fold = game_pk % folds
    for k in range(folds):
        te = TargetEncoder(alpha).fit(*(a[fold != k] for a in (pitcher, batter, balls, strikes, stand, y)))
        sel = fold == k
        enc = te.transform(pitcher[sel], batter[sel], balls[sel], strikes[sel], stand[sel])
        if parts is None:
            parts = {key: np.zeros((len(y),) + v.shape[1:]) for key, v in enc.items()}
        for key, v in enc.items():
            parts[key][sel] = v
    return parts


# ---------------------------------------------------------------- tabular features (MLR / GBT)
def token_features(tok):
    v = tok['valid']; t9 = np.where(v, TYPE9[tok['type']], -2)        # -1 = outside 9 groups, -2 = no token
    n = len(v); f = {}
    def onehot(col, k, offset):
        o = np.zeros((n, k)); o[np.arange(n), col + offset] = 1.0; return o
    f['last1_type'] = onehot(t9[:, 0], 11, 2); f['last2_type'] = onehot(t9[:, 1], 11, 2)
    f['last1_out'] = onehot(np.where(v[:, 0], tok['out'][:, 0], 10), 11, 0)
    f['last1_same'] = (v[:, 0] & (tok['same'][:, 0] == 1)).astype(float)[:, None]
    npv = v.sum(1); f['n_prev'] = (npv / CONTEXT)[:, None]
    mix = np.zeros((n, K9))
    for j in range(CONTEXT):
        m = t9[:, j] >= 0; mix[np.flatnonzero(m), t9[m, j]] += 1
    f['game_mix'] = mix / np.maximum(npv, 1)[:, None]
    same = v & (tok['same'] == 1)
    pa_len = np.argmin(np.column_stack((same, np.zeros(n, bool))), axis=1)   # consecutive same-batter tokens from the most recent
    pam = np.zeros((n, K9))
    for j in range(CONTEXT):
        m = (j < pa_len) & (t9[:, j] >= 0); pam[np.flatnonzero(m), t9[m, j]] += 1
    f['pa_n'] = (pa_len / 10.0)[:, None]; f['pa_mix'] = pam / np.maximum(pa_len, 1)[:, None]
    return f


def tabular(state11, balls, strikes, enc, tok):
    n = len(state11)
    cnt = np.zeros((n, 12)); cnt[np.arange(n), np.clip(balls, 0, 3) * 3 + np.clip(strikes, 0, 2)] = 1.0
    lg = lambda q: np.log(np.clip(q, 1e-6, 1.0))  # noqa: E731
    tf = token_features(tok)
    cols = [state11.astype(np.float64), cnt, lg(enc['p']), lg(enc['pc']), lg(enc['ps']), lg(enc['b']), enc['exposure'][:, None]]
    cols += [tf[k] for k in ('last1_type', 'last2_type', 'last1_out', 'last1_same', 'n_prev', 'game_mix', 'pa_n', 'pa_mix')]
    X = np.concatenate(cols, axis=1).astype(np.float32)
    need(np.isfinite(X).all(), 'tabular features finite')
    return X


def vocab(ids_fit):
    return np.unique(ids_fit)


def index(ids, vocabulary):
    pos = np.searchsorted(vocabulary, ids); pos = np.minimum(pos, len(vocabulary) - 1)
    return np.where(vocabulary[pos] == ids, pos + 1, 0).astype(np.int64)   # 0 = unknown
