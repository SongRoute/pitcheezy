"""EXP-LIT-BASE-001 기준 모델 4+1가지. 모두 투구 전 정보만 받는다(data.py 정보 집합).

  B-TAB  투수×카운트 빈도표(Dirichlet 계층 평활, gpu14h_prepitch frequency와 같은 식; alpha만 조정)
  B-MLR  다항 로지스틱 회귀(표 특징 + 선택적 투수별 절편), L-BFGS 전체 배치, L2
  B-GBT  LightGBM 다중 분류(표 특징), DEV 조기 종료
  B-SEQ  GRU(같은 투수 같은 경기 앞선 투구 32개) + state11 + 투수·타자 임베딩
  B-TAK  Takamido & Nakamoto(2026)식 Transformer 인코더(앞선 6구, 임베딩 256, 헤드 4, FF 512, 2블록, 맥락 dense 32,
         결합 dense 64) - 표적을 9구종으로 바꾸고, 물리 피처는 R 정보 집합에 없어 빼고, 투수·타자 임베딩을 더했다.

새 기준 모델은 configs 항목 + 이 모듈의 FAMILIES에 함수 하나를 더하면 된다.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import data as D

K9 = D.K9


def softmax_np(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def nll(q, y):
    return float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-300, 1)).mean())


def temper(q, tau):
    return softmax_np(np.log(np.clip(q, 1e-15, 1.0)) / tau)


def fit_temperature(q, y, lo=0.25, hi=4.0, iters=60):
    """1-D golden-section search for tau minimising NLL(q^(1/tau)) on DEV."""
    g = (math.sqrt(5) - 1) / 2; a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(temper(q, c), y), nll(temper(q, d), y)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc; c = b - g * (b - a); fc = nll(temper(q, c), y)
        else:
            a, c, fc = c, d, fd; d = a + g * (b - a); fd = nll(temper(q, d), y)
    return (a + b) / 2


# ---------------------------------------------------------------- B-TAB
def tab_predict(train, evalset, hp):
    te = D.TargetEncoder(hp['alpha']).fit(train['pitcher'], train['batter'], train['balls'], train['strikes'], train['stand'], train['y'])
    return te.transform(evalset['pitcher'], evalset['batter'], evalset['balls'], evalset['strikes'], evalset['stand'])['pc'], {}


# ---------------------------------------------------------------- B-MLR
def mlr_fit_predict(Xtr, ytr, ptr, Xev, pev, hp, seed, threads, log):
    import torch
    torch.set_num_threads(threads); torch.manual_seed(seed)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-6] = 1.0
    xt = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32); yt = torch.as_tensor(ytr, dtype=torch.long)
    d = xt.shape[1]; W = torch.zeros(d, K9, requires_grad=True); b = torch.zeros(K9, requires_grad=True)
    npit = int(max(ptr.max(), pev.max())) + 1 if hp['pitcher_intercepts'] else 0
    P = torch.zeros(npit, K9, requires_grad=True) if npit else None
    pt = torch.as_tensor(ptr, dtype=torch.long)
    params = [W, b] + ([P] if P is not None else [])
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=hp['max_iter'], history_size=20, tolerance_grad=1e-7, tolerance_change=1e-10,
                            line_search_fn='strong_wolfe')
    lam = hp['l2']; lam_p = hp.get('l2_pitcher', lam)
    def closure():
        opt.zero_grad()
        z = xt @ W + b
        if P is not None:
            z = z + P[pt]
        loss = torch.nn.functional.cross_entropy(z, yt) + lam * (W ** 2).sum() + (lam_p * (P ** 2).sum() if P is not None else 0)
        loss.backward(); return loss
    t0 = time.monotonic(); loss0 = float(opt.step(closure))      # LBFGS.step returns the loss at the START
    with torch.no_grad():
        z = xt @ W + b + (P[pt] if P is not None else 0)
        loss = float(torch.nn.functional.cross_entropy(z, yt))
    st = opt.state[opt._params[0]]
    log(f'mlr fit start {loss0:.5f} end CE {loss:.5f} iters {st.get("n_iter")} evals {st.get("func_evals")} {time.monotonic() - t0:.0f}s')
    with torch.no_grad():
        z = torch.as_tensor((Xev - mu) / sd, dtype=torch.float32) @ W + b
        if P is not None:
            z = z + P[torch.as_tensor(pev, dtype=torch.long)]
    return softmax_np(z.double().numpy()), {'train_ce_end': loss, 'lbfgs_iters': st.get('n_iter'), 'func_evals': st.get('func_evals')}


# ---------------------------------------------------------------- B-GBT
def gbt_fit_predict(Xtr, ytr, Xev, hp, seed, threads, log, dev=None, rounds=None):
    import lightgbm as lgb
    params = dict(objective='multiclass', num_class=K9, learning_rate=hp['learning_rate'], num_leaves=hp['num_leaves'],
                  min_data_in_leaf=hp['min_data_in_leaf'], feature_fraction=hp['feature_fraction'], lambda_l2=hp['lambda_l2'],
                  max_bin=hp.get('max_bin', 255), num_threads=threads, seed=seed, deterministic=True, force_row_wise=True, verbose=-1)
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
    t0 = time.monotonic(); info = {}
    if dev is not None:
        dv = lgb.Dataset(dev[0], dev[1], reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=hp['max_rounds'], valid_sets=[dv],
                            callbacks=[lgb.early_stopping(hp['early_stopping_rounds'], verbose=False)])
        info['best_iteration'] = int(booster.best_iteration)
    else:
        booster = lgb.train(params, dtr, num_boost_round=int(rounds))
        info['rounds'] = int(rounds)
    log(f'gbt fit {time.monotonic() - t0:.0f}s {info}')
    q = booster.predict(Xev, num_iteration=info.get('best_iteration') or None)
    return np.asarray(q, np.float64), info


# ---------------------------------------------------------------- B-SEQ / B-TAK (torch)
def build_seq(hp, n_pitcher, n_batter, seed):
    import torch
    nn = torch.nn
    torch.manual_seed(seed)

    class TokEmb(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.t = nn.Embedding(len(D.TYPES), d); self.o = nn.Embedding(11, max(4, d // 4)); self.b = nn.Embedding(4, 4)
            self.s = nn.Embedding(3, 4); self.sm = nn.Embedding(2, 4)
            self.dim = d + max(4, d // 4) + 13

        def forward(self, tok):
            v = tok['valid'].unsqueeze(-1).float()
            e = torch.cat((self.t(tok['type']), self.o(tok['out'].clamp(0, 10)), self.b(tok['balls'].clamp(0, 3)),
                           self.s(tok['strikes'].clamp(0, 2)), self.sm(tok['same'].clamp(0, 1)), v), -1)
            return e * v

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = hp['encoder']; self.window = hp['window']
            self.emb = TokEmb(hp['type_dim'])
            self.pid = nn.Embedding(n_pitcher + 1, hp['pitcher_dim']); self.bid = nn.Embedding(n_batter + 1, hp['batter_dim'])
            if self.kind == 'gru':
                self.enc = nn.GRU(self.emb.dim, hp['hidden'], batch_first=True); seq_out = hp['hidden']
            else:
                dm = hp['d_model']; self.proj = nn.Linear(self.emb.dim, dm); self.posemb = nn.Embedding(self.window, dm)
                layer = nn.TransformerEncoderLayer(dm, hp['heads'], hp['ff'], hp['dropout'], batch_first=True)
                self.enc = nn.TransformerEncoder(layer, hp['blocks'], enable_nested_tensor=False); seq_out = dm
            self.ctx = nn.Sequential(nn.Linear(11, hp['context_dim']), nn.ReLU())
            cat = seq_out + hp['context_dim'] + hp['pitcher_dim'] + hp['batter_dim']
            self.head = nn.Sequential(nn.Dropout(hp['dropout']), nn.Linear(cat, hp['head_hidden']), nn.ReLU(), nn.Dropout(hp['dropout']),
                                      nn.Linear(hp['head_hidden'], K9))

        def forward(self, x, pi, bi, tok):
            w = self.window
            t = {k: v[:, :w] for k, v in tok.items()}
            e = self.emb(t)                                   # (B, w, d); slot 0 = most recent
            if self.kind == 'gru':
                _, h = self.enc(torch.flip(e, dims=[1])); s = h[-1]    # oldest -> newest, last step = most recent
            else:
                h = self.proj(e) + self.posemb.weight[None, :w]
                pad = ~t['valid']; allpad = pad.all(1)
                pad = pad.clone(); pad[allpad, 0] = False             # keep one slot so attention is defined
                h = self.enc(h, src_key_padding_mask=pad)
                m = (~pad).unsqueeze(-1).float(); s = (h * m).sum(1) / m.sum(1).clamp(min=1)
                s = s * (~allpad).unsqueeze(-1).float()
            return self.head(torch.cat((s, self.ctx(x), self.pid(pi), self.bid(bi)), 1))

    return Net()


class SeqData:
    def __init__(self, torch, device, state11, pidx, bidx, tok, y=None):
        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)  # noqa: E731
        self.x = t(state11, torch.float32); self.p = t(pidx, torch.long); self.b = t(bidx, torch.long)
        self.tok = {k: t(v, torch.bool if k == 'valid' else torch.long) for k, v in tok.items()}
        self.y = None if y is None else t(y, torch.long)

    def batch(self, idx):
        return self.x[idx], self.p[idx], self.b[idx], {k: v[idx] for k, v in self.tok.items()}


def seq_predict(torch, model, data, rows, bs):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            idx = rows[s:s + bs]; z = model(*data.batch(idx))
            out.append(torch.softmax(z.float(), 1).cpu().double().numpy())
    q = np.concatenate(out) if out else np.zeros((0, K9))
    return q / q.sum(1, keepdims=True)


def seq_fit_predict(tr, ev, hp, seed, device, threads, log, epochs=None, dev=None, max_seconds=None):
    """tr/ev: dict(state11, pidx, bidx, tok, y). dev given -> early stopping on dev NLL; returns the best-epoch DEV predictions
    (ev is ignored). dev None -> train exactly `epochs` epochs and predict ev."""
    import torch
    torch.set_num_threads(threads); torch.manual_seed(seed); np.random.seed(seed)
    dev_t = torch.device(device)
    if dev_t.type == 'mps':
        torch.mps.manual_seed(seed)
    n_p = int(tr['pidx'].max()); n_b = int(tr['bidx'].max())   # vocab = training rows; unknown = 0
    model = build_seq(hp, n_p, n_b, seed).to(dev_t)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    T = SeqData(torch, dev_t, tr['state11'], tr['pidx'], tr['bidx'], tr['tok'], tr['y'])
    V = SeqData(torch, dev_t, ev['state11'], ev['pidx'], ev['bidx'], ev['tok']) if dev is None else None
    G = SeqData(torch, dev_t, dev['state11'], dev['pidx'], dev['bidx'], dev['tok']) if dev is not None else None
    rows = np.arange(len(tr['y'])); bs = hp['batch_size']; pdrop = hp['id_dropout']
    gen = torch.Generator().manual_seed(seed)
    max_ep = hp['max_epochs'] if epochs is None else int(epochs)
    best = (math.inf, 0, None); trace = []; bad = 0; t0 = time.monotonic()
    for ep in range(1, max_ep + 1):
        model.train(); order = torch.randperm(len(rows), generator=gen).numpy(); tot = 0.0
        for s in range(0, len(order), bs):
            idx = torch.as_tensor(order[s:s + bs], device=dev_t)
            x, p, b, tok = T.batch(idx)
            if pdrop > 0:
                p = torch.where(torch.rand(p.shape, device=dev_t) < pdrop, torch.zeros_like(p), p)
                b = torch.where(torch.rand(b.shape, device=dev_t) < pdrop, torch.zeros_like(b), b)
            loss = torch.nn.functional.cross_entropy(model(x, p, b, tok), T.y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        rec = {'epoch': ep, 'train_loss': tot / len(rows), 'seconds': time.monotonic() - t0}
        if G is not None:
            qd = seq_predict(torch, model, G, torch.arange(len(dev['y']), device=dev_t), 8192); dn = nll(qd, dev['y']); rec['dev_nll'] = dn
            if dn < best[0] - 1e-5:
                best = (dn, ep, qd); bad = 0
            else:
                bad += 1
        trace.append(rec); log(f'seq ep {ep} {rec}')
        if G is not None and bad >= hp['patience']:
            break
        if max_seconds and time.monotonic() - t0 > max_seconds:
            log('seq wall budget reached'); rec['stopped'] = 'wall_budget'; break
    if G is not None:
        return best[2], {'best_epoch': best[1], 'best_dev_nll': best[0], 'trace': trace}
    q = seq_predict(torch, model, V, torch.arange(len(ev['pidx']), device=dev_t), 8192)
    return q, {'epochs': max_ep, 'trace': trace}
