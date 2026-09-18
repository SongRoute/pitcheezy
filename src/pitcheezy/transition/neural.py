"""전이 모델 ⓑ — 공유 신경망 + 투수 임베딩 (design.md ② ⓑ, D24). 산출은 count.fit 과 같은 TransitionTensor.

입력 = 범주 임베딩 [카운트 12, 주자아웃 24, 타자 군집 K, 구종 9, 위치 25] + 투수 임베딩 (+ 선택: 투수×구종 임베딩)
      → MLP → 결과 11 로짓. 규칙 마스크(카운트별 불허 결과)는 로짓에서 −inf 로 막는다.
모든 투수가 몸통(MLP)을 공유하고, 투수 고유 정보는 임베딩 벡터에만 들어간다. pitcher_dim = 0 이면 리그 공통 모델(대조군).
에폭 수는 2023–24 학습 → 2025 홀드아웃 NLL 로 고른다 (count 모델의 α 스크리닝과 같은 자리, 2026 은 쓰지 않음).
valid·n_obs 는 count 모델과 같은 규칙. 난수가 있으므로 시드마다 텐서가 다르다.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces import states as S
from pitcheezy.interfaces.grid import N_ACTIONS, N_LOC, decode_action
from pitcheezy.interfaces.pitch_types import N_PITCH
from pitcheezy.interfaces.tensor import DEFAULT_REPERTOIRE_MIN_PITCHES, DEFAULT_ROW_SUM_TOL, SPEC_VERSION, TransitionTensor
from pitcheezy.transition.count import count_transitions, repertoire_counts

log = logging.getLogger("pitcheezy")

HP_DEFAULTS = {
    "pitcher_dim": 16, "pitcher_pitch_dim": 0, "cat_dim": 8, "hidden": 256, "n_layers": 2, "dropout": 0.0,
    "lr": 2e-3, "weight_decay": 1e-4, "embed_l2": 0.0, "batch_size": 4096, "max_epochs": 30, "patience": 3, "device": "auto",
}


def hparams(cfg: dict | None) -> dict:
    hp = {**HP_DEFAULTS, **(cfg or {})}
    unknown = set(hp) - set(HP_DEFAULTS)
    if unknown:
        raise ValueError(f"neural 하이퍼파라미터 모름: {sorted(unknown)}")
    return hp


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


class SharedNet(nn.Module):
    def __init__(self, n_pitchers: int, K: int, hp: dict):
        super().__init__()
        d = int(hp["cat_dim"])
        self.e_count = nn.Embedding(S.N_COUNTS, d)
        self.e_bo = nn.Embedding(S.N_BASE_OUT, d)
        self.e_cl = nn.Embedding(K, d)
        self.e_pitch = nn.Embedding(N_PITCH, d)
        self.e_loc = nn.Embedding(N_LOC, d)
        self.dp, self.dpp = int(hp["pitcher_dim"]), int(hp["pitcher_pitch_dim"])
        self.e_p = nn.Embedding(n_pitchers, self.dp) if self.dp else None
        self.e_pp = nn.Embedding(n_pitchers * N_PITCH, self.dpp) if self.dpp else None
        for e in (self.e_p, self.e_pp):  # 0 에서 출발 = 리그 공통 모델에서 출발, 데이터가 있을 때만 벌어진다
            if e is not None:
                nn.init.zeros_(e.weight)
        layers, w = [], 5 * d + self.dp + self.dpp
        for _ in range(int(hp["n_layers"])):
            layers += [nn.Linear(w, int(hp["hidden"])), nn.GELU()]
            if hp["dropout"]:
                layers.append(nn.Dropout(float(hp["dropout"])))
            w = int(hp["hidden"])
        layers.append(nn.Linear(w, O.N_OUTCOMES))
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("rule", torch.from_numpy(O.rule_mask_table().astype(bool)))

    def forward(self, p, c, b, k, g, l):
        x = [self.e_count(c), self.e_bo(b), self.e_cl(k), self.e_pitch(g), self.e_loc(l)]
        if self.e_p is not None:
            x.append(self.e_p(p))
        if self.e_pp is not None:
            x.append(self.e_pp(p * N_PITCH + g))
        logits = self.mlp(torch.cat(x, dim=-1))
        return logits.masked_fill(~self.rule[c], float("-inf"))

    def embed_l2(self) -> torch.Tensor:
        z = [e.weight.pow(2).mean() for e in (self.e_p, self.e_pp) if e is not None]
        return torch.stack(z).sum() if z else torch.zeros((), device=self.rule.device)


def _rows(df: pd.DataFrame, state_id: np.ndarray, K: int):
    """행동·결과가 있는 투구 → (p, c, b, k, g, l, o) int64 배열. 상태 성분은 state_id 에서 복원 (B1 의 주자아웃 붕괴와 일치)."""
    ok = (df["action_id"].to_numpy() >= 0) & (df["outcome_id"].to_numpy() >= 0)
    # 규칙 마스크가 막는 결과가 찍힌 행(데이터 이상, 시즌당 2~5구)은 뺀다: 마스크된 로짓이 정답이면 손실이 inf
    cnt = S.decode_state(state_id, K)[0]
    ok &= O.rule_mask_table()[cnt, np.maximum(df["outcome_id"].to_numpy(dtype=np.int64), 0)]
    c, b, k = S.decode_state(state_id[ok], K)
    g, l = decode_action(df["action_id"].to_numpy(dtype=np.int64)[ok])
    cols = (df["pitcher_idx"].to_numpy(dtype=np.int64)[ok], c, b, k, g, l, df["outcome_id"].to_numpy(dtype=np.int64)[ok])
    return tuple(np.ascontiguousarray(x, dtype=np.int64) for x in cols)


def _nll(net: SharedNet, rows, dev, bs: int = 65536) -> float:
    net.eval()
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(rows[0]), bs):
            t = [torch.from_numpy(x[i:i + bs]).to(dev) for x in rows]
            tot += float(nn.functional.cross_entropy(net(*t[:6]), t[6], reduction="sum"))
    return tot / max(len(rows[0]), 1)


def train(train_rows, n_pitchers: int, K: int, hp: dict, seed: int, *, eval_rows=None, n_epochs: int | None = None):
    """n_epochs 가 없으면 eval_rows NLL 로 조기 종료해 최적 에폭을 찾는다. 반환 (net, 이력). 이력.best_epoch = 쓸 에폭 수."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device(hp["device"])
    net = SharedNet(n_pitchers, K, hp).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    data = [torch.from_numpy(x).to(dev) for x in train_rows]
    n, bs = len(train_rows[0]), int(hp["batch_size"])
    max_epochs = int(n_epochs) if n_epochs else int(hp["max_epochs"])
    hist, best, best_state, bad = [], (None, float("inf")), None, 0
    for ep in range(1, max_epochs + 1):
        net.train()
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            t = [x[idx] for x in data]
            loss = nn.functional.cross_entropy(net(*t[:6]), t[6])
            if hp["embed_l2"]:
                loss = loss + float(hp["embed_l2"]) * net.embed_l2()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        rec = {"epoch": ep, "train_nll": tot / n}
        if eval_rows is not None and not n_epochs:
            rec["eval_nll"] = _nll(net, eval_rows, dev)
            if rec["eval_nll"] < best[1] - 1e-5:
                best, bad = (ep, rec["eval_nll"]), 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
        hist.append(rec)
        log.info("neural ep %d train NLL %.4f%s", ep, rec["train_nll"], f" eval NLL {rec['eval_nll']:.4f}" if "eval_nll" in rec else "")
        if bad >= int(hp["patience"]):
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, {"epochs": hist, "best_epoch": best[0] if best[0] else max_epochs, "device": str(dev)}


def predict_tensor(net: SharedNet, n_pitchers: int, K: int, dev) -> np.ndarray:
    """P [P, S, A, O] float32. 투수 한 명씩 (S×A 행) 추론."""
    S_ = S.n_states(K)
    c, b, k = S.decode_state(np.repeat(np.arange(S_), N_ACTIONS), K)
    g, l = decode_action(np.tile(np.arange(N_ACTIONS), S_))
    fixed = [torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64)).to(dev) for x in (c, b, k, g, l)]
    out = np.empty((n_pitchers, S_, N_ACTIONS, O.N_OUTCOMES), dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for p in range(n_pitchers):
            pp = torch.full((S_ * N_ACTIONS,), p, dtype=torch.int64, device=dev)
            chunks = [torch.softmax(net(pp[i:i + 131072], *[x[i:i + 131072] for x in fixed]).float(), dim=-1) for i in range(0, S_ * N_ACTIONS, 131072)]
            pr = torch.cat(chunks).cpu().numpy().astype(np.float64)
            pr /= pr.sum(-1, keepdims=True)  # float32 softmax 의 합 오차를 계약 허용치 안으로
            out[p] = pr.reshape(S_, N_ACTIONS, O.N_OUTCOMES)
    return out


def fit(
    df: pd.DataFrame, state_id: np.ndarray, pitchers: pd.DataFrame, K: int, *, hp: dict, seed: int, n_epochs: int | None = None,
    eval_df: pd.DataFrame | None = None, eval_state_id: np.ndarray | None = None,
    repertoire_min: int = DEFAULT_REPERTOIRE_MIN_PITCHES, valid_states: np.ndarray | None = None, meta: dict | None = None,
) -> TransitionTensor:
    t0 = time.time()
    n_p, S_ = len(pitchers), S.n_states(K)
    eval_rows = _rows(eval_df, eval_state_id, K) if eval_df is not None else None
    net, info = train(_rows(df, state_id, K), n_p, K, hp, seed, eval_rows=eval_rows, n_epochs=n_epochs)
    P = predict_tensor(net, n_p, K, pick_device(hp["device"]))
    group = decode_action(np.arange(N_ACTIONS))[0]
    valid = (repertoire_counts(df, n_p) >= repertoire_min)[:, group]
    valid = np.broadcast_to(valid[:, None, :], (n_p, S_, N_ACTIONS)).copy()
    if valid_states is not None:
        valid &= np.asarray(valid_states, dtype=bool)[None, :, None]
    P[~valid] = 0.0
    n = count_transitions(df, state_id, n_p, K)
    n_obs = np.empty(n.shape[:3], dtype=np.int32)
    for i in range(n_p):
        n_obs[i] = n[i].sum(-1, dtype=np.int64)
    m = {
        "spec_version": SPEC_VERSION, "model_arch": "shared_mlp_pitcher_embedding", "K": int(K), "repertoire_min_pitches": int(repertoire_min),
        "row_sum_tol": DEFAULT_ROW_SUM_TOL, "neural": {**hp, **info, "fit_seconds": time.time() - t0},
        "n_train_pitches_with_action": int(n.sum()), "holdout_nll": None, "holdout_ece": None, "holdout_ece_hr": None, "excluded_pitchers": [],
    }
    if meta:
        m.update(meta)
    return TransitionTensor(P=P, valid=valid, n_obs=n_obs, pitchers=pitchers, meta=m)
