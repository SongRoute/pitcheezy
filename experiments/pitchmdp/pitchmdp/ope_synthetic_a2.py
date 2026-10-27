"""EXP-OPE-RES-01 synthetic certification: the TAB table world and the S5-X layered S5 world.

Both worlds add the same declared layers to a pitch-outcome model: in play -> batted-ball quality
x ~ P_W(x | pitch group, two strikes) (TRAIN empirical), class e ~ P_kappa(e | x, dp, g) proportional
to P_hat_TRAIN(e | x, dp) exp(kappa u(g) 1[e is a hit]), advancement drawn from the frozen C0
EmpiricalAdvancement outcomes (zeta tilts the logit of the scoring mass of an out with a runner on
third and < 2 outs by zeta u'(g)), realized reward = defensive C0 WE of the next state. kappa = 0
removes the direct action dependence of the class law; full R1_b exclusion needs kappa = zeta = 0.

TAB (design 4.3): exact dynamic programming gives the realized-WE truth Delta_true(r) and the exact
expectation of the baseline DR (beta_0 = 0 under V2: a deterministic integrity check). Every
replication draws a fresh synthetic TRAIN of the registered in-play count (R1_b table re-estimated
with its m selection), a 10,131-PA dataset of 150 pseudo-games, and runs the whole pipeline (R1,
R2, lambda/M selection with the closed-form cluster SE, SN normalisers). The certified object is the
combined arm: b_hat = mean_j(Delta_hat_p - Delta_hat_0) + beta_0, SE = SD_j / sqrt(R).

S5-X (design 4.2/4.6): the S5 world (bound G0, TRAIN delivery pools, frozen WE) wrapped by
``QualityLayerWorld``; run k has its own world seed role_seed(v2) XOR k; logs SeedSequence([ws, 0,
i, j]), truth SeedSequence([ws, 1, i]) (run_world layout); added layers from seed role s5x_world
with key [k, layer, i, draw] (layer 0 log / draw = log j, layer 1 truth / draw = rollout), the
candidate and reference truth rollouts sharing the same layer draws (CRN). Six repetitions over the
fixed starts: repetition r uses log r - 1 and rollouts 2r - 2, 2r - 1; each recomputes the whole
pipeline; acceptance SE = SD_r(gap_r) / sqrt(6). S-2 uses the Rao-Blackwellized CRN surrogate bias.
"""
from __future__ import annotations

import math

import numpy as np

from .game import GameState
from .ope_diagnostics import affine_padded
from .ope_redesign_registry import (BASELINE, BIP_CLASSES, HIT_CLASSES, INF, INTEGRITY_TOL, LAMBDA_GRID, M_GRID,
                                    N_CELLS, N_X, PAIRS, R1, S5X_REPETITIONS, TAB_DATASET_PAS,
                                    TAB_ECE_FACTOR, TAB_ECE_RECORDED, TAB_PLANNING_SIGMA, TAB_PSEUDO_GAMES,
                                    TAB_TARGET_KL, TERMINALS9, Z_BOUND, cell_outcome, cluster_se_index,
                                    inconclusive_invalid, pair_id, role_sequence, select_minimizer)
from .ope_surrogate import N_CLASSES, fit_table, select_m, tilted_out_mass
from .planner import OUTCOMES
from .policy_artifacts import IntegrityError, _require

HIT = np.array([c in HIT_CLASSES for c in BIP_CLASSES])
BIP10 = np.array([OUTCOMES.index(c) for c in BIP_CLASSES])  # positions of the six in-play classes in OUTCOMES
T9 = {name: i for i, name in enumerate(TERMINALS9)}
BIP_TO_T9 = np.array([TERMINALS9.index(c) for c in BIP_CLASSES], dtype=np.int64)
OUT_BIP = BIP_CLASSES.index('out')
N_COUNTS = 12  # count index = balls * 3 + strikes


# ---------------------------------------------------------------- shared layer math
def class_law(table, signs, kappa):
    """P_kappa(e | cell, g) [960, G, 6] from the TRAIN table and the per-group signs u(g)."""
    table = np.asarray(table, dtype=np.float64)
    u = np.asarray(signs, dtype=np.float64)
    if kappa == 0:
        return np.repeat(table[:, None, :], len(u), axis=1)
    tilt = np.exp(kappa * u[None, :, None] * HIT[None, None, :])
    raw = table[:, None, :] * tilt
    return raw / raw.sum(axis=-1, keepdims=True)


def drifted_table(table, first_half, second_half):
    """Era drift: clip(P_hat + (P_hat_second - P_hat_first), 0) renormalised (dp = 0 keeps no double play)."""
    shifted = np.clip(np.asarray(table) + np.asarray(second_half) - np.asarray(first_half), 0., None)
    shifted[0::2, BIP_CLASSES.index('double_play')] = 0.
    return shifted / shifted.sum(axis=1, keepdims=True)


class Situation:
    """One fixed game state with its frozen C0 terminal outcome distributions per class.

    ``outcomes[e]`` = [(prob, defensive WE of the next state, runs)] from EmpiricalAdvancement
    (deterministic apply_terminal for walk, strikeout, hbp, home run); ``tv[e]`` = their expectation
    computed exactly as game.terminal_values does."""
    def __init__(self, state, outcomes):
        self.state = state
        self.outcomes = {e: [(float(p), float(w), int(r)) for p, w, r in outcomes[e]] for e in TERMINALS9}
        self.tv = {e: float(sum(p * w for p, w, _ in self.outcomes[e])) for e in TERMINALS9}
        self.dp = bool(state.bases & 1 and state.outs < 2)
        self.zeta_eligible = bool(state.bases & 4 and state.outs < 2)

    @classmethod
    def from_models(cls, state, we, advancement):
        outcomes = {}
        for e in TERMINALS9:
            before = state.home_score if state.half == 'Bot' else state.away_score
            rows = []
            for prob, nxt in advancement.distribution(state, e):
                after = nxt.home_score if state.half == 'Bot' else nxt.away_score
                rows.append((prob, we.predict_defense(nxt, state.defender_is_home), after - before))
            outcomes[e] = rows
        return cls(state, outcomes)

    def out_outcomes(self, zeta, sign):
        """The 'out' outcome law after the zeta tilt (identity unless eligible and zeta * sign != 0)."""
        rows = self.outcomes['out']
        if not self.zeta_eligible or zeta == 0 or sign == 0:
            return rows
        p_score = sum(p for p, _, r in rows if r > 0)
        q = tilted_out_mass(p_score, zeta, sign)
        if q == p_score:
            return rows
        return [(p * (q / p_score if r > 0 else (1 - q) / (1 - p_score)), w, r) for p, w, r in rows]

    def class_values(self, zeta, sign):
        """E_zeta[WE(next) | s, e, g] for the six in-play classes."""
        out = []
        for e in BIP_CLASSES:
            rows = self.out_outcomes(zeta, sign) if e == 'out' else self.outcomes[e]
            out.append(sum(p * w for p, w, _ in rows))
        return np.array(out)


# ---------------------------------------------------------------- TAB world
class TabMarginals:
    """TRAIN marginals of the TAB world (A2-S0 output; synthetic in tests).

    outcome_freq [12, G, 5] (ball, strike, foul, hbp, in play) per count and group; count_group_freq
    [12, G] league P(g | count); league_shares [G]; archetype_shares [8, G]; x_dist [G, 2, 480]
    P(x | g, two strikes); bip_mix [G, 2, 2] P(g, two strikes, dp) of the TRAIN in-play rows;
    class_table [960, 6] (selected m), first/second half tables; situations [64 Situation]; signs u,
    u' per group; kappa_p, zeta_p (TAB groups); n_train_bip; m."""
    FIELDS = ('groups', 'outcome_freq', 'count_group_freq', 'league_shares', 'archetype_shares', 'x_dist', 'bip_mix',
              'class_table', 'first_half_table', 'second_half_table', 'situations', 'signs', 'signs_prime',
              'kappa_p', 'zeta_p', 'n_train_bip')

    def __init__(self, **values):
        _require(set(values) == set(self.FIELDS), f'TAB marginals need exactly {self.FIELDS}')
        self.__dict__.update(values)
        G = len(self.groups)
        _require(np.shape(self.outcome_freq) == (N_COUNTS, G, 5) and np.shape(self.x_dist) == (G, 2, N_X)
                 and np.shape(self.class_table) == (N_CELLS, N_CLASSES) and np.shape(self.bip_mix) == (G, 2, 2)
                 and np.shape(self.archetype_shares)[1] == G and len(self.situations) >= 1,
                 'TAB marginal shapes')


def _norm(p, axis=-1):
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(axis=axis, keepdims=True)


def safe_cdf(p):
    """Cumulative probabilities along the last axis with +inf from the last positive category on, so
    an inverse-CDF draw (count of entries <= u) never lands on a zero-probability category."""
    p = np.asarray(p, dtype=np.float64)
    cdf = np.cumsum(p, axis=-1)
    positive = p > 0
    last = p.shape[-1] - 1 - np.argmax(positive[..., ::-1], axis=-1)
    cdf[np.arange(p.shape[-1]) >= last[..., None]] = np.inf
    return cdf


def draw_index(cdf_rows, u):
    """Inverse-CDF categorical draw for rows of ``safe_cdf`` and uniforms u in [0, 1)."""
    return (np.asarray(u)[:, None] >= cdf_rows).sum(axis=1)


def _solve(policy, pb, ps, pf, imm, walk, strikeout):
    """Exact PA values V [4, 3, S, A] and Q [12, S, A, G] for a fixed policy; two-strike fouls loop."""
    S, A = policy.shape[1], policy.shape[2]
    V, Q = np.zeros((4, 3, S, A)), np.zeros(policy.shape)
    for b in (3, 2, 1, 0):
        for st in (2, 1, 0):
            c = b * 3 + st
            ball = (walk[:, None] if b == 3 else V[b + 1, st])[..., None]
            if st == 2:
                rest = pb[c] * ball + ps[c] * strikeout[:, None, None] + imm[c]
                V[b, 2] = (policy[c] * rest).sum(-1) / (1 - (policy[c] * pf[c]).sum(-1))
                Q[c] = rest + pf[c] * V[b, 2][..., None]
            else:
                nxt = V[b, st + 1][..., None]
                Q[c] = pb[c] * ball + ps[c] * nxt + pf[c] * nxt + imm[c]
                V[b, st] = (policy[c] * Q[c]).sum(-1)
    return V, Q


def _expected_dr(policy, logging, pi_b, q_hat, pb, ps, pf, imm, walk, strikeout):
    """Exact expectation Y [4, 3, S, A] of the plain DR under the logging law (pi_b_hat = pi_b)."""
    S, A = policy.shape[1], policy.shape[2]
    Y = np.zeros((4, 3, S, A))
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = np.where(pi_b > 0, logging * policy / pi_b, 0.)
    for b in (3, 2, 1, 0):
        for st in (2, 1, 0):
            c = b * 3 + st
            v_hat = (policy[c] * q_hat[c]).sum(-1)
            ball = (walk[:, None] if b == 3 else Y[b + 1, st])[..., None]
            if st == 2:
                rest = pb[c] * ball + ps[c] * strikeout[:, None, None] + imm[c]
                num = v_hat + (weight[c] * (rest - q_hat[c])).sum(-1)
                Y[b, 2] = num / (1 - (weight[c] * pf[c]).sum(-1))
            else:
                nxt = Y[b, st + 1][..., None]
                g = pb[c] * ball + ps[c] * nxt + pf[c] * nxt + imm[c]
                Y[b, st] = v_hat + (weight[c] * (g - q_hat[c])).sum(-1)
    return Y


def kl_policy_batch(q, pi_b, mask, tau):
    """rollout_policy.kl_policy over a batch: pi proportional to pi_b exp((q - max q) / tau) on the mask."""
    with np.errstate(divide='ignore', invalid='ignore'):
        logw = np.where(mask, np.log(np.where(mask, pi_b, 1.)) + (q - np.where(mask, q, -np.inf).max(-1, keepdims=True)) / tau,
                        -np.inf)
    logw -= logw.max(-1, keepdims=True)
    w = np.where(mask, np.exp(logw), 0.)
    return w / w.sum(-1, keepdims=True)


def _visitation(policy, pb, pf, ps):
    """Expected visits [12, S, A] of each count from 0-0 under ``policy`` (uniform start weights)."""
    S, A = policy.shape[1], policy.shape[2]
    occ = np.zeros((4, 3, S, A))
    occ[0, 0] = 1.
    for total in range(6):
        for b in range(4):
            st = total - b
            if not 0 <= st <= 2:
                continue
            c = b * 3 + st
            inflow = occ[b, st]
            if st == 2:
                inflow = inflow / (1 - (policy[c] * pf[c]).sum(-1))
                occ[b, st] = inflow
            p_ball = (policy[c] * pb[c]).sum(-1)
            p_adv = (policy[c] * (ps[c] + (pf[c] if st < 2 else 0.))).sum(-1)
            if b < 3:
                occ[b + 1, st] += inflow * p_ball
            if st < 2:
                occ[b, st + 1] += inflow * p_adv
    return occ.reshape(N_COUNTS, S, A)


def top_label_ece(model, world, weights, bins=10):
    """Top-label ECE10 with exact world probabilities (expected accuracy) and decision weights."""
    model = model.reshape(-1, model.shape[-1])
    world = world.reshape(-1, world.shape[-1])
    w = np.asarray(weights, dtype=np.float64).ravel()
    top = model.argmax(axis=1)
    conf = model[np.arange(len(model)), top]
    acc = world[np.arange(len(world)), top]
    index = np.minimum((conf * bins).astype(int), bins - 1)
    gap = np.bincount(index, weights=w * (acc - conf), minlength=bins)
    return float(np.abs(gap).sum() / w.sum())


def _bisect_up(fn, target, lo=0.0, hi=1.0, hi_max=1e3, iters=200, rel=1e-12):
    """The upper end of the final bracket: fn(result) >= target and fn(result - rel * max(1, result)) <
    target. Deterministic also when fn jumps (the binned ECE is a step function of eta)."""
    while fn(hi) < target:
        hi *= 2
        _require(hi <= hi_max, 'TAB calibration target not reachable inside the registered bracket')
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= rel * max(1.0, hi):
            break
    return hi


class TabWorld:
    """The exact TAB world of one cell (law, eta, kappa, zeta, drift) on fixed draws of seed role tab_world."""
    def __init__(self, marginals, *, law, eta, kappa, zeta, drift=False, perturbation_sd=0.1,
                 sigma=TAB_PLANNING_SIGMA, target_kl=TAB_TARGET_KL, seed_role='tab_world'):
        mg = marginals
        self.mg, self.law, self.eta, self.kappa, self.zeta, self.drift = mg, law, float(eta), float(kappa), float(zeta), drift
        G = len(mg.groups)
        S, A = len(mg.situations), len(mg.archetype_shares)
        self.S, self.A, self.G = S, A, G
        self.seed_role = seed_role
        draw = lambda component, shape, scale=1.0: scale * np.random.default_rng(
            role_sequence(seed_role, 0, component)).standard_normal(shape)
        perturb = draw(2, (A, G, 5), perturbation_sd)
        self.p5 = _norm(np.asarray(mg.outcome_freq)[:, None] * np.exp(perturb)[None])  # [12, A, G, 5]
        shares = np.asarray(mg.archetype_shares, dtype=np.float64)
        self.support = shares > 0  # [A, G]
        pi_b = np.asarray(mg.count_group_freq)[:, None, :] * np.where(self.support, shares / np.asarray(mg.league_shares), 0.)[None]
        self.pi_b = _norm(pi_b)  # [12, A, G]
        logging = np.where(self.support, self.pi_b ** 0.5, 0.) if law == 'V3' else self.pi_b
        self.logging = _norm(logging)
        base_table = drifted_table(mg.class_table, mg.first_half_table, mg.second_half_table) if drift else mg.class_table
        self.world_table = np.asarray(base_table, dtype=np.float64)
        self.law_k = class_law(self.world_table, mg.signs, self.kappa)  # [960, G, 6]
        dp = np.array([s.dp for s in mg.situations])
        x = np.asarray(mg.x_dist, dtype=np.float64)  # [G, 2, 480]
        law_x = self.law_k.reshape(N_X, 2, G, N_CLASSES)  # cell = x * 2 + dp
        self.pcls = np.einsum('gtx,xdge->gtde', x, law_x)  # [G, 2(ts), 2(dp), 6]
        ev = np.array([[s.class_values(self.zeta, mg.signs_prime[g]) for g in range(G)] for s in mg.situations])  # [S,G,6]
        self.ev_world = ev
        self.tv = np.array([[s.tv[e] for e in TERMINALS9] for s in mg.situations])  # [S, 9]
        ts = np.array([b_s % 3 == 2 for b_s in range(N_COUNTS)])
        self.ts = ts
        pcls_s = self.pcls[:, ts.astype(int)][:, :, dp.astype(int)]  # [G, 12, S, 6]
        pcls_s = np.moveaxis(pcls_s, 0, 2)  # [12, S, G, 6]
        self.ip_world = (pcls_s * ev[None]).sum(-1)  # [12, S, G] expected in-play value
        p5 = self.p5[:, None]  # [12, 1, A, G, 5]
        self.pb, self.ps, self.pf = (np.broadcast_to(p5[..., k], (N_COUNTS, S, A, G)) for k in (0, 1, 2))
        self.walk, self.strikeout, self.hbp = self.tv[:, T9['walk']], self.tv[:, T9['strikeout']], self.tv[:, T9['hbp']]
        self.imm_world = p5[..., 3] * self.hbp[None, :, None, None] + p5[..., 4] * self.ip_world[:, :, None, :]
        # 10-class world law and the misspecified planner model
        world10 = np.zeros((N_COUNTS, S, A, G, 10))
        world10[..., 0], world10[..., 1], world10[..., 2] = self.pb, self.ps, self.pf
        world10[..., OUTCOMES.index('hbp')] = np.broadcast_to(p5[..., 3], (N_COUNTS, S, A, G))
        world10[..., BIP10] = p5[..., 4][..., None] * pcls_s[:, :, None]
        self.world10 = world10
        eps = draw(3, world10.shape)
        self.eps = eps
        with np.errstate(divide='ignore'):
            logits = np.log(world10) + self.eta * eps
        logits -= logits.max(-1, keepdims=True)
        model = np.exp(logits)
        self.model10 = model / model.sum(-1, keepdims=True)
        tv_bip = self.tv[:, [T9[e] for e in BIP_CLASSES]]  # [S, 6]
        m = self.model10
        imm_model = m[..., OUTCOMES.index('hbp')] * self.hbp[None, :, None, None] + (m[..., BIP10] * tv_bip[None, :, None, None]).sum(-1)
        pi_b_full = np.broadcast_to(self.pi_b[:, None], (N_COUNTS, S, A, G))
        self.pi_b_full = pi_b_full
        self.mask = np.broadcast_to(self.support[None, None], (N_COUNTS, S, A, G))
        _, self.q_model = _solve(pi_b_full, m[..., 0], m[..., 1], m[..., 2], imm_model, self.walk, self.strikeout)
        shape = (N_COUNTS, S, A, G)
        self.q_plan = self.q_model + draw(4, shape, sigma)
        self.q_hat = self.q_model + draw(5, shape, sigma)
        self.visits = _visitation(pi_b_full, self.pb, self.pf, self.ps)  # [12, S, A] under pi_b
        self.tau = self._tune_tau(target_kl)
        self.pi_c = kl_policy_batch(self.q_plan, pi_b_full, self.mask, self.tau)
        self.v_true = {name: _solve(pol, self.pb, self.ps, self.pf, self.imm_world, self.walk, self.strikeout)[0]
                       for name, pol in (('candidate', self.pi_c), ('reference', pi_b_full))}
        self.delta_true = float((self.v_true['candidate'][0, 0] - self.v_true['reference'][0, 0]).mean())
        logging_full = np.broadcast_to(self.logging[:, None], shape)
        y = {name: _expected_dr(pol, logging_full, pi_b_full, self.q_hat, self.pb, self.ps, self.pf, self.imm_world,
                                self.walk, self.strikeout)
             for name, pol in (('candidate', self.pi_c), ('reference', pi_b_full))}
        self.expected_delta_hat_0 = float((y['candidate'][0, 0] - y['reference'][0, 0]).mean())
        self.beta_0 = self.expected_delta_hat_0 - self.delta_true
        self.v_c, self.v_r = (self.pi_c * self.q_hat).sum(-1), (pi_b_full * self.q_hat).sum(-1)
        self.logging_cdf = safe_cdf(self.logging)
        self.p5_cdf = safe_cdf(self.p5)
        self.x_cdf = safe_cdf(x)
        self.law_cdf = safe_cdf(self.law_k)
        self.adv = self._advancement_tables()

    def surrogate_truth(self):
        """Descriptive exact DP decomposition: Delta_true(r~) for R1_a (C0 class values of the realized
        class) and R1_b (the A2-S0 TRAIN table over the world's x law), and the surrogate share
        Delta_true(r~) - Delta_true(r) (realized WE)."""
        mg, S, G = self.mg, self.S, self.G
        dp = np.array([s.dp for s in mg.situations]).astype(int)
        ts = self.ts.astype(int)
        tv_bip = self.tv[:, [T9[e] for e in BIP_CLASSES]]  # [S, 6]
        table_x = np.asarray(mg.class_table, dtype=np.float64).reshape(N_X, 2, N_CLASSES)
        tcls = np.einsum('gtx,xde->gtde', np.asarray(mg.x_dist, dtype=np.float64), table_x)  # [G, ts, dp, 6]
        p5 = self.p5[:, None]
        out = {'R1_0_realized': self.delta_true}
        for name, cls in (('R1_a', self.pcls), ('R1_b', tcls)):
            law = np.moveaxis(cls[:, ts][:, :, dp], 0, 2)  # [12, S, G, 6]
            imm = p5[..., 3] * self.hbp[None, :, None, None] + p5[..., 4] * ((law * tv_bip[None, :, None]).sum(-1))[:, :, None, :]
            values = {pol: _solve(policy, self.pb, self.ps, self.pf, imm, self.walk, self.strikeout)[0][0, 0]
                      for pol, policy in (('candidate', self.pi_c), ('reference', self.pi_b_full))}
            delta = float((values['candidate'] - values['reference']).mean())
            out[name] = delta
            out[f'{name}_surrogate_share'] = delta - self.delta_true
        return out

    def _tune_tau(self, target):
        w = self.visits  # [12, S, A] expected visits under pi_b

        def mean_kl(tau):
            pi = kl_policy_batch(self.q_plan, self.pi_b_full, self.mask, tau)
            with np.errstate(divide='ignore', invalid='ignore'):
                kl = np.where(self.mask & (pi > 0), pi * np.log(pi / self.pi_b_full), 0.).sum(-1)
            return float((w * kl).sum() / w.sum())
        lo, hi = 1e-7, 10.0
        _require(mean_kl(lo) > target > mean_kl(hi), 'TAB tau target outside the bisection bracket')
        for _ in range(300):
            mid = math.sqrt(lo * hi)
            if mean_kl(mid) > target:
                lo = mid
            else:
                hi = mid
            if hi / lo - 1 < 1e-13:
                break
        return math.sqrt(lo * hi)

    def _advancement_tables(self):
        """Per (situation, class 9, group): cumulative outcome probabilities and next-state WE (padded)."""
        S, G = self.S, self.G
        rows = {}
        width = 1
        for s, sit in enumerate(self.mg.situations):
            for e in TERMINALS9:
                for g in range(G):
                    out = sit.out_outcomes(self.zeta, self.mg.signs_prime[g]) if e == 'out' else sit.outcomes[e]
                    rows[s, T9[e], g] = out
                    width = max(width, len(out))
        probs = np.zeros((S, 9, G, width))
        we = np.zeros((S, 9, G, width))
        for (s, e, g), out in rows.items():
            probs[s, e, g, :len(out)] = [o[0] for o in out]
            we[s, e, g, :len(out)] = [o[1] for o in out]
            we[s, e, g, len(out):] = out[-1][1]
        return safe_cdf(probs), we

    @classmethod
    def eta_p(cls, marginals, *, perturbation_sd=0.1, seed_role='tab_world'):
        """eta at which the planner model's exact top-label ECE10 equals 2 x 0.0087 (V2, kappa = zeta = 0)."""
        base = cls(marginals, law='V2', eta=0.0, kappa=0.0, zeta=0.0, perturbation_sd=perturbation_sd,
                   seed_role=seed_role)
        eps = base.eps
        weights = base.visits[..., None] * base.pi_b_full

        def ece(eta):
            with np.errstate(divide='ignore'):
                logits = np.log(base.world10) + eta * eps
            logits -= logits.max(-1, keepdims=True)
            model = np.exp(logits)
            model /= model.sum(-1, keepdims=True)
            return top_label_ece(model, base.world10, weights)
        return _bisect_up(ece, TAB_ECE_FACTOR * TAB_ECE_RECORDED, hi_max=64.0)

    # ------------------------------------------------------------ one replication
    def simulate(self, rng, n_pas=TAB_DATASET_PAS, cap=64):
        """Logged PAs under the cell's logging law: padded DR inputs and terminal records."""
        S, A, G = self.S, self.A, self.G
        sit = rng.integers(S, size=n_pas)
        arch = rng.integers(A, size=n_pas)
        b = np.zeros(n_pas, dtype=np.int64)
        st = np.zeros(n_pas, dtype=np.int64)
        active = np.ones(n_pas, dtype=bool)
        steps = {k: np.zeros((n_pas, cap)) for k in ('v_c', 'v_r', 'q_a', 'rho_c')}
        lengths = np.zeros(n_pas, dtype=np.int64)
        reward = np.full(n_pas, np.nan)
        term9 = np.full(n_pas, -1, dtype=np.int64)
        cell = np.full(n_pas, -1, dtype=np.int64)
        dp = np.array([s.dp for s in self.mg.situations])
        adv_cdf, adv_we = self.adv
        for t in range(cap):
            idx = np.flatnonzero(active)
            if not len(idx):
                break
            u = rng.random((len(idx), 5))
            c = b[idx] * 3 + st[idx]
            s_i, a_i = sit[idx], arch[idx]
            g = draw_index(self.logging_cdf[c, a_i], u[:, 0])
            steps['v_c'][idx, t] = self.v_c[c, s_i, a_i]
            steps['v_r'][idx, t] = self.v_r[c, s_i, a_i]
            steps['q_a'][idx, t] = self.q_hat[c, s_i, a_i, g]
            steps['rho_c'][idx, t] = self.pi_c[c, s_i, a_i, g] / self.pi_b[c, a_i, g]
            lengths[idx] += 1
            o = draw_index(self.p5_cdf[c, a_i, g], u[:, 1])
            ball, strike, foul, hbp, ip = (o == k for k in range(5))
            walk = ball & (b[idx] == 3)
            k_out = strike & (st[idx] == 2)
            ends = walk | k_out | hbp | ip
            for mask_, e in ((walk, 'walk'), (k_out, 'strikeout'), (hbp, 'hbp')):
                rows = idx[mask_]
                term9[rows] = T9[e]
                k_adv = draw_index(adv_cdf[sit[rows], T9[e], g[mask_]], u[mask_, 4])  # C0: one deterministic outcome
                reward[rows] = adv_we[sit[rows], T9[e], g[mask_], k_adv]
            if ip.any():
                rows = idx[ip]
                gi, ts_i = g[ip], (st[rows] == 2).astype(int)
                x = draw_index(self.x_cdf[gi, ts_i], u[ip, 2])
                cl = x * 2 + dp[sit[rows]].astype(int)
                e = draw_index(self.law_cdf[cl, gi], u[ip, 3])
                e9 = BIP_TO_T9[e]
                k_adv = draw_index(adv_cdf[sit[rows], e9, gi], u[ip, 4])
                reward[rows] = adv_we[sit[rows], e9, gi, k_adv]
                term9[rows], cell[rows] = e9, cl
            b[idx] = np.where(ball & ~walk, b[idx] + 1, b[idx])
            st[idx] = np.where((strike & ~k_out) | (foul & (st[idx] < 2)), st[idx] + 1, st[idx])
            active[idx[ends]] = False
        _require(not active.any(), 'a TAB PA reached the registered cap (deterministic integrity)')
        L = max(int(lengths.max()), 1)
        steps = {k: v[:, :L] for k, v in steps.items()}
        return {'steps': steps, 'lengths': lengths, 'reward': reward, 'term9': term9, 'cell': cell, 'sit': sit,
                'game': np.arange(n_pas) % TAB_PSEUDO_GAMES}

    def synthetic_train(self, rng, n):
        """Multinomial counts over (fold, dp, x, e) of n synthetic TRAIN in-play rows (undrifted world
        class law, TRAIN mix of group / two strikes / dp); fold = iid half."""
        law = class_law(self.mg.class_table, self.mg.signs, self.kappa).reshape(N_X, 2, self.G, N_CLASSES)
        mix = np.asarray(self.mg.bip_mix, dtype=np.float64)  # [G, ts, dp]
        x = np.asarray(self.mg.x_dist, dtype=np.float64)  # [G, ts, x]
        joint = np.einsum('gtd,gtx,xdge->dxe', mix, x, law)  # [dp, x, e]
        joint = np.moveaxis(joint, 0, 1).reshape(N_CELLS, N_CLASSES)  # cell = x * 2 + dp
        p = np.concatenate([0.5 * joint.ravel(), 0.5 * joint.ravel()])
        counts = rng.multinomial(int(n), p / p.sum())
        half = N_CELLS * N_CLASSES
        return counts[:half].reshape(N_CELLS, N_CLASSES), counts[half:].reshape(N_CELLS, N_CLASSES)


def _transform_array(kind, param):
    if kind == 'identity' or param == INF:
        return None
    lam = float(param)
    if kind == 'shrink':
        return lambda r: np.where(r == 0, r, lam * r / (r * r + lam))
    return lambda r: np.where(r <= lam, r, 0.)


def _sn_inverse(rho, lengths):
    """Inverse SN normalisers [L + 1] from padded ratios (held W beyond each PA's end)."""
    n, L = rho.shape
    active = np.arange(L)[None, :] < lengths[:, None]
    held = np.where(active, rho, 1.)
    w = np.concatenate([np.ones((n, 1)), np.cumprod(held, axis=1)], axis=1)
    n_bar = w.mean(axis=0)
    valid = bool(np.isfinite(n_bar).all() and (n_bar > 0).all())
    with np.errstate(divide='ignore'):
        return (1.0 / n_bar if valid else None), n_bar


def pipeline(data, rewards, *, smoke=None, per_pa=False):
    """Delta_hat of every pair on one dataset of complete PAs (the whole registered pipeline):
    rewards {R1: [n]}; R2_1 / R2_2 choose lambda / M with the closed-form cluster SE over
    ``data['game']`` (pseudo-games in TAB, starts in S5-X); R2_3 uses the dataset's own normalisers.
    ``data['steps']`` holds padded v_c, v_r, q_a, rho_c and optionally rho_r (default 1: reference =
    pi_b_hat). ``smoke`` = {'lambda': x, 'M': y}: fixed grid values, no selection."""
    s, lengths = data['steps'], data['lengths']
    _, inverse = np.unique(np.asarray(data['game']), return_inverse=True)
    n_clusters = int(inverse.max()) + 1 if len(inverse) else 0
    se_cl = lambda v: cluster_se_index(v, inverse, n_clusters)
    active = np.arange(s['rho_c'].shape[1])[None, :] < lengths[:, None]
    ones = np.where(active, 1., 0.)
    rho_r = s.get('rho_r', ones)
    lines = {}

    def line(f, u=None, u_r=None):
        rc = s['rho_c'] if f is None else f(s['rho_c'])
        rr = rho_r if f is None else f(rho_r)
        return (affine_padded(s['v_c'], s['q_a'], rc, lengths, u),
                affine_padded(s['v_r'], s['q_a'], rr, lengths, u if u_r is None else u_r))
    lines['identity'] = line(None)
    grids = {'shrink': LAMBDA_GRID, 'switch': M_GRID} if smoke is None else {'shrink': (smoke['lambda'],),
                                                                             'switch': (smoke['M'],)}
    for kind, grid in grids.items():
        for v in grid:
            if v != INF:
                lines[(kind, v)] = line(_transform_array(kind, v))
    u_c, _ = _sn_inverse(s['rho_c'], lengths)
    u_r, _ = _sn_inverse(rho_r, lengths)
    sn_valid = u_c is not None and u_r is not None
    if sn_valid:
        lines['sn'] = line(None, u_c, u_r)
    deltas = lambda key, c: (lines[key][0][0] + lines[key][0][1] * c) - (lines[key][1][0] + lines[key][1][1] * c)
    out, selected, invalid, rows = {}, {}, [], {}
    for comp in R1:
        c = rewards[comp]
        base = deltas('identity', c)
        out[pair_id(comp, 'R2_0')] = float(base.mean())
        rows[pair_id(comp, 'R2_0')] = base
        for kind, r2 in (('shrink', 'R2_1'), ('switch', 'R2_2')):
            name = pair_id(comp, r2)
            grid = grids[kind]
            if smoke is not None:
                chosen = grid[0]
            else:
                values = {v: (base if v == INF else deltas((kind, v), c)) for v in grid}
                objectives = []
                for v in grid:
                    diff = values[v] - base
                    bias = abs(float(diff.mean())) + Z_BOUND * se_cl(diff)
                    value = bias * bias + se_cl(values[v]) ** 2
                    objectives.append(value if math.isfinite(value) else math.inf)
                chosen = select_minimizer(grid, objectives)
            if chosen is None:
                invalid.append(name)
                out[name] = math.nan
                continue
            selected[name] = chosen
            rows[name] = base if chosen == INF else deltas((kind, chosen), c)
            out[name] = float(rows[name].mean())
        name = pair_id(comp, 'R2_3')
        if sn_valid:
            rows[name] = deltas('sn', c)
            out[name] = float(rows[name].mean())
        else:
            invalid.append(name)
            out[name] = math.nan
    for name, value in out.items():
        if not math.isfinite(value) and name not in invalid:
            invalid.append(name)
    if per_pa:
        return out, selected, invalid, rows
    return out, selected, invalid


def tab_rewards(world, data, table):
    """R1_0 realized WE, R1_a class-expected C0 value, R1_b quality-expected value (replication table;
    NaN when the replication's table could not be fitted)."""
    tv = world.tv  # [S, 9]
    sit, term9, cell = data['sit'], data['term9'], data['cell']
    r0 = data['reward']
    ra = tv[sit, term9]
    if table is None:
        return {'R1_0': r0, 'R1_a': ra, 'R1_b': np.full_like(ra, np.nan)}
    rb = ra.copy()
    ip = cell >= 0
    tv_bip = tv[:, [T9[e] for e in BIP_CLASSES]]
    rb[ip] = (np.asarray(table)[cell[ip]] * tv_bip[sit[ip]]).sum(1)
    return {'R1_0': r0, 'R1_a': ra, 'R1_b': rb}


def tab_replication(world, cell_index, replication, *, n_train, smoke=None):
    """One full replication of one cell: synthetic TRAIN (+ m selection), dataset, every pair. Registered
    stream [1, cell_index, replication] of role tab_world; the smoke uses its own role 'smoke' with key
    [1, replication] (never a registered stream). If every m is nonfinite the R1_b table is not fitted:
    only the R1_b pairs are NOT_EVALUABLE in this replication; R1_0 and R1_a continue."""
    if world.seed_role == 'tab_world':
        rng = np.random.default_rng(role_sequence('tab_world', 1, cell_index, replication))
    else:
        _require(world.seed_role == 'smoke', 'TAB worlds use the tab_world or the smoke seed role')
        rng = np.random.default_rng(role_sequence('smoke', 1, replication))
    folds = world.synthetic_train(rng, n_train)
    if smoke is None:
        try:
            m, _ = select_m(folds)
        except IntegrityError:
            m = None
    else:
        m = smoke['m']
    table = None if m is None else fit_table(folds[0] + folds[1], m)
    data = world.simulate(rng)
    rewards = tab_rewards(world, data, table)
    estimates, selected, invalid = pipeline(data, rewards, smoke=smoke)
    not_evaluable = [] if table is not None else [p for p in PAIRS if p.startswith('R1_b')]
    invalid = [p for p in invalid if p not in not_evaluable]
    return estimates, {**selected, 'm': m}, invalid, not_evaluable


def tab_cell(world, cell, *, n_train, replications=None, deadline=None):
    """Combined realized-WE bias of every pair in one cell (design 4.5): b_hat, SE, SD, RMSE, the
    lambda / M / m selections, invalid and not-evaluable replications, the exact DP truth and the
    descriptive surrogate-truth decomposition; acceptance outcome for acceptance cells (the baseline's
    own row is the exact DP value beta_0, marked EXACT_DP, not a certification cell)."""
    R = cell['R'] if replications is None else replications
    estimates = {p: np.full(R, np.nan) for p in PAIRS}
    picks, invalid, not_evaluable = {}, {p: 0 for p in PAIRS}, {p: 0 for p in PAIRS}
    for j in range(R):
        if deadline is not None:
            deadline()
        est_j, sel, bad, skipped = tab_replication(world, cell['index'], j, n_train=n_train)
        for p in PAIRS:
            estimates[p][j] = est_j[p]
            invalid[p] += p in bad
            not_evaluable[p] += p in skipped
        for name, value in sel.items():
            label = 'none' if value is None else ('inf' if value == INF else str(value))
            picks.setdefault(name, {}).setdefault(label, 0)
            picks[name][label] += 1
    base = estimates[BASELINE]
    out = {}
    for p in PAIRS:
        diff = estimates[p] - base
        valid = invalid[p] == 0 and invalid[BASELINE] == 0 and not_evaluable[p] == 0 and np.isfinite(diff).all()
        if p == BASELINE:
            b_hat, se = world.beta_0, 0.0
        else:
            b_hat = float(np.mean(diff)) + world.beta_0 if valid else None
            se = float(np.std(diff, ddof=1) / math.sqrt(R)) if valid and R > 1 else None
        finite = np.isfinite(estimates[p])
        record = {'b_hat': b_hat, 'se': se, 'invalid_replications': invalid[p],
                  'not_evaluable_replications': not_evaluable[p], 'R': R,
                  'sd': float(np.std(estimates[p][finite], ddof=1)) if finite.sum() > 1 else None,
                  'rmse': float(np.sqrt(np.mean((estimates[p][finite] - world.delta_true) ** 2))) if finite.any() else None}
        if cell['role'] == 'acceptance':
            if p == BASELINE:
                record.update(outcome='EXACT_DP', note='beta_0 by exact dynamic programming; not a certification cell')
            elif not_evaluable[p]:
                record.update(cell_outcome(None, None, not_evaluable=True))
            elif invalid[p] or invalid[BASELINE]:
                record.update(inconclusive_invalid(f'{invalid[p] + invalid[BASELINE]} invalid replications'),
                              b_hat=b_hat, se=se)
            else:
                record.update(cell_outcome(b_hat, se, beta_0=world.beta_0 if cell['law'] == 'V3' else None, valid=valid))
        out[p] = record
    return {'cell': {k: cell[k] for k in ('id', 'law', 'eta', 'kappa', 'zeta', 'R', 'role', 'drift', 'index')},
            'delta_true': world.delta_true, 'beta_0': world.beta_0, 'tau': world.tau,
            'expected_delta_hat_0': world.expected_delta_hat_0, 'delta_true_surrogate': world.surrogate_truth(),
            'pairs': out, 'selections': picks}


def tab_integrity(world):
    """Deterministic check of a V2 cell: the baseline's exact DR expectation equals the DP truth (1e-12)."""
    ok = world.law != 'V2' or abs(world.expected_delta_hat_0 - world.delta_true) <= INTEGRITY_TOL
    if not ok:
        raise IntegrityError(f'TAB V2 baseline DP expectation differs from the DP truth: '
                             f'{world.expected_delta_hat_0 - world.delta_true}')
    return True


# ================================================================ S5-X: the S5 world with the A2 layers
class S5XLayers:
    """Layer parameters of one S5-X run: the five S5-X pitch groups' x law [5, 2, 480], the TRAIN class
    table [960, 6], signs u / u' per group, kappa, zeta and the frozen C0 outcome law of every bound
    context (Situation per context_key, checked against FrozenWE.terminal_tables bit for bit)."""
    def __init__(self, *, groups, x_dist, class_table, signs, signs_prime, kappa, zeta, situations):
        self.groups, self.kappa, self.zeta = tuple(groups), float(kappa), float(zeta)
        self.x_dist = np.asarray(x_dist, dtype=np.float64)
        self.table = np.asarray(class_table, dtype=np.float64)
        self.signs, self.signs_prime = np.asarray(signs, dtype=np.float64), np.asarray(signs_prime, dtype=np.float64)
        _require(self.x_dist.shape == (len(self.groups), 2, N_X) and self.table.shape == (N_CELLS, N_CLASSES),
                 'S5-X layer shapes')
        self.law = class_law(self.table, self.signs, self.kappa)  # [960, G, 6]
        self.x_cdf, self.law_cdf = safe_cdf(self.x_dist), safe_cdf(self.law)
        law_x = self.law.reshape(N_X, 2, len(self.groups), N_CLASSES)
        self.pcls = np.einsum('gtx,xdge->gtde', self.x_dist, law_x)  # [G, ts, dp, 6]
        self.situations = dict(situations)

    @classmethod
    def from_components(cls, components, game_values, **kwargs):
        """Situations of every bound context from the pinned WE/advancement; FrozenWE must agree exactly."""
        situations = {}
        for key, row in components.inputs.rows.items():
            sit = Situation.from_models(GameState.from_row(row), game_values['we'], game_values['advancement'])
            frozen = components.we.terminal_tables[key]
            _require(all(sit.tv[e] == frozen[e] for e in TERMINALS9),
                     'S5-X advancement layer differs from the frozen terminal values (declared matching model)')
            situations[key] = sit
        return cls(situations=situations, **kwargs)

    def group(self, action):
        from .ope_redesign_registry import pitch_group
        g = pitch_group(action)
        _require(g in self.groups, f'action {action!r} has no S5-X pitch group')
        return self.groups.index(g)

    def gamma(self, state, g):
        """Per-unit-in-play surrogate gaps (D_a, D_b) at a state for group g (exact bin sums):
        D_b = E_x[r~_b(x, s) - m(x, a, s)], D_a = E_x[sum_e P_kappa(e) tv(s)[e] - m(x, a, s)]."""
        sit = self.situations[state.context_key]
        ts, dp = int(state.strikes == 2), int(sit.dp)
        tv = np.array([sit.tv[e] for e in BIP_CLASSES])
        ev = sit.class_values(self.zeta, self.signs_prime[g])
        x = self.x_dist[g, ts]
        law = self.law.reshape(N_X, 2, len(self.groups), N_CLASSES)[:, dp, g]  # [480, 6]
        table = self.table.reshape(N_X, 2, N_CLASSES)[:, dp]  # [480, 6]
        m = law @ ev
        d_b = float(x @ (table @ tv - m))
        d_a = float(self.pcls[g, ts, dp] @ (tv - ev))
        return d_a, d_b


class QualityLayerWorld:
    """JointSimulator semantics (delivery -> G0 10-class outcome -> history) plus the A2 layers on
    in-play outcomes. ``step`` takes per-state layer uniforms [N, 3] (x, class, advancement); on the
    log path (``generate_pa``, one state per call) they come from the PA's bound layer stream."""
    def __init__(self, pool, predictor, terminal, budget, layers):
        self.pool, self.predictor, self.terminal, self.budget, self.layers = pool, predictor, terminal, budget, layers
        self._bound, self._t, self.last = None, 0, None

    def bind(self, layer_uniforms):
        self._bound, self._t, self.last = np.asarray(layer_uniforms, dtype=np.float64), 0, None

    def step(self, states, actions, uniforms, layer_u=None):
        from .rollout_policy import PAState, PastPitch, Step, _draw, _we, probabilities
        u = np.asarray(uniforms, dtype=float)
        _require(len(states) == len(actions) and u.shape == (len(states), 2), 'one action and two uniforms per state')
        if not states:
            return []
        if layer_u is None:
            _require(self._bound is not None and len(states) == 1, 'log steps need a bound layer stream')
            layer_u = self._bound[self._t:self._t + 1]
            self._t += 1
        layer_u = np.asarray(layer_u, dtype=np.float64)
        delivery = np.stack([self.pool(s, a).values[int(x * 400)] for s, a, x in zip(states, actions, u[:, 0])])
        self.budget.consume(len(states))
        p = probabilities(self.predictor(states, actions, delivery), 10)
        out = []
        for s, a, d, probs, x, lu in zip(states, actions, delivery, p, u[:, 1], layer_u):
            outcome = OUTCOMES[_draw(probs, x)]
            terminal = ('walk' if outcome == 'ball' and s.balls == 3 else
                        'strikeout' if outcome == 'strike' and s.strikes == 2 else
                        outcome if outcome in OUTCOMES[3:] else None)
            physical = tuple(float(v) for v in d)
            if terminal in BIP_CLASSES:
                value, e, cell = self._in_play(s, a, lu)
                self.last = {'class': e, 'cell': cell, 'context_key': s.context_key}
                out.append(Step(None, _we(value), a, physical, e))
            elif terminal:
                self.last = {'class': terminal, 'cell': -1, 'context_key': s.context_key}
                out.append(Step(None, _we(self.terminal(s, terminal)), a, physical, outcome))
            else:
                token = PastPitch(a, physical, outcome, s.balls, s.strikes)
                child = PAState(s.balls + int(outcome == 'ball'), min(2, s.strikes + int(outcome in ('strike', 'foul'))),
                                s.pitcher, s.batter_side, s.history + (token,), s.context_key)
                out.append(Step(child, None, a, physical, outcome))
        return out

    def _in_play(self, state, action, lu):
        layers = self.layers
        sit = layers.situations[state.context_key]
        g = layers.group(action)
        ts = int(state.strikes == 2)
        x = int(draw_index(layers.x_cdf[g, ts][None], lu[0:1])[0])
        cell = x * 2 + int(sit.dp)
        e = BIP_CLASSES[int(draw_index(layers.law_cdf[cell, g][None], lu[1:2])[0])]
        rows = sit.out_outcomes(layers.zeta, layers.signs_prime[g]) if e == 'out' else sit.outcomes[e]
        k = int(draw_index(safe_cdf(np.array([r[0] for r in rows]))[None], lu[2:3])[0])
        return rows[k][1], e, cell


def rollouts_x(world, states, policy, uniforms, layer_uniforms, cutoff, record=None):
    """rollout_policy.rollouts with per-(trajectory, depth) layer uniforms (CRN across policies) and an
    optional record(i, depth, state, names, probs) callback for the Rao-Blackwellized bias."""
    from .rollout_policy import _draw, _we, probabilities
    u = np.asarray(uniforms, dtype=float)
    lu = np.asarray(layer_uniforms, dtype=float)
    n = len(states)
    _require(u.shape[0] == n and u.shape[2] == 3 and lu.shape[:2] == u.shape[:2] and lu.shape[2] == 3,
             'rollout and layer uniforms must align')
    current, values = list(states), np.zeros(n)
    for depth in range(u.shape[1]):
        active = [i for i, s in enumerate(current) if s is not None]
        if not active:
            break
        actions = []
        for i in active:
            names, p = policy(current[i], depth)
            p = probabilities(p, len(names))
            if record is not None:
                record(i, depth, current[i], names, p)
            actions.append(names[_draw(p, u[i, depth, 0])])
        steps = world.step([current[i] for i in active], actions, u[active, depth, 1:], layer_u=lu[active, depth])
        for i, step in zip(active, steps):
            current[i] = step.state
            if step.value is not None:
                values[i] = step.value
    for i, s in enumerate(current):
        if s is not None:
            values[i] = _we(cutoff(s))
    return values


def _padded_lines(lines):
    """Padded DR inputs (v_c, v_r, q_a, rho_c, rho_r) of compact ope_variants lines (complete PAs)."""
    from .ope_diagnostics import padded
    arr, lengths = padded([np.array(line['steps'], dtype=np.float64).reshape(-1, 5) for line in lines], 5)
    with np.errstate(invalid='ignore'):
        steps = {'v_c': arr[..., 0], 'v_r': arr[..., 1], 'q_a': np.nan_to_num(arr[..., 2], nan=0.0),
                 'rho_c': arr[..., 3], 'rho_r': arr[..., 4]}
    return steps, lengths


def s5x_rewards(lines, ends, layers):
    """R1 rewards of the S5-X logs: R1_0 realized (cutoff at the cap / absorbed), R1_a class-expected
    C0 value of the realized terminal class, R1_b quality-expected value of an in-play terminal."""
    r0 = np.array([line['reward'] for line in lines], dtype=np.float64)
    ra, rb = r0.copy(), r0.copy()
    for i, end in enumerate(ends):
        if end is None:
            continue
        sit = layers.situations[end['context_key']]
        ra[i] = sit.tv[end['class']]
        rb[i] = ra[i]
        if end['cell'] >= 0:
            p = layers.table[end['cell']]
            rb[i] = float(sum(p[k] * sit.tv[e] for k, e in enumerate(BIP_CLASSES)))
    return {'R1_0': r0, 'R1_a': ra, 'R1_b': rb}


def repetition_streams(r, repetitions=S5X_REPETITIONS):
    """Repetition r (1..6) of an S5-X run: log j = r - 1 and truth rollouts 2r - 2, 2r - 1 of every start."""
    _require(type(r) is int and 1 <= r <= repetitions, 'repetition index is 1..6')
    return {'log': r - 1, 'truth': (2 * r - 2, 2 * r - 1)}


def run_s5x(runtime, components, starts, *, run, law, layers, world_predictor=None, logs_per_start=6,
            truth_rollouts=12, cap=12, repetitions=S5X_REPETITIONS, gamma=False, invariant=False, budget=None,
            deadline=None):
    """One registered S5-X run (``run`` = an entry of registry.S5X_RUNS): logs through the candidate
    runtime ledger, CRN truth, six complete repetitions, the S-2 Rao-Blackwellized surrogate bias when
    ``gamma`` and the gamma_a invariant when ``invariant`` (zeta = 0 runs)."""
    from . import ope_variants as ov
    from .ope_redesign_registry import world_seed
    from .policy_runtime import OUTSIDE_POLICY_SUPPORT, DecisionRequest
    from .policy_semisynthetic import generate_pa
    _require(runtime.components is components and runtime.improvement is not None, 'candidate runtime required')
    _require(logs_per_start == repetitions and truth_rollouts == 2 * repetitions, 'repetitions split the S5 budget')
    k = run['k']
    ws = world_seed(k)
    budget = runtime.improvement.simulator.budget if budget is None else budget
    world = QualityLayerWorld(components.inputs.pool, world_predictor or components.g0, components.we.terminal, budget,
                              layers)
    pas, ends, excluded, kept = {}, {}, [], []
    for i, (start, hand) in enumerate(starts):
        inside, reason = runtime.start_population(start, hand)
        if not inside:
            excluded.append({'start': i, 'reason': reason})
            continue
        kept.append(i)
        for j in range(logs_per_start):
            if deadline is not None:
                deadline()
            pa_id = f'w{i}:{j}'
            rng = np.random.default_rng(np.random.SeedSequence([ws, 0, i, j]))
            world.bind(np.random.default_rng(role_sequence('s5x_world', k, 0, i, j)).random((cap, 3)))
            states, actions, (end, value), last = generate_pa(runtime, world, start, law, rng, cap)
            rows = [runtime.submit(DecisionRequest(f'{pa_id}:{t}', pa_id, t, state, action, runtime.sha256, hand))
                    for t, (state, action) in enumerate(zip(states, actions))]
            _require(all(r['status'] in ('SUPPORTED', OUTSIDE_POLICY_SUPPORT) for r in rows),
                     'a generated in-E0 log was refused by the runtime')
            if end in ('absorbed', 'cap'):
                value = float(components.we.cutoff(last))
            pas[pa_id] = {'game': f'start{i}', 'in_population': True, 'reward': float(value), 'end_kind': end}
            ends[pa_id] = world.last if end == 'terminal' else None
    _require(kept, 'no registered start is inside E0')
    by_pa = {}
    for row in runtime.ledger.iter_decisions():
        by_pa.setdefault(row['pa_id'], []).append(row)
    lines = {pa_id: ov.pa_line(sorted(by_pa[pa_id], key=lambda d: d['decision_index']), info)
             for pa_id, info in pas.items()}
    # truth: CRN rollouts, the run_world stream layout, layer draws shared by both policies
    improvement = runtime.improvement
    play = getattr(runtime, 'planning_policy', None) or (lambda state: runtime.candidate(state)[0])
    policies = {'reference': lambda state, depth: (improvement.bc.actions, improvement.bc.probabilities(state)),
                'candidate': lambda state, depth: (improvement.bc.actions, play(state))}
    truth, g_sums = {}, {}
    gamma_cache, invariant_max = {}, 0.0
    for i in kept:
        if deadline is not None:
            deadline()
        seed = int(np.random.SeedSequence([ws, 1, i]).generate_state(1)[0])
        uniforms = np.random.default_rng(seed).random((truth_rollouts, cap, 3))
        layer = np.stack([np.random.default_rng(role_sequence('s5x_world', k, 1, i, r)).random((cap, 3))
                          for r in range(truth_rollouts)])
        values = {}
        for name, policy in policies.items():
            sums = np.zeros((truth_rollouts, 2))
            record = None
            if gamma or invariant:
                def record(r, depth, state, names, probs, sums=sums):
                    nonlocal invariant_max
                    for a, p in zip(names, probs):
                        if p <= 0:
                            continue
                        key = (state, a)
                        if key not in gamma_cache:
                            gamma_cache[key] = _gamma(components, world, state, a, budget)
                        g_a, g_b = gamma_cache[key]
                        invariant_max = max(invariant_max, abs(g_a))
                        sums[r] += p * np.array([g_a, g_b])
            values[name] = rollouts_x(world, [starts[i][0]] * truth_rollouts, policy, uniforms, layer,
                                      components.we.cutoff, record=record)
            g_sums[i, name] = sums
        truth[i] = values
    if invariant and invariant_max > INTEGRITY_TOL:
        raise IntegrityError(f'S-2 invariant violated: |gamma_a| = {invariant_max} > 1e-12 at zeta = 0 '
                             f'(deterministic integrity)')
    runtime.verify_components()
    # six complete repetitions over the fixed starts
    reps = []
    for r in range(1, repetitions + 1):
        streams = repetition_streams(r, repetitions)
        ids = [f'w{i}:{streams["log"]}' for i in kept]
        rep_lines = [lines[pa] for pa in ids]
        steps, lengths = _padded_lines(rep_lines)
        data = {'steps': steps, 'lengths': lengths, 'game': np.array(kept)}
        rewards = s5x_rewards(rep_lines, [ends[pa] for pa in ids], layers)
        estimates, selected, bad = pipeline(data, rewards)
        pick = list(streams['truth'])
        true_r = float(np.mean([np.mean(truth[i]['candidate'][pick] - truth[i]['reference'][pick]) for i in kept]))
        reps.append({'gaps': {p: estimates[p] - true_r for p in PAIRS}, 'truth': true_r, 'invalid': sorted(bad),
                     'selected': {n: ('inf' if v == INF else str(v)) for n, v in selected.items()}})
    gaps = {p: np.array([rep['gaps'][p] for rep in reps]) for p in PAIRS}
    invalid_reps = {p: sum(not math.isfinite(rep['gaps'][p]) or p in rep['invalid'] for rep in reps) for p in PAIRS}
    result = {'run': dict(run), 'world_seed': ws, 'starts': len(starts), 'kept_starts': len(kept),
              'excluded_starts': excluded, 'repetitions': reps, 'pairs': {}}
    for p in PAIRS:
        valid = invalid_reps[p] == 0 and np.isfinite(gaps[p]).all()
        result['pairs'][p] = {'gap': float(gaps[p].mean()) if valid else None,
                              'se': float(gaps[p].std(ddof=1) / math.sqrt(repetitions)) if valid else None,
                              'valid': bool(valid), 'invalid_repetitions': int(invalid_reps[p])}
    result['inherited_s5_se'] = _inherited_se(lines, ends, layers, kept, truth, logs_per_start)
    if gamma:
        result['s2'] = {}
        exact = invariant and run.get('zeta') == '0'  # the registered gamma_a = 0 invariant at zeta = 0
        for col, comp in ((0, 'R1_a'), (1, 'R1_b')):
            diffs = np.array([g_sums[i, 'candidate'][:, col] - g_sums[i, 'reference'][:, col] for i in kept])
            n = len(kept)
            b_hat = float(diffs.mean(axis=1).sum() / n)
            se = float(math.sqrt(float((diffs.std(axis=1, ddof=1) ** 2 / truth_rollouts).sum())) / n)
            result['s2'][comp] = {'b_hat': b_hat, 'se': se,
                                  **cell_outcome(b_hat, se, exact_zero_allowed=bool(exact and comp == 'R1_a'))}
        result['s2_pairs'] = s2_pair_cells(result['s2'], result['pairs'], k)
        result['first_order_reference_approximation'] = {
            'status': 'NOT_RUN', 'reason': 'sum_a (pi_c - pi_r)(a | s) gamma(s, a) along the reference rollouts needs the '
                                          'candidate policy (planner calls) at every reference-visited state; descriptive '
                                          'only in the design and omitted by the registered implementation choice'}
    result['invariant'] = None if not invariant else {'max_abs_gamma_a': invariant_max, 'outcome': 'PASS'}
    return result


def s2_pair_cells(s2, pairs, k):
    """Per R1_a / R1_b pair: the component's S-2 cell of run k only if the pair has no invalid repetition
    (SN normaliser, lambda / M selection or nonfinite estimate) in that run; otherwise INCONCLUSIVE with
    the invalid-repetition flag (PM decision, review #1: never PASS, the pair is not adoptable)."""
    out = {}
    for p in PAIRS:
        comp = p.split('x')[0]
        if comp not in ('R1_a', 'R1_b'):
            continue
        entry = pairs[p]
        out[p] = (dict(s2[comp]) if entry['valid'] else
                  inconclusive_invalid(f"{entry['invalid_repetitions']} invalid repetitions in run {k}"))
    return out


def _gamma(components, world, state, action, budget):
    """gamma_a and gamma_b at (s, a): the exact 400-draw in-play probability of the world predictor
    times the exact bin-sum surrogate gaps of the layers."""
    from .policy_identity import integrated_predictions
    predictor = world.predictor
    p = integrated_predictions(predictor, [state], [action])[0]
    budget.consume(len(components.inputs.pool(state, action, count_access=False).values))
    p_in_play = float(p[BIP10].sum())
    d_a, d_b = world.layers.gamma(state, world.layers.group(action))
    return p_in_play * d_a, p_in_play * d_b


def _inherited_se(lines, ends, layers, kept, truth, logs):
    """Descriptive only: the S5 per-run SE (within-start variances summed without cross-start
    covariance) of the plain R1_0 x R2_0 gap; never an acceptance input."""
    from .ope_variants import line_value
    per_start = []
    for i in kept:
        deltas = [line_value(lines[f'w{i}:{j}'])['delta'] for j in range(logs)]
        d = np.asarray(deltas)
        t = truth[i]['candidate'] - truth[i]['reference']
        per_start.append(d.std(ddof=1) ** 2 / len(d) + t.std(ddof=1) ** 2 / len(t))
    return float(math.sqrt(sum(per_start)) / len(per_start))


def nets(runs):
    """S-1 (X0-V2, k = 1..3) and S-4 (X2-V2, k = 6..8) pooled nets per pair (registry.net_outcome)."""
    from .ope_redesign_registry import net_outcome
    by_k = {r['run']['k']: r for r in runs}
    out = {}
    for name, ks in (('S1', (1, 2, 3)), ('S4', (6, 7, 8))):
        out[name] = {}
        for p in PAIRS:
            present = all(k in by_k for k in ks)
            entries = [by_k[k]['pairs'][p] for k in ks] if present else []
            valid = present and all(e['valid'] for e in entries)
            out[name][p] = net_outcome([e['gap'] for e in entries], [e['se'] for e in entries], valid=valid)
    return out


class SingleMemberPredictor:
    """X2 world predictor kind 'g0_single_member': one member of the bound frozen G0 family (its own May
    temperature, the same frozen frequency baseline and June blend), evaluated through exactly the
    FrozenGEnsemble.__call__ path with one member instead of five. The member seed is a registration pin."""
    def __init__(self, g0, seed):
        seeds = [m.seed for m in g0.models]
        _require(seed in seeds, f'G0 member seed {seed} is not bound')
        k = seeds.index(seed)
        self.g0, self.seed = g0, seed
        self.inputs, self.member, self.temperature = g0.inputs, g0.models[k], g0.temperatures[k]
        self.identity = {'kind': 'g0_single_member', 'seed': seed, 'g0_identity_sha256': g0.identity.get('sha256')}

    def __call__(self, states, actions, physical):
        from scipy.special import softmax
        from .rollout_policy import calibrated_conditional
        g0 = self.g0
        arrays = self.inputs.arrays(states, actions, physical)
        logits = np.stack([self.member.logits(arrays)])
        keys = [(s.context_key, s.balls, s.strikes, a) for s, a in zip(states, actions)]
        missing = {}
        for key, state, action in zip(keys, states, actions):
            if key not in g0.frequency_cache:
                missing[key] = self.inputs.query(state, action)
        if missing:
            import pandas as pd
            raw = g0.baseline.predict(pd.DataFrame(list(missing.values())))
            freq = softmax(np.log(np.clip(raw, 1e-12, 1)) / g0.baseline_temperature, axis=-1)
            g0.frequency_cache.update(zip(missing, freq))
        frequency = np.stack([g0.frequency_cache[key] for key in keys])
        p = calibrated_conditional(logits, [self.temperature], frequency, g0.neural_weight)
        _require(p.shape == (len(states), 10), 'X2 world predictor output shape')
        return p
