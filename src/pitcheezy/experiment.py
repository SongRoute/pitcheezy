"""실험 파이프라인 (P0): 준비 → 전이 텐서(홀드아웃·전체) → VI·완화 → π_b·IPS OPE → results/{ID}.json.

config 스키마는 configs/_template.yaml. 산출 runs/{ID}/s{seed}/ (transition·value 는 결정론적이라 s0 에만 두고 다른 시드는 참조).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pitcheezy.data import prepare as PR
from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces.re24 import RE24Table
from pitcheezy.interfaces.states import N_BASE_OUT, decode_state, n_states
from pitcheezy.interfaces.tensor import TransitionTensor
from pitcheezy.interfaces.validate import validate_transition, validate_value
from pitcheezy.interfaces.value import ValueBundle
from pitcheezy.ope import behavior as BH
from pitcheezy.ope import ips as IPS
from pitcheezy.policy import vi as VI
from pitcheezy.transition import count as TC

log = logging.getLogger("pitcheezy")
REPO = Path(__file__).resolve().parents[2]
HOLDOUT_SEASON_FLOOR = 2026  # 이 시즌부터는 학습·튜닝 금지


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return out + ("-dirty" if dirty else "")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["id"] != Path(path).stem:
        raise ValueError(f"config 파일명 {Path(path).stem} ≠ id {cfg['id']}")
    p = cfg["params"]
    for s in p["transition"]["train_seasons"] + p["transition"]["holdout"]["train_seasons"] + [p["transition"]["holdout"]["eval_season"]]:
        if int(s) >= HOLDOUT_SEASON_FLOOR:
            raise ValueError(f"{s} 시즌은 학습·튜닝에 쓸 수 없음 (OPE 전용)")
    return cfg


class Experiment:
    def __init__(self, cfg: dict, seed: int, data_dir: Path, runs_dir: Path):
        self.cfg, self.seed, self.data_dir, self.runs_dir = cfg, int(seed), Path(data_dir), Path(runs_dir)
        self.id = cfg["id"]
        self.p = cfg["params"]
        self.K = int(self.p["state"]["K"])
        self.collapse = bool(self.p["state"].get("collapse_base_out", False))
        self.seed_dir = self.runs_dir / self.id / f"s{self.seed}"
        # transition·value 는 결정론적 → s0 에만. transition.reuse_from 이 있으면 그 실험의 s0 을 그대로 쓴다 (예: 003 기준선은 001 텐서)
        self.s0_dir = self.runs_dir / self.p["transition"].get("reuse_from", self.id) / "s0"
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        self.commit = git_commit()
        self.pool_dir = REPO / "data" / "pitchers" / cfg["pool_version"]
        self.re24 = RE24Table.load(REPO / "data" / "re24" / cfg["re24_version"])
        pool = pd.read_parquet(self.pool_dir / "pitchers.parquet")
        self.pitcher_index = PR.pitcher_index_from_pool(pool)
        self.pool = pool
        self.n_p = len(pool)
        self.cluster_version = self.p["state"].get("cluster_version")
        self.clusters = None
        if self.cluster_version:
            from pitcheezy.data.batters import load_cluster_map
            self.clusters = load_cluster_map(REPO / "data" / "batters" / self.cluster_version)
            if self.K == 1:
                raise ValueError("cluster_version 이 있으면 K > 1 이어야 함")
        self.timing: dict[str, float] = {}

    # ------------------------------------------------------------ 데이터
    def pitches(self, season: int, *, holdout: bool = False) -> pd.DataFrame:
        return PR.load_or_prepare(self.runs_dir, self.data_dir, self.cfg["data_version"], self.cfg["pool_version"], season, self.pitcher_index, holdout=holdout,
                                  cluster_version=self.cluster_version, clusters=self.clusters)

    def sid(self, df: pd.DataFrame) -> np.ndarray:
        return PR.state_ids(df, self.K, collapse_base_out=self.collapse)

    def pitchers_table(self, train: pd.DataFrame) -> pd.DataFrame:
        n = train.groupby("pitcher_idx").size().reindex(range(self.n_p)).fillna(0).astype(np.int32)
        return pd.DataFrame({"pitcher_idx": np.arange(self.n_p, dtype=np.int32), "mlbam_id": self.pool["mlbam_id"].to_numpy(dtype=np.int64),
                             "name": self.pool["name"].to_numpy(dtype=object), "n_pitches_train": n.to_numpy()})

    def valid_states(self) -> np.ndarray | None:
        if not self.collapse:
            return None
        return decode_state(np.arange(n_states(self.K)), self.K)[1] == 0

    # ------------------------------------------------------------ 전이
    def stage_transition(self) -> dict:
        t0 = time.time()
        tp = self.p["transition"]
        out_h, out_f = self.s0_dir / "transition_holdout", self.s0_dir / "transition"
        if (out_f / "sha256.txt").exists() and (out_h / "meta.json").exists():
            log.info("전이 텐서 재사용 %s", out_f)
            meta = json.loads((out_f / "meta.json").read_text())
            return {"reused": True, **{k: meta.get(k) for k in ("alpha", "alpha_screen", "holdout_nll", "holdout_ece", "holdout_ece_hr", "excluded_pitchers", "holdout_metrics")}}
        common = {"data_version": self.cfg["data_version"], "pool_version": self.cfg["pool_version"], "seed": 0, "train_commit": self.commit,
                  "cluster_file_version": self.cluster_version or f"K{self.K}-none", "pitch_type_map_version": "v1", "collapse_base_out": self.collapse}
        # 홀드아웃: 2023–24 → 2025
        htr = pd.concat([self.pitches(s) for s in tp["holdout"]["train_seasons"]], ignore_index=True)
        hev = self.pitches(tp["holdout"]["eval_season"])
        present = set(htr["pitcher_idx"].unique())
        excluded = [int(m) for i, m in enumerate(self.pool["mlbam_id"]) if i not in present]
        hev = hev[hev["pitcher_idx"].isin(present)]
        pitchers_h = self.pitchers_table(htr)
        alphas = tp["alpha_grid"] if tp["alpha"] == "auto" else [tp["alpha"]]
        screen = {}
        best = None
        for a in alphas:
            t = TC.fit(htr, self.sid(htr), pitchers_h, self.K, alpha=float(a), repertoire_min=tp["repertoire_min_pitches"], valid_states=self.valid_states(),
                       meta={**common, "season_window": f"{tp['holdout']['train_seasons'][0]}-{tp['holdout']['train_seasons'][-1]}", "holdout_split": f"season:{tp['holdout']['eval_season']}", "excluded_pitchers": excluded})
            m = TC.holdout_metrics(t, hev, self.sid(hev))
            screen[str(a)] = m
            log.info("α=%s 홀드아웃 NLL %.4f ECE %.4f ECE_HR %.4f (n=%d)", a, m["holdout_nll"], m["holdout_ece"], m["holdout_ece_hr"], m["holdout_n_pitches"])
            if best is None or m["holdout_nll"] < best[1]["holdout_nll"]:
                best = (float(a), m, t)
        alpha, hm, th = best
        th.meta.update({k: hm[k] for k in ("holdout_nll", "holdout_ece", "holdout_ece_hr")})
        th.meta["holdout_metrics"] = hm
        th.meta["alpha_screen"] = {a: {"holdout_nll": v["holdout_nll"], "holdout_ece": v["holdout_ece"], "holdout_ece_hr": v["holdout_ece_hr"]} for a, v in screen.items()}
        pr = validate_transition(th)
        if pr:
            raise RuntimeError(f"홀드아웃 텐서 계약 위반: {pr}")
        if tp.get("save_holdout_tensor", True):
            th.save(out_h)
        else:  # 큰 텐서(K>1)는 meta·지표만 (D21). 재현은 같은 config 로 재실행
            out_h.mkdir(parents=True, exist_ok=True)
            (out_h / "meta.json").write_text(json.dumps({**th.meta, "tensor_saved": False}, ensure_ascii=False, indent=2))
        del th
        # 전체: 2023–25
        ftr = pd.concat([self.pitches(s) for s in tp["train_seasons"]], ignore_index=True)
        tf = TC.fit(ftr, self.sid(ftr), self.pitchers_table(ftr), self.K, alpha=alpha, repertoire_min=tp["repertoire_min_pitches"], valid_states=self.valid_states(),
                    meta={**common, "season_window": f"{tp['train_seasons'][0]}-{tp['train_seasons'][-1]}", "holdout_split": "none",
                          "holdout_nll": hm["holdout_nll"], "holdout_ece": hm["holdout_ece"], "holdout_ece_hr": hm["holdout_ece_hr"], "holdout_metrics": hm,
                          "holdout_tensor_dir": str(out_h), "alpha_screen": th_screen(screen), "excluded_pitchers": excluded})
        pr = validate_transition(tf)
        if pr:
            raise RuntimeError(f"전체 텐서 계약 위반: {pr}")
        tf.save(out_f)
        self.timing["transition"] = time.time() - t0
        log.info("전이 텐서 저장 %s (α=%s, %.0fs)", out_f, alpha, self.timing["transition"])
        return {"reused": False, "alpha": alpha, "alpha_screen": tf.meta["alpha_screen"], "holdout_metrics": hm, "excluded_pitchers": excluded,
                "holdout_nll": hm["holdout_nll"], "holdout_ece": hm["holdout_ece"], "holdout_ece_hr": hm["holdout_ece_hr"]}

    # ------------------------------------------------------------ 가치
    def stage_value(self) -> dict:
        t0 = time.time()
        pp = self.p["policy"]
        out = self.s0_dir / "value"
        if (out / "sha256.txt").exists():
            log.info("가치함수 재사용 %s", out)
            return {"reused": True, **json.loads((out / "meta.json").read_text()).get("vi", {})}
        t = TransitionTensor.load(self.s0_dir / "transition", mmap=True)
        R = VI.reward_table(self.re24.dRE24, self.K)
        nxt = VI.next_state_table(self.K)
        Q, V, iters, delta = VI.value_iteration(t.P, t.valid, R, nxt)
        kind = pp.get("kind", "softmax")
        # tilt 는 평가 시즌 π_b 가 필요해 여기서 못 만든다 → value/policy.npy 에는 같은 τ 의 softmax 를 저장 (meta.relax 에 기록)
        pol = VI.relax(Q, t.valid, method=kind if kind in ("softmax", "topk", "greedy") else "softmax", temperature=float(pp.get("temperature", 0.05)), top_k=int(pp.get("top_k", 5)))
        sha = dict(line.split()[::-1] for line in (self.s0_dir / "transition" / "sha256.txt").read_text().splitlines() if line.strip())
        meta = {"transition_dir": str(self.s0_dir / "transition"), "transition_sha256": sha, "re24_version": self.cfg["re24_version"], "dre24_version": self.cfg["re24_version"],
                "terminal_reward": "-dRE24[outcome, base_out] (투수 관점)", "in_play_reward": "actual_league_mean (ASM-7)", "gamma": 1,
                "relax": {"method": kind if kind in ("softmax", "topk", "greedy") else f"softmax (stored stand-in for {kind}; tilt is built in OPE stage)", "temperature": pp.get("temperature"), "top_k": pp.get("top_k")}, "lookup_mode": "snap", "seed": 0, "train_commit": self.commit,
                "vi": {"iters": int(iters), "max_delta": float(delta), "V_mean": float(V.mean()), "V_min": float(V.min()), "V_max": float(V.max()),
                       "n_states_no_valid_action": int((~t.valid.any(-1)).sum())}}
        vb = ValueBundle(Q=Q.astype(np.float32), V=V.astype(np.float32), policy=pol, meta=meta)
        pr = validate_value(vb, t.valid)
        if pr:
            raise RuntimeError(f"가치함수 계약 위반: {pr}")
        vb.save(out)
        self.timing["value"] = time.time() - t0
        log.info("VI %d회 수렴 (Δ=%.1e) V 평균 %.4f, %.0fs", iters, delta, V.mean(), self.timing["value"])
        return {"reused": False, **meta["vi"]}

    # ------------------------------------------------------------ OPE
    def policy_menu(self, vb: ValueBundle, valid: np.ndarray) -> dict:
        """평가할 정책 메뉴 (모든 실험 공통, 싸다). 값: None = π_b 자체, "tilt" = 폴드별 π_b 로 stage_ope 에서 생성, callable = 지연 생성."""
        op = self.p["ope"]
        Q = vb.Q.astype(np.float64)
        menu: dict = {}
        for kind in op.get("menu", ["primary"]):
            if kind == "primary":
                kind = self.primary_name()
            if kind in menu:
                continue
            if kind == "behavior":
                menu[kind] = None
            elif kind == "uniform":
                menu[kind] = lambda: VI.relax(Q, valid, method="uniform")
            elif kind == "greedy":
                menu[kind] = lambda: VI.relax(Q, valid, method="greedy")
            elif kind.startswith("topk"):
                menu[kind] = (lambda k=int(kind[4:]): VI.relax(Q, valid, method="topk", top_k=k))
            elif kind.startswith("softmax_t"):
                menu[kind] = (lambda t=float(kind[9:]): VI.relax(Q, valid, method="softmax", temperature=t))
            elif kind.startswith("tilt_t"):
                menu[kind] = "tilt"
            else:
                raise ValueError(f"메뉴 항목 모름: {kind}")
        return menu

    def primary_name(self) -> str:
        pp = self.p["policy"]
        k = pp.get("kind", "softmax")
        if k in ("softmax", "tilt"):
            return f"{k}_t{pp['temperature']}"
        if k == "topk":
            return f"topk{pp['top_k']}"
        return k

    def eval_support(self, ev: pd.DataFrame, valid: np.ndarray) -> np.ndarray:
        """공통 지지 [P,S,A] = valid ∩ (평가 시즌 투수×구종 투구 수 ≥ support_min_pitches_eval)."""
        from pitcheezy.transition.count import repertoire_counts
        from pitcheezy.interfaces.grid import decode_action
        rep = repertoire_counts(ev, self.n_p) >= int(self.p["ope"].get("support_min_pitches_eval", 0))  # [P, 9]
        group = decode_action(np.arange(valid.shape[-1]))[0]
        return valid & rep[:, group][:, None, :]

    def stage_ope(self) -> dict:
        t0 = time.time()
        op = self.p["ope"]
        out = self.seed_dir / "ope"
        out.mkdir(exist_ok=True)
        ev = self.pitches(op["eval_season"], holdout=True).reset_index(drop=True)
        sid = self.sid(ev)
        t = TransitionTensor.load(self.s0_dir / "transition", check_hash=False, mmap=True)
        vb = ValueBundle.load(self.s0_dir / "value", check_hash=False)
        support = self.eval_support(ev, t.valid)
        menu = self.policy_menu(vb, t.valid)
        tilts = {name: (vb.Q, support, float(name[6:])) for name, pol in menu.items() if isinstance(pol, str) and pol == "tilt"}
        groups = IPS.coarse_groups()
        pb_logged, pe_tilt, pb_coarse, pe_tilt_coarse = BH.crossfit_logged(ev, sid, self.n_p, self.K, alpha=float(op["behavior_alpha"]), n_folds=int(op["n_folds"]), tilts=tilts, groups=groups)
        pb_full = BH.fit_behavior(ev, sid, self.n_p, self.K, alpha=float(op["behavior_alpha"]))  # 진단(모델 내 가치)용
        pa = PR.pa_rewards(ev, self.re24.RE24)
        keep = pa["n_pitchers"] == 1
        pa = pa[keep].reset_index(drop=True)
        r = -pa["delta_re24"].to_numpy()  # 투수 관점
        games = pa["game_pk"].to_numpy()
        key = pa[["game_pk", "at_bat_number"]]
        R = VI.reward_table(self.re24.dRE24, self.K)
        nxt = VI.next_state_table(self.K)
        first = ev.groupby(["game_pk", "at_bat_number"], sort=False).head(1)
        f_p, f_s = first["pitcher_idx"].to_numpy(dtype=np.int64), self.sid(first)
        has_a = ev["action_id"] >= 0
        two_strike = (ev["count_id"].to_numpy() % 3 == 2)
        n_dec = key.merge(ev[has_a].groupby(["game_pk", "at_bat_number"]).size().rename("n").reset_index(), on=["game_pk", "at_bat_number"], how="left")["n"].fillna(0).to_numpy(dtype=float)
        n_dec_2s = key.merge(ev[has_a & two_strike].groupby(["game_pk", "at_bat_number"]).size().rename("n").reset_index(), on=["game_pk", "at_bat_number"], how="left")["n"].fillna(0).to_numpy(dtype=float)
        results = {}
        variants = [("traj_clip", "w_traj", op.get("clip"), "fine"), ("traj_noclip", "w_traj", None, "fine"), ("onestep_clip", "w_onestep", op.get("clip"), "fine"),
                    ("onestep2s_clip", "w_onestep_slice", op.get("clip"), "fine"), ("traj_coarse_clip", "w_traj", op.get("clip"), "coarse"), ("onestep_coarse_clip", "w_onestep", op.get("clip"), "coarse")]
        for name, pol in menu.items():
            model_value = None
            if isinstance(pol, str):  # tilt
                pol_arr = BH.tilt(pb_full, vb.Q, support, tilts[name][2])
                pe_logged, pe_logged_coarse = pe_tilt[name], pe_tilt_coarse[name]
            elif pol is not None:
                pol_arr = IPS.restrict_support(pol() if callable(pol) else pol, support)
                pe_logged = IPS.logged_probs(ev, sid, pol_arr)
                pe_logged_coarse = IPS.logged_probs(ev, sid, IPS.coarsen(pol_arr, groups))
            else:
                pol_arr = None
            if pol_arr is not None:
                model_value = float(VI.policy_evaluation(t.P, pol_arr, R, nxt)[f_p, f_s].mean())
                pol_arr = pol_arr.astype(np.float32)
            for vname, col, clip, level in variants:
                if pol_arr is None:  # π_b 자체: 궤적은 타석당 1, 1스텝은 결정 수 (다른 정책과 같은 가중 방식)
                    if level == "coarse":
                        continue
                    w = np.ones(len(pa)) if col == "w_traj" else (n_dec_2s if col == "w_onestep_slice" else n_dec); nd = None
                else:
                    pe_l, pb_l = (pe_logged_coarse, pb_coarse) if level == "coarse" else (pe_logged, pb_logged)
                    pw = key.merge(IPS.pa_weights(ev, pe_l, pb_l, clip=clip, slice_mask=two_strike), on=["game_pk", "at_bat_number"], how="left")
                    w = pw[col].fillna(1.0 if col == "w_traj" else 0.0).to_numpy(); nd = pw
                est = IPS.estimate(w, r)
                est.update(IPS.bootstrap(w, r, games, n_boot=int(op["n_boot"]), seed=self.seed))
                est["model_value"] = model_value
                if nd is not None:
                    est["n_decisions_mean"] = float(nd["n_decisions"].mean()); est["frac_pa_with_zero_rho"] = float((nd["n_zero"] > 0).mean())
                results[f"{name}/{vname}"] = est
                log.info("OPE %-16s %-12s SN %+.4f [%+.4f, %+.4f] wmean %8.3f ESS %5.1f%% wmax %9.1f zero %4.1f%% model %s",
                         name, vname, est["snips"], est["ci_low"], est["ci_high"], est["w_mean"], 100 * est["ess_frac"], est["w_max"], 100 * est["frac_zero_w"],
                         "-" if model_value is None else f"{model_value:+.4f}")
                if pol_arr is None and vname == "traj_noclip":
                    results.pop(f"{name}/{vname}", None)
        summary = {"seed": self.seed, "eval_season": op["eval_season"], "n_pa": int(len(pa)), "n_pa_dropped_multi_pitcher": int((~keep).sum()), "n_games": int(len(np.unique(games))),
                   "n_pitches": int(len(ev)), "frac_pitches_no_action": float((ev["action_id"] < 0).mean()),
                   "support_min_pitches_eval": op.get("support_min_pitches_eval", 0), "frac_support_actions": float(support.sum() / max(t.valid.sum(), 1)),
                   "mean_reward_behavior": float(r.mean()), "behavior_alpha": op["behavior_alpha"], "n_folds": op["n_folds"], "clip": op.get("clip"), "n_boot": op["n_boot"],
                   "primary": f"{self.primary_name()}/{op.get('primary_variant', 'onestep_clip')}", "results": results}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        self.timing["ope"] = time.time() - t0
        return summary

    # ------------------------------------------------------------ 실행
    def run(self) -> dict:
        log.info("=== %s seed %d commit %s ===", self.id, self.seed, self.commit)
        tr = self.stage_transition()
        va = self.stage_value()
        op = self.stage_ope()
        rec = {"id": self.id, "seed": self.seed, "commit": self.commit, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "transition": tr, "value": va, "ope": op, "timing_seconds": self.timing}
        (self.seed_dir / "run.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
        return rec


def th_screen(screen: dict) -> dict:
    return {a: {"holdout_nll": v["holdout_nll"], "holdout_ece": v["holdout_ece"], "holdout_ece_hr": v["holdout_ece_hr"]} for a, v in screen.items()}


def aggregate(cfg: dict, runs_dir: Path, results_dir: Path) -> dict:
    """runs/{ID}/s*/run.json → results/{ID}.json. 정책×클립별로 시드 평균·CI 범위."""
    runs = []
    for d in sorted((Path(runs_dir) / cfg["id"]).glob("s*/run.json")):
        runs.append(json.loads(d.read_text()))
    if not runs:
        raise FileNotFoundError("run.json 없음")
    r0 = runs[0]
    primary = r0["ope"].get("primary")
    pol = {}
    for name in r0["ope"]["results"]:
        per = {str(r["seed"]): r["ope"]["results"][name] for r in runs if name in r["ope"]["results"]}
        pol[name] = {"point_snips": r0["ope"]["results"][name]["snips"], "point_ips": r0["ope"]["results"][name]["ips"],
                     "ci_low_min": min(v["ci_low"] for v in per.values()), "ci_high_max": max(v["ci_high"] for v in per.values()),
                     "boot_std_mean": float(np.mean([v["boot_std"] for v in per.values()])), "ess_frac": r0["ope"]["results"][name]["ess_frac"],
                     "w_max": r0["ope"]["results"][name]["w_max"], "w_mean": r0["ope"]["results"][name]["w_mean"], "frac_zero_w": r0["ope"]["results"][name]["frac_zero_w"],
                     "model_value": r0["ope"]["results"][name].get("model_value"), "seeds": per}
    out = {"id": cfg["id"], "phase": cfg["phase"], "baseline": cfg.get("baseline"), "change": cfg.get("change"), "data_version": cfg["data_version"],
           "pool_version": cfg["pool_version"], "re24_version": cfg["re24_version"], "commits": sorted({r["commit"] for r in runs}), "seeds": [r["seed"] for r in runs],
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "params": cfg["params"],
           "transition": {k: r0["transition"].get(k) for k in ("alpha", "alpha_screen", "holdout_nll", "holdout_ece", "holdout_ece_hr", "excluded_pitchers")},
           "holdout_metrics": r0["transition"].get("holdout_metrics"), "value": r0["value"],
           "ope": {"eval_season": r0["ope"]["eval_season"], "n_pa": r0["ope"]["n_pa"], "n_games": r0["ope"]["n_games"], "mean_reward_behavior": r0["ope"]["mean_reward_behavior"],
                   "n_pitches": r0["ope"].get("n_pitches"), "support_min_pitches_eval": r0["ope"].get("support_min_pitches_eval"), "clip": r0["ope"]["clip"], "n_folds": r0["ope"].get("n_folds"), "n_boot": r0["ope"]["n_boot"], "primary": primary, "policies": pol},
           "timing_seconds": r0["timing_seconds"]}
    Path(results_dir).mkdir(exist_ok=True)
    (Path(results_dir) / f"{cfg['id']}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def log_wandb(cfg: dict, rec: dict) -> None:
    w = cfg.get("wandb") or {}
    if w.get("mode", "online") == "disabled":
        return
    try:
        import wandb
        run = wandb.init(project=w.get("project", "pitcheezy"), name=f"{cfg['id']}/s{rec['seed']}", config=cfg, mode=w.get("mode", "online"),
                         settings=wandb.Settings(init_timeout=60), reinit=True)
        flat = {"holdout/nll": rec["transition"].get("holdout_nll"), "holdout/ece": rec["transition"].get("holdout_ece"), "holdout/ece_hr": rec["transition"].get("holdout_ece_hr"), "alpha": rec["transition"].get("alpha")}
        for name, est in rec["ope"]["results"].items():
            for k in ("snips", "ips", "ci_low", "ci_high", "ess_frac", "w_max"):
                flat[f"ope/{name}/{k}"] = est[k]
        run.log(flat)
        run.finish()
    except Exception as exc:  # noqa: BLE001
        log.warning("W&B 기록 실패 (무시): %s", exc)
