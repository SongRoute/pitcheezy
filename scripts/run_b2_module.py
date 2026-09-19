"""B2 모듈 층 재구현 실행 (EXP-P1-006, docs/baselines.md B2).

    .venv/bin/python scripts/run_b2_module.py --config configs/EXP-P1-006.yaml --seed 0
    .venv/bin/python scripts/run_b2_module.py --config configs/EXP-P1-006.yaml --aggregate

2023–24 학습 → 2025 홀드아웃 NLL·ECE·ECE_HR·top-1. 텐서·VI·OPE 없음 (정책 층은 범위 밖).
지표는 두 행 집합에서 낸다: b2_rows(적격 2025 전부) / common_rows(그중 기준 실험의 valid[p,s,a] 가 True 인 행)
— common_rows 에서는 기준 실험(EXP-P0-006)의 홀드아웃 텐서로 우리 모델 지표도 다시 계산해 같은 행에서 비교한다.
산출 $PITCHEEZY_RUNS_DIR/{ID}{suffix}/s{seed}/b2/{model.pt, meta.json, run.json}, 요약 results/{ID}.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pitcheezy.baselines import b2 as B2  # noqa: E402
from pitcheezy.data import prepare as PR  # noqa: E402
from pitcheezy.experiment import HOLDOUT_SEASON_FLOOR, git_commit  # noqa: E402
from pitcheezy.interfaces import outcomes as O  # noqa: E402

log = logging.getLogger("pitcheezy")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["id"] != Path(path).stem:
        raise ValueError(f"config 파일명 {Path(path).stem} ≠ id {cfg['id']}")
    p = cfg["params"]
    for s in list(p["train_seasons"]) + [p["eval_season"]]:
        if int(s) >= HOLDOUT_SEASON_FLOOR:
            raise ValueError(f"{s} 시즌은 학습·튜닝에 쓸 수 없음 (OPE 전용)")
    return cfg


def concat(dss: list[B2.Dataset]) -> B2.Dataset:
    return B2.Dataset(**{k: np.concatenate([getattr(d, k) for d in dss]) for k in B2.Dataset.__slots__})


def build(cfg: dict, data_dir: Path, runs_dir: Path, window: int):
    """(train_ds, eval_ds, 메타). 학습 = params.train_seasons, 평가 = eval_season (학습에 나온 투수만)."""
    pool = pd.read_parquet(REPO / "data" / "pitchers" / cfg["pool_version"] / "pitchers.parquet")
    pindex = PR.pitcher_index_from_pool(pool)
    p = cfg["params"]
    seasons = sorted({*map(int, p["train_seasons"]), int(p["eval_season"])})
    stats = {s: B2.load_batter_stats(data_dir, cfg["data_version"], s) for s in seasons}
    means = {s: B2.league_means(stats[s]) for s in seasons}

    def season_ds(season: int) -> B2.Dataset:
        ids = PR.load_or_prepare(runs_dir, data_dir, cfg["data_version"], cfg["pool_version"], season, pindex)
        raw = B2.load_raw_features(data_dir, cfg["data_version"], season)
        prev = stats.get(season - 1)  # 2023 행은 2022 가 없다 → 리그 평균 + 결측 플래그
        return B2.build_dataset(ids, raw, prev, means[season if prev is None else season - 1], window)

    tr = concat([season_ds(s) for s in map(int, p["train_seasons"])])
    ev = season_ds(int(p["eval_season"]))
    present = np.isin(ev.pitcher_idx, np.unique(tr.pitcher_idx))  # 006·001 과 같은 규칙: 학습 창에 없는 투수는 평가에서 뺀다
    excluded = [int(m) for i, m in enumerate(pool["mlbam_id"]) if i not in set(np.unique(tr.pitcher_idx).tolist())]
    ev = ev.take(present)
    meta = {"n_train_rows": len(tr), "n_eval_rows": len(ev), "n_eval_rows_dropped_pitcher_absent": int((~present).sum()),
            "excluded_pitchers": excluded, "prev_season_missing_frac_eval": float(ev.ctx[:, B2.CTX_NAMES.index("prev_season_missing")].mean()),
            "prev_season_missing_frac_train": float(tr.ctx[:, B2.CTX_NAMES.index("prev_season_missing")].mean())}
    return tr, ev, meta


def ref_rows(runs_dir: Path, ref_id: str, ev: B2.Dataset):
    """기준 실험의 홀드아웃 valid[p,s,a] → common 마스크. 가능하면 P.npy 로 기준 모델 확률도 gather."""
    hold, full = Path(runs_dir) / ref_id / "s0" / "transition_holdout", Path(runs_dir) / ref_id / "s0" / "transition"
    src = hold if (hold / "valid.npy").exists() else full
    if not (src / "valid.npy").exists():
        return None, {"ref_valid_source": None, "note": f"{ref_id} 의 valid.npy 없음 — common_rows 계산 불가"}
    valid = np.load(src / "valid.npy", mmap_mode="r")
    common = np.asarray(valid[ev.pitcher_idx, ev.state_id, ev.action_id], dtype=bool)
    info = {"ref_valid_source": str(src), "ref_valid_is_holdout_window": src == hold, "n_common_rows": int(common.sum())}
    probs = None
    if (hold / "P.npy").exists():
        P = np.load(hold / "P.npy", mmap_mode="r")
        p, s, a = ev.pitcher_idx[common], ev.state_id[common], ev.action_id[common]
        probs = np.empty((len(p), O.N_OUTCOMES), dtype=np.float64)
        for i in range(0, len(p), 200_000):
            probs[i:i + 200_000] = P[p[i:i + 200_000], s[i:i + 200_000], a[i:i + 200_000]]
        info["ref_P_source"] = str(hold / "P.npy")
    else:
        info["ref_P_source"] = None
        info["ref_P_note"] = "홀드아웃 텐서 P.npy 가 없어 기준 모델 지표는 생략 (전체 창 텐서는 2025 를 학습해 쓸 수 없음)"
    return common, {**info, "probs": probs}


def run_seed(cfg: dict, seed: int, data_dir: Path, runs_dir: Path, out_id: str, max_rows: int | None, max_epochs: int | None) -> dict:
    t0 = time.time()
    hp = B2.hparams(cfg["params"].get("b2"))
    timing: dict[str, float] = {}
    td = time.time()
    tr, ev, dmeta = build(cfg, data_dir, runs_dir, int(hp["window"]))
    timing["build"] = time.time() - td
    if max_rows:  # 스모크용. 학습·평가 모두 줄인다 (결과는 results/ 에 넣지 않는다)
        rng = np.random.default_rng(0)
        if len(tr) > max_rows:
            tr = tr.take(np.sort(rng.choice(len(tr), max_rows, replace=False)))
        if len(ev) > max_rows:
            ev = ev.take(np.sort(rng.choice(len(ev), max_rows, replace=False)))
        dmeta["smoke_max_rows"] = int(max_rows)
    sc = B2.Scaler.fit(tr)
    tr, ev = sc.apply(tr), sc.apply(ev)
    log.info("b2 학습 %d행 / 평가 %d행 (전 시즌 결측 학습 %.3f 평가 %.3f)", len(tr), len(ev), dmeta["prev_season_missing_frac_train"], dmeta["prev_season_missing_frac_eval"])
    tt = time.time()
    net, info = B2.train(tr, ev, hp, seed, max_epochs=max_epochs)
    timing["train"] = time.time() - tt
    dev = B2.pick_device(hp["device"])
    probs = B2.predict_probs(net, ev, dev)
    m_all = B2.metrics(probs, ev.y)
    log.info("b2 홀드아웃(b2_rows) NLL %.4f ECE %.4f ECE_HR %.4f top1 %.4f (n=%d, 에폭 %d)",
             m_all["holdout_nll"], m_all["holdout_ece"], m_all["holdout_ece_hr"], m_all["holdout_accuracy_top1"], m_all["holdout_n_pitches"], info["best_epoch"])
    common, cinfo = ref_rows(runs_dir, cfg["params"]["ref_experiment"], ev)
    hm = {"b2_rows": m_all}
    if common is not None and common.any():
        hm["common_rows"] = {"b2": B2.metrics(probs[common], ev.y[common])}
        if cinfo.get("probs") is not None:
            hm["common_rows"][f"ref_{cfg['params']['ref_experiment']}"] = B2.metrics(cinfo.pop("probs"), ev.y[common])
        r = hm["common_rows"].get(f"ref_{cfg['params']['ref_experiment']}")
        log.info("common_rows n=%d  b2 NLL %.4f / 기준 NLL %s", int(common.sum()), hm["common_rows"]["b2"]["holdout_nll"], f"{r['holdout_nll']:.4f}" if r else "없음")
    cinfo.pop("probs", None)
    out = Path(runs_dir) / out_id / f"s{seed}" / "b2"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "hp": hp, "scaler": sc.to_meta()}, out / "model.pt")
    meta = {"arch": "b2_transformer", "id": cfg["id"], "seed": seed, "commit": git_commit(), "data_version": cfg["data_version"], "pool_version": cfg["pool_version"],
            "train_seasons": cfg["params"]["train_seasons"], "eval_season": cfg["params"]["eval_season"], "b2": hp, "scaler": sc.to_meta(),
            "train_info": info, "ref": cinfo, **dmeta}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    timing["total"] = time.time() - t0
    rec = {"id": cfg["id"], "seed": seed, "commit": meta["commit"], "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "transition": {"arch": "b2_transformer", "holdout_nll": m_all["holdout_nll"], "holdout_ece": m_all["holdout_ece"], "holdout_ece_hr": m_all["holdout_ece_hr"],
                          "n_epochs": int(info["best_epoch"]), "b2": hp, "excluded_pitchers": dmeta["excluded_pitchers"]},
           "holdout_metrics": hm, "ref": cinfo, "data": {k: v for k, v in dmeta.items() if k != "excluded_pitchers"},
           "value": None, "ope": {"note": "모듈층만 — OPE 해당 없음"}, "timing_seconds": timing}
    (out / "run.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    return rec


def aggregate(cfg: dict, runs_dir: Path, results_dir: Path) -> dict:
    runs = [json.loads(p.read_text()) for p in sorted((Path(runs_dir) / cfg["id"]).glob("s*/b2/run.json"))]
    if not runs:
        raise FileNotFoundError("run.json 없음")
    r0 = runs[0]
    out = {"id": cfg["id"], "phase": cfg["phase"], "baseline": cfg.get("baseline"), "change": cfg.get("change"), "data_version": cfg["data_version"],
           "pool_version": cfg["pool_version"], "re24_version": cfg["re24_version"], "commits": sorted({r["commit"] for r in runs}),
           "seeds": [r["seed"] for r in runs], "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "params": cfg["params"],
           "transition": r0["transition"], "holdout_nll_by_seed": {str(r["seed"]): r["transition"]["holdout_nll"] for r in runs},
           "holdout_metrics": r0["holdout_metrics"], "ref": r0["ref"], "data": r0["data"],
           "value": None, "ope": {"note": "모듈층만 — OPE 해당 없음"}, "timing_seconds": r0["timing_seconds"]}
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
        t, hm = rec["transition"], rec["holdout_metrics"]
        flat = {"holdout/nll": t["holdout_nll"], "holdout/ece": t["holdout_ece"], "holdout/ece_hr": t["holdout_ece_hr"],
                "holdout/top1": hm["b2_rows"]["holdout_accuracy_top1"], "holdout/n_pitches": hm["b2_rows"]["holdout_n_pitches"], "n_epochs": t["n_epochs"]}
        for k, v in (hm.get("common_rows") or {}).items():
            flat[f"common/{k}/nll"] = v["holdout_nll"]
            flat[f"common/{k}/ece"] = v["holdout_ece"]
        run.log(flat)
        run.finish()
    except Exception as exc:  # noqa: BLE001
        log.warning("W&B 기록 실패 (무시): %s", exc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true", help="시드 실행 없이 results/{ID}.json 만 갱신")
    ap.add_argument("--data-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_DATA_DIR", REPO / "data")))
    ap.add_argument("--runs-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_RUNS_DIR", REPO / "runs")))
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None, help="스모크: 학습·평가 행 수 상한 (결과를 results/ 에 넣지 않는다)")
    ap.add_argument("--max-epochs", type=int, default=None, help="스모크: config 의 max_epochs 를 덮어쓴다")
    ap.add_argument("--run-id-suffix", type=str, default="", help="스모크: 산출 디렉터리를 {ID}{suffix} 로 (실제 실행 디렉터리를 더럽히지 않게)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    cfg = load_config(a.config)
    if a.no_wandb:
        cfg.setdefault("wandb", {})["mode"] = "disabled"
    smoke = bool(a.run_id_suffix or a.max_rows or a.max_epochs)
    if not a.aggregate:
        if a.seed is None:
            ap.error("--seed 필요 (또는 --aggregate)")
        rec = run_seed(cfg, a.seed, a.data_dir, a.runs_dir, cfg["id"] + a.run_id_suffix, a.max_rows, a.max_epochs)
        log_wandb(cfg, rec)
        if smoke:
            m = rec["holdout_metrics"]["b2_rows"]
            print(f"SMOKE {cfg['id']}{a.run_id_suffix} s{a.seed}  NLL {m['holdout_nll']:.4f} ECE {m['holdout_ece']:.4f} top1 {m['holdout_accuracy_top1']:.4f} "
                  f"n={m['holdout_n_pitches']}  epochs={rec['transition']['n_epochs']}  {rec['timing_seconds']}")
            return 0
    out = aggregate(cfg, a.runs_dir, REPO / "results")
    m = out["holdout_metrics"]["b2_rows"]
    print(f"OK results/{cfg['id']}.json  seeds={out['seeds']}  b2_rows: NLL {m['holdout_nll']:.4f} ECE {m['holdout_ece']:.4f} top1 {m['holdout_accuracy_top1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
