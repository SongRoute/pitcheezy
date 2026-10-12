"""Reviewed C/base package materializer; source projection before role/filter construction.

This is a separate GPU-run data job, never one of the 87 training slots. The
grant must pin the original S2 parquet/manifest, approved scale source bundle,
base input package, C plan, memory/wall bounds and fresh output directory.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from experiments.ws_campaign import data, runtime
from experiments.ws_seq import train_seq as base
from experiments.ws_seq.train_tm002 import supervise_command


def source_path(spec):
    path = Path(spec["path"])
    if not path.is_absolute():
        path = runtime.ROOT / path
        runtime.require(path.resolve().is_relative_to(runtime.ROOT), "source path escapes snapshot")
    return path


def checked_grant(path, window):
    grant = json.loads(path.read_text())
    runtime.require(grant.get("execute") is True and grant.get("verdict") == "PASS"
                    and grant.get("distinct_from_author") is True
                    and grant.get("author") and grant.get("reviewer")
                    and grant["author"] != grant["reviewer"]
                    and grant.get("window") == window
                    and grant.get("sources_sha256") == runtime.source_hashes(), "independent data preparation grant")
    runtime.require(grant.get("data_role") == "2026 EXPOSED DEVELOPMENT"
                    and grant.get("data_version") == "d20260930-h2026f", "data role/version")
    runtime.require(grant.get("max_wall_seconds", 0) > 0 and grant.get("max_rss_bytes", 0) > 0,
                    "preparation caps")
    plan_path = runtime.ROOT / "experiments/ws_campaign/C-PLAN.json"
    runtime.require(runtime.sha(plan_path) == grant.get("plan_sha256"), "C plan hash")
    for name in ("scale_registration", "source_parquet", "source_manifest", "base_inputs_manifest",
                 "exposure_ledger"):
        spec = grant["inputs"][name]
        location = source_path(spec) if name == "scale_registration" else Path(spec["path"])
        runtime.require(location.is_absolute() and len(spec["sha256"]) == 64,
                        "pinned preparation input descriptor " + name)
    for name, spec in grant["scale_sources"].items():
        location = source_path(spec)
        runtime.require(location.is_absolute() and len(spec["sha256"]) == 64,
                        "pinned source descriptor " + name)
    runtime.require(Path(grant["output"]).is_absolute() and not Path(grant["output"]).exists(),
                    "fresh absolute output")
    plan = json.loads(plan_path.read_text())
    bound_plan = dict(plan, source_sha256=grant["inputs"]["source_parquet"]["sha256"],
                      exposure_ledger_sha256=grant["inputs"]["exposure_ledger"]["sha256"],
                      codec_source_sha256=grant["inputs"]["scale_registration"]["sha256"])
    data.validate_plan(bound_plan)
    runtime.require(grant.get("gpu_index") is not None and grant.get("gpu_uuid")
                    and Path(grant.get("common_lock", "")).is_absolute()
                    and grant.get("min_free_host_bytes", 0) >= grant["max_rss_bytes"],
                    "preparation resource ownership")
    return grant, plan


def load_scale(grant):
    """Load the exact reviewed closure from staged, hash-pinned original files."""
    files = grant["scale_sources"]
    reg = json.loads(source_path(grant["inputs"]["scale_registration"]).read_text())
    for name, old in reg["pinned"].items():
        runtime.require(files[name]["sha256"] == old[1], "original pinned helper " + name)
        old[0] = str(source_path(files[name]))
    source = reg["source"]
    runtime.require(grant["inputs"]["source_parquet"]["sha256"] == source["parquet_sha256"]
                    and grant["inputs"]["source_manifest"]["sha256"] == source["manifest_sha256"],
                    "original S2 source bytes")
    source["parquet"] = grant["inputs"]["source_parquet"]["path"]
    source["manifest"] = grant["inputs"]["source_manifest"]["path"]
    spec = importlib.util.spec_from_file_location("campaign_scale_runner", source_path(files["scale_runner"]))
    scale = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scale)
    scale.HERE = source_path(files["scale_safe"]).parent
    M = scale.closure(reg)
    return scale, M, reg


def package(grant, plan, window):
    import gc
    scale, M, original = load_scale(grant)
    raw, binding = M.cp.read_source(original)
    g, _, _, _, _, _ = scale.project(original, raw, M)
    del raw; gc.collect()
    source_keys = g[M.cp.KEY].to_numpy("<i8")
    base_dir = Path(grant["inputs"]["base_inputs_manifest"]["path"]).parent
    original_manifest = json.loads((base_dir / "manifest.json").read_text())
    for name in ("x-prequential.npy", "corpus-subset.npz", "seq-inputs.npz"):
        runtime.require(runtime.sha(base_dir / name) == original_manifest["files"][name]["sha256"],
                        "base package file " + name)
    with np.load(base_dir / "corpus-subset.npz", allow_pickle=False) as z:
        corpus = {name: np.array(z[name]) for name in base.CORPUS_MEMBERS}
    with np.load(base_dir / "seq-inputs.npz", allow_pickle=False) as z:
        seq = {name: np.array(z[name]) for name in base.SEQ_MEMBERS}
    runtime.require(np.array_equal(source_keys, corpus["keys"])
                    and np.array_equal(g.pitcher.to_numpy(dtype=np.int64), seq["pitcher"].astype(np.int64)),
                    "projected rows differ from accepted base package")
    days = g.game_date.dt.date.astype(str).to_numpy()
    game_date_count = g.groupby("game_pk").game_date.nunique()
    runtime.require(int(game_date_count.max()) == 1, "game crosses calendar date")
    if window == "base":
        masks = {role: corpus[key].astype(bool) for role, key in
                 (("FIT", "fit"), ("SELECT", "tuning"), ("CAL", "calibration"), ("EVAL", "evaluation"))}
        selected = np.ones(len(g), dtype=bool)
        heldout = []
    else:
        spec = plan["windows"][window]
        heldout = data.select_heldout(seq["pitcher"], spec["heldout_selection"]) if window == "pitcher_holdout" else []
        spec = dict(spec, heldout_pitchers=heldout)
        masks = data.role_masks(days, seq["pitcher"], spec, heldout=bool(heldout))
        selected = np.logical_or.reduce(list(masks.values()))
    runtime.require(selected.any(), "empty selected population")
    role_to_member = {"FIT": "fit", "SELECT": "tuning", "CAL": "calibration", "EVAL": "evaluation"}
    new_corpus = {name: value[selected] for name, value in corpus.items()}
    for role, member in role_to_member.items():
        new_corpus[member] = masks[role][selected]
    new_seq = {name: value[selected] for name, value in seq.items()}
    # The package contains only rows in one declared role. Same-game contexts
    # never need excluded prior-season games; the source game/date check is above.
    runtime.require(np.all(sum(new_corpus[k].astype(int) for k in role_to_member.values()) == 1),
                    "unassigned or overlapping package row")
    out = Path(grant["output"])
    out.mkdir(parents=True, exist_ok=False)
    if window == "pitcher_holdout":
        donor = g.loc[masks["FIT"], M.safe.DONOR].reset_index(drop=True)
        runtime.require(not donor.pitcher.isin(heldout).any(), "heldout profile donor")
        joined, _ = scale.features(donor, g, original["history_cutoff"], M)
        # A second pass recovers ONLY heldout pitchers' own strictly-prior
        # physics/outcome histories for their EVAL queries. Other columns stay
        # from the clean pool, including batter, cross-pitcher donor and
        # similarity-abstention features. Thus no heldout row influences FIT,
        # SELECT, CAL or the encoder. Same-day sequence history is separate.
        own_rows = g.pitcher.isin(heldout).to_numpy()
        own_pool = __import__("pandas").concat([donor, g.loc[own_rows, M.safe.DONOR]], ignore_index=True)
        eval_rows = masks["EVAL"]
        owned, _ = scale.features(own_pool, g.loc[eval_rows], plan["windows"][window]["roles"]["EVAL"][1], M)
        runtime.require(owned.shape == (int(eval_rows.sum()), 91), "heldout own-history shape")
        joined = data.splice_heldout_own_history(joined, owned, eval_rows, masks["FIT"])
        del own_pool, owned
        del donor; gc.collect()
        fitx, encoder = M.fsf.encode(joined[masks["FIT"]], np.ones(int(masks["FIT"].sum()), bool))
        runtime.require(fitx.shape[1] == 199, "heldout-free codec dimension")
        features = np.lib.format.open_memmap(out / "x-prequential.npy", mode="w+", dtype=np.float32,
                                              shape=(int(selected.sum()), 199))
        selected_index = np.flatnonzero(selected)
        for start in range(0, len(selected_index), 16384):
            idx = selected_index[start:start + 16384]
            features[start:start + len(idx)] = M.safe.apply_encoder(joined[idx], encoder)
        del features, joined, fitx; gc.collect()
        (out / "encoder.json").write_text(json.dumps(encoder, sort_keys=True) + "\n")
    else:
        x = np.load(base_dir / "x-prequential.npy", mmap_mode="r", allow_pickle=False)
        np.save(out / "x-prequential.npy", x[selected], allow_pickle=False)
    support = data.fit_prior_support(days[selected], new_seq["pitcher"], new_seq["game_pk"],
                                     new_seq["at_bat_number"], new_seq["pitch_number"], new_corpus["fit"])
    np.save(out / "fit-prior-support.npy", support, allow_pickle=False)
    np.save(out / "game-date.npy", days[selected].astype("U10"), allow_pickle=False)
    np.savez(out / "corpus-subset.npz", **new_corpus)
    np.savez(out / "seq-inputs.npz", **new_seq)
    files = {name: {"sha256": runtime.sha(out / name), "bytes": (out / name).stat().st_size}
             for name in ("x-prequential.npy", "corpus-subset.npz", "seq-inputs.npz",
                          "fit-prior-support.npy", "game-date.npy")}
    receipt = {"status": "COMPLETED", "window": window, "data_role": plan["data_role"],
               "data_version": plan["data_version"], "source_binding": binding,
               "heldout_pitchers_sha256": __import__("hashlib").sha256(json.dumps(heldout).encode()).hexdigest(),
               "heldout_pitchers": heldout,
               "rows": {role: int(mask.sum()) for role, mask in masks.items()},
               "eval_scores_opened": False, "files": files}
    manifest = {"id": base.EXP_ID, "campaign_id": plan["campaign_id"], "window": window,
                "data_role": plan["data_role"], "data_version": plan["data_version"],
                "plan_sha256": grant["plan_sha256"], "files": files,
                "bound_sources": {"train_seq.py": runtime.sha(runtime.ROOT / "experiments/ws_seq/train_seq.py"),
                                  "config": runtime.sha(runtime.ROOT / "configs/EXP-WS-CAMPAIGN-001.json")},
                "receipt": receipt}
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    roles = plan["windows"][window]["roles"] if window != "base" else plan["windows"]["pitcher_holdout"]["roles"]
    ledger = {"campaign_id": plan["campaign_id"], "window": window,
              "data_role": plan["data_role"], "data_version": plan["data_version"],
              "roles": roles, "rows": receipt["rows"],
              "input_manifest_sha256": runtime.sha(out / "manifest.json"),
              "source_exposure_ledger_sha256": grant["inputs"]["exposure_ledger"]["sha256"],
              "eval_scores_opened": False, "independent_test": False}
    (out / "exposure-ledger.json").write_text(json.dumps(ledger, sort_keys=True, indent=2) + "\n")
    (out / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


def owned_worker(grant, plan, window):
    job_env = Path(os.environ["GPU_JOB_DIR"]) / "job.env"
    env = dict(line.split("=", 1) for line in job_env.read_text().splitlines() if "=" in line)
    short = env.get("COMMIT", "")
    runtime.require(len(short) >= 7 and grant["commit"].startswith(short)
                    and len(grant["commit"]) == 40, "preparation source commit")
    runtime.require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(grant["gpu_index"]), "preparation GPU index")
    import subprocess
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(grant["gpu_index"]), "--query-gpu=uuid",
                                    "--format=csv,noheader"], text=True, timeout=10).strip()
    runtime.require(uuid == grant["gpu_uuid"], "preparation GPU UUID")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                         "--format=csv,noheader"], text=True, timeout=10)
    rows = [line.split(",") for line in processes.splitlines() if line.strip()]
    runtime.require(all(len(row) == 2 and row[0].strip().isdigit() for row in rows)
                    and not any(row[1].strip() == uuid for row in rows), "preparation GPU occupancy")
    available = next((int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemAvailable:")), 0)
    runtime.require(available >= grant["min_free_host_bytes"], "preparation host memory headroom")
    lock_path = Path(grant["common_lock"])
    runtime.require(lock_path.is_file(), "existing common GPU lock")
    with lock_path.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, spec in grant["inputs"].items():
            location = source_path(spec) if name == "scale_registration" else Path(spec["path"])
            runtime.require(runtime.sha(location) == spec["sha256"], "preparation input hash " + name)
        for name, spec in grant["scale_sources"].items():
            runtime.require(runtime.sha(source_path(spec)) == spec["sha256"],
                            "preparation source hash " + name)
        package(grant, plan, window)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grant", type=Path, required=True)
    p.add_argument("--window", choices=("base", "chrono1", "chrono2", "pitcher_holdout"), required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    try:
        grant, plan = checked_grant(args.grant, args.window)
        runtime.require(sys.platform == "linux" and os.environ.get("GPU_JOB_ID"), "gpu run preparation only")
        if args.worker:
            owned_worker(grant, plan, args.window)
            return 0
        return supervise_command([sys.executable, "-m", "experiments.ws_campaign.prepare",
                                  "--grant", str(args.grant), "--window", args.window, "--worker"],
                                 Path(grant["output"]), grant["max_wall_seconds"],
                                 grant["max_rss_bytes"], runtime.sha(args.grant))
    except (data.DataStop, runtime.Stop, OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 87


if __name__ == "__main__":
    sys.exit(main())
