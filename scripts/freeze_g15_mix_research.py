"""Build or verify a research-only, hash-bound inventory of the confirmed G15 mix.

The G15 mix is the equal-weight probability mix of the direct pre-pitch head and
the frozen G0-global member of the same seed (EXP-P15-002, confirmed on five
seeds by EXP-P16-001, D143).  It becomes the research prediction baseline; the
frozen G0 bundle is one of its components and is not modified.

This command reads existing artifacts as bytes/JSON. It never deserializes model
checkpoints, loads prediction arrays, fits, or writes inside a run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_g0_research import REPO, RUNS, digest, read, require, verify_files  # noqa: E402

BUNDLE = REPO / "configs/G15-MIX-RESEARCH-FROZEN-v1.json"
RESULT = REPO / "results/G15-MIX-RESEARCH-FROZEN-v1.json"
G0_BUNDLE = REPO / "configs/G0-RESEARCH-FROZEN-v1.json"
ARM = "G15-direct-g0-mix"
FIT_COMMIT = "12f4fe944d2faa825dc4ea2b3c2668447b6e74eb"       # EXP-P15-001 fit/predict code (seeds 0-2)
CONFIRM_COMMIT = "fbcdaacd8ddb1ca987632cc315fd6a041a5b8bc5"   # EXP-P16-001 runner (seeds 3-4, five-seed scoring)
SEEDS = range(5)
MEMBER_FILES = {"checkpoint": "model.pt", "fit_report": "fit.json", "fit_state": "fit_state.json",
                "may_calibration": "calibration.json", "prediction_state": "prediction_state.json",
                "mlb_prediction_state": "mlb_prediction_state.json"}
FLAGS = {"research_only": True, "service_promotion": False, "independent_confirmation": False, "policy_validated": False}


def run_of(seed: int) -> str:
    return "EXP-P15-001" if seed < 3 else "EXP-P16-001"


def frozen_paths() -> dict[str, Path]:
    paths = {"g0_bundle": G0_BUNDLE,
             "results_p15": REPO / "results/EXP-P15-001.json", "results_p16": REPO / "results/EXP-P16-001.json"}
    for run in ("EXP-P15-001", "EXP-P16-001"):
        tag = run[4:7].lower()
        paths[f"{tag}_preparation"] = RUNS / run / "preparation.json"
        paths[f"{tag}_registered_config"] = RUNS / run / "registered_config.json"
        paths[f"{tag}_analysis_results"] = RUNS / run / "analysis/results.json"
    for seed in SEEDS:
        for role, name in MEMBER_FILES.items():
            paths[f"seed{seed}_{role}"] = RUNS / run_of(seed) / str(seed) / name
    for relative in sorted(read(paths["p16_preparation"])["identity"]["source_hashes"]):
        paths[f"source/{relative}"] = RUNS / "EXP-P16-001/source" / relative
    return paths


def verify_semantics(bundle: dict) -> None:
    require(bundle["protocol"] == "g15_mix_research_frozen_v1", "protocol")
    for flag, value in FLAGS.items():
        require(bundle[flag] is value, f"flag: {flag}")
    require(bundle["definition"]["arm"] == ARM and bundle["definition"]["direct_weight"] == .5, "mix definition")
    require(list(bundle["members"]) == [str(seed) for seed in SEEDS], "member order")
    files = bundle["files"]
    for seed in SEEDS:
        member, state = bundle["members"][str(seed)], read(Path(files[f"seed{seed}_prediction_state"]["path"]))
        require(member["source_run"] == run_of(seed), f"member run: {seed}")
        require(member["model_sha256"] == files[f"seed{seed}_checkpoint"]["sha256"] == state["identity"]["model_sha256"],
                f"member checkpoint: {seed}")
        require(state["identity"]["seed"] == seed and state["identity"]["smoke"] is False, f"member state: {seed}")
    result = read(Path(files["results_p16"]["path"]))
    require(files["results_p16"]["sha256"] == files["p16_analysis_results"]["sha256"], "repository result differs from the run")
    require(files["results_p15"]["sha256"] == files["p15_analysis_results"]["sha256"], "repository result differs from the run")
    require(result["confirmation"]["status"] == "confirmed" and result["confirmation"]["arm"] == ARM, "confirmation")
    require(result["git_commit"] == CONFIRM_COMMIT and bundle["code"]["confirmation_commit"] == CONFIRM_COMMIT, "code commit")
    require(result["frozen_direct_preparation_sha256"] == files["p15_preparation"]["sha256"], "frozen direct run")
    require(result["preparation_sha256"] == files["p16_preparation"]["sha256"], "confirmation run")
    require(bundle["g0_component"]["bundle_sha256"] == files["g0_bundle"]["sha256"], "G0 component")


def evidence(row: dict) -> dict:
    return {"delta_nll": row["paired"]["nll"]["delta"], "ci95": row["paired"]["nll"]["ci95"],
            "seed_deltas": row["seed_deltas"], "N_status": row["N"]["status"]}


def build() -> dict:
    paths = frozen_paths()
    files = {role: {"path": str(path), "sha256": digest(path)} for role, path in paths.items()}
    result, config = read(paths["results_p16"]), read(paths["p16_registered_config"])
    report = result["cpanel_dev"]["reports"][ARM]
    bundle = {
        "protocol": "g15_mix_research_frozen_v1", **FLAGS,
        "role": "Research prediction baseline from D145: the control arm of later prediction experiments. "
                "The service, the demo and the validated policy identifier still use G0-global alone.",
        "scope": "Five direct-head members mixed with the five frozen G0-global members; 2025 exposed DEV evaluation; no new fit or inference",
        "availability": "Reference inventory only: external SSD checkpoints and the experiment branch code remain required; "
                        "no model or data bytes are copied into this repository.",
        "builder_sha256": digest(Path(__file__)),
        "definition": {"arm": ARM, "direct_weight": config["mix"]["direct_weight"], "rule": config["mix"]["rule"],
                       "ensemble": "five-seed mean of the member-wise mixes",
                       "june_frequency_blend_model_weight": report["selection"]["model_weight"],
                       "one_change_of_the_direct_head": config["one_change"]},
        "g0_component": {"bundle": str(G0_BUNDLE.relative_to(REPO)), "bundle_sha256": files["g0_bundle"]["sha256"]},
        "code": {"branch": "claude/exp-p16-direct-mix-confirm", "fit_predict_commit": FIT_COMMIT,
                 "confirmation_commit": CONFIRM_COMMIT},
        "decision_evidence": {
            "confirmation": result["confirmation"],
            "cpanel_dev_after_blend": evidence(result["cpanel_dev"]["comparison"][ARM]),
            "whole_mlb_dev_blended": evidence(result["mlb_dev_blended"]["comparison"][ARM]),
            "limits": result["limits"]},
        "members": {str(seed): {
            "source_run": run_of(seed), "model_sha256": files[f"seed{seed}_checkpoint"]["sha256"],
            "delivery_temperature": read(paths[f"seed{seed}_may_calibration"])["delivery_temperature"],
            "best_epoch": read(paths[f"seed{seed}_fit_report"])["report"]["best_epoch"],
            "june_mix_model_weight": report["seeds"][seed]["blend_selection"]["model_weight"],
        } for seed in SEEDS},
        "files": files,
    }
    verify_semantics(bundle)
    verify_files(files)
    return bundle


def verify(bundle: dict) -> dict:
    verify_semantics(bundle)
    verify_files(bundle["files"])
    return {"protocol": "g15_mix_research_frozen_verification_v1", "status": "PASS",
            "bundle_sha256": digest(BUNDLE), "file_count": len(bundle["files"]), "member_count": len(bundle["members"]),
            "fit_predict_commit": FIT_COMMIT, "confirmation_commit": CONFIRM_COMMIT, **FLAGS,
            "new_fit_count": 0, "inference_performed": False, "self_contained": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    if args.command == "build":
        require(not BUNDLE.exists(), f"bundle already exists: {BUNDLE}")
        require(not RESULT.exists(), f"result already exists: {RESULT}")
        BUNDLE.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = verify(read(BUNDLE))
    if args.command == "build":
        RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        require(read(RESULT) == result, "stored verification result diverged")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
