"""Build or verify a research-only, hash-bound G0 five-seed inventory.

This command reads existing artifacts as bytes/JSON. It never deserializes model
checkpoints, loads prediction arrays, fits, or writes inside a parent run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNS = Path("/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924")
BUNDLE = REPO / "configs/G0-RESEARCH-FROZEN-v1.json"
RESULT = REPO / "results/G0-RESEARCH-FROZEN-v1.json"
CODE_COMMIT = "5e82bebc239ced378911919de48c106034ee24b0"
SOURCE_HEAD = "eef5417c0a035f11412fb7fe535bf32e57fe7f8e"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_member_paths(seed: int) -> dict[str, Path]:
    arm = "EXP-P4-001" if seed < 3 else "EXP-P10-001"
    fit = RUNS / arm / "fits" / f"seed{seed}" / "global"
    member = RUNS / arm / "members" / "G0-global" / f"seed{seed}"
    return {
        "checkpoint": fit / "model.pt",
        "fit_report": fit / "fit.json",
        "fit_state": fit / "state.json",
        "may_calibration": member / "calibration.json",
        "prediction_state": member / "prediction_state.json",
    }


def frozen_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {
        "config_p4": REPO / "configs/EXP-P4-001.yaml",
        "config_p10": REPO / "configs/EXP-P10-001.yaml",
        "config_p11": REPO / "configs/EXP-P11-001.yaml",
        "config_p11_calibration": REPO / "configs/EXP-P11-002.yaml",
        "p4_preparation": RUNS / "EXP-P4-001/preparation.json",
        "p4_registered_config": RUNS / "EXP-P4-001/registered_config.json",
        "p4_architecture_preparation": RUNS / "EXP-P4-001/architecture_preparation.json",
        "p4_parent_preparation": RUNS / "EXP-P4-001/parent_preparation.json",
        "p4_auxiliary": RUNS / "EXP-P4-001/aux.pkl",
        "p4_panel": RUNS / "EXP-P4-001/panel.json",
        "p4_clusters": RUNS / "EXP-P4-001/clusters.json",
        "p4_frequency": RUNS / "EXP-P4-001/baseline_predictions.npz",
        "p4_train_keys": RUNS / "EXP-P4-001/train_keys.parquet",
        "p4_earlystop_keys": RUNS / "EXP-P4-001/earlystop_keys.parquet",
        "p4_may_keys": RUNS / "EXP-P4-001/temperature_keys.parquet",
        "p4_june_keys": RUNS / "EXP-P4-001/blend_keys.parquet",
        "p4_dev_keys": RUNS / "EXP-P4-001/dev_keys.parquet",
        "p4_mlb_dev_keys": RUNS / "EXP-P4-001/mlb_dev_keys.parquet",
        "p10_preparation": RUNS / "EXP-P10-001/preparation.json",
        "p10_registered_config": RUNS / "EXP-P10-001/registered_config.json",
        "p10_analysis_results": RUNS / "EXP-P10-001/analysis/results.json",
        "p10_analysis_manifest": RUNS / "EXP-P10-001/analysis/manifest.json",
        "p11_preparation": RUNS / "EXP-P11-001/preparation.json",
        "p11_registered_config": RUNS / "EXP-P11-001/registered_config.json",
        "p11_frozen_calibration": RUNS / "EXP-P11-001/frozen_calibration.json",
        "p11_dev_keys": RUNS / "EXP-P11-001/dev_keys.parquet",
        "p11_dev_metadata": RUNS / "EXP-P11-001/dev_metadata.parquet",
        "p11_cpanel_keys": RUNS / "EXP-P11-001/cpanel_dev_keys.parquet",
        "p11_frequency": RUNS / "EXP-P11-001/baseline_predictions.npz",
        "p11_predictions": RUNS / "EXP-P11-001/analysis/predictions.npz",
        "p11_results": RUNS / "EXP-P11-001/analysis/results.json",
        "p11_manifest": RUNS / "EXP-P11-001/analysis/manifest.json",
        "p11_calibration_activation": RUNS / "EXP-P11-002/activation.json",
        "p11_calibration_fit": RUNS / "EXP-P11-002/fits/I1.json",
        "p11_calibration_results": RUNS / "EXP-P11-002/analysis/results.json",
        "p11_calibration_manifest": RUNS / "EXP-P11-002/analysis/manifest.json",
    }
    for seed in range(5):
        for role, path in expected_member_paths(seed).items():
            paths[f"seed{seed}_{role}"] = path
    preparation = read(paths["p11_preparation"])
    for relative in sorted(preparation["identity"]["source_hashes"]):
        paths[f"source/{relative}"] = RUNS / "EXP-P11-001/source" / relative
    scoring = read(paths["p11_manifest"])["scoring_sources"]
    code_root = Path("/Users/song/Projects/pitcheezy-worktrees/g0-whole-mlb-execution/experiments/pitchmdp")
    for relative in sorted(scoring):
        paths[f"scoring_source/{relative}"] = code_root / relative
    return paths


def verify_files(files: dict[str, dict[str, str]]) -> None:
    require(files and len(files) == len(set(files)), "empty or duplicate file roles")
    seen: set[str] = set()
    for role, entry in files.items():
        require(set(entry) == {"path", "sha256"}, f"bad file entry: {role}")
        path = Path(entry["path"])
        require(path.is_absolute() and path.is_file(), f"missing file: {role}")
        require(str(path) not in seen, f"duplicate file path: {role}")
        seen.add(str(path))
        require(digest(path) == entry["sha256"], f"hash mismatch: {role}")


def verify_members(members: dict, files: dict, prep: dict, frozen: dict) -> None:
    require(list(members) == [str(i) for i in range(5)], "member order")
    for seed in range(5):
        key = str(seed)
        member = prep["members"][key]
        require(member["seed"] == seed and member["source_arm"] == ("g" if seed < 3 else "c1"), f"member identity: {seed}")
        require(members[key] == {
            "source_arm": member["source_arm"], "model_sha256": member["model_sha256"],
            "delivery_temperature": member["delivery_temperature"],
            "june_model_weight": frozen["june_seed_model_weights"][key],
        }, f"member semantics: {seed}")
        for role, field in (("checkpoint", "model_sha256"), ("fit_report", "fit_sha256"),
                            ("fit_state", "fit_state_sha256"), ("may_calibration", "calibration_sha256"),
                            ("prediction_state", "prediction_state_sha256")):
            require(files[f"seed{seed}_{role}"]["sha256"] == member[field], f"member link: {seed} {role}")
        cal = read(Path(files[f"seed{seed}_may_calibration"]["path"]))
        require(cal["delivery_temperature"] == member["delivery_temperature"] == frozen["may_delivery_temperatures"][key], f"May temperature: {seed}")
        fit = read(Path(files[f"seed{seed}_fit_report"]["path"]))
        require(fit["report"]["seed"] == seed, f"fit seed: {seed}")


def verify_semantics(bundle: dict) -> None:
    require(bundle["protocol"] == "g0_research_frozen_v1", "wrong protocol")
    require(bundle["research_only"] is True and bundle["service_promotion"] is False, "research-only boundary")
    require(bundle["independent_confirmation"] is False, "independent confirmation boundary")
    require(bundle["execution_code_commit"] == CODE_COMMIT, "code commit changed")
    require(bundle["evidence_head"] == SOURCE_HEAD, "evidence head changed")
    files = bundle["files"]
    expected = frozen_paths()
    require(set(files) == set(expected), "file roles changed")
    for role, path in expected.items():
        require(files[role]["path"] == str(path), f"path alias changed: {role}")
    prep = read(Path(files["p11_preparation"]["path"]))
    p4 = read(Path(files["p4_preparation"]["path"]))
    p10 = read(Path(files["p10_preparation"]["path"]))
    frozen = read(Path(files["p11_frozen_calibration"]["path"]))
    c1_result = read(Path(files["p10_analysis_results"]["path"]))
    config = read(Path(files["config_p11"]["path"]))
    calibration_config = read(Path(files["config_p11_calibration"]["path"]))
    manifest = read(Path(files["p11_manifest"]["path"]))
    require(config["experiment_id"] == prep["experiment_id"] == "EXP-P11-001", "P11 identity")
    require(config["registration"]["execution_code_commit"] == CODE_COMMIT, "P11 code pin")
    require(read(Path(files["p11_registered_config"]["path"])) == config, "P11 registered config")
    require(read(Path(files["p4_registered_config"]["path"])) == read(Path(files["config_p4"]["path"])), "P4 registered config")
    require(read(Path(files["p10_registered_config"]["path"])) == read(Path(files["config_p10"]["path"])), "P10 registered config")
    require(hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == prep["identity"]["config_sha256"], "P11 canonical config")
    require(prep["new_fits"] == 0 and prep["reused_fits"] == 5, "P11 fit count")
    require(prep["identity"]["source_hashes"] == {
        role.removeprefix("source/"): entry["sha256"]
        for role, entry in files.items() if role.startswith("source/")
    }, "P11 source closure")
    require(manifest["scoring_sources"] == {
        role.removeprefix("scoring_source/"): entry["sha256"]
        for role, entry in files.items() if role.startswith("scoring_source/")
    }, "P11 scoring source closure")
    require(bundle["environment"] == prep["identity"], "environment identity changed")
    require(bundle["builder_sha256"] == digest(Path(__file__)), "builder source changed")
    require(bundle["archived_references"] == {
        "p4_external_hashes": p4["external_hashes"],
        "p4_architecture_dataset_identity": read(Path(files["p4_architecture_preparation"]["path"]))["dataset_identity"],
    }, "archived source data identity changed")
    require(bundle["data_identity"] == {
        "p4_samples": p4["samples"], "p11_samples": prep["samples"],
        "p11_population": prep["population"], "p11_cpanel_overlap": prep["cpanel_overlap"],
    }, "data identity changed")
    require(prep["features"] == p4["features"] == p10["features"], "feature/normalizer/delivery metadata diverged")
    require(prep["clusters"] == p4["clusters"] == p10["clusters"], "cluster encoder diverged")
    require(prep["panel"] == p4["panel"] == p10["panel"], "panel diverged")
    require(prep["frozen_calibration"] == frozen, "frozen calibration diverged")
    require(frozen["refit_on_whole_mlb"] is False, "whole MLB refit")
    require(frozen["june_ensemble_model_weight"] == c1_result["reports"]["G0-global"]["selection"]["model_weight"], "June ensemble weight")
    seed_reports = c1_result["reports"]["G0-global"]["seeds"]
    require(len(seed_reports) == 5 and p10["seeds"] == list(range(5)), "C1 seed report order")
    require(frozen["june_seed_model_weights"] == {
        str(seed): row["blend_selection"]["model_weight"] for seed, row in enumerate(seed_reports)
    }, "June seed weights")
    verify_members(bundle["members"], files, prep, frozen)
    require(files["p11_results"]["sha256"] == manifest["results_sha256"], "P11 result link")
    require(files["p11_predictions"]["sha256"] == manifest["predictions_sha256"], "P11 prediction link")
    require(files["p11_frequency"]["sha256"] == manifest["inputs"][files["p11_frequency"]["path"]], "P11 frequency link")
    require(files["p11_dev_metadata"]["sha256"] == manifest["inputs"][files["p11_dev_metadata"]["path"]], "P11 metadata link")
    require(config["parent_g"]["preparation_sha256"] == files["p4_preparation"]["sha256"], "P4 parent link")
    require(config["parent_c1"]["preparation_sha256"] == files["p10_preparation"]["sha256"], "C1 parent link")
    require(config["parent_c1"]["analysis_results_sha256"] == files["p10_analysis_results"]["sha256"], "C1 result link")
    require(config["parent_g"]["baseline_predictions_sha256"] == files["p4_frequency"]["sha256"], "frequency parent link")
    whole_result = read(Path(files["p11_results"]["path"]))
    require(whole_result["experiment_id"] == "EXP-P11-001" and whole_result["execution_code_commit"] == CODE_COMMIT, "whole evaluation identity")
    require(whole_result["new_fits"] == 0 and whole_result["reused_fits"] == 5, "whole evaluation fit count")
    require(whole_result["estimands"]["whole"]["N"]["status"] == "development_advantage_over_frequency", "whole evaluation status")
    require(whole_result["estimands"]["non_cpanel"]["N"]["status"] == "development_advantage_over_frequency", "non-Cpanel evaluation status")
    calibration_result = read(Path(files["p11_calibration_results"]["path"]))
    activation = read(Path(files["p11_calibration_activation"]["path"]))
    calibration_manifest = read(Path(files["p11_calibration_manifest"]["path"]))
    require(calibration_config["experiment_id"] == calibration_result["experiment_id"] == "EXP-P11-002", "calibration ID")
    require(calibration_result["activation"] == activation["activation"] == {"I1": True, "I2": False}, "calibration activation")
    require(calibration_result["candidates"]["I1"]["N"]["status"] == "worse_or_guardrail_failure", "I1 negative evidence")
    require(calibration_result["candidates"]["I2"]["status"] == "not_activated" and calibration_result["candidates"]["I2"]["N"] is None, "I2 unmeasured evidence")
    require(calibration_result["stage2_binding_sha256"] == activation["stage2_binding_sha256"], "stage2 binding")
    require(calibration_result["activation_sha256"] == files["p11_calibration_activation"]["sha256"], "activation hash link")
    require(calibration_manifest["results_sha256"] == files["p11_calibration_results"]["sha256"], "calibration result hash link")
    require(bundle["decision_evidence"] == {
        "whole_N_status": whole_result["estimands"]["whole"]["N"]["status"],
        "non_cpanel_N_status": whole_result["estimands"]["non_cpanel"]["N"]["status"],
        "I1_N_status": calibration_result["candidates"]["I1"]["N"]["status"],
        "I2_status": calibration_result["candidates"]["I2"]["status"],
    }, "decision evidence changed")


def build() -> dict:
    paths = frozen_paths()
    files = {role: {"path": str(path), "sha256": digest(path)} for role, path in paths.items()}
    prep = read(paths["p11_preparation"])
    p4 = read(paths["p4_preparation"])
    frozen = read(paths["p11_frozen_calibration"])
    whole_result = read(paths["p11_results"])
    calibration_result = read(paths["p11_calibration_results"])
    bundle = {
        "protocol": "g0_research_frozen_v1", "research_only": True, "service_promotion": False,
        "independent_confirmation": False, "execution_code_commit": CODE_COMMIT,
        "evidence_head": SOURCE_HEAD,
        "builder_sha256": digest(Path(__file__)),
        "scope": "Five existing G0-global members; 2025 exposed DEV evaluation; no new fit or inference",
        "availability": "Reference inventory only: external SSD model and cached input dependencies plus execution worktree scoring source remain required; no model or data bytes are copied into this repository.",
        "archived_references": {
            "p4_external_hashes": p4["external_hashes"],
            "p4_architecture_dataset_identity": read(paths["p4_architecture_preparation"])["dataset_identity"],
        },
        "decision_evidence": {
            "whole_N_status": whole_result["estimands"]["whole"]["N"]["status"],
            "non_cpanel_N_status": whole_result["estimands"]["non_cpanel"]["N"]["status"],
            "I1_N_status": calibration_result["candidates"]["I1"]["N"]["status"],
            "I2_status": calibration_result["candidates"]["I2"]["status"],
        },
        "environment": prep["identity"],
        "data_identity": {"p4_samples": p4["samples"], "p11_samples": prep["samples"],
                          "p11_population": prep["population"], "p11_cpanel_overlap": prep["cpanel_overlap"]},
        "members": {str(i): {
            "source_arm": prep["members"][str(i)]["source_arm"],
            "model_sha256": prep["members"][str(i)]["model_sha256"],
            "delivery_temperature": prep["members"][str(i)]["delivery_temperature"],
            "june_model_weight": frozen["june_seed_model_weights"][str(i)],
        } for i in range(5)},
        "files": files,
    }
    verify_semantics(bundle)
    verify_files(files)
    return bundle


def verify(bundle: dict) -> dict:
    verify_semantics(bundle)
    verify_files(bundle["files"])
    return {
        "protocol": "g0_research_frozen_verification_v1", "status": "PASS",
        "bundle_sha256": digest(BUNDLE), "file_count": len(bundle["files"]),
        "member_count": len(bundle["members"]), "execution_code_commit": CODE_COMMIT,
        "evidence_head": SOURCE_HEAD, "research_only": True,
        "new_fit_count": 0, "inference_performed": False,
        "independent_confirmation": False, "service_promotion": False,
        "archived_raw_identity_verified": False,
        "runtime_binaries_verified": False,
        "self_contained": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    if args.command == "build":
        require(not BUNDLE.exists(), f"bundle already exists: {BUNDLE}")
        require(not RESULT.exists(), f"result already exists: {RESULT}")
        bundle = build()
        BUNDLE.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = verify(read(BUNDLE))
    if args.command == "build":
        RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        require(read(RESULT) == result, "stored verification result diverged")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
