"""Synthetic boundary checks for the research-only G15 mix hash inventory."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_g15_mix_research.py"
SPEC = importlib.util.spec_from_file_location("freeze_g15_mix_research", SCRIPT)
assert SPEC and SPEC.loader
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


def _write(path: Path, value) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _bundle(root: Path) -> dict:
    files = {"g0_bundle": _write(root / "g0.json", {"protocol": "g0"}),
             "p15_preparation": _write(root / "p15/preparation.json", {"run": 15}),
             "p16_preparation": _write(root / "p16/preparation.json", {"run": 16})}
    members = {}
    for seed in freeze.SEEDS:
        files[f"seed{seed}_checkpoint"] = _write(root / f"{seed}/model.pt", f"weights{seed}".encode())
        model = files[f"seed{seed}_checkpoint"]["sha256"]
        files[f"seed{seed}_prediction_state"] = _write(
            root / f"{seed}/prediction_state.json", {"identity": {"seed": seed, "smoke": False, "model_sha256": model}})
        members[str(seed)] = {"source_run": freeze.run_of(seed), "model_sha256": model}
    result = {"confirmation": {"status": "confirmed", "arm": freeze.ARM}, "git_commit": freeze.CONFIRM_COMMIT,
              "frozen_direct_preparation_sha256": files["p15_preparation"]["sha256"],
              "preparation_sha256": files["p16_preparation"]["sha256"]}
    for tag, value in (("p15", {"three": "seeds"}), ("p16", result)):
        files[f"results_{tag}"] = _write(root / f"repo/{tag}.json", value)
        files[f"{tag}_analysis_results"] = _write(root / f"run/{tag}.json", value)
    return {"protocol": "g15_mix_research_frozen_v1", **freeze.FLAGS,
            "definition": {"arm": freeze.ARM, "direct_weight": .5},
            "g0_component": {"bundle_sha256": files["g0_bundle"]["sha256"]},
            "code": {"confirmation_commit": freeze.CONFIRM_COMMIT}, "members": members, "files": files}


def test_semantics_accept_the_confirmed_mix_and_reject_promotion_or_drift(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    freeze.verify_semantics(bundle)
    freeze.verify_files(bundle["files"])
    for flag in ("service_promotion", "policy_validated", "independent_confirmation"):
        promoted = copy.deepcopy(bundle)
        promoted[flag] = True
        with pytest.raises(ValueError, match="flag: " + flag):
            freeze.verify_semantics(promoted)
    weight = copy.deepcopy(bundle)
    weight["definition"]["direct_weight"] = .7
    with pytest.raises(ValueError, match="mix definition"):
        freeze.verify_semantics(weight)
    swapped = copy.deepcopy(bundle)
    swapped["members"]["3"]["model_sha256"] = bundle["members"]["4"]["model_sha256"]
    with pytest.raises(ValueError, match="member checkpoint: 3"):
        freeze.verify_semantics(swapped)
    wrong_run = copy.deepcopy(bundle)
    wrong_run["members"]["3"]["source_run"] = "EXP-P15-001"
    with pytest.raises(ValueError, match="member run: 3"):
        freeze.verify_semantics(wrong_run)
    other_g0 = copy.deepcopy(bundle)
    other_g0["g0_component"]["bundle_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="G0 component"):
        freeze.verify_semantics(other_g0)


def test_unconfirmed_or_edited_results_are_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    path = Path(bundle["files"]["results_p16"]["path"])
    result = json.loads(path.read_text())
    result["confirmation"]["status"] = "not_confirmed"
    edited = copy.deepcopy(bundle)
    edited["files"]["results_p16"] = _write(path, result)
    with pytest.raises(ValueError, match="repository result differs from the run"):
        freeze.verify_semantics(edited)
    edited["files"]["p16_analysis_results"] = _write(Path(bundle["files"]["p16_analysis_results"]["path"]), result)
    with pytest.raises(ValueError, match="confirmation"):
        freeze.verify_semantics(edited)
