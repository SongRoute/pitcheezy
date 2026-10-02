"""Synthetic boundary checks for the research-only G0 hash inventory."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_g0_research.py"
SPEC = importlib.util.spec_from_file_location("freeze_g0_research", SCRIPT)
assert SPEC and SPEC.loader
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


def _entry(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_byte_verification_detects_mutation_and_duplicate_alias(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"frozen")
    second.write_bytes(b"other")
    files = {"first": _entry(first), "second": _entry(second)}
    freeze.verify_files(files)
    first.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch: first"):
        freeze.verify_files(files)
    first.write_bytes(b"frozen")
    aliased = copy.deepcopy(files)
    aliased["second"] = _entry(first)
    with pytest.raises(ValueError, match="duplicate file path"):
        freeze.verify_files(aliased)


def _members(tmp_path: Path) -> tuple[dict, dict, dict, dict]:
    bundle_members = {}
    files = {}
    prep = {"members": {}}
    frozen = {"june_seed_model_weights": {}, "may_delivery_temperatures": {}}
    for seed in range(5):
        key = str(seed)
        arm = "g" if seed < 3 else "c1"
        model_hash = f"model-{seed}"
        temperature = 1.0 + seed / 100
        june = 0.7 + seed / 100
        cal = tmp_path / f"cal-{seed}.json"
        fit = tmp_path / f"fit-{seed}.json"
        cal.write_text(json.dumps({"delivery_temperature": temperature}))
        fit.write_text(json.dumps({"report": {"seed": seed}}))
        prep["members"][key] = {
            "seed": seed, "source_arm": arm, "model_sha256": model_hash,
            "fit_sha256": _entry(fit)["sha256"], "fit_state_sha256": f"state-{seed}",
            "calibration_sha256": _entry(cal)["sha256"],
            "prediction_state_sha256": f"prediction-{seed}",
            "delivery_temperature": temperature,
        }
        frozen["june_seed_model_weights"][key] = june
        frozen["may_delivery_temperatures"][key] = temperature
        bundle_members[key] = {"source_arm": arm, "model_sha256": model_hash,
                               "delivery_temperature": temperature, "june_model_weight": june}
        for role, value in (("checkpoint", model_hash), ("fit_state", f"state-{seed}"),
                            ("prediction_state", f"prediction-{seed}")):
            files[f"seed{seed}_{role}"] = {"path": str(tmp_path / f"{role}-{seed}"), "sha256": value}
        files[f"seed{seed}_fit_report"] = _entry(fit)
        files[f"seed{seed}_may_calibration"] = _entry(cal)
    return bundle_members, files, prep, frozen


def test_member_order_and_weight_swaps_fail_closed(tmp_path: Path) -> None:
    members, files, prep, frozen = _members(tmp_path)
    freeze.verify_members(members, files, prep, frozen)
    swapped = copy.deepcopy(members)
    swapped["0"], swapped["1"] = swapped["1"], swapped["0"]
    with pytest.raises(ValueError, match="member semantics"):
        freeze.verify_members(swapped, files, prep, frozen)
    reversed_members = dict(reversed(list(members.items())))
    with pytest.raises(ValueError, match="member order"):
        freeze.verify_members(reversed_members, files, prep, frozen)
    bad_hash = copy.deepcopy(files)
    bad_hash["seed3_checkpoint"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="member link: 3 checkpoint"):
        freeze.verify_members(members, bad_hash, prep, frozen)


def test_path_alias_rejected_before_any_parent_read(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = Path("/archived/seed0/model.pt")
    monkeypatch.setattr(freeze, "frozen_paths", lambda: {"checkpoint": expected})
    bundle = {"protocol": "g0_research_frozen_v1", "research_only": True,
              "service_promotion": False, "independent_confirmation": False,
              "execution_code_commit": freeze.CODE_COMMIT,
              "evidence_head": freeze.SOURCE_HEAD,
              "files": {"checkpoint": {"path": "/archived/seed1/model.pt", "sha256": "same"}}}
    with pytest.raises(ValueError, match="path alias changed"):
        freeze.verify_semantics(bundle)


def test_independent_confirmation_claim_rejected_before_parent_read() -> None:
    bundle = {"protocol": "g0_research_frozen_v1", "research_only": True,
              "service_promotion": False, "independent_confirmation": True}
    with pytest.raises(ValueError, match="independent confirmation boundary"):
        freeze.verify_semantics(bundle)
