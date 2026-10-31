"""Metadata-only June 2025 calibration support and cost audit (ML-JUNE-SUPPORT-AUDIT-v1).

Reads only registered, hash-pinned files with explicit column projections. No training,
inference, calibration, scoring, labels, outcomes or coordinates. Eligibility for the
non-Cpanel June population is not reconstructed; it is reported as unknown with bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_REL = "scripts/audit_june_calibration_support.py"
CONFIG_REL = "configs/ML-JUNE-SUPPORT-AUDIT-v1.json"
KEY = ["game_pk", "at_bat_number", "pitch_number"]
FORBIDDEN_COLUMN_HINTS = (
    "event", "description", "label", "outcome", "prob", "loss", "plate_", "pfx_", "release_",
    "launch_", "sz_", "type", "score", "terminal", "supported_pa", "spin", "speed",
)


class AuditBoundaryError(RuntimeError):
    """Raised when a request would cross a registered data or privacy boundary."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_window(window: dict) -> tuple[date, date]:
    lo = date.fromisoformat(window["date_min"])
    hi = date.fromisoformat(window["date_max"])
    reject = int(window["reject_year_min"])
    if lo > hi:
        raise AuditBoundaryError("window date_min after date_max")
    if hi.year >= reject or lo.year >= reject or int(window["season"]) >= reject:
        raise AuditBoundaryError(f"window reaches {reject}+ data, which is OPE-only and not auditable here")
    if lo.year != int(window["season"]) or hi.year != int(window["season"]):
        raise AuditBoundaryError("window must stay inside the registered season")
    return lo, hi


def check_columns(requested: list[str], allowed: list[str]) -> list[str]:
    bad = [c for c in requested if c not in allowed]
    if bad:
        raise AuditBoundaryError(f"disallowed columns requested: {bad}")
    return list(requested)


def validate_config(cfg: dict) -> None:
    validate_window(cfg["window"])
    allowed = cfg["allowed_parquet_columns"]
    for name in ("event", "description", "type"):
        if name in allowed:
            raise AuditBoundaryError(f"allowlist must not contain {name}")
    for column in allowed:
        if column == "game_type":
            continue
        if any(hint in column for hint in FORBIDDEN_COLUMN_HINTS):
            raise AuditBoundaryError(f"allowlist column looks like outcome/coordinate data: {column}")
    for spec in cfg["sources"].values():
        if "columns" in spec:
            check_columns(spec["columns"], allowed)
    rule = cfg["support_rule"]
    if rule.get("combine") != "AND":
        raise AuditBoundaryError("support rule must combine games AND pitches")


def verified_path(spec: dict) -> Path:
    path = Path(spec["path"])
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise AuditBoundaryError(f"hash mismatch for {path}: {actual} != {spec['sha256']}")
    return path


def verify_sources(sources: dict) -> dict:
    """Hash every declared source before any read; any mismatch stops the audit."""
    return {name: str(verified_path(spec)) for name, spec in sources.items()}


def check_output(output: Path, cfg: dict) -> None:
    if output.exists():
        raise AuditBoundaryError(f"refusing to overwrite {output}")
    resolved = output.resolve()
    roots = [(ROOT / "results").resolve(), Path(cfg["outputs"]["allowed_attempt_root"]).resolve()]
    if not any(resolved.is_relative_to(root) for root in roots):
        raise AuditBoundaryError(f"output outside registered roots: {resolved}")


def read_projected(path: Path, columns: list[str], allowed: list[str], filters=None) -> pd.DataFrame:
    columns = check_columns(columns, allowed)
    table = pq.read_table(path, columns=columns, filters=filters)
    if table.column_names != columns:
        raise AuditBoundaryError(f"projection mismatch: {table.column_names}")
    return table.to_pandas()


def assert_unique_keys(frame: pd.DataFrame, name: str) -> None:
    dup = int(frame.duplicated(KEY).sum())
    if dup:
        raise AuditBoundaryError(f"{name}: {dup} duplicate pitch keys")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditBoundaryError(message)


def filter_window(frame: pd.DataFrame, window: dict) -> pd.DataFrame:
    lo, hi = validate_window(window)
    stamps = pd.to_datetime(frame["game_date"])
    if (stamps.dt.year >= int(window["reject_year_min"])).any():
        raise AuditBoundaryError("rows at or after the rejected year reached the audit")
    dates = stamps.dt.date
    keep = (dates >= lo) & (dates <= hi) & frame["game_type"].isin(window["game_types"])
    return frame.loc[keep].reset_index(drop=True)


def check_expected(items: list[tuple[str, object, object]]) -> dict:
    """Fail closed on any mismatch; an absent expected value stays 'unknown', never a pass."""
    out = {}
    for name, expected, actual in items:
        if expected is None:
            out[name] = {"expected": None, "actual": actual, "status": "unknown"}
            continue
        if expected != actual:
            raise AuditBoundaryError(f"expected count mismatch for {name}: {expected} != {actual}")
        out[name] = {"expected": expected, "actual": actual, "status": "match"}
    unknown = [k for k, v in out.items() if v["status"] == "unknown"]
    out["overall"] = "passed" if not unknown else "unknown_items_present"
    out["unknown_items"] = unknown
    return out


def support_table(frame: pd.DataFrame, column: str, levels: list, rule: dict) -> dict:
    out = {}
    for level in levels:
        part = frame.loc[frame[column] == level]
        pitches, games = int(len(part)), int(part["game_pk"].nunique())
        out[str(level)] = {
            "pitches": pitches,
            "games": games,
            "supported": bool(games >= rule["min_games"] and pitches >= rule["min_pitches"]),
        }
    return out


def bounded_status(lower: dict | None, upper: dict) -> str:
    """Eligible support status from a lower (known-eligible) and upper bound."""
    if lower is not None and lower["supported"]:
        return "supported"
    if not upper["supported"]:
        return "unsupported"
    return "indeterminate"


def add_groups(frame: pd.DataFrame, volume_of: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame["volume"] = frame["pitcher"].map(volume_of).fillna("zero")
    frame["game_role"] = (frame["pitcher"] == frame["starter_pitcher"]).map({True: "starter", False: "relief"})
    frame["hand"] = frame["p_throws"]
    frame["role_x_volume"] = frame["game_role"] + "|" + frame["volume"]
    frame["hand_x_volume"] = frame["hand"].astype(str) + "|" + frame["volume"]
    return frame


def group_levels(cfg: dict) -> dict:
    g = cfg["groups"]
    vol, role, hand = g["volume"]["levels"], g["game_role"]["levels"], g["hand"]["levels"]
    return {
        "volume": vol,
        "game_role": role,
        "hand": hand,
        "role_x_volume": [f"{r}|{v}" for r in role for v in vol],
        "hand_x_volume": [f"{h}|{v}" for h in hand for v in vol],
    }


def check_frozen_volume(panel: dict, expected_levels: list[str]) -> dict:
    """Check stored TRAIN-volume labels against the frozen cutoffs; any disagreement stops the audit."""
    thresholds = panel["volume_thresholds"]
    q25, q75 = float(thresholds["q25"]), float(thresholds["q75"])
    mismatches = []
    volume_of = {}
    for player in panel["train_players"]:
        n = player["train_pitches"]
        rederived = "low" if n <= q25 else ("middle" if n <= q75 else "high")
        if player["train_volume"] not in expected_levels:
            raise AuditBoundaryError(f"unknown volume level {player['train_volume']}")
        if rederived != player["train_volume"]:
            mismatches.append(int(player["pitcher"]))
        volume_of[int(player["pitcher"])] = player["train_volume"]
    require(not mismatches, f"stored TRAIN-volume labels disagree with frozen cutoffs for {mismatches[:10]}")
    return {"q25": q25, "q75": q75, "method": thresholds["method"], "players": len(volume_of),
            "stored_label_vs_frozen_cutoff_mismatches": 0, "volume_of": volume_of}


def official_predict_walls(status: dict, members: int) -> dict:
    require(status.get("status") == "completed", "official queue status is not completed")
    steps = status["steps"]
    for step in steps:
        require(step["status"] == "completed" and step["exit_code"] == 0, f"official step not clean: {step['stage']}")
    predict = [s for s in steps if s["stage"].startswith("g0-predict-")]
    require(len(predict) == members, f"expected {members} official predict steps, found {len(predict)}")
    return {
        "per_member_worker_wall_seconds": {s["stage"]: s["worker_wall_seconds"] for s in predict},
        "other_evaluation_steps_worker_wall_seconds": {
            s["stage"]: s["worker_wall_seconds"] for s in steps if s["phase"] == "evaluation" and s not in predict},
        "authoritative_worker_charged_seconds": status["authoritative_worker_charged_seconds"],
        "full_outer_queue_wall_seconds": status["full_outer_queue_wall_seconds"],
    }


def internal_ledger_crosscheck(ledger: list[dict]) -> dict:
    starts = {e["id"]: e for e in ledger if e.get("event") == "start"}
    stages = []
    for end in (e for e in ledger if e.get("event") == "end"):
        start = starts.get(end["id"], {})
        stages.append({"stage": start.get("stage"), "seed": start.get("seed"), "status": end.get("status"),
                       "seconds": end.get("seconds")})
    predict = sum(float(s["seconds"]) for s in stages if s["stage"] == "predict" and s["status"] == "completed")
    return {"label": "internal run ledger; cross-check only, NOT authoritative", "stages": stages,
            "predict_total_seconds": predict}


def cost_projection(cfg: dict, profile: dict, status: dict, ledger: list[dict], row_counts: dict) -> dict:
    cost = cfg["cost"]
    require(int(profile["draws"]) == int(cost["draws"]), "profile draws differ from registered draws")
    official = official_predict_walls(status, cost["members"])
    walls = list(official["per_member_worker_wall_seconds"].values())
    predict_total = float(sum(walls))
    ref_rows = cost["reference_eligible_rows"]
    worker_rate = predict_total / ref_rows
    profile_rate = float(profile["projection"]["seconds_per_row"])
    estimates = {}
    for name, rows in row_counts.items():
        estimates[name] = {
            "rows": rows,
            "profile_inference_only_seconds_5_members": profile_rate * rows * cost["members"],
            "full_worker_linear_proxy_seconds_5_members": worker_rate * rows,
        }
    return {
        "measured_components": {
            "authoritative_source": "official supervisor queue status.json (SHA-pinned)",
            "official": official,
            "observed_predict_bound": {
                "rows_per_member": ref_rows,
                "min_member_worker_wall_seconds": min(walls),
                "max_member_worker_wall_seconds": max(walls),
                "sum_5_member_worker_wall_seconds": predict_total,
            },
            "profile": {"draws": profile["draws"], "measured": profile["measured"],
                        "inference_seconds_per_row_per_member": profile_rate},
            "internal_ledger_crosscheck": internal_ledger_crosscheck(ledger),
        },
        "extrapolated_inference": {
            "profile_inference_only": "profile inference rate x rows x 5; excludes per-member interpreter/import/model-load/verification fixed overhead",
            "full_worker_linear_proxy": "sum of 5 official predict worker walls / 311,721 x rows; scales fixed load/import overhead linearly with rows, so it is a proxy, not a guaranteed upper cost bound even when rows are an upper bound",
            "estimates": estimates,
        },
        "unknown_costs": {
            "june_eligible_inventory_and_feature_preparation": "미측정",
            "fit": "미측정 (no fit is implied by this audit; any new fit is out of scope)",
            "scoring_and_bootstrap": "미측정",
            "calibration_fit": "미측정",
        },
        "guarantee": "none; extrapolations are not measurements and may not hold for different row mixes, device state or I/O",
    }


def implementation_provenance(cfg: dict) -> dict:
    script_sha = sha256_file(ROOT / SCRIPT_REL)
    require(script_sha == cfg["implementation"]["script_sha256"],
            f"executed script {script_sha} differs from registered {cfg['implementation']['script_sha256']}")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", SCRIPT_REL, CONFIG_REL], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    require(not dirty, f"script/config not committed: {dirty}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.strip()
    return {"executed_source_commit": commit, "executed_script_sha256": script_sha,
            "executed_config_sha256": sha256_file(ROOT / CONFIG_REL)}


def run(config_path: Path, output: Path) -> dict:
    t_start = time.perf_counter()
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    check_output(output, cfg)
    provenance = implementation_provenance(cfg)
    allowed, rule, src = cfg["allowed_parquet_columns"], cfg["support_rule"], cfg["sources"]
    levels = group_levels(cfg)

    t_verify = time.perf_counter()
    paths = {name: Path(p) for name, p in verify_sources(src).items()}
    verify_seconds = time.perf_counter() - t_verify

    t_work = time.perf_counter()
    panel = json.loads(paths["panel"].read_text())
    frozen = check_frozen_volume(panel, levels["volume"])
    volume_of = frozen.pop("volume_of")
    panel_ids = {int(p) for p in panel["pitcher_ids"]}

    lo, hi = validate_window(cfg["window"])
    filters = [("game_date", ">=", pd.Timestamp(lo)), ("game_date", "<=", pd.Timestamp(hi))]
    raw = read_projected(paths["processed_pitches"], src["processed_pitches"]["columns"], allowed, filters=filters)
    rows_read = int(len(raw))
    game_type_counts = {str(k): int(v) for k, v in raw["game_type"].value_counts().items()}
    june = filter_window(raw, cfg["window"])
    assert_unique_keys(june, "june_requested")
    june["pitcher"] = june["pitcher"].astype("int64")
    june = add_groups(june, volume_of)
    june["in_cpanel"] = june["pitcher"].isin(panel_ids)

    blend_keys = read_projected(paths["blend_keys"], src["blend_keys"]["columns"], allowed)
    blend_meta = read_projected(paths["blend_metadata"], src["blend_metadata"]["columns"], allowed)
    assert_unique_keys(blend_keys, "blend_keys")
    assert_unique_keys(blend_meta, "blend_metadata")
    require(blend_keys[KEY].reset_index(drop=True).equals(blend_meta[KEY].reset_index(drop=True)),
            "blend_keys and blend_metadata key order differ")

    joined = blend_meta.merge(june, on=KEY, how="left", suffixes=("_blend", ""), indicator=True)
    missing = int((joined["_merge"] != "both").sum())
    require(missing == 0, f"{missing} blend keys absent from June requested population")
    alignment = {
        "blend_keys_equal_blend_metadata_order": True,
        "blend_rows": int(len(blend_meta)),
        "blend_keys_missing_from_june_requested": missing,
        "pitcher_mismatch": int((joined["pitcher_blend"].astype("int64") != joined["pitcher"]).sum()),
        "batter_mismatch": int((joined["batter_blend"] != joined["batter"]).sum()),
        "in_cpanel_flag_false": int((~joined["in_cpanel_blend"].astype(bool)).sum()),
        "not_panel_pitcher": int((~joined["in_cpanel"]).sum()),
        "train_volume_mismatch": int((joined["train_volume"] != joined["volume"]).sum()),
        "game_role_mismatch": int((joined["game_role_blend"] != joined["game_role"]).sum()),
        "throwing_hand_mismatch": int((joined["throwing_hand"] != joined["hand"]).sum()),
        "month_values": sorted(joined["month"].astype(str).unique().tolist()),
    }
    failed = [k for k, v in alignment.items() if k.endswith(("mismatch", "_false", "_pitcher")) and v]
    require(not failed, f"Cpanel alignment failed: {failed}")

    blend_set = pd.MultiIndex.from_frame(blend_meta[KEY])
    june["cpanel_eligible"] = pd.MultiIndex.from_frame(june[KEY]).isin(blend_set)
    known_ineligible = june["in_cpanel"] & ~june["cpanel_eligible"]
    populations = {
        "full_june_requested": june,
        "cpanel_pitcher_requested": june.loc[june["in_cpanel"]],
        "cpanel_eligible_known_keys": june.loc[june["cpanel_eligible"]],
        "cpanel_pitcher_known_ineligible": june.loc[known_ineligible],
        "outside_cpanel_pitcher_requested": june.loc[~june["in_cpanel"]],
        "complement_of_eligible_cpanel_keys": june.loc[~june["cpanel_eligible"]],
        "full_june_tight_eligible_upper": june.loc[~known_ineligible],
    }
    counts = {name: {"pitches": int(len(f)), "games": int(f["game_pk"].nunique()),
                     "pitchers": int(f["pitcher"].nunique())} for name, f in populations.items()}
    games = {name: set(f["game_pk"]) for name, f in populations.items()}
    n = {name: c["pitches"] for name, c in counts.items()}
    partitions = {
        "cpanel_pitcher_requested + outside_cpanel_pitcher_requested == full":
            n["cpanel_pitcher_requested"] + n["outside_cpanel_pitcher_requested"] == n["full_june_requested"],
        "cpanel_eligible_known_keys + complement_of_eligible_cpanel_keys == full":
            n["cpanel_eligible_known_keys"] + n["complement_of_eligible_cpanel_keys"] == n["full_june_requested"],
        "outside_cpanel_pitcher_requested + cpanel_pitcher_known_ineligible == complement_of_eligible_cpanel_keys":
            n["outside_cpanel_pitcher_requested"] + n["cpanel_pitcher_known_ineligible"]
            == n["complement_of_eligible_cpanel_keys"],
    }
    require(all(partitions.values()), f"key partition failed: {partitions}")
    overlap = {
        "exact_key_partitions": partitions,
        "games_shared": {
            "cpanel_eligible_known_keys & complement_of_eligible_cpanel_keys":
                len(games["cpanel_eligible_known_keys"] & games["complement_of_eligible_cpanel_keys"]),
            "cpanel_eligible_known_keys & outside_cpanel_pitcher_requested":
                len(games["cpanel_eligible_known_keys"] & games["outside_cpanel_pitcher_requested"]),
            "cpanel_pitcher_requested & outside_cpanel_pitcher_requested":
                len(games["cpanel_pitcher_requested"] & games["outside_cpanel_pitcher_requested"]),
        },
        "outside_only_games": len(games["outside_cpanel_pitcher_requested"] - games["cpanel_pitcher_requested"]),
        "known_ineligible_cpanel_pitches": n["cpanel_pitcher_known_ineligible"],
    }

    p4 = json.loads(paths["p4_preparation"].read_text())
    p11 = json.loads(paths["p11_preparation"].read_text())
    blend_cov = p4.get("coverage", {}).get("blend", {})
    first = cfg["first_attempt_counts"]
    expected = check_expected([
        ("cpanel_eligible_pitches (config)", cfg["expected_counts"]["cpanel_eligible_pitches"],
         n["cpanel_eligible_known_keys"]),
        ("cpanel_eligible_games (config)", cfg["expected_counts"]["cpanel_eligible_games"],
         counts["cpanel_eligible_known_keys"]["games"]),
        ("cpanel_requested_pitches (config)", cfg["expected_counts"]["cpanel_requested_pitches"],
         n["cpanel_pitcher_requested"]),
        ("p4 coverage.blend.requested_pitches", blend_cov.get("requested_pitches"), n["cpanel_pitcher_requested"]),
        ("p4 coverage.blend.eligible_pitches", blend_cov.get("eligible_pitches"), n["cpanel_eligible_known_keys"]),
        ("p4 samples.blend.games", p4.get("samples", {}).get("blend", {}).get("games"),
         counts["cpanel_eligible_known_keys"]["games"]),
        ("first attempt full_june_requested", first["full_june_requested"], n["full_june_requested"]),
        ("first attempt outside_cpanel_pitcher_requested", first["outside_cpanel_pitcher_requested"],
         n["outside_cpanel_pitcher_requested"]),
        ("full_june_eligible_pitches (no frozen June eligible inventory)", None, "unknown"),
    ])
    require(p11["coverage"] == p4["coverage"]["mlb_dev"], "p11 coverage disagrees with p4 mlb_dev coverage")

    support = {}
    for group, lvls in levels.items():
        tables = {name: support_table(frame, group, lvls, rule) for name, frame in populations.items()}
        rows = {}
        for level in map(str, lvls):
            rows[level] = {name: tables[name][level] for name in tables}
            rows[level]["full_june_eligible_status"] = bounded_status(
                tables["cpanel_eligible_known_keys"][level], tables["full_june_tight_eligible_upper"][level])
            rows[level]["non_cpanel_eligible_status"] = bounded_status(
                None, tables["outside_cpanel_pitcher_requested"][level])
        support[group] = rows

    profile = json.loads(paths["p11_profile"].read_text())
    status = json.loads(paths["official_queue_status"].read_text())
    ledger = [json.loads(line) for line in paths["p11_ledger"].read_text().splitlines() if line.strip()]
    cost = cost_projection(cfg, profile, status, ledger, {
        "full_june_tight_eligible_upper": n["full_june_tight_eligible_upper"],
        "full_june_requested": n["full_june_requested"],
        "outside_cpanel_pitcher_requested": n["outside_cpanel_pitcher_requested"],
        "cpanel_eligible_known_keys": n["cpanel_eligible_known_keys"],
    })
    work_seconds = time.perf_counter() - t_work

    result = {
        "audit_id": cfg["audit_id"],
        "attempt": cfg["attempt"],
        "implementation": provenance,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "seed": "not applicable (deterministic metadata counts; no sampling)",
        "data_version": {k: v["sha256"] for k, v in src.items()},
        "sources_verified_before_read": sorted(paths),
        "columns_read": {k: v.get("columns", v.get("fields")) for k, v in src.items()},
        "window": cfg["window"],
        "rows_read_after_date_pushdown": rows_read,
        "game_type_counts_in_date_window": game_type_counts,
        "frozen_volume": frozen,
        "population_definitions": cfg["population_definitions"],
        "eligibility": {
            "cpanel": "known: EXP-P4-001 blend_keys (4,821 eligible of 5,212 requested; 391 known ineligible)",
            "full_june_and_non_cpanel": "unknown",
            "reason": "No frozen eligible whole-MLB June key inventory exists: EXP-P4-001/EXP-P11-001 mlb_dev covers 2025-07..2025-09 only; eligibility needs outcome/coordinate fields that this audit must not read.",
            "bounds": "full-June eligible: lower = cpanel_eligible_known_keys, tight upper = full_june_tight_eligible_upper (requested minus 391 known-ineligible Cpanel rows). Non-Cpanel eligible (the eligible part of both complements, identical): lower 0, upper = outside_cpanel_pitcher_requested.",
            "mlb_dev_reference": {"samples": p4["samples"]["mlb_dev"], "coverage": p4["coverage"]["mlb_dev"]},
        },
        "counts": counts,
        "overlap_and_exclusions": overlap,
        "cpanel_alignment": alignment,
        "expected_count_check": expected,
        "support_rule": rule,
        "support": support,
        "cost": cost,
        "audit_only_cost": {
            "label": "audit-only; not a candidate experiment cost",
            "in_process_source_verification_seconds": verify_seconds,
            "in_process_read_and_aggregate_seconds": work_seconds,
            "in_process_total_seconds_before_write": time.perf_counter() - t_start,
            "excluded": "interpreter start, imports and result write; external command wall is measured outside and recorded in the report",
        },
        "prior_exposure_note": "EXP-P11-001 (full G0 evaluation) and EXP-P11-002 (bounded correction) are existing exposed evidence; this audit reads none of their scores or predictions. Independence of any future June data is not claimed here (exposure audit owned by Astra).",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / CONFIG_REL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"counts": result["counts"], "audit_only_cost": result["audit_only_cost"]}, indent=2))


if __name__ == "__main__":
    main()
