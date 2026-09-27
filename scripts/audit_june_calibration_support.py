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


def read_projected(spec: dict, allowed: list[str], filters=None) -> pd.DataFrame:
    columns = check_columns(spec["columns"], allowed)
    table = pq.read_table(verified_path(spec), columns=columns, filters=filters)
    if table.column_names != columns:
        raise AuditBoundaryError(f"projection mismatch: {table.column_names}")
    return table.to_pandas()


def assert_unique_keys(frame: pd.DataFrame, name: str) -> None:
    dup = int(frame.duplicated(KEY).sum())
    if dup:
        raise AuditBoundaryError(f"{name}: {dup} duplicate pitch keys")


def filter_window(frame: pd.DataFrame, window: dict) -> pd.DataFrame:
    lo, hi = validate_window(window)
    dates = pd.to_datetime(frame["game_date"]).dt.date
    if (pd.to_datetime(frame["game_date"]).dt.year >= int(window["reject_year_min"])).any():
        raise AuditBoundaryError("rows at or after the rejected year reached the audit")
    keep = (dates >= lo) & (dates <= hi) & frame["game_type"].isin(window["game_types"])
    return frame.loc[keep].reset_index(drop=True)


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
    """Eligible support status from a lower (known-eligible) and upper (requested) bound."""
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
    thresholds = panel["volume_thresholds"]
    q25, q75 = float(thresholds["q25"]), float(thresholds["q75"])
    mismatches = 0
    volume_of = {}
    for player in panel["train_players"]:
        n = player["train_pitches"]
        rederived = "low" if n <= q25 else ("middle" if n <= q75 else "high")
        mismatches += int(rederived != player["train_volume"])
        if player["train_volume"] not in expected_levels:
            raise AuditBoundaryError(f"unknown volume level {player['train_volume']}")
        volume_of[int(player["pitcher"])] = player["train_volume"]
    return {"q25": q25, "q75": q75, "method": thresholds["method"], "players": len(volume_of),
            "stored_label_vs_frozen_cutoff_mismatches": mismatches, "volume_of": volume_of}


def cost_projection(cfg: dict, profile: dict, ledger: list[dict], row_counts: dict) -> dict:
    cost = cfg["cost"]
    if int(profile["draws"]) != int(cost["draws"]):
        raise AuditBoundaryError("profile draws differ from registered draws")
    ends = [e for e in ledger if e.get("event") == "end"]
    starts = {e["id"]: e for e in ledger if e.get("event") == "start"}
    stages = []
    for end in ends:
        start = starts.get(end["id"], {})
        stages.append({"stage": start.get("stage"), "seed": start.get("seed"), "status": end.get("status"),
                       "seconds": end.get("seconds")})
    predict = [s for s in stages if s["stage"] == "predict" and s["status"] == "completed"]
    if len(predict) != cost["members"]:
        raise AuditBoundaryError(f"expected {cost['members']} completed predict stages, found {len(predict)}")
    predict_total = sum(float(s["seconds"]) for s in predict)
    ledger_rate = predict_total / cost["reference_eligible_rows"]
    profile_rate = float(profile["projection"]["seconds_per_row"])
    extrapolated = {}
    for name, rows in row_counts.items():
        extrapolated[name] = {
            "rows": rows,
            "five_member_seconds_from_profile_rate": profile_rate * rows * cost["members"],
            "five_member_seconds_from_measured_ledger_rate": ledger_rate * rows,
        }
    return {
        "measured_components": {
            "source": "EXP-P11-001 profile.json and ledger.jsonl (existing; no new profile)",
            "profile_draws": profile["draws"],
            "profile_measured": profile["measured"],
            "profile_seconds_per_row_per_member": profile_rate,
            "ledger_stages": stages,
            "ledger_predict_total_seconds_5_members": predict_total,
            "ledger_predict_rows_per_member": cost["reference_eligible_rows"],
            "ledger_seconds_per_row_5_members": ledger_rate,
        },
        "extrapolated_inference": {
            "method": "linear in rows at 400 draws on the same mps host; rows for full/complement June are requested (upper bound), not eligible",
            "estimates": extrapolated,
        },
        "unknown_costs": {
            "june_eligible_inventory_and_feature_preparation": "미측정",
            "fit": "미측정 (no fit is implied by this audit; any new fit is out of scope)",
            "scoring_and_bootstrap": "미측정",
            "calibration_fit": "미측정",
        },
        "guarantee": "none; extrapolations are not measurements and may not hold for different row mixes, device state or I/O",
    }


def run(config_path: Path, output: Path, force: bool = False) -> dict:
    wall_start = time.perf_counter()
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    if output.exists() and not force:
        raise AuditBoundaryError(f"refusing to overwrite {output}")
    runs_root = "/Volumes/T7 Shield/pitcheezy/pitchmdp/runs"
    if str(output.resolve()).startswith(runs_root):
        raise AuditBoundaryError("output must not be written into existing run directories")
    allowed, rule, src = cfg["allowed_parquet_columns"], cfg["support_rule"], cfg["sources"]
    levels = group_levels(cfg)

    panel = json.loads(verified_path(src["panel"]).read_text())
    frozen = check_frozen_volume(panel, levels["volume"])
    volume_of = frozen.pop("volume_of")
    panel_ids = {int(p) for p in panel["pitcher_ids"]}

    lo, hi = validate_window(cfg["window"])
    filters = [("game_date", ">=", pd.Timestamp(lo)), ("game_date", "<=", pd.Timestamp(hi))]
    raw = read_projected(src["processed_pitches"], allowed, filters=filters)
    rows_read = int(len(raw))
    june = filter_window(raw, cfg["window"])
    excluded_game_type = {str(k): int(v) for k, v in raw["game_type"].value_counts().items()}
    assert_unique_keys(june, "june_requested")
    june["pitcher"] = june["pitcher"].astype("int64")
    june = add_groups(june, volume_of)
    june["in_cpanel"] = june["pitcher"].isin(panel_ids)

    blend_keys = read_projected(src["blend_keys"], allowed)
    blend_meta = read_projected(src["blend_metadata"], allowed)
    assert_unique_keys(blend_keys, "blend_keys")
    assert_unique_keys(blend_meta, "blend_metadata")
    key_order_equal = bool(blend_keys[KEY].reset_index(drop=True).equals(blend_meta[KEY].reset_index(drop=True)))

    joined = blend_meta.merge(june, on=KEY, how="left", suffixes=("_blend", ""), indicator=True)
    missing = int((joined["_merge"] != "both").sum())
    if missing:
        raise AuditBoundaryError(f"{missing} blend keys absent from June requested population")
    alignment = {
        "blend_keys_equal_blend_metadata_order": key_order_equal,
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
    if any(alignment[k] for k in ("pitcher_mismatch", "batter_mismatch", "in_cpanel_flag_false", "not_panel_pitcher",
                                   "train_volume_mismatch", "game_role_mismatch", "throwing_hand_mismatch")):
        raise AuditBoundaryError(f"Cpanel alignment failed: {alignment}")

    blend_set = pd.MultiIndex.from_frame(blend_meta[KEY])
    june_idx = pd.MultiIndex.from_frame(june[KEY])
    june["cpanel_eligible"] = june_idx.isin(blend_set)
    populations = {
        "full_june_requested": june,
        "cpanel_requested": june.loc[june["in_cpanel"]],
        "cpanel_eligible": june.loc[june["cpanel_eligible"]],
        "complement_requested": june.loc[~june["in_cpanel"]],
    }
    p4 = json.loads(verified_path(src["p4_preparation"]).read_text())
    counts = {}
    for name, frame in populations.items():
        counts[name] = {"pitches": int(len(frame)), "games": int(frame["game_pk"].nunique()),
                        "pitchers": int(frame["pitcher"].nunique())}
    cp_games = set(populations["cpanel_eligible"]["game_pk"])
    cp_req_games = set(populations["cpanel_requested"]["game_pk"])
    comp_games = set(populations["complement_requested"]["game_pk"])
    overlap = {
        "cpanel_eligible_games_shared_with_complement": len(cp_games & comp_games),
        "cpanel_requested_games_shared_with_complement": len(cp_req_games & comp_games),
        "complement_only_games": len(comp_games - cp_req_games),
        "cpanel_requested_not_eligible_pitches": int(len(populations["cpanel_requested"]) - len(populations["cpanel_eligible"])),
        "key_partition_exact": bool(len(populations["cpanel_requested"]) + len(populations["complement_requested"]) == len(june)),
    }
    expected = cfg["expected_counts"]
    expected_check = {
        "cpanel_eligible_pitches": [expected["cpanel_eligible_pitches"], counts["cpanel_eligible"]["pitches"]],
        "cpanel_eligible_games": [expected["cpanel_eligible_games"], counts["cpanel_eligible"]["games"]],
        "cpanel_requested_pitches": [expected["cpanel_requested_pitches"], counts["cpanel_requested"]["pitches"]],
        "p4_coverage_blend": p4["coverage"]["blend"],
    }
    expected_check["all_match"] = all(a == b for a, b in list(expected_check.values())[:3])

    support = {}
    for group, lvls in levels.items():
        tables = {name: support_table(frame, group, lvls, rule) for name, frame in populations.items()}
        rows = {}
        for level in map(str, lvls):
            rows[level] = {name: tables[name][level] for name in tables}
            rows[level]["full_june_eligible_status"] = bounded_status(tables["cpanel_eligible"][level],
                                                                      tables["full_june_requested"][level])
            rows[level]["complement_eligible_status"] = bounded_status(None, tables["complement_requested"][level])
        support[group] = rows

    profile = json.loads(verified_path(src["p11_profile"]).read_text())
    ledger = [json.loads(line) for line in verified_path(src["p11_ledger"]).read_text().splitlines() if line.strip()]
    cost = cost_projection(cfg, profile, ledger, {
        "full_june_requested_upper_bound": counts["full_june_requested"]["pitches"],
        "complement_requested_upper_bound": counts["complement_requested"]["pitches"],
        "cpanel_eligible_known": counts["cpanel_eligible"]["pitches"],
    })

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    result = {
        "audit_id": cfg["audit_id"],
        "config_sha256": sha256_file(config_path),
        "code_commit_at_run": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "seed": "not applicable (deterministic metadata counts; no sampling)",
        "data_version": {k: v["sha256"] for k, v in src.items()},
        "columns_read": {k: v.get("columns", v.get("fields")) for k, v in src.items()},
        "window": cfg["window"],
        "rows_read_after_date_pushdown": rows_read,
        "game_type_counts_in_date_window": excluded_game_type,
        "frozen_volume": frozen,
        "eligibility": {
            "cpanel": "known: EXP-P4-001 blend_keys (4,821 eligible of 5,212 requested)",
            "full_june_and_complement": "unknown",
            "reason": "No frozen eligible whole-MLB June key inventory exists: EXP-P4-001/EXP-P11-001 mlb_dev covers 2025-07..2025-09 only; eligibility needs outcome/coordinate fields that this audit must not read. Full/complement eligible support is bounded: lower = Cpanel eligible subset (full) or 0 (complement), upper = requested counts.",
            "mlb_dev_reference": {"samples": p4["samples"]["mlb_dev"], "coverage": p4["coverage"]["mlb_dev"]},
        },
        "counts": counts,
        "overlap_and_exclusions": overlap,
        "cpanel_alignment": alignment,
        "expected_count_check": expected_check,
        "support_rule": rule,
        "support": support,
        "cost": cost,
        "prior_exposure_note": "EXP-P11-001 (full G0 evaluation) and EXP-P11-002 (bounded correction) are existing exposed evidence; this audit reads none of their scores or predictions. Independence of any future June data is not claimed here (exposure audit owned by Astra).",
    }
    result["audit_wall_seconds"] = time.perf_counter() - wall_start
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ML-JUNE-SUPPORT-AUDIT-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ML-JUNE-SUPPORT-AUDIT-v1.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, force=args.force)
    print(json.dumps({"counts": result["counts"], "overlap": result["overlap_and_exclusions"],
                      "audit_wall_seconds": result["audit_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
