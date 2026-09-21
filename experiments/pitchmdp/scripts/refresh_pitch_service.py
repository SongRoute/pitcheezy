"""Immutable dated profiles/repertoire over the frozen historical service.

No fitting, source rewriting, raw downloads, or extension of model support.
Run with the configured project .venv. Only trusted local bundles may be loaded.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import numpy as np
import pandas as pd

from pitchmdp.archetypes import (
    RELIABILITY_COLUMNS, STYLE_COLUMNS, STYLE_NAMES, add_batter_style_history,
)
from pitchmdp.data import KEY, RAW_ALLOWLIST, hash_file, utc_now
from minimal_pitch_service import Engine, RequestError, _iso_date, validate_request

VOLUME = Path("/Volumes/T7 Shield")
RUNS = VOLUME / "pitcheezy/pitchmdp/runs"
MODEL_CALIBRATION_CUTOFF = "2025-06-30"
LOOKBACK_DAYS = 90
MIN_PITCHES = 20
LOW_RELIABILITY = .5


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def metadata_hash(metadata):
    """Bind a sidecar to the exact original metadata, including model provenance."""
    return hashlib.sha256(_json_bytes(metadata)).hexdigest()


def _prior_frame(frame, as_of):
    effective = pd.Timestamp(_iso_date(as_of, "as_of"))
    dates = pd.to_datetime(frame.game_date, errors="raise").dt.normalize()
    if dates.isna().any() or not dates.dt.year.isin([2023, 2024, 2025]).all():
        raise ValueError("Only approved 2023–2025 source dates are allowed")
    result = frame.loc[dates.lt(effective)].copy()
    result["game_date"] = dates.loc[result.index]
    return result, effective


def refresh_profiles(frame, as_of):
    """Return cumulative, strictly prior-date snapshots from the complete pitch pool.

    Zero-observation probes invoke the existing style pipeline after all prior
    dates, including every pitch on the last date. For a 2026+ effective date,
    translate the entire synthetic timeline backward by a constant duration so
    the canonical pipeline's real-data year guard is preserved. Actual input
    dates are separately restricted to 2023–2025 before this translation.
    """
    prior, effective = _prior_frame(frame, as_of)
    columns = ["batter", "game_date", "events", "description", "is_pa_terminal"]
    if "launch_angle" in prior:
        columns.append("launch_angle")
    missing = set(columns).difference(prior.columns)
    if missing:
        raise ValueError(f"Missing complete-pool style columns: {sorted(missing)}")
    ids = pd.to_numeric(prior.batter, errors="raise")
    if ids.isna().any() or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError("Historical batter IDs must be positive integers")
    batters = sorted(ids.astype("int64").unique().tolist())
    history = prior[columns].copy()
    history["batter"] = ids.astype("int64")
    # ID 0 is solely a synthetic unknown-batter probe, never an input player.
    probes = pd.DataFrame({"batter": [*batters, 0], "game_date": effective,
                           "events": "", "description": "", "is_pa_terminal": False})
    combined = pd.concat([history, probes], ignore_index=True)
    shift = max(pd.Timedelta(0), effective - pd.Timestamp("2025-12-31"))
    combined["game_date"] = pd.to_datetime(combined.game_date) - shift
    computed = add_batter_style_history(combined).tail(len(probes))
    source_max = prior.game_date.max().date().isoformat() if len(prior) else None
    source_min = prior.game_date.min().date().isoformat() if len(prior) else None
    cutoff = source_max or (effective.date() - timedelta(days=1)).isoformat()
    last_seen = prior.groupby("batter").game_date.max()
    profiles = {}
    default = None
    for _, row in computed.iterrows():
        batter = int(row.batter)
        reliability = [float(row[k]) for k in RELIABILITY_COLUMNS]
        flags = []
        if np.mean(reliability) < LOW_RELIABILITY:
            flags.append("low_batter_profile_reliability")
        last = last_seen.get(batter)
        if last is not None and (effective - last).days > LOOKBACK_DAYS:
            flags.append("stale_batter_observations")
        profile = {"rates": [float(row[k]) for k in STYLE_COLUMNS],
                   "reliabilities": reliability, "as_of": cutoff,
                   "last_observed_date": last.date().isoformat() if last is not None else None,
                   "flags": flags}
        if batter == 0:
            profile["reliabilities"] = [0.] * len(STYLE_NAMES)
            profile["flags"].append("missing_batter_zero_reliability")
            default = profile
        else:
            profiles[str(batter)] = profile
    return {"profiles": profiles, "default_profile": default,
            "profile_cutoff": cutoff, "effective_date": as_of,
            "source_date_min": source_min, "source_date_max": source_max,
            "profile_order": list(STYLE_NAMES), "source_rows_before_date": len(prior)}


def build_sidecar(frame, as_of, original_metadata, source_identity=None):
    """Keep original pitcher/action support; mask actions below 20 pitches/90 days.

    A pitcher with no recently supported original type remains listed but cannot
    be recommended. There is deliberately no stale or newly-seen-type fallback.
    """
    snapshot = refresh_profiles(frame, as_of)
    prior, effective = _prior_frame(frame, as_of)
    missing = {"pitcher", "pitch_type", "p_throws"}.difference(prior.columns)
    if missing:
        raise ValueError(f"Missing repertoire columns: {sorted(missing)}")
    original_pitchers = original_metadata.get("pitchers", {})
    if not original_pitchers:
        raise ValueError("Original bundle has no supported pitchers")
    recent = prior.loc[prior.game_date.ge(effective - pd.Timedelta(days=LOOKBACK_DAYS))]
    pitchers, counts = {}, {}
    for pid, original in original_pitchers.items():
        rows = prior.loc[prior.pitcher.eq(int(pid))]
        recent_rows = recent.loc[recent.pitcher.eq(int(pid))]
        observed = recent_rows.pitch_type.value_counts()
        original_types = original["pitch_types"]
        allowed = [kind for kind in original_types if int(observed.get(kind, 0)) >= MIN_PITCHES]
        hands = rows.loc[rows.p_throws.isin(["L", "R"]), "p_throws"]
        p_throws = str(hands.mode().iloc[0]) if len(hands) else None
        flags = []
        if not len(rows):
            flags.append("missing_pitcher_observations")
        if p_throws is None:
            flags.append("missing_pitcher_handedness")
        elif hands.nunique() != 1 or p_throws != original["p_throws"]:
            flags.append("pitcher_handedness_conflicts_with_frozen_support")
        if len(allowed) < len(original_types):
            flags.append("original_pitch_types_masked_for_recent_support")
        if not allowed:
            flags.append("no_recent_supported_pitch_types")
        unavailable = p_throws is None or hands.nunique() != 1 or p_throws != original["p_throws"] or not allowed
        pitchers[pid] = {**deepcopy(original), "p_throws": p_throws,
                         "pitch_types": allowed, "available": not unavailable,
                         "original_pitch_types": list(original_types),
                         "masked_pitch_types": [kind for kind in original_types if kind not in allowed],
                         "unmodeled_recent_pitch_types": sorted(set(observed.index) - set(original_types)),
                         "recent_observed_counts": {str(kind): int(n) for kind, n in observed.items()},
                         "last_observed_date": rows.game_date.max().date().isoformat() if len(rows) else None,
                         "flags": flags}
        counts[pid] = {kind: int(observed[kind]) for kind in allowed}
    return {"schema_version": 1, "created_at_utc": utc_now(), **snapshot,
            "original_metadata_sha256": metadata_hash(original_metadata),
            "model_calibration_cutoff": max(MODEL_CALIBRATION_CUTOFF,
                original_metadata.get("training_cutoff", MODEL_CALIBRATION_CUTOFF),
                original_metadata.get("calibration_cutoff", MODEL_CALIBRATION_CUTOFF)),
            "pitchers": pitchers, "repertoire_counts": counts,
            "source_identity": deepcopy(source_identity or {}),
            "repertoire_rule": {"lookback_days": LOOKBACK_DAYS, "minimum_type_count": MIN_PITCHES,
                                "interval": "[effective_date - 90 days, effective_date)",
                                "fallback": "none; unavailable pitchers raise an error"},
            "scope": "Historical 2023–2025 observations only; no 2026/current feed. "
                     "Models, calibration, delivery pools, continuation and action support stay frozen."}


def _validate_profile(profile, cutoff):
    if not isinstance(profile, dict):
        raise ValueError("Invalid profile")
    for key, upper in (("rates", [1, 1, 1, 1, 3, 1]), ("reliabilities", [1] * 6)):
        values = profile.get(key)
        if not isinstance(values, list) or len(values) != 6 or any(
            type(x) not in (float, int) or not np.isfinite(x) or x < 0 or x > hi
            for x, hi in zip(values, upper)
        ):
            raise ValueError(f"Invalid profile {key}")
    if _iso_date(profile.get("as_of"), "profile.as_of") > cutoff:
        raise ValueError("Profile includes data after sidecar cutoff")


def validate_sidecar(sidecar, original_metadata=None):
    if sidecar.get("schema_version") != 1:
        raise ValueError("Unsupported sidecar schema_version")
    effective = _iso_date(sidecar.get("effective_date"), "effective_date")
    cutoff = _iso_date(sidecar.get("profile_cutoff"), "profile_cutoff")
    model_cutoff = _iso_date(sidecar.get("model_calibration_cutoff"), "model_calibration_cutoff")
    if cutoff >= effective or model_cutoff < date.fromisoformat(MODEL_CALIBRATION_CUTOFF):
        raise ValueError("Invalid sidecar cutoff")
    if sidecar.get("profile_order") != list(STYLE_NAMES):
        raise ValueError("Sidecar profile order differs from frozen model")
    for key in ("source_date_min", "source_date_max"):
        if sidecar.get(key) is not None:
            parsed = _iso_date(sidecar[key], key)
            if parsed.year not in (2023, 2024, 2025) or parsed > cutoff:
                raise ValueError("Invalid sidecar source date")
    for profile in [*sidecar["profiles"].values(), sidecar["default_profile"]]:
        _validate_profile(profile, cutoff)
    if any(sidecar["default_profile"]["reliabilities"]):
        raise ValueError("Unknown batter fallback must have zero reliability")
    for pid, info in sidecar["pitchers"].items():
        kinds = info["pitch_types"]
        counts = sidecar["repertoire_counts"][pid]
        if len(kinds) != len(set(kinds)) or set(counts) != set(kinds) or any(
            type(n) is not int or n < MIN_PITCHES for n in counts.values()
        ):
            raise ValueError("Invalid refreshed repertoire counts")
        if info["available"] and (not kinds or info["p_throws"] not in ("L", "R")):
            raise ValueError("Available pitcher lacks repertoire or handedness")
    if original_metadata is not None:
        if metadata_hash(original_metadata) != sidecar["original_metadata_sha256"]:
            raise ValueError("Sidecar belongs to a different original bundle")
        if set(sidecar["pitchers"]) != set(original_metadata["pitchers"]):
            raise ValueError("Sidecar must retain exactly the original supported pitchers")
        for pid, info in sidecar["pitchers"].items():
            original = original_metadata["pitchers"][pid]
            if not set(info["pitch_types"]).issubset(original["pitch_types"]):
                raise ValueError("Sidecar introduces unsupported model actions")
            if info["available"] and info["p_throws"] != original["p_throws"]:
                raise ValueError("Sidecar changes supported pitcher handedness")
    return sidecar


def new_run_directory(output):
    """Reserve a fresh direct child of the approved, mounted runs directory."""
    output = Path(output)
    if not VOLUME.is_mount() or output.parent.resolve() != RUNS.resolve():
        raise ValueError(f"Output must be a new directory directly under mounted {RUNS}")
    if output.name in ("", ".", "..") or output.is_symlink():
        raise ValueError("Invalid output run directory")
    output.mkdir(exist_ok=False)
    return output


def save_sidecar(sidecar, output):
    validate_sidecar(sidecar)
    payload = _json_bytes(sidecar)
    output = new_run_directory(output)
    with (output / "snapshot.json").open("xb") as stream:
        stream.write(payload)
    manifest = {"schema_version": 1, "created_at_utc": utc_now(),
                "sha256": {"snapshot.json": hashlib.sha256(payload).hexdigest()},
                "effective_date": sidecar["effective_date"],
                "source_date_max": sidecar["source_date_max"],
                "original_metadata_sha256": sidecar["original_metadata_sha256"]}
    with (output / "snapshot_manifest.json").open("xb") as stream:
        stream.write(_json_bytes(manifest))
    return output


def load_sidecar(path, original_metadata=None):
    path = Path(path)
    manifest = json.loads((path / "snapshot_manifest.json").read_text())
    payload = (path / "snapshot.json").read_bytes()
    if manifest.get("schema_version") != 1 or manifest.get("sha256") != {
        "snapshot.json": hashlib.sha256(payload).hexdigest()
    }:
        raise ValueError("Sidecar hash mismatch")
    sidecar = validate_sidecar(json.loads(payload), original_metadata)
    for key in ("effective_date", "source_date_max", "original_metadata_sha256"):
        if manifest.get(key) != sidecar[key]:
            raise ValueError(f"Sidecar manifest mismatch: {key}")
    return sidecar


def validate_calibration(calibration, original_metadata, requested):
    """A ten-intercept repair is usable only after its independent retention gate."""
    if not isinstance(calibration, dict) or type(calibration.get("accepted")) is not bool:
        raise ValueError("Calibration must declare accepted as a boolean")
    if not calibration["accepted"]:
        return None
    if calibration.get("original_metadata_sha256") != metadata_hash(original_metadata):
        raise ValueError("Calibration belongs to a different original bundle")
    available = _iso_date(calibration.get("available_from"), "calibration.available_from")
    if available < date(2025, 8, 16):
        raise ValueError("Calibration availability precedes the frozen retention gate")
    if requested < available:
        raise RequestError("date precedes accepted calibration availability", "calibration_date_conflict")
    bias = calibration.get("bias")
    if not isinstance(bias, list) or len(bias) != 10 or any(
        type(x) not in (float, int) or not np.isfinite(x) for x in bias
    ):
        raise ValueError("Calibration requires ten finite class intercepts")
    return np.asarray(bias, dtype=float)


def recommend_with_sidecar(engine, request, sidecar, calibration=None):
    """Apply metadata inside the engine lock, restoring it even on inference errors."""
    if not isinstance(request, dict):
        raise RequestError("Expected a JSON object")
    requested = _iso_date(request.get("date"), "date")
    with engine._lock:
        original = engine.metadata
        validate_sidecar(sidecar, original)
        if requested <= _iso_date(sidecar["model_calibration_cutoff"], "model cutoff"):
            raise RequestError("date must follow frozen model/calibration cutoff", "model_date_conflict")
        if requested < _iso_date(sidecar["effective_date"], "snapshot effective_date"):
            raise RequestError("date must be on or after snapshot effective_date", "profile_date_conflict")
        bias = validate_calibration(calibration, original, requested) if calibration is not None else None
        if "batter_profile" in request:
            profile = request["batter_profile"]
            if not isinstance(profile, dict):
                raise RequestError("batter_profile must be an object")
            profile_date = _iso_date(profile.get("as_of"), "batter_profile.as_of")
            if profile_date >= requested:
                raise RequestError("Explicit profile as_of must strictly precede date", "profile_date_conflict")
            for key in ("source_date_max", "last_observed_date"):
                if profile.get(key) is not None and _iso_date(profile[key], key) > profile_date:
                    raise RequestError("Explicit profile source dates exceed its as_of", "profile_date_conflict")
        refreshed = deepcopy(original)
        refreshed.update({key: deepcopy(sidecar[key]) for key in
                          ("profiles", "default_profile", "profile_cutoff", "repertoire_counts")})
        refreshed["pitchers"] = {pid: deepcopy(info) for pid, info in sidecar["pitchers"].items()
                                 if info["available"]}
        info = sidecar["pitchers"].get(str(request.get("pitcher_id")))
        if info is not None and not info["available"]:
            raise RequestError("Pitcher has no usable prior-date supported repertoire/handedness", "unavailable_pitcher")
        row = validate_request(request, refreshed)
        profile = request.get("batter_profile")
        if profile is None:
            profile = refreshed["profiles"].get(str(request.get("batter_id")), refreshed["default_profile"])
        flags = list(profile.get("flags", [])) + list(info.get("flags", []))
        if np.mean(profile["reliabilities"]) < LOW_RELIABILITY:
            flags.append("low_batter_profile_reliability")
        if row["profile_source"] == "default_profile_zero_reliability":
            flags.append("missing_batter_zero_reliability")
        source_max = sidecar["source_date_max"]
        if source_max is None or (requested - _iso_date(source_max, "source_date_max")).days > LOOKBACK_DAYS:
            flags.append("stale_historical_snapshot")
        if "batter_profile" in request:
            flags.append("user_supplied_profile_provenance_not_independently_verified")
        predict_original = engine.predict_counts
        had_predict_override = "predict_counts" in engine.__dict__
        predict_override = engine.__dict__.get("predict_counts")
        if bias is not None:
            def calibrated_predict_counts(value):
                predicted = predict_original(value)
                probabilities = predicted["probabilities"]["blend"]
                adjusted = probabilities * np.exp(bias - bias.max())
                total = adjusted.sum(axis=-1, keepdims=True)
                if not np.isfinite(adjusted).all() or (total <= 0).any():
                    raise ValueError("Calibration produced invalid outcome probabilities")
                predicted["probabilities"]["blend"] = adjusted / total
                return predicted
            engine.predict_counts = calibrated_predict_counts
        try:
            engine.metadata = refreshed
            result = engine.recommend(request)
        finally:
            engine.metadata = original
            if bias is not None:
                if had_predict_override:
                    engine.predict_counts = predict_override
                else:
                    del engine.predict_counts
        result["baseline_policy"] = "prior_90_day_supported_repertoire_frequency"
        result["assumptions"] = [value for value in result.get("assumptions", [])
                                 if not value.startswith("Full-PA defensive WE")]
        result["assumptions"].append("Full-PA defensive WE uses frozen continuation/advancement; "
                                     "baseline follows the dated 90-day supported repertoire counts.")
        result["refresh"] = {"effective_date": sidecar["effective_date"], "source_date_max": source_max,
                             "model_calibration_cutoff": sidecar["model_calibration_cutoff"],
                             "flags": sorted(set(flags)), "scope": sidecar["scope"],
                             "snapshot_sha256": hashlib.sha256(_json_bytes(sidecar)).hexdigest()}
        result["outcome_calibration"] = {"applied": bias is not None,
            "available_from": calibration.get("available_from") if bias is not None else None,
            "kind": "accepted_class_intercept_repair" if bias is not None else "frozen_original_blend"}
        return result


def load_approved_profile_frame(config):
    """Read the pinned complete processed pool; verify approved raw hashes read-only.

    Unlike sequence prepare_frame, refresh needs no physics cache and writes none.
    """
    if config["raw_allowlist"] != RAW_ALLOWLIST:
        raise ValueError("Only the three exact approved raw files may be opened")
    root = Path(config["artifact_root"])
    if not VOLUME.is_mount() or root.resolve() != RUNS.parent.resolve():
        raise ValueError("Mounted approved artifact root is required")
    quality_path = root / "reports/data_quality.json"
    quality = json.loads(quality_path.read_text())
    sources = quality["sources"]
    if [item["file"] for item in sources] != RAW_ALLOWLIST:
        raise ValueError("Data-quality manifest does not pin the approved source list")
    for item in sources:
        source = Path(config["raw_root"]) / item["file"]
        if source.stat().st_size != item["bytes"] or hash_file(source) != item["sha256"]:
            raise ValueError(f"Raw source hash mismatch: {item['file']}")
    processed = root / "processed/pitches.parquet"
    if hash_file(processed) != quality["processed_sha256"]:
        raise ValueError("Processed source hash mismatch")
    frame = pd.read_parquet(processed, columns=[*KEY, "game_date", "batter", "pitcher", "pitch_type",
        "p_throws", "description", "events", "is_pa_terminal", "launch_angle"])
    if len(frame) != quality["rows"] or frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Processed source row/key mismatch")
    identity = {"sources": sources, "processed_sha256": quality["processed_sha256"],
                "quality_manifest_sha256": hash_file(quality_path), "rows": len(frame),
                "key": KEY, "read_scope": "complete approved pool before any support/outcome filter"}
    return frame, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    refresh = sub.add_parser("refresh", help="Create a new immutable dated sidecar")
    refresh.add_argument("--as-of", required=True, help="Exclusive source cutoff / first valid request date")
    refresh.add_argument("--output", required=True, type=Path)
    recommend = sub.add_parser("recommend", help="Call frozen Engine with a sidecar")
    recommend.add_argument("--snapshot", required=True, type=Path)
    recommend.add_argument("--request", required=True, type=Path)
    recommend.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    recommend.add_argument("--calibration", type=Path, help="Optional accepted dated ten-intercept repair JSON")
    recommend.add_argument("--output", type=Path, help="Optional new run directory for request/response JSON")
    for command in (refresh, recommend):
        command.add_argument("--bundle", type=Path, default=RUNS / "minimal-pitch-service-v1")
        command.add_argument("--config", type=Path, default=PROJECT / "configs/local.json")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        # Prefix keeps the .venv identity even when python is a symlink to its base interpreter.
        if Path(sys.prefix).resolve() != Path(config["python"]).parent.parent.resolve():
            raise ValueError("Use the configured project .venv/bin/python")
        if args.command == "refresh":
            metadata_path = args.bundle / "metadata.json"
            manifest = json.loads((args.bundle / "bundle_manifest.json").read_text())
            if hash_file(metadata_path) != manifest["sha256"].get("metadata.json"):
                raise ValueError("Original bundle metadata hash mismatch")
            metadata = json.loads(metadata_path.read_text())
            if len(metadata["pitchers"]) != 6:
                raise ValueError("CLI requires the original six-pitcher service bundle")
            # Capture bytes before the potentially long source read, so copied
            # provenance and snapshot-bound hashes refer to identical content.
            source_payloads = {f"source/{rel}": (PROJECT / rel).read_bytes() for rel in
                               ("scripts/refresh_pitch_service.py", "pitchmdp/archetypes.py")}
            source_hashes = {rel: hashlib.sha256(payload).hexdigest()
                             for rel, payload in source_payloads.items()}
            frame, identity = load_approved_profile_frame(config)
            identity["refresh_code_sha256"] = source_hashes
            snapshot = build_sidecar(frame, args.as_of, metadata, identity)
            output = save_sidecar(snapshot, args.output)
            for rel, payload in source_payloads.items():
                destination = output / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(payload)
            with (output / "source_hashes.json").open("xb") as stream:
                stream.write(_json_bytes({"schema_version": 1, "sha256": source_hashes}))
            print(json.dumps({"snapshot": str(output), "effective_date": args.as_of,
                              "source_date_max": snapshot["source_date_max"],
                              "profile_count": len(snapshot["profiles"]),
                              "available_pitchers": [pid for pid, info in snapshot["pitchers"].items() if info["available"]]}))
        else:
            engine = Engine(args.bundle, args.device)
            snapshot = load_sidecar(args.snapshot, engine.metadata)
            request = json.loads(args.request.read_text())
            calibration = json.loads(args.calibration.read_text()) if args.calibration else None
            response = recommend_with_sidecar(engine, request, snapshot, calibration)
            if args.output is not None:
                output = new_run_directory(args.output)
                for filename, value in (("request.json", request), ("response.json", response)):
                    with (output / filename).open("xb") as stream:
                        stream.write(_json_bytes(value))
            print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"error": getattr(exc, "code", "invalid_refresh"), "message": str(exc)}))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
