"""Registered June 2025 eligibility preparation (ML-JUNE-ELIGIBILITY-v1).

Single worker. Reads the pinned processed pitches with an explicit column projection and
date/game_type pushdown, applies the frozen EXP-P4-001 ``eligible`` rule extracted from the
hash-pinned archived source (pure functions only, no torch import), replays the frozen Cpanel
blend keys exactly, and writes eligibility keys, metadata, exclusion reason flags, support
counts, result.json and finally manifest.json. No model, inference, calibration fit, quality
metric, class label or class frequency is produced or persisted.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_REL = "scripts/prepare_june_eligibility.py"
KEY = ["game_pk", "at_bat_number", "pitch_number"]
# Contract §2 upper bound on decoded columns; the config may not widen it.
CONTRACT_METADATA_COLUMNS = ("game_pk", "at_bat_number", "pitch_number", "game_date", "game_type", "split",
                             "pitcher", "batter", "starter_pitcher", "p_throws")
CONTRACT_ELIGIBILITY_COLUMNS = ("description", "events", "balls", "strikes", "supported_pa", "pitch_type",
                                "plate_x", "plate_z")
REASONS = ("unmapped_outcome", "unsupported_pa", "missing_type", "missing_plate_x", "missing_plate_z",
           "invalid_balls", "invalid_strikes")
EXTRACTED_NAMES = ("OUTCOMES", "outcome_labels", "eligible")
METADATA_OUT = ["game_date", "research_split", "pitcher", "batter", "in_cpanel", "train_volume", "game_role",
                "throwing_hand", "month"]
BLEND_META_COMPARE = ["pitcher", "batter", "in_cpanel", "train_volume", "game_role", "throwing_hand", "month"]
PERSIST_FORBIDDEN = set(CONTRACT_ELIGIBILITY_COLUMNS) | {"label", "labels", "y", "outcome", "pitch_outcome"}


class PreparationError(RuntimeError):
    """Fail-closed boundary or gate violation. Messages carry names and counts, never row values."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def ordered_key_hash(frame: pd.DataFrame) -> str:
    """Same definition as archived matrix_data.ordered_key_hash: int64 C-contiguous KEY bytes."""
    require(not frame[KEY].isna().any().any() and not frame.duplicated(KEY).any(), "keys missing or duplicated")
    return hashlib.sha256(np.ascontiguousarray(frame[KEY].to_numpy(np.int64)).tobytes()).hexdigest()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- config

def validate_window(window: dict) -> tuple[date, date]:
    lo, hi = date.fromisoformat(window["date_min"]), date.fromisoformat(window["date_max"])
    reject, season = int(window["reject_year_min"]), int(window["season"])
    require(lo <= hi, "window date_min after date_max")
    require(max(lo.year, hi.year, season) < reject, f"window reaches {reject}+ (OPE-only data)")
    require(lo.year == season == hi.year, "window must stay inside the registered season")
    require(list(window["game_types"]) == ["R"], "requested population is regular season only")
    return lo, hi


def validate_config(cfg: dict) -> dict:
    lo, hi = validate_window(cfg["window"])
    fold = cfg["research_fold"]
    require((date.fromisoformat(fold["date_min"]), date.fromisoformat(fold["date_max"])) == (lo, hi),
            "research fold dates differ from the requested window")
    require(list(cfg["key"]) == KEY and list(cfg["sort"]) == ["game_date", *KEY], "key/sort order differs from P4")
    cols = cfg["columns"]
    require(list(cols["metadata"]) == list(CONTRACT_METADATA_COLUMNS), "metadata columns differ from contract")
    require(list(cols["eligibility"]) == list(CONTRACT_ELIGIBILITY_COLUMNS), "eligibility columns differ from contract")
    require(list(cfg["eligibility_rule"]["reasons"]) == list(REASONS), "reason flag order differs")
    require(list(cfg["eligibility_rule"]["extract"]) == list(EXTRACTED_NAMES), "extracted names differ")
    rule = cfg["support_rule"]
    require(rule.get("combine") == "AND" and int(rule["min_games"]) > 0 and int(rule["min_pitches"]) > 0,
            "support rule must be positive games AND pitches")
    policy = cfg["unknown_policy"]
    require(policy.get("missing_pitcher") == "fail", "missing pitcher identity must fail")
    for name in ("missing_starter_pitcher", "hand_outside_levels"):
        require(policy.get(name) in ("fail", "unknown"), f"unknown_policy.{name} must be fail or unknown")
    g = cfg["groups"]
    require(list(g["volume"]["levels"]) == ["zero", "low", "middle", "high"], "volume levels differ")
    require(list(g["game_role"]["levels"]) == ["starter", "relief"], "role levels differ")
    require(list(g["hand"]["levels"]) == ["L", "R"], "hand levels differ")
    require(cfg["outputs"].get("allowed_attempt_root"), "outputs.allowed_attempt_root is not registered")
    names = list(cfg["outputs"]["files"].values())
    require(len(set(names)) == len(names) and all("/" not in n for n in names), "output file names invalid")
    return {"lo": lo, "hi": hi}


def claim_output(output: Path, cfg: dict) -> Path:
    """Refuse any existing path; claim a fresh directory strictly inside the registered root."""
    root = resolve(cfg["outputs"]["allowed_attempt_root"]).resolve()
    target = output.resolve()
    require(target != root and target.is_relative_to(root), "output outside registered attempt root")
    require(not output.exists() and not output.is_symlink(), "refusing to reuse an existing output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    return output


@contextmanager
def heavy_lock(cfg: dict):
    """Non-blocking exclusive hold on the existing shared lock; never creates it."""
    path = resolve(cfg["heavy_lock"])
    env = os.environ.get("PITCHEEZY_HEAVY_LOCK")
    require(env is None or Path(env).resolve() == path.resolve(), "PITCHEEZY_HEAVY_LOCK differs from config")
    require(path.is_file(), "shared heavy lock file does not exist")
    with open(path, "rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PreparationError("another heavy job holds the shared lock") from exc
        try:
            yield str(path)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------- provenance

def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def implementation_provenance(cfg: dict, config_path: Path) -> dict:
    """Registered code commit C must be an ancestor of HEAD with identical script bytes; files clean."""
    reg = cfg["registration"]
    require(reg.get("code_commit") and reg.get("script_sha256"), "config is an unregistered draft")
    script_sha = sha256_file(ROOT / SCRIPT_REL)
    require(script_sha == reg["script_sha256"], "executed script differs from registered SHA256")
    require(_git("merge-base", "--is-ancestor", reg["code_commit"], "HEAD").returncode == 0,
            "registered code commit is not an ancestor of HEAD")
    require(_git("diff", "--quiet", reg["code_commit"], "HEAD", "--", SCRIPT_REL).returncode == 0,
            "script changed since registered code commit")
    cfg_resolved = config_path.resolve()
    require(cfg_resolved.is_relative_to(ROOT.resolve()), "config must live in the execution checkout")
    cfg_rel = str(cfg_resolved.relative_to(ROOT.resolve()))
    require(_git("ls-files", "--error-unmatch", cfg_rel).returncode == 0, "config is not tracked")
    dirty = _git("status", "--porcelain", "--", SCRIPT_REL, cfg_rel).stdout.strip()
    require(not dirty, "script/config have uncommitted changes")
    return {"registered_code_commit": reg["code_commit"],
            "execution_head": _git("rev-parse", "HEAD").stdout.strip(),
            "executed_script_sha256": script_sha, "config_path": cfg_rel}


def verify_pins(cfg: dict) -> dict:
    """Hash every pinned source and read-only guard. Any mismatch stops before any parquet decode."""
    out = {}
    for group in ("sources", "read_only_guards"):
        for name, spec in cfg[group].items():
            path = resolve(spec["path"])
            require(spec.get("sha256"), f"{group}.{name} has no pinned sha256")
            require(path.is_file(), f"{group}.{name} missing")
            actual = sha256_file(path)
            require(actual == spec["sha256"], f"hash mismatch for {group}.{name}")
            out[f"{group}.{name}"] = actual
    model = cfg["sources"]["p4_model_source"]
    data = resolve(model["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == model["sha256"], "model source changed after hashing")
    require(git_blob_sha1(data) == model["git_blob_sha1"], "archived model source differs from approved git blob")
    approved = _git("rev-parse", f"HEAD:{model['approved_repo_path']}").stdout.strip()
    require(approved == model["git_blob_sha1"], "approved repo model source blob differs from archived source")
    out["p4_model_source.git_blob_sha1"] = approved
    return out


def extract_eligibility(source: bytes) -> dict:
    """Execute only OUTCOMES/outcome_labels/eligible from the archived module, with numpy/pandas only."""
    tree = ast.parse(source)
    picked = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "OUTCOMES" for t in node.targets):
            picked.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in EXTRACTED_NAMES[1:]:
            picked.append(node)
    found = [n.name if isinstance(n, ast.FunctionDef) else "OUTCOMES" for n in picked]
    require(sorted(found) == sorted(EXTRACTED_NAMES), "archived source lacks exactly one of each pure function")
    for node in picked:
        for sub in ast.walk(node):
            require(not isinstance(sub, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)),
                    "extracted functions must be import-free")
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                require(sub.id in {"np", "pd", "OUTCOMES", "outcome_labels", "frame", "d", "e", "labels", "name",
                                   "range", "len"},
                        f"extracted function references unexpected name {sub.id}")
    module = ast.Module(body=picked, type_ignores=[])
    namespace = {"np": np, "pd": pd, "__builtins__": {"range": range, "len": len}}
    exec(compile(module, "<p4-model-eligibility>", "exec"), namespace)
    return {name: namespace[name] for name in EXTRACTED_NAMES}


# ---------------------------------------------------------------- data

def read_projection(path: Path, cfg: dict, lo: date, hi: date) -> pd.DataFrame:
    columns = [*cfg["columns"]["metadata"], *cfg["columns"]["eligibility"]]
    schema = pq.read_schema(path).names
    missing = [c for c in columns if c not in schema]
    require(not missing, f"processed parquet lacks registered columns {missing}")
    filters = [("game_date", ">=", pd.Timestamp(lo)), ("game_date", "<=", pd.Timestamp(hi)),
               ("game_type", "in", list(cfg["window"]["game_types"]))]
    table = pq.read_table(path, columns=columns, filters=filters)
    require(table.column_names == columns, "decoded projection differs from registered columns")
    return table.to_pandas()


def check_window_rows(frame: pd.DataFrame, cfg: dict, lo: date, hi: date) -> None:
    stamps = pd.to_datetime(frame["game_date"])
    require(not stamps.isna().any(), "decoded rows have missing dates")
    require(not (stamps.dt.year >= int(cfg["window"]["reject_year_min"])).any(), "decoded rows reach rejected year")
    days = stamps.dt.date
    require(bool(((days >= lo) & (days <= hi)).all()), "decoded rows outside the registered window")
    require(bool(frame["game_type"].isin(cfg["window"]["game_types"]).all()), "decoded rows outside game_type R")
    split = cfg["source_split"]
    require(bool(frame[split["column"]].isin(split["expected"]).all()),
            "source split outside the registered legacy split")


def check_identity(frame: pd.DataFrame, policy: dict) -> dict:
    require(not frame[KEY].isna().any().any(), "requested keys incomplete")
    require(not frame.duplicated(KEY).any(), "requested keys duplicated")
    require(not frame["pitcher"].isna().any(), "requested pitcher identity missing")
    require(not frame["batter"].isna().any(), "requested batter identity missing")
    require(int(frame.groupby("game_pk")["game_date"].nunique().max()) == 1, "a game spans more than one date")
    unknown = {"missing_starter_pitcher": int(frame["starter_pitcher"].isna().sum()),
               "hand_outside_levels": int((~frame["p_throws"].isin(["L", "R"])).sum())}
    for name, count in unknown.items():
        require(not (count and policy[name] == "fail"), f"{name}: {count} rows and policy is fail")
    return unknown


def add_metadata(frame: pd.DataFrame, cfg: dict, volume_of: dict, panel_ids: set) -> pd.DataFrame:
    frame = frame.sort_values(cfg["sort"], kind="mergesort").reset_index(drop=True)
    for col in ("game_pk", "at_bat_number", "pitch_number", "pitcher", "batter"):
        frame[col] = frame[col].astype("int64")
    frame["research_split"] = cfg["research_fold"]["name"]
    frame["in_cpanel"] = frame["pitcher"].isin(panel_ids)
    frame["train_volume"] = frame["pitcher"].map(volume_of).fillna("zero").astype(str)
    known = frame["starter_pitcher"].notna()
    frame["game_role"] = np.where(known, np.where(frame["pitcher"].eq(frame["starter_pitcher"]), "starter", "relief"),
                                  "unknown")
    frame["throwing_hand"] = frame["p_throws"].where(frame["p_throws"].isin(["L", "R"]), "unknown").astype(str)
    frame["month"] = pd.to_datetime(frame["game_date"]).dt.to_period("M").astype(str)
    return frame


def eligibility_flags(frame: pd.DataFrame, fns: dict) -> pd.DataFrame:
    """Frozen conjuncts as seven reason flags; labels are transient and never returned."""
    mapped = fns["outcome_labels"](frame) >= 0
    flags = pd.DataFrame({
        "unmapped_outcome": ~mapped,
        "unsupported_pa": ~np.asarray(frame.supported_pa.fillna(False).to_numpy(), dtype=bool),
        "missing_type": ~frame.pitch_type.notna().to_numpy(),
        "missing_plate_x": ~frame.plate_x.notna().to_numpy(),
        "missing_plate_z": ~frame.plate_z.notna().to_numpy(),
        "invalid_balls": ~frame.balls.between(0, 3).to_numpy(),
        "invalid_strikes": ~frame.strikes.between(0, 2).to_numpy(),
    }, index=frame.index)
    del mapped
    authoritative = np.asarray(fns["eligible"](frame), dtype=bool)
    union = flags[list(REASONS)].any(axis=1).to_numpy()
    require(bool((authoritative == ~union).all()), "reason union differs from authoritative eligible()")
    flags.insert(0, "eligible", authoritative)
    return flags


def reason_summary(frame: pd.DataFrame) -> dict:
    excluded = frame.loc[~frame["eligible"]]
    bits = np.zeros(len(excluded), dtype=np.int64)
    for i, name in enumerate(REASONS):
        bits |= excluded[name].to_numpy(np.int64) << i
    patterns = pd.Series(bits).value_counts().sort_index()
    disjoint = {format(int(k), "07b")[::-1]: int(v) for k, v in patterns.items()}
    require(sum(disjoint.values()) == len(excluded) and 0 not in patterns.index,
            "disjoint bitmask counts do not partition the excluded rows")
    coords = excluded["missing_plate_x"] | excluded["missing_plate_z"]
    return {
        "overlapping": {name: {"pitches": int(excluded[name].sum()),
                               "games": int(excluded.loc[excluded[name], "game_pk"].nunique())} for name in REASONS},
        "missing_coordinates": {"pitches": int(coords.sum()), "games": int(excluded.loc[coords, "game_pk"].nunique())},
        "disjoint_bitmask": {"bit_order": list(REASONS),
                             "encoding": "string position i = reason i (1 = set); patterns partition excluded rows",
                             "counts": disjoint, "sum": int(sum(disjoint.values()))},
        "note": "overlapping reason counts are not unique exclusion counts",
    }


def verify_frozen_volume(panel: dict, cfg: dict) -> tuple[dict, dict]:
    thresholds = panel["volume_thresholds"]
    q25, q75 = float(thresholds["q25"]), float(thresholds["q75"])
    exp = cfg["expected"]["frozen_volume"]
    require((q25, q75) == (float(exp["q25"]), float(exp["q75"])), "frozen volume cutoffs differ from registration")
    volume_of = {}
    for player in panel["train_players"]:
        n = player["train_pitches"]
        rederived = "low" if n <= q25 else ("middle" if n <= q75 else "high")
        require(player["train_volume"] == rederived, "stored TRAIN-volume label disagrees with frozen cutoffs")
        volume_of[int(player["pitcher"])] = player["train_volume"]
    return volume_of, {"q25": q25, "q75": q75, "players": len(volume_of), "mismatches": 0}


def replay_cpanel(frame: pd.DataFrame, blend_keys: pd.DataFrame, blend_meta: pd.DataFrame, p4_rows_sha: str) -> dict:
    require(blend_keys[KEY].reset_index(drop=True).equals(blend_meta[KEY].reset_index(drop=True)),
            "blend_keys and blend_metadata key order differ")
    derived = frame.loc[frame["in_cpanel"] & frame["eligible"]].reset_index(drop=True)
    frozen = blend_keys[KEY].astype("int64").reset_index(drop=True)
    require(len(derived) == len(frozen), f"derived Cpanel eligible count {len(derived)} != frozen {len(frozen)}")
    require(bool((derived[KEY].to_numpy() == frozen.to_numpy()).all()), "derived Cpanel ordered keys differ from P4")
    require(ordered_key_hash(frozen) == p4_rows_sha, "frozen blend keys differ from P4 samples.blend.rows_sha256")
    meta = blend_meta.reset_index(drop=True)
    require(not meta[BLEND_META_COMPARE].isna().any().any(), "blend_metadata has missing values")
    as_cmp = {"pitcher": "int64", "batter": "int64", "in_cpanel": bool}
    mismatched = [c for c in BLEND_META_COMPARE
                  if not (derived[c].astype(as_cmp.get(c, str)).to_numpy()
                          == meta[c].astype(as_cmp.get(c, str)).to_numpy()).all()]
    require(not mismatched, f"Cpanel metadata differs from blend_metadata in {mismatched}")
    return {"ordered_keys_equal": True, "rows": len(derived), "ordered_key_sha256": p4_rows_sha,
            "metadata_columns_equal": BLEND_META_COMPARE}


def census(part: pd.DataFrame) -> dict:
    return {"pitches": int(len(part)), "games": int(part["game_pk"].nunique()),
            "pitchers": int(part["pitcher"].nunique())}


def support_table(part: pd.DataFrame, column: str, levels: list, rule: dict, judge: bool) -> dict:
    out = {}
    for level in levels:
        sub = part.loc[part[column] == level]
        row = {"pitches": int(len(sub)), "games": int(sub["game_pk"].nunique()),
               "pitchers": int(sub["pitcher"].nunique())}
        if judge:
            row["supported"] = bool(row["games"] >= int(rule["min_games"]) and row["pitches"] >= int(rule["min_pitches"]))
        out[str(level)] = row
    return out


def group_levels(cfg: dict) -> dict:
    g = cfg["groups"]
    vol, role, hand = g["volume"]["levels"], g["game_role"]["levels"], g["hand"]["levels"]
    return {"train_volume": vol, "game_role": role, "throwing_hand": hand,
            "role_x_volume": [f"{r}|{v}" for r in role for v in vol],
            "hand_x_volume": [f"{h}|{v}" for h in hand for v in vol]}


def populations(frame: pd.DataFrame) -> dict:
    elig, panel, frozen = frame["eligible"], frame["in_cpanel"], frame["frozen_cpanel_eligible"]
    return {
        "requested": frame, "eligible": frame.loc[elig], "excluded": frame.loc[~elig],
        "panel_requested": frame.loc[panel], "panel_eligible": frame.loc[panel & elig],
        "panel_excluded": frame.loc[panel & ~elig],
        "outside_requested": frame.loc[~panel], "outside_eligible": frame.loc[~panel & elig],
        "outside_excluded": frame.loc[~panel & ~elig],
        "requested_minus_frozen_cpanel": frame.loc[~frozen],
        "eligible_minus_frozen_cpanel": frame.loc[elig & ~frozen],
    }


def partition_gates(frame: pd.DataFrame, pops: dict, blend_keys: pd.DataFrame) -> dict:
    keys = {name: set(map(tuple, p[KEY].to_numpy().tolist())) for name, p in pops.items()}
    frozen = set(map(tuple, blend_keys[KEY].astype("int64").to_numpy().tolist()))
    checks = {
        "eligible ⊔ excluded = requested": (keys["eligible"] | keys["excluded"] == keys["requested"]
                                            and not keys["eligible"] & keys["excluded"]),
        "panel ⊔ outside = requested": (keys["panel_requested"] | keys["outside_requested"] == keys["requested"]
                                        and not keys["panel_requested"] & keys["outside_requested"]),
        "panel_eligible = frozen blend keys": keys["panel_eligible"] == frozen,
        "panel_excluded = panel_requested − frozen": keys["panel_excluded"] == keys["panel_requested"] - frozen,
        "requested_minus_frozen = requested − frozen": keys["requested_minus_frozen_cpanel"] == keys["requested"] - frozen,
        "eligible_minus_frozen = outside_eligible": keys["eligible_minus_frozen_cpanel"] == keys["outside_eligible"],
        "outside_eligible ⊔ outside_excluded = outside": (
            keys["outside_eligible"] | keys["outside_excluded"] == keys["outside_requested"]
            and not keys["outside_eligible"] & keys["outside_excluded"]),
        "frozen ⊂ requested": frozen <= keys["requested"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    require(not failed, f"key partition gates failed: {failed}")
    return {name: True for name in checks}


def expected_gates(counts: dict, reasons_panel: dict, cfg: dict, p4_cov: dict) -> dict:
    exp = cfg["expected"]
    items = []
    for pop, fields in exp["populations"].items():
        for field, value in fields.items():
            items.append((f"{pop}.{field}", value, counts[pop][field]))
    p4_map = {"requested_pitches": counts["panel_requested"]["pitches"],
              "eligible_pitches": counts["panel_eligible"]["pitches"],
              "unsupported_pa": reasons_panel["overlapping"]["unsupported_pa"]["pitches"],
              "unmapped_outcome": reasons_panel["overlapping"]["unmapped_outcome"]["pitches"],
              "missing_type": reasons_panel["overlapping"]["missing_type"]["pitches"],
              "missing_coordinates": reasons_panel["missing_coordinates"]["pitches"]}
    for field, actual in p4_map.items():
        items.append((f"p4 coverage.blend.{field}", p4_cov[field], actual))
    out = {}
    for name, expected, actual in items:
        require(expected == actual, f"expected count mismatch for {name}: {expected} != {actual}")
        out[name] = {"expected": expected, "actual": actual, "status": "match"}
    n = counts["eligible"]["pitches"]
    lo, hi = exp["eligible_bounds"]["min"], exp["eligible_bounds"]["max"]
    require(lo <= n <= hi, f"eligible count outside registered bounds [{lo}, {hi}]")
    out["eligible within bounds"] = {"bounds": [lo, hi], "status": "match"}
    return out


# ---------------------------------------------------------------- write

def write_parquet(frame: pd.DataFrame, path: Path) -> dict:
    bad = PERSIST_FORBIDDEN & set(frame.columns)
    require(not bad, f"refusing to persist outcome/coordinate columns {sorted(bad)}")
    with open(path, "xb") as handle:
        frame.to_parquet(handle, index=False)
    return {"sha256": sha256_file(path), "rows": int(len(frame)), "columns": list(frame.columns),
            "ordered_key_sha256": ordered_key_hash(frame),
            "games": int(frame["game_pk"].nunique())}


def write_json(value: dict, path: Path) -> str:
    with open(path, "x") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    return sha256_file(path)


def runtime_versions() -> dict:
    return {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__}


def prepare(cfg: dict, config_path: Path, output: Path, timings: dict, stage: list) -> dict:
    t0 = time.perf_counter()
    files = cfg["outputs"]["files"]
    src = cfg["sources"]
    bounds = validate_config(cfg)
    stage[0] = "lock"
    with heavy_lock(cfg) as lock_path:
        timings["lock_acquired_s"] = time.perf_counter() - t0
        stage[0] = "provenance"
        provenance = implementation_provenance(cfg, config_path)
        config_sha = sha256_file(config_path)
        stage[0] = "verify_pins"
        t = time.perf_counter()
        pins = verify_pins(cfg)
        timings["verify_pins_s"] = time.perf_counter() - t

        stage[0] = "parents"
        t = time.perf_counter()
        fns = extract_eligibility(resolve(src["p4_model_source"]["path"]).read_bytes())
        panel = json.loads(resolve(src["panel"]["path"]).read_text())
        volume_of, frozen_volume = verify_frozen_volume(panel, cfg)
        panel_ids = {int(p) for p in panel["pitcher_ids"]}
        p4 = json.loads(resolve(src["p4_preparation"]["path"]).read_text())
        blend_keys = pd.read_parquet(resolve(src["blend_keys"]["path"]), columns=KEY)
        blend_meta = pd.read_parquet(resolve(src["blend_metadata"]["path"]), columns=[*KEY, *BLEND_META_COMPARE])
        timings["parents_s"] = time.perf_counter() - t

        stage[0] = "read_projection"
        t = time.perf_counter()
        read_utc = datetime.now(timezone.utc).isoformat()
        raw = read_projection(resolve(src["processed_pitches"]["path"]), cfg, bounds["lo"], bounds["hi"])
        timings["read_projection_s"] = time.perf_counter() - t

        stage[0] = "eligibility"
        t = time.perf_counter()
        check_window_rows(raw, cfg, bounds["lo"], bounds["hi"])
        unknown = check_identity(raw, cfg["unknown_policy"])
        frame = add_metadata(raw, cfg, volume_of, panel_ids)
        del raw
        flags = eligibility_flags(frame, fns)
        frame = pd.concat([frame.drop(columns=list(CONTRACT_ELIGIBILITY_COLUMNS)), flags], axis=1)
        frame["source_split"] = frame.pop(cfg["source_split"]["column"])
        frozen_set = pd.MultiIndex.from_frame(blend_keys[KEY].astype("int64"))
        frame["frozen_cpanel_eligible"] = pd.MultiIndex.from_frame(frame[KEY]).isin(frozen_set)
        timings["eligibility_s"] = time.perf_counter() - t

        stage[0] = "gates"
        t = time.perf_counter()
        cpanel = replay_cpanel(frame, blend_keys, blend_meta, p4["samples"]["blend"]["rows_sha256"])
        pops = populations(frame)
        partitions = partition_gates(frame, pops, blend_keys)
        counts = {name: census(p) for name, p in pops.items()}
        reasons = {"requested": reason_summary(frame), "panel_requested": reason_summary(pops["panel_requested"]),
                   "outside_requested": reason_summary(pops["outside_requested"])}
        expected = expected_gates(counts, reasons["panel_requested"], cfg, p4["coverage"]["blend"])
        timings["gates_s"] = time.perf_counter() - t

        stage[0] = "aggregate"
        t = time.perf_counter()
        frame["role_x_volume"] = frame["game_role"] + "|" + frame["train_volume"]
        frame["hand_x_volume"] = frame["throwing_hand"] + "|" + frame["train_volume"]
        pops = populations(frame)
        levels, rule = group_levels(cfg), cfg["support_rule"]
        judged = set(cfg["support_judged_populations"])
        support = {group: {name: support_table(p, group, lv, rule, name in judged) for name, p in pops.items()}
                   for group, lv in levels.items()}
        unknown_diag = {name: census(frame.loc[frame[col] == "unknown"])
                        for name, col in (("game_role_unknown", "game_role"), ("hand_unknown", "throwing_hand"))}
        games = {name: set(p["game_pk"]) for name, p in pops.items()}
        shared = {f"{a} & {b}": len(games[a] & games[b]) for a, b in cfg["shared_game_pairs"]}
        timings["aggregate_s"] = time.perf_counter() - t

        stage[0] = "write"
        t = time.perf_counter()
        meta_cols = [*KEY, *METADATA_OUT]
        inventory_cols = [*meta_cols, "source_split", "frozen_cpanel_eligible", "eligible", *REASONS]
        eligible = frame.loc[frame["eligible"]]
        complement = frame.loc[frame["eligible"] & ~frame["frozen_cpanel_eligible"]]
        artifacts = {
            files["inventory"]: write_parquet(frame[inventory_cols], output / files["inventory"]),
            files["eligible_keys"]: write_parquet(eligible[KEY], output / files["eligible_keys"]),
            files["eligible_metadata"]: write_parquet(eligible[[*meta_cols, "frozen_cpanel_eligible"]],
                                                      output / files["eligible_metadata"]),
            files["complement_eligible_keys"]: write_parquet(complement[KEY], output / files["complement_eligible_keys"]),
            files["complement_eligible_metadata"]: write_parquet(complement[meta_cols],
                                                                 output / files["complement_eligible_metadata"]),
            files["excluded"]: write_parquet(frame.loc[~frame["eligible"], [*KEY, "in_cpanel", *REASONS]],
                                             output / files["excluded"]),
        }
        timings["write_parquet_s"] = time.perf_counter() - t

        stage[0] = "reverify"
        after = verify_pins(cfg)
        require(after == pins, "a pinned parent/source/guard changed during preparation")
        require(sha256_file(config_path) == config_sha, "config changed during preparation")
        require(sha256_file(ROOT / SCRIPT_REL) == provenance["executed_script_sha256"], "script changed during run")
        timings["worker_internal_total_before_result_s"] = time.perf_counter() - t0

        result = {
            "prep_id": cfg["prep_id"], "status": "prepared",
            "config": {"path_as_passed": str(config_path), "resolved": str(config_path.resolve()), "sha256": config_sha},
            "implementation": provenance, "runtime": runtime_versions(), "heavy_lock": lock_path,
            "pins_verified_before_read": pins, "pins_reverified_after_write": True,
            "window": cfg["window"], "research_fold": cfg["research_fold"], "source_split": cfg["source_split"],
            "exposure": {
                "record": "June 2025 regular-season eligibility fields newly decoded for preparation",
                "fields_newly_read": list(CONTRACT_ELIGIBILITY_COLUMNS),
                "metadata_fields_read": list(CONTRACT_METADATA_COLUMNS),
                "file": src["processed_pitches"]["path"], "file_sha256": src["processed_pitches"]["sha256"],
                "rows_window": [str(bounds["lo"]), str(bounds["hi"])], "game_types": cfg["window"]["game_types"],
                "decoded_utc": read_utc, "purpose": "eligibility mask and exclusion reason flags only",
                "persisted_outcome_values": False, "class_counts_reported": False,
                "freshness_claim": None,
            },
            "frozen_volume": frozen_volume, "unknown_metadata": {"rows": unknown, "diagnostic_census": unknown_diag,
                                                                 "policy": cfg["unknown_policy"]},
            "counts": counts, "cpanel_replay": cpanel, "partition_gates": partitions, "expected_gates": expected,
            "exclusion_reasons": reasons, "shared_games": shared,
            "support_rule": rule, "support_judged_populations": sorted(judged), "support": support,
            "artifacts": artifacts,
            "timings_internal_nonadditive": timings,
            "budget": {"external_worker_cap_seconds": cfg["budget"]["external_worker_seconds"],
                       "enforced_by": cfg["budget"]["enforced_by"],
                       "note": "internal timings exclude interpreter start/imports and are cross-checks only"},
            "g0_weights_changed": False, "new_fit_count": 0, "new_inference_count": 0,
            "independent_confirmation": None, "policy_effect": None, "service_adoption": None,
        }
        stage[0] = "result"
        result_sha = write_json(result, output / files["result"])
        stage[0] = "manifest"
        manifest = {
            "prep_id": cfg["prep_id"], "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha, "executed_script_sha256": provenance["executed_script_sha256"],
            "registered_code_commit": provenance["registered_code_commit"],
            "execution_head": provenance["execution_head"],
            "parent_hashes": pins, "artifacts": {**artifacts, files["result"]: {"sha256": result_sha}},
        }
        write_json(manifest, output / files["manifest"])
    return result


def run(config_path: Path, output: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    output = claim_output(output, cfg)
    timings: dict = {}
    stage = ["start"]
    try:
        return prepare(cfg, config_path, output, timings, stage)
    except BaseException as exc:
        files = cfg["outputs"]["files"]
        failure = {"prep_id": cfg.get("prep_id"), "status": "failed", "stage": stage[0],
                   "error_type": type(exc).__name__, "message": str(exc)[:2000],
                   "failed_utc": datetime.now(timezone.utc).isoformat(),
                   "timings_internal_nonadditive": timings, "partial_outputs_preserved": True}
        write_json(failure, output / files["failure"])
        with open(output / files["failure_log"], "x") as handle:
            handle.write("".join(traceback.format_exception_only(type(exc), exc)))
            handle.write("".join(traceback.format_tb(exc.__traceback__)))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
