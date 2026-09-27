"""Synthetic boundary tests for scripts/prepare_june_eligibility.py (no real June field reads)."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts/prepare_june_eligibility.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

MODEL_SRC = ROOT / "experiments/pitchmdp/pitchmdp/model.py"
DRAFT = ROOT / "configs/ML-JUNE-ELIGIBILITY-DRAFT-v1.json"
KEY = prep.KEY

# Independent transcription of the frozen P4 mapping (contract §3), not derived from the source.
BALL = {"ball", "blocked_ball", "pitchout", "intent_ball"}
STRIKE = {"called_strike", "swinging_strike", "swinging_strike_blocked", "missed_bunt", "foul_tip", "bunt_foul_tip"}
INPLAY = {"single": 4, "double": 5, "triple": 6, "home_run": 7, "field_out": 3, "force_out": 3,
          "fielders_choice_out": 3, "sac_fly": 3, "sac_bunt": 3, "double_play": 9,
          "grounded_into_double_play": 9, "sac_fly_double_play": 9, "sac_bunt_double_play": 9}


def ref_label(desc, event, strikes):
    if desc in BALL:
        return 0
    if desc in STRIKE:
        return 1
    if desc == "foul_bunt":
        return 1 if strikes == 2 else 2
    if desc == "foul":
        return 2
    if desc == "hit_by_pitch":
        return 8
    if desc == "hit_into_play":
        return INPLAY.get(event, -1)
    return -1


def ref_eligible(row) -> bool:
    return bool(ref_label(row["description"], row["events"], row["strikes"]) >= 0
                and row["supported_pa"] is True and pd.notna(row["pitch_type"])
                and pd.notna(row["plate_x"]) and pd.notna(row["plate_z"])
                and pd.notna(row["balls"]) and 0 <= row["balls"] <= 3
                and pd.notna(row["strikes"]) and 0 <= row["strikes"] <= 2)


def fns():
    return prep.extract_eligibility(MODEL_SRC.read_bytes())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ synthetic world

PANEL_IDS = [1, 2]
VOLUME = {1: "low", 2: "high", 3: "middle"}           # pitcher 4 absent from TRAIN -> zero
HAND = {1: "R", 2: "L", 3: "R", 4: "L"}
GAMES = {101: ("2025-06-01", 1, 3), 102: ("2025-06-03", 2, 4), 103: ("2025-06-10", 3, 1),
         104: ("2025-06-30", 4, 2), 99: ("2025-05-31", 1, 3), 105: ("2025-07-01", 2, 4)}
DEFECTS = [  # (game, pitcher, pitch index) -> overrides
    ((101, 1, 0), dict(description="hit_into_play", events="sac_bunt_double_play")),
    ((101, 1, 1), dict(supported_pa=None)),
    ((101, 3, 0), dict(plate_x=np.inf)),
    ((102, 2, 0), dict(description="foul_bunt", strikes=2)),
    ((102, 2, 1), dict(description="hit_into_play", events="catcher_interf", plate_z=np.nan)),
    ((102, 4, 1), dict(balls=4)),
    ((103, 1, 2), dict(strikes=3, pitch_type=None)),
    ((103, 3, 1), dict(supported_pa=False)),
    ((104, 2, 2), dict(plate_x=np.nan, plate_z=np.nan)),
    ((104, 4, 0), dict(description=None, events=None)),
]


def build_rows():
    rows, defects = [], dict(DEFECTS)
    for game, (day, starter, reliever) in GAMES.items():
        ab = 0
        for pitcher in (starter, reliever):
            ab += 1
            for i in range(3):
                row = dict(game_pk=game, at_bat_number=ab, pitch_number=i + 1, game_date=pd.Timestamp(day),
                           game_type="R", split="calibration" if day <= "2025-06-30" else "dev",
                           pitcher=pitcher, batter=900 + pitcher, starter_pitcher=starter, p_throws=HAND[pitcher],
                           description="ball", events=None, balls=1.0, strikes=1.0, supported_pa=True,
                           pitch_type="FF", plate_x=0.1, plate_z=2.5)
                row.update(defects.get((game, pitcher, i), {}))
                rows.append(row)
    rows.append({**rows[0], "game_pk": 106, "game_date": pd.Timestamp("2025-06-05"), "game_type": "S"})
    return rows


def write_world(tmp: Path, rows=None, blend_mutator=None, **cfg_over):
    tmp.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows or build_rows())
    frame["supported_pa"] = frame["supported_pa"].astype(object)
    frame.iloc[::-1].to_parquet(tmp / "pitches.parquet", index=False)  # reversed: script must sort
    req = frame.loc[frame.game_date.between("2025-06-01", "2025-06-30") & frame.game_type.eq("R")].copy()
    req = req.sort_values(["game_date", *KEY]).reset_index(drop=True)
    req["eligible"] = [ref_eligible(r) for _, r in req.iterrows()]
    req["in_cpanel"] = req.pitcher.isin(PANEL_IDS)
    blend = req.loc[req.in_cpanel & req.eligible].reset_index(drop=True)
    meta = blend[KEY].copy()
    meta["pitcher"], meta["batter"], meta["in_cpanel"] = blend.pitcher, blend.batter, True
    meta["train_volume"] = blend.pitcher.map(VOLUME).fillna("zero")
    meta["game_role"] = np.where(blend.pitcher.eq(blend.starter_pitcher), "starter", "relief")
    meta["throwing_hand"], meta["month"] = blend.p_throws, "2025-06"
    keys = blend[KEY].copy()
    if blend_mutator:
        keys, meta = blend_mutator(keys, meta)
    keys.to_parquet(tmp / "blend_keys.parquet", index=False)
    meta.to_parquet(tmp / "blend_metadata.parquet", index=False)
    panel = {"version": "cpanel_train_stratified_v1", "pitcher_ids": PANEL_IDS,
             "train_players": [{"pitcher": p, "train_pitches": n, "train_volume": VOLUME[p]}
                               for p, n in ((1, 100), (2, 2000), (3, 500))],
             "volume_thresholds": {"q25": 170.0, "q75": 1514.0, "method": "frozen"}}
    (tmp / "panel.json").write_text(json.dumps(panel))
    panel_req = req.loc[req.in_cpanel]
    labels = [ref_label(r.description, r.events, r.strikes) for r in panel_req.itertuples()]
    coverage = {"requested_pitches": len(panel_req), "eligible_pitches": int(panel_req.eligible.sum()),
                "unsupported_pa": int((panel_req.supported_pa != True).sum()),  # noqa: E712
                "unmapped_outcome": int((np.array(labels) < 0).sum()),
                "missing_type": int(panel_req.pitch_type.isna().sum()),
                "missing_coordinates": int((panel_req.plate_x.isna() | panel_req.plate_z.isna()).sum())}
    rows_sha = hashlib.sha256(np.ascontiguousarray(blend[KEY].to_numpy(np.int64)).tobytes()).hexdigest()
    (tmp / "p4.json").write_text(json.dumps({"samples": {"blend": {"rows_sha256": rows_sha}},
                                             "coverage": {"blend": coverage}}))
    (tmp / "guard.json").write_text("{}")
    (tmp / ".heavy.lock").touch()
    (tmp / "attempts").mkdir(exist_ok=True)

    def census(part):
        return {"pitches": len(part), "games": part.game_pk.nunique(), "pitchers": part.pitcher.nunique()}

    frozen = set(map(tuple, blend[KEY].to_numpy().tolist()))
    in_frozen = req[KEY].apply(tuple, axis=1).isin(frozen)
    cfg = json.loads(DRAFT.read_text())
    cfg["sources"] = {name: {"path": str(tmp / f), "sha256": sha(tmp / f)} for name, f in (
        ("processed_pitches", "pitches.parquet"), ("blend_keys", "blend_keys.parquet"),
        ("blend_metadata", "blend_metadata.parquet"), ("panel", "panel.json"), ("p4_preparation", "p4.json"))}
    cfg["sources"]["p4_model_source"] = {"path": str(MODEL_SRC), "sha256": sha(MODEL_SRC),
                                         "git_blob_sha1": "07f3389cf7c9300e122cacbf97b5c518921b2043",
                                         "approved_repo_path": "experiments/pitchmdp/pitchmdp/model.py"}
    cfg["read_only_guards"] = {"g0": {"path": str(tmp / "guard.json"), "sha256": sha(tmp / "guard.json")}}
    cfg["heavy_lock"] = str(tmp / ".heavy.lock")
    cfg["outputs"]["allowed_attempt_root"] = str(tmp / "attempts")
    cfg["expected"]["populations"] = {
        "requested": census(req), "panel_requested": census(panel_req),
        "panel_eligible": census(blend), "panel_excluded": census(panel_req.loc[~panel_req.eligible]),
        "outside_requested": census(req.loc[~req.in_cpanel]),
        "requested_minus_frozen_cpanel": census(req.loc[~in_frozen])}
    cfg["expected"]["eligible_bounds"] = {"min": len(blend), "max": len(req)}
    for dotted, value in cfg_over.items():
        target = cfg
        *path, last = dotted.split("__")
        for part in path:
            target = target[part]
        target[last] = value
    (tmp / "config.json").write_text(json.dumps(cfg))
    return tmp / "config.json", req


@pytest.fixture(autouse=True)
def _stub_provenance(monkeypatch):
    monkeypatch.delenv("PITCHEEZY_HEAVY_LOCK", raising=False)
    monkeypatch.setattr(prep, "implementation_provenance", lambda cfg, path: {
        "registered_code_commit": "synthetic", "execution_head": "synthetic",
        "executed_script_sha256": prep.sha256_file(prep.ROOT / prep.SCRIPT_REL), "config_path": str(path)})


# ------------------------------------------------------------------ eligibility semantics

def test_extracted_mapping_equals_independent_table():
    descs = [*BALL, *STRIKE, "foul", "foul_bunt", "hit_by_pitch", "hit_into_play", "automatic_ball",
             "pickoff_1b", "", None]
    events = [*INPLAY, "catcher_interf", "field_error", "strikeout", None]
    grid = [(d, e, s) for d in descs for e in events for s in (0, 1, 2)]
    frame = pd.DataFrame(grid, columns=["description", "events", "strikes"])
    got = fns()["outcome_labels"](frame)
    want = np.array([ref_label(d, e, s) for d, e, s in grid])
    assert (got == want).all()
    assert got[(frame.description.eq("foul_bunt") & frame.strikes.eq(2)).to_numpy()].tolist() == [1] * len(events)
    assert set(got[frame.events.eq("sac_bunt_double_play") & frame.description.eq("hit_into_play")]) == {9}


def test_eligible_edges_reason_union_and_bitmask_partition():
    base = dict(description="ball", events=None, balls=1, strikes=1, supported_pa=True, pitch_type="FF",
                plate_x=0.0, plate_z=2.0, game_pk=1)
    cases = [  # overrides, eligible, reasons
        ({}, True, set()),
        ({"plate_x": np.inf, "plate_z": -np.inf}, True, set()),           # notna, not finite
        ({"balls": 3, "strikes": 2}, True, set()),                        # inclusive bounds
        ({"supported_pa": None}, False, {"unsupported_pa"}),
        ({"supported_pa": np.nan}, False, {"unsupported_pa"}),
        ({"balls": 4}, False, {"invalid_balls"}),
        ({"balls": -1}, False, {"invalid_balls"}),
        ({"balls": np.nan}, False, {"invalid_balls"}),
        ({"strikes": 3, "pitch_type": None}, False, {"invalid_strikes", "missing_type"}),
        ({"description": "hit_into_play", "events": "catcher_interf", "plate_z": np.nan},
         False, {"unmapped_outcome", "missing_plate_z"}),
    ]
    frame = pd.DataFrame([{**base, **o} for o, _, _ in cases])
    frame["supported_pa"] = frame["supported_pa"].astype(object)
    flags = prep.eligibility_flags(frame, fns())
    assert flags["eligible"].tolist() == [e for _, e, _ in cases]
    for i, (_, _, reasons) in enumerate(cases):
        assert {r for r in prep.REASONS if flags.loc[i, r]} == reasons
    summary = prep.reason_summary(pd.concat([frame[["game_pk"]], flags], axis=1))
    assert summary["disjoint_bitmask"]["sum"] == int((~flags.eligible).sum())
    assert sum(v["pitches"] for v in summary["overlapping"].values()) > summary["disjoint_bitmask"]["sum"]


def test_reason_union_must_match_authoritative(monkeypatch):
    frame = pd.DataFrame([dict(description="ball", events=None, balls=1, strikes=1, supported_pa=True,
                               pitch_type="FF", plate_x=0.0, plate_z=2.0)])
    broken = dict(fns())
    broken["eligible"] = lambda f: np.zeros(len(f), dtype=bool)
    with pytest.raises(prep.PreparationError, match="reason union"):
        prep.eligibility_flags(frame, broken)


def test_support_rule_is_and_at_edges():
    def part(pitches, games):
        return pd.DataFrame({"g": "x", "game_pk": np.arange(pitches) % games, "pitcher": 1})
    rule = {"min_games": 30, "min_pitches": 500}
    got = {k: prep.support_table(part(*k), "g", ["x"], rule, True)["x"]["supported"]
           for k in ((500, 30), (499, 30), (500, 29), (10_000, 29))}
    assert got == {(500, 30): True, (499, 30): False, (500, 29): False, (10_000, 29): False}


# ------------------------------------------------------------------ end to end

def test_end_to_end_synthetic_success(tmp_path):
    cfg_path, req = write_world(tmp_path)
    out = tmp_path / "attempts" / "a1"
    result = prep.run(cfg_path, out)
    files = json.loads(cfg_path.read_text())["outputs"]["files"]
    manifest = json.loads((out / files["manifest"]).read_text())
    assert manifest["status"] == "complete" and not (out / files["failure"]).exists()
    for name, entry in manifest["artifacts"].items():
        assert sha(out / name) == entry["sha256"]
    inv = pd.read_parquet(out / files["inventory"])
    assert inv[KEY].equals(req[KEY].astype("int64"))                          # pushdown + sort; S/May/July gone
    assert inv["eligible"].tolist() == req["eligible"].tolist()
    assert (inv.research_split == "blend").all() and (inv.source_split == "calibration").all()
    comp = pd.read_parquet(out / files["complement_eligible_keys"])
    outside = req.loc[req.eligible & ~req.in_cpanel, KEY].astype("int64").reset_index(drop=True)
    assert comp.equals(outside)
    for name in files.values():
        if name.endswith(".parquet"):
            cols = set(pq.read_schema(out / name).names)
            assert not cols & (set(prep.CONTRACT_ELIGIBILITY_COLUMNS) | {"label", "y", "pitch_outcome"})
    assert result["counts"]["eligible"]["pitches"] == int(req.eligible.sum())
    assert inv.loc[inv.pitcher.eq(4), "train_volume"].eq("zero").all()
    assert result["independent_confirmation"] is None and result["new_inference_count"] == 0
    assert result["exposure"]["freshness_claim"] is None
    with pytest.raises(prep.PreparationError, match="existing output"):
        prep.run(cfg_path, out)


def test_cpanel_order_mismatch_aborts_and_preserves_partial(tmp_path):
    def swap(keys, meta):
        return keys.iloc[[1, 0, *range(2, len(keys))]].reset_index(drop=True), \
            meta.iloc[[1, 0, *range(2, len(meta))]].reset_index(drop=True)
    cfg_path, _ = write_world(tmp_path, blend_mutator=swap)
    out = tmp_path / "attempts" / "bad"
    with pytest.raises(prep.PreparationError, match="ordered keys differ"):
        prep.run(cfg_path, out)
    failure = json.loads((out / "failure.json").read_text())
    assert failure["stage"] == "gates" and not (out / "manifest.json").exists()
    assert (out / "failure.log").exists()


def test_cpanel_metadata_mismatch_aborts(tmp_path):
    def relabel(keys, meta):
        meta = meta.copy()
        meta.loc[0, "game_role"] = "relief" if meta.loc[0, "game_role"] == "starter" else "starter"
        return keys, meta
    cfg_path, _ = write_world(tmp_path, blend_mutator=relabel)
    with pytest.raises(prep.PreparationError, match="game_role"):
        prep.run(cfg_path, tmp_path / "attempts" / "m")


def test_known_excluded_count_mismatch_aborts(tmp_path):
    cfg_path, _ = write_world(tmp_path)
    cfg = json.loads(cfg_path.read_text())
    cfg["expected"]["populations"]["panel_excluded"]["pitches"] += 1
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(prep.PreparationError, match="panel_excluded.pitches"):
        prep.run(cfg_path, tmp_path / "attempts" / "x")


def test_pin_mutation_stops_before_any_decode(tmp_path, monkeypatch):
    cfg_path, _ = write_world(tmp_path)
    monkeypatch.setattr(prep, "read_projection", lambda *a, **k: pytest.fail("decoded before pins verified"))
    with open(tmp_path / "blend_metadata.parquet", "ab") as handle:
        handle.write(b"\0")
    with pytest.raises(prep.PreparationError, match="hash mismatch for sources.blend_metadata"):
        prep.run(cfg_path, tmp_path / "attempts" / "p")
    cfg_path, _ = write_world(tmp_path / "w2", sources__p4_model_source__git_blob_sha1="0" * 40)
    with pytest.raises(prep.PreparationError, match="approved git blob"):
        prep.run(cfg_path, tmp_path / "w2" / "attempts" / "p")


def test_config_boundaries(tmp_path):
    cfg_path, _ = write_world(tmp_path)
    good = json.loads(cfg_path.read_text())
    bad_cases = {
        "2026": ("window", {**good["window"], "date_min": "2026-06-01", "date_max": "2026-06-30",
                            "season": 2026}),
        "extra column": ("columns", {**good["columns"], "eligibility": [*good["columns"]["eligibility"],
                                                                        "pitch_outcome"]}),
        "fold": ("research_fold", {**good["research_fold"], "date_max": "2025-06-29"}),
        "OR support": ("support_rule", {**good["support_rule"], "combine": "OR"}),
        "starter to relief": ("unknown_policy", {**good["unknown_policy"], "missing_starter_pitcher": "relief"}),
    }
    for label, (field, value) in bad_cases.items():
        cfg = copy.deepcopy(good)
        cfg[field] = value
        with pytest.raises(prep.PreparationError):
            prep.validate_config(cfg)
    with pytest.raises(prep.PreparationError, match="allowed_attempt_root"):
        prep.validate_config(json.loads(DRAFT.read_text()))
    assert not list((tmp_path / "attempts").iterdir())


def test_unregistered_draft_provenance_refuses():
    spec2 = importlib.util.spec_from_file_location("prep_raw", ROOT / "scripts/prepare_june_eligibility.py")
    raw = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(raw)
    with pytest.raises(raw.PreparationError, match="unregistered draft"):
        raw.implementation_provenance(json.loads(DRAFT.read_text()), DRAFT)


def test_missing_starter_and_legacy_split_fail_closed(tmp_path):
    rows = build_rows()
    rows[7]["starter_pitcher"] = None
    cfg_path, _ = write_world(tmp_path / "s", rows=rows)
    with pytest.raises(prep.PreparationError, match="missing_starter_pitcher"):
        prep.run(cfg_path, tmp_path / "s" / "attempts" / "a")
    rows = build_rows()
    rows[7]["split"] = "dev"
    cfg_path, _ = write_world(tmp_path / "l", rows=rows)
    with pytest.raises(prep.PreparationError, match="legacy split"):
        prep.run(cfg_path, tmp_path / "l" / "attempts" / "a")


def test_busy_heavy_lock_fails_without_waiting(tmp_path):
    cfg_path, _ = write_world(tmp_path)
    with open(tmp_path / ".heavy.lock", "rb") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(prep.PreparationError, match="holds the shared lock"):
            prep.run(cfg_path, tmp_path / "attempts" / "busy")
    assert json.loads((tmp_path / "attempts" / "busy" / "failure.json").read_text())["stage"] == "lock"
