"""Approved Statcast preparation, chronological histories, and frozen LAD cohort.

All current-pitch realization columns are retained only for conditional outcome
model fitting/diagnostics. They are not a pre-pitch policy feature contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RAW_ALLOWLIST = [f"statcast_{year}.parquet" for year in (2023, 2024, 2025)]
KEY = ["game_pk", "at_bat_number", "pitch_number"]
PA_KEY = ["game_pk", "at_bat_number"]
RAW_COLUMNS = [
    "game_date", "game_pk", "at_bat_number", "pitch_number", "game_type",
    "pitch_type", "player_name", "pitcher", "batter", "stand", "p_throws",
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "home_team", "away_team", "home_score", "away_score", "bat_score", "fld_score",
    "post_home_score", "post_away_score", "post_bat_score", "post_fld_score",
    "on_1b", "on_2b", "on_3b", "plate_x", "plate_z", "sz_top", "sz_bot",
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "events",
    "description", "type", "launch_speed", "launch_angle",
]
EVENT_OUTS = {
    "field_out": 1, "strikeout": 1, "force_out": 1,
    "fielders_choice_out": 1, "sac_fly": 1, "sac_bunt": 1,
    "grounded_into_double_play": 2, "double_play": 2,
    "strikeout_double_play": 2, "sac_fly_double_play": 2, "triple_play": 3,
}
SUPPORTED_EVENTS = {
    "field_out", "strikeout", "force_out", "fielders_choice_out", "sac_fly",
    "grounded_into_double_play", "double_play", "sac_fly_double_play",
    "single", "double", "triple", "home_run", "walk", "hit_by_pitch",
}
DESCRIPTION_LABELS = {
    "ball": "ball", "blocked_ball": "ball", "pitchout": "ball",
    "called_strike": "strike", "swinging_strike": "strike",
    "swinging_strike_blocked": "strike", "foul_tip": "strike", "foul": "foul",
    "hit_by_pitch": "hbp",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def add_splits(df: pd.DataFrame) -> None:
    date = df["game_date"]
    df["split"] = np.select(
        [date.lt("2023-05-15"), date.le("2025-04-30"), date.le("2025-06-30")],
        ["history", "train", "calibration"], default="dev",
    )


def reconstruct_outs(df: pd.DataFrame) -> pd.DataFrame:
    """Credit interval outs to the pitcher on its left; expose substitution ambiguity.

    Statcast is a pitch log, so non-pitch runner outs may be seen only in the next
    pitch's pre-state. At a pitcher substitution those extra outs cannot always
    be assigned uniquely. Lower/upper starter bounds expose this limitation.
    """
    nxt = df.shift(-1)
    same_game = df.game_pk.eq(nxt.game_pk)
    same_half = same_game & df.inning.eq(nxt.inning) & df.inning_topbot.eq(nxt.inning_topbot)
    implied = df.events.map(EVENT_OUTS).fillna(0).astype(int)
    delta = np.where(same_half, nxt.outs_when_up - df.outs_when_up,
                     np.where(same_game, 3 - df.outs_when_up, implied))
    delta = pd.Series(delta, index=df.index).astype(int)
    invalid = delta.lt(0) | delta.gt(3) | (df.outs_when_up + delta).gt(3)
    if invalid.any():
        raise ValueError(f"Invalid outs deltas in {int(invalid.sum())} intervals")
    substitution = same_half & df.pitcher.ne(nxt.pitcher)
    ambiguous_extra = np.where(substitution, np.maximum(delta - implied, 0), 0)
    return pd.DataFrame({
        "outs_recorded": delta,
        "outs_ambiguous_extra": ambiguous_extra,
        "outs_interval_pitcher_change": substitution,
        "outs_event_implied": implied,
    }, index=df.index)


def freeze_cohort(df: pd.DataFrame, config: dict, out: Path) -> dict:
    df["fielding_team"] = df.home_team.where(df.inning_topbot.eq("Top"), df.away_team)
    first = df.groupby(["game_pk", "inning_topbot"], sort=False).head(1)
    first = first.loc[first.inning.eq(1)].copy()
    if first.outs_when_up.ne(0).any():
        raise ValueError("A first defensive pitch has nonzero outs: starter identification ambiguous")
    starter_lookup = first.set_index(["game_pk", "inning_topbot"])["pitcher"]
    index = pd.MultiIndex.from_frame(df[["game_pk", "inning_topbot"]])
    df["starter_pitcher"] = starter_lookup.reindex(index).to_numpy()
    df["is_lad_start"] = df.fielding_team.eq("LAD") & df.pitcher.eq(df.starter_pitcher)
    cohort = config["cohort"]
    selected_dates = df.game_date.between(cohort["period_start"], cohort["period_end"])
    selected = df.loc[df.is_lad_start & selected_dates]
    starts = selected.groupby(["pitcher", "game_pk"], sort=True).agg(
        player_name=("player_name", "first"), game_date=("game_date", "first"),
        outs=("outs_recorded", "sum"), uncertain_outs=("outs_ambiguous_extra", "sum"),
        pitches=("pitch_number", "size"),
    ).reset_index()
    rank = starts.groupby("pitcher", sort=True).agg(
        player_name=("player_name", "first"), starts=("game_pk", "size"),
        outs=("outs", "sum"), uncertain_outs=("uncertain_outs", "sum"),
    ).reset_index().sort_values(["outs", "starts", "pitcher"], ascending=[False, False, True])
    rank["outs_lower_bound"] = rank.outs - rank.uncertain_outs
    rank["innings"] = rank.outs.map(lambda x: f"{x // 3}.{x % 3}")
    count = cohort["count"]
    chosen = rank.head(count)
    # Conservatively refuse a freeze if substitution attribution could change membership.
    stable = len(rank) >= count and chosen.outs_lower_bound.min() > rank.iloc[count:].outs.max()
    if not stable:
        raise ValueError("Cohort boundary is unstable under runner-out attribution bounds")
    ids = chosen.pitcher.astype(int).tolist()
    df["cohort_pitcher"] = df.pitcher.isin(ids)
    starts["game_date"] = starts.game_date.dt.strftime("%Y-%m-%d")
    manifest = {
        "frozen_at_utc": utc_now(), "status": "frozen_before_model_evaluation",
        "period_start": cohort["period_start"], "period_end": cohort["period_end"],
        "team": "LAD", "role": "first pitcher on team's first defensive pitch",
        "ranking": "sum reconstructed outs in LAD starts; starts descending then MLBAM ID ascending",
        "outs_method": "next pre-pitch outs minus current outs, or 3-current at half-inning boundary; final rows event-implied",
        "outs_uncertainty": "non-pitch outs between a substitution and its next pitch may belong to incoming pitcher; outgoing attribution is an upper bound",
        "boundary_stable_under_outs_uncertainty": bool(stable),
        "pitcher_ids": ids, "selected": chosen.to_dict("records"),
        "ranking_all": rank.to_dict("records"), "starts": starts.to_dict("records"),
        "batter_filter": None,
    }
    if out.exists():
        old = json.loads(out.read_text())
        if old["pitcher_ids"] != ids:
            raise ValueError("Existing frozen cohort differs; refusing silent replacement")
        manifest["frozen_at_utc"] = old["frozen_at_utc"]
    write_json(out, manifest)
    return manifest


def add_history(df: pd.DataFrame) -> None:
    groups = df.groupby(PA_KEY, sort=False)
    df["prev_pitch_type"] = groups.pitch_type.shift().fillna("START")
    df["prev_description"] = groups.description.shift().fillna("START")
    df["prev_plate_x"] = groups.plate_x.shift()
    df["prev_plate_z"] = groups.plate_z.shift()
    # Aggregate entire dates before shifting: same-day doubleheaders never see one another.
    terminal = df.loc[df.is_pa_terminal].copy()
    terminal["obp_reach"] = terminal.events.isin(["single", "double", "triple", "home_run", "walk", "intent_walk", "hit_by_pitch"]).astype(int)
    terminal["obp_denom"] = ~terminal.events.isin(["sac_bunt", "catcher_interf"])
    terminal["strikeout"] = terminal.events.isin(["strikeout", "strikeout_double_play"]).astype(int)
    daily = terminal.groupby(["batter", "game_date"], sort=True).agg(
        pa=("events", "size"), on_base=("obp_reach", "sum"),
        obp_denominator=("obp_denom", "sum"), strikeouts=("strikeout", "sum"),
    )
    # Reindex all pitcher/batter dates, including days with truncated-only PA logs.
    all_dates = pd.MultiIndex.from_frame(df[["batter", "game_date"]].drop_duplicates()).sort_values()
    daily = daily.reindex(all_dates, fill_value=0)
    prior = daily.groupby(level="batter").cumsum() - daily
    prior["batter_pa_prior"] = prior.pa
    prior["batter_obp_prior"] = (prior.on_base + 30 * .320) / (prior.obp_denominator + 30)
    prior["batter_k_prior"] = (prior.strikeouts + 30 * .220) / (prior.pa + 30)
    lookup = pd.MultiIndex.from_frame(df[["batter", "game_date"]])
    for col in ["batter_pa_prior", "batter_obp_prior", "batter_k_prior"]:
        df[col] = prior[col].reindex(lookup).to_numpy()


def add_transitions_and_support(df: pd.DataFrame) -> dict:
    df["bases"] = df.on_1b.notna().astype(int) + 2 * df.on_2b.notna().astype(int) + 4 * df.on_3b.notna().astype(int)
    group = df.groupby(PA_KEY, sort=False)
    last = ~df.duplicated(PA_KEY, keep="last")
    df["is_pa_terminal"] = last & df.events.notna() & df.events.ne("truncated_pa")
    df["terminal_event"] = df.events.where(df.is_pa_terminal)
    nxt = df.shift(-1)
    observed = df.is_pa_terminal & df.game_pk.eq(nxt.game_pk)
    for target, source in {
        "next_outs": "outs_when_up", "next_bases": "bases", "next_home_score": "home_score",
        "next_away_score": "away_score", "next_inning": "inning", "next_half": "inning_topbot",
    }.items():
        df[target] = nxt[source].where(observed)

    game_last = df.groupby("game_pk", sort=False).tail(1).copy()
    implied = game_last.events.map(EVENT_OUTS).fillna(0)
    half_complete = (game_last.outs_when_up + implied).eq(3)
    home_leads = game_last.post_home_score.gt(game_last.post_away_score)
    away_leads = game_last.post_home_score.lt(game_last.post_away_score)
    top = game_last.inning_topbot.eq("Top")
    complete = game_last.inning.ge(9) & game_last.is_pa_terminal & (
        (top & half_complete & home_leads) | (~top & half_complete & away_leads) |
        (~top & home_leads)
    )
    game_last["complete_game"] = complete
    game_last["final_home_win"] = home_leads.astype(float).where(complete)
    for col in ["complete_game", "final_home_win"]:
        df[col] = df.game_pk.map(game_last.set_index("game_pk")[col])

    df["pitch_outcome"] = df.description.map(DESCRIPTION_LABELS)
    inplay = df.description.eq("hit_into_play")
    event_labels = {
        "field_out": "out", "force_out": "out", "fielders_choice_out": "out", "sac_fly": "out",
        "grounded_into_double_play": "double_play", "double_play": "double_play", "sac_fly_double_play": "double_play",
        "single": "single", "double": "double", "triple": "triple", "home_run": "home_run",
    }
    df.loc[inplay, "pitch_outcome"] = df.loc[inplay, "events"].map(event_labels)
    # The support decision belongs to the whole PA, so future events are used only
    # for coverage/evaluation eligibility, never in a pre-pitch feature vector.
    first = group.head(1)
    state_cols = ["outs_when_up", "on_1b", "on_2b", "on_3b", "home_score", "away_score"]
    state_changes = group[state_cols].nunique(dropna=False).gt(1).any(axis=1)
    first_index = pd.MultiIndex.from_frame(first[PA_KEY])
    first_valid = pd.Series(first.balls.eq(0).to_numpy() & first.strikes.eq(0).to_numpy(), index=first_index)
    terminal = df.loc[last].set_index(PA_KEY)
    irregular_strikeout = terminal.events.eq("strikeout") & terminal.outs_recorded.ne(1)
    invalid_out = terminal.pitch_outcome.eq("out") & terminal.outs_recorded.lt(1)
    invalid_dp = terminal.pitch_outcome.eq("double_play") & terminal.outs_recorded.ne(2)
    inconsistent_label = ((terminal.events.eq("strikeout") & ~(terminal.pitch_outcome.eq("strike") & terminal.strikes.eq(2))) |
                          (terminal.events.eq("walk") & ~(terminal.pitch_outcome.eq("ball") & terminal.balls.eq(3))))
    inconsistent_label |= (terminal.pitch_outcome.eq("strike") & terminal.events.ne("strikeout"))
    inconsistent_label |= (terminal.pitch_outcome.eq("ball") & terminal.events.ne("walk"))
    # Solver treatment of awards, strikeouts, and home runs is deterministic.
    # Remove observed trajectories containing extra runner movement on that play.
    deterministic = terminal.events.isin(["walk", "hit_by_pitch", "strikeout", "home_run"])
    base = terminal.bases.astype(int)
    expected_bases = base.copy()
    awards = terminal.events.isin(["walk", "hit_by_pitch"])
    expected_bases.loc[awards] = (base | 1 | ((base & 1) * 2) | ((base & 3).eq(3).astype(int) * 4)).loc[awards]
    home_runs = terminal.events.eq("home_run")
    expected_bases.loc[home_runs] = 0
    runs = (awards & base.eq(7)).astype(int)
    runs.loc[home_runs] = base.map(int.bit_count).loc[home_runs] + 1
    expected_outs = terminal.outs_when_up + terminal.events.eq("strikeout").astype(int)
    half_end = expected_outs.eq(3)
    expected_outs = expected_outs.where(~half_end, 0)
    expected_inning = terminal.inning + (half_end & terminal.inning_topbot.eq("Bot")).astype(int)
    expected_half = terminal.inning_topbot.where(~half_end, terminal.inning_topbot.map({"Top": "Bot", "Bot": "Top"}))
    expected_bases = expected_bases.where(~half_end, expected_inning.ge(10).astype(int) * 2)
    expected_home_score = terminal.home_score + runs * terminal.inning_topbot.eq("Bot")
    expected_away_score = terminal.away_score + runs * terminal.inning_topbot.eq("Top")
    unsupported_deterministic = deterministic & (
        terminal.next_outs.ne(expected_outs) | terminal.next_bases.ne(expected_bases) |
        terminal.next_inning.ne(expected_inning) | terminal.next_half.ne(expected_half) |
        terminal.next_home_score.ne(expected_home_score) | terminal.next_away_score.ne(expected_away_score)
    )
    # A foul with two strikes is a self-loop. No observed future count is used
    # as a model feature; this check only detects unsupported/log-inconsistent PAs.
    same_pa = df.game_pk.eq(nxt.game_pk) & df.at_bat_number.eq(nxt.at_bat_number)
    expected_balls = df.balls + df.pitch_outcome.eq("ball").astype(int)
    expected_strikes = df.strikes + df.pitch_outcome.eq("strike").astype(int) + (df.pitch_outcome.eq("foul") & df.strikes.lt(2)).astype(int)
    bad_count_path = same_pa & (
        ~df.pitch_outcome.isin(["ball", "strike", "foul"]) |
        nxt.balls.ne(expected_balls) | nxt.strikes.ne(expected_strikes)
    )
    bad_count_by_pa = bad_count_path.groupby([df.game_pk, df.at_bat_number]).any()
    pitch_valid = (df.pitch_outcome.notna() & df.pitch_type.notna() & df.plate_x.notna() & df.plate_z.notna() &
                   df.sz_top.gt(df.sz_bot) & df.balls.between(0, 3) & df.strikes.between(0, 2))
    valid_by_pa = pitch_valid.groupby([df.game_pk, df.at_bat_number]).all()
    regular = df.game_type.eq("R") & df.inning.between(1, 9)
    regular_by_pa = regular.groupby([df.game_pk, df.at_bat_number]).all()
    reasons = pd.DataFrame({
        "mid_pa_state_change": state_changes,
        "not_zero_count_start": ~first_valid,
        "unsupported_terminal_event": ~terminal.events.isin(SUPPORTED_EVENTS),
        "missing_or_unsupported_pitch": ~valid_by_pa,
        "pitcher_or_batter_changes": group[["pitcher", "batter"]].nunique().gt(1).any(axis=1),
        "outside_regular_innings": ~regular_by_pa,
        "unobserved_terminal_transition": terminal.next_outs.isna(),
        "irregular_terminal_outs": irregular_strikeout | invalid_out | invalid_dp,
        "inconsistent_terminal_count_label": inconsistent_label,
        "inconsistent_with_deterministic_terminal_rule": unsupported_deterministic,
        "inconsistent_with_observed_count_path": bad_count_by_pa,
        "incomplete_game": ~terminal.complete_game,
    }).fillna(True)
    supported = ~reasons.any(axis=1)
    pa_index = pd.MultiIndex.from_frame(df[PA_KEY])
    df["supported_pa"] = supported.reindex(pa_index).to_numpy()
    # Detect unobserved runner movement after the logged final pitch. A positive
    # discrepancy can contain a steal, pickoff, or score correction and is excluded.
    score_mismatch = observed & (df.next_home_score.ne(df.post_home_score) | df.next_away_score.ne(df.post_away_score))
    mismatched_pas = pd.MultiIndex.from_frame(df.loc[score_mismatch, PA_KEY])
    if len(mismatched_pas):
        df.loc[pa_index.isin(mismatched_pas), "supported_pa"] = False
    counts = {str(k): int(v) for k, v in reasons.sum().items()}
    counts["post_pitch_score_disagrees_next_pitch"] = int(score_mismatch.sum())
    counts["excluded_unique_pas"] = int((~df.groupby(PA_KEY).supported_pa.first()).sum())
    counts["total_pas"] = int(len(terminal))
    return counts


def prepare(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config["raw_allowlist"] != RAW_ALLOWLIST:
        raise ValueError("Only the three exact approved raw files may be opened")
    volume = Path("/Volumes/T7 Shield")
    root = Path(config["artifact_root"])
    if not volume.is_mount() or not root.resolve().is_relative_to(volume.resolve()):
        raise RuntimeError("Mounted T7 Shield artifact root is required")
    for subdir in ["reports", "processed"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    began = utc_now()
    sources, frames = [], []
    for name in RAW_ALLOWLIST:
        path = Path(config["raw_root"]) / name
        print(f"Reading approved source {name}", flush=True)
        sources.append({"file": name, "bytes": path.stat().st_size, "sha256": hash_file(path)})
        frames.append(pq.read_table(path, columns=RAW_COLUMNS).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["game_date"] = df.game_date.dt.normalize()
    df.sort_values(["game_date", *KEY], inplace=True, ignore_index=True)
    if df.duplicated(KEY).any():
        raise ValueError("Duplicate game/PA/pitch keys")
    if df.game_date.dt.year.eq(2026).any():
        raise ValueError("Forbidden 2026 rows in approved files")
    for column, values in reconstruct_outs(df).items():
        df[column] = values
    add_splits(df)
    manifest = freeze_cohort(df, config, root / "reports/cohort_manifest.json")
    print("Frozen cohort: " + ", ".join(x["player_name"] for x in manifest["selected"]), flush=True)
    exclusions = add_transitions_and_support(df)
    add_history(df)
    # A strict first-pitch history reset is a hard guard, not merely a metric.
    first = df.groupby(PA_KEY, sort=False).head(1)
    if not first.prev_pitch_type.eq("START").all() or first.prev_plate_x.notna().any():
        raise ValueError("Previous pitch crosses PA boundary")
    cohort_df = df.loc[df.is_lad_start & df.cohort_pitcher]
    summary = cohort_df.groupby(["split", "pitcher", "player_name"]).agg(
        pitches=("pitch_number", "size"), games=("game_pk", "nunique"),
        batters=("batter", "nunique"), supported_pitches=("supported_pa", "sum"),
    ).reset_index().to_dict("records")
    pa_first = df.groupby(PA_KEY, sort=False).head(1)
    pa_summary = pa_first.groupby(["split", "cohort_pitcher", "is_lad_start"]).agg(
        pas=("game_pk", "size"), supported_pas=("supported_pa", "sum"),
        games=("game_pk", "nunique"), batters=("batter", "nunique"),
    ).reset_index().to_dict("records")
    report = {
        "created_at_utc": utc_now(), "started_at_utc": began, "sources": sources,
        "raw_columns": RAW_COLUMNS, "rows": len(df), "games": int(df.game_pk.nunique()),
        "unique_pitch_key": KEY, "duplicate_pitch_keys": 0,
        "date_min": str(df.game_date.min().date()), "date_max": str(df.game_date.max().date()),
        "split_rows": {str(k): int(v) for k, v in df.split.value_counts().items()},
        "split_contract": {"history": "before 2023-05-15", "train": "2023-05-15 through 2025-04-30",
                           "calibration": "2025-05-01 through 2025-06-30", "dev": "2025-07-01 through 2025-12-31"},
        "exclusion_reason_pa_counts_overlapping": exclusions,
        "cohort_samples": summary, "pa_samples": pa_summary,
        "complete_games": int(df.groupby("game_pk").complete_game.first().sum()),
        "pitch_outcome_counts": {str(k): int(v) for k, v in df.pitch_outcome.value_counts(dropna=False).items()},
        "batter_prior": "strictly before game_date, same-day games excluded; fixed 30 PA smoothing to OBP .320 and K .220",
        "history_scope": "previous pitch in same PA only; START otherwise",
        "names": "player_name is pitcher name; no batter-name field is available in approved raw schema",
        "support_warning": "Eligibility excludes special/missing/runner-changing PAs by observed trajectory; this restricted-population selection is not causal-policy validation",
        "output": str(root / "processed/pitches.parquet"),
    }
    output = root / "processed/pitches.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    df.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)
    report["processed_bytes"] = output.stat().st_size
    report["processed_sha256"] = hash_file(output)
    write_json(root / "reports/data_quality.json", report)
    print(json.dumps({"output": str(output), "rows": len(df), "cohort": manifest["selected"], "exclusions": exclusions}, indent=2), flush=True)
    return report
