#!/usr/bin/env python3
"""Freeze descriptive MLB starter cohort proposals before new model evaluation.

Reads only the previously prepared 2023–2025 pitch table. Does not train models,
read result metrics, overwrite the original LAD manifest, or alter pitch data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

PERIODS = {
    "train": ["2023-05-15", "2025-04-30"],
    "calibration": ["2025-05-01", "2025-06-30"],
    "dev": ["2025-07-01", "2025-09-28"],
}
COLS = ["game_date", "game_pk", "at_bat_number", "pitcher", "starter_pitcher",
        "player_name", "batter", "split", "fielding_team", "game_type",
        "outs_recorded", "outs_ambiguous_extra", "supported_pa"]


def ip(outs: int) -> str:
    return f"{outs // 3}.{outs % 3}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/Volumes/T7 Shield/pitcheezy/pitchmdp"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--min-eval-ip", type=int, default=50)
    args = parser.parse_args()
    if args.count < 1 or args.min_eval_ip < 1:
        raise ValueError("Positive cohort size and evaluation innings required")
    root = args.artifact_root
    source = root / "processed/pitches.parquet"
    source_hash = sha256(source)
    df = pd.read_parquet(source, columns=COLS)
    if not df.game_date.dt.year.isin([2023, 2024, 2025]).all():
        raise ValueError("Only 2023–2025 rows are authorized")
    # All-team starters: pitcher identity persists through trades. LAD flags ignored.
    s = df.loc[df.pitcher.eq(df.starter_pitcher) & df.split.isin(PERIODS)].copy()
    if not s.game_type.eq("R").all():
        raise ValueError("Expected regular-season-only processed data")
    for split, (start, end) in PERIODS.items():
        if not s.loc[s.split.eq(split), "game_date"].between(start, end).all():
            raise ValueError(f"Unexpected dates in {split}")
    agg = s.groupby(["pitcher", "split"]).agg(
        starts=("game_pk", "nunique"), outs=("outs_recorded", "sum"),
        ambiguous_outs=("outs_ambiguous_extra", "sum"), batters=("batter", "nunique"),
    )
    pa = s.drop_duplicates(["pitcher", "game_pk", "at_bat_number"])
    pa_agg = pa.groupby(["pitcher", "split"]).agg(
        pas=("game_pk", "size"), eligible_pas=("supported_pa", "sum"),
    )
    agg = agg.join(pa_agg)
    eligible_batters = pa.loc[pa.supported_pa].groupby(["pitcher", "split"]).batter.nunique()
    agg["eligible_batters"] = eligible_batters.reindex(agg.index, fill_value=0)
    wide = agg.unstack("split", fill_value=0)
    records = []
    for pitcher, group in s.groupby("pitcher", sort=True):
        record = {"pitcher": int(pitcher), "player_name": str(group.player_name.iloc[-1])}
        for split in PERIODS:
            period = {field: int(wide.loc[pitcher, (field, split)]) for field in agg.columns}
            period.update(innings=ip(period["outs"]), innings_decimal=period["outs"] / 3,
                          outs_lower_bound=period["outs"] - period["ambiguous_outs"],
                          innings_lower_bound=ip(period["outs"] - period["ambiguous_outs"]))
            period["teams"] = sorted(group.loc[group.split.eq(split), "fielding_team"].unique().tolist())
            record[split] = period
        starts = group[["game_date", "game_pk", "fielding_team"]].drop_duplicates().sort_values(["game_date", "game_pk"])
        changes = starts.loc[starts.fielding_team.ne(starts.fielding_team.shift())]
        record["team_sequence"] = [{"first_observed_start": str(row.game_date.date()), "team": str(row.fielding_team)} for row in changes.itertuples()]
        record["teams"] = sorted(starts.fielding_team.unique().tolist())
        record["team_changes_observed"] = max(0, len(changes) - 1)
        records.append(record)
    # Eligibility uses a conservative out bound, so an ambiguous runner out
    # cannot qualify someone whose true recorded innings might fall below 50.
    eligible = [r for r in records if r["train"]["outs"] > 0 and r["dev"]["outs_lower_bound"] >= 3 * args.min_eval_ip]
    ranked = sorted(eligible, key=lambda r: (-r["train"]["outs"], -r["train"]["starts"], r["pitcher"]))
    if len(ranked) < args.count:
        raise ValueError("Insufficient eligible starters")
    rankframe = pd.DataFrame({"pitcher": [r["pitcher"] for r in eligible],
                              "train": [r["train"]["outs"] for r in eligible],
                              "dev": [r["dev"]["outs"] for r in eligible]}).set_index("pitcher")
    percentiles = rankframe.rank(method="average", pct=True, ascending=True)
    for i, record in enumerate(ranked, 1):
        record["train_rank_among_eligible"] = i
        record["train_workload_percentile"] = float(percentiles.loc[record["pitcher"], "train"])
        record["eval_workload_percentile"] = float(percentiles.loc[record["pitcher"], "dev"])
        record["balanced_score"] = min(record["train_workload_percentile"], record["eval_workload_percentile"])
    balanced = sorted(eligible, key=lambda r: (-r["balanced_score"], -r["train"]["outs"], -r["train"]["starts"], r["pitcher"]))
    for i, record in enumerate(balanced, 1):
        record["balanced_rank_among_eligible"] = i
    chosen = ranked[:args.count]
    outsiders = ranked[args.count:]
    boundary_stable = not outsiders or min(r["train"]["outs_lower_bound"] for r in chosen) > max(r["train"]["outs"] for r in outsiders)
    if not boundary_stable:
        raise ValueError("Train-top membership sensitive to reconstructed-out attribution")
    limitations = [
        "Evaluation exposure is used in eligibility. This is a retrospective surviving high-workload population, not a prospective no-lookahead cohort.",
        "Exposure-only selection avoids choosing pitchers by new model metrics, but availability/workload can depend on health, performance and team decisions; MLB-wide causal or prospective generalization is unsupported.",
        "Balanced ranking uses evaluation workload continuously and is descriptive; train-top with a fixed 50-IP eligibility threshold is the recommended simpler rule.",
        "Out totals are exact integer sums of existing reconstructed outs, not pitch-count proxies or independently reconciled official box-score IP. Substitution-attribution uncertainty is reported as lower/upper bounds.",
        "Eligible PA excludes observed unsupported trajectories; this is outcome-dependent support restriction. All batter identities are retained without an individual sample threshold.",
        "The July–September 2025 period is development evaluation, not an untouched confirmatory test. No 2026 data is read.",
        "Training period is longer than evaluation; raw IP totals should not be compared as equal-duration rates. Balanced score uses within-period ranks over the same eligible population.",
    ]
    frozen = datetime.now(timezone.utc).isoformat()
    report_path = root / "reports/mlb_cohort_candidates.json"
    proposal_path = Path(__file__).resolve().parents[1] / "configs/mlb_cohort_proposal.json"
    proposal = {
        "status": "selection_proposal_frozen_before_new_MLB_cohort_model_metrics",
        "frozen_at_utc": frozen, "artifact_root": str(root),
        "processed_source": str(source), "processed_sha256": source_hash,
        "count": args.count, "team_filter": None, "role": "pitcher equals first defensive pitcher of game",
        "periods": PERIODS, "eval_split": "dev", "min_eval_innings": args.min_eval_ip,
        "eligibility": f"positive train starter outs and dev starter outs lower bound >= {3 * args.min_eval_ip} ({args.min_eval_ip} IP)",
        "selection_rule": "train starter outs descending; train starts descending; MLBAM ID ascending",
        "pitcher_ids": [r["pitcher"] for r in chosen], "batter_filter": None,
        "boundary_stable_under_outs_uncertainty": boundary_stable,
        "retrospective_exposure_selection": True, "causal_effect_claim": False,
        "raw_2026_access": False, "new_model_metrics_read": False,
        "cohort_training_filter": "all teams, selected persistent pitcher IDs, starter appearances",
        "shared_model_training_pool": "league-wide training data remains permitted; distinguish it from reporting cohort",
        "original_LAD_artifacts": "preserved; this proposal does not activate a new model run",
        "report": str(report_path), "limitations": limitations,
    }
    if proposal_path.exists():
        old = json.loads(proposal_path.read_text())
        compare = ["processed_sha256", "count", "min_eval_innings", "pitcher_ids", "selection_rule"]
        if any(old.get(key) != proposal[key] for key in compare):
            raise ValueError("Existing frozen proposal differs; refusing silent replacement")
        proposal["frozen_at_utc"] = old["frozen_at_utc"]
    report = {
        **proposal, "evaluated_new_models": False,
        "selected": [{**r, "starts": r["train"]["starts"], "outs": r["train"]["outs"],
                      "innings": r["train"]["innings"], "uncertain_outs": r["train"]["ambiguous_outs"],
                      "outs_lower_bound": r["train"]["outs_lower_bound"]} for r in chosen],
        "starter_pitchers_any_period": len(records), "eligible_pitchers": len(eligible),
        "innings_notation": "X.Y means X whole innings and Y outs (Y in 0,1,2), not a decimal fraction",
        "candidate_designs": {
            "train_top_with_eval_eligibility": {"recommended": True, "selected": chosen},
            "balanced_retrospective": {
                "recommended": False, "rule": "maximize min(train outs percentile, dev outs percentile), both over eligible pitchers; average ranks / eligible N; ties use train outs, starts, MLBAM ID",
                "selected": balanced[:args.count]},
        },
        "selected_union_samples": {},
        "eligible_ranking_by_train": ranked,
        "all_starters": sorted(records, key=lambda r: (-r["train"]["outs"], r["pitcher"])),
    }
    for split in PERIODS:
        selected_pa = pa.loc[pa.pitcher.isin(proposal["pitcher_ids"]) & pa.split.eq(split)]
        report["selected_union_samples"][split] = {
            "starts": sum(r[split]["starts"] for r in chosen),
            "outs": sum(r[split]["outs"] for r in chosen),
            "innings": ip(sum(r[split]["outs"] for r in chosen)),
            "pas": len(selected_pa), "eligible_pas": int(selected_pa.supported_pa.sum()),
            "unique_batters": int(selected_pa.batter.nunique()),
            "eligible_unique_batters": int(selected_pa.loc[selected_pa.supported_pa, "batter"].nunique()),
        }
    write_json(proposal_path, proposal)
    write_json(report_path, report)
    lines = ["# MLB all-team starter cohort candidates", "", f"Frozen: {proposal['frozen_at_utc']}", "", 
             f"Recommended: top {args.count} by training starter outs among {len(eligible)} pitchers with ≥{args.min_eval_ip} dev IP (conservative lower bound).",
             "Persistent MLBAM IDs follow trades; all teams count. Calibration remains separate. Selection uses no new model metrics.", "",
             "Training: 2023-05-15–2025-04-30. Calibration: 2025-05-01–06-30. Evaluation: 2025-07-01–09-28 (available regular-season data).", ""]
    for name, design in report["candidate_designs"].items():
        lines += [f"## {name}", "", "| Pitcher (MLBAM) | Teams observed | Train starts / IP | Calibration starts / IP | Eval starts / IP | Train PA / eligible / batters | Eval PA / eligible / batters |", "|---|---|---:|---:|---:|---:|---:|"]
        for r in design["selected"]:
            t, c, e = r["train"], r["calibration"], r["dev"]
            lines.append(f"| {r['player_name']} ({r['pitcher']}) | {' → '.join(x['team'] for x in r['team_sequence'])} | {t['starts']} / {t['innings']} | {c['starts']} / {c['innings']} | {e['starts']} / {e['innings']} | {t['pas']} / {t['eligible_pas']} / {t['batters']} | {e['pas']} / {e['eligible_pas']} / {e['batters']} |")
        lines += [""]
    lines += ["## Interpretation", "", *[f"- {item}" for item in limitations], "", "IP is baseball notation; 62.0 = 62 innings, 85.1 = 85 innings + 1 out. Full integer out sums, bounds, all candidate ranks, team changes and unique batters are in the JSON report.", "", f"Recommended membership stable under reconstructed-out uncertainty: {boundary_stable}.", "", "Original LAD manifest, configurations, model and first-result artifacts are unchanged by this script.", ""]
    (root / "reports/mlb_cohort_candidates.md").write_text("\n".join(lines))
    print(json.dumps({"recommended": [{"pitcher": r["pitcher"], "name": r["player_name"], "train_ip": r["train"]["innings"], "eval_ip": r["dev"]["innings"]} for r in chosen], "balanced": [r["player_name"] for r in balanced[:args.count]], "samples": report["selected_union_samples"], "report": str(report_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
