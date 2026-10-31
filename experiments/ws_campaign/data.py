"""C split and history contracts; pure metadata operations, never scores."""
from datetime import date
import hashlib
import json

import numpy as np


class DataStop(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise DataStop("STOP " + reason)


def parse_day(value):
    require(isinstance(value, str), "date must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise DataStop("STOP invalid date") from exc
    require(parsed.isoformat() == value, "noncanonical date")
    return parsed


def validate_plan(plan):
    """Registration must fix all boundaries and heldout IDs before any payload access."""
    require(plan.get("data_role") == "2026 EXPOSED DEVELOPMENT", "data role")
    require(plan.get("source_sha256") and plan.get("exposure_ledger_sha256")
            and plan.get("codec_source_sha256"), "source and exposure bindings")
    require(set(plan.get("windows", {})) == {"chrono1", "chrono2", "pitcher_holdout"}, "C windows")
    for name, spec in plan["windows"].items():
        roles = spec.get("roles", {})
        require(set(roles) == {"FIT", "SELECT", "CAL", "EVAL"}, "four roles in " + name)
        spans = [(parse_day(roles[r][0]), parse_day(roles[r][1])) for r in ("FIT", "SELECT", "CAL", "EVAL")]
        require(all(start <= end for start, end in spans)
                and all(spans[i][1] < spans[i + 1][0] for i in range(3)), "chronological disjoint roles")
        require(spec.get("codec_fit_end") and parse_day(spec["codec_fit_end"]) <= spans[0][1],
                "codec fit cutoff")
        if name.startswith("chrono"):
            require(spec.get("codec_mode") == "frozen_pre_fit" and
                    parse_day(spec["codec_fit_end"]) < spans[1][0], "frozen codec precedes SELECT")
        else:
            require(spec.get("codec_mode") == "refit_excluding_heldout", "heldout-free codec")
            rule = spec.get("heldout_selection")
            require(rule == {"method": "sha256_mod", "namespace": "EXP-WS-CAMPAIGN-001",
                             "modulus": 5, "remainder": 0}, "fixed heldout selection rule")
    return plan


def select_heldout(pitcher_ids, rule):
    """ID-only deterministic partition; call before label/score access."""
    require(rule == {"method": "sha256_mod", "namespace": "EXP-WS-CAMPAIGN-001",
                     "modulus": 5, "remainder": 0}, "heldout selection rule")
    ids = []
    for value in pitcher_ids:
        require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                and value >= 0, "pitcher ID integer domain")
        ids.append(str(int(value)))
    ids = sorted(set(ids))
    require(ids, "empty pitcher inventory")
    heldout = [int(v) for v in ids if int(hashlib.sha256((rule["namespace"] + ":" + v).encode()).hexdigest(), 16)
               % rule["modulus"] == rule["remainder"]]
    require(heldout and len(heldout) < len(ids), "degenerate heldout selection")
    return heldout


def role_masks(days, pitchers, spec, *, heldout=False):
    """An assigned row has exactly one role; holdout pitchers enter EVAL only."""
    days = np.asarray(days)
    pitchers = np.asarray(pitchers)
    require(days.ndim == pitchers.ndim == 1 and len(days) == len(pitchers), "row metadata shape")
    parsed = np.array([parse_day(str(v)) for v in days], dtype=object)
    masks = {}
    for role, (lo, hi) in spec["roles"].items():
        masks[role] = (parsed >= parse_day(lo)) & (parsed <= parse_day(hi))
    if heldout:
        held = np.isin(pitchers, spec["heldout_pitchers"])
        for role in ("FIT", "SELECT", "CAL"):
            masks[role] &= ~held
        masks["EVAL"] &= held
        require(not any(np.any(masks[r] & held) for r in ("FIT", "SELECT", "CAL")),
                "heldout in model fitting or selection")
    require(np.max(sum(mask.astype(int) for mask in masks.values()), initial=0) <= 1,
            "overlapping data roles")
    require(all(np.any(mask) for mask in masks.values()), "empty role")
    return masks


def check_prequential(query_day, donor_day, donor_pitcher, heldout_pitchers=()):
    """The source feature builder must expose every donor-to-query edge for this audit."""
    query_day = np.asarray(query_day)
    donor_day = np.asarray(donor_day)
    donor_pitcher = np.asarray(donor_pitcher)
    require(query_day.shape == donor_day.shape == donor_pitcher.shape, "donor edge shape")
    require(all(parse_day(str(d)) < parse_day(str(q)) for q, d in zip(query_day, donor_day)),
            "same-day/future donor")
    require(not np.isin(donor_pitcher, heldout_pitchers).any(), "heldout donor in feature history")
    return {"edges": int(len(query_day)), "strictly_prior": True, "heldout_free": True}


def expanded_history_columns(raw_names, continuous, knots, history_names):
    """Map registered 91 raw feature names through frozen hinge expansion to 199 indices."""
    raw_names = list(raw_names)
    continuous = [int(i) for i in continuous]
    history_names = set(history_names)
    require(len(raw_names) == 91 and len(set(raw_names)) == 91, "raw feature schema")
    require(history_names and history_names <= set(raw_names), "registered history names")
    require(len(continuous) == len(set(continuous)) and all(0 <= i < 91 for i in continuous),
            "encoder continuous columns")
    require(91 + len(continuous) * len(knots) == 199, "encoded dimension")
    indices = [i for i, name in enumerate(raw_names) if name in history_names]
    width = len(continuous)
    for k in range(len(knots)):
        indices.extend(91 + k * width + j for j, raw_i in enumerate(continuous)
                       if raw_names[raw_i] in history_names)
    digest = hashlib.sha256(json.dumps({"raw_names": raw_names, "continuous": continuous,
                                         "knots": list(knots), "history_names": sorted(history_names),
                                         "indices": indices}, sort_keys=True).encode()).hexdigest()
    return {"indices": indices, "sha256": digest, "raw_names": sorted(history_names)}


def fit_prior_support(days, pitchers, games, at_bat, pitch_no, fit):
    """Count only earlier FIT pitches: prior dates, plus earlier pitches in the same game.

    Different games on one date have no registered start-time ordering and do not
    contribute to each other's support. This conservative tie rule is fixed here.
    """
    arrays = [np.asarray(v) for v in (days, pitchers, games, at_bat, pitch_no, fit)]
    require(all(a.ndim == 1 and len(a) == len(arrays[0]) for a in arrays), "support row shape")
    n = len(arrays[0]); require(n > 0, "empty support population")
    days, pitchers, games, at_bat, pitch_no, fit = arrays
    require(fit.dtype == np.bool_, "FIT mask type")
    parsed = np.array([parse_day(str(v)).toordinal() for v in days], dtype=np.int64)
    result = np.zeros(n, dtype=np.int64)
    order = np.lexsort((pitch_no, at_bat, games, parsed, pitchers))
    last_pitcher = None; last_day = None; last_game = None
    previous_dates = 0; today = 0; game_seen = 0
    for i in order:
        pitcher, day, game = int(pitchers[i]), int(parsed[i]), int(games[i])
        if pitcher != last_pitcher:
            previous_dates = today = game_seen = 0
        elif day != last_day:
            previous_dates += today; today = game_seen = 0
        elif game != last_game:
            game_seen = 0
        result[i] = previous_dates + game_seen
        if fit[i]:
            today += 1; game_seen += 1
        last_pitcher, last_day, last_game = pitcher, day, game
    return result


OWN_HISTORY_COLUMNS = tuple(range(50, 59)) + tuple(range(66, 78))


def splice_heldout_own_history(clean, enriched, eval_mask, fit_mask):
    """Only pitcher-owned features may cross from the EVAL-only history pass."""
    clean = np.asarray(clean)
    enriched = np.asarray(enriched)
    eval_mask = np.asarray(eval_mask, bool)
    fit_mask = np.asarray(fit_mask, bool)
    require(clean.ndim == enriched.ndim == 2 and clean.shape[1] == enriched.shape[1] == 91
            and len(eval_mask) == len(fit_mask) == len(clean)
            and len(enriched) == int(eval_mask.sum()), "own-history splice shape")
    require(not (eval_mask & fit_mask).any(), "EVAL/FIT overlap")
    require(np.all(clean[eval_mask, 90] == 1), "heldout clean-pool similarity abstention")
    before = clean[fit_mask].copy()
    clean[np.ix_(np.flatnonzero(eval_mask), OWN_HISTORY_COLUMNS)] = enriched[:, OWN_HISTORY_COLUMNS]
    require(np.array_equal(before, clean[fit_mask], equal_nan=True), "heldout pass changed FIT features")
    require(np.all(clean[eval_mask, 90] == 1), "heldout similarity abstention changed")
    return clean
