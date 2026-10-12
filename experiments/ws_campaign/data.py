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
            heldout = spec.get("heldout_pitchers")
            require(isinstance(heldout, list) and heldout and len(heldout) == len(set(heldout)),
                    "fixed heldout pitcher IDs")
            require(spec.get("heldout_selection") == "deterministic_pre_score", "heldout selection rule")
    return plan


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
