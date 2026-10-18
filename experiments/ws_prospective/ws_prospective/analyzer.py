"""Single sealed performance look + operational metadata checks.

``ops_check`` (allowed during the Series) reads only ledger metadata: counts
of intents/forecasts by status and provisional timing, compute latency,
missing results, resource events.  It never opens the store's pitch bodies,
never builds labels and never computes a metric.

``sealed_look`` refuses unless ``seal.check_unsealed`` passes, then creates the
single-look marker *before* reading any label, builds labels from the store at
the registered revision, applies the registered eligibility rules and computes
the registered paired comparisons.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import metrics as M
from .ledger import read_rows
from .seal import check_unsealed, file_sha256, take_look

LabelFn = Callable[[dict], Optional[str]]   # playEvent -> leaf name or None
TypeFn = Callable[[dict], Optional[str]]    # playEvent -> realized pitch type code


def ops_check(ledger_path: Path) -> dict:
    intents = read_rows(ledger_path, "intents")
    forecasts = {r["intent_id"]: r for r in read_rows(ledger_path, "forecasts")}
    events = read_rows(ledger_path, "events")
    sessions = read_rows(ledger_path, "sessions")
    by = Counter()
    lat: Dict[str, List[float]] = defaultdict(list)
    for it in intents:
        f = forecasts.get(it["intent_id"])
        if f is None:
            by[(it["model_id"], "INCOMPLETE_INTENT", "-")] += 1
            continue
        by[(it["model_id"], f["status"], f["provisional_timing"])] += 1
        lat[it["model_id"]].append(f["compute_ns"] / 1e6)
    return {
        "kind": "OPS_METADATA_ONLY_NO_LABELS_NO_METRICS",
        "sessions": len(sessions),
        "counts": [{"model_id": k[0], "status": k[1], "timing": k[2], "n": v}
                   for k, v in sorted(by.items())],
        "compute_ms": {m: {"n": len(v), "p50": float(np.median(v)), "max": float(np.max(v))}
                       for m, v in lat.items() if v},
        "events": dict(Counter(e["kind"] for e in events)),
    }


def _final_sequence(label_rows: Dict[str, dict]) -> List[tuple]:
    """[(slot_key, pitch_uid, row)] in (at_bat_number, pitch_number) order."""
    ordered = sorted(label_rows.items(),
                     key=lambda kv: (kv[1]["at_bat_number"], kv[1]["pitch_number"]))
    out, prev = [], None
    for uid, row in ordered:
        out.append((f"after:{prev}" if prev else "game_start", uid, row))
        prev = uid
    return out


def grade_game(ledger_path, label_rows: Dict[str, dict], *, leaf_of: LabelFn,
               type_of: TypeFn, leaf_order: List[str], verified_clock: str = "observer_local"):
    """Return ({model_id: {slot_key: (probs_vector, y)}}, exclusion counter).

    ``ledger_path`` may be one path or a list (one ledger per runner process);
    a model_id may appear in only one of them.
    """
    paths = [ledger_path] if isinstance(ledger_path, (str, Path)) else list(ledger_path)
    intents, forecasts, owner = [], {}, {}
    for lp in paths:
        rows = read_rows(lp, "intents")
        for it in rows:
            if owner.setdefault(it["model_id"], str(lp)) != str(lp):
                raise ValueError(f"model {it['model_id']} appears in two ledgers")
        intents += rows
        forecasts.update({r["intent_id"]: r for r in read_rows(lp, "forecasts")})
    by_slot = defaultdict(lambda: defaultdict(list))
    for it in intents:
        by_slot[it["slot_key"]][it["model_id"]].append(it)
    models = sorted({it["model_id"] for it in intents})
    graded: Dict[str, Dict[str, tuple]] = {m: {} for m in models}
    excl = Counter()
    for key, uid, row in _final_sequence(label_rows):
        ev = row["event"]
        leaf, ptype = leaf_of(ev), type_of(ev)
        for m in models:
            cands = by_slot.get(key, {}).get(m, [])
            match = [c for c in cands if (c["target_at_bat_number"], c["target_pitch_number"])
                     == (row["at_bat_number"], row["pitch_number"])]
            it = match[0] if match else (cands[0] if cands else None)
            reason = None
            f = forecasts.get(it["intent_id"]) if it else None
            if it is None:
                reason = "NO_FORECAST"
            elif f is None:
                reason = "INCOMPLETE_INTENT"
            elif f["status"] != "OK":
                reason = f"MODEL_{f['status']}"
            elif (it["target_at_bat_number"], it["target_pitch_number"]) != (
                    row["at_bat_number"], row["pitch_number"]):
                reason = "TARGET_KEY_MISMATCH"
            elif f["provisional_timing"] != "PRE_RECEIPT":
                reason = "POST_HOC_REPLAY"
            elif row.get("receipt_clock") != verified_clock or row.get("first_received_ns") is None:
                reason = "RECEIPT_CLOCK_UNVERIFIED"
            elif not f["generated_wall_ns"] < row["first_received_ns"]:
                reason = "GENERATED_AFTER_FIRST_RECEIPT"
            elif json.loads(f["leaf_order"]) != list(leaf_order):
                reason = "LEAF_ORDER_MISMATCH"
            elif leaf is None or leaf not in leaf_order:
                reason = "LABEL_UNMAPPED"
            else:
                probs = json.loads(f["probs_json"])
                vec = probs.get(ptype) if ptype in probs else probs.get("*")
                if vec is None:
                    reason = "REALIZED_TYPE_NOT_IN_CANDIDATES"
                else:
                    graded[m][key] = (np.asarray(vec, float), leaf_order.index(leaf))
            excl[(m, reason or "ELIGIBLE")] += 1
    return graded, excl


def sealed_look(registration_path: Path, job_dir: Path, games: Dict[int, dict], *,
                leaf_of: LabelFn, type_of: TypeFn, label_loader, now_utc=None) -> dict:
    """games: {game_pk: {"ledger": path, "store": path}}; label_loader(store, pk, revision=...)."""
    reg = check_unsealed(registration_path, job_dir, now_utc=now_utc)
    seal = reg["analysis"]["seal"]
    take_look(job_dir, seal["look_marker"],
              {"registration_sha256": reg["_sha256"], "games": sorted(games),
               "ledger_sha256": {str(k): {str(lp): file_sha256(lp) for lp in v["ledgers"]}
                                 for k, v in games.items()},
               "store_sha256": {str(k): file_sha256(v["store"]) for k, v in games.items()}})
    leaf_order = list(reg["targets"]["leaf_order"])
    an = reg["analysis"]
    result = {"registration_sha256": reg["_sha256"], "revisions": {}}
    for revision in [an["label_revision_primary"], *an.get("label_revision_sensitivity", [])]:
        pooled: Dict[str, Dict[tuple, tuple]] = defaultdict(dict)
        excl = Counter()
        for pk, g in sorted(games.items()):
            graded, ex = grade_game(g["ledgers"], label_loader(g["store"], pk, revision=revision),
                                    leaf_of=leaf_of, type_of=type_of, leaf_order=leaf_order)
            excl.update(ex)
            for m, slots in graded.items():
                for k, v in slots.items():
                    pooled[m][(pk, k)] = v
        comps = []
        for c in an["comparisons"]:
            a, b = pooled.get(c["candidate"], {}), pooled.get(c["comparator"], {})
            keys = sorted(set(a) & set(b))
            entry = {"candidate": c["candidate"], "comparator": c["comparator"],
                     "role": c.get("role", "primary"), "n_paired": len(keys)}
            if len(keys) >= an.get("min_paired_pitches", 1) and len({k[0] for k in keys}) >= 1:
                y = np.array([a[k][1] for k in keys])
                PA = np.vstack([a[k][0] for k in keys])
                PB = np.vstack([b[k][0] for k in keys])
                M.validate_probs(PA)
                M.validate_probs(PB)
                entry["metrics"] = M.cluster_bootstrap_paired(
                    PA, PB, y, [k[0] for k in keys], seed=an["bootstrap"]["seed"],
                    B=an["bootstrap"]["B"])
            else:
                entry["metrics"] = "INSUFFICIENT_PAIRED_PITCHES"
            comps.append(entry)
        result["revisions"][revision] = {
            "exclusions": [{"model_id": k[0], "reason": k[1], "n": v}
                           for k, v in sorted(excl.items())],
            "comparisons": comps}
    out = Path(job_dir) / an.get("result_file", "RESULT.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
