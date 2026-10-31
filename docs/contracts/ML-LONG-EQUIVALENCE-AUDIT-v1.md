# Optional F4 MPS numerical equivalence audit v1

2026-09-24. **Source implementation and synthetic CPU tests only; not executed or adopted.** This additive audit depends on the separately reviewed optional inference change in commit `335d878` and the identity-only long-history repair. The original registered unoptimized F4 profiles must run first. Preserve their results or timeout evidence. A resource decision may authorize this distinct audit before any optimized DEV evaluation; CPU equivalence alone does not authorize backend adoption.

## Frozen numerical screen

Fit one seed-0 width-128 profile network for each H0/32/128 arm on exactly 8,192 TRAIN and 2,048 early-stop rows, selected with the original deterministic evenly spaced rule. Use two epochs, patience two, batch 256, learning rate .0005 and the unchanged model training method. Fail if a partition is too small; never silently reduce counts. Save the fitted checkpoint **before calibration**, then use those same live weights for reference `reuse_long=False` and optimized `reuse_long=True` paths. No profile fit is a completed full-screen member.

Use the original 16 evenly spaced May temperature queries and all 400 frozen delivery vectors per query. Compare raw logits and float64 integrated probabilities at fixed identical temperatures .5, 1 and 2.5 (temperature 1 is raw probability). Independently optimize temperature for each path's cached logits with the original float32 objective, bounded SciPy minimizer and [.5,2.5] bounds. Compare fitted temperatures, final float64 probabilities and objective difference. Calibration arrays/labels and both numerical outputs are retained; no DEV quality is read or compared.

The root-approved **absolute tolerances**, with relative tolerance zero, are fixed before execution:

| Quantity | Maximum absolute difference/error |
|---|---:|
| Conditional logits | .00002 |
| Integrated probabilities at a fixed temperature | .000002 |
| Probability row-sum error from 1 | .000001 |
| Independently fitted temperature | .001 |
| Probabilities using each independently fitted temperature | .00002 |
| Independently fitted objective | .000002 |

CPU random-weight fixtures measured zero differences in the earlier screen; these thresholds allow small MPS float32 batching effects. No threshold is automatically relaxed. A flat objective may yield a fitted-temperature failure despite small probability differences; preserve that discrepancy and review it rather than changing the rule after seeing it.

Additional fixed probes use the same 16 rows in reverse order with pitch chunk 7/draw batch 257, and six candidate queries: the first two May query rows crossed with the first two TRAIN vocabulary types plus one explicit unknown-action sentinel. Candidate probes use pitch chunk 3/draw batch 257. The sentinel must use the frozen global fallback tier -1; this is a numerical coverage probe, not a recommended action or policy support expansion. Compare logits and fixed-temperature probabilities, preserve candidate actions and fallback metadata, and require exact delivery-vector/tier correspondence under every query ordering/chunk. Current pitch physics and candidate types remain paired. The main reference and optimized paths use original chunks 16/256.

Each inference call verifies unchanged CPU NumPy/Torch and MPS RNG states and network weights/buffers. Forward hooks count actual long-token, H5 and head rows. For 16 main queries, reference long-token rows must equal `16*400*128`, optimized long-token rows `16*128`, and both H5/head rows `16*400`. Main paths plus reverse-optimized and two six-query candidate paths total 60 query evaluations / 24,000 conditional rows per arm. This is a backend numerical screen on May16 and six candidate probes, **not proof for every possible input, rare event or eventual fitted full model**.

## Registration, provenance and execution

Run only under the common exclusive matrix heavy lock. The audit directory must be a distinct sibling of the original F4 run and outside any `members` directory. `prepare` seals exact config, current source copies, environment identity and original preparation metadata before fitting. It verifies the original frozen artifact/source hashes and permits only the reviewed lazy-model source difference: reference SHA256 `8db3d08d4bfab5539baffb1450290a71800a5ed8482c11b18c9c713c8812515b`, optimized SHA256 `4203aa36099c0fb315b40a541b25d7cd475023025a90d8a557f490cabb86fb78`. Every other inherited source must match. Original preparation must include `batter_dual_stream_v2`.

The exact config schema is below. Replace all paths and SHA placeholders with verified values. Each original profile evidence record must point to a distinct sealed original profile manifest or preserved caller timeout/failure ledger. Completed manifests must be at the exact original `profiles/<cell>/manifest.json` path, bind the original preparation SHA and cell, and verify the original `profile.json` hash. A caller ledger must use the exact schema below and bind the same preparation, cell, profile command and original source hashes, with positive finite elapsed seconds and timeout/failure outcome. An unrelated or copied evidence file cannot satisfy multiple arms. No original failed attempt is overwritten.

```json
{
  "protocol": "ml_long_equivalence_v1",
  "original_f4_run": "/absolute/original-f4-run",
  "original_preparation_sha256": "<64 hex characters>",
  "original_profile_evidence": {
    "F4-H0": {"kind": "completed_profile_manifest", "path": "/absolute/original-f4-run/profiles/F4-H0/manifest.json", "sha256": "<64 hex characters>"},
    "F4-32": {"kind": "caller_failure_ledger", "path": "/absolute/original-32-timeout.json", "sha256": "<64 hex characters>"},
    "F4-128": {"kind": "caller_failure_ledger", "path": "/absolute/original-128-timeout.json", "sha256": "<64 hex characters>"}
  },
  "device": "mps",
  "cells": ["F4-H0", "F4-32", "F4-128"],
  "tolerances": {
    "logits": 0.00002,
    "fixed_probability": 0.000002,
    "probability_sum": 0.000001,
    "fitted_temperature": 0.001,
    "refitted_probability": 0.00002,
    "objective": 0.000002
  },
  "limits": {"train": 8192, "earlystop": 2048, "temperature": 16},
  "fixed_temperatures": [0.5, 1.0, 2.5],
  "cell_seconds": 1200
}
```

For a failed original profile, preserve its caller outcome in this exact schema (one distinct ledger per arm), derived from the actual process outcome and registered original source identity:

```json
{
  "protocol": "ml_long_profile_caller_outcome_v1",
  "original_f4_run": "/absolute/original-f4-run",
  "preparation_sha256": "<original preparation SHA256>",
  "cell": "F4-32",
  "command": "profile",
  "outcome": "timeout",
  "elapsed_seconds": 600.1,
  "source_hashes": {"<each original source path>": "<its registered SHA256>"}
}
```

`outcome` may be `timeout` or `failure`; the elapsed value above is illustrative. Do not invent a failed attempt or elapsed time to satisfy this gate. The loader may hold the full processed source frame to build legal as-of history links, but it never reads DEV/blend query lists, gathers their features, or reads their quality scores.

The 1,200-second per-arm cap covers load/fit/numerical work cooperatively; the sole caller must impose the same hard subprocess timeout, preserve elapsed time/partial files after external termination, and stop on cap failure. No source adoption or count reduction occurs automatically. A changed budget requires a fresh explicit source/config review and registration. The maximum three-arm audit allocation is 3,600 seconds, separately accounted from original profiles/full fits; registration/hash overhead is also reported by the caller's process ledger.

Using the original registered local config and native-library/Python environment:

```sh
python experiments/pitchmdp/scripts/audit_ml_long_equivalence.py --config "$AUDIT_CONFIG" --local-config "$LOCAL_CONFIG" --output "$AUDIT_RUN" prepare
python experiments/pitchmdp/scripts/audit_ml_long_equivalence.py --config "$AUDIT_CONFIG" --local-config "$LOCAL_CONFIG" --output "$AUDIT_RUN" cell --cell F4-H0
python experiments/pitchmdp/scripts/audit_ml_long_equivalence.py --config "$AUDIT_CONFIG" --local-config "$LOCAL_CONFIG" --output "$AUDIT_RUN" cell --cell F4-32
python experiments/pitchmdp/scripts/audit_ml_long_equivalence.py --config "$AUDIT_CONFIG" --local-config "$LOCAL_CONFIG" --output "$AUDIT_RUN" cell --cell F4-128
python experiments/pitchmdp/scripts/audit_ml_long_equivalence.py --config "$AUDIT_CONFIG" --local-config "$LOCAL_CONFIG" --output "$AUDIT_RUN" summarize
```

Each arm preserves sample keys/hashes, shared checkpoint, original source/input dependencies, logits, raw/fixed/refitted probabilities, fitted temperatures, candidate/order arrays, pool/tier hashes, counts, synchronized wall times, process peak RSS and MPS allocator snapshots. RSS is process-lifetime peak, not isolated allocation peak. Timings use a fixed reference-first order without an independent warm-up/crossover benchmark; cache/warm-up effects limit speedup interpretation. All three sealed arms must complete before the family summary reports `all_three_equivalent`; an incomplete or failed arm cannot count as a pass. A pass still leaves adoption null. Root review and a fresh optimized preparation/resource profile are required before any optimized full experiment, with original attempts preserved. No G/P3 sources are modified.
