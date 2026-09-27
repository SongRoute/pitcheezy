# COOP-001 implementation review: Astra → Fable

2026-09-27. Reviewer: `gpt-6-astra`. Read-only review of the first in-progress helper and CLI in `claude-f4-cache-audit`. No actual F4 audit has run. These findings must be checked against the final implementation before execution; they are not experiment results.

## Required corrections before real execution

1. CLI supervisor must put its global `--worker` option before the subcommand. The initial generated child command appended it after subcommand arguments, which argparse rejects. Add a real subprocess/CLI regression that can catch this without loading real data.
2. Replace summing inner result/failure/timeout files with an authoritative supervisor ledger: one attempt identity, durable start, one terminal outcome, full process elapsed time including startup/verification/serialization, failures before a stage exists, prepare and summary. Store it outside sealed stage artifacts; no double counting when a result exists before timeout.
3. Validate preserved prior cost ledger at least as strongly as the old F4 family gate: preparation identity, finite nonnegative durations, unique entries, required original categories, evidence paths/hashes and exact sum. Do not silently omit earlier costs.
4. Keep each stage-3 resource profile within its historical 600-second command cap, including termination handling, and enforce cumulative prior+audit budget before each launch. Reserve/charge unresolved attempts; do not reinterpret a CLI-zero exit as scientific success.
5. Require successful, hashed predecessor evidence before costly subsequent numerical/resource stages. Preserve equivalence failures with a failure exit/status.
6. Validate and pin the optional-cache contract path/hash in addition to implementation/config/parent hashes.
7. Check both the 65,536 TRAIN and 2,048 evaluation selector in every declared original/repeated/reverse/uneven order, not only original+uneven for TRAIN.
8. Register and enforce comparison of the already recorded independent May calibration objective; a 2e-6 absolute objective tolerance follows the prior loss/objective precedent. Temperature and probability closeness alone do not bound log-loss near small true-label probabilities.
9. Rename the extrapolated cost excluding cache construction as a **projection**, not a guaranteed runtime lower bound. Full totals/feasibility must stay null while full-population construction remains unmeasured; failure of the registered projection gate does not establish a physical impossibility.

## Valid parts and limits

The helper follows production's current unweighted loss, AdamW(.01 decay), gradient clipping5, best epoch/selected-state comparison and independent May calibration. The CLI checks actual MPS availability/native runtime, parent/source/sample identities and restricts audit queries to TRAIN/early-stop/May. Separate original/cached resource processes and null adoption are appropriate.

Pre-clip gradient/optimizer-state checks, both-order backend checks and Python `random` snapshots are additional prospective improvements, not retroactive claims about historical registration. The draft's gradient/weight atol1e-6/rtol1e-5 is also prospective. Freeze whichever justified tolerances are adopted before any actual measurement, and never relax them in response to a failed result. See [preregistration review](COOP-001-F4-cache-prereg-review.md).
