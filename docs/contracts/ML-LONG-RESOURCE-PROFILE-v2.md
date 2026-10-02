# F4 resource-profile amendment v2

2026-09-24. **Source-only amendment, held from integration until the optional MPS equivalence proof has completed.** Do not modify the original frozen source/run while the proof compares it. This amendment requires a fresh F4 attempt/preparation and `ml_long_history_v2` configuration; neither original preparation nor its profiles can be resumed under these sources. No revised real resource measurement or model-quality result has been observed by the implementation agent.

**Integration update:** all three actual MPS numerical screens completed under the original source identity, and independent review passed17 synthetic tests. Root adopted long-encoding reuse and this amendment for fresh `configs/EXP-P4-002-v3.yaml` only. Original attempts remain immutable; fresh resource gates are still required before full fitting. The numerical screen is bounded to the registered probes, not every possible input.

The sole owner's original v2-identity-history preparation is `24394218e89ed3bb2d4e0e032ae1061b37f7f79d8466b0ff34a9e55e331d72b5`. All three original profiles completed and failed the 7,200-second projection gate. Their 8,192-row/two-epoch cold fits took about 3.65–3.89 seconds, projecting approximately 8,380–8,935 fit seconds, 136 calibration seconds and 872–899 prediction seconds per full member. Those are resource measurements, not quality findings. Fit projection dominates; inference reuse alone cannot make the original projection affordable. Preserve these measurements and costs.

## Fixed revised measurement

For every H0/32/128 arm, use the same frozen parent partitions, normalizer/context, chronological evenly spaced sample rule, width 128, batch 256, learning rate .0005 and exact 400 delivery draws. Insufficient samples cause failure; do not shrink them.

1. Construct a fresh seed-0 model. Warm it on 8,192 TRAIN rows for exactly one epoch/32 optimizer updates. The unchanged training method requires an epoch-end evaluation, so use an evenly spaced 2,048-row subset of those same TRAIN rows for this discarded warmup's evaluation. No real early-stop, May or DEV rows enter warmup. Synchronize and record the complete cold warmup cost, then discard its model/optimizer and collect garbage.
2. Construct a separate fresh seed-0 model and optimizer. Measure 65,536 TRAIN rows for exactly four epochs, patience four and 1,024 optimizer updates, with the fixed 2,048-row actual early-stop sample. No warmup weights, optimizer state or checkpoint is reused. The model's training implementation and initialization are unchanged.
3. Measure calibration on the same 16 May temperature queries with all 400 draws, followed by prediction on those same 16 queries. Do not gather DEV/blend inputs or compute their quality. Their frozen row counts are used only to project the later workload. The source-pinned inference implementation determines whether separately approved reuse is present; this resource amendment does not itself approve it.

Report load, warmup, measured-fit, calibration and inference times; sample counts and ordered key hashes; actual epochs/updates; width/parameter count and maximum expanded minibatch rows; process peak RSS and per-stage MPS allocator snapshots. Peak RSS is a process-lifetime high-water mark, not isolated allocation peak. Omit training/early-stop/calibration quality scores from the resource report. Profile weights are discarded and never used as full-screen fits.

## Conservative projection and gates

Let `U_full = 30 * ceil(full_TRAIN_rows / 256)` and `U_profile = 1024`. The projected fit cost is

`cold_warmup_seconds + measured_four_epoch_fit_seconds / U_profile * U_full`.

The measured fit time includes all four early-stop passes; do not subtract them. Before projecting, verify that full `ceil(early_rows/256)/ceil(train_rows/256)` is no larger than the profile's `8/256`. The actual frozen full population of 16,000 early-stop / 1,252,824 TRAIN rows satisfies this conservative validation-work ratio. A different population that violates it requires explicit replanning, not a falsely conservative projection. No early-stop savings or reduced epoch count are assumed.

Add May calibration time scaled from 16 queries, June+DEV inference time scaled from 16 queries, and **two measured data loads per member** because `fit` and `predict` each load the data. Add the cold warmup term separately for every member; do not amortize it across nine independently launched processes. Each arm's whole-member projection must be at most 7,200 seconds. All three versioned profiles must be complete, have matching preparation/artifact identities, use identical warmup/measured sample hashes, and pass before any full fit or prediction begins. The runner recomputes each saved projection from recorded costs/counts and the frozen formula.

The family forecast is `3 * sum(H0_member_seconds, H32_member_seconds, H128_member_seconds)`. Add actual prior preparation, profiles, cold/failed attempts and optional-equivalence costs from an owner-maintained evidence-backed ledger. The total must fit the original **28,800-second family cap**. Root freezes the exact ledger, per-arm projections and budget only after the revised profiles. If they do not fit, explicitly replan before fitting. No automatic change to full data, batch 256, maximum 30 epochs, patience five, seeds 0/1/2, three arms, 400 draws or statistical comparisons is permitted.

These are projections, not runtime guarantees. The sole owner must enforce a 600-second hard subprocess cap for each profile, the remaining portion of the combined 7,200-second fit+prediction member cap, and an authoritative cumulative family ledger including failures. The frozen prior-cost ledger is a registration snapshot; record later actual costs separately rather than rewriting its hashed file.

## Exact registration

The existing F4 config changes `protocol` to `ml_long_history_v2` and adds the exact `profile` object below. Other parent hashes, samples, seeds, full `budget` and model fields remain fixed. Source identity makes a fresh preparation mandatory.

```json
{
  "version": "f4_resource_profile_v2",
  "warmup_train": 8192,
  "warmup_epochs": 1,
  "warmup_eval_train": 2048,
  "measured_train": 65536,
  "measured_epochs": 4,
  "measured_patience": 4,
  "earlystop": 2048,
  "temperature": 16,
  "seed": 0,
  "width": 128,
  "batch_size": 256,
  "learning_rate": 0.0005,
  "draws": 400,
  "full_epochs": 30,
  "sample_rule": "chronological_linspace_no_shrink",
  "projection": "cold_once_plus_all_measured_fit_cost_per_update_times_full30_updates",
  "data_loads_per_member": 2,
  "profile_seconds": 600,
  "member_seconds": 7200
}
```

Before any full `fit` or `predict`, supply `--family-budget /absolute/frozen-family-budget.json` with this exact schema. Placeholder values below must come from actual completed profiles and the owner's elapsed-time ledger, never invented estimates.

```json
{
  "protocol": "ml_long_family_budget_v2",
  "preparation_sha256": "<fresh preparation SHA256>",
  "profile_manifest_sha256": {
    "F4-H0": "<fresh H0 profile manifest SHA256>",
    "F4-32": "<fresh 32 profile manifest SHA256>",
    "F4-128": "<fresh 128 profile manifest SHA256>"
  },
  "owner_budget_ledger": {"path": "/absolute/prior-cost-ledger.json", "sha256": "<ledger SHA256>"},
  "prior_seconds": "REPLACE WITH POSITIVE ACTUAL NUMERIC TOTAL",
  "family_seconds": 28800
}
```

The referenced prior-cost ledger has exactly `protocol: "ml_long_owner_budget_ledger_v1"`, the same `preparation_sha256`, an `entries` list and numeric `elapsed_seconds_total`. Each entry has exactly `category`, a unique nonempty `attempt`, finite nonnegative numeric `seconds`, and a nonempty `evidence` list of `{ "path": "/absolute/process-evidence", "sha256": "<hash>" }`. All four categories—`preparation`, `profiles`, `cold_failed_attempts`, `equivalence`—must be represented. Evidence can include preserved process logs, timeout records and completed profile/audit manifests; retain real elapsed times including failed attempts. The entry sum must match both the ledger total and positive registered `prior_seconds` within one microsecond. Zero for a category is permitted only when its evidence supports no applicable spent time; no category may silently disappear.

The runner verifies all hashes and the all-three gates before loading full-member data or models, then writes immutable `family_budget.json` and `family_budget_projection.json`. Later calls must match them exactly. Missing, over-budget or changed artifacts abort. A partial budget freeze is preserved for review. This amendment changes resource estimation and execution gates only; the original scientific experiment and no-DEV-selection rules remain intact.
