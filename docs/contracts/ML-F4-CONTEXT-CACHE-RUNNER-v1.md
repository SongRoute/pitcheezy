# F4 observed-context cache audit runner (COOP-001)

2026-09-27. Implementation contract for the real-data audit required by `docs/contracts/ML-F4-CONTEXT-CACHE-OPTIONAL-v1.md`. Implemented by Claude Code; reviewed by Astra (`docs/reviews/COOP-001-Fable-code-review.md`), frozen and executed by the Codex coordinator. **At registration time, no audit has yet been executed; subsequent results are recorded in `docs/reports/AI-collaboration-2026-09-27.md`.** The cache stays unadopted. Draft config: `configs/EXP-P4-002-v3-cache-audit-v2.yaml`. The historical v1 registration `configs/EXP-P4-002-v3-cache-audit.yaml` is preserved and pinned by hash.

## Owned files

- `experiments/pitchmdp/scripts/audit_ml_context_cache.py` — CLI runner (supervisor + worker).
- `experiments/pitchmdp/pitchmdp/matrix_context_cache_audit.py` — backend-agnostic stage helpers and owner-ledger validator.
- `experiments/pitchmdp/tests/test_context_cache_audit.py` — synthetic CPU regression tests.
- This contract and the v2 config. Frozen scientific/runner sources are imported as building blocks and unchanged.

## CLI

```
audit_ml_context_cache.py --config C --local-config L --output O prepare --attempt N
audit_ml_context_cache.py --config C --local-config L --output O stage1 --arm {F4-H0,F4-32,F4-128} --attempt N
audit_ml_context_cache.py --config C --local-config L --output O stage2 --arm A --attempt N
audit_ml_context_cache.py --config C --local-config L --output O stage3 --arm A --path {original,cached} --attempt N
audit_ml_context_cache.py --config C --local-config L --output O compare --arm A --attempt N
audit_ml_context_cache.py --config C --local-config L --output O summary --attempt N
```

Real run order (root-owned, single heavy job at a time): `prepare`; per arm `stage1`, `stage2`, `stage3 --path original`, `stage3 --path cached`, `compare`; then `summary`. Seventeen commands.

### Supervisor and wall ledger

Every invocation without `--worker` is a supervisor. It re-launches itself as a worker with `--worker` placed **before** the subcommand (argparse rejects it afterwards; regression-tested on the generated argv). Before launch, under a file lock on the registered `limits.wall_ledger_dir` (must be under `artifact_root` and disjoint from the audit output), it:

1. validates the owner prior-cost ledger (below) and reads the wall ledger;
2. refuses to launch if any prior job has no terminal record (an unresolved job reserves its full cap until root writes its end record), if the same command/arm/path/attempt was already launched (new attempt number only; no metadata overwrite), or if `prior + charged audit wall + this command's cap > 28800`;
3. writes a durable start record `jobs/<job_id>.json` (command, arm, path, attempt, cap, grace, argv, budget snapshot, launch time).

It then launches the worker, waiting only for the cap remaining after preflight and reserved termination grace. SIGTERM and SIGINT to the supervisor become catchable interruptions. On timeout or supervision interruption it sends SIGTERM, waits up to half of one remaining grace interval, then SIGKILL and uses the rest of that same interval for a final bounded reap attempt. One terminal record `ends/<job_id>.json` is written exactly once only when a launched child has been reaped (or launch never produced a child), with the outcome (`completed`, `not_equivalent`, `failed`, `timeout`), exit code and the full caller wall from entry to the supervisor, including prior-ledger validation and startup. If child death cannot be established, the start remains unresolved and reserves its full cap. Exit code 124 for timeouts, 3 for a sealed but non-equivalent verdict, the worker's code otherwise. The ledger lives outside sealed stage artifacts; `results.json`/`failure.json` inside attempts are evidence, never cost sources. Ledger enumeration excludes only AppleDouble files verified by prefix and binary magic; a malformed ordinary JSON record remains an error.

The summary charges every job at its ended wall, and every unresolved job (including the summary itself while it runs) at its full cap. This is conservative by construction; the summary's own final wall is written by its supervisor after the summary artifacts are sealed.

### Caps

`limits.command_seconds` per command (each ≤ 7200); `stage3` must be ≤ the historical 600-second profile cap (`profile_command_cap_seconds`), termination grace included, because a 7200-second member limit never authorizes a 7200-second profile. Grace must be smaller than every cap. Actual wall is recorded even when OS overhead exceeds the cap, and the ledger marks `within_cap` truthfully.

## Identity and refusal rules

- `config_check(real=True)` rejects any null pin (`repo_commit`, both contract digests, historical registration, parent config/preparation/profile digests, source digests, sample row hashes, owner ledger). The draft pins the optional-cache contract and the owner ledger now; `repo_commit`, `sources` and the runner contract digest stay null until the root freeze at the final implementation commit.
- Worker start: `check_location`, `validate_native_runtime`, actual MPS, `verify_pins` (sources, both contracts, historical config, `git rev-parse HEAD == repo_commit`), then the shared `.heavy.lock`.
- Registration identity includes config, local config, environment, all source hashes, both contract hashes and `repo_commit`. `prepare` copies sources and both contracts into the output and seals `registration_manifest.json`.
- `verify_parent`: parent config file hash, `preparation.json` hash and artifact hashes, `batter_dual_stream_v2`, closed DEV scores, each v3 pinned source unchanged in the parent copy and the working tree, each profile manifest/profile hash, identity, long length, and profile-recorded TRAIN/early-stop/May/warmup-evaluation sample hashes equal to the registered selectors.
- Owner prior-cost ledger (`ml_long_owner_budget_ledger_v2`): pinned path+hash; `preparation_sha256` equal to the parent pin; exactly the categories preparation, profiles, cold_failed_attempts, equivalence, cache_audit; unique nonempty attempts; finite nonnegative seconds; nonempty evidence with verified path hashes; exact total (and category totals/entry count if present).
- Selectors are recomputed as in the v3 profile and their ordered key hashes must match before any work. Only TRAIN/early-stop/May batches are loaded.
- Stage attempts `O/<stage>/<arm>[/<path>]/attempt<N>` refuse non-AppleDouble content; `failure.json` is preserved and a new attempt is required. AppleDouble detection uses `matrix_policy_artifacts.is_appledouble`.

### Predecessors

Each stage identity carries `dependencies` = manifest digests of its required predecessors, so a tampered or re-sealed predecessor invalidates every successor:

- `stage2` requires one sealed, **passed** `stage1` of the same arm.
- `stage3` (either path) requires passed `stage1` and `stage2` of the same arm.
- `compare` requires both sealed `stage3` paths bound to that chain.
- `summary` requires all three sealed, **passed** comparisons plus their chains.

A sealed result whose verdict fails is preserved and the command exits 3; it is never a predecessor.

## Stages

1. `stage1`: build `FrozenObservedContext` over TRAIN ∪ evaluation rows (chunk 8192); bitwise comparison of cached vs original context on **both** the 65,536 TRAIN and 2,048 evaluation samples in original, repeated, reverse and uneven-chunk orders, consumed in bounded chunks with per-sample row hashes and chunk digest chains; bitwise comparison of all five `LazyPitchBatch.gather` outputs under candidate pitch-type and current-physics overrides; count change, style change and undeclared row must raise. Timing: construction, context-only transform/lookup and whole lazy gather over the evaluation rows in 256-row minibatches, each path warmed once, then alternating original/cached order. RNG state must be unchanged by construction.
2. `stage2`: `torch.manual_seed(0)`, one `DualStreamNetwork` width128 deep-copied for both paths, the same first 8×256 minibatches of the seed-0 permutation `LazyMatrixModel.fit` would use, fresh AdamW per path. Per step: bitwise input context, forward logits, loss, all parameter gradients and updated weights under the registered tolerances; numpy/torch-cpu/torch-mps RNG state unchanged by each step.
3. `stage3 --path original|cached`: v3-style resource fit per path in its own process: discarded 8192-row/1-epoch TRAIN warmup on the evaluation rows, fresh seed-0 width128 model, 65,536×4 epochs/patience 4/batch 256/lr .0005, May16×400 calibration and inference. Records sync-aware timings, cache construction (cached path: warmup ∪ evaluation ∪ TRAIN ∪ early-stop ∪ May rows), peak RSS and MPS allocator snapshots. `compare` checks selected state, history, best epoch, update counts, delivery temperature, the independently recorded May calibration objective, calibrated/raw probabilities, sums and tiers.
4. `summary`: all-arm pass flags plus `cost_summary`: original v3 profile projection, this audit's original and cached projections via the frozen `resource_projection` formula, measured cache construction for audit rows, owner prior ledger and the wall ledger. Reported as `family_projection_excluding_cache_construction_seconds` with gate `family_projection_gate_excluding_cache_construction`: a frozen-formula projection plus ledgers, **not** a guaranteed or minimum physical runtime. Full-population cache construction is `unmeasured`; per-member cached totals and family feasibility stay `null`; `decision` is always `not_adopted`.

## Registered tolerances (not relaxable after execution)

| Quantity | atol | rtol |
|---|---|---|
| stage1 context, gather outputs, stage2 input context, initial weights | bitwise | — |
| stage2 forward logits | 2e-5 | 0 |
| stage2 loss | 2e-6 | 0 |
| stage2 gradients | 1e-6 | 1e-5 |
| stage2 updated weights | 1e-6 | 1e-5 |
| stage3 selected state | 1e-6 | 1e-5 |
| stage3 history NLL | 2e-6 | 0 |
| stage3 delivery temperature | 1e-3 | 0 |
| stage3 probabilities | 2e-5 | 0 |
| stage3 probability sum | 1e-6 | 0 |
| stage3 calibration objective (integrated log loss) | 2e-6 | 0 |

Logit/probability/temperature/objective values follow the `ml_long_equivalence_v1` precedent; gradient/weight tolerances are prospective fixed backend values. Inputs are bitwise identical, so any nonzero difference is attributable to backend nondeterminism, not to the cache.

## Limits

- Synthetic CPU tests cannot establish MPS behaviour, speedup or runtime feasibility.
- Stage 3 measures 65,536-row fits; the 30-epoch projection uses the same frozen formula as v3 and inherits its limits.
- Full-population cache construction cost is not measured and not extrapolated; adoption needs a separate root decision, full-population timing and a fresh F4 preparation/profile version.
- An unresolved ledger job (supervisor crash) must be resolved by root writing its end record; the runner never overwrites ledger records.
- The runner never writes into the parent v3 run or any `members` directory.
