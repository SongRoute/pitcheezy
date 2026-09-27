# F4 observed-context cache audit runner (COOP-001)

2026-09-27. Draft implementation contract for the real-data audit required by `docs/contracts/ML-F4-CONTEXT-CACHE-OPTIONAL-v1.md`. Implemented by Claude Code; reviewed, frozen and executed by the Codex coordinator. **No audit has been executed under this contract.** The cache stays unadopted. Draft config: `configs/EXP-P4-002-v3-cache-audit-v2.yaml`. The historical v1 registration `configs/EXP-P4-002-v3-cache-audit.yaml` is preserved and pinned by hash.

## Owned files

- `experiments/pitchmdp/scripts/audit_ml_context_cache.py` — CLI runner.
- `experiments/pitchmdp/pitchmdp/matrix_context_cache_audit.py` — backend-agnostic stage helpers.
- `experiments/pitchmdp/tests/test_context_cache_audit.py` — synthetic CPU regression tests.
- This contract and the v2 config. Frozen scientific/runner sources are imported as building blocks and unchanged.

## CLI

```
audit_ml_context_cache.py --config C --local-config L --output O prepare
audit_ml_context_cache.py --config C --local-config L --output O stage1 --arm {F4-H0,F4-32,F4-128} --attempt N
audit_ml_context_cache.py --config C --local-config L --output O stage2 --arm A --attempt N
audit_ml_context_cache.py --config C --local-config L --output O stage3 --arm A --path {original,cached} --attempt N
audit_ml_context_cache.py --config C --local-config L --output O compare --arm A --attempt N
audit_ml_context_cache.py --config C --local-config L --output O summary --attempt N
```

Real run order: `prepare`; per arm `stage1`, `stage2`, `stage3 --path original`, `stage3 --path cached`, `compare`; then `summary`. Twenty commands total. Every command re-launches itself as a worker subprocess; the supervisor kills the worker (SIGTERM, then SIGKILL after 15 s) at `limits.command_seconds[command]` and writes `timeout.json` into the attempt directory with exit code 124. This is a hard kill, not a cooperative check; the worker additionally raises if it observes the deadline itself. The worker holds the shared `.heavy.lock` under `artifact_root` for every command, requires the actual MPS backend, `validate_native_runtime`, and `git rev-parse HEAD == repo_commit`.

## Identity and refusal rules

- `config_check(real=True)` rejects any null pin (`repo_commit`, historical registration, parent config/preparation/profile digests, source digests, sample row hashes, owner ledger). The draft config carries nulls for `repo_commit`, all `sources` and `prior_cost_ledger` until the root freeze. Parent pins and sample row hashes are carried over from the v1 registration.
- `verify_parent` checks: parent config file hash, `preparation.json` hash and artifact hashes, `batter_dual_stream_v2`, closed DEV scores, each v3 pinned source unchanged both in the parent `source/` copy and in the working tree, each profile `manifest.json`/`profile.json` hash, profile identity, long length, and that the profile-recorded TRAIN/early-stop/May sample hashes and the warmup `evaluation_train` hash equal the registered selectors.
- Sample selectors are recomputed as in the v3 profile (`linspace` over each split; evaluation rows are a `linspace` over the 8192 TRAIN warmup) and their ordered key hashes must equal the registered values before any work.
- Only TRAIN/early-stop/May batches are loaded (`load_batches` from the equivalence audit). No DEV/June/blend rows, cache, features or metrics.
- `prepare` refuses an output containing any non-AppleDouble file. Stage attempts are `O/<stage>/<arm>[/<path>]/attempt<N>` and refuse non-AppleDouble content; `failure.json`/`timeout.json` are preserved and a new attempt number is required. `compare` and `summary` require exactly one sealed attempt per dependency and record its manifest digest. AppleDouble detection uses the existing `matrix_policy_artifacts.is_appledouble` helper (prefix plus magic).
- Sealed manifests list every non-AppleDouble file; hash mismatch or extra/missing files fail verification.

## Stages

1. `stage1`: build `FrozenObservedContext` over TRAIN ∪ evaluation rows (chunk 8192); bitwise comparison of cached vs original context in original, repeated, reverse and uneven-chunk orders (also original order over all 65,536 TRAIN rows); bitwise comparison of all five `LazyPitchBatch.gather` outputs under candidate pitch-type overrides and current-physics replacements; count change, style change and undeclared row must raise. Timing: cache construction, context-only transform/lookup and whole lazy gather over the 2,048 evaluation rows in 256-row minibatches, each path warmed once, then alternating original/cached order for `timing_repetitions` repetitions. RNG state must be unchanged by construction.
2. `stage2`: `torch.manual_seed(0)`, one `DualStreamNetwork` width128 deep-copied for both paths (bitwise identical initial weights), the same first `minibatches` (8×256) of the seed-0 permutation that `LazyMatrixModel.fit` would use, fresh AdamW per path. Per step: bitwise input context, forward logits, loss, all parameter gradients and updated weights compared under the registered tolerances; numpy/torch-cpu/torch-mps RNG state must be unchanged by each step.
3. `stage3 --path original|cached`: v3-style resource fit in a separate process per path: discarded 8192-row/1-epoch TRAIN warmup evaluated on the evaluation rows, fresh seed-0 width128 model, 65,536×4 epochs/patience 4/batch 256/lr .0005, May16×400 calibration and inference. Records sync-aware MPS timings, cache construction (cached path builds over warmup ∪ evaluation ∪ TRAIN ∪ early-stop ∪ May rows), peak RSS and MPS allocator snapshots. Saves state/probabilities only for comparison (`artifacts.npz`); no model archive, never reused as a member. `compare` checks selected state, history, best epoch, update counts, delivery temperature, calibrated/raw probabilities, sums and tiers.
4. `summary`: all-arm pass flags plus `cost_summary`: original v3 profile projection, this audit's original and cached projections via the frozen `resource_projection` formula (full30 epochs, two loads/member), measured cache construction for audit rows, owner prior-cost ledger, every attempt cost (sealed, failed and killed). Full-population cache construction for fit and prediction processes is reported as `unmeasured`; per-member cached total and family projection are `null`; only a family lower bound excluding cache construction is computed. `decision` is always `not_adopted` from this runner.

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

Logit/probability/temperature values follow the `ml_long_equivalence_v1` precedent; gradient/weight tolerances are the typical fixed backend values the coordinator requested. Inputs are bitwise identical, so any nonzero difference is attributable to backend nondeterminism, not to the cache.

## Limits

- Synthetic CPU tests cannot establish MPS behaviour, speedup or runtime feasibility.
- Stage 3 measures 65,536-row fits; the 30-epoch full-data projection uses the same frozen formula as v3 and inherits its limits.
- The cache's full-population construction cost is not measured here and is not extrapolated. Any adoption requires a separate root decision, full-population timing and a fresh F4 preparation/profile version.
- The runner never writes into the parent v3 run or any `members` directory.
