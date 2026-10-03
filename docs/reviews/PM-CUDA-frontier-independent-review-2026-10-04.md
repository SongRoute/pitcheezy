# CUDA frontier batch — independent review (2026-10-04)

Reviewer: Claude Opus 5.5 (`claude-opus-5-5`), separate from the authoring call. Scope: `61bbd67` + `906f16b`
(`direct_frontier_model.py`, `direct_frontier_registry.py`, `run_cuda_direct.py` diff, `score_cuda_direct.py`,
the two DRAFT configs and their tests). Code review and synthetic CPU checks only: no SSH, GPU, real export,
fit or 2026 access. First-batch code `edfd4cc` was not touched. No scientific verdict is made or changed.

## Verdict

**No material bug found; no code change.** Ready for root's diff review and GPU smoke.

## What was checked

- **One change per arm vs. the same control.** The capacity arms are plain `MatrixModel('flatten_mlp', width)`.
  Re-derived the parameter count by hand: `h² + (n_flat+12)h + 10`, with `n_flat = 6·39 + 52 = 286`. It matches
  the residual and periodic targets within 0.06 % and 0.32 %. The information arms use the control network.
  The regularization arms mirror `MatrixModel.fit` with only the AdamW weight decay and dropout changed. The
  budget, early stop, calibration and inputs are unchanged. `check_config` and `frontier_arm_check` require each
  arm to equal its registry entry exactly.
- **Mask and context columns.** The column order was traced to source. `SequenceContext.transform` produces 11
  game-state columns, then `Archetypes.numeric_features`, which is `column_stack([style z, evidence, memberships])`.
  So `no_batter` = 11–27 and `no_batter_cluster` = 23–27 are correct. The runner passes names to the information
  arms only after the export manifest's `context_layout` equals the config's expected layout. The base arms receive
  no names, and the build refuses a mismatch.
- **Leakage and invariant probe (synthetic, ad hoc).** For each information arm, perturbing the dropped context
  columns did not change the logits at all (`atol=0`). For `no_history`, randomizing past tokens and flipping their
  valid bits also left the logits unchanged. Changing a kept context column, or the current token, did change the
  logits. For the dropout arms, logits are deterministic after fit and stay identical even when the net is put back
  into train mode, because `MatrixModel.logits` calls `eval()`. The current-token physics and outcome leak guards in
  `arm_inputs` are unchanged.
- **Scorer.** Before any prediction is read, it checks:
  - the config SHA and that the config is REGISTERED and enabled;
  - the export manifest pin, including the new config `manifest_sha256` pin;
  - the hashes of the `keys`, `y` and `game_pk` arrays;
  - that every arm × seed is complete, with an exact identity, the expected source map and intact artifacts;
  - that all tasks share one CUDA runtime signature.

  Each member's keys, labels and game ids must equal the export rows in order. Comparisons are always against the
  same-family control. The output directory is immutable, and no decision or ranking is written.
- **First batch through the scorer.**
  - The new `config_check` accepts the base config unchanged.
  - The base config has no `export.manifest_sha256`, so the new pin check falls back to the CLI pin.
  - `--source-manifest` accepts the exact old 4-file map (a subset of the new names) and rejects unknown files and
    malformed hashes.
  - It is covered by `test_scorer_base_family_with_pinned_old_source_manifest`. The default (current) sources
    correctly refuse first-batch tasks.

## Limitations / notes for root (no change made)

1. The scorer refuses an **incomplete** family. For the first batch, `residual`/`periodic` are priority 2, so the
   portable scorer cannot score it until all 5 arms × 3 seeds have finished. This is intentional fail-closed
   behavior, not reduced.
2. `--source-manifest` is recorded with its own SHA but not pinned by a CLI hash. A wrong map can only fail closed,
   because task identities must match it exactly.
3. `no_history` raises at fit if any real row has `valid[:, -1] == False`. This was not checked on the real export,
   and the GPU smoke will show it.
4. The ensemble is the mean of per-seed **calibrated** probabilities, as disclosed. The bootstrap is conditional on
   fixed predictions, as stated in the score's `limits`.

## Tests actually run

- `pytest -q tests/test_direct_frontier_model.py tests/test_cuda_frontier_run.py` → **45 passed** (7.7 s; main-repo
  `.venv` Python).
- Ad hoc probe (`/tmp/frontier_probe.py`, not committed) covering the mask/eval invariants above → all assertions
  passed.
- Not run: the full pitchmdp suite, GPU/CUDA, real export or manifest.
