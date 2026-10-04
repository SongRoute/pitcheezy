# DISPOSITION — same-model review of the P35 PLE module (c953bfd → 840c771), 2026-10-04

Label: `same_model_reviewed; Codex_review_pending`. Author: Opus child 81676. Reviewer: a separate Opus child, 84467
(returned 09:51:15Z, exit 0, no permission denials, only claude-opus-5-5 observed). Disposition: the delegated PM
worker, v4 cycle-018 (T67). No performance claim. No GPU, export or real data was used.

## Decision

Accepted. `codex/p35-module-1004` was fast-forwarded to `840c771` (reviewer commits `a831f8d` tests, `cac12ea` fix,
`840c771` review doc). The final module is `pitchmdp/direct_ple_model.py`
`d3575019f971b06ba26060a1fa31e3894a070fd3b4b88191393743cb16ea175e`. The author pin was `6a1541a3…7425`.

| Finding | Disposition |
|---|---|
| F1 MINOR: TabM fit kept an extra host copy | accepted (fixed in `cac12ea`; reviewer reports bitwise-equal logits/history before and after) |
| F2 MINOR: class-level `KIND`/`FORMAT` inherited P31 values | accepted (fixed in `cac12ea`) |
| F3 MINOR: mutants X9 (label order) and X14 (early-stop physics guard) survived | accepted (tests in `a831f8d`) |
| N-a: package deviations (i) no row floor in `compute_bins`, (ii) no W draw, (iii) silent on single-bin columns | accepted as disclosure only; add to the DRAFT §2 disclosure at the next DRAFT edit, before registration. No code change |
| N-b: B1 reachability in the smoke slice is 미측정 | carried as W5: the smoke prints per-position valid counts |
| N-c, N-d | noted; GPU memory/time stay 미측정 until the DRAFT §6 probe |
| N-e, N-f | optional hardening; not done (no output effect). Recorded, not hidden |
| W1–W5 | carried into the wiring disposition: W1 re-pin `MODEL_SOURCES` to `d3575019…`, W2 record `ple_effective_parameter_count` next to the allocated count, W3 label → export-column map, W4 dispatch-first, W5 smoke on the TRAIN slice + valid counts |

## My checks (PM rerun, not the reviewer's word)

- `git merge --ff-only 840c771` on a clean worktree, exit 0.
- `shasum -a 256`: module `d3575019…a175e`; frozen `direct_periodic_residual_model.py` `ee1b3d3f…`,
  `direct_tabm_residual_model.py` `6a9daff6…`, `direct_tabm_model.py` `c968aec9…` (unchanged).
- 18:52:03–06 KST: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 perl -e 'alarm 600; exec @ARGV'
  .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_direct_ple_model.py
  tests/test_direct_periodic_residual_model.py tests/test_direct_tabm_residual_model.py tests/test_direct_tabm_model.py`
  → **83 passed**, exit 0.
- Negative control: `git archive 840c771 experiments/pitchmdp` into `/tmp/pz-t67-negctl-ple`, module replaced by the
  c953bfd version (`6a1541a3…`), new test file → **1 failed / 26 passed**, exit 1. The failing test is
  `test_tabm_fit_hands_the_callers_arrays_to_the_frozen_loop_and_class_kinds`. This matches the reviewer.

Not rerun by me: the reviewer's package-bitwise comparison and mutation harness (they live in the reviewer's
`/tmp/pz-p35-review/`). They are the reviewer's evidence, not mine.

## Next

The wiring author (84755) started from c953bfd and pins `6a1541a3…`. When it returns: merge onto `840c771`, re-pin to
`d3575019…` (W1), then W2–W5, its focused tests, a distinct wiring review, and only after P30-002 is terminal the CUDA
smoke and the timing probe with the unchanged gates.
