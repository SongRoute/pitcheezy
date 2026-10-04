# Disposition — same-model review of EMA arms f7282bb (review `ff8c9db`)

Status: `same_model_reviewed; Codex_review_pending`. DRAFT, execution disabled. No smoke, no fit, no server or GPU use.
Author: delegated Opus 5.5 PM worker, cycle-022 (2026-10-04 KST). Review: separate Opus 5.5 session, helper pid 16374, `returned_for_review`.

## Commits
- `2b93e0b`: cherry-pick of `ff8c9db` (review doc + reviewer tests T1–T5). Conflict: the reviewer's registry change (F4) overlapped with `cee0935`, which had already added `SEEDS`, `FITS`, `same()` and `check_seeds`. Kept `cee0935`'s registry. Test-fixture conflict resolved to the `cee0935` side (same content). At this commit one reviewer test fails (see F4), fixed in the next commit.
- next commit (this file): fixes below.

## Findings
- F1 (medium, wording) — **applied**. Design note: EMA can stop later and run more updates; per-seed Δbest_epoch and Δoptimizer_updates are reported.
- F2 (low) — **applied, changed**. The reviewer's sketch kept updating the raw best after the raw stop (no `break`). Added a pure `replay_stop_rule(scores, patience)` that returns at the stop, and report fields `raw_rule_best_epoch`, `raw_rule_stop_epoch` (None if the EMA run ended first). Tests: hand-made list `[3,2,2.5,2.6,1]` → (2, 4) (the late 1.0 must not count), min-delta case, no-stop case; T1 vanishing decay → raw-rule epochs equal the base arm; T2 and a 3-seed synthetic check equal the reference `stop_rule`. Mutant "no return at stop" → 3 fail. A synthetic probe (12 settings, 4 with the raw rule stopping inside the EMA run) matched the reference in 12/12. Diagnostic only, not a DEV evaluation.
- F3 (low, wording) — **applied**. Warm-up: .999 reached at step 8,990 ≈ 7.3 epochs; mean lag 0.1 epoch at epoch 1, 0.6 at epoch 6.
- F4 (medium) — **seeds/fits already in `cee0935`; one remaining gap fixed**. Reviewer test T4 found `priority: True` still passed (`True != 1` is False). `check_arm` and the `dropout` check in `check_family` now use `same()` for `priority` and `workload`. The DRAFT config still passes `check_family` and `check_contrasts`.
- F5 (low) — **already in `cee0935`**. `config_check` checks `uses_ema` first (before tabm/embed-residual/workload/frontier), and dispatch calls `check_arm` per EMA arm.
- F6 (low, CUDA) — **carried to the smoke step**. The smoke must assert per seed over overlapping epochs: `ema.train_weighted_nll == control.train_weighted_nll`, `ema.raw_earlystop_conditional_nll == control.earlystop_conditional_nll`, same for `dropout_ema` vs `dropout`. If not bit-equal, the note's claim becomes "same on CPU; CUDA within X".
- F7 — no change (no defect).
- F8 (info) — **applied as note wording**: in EMA arms `history[*].earlystop_conditional_nll` is the EMA score; cross-arm comparisons use `raw_earlystop_conditional_nll`.
- F9 (info) — **applied**: worst case ≈ 250 s/fit, ≈ 5,000 worker-s for 20 fits, next to the 2,000 point estimate (both estimates, 미측정).
- F10 (info) — **applied**: Izmailov et al. marked as SWA (equal average), not the same method as EMA.

## Hashes after fixes
- `direct_ema_model.py` d3dc5467…2baa
- `direct_ema_registry.py` fe8874fa…4d905
- `tests/test_direct_ema_model.py` 3aebb0e0…e2fd2
- design note 064eec0b9776223ebeea99e2f00773e80fcd48963f466195187778fbc2ba52f8 (re-pinned in `configs/DRAFT-CUDA-EMA-20261004.json`, now a6cf64eb…de7)
- Frozen references unchanged: `git diff eee5c64 -- direct_frontier_model.py direct_frontier_registry.py direct_residual_model.py matrix_models.py` → 0 lines.

## Checks
```
cd experiments/pitchmdp && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_cuda_*.py tests/test_direct_*.py -q -p no:cacheprovider
# 408 passed in 26.08s (Mac, light)
```

## Still open before registration
- A distinct review of the wiring `f7282bb..0d78ce7` (runner/scorer dispatch, DRAFT config) — not covered by this review.
- CUDA smoke of all 20 tasks with the F6 identity assertions.
