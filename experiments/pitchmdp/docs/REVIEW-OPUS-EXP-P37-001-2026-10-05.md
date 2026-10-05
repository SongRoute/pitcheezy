# OPUS review — EXP-P37-001 DRAFT (embedding-map control, single-GPU)

Reviewer: Claude Opus 5.5, distinct from both authors (next-research author-001 `ebf70e1`, single-gpu author-001 `1fbd9f1`).
Job `pm-jobs/20261005-p37-review`, 2026-10-05 14:21–14:30 KST. Same-model distinct review, NOT an independent-model review.
No GPU, SSH, server, registration, launch, code-f429318, v5/v6, 2026 data, push or merge.

**Verdict: ACCEPT WITH FIXES. 0 blocker, 1 major (prereg §4 timing gate), 5 minor.** Only prereg text needs fixing
before registration. Code and config need no rework. One test re-scope is committed on this branch (`90d9dde`) for the
author or root to take.

## Target (pinned)
- Final source `codex/next-research-1005` @ `1fbd9f1dfce67ae6afd4f3e08173898f32cf14ae` (parent `ebf70e1`, base `f8a9a1b`).
  The single-gpu helper state is `returned_for_review`, exit 0, with no permission denials. This review branch
  `codex/p37-review-1005` was created at that SHA.
- sha256 recomputed at `1fbd9f1` (all equal to review-request-002): config `75265fc7…ace2`, prereg `2c9fec03…86db`,
  generator `231f2ea4…14f0`, registry `7771b9fa…4102`, tests `69b6c43f…2a67`, runner `132ef47a…eb`, scorer
  `bd40910a…4bb4`, queue `3b9adb3d…`, P36 source config `b6479afa…`.
- Generator was run twice. Both runs exited 0 and gave `75265fc7…` (deterministic, worktree clean).
- Runtime delta vs `code-07ab288`: exactly the new registry plus runner and scorer dispatch (`git diff --stat 07ab288 1fbd9f1`, 3 files).

## Science and data boundaries — pass
- **Arms**: the exact earlier registered entries. The config differs from `ebf70e1` only in `device` and `queue`.
  protocol, export, training (cap 60, patience 5, lr 5e-4, bs 1024, hpo 0), calibration and data_rules are JSON-equal
  to EXP-P36-001. Seeds 39–43 are disjoint from blocks 0–38 (tested). 15 fits.
- **Contrast**: one pre-registered pair, periodic − linear, isolating `embedding_map`. The +264 parameters are exactly
  the shared `Linear(32, 8)` (32·8+8), so "parameter-near-matched, NOT compute-matched" is correct. The pair is reported
  on its own, with no scientific verdict, ranking or winner. `decision`/`ranking` are None.
- **Novelty vs P32**: P32 ran the same pair at cap 30 with 3 seeds. dev favoured periodic and mlb_dev covered 0. P37
  asks the open question at the cap-60 condition, where P36's periodic gain was seen. It is labelled a repeat on the same
  rows and not independent, and must never be pooled with P32. It also says seeds are new model RNG, not new data.
- **Data**: same export manifest pin as P36. No 2026 (exporter guard + runner check). DEV/mlb_dev are predicted only.
  The smoke pilot fits on TRAIN 8,192 / earlystop 2,048 and predicts 256 DEV rows that are never scored.
- **Frozen training**: no model, factory or loop file changed. Routing for P30–P36 is checked against the base runner
  `f8a9a1b`: config_check, training_budget and uses_epoch_cap give identical output for all 7 registered configs. The
  marker cannot be added to an earlier family (tested). P37 without the marker is refused by the P32 check (tested).
- **Routing order**: the embed-type predicate comes first in config_check, training_budget, uses_epoch_cap and scorer
  `score()`. The scorer refuses embed_type contrasts outside the family, and refuses the family with any other contrast
  key. There is no path into the P32 six-arm check, or back out of it.

## Resource guard proof — pass
- Config device = `{cuda, [GPU3 GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9], max_parallel 1, cpu_fallback false}`.
- `check_device`, called from `check_family` and so from runner `config_check` and queue `preflight`, refuses:
  GPU4 added, GPU4 only, GPU4 + parallel 2, parallel 2, another GPU, and the P36 device block. cpu_fallback is refused
  by the generic runner check. The queue constructor refuses `[GPU4]`, `[GPU3, GPU4]`, duplicates and empty.
  `bind_gpu(GPU4)` raises before `CUDA_VISIBLE_DEVICES` is set. These tests hit the real failure paths.
- **Mutant (mine)**: replacing the `check_device(config)` call with `pass` gives 6 failed / 47 passed.
- All 15 fits run on one GPU, so every pair is same-GPU by design. Other families keep their historic ≤2 generic limit,
  which is unchanged.

## Tests (CPU, OMP_NUM_THREADS=1, `/Users/song/Projects/pitcheezy/.venv/bin/python -m pytest -q -p no:cacheprovider`)
- `tests/test_cuda_embed_type_wiring.py` @1fbd9f1: **53 passed**, exit 0.
- 3 known failures, checked at both commits:
  - `test_cuda_ple_wiring::test_config_is_draft_exact_5_arm…` and `test_cuda_robust_clip_wiring::test_config_is_draft_exact_4_arm…`
    fail on **f8a9a1b too** (2 failed / 322 passed there). They are stale "registered config absent" asserts left over
    from P35/P36 registration. They are not caused by P37. See m4.
  - `test_timing_probe_robust_clip::test_runtime_code_on_this_lineage_equals_the_archived_probe_commit`: new on the
    P37 lineage, as expected, because the runner, scorer and registry differ from code-07ab288. **Re-scoped, not
    disabled**, in `90d9dde`. It now allows exactly the declared P37 delta (A registry, M runner, M scorer) plus new
    configs. Everything else must stay byte-equal to 07ab288. Mutants that fail as intended: a frozen
    `direct_periodic_residual_model.py` is touched, the queue is touched, or the runner is reverted to 07ab288.
- Suite `tests/test_cuda_*.py tests/test_timing_probe_*.py tests/*registry*.py` @90d9dde: **1001 passed, 2 failed**
  (the 2 failures from base), exit 1, 63 s. Run once. The full ~2,000-test suite was not rerun (no new need).

## Findings

### MAJOR
**M1 — §4 timing gate is not operational as written. Fix the prereg text before registration.**
"if the smoke shows any per-epoch time of linear > 1.5 × periodic" has three problems:
(a) No field is named. `fit.json fit_seconds` starts before `verify_export`, which hashes the full export arrays, and
before load and build. In a smoke of 8,192 rows × 2 epochs (≈16 updates), that fixed overhead dominates and pulls the
ratio toward 1, so the gate cannot detect what it is meant to detect.
(b) "any" over 5 seeds × 2 epochs fires on noise such as CUDA warm-up or co-tenant CPU load. It is not
proportionate, and leaves room for a discretionary stop or go.
(c) Smoke cannot measure linear at cap 60, and the prereg admits this. The real bound is the clock: kill at 21:45 and
report an incomplete family.
**Fix (text only; no DEV involved; fixed before any P37 smoke):**
- Statistic = median over seeds 39–43 of `report.seconds / report.epochs_run`. This is the training-loop wall on TRAIN
  and earlystop rows only, and is already in fit.json. Compare linear against periodic. The gate is ratio > 1.5 →
  stop and re-plan. Use only this statistic and do not pick seeds.
- Add a schedule check: launch only if launch time + 15 × (P36 periodic max 235.2 s × max(1, ratio)) ends before 21:00.
- State that the smoke gate is a sanity check and not a cap-60 cost measurement.

### MINOR
- **m1 — cost record.** `fit_seconds` and `seconds_per_update` include export hashing and loading (same fixed overhead
  in both arms), which dilutes the ratio toward 1. Add `train_seconds = report.seconds` and `train_seconds_per_update`
  next to them, and keep the existing fields. This is a small scorer and registry change plus a test.
- **m2 — §7 score step.** The schedule fits but does not spell out the score step. Worst case: kill at 21:45, then
  pinned scorer (P36 pipeline timeout 900 s) finishes by ≤22:00, aggregate by 22:30, pull and report by 23:00. Write
  "score by 22:30 or report incomplete" into §7 and the P37 pipeline.
- **m3 — gate 3 (pipeline).** The P36 TEMPLATE hardcodes `--gpu GPU3 --gpu GPU4` (line 42). The P37 pipeline needs
  exactly one `--gpu GPU-c334d905…`, the shared `--lock-dir …/gpu-locks`, and an assertion that the smoke set equals
  {GPU3}. The queue default lock dir is per-queue, not the common lock, so the shared `--lock-dir` is required.
  Keep the pgrep start_after check.
- **m4 — stale presence asserts.** The 2 pre-existing failures are draft-era "config absent" asserts in P35/P36
  wiring tests. The owners of those tests should scope them to their draft window. Until then they hide new
  regressions. Out of P37 scope.
- **m5 — wording and defence in depth.**
  - §5 reading guide, bullet 2: "capacity/embedding presence is the simpler account" is a cross-family inference,
    because residual_counts is not re-run in P37. Soften it to "not attributable to the periodic map over LR on these rows".
  - Optional: the scorer could assert that every fit's runtime UUID equals RETAINED_GPU. `bind_gpu` already guarantees it.
  - `test_earlier_registered_configs_route_and_budget_exactly_as_before` checks only config_check. I checked
    training_budget equality outside the suite (identical). A test could pin it.

## Remaining gates (none done)
1. Root accepts this review. The author fixes M1 (prereg text, which changes the prereg sha) and m2, plus m1/m3 as
   chosen. Take `90d9dde`.
2. Write the P37 launch pipeline from the P36 TEMPLATE (m3), then review it.
3. CUDA smoke 15/15 on GPU3 only, identity checker exit 0, then the §4 gate as fixed in M1. Smoke wall time is 미측정.
4. Registration commit on the real clock by 19:00, then launch under the GPU3 lock. Actual fit ETA is 미측정. The
   40–60 min figure is an estimate from P36, not a promise.
