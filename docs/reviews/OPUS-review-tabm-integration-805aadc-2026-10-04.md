# Review — TabM-mini runner/scorer integration (805aadc)

Label: `same_model_review; Codex_review_pending`
Reviewer: Opus 5.5, separate from the author (same model). No other model was called.
Date: 2026-10-04 08:08 KST (`date`). Worktree `codex-tabm-mini-integration-review-1004`,
diff `0afe1dd..805aadc`. No GPU, ssh, server or production access. No score was read.

## Disposition

- **CUDA smoke: not blocked.** No correctness defect found in the family guards,
  dispatch, factory guards, smoke path or scorer. Findings 1–2 have small fixes
  in this commit. They are hardening, not bug fixes.
- **Registration: not blocked**, with conditions. Before registering, resolve
  finding 3 (GPU sharing and deadlines with the running P30) and finding 4 (stale
  status line in the design note). Record the measured smoke seconds/memory in place
  of the estimates in finding 6.

## What was checked and holds

1. **Family guards.** `config_check` accepts the DRAFT 5-arm family on `[13, 14, 15]`.
   It refuses all 23 drift cases in `test_cuda_tabm_mini.py`. My probes (scratch
   script, not committed) also showed these are refused:
   - `tabm: true` on `control`
   - `tabm: true` on `periodic_counts_d10`
   - `tabm: false` on an arm of a P30 config (`uses_tabm` tests whether the key
     exists, so any `tabm` key sends the config to the TabM family check, which
     refuses it)
   - P30 config plus one TabM arm
   - `cpu_fallback: true`
   - `last_year: 2026`

   Running the TabM check before the P30 check opens no hole. After
   `check_family`, the per-arm loop still sends `periodic_counts_d10` to
   `direct_workload_dropout_registry.check_arm` and `periodic_counts` to
   `combo_arm_check`, and both check the flag strictly (`is True`).
   `DRAFT-CUDA-DIRECT`, `DATA-COMBO`, `WORKLOAD-DROPOUT` DRAFT and `EXP-P30-001`
   still pass unchanged, and each refuses seeds 13–15 and an added TabM arm (tested).
2. **Exactness.** The config text of `control`, `periodic_counts` and
   `periodic_counts_d10` is identical to the `EXP-P30-001` entries
   (sha `7152ebca…`). I checked with `json.dumps` equality and confirmed every
   scalar has the same type. `export`, `training`, `calibration` and `device` are
   equal to P30. These arms dispatch to `DirectPeriodicModel` and
   `PeriodicHeadDropoutModel` (tested). The TabM arms' settings match
   `direct_tabm_model.ENSEMBLE` and the design note:
   - continuous map: `wr.CONTINUOUS['counts']` (40 base indices + 52)
   - workload: `['prior_records_per100']`
   - `k=8`, `scaling_init='normal'`
   - context width 53
3. **Factory.** `load_tabm` refuses an archive of another kind/format (through
   `DirectTabMModel.load`), or with a different k, dropout, scaling init,
   continuous map or context width. It also refuses a single-network archive
   (new gap test, passes without any fix).
4. **Source map.** The 3 new `MODEL_SOURCES` change `own_hashes()`. Scoring any
   older family with this branch's scorer therefore needs `--source-manifest` with
   that snapshot's map. This was already true: `run_cuda_direct.py`'s own hash
   differs between snapshots anyway. `task_sources` accepts an older map whose
   keys are a subset of the current ones (tested generically in
   `test_scorer_base_family_with_pinned_old_source_manifest`). P30, running from
   its own `5227ee1` snapshot, is unaffected.
5. **Scorer.**
   - Contrasts are checked before anything is read: `tabm_registry.check_contrasts`
     plus the TabM branch run before `task_sources`/`export_rows`/`verify_family`.
   - The return tuple unpacks correctly in every case:
     - base: `joint=[]`
     - workload_dropout: `joint=[joint]` → `joint[0]`
     - TabM: `joint=[None, head]` → `joint[1]`
   - TabM together with workload_dropout is refused. `seed_block` cannot collide.
   - `decision`/`ranking` are `None`. No winner or verdict fields.
6. **Queue.** `cuda_direct_queue.plan` handles the TabM config generically: 15
   tasks, seed-major, without importing torch (probe).
7. **Framing.** Config and note state all of the following:
   - post-selection (`counts` chosen after reading P27)
   - repeated DEV exposure
   - new RNG seeds ≠ new data
   - not compute-matched (k=8 in one fit vs 1)
   - not a faithful TabM reproduction
   - no claim of validation on unseen data
   - deadlines 19:00/19:30/20:00 KST on 2026-10-04
   - task 3600 s, family 36000 worker-s, ≤2 attempts
   - execution disabled

## Findings

### 1. Low — "exact earlier entry" compared values, not JSON text (fixed here)

`pitchmdp/direct_tabm_registry.py:70,72,79,81,92,115` (805aadc line numbers).
`check_arm`/`check_family`/`check_seeds` used Python `==`, so
`True == 1`, `8.0 == 8`, `False == 0.` and `13.0 == 13`.

**Failure scenario:** a registered config with `"priority": true`, `"k": 8.0`,
`"head_dropout": false`, `"width": 128.0` on `periodic_counts`,
`"seeds": [13.0, 14, 15]` or `"fits": 15.0` passed `config_check`. Each of these
was ACCEPTED in the probe. The "exact earlier P30 entry" claim then held only up
to value equality. Execution would not change (TabM arms build from the registry
spec, and flags are checked with `is True` downstream). However, `identity.json`
and the config digest would record a config that differs from P30's.

**Fix applied:** `same(value, expected)` compares `json.dumps(..., sort_keys=True)`.
It is used for arms, BASELINES, workload, priority, seeds, fits and the contrast
list. 7 failing-then-fixed cases are in `tests/test_cuda_tabm_mini_review.py`.
The DRAFT config still passes. P30 and the other registries are untouched; the
same value-equality looseness remains in `direct_workload_dropout_registry`,
which has no consequence for the running P30.

### 2. Low — Korean table: compute note placed over the wrong rows; no seed or exposure line (fixed here)

`scripts/score_cuda_direct.py:313-316` (805aadc). The sentence "TabM-mini 1회
학습(구성원 8)과 단일 신경망 1회 학습의 비교입니다" sat above all 3 rows. One of
them, `periodic_counts_d10_vs_periodic_counts`, compares two single networks.
`score.json` has the `seed_block` note (post-selection, not new data), but
`table_ko.md` — the file Song reads on the phone — did not.

**Fix applied:** the note now says it applies to the `ensemble_head` rows, and
that the `head_dropout` row compares two single networks. A new line says seeds
13, 14, 15 are new model RNG seeds, not new data (same DEV rows, repeatedly
exposed), and that `counts` is a post-selection choice made after reading P27.
These lines appear only in the TabM section; tables for older families are
unchanged. The substring the existing test asserts ('연산량은 같지 않습니다')
is kept. A failing-then-fixed test was added.

### 3. Medium (registration) — same 2 GPUs and same deadlines as the running P30

`configs/DRAFT-CUDA-TABM-MINI-20261004.json` `device`/`queue`.
`allowed_gpu_uuids` equals P30's (the design note's "GPU 3/4"), `max_parallel` is
2, and `no_new_fit_after` 19:00 is the same as P30's.

**Failure scenario:** a TabM queue is started while the P30 queue still runs. Up
to 4 processes then share 2 GPUs, because each queue sees only its own tasks. Each
task's `fit_seconds` then includes contention. Those seconds are exactly the
compute disclosure the design note asks to "report next to every number". Both
families also compete for the 19:00 launch cutoff, so P30 could be left with
`deadline_no_new_fit` tasks.

**Suggested change:** add a queue precondition to the config and to the
registration step, for example:
```json
"queue": { ..., "start_after": "EXP-P30-001 queue terminal (all 50 done or failed_final); never concurrent with it" }
```
Alternatively, record that tasks were co-scheduled next to every reported fit
time. Smoke runs are light and may share the GPUs. Treat their seconds as an upper
bound.

### 4. Low — design note status line is stale

`experiments/pitchmdp/docs/DRAFT-TABM-MINI-DESIGN-20261004.md:3`. The line says
"DRAFT design + model module only. Not registered, not in the CUDA runner". As of
805aadc the module is in the runner. Step 2 also says "08:0x KST"; the commit time
is 08:01:47 KST (`git log`).
**Suggested diff:**
```
-Status: **DRAFT design + model module only.** Not registered, not in the CUDA runner,
-no GPU run, no score read.
+Status: **DRAFT; runner/scorer integration 805aadc (smoke only, execution disabled).**
+Not registered, no GPU run, no score read.
...
-2. Registry/runner/scorer integration (cycle-005, 2026-10-04 08:0x KST): new
+2. Registry/runner/scorer integration (cycle-005, committed 2026-10-04 08:01:47 KST): new
```
Not edited here; the author owns this file.

### 5. Low / informational — scorer robustness

- `score_cuda_direct.py:341-347`: the TabM branch does not explicitly refuse
  `factorial_contrasts` or `regularization_contrasts` on a TabM config. These are
  refused in practice. Probe: the dropout registry's `check_family` refuses, and a
  factorial pair over TabM-family arms never validates. Still, the guard
  `contrasts is not None or regularization is not None` would make the refusal
  explicit. Not changed.
- `score_family` returns 3, 4 or 5 values by position, and `joint[1]` relies on
  the slot order. A dict return value would be safer the next time a family is
  added. Not changed (refactor, not a fix).

### 6. Estimates (labeled ESTIMATE; replace with smoke measurements)

All figures below are estimates from CPU instantiation, not measurements.

- **Network size:** backbone input width D = 854 (206 raw + 81 embedded × 8).
  Parameters: 313,624 for TabM vs 288,802 for the single network.
- **Activation memory:** about 62 MB per training batch (1024 × 8 × (D + 4·256)
  fp32) and about 179 MB per inference batch of 4096. This is negligible next to
  24 GB. The resident TRAIN tensors are the same as for P30 arms.
- **Time:** the two shared head Linear layers run on 8× rows. Per-step compute is
  at most ~8× the single head, and likely less in wall time, since the GPU spends
  much of the time launching kernels.
  - Using the design note's P27 mean of 84 s/fit, the ESTIMATE is ~130–670 s per
    TabM fit.
  - Family total ≲ 9 × ~84 s + 6 × 670 s ≈ 4.8k worker-s, inside the 36000
    worker-s cap and the 3600 s task cap.
  - Wall time at 2 GPUs is ≲ 45 min if P30 is not co-running. Registration must
    therefore happen well before 18:00 KST to fit the 19:00 no-new-fit cutoff.

### 7. Test adequacy

Added in `tests/test_cuda_tabm_mini_review.py`:
- 7 type-drift cases (failing → fixed, finding 1)
- a Korean table scope/seed test (failing → fixed, finding 2)
- 2 cases where a single-network archive is loaded into a TabM arm (pass with no
  fix; the guard already exists)

Still missing (not added):
- a scorer test refusing a TabM config that also carries `regularization_contrasts`
  (finding 5)
- an end-to-end check that this branch's scorer scores a P30-shaped family whose
  tasks carry the 14-key source map of `5227ee1`. Only the generic base-family
  version exists.

## Changes in this commit (separate from the reviewed code)

1. `experiments/pitchmdp/pitchmdp/direct_tabm_registry.py`: `same()` gives
   JSON-exact equality in `check_arm`, `check_seeds`, `check_family` and
   `check_contrasts` (finding 1). Registry sha changes. No TabM task exists yet,
   so no recorded identity is affected.
2. `experiments/pitchmdp/scripts/score_cuda_direct.py`: TabM section of
   `korean_table` only (finding 2). This file is not a task source; it is recorded
   only in scorer provenance.
3. `experiments/pitchmdp/tests/test_cuda_tabm_mini_review.py`: 10 new tests.
4. This review file.

## Tests run

```
cd experiments/pitchmdp && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest \
  tests/test_cuda_tabm_mini.py tests/test_cuda_workload_dropout.py tests/test_cuda_frontier_run.py tests/test_cuda_direct.py \
  tests/test_cuda_dropout.py tests/test_cuda_data_combo.py tests/test_direct_tabm_model.py tests/test_direct_tabm_model_review.py -q -p no:cacheprovider
```
- At 805aadc, before changes: `245 passed in 23.68s`.
- New review file before fixes: `8 failed, 1 passed` (the 1 pass was a case
  already refused downstream; it was dropped).
- After fixes, same command plus `tests/test_cuda_tabm_mini_review.py`:
  `255 passed in 23.42s`.

The full suite was not run, as instructed.
