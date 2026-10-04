# Review — tentative EXP-P36-001 robust-clip pre-registration DRAFT + wiring (8311fa4..a23e77b), T104

Label: `same_model_review; Codex_review_pending`. Reviewer: Opus 5.5, distinct from the wiring author (T101b), the module
author (T100), the module reviewer (T100b) and the PM worker who merged and re-pinned. Written 2026-10-04 on
`codex/robust-clip-wiring-review-1004` (base `a23e77b`). Mac CPU only, synthetic data only. No ssh, GPU, network, real
data, 2026 data, score or results directory. Nothing registered.

## 요약 (Song용)

- 사전 등록과 연결 코드에 실험을 막는 결함(BLOCKER)은 없다. **P35가 끝난 뒤 CUDA smoke 20개를 돌려도 된다.**
- 등록 전에 꼭 고칠 것 1건(MAJOR): 사전 등록 §4의 시간 관문이 "smoke의 epoch당 시간 × 60"인데, smoke는 TRAIN
  8,192행 × 2 epoch라 실제 시간(약 125만 행)을 재지 못한다. P35처럼 전체 TRAIN 시간 측정(probe)이 필요하다.
- 문구·기록 보완 5건(MINOR): 마감 시각 규칙(10-05 마감이면 늦어도 16:30 시작), 시드 34가 P35 시간 측정에 쓰인 사실,
  모듈 해시 표기, "five flags"→"six flags", P35 정식 설정 파일과의 대조.
- 테스트가 놓친 변이 2개를 찾아 테스트 15개를 새로 넣었다(`204929e`). 집중 테스트 8개 파일 428개 통과.

## Findings

### MAJOR

**M1. The §4 timing gate cannot be measured by a smoke (blocking-for-registration, not for smoke).**
`PREREG-DRAFT-EXP-P36-001-robust-clip-2026-10-04.md:99-101` and the DRAFT `queue.stop_rule` gate on
"smoke s/epoch × 60 + measured setup ≤ 1,800 s". A smoke task loads only `SMOKE['train'] = 8,192` TRAIN rows and runs
`SMOKE['epochs'] = 2` (`scripts/run_cuda_direct.py:152`, `:783`, `:791`), i.e. 8 updates per epoch against 1,224 in
the full run. The P35 amendment already found that smoke times "are dominated by warm-up and are not a projection"
(`docs/AMENDMENT-P35-TIMING-PROBE-20261004.md:15`). Failure: the gate passes for any arm, the full-TRAIN
`compute_stats` time (81 float64 quantile passes over ≈1.25 M rows; unmeasured) and the full-TRAIN host RSS with the
extra guarded copy (module review n4) are never measured before registration, and "no separate timing probe unless
the smoke projection is within 10 % of 1,800 s" then never triggers. Today the only real bound is the assumption that
`residual_counts` and the robust-clip arm are ≤ 1,046 s/task (P35 probe, periodic arm), which P32 supports
(residual 163 s vs periodic 217 s per fit, `RESULT-EXP-P32-001-20261004.md:16/:19`) but which the prereg itself calls
"an assumption until the smoke measures it". Suggested replacement for §4 bullet 3 (and the same sentence in
`queue.stop_rule`):

> - Gate before registration: (i) CUDA smoke of all 20 tasks on GPU3/4 (fresh archive/code dir), exit 0 — wiring and
>   identities only, smoke times are not a projection; (ii) timing probe by the P35 Amendment-2 method for
>   `residual_counts` and `robust_clip_residual_counts` on each of GPU3/GPU4, probe seed 39 (outside 0–38; no artifact
>   kept, no DEV/MLB DEV row touched): task_projected = startup + prepare (incl. `compute_stats` on the full TRAIN
>   argument, reported separately) + 60 × per_epoch (warm full-TRAIN 2-epoch fit after a 1-epoch warm-up,
>   max(wall, report)/2) + predict + 30 s, slower GPU per arm; `control` and `periodic_residual_counts` take the P35
>   probe values 464 s / 1,046 s (`RESULT-P35-TIMING-PROBE-3b317f2-20261004.md:30-31`). Gates: every arm ≤ 1,800 s,
>   family = Σ_arms 5 × slowest ≤ 36,000 worker-s, probe exit 0; peak device bytes and peak host RSS
>   (`resource.getrusage(RUSAGE_SELF).ru_maxrss`) printed per arm. Any gate false → no registration, record kept.

(If the PM prefers no probe, the honest alternative is to state the gate as the P35-probe bound by assumption and drop
the words "smoke s/epoch"; the RSS and `compute_stats` time then stay unmeasured until the first real task.)

### MINOR

**m1. Launch-time rule is not fixed in numbers (blocking-for-registration).** `PREREG…:109` "latest launch =
aggregate_by ≤ 23:00 under the fixed L-relative rule" and the DRAFT `queue.date_note` (`configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json:455`)
defer the offsets to "the measured smoke projection". The P35 pack rule is no_new = L + 5 h, kill = no_new + 1 h,
aggregate = kill + 30 min (`docs/launch/LAUNCH-PLAN-p35-20261005.md:51`), i.e. latest L = 16:30 for 23:00. The DRAFT
placeholders 22:00 / 22:30 / 23:00 do not follow that spacing (under it, no_new 22:00 ⇒ aggregate 23:30). Failure: at
registration the offsets could be chosen after P35 timings are seen, or a launch after 16:30 on 10-05 could be
rationalised. The family budget 36,000 worker-s / 2 GPUs = 5 h is exactly the L + 5 h no-new window, so the P35 offsets
fit this family unchanged. Suggested §5 text:

> - Deadlines on ONE KST date, fixed now: no_new_fit_after = L + 5 h, kill_own_after = no_new + 1 h,
>   aggregate_by = kill + 30 min (the P35 pack rule). Coordinator deadline 2026-10-05 23:00 KST ⇒ latest launch
>   L = 16:30 KST on 2026-10-05. If P35 is not terminal, or the smoke, probe, reviews and registration are not done by
>   then, P36 does not launch on 10-05 and waits for a new dated go.

**m2. Seed 34 was already used by the P35 timing probe (blocking-for-registration, disclosure only).** The probe used
"Probe seed 34 … no seed-34 artifact is kept" (`docs/AMENDMENT-P35-TIMING-PROBE-20261004.md:44-45`; TRAIN only,
"No DEV/MLB DEV row … is touched", `:142`). The P36 seeds stay legal (no DEV prediction or selection used seed 34),
but `PREREG…:76` "disjoint from P24–P35 (0–33)" and the DRAFT `seed_rule` (`:270`) read as if 34 were never drawn.
Suggested addition to §3 Seeds: "Seed 34 was drawn once as the P35 timing-probe seed (3 TRAIN epochs per probed arm,
no artifact kept, no DEV/MLB DEV row touched); it never produced a prediction or a score." Any P36 probe must use a
seed outside 0–38 (39 above).

**m3. Prereg names the author module, not the pinned one (blocking-for-registration, text).** `PREREG…:47` "module
`5f1864b`" and `:131` (step 1 "running") are stale; the registry pin is `a3d6cbce…52fa` (reviewed `77cc453`,
disposition `55938c7`), recomputed equal to the file. Suggested §2 text: "(module `direct_robust_clip_model.py`
sha256 `a3d6cbce…52fa`, reviewed `77cc453`, disposition `55938c7`; transform in its docstring)". Mark §8 steps 1–2 done.

**m4. "five flags" is six (text).** `PREREG…:70`: context 0–10 are balls, strikes, outs, inning,
defensive_score_lead (5) + defender_is_home, first_base, second_base, third_base, batter_left, pitcher_left (6)
(`pitchmdp/direct_frontier_registry.py` `BASE_CONTEXT_NAMES`). Replace "five flags" by "six flags (defender_is_home,
first/second/third_base, batter_left, pitcher_left)". The rest of the m1 disclosure is accurate against the code:
transformed = 5 past positions × 8 physics channels on valid rows + 41 continuous context columns (11–31, 33–52,
incl. the workload scalar 52) = 81; raw = past type (19) / outcome (11) one-hots, the whole current token, context 0–10
and 32 (`left_share`). Now pinned by `test_prereg_m1_column_split_matches_the_registry`. Worth adding to the
disclosure (note): the transformed set includes bounded proportions — batter archetype memberships (23–27) and pitcher
pitch-type shares — not only unbounded physics.

**m5. "Exact EXP-P35-001 entry" is checked against the P35 DRAFT only (blocking-for-registration).** `configs/EXP-P35-001.json`
does not exist yet (generated at P35 launch), so `BASELINES` and `source_configs.ple_draft` (`:60`) point at
`DRAFT-CUDA-PLE-RESIDUAL-20261004.json`. I recomputed: `control`, `residual_counts`, `periodic_residual_counts` are
JSON-exact to `EXP-P32-001.json` (all three), `control`/`periodic_residual_counts` to `EXP-P34-001.json` and the P35
DRAFT; export, calibration and device sections equal P32/P34/P35. At registration add
`source_configs.ple_registered = configs/EXP-P35-001.json` (+ sha) and a test that `control` and
`periodic_residual_counts` equal its entries (expected to hold: the P35 registered-config test limits the diff to
identity/status/prose/deadlines).

**m6. Two surviving mutants = test gaps (fixed in `204929e`).** R1 (scorer computes `check_one_stats_hash` after
`score_family`, i.e. reads probabilities first) and F2 (`check_family` compares reference arms by `model` only) both
passed all 142 wiring tests. Code order is correct (`scripts/score_cuda_direct.py:624` before `:625`); F2 is masked in
the system by the runner's per-arm loop, but the registry contract was untested. New file
`tests/test_cuda_robust_clip_wiring_review.py` (15 tests) kills both.

### NOTE

- **n1. §7 vs "This draft does not choose".** §7 makes P36 the default next unit under (a)/(b). Suggest one sentence:
  "The choice itself is recorded at registration by the PM, citing the complete P35 descriptive report; §7 fixes only
  the conditions and forbids design changes from P35 magnitudes."
- **n2. Host RSS is not recorded by the runner** (`fit.json` has `peak_device_bytes` only, `run_cuda_direct.py:831`).
  The smoke (8,192 rows) RSS is not informative; measure it in the probe (M1). An additive `peak_host_rss_bytes` field
  would touch every family's fit.json; not recommended now.
- **n3. Equivalent mutants.** Author M2 is equivalent: `RobustClipResidualNetwork.__init__` sets
  `config['rc_stats_sha256'] = stats_sha256(center, scale)` from the same arrays stored in `model.rc_stats`, so either
  half of the condition alone refuses (and with `recorded=None` the remaining half still refuses). My F1 (drop the
  `spec`/`model.context_continuous` map checks, keep `net.config`) is equivalent for the same reason (load builds
  `model.context_continuous` from `config['context_continuous']`). Defence in depth; keep.
- **n4. Smoke families are never scored** (`score_cuda_direct.py:239`), so the cross-GPU one-hash check runs only on
  the real family; in the smoke it must be read by hand (smoke list item 3).
- **n5. Citations.** §1 P32 deltas match `RESULT-EXP-P32-001-20261004.md:34/:37/:38/:40/:43/:44`; 464 / 1,046 s match
  the P35 probe `:30-31`; 5 × 464 + 15 × 1,046 = 18,010 is right; 258 = 233 + 25. Coordinator reports (T84, T89a,
  T90b) are outside this repo and were not checked here.
- **n6. P35 isolation.** P35 runs from its own server code dir `code-8864389` (`docs/launch/run-p35-pipeline.TEMPLATE.py:3`),
  so the runner self-hash and `MODEL_SOURCES` change here do not touch P35. P36 needs its own fresh code dir; an older
  scorer refuses this config (it routes to the P32 family check and fails), so mixing trees fails closed.
- **n7. One-factor claim holds.** `DirectRobustClipResidualModel.__init__` sets exactly the fields of
  `DirectResidualModel.__init__`; `compute_stats` is numpy only (no RNG draw) and runs before the frozen fit seeds and
  builds the network; the network only adds buffers; `residual_counts` builds `DirectResidualModel(seed, device)` via
  `build_combo` (`direct_combo_model.py`). The `_no_current_physics` guard does not change accepted inputs. Initial
  parameters are verified equal on CPU only; CUDA is a smoke item.
- **n8. Records.** `rc_stats_rows` uses the PLE row rule, which is the right count: `compute_stats` takes
  `tokens[valid[:, p], p, c]` for each past position p and every TRAIN row for context columns, and
  `ple_bin_rows` returns all TRAIN rows + `valid[:, :-1].sum(0)` on the same `train_inputs`. The scorer checks one
  statistics hash + one TRAIN row hash + one full record across the 5 fits before `score_family` (now tested), and
  `rc_module_sha256` == pin == identity source map (runner). No winner/verdict/ranking/promotion/selection keys
  (`decision`/`ranking` stay `None`; tested with `FORBIDDEN_KEYS`).

## Checks against the brief

1. **Prereg.** Primary contrast is one factor (n7); secondary is labelled not compute-matched and "never used to pick"
   in §2, the registry docstring, the DRAFT `robust_clip_scope` and the scorer note/table. Repeated-DEV and
   post-selection exposure (§6), 20 fits, attempts, budgets, stop conditions (§5) and §7 are stated before any P35/P36
   score (no P35 fit exists). Gaps: M1, m1, m2, m3, m4, n1.
2. **Registry.** 4 arms in prereg order; references JSON-exact (m5 for P35 registered); contrasts exact, `CHANGED` =
   `{numeric_input}` for both, `HELD` workload/backbone/dropout/head; a drifted contrast or factor table is refused
   (author tests + my F8: secondary `compute_matched` flipped → 10 failed). One pin, equal to the file.
3. **Dispatch-first.** Robust-clip is tested first in `training_budget`, `config_check` (family and per-arm loop),
   `build_model`, `load_model`, `run_task` record block, scorer `score()` and `verify_family`; earlier families
   (10 DRAFTs, P30-001…P34-001, P30-002) route as before (author parametrised test, passes).
4. **Records.** See n8.
5. **Factory strict load.** Refuses other class/kind/format (class loader), architecture, continuous map (same length
   too), context width, hash ≠ run record, missing fit.json or missing hash. Author M2 claim: correct (n3).
6. **Config.** DRAFT, `execution.enabled` false; GPU UUIDs `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`,
   `GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99`, `max_parallel` 2, no CPU fallback; export manifest `49097109…`;
   task 3,600 s, family 36,000 s, 2 attempts; training = P35 budget; `start_after` requires EXP-P35-001 terminal and the
   common lock dir. All 15 `source_configs` hashes recomputed equal to the files (incl. design_note `ac0d6cc7…6c46`).
   DRAFT sha256 `7da55186…278c` (recomputed). Deadline placeholders: m1.

## Commands (from `experiments/pitchmdp`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10`, `perl alarm 600`)

- 7 focused files (`test_cuda_robust_clip_wiring`, `test_direct_robust_clip_model`, `test_cuda_ple_wiring`,
  `test_cuda_ple_wiring_review`, `test_direct_ple_model`, `test_cuda_tabm_residual`,
  `test_cuda_embed_residual_registered`) at `a23e77b`: exit 0, **413 passed**.
- Same 7 + new `test_cuda_robust_clip_wiring_review.py` at `204929e`: exit 0, **428 passed**.
- `tests/test_cuda_ple_registered.py`: exit 1, 4 failed / 1 passed (known, `configs/EXP-P35-001.json` absent; not this
  change).
- Hash recomputation (python3 hashlib): module `a3d6cbce…52fa` == `MODULE_SHA256`; 15/15 `source_configs` equal;
  reference arms / export / calibration / device JSON-exact as in m5.

## Mutations (each alone, restored with `git checkout --`; wiring test file unless noted)

| id | file | mutation | result |
|---|---|---|---|
| R1 | score_cuda_direct.py | `check_one_stats_hash` moved after `score_family` | **survived** (142 passed) → killed by new test (2 failed) |
| R2 | run_cuda_direct.py | `load_model` falls back to `'0'*64` instead of the fit.json record | killed (1 failed) |
| R3 | run_cuda_direct.py | `rc_stats_rows` = all rows for every past position | killed (1 failed) |
| F1 | direct_robust_clip_factory.py | drop `spec`/`model.context_continuous` map checks | survived — equivalent (n3) |
| F2 | direct_robust_clip_registry.py | reference arms compared by `model` only | **survived** (142 passed) → killed by new test (12 failed) |
| F3 | direct_robust_clip_registry.py | `fit_record` ignores the continuous map | killed (1 failed) |
| F5 | direct_robust_clip_registry.py | `fits` check removed | killed (1 failed) |
| F6 | direct_robust_clip_registry.py | P35 block dropped from `EARLIER_SEEDS` | killed (1 failed) |
| F7 | direct_robust_clip_factory.py | context-width check removed | killed (1 failed) |
| F8 | direct_robust_clip_registry.py | secondary `compute_matched` True | killed (10 failed) |

`git status` after all mutations: clean apart from the new test file (committed `204929e`) and this review.

## Author's open questions

1. **Re-pin.** Done at `a23e77b`; `MODULE_SHA256` = `a3d6cbce…52fa` equals the file; `config_check` of the DRAFT passes.
2. **`MODEL_SOURCES` +3.** Acceptable. P35 runs from its own code dir (n6); P36 tasks are scored by the scorer of the
   same tree; cross-tree scoring needs `--source-manifest`, as after the P35 wiring. No action.
3. **Runner-side recompute of statistics.** Not in the runner (a second ≈1.25 M-row pass per task, duplicates the module
   path; the scorer's one-hash-across-5-fits rule plus the one TRAIN row hash is the stronger cross-check). Do it once
   instead: smoke item 4 (8,192 rows) and in the M1 probe on full TRAIN (which also times `compute_stats`).
4. **`verify_family` `uses_ple` call.** Harmless (returns False; no PLE arm). Leave it.
5. **`s_j = 0` columns.** Agree, matches the paper's "else 0"; such a column is constant on TRAIN so carries no TRAIN
   information. Print `rc_rule_columns` in the smoke and probe; a `constant` physics or workload column on full TRAIN
   would be a data-contract finding to record before registration (§7 b).

## CUDA smoke list (after P35 is terminal; all 20 tasks; nothing here was run)

For **every one of the 4 arms** (5 seeds each):
1. Exit 0 and `CUDA_DIRECT_TASK_COMPLETE` for all 20; `fit.json` `device` cuda; `runtime.device_uuid` ∈ the two pinned
   UUIDs, both GPUs used; identity `source_sha256['pitchmdp/direct_robust_clip_model.py']` = `a3d6cbce…52fa`.
2. `fit.json.epoch_cap` present (smoke cap 2); `peak_device_bytes` printed; host `ru_maxrss` printed per task if the
   launcher can (informative only at 8,192 rows — the real RSS comes from the probe, M1).
3. Robust-clip arm: `RC_STATS_ROWS` line; one `rc_stats_sha256` across all 5 seeds **and both GPUs**; `rc_stats_rows`
   identical (`train_rows` 8,192, 5 past-position counts); `rc_rule_counts` sums to 81; `rc_rule_columns` listed;
   `rc_module_sha256` = pin.
4. Independent CPU recompute: `compute_stats(arm_inputs(first 8,192 TRAIN rows), CONTINUOUS)` → `stats_sha256` equals
   the smoke hash (TRAIN-only reproduction).
5. Reference arms carry no `rc_*` fields; `residual_counts` archive loads through `load_combo`, `periodic_residual_counts`
   through the embed-residual factory, `control` through `MatrixModel`.
6. Strict reload on the server: `load_model` of each arm on CUDA reproduces the saved raw `dev` probabilities (bitwise or
   a recorded max-abs tolerance); `rc_center`/`rc_scale` on the CUDA device.
7. `index_copy` on CUDA (module n5): robust-clip `transform` of one smoke batch on CUDA equals the CPU transform within
   1e-6 (max-abs printed).
8. Initial parameters on CUDA: for one seed, the state_dict of `robust_clip_residual_counts` without `rc_*` buffers
   equals `residual_counts` at build (sha printed).
9. Identity-statistics reproduction (one-off script, not a runner task): `smooth_clip` patched to identity, center 0,
   scale 1, same seed — 1-epoch smoke logits equal `residual_counts` (bitwise under the runner's CUBLAS setting, else
   max-abs recorded).
10. Smoke times are **not** a projection; the timing gate is the M1 probe.

## Disposition

- **CUDA smoke of all 20 tasks after P35 is terminal: ready** (0 BLOCKER). Module pin, DRAFT hashes and reference arms
  verified; dispatch-first holds; records and strict load fail closed.
- **Before registration:** M1 (full-TRAIN probe or an honest assumption-only gate), m1 (fixed L-relative offsets ⇒
  latest launch 16:30 on 2026-10-05), m2 (seed-34 disclosure), m3 (module sha in §2, §8 updated), m4 (six flags), m5
  (EXP-P35-001 registered config pinned and compared), n1 (who chooses), re-dated deadlines, prereg sha re-pinned in the
  config after the text edits (`source_configs.design_note`), the Codex review, and Song's explicit go.
