# Opus 5.5 review — EMA runner/scorer wiring, DRAFT config and disposition (f7282bb..b79888f)

Label: `same_model_review; Codex_review_pending`
Reviewer: Opus 5.5, a separate session from the author. No other model was called and no further Claude process was started.
Target: `f7282bb..b79888f` in worktree `codex-ema-wiring-review-1004` (branch `codex/ema-wiring-review-1004`): `cee0935` (DRAFT config, runner/scorer dispatch, `tests/test_cuda_ema.py`), `0d78ce7` (re-pin), `2b93e0b` + `b79888f` (cherry-picked review and disposition). The module itself was reviewed in `ff8c9db`.
Scope kept: Mac in protected recording mode. Light CPU, single thread, focused tests only. No server, GPU, ssh, ports or tmux. No score or result file was opened. The new checker reads only TRAIN / early-stop losses of synthetic CPU tasks.

## Disposition

**Non-blocking** for a CUDA smoke of all 20 tasks (4 arms × seeds 19–23).
- No path found where an EMA config reaches a non-EMA branch, a non-EMA config reaches an EMA branch, or a mixed config passes. Every route fails closed (Q1).
- The config blocks equal the earlier families. All 8 `source_configs` pins match the files at `b79888f` (Q2).
- The scorer treats EMA arms exactly like other arms. It never reads `history`, so F8 cannot leak. There is no winner, ranking or verdict field (Q3).
- Before registration (not before smoke):
  - run the new F6 checker on the smoke output and record the result (R1);
  - fix the scorer's disclosure wording (R2).

Findings below are R1–R7. Reviewer changes are listed separately at the end. No author file was edited.

## Findings

### R1 — low → **checker added**. The F6 CUDA raw-trajectory identity check had no code
- Disposition F6 carries the assertion "to the smoke step", but nothing in the runner, queue or scorer performs it. The runner cannot do it per task, because `run_task` sees one task only. The queue is family-agnostic.
- The natural place is a separate, additive, torch-free post-hoc checker. It needs no edit to frozen modules, the runner or the scorer, and changes no pinned hash (it is not in `MODEL_SOURCES` / `SCORER_SOURCES`).
- **Added** `experiments/pitchmdp/scripts/ema_trajectory_check.py`:
  - First `rc.config_check`, then it requires `uses_ema`.
  - Per seed and pair (`ema`/`control`, `dropout_ema`/`dropout`):
    - `rc.verify_task` against the expected identity;
    - one shared `source_sha256` map across all tasks;
    - the candidate report has `weights == 'ema'` and the base report does not;
    - over the overlapping epochs (epochs must be 1..n): `cand.train_weighted_nll == base.train_weighted_nll` and `cand.raw_earlystop_conditional_nll == base.earlystop_conditional_nll`. The EMA arm's `earlystop_conditional_nll` (the EMA score, F8) is never compared.
  - Prints JSON with, per pair and seed: `exact`, `max_abs_diff`, `epochs_compared`, `epochs_run` and both `device_uuid`s (GPU3 vs GPU4 matters). Exit code 1 if any pair is not exact.
  - Reads only identity/state and `fit.json`. Never reads predictions, DEV or mlb_dev.
- Smoke usage, after the queue smoke finishes:
  ```
  PYTHONPATH=. python scripts/ema_trajectory_check.py --config ../../configs/DRAFT-CUDA-EMA-20261004.json \
      --export-manifest-sha256 49097109…1f578 --family <runner --output ending -smoke> --smoke
  ```
- If `all_exact` is false, write the note's claim as "same on CPU tests; CUDA within `max_abs_diff`".
- Smoke covers only 2 epochs (`SMOKE['epochs']`, patience 5, so no stop). The full run gives the overlap up to the shorter run. The checker works for both.

### R2 — low (disclosure). Scorer text repeats the pre-F1 wording and states the CUDA trajectory claim as fact
- `scripts/score_cuda_direct.py:439` (`seed_block.note`) and `:362` (Korean table) say the raw trajectory is shared and that "only the evaluated, stopped and saved weights differ".
- Two problems:
  - The design note was corrected for F1: EMA can stop later and run more updates. The scorer text was not.
  - Trajectory equality is shown on CPU only until R1 runs on CUDA.
- These strings go into `score.json` and `table_ko.md`. Suggested diff (not applied; the scorer is the author's and the text should follow the R1 outcome):
```diff
-                          'EMA arms share the raw trajectory of their base arm per seed, only the evaluated, '
-                          'stopped and saved weights differ'}} if ema else {}),
+                          'EMA arms share the raw trajectory of their base arm per seed (CPU tests; CUDA: '
+                          'ema_trajectory_check), the evaluated, stopped and saved weights differ, so epochs '
+                          'run and optimizer updates can differ (EMA usually stops later)'}} if ema else {}),
-                  'EMA 팔은 같은 시드에서 원래 가중치의 학습 경로가 기준 팔과 같고, 평가·멈춤·저장하는 가중치만 다릅니다. '
+                  'EMA 팔은 같은 시드에서 원래 가중치의 학습 경로가 기준 팔과 같고(CPU 테스트, CUDA는 스모크 확인), '
+                  '평가·멈춤·저장하는 가중치가 다릅니다. 그래서 멈추는 epoch와 update 수가 달라질 수 있습니다. '
```

### R3 — low (test adequacy) → **test added**. The EMA config's device, calibration and source pins were not tested
- Sibling families pin these blocks to the earlier config in their tests (`test_cuda_tabm_mini.py:60-67`, `test_cuda_embed_residual.py:79`). `test_cuda_ema.py` does not.
- `config_check` only checks `len(allowed_gpu_uuids) <= 2` and the calibration `bounds`. It does not check which UUIDs, the calibration `method`/`chain`, or the `source_configs` hashes.
- Shown with three config mutants (temporary edit, restored byte-identical; output below). All 36 author tests passed under each one:
  - C1: a third GPU UUID replaces one of the two;
  - C2: the calibration chain is changed;
  - C3: the design-note pin is stale.
- **Added** `test_config_blocks_equal_the_earlier_family_and_every_pin_is_current`. It checks that:
  - training, calibration, export and device equal `DRAFT-CUDA-EMBED-RESIDUAL-20261004.json`;
  - the queue limits (timeout, budget, attempts) are equal;
  - the config is DRAFT with execution disabled;
  - every `source_configs.sha256` equals the file.
- It fails on C1, C2 and C3 and passes on `b79888f`.

### R4 — info. Smoke identities can be linked to a later full run only through the source map
- Every task identity holds the config digest and `smoke: true`. A REGISTERED config has a different digest (status and execution change). A full run therefore cannot verify a smoke identity record as a whole.
- It can verify the smoke's `source_sha256` map.
  - That map does include `pitchmdp/direct_ema_model.py` and `pitchmdp/direct_ema_registry.py`: `MODEL_SOURCES` at `run_cuda_direct.py:101`. It is asserted for a real CPU smoke task in `test_cuda_ema.py`.
  - The R1 report prints that map.
- Suggested registration test: the REGISTERED config minus `{status, decision, execution, queue, registration}` equals the smoke config minus the same keys, and the smoke checker's `source_sha256` equals the queue's pinned map without `scripts/cuda_direct_queue.py`.

### R5 — info. Queue date is today; a queued smoke after 19:00 KST needs a re-dated config
- `configs/DRAFT-CUDA-EMA-20261004.json:334-336`: `date 2026-10-04`, `no_new_fit_after 19:00`. `cuda_direct_queue.Deadlines` refuses every launch after that time.
- The Mac is in protected recording mode today, so the smoke will probably run later. That needs a re-dated config, which means a new config digest; R3's test compares only the queue limits, so it still passes.
- Direct `run_cuda_direct.py task --smoke` is not time-gated. This is the same pattern as the sibling configs.

### R6 — info. Two guards are redundant (equivalent mutants), not defects
- M2: `score_cuda_direct.py:399` (`elif ema is not None: raise`) cannot be reached. `ema_registry.check_contrasts` calls `check_family`, which already raises for any non-EMA arm list.
- M3: the per-arm `check_arm` at `run_cuda_direct.py:227` repeats the family check's `check_arm` over all `ARMS`.
- Both survive mutation without changing behaviour. Keep them as defence in depth.

### R7 — info. EMA build/load use only the arm name
- `run_cuda_direct.py:436` and `:480` call `build_ema(arm['name'], …)` and `load_ema(arm['name'], …)`. These build from the registry and ignore the arm's other fields.
- An arm that differs from the registry is reachable only if `run_task` is called without `config_check`. `main` (`:722`), the queue `preflight` and the scorer all call `config_check` first. `load_ema` refuses a mismatched archive.
- This matches the TabM and embedded-residual branches. Optional defence in depth (not applied, to keep the runner hash):
```diff
     if arm.get('ema') is True:
-        from pitchmdp.direct_ema_model import build_ema
+        from pitchmdp.direct_ema_model import build_ema, check_arm
+        check_arm(arm)
```

## Checks per question

**Q1. Dispatch order and fail-closed behaviour.**
- `config_check` tests `uses_ema` (`'ema' in arm`, the key's presence) before tabm, embed-residual, workload-dropout and the seeds 0–2 fallback. `check_family` then requires the arm names to be exactly `control, ema, dropout, dropout_ema`.
- Mixed configs:
  - A foreign arm in an EMA config fails `check_family`.
  - An `ema` key in another family routes that config to `check_family`, which fails. This is tested for 5 sibling configs.
  - `ema: false` or `ema: 1` fails `check_arm`.
  - An `ema` arm that also carries `tabm` or `frontier` fails `same(rest, ARMS[name])`.
- The EMA `dropout` arm carries `frontier: true` but no `ema` key, so it takes the frontier path in `config_check`, `build_model` and `load_model`.
- `build_model` and `load_model` test `ema is True` first. A non-EMA arm can never reach `load_ema`. An EMA archive is refused by `load_frontier` and `MatrixModel.load` (tested), and the other EMA arm's archive is refused by `load_ema`.
- The context-name gate at `config_check`'s end runs for this config (because of the frontier `dropout` arm) and passes.
- 8 wiring mutants were run:
  - 6 are caught by the author tests: M1 (scorer accepts a missing `ema_contrasts`), M4, M5, M6, M7 and M8.
  - 2 survive but are equivalent (R6).

**Q2. Config.**
- Arms: exactly control / ema / dropout / dropout_ema. `control` and `dropout` are identical to the FRONTIER-RUN entries.
- Seeds and fits: seeds `[19..23]`, fits 20, no `repeat_tag`. The seeds are disjoint from 0–2, 3–7, 8–12, 13–15 and 16–18; each sibling registry pins its own seed list exactly, so no sibling family can take 19–23.
- Contrasts: the fixed 3, each with `held`.
- These blocks equal the EMBED-RESIDUAL, TABM-MINI, FRONTIER-RUN and EXP-P30-001 configs:
  - `training` equals `rc.TRAINING` (30/5/1024/5e-4/uniform/0);
  - calibration (bounded `minimize_scalar`, [.5, 2.5], G0 chain, temperature split);
  - `export`;
  - `device`: the two GPU3/GPU4 UUIDs, `max_parallel 2`, `cpu_fallback false`;
  - the queue limits.
- `execution.enabled` is `false` and the status is DRAFT. Smoke is allowed and real fits are refused (tested).
- The 8 `source_configs` sha256 values were recomputed: all match (output below).
- Budget:
  - worst case ≈ 250 s per fit, below the task timeout of 3,600 s;
  - 2,000–5,000 worker-s, below the family budget of 36,000;
  - at 2 GPUs, a wall time of about 1,000–2,500 s (estimate; the smoke's time is 미측정).
  - The EMA overhead is one extra early-stop pass per epoch plus a per-step `lerp_` over about 142k parameters. That is small next to the 1.1× factor already assumed.

**Q3. Scorer.**
- EMA arms take the same `score_family` path as every other arm:
  - `prediction_metrics` (NLL, Brier, accuracy, top-label ECE10) on `dev` and `mlb_dev`;
  - `member` checks keys, labels, game ids and the no-physics level against the export;
  - `paired_block` uses the game-block bootstrap (10,000 draws, seed 20260924).
- The 3 EMA contrasts reuse `paired_block`. The test checks that ensemble Δ = Δ(cand vs control) − Δ(ref vs control).
- F8: the scorer reads only these fields:
  - from `fit.json`: `device`, `runtime`, `train_subset` and `train_rows_sha256`;
  - from `calibration.json`: `delivery_temperature`.
- It never reads `history`, so the EMA-score key is never compared across arms. Outside the CUDA pipeline, only MPS-era `run_ml_*` scripts read history keys, and they never see EMA tasks.
- `decision` and `ranking` are `None`. There is no winner, promotion or verdict field.

**Q4. Identity.** See R4 (the source map includes both EMA files) and R1 (checker added).

**Q5. Tests.** See R3 (added), R1 (added) and the mutant table below.

## Mutant table
Mutants were applied by string replacement and restored from the in-memory original. The author tests are `tests/test_cuda_ema.py` and `tests/test_direct_ema_model.py`, run with `-x`.

| mutant | author tests | review tests |
|---|---|---|
| M1 scorer: EMA config without `ema_contrasts` accepted | 1 fail | – |
| M2 scorer: drop `elif ema is not None` guard | 0 fail (equivalent, R6) | – |
| M3 runner: drop per-arm `check_arm` | 0 fail (equivalent, R6) | – |
| M4 `load_model` loads every EMA arm as `dropout_ema` | 1 fail | – |
| M5 scorer EMA contrast candidate/reference swapped | 1 fail | – |
| M6 `config_check` EMA branch skips `check_family` | 1 fail | – |
| M7 `build_model` builds every EMA arm as `ema` | 1 fail | – |
| M8 `build_model` EMA branch unreachable | 1 fail | – |
| C1 config: third GPU UUID | **0 fail** | 1 fail (R3) |
| C2 config: calibration chain changed | **0 fail** | 1 fail (R3) |
| C3 config: stale design-note pin | **0 fail** | 1 fail (R3) |

## Changes made (separately)
1. `experiments/pitchmdp/scripts/ema_trajectory_check.py` (new, R1). Additive and torch-free. It is not part of any pinned source list.
2. `experiments/pitchmdp/tests/test_cuda_ema_wiring_review.py` (new, 9 tests):
   - R3 config pins;
   - a real CPU smoke of the 4 arms at seed 19 through `run_task`: the checker reports exact equality (max |Δ| = 0) on both pairs over the 2 smoke epochs, and it refuses missing seeds and an unregistered seed;
   - fake-task histories: a 1e-7 difference is reported, the EMA score is not compared, and the overlap is the shorter run;
   - it fails closed on a base flagged EMA, a candidate that is not EMA, a tampered `fit.json`, a source-map mismatch, misnumbered epochs and a non-EMA config.
3. This review file.
- No author file was edited. Frozen references are unchanged: `git diff f7282bb..HEAD` over `direct_frontier_model/registry`, `direct_residual_model`, `matrix_models` and the tabm / embed-residual / workload-dropout / dropout / combo registries → 0 lines.

## Commands run (output)
```
$ cd experiments/pitchmdp && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
    /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_cuda_*.py tests/test_direct_*.py -q -p no:cacheprovider
# at b79888f:             408 passed in 26.47s
# with reviewer changes:  417 passed in 26.28s

$ python - (recompute configs/DRAFT-CUDA-EMA-20261004.json source_configs sha256)
base_cuda True / frontier_run True / data_combo_run True / dropout_run True / workload_dropout_run True /
tabm_run True / design_note True / embed_residual_run True

$ python - (compare blocks with sibling configs)
EMBED-RESIDUAL, TABM-MINI, FRONTIER-RUN, EXP-P30-001: training/calibration/device/export/protocol all True; queue diff {}

$ python /tmp/ema_mut.py   (mutants, -x; restored; git status shows only the two new files)
M1 scorer drops "ema is None" | author tests: 1 failed, 20 passed in 2.19s
M2 scorer drops "elif ema is not None" guard | author tests: 36 passed in 2.43s
M3 runner per-arm check_arm removed | author tests: 36 passed in 1.94s
M4 load_model loads every EMA arm as dropout_ema | author tests: 1 failed, 18 passed in 1.27s
M5 scorer EMA contrast candidate/reference swapped | author tests: 1 failed, 19 passed in 1.39s
M6 config_check EMA branch skips check_family | author tests: 1 failed, 6 passed in 0.54s
M7 build_model builds every EMA arm as ema | author tests: 1 failed, 17 passed in 0.52s
M8 build_model EMA branch after frontier (moved: ema->never reached) | author tests: 1 failed, 17 passed, 1 warning in 0.74s
C1 config: third GPU UUID replaces one | author tests: 36 passed in 1.88s | + review tests: 1 failed in 0.40s
C2 config: calibration chain changed | author tests: 36 passed in 1.86s | + review tests: 1 failed in 0.40s
C3 config: stale design-note pin | author tests: 36 passed in 1.86s | + review tests: 1 failed in 0.40s

$ pytest tests/test_cuda_ema_wiring_review.py -q
9 passed in 1.80s

$ git diff --stat f7282bb..HEAD -- <frozen models + sibling registries> | wc -l
0
```
