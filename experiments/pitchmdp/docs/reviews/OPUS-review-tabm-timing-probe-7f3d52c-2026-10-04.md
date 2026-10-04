# Review — TabM-residual Amendment 2 timing-only TRAIN probe (`ee954a5..7f3d52c`)

Label: `same_model_review; Codex_review_pending`. Reviewer: Opus 5.5, distinct from the author.
2026-10-04. Worktree `codex-tabm-timing-probe-review-1004`, branch `codex/tabm-timing-probe-review-1004`.
No ssh, GPU, port, real-data fit or 2026 data was used. Mac commands ran with `OMP_NUM_THREADS=1
MKL_NUM_THREADS=1` and `perl -e 'alarm 600; exec @ARGV'`. Only the probe test file was run, not the
full suite.

## Disposition

**Ready to run once on GPU3/4 at `5c9c8aa`** (probe script sha256
`da8146f46daa6d98be6da80fef711bee71328de87da89a44b1dee3c7a7e5ff95`) after Codex review, provided
root/author settle the open pre-registration items N4, N5 and N8 *before* the run. They change doc
wording or the projection basis, not safety. If N5 is adopted, the one-line code and doc change
needs a quick re-check.

`7f3d52c` as submitted was **not** ready because of B1: on an interrupt it released the GPU locks
while the workers were still alive. B1 is fixed in `71f65c9`.

The method itself is sound:
- TRAIN rows are a faithful cost proxy (same row counts and shapes, no data-dependent cost).
- Every arm goes through the same build/inputs/fit/logits path as `run_task`.
- The projection is conservative: per-epoch time includes the one-time build and upload costs, every
  fit is counted at the 60 cap, the slower GPU is used, and there is a 30 s allowance against
  6.1–7.9 s measured context.
- No score is read or written.

Commits added by this review (each separate; no push, no merge):

| commit | what |
|---|---|
| `1bf354c` | test: worker TRAIN-only + manifest stand-ins; worker refuses changed sources before loading; parent preflight + GpuLock never import torch |
| `71f65c9` | fix B1: signal handlers + stop every worker group before any lock release; tests (interrupt, timeout) |
| `5c9c8aa` | fix N1–N3: worker re-checks config SHA; worker self-alarm; `run` exits 1 when incomplete; test |

Frozen files: `git diff --name-only ee954a5 HEAD` over `pitchmdp/`, `run_cuda_direct.py`,
`cuda_direct_queue.py`, `configs/`, the smoke record and the design note → **0 files**. The
pre-registration doc `AMENDMENT2-…` is not edited by this review. Suggested wording is below.

## Findings

### B1 (blocking; fixed `71f65c9`): GPU locks released while worker groups are still alive

`scripts/timing_probe_tabm_residual.py@7f3d52c:238-275`.

**Problem.** Workers start with `start_new_session=True`. The parent installed no signal handler,
and its `finally` released the flocks without stopping the workers. Python's default SIGTERM/SIGHUP
handling kills the parent without running `finally`, and the kernel drops its flocks.

**Cases.**
- (a) `timeout 1700` fires (GNU timeout signals the parent's process group, not the workers' new
  sessions).
- (b) The ssh session drops (SIGHUP).
- (c) Ctrl-C (KeyboardInterrupt).
- (d) An exception in the launch loop, e.g. the second `open` or `Popen` fails after worker 1
  started.

In every case a worker keeps running on GPU3/4 while the common lock is free. Another family queue
could then launch on the same GPU. That breaks the contract `cuda_direct_queue` keeps ("stop own
groups BEFORE any lock is released") and corrupts both timings.

**Shown.** The new test failed on the old code: `alive_at_release == [55770, 55771, 55770, 55771]`,
i.e. both groups were alive at both releases.

**Fix.**
- `SIGNALS` → `_interrupt` raises into the cleanup, as the queue does.
- The `finally` first calls `stop_group` on every launched worker (SIGTERM, 20 s grace, SIGKILL,
  reap), then closes logs, then releases locks, then restores handlers.
- The timeout path uses the same `stop_group`.
- Timing method and projection are unchanged.

```diff
-    finally:
-        for lock in held:
-            lock.release()
+    finally:
+        for proc in procs.values():  # never release a GPU lock while an own worker group is alive
+            stop_group(proc)
+        for log in logs:
+            log.close()
+        for lock in held:
+            lock.release()
+        for sig, handler in previous.items():
+            signal.signal(sig, handler)
```

### N1 (non-blocking; fixed `5c9c8aa`): worker did not re-check the config SHA

`@7f3d52c:195-205`. Only the parent compared `sha256(config)` with `--config-sha256`. The worker
re-hashed the sources but read the config file unchecked. The worker now refuses a differing config
before `verify_export` or any split load. Test: `test_worker_refuses_changed_config_before_loading_data`.

### N2 (non-blocking; fixed `5c9c8aa`): an orphaned worker had no bound of its own

**Problem.** SIGKILL from `timeout --kill-after=30` cannot be handled. Before this fix a worker
orphaned that way ran until it finished (expected 3–6 min, unbounded if hung).

**Fix.** Each worker now calls `signal.alarm(WORKER_TIMEOUT_SECONDS + 60)` = 1,560 s, and SIGALRM's
default action ends it.

**Remaining exposure.** If the parent is SIGKILLed, the flock is still released while a worker is
still running. The worker bound limits that to ≤ 1,560 s; it cannot be closed fully.

**Timeline.** Parent wall = preflight (config check + full export hash) + ≤ 1,500 s + ≤ 20 s grace.
The 1,700 s wrapper leaves 180 s minus preflight. Suggest either recording the preflight duration
or raising the wrapper to 1,800 s (doc line 76).

### N3 (non-blocking; fixed `5c9c8aa`): `run` exited 0 when incomplete

An incomplete probe (timeout or worker error) printed `TIMING_PROBE_DONE complete False` but exited
0. It now exits 1, so a foreground operator or script cannot mistake it for success.

### N4 (non-blocking, doc wording; open): "Not opened: earlystop, temperature, blend, dev, mlb_dev"

`AMENDMENT2…:42`.

**What actually happens.** `verify_export(..., hash_arrays=True)` reads the bytes of every split's
array files, including dev and mlb_dev, for SHA256. This happens twice: in the parent preflight
(`:232`) and in the worker (`:205`).

**What does not happen.** None of those splits is `np.load`ed, parsed or used. `row_counts` reads
only the manifest `n` metadata. This is identical to every real task and to the smoke, and the code
behaviour is correct.

**Suggested diff:**

```diff
-Data opened: TRAIN only. Not opened: earlystop, temperature, blend, dev, mlb_dev.
+Data loaded: TRAIN only. earlystop/temperature/blend/dev/mlb_dev are never loaded or parsed; their
+array files are only byte-hashed by verify_export (integrity, as in every real task) and their
+manifest row counts `n` are read.
```

### N5 (non-blocking, projection basis; open, decide before the run): `report['seconds']` omits each fit's pre-timer input checks

`probe_arm` `:128,138`; frozen `matrix_models.py:114-129`, `direct_residual_model.py:84-102`,
`direct_tabm_model.py:185-203`, `direct_periodic_model.py:114-118,140`.

**What is excluded.** Every frozen `fit` validates its inputs before `start = time.perf_counter()`:
- `_validate`: `isfinite` over all TRAIN tokens plus an `astype(float32)` copy (~1.1 GB at 1,253k × 6
  × 38 float32);
- `np.isin` over the labels.

For `periodic_residual_counts` this happens twice: `_no_current_physics` in the wrapper, then
`_validate` in `DirectResidualModel.fit`. A real task pays this once per fit. The probe records it
only in `wall_seconds`, which is not used by the projection. The doc's sentence "includes net build,
TRAIN tensor upload and per-epoch validation" is accurate. The cost is likely a few seconds per fit
(**미측정**). The 30 s allowance covers it numerically but does not name it.

**Suggested diff (one line + doc).** `max(wall, report)` is always ≥ `report['seconds']`, adds the
pre-timer checks and the final sync, and multiplies a one-time cost ×30, so it is more conservative:

```diff
-    record['measured_seconds_per_epoch'] = m['fit_report_seconds'] / m['epochs']
+    record['measured_seconds_per_epoch'] = max(m['wall_seconds'], m['fit_report_seconds']) / m['epochs']
```

If this is declined, add "pre-timer input validation in each fit (once per fit)" to the allowance
list at doc line 54.

### N6 (non-blocking, no change needed): is the 30 s allowance defensible?

Real-task wall time not covered by the probe's measured terms:
- interpreter start and pre-`started` imports (numpy, runner; ~1 s; `import torch` is inside
  `bind_gpu`, so it is measured in `startup`);
- queue poll latency ≤ 5 s (`Queue.poll=5`; the task's seconds run from `Popen` to the reap);
- `before_launch` (manifest-only `verify_export`);
- earlystop `load_split` + `arm_inputs`;
- temperature load + `arm_inputs` + `logits` + bounded `minimize_scalar` over softmax;
- per-split `arm_inputs`, two softmaxes, `np.savez_compressed` of the float64 calibrated/raw
  probabilities;
- `model.save`, artifact SHA256, `own_hashes` ×2.

`is_complete`/`verify_task` hashing runs after the queue has stored the elapsed time, so it is not
counted.

Context from the smoke record: P31/P32 full tasks measured 6.1–7.9 s for `seconds_total −
report.seconds`. That covers everything inside `run_task` outside the fit loop, including the full
`verify_export` hash and the TRAIN load. The probe counts `startup`, `prepare` and `predict`
separately *and* adds 30 s. Those terms largely overlap with the 6.1–7.9 s, so the probe counts them
twice. Even after adding the poll and interpreter start (~6 s), the allowance has roughly 3× headroom.
30 s is defensible as an assumption; it does not need to be measured or raised.

### N7 (non-blocking, confirmation): timing faithfulness against `run_task`

**Same path per arm class**
- control → `MatrixModel`.
- `periodic_residual_counts` → `build_embed_residual` → `DirectPeriodicResidualModel`.
- `tabm_counts`/`_d10` → `build_tabm`.
- `tabm_residual_counts` → `build_tabm_residual`.

All of these go through `rc.build_model(arm, seed, 'cuda')`. In addition:
- `train_subset` is a no-op (no `train_fraction`) and `context_names` is None for every arm, exactly
  as in `run_task`.
- `arm_inputs` is used with the config's `workload_columns`.
- Batch size, learning rate and patience come from `config['training']`.

**No data-dependent cost**
- The flatten, periodic, residual and TabM forwards have no boolean indexing and no data-dependent
  control flow. `np.where` appears only in `_flat` (lightgbm) and the recurrent kinds.
- Row counts come from the manifest, so updates per epoch and validation size are identical. The
  real validation size is `n_early` and the real TRAIN gives 1,224 updates per epoch.
- Patience 5 cannot stop a 1- or 2-epoch fit, and `epochs_run` is asserted.
- `predict` is one `logits` call where the real task makes four; the difference is ≤ 4 partial
  batches of 4,096 rows, negligible.

**Conservative**
- `per_epoch` = 2-epoch `report['seconds']` / 2. That includes net build, AdamW init, the full TRAIN
  H2D upload, two validations and up to two `best_state` CPU copies, and each one-time cost is
  multiplied ×30 in the projection.
- Every fit is counted at the 60 cap.
- The slower GPU is used per arm.

**Warm-up**
- One warm-up epoch is 1,224 updates, far beyond the CUDA/cuBLAS warm-up that dominated the
  16-update smoke.
- Each arm gets its own warm-up, so per-network kernels are warm. Carrying the CUDA context and
  allocator cache across arms only removes per-process warm-up. A real task pays that inside its
  first epoch, a few seconds per the smoke, which falls inside the allowance.
- Two measured epochs are enough: there is no LR schedule, the batch count is constant, nothing in
  the loop grows with the epoch, `.item()` syncs every batch, and validation `.cpu()` syncs every
  epoch.

**Possible understatement (미측정; recorded, not blocking)**
- (a) Clocks, thermals and host contention over a ~20 min fit versus a ~1 min window.
- (b) Pairing: the probe runs the same arm on both GPUs in lockstep, while the queue pairs different
  arms (order seed → arm). The concurrency is the same (2 workers). The per-batch loop is bound by
  host-side kernel launches and syncs, so contention depends on free host cores.
- (c) Environment: `torch.set_num_threads(4)` inside every fit overrides `OMP_NUM_THREADS` for torch
  ops; numpy ops in the loop are single-threaded.

Covered by N8.

### N8 (non-blocking, doc/launch; open): pin the launch environment and record host context

**Why.** The doc fixes neither `OMP_NUM_THREADS`/`MKL_NUM_THREADS` (the smoke used 2) nor the host
state. Suggest fixing both now, before the run.

**Suggested doc lines:**
- Launch env = the env planned for the family queue (`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`, as the
  smoke).
- Evidence records `nproc`, `uptime` (loadavg) and `nvidia-smi` (GPU3/4 compute apps) immediately
  before and after the probe.
- If any arm's projection lands within 10 % of 1,800 s, the report calls it marginal. The rule
  itself is unchanged.

### N9 (minor; open): no single combined gate field

`summarize` `:154-175` returns three booleans. The doc requires all three, but no field states the
combined result. Suggest adding `'register_allowed': tabm_residual_within and every_arm_within and
family_within` so the reading cannot be partial. Arithmetic checked:
- task = startup + prepare + 60 × per_epoch + predict + 30;
- slowest GPU per arm;
- family = seeds (5) × Σ slowest.

All three match the doc (lines 49–58) and are tested. Mutations M4–M6 are caught.

### N10 (minor; open): a preflight failure leaves an empty output directory with no error file

`run` creates the output directory before `preflight` (`:264-265`), so a preflight refusal leaves
the directory empty with no record. That counts against "at most 2 attempts" and reuse is refused,
which is correct. Suggest writing `preflight-failed.json` (error repr + UTC), as the queue does.

### N11 (tests; addressed): the original 13 tests missed the worker and the lock path

Three of ten mutations of `7f3d52c` survived: M1, M7 and M10 (table below). This review adds 6
tests; at HEAD every listed mutation fails at least one test.

## Data isolation (item 1), verified

**Splits loaded.** Only `load_split(..., 'train')` is called (`:207`). Tests prove it
(`test_worker_opens_train_only…`, `test_probe_arm_reads_train_only…`).

**Score values stay in memory.** The frozen fit computes the early-stop NLL on the TRAIN stand-in and
keeps `history`/`best_epoch` in `model.report` in memory. The probe copies only `epochs_run`
(asserted), `optimizer_updates` and `seconds`.

**What gets written.**
- `worker-*.json` and `summary.json` pass `no_forbidden_keys`.
- The error path writes `repr(error)` + traceback. The frozen loop's only score-related error is
  "Nonfinite early-stop loss" and carries no value.
- The frozen loops have no `print`.
- There is no `save`/`checkpoint`, no temperature, and no probabilities. Post-fit logits are
  discarded.

**Parent.** The parent reads only worker timing JSON and never imports torch (new subprocess test).

## Commands run (outputs abridged)

```
$ git diff --stat ee954a5 7f3d52c
 AMENDMENT2-TABM-RESIDUAL-TIMING-PROBE-20261004.md | 87, timing_probe_tabm_residual.py | 322,
 test_timing_probe_tabm_residual.py | 194  (3 files, 603 insertions)
$ pytest tests/test_timing_probe_tabm_residual.py -q -p no:cacheprovider     # at 7f3d52c
13 passed in 2.24s
$ /tmp/mut_tp.sh <name> <perl edit>   # copy, mutate, run the test file, restore, `git diff --quiet` check
$ pytest ... (after 1bf354c / 71f65c9 / 5c9c8aa)
16 passed / 18 passed / 19 passed in 3.94s;  pgrep "sleep 60" -> none (no stray test workers)
$ git diff --name-only ee954a5 HEAD -- experiments/pitchmdp/pitchmdp …run_cuda_direct.py …cuda_direct_queue.py configs …SMOKE-… …DRAFT-TABM-RESIDUAL-DESIGN-… | wc -l
0
```

Mutation results. "Original" = the 13 tests at `7f3d52c`; "at HEAD" = all 19 tests at `5c9c8aa`.

| mutation | original | at HEAD |
|---|---|---|
| M1 worker also loads `earlystop` | **survived** | killed (`test_worker_opens_train_only…`) |
| M2 early stand-in = last TRAIN rows | killed (`…standins_are_train_rows…`) | killed |
| M3 `no_forbidden_keys` returns early | killed (`test_forbidden_output_keys_refused`) | killed |
| M4 projection drops 30 s allowance | killed (`test_projection_and_gate_arithmetic`) | killed |
| M5 fit projected at 30, not 60 epochs | killed (same) | killed |
| M6 fastest instead of slowest GPU | killed (same) | killed |
| M7 locks released before workers stopped | **survived** | killed (`test_run_interrupt…`) |
| M8 worker record carries `model.report` | killed (5 × `test_probe_arm…`) | killed |
| M9 `PREDICT_SPLITS` drops mlb_dev | killed (row_counts, preflight) | killed |
| M10 worker skips the source-hash check | **survived** | killed (`test_worker_refuses_changed_sources…`) |
| M11 worker passes n_train as n_early | — | killed (`test_worker_opens_train_only…`) |
| M12 no signal handlers in `run` | — | killed (`test_run_interrupt…`) |
| M13 worker skips the config-SHA check | — | killed (`test_worker_refuses_changed_config…`) |

## Before the run (checklist for root)

1. Codex review of `7f3d52c..5c9c8aa` (this review is same-model).
2. Decide N4/N5/N8 (doc wording / projection basis / env + host record) and commit them *before*
   the run; ship `git archive` of that commit, not `7f3d52c`.
3. Wrapper: `timeout --kill-after=30 1800` (or record the preflight duration) — N2.
4. A non-zero exit or `complete: false` counts as a failed attempt (≤ 2).
