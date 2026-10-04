# Amendment — tentative EXP-P36-001 DRAFT (robust clip): timing-only warm TRAIN probe (pre-registration)

Status: written 2026-10-04 (`date` 23:31 KST) by an Opus 5.5 author worker (v5 cycle-027, T106c) together with the
probe script, BEFORE any probe run. `author_only; same_model_review_pending; Codex_review_pending`. Code
`scripts/timing_probe_robust_clip.py` + tests `tests/test_timing_probe_robust_clip.py` + fake worker
`tests/fake_robust_clip_probe_worker.py` (commit `36d84ab`, base `08da77f`). Family config
`configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json` (sha `644239f3…470a`) stays DRAFT and unedited; the pre-registration,
every frozen model/registry/factory/runner/queue/scorer file and both earlier probe scripts stay unedited.

## Why

Same-model review a23e77b M1 (accepted in `DISPOSITION-robust-clip-wiring-review-a23e77b-2026-10-04.md`): a CUDA smoke
runs 8,192 TRAIN rows × 2 epochs and cannot project a full task. PREREG-DRAFT-EXP-P36-001 §4 gate (ii) therefore needs
a timing probe by the P35 Amendment-2 method for `residual_counts` and `robust_clip_residual_counts`, probe seed 39,
with `compute_stats` on the full TRAIN argument timed separately and peak host RSS printed. The P35 probe script is
P35-only (fixed family name, 5 arms, the `ple_arms_within` gate key, seed 34), so a new script is needed; it copies
the P35 method and parent/worker code with attribution and imports nothing from the P35 file.

## What is measured (all fixed now)

On newsrv GPU3 `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9` and GPU4 `GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99`
concurrently (one worker process per GPU, by UUID only; the probe refuses any other GPU set), each worker does the two
probed arms in config order (`residual_counts`, then `robust_clip_residual_counts`):

0. identity first (parent preflight and each worker): the config file must hash to the pinned DRAFT sha256
   `644239f3b2aaa4410391211ab1e4d972f13e1a6b025a026b058f0dff319c470a`, and both the registry pin and the file
   `pitchmdp/direct_robust_clip_model.py` must equal `a3d6cbcef8350586a23d1a6d35213e36ba8e0ccf21288288464f8ce7dcbf52fa`;
   only then the launch env, GPU binding, source hashes (= preflight), export and data.
1. startup (measured): `bind_gpu`, `verify_export` with full array hashes (same as a real task), `load_split(train)`.
2. per arm, prepare (measured): `arm_inputs` of the full TRAIN split; for the robust-clip arm also ONE
   `direct_robust_clip_model.compute_stats` call on those TRAIN inputs — the same call with the same arguments the
   frozen robust-clip `fit` makes (`_no_current_physics(train)`, the registry `context_continuous`, 41 columns). Its
   seconds are inside `prepare_seconds` and also recorded alone (`rc.compute_stats_seconds`). Recorded with it
   (TRAIN-derived, no label used): statistics sha256, rule counts (iqr / range / constant, sum 81), the `range` and
   `constant` column labels (review a23e77b open question 5), TRAIN rows and valid rows per past position, module sha.
3. per arm, warm-up: a fresh model from `run_cuda_direct.build_model(arm, 39, 'cuda')`, the arm's own frozen `fit`
   for **1 epoch** (all TRAIN rows, batch 1024), discarded.
4. per arm, measured: a NEW fresh model, the same frozen `fit` for **2 epochs**; recorded `report['seconds']`, wall
   seconds, `optimizer_updates`, peak device bytes (reset before each fit), peak host RSS. Per-epoch time =
   `max(wall_seconds, report['seconds']) / 2`. For the robust-clip arm the probe checks that the fit's own
   `rc_stats_sha256` and rule counts equal the prepare-step values (refuses otherwise).
5. per arm, predict (measured): `model.logits` over N_pred TRAIN rows, N_pred = manifest row count of
   temperature + blend + dev + mlb_dev (values discarded) — exactly the P35 probe's stand-in; no row of those splits
   is loaded.
6. memory: peak device bytes after each fit and after predict (`torch.cuda.max_memory_allocated`); peak host RSS
   `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` after startup, prepare, each fit and predict, recorded as
   `{ru_maxrss, unit, bytes}` with unit **KiB on Linux** (the server) and bytes on macOS. `ru_maxrss` is a
   process-lifetime peak: the robust-clip arm's value is the maximum over both arms of that worker (it runs second).

Validation stand-in: the first N_early TRAIN rows (N_early = manifest `earlystop.n`), as in P35. Patience 5 cannot stop
a 1- or 2-epoch fit.

**Probe seed 39 only** (any other seed is refused, and 39 is outside the family seeds 34–38 and every earlier block
0–33). Timing does not depend on the seed; the probe is not a fit of the family and no seed-39 artifact is kept.

**Launch env fixed (N8):** `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`; parent and worker refuse any
other value. Host snapshots (nproc, load average, `nvidia-smi` GPU memory/utilisation and compute apps) go to
`preflight.json` (`host_pre`) and `summary.json` (`host_post`).

## Data and outputs

Loaded: TRAIN only. Other splits are not loaded or parsed; their array bytes are hashed by `verify_export` for
integrity, and manifest row counts are read. The statistics use TRAIN only. No model saved, no calibration, no
probabilities, no logits, and no loss/history/best-epoch/score value written (every output is checked for such keys and
refused). Written files, in a NEW output directory whose name must contain `probe` and end `-timing` (an existing
directory is refused): `preflight.json`, `worker-<uuid>.json` ×2, `worker-<uuid>.log` ×2, `summary.json`, plus
`worker-<uuid>-error.json` on a worker failure. No `runs/<family>/<arm>/<seed>` task output is written.

## Projection (fixed now; measurement vs assumption named)

Per (GPU, probed arm):

    task_projected = startup (measured) + prepare (measured, incl. compute_stats for the robust-clip arm)
                     + 60 × per_epoch (measured, projected to the cap) + predict (measured)
                     + 30 s fixed allowance (assumption)

- Slower GPU per probed arm. `control` and `periodic_residual_counts` are NOT re-probed: they take the P35 probe
  projections **464 s** and **1,046 s** (`RESULT-P35-TIMING-PROBE-3b317f2-20261004.md:30-31`), fixed constants in the
  script with that citation (`fixed_task_seconds.measured_here: false`).
- Family = Σ over the 4 arms of 5 seeds × slowest task_projected (every fit at the cap; early stopping not credited).
- Conservative by construction: the robust-clip `fit` computes the statistics before the frozen fit's timer starts,
  so the measured wall time of each robust-clip fit already contains one `compute_stats`; the projection therefore
  carries ≈ 30 × compute_stats seconds on top of the one in prepare. Recorded per row
  (`compute_stats_seconds_inside_projection_not_credited`) and **not** subtracted; it never rescues a false gate.
- Context, not a measurement of these arms: P32 per-fit residual 163 s vs periodic 217 s
  (`RESULT-EXP-P32-001-20261004.md`), the reason the prereg assumed ≤ 1,046 s/task.

## Gates (fixed now; PREREG §4 (ii), unchanged)

`summary.json` `gate` (written only when `complete` is true) carries exactly:

1. `every_arm_within`: every one of the 4 arms ≤ 1,800 s on the slower GPU (the two constants included);
2. `family_within`: family ≤ 36,000 worker-s;
3. `probe_exit_zero`: both worker exit codes 0. The operator also checks the probe process exit code 0 (the parent
   exits 1 when `complete` is false).

All must hold (P34 N9: no single gate alone permits anything). Note: with 464 + 1,046 fixed, two arms ≤ 1,800 s give
at most 5 × 5,110 = 25,550 worker-s, so `family_within` can only fail together with `every_arm_within`. Arms within
10 % of 1,800 s are listed in `marginal_arms` (gate unchanged).

- All true → PREREG §4 (ii) is met. Registration still needs §4 (i) (CUDA smoke 20/20), the other before-registration
  items of the disposition, the Codex review and Song's explicit go. This amendment does not register anything.
- Any false → no registration; the probe record is kept; gates are not raised, the family is not silently shortened.
- Nothing in the probe reads a score; no timing, epoch or arm choice may use any score.

## Preconditions (all before the first attempt)

- **EXP-P35-001 is terminal** (complete or stopped; DRAFT `queue.start_after`) and no other family queue, smoke, score
  or probe runs on GPU3/4.
- **A distinct same-model review of this script, tests and amendment exists and is dispositioned**; the probe runs from
  the reviewed commit `<R>` (or a later commit whose diff to the reviewed one touches no script/config/model file).
- GPU3/4 idle by `nvidia-smi` (no compute app on 3/4); the common lock directory
  `/home/sds733/pitcheezy-autonomy/gpu-locks` (busy lock → refuse, nothing waits). Never concurrent with the P36 smoke.
- The probe must end in time for the P36 one-date rule (latest launch L = 16:30 KST on 2026-10-05, PREREG §5).

## Command template (for a later cycle; nothing here was run)

- Code: `git archive <R> configs experiments/pitchmdp | gzip -n` of the reviewed commit (archive sha recorded) → a NEW
  read-only `code-<commit7>` directory on the server root (must not exist before). Output: a NEW
  `runs/DRAFT-ROBUST-CLIP-probe-<commit7>-timing` (must not exist).
- Foreground bounded run from the server root (no background launch, no `nohup`):

      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
      timeout --kill-after=30 1800 /home/sds733/venv/bin/python \
        code-<commit7>/experiments/pitchmdp/scripts/timing_probe_robust_clip.py \
        --config code-<commit7>/configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json \
        --config-sha256 644239f3b2aaa4410391211ab1e4d972f13e1a6b025a026b058f0dff319c470a \
        --export export-49097109 \
        --export-manifest-sha256 49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578 \
        --output runs/DRAFT-ROBUST-CLIP-probe-<commit7>-timing \
        --lock-dir /home/sds733/pitcheezy-autonomy/gpu-locks \
        --gpu GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9 --gpu GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99 run

- Bounds (as P35 / Amendment 2): wrapper `timeout --kill-after=30 1800`; worker hard timeout 1,500 s (one shared
  deadline for both workers) + each worker's own 1,560 s alarm; the parent kills each worker's process group
  (SIGTERM, SIGKILL after 20 s) before releasing its GPU lock, on timeout, failure or SIGTERM/SIGHUP/SIGINT.
- Expected cost per worker: startup (array hashing; ≈ the P35 startup) + 2 × 3 TRAIN epochs + 3 TRAIN `compute_stats`
  (1 in prepare, 1 inside each of the 2 robust-clip fits) + 2 predict passes. At the P35 periodic per-epoch time
  (16.85 s, context only; these arms 미측정) a few minutes per GPU; full-TRAIN `compute_stats` time 미측정.
- Expected files: `preflight.json`, `worker-GPU-c334d905-….json`, `worker-GPU-ae803a24-….json`, the two
  `worker-<uuid>.log`, `summary.json` (with `gate`, `per_arm`, `slowest_task_seconds_projected`,
  `probed_task_seconds_projected`, `fixed_task_seconds`, `family_worker_seconds_projected`, `marginal_arms`).
  Copy all to the Mac evidence directory with sha256 (as P35), keep them read-only.
- **At most 2 attempts**, a second one only if attempt 1 is incomplete (`complete` false or a non-zero exit, e.g. an
  infrastructure failure). A complete attempt is final whatever its gate values. A failed attempt keeps its output
  (never reused or deleted) and is reported; a second attempt uses a NEW output
  `runs/DRAFT-ROBUST-CLIP-probe-<commit7>-a2-timing`. Two failures end the probe; the family then does not register.
- Review T109 m2 (disposition 2026-10-04 23:49 KST, before any run): `run()` creates the output directory before the
  pin/name/GPU/export checks (inherited from P35; script blob kept as reviewed). A refusal before `preflight.json` is
  written is NOT an attempt: the empty directory is kept (never deleted) and reported, and the next run still uses a
  NEW output name (`...-r2-timing`). The operator types the output name from this document, never a family `runs/` path.

## Exposure

Timing only. No DEV/MLB DEV row, label-derived value or score is touched; the result cannot use any score. Seed 39 is
not a family seed. The family's exposure statement (post-selection, repeated DEV, descriptive only, no 2026) is
unchanged.
