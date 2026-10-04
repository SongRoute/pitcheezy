# Amendment — EXP-P35-001 DRAFT (PLE embedding): timing-only warm TRAIN probe (pre-registration)

Status: written 2026-10-04 (~19:35 KST) by an Opus 5.5 author worker together with the probe script, BEFORE any
probe run. `author_self_tested; same_model_review_pending; Codex_review_pending`. Code
`scripts/timing_probe_ple_residual.py` + tests `tests/test_timing_probe_ple_residual.py` (commit `12925d9`).
A separate review of method and source is required before the probe runs (DRAFT §10.7). Family config
`configs/DRAFT-CUDA-PLE-RESIDUAL-20261004.json` (sha `f0ae4e91…b9b2`) stays DRAFT and unedited; every frozen
model/registry/factory/runner/queue/scorer file stays unedited.

## Why

DRAFT §6 "Timing gate before registration — DECIDED (review B2 option 1)": the only projection method is the P34
Amendment-2 method (`AMENDMENT2-TABM-RESIDUAL-TIMING-PROBE-20261004.md`, final reviewed method at `9de203a`, incl.
N4/N5 max(wall, report)/2 and the N8 env/host records); gates are not raised. The 25/25 CUDA smoke
(`SMOKE-PLE-8864389-20261004.md`) times are dominated by warm-up and are not a projection. The P34 probe script is
P34-only (fixed family name, arms and the `tabm_residual` gate key), so a new script is needed; it copies the P34
method and parent/worker code with attribution and imports nothing from the P34 file.

## What is measured (all fixed now)

On newsrv GPU3 `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9` and GPU4 `GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99`
concurrently (one worker process per GPU, as the family queue runs two fits at once; the probe refuses any other GPU
set — exactly these two allow-listed UUIDs), each worker does all 5 arms in config order
(`control`, `periodic_residual_counts`, `ple_residual_counts`, `tabm_residual_counts`, `ple_tabm_residual_counts`):

1. startup (measured): `bind_gpu`, `verify_export` with full array hashes (same as a real task), `load_split(train)`.
2. per arm, prepare (measured): `arm_inputs` of the full TRAIN split; for the two PLE arms also ONE
   `direct_ple_model.compute_bins` call on those TRAIN inputs — the same call with the same arguments the frozen PLE
   `fit` makes (`_no_current_physics(train)`, the arm's `context_continuous`, n_bins 48). Its seconds are inside
   `prepare_seconds` and are also recorded alone (`ple.bins_seconds`). Recorded with it (TRAIN-derived, no label
   used): bins sha256, ΣT_j, number of T_j = 0 columns, valid rows per past position.
3. per arm, warm-up: a fresh model from `run_cuda_direct.build_model(arm, 34, 'cuda')`, the arm's own frozen `fit`
   for **1 epoch** (all TRAIN rows, batch 1024), discarded.
4. per arm, measured: a NEW fresh model, the same frozen `fit` for **2 epochs**; recorded `report['seconds']`,
   wall seconds, `optimizer_updates`, peak device bytes. Per-epoch time = `max(wall_seconds, report['seconds']) / 2`.
   For PLE arms the probe checks that the fit's own bins hash equals the prepare-step bins hash (refuses otherwise).
5. per arm, predict (measured): `model.logits` over N_pred TRAIN rows, N_pred = manifest row count of
   temperature + blend + dev + mlb_dev (values discarded).

Validation stand-in: the frozen loops evaluate the early-stop rows after every epoch. The probe passes the first
N_early TRAIN rows (N_early = manifest `earlystop.n`) as that argument, so the validation cost has the real row count
without opening the early-stop split. Patience 5 cannot stop a 1- or 2-epoch fit.

**Probe seed 34** (outside the family seeds 29–33; refused if inside them). Timing does not depend on the seed; the
probe is not a fit of the family and no seed-34 artifact is kept.

**Launch env fixed (N8):** `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2` (= the later family queue);
parent and worker refuse any other value. Host snapshots (nproc, load average, `nvidia-smi` GPU memory/utilisation
and compute apps) are written to `preflight.json` (`host_pre`) and `summary.json` (`host_post`).

## Data and outputs

Loaded: TRAIN only. Other splits are not loaded or parsed; their array bytes are hashed by `verify_export` for
integrity, and manifest row counts are read. The bins use TRAIN only (the frozen fit's TRAIN argument). No model
saved, no calibration, no probabilities, no logits, and no loss/history/best-epoch/score value written (every output
is checked for such keys and refused). Written files: `preflight.json`, `worker-<uuid>.json`, `worker-<uuid>.log`,
`summary.json` (plus `worker-<uuid>-error.json` on failure) in a NEW output directory whose name must contain `probe`
and end `-timing`; an existing directory is refused. No `runs/<family>/<arm>/<seed>` task output is written.

## Projection (fixed now; measurement vs assumption named)

Per (GPU, arm):

    task_projected = startup (measured) + prepare (measured, incl. TRAIN binning for PLE arms)
                     + 60 × per_epoch (measured, projected to the cap) + predict (measured)
                     + 30 s fixed allowance (assumption)

- Measurement: startup, prepare, bins seconds, the 1+2 warm TRAIN epochs, the TRAIN forward pass, peak memory.
- Projection: 60 × per_epoch (every fit at the cap; early stopping not credited), slower GPU per arm, family =
  Σ_arms 5 seeds × slowest task_projected.
- Assumption: the 30 s allowance (process start/imports, temperature fit, save, artifact hashing, rename; P34 used the
  same value).
- Conservative by construction: the frozen PLE `fit` recomputes the bins before its own timer starts, so the measured
  wall time of each PLE fit already contains one binning; per_epoch therefore carries ≈ bins/2 and the projection
  ≈ 30 × bins seconds on top of the binning in prepare. This is recorded per row
  (`ple_binning_seconds_inside_projection_not_credited`) and **not** subtracted.
- Context, not a measurement of the PLE arms: P34 probe projections control 469 s, periodic_residual_counts 994 s,
  tabm_residual_counts 975 s (`RESULT-AMENDMENT2-TIMING-PROBE-9de203a-20261004.md`).

## Gates (fixed now; DRAFT §6, unchanged)

All of the following must hold; no single gate alone permits anything (P34 N9):

1. `complete` true and both worker exit codes 0;
2. `gate.ple_arms_within`: both PLE arms (`ple_residual_counts`, `ple_tabm_residual_counts`) ≤ 1,800 s
   (the P35 counterpart of the P34 third gate `tabm_residual_within`, which named the family's new arm);
3. `gate.every_arm_within`: every arm ≤ 1,800 s;
4. `gate.family_within`: family ≤ 36,000 worker-s.

Arms whose projection is within 10 % of 1,800 s are listed in `marginal_arms` and reported as marginal; the gate
value is not changed.

The per-row `ple_binning_seconds_inside_projection_not_credited` is reported next to any PLE projection but never
subtracted and never rescues a false gate. If `ple_arms_within` is false, the record is reported as measured; a later
redesign may move the binning cost out of the per-epoch measurement only through a new reviewed amendment, never by
re-reading this probe. (Review F1, adopted before any run.)

- All true → the probe gate of DRAFT §6 is met. Registration still needs the other before-registration items
  (wiring review "Before registration": pre-score extractor §8, re-dated deadlines from this probe, explicit go
  from Song as `결정 필요`). This amendment does not register anything.
- Any false → no registration; the probe record is kept; the family is redesigned or dropped (not re-probed until
  a redesign passes its own review). Gates are not raised and the family is not silently shortened.
- Nothing in the probe reads a score; no timing, epoch or arm choice may use any score.

## Bounds, preconditions and the command

- Expected cost: per worker ≈ startup (array hashing) + 5 × 3 TRAIN epochs + 6 TRAIN binnings (2 in prepare, 1 inside
  each of the 4 PLE fits) + 5 predict passes (review F2); at the
  P34 per-epoch times (context; PLE arms 미측정) about 4–6 min per GPU. Worker hard timeout 1,500 s (+ own
  1,560 s alarm); the parent kills each worker's process group (SIGTERM, SIGKILL after 20 s) before releasing its
  GPU lock, on timeout, failure or SIGTERM/SIGHUP/SIGINT.
- Preconditions: EXP-P30-002 and every other family queue terminal; GPU3/4 idle (`nvidia-smi`, no queue/score/probe
  process); the common lock directory `/home/sds733/pitcheezy-autonomy/gpu-locks` (busy lock → refuse, nothing
  waits).
- Code: `git archive <R> configs experiments/pitchmdp | gzip -n` of the reviewed commit `<R>` → a NEW read-only
  `code-<R>` directory on the server (archive sha recorded). Output: a NEW `runs/DRAFT-PLE-RESIDUAL-probe-<R>-timing`
  (must not exist).
- Command (foreground, from the server root; no background launch):

      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
      timeout --kill-after=30 1800 /home/sds733/venv/bin/python \
        code-<R>/experiments/pitchmdp/scripts/timing_probe_ple_residual.py \
        --config code-<R>/configs/DRAFT-CUDA-PLE-RESIDUAL-20261004.json \
        --config-sha256 f0ae4e9116d3efe90d119fcc84616172bde183a981206caacc4804d68a31b9b2 \
        --export export-49097109 \
        --export-manifest-sha256 49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578 \
        --output runs/DRAFT-PLE-RESIDUAL-probe-<R>-timing \
        --lock-dir /home/sds733/pitcheezy-autonomy/gpu-locks \
        --gpu GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9 --gpu GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99 run

  The script refuses any config other than `DRAFT-CUDA-PLE-RESIDUAL-*` hashing to exactly
  `f0ae4e91…b9b2`, any other GPU set, any other env, and an existing output.
- At most 2 attempts, and a second attempt only if attempt 1 is incomplete (`complete` false or a non-zero exit, e.g.
  an infrastructure failure). A complete attempt is final whatever its gate values (DRAFT §6: the probe runs once;
  review F3). A failed attempt keeps its output (worker error JSON + logs + summary when the parent was not
  interrupted; review F8) and is reported; a second
  attempt uses a NEW output name (`runs/DRAFT-PLE-RESIDUAL-probe-<R>-a2-timing`). Two failures end the probe; the
  family is then redesigned or dropped as above.

## Exposure

Timing only. No DEV/MLB DEV row, label-derived value or score is touched; the result cannot use any score. The
family's exposure statement (post-selection, repeated DEV, descriptive only, no 2026) is unchanged.
