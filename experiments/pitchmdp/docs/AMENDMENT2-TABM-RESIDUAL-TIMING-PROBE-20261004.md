# Amendment 2 — TabM-residual family: timing-only warm steady-state TRAIN probe (pre-registration)

Status: written 2026-10-04 16:00 KST (`date` 16:01:35 at commit) (v3 coordinator cycle-021, T49) together with the probe
script, BEFORE any probe run. Code `scripts/timing_probe_tabm_residual.py` + tests
`tests/test_timing_probe_tabm_residual.py`. Not reviewed yet: a separate Opus review of method
and source is required before the probe runs (same-model review; Codex review pending).
Family config `configs/DRAFT-CUDA-TABM-RESIDUAL-20261004.json` (sha 5e3ce3ad…6d34) stays DRAFT and
unedited; the design note and every frozen module/runner/queue file stay unedited.

## Why

The pre-registered gate of Amendment 1 (smoke formula, `SMOKE-TABM-RESIDUAL-b9f28dc-20261004.md`)
projected tabm_residual_counts at 7,119–7,549 s > 1800 s → "redesign before registration". That
smoke record and its projection stay as they are; this amendment does not override it. The same
formula overstates the three arms whose full fits exist by about 7–16× (16 smoke updates are
dominated by CUDA warm-up), but parents' full-fit numbers are not measurements of the new arm.
Root disposition (next-work-ledger, 2026-10-04 15:45 KST): measure, do not borrow.

## What is measured (all fixed now)

On newsrv GPU3 `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9` and GPU4
`GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99` concurrently (one worker process per GPU, as the
family queue runs two fits at once), each worker does all 5 arms in config order:

1. startup (measured): `bind_gpu`, `verify_export` with full array hashes (same as a real task),
   `load_split(train)`.
2. per arm, prepare (measured): `arm_inputs` of the full TRAIN split.
3. per arm, warm-up: a fresh model from `run_cuda_direct.build_model(arm, seed 1000, 'cuda')`, the
   arm's own frozen `fit` for **1 epoch** (one pass over all TRAIN rows, batch 1024), discarded.
4. per arm, measured: a NEW fresh model, the same frozen `fit` for **2 epochs**; recorded
   `report['seconds']`, `optimizer_updates`, peak device bytes. Per-epoch time =
   `report['seconds'] / 2` (includes net build, TRAIN tensor upload and per-epoch validation, so it
   over-states a steady epoch; conservative).
5. per arm, predict (measured): `model.logits` over N_pred TRAIN rows, N_pred = manifest row
   count of temperature + blend + dev + mlb_dev (values discarded).

Validation stand-in: the frozen loops evaluate the early-stop rows after every epoch. The probe
passes the first N_early TRAIN rows (N_early = manifest `earlystop.n`) as that argument, so the
validation cost has the real row count without opening the early-stop split. Patience 5 cannot
stop a 1- or 2-epoch fit.

Data opened: TRAIN only. Not opened: earlystop, temperature, blend, dev, mlb_dev. No model saved,
no calibration, no probabilities, and no loss/history/best-epoch value written (the output is
checked for such keys and refused). Probe seed 1000 is not a registered seed; the probe is not a
fit of the family. Batch size, learning rate and patience come from the config `training` block.

## Projection (fixed now; measurement vs assumption named)

Per (GPU, arm):

    task_projected = startup (measured) + prepare (measured) + 60 × per_epoch (measured, projected to cap)
                     + predict (measured) + 30 s fixed allowance (assumption)

The 30 s allowance covers process start/imports, temperature fit, save, artifact hashing and
rename (assumption; the measured non-fit seconds of P31/P32 full tasks were 6.1–7.9 s, cited
as context only). Every fit is projected at the 60-epoch cap (real fits may stop earlier by
patience; not credited). Per arm the slower GPU is used. Family =
Σ_arms 5 seeds × slowest task_projected.

## Decision rule (fixed now)

- If `tabm_residual_task_seconds_projected` ≤ 1800 s AND every arm ≤ 1800 s AND family ≤ 36000
  worker-s: register a NEW family config `EXP-P34-001` derived from DRAFT 5e3ce3ad (status
  REGISTERED, execution enabled, realistic family deadline on the launch day; unchanged: 5 arms,
  seeds 24–28, 25 fits, epoch cap 60, task hard runtime ≤ 3600 s, ≤ 2 attempts, family budget
  36000, contrasts, calibration chain, scientific contracts), new output and queue dirs, then
  launch after the usual preflight/identity checks.
- Otherwise: no registration. Either an explicitly new bounded redesign (new pre-registration,
  new smoke) or end this candidate, recorded.
- No timing/epoch selection from scores; nothing in the probe reads a score.

## Bounds and stop

- Expected cost: per worker ≈ startup (array hashing) + 5 × 3 epochs + 5 predict passes. From
  parents' measured per-update times (context, not a measurement of the new arm) about 3–6 min
  per GPU; worker hard timeout 1500 s; foreground command bounded by `timeout --kill-after=30 1700`.
- Preconditions: GPU3/4 free (nvidia-smi + no queue/probe process), the common lock directory
  `/home/sds733/pitcheezy-autonomy/gpu-locks` (busy lock → refuse, nothing waits).
- Code shipped as `git archive` of the reviewed commit → a NEW read-only `code-<sha>` dir; output
  a NEW `runs/DRAFT-TABM-RESIDUAL-probe-<sha>-timing` dir (must not exist).
- At most 2 attempts. A failed attempt is kept (worker error JSON + log) and reported. Two
  failures end the probe; the candidate is then redesigned or ended as above.

## Exposure

Timing only. No DEV/MLB DEV row or score is touched. The family's own exposure statement
(post-selection, repeated DEV, descriptive only, no 2026) is unchanged.
