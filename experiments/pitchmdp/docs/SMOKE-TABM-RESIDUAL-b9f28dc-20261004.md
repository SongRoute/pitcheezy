# CUDA smoke — TabM-residual family, all 25 tasks (`b9f28dc`, 2026-10-04)

Label: smoke only. Config DRAFT `5e3ce3ad…6d34`, NOT registered, no full fit, no score, no
prediction read. Code `same_model_reviewed; Codex_review_pending` (wiring review disposition
`b9f28dc`). v3 coordinator cycle-007, T35b.

## What ran

- Archive `git archive b9f28dc configs experiments/pitchmdp | gzip -n` sha256
  `f032ecef94a38cd3a48981cc89745fd142d3fc93aa9aebf4b69c2396f59d740f` → server
  `cuda-tabm-residual-b9f28dc.tar.gz` (same sha, a-w) → NEW read-only `code-b9f28dc`.
  On server: config `5e3ce3ad…`, queue `3b9adb3d…57a1`, runner `d8d1fab6…416a`, module
  `6a9daff6…87cf`, export manifest `49097109…f578` (= worktree).
- Precheck 14:14:08 KST: GPU3/4 2 MiB 0 %, no compute app on 3/4, no cuda_direct/score process.
- Foreground, bounded (no background launch, no launch-record file, no remote wait loop):
  `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout --kill-after=30 540 /home/sds733/venv/bin/python code-b9f28dc/experiments/pitchmdp/scripts/cuda_direct_queue.py --config code-b9f28dc/configs/DRAFT-CUDA-TABM-RESIDUAL-20261004.json --config-sha256 5e3ce3ad… --export export-49097109 --export-manifest-sha256 49097109… --output runs/DRAFT-TABM-RESIDUAL-smoke --queue-dir queue-tabm-residual-smoke-001 --lock-dir /home/sds733/pitcheezy-autonomy/gpu-locks --gpu GPU-c334d905-… --gpu GPU-ae803a24-… --smoke run`
  14:14:30 → 14:16:43 KST, exit 0, `CUDA_QUEUE_DONE {"done": 25} comparison_allowed True`
  (completeness only). GPU3/4 back to 2 MiB 0 % at 14:17:44, no queue process.
- `summary.json` sha `4e6c90ae…efe32c`: family_seconds 250.2 / 36000, on_time. No `failed/`;
  25/25 done, 1 attempt each.
- Identity: preflight 24 source hashes = blobs of `b9f28dc` (0 mismatch, Mac check); 575
  per-task source-hash checks vs `code-b9f28dc` (0 mismatch); every task config `5e3ce3ad`,
  `smoke: true`, device cuda, bound UUID = one of GPU3/4.
- Evidence copy (79 files + SHA256SUMS): `/Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/t35-smoke/evidence/`.

## Per arm (5 seeds 24–28 each)

| arm | params | peak MiB | smoke fit s | train s / 16 updates | non-train s |
|---|---|---|---|---|---|
| control | 141,834 | 86.3 | 2.77–3.62 | 1.04–1.35 | 1.93–2.53 |
| periodic_residual_counts | 244,898 | 133.5 | 3.80–4.09 | 1.49–1.59 | 2.50–2.71 |
| tabm_counts | 313,624 | 205.9 | 3.18–4.03 | 1.44–1.57 | 1.93–2.70 |
| tabm_counts_d10 | 313,624 | 206.7 | 3.45–3.92 | 1.28–1.64 | 2.20–2.59 |
| tabm_residual_counts | 260,760 | 205.9 | 2.93–4.13 | 1.17–1.64 | 1.91–2.68 |

All 25 `epoch_cap` records: cap 2, epochs_run 2, best 2, `stopped_by: cap` (expected for the
2-epoch smoke). Params of the four inherited arms equal their P31/P32/P33 values.

## Pre-registered timing gate: TRIGGERED

Fixed formula (design note, Amendment 1): `(smoke train s / smoke updates) × 1,224 × 60 +
smoke non-train s`, per GPU, slowest of the 5 seeds.

- tabm_residual_counts: GPU ae803a24 (seeds 24, 26, 28) slowest **7,549 s**; GPU c334d905
  (seeds 25, 27) slowest **7,119 s**. Both > 1800 s → the rule says **redesign before
  registration**. Not registered. Nothing was shortened.

Same formula applied to arms whose full fits on this export already exist (calibration of the
formula, not of the new arm; full-fit numbers are `report.seconds / report.optimizer_updates`
from completed tasks):

| arm | formula on this smoke | measured full per-update s | full-fit s at cap 60 implied | overestimate |
|---|---|---|---|---|
| control | 4,779–6,206 | 0.0056–0.0059 (EXP-P31-001, 3 fits) | ≈ 420–440 | ≈ 11–14× |
| periodic_residual_counts | 6,856–7,293 | 0.0134–0.0139 (EXP-P32-001, 3 fits) | ≈ 990–1,020 | ≈ 7× |
| tabm_counts | 6,609–7,202 | 0.0099–0.0101 (EXP-P31-001, 3 fits) | ≈ 730–750 | ≈ 9–10× |

Non-train seconds of those full tasks: 6.1–7.9 s (vs 1.9–2.7 s in smoke); this is review N2's
cross-check, printed here and not part of the rule.

Reading: the smoke's 16 updates are dominated by CUDA warm-up (review N2 predicted an
overestimate), so the fixed formula overstates every arm by roughly 7–14×. The new arm's full
per-update time is **미측정**. Bounds from its parents (assumption, not measurement): between
TabM 0.0100 and an additive guess periodic_residual 0.0139 + TabM overhead 0.0043 = 0.0182
s/update → about 730–1,340 s at cap 60, under 1800 s. This is why the gate is put to a
decision rather than silently overridden.

## Decision needed before any registration (Song / root)

The gate is pre-registered; changing it after seeing smoke timings is a design amendment, even
though no score or prediction has been read. Options are in the coordinator ledger (T35b).
