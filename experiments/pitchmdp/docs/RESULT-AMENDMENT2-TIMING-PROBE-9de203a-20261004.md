# Amendment 2 timing probe — result (9de203a, 2026-10-04) and EXP-P34-001 registration

Label: timing measurement + projection only. No model of the family was fit, saved, calibrated or scored;
only TRAIN was loaded (other split files byte-hashed for integrity). Pre-registration:
`AMENDMENT2-TABM-RESIDUAL-TIMING-PROBE-20261004.md` (Codex PM disposition 16:16 KST at 9de203a).

## Run (attempt 1 of ≤ 2; no failed attempt)

- Code: `git archive 9de203a configs experiments/pitchmdp | gzip -n` sha 60281296…03a4 → newsrv
  `cuda-timing-probe-9de203a.tar.gz` (0444) → NEW read-only `code-9de203a`. Config
  `DRAFT-CUDA-TABM-RESIDUAL-20261004.json` 5e3ce3ad…6d34 (unedited), probe script 4bef4663…1bc2.
  Frozen model/runner/queue/config files identical to b9f28dc (git diff b9f28dc..9de203a = 5 added files only).
- Command (foreground, from the server root):
  `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 timeout --kill-after=30 1800
  /home/sds733/venv/bin/python code-9de203a/experiments/pitchmdp/scripts/timing_probe_tabm_residual.py --config
  code-9de203a/configs/DRAFT-CUDA-TABM-RESIDUAL-20261004.json --config-sha256 5e3ce3ad… --export export-49097109
  --export-manifest-sha256 49097109… --output runs/DRAFT-TABM-RESIDUAL-probe-9de203a-timing --lock-dir
  /home/sds733/pitcheezy-autonomy/gpu-locks --gpu GPU-c334d905-… --gpu GPU-ae803a24-… run`
- 16:19:14 → 16:22:45 KST, exit 0, `complete: true`, worker exits 0/0 (GPU4 207.2 s, GPU3 185.0 s).
- Host: nproc 64; load average before 27.59/26.35/24.91, after 28.60/27.47/25.65 (other users' work);
  GPU0–2 busy with other users' processes (not touched), GPU3/4 2 MiB / 0 % before and after.
- Evidence (copied, a-w, SHA256SUMS): `/Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/t50-timing-probe/evidence/`
  summary.json 160ed14b4643d0e3e8d230fe9ada0fea7ee1843d16b8740e712b5619ded5930e, preflight c34e84a3…, workers
  5e21bbb6… (GPU4) / 00c01598… (GPU3); wrapper log with pre/post nvidia-smi `attempt1-wrapper.log`.

## Measured per epoch (max(wall, fit report)/2; warm, 1,224 updates/epoch, batch 1024) and projection

| arm | params | s/epoch GPU3 / GPU4 | task projected (slower GPU, 60-epoch cap + 30 s assumed) | peak allocated |
|---|---|---|---|---|
| control | 141,834 | 6.20 / 7.21 | 469 s | 1.41 GiB |
| periodic_residual_counts | 244,898 | 14.38 / 15.98 | 994 s | 1.51 GiB |
| tabm_counts | 313,624 | 10.93 / 10.54 | 691 s | 1.59 GiB |
| tabm_residual_counts | 260,760 | 14.81 / 15.67 | 975 s | 1.54 GiB |
| tabm_counts_d10 | 313,624 | 10.85 / 10.98 | 694 s | 1.59 GiB |

Family (5 seeds × slowest per arm, every fit at the cap) = 19,118 worker-s.

## Gates (all three plus complete/zero exit, N9)

- complete true, exit codes 0/0
- tabm_residual_counts 975 s ≤ 1800 → true (54 % of the gate; not within 10 %, so not marginal)
- every arm ≤ 1800 (max 994 s) → true
- family 19,118 ≤ 36,000 worker-s → true

The original smoke-formula projection (7,119–7,549 s, `SMOKE-TABM-RESIDUAL-b9f28dc-20261004.md`) stays recorded
unchanged; it was dominated by CUDA warm-up of 16 smoke updates. The probe is a separate measurement.
Limits: two-GPU concurrent probe on a shared host (load ≈ 28/64); per-epoch time includes net build,
TRAIN upload and per-epoch validation (conservative); real fits are not credited for early stopping.

## Registration and launch (pre-registered rule applied)

- `configs/EXP-P34-001.json` sha 2bfd0e03aa8cdba6923a237b748ec7f59ca23c44331e1ba1149da23ef2111909 (commit a792356):
  DRAFT 5e3ce3ad with only experiment_id/status/execution/decision/registration and queue
  batch/date_note/no_new_fit_after 21:30/kill_own_after 22:30/aggregate_by 23:00 changed (launch day 2026-10-04;
  projected wall ≈ 19,118 / 2 ≈ 2.7 h). Unchanged: 5 arms, seeds 24–28, 25 fits, epoch cap 60, task timeout 3600,
  family budget 36000, ≤ 2 attempts, contrasts, calibration, start_after and stop_rule.
  Test `tests/test_cuda_tabm_residual_registered.py` 4 passed (negative control: 2 fail); with neighbours 123 passed.
- Pipeline `docs/launch/run-p34-pipeline.py` sha ab517bb4…bc22: re-checks probe summary sha + three gates, 25/25
  smoke identities, P30–P33 terminal, no P30-002, no other queue/probe; runs smoke-verified `code-b9f28dc` with
  OMP/MKL/OPENBLAS 2 (N8); scores into NEW `scores/EXP-P34-001`.
- Launched 16:23:55 KST: timeout 259096 / pipeline 259097 / queue 259172; preflight-259172 24 source hashes ==
  b9f28dc blobs (0 mismatches), runner d8d1fab6… == b9f28dc. Launch record kept on the Mac
  (`t50-timing-probe/p34-launch-record.json`), no server launch json.

Status: RUNNING (not a result). same_model_reviewed; Codex_review_pending. Exploratory, repeated DEV, post-selection;
no promotion or verdict; D131/D155 unchanged.
