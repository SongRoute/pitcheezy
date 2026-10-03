# TabM-mini CUDA smoke record — source 2d98f95 (2026-10-04)

Status: DRAFT config, smoke only. `same_model_reviewed; Codex_review_pending`. Not registered, no full fit, no score, no decision.

## Identity
- Source: commit `2d98f95` (`git archive --format=tar.gz 2d98f95 configs experiments/pitchmdp`) → newsrv `cuda-tabm-2d98f95.tar.gz` sha256 `31102ec02f3cba3bf6487390dfb494999a3d19e22b61a71c26e40964ddec5ee6`, extracted to new read-only `code-2d98f95`.
- Config `configs/DRAFT-CUDA-TABM-MINI-20261004.json` sha256 `334528a5904bd78f9f4a70955114d3434f3085b99a079a699daff07d252eae70`; export `export-49097109` manifest `49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578`.
- Preflight `queue-tabm-smoke-001/preflight-87724.json`: runner sha `ad5f5897…f9af` and all 16 pinned source hashes equal the blobs of `2d98f95` (checked on the Mac with `git show 2d98f95:experiments/pitchmdp/<path> | sha256`; 0 mismatches).

## Precondition (config `queue.start_after`)
- 08:23:13 KST: `queue-p30-001` = {done: 50}; no `queue-p30-002`; no other cuda_direct queue/runner process; GPU3/GPU4 2 MiB, no compute apps on their UUIDs.

## Launch
- 08:23:19 KST, pid 87722, `timeout --kill-after=60 3600`, PYTHONDONTWRITEBYTECODE=1, OMP/MKL threads 4; record `tabm-smoke-launch.json`; output `runs/DRAFT-TABM-smoke`; queue `queue-tabm-smoke-001`; common `--lock-dir /home/sds733/pitcheezy-autonomy/gpu-locks`; GPUs by UUID GPU-c334d905… (GPU3) and GPU-ae803a24… (GPU4).
- Exit 08:24:42 KST: `CUDA_QUEUE_DONE {"done": 15} comparison_allowed True` (queue completeness only). `summary` 08:24:52: done 15, family_seconds 150.1, budget 36000, on_time true. `runs/DRAFT-TABM-smoke/failed` absent. One attempt per task.

## Per task (SMOKE limits: 8192 train / 2048 earlystop / 256 temperature / 256 predict rows, 2 epochs)
```
control-smoke 13 fit_s=2.56 total_s=2.85 peak_MiB=86.3 params=141834 T=0.5000 epochs=2 state=None
control-smoke 14 fit_s=3.56 total_s=3.82 peak_MiB=86.3 params=141834 T=0.5000 epochs=2 state=None
control-smoke 15 fit_s=3.50 total_s=3.77 peak_MiB=86.3 params=141834 T=0.5004 epochs=2 state=None
periodic_counts-smoke 13 fit_s=3.66 total_s=3.84 peak_MiB=134.2 params=288802 T=1.1520 epochs=2 state=None
periodic_counts-smoke 14 fit_s=4.10 total_s=4.29 peak_MiB=134.2 params=288802 T=1.0937 epochs=2 state=None
periodic_counts-smoke 15 fit_s=3.94 total_s=4.12 peak_MiB=134.2 params=288802 T=1.0959 epochs=2 state=None
periodic_counts_d10-smoke 13 fit_s=3.98 total_s=4.16 peak_MiB=134.2 params=288802 T=1.1483 epochs=2 state=None
periodic_counts_d10-smoke 14 fit_s=3.72 total_s=3.85 peak_MiB=134.2 params=288802 T=1.0878 epochs=2 state=None
periodic_counts_d10-smoke 15 fit_s=4.05 total_s=4.23 peak_MiB=134.2 params=288802 T=1.0897 epochs=2 state=None
tabm_counts-smoke 13 fit_s=4.03 total_s=4.22 peak_MiB=205.9 params=313624 T=0.5000 epochs=2 state=None
tabm_counts-smoke 14 fit_s=4.01 total_s=4.20 peak_MiB=205.9 params=313624 T=0.5000 epochs=2 state=None
tabm_counts-smoke 15 fit_s=3.99 total_s=4.17 peak_MiB=205.9 params=313624 T=0.5000 epochs=2 state=None
tabm_counts_d10-smoke 13 fit_s=4.01 total_s=4.15 peak_MiB=206.7 params=313624 T=0.5000 epochs=2 state=None
tabm_counts_d10-smoke 14 fit_s=3.90 total_s=4.09 peak_MiB=206.7 params=313624 T=0.5000 epochs=2 state=None
tabm_counts_d10-smoke 15 fit_s=3.88 total_s=4.07 peak_MiB=206.7 params=313624 T=0.5000 epochs=2 state=None
failed: none
```
(`state=None`: the reader looked for a state/status key the runner does not write; completion is from the queue status and `summary`.)

## Observations (smoke only; not performance evidence)
- Peak CUDA allocation: control 86.3 MiB, periodic single 134.2 MiB, TabM k=8 205.9–206.7 MiB at batch 1024. Parameters: control 141,834; periodic 288,802; TabM 313,624.
- Smoke fit seconds (2 epochs, 8192 rows) are wiring times, NOT full-fit estimates; full-fit time remains 미측정 (review F6 asked for measured numbers: memory now measured at smoke size, full time still not).
- Delivery temperature hits the lower bound 0.5 for control and both TabM arms in smoke; periodic arms ≈1.09–1.15. The P30 smoke control also hit 0.5 on all 5 seeds while full P30 control/periodic_counts seed 3 gave 0.999/1.0217, so this is consistent with 2-epoch under-training. A full run must report any temperature at a bound per arm; the bounds belong to the shared calibration chain and are not changed here.

## Not done
- No registration, no full run, no score, no comparison. Registration (REGISTERED + execution.enabled in a new config identity, start_after re-check, F7 14-key scorer test) is root's step.
