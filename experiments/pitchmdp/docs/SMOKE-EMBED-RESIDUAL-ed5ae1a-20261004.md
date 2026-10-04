# Embedded-residual CUDA smoke record — source ed5ae1a (2026-10-04)

Status: DRAFT config, smoke only. `same_model_reviewed; Codex_review_pending`. Not registered. No full fit, no score, no decision.

## Identity
- Source: commit `ed5ae1a` (`git archive --format=tar.gz ed5ae1a configs experiments/pitchmdp`) → newsrv `cuda-embed-ed5ae1a.tar.gz` sha256 `7b20c0fd7ddcdf9bc5f05f6ca13e6b0d564aa7470a88fc94858a4837cd9347c3` (a-w), extracted to new read-only `code-ed5ae1a`.
- Config `configs/DRAFT-CUDA-EMBED-RESIDUAL-20261004.json` sha256 `bb0ce6209db7b5b6bbc9243cb57e7ba4913dc1cf567756c67feacf2f508ac14e` (verified inside the archive). Config digest in task identities: `76e923e2834a…`.
- Export `export-49097109`, manifest `49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578` (verified).
- Preflight `queue-embed-smoke-001/preflight-116450.json`: 19 source hashes equal the blobs of `ed5ae1a`. Checked on the Mac with `git show ed5ae1a:experiments/pitchmdp/<path>`; 0 mismatches.
- Every task's `identity.json`: 18 source hashes equal the files in `code-ed5ae1a`; 0 mismatches. Seed, smoke flag and config digest match.

## Precondition (config `queue.start_after`)
- 09:38:36 KST and 09:46:07 KST (server `date`): GPU3 and GPU4 at 2 MiB, 0 %. No compute apps on their UUIDs. No cuda_direct queue or runner process. No new queue directory, so no root queue was running.

## Launch
- 09:46:07 KST (`2026-10-04T00:46:07Z`).
  - pid 116448, `timeout --kill-after=60 3600`.
  - Environment: PYTHONDONTWRITEBYTECODE=1, OMP and MKL threads 4.
  - Record `embed-smoke-launch.json`, sha `93e93afb…ae22`, a-w.
- Output `runs/DRAFT-EMBED-RESIDUAL-smoke`, queue `queue-embed-smoke-001`, common `--lock-dir /home/sds733/pitcheezy-autonomy/gpu-locks`. GPUs by UUID: GPU-c334d905… (GPU3) and GPU-ae803a24… (GPU4).
- Exit about 09:47:39 KST with `CUDA_QUEUE_DONE {"done": 18} comparison_allowed True`. That message only means the queue completed.
- `summary.json` at 09:47:39 KST (sha `3b448c43…645c`): done 18, family_seconds 180.1 of a 36000 budget, on_time true, priority1_complete true.
- `failed/` is absent. Every task ran in one attempt. GPU3/4 were back at 2 MiB afterwards.

## Per task
SMOKE limits: 8192 train / 2048 earlystop / 256 temperature / 256 predict rows, 2 epochs.

Columns: arm seed | params | context width | peak CUDA MiB | fit s | total s | delivery T | GPU
```
control 16/17/18                  141834  52   86.3  3.49/3.53/3.62  3.78/3.82/3.92  0.5/0.5/0.5           GPU3
residual_counts 16/17/18          170762  53   87.1  3.96/3.88/3.99  4.15/4.03/4.19  0.8341/0.8187/0.8610  GPU4
periodic_counts 16/17/18          288802  53  134.2  4.13/3.89/4.07  4.31/4.07/4.25  1.1370/1.1776/0.8850  GPU3
periodic_counts_d10 16/17/18      288802  53  134.2  4.04/4.07/4.02  4.23/4.26/4.20  1.1275/1.1749/0.8841  GPU4
periodic_residual_counts 16/17/18 244898  53  133.5  4.10/4.11/4.04  4.29/4.29/4.23  0.9187/0.8482/0.8438  GPU3
linear_residual_counts 16/17/18   244634  53   94.0  3.99/4.10/3.98  4.17/4.34/4.18  0.9204/0.8524/0.8411  GPU4
```

## Observations
These come from the smoke run and are not performance evidence.
- Parameter counts on the real export equal the design note's formulas:
  - periodic_residual 244,898
  - linear_residual 244,634
  - periodic 288,802
  - residual 170,762
- The control's delivery temperature sits at the 0.5 lower bound on all 3 seeds. This matched the P30 and P31 smokes, where it fit the 2-epoch under-training pattern. The full run must report every per-arm bound hit. The bounds stay unchanged.
- Smoke fit seconds measure wiring only. Full-fit time is 미측정 (not measured).
- The server-side helper `/tmp/embed_smoke_facts_cycle016.py` was used read-only for the table above. It was not deleted: server deletion needs approval first.
