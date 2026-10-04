# EXP-P36-001 launch record (T120, v5 cycle-041-20261004T162217Z-ef81dc5d, 2026-10-05 KST)

Exploratory execution only. No promotion, verdict or policy claim. Family code and results: same_model_reviewed;
Codex_review_pending. Go reference: `registration.go` (v5 runbook e73f4402…3b96 first section + PREREG P36 line 10 +
§7 rule met, P35 report 3218d78; cycle id). Lineage `codex/p36-smoke-code-1005`; server code dir `code-07ab288`.

## Steps (README-p36-registration-pack.md)
1. `date` 01:22:20 → L = now+20 min rounded up = **01:45**; no_new 06:45, kill 07:45, aggregate 08:15 (≤ 23:00,
   L ≤ 16:30), one KST date 2026-10-05.
2. Read-only preflight (server 01:23:00): GPU3 `c334d905` / GPU4 `ae803a24` 2 MiB 0 %, no compute app; no
   queue/probe/score process; the 7 P36 paths absent; `score-p35-state.json` complete; 104 G free; locks hold stale
   pid 443102 (smoke queue r2); server EXP-P35-001.json 0a7cefda…, DRAFT 644239f3… in code-07ab288; probe dir and
   smoke queue r2 present.
3. `make_p36_config.py` (probe summary 0edb45bf…, configs/EXP-P35-001.json = byte copy of 30588ad 0a7cefda…) →
   `configs/EXP-P36-001.json` sha **b6479afa43a59df91774e0b714da2397692bdb06959fabb7716a129a5826b87e**; extractor
   `config_diff` 11 paths, 0 outside REGISTRATION_PATHS; new `tests/test_cuda_robust_clip_registered.py` 6 passed
   (epochs 61 + aggregate 23:30 mutant → 3 failed, restored, sha re-checked); with the pack test 87 passed. Commit c834504.
4. `run-p36-pipeline.py` = TEMPLATE with the six placeholder lines filled (diff = those lines only), py_compile ok,
   sha **d1b8bde2f233ae166f33487e82e0e0ff893ec6a8861b00ff60f7f24988d52cf1**. Commit c80dd52.
5. One scp (both absent at step 2) → separate `sha256sum` on server equals both shas and the probe summary 0edb45bf…
   → separate `chmod a-w` (-r--r--r--).
6. Launch 01:24:25 KST (one ssh call): `cd /home/sds733/pitcheezy-autonomy/20261004 && nohup setsid timeout
   --kill-after=60 24900 /home/sds733/venv/bin/python run-p36-pipeline.py < /dev/null > p36-pipeline.log 2>&1 &`
   (24,900 = 08:15 − 01:45 + 900 + 600). Remote printed `launched`; the local ssh client did not close within 90 s
   (detached remote process; not a wait loop, same as P35).
7. Confirmation read: below.

## Confirmation read (ONE, server 01:26:47 KST)
- `score-p36-state.json`: status `p36-full-001`, pipeline pid 446651, queue pid 446718, updated 16:24:25Z → every
  in-pipeline check before the queue passed.
- `queue-p36-001/`: logs, plan.json, preflight-446718.json, queue.lock, status (control-34, periodic_residual_counts-34,
  residual_counts-34 status files present).
- GPU locks: both UUIDs held by 446718. GPU3 1,908 MiB 8 % / GPU4 1,796 MiB 5 %.
- State: **RUNNING** (no score). Probe projection 16,852 worker-s / 2 GPUs ≈ 2.3 h (미측정 until done); before no_new 06:45.

## Next
One status snapshot per later cycle (no wait loop). After `complete`: read-only pull of scores + descriptive report
(every arm and seed, repeated DEV exposure stated, no winner).
