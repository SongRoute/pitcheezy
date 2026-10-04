# EXP-P35-001 launch plan — go recorded, launch deferred to 2026-10-05 (T78, v5 cycle-001, 2026-10-04 20:31–20:33 KST)

Status: NOT REGISTERED, NOT LAUNCHED. Exploratory execution only; no promotion, verdict or policy claim. Family code and
results: same_model_reviewed; Codex_review_pending.

## Go reference

Root directive at the top of the pinned continuous runbook v5
(`/Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/continuous-runbook-v5.md`, sha256
`e73f44022593792efa326df31440e7c6f88752bd455d62eb47f9b2e8a59f3b96`): "Root authorizes exploratory P35 registration and
the fixed 25-fit family under the already-given Song instruction to continue performance experiments; this is NOT policy
adoption or a verdict change." Pass this as the `<go reference>` argument:

    root v5 runbook directive sha256 e73f4402...3b96 (2026-10-04 20:29 KST), under Song's standing instruction to continue performance experiments; exploratory only

Wording note: `make_p35_config.py` (reviewed 859f315) hard-codes the `decision` sentence "Song's explicit go ... plus the
pinned runbook v4 delegation". The actual go is the v5 root directive above, which `registration.go` records exactly. The
reviewed generator is not edited; this note is the disclosure.

## Why not tonight

The reviewed pack (MINOR-1, ed02276) fixes L+5 h / +1 h / +30 min on ONE KST date (the queue's `Deadlines` read HH:MM on
`queue.date`). At 20:31 on 10-04, L + 6 h 30 min crosses midnight, so the pack guard is kept and the next valid launch time
is 2026-10-05 00:00 or later. The DRAFT `queue.date` is already `2026-10-05`, so the registered date stays equal to the
DRAFT. Latest valid launch: 16:30 (aggregate ≤ 23:00 coordinator deadline).

## Gates verified 2026-10-04 20:31 KST (read-only)

- newsrv 20:31:34: GPU3 `c334d905` 2 MiB 0 %, GPU4 `ae803a24` 2 MiB 0 %, no compute app on 3/4; no
  `cuda_direct_queue.py|timing_probe_|score_cuda_direct` process; lock files hold pid 335823 (probe), not running.
- DRAFT `f0ae4e91…b9b2` identical in `code-8864389` and `code-3b317f2` (and locally); `git diff 8864389 3b317f2` over
  pitchmdp/, run_cuda_direct, cuda_direct_queue, score_cuda_direct, configs: empty.
- Probe summary `311f0d4f…7ff5`: complete, exit 0/0, `ple_arms_within`/`every_arm_within`/`family_within` true;
  slowest per arm 464/1,046/1,149/1,225/1,150 s (≤ 1,800), family 25,171 worker-s (≤ 36,000).
- `queue-ple-smoke-001` {done: 25}; terminal p30-001 50, p31-001 15, p32-001 18, p33-001 20, p34-001 25, p30-002 50.
- No P35 path exists (queue-p35-001, runs/EXP-P35-001, scores/EXP-P35-001, score-p35-state.json, EXP-P35-001.json,
  run-p35-pipeline.py).
- Export manifest `49097109…f578`. Server disk: 28 GB free (99 % used, shared volume); P34 family output 1.3 GB.
- Budgets unchanged in the DRAFT: task 3600 s, family 36,000 worker-s, attempts 2, epoch cap 60.

## Dry run (T78, scratch `/tmp/pz-t78-dry-fLoMyf`, left in place)

`make_p35_config.py probe-summary.json configs/EXP-P35-001.json 2026-10-05 00:30 05:30 06:30 07:00 dry-run-t78 "DRY RUN
ONLY (T78)"` → exit 0, sha `3c569d79…6b5c`. `tests/test_cuda_ple_registered.py` (3d26401) on it: 5 passed. Extractor
`config_diff` (coordinator `reports/research-p35/extract_p35.py`): 10 paths, 0 outside `REGISTRATION_PATHS`. Mutants:
epochs 61 → 2 failed; kill < no_new + 1 h → 1 failed; family budget 40,000 → 2 failed; empty go → 1 failed.

## Launch-day steps (first cycle at or after 2026-10-05 00:00 KST)

1. `date`. Choose L = now + 20 min rounded UP to 5 min (L must be ≥ the actual launch; declaring later only widens the
   real margin). no_new = L+5 h, kill = no_new+1 h, aggregate = kill+30 min. If aggregate > 23:00 → do not launch.
2. Read-only preflight (separate ssh call): the same reads as above (GPU3/4 by UUID, no queue/probe/score process, no P35
   path, disk ≥ 5 GB, export manifest).
3. In this worktree: `make_p35_config.py <probe summary 311f0d4f> configs/EXP-P35-001.json 2026-10-05 <L> <no_new>
   <kill> <agg> <cycle id> "<go reference>"`; run `tests/test_cuda_ple_registered.py` + extractor `config_diff`
   (0 outside); commit config.
4. `cp run-p35-pipeline.TEMPLATE.py run-p35-pipeline.py`, set `configsha` to the generated sha, `py_compile`, commit.
5. Server copies as NEW files with scp (one call), then a separate read-only `sha256sum` (config and pipeline equal the
   commit). Same pattern as P34/P30-002 (`a-w` after copy, in its own call; no launch json, never p33-launch.json).
6. Launch (one call, stdin/stdout detached so ssh returns; no remote wait loop):
   `cd /home/sds733/pitcheezy-autonomy/20261004 && nohup setsid timeout --kill-after=60 <budget> /home/sds733/venv/bin/python run-p35-pipeline.py < /dev/null > p35-pipeline.log 2>&1 &`
   with `<budget>` = (aggregate − L) + 900 + 600 s, and ≤ 40,000 (e.g. L 00:30 → 23,400 + 1,500 = 24,900). Plain python,
   no `-O`; the command string must not contain `cuda_direct_queue.py` or `timing_probe_`.
7. ONE confirmation read a few minutes later (state file + queue status), then return WAITING. Mac-side launch record
   only (git doc on this branch). Later cycles: one status snapshot each; after `complete`, copy scores read-only and run
   the committed extractor (fadcb6f) — descriptive only, every arm and seed, repeated DEV exposure stated.
