# P37 (embed-type) execution packet — PREPARED, NOT RUN (author_only; distinct Opus review pending)

Author job `pm-jobs/20261005-p37-delivery/author-001`, branch `codex/p37-delivery-1005` (base 6b66436 = review 90d9dde +
review doc). Nothing here touched a GPU, SSH or the server. `<C>` = the delivery commit that adds this file (full sha
in the author REPORT). Base dir `B=/home/sds733/pitcheezy-autonomy/20261004` (export-49097109, P36 queue there).

Pins: DRAFT `configs/DRAFT-CUDA-EMBED-TYPE-20261005.json` sha `75265fc715f0fec998f609f0316ffe1de486c609677588d0fbd7241fc427ace2`
(regenerated at `<C>`: same bytes); export manifest `49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578`;
prereg sha at `<C>` recorded by the registration helper (`registration.preregistration.sha256`). GPU3 only:
`GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`; GPU4 `GPU-ae803a24-…` never appears in any command. Lock dir is always the
COMMON `/home/sds733/pitcheezy-autonomy/gpu-locks`.

## Files
- `p37_smoke_gate.py` — TRAIN-only smoke gate (prereg §4): 15/15 identities via `runner.verify_task`, smoke bound-UUID
  set == {GPU3}, median over 5 seeds of `report.seconds / report.epochs_run`, linear/periodic ≤ 1.5, invalid fields
  refuse; fit-wall overhead recorded, not gated; projection constant 6,881.4 s (P36 actuals, not the smoke). Exit 0/3.
- `make_p37_registered_config.py` — DRAFT + passing gate summary → `EXP-P37-001.json`; only status/decision/
  execution.enabled/registration change; registered HH:MM ≤ 19:00, within 15 min of the real clock, never future;
  registered + 6,881.4 s ≤ 21:00. Deterministic.
- `run-p37-pipeline.TEMPLATE.py` — from the P36 template (save/interrupted/execute AST-identical, tested). Placeholders
  `configsha codecommit gatesha smokeq smokeruns` fail closed. Exactly one `--gpu` GPU3 + common lock (asserted);
  smoke UUID set == {GPU3}; sources == gate `source_sha256`; launch ≤ 19:00 and launch + projection ≤ 21:00; queue
  timeout = until 21:45 + 300 s (the queue itself stops new fits at 21:00 and kills own fits at 21:45); scorer only after
  15/15 and only if it can end by 22:00 (900 s); otherwise state `incomplete` (report by 22:30).
- Tests `tests/test_cuda_embed_type_launch_pack.py` (23) + `tests/test_cuda_embed_type_wiring.py` (53).

## Steps (executor; one ssh call per step, no remote wait loop, no `p33-launch.json`, never `-O`)
0. `date` (KST). Read-only preflight (separate ssh): `nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu
   --format=csv` and `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv` — GPU3 must have no compute app (if
   busy: wait/report, never another GPU); `pgrep -af 'cuda_direct_queue.py|timing_probe_|score_cuda_direct'` empty;
   `ls /home/sds733/pitcheezy-autonomy/gpu-locks`; absent: `$B/code-<C>`, `$B/runs/DRAFT-EMBED-TYPE-<C7>-smoke`,
   `$B/queue-embed-type-smoke-<C7>`, `$B/p37-smoke-gate.json`, `$B/EXP-P37-001.json`, `$B/run-p37-pipeline.py`,
   `$B/score-p37-state.json`, `$B/queue-p37-001`, `$B/runs/EXP-P37-001`, `$B/scores/EXP-P37-001`; `queue-p36-001`
   summary `{done: 20}`; disk ≥ 5 GB.
1. Immutable source (Mac): `git -C <worktree> archive --format=tar <C> | ssh newsrv 'mkdir $B/code-<C7> && tar -x -C
   $B/code-<C7> && chmod -R a-w $B/code-<C7>'`; then a separate read-only `sha256sum` of
   `code-<C7>/configs/DRAFT-CUDA-EMBED-TYPE-20261005.json` (= 75265fc7…) and the 4 launch files.
2. Smoke (one call, detached; ≤ 1,800 s; P36 smoke 20 tasks took 92 s on 2 GPUs, P37 smoke time 미측정):
   `cd $B/code-<C7>/experiments/pitchmdp && nohup setsid timeout --kill-after=60 1800 env OMP_NUM_THREADS=2
   OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/sds733/venv/bin/python
   scripts/cuda_direct_queue.py --config $B/code-<C7>/configs/DRAFT-CUDA-EMBED-TYPE-20261005.json --config-sha256
   75265fc715f0fec998f609f0316ffe1de486c609677588d0fbd7241fc427ace2 --export $B/export-49097109
   --export-manifest-sha256 49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578 --output
   $B/runs/DRAFT-EMBED-TYPE-<C7>-smoke --queue-dir $B/queue-embed-type-smoke-<C7> --lock-dir
   /home/sds733/pitcheezy-autonomy/gpu-locks --gpu GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9 --smoke run
   < /dev/null > $B/p37-smoke-<C7>.log 2>&1 &`
   Confirm later with ONE read of `queue-embed-type-smoke-<C7>/summary.json` (`{done: 15}`) and the log tail.
3. Gate (one call): `cd $B/code-<C7>/experiments/pitchmdp && /home/sds733/venv/bin/python docs/launch/p37_smoke_gate.py
   $B/code-<C7>/configs/DRAFT-CUDA-EMBED-TYPE-20261005.json $B/queue-embed-type-smoke-<C7>
   $B/runs/DRAFT-EMBED-TYPE-<C7>-smoke $B/p37-smoke-gate.json` → `P37_SMOKE_GATE PASS <gatesha>`; exit 3 = refused,
   stop and report (no registration). scp the summary to the Mac job dir; local sha must equal `<gatesha>`.
4. Register (Mac, real clock ≤ 19:00): `python experiments/pitchmdp/docs/launch/make_p37_registered_config.py
   <local gate summary> configs/EXP-P37-001.json <HH:MM now> <C7> "<go reference>"` → `<configsha>` (`<C7>` = the
   same 7-char string as `code-<C7>`, because the pipeline asserts `registration.source_commit == codecommit`).
   `cp run-p37-pipeline.TEMPLATE.py run-p37-pipeline.py`; fill exactly the 5 placeholder lines (`configsha`,
   `codecommit='<C7>'`; `gatesha`; `smokeq='queue-embed-type-smoke-<C7>'`; `smokeruns='runs/DRAFT-EMBED-TYPE-<C7>-smoke'`);
   `py_compile`; commit both (config + pipeline) on the executor branch.
5. scp the two NEW files to `$B/EXP-P37-001.json`, `$B/run-p37-pipeline.py`; separate `sha256sum` (= commit);
   separate `chmod a-w`.
6. Launch (one call, ≤ 19:00): `cd $B && nohup setsid timeout --kill-after=60 <budget> /home/sds733/venv/bin/python
   run-p37-pipeline.py < /dev/null > p37-pipeline.log 2>&1 &` with `<budget>` = (22:00 − L) + 600 s. The command must
   not contain `cuda_direct_queue.py`, `timing_probe_` or `score_cuda_direct` (pipeline pgrep self-block).
7. ONE confirmation read (`score-p37-state.json` status `p37-full-001`, `queue-p37-001/` status files, lock owner pid).
   Status later: read `score-p37-state.json` + `queue-p37-001/summary.json`. Stop (own only): `kill -TERM <pid>` with
   the pipeline pid from `score-p37-state.json` (its handler SIGTERMs only its own child process group).
8. After `complete`: read-only pull of `scores/EXP-P37-001` + descriptive report (every arm and seed, repeated DEV, no
   winner, no causal capacity claim). `incomplete`/`blocked`: report incomplete by 22:30.
