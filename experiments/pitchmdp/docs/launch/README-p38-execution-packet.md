# P38 (pitcher-ID) execution packet — PREPARED, NOT RUN (author_only; distinct Opus review pending)

> **Recovery (2026-10-05 p38-recovery-001):** the v1 smoke gate REFUSED at 1.2528 (15:44). The next step is the ONE v2
> TRAIN-only fixed-step resource measurement in `README-p38-recovery-v2.md`; registration now needs v1 validity + v2 pass.

Job `pm-jobs/20261005-next-information/integration-002`, branch `codex/p38-integration-1005` (base = P37 delivery
`2265f444be87e65adf9514173e3c08998a31700c`, P38 model `684f4b4` cherry-picked with `-x`). Nothing here touched a GPU, SSH
or the server. `<C>` = the review-003 commit on `codex/p38-review-1005` (P38 on final P37 head c44d5fc; full sha in REPORT-review-003.md). `B=/home/sds733/pitcheezy-autonomy/20261004`.

Pins: DRAFT `configs/DRAFT-CUDA-PITCHER-ID-20261005.json` sha `8518f754236447fd054550dee75a5c3da7c510c3dd646f028662f097a02ee537`
(deterministic `make_p38_config.py`); export manifest `49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578`
(same export as P35-P37; `pitcher`/`date` arrays already in it, no new export). GPU3 only
`GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`; GPU4 never appears in any command. Lock dir always the COMMON
`/home/sds733/pitcheezy-autonomy/gpu-locks`. 2 arms x seeds 44-48 = 10 fits, cap 60, one at a time.

PREREQUISITE (hard, checked by the pipeline): P37 pipeline state `score-p37-state.json` complete/incomplete/blocked, no
`cuda_direct_queue.py|timing_probe_|score_cuda_direct|run-p37-pipeline` process, GPU3 without compute app. P37 keeps
priority and exclusive GPU3 ownership until then. If P37 review changes runner/scorer after `2265f44`, rebase this
branch on the final reviewed P37 head (or root integration) and re-run the tests BEFORE step 1; the archived code must be
one commit containing both final P37 and P38.

## Files
- `pitchmdp/direct_pitcher_id_registry.py` (torch-free): family/arm/seed/device/contrast checks, TRAIN-only `build_vocab`
  / `map_ids`, split-date pin (TRAIN max_date <= 20250430 < later min_date), per-fit record check, resource-gate constants.
- `pitchmdp/direct_pitcher_id_model.py`: network/model (684f4b4) + `build_pitcher_id` / `load_pitcher_id` (vocab hash).
- `scripts/run_cuda_direct.py`: marker-first dispatch; vocabulary from loaded TRAIN rows; index column for every split
  (train/earlystop/temperature/blend/dev/mlb_dev); `fit.json.pitcher_id` record; load checks archived vocab hash.
- `scripts/score_cuda_direct.py`: two-arm path (no control), vocabulary RE-DERIVED from export TRAIN arrays, all / in-vocab /
  UNK subsets, every seed + ensemble, `decision` null.
- `p38_smoke_gate.py` (TRAIN-only resource gate: candidate/reference median train s/epoch <= 1.25 + validity checks),
  `make_p38_registered_config.py`, `run-p38-pipeline.TEMPLATE.py` (save/interrupted/execute/queue AST-identical to P37).
- Tests: `tests/test_cuda_pitcher_id_wiring.py`, `tests/test_direct_pitcher_id_model.py`.

## Steps (executor; one ssh call per step, no remote wait loop, never `-O`)
0. `date` (KST). Read-only preflight: `nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv`,
   `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv` (GPU3 must be empty; else wait/report),
   `cat $B/score-p37-state.json` (terminal), `pgrep -af 'cuda_direct_queue.py|timing_probe_|score_cuda_direct|run-p3'`
   empty; absent: `$B/code-<C7>`, `$B/runs/DRAFT-PITCHER-ID-<C7>-smoke`, `$B/queue-pitcher-id-smoke-<C7>`,
   `$B/p38-smoke-gate.json`, `$B/EXP-P38-001.json`, `$B/run-p38-pipeline.py`, `$B/score-p38-state.json`,
   `$B/queue-p38-001`, `$B/runs/EXP-P38-001`, `$B/scores/EXP-P38-001`; disk >= 5 GB.
1. Immutable source: `git archive --format=tar <C> | ssh newsrv 'mkdir $B/code-<C7> && tar -x -C $B/code-<C7> && chmod -R a-w $B/code-<C7>'`;
   separate `sha256sum $B/code-<C7>/configs/DRAFT-CUDA-PITCHER-ID-20261005.json` (= 8518f754…).
2. Smoke (detached, <= 1,200 s): `cd $B/code-<C7>/experiments/pitchmdp && nohup setsid timeout --kill-after=60 1200 env
   OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/sds733/venv/bin/python
   scripts/cuda_direct_queue.py --config $B/code-<C7>/configs/DRAFT-CUDA-PITCHER-ID-20261005.json --config-sha256
   8518f754236447fd054550dee75a5c3da7c510c3dd646f028662f097a02ee537 --export $B/export-49097109 --export-manifest-sha256
   49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578 --output $B/runs/DRAFT-PITCHER-ID-<C7>-smoke
   --queue-dir $B/queue-pitcher-id-smoke-<C7> --lock-dir /home/sds733/pitcheezy-autonomy/gpu-locks --gpu
   GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9 --smoke run < /dev/null > $B/p38-smoke-<C7>.log 2>&1 &`.
   The queue clock (DRAFT queue.date/no_new_fit_after) also bounds the smoke: if the date passed, re-date first (below).
3. Gate: `cd $B/code-<C7>/experiments/pitchmdp && /home/sds733/venv/bin/python docs/launch/p38_smoke_gate.py
   $B/code-<C7>/configs/DRAFT-CUDA-PITCHER-ID-20261005.json $B/queue-pitcher-id-smoke-<C7> $B/runs/DRAFT-PITCHER-ID-<C7>-smoke
   $B/p38-smoke-gate.json` → `P38_SMOKE_GATE PASS <gatesha>`; exit 3 = refused: stop, report, no registration. The
   `PITCHER_ID_VOCAB` log lines give the smoke vocabulary (8,192 TRAIN rows; not the full-TRAIN vocabulary).
4. Register (Mac, real clock <= launch_by): `python experiments/pitchmdp/docs/launch/make_p38_registered_config.py <local gate
   summary> configs/EXP-P38-001.json <HH:MM now> <C7> "<go reference>"` → `<configsha>`; fill the 5 placeholders of a copy
   `run-p38-pipeline.py`; `py_compile`; commit both on the executor branch.
5. scp both to `$B/EXP-P38-001.json`, `$B/run-p38-pipeline.py`; separate `sha256sum`; separate `chmod a-w`.
6. Launch (<= launch_by): `cd $B && nohup setsid timeout --kill-after=60 <budget> /home/sds733/venv/bin/python run-p38-pipeline.py
   < /dev/null > p38-pipeline.log 2>&1 &`, `<budget>` = (aggregate_by − L) + 600 s.
7. ONE confirmation read (`score-p38-state.json`, `queue-p38-001/` status, lock owner pid). Stop own only: `kill -TERM <pid>`.
8. After `complete`: pull `scores/EXP-P38-001`, descriptive report (every seed, both splits, all/in-vocab/UNK, vocabulary
   size / min count / UNK shares / dates / row hashes, extra parameters; no verdict, no causal capacity claim).

Re-dating (only if the default 2026-10-05 20:45 window is missed): `python experiments/pitchmdp/docs/launch/make_p38_config.py
--clock YYYY-MM-DD HH:MM` (only queue clock fields change; tested), commit, update `DRAFT_SHA` in `p38_smoke_gate.py`,
`draftsha` in the template and `RUN_SHA256` in the test, and get the clock diff reviewed before step 1.
