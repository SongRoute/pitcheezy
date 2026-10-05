# P38 recovery — v2 TRAIN-only fixed-step resource measurement (p38-recovery-001, AUTHOR ONLY, NOT RUN)

Job `pm-jobs/20261005-evening-resume/p38-recovery-001` (Claude Opus5.5 worker, root dispatch 2026-10-05 18:45 KST).
Branch `codex/p38-recovery-1005`, base = P38 review-003 final `5b3335e`, plus the GPU3 two-slot scheduler line
cherry-picked with `-x` (2ffe51f, 7566828, da74997, c17a092). No GPU, SSH, server, registration or launch in this job.
`B=/home/sds733/pitcheezy-autonomy/20261004`, `R` = this packet's commit (REPORT-p38-recovery-001.md).

## Why (and what this is NOT)
- The v1 smoke gate (`p38_train_smoke_gate_v1`, 8,192 TRAIN rows x 2 epochs) REFUSED on 2026-10-05 15:44:01 KST:
  candidate/reference 0.7587 / 0.6056 s per epoch = **1.2528 > 1.25**, errors [] (all validity checks passed).
  Gate summary sha `27cc8f78…96d3` (`pm-jobs/20261005-next-information/execute-004/p38-smoke-gate.json`). That refusal
  stays the original record. It was not rerun and the 1.25 limit is not raised.
- v2 (`p38_train_fixed_step_resource_v2`) changes the **measurement method only**: full TRAIN rows, full TRAIN pitcher
  vocabulary, batch 1024, fixed 3 epochs (patience == epochs), 2 arms x 2 seeds (902/903) in ABBA order, serial.
  Same metric (report.seconds / epochs_run, per-arm median) and the same 1.25 threshold.
- It was written **after observing** the v1 resource result (executor option 2 of REPORT-execute-004). It is not the
  original pre-registration, not an independent validation, and not evidence that v1 was wrong. It is a resource rule,
  not a scientific or performance claim.
- **Exactly one measurement.** The out dir `$B/p38-resource-v2-001` is fixed in the spec; the CLI refuses any other out
  and an existing one. A preflight refusal (P37 not terminal, other Pitcheezy GPU work, GPU3 busy, lock held, memory)
  happens before any CUDA and does not consume it. If the measurement REFUSES, P38 stays blocked: no rerun, no new
  method, no threshold change in this window.

## Files
- `scripts/p38_resource_measure.py`: spec check, pure `evaluate`, `preflight` (P38 not started + gpu3_two_slot's
  preflight), `measure` (reviewed `TwoSlotSupervisor` with **slots=1**: common GPU3 flock, own-session child, own-group
  cleanup on timeout/SIGTERM, lock release, exclusive `measure.json`).
- `scripts/p38_resource_child.py`: one fit through the frozen runner route incl. `pitcher_vocab` / `id_inputs` /
  `build_model(..., vocab ids)` (same rule as `run_task`), reusing `gpu3_bench_child.verify_train_only` and
  `fit_and_record`. TRAIN/earlystop only, no model saved, no scoring, no DEV/mlb_dev/2026.
- `docs/launch/p38-resource-v2-001.json`: the predeclared spec. Source pins = `own_hashes()` of 5b3335e (equal to the
  v1 smoke gate's `source_sha256`, checked); config = staged `code-5b3335e` DRAFT 8518f754; manifest 49097109.
- `make_p38_registered_config.py`: now `<v1 gate> <v2 measure.json> <out> <HH:MM> <commit> <go>`; requires v1 VALIDITY
  (errors [], 10 tasks, {GPU3}, DRAFT/manifest sha; pass flag consistent with its ratio) and a PASSING v2 measurement on
  the same source map and this commit's spec sha. `registration.smoke` keeps the v1 verdict/ratio; new
  `registration.resource` records the v2 sha/ratio and `method_revised_after_observing_v1: true`.
- `run-p38-pipeline.TEMPLATE.py`: new placeholder `ressha`; checks v1 validity + v2 pass (ratio <= 1.25, {GPU3},
  max_live 1, same sources); still `max_parallel == 1`; pgrep also refuses a live `p38_resource_child|gpu3_bench_child`.

## Resource protocol scope (this branch only; main/global rules not changed here)
- Song's limit is ONE card: GPU3 `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`. GPU4 is never used, probed or a fallback.
- "Max 1 fit" is the currently approved Pitcheezy protocol and stays in force for the v2 measurement and the P38 full run.
- Two fits on GPU3 (two-slot scheduler, bench-003: exact equivalence, 133.3 s serial vs 92.5 s concurrent for ONE
  control/periodic pair, 1.441; a candidate, not a universal speedup) is **proposed only after its own integration
  review**. Under the explicit autonomy delegation no extra Song approval is needed for that future step, but it is not
  adopted here.
- **2-slot integration into P38 is DEFERRED.** Reason: it needs a P38-only queue entry point on `TwoSlotSupervisor`, a new
  DRAFT sha with `max_parallel: 2` + `scheduler: gpu3_two_slot_v1`, a registry device check and a concurrent projection.
  A new DRAFT sha changes every smoke task identity (config digest), so the 10/10 smoke would have to be redone and
  reviewed — not bounded before the 20:45 launch_by. The serial packet keeps P38 runnable today; the scheduler code is
  on this branch (provenance `-x`) and used here only with slots=1.

## Executor steps (after a DISTINCT review accepts `R`; one ssh call per step, no remote wait loop)
0. Read-only preflight: `cat $B/score-p37-state.json` terminal; `pgrep -af 'cuda_direct_queue.py|timing_probe_|score_cuda_direct|gpu3_bench_child|p38_resource_child|run-p3[0-9]-pipeline'`
   empty (run it in its OWN ssh call: the pattern must not sit in the same command line as a launch);
   `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv` has no GPU3 row; absent: `$B/EXP-P38-001.json`,
   `$B/run-p38-pipeline.py`, `$B/score-p38-state.json`, `$B/queue-p38-001`, `$B/runs/EXP-P38-001`, `$B/p38-resource-v2-001`;
   present read-only `$B/code-5b3335e` (DRAFT sha 8518f754…). Any failure: wait/report, never another GPU.
1. Stage: `git archive --format=tar R configs experiments/pitchmdp | ssh newsrv 'mkdir $B/code-R7 && tar -x -C $B/code-R7 && chmod -R a-w $B/code-R7'`;
   separate `sha256sum` of `docs/launch/p38-resource-v2-001.json`, `scripts/p38_resource_{measure,child}.py`,
   `scripts/gpu3_{two_slot,bench_child}.py` == commit bytes; `own_hashes()` in `code-R7` == spec pins.
2. Measure (ONE run; real-clock start <= 20:20 KST so registration still fits before 20:45):
   `cd $B/code-R7/experiments/pitchmdp && nohup setsid timeout --kill-after=30 700 env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/sds733/venv/bin/python scripts/p38_resource_measure.py measure --spec docs/launch/p38-resource-v2-001.json --out $B/p38-resource-v2-001 < /dev/null > $B/p38-resource-v2-001.log 2>&1 &`
   Then single finite reads (<= 60 s apart) of the log's last line: `P38_RESOURCE_V2 PASS|REFUSED <sha>`.
   Projection (not measured): ~260 s for 4 fits, wall cap 600 s, per-fit timeout 200 s.
3. REFUSED (exit 3) or aborted: stop. P38 stays not registered; copy `measure.json` to the job dir and report. No rerun.
4. PASS: pull `measure.json` (hash). On the executor branch from `R`:
   `python experiments/pitchmdp/docs/launch/make_p38_registered_config.py <execute-004/p38-smoke-gate.json> <measure.json> configs/EXP-P38-001.json <HH:MM now> 5b3335e "<go ref>"`
   (source commit stays 5b3335e: the queue runs the smoke-verified `code-5b3335e`; its `own_hashes()` equal R's).
   Fill `run-p38-pipeline.py` from the TEMPLATE at `R`: configsha, codecommit=5b3335e, gatesha=<full 27cc8f78… sha>,
   smokeq=`queue-pitcher-id-smoke-5b3335e`, smokeruns=`runs/DRAFT-PITCHER-ID-5b3335e-smoke`, ressha=<measure.json sha>;
   `py_compile`; commit.
5. `scp` → `$B/EXP-P38-001.json`, `$B/run-p38-pipeline.py` (new files), separate `sha256sum`, `chmod a-w`.
6. Launch at real clock <= 20:45: `cd $B && nohup setsid timeout --kill-after=60 <budget> /home/sds733/venv/bin/python run-p38-pipeline.py < /dev/null > p38-pipeline.log 2>&1 &`,
   budget = (22:50 - actual start) + 600 s. One confirmation read. Scoring is automatic after 10/10, decision null.

## Clock (real 2026-10-05 KST; DRAFT queue fields unchanged)
- launch_by 20:45, no_new_fit_after 22:15, kill_own_after 22:30, aggregate_by 22:50. Family projection 5,292 s
  (P36 actuals, fixed) → a start at 20:45 ends by 22:13.
- Latest sensible v2 start ≈ 20:20 (700 s cap + pull/registration/scp ≈ 10 min). If the window is missed, P38 is
  blocked for today: re-dating needs `make_p38_config.py --clock` → new DRAFT sha → new smoke identities → new smoke and
  review. Do not backdate.

## Remaining objective gates
distinct review of `R` → P37 terminal (yes since 15:37:38, re-check) + GPU3 free + common lock free → stage identity →
ONE v2 measurement PASS (<= 1.25) → registration at real clock <= 20:45 → 10/10 → scorer. Science (arms, seeds 44-48,
data, model math, batch, cap 60, calibration) is byte-identical to DRAFT 8518f754.
