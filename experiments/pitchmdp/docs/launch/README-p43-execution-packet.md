# P43 (batter ID) execution packet — author_only (next-research), distinct review + Root acceptance pending

Pre-registration `docs/PREREG-DRAFT-EXP-P43-001-batter-id-2026-10-06.md`. Undated DRAFTs (deterministic
`make_p43_config.py`): main `configs/DRAFT-CUDA-BATTER-ID-20261006.json` sha `bc94ae86a2482c920ebd3cb8a898774bcbbb4c5862d58639556621f4b7a442e2`
(EXP-P43-001, 5 arms × seeds 70–72 = 15 fits), extension `configs/DRAFT-CUDA-BATTER-ID-EXT-20261006.json` sha
`389c81b369a206460202aeda8b683a66da5df3e565dda0c380417e61f998782a` (EXP-P43-002, seeds 73–74 = 10 fits, only after 001
is complete and only if the window fits). Export 49097109 unchanged. Batter sidecar `sidecar.json` sha
`ad2eacc64d46985a57527167486d7762a435f02d51fc5ed86a9e2e2644c160a4`, Mac copy
`/Users/song/Projects/pitcheezy-pm-jobs/20261006-four-hour/next-research/batter-sidecar-49097109/`.
`B=/home/sds733/pitcheezy-autonomy/20261004`, GPU3 UUID `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9` only, common lock dir
`/home/sds733/pitcheezy-autonomy/gpu-locks`. `<C>` = reviewed commit, `<C7>` its prefix. Executor: GPU lead (this
author does not own GPU allocations).

## Code changes vs 3ec8df1 (registered P40 source)
- NEW `pitchmdp/batter_sidecar.py`: id-only key join (keys, batter, pitcher), row-exact (0 missing, raw pitcher ==
  export pitcher), refuses non-`statcast_YYYY.parquet` / YYYY > 2025 / `holdout` paths; `verify` / `load`.
- NEW `pitchmdp/direct_pitcher_batter_id_model.py`: two-path network (pitcher path = P38 verbatim, batter path private
  generator seed + 7919 + 1009), single batter model (P38 network / archive format, batter projection seed).
- `pitchmdp/direct_pitcher_id_registry.py`: designs `p43_batter_id_v1` / `p43_batter_id_ext_v1`, 3 new arms, 6 contrasts,
  sidecar pin check, P43 fit / per-column vocabulary checks. P38 / P40 designs unchanged.
- `scripts/run_cuda_direct.py`: SOURCES + 2 files; `verify_export` verifies the sidecar for P43 configs; `with_ids`
  attaches `batter`; per-column TRAIN vocabularies; P43 index columns / fit record / build / load dispatch. Other families'
  paths unchanged.
- `scripts/score_cuda_direct.py`: P43 branch of the pitcher-ID scorer (per-column vocab re-derivation from TRAIN,
  verified sidecar, batter subsets, per-arm parameters, co-primary marks, P43 note / Korean table).
- Tests: NEW `tests/test_p43_batter_id.py` (15); `tests/test_timing_probe_robust_clip.py` declares the P43 two-file
  runtime delta (P38 precedent) instead of skipping.
Evidence: P43 15 passed; 31 affected test files 1120 passed, 3 failures identical at untouched 3ec8df1 (historical
"not yet registered" assertions in test_cuda_ple_wiring / test_cuda_robust_clip_registered_pack / test_cuda_robust_clip_wiring);
7 meaningful mutants all caught.

## Steps (one ssh call per step, never `-O`, GPU3-only nvidia-smi queries; start only when P40 is terminal and P42 is not running)
0. Preflight: `date`; `cat $B/score-p40-state.json` status in complete/incomplete/blocked; P42 not started (or terminal);
   `nvidia-smi --id=GPU3UUID --query-compute-apps=pid --format=csv,noheader` empty; pgrep in its OWN ssh call
   `[c]uda_direct_queue|[s]core_cuda_direct|[r]un-p40-pipeline|[r]un-p42-pipeline` empty; `$B/code-<C7>`,
   `$B/batter-sidecar-49097109`, `$B/runs/DRAFT-P43-<C7>-smoke`, `$B/queue-p43-smoke-<C7>`, `$B/p43-001-smoke-check.json`,
   `$B/EXP-P43-001.json`, `$B/run-p43-pipeline.py`, `$B/score-p43-001-state.json`, `$B/queue-p43-001`,
   `$B/runs/EXP-P43-001`, `$B/scores/EXP-P43-001` absent.
1. Sidecar (≈10.8 MB, NEW dir): `tar -C /Users/song/Projects/pitcheezy-pm-jobs/20261006-four-hour/next-research -cf -
   batter-sidecar-49097109 | ssh newsrv 'tar -x -C $B && chmod -R a-w $B/batter-sidecar-49097109 && sha256sum
   $B/batter-sidecar-49097109/sidecar.json'` → must print ad2eacc6….
2. Dated DRAFT on the Mac (repo root, branch commit `<C>`): `python experiments/pitchmdp/docs/launch/make_p43_config.py
   --clock 2026-10-06 <HH:MM launch_by> --out configs/DRAFT-CUDA-BATTER-ID-20261006-dated-<HHMM>.json`, then
   `python experiments/pitchmdp/docs/launch/p43_launch.py dated-check <that file>` → `P43_DATED_OK <sha>`; commit it
   (that commit is `<C>`). launch_by + 4,588 s, ceil 5 min = no_new_fit_after; +15 kill; +20 aggregate.
3. `git archive <C> configs experiments/pitchmdp | ssh newsrv 'mkdir $B/code-<C7> && tar -x -C $B/code-<C7> && chmod -R a-w $B/code-<C7>'`.
4. Smoke (TRAIN-only, detached, ≤ 1,800 s): as P40 step 2 with `--config $B/code-<C7>/configs/<dated DRAFT>
   --config-sha256 <dated sha> --output $B/runs/DRAFT-P43-<C7>-smoke --queue-dir $B/queue-p43-smoke-<C7>` (export
   `$B/export-49097109`; the runner finds the sidecar as the export's sibling).
5. `python docs/launch/p43_launch.py smoke-check <dated DRAFT> $B/queue-p43-smoke-<C7> $B/runs/DRAFT-P43-<C7>-smoke
   $B/p43-001-smoke-check.json` → `P43_SMOKE_CHECK VALID <sha>`.
6. After review + Root acceptance: copy the check to the Mac; `python experiments/pitchmdp/docs/launch/p43_launch.py register
   <dated DRAFT> <check> <HH:MM now> <C7> "<root go>"` → `configs/EXP-P43-001.json`. Fill a copy of
   `run-p43-pipeline.TEMPLATE.py` as `run-p43-pipeline.py` (configsha, codecommit=<C7>, smokesha, familyid='EXP-P43-001',
   smokeq, smokeruns, draftsha, draftname), `py_compile`, commit.
7. scp both to `$B` as NEW files; sha256 check; `chmod a-w`.
8. Launch ≤ launch_by: `cd $B && nohup setsid timeout --kill-after=60 <(aggregate_by − now) + 600> /home/sds733/venv/bin/python
   run-p43-pipeline.py < /dev/null > p43-001-pipeline.log 2>&1 &`.
9. Finite progress reads only. After `complete`, pull `scores/EXP-P43-001` (`gpu pull` or scp to the Mac) and write a
   descriptive report: both co-primaries, every contrast, split, subset and seed; no verdict.
Extension EXP-P43-002: repeat 2–8 with `--extension`, familyid 'EXP-P43-002', file names `p43-002-*`, only when
`score-p43-001-state.json` is `complete` and the 10-fit window (3,059 s + scoring) still fits.

## Not allowed
No GPU4, no second concurrent fit on GPU3 without a fresh distinct review + TRAIN-only bench on this exact source,
no DEV read before scoring, no 2026, no pooling of 73–74 into the 70–72 bootstrap, no edits to P40 / P41 / P42 sources.

## Expected values (computed 01:30 KST from TRAIN ids only on the Mac copy; the full fits must reproduce them)
- TRAIN vocabularies (≥100 rows, dates ≤ 20250430): pitcher 920 of 1,137 (UNK TRAIN rows 9,903; ids sha 9fbe6f45…, P38's
  8,392 id parameters), batter 706 of 806 (UNK TRAIN rows 4,425; ids sha 0a714915…).
- Allocated parameters: reference 244,898; pitcher 253,290; batter 251,578; pitcher+batter 259,970; batter shuffle 251,578.
- Local CPU TRAIN smoke on the REAL export + sidecar (01:29 KST, 5 arms × seed 70, smoke limits, `/tmp/nr-p43-smoke`):
  all 5 tasks completed (0.3 s each), dev_used false, parameter counts consistent. Smoke limitation: the first 8,192
  TRAIN rows contain no batter with ≥100 rows, so the smoke batter vocabulary is EMPTY (n_vocab 0, every batter UNK);
  the GPU smoke therefore exercises the batter path only through the UNK row. The real vocabulary path is covered by
  the CPU tests (synthetic export, non-empty vocab) and by the numbers above. Prediction archives were never opened.
