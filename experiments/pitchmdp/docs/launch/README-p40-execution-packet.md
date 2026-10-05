# P40 (pitcher-ID controls) execution packet — author_only, distinct review + Root acceptance pending

Pre-registration `docs/PREREG-DRAFT-EXP-P40-001-pitcher-id-controls-2026-10-06.md`. DRAFT
`configs/DRAFT-CUDA-PITCHER-ID-CONTROLS-20261006.json` sha `3dba5445b3a76dbc5f19c73f1cd60f6a69fd18312c775a6f82e06db656ab6944`
(deterministic `make_p40_config.py`). Export 49097109 (unchanged). `B=/home/sds733/pitcheezy-autonomy/20261004`. GPU3 UUID
`GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9` only; common lock dir `/home/sds733/pitcheezy-autonomy/gpu-locks`. `<C>` = the
reviewed commit, `<C7>` its 7-char prefix.

## Code changes vs ba0a8f1 (P38 head)
- `pitchmdp/direct_pitcher_id_registry.py`: `design(config)` (no key = P38 unchanged; `pitcher_id_design:
  p40_controls_v1` = P40), 2 new arms, seeds 60-64, 4 contrasts, `id_spec`, projection 6,117.5 s; `check_fit` /
  `check_one_vocab` over every id arm.
- `pitchmdp/direct_pitcher_id_model.py`: `shuffled_pitcher` + `id_inputs(..., split, index_source, shuffle_seed)`;
  network `id_dropout` (train-only, private device generator; config key only when > 0, so P38 archives are unchanged);
  archive round trip and an arm/archive dropout mismatch check.
- `scripts/run_cuda_direct.py`: passes the split / index source to `id_inputs`; fit.json records mode fields for P40 controls only.
- `scripts/score_cuda_direct.py`: pitcher-ID scorer loops over the design's arms and contrasts, adds per-arm subset
  metrics and a P40 note/table text. The P38 output layout is unchanged for P38 configs.
- Tests: `tests/test_p40_pitcher_id_controls.py` (new). Five P38-era source-pin tests are skipped with a reason: they assert
  the tree equals the frozen 5b3335e map, which is true at ba0a8f1 and false by design here.

## Steps (one ssh call per step, never `-O`, GPU3-only nvidia-smi queries)
0. Preflight: `date`; `nvidia-smi --id=GPU3UUID --query-compute-apps=pid --format=csv,noheader` empty; pgrep (its own ssh call)
   `[c]uda_direct_queue|[s]core_cuda_direct|[r]un-p3|[r]un-p4` empty; `$B/code-<C7>`, `$B/runs/DRAFT-P40-<C7>-smoke`,
   `$B/queue-p40-smoke-<C7>`, `$B/p40-smoke-check.json`, `$B/EXP-P40-001.json`, `$B/run-p40-pipeline.py`,
   `$B/score-p40-state.json`, `$B/queue-p40-001`, `$B/runs/EXP-P40-001`, `$B/scores/EXP-P40-001` absent.
1. `git archive <C> configs experiments/pitchmdp | ssh newsrv 'mkdir $B/code-<C7> && tar -x -C $B/code-<C7> && chmod -R a-w $B/code-<C7>'`; sha256 of the DRAFT = 3dba5445….
2. Smoke (TRAIN-only, detached, ≤ 1,800 s): `cd $B/code-<C7>/experiments/pitchmdp && nohup setsid timeout --kill-after=60 1800 env
   OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/sds733/venv/bin/python
   scripts/cuda_direct_queue.py --config $B/code-<C7>/configs/DRAFT-CUDA-PITCHER-ID-CONTROLS-20261006.json --config-sha256 3dba5445…
   --export $B/export-49097109 --export-manifest-sha256 49097109… --output $B/runs/DRAFT-P40-<C7>-smoke --queue-dir
   $B/queue-p40-smoke-<C7> --lock-dir /home/sds733/pitcheezy-autonomy/gpu-locks --gpu GPU-c334d905-… --smoke run < /dev/null > $B/p40-smoke-<C7>.log 2>&1 &`
3. `python docs/launch/p40_launch.py smoke-check <DRAFT> $B/queue-p40-smoke-<C7> $B/runs/DRAFT-P40-<C7>-smoke $B/p40-smoke-check.json` → `P40_SMOKE_CHECK VALID <sha>`.
4. After review + Root acceptance: copy the check to the Mac, then `python experiments/pitchmdp/docs/launch/p40_launch.py register <check> <HH:MM now> <C7> "<root go>"`.
   Fill a copy of the template as `run-p40-pipeline.py` (configsha, codecommit=<C7>, smokesha, smokeq, smokeruns), run `py_compile`, commit.
5. scp both to `$B` as NEW files; separate sha256 check; `chmod a-w`.
6. Launch ≤ 02:05 (else re-date with `make_p40_config.py --clock` + review of the clock diff + new smoke): `cd $B && nohup setsid timeout
   --kill-after=60 <(04:25 − now) + 600> /home/sds733/venv/bin/python run-p40-pipeline.py < /dev/null > p40-pipeline.log 2>&1 &`.
7. Single finite progress reads only. After `complete`, pull `scores/EXP-P40-001` and write a descriptive report (all contrasts, splits, subsets, seeds; no verdict).
