# Author report — P36 robust-clip timing-probe script (review a23e77b M1), T106c

Label: `author_only; same_model_review_pending; Codex_review_pending`. Author: Opus 5.5 (bounded child of the v5
cycle-027 PM worker). Written 2026-10-04 ~23:35 KST on `codex/robust-clip-probe-1004` (base `08da77f`). Mac CPU only,
synthetic data only. No ssh, GPU, network, real data, 2026 data, score or results directory. Nothing registered,
nothing run on a GPU.

## 요약 (Song용)

- 리뷰 M1이 요구한 전체 TRAIN 시간 측정 스크립트를 새로 만들었다. P35 측정 스크립트를 그대로 본떠서
  `residual_counts`와 robust-clip 팔 두 개만 GPU3/4에서 재고, control 464초·periodic 1,046초는 P35 측정값을 출처와 함께 고정값으로 쓴다.
- 합성 데이터 CPU 테스트 50개 통과, 일부러 넣은 버그 9개를 테스트가 모두 잡았다. 관련 6개 파일: 279개 통과, 1개 실패.
  실패 1개는 내 변경과 무관한 기존 실패로, 손대기 전 기준점에서도 똑같이 실패했다.
- 아직 실행하지 않았다. P35가 끝나고, 별도 리뷰를 받은 뒤에만 돌린다.

## What was built (commit `36d84ab` code + tests; docs in the next commit)

- `scripts/timing_probe_robust_clip.py` (new). Copied from `scripts/timing_probe_ple_residual.py` (same CLI, parent /
  worker split, per-GPU flock, one shared worker deadline, own alarm, process-group cleanup before lock release, N8 env,
  host snapshots, forbidden-key check, `-timing`/`probe` output rule, source hashes vs preflight). P36 specifics:
  - DRAFT sha `644239f3…470a` and module pin `a3d6cbce…52fa` (registry constant AND file hash) checked first in the
    parent preflight and in each worker, before env, GPU binding, sources, export and data.
  - Probed arms `residual_counts`, `robust_clip_residual_counts` only; config must be the exact 4-arm DRAFT family
    (`DRAFT-CUDA-ROBUST-CLIP-` prefix, `uses_robust_clip`, epoch cap 60, seeds 34–38).
  - Seed 39 only: any other seed refused; 39 also refused if it ever fell into the family or an earlier block.
  - Prepare: `arm_inputs` + ONE `compute_stats(_no_current_physics(TRAIN inputs), registry continuous map)` for the
    robust-clip arm, timed separately; stats sha, rule counts, `range`/`constant` column labels, stats rows
    (`check_stats_rows`) and module sha recorded; each fit's `rc_stats_sha256` and rule counts must equal them.
  - 1-epoch warm-up, warm 2-epoch measured fit, per_epoch = max(wall, report)/2; predict over N_pred TRAIN rows
    (N_pred = temperature+blend+dev+mlb_dev manifest counts) — exactly what the P35 probe did.
  - Peak device bytes per fit and after predict; peak host RSS (`ru_maxrss`, `{ru_maxrss, unit, bytes}`, KiB on Linux).
  - Summary: `FIXED_TASK_SECONDS = {control: 464, periodic_residual_counts: 1046}` with
    `FIXED_SOURCE = docs/RESULT-P35-TIMING-PROBE-3b317f2-20261004.md:30-31`; family = Σ over 4 arms of 5 × slowest;
    `gate` = `{task_seconds_max, family_seconds_max, every_arm_within, family_within, probe_exit_zero}`.
- `tests/test_timing_probe_robust_clip.py` (50 tests) + `tests/fake_robust_clip_probe_worker.py` (new; the P35 fake
  worker hard-codes the P35 arms and is not edited).
- `docs/AMENDMENT-P36-TIMING-PROBE-20261004.md`: method, gates, preconditions, command template, bounds, attempts.

## Tests (from `experiments/pitchmdp`)

Command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest
<files> -q -p no:cacheprovider`.

- `tests/test_timing_probe_robust_clip.py`: **50 passed** (9.3 s).
- The six required files (`test_timing_probe_robust_clip`, `test_cuda_robust_clip_wiring`,
  `test_cuda_robust_clip_wiring_review`, `test_direct_robust_clip_model`, `test_timing_probe_ple_residual`,
  `test_timing_probe_tabm_residual`): **279 passed, 1 failed** (23.5 s).
- The 1 failure is **pre-existing and not caused by this change**:
  `test_timing_probe_ple_residual.py::test_frozen_runner_queue_models_and_configs_unchanged_in_the_probe_window`
  diffs frozen paths against its window base `0978558`; the robust-clip wiring (merged before `08da77f`) added
  `configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json`, the three `pitchmdp/direct_robust_clip_*.py` files and changed
  `run_cuda_direct.py` / `score_cuda_direct.py` after that base. I ran the two earlier probe test files on the clean
  base `08da77f` before writing anything: same failure, 1 failed / 58 passed. Not edited here (existing probe tests
  are out of scope); the fix is a re-scope of that window by whoever owns the P35 probe record (see open question 1).

Coverage of the required points: seed 39 only (10 refused seeds, also inside `probe_arm`); wrong DRAFT sha (other
pin, edited file, self-consistent pin of an edited file); wrong module sha (probe constant, registry constant, file
hash) refused in preflight before export/config/env and in the worker before env/GPU/data; no model file (all `save`
methods and `torch.save` replaced by a raising stub, both arms probed, no file appears, no `.save(`/`checkpoint`/
`savez` in the script); no DEV/MLB DEV (worker end-to-end on CPU with the real `probe_arm`: `load_split` refuses any
split but `train`, every `np.load` path is a TRAIN array file, output dir holds only preflight + worker JSON);
projection arithmetic, 1,800 edge (≤), family 36,000 edge (≤) and 36,005 (false), failing `every_arm_within`
(new arm and reference arm), failing `family_within`, failing `probe_exit_zero` (non-zero, timeout, empty);
constants 464/1,046 and their citation lines read from the cited file; frozen paths unchanged since `08da77f`.

## Mutants (each alone on the script, restored with `git checkout --`; worktree clean after)

| id | mutation | result |
|---|---|---|
| M1 | per_epoch `min(wall, report)` instead of `max` | killed (4 failed) |
| M2 | periodic constant 1,046 → 994 | killed (5 failed) |
| M3 | `check_seed` accepts any unused seed (drops `seed != 39`) | killed (2 failed) |
| M4 | fit-vs-prepare statistics hash check disabled | killed (1 failed) |
| M5 | worker module-pin check removed | killed (1 failed) |
| M6 | family sums only the probed arms (drops 464/1,046) | killed (3 failed) |
| M7 | task gate `<` instead of `<=` | killed (1 failed) |
| M8 | `probe_exit_zero` always true | killed (1 failed) |
| M9 | predict over N_early rows instead of N_pred | killed (1 failed) |

## Deviations from the P35 probe and why

1. Check order: DRAFT sha and module pin come first in the worker (P35: env → bind → sources → config sha). Task
   requirement "before anything else"; config/module refusal now also happens before `bind_gpu`.
2. Gate keys: `every_arm_within`, `family_within`, `probe_exit_zero` (P35 had `ple_arms_within` instead of the exit
   flag). Exactly the PREREG §4 (ii) gates. `probe_exit_zero` is only written when `complete` (as P35 writes no gate
   for a partial run), so in a written gate it is true by construction; the operator still checks the process exit.
3. Two probed arms + fixed constants for the other two (P35 probed all 5). Per PREREG §4 (ii).
4. New fields: `rc` block, `host_peak_rss*`, `peak_device_bytes_measured_and_predict` (P35 read peak only after each
   fit, so predict memory was not covered), `module_sha256` in the preflight, `fixed_task_seconds` in the summary.
5. `probe_arm` refuses an arm that is not probed (control / periodic) — P35 accepted any config arm.
6. New fake worker file instead of the P35 one (which hard-codes P35 arms and must not be edited).

## Open questions (for the reviewer / PM)

1. The pre-existing P35 frozen-window test failure (above) will show in any focused run that includes
   `test_timing_probe_ple_residual.py` on this lineage. Re-scope its window end to the archived probe commit
   `3b317f2` (as its own comment says) in a separate, reviewed commit?
2. `ru_maxrss` is a process-lifetime peak, so the robust-clip arm's RSS includes the `residual_counts` arm's peak.
   Isolating it would need one process per arm (more startup cost, a different structure from P35). I kept the P35
   structure and documented the attribution; acceptable for an RSS bound, not for a per-arm difference.
3. With 464 + 1,046 fixed, `family_within` cannot fail unless an arm gate also fails (max 25,550 worker-s). It is
   kept because the prereg names it; the reviewer may want that stated in the prereg too.
4. The 30 × `compute_stats` seconds inside the robust-clip projection are not credited (as PLE binning in P35). If
   full-TRAIN `compute_stats` turned out large (81 float64 quantile passes over ≈1.25 M rows; 미측정), it inflates
   the projection only, never rescues a gate.
5. The P35 probe's 30 s allowance and the predict stand-in were copied unchanged; the robust-clip `fit` also makes
   one extra guarded input copy (`_no_current_physics`), which shows up in wall time and RSS, not in the allowance.

## Not done

No GPU run, no ssh, no smoke, no registration, no edit to any frozen/config/prereg/launch file.
