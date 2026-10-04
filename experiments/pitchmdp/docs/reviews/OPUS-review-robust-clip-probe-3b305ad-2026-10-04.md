# Review — P36 robust-clip timing-probe script (08da77f..3b305ad), T109

Label: `same_model_review; Codex_review_pending`. Reviewer: Opus 5.5, distinct from the probe author (T106c) and the
PM worker. Written 2026-10-04 ~23:45 KST on `codex/robust-clip-probe-review-1004` (at `3b305ad`, plus one test
commit `da750d1`). Mac CPU only, synthetic data only. No ssh, GPU, port, network, real data, 2026 data, score or results
directory. No frozen file edited.

Reviewed blobs: `scripts/timing_probe_robust_clip.py` sha256 `6fe717f5…cbb75` (unchanged since `36d84ab`),
`tests/fake_robust_clip_probe_worker.py` `287e23d3…ae84`. Pins re-hashed: prereg `947e56c6…4750`, DRAFT
`644239f3…470a`, module `a3d6cbce…52fa`. All three match.

## 요약 (Song용)

- 차단 문제(BLOCKER)와 큰 문제(MAJOR)는 없다. P35에서 안전하게 돈 측정 스크립트와 동작이 같고, 바뀐 부분 10곳은 모두 사전등록 §4 (ii)에 맞는다.
- 테스트가 놓친 변형 1개(규칙 개수 비교만 빠짐)를 찾아 테스트를 하나 추가했다 (`da750d1`, 51개 통과).
- 등록 묶음(다른 브랜치)이 이 스크립트가 쓰지 않는 게이트 이름 `robust_clip_arms_within`을 요구한다. 그대로면 측정이 통과해도 등록이 막힌다. 고칠 쪽은 등록 묶음이다.
- 판정: P35가 끝난 뒤 한 번 돌려도 된다(아래 명령, 예상 2~5분, 상한 1,830초).

## Findings

| # | sev | where | one line |
|---|---|---|---|
| X1 | MAJOR (other branch, not this probe) | pack `ac097b4` `docs/launch/make_p36_config.py:36` | requires gate key `robust_clip_arms_within` the probe never writes → a passing probe fails closed at registration |
| m1 | MINOR (fixed, test only) | `tests/test_timing_probe_robust_clip.py` | mutant R7 (fit-vs-prepare rule-count check dropped, hash kept) survived 50 tests → new test `da750d1` |
| m2 | MINOR (inherited from P35, not applied) | `scripts/timing_probe_robust_clip.py:401-402,360` | output-name rule and pins are checked after `mkdir`: a wrong `--output` creates the directory, then refuses |
| m3 | MINOR (not applied) | `scripts/timing_probe_robust_clip.py:284-285` | summary `per_arm` rows carry no memory fields; prereg "printed per arm" is met only in `worker-*.json` |
| m4 | MINOR (sequencing) | `tests/test_timing_probe_robust_clip.py:38-46,701-707` | this probe's own frozen-window test will break the same way P35's did once the pack or the P35-test fix lands on this lineage |
| n1 | NOTE | `:300` | `probe_exit_zero` is true by construction when a gate is written; parent exit code + `complete` are the real checks |
| n2 | NOTE | `:291,298-299` | `family_within` cannot fail unless `every_arm_within` fails (max 5 × 5,110 = 25,550) |
| n3 | NOTE | `:71` | fixed 464/1,046 were measured at host load ≈ 15; not re-measured |
| n4 | NOTE | `:262-269` | real-task costs not in the probe are covered only by the 30 s allowance; outweighed by conservative terms |
| n5 | NOTE | `:159-163` | `ru_maxrss` is cumulative per worker (robust-clip row = max of both arms) |
| n6 | NOTE | `:378-390` | `stop_group` returns if the leader already exited (inherited; worker spawns no children) |

### X1 — MAJOR for the registration pack (do not edit the pack here)

`make_p36_config.py:36` `GATE_KEYS = ('robust_clip_arms_within', 'every_arm_within', 'family_within')` and
`:101-103` assert every key present and `True`. `summarize()` (`timing_probe_robust_clip.py:297-300`) writes exactly
`task_seconds_max, family_seconds_max, every_arm_within, family_within, probe_exit_zero`. Failure: the probe passes,
`make_p36_config.py` raises `probe gate key missing: robust_clip_arms_within`, registration is blocked (fails closed,
not unsafe).

**The probe's names are the right ones.** PREREG §4 (ii) (`PREREG-DRAFT-EXP-P36-001…md:112-113`) names exactly three
gates: "every arm ≤ 1,800 s, family = Σ_arms 5 × slowest ≤ 36,000 worker-s, probe exit 0" — i.e.
`every_arm_within`, `family_within`, `probe_exit_zero`. There is no robust-clip-only arm gate in the prereg (P35 had
`ple_arms_within` because its DRAFT §6 named the new arms separately). Suggested pack change (for the pack reviewer):

```diff
-GATE_KEYS = ('robust_clip_arms_within', 'every_arm_within', 'family_within')  # P35: ple_arms_within -> P36 name
+GATE_KEYS = ('every_arm_within', 'family_within', 'probe_exit_zero')  # PREREG-DRAFT-EXP-P36-001 section 4 (ii)
```

Every other pack key matches the probe output: `complete`, `exit_codes` (ints from `Popen.wait`),
`preflight.config_sha256`, `preflight.export_manifest_sha256`, `preflight.gpus`, `slowest_task_seconds_projected`
(holds all 4 arms incl. the two constants; the pack's "every value ≤ 1,800" check is then the same as
`every_arm_within`), `family_worker_seconds_projected`.

### m1 — test gap, fixed in `da750d1`

Mutant R7: delete `or report.get('rc_rule_counts') != rule_counts` at `:240-241` (hash comparison kept) → 50 passed.
Not strictly equivalent: `stats_sha256` hashes (center, scale) bytes, and an `iqr` and a `range` column could carry
the same scale bytes. Added `test_probe_arm_refuses_rule_counts_that_differ_from_the_fit_at_the_same_stats_hash`
(same stats, one rule label changed → `RuntimeError`). R7 is now killed; the file passes 51/51.

### m2 — name rule and pins after `mkdir` (inherited, optional)

`run()` does `args.output.mkdir(parents=True, exist_ok=False)` (`:401`) before `preflight()` checks the pins and the
`probe`/`-timing` name (`:356-361`). Failure: a mistyped `--output runs/EXP-P36-001` creates that (family) directory
and then refuses; a later family launch that requires a fresh output would refuse it in turn. Also a pin refusal leaves
an empty output directory, and the amendment does not say whether that counts as attempt 1. Identical in the P35 script
(`timing_probe_ple_residual.py` same lines), so not a regression. Suggested diff (not applied, to keep the reviewed
script blob identical to `36d84ab`):

```diff
 def run(args, config):
     from cuda_direct_queue import GpuLock  # the common per-GPU flock of the family queue
+    if not args.output.name.endswith('-timing') or 'probe' not in args.output.name:
+        raise ValueError('Probe output must be a NEW <...>probe<...>-timing directory (never a family runs/ dir)')
     args.output.mkdir(parents=True, exist_ok=False)  # never reuse an output
```

Amendment text suggestion: "a refusal before `preflight.json` is written is not an attempt; the empty directory is
kept and the next attempt still uses a new name".

### m3 — memory per arm only in worker JSON (optional)

Prereg §4 (ii): "peak device bytes and peak host RSS (`ru_maxrss`) printed per arm". `probe_arm` records them
(`measured.peak_device_bytes`, `peak_device_bytes_measured_and_predict`, `host_peak_rss`), but `summarize()` copies
only the projection fields into `per_arm`. The result writer must read `worker-*.json` (as the P35 result table did).
Optional diff at `:284-285`: add `'peak_device_bytes': record['peak_device_bytes_measured_and_predict'],
'host_peak_rss': record['host_peak_rss']` to each row (passes the forbidden-key check). Not applied.

### m4 — window tests collide (sequencing for the PM)

The new test's `FROZEN` list contains `docs/launch`, `configs` and `tests/test_timing_probe_ple_residual.py`, window
base `08da77f` .. working tree. Merging the P36 pack (adds `docs/launch/make_p36_config.py`, later
`configs/EXP-P36-001.json`) or applying the P35-test fix below onto this lineage makes this test fail exactly as the
P35 one does now. Recommend: in the commit that records the P36 probe result, re-scope this test's window end to the
archived probe commit (`PROBE_RUN_COMMIT`, as T70c did for P34 at `test_timing_probe_tabm_residual.py:196`) and apply
the P35 fix in the same commit.

### Notes

- n1: `summarize` runs only when `complete` (both codes `== 0` and both worker files), so a written `gate` always has
  `probe_exit_zero: true`. Harmless; the operator checks the probe exit code (1 when incomplete) and `complete`. No
  path reports a failure as pass: timeout → `'timeout'` code → incomplete → no `gate`; worker crash → no JSON/non-zero;
  wrong arm order/count → `summarize` raises (no `summary.json`, parent non-zero); NaN projection → comparisons false.
- n2: with 464 + 1,046 fixed, family ≤ 25,550 whenever both probed arms ≤ 1,800. Kept because the prereg names it.
- n3: P34→P35 drift on the same arms was +5 % (periodic 15.98 → 16.85 s/epoch). 1,046 s has 1.72× headroom to 1,800.
- n4: not measured by the probe: `arm_inputs` + `load_split` of earlystop/temperature/blend/dev/mlb_dev,
  `fit_temperature`, `savez_compressed`, `model.save`, a cold first epoch. Conservative terms on the other side:
  startup counted in full for every arm, `compute_stats` ≈ 30× inside the projection (plus once in prepare), the fit's
  `_no_current_physics` copy also ≈ 30×, every fit at the 60-epoch cap. P34 actual tasks were ≤ 0.78× projection.
- n5/n6: see open questions 2; `stop_group` same as P35/queue.

## 1. Fidelity to P35 — every behavioural difference (`diff timing_probe_ple_residual.py timing_probe_robust_clip.py`)

Unchanged (verified line by line): per-GPU `GpuLock` in the common lock dir, refuse-if-busy; one shared deadline
(`:424`); worker `signal.alarm(1560)` (`:478`); `start_new_session` + `stop_group` (SIGTERM, 20 s, SIGKILL) for every
worker **before** any lock release in `finally` (`:430-436`); SIGTERM/SIGHUP/SIGINT handler; N8 env check in parent
and worker; host snapshots pre/post; GPU set must equal the DRAFT allow-list (the two UUIDs); `bind_gpu` before torch;
sources = preflight, unchanged at end; `verify_export` full hash; `load_split(..., 'train')` only; no `save`; forbidden
key check on worker and summary; `-timing`/`probe` output rule; `exist_ok=False`.

| # | difference | judgement |
|---|---|---|
| D1 | worker: DRAFT sha + module pin first (before env/bind/sources); P35 checked config sha after bind+sources | better; prereg/task order. Killed by R1/R9 tests |
| D2 | `check_module_pin` (registry constant AND file hash) in preflight, worker and `probe_arm` | correct; R2 killed |
| D3 | 4-arm config, 2 probed arms; `probe_arm` refuses non-probed arms | per §4 (ii) |
| D4 | fixed constants 464/1,046 for control/periodic, cited `RESULT-P35…md:30-31` (test reads the lines) | correct: DRAFT `control` and `periodic_residual_counts` entries, `training`, `export`, `device` are byte-equal to the P35 DRAFT (checked) |
| D5 | seed 39 only; refused if in family 34-38 or `EARLIER_SEEDS` (union = 0-33, checked) | correct |
| D6 | prepare: one `compute_stats(_no_current_physics(inputs), registry continuous)`, separately timed; fit's `rc_stats_sha256` + `rc_rule_counts` must equal it | same call as `direct_robust_clip_model.fit:175-178`; R7 needed m1 |
| D7 | `rc` record: stats sha, rule counts/columns, `check_stats_rows(ple_bin_rows(inputs))` (= runner `:802,845`), module sha | correct, TRAIN-derived only |
| D8 | host RSS (`ru_maxrss`, KiB on Linux) after startup/prepare/each fit/predict; device peak after predict | new info fields; n5 |
| D9 | gate keys `every_arm_within`, `family_within`, `probe_exit_zero` (P35: `ple_arms_within`, no exit flag) | per §4 (ii); see X1 |
| D10 | `ple` effective-parameter field removed; `check_config_file` raises `ValueError` (P35 wrapped to `RuntimeError`) | harmless; error JSON records `repr` either way |

## 2. Projection

`task = startup + prepare + 60 × max(wall, report)/2 + predict(N_pred TRAIN rows, N_pred = temperature+blend+dev+mlb_dev
manifest n) + 30` (`:262-269`), slowest GPU per probed arm (`max`, R4 killed), constants for the other two, family =
`5 × Σ_4 arms slowest` (`:291`), gates `≤ 1800` / `≤ 36000` (inclusive, edges tested). Matches prereg §4 (ii) and the
amendment. Constants cited correctly (`RESULT-P35…md:30` `| 464 s |`, `:31` `| 1,046 s |`). Real-task row counts:
the runner fits with earlystop rows (N_early) as validation and predicts exactly temperature+blend+dev+mlb_dev
(`run_cuda_direct.py:805-816`), so the stand-ins are size-exact. `compute_stats` is credited once in prepare and
≈ 30× inside the fit term (conservative, not subtracted). Nothing found that is optimistic in aggregate (n4).

## 3. Summary schema (exact keys written)

`protocol, exit_codes, complete, written_utc, preflight{…, config_sha256, export_manifest_sha256, module_sha256, gpus,
fixed{…}}, host_post` and, only when complete: `per_arm, slowest_task_seconds_projected (4 arms),
family_worker_seconds_projected, probed_task_seconds_projected, fixed_task_seconds{arms, source, measured_here},
marginal_arms, gate{task_seconds_max, family_seconds_max, every_arm_within, family_within, probe_exit_zero}, label`.
These are the prereg names; the pack should align to them (X1).

## 4. Pre-existing P35 test failure

`test_timing_probe_ple_residual.py::test_frozen_runner_queue_models_and_configs_unchanged_in_the_probe_window` fails
identically at `08da77f` and at `3b305ad`: `git diff --name-only 0978558 <rev> -- FROZEN` lists the same 6 paths in
both (`configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json`, `pitchmdp/direct_robust_clip_{factory,model,registry}.py`,
`scripts/run_cuda_direct.py`, `scripts/score_cuda_direct.py`); `git diff --name-only 08da77f 3b305ad -- FROZEN` is
empty. Cause: the robust-clip wiring legitimately changed frozen paths after the P35 probe ran. `git diff --name-only
0978558 3b317f2 -- FROZEN` is empty and `3b317f2` is an ancestor of HEAD. Recommended fix (not applied; P35 records
frozen), the P34 precedent:

```diff
+PROBE_RUN_COMMIT = '3b317f2'  # the archived P35 probe run commit (RESULT-P35-TIMING-PROBE-3b317f2-20261004.md)
 ...
-    changed = subprocess.run(['git', 'diff', '--name-only', WINDOW_BASE, '--', *FROZEN], cwd=REPO,
+    changed = subprocess.run(['git', 'diff', '--name-only', WINDOW_BASE, PROBE_RUN_COMMIT, '--', *FROZEN], cwd=REPO,
```

Apply it together with the m4 re-scope (the P35 test file is in this probe's `FROZEN` list).

## 5. Commands, counts, mutations

From `experiments/pitchmdp`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 perl -e 'alarm 600; exec @ARGV'
/Users/song/Projects/pitcheezy/.venv/bin/python -m pytest <file> -q -p no:cacheprovider`:

- `tests/test_timing_probe_robust_clip.py` at `3b305ad`: **50 passed**, exit 0 (9.6 s).
- `tests/test_timing_probe_ple_residual.py`: **1 failed, 39 passed**, exit 1 — the §4 failure only.
- after `da750d1`: `tests/test_timing_probe_robust_clip.py` **51 passed**, exit 0.
- window comparisons: `git diff --name-only` as in §4 (bash, explicit path list).

Own mutations (each alone on the script, restored by rewriting the original text; `git status` clean afterwards except
the committed test):

| id | mutation | result |
|---|---|---|
| R1 | worker module pin moved after `bind_gpu` | killed (1 failed) |
| R2 | preflight module pin removed | killed |
| R3 | `finally`: locks released before worker groups stopped | killed |
| R4 | slowest GPU `max` → `min` | killed |
| R5 | output rule: `probe` substring dropped | killed |
| R6 | startup dropped from the projection | killed |
| R7 | fit-vs-prepare rule-count comparison dropped (hash kept) | **survived** → `da750d1` → killed |
| R8 | allowance 30 → 0 | killed |
| R9 | worker env check removed | killed |

## 6. Answers to the author's open questions

1. Yes — re-scope the P35 window end to `3b317f2` in a separate reviewed commit, but in the same commit re-scope this
   probe's window (m4), because the P35 test file is frozen here. Best done in the P36 probe-result commit.
2. Acceptable. The prereg asks for RSS "printed per arm", not a per-arm difference; the cumulative peak is an upper
   bound for the second arm and is labelled. One process per arm is not worth breaking P35 fidelity.
3. Agree it is implied (n2). No prereg edit needed (frozen); stating it in the result file is enough.
4. Correct and conservative. If `compute_stats` were large it can only push a gate to false, never to true. If the
   robust-clip arm then fails by a margin smaller than ≈ 30 × `compute_stats_seconds`, that is a recorded fail (no
   re-run, no subtraction) per the amendment.
5. The `_no_current_physics` copy inside `fit` is inside the measured wall, so it is carried ≈ 30× in the projection,
   not just once; RSS includes it. No allowance change needed.

## 7. Exact one-shot server command (after EXP-P35-001 is terminal)

Code: a commit whose script/config/model files equal `36d84ab` — e.g. `da750d1` (this branch; diff to `3b305ad` is one
test file) or the PM's later disposition commit (then substitute its 7-char sha for `da750d1` in both places).
`git archive da750d1 configs experiments/pitchmdp | gzip -n` → record sha → NEW read-only `code-da750d1` in the same
server root as the P35 probe (where `export-49097109` and `runs/` live). Preconditions: GPU3/4 no compute app, common
locks free, no P36 smoke/queue/score/probe running, `runs/DRAFT-ROBUST-CLIP-probe-da750d1-timing` absent, no
`CUDA_VISIBLE_DEVICES` in the shell.

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
    timeout --kill-after=30 1800 /home/sds733/venv/bin/python \
      code-da750d1/experiments/pitchmdp/scripts/timing_probe_robust_clip.py \
      --config code-da750d1/configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json \
      --config-sha256 644239f3b2aaa4410391211ab1e4d972f13e1a6b025a026b058f0dff319c470a \
      --export export-49097109 \
      --export-manifest-sha256 49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578 \
      --output runs/DRAFT-ROBUST-CLIP-probe-da750d1-timing \
      --lock-dir /home/sds733/pitcheezy-autonomy/gpu-locks \
      --gpu GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9 --gpu GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99 run

Foreground, one attempt (a second only if incomplete, new `-a2-timing` output). Expected wall ≈ 2–5 min (P35 did
5 arms × 3 epochs in 266–267 s per worker; here 2 arms × 3 epochs + 3 `compute_stats`, whose time is 미측정). Hard
bound 1,830 s (timeout 1,800 + kill-after 30; workers end by the shared 1,500 s deadline + 20 s grace, alarm 1,560 s).
Then: `sha256sum` the 6–7 files, copy to the Mac evidence dir, write `RESULT-P36-TIMING-PROBE-<sha>-…md`.

## 8. Disposition

**Ready to run once after EXP-P35-001 is terminal**, subject to the PM disposition of this review and the amendment's
preconditions. No blocker or major finding in the probe; m2/m3 are optional and, if applied, need the amendment's
"reviewed commit" to move to that commit. Before the summary is consumed for registration, the pack must drop
`robust_clip_arms_within` (X1). Nothing was registered, run on a GPU or pushed.
