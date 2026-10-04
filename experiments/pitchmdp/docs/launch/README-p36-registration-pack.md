# P36 (robust-clip) registration pack — PREPARED, NOT REGISTERED, NOT LAUNCHED (T107, 2026-10-04)

Label: `author_only; same_model_review_pending; Codex_review_pending`. Written by a bounded Opus 5.5 author child on
`codex/p36-register-pack-1005` (base `08da77f`). Nothing here was registered, copied to a server or launched; no
`configs/EXP-P36-001.json` or `configs/EXP-P35-001.json` exists in the repo. Mirrors the P35 pack
(`make_p35_config.py`, `run-p35-pipeline.TEMPLATE.py`, `README-p35-registration-pack.md`, `LAUNCH-PLAN-p35-20261005.md`;
review ed02276 + disposition).

## Preconditions (all, before the generator is run for real)

1. P35 terminal: `queue-p35-001` `{done: 25}`, `score-p35-state.json` `complete`, and the complete P35 descriptive
   report written (pre-registration §7 (a)/(b); no pipeline/data-contract problem on the shared backbone/export).
2. P36 CUDA smoke 20/20 (4 arms × seeds 34–38) on GPU3/4 from a fresh archive/code dir, passing the coordinator checker
   `reports/research-candidate3-smoke/check_p36_smoke.py` (coordinator worktree, commit f988f0a): one `rc_stats_sha256`
   across seeds and both GPUs, module sha = pin `a3d6cbce…52fa`, sources = commit.
3. P36 timing probe (`scripts/timing_probe_robust_clip.py`, seed 39, other branch) reviewed + dispositioned, complete,
   exit 0 on both GPUs, every gate true (pre-registration §4 (ii)).
4. Same-model review of THIS pack dispositioned (and the probe-key table below checked).
5. An explicit dated go; L ≤ 16:30 KST on 2026-10-05 (else P36 waits for a new dated go, pre-registration §5).

## Files

- `make_p36_config.py` — builds `configs/EXP-P36-001.json` from DRAFT `644239f3…470a` + the P36 probe summary + the
  registered `configs/EXP-P35-001.json`. Changes only `REGISTRATION_PATHS` (the P35 set + `source_configs.ple_registered`).
- `run-p36-pipeline.TEMPLATE.py` — server-side bounded pipeline derived mechanically from `run-p35-pipeline.TEMPLATE.py`.
  `configsha='FILL_AT_REGISTRATION'` fails closed (AssertionError, exit 1), as do five more placeholders.
- `../../tests/test_cuda_robust_clip_registered_pack.py` — CPU tests (dry runs into tmp only).
- Author report: `../REPORT-p36-register-pack-author-2026-10-04.md`.

## Probe-summary keys read by the generator (to be checked against the actual P36 probe summary at disposition)

The P36 probe was written on another branch and not seen by this author. The keys below are derived from the P35 probe
summary schema (`scripts/timing_probe_ple_residual.py` `run()`/`summarize()`) and live in ONE table, `PROBE_KEYS` +
`GATE_KEYS` + `PROBED_ARMS` at the top of `make_p36_config.py` (the template repeats `GATE_KEYS`; a test pins them equal).
A missing key fails closed.

- `complete` — must be `True`
- `exit_codes` — `{gpu uuid: int}`, exactly 2 entries, all `0`
- `preflight.config_sha256` — must equal the DRAFT `644239f3…`
- `preflight.export_manifest_sha256` — must equal `49097109…f578`
- `preflight.gpus` — must equal the DRAFT GPU allow-list (and the `exit_codes` keys)
- `slowest_task_seconds_projected` — dict containing `residual_counts` and `robust_clip_residual_counts`; every value ≤ 1,800
- `family_worker_seconds_projected` — ≤ 36,000
- `gate` — `robust_clip_arms_within`, `every_arm_within`, `family_within` present and `True` (P35 used `ple_arms_within`;
  the P36 name is a GUESS), and every other boolean in `gate` also `True`

If the real P36 summary names a key differently, change the table (and `GATE_KEYS` in the template) in a reviewed commit
before registration; do not relax a gate.

## Differences from the P35 pack (every one)

Generator (`make_p36_config.py` vs `make_p35_config.py`):
1. 11 arguments instead of 9: adds `<EXP-P35-001.json>` (review a23e77b m5) and `<archive commit>` (P35 hard-coded the
   known commit `8864389`; the P36 smoke commit does not exist yet). Commit must be 7–40 lowercase hex.
2. Probe summary sha is not pinned in the generator (the P36 probe has not run); it is recorded as
   `registration.timing_probe.summary_sha256` and pinned in the template (`probesha`).
3. Probe checks read the key table above; the gate keys are the P36 names; extra checks: `preflight.gpus`, exactly two
   exit codes, every gate boolean true, task ≤ 1,800 and family ≤ 36,000 recomputed from the projections.
4. Extra deadline checks: `aggregate_by ≤ 23:00` always, and `L ≤ 16:30` when the date is 2026-10-05 (pre-registration
   §5). Strict `YYYY-MM-DD` regex in addition to `strptime`. Check order: format → rule → 16:30 → 23:00 → past date →
   existing output (so each refusal has its own message).
5. Existing output refused before any read (`assert not exists`) and again by `open(out, 'x')`.
6. `source_configs.ple_registered = {path: configs/EXP-P35-001.json, sha256}` appended at the end of `source_configs`;
   the P35 config must be named `EXP-P35-001.json`, have `experiment_id` EXP-P35-001, status REGISTERED,
   `registration.draft_sha256` = `f0ae4e91…`, and `control` / `periodic_residual_counts` JSON-exact to the DRAFT.
7. `decision` text is P36-specific and does NOT claim "Song's explicit go"; it points at `registration.go` (the P35
   LAUNCH-PLAN wording note). `registration` adds `preregistration`, drops `smoke`/`timing_probe.commit`/`result`
   (unknown now; record them in the launch record).

Template (`run-p36-pipeline.TEMPLATE.py` vs `run-p35-pipeline.TEMPLATE.py`):
1. Code dir `code-<codecommit>` (placeholder) instead of `code-8864389`; no second probe-commit code dir check (the probe
   summary's `preflight.config_sha256` is checked instead).
2. Placeholders `codecommit`, `probedir`, `probesha`, `smokeq`, `smokeruns` (all `FILL_AT_REGISTRATION`, one fail-closed
   assert after the `configsha` assert); `codecommit`/`probesha` must equal the registered config's `registration`.
3. DRAFT `DRAFT-CUDA-ROBUST-CLIP-20261005.json` / `644239f3…`; config `EXP-P36-001.json`; state `score-p36-state.json`.
4. New: server `EXP-P35-001.json` sha must equal `source_configs.ple_registered.sha256`.
5. Probe gates: `GATE_KEYS` all `True` + every gate boolean `True` + exactly two exit codes.
6. Smoke `{done: 20}`; terminal families add `('p35-001', 25)`; new check `score-p35-state.json` status `complete`.
7. No P36 output: `queue-p36-001`, `runs/EXP-P36-001`, `scores/EXP-P36-001`.
8. `pgrep -f` pattern adds `score_cuda_direct` (no scorer running either).
9. Smoke identities 4 arms × 5 seeds with `own_hashes()` computed once; module pin `a3d6cbce…52fa` checked in the registry,
   `own_hashes()`, and every robust-clip smoke `fit.json` `rc_module_sha256`; ONE `rc_stats_sha256` across the 5
   robust-clip smoke fits; the 20 smoke `fit.json` `runtime.bound_uuid` values are exactly the two pinned UUIDs.
10. Queue `p36-full-001` into `runs/EXP-P36-001` / `queue-p36-001`, `{done: 20}`; scorer `score-p36-001` into a NEW
    `scores/EXP-P36-001`. `save`, `interrupted`, `execute`, `queue` functions are byte-identical to P35 (tested).

## Launch-day steps (mirror `LAUNCH-PLAN-p35-20261005.md`; only after all preconditions)

1. `date`. Choose L = now + 20 min rounded UP to 5 min; no_new = L+5 h, kill = no_new+1 h, aggregate = kill+30 min, one
   KST date. If aggregate > 23:00 or (date 2026-10-05 and L > 16:30) → do not launch.
2. Read-only preflight in a SEPARATE ssh call: GPU3/4 by UUID idle, no `cuda_direct_queue.py|timing_probe_|
   score_cuda_direct` process, no P36 path (queue-p36-001, runs/EXP-P36-001, scores/EXP-P36-001, score-p36-state.json,
   EXP-P36-001.json, run-p36-pipeline.py), P35 state complete, disk ≥ 5 GB, export manifest.
3. In the worktree: `python make_p36_config.py <P36 probe summary> configs/EXP-P35-001.json configs/EXP-P36-001.json
   <date> <L> <no_new> <kill> <agg> <archive commit> <cycle id> "<go reference>"`; run the extractor `config_diff`
   (0 paths outside `REGISTRATION_PATHS`) and a registered-config test (P35 pattern `test_cuda_ple_registered.py`: queue
   timeout/budget/attempts/start_after/stop_rule, `protocol`, `seed_rule`, `source_configs` minus `ple_registered` equal
   the DRAFT; `control`/`periodic_residual_counts` equal `configs/EXP-P35-001.json`; `registration.go` non-empty);
   commit config + test.
4. `cp run-p36-pipeline.TEMPLATE.py run-p36-pipeline.py`, fill `configsha` (generator output) and the five other
   placeholders, `py_compile`, commit.
5. Server copies as NEW files with scp (one call), then a separate read-only `sha256sum` (config and pipeline equal the
   commit); `a-w` in its own call. No server launch json; never `p33-launch.json`.
6. Launch (one call, stdin/stdout detached so ssh returns; no remote wait loop):

   `cd /home/sds733/pitcheezy-autonomy/20261004 && nohup setsid timeout --kill-after=60 <budget> /home/sds733/venv/bin/python run-p36-pipeline.py < /dev/null > p36-pipeline.log 2>&1 &`

   with `<budget>` = (aggregate − L) + 900 + 600 s, ≤ 40,000 (e.g. L 16:30, aggregate 23:00 → 23,400 + 1,500 = 24,900).
   Plain `python`, never `-O`/`PYTHONOPTIMIZE` (all checks are asserts). The command string must not contain
   `cuda_direct_queue.py`, `timing_probe_` or `score_cuda_direct`, and no tail/less on probe/queue logs may be open
   (newsrv procps `pgrep -f` sees every user's processes and does not exclude ancestors: fail-closed false block only).
7. ONE confirmation read a few minutes later (state file + queue status), then return. Mac-side launch record only (git
   doc on the branch). After `complete`: copy scores read-only, descriptive report only (every arm and seed, repeated
   DEV exposure stated, no winner).
