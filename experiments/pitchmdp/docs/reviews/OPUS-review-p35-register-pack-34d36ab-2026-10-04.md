# Distinct same-model review — P35 registration pack at 34d36ab (2026-10-04)

Label: `same_model_review; Codex_review_pending`. Reviewer: Opus 5.5 (distinct session, not the author, not Codex).
Scope: `docs/launch/make_p35_config.py`, `docs/launch/run-p35-pipeline.TEMPLATE.py`,
`docs/launch/README-p35-registration-pack.md`. Read only. No ssh/newsrv/GPU/network, no registration, no launch, no
score read, no 2026 data. Temp output under `/tmp/pz-p35-pack-review/` (kept).

**Verdict: 0 BLOCKER, 0 MAJOR, 6 MINOR, 6 NOTE. No fix commits.** The pack is fit to turn a go into a registration
once the PM fixes the deadline rule and outer timeout (결정 필요 below). Nothing in it claims more than timing and
registration readiness.

## Answers

1. **Generated config vs DRAFT.** It differs only at the extractor's `REGISTRATION_PATHS` for every valid argv I tried
   (3 argv sets, incl. a different date, a 23:59 aggregate, a quoted cycle id). Every other key is copied by reference
   (`out[k] = v`); `queue` is a shallow copy with only the 6 listed keys changed; `execution` is `{**v, 'enabled': True}`;
   `registration` is appended last. Top-level and queue key order equal the DRAFT. Arms, seeds, fits, training
   (epochs 60, patience 5), calibration, comparison (7 `ple_contrasts`, headline), task timeout 3600, family budget
   36000, max_attempts 2, order, start_after and stop_rule are untouched. Negative controls (training.epochs 61,
   queue.task_timeout_seconds, queue.start_after, seeds, comparison.ple_headline) are all caught by `config_diff`.
   Dry run sha with the README argv = `0336100743d0b3e360ad44a324734ad8f4a504acfddd593a65e051248b52dd45` (= README).
2. **Pipeline template.** `diff run-p34-pipeline.py run-p35-pipeline.TEMPLATE.py` shows only the expected
   substitutions: code dir `code-8864389`; DRAFT sha f0ae4e91… checked in `code-8864389` and `code-3b317f2`; probe
   path/sha 311f0d4f…; smoke `queue-ple-smoke-001` {done: 25} and `runs/DRAFT-PLE-smoke`; gate `ple_arms_within`;
   terminal families now include `p34-001` 25 and `p30-002` 50 (the P34 "no queue-p30-002" check is correctly
   removed); no `queue-p35-001` / `runs/EXP-P35-001` / `scores/EXP-P35-001`; pgrep widened to `timing_probe_`; state
   file `score-p35-state.json`; queue/score into `runs/EXP-P35-001` and new `scores/EXP-P35-001`. No leftover
   P34/TabM path, sha or gate name (only the TabM *arm* names, which are in the DRAFT). With the placeholder the script
   stops at line 11 (`AssertionError: template: configsha not filled`, exit 1) before the state check, before any
   `save()` and before any file read. The smoke identity check matches how the smoke ran: smoke used
   `--config code-8864389/configs/DRAFT-…json --output runs/DRAFT-PLE-smoke --smoke`
   (`SMOKE-PLE-8864389-20261004.md`); the queue's `is_complete` verifies with
   `expected_identity(config, manifest, arm, seed, smoke, 'cuda', own_hashes())` (`cuda_direct_queue.py` at 8864389,
   `is_complete`: pinned sources minus the queue file = `run_cuda_direct.own_hashes()`), and `task_dir(out, arm, seed,
   True)` = `out/<arm>-smoke/<seed>` (`run_cuda_direct.py:705`). The template calls the same functions with the same
   arguments on the DRAFT parsed from `code-8864389`, so the 25 checks reproduce the queue's own check. The scorer at
   8864389 handles P35: `score()` requires REGISTERED/enabled, calls `ple_registry.check_contrasts(config)`, tests
   `rc.uses_ple` FIRST and refuses any other contrast family (`score_cuda_direct.py` `if rc.uses_ple(config): # first`),
   requires one bins hash across PLE fits (`check_one_bins_hash`), writes `ple_contrasts`, `ple_bins`, `seed_block`
   with `headline`, `same_gpu_pairs`, `decision: None`, `ranking: None`, and refuses an existing destination. Local
   check on the dry config: `rc.config_check` OK, `require_executable(smoke=False)` OK, `uses_ple` True, 7 contrasts,
   queue `Deadlines` OK; the DRAFT still refuses real fits. `git diff --stat 8864389 34d36ab` over pitchmdp/,
   scripts, configs = only the new `timing_probe_ple_residual.py` (not in `code-8864389`).
3. **pgrep.** `subprocess.run(['pgrep','-f',…])` has no shell wrapper, pgrep never reports itself, and the pipeline's
   own command line (`python run-p35-pipeline.py`) does not contain either pattern; no child exists yet at that line.
   So it cannot match itself. Caveat MINOR-3: on Linux procps (newsrv) ancestors are NOT excluded (macOS/BSD pgrep
   excludes ancestors by default, `man pgrep -a`), and `-f` matches every user's processes. Both only cause a false
   block (fail closed), never a false pass.
4. **Score / 2026 influence.** None found. The generator reads only the DRAFT and the sha-pinned probe summary (timing
   numbers only); the pipeline scores only after a complete 25/25 family and writes to a new directory; the extractor
   is pinned before any score. Data rules, export manifest 49097109… and the runner's year guard (`run_cuda_direct.py`
   `date // 10000 > 2025` check) are unchanged. Wording stays at timing/registration readiness; `decision`,
   `exposure` and `review` keep "No scientific N/G decision", "descriptive only", `Codex_review_pending`.
5. **README gaps.** See MINOR-1, MINOR-2, MINOR-4 and NOTE-5.

## Findings (most severe first)

- **MINOR-1 — no fixed deadline rule** (`README-p35-registration-pack.md:19-20`). Step 1 says "choose deadlines from
  the probe … with margin" and quotes "P34 actual was 0.43 × projection". That invites a deadline tighter than the
  cap-based projection. If `kill_own_after` falls before the family ends, tasks are killed, the pipeline blocks at
  `full family incomplete`, and a rerun needs a new queue dir and a re-registration (new sha). Proposed rule (fixed
  before go, from the probe only, not from P34's 0.43): with launch time L, `no_new_fit_after ≥ L + 5 h` (family budget
  36,000 worker-s / 2 GPUs), `kill_own_after ≥ no_new + 1 h` (task timeout 3600), `aggregate_by ≥ kill + 30 min`,
  all on one KST date. (Cap-based makespan bound: 25,172/2 + 1,225 ≈ 13,811 s ≈ 3.8 h, so 5 h already has margin.)
- **MINOR-2 — outer `timeout <budget>` not specified** (`README…:23`). If it is shorter than the queue run + scorer,
  SIGTERM blocks the family midway. Proposed: budget ≥ (aggregate_by − L) + 900 s scorer + 600 s, and ≤ the
  pipeline's own 40,000 s queue wait.
- **MINOR-3 — pgrep false blocks on newsrv** (`run-p35-pipeline.TEMPLATE.py:55`). procps pgrep does not exclude
  ancestors, and `timing_probe_` also matches e.g. `tail -f …timing_probe_…log` or another user's process. Fail
  closed only. Operational rule: the launching shell string must not contain `cuda_direct_queue.py` or
  `timing_probe_` (do the preflight pgrep in a separate ssh call), and close any tail/less on probe logs.
- **MINOR-4 — the generator records "Song's explicit go" without a pointer to it** (`make_p35_config.py:33`). The
  claim is baked in; running the script before a go produces a config that falsely states one. Proposed: take a go
  reference argv (ledger path + line or commit + KST time) and store it as `registration.go`; or have the
  registered-config test assert that field is non-empty. (Does not change any non-registration path.)
- **MINOR-5 — argv not validated** (`make_p35_config.py:13,24`). Date and HH:MM are free strings and compared as
  strings: `garbage 15:00 16:00 16:30` → exit 0 (sha 51518544…); `10:00 9:00 9:30` would pass `no_new < kill < agg`
  as strings while being out of order numerically; extra argv are ignored. The queue's `Deadlines`
  (`cuda_direct_queue.py`, `strptime('%Y-%m-%d')`, ordered check) still fails closed — but only on the server, after
  the config is committed and hashed, costing a re-registration. Proposed: `datetime.strptime(date,'%Y-%m-%d')`,
  `re.fullmatch(r'\d\d:\d\d')` for the three times, `len(sys.argv) == 8`, and date ≥ today KST.
- **MINOR-6 — output can overwrite any file** (`make_p35_config.py:72`). `write_text` without `'x'`; passing the
  DRAFT path as `<out.json>` silently replaces the DRAFT in the worktree. Proposed: `open(out_path, 'x')`.
- **NOTE-1 — `assert` under `python -O`.** All checks in both scripts are `assert`s (same as P34). With `-O` or
  `PYTHONOPTIMIZE` they vanish; the placeholder would still be refused by the queue's config-sha preflight, but the
  filled pipeline's smoke/terminal/pgrep/probe checks would be skipped. README's command uses plain `python`; keep it so.
- **NOTE-2 — pipeline compares 11 top-level keys** (`TEMPLATE:44`), not `queue` timeout/budget/attempts/start_after/
  stop_rule or `protocol`/`seed_rule`/`source_configs`. Covered by the configsha pin + generator + extractor
  `config_diff`; same as P34. The registered-config test should assert them (P34 test pattern
  `test_cuda_tabm_residual_registered.py::test_only_identity_status_prose_and_launch_day_deadlines_differ`).
- **NOTE-3 — terminal check is "done == N" exactly.** If P30-002 or P34 had ended with a failed task, P35 can never
  start under this template (fail closed). I could not check the server; the P30-002 extractor exists and requires
  {done: 50}.
- **NOTE-4 — `registration.review`** cites "dispositions … fe7ee46+8864389": fe7ee46 is the wiring review commit
  (with W5/W2 fixes), 8864389 the disposition. Accurate enough; wording only.
- **NOTE-5 — README says "without new code"** (`README…:5`) but step 1 adds a registered-config test. Suggest writing
  that test now against a fixture (dry config), so launch day only swaps in the real file. Step 1 should also say:
  run the extractor's `config_diff` (0 paths outside `REGISTRATION_PATHS`) on the final file before commit.
- **NOTE-6 — scorer timeout 900 s** is the P34 value; P35 adds the bins check only. Duration for P35: 미측정.

## Commands (all on the Mac, OMP/MKL/OPENBLAS=1, `perl -e 'alarm 600; exec @ARGV'`, `.venv/bin/python`)

- `diff run-p34-pipeline.py run-p35-pipeline.TEMPLATE.py` → exit 1 (differences listed above, all expected).
- `diff make_p34_config.py make_p35_config.py` → exit 1 (expected).
- `shasum -a 256 configs/DRAFT-CUDA-PLE-RESIDUAL-20261004.json` → f0ae4e91…b9b2; probe summary → 311f0d4f…7ff5.
- `make_p35_config.py <summary> dry1.json 2026-10-05 15:00 16:00 16:30 dry-run` → exit 0, sha 03361007…2dd45.
- same with `2026-10-06 09:00 10:00 23:59 'cyc "x" 2'` → exit 0; `9:00 10:00 11:00` → AssertionError exit 1;
  `16:00 15:00 16:30` → AssertionError; `garbage … extra-arg` → exit 0; `15:00 16:00 1630` → AssertionError exit 1.
- extractor `config_diff` on dry1/dry2/dry5 → 10/11/11 paths, 0 outside `REGISTRATION_PATHS`, key order equal → exit 0.
- 5 negative controls on dry1 → all caught → exit 0.
- `rc.config_check`, `require_executable(False)`, `uses_ple`, `direct_ple_registry.check_contrasts`, queue `Deadlines`
  on dry1/dry2 → OK; DRAFT `require_executable(False)` → refused; dry5 `Deadlines` → ValueError → exit 0.
- `python run-p35-pipeline.TEMPLATE.py` → `AssertionError: template: configsha not filled`, exit 1 (no file touched;
  `/home/sds733` does not exist on the Mac).
- `git diff --stat 8864389 3b317f2 -- pitchmdp scripts(3) configs` → empty; `8864389..34d36ab` → only the probe script.
- pgrep demo on macOS → `b''` with and without a matching parent (BSD excludes ancestors; procps does not).
