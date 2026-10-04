# P36 registration pack — author report (T107, v5 cycle-028, 2026-10-04 ~23:35 KST)

Label: `author_only; same_model_review_pending; Codex_review_pending`. Author: bounded Opus 5.5 child of the delegated PM
worker, branch `codex/p36-register-pack-1005` (base `08da77f`). Mac CPU only, synthetic inputs only. No ssh, GPU,
network, install, 2026 data, score or results directory. NOT registered, NOT archived, NOT launched. No
`configs/EXP-P36-001.json` / `configs/EXP-P35-001.json` written in the repo (dry runs go to pytest tmp dirs).

## What was built (commit 6e3079e)

- `docs/launch/make_p36_config.py` — P35-style generator. Verifies DRAFT sha `644239f3…470a`; checks the registered
  `EXP-P35-001.json` (name, id, REGISTERED, built from P35 DRAFT `f0ae4e91…`, `control` + `periodic_residual_counts`
  JSON-exact to the DRAFT) and records `source_configs.ple_registered` = path + sha256 (prereg §8 item 5 / review m5);
  requires probe complete, exactly two exit codes all 0, preflight config/manifest/GPUs, every `GATE_KEYS` entry present
  and `True`, every other gate boolean `True`, probed arms projected, task ≤ 1,800, family ≤ 36,000 (missing key → fail);
  one-date deadline rule (no_new ≥ L+5 h, kill ≥ no_new+1 h, agg ≥ kill+30 min, agg ≤ 23:00, L ≤ 16:30 on 2026-10-05);
  refuses wrong argc, malformed/past date, non-HH:MM, empty cycle/go, non-hex commit, existing output. Changes only
  `REGISTRATION_PATHS`. Probe-summary keys in one table (`PROBE_KEYS`, `GATE_KEYS`, `PROBED_ARMS`).
- `docs/launch/run-p36-pipeline.TEMPLATE.py` — derived from the P35 template (differences listed in the README).
  `configsha='FILL_AT_REGISTRATION'` → `AssertionError: template: configsha not filled`, exit 1, nothing written.
- `tests/test_cuda_robust_clip_registered_pack.py` — 57 CPU tests.
- `docs/launch/README-p36-registration-pack.md` — status, preconditions, probe-key caveat, every P35→P36 difference,
  launch-day steps.

## Test commands and counts

All from `experiments/pitchmdp`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10
/Users/song/Projects/pitcheezy/.venv/bin/python -m pytest <files> -q -p no:cacheprovider`:

- `tests/test_cuda_robust_clip_registered_pack.py` → **57 passed** (1.2 s).
- That file + `tests/test_cuda_robust_clip_wiring.py` + `tests/test_cuda_robust_clip_wiring_review.py` → **214 passed**
  (5.6 s; i.e. 157 wiring + wiring-review tests unchanged).
- `tests/test_cuda_ple_registered.py` → **4 failed, 1 passed**: every failure is `FileNotFoundError` for
  `configs/EXP-P35-001.json`, which exists only after P35 registration (same as review a23e77b reported; I may not create
  it). Covered instead by `test_the_p35_fixture_passes_the_p35_registered_config_tests`: 4 of its 5 functions (3 + the
  dated deadline test while the date is 2026-10-05) pass against the tmp `make_p35_config.py` dry output; the remaining
  one pins the real probe summary sha `311f0d4f…` and cannot pass on a synthetic summary.

How the fixture works: a synthetic P35-schema summary is fed to `make_p35_config.py` executed from its own path with
the pinned `PROBE_SHA` string swapped in memory (`exec` of the patched source; the file is never edited), producing a tmp
`configs/EXP-P35-001.json`; then `make_p36_config.py` runs on a synthetic P36 summary. Both run under plain python.

## Mutants (each alone, `git checkout --` after; all killed; `git status` clean afterwards)

| id | file | mutation | result |
|---|---|---|---|
| M1 | make_p36_config.py | `P35_SHARED_ARMS = ('control',)` | 2 failed |
| M2 | make_p36_config.py | no_new rule L + 240 min | 1 failed |
| M3 | make_p36_config.py | 16:30 rule disabled | 1 failed |
| M4 | make_p36_config.py | missing gate key treated as true | 2 failed |
| M5 | make_p36_config.py | `training.epochs` 61 | 3 failed |
| M6 | make_p36_config.py | aggregate ≤ 23:30 | 1 failed |
| M7 | make_p36_config.py | `ple_registered.sha256` = DRAFT sha | 1 failed |
| M8 | make_p36_config.py | "every gate boolean true" check removed | 1 failed |
| M9 | run-p36-pipeline.TEMPLATE.py | configsha assert removed | 1 failed |
| M10 | run-p36-pipeline.TEMPLATE.py | `GATE_KEYS` without `robust_clip_arms_within` | 1 failed |
| M11 | run-p36-pipeline.TEMPLATE.py | P35 terminal count 20 | 1 failed |

## Deviations from the P35 pack (and why)

1. **11 argv instead of the 10 listed in the brief**: added `<archive commit>` (stored as `registration.source_commit`,
   template `codecommit`). P35 hard-coded the known commit `8864389`; the P36 smoke commit does not exist yet, and a
   registered config should name its code. Easy to drop if the PM prefers 10.
2. Probe summary sha not pinned in the generator (probe not run); recorded in `registration.timing_probe.summary_sha256`
   and pinned in the template `probesha`.
3. Probe gate names are guesses (`robust_clip_arms_within` replaces `ple_arms_within`); plus the stricter all-booleans
   rule, exact two exit codes, GPU list, recomputed task/family bounds.
4. Extra deadline checks (agg ≤ 23:00 always; L ≤ 16:30 on 2026-10-05) and a strict date regex; checks ordered so each
   refusal has its own message (the 16:30 refusal keeps its message after 10-05 too).
5. `decision` text no longer says "Song's explicit go … runbook v4" (the P35 LAUNCH-PLAN wording note); it points at
   `registration.go`.
6. `registration` drops `smoke`, `timing_probe.commit/result/preregistration` (unknown now), adds `preregistration`.
7. Template: five extra placeholders, `score_cuda_direct` in the `pgrep` pattern, P35 state `complete` required,
   server `EXP-P35-001.json` sha checked against `ple_registered`, smoke `rc_stats_sha256` one value / `rc_module_sha256`
   pin / both GPUs by `runtime.bound_uuid`. The process-control functions are byte-identical to P35 (tested).

## Open questions (for the reviewer / PM)

1. Gate key names and whether the P36 summary's `slowest_task_seconds_projected` includes `control` /
   `periodic_residual_counts` (prereg §4 takes them from the P35 probe, 464 / 1,046 s). The generator only requires the
   two probed arms and trusts `family_worker_seconds_projected` ≤ 36,000; should it also recompute
   family = 5 × (464 + 1,046 + both probed slowest)? Not done (would hard-code P35 numbers).
2. Smoke queue/runs dir names and probe dir are placeholders (not guessed). `runtime.bound_uuid` in smoke `fit.json` is
   assumed from `bind_gpu` (`run_cuda_direct.py:721-739`); confirm on the real smoke evidence / f988f0a checker.
3. `registration.timing_probe.summary` stores the summary path exactly as passed; pass the server-relative path
   (`runs/<probedir>/summary.json`) at registration for a stable record.
4. Requiring `score-p35-state.json` `complete` blocks P36 if the P35 scorer failed even with 25/25 fits — intended
   (prereg §7 needs the complete P35 descriptive report), but the PM should confirm.

## Found in files I may not edit (report only)

- DRAFT `decision` (configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json) still says "a CUDA smoke of all 20 tasks on GPU3/4
  passes the pre-registration section 4 timing gate"; after the M1 disposition the gate is the timing probe (smoke =
  wiring only), as `queue.stop_rule` already says. The registered config replaces `decision`, so it does not reach
  EXP-P36-001, but the DRAFT text is stale.
- `tests/test_cuda_ple_registered.py` fails (4) until `configs/EXP-P35-001.json` exists — expected, not a defect.
