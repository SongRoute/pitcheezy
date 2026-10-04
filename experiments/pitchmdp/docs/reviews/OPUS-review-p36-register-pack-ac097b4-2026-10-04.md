# Distinct Opus 5.5 review — P36 registration pack 08da77f..ac097b4 (T109, 2026-10-04)

Label: `same_model_review; Codex_review_pending`. Reviewer: Opus 5.5, distinct from the pack author (T107) and the PM
worker. Branch `codex/p36-register-pack-review-1005`, base `ac097b4`. Mac CPU only, synthetic inputs only; no ssh,
GPU, network, real data, 2026 data, score or results dir. Nothing registered or launched; no `configs/EXP-P36-001.json`
or `configs/EXP-P35-001.json` written. Frozen files (DRAFT, prereg, modules, registries, earlier configs, P35 launch
files) untouched.

Review target: `docs/launch/make_p36_config.py`, `docs/launch/run-p36-pipeline.TEMPLATE.py`,
`docs/launch/README-p36-registration-pack.md`, `tests/test_cuda_robust_clip_registered_pack.py`, author report
`docs/REPORT-p36-register-pack-author-2026-10-04.md` (not evidence). Context read: probe `3b305ad:
scripts/timing_probe_robust_clip.py`, P35 pack in this tree (`make_p35_config.py` / `run-p35-pipeline.TEMPLATE.py`
byte-identical to 387acd7: sha256 `be4485cd…0356` / `b16d253c…58f8`), `LAUNCH-PLAN-p35-20261005.md`, prereg
(sha256 `947e56c6…4750` verified) §4/§5/§7/§8, DRAFT (`644239f3…470a` verified), smoke checker
`check_p36_smoke.py` @ f988f0a (coordinator worktree, read only), `run_cuda_direct.py`, `cuda_direct_queue.py`.

## Fix commits (new pack files/tests only)

| commit | finding |
|---|---|
| `bfe48bc` | BLOCKER-1 gate keys = real probe schema |
| `0dceba7` | MAJOR-2 pin the real EXP-P35-001 registration |
| `96e1d13` | MINOR-3 bind the probe summary (protocol, module, 4 arms, P35 constants, family recompute) |
| `4467e46` | MINOR-4 template one-date guard on the actual launch time + probe identity |
| `b6fbfe0` | README: template item order, P35 pin line |

## Findings

**BLOCKER-1 — gate names did not exist in the probe (fixed `bfe48bc`).** `make_p36_config.py:36` and TEMPLATE `:20`
required `GATE_KEYS = ('robust_clip_arms_within','every_arm_within','family_within')`. The probe's `summarize()`
(3b305ad `:297-300`) writes `gate = {task_seconds_max, family_seconds_max, every_arm_within, family_within,
probe_exit_zero}`. A real passing summary → `AssertionError: probe gate key missing: robust_clip_arms_within`
(reproduced: the original generator run on a probe-schema summary makes the module fixture fail; 53 errors + 1 fail).
Fix: `GATE_KEYS = ('every_arm_within', 'family_within', 'probe_exit_zero')` in both files (generator `:43`, template
`:20`), fail-closed kept (missing → "key missing", false/1 → "gate false", any other false boolean → refused; numbers
`task_seconds_max`/`family_seconds_max` are skipped by the boolean rule and the bounds are recomputed instead). Test
`p36_summary()` now copies the probe's dict construction (`run()` :446-449, `summarize()` :289-303, `preflight()`
:366-375, gate computed with the probe's own expressions). New tests: `test_gate_keys_are_the_probe_schema_booleans`,
`test_probe_schema_failing_gate_refused` (probed arm 1,850 s → `every_arm_within` False → refused),
`test_old_guessed_gate_name_is_refused` (mutant: in-memory old `GATE_KEYS` refuses a passing real-schema summary; and
a summary carrying `robust_clip_arms_within` instead of `probe_exit_zero` is refused by the fixed generator).
```diff
-GATE_KEYS = ('robust_clip_arms_within', 'every_arm_within', 'family_within')  # P35: ple_arms_within -> P36 name
+GATE_KEYS = ('every_arm_within', 'family_within', 'probe_exit_zero')
-GATE_KEYS=('robust_clip_arms_within','every_arm_within','family_within')
+GATE_KEYS=('every_arm_within','family_within','probe_exit_zero')  # timing_probe_robust_clip.py @ 3b305ad :297-300
```

**MAJOR-2 — any "REGISTERED" file named EXP-P35-001.json was accepted (fixed `0dceba7`).** The P35 check
(`make_p36_config.py:80-88` at ac097b4) used name + `experiment_id` + `status` + `registration.draft_sha256` + the two
shared arms. The pack's own tmp fixture (a `make_p35_config.py` dry run with a synthetic summary and go "DRY RUN P35")
passes all of these, so P36 could be registered against a dry-run P35 config and record its sha as `ple_registered`
(the template would then block on the server sha mismatch, but the committed P36 config would already be wrong).
Fix: also require `registration.source_commit == '8864389'` and `registration.timing_probe.summary_sha256 ==
'311f0d4f…'` — both literals of the frozen `make_p35_config.py` (`:17`, `:66`), so the real file matches by
construction (`make_p36_config.py:22-25, 96-99`). Tests: the fixture swaps the pin in memory (multi-patch `exec`);
`test_dry_run_p35_config_refused_by_the_unpatched_generator`, `test_p35_registration_identity_mismatch_refused` (2).
```diff
+p35_reg = p35['registration']
+assert p35_reg.get('source_commit') == P35_SOURCE_COMMIT and \
+    p35_reg.get('timing_probe', {}).get('summary_sha256') == P35_PROBE_SHA, \
+    'EXP-P35-001 is not the code-8864389 / probe 311f0d4f registration (dry run?)'
```

**MINOR-3 — probe summary not bound to the P36 probe; family not recomputed; bools pass as seconds (fixed
`96e1d13`).** Without a pinnable summary sha (unlike P35), the generator did not check `protocol` or the module pin,
accepted any extra/missing non-probed arms in `slowest_task_seconds_projected`, trusted `family_worker_seconds_
projected`, and `isinstance(True, (int, float))` let `True` pass the task bound. Fix (`make_p36_config.py:30, 36,
45-47, 106, 114, 123-127`): `protocol == 'robust_clip_timing_probe_v1'` (probe :62), `preflight.module_sha256 ==
a3d6cbce…` (probe :367 / registry pin), projection arms == the 4 DRAFT arms, control/periodic == 464/1,046 (prereg §4
(ii) fixes these literally, probe :71), `type(v) in (int, float)`, `family == seeds × Σ slowest` (rel. 1e-6). 9 new
refusal cases + `test_probe_constants_match_the_probe_branch_values`.

**MINOR-4 — declared L is text only; a late or off-date launch shrank the pre-registered window (fixed `4467e46`).**
The generator checks `no_new ≥ L+5 h` on the DECLARED L, which is stored only inside `queue.date_note`. Neither the
generator nor the template checked the ACTUAL launch: run on `queue.date` after L (e.g. L 09:00, no_new 14:00, launch
11:00) the queue (`cuda_direct_queue.py:95-96`, `may_launch = now < no_new`) gets 3 h for a 5 h projection → likely
incomplete family → "full family incomplete", no score; run the evening before, the window spans two dates. Fix: two
template lines right before the queue starts (`run-p36-pipeline.TEMPLATE.py:81-82`): now (KST) must be on
`queue.date` and `now + 5 h ≤ no_new_fit_after`. Tested by executing exactly those lines with a fixed clock (5 cases:
08:55 ok, 09:00 ok, 09:01 refused, previous day 23:50 refused, next day refused). Also `:56`: probe `protocol`,
`preflight.module_sha256`, `preflight.gpus` checked. Process-control functions unchanged (test still byte-compares).

**NOTE-5 — one-date rule otherwise sound.** HH:MM regex `([01]\d|2[0-3]):[0-5]\d`, monotonic `no_new ≥ L+300`,
`kill ≥ no_new+60`, `agg ≥ kill+30`, `agg ≤ 23:00` ⇒ L ≤ 16:30 on every date and no value can cross midnight
(`24:00`, `23:60` refused by the regex). The explicit "L ≤ 16:30 on 2026-10-05" check (`:82`) is therefore implied by
`:83` (kept: own message). Past date refused; a far-future date is accepted by the generator and now refused by the
template guard unless launched on that date. Prereg §5 / DRAFT `date_note` say "no_new = L + 5 h" (equality); the
pack uses ≥ as the P35 pack and LAUNCH-PLAN ("declaring later only widens the real margin"). Not changed.

**NOTE-6 — `REGISTRATION_PATHS` only.** `test_dry_run_differs_from_the_draft_only_in_registration_paths` diffs every
top-level key and the nested `queue`/`execution`/`source_configs` keys; exact equality to the allowed set (minus
`queue.date` when unchanged). Key order preserved, `registration` appended last, `ple_registered` appended last in
`source_configs`. Existing output refused twice (`assert not exists`, `open(..., 'x')`); empty go/cycle refused.

**NOTE-7 — 11 argv (archive commit): keep.** The P36 smoke commit is unknown now; P35 could hard-code `8864389`.
The commit flows into `registration.source_commit`, the template asserts `source_commit == codecommit` (`:52`), and
the smoke identities are verified with `own_hashes()` of `code-<codecommit>` (`:69-75`, `expected_identity` includes
`source_sha256`), so a commit whose code differs from the smoke's fails closed on the server. The generator cannot
prove the commit is the smoke commit; run `check_p36_smoke.py --commit <same>` (f988f0a) before registering.

**NOTE-8 — template vs smoke checker (f988f0a).** Consistent: 20 tasks `{done: 20}`, `fit.json`
`runtime.bound_uuid` (`run_cuda_direct.py:746`, `:831`), `rc_module_sha256` (`:846`), `rc_stats_sha256` one value.
The checker additionally requires the stats hash on BOTH GPUs and one `rc_stats_rows`; the template does not — run the
checker (README precondition 2). Template pins GPU3/4 by UUID only (`:42`, = DRAFT allow-list); the queue enforces
the allow-list again.

**NOTE-9 — `pgrep -f 'cuda_direct_queue.py|timing_probe_|score_cuda_direct'` (`:65`).** ERE alternation is correct;
pgrep excludes itself; the launch command (`timeout … python run-p36-pipeline.py`) does not match. Any process of any
user whose command line contains those strings (an open `tail`, a monitoring `bash -c 'pgrep -f …'`) blocks the launch
— fail-closed, documented in README step 6.

**NOTE-10 — process control byte-identical.** `save`, `interrupted`, `execute`, `queue` sources equal P35 (AST
segment test, 4 passed; manual `diff` of the files shows only the listed P36 lines). P35 terminal requires
`queue-p35-001 {done: 25}` and `score-p35-state.json` `complete` (`:60-63`); P36 paths `queue-p36-001`,
`runs/EXP-P36-001`, `scores/EXP-P36-001`, state `score-p36-state.json` must not pre-exist (`:23`, `:64`).

**NOTE-11 — fixture design.** The in-memory `exec` swaps only named constants (each target asserted unique; now a list
of pairs), never edits a script, runs plain `python` (asserts active). It cannot mask drift in the REAL
`EXP-P35-001.json`: the fixture is produced by the frozen `make_p35_config.py` from the frozen P35 DRAFT, so it
differs from the real file only in summary-derived values, go, cycle and deadlines. Drift of the real file is caught by
(a) the generator's shared-arm + draft-sha + (new) commit/probe-sha checks, (b) the template's server-copy sha ==
`ple_registered.sha256`, and (c) `test_cuda_ple_registered.py`, which will run 5/5 on the real file once it exists —
run it at P36 registration (it fails 4/5 with FileNotFoundError until then, expected).

**NOTE-12 — `registration.timing_probe.summary` stores argv verbatim** (author Q3). Pass the server-relative
`runs/<probedir>/summary.json` (the probe requires an output name containing `probe` and ending `-timing`, :360); the
sha is the binding value. `registration` no longer records `smoke`/probe commit/result; put them in the launch record.

**NOTE-13 — stale DRAFT `decision` text** (author report): still says the smoke passes the §4 timing gate. Frozen;
the registered config replaces `decision`. Report only.

## Key-by-key schema check vs the probe (3b305ad `scripts/timing_probe_robust_clip.py`)

| pack key (reader) | probe writes | probe file:line | status |
|---|---|---|---|
| `protocol` (gen `:106`, tmpl `:56`) | `'robust_clip_timing_probe_v1'` | :62, :446 | match (added) |
| `complete` (gen, tmpl `:55`) | `len(workers) == len(args.gpu)` | :444, :446 | match |
| `exit_codes` (gen, tmpl) | `{gpu: proc code / 'timeout' / None}` | :425-429, :445-446 | match; non-int refused |
| `preflight.config_sha256` | `args.config_sha256` | :366, :447 | match |
| `preflight.export_manifest_sha256` | `args.export_manifest_sha256` | :366 | match |
| `preflight.module_sha256` (gen `:114`, tmpl `:56`) | `check_module_pin()` | :357, :367 | match (added) |
| `preflight.gpus` (gen, tmpl `:56`) | `list(args.gpu)` | :375 | match |
| `slowest_task_seconds_projected` | 4 arms, control/periodic fixed | :290, :293, :71 | match; now exactly 4 arms |
| `family_worker_seconds_projected` | `Σ seeds × slowest` | :291, :293 | match; recomputed (added) |
| `gate.every_arm_within` | `all(s <= 1800)` | :298 | match |
| `gate.family_within` | `family <= 36000` | :299 | match |
| `gate.probe_exit_zero` | `all(c == 0)` | :300 | match (was missing) |
| `gate.task_seconds_max` / `family_seconds_max` | 1800 / 36000 (numbers) | :297 | not bool, skipped; bounds recomputed |
| `gate.robust_clip_arms_within` | — not written | — | **was required → BLOCKER-1** |
| `per_arm` | per-GPU projections | :284-285, :293 | not read by the pack (OK) |
| summary path | `args.output/'summary.json'` | :450, :360 | tmpl `runs/<probedir>/summary.json` |

Summarize keys exist only when `complete` (:448-449); the pack checks `complete` first, so an incomplete summary is
refused with "timing probe incomplete" (generator) / assertion (template).

## Commands (from `experiments/pitchmdp`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 perl -e 'alarm 600; exec
@ARGV' /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest <files> -q -p no:cacheprovider`)

- `tests/test_cuda_robust_clip_registered_pack.py` at ac097b4 → 57 passed, exit 0.
- same file, new tests against the ORIGINAL generator/template (restored after) → 1 failed, 9 passed, 53 errors
  (fixture: `AssertionError: probe gate key missing: robust_clip_arms_within`).
- same file after `bfe48bc` → 63 passed; `0dceba7` → 66; `96e1d13` → 75; `4467e46` → **81 passed**, exit 0.
- `+ tests/test_cuda_robust_clip_wiring.py tests/test_cuda_robust_clip_wiring_review.py` → **238 passed**, exit 0.
- `tests/test_cuda_ple_registered.py` → 4 failed, 1 passed (FileNotFoundError `configs/EXP-P35-001.json`; expected).
- sha256 checks: prereg `947e56c6…4750`, DRAFT `644239f3…470a`, P35 make/template equal 387acd7.

## Reviewer mutations (each alone on the fixed tree, restored with `git checkout --`; `git status` clean after)

| file | mutation | result |
|---|---|---|
| template | guard `hours=5` → `hours=4` | 1 failed (killed) |
| template | guard date-equality removed | 1 failed (killed) |
| generator | family recompute removed | 1 failed (killed) |
| generator | P35 commit/probe pin bypassed | 3 failed (killed) |
| template | `GATE_KEYS` back to the guessed names | 2 failed (killed) |
| generator | "every gate boolean true" removed | 1 failed (killed) |
| generator | module-sha check removed | 1 failed (killed) |
| template | `score_cuda_direct` dropped from `pgrep` | 1 failed (killed) |
| generator | `agg ≤ 23:59` | 1 failed (killed) |

9/9 killed.

## Answers to the author's open questions

1. Gate names: the real ones are `every_arm_within`, `family_within`, `probe_exit_zero` (BLOCKER-1). The summary's
   `slowest_task_seconds_projected` DOES include `control`/`periodic_residual_counts` (P35 constants, probe :290). Yes,
   recompute: the generator now requires the 4 DRAFT arms, the constants 464/1,046 (fixed by prereg §4 (ii), not a new
   hard-coding choice) and family = 5 × Σ (MINOR-3).
2. `runtime.bound_uuid`, `rc_module_sha256`, `rc_stats_sha256` are the real `fit.json` names
   (`run_cuda_direct.py:746/831/846`) and the f988f0a checker reads the same ones. Smoke queue/runs/probe dirs stay
   placeholders (fail closed).
3. Pass the server-relative `runs/<probedir>/summary.json`; the sha is binding (NOTE-12).
4. Yes, intended: prereg §7 (a) needs the complete P35 descriptive report, which needs the scorer. A P35 scorer
   failure with 25/25 fits blocks P36 until a scorer rerun completes; record that in the PM ledger, do not relax.

## Disposition

Usable for registration **after** (a) P35 terminal (25/25 + `score-p35-state.json` complete + descriptive report),
(b) the real `configs/EXP-P35-001.json` exists and `test_cuda_ple_registered.py` passes 5/5 on it, (c) P36 smoke 20/20
passing `check_p36_smoke.py` (f988f0a) with `--commit` = the archive commit given to the generator, (d) the reviewed
P36 probe run complete with every gate true, and (e) PM disposition of this review (fix commits `bfe48bc`, `0dceba7`,
`96e1d13`, `4467e46`, `b6fbfe0` accepted or reverted). Open items: none blocking. Same-model only; Codex review pending.
