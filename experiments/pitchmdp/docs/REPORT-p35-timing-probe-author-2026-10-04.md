# Author report — P35 PLE timing-only probe script, amendment, DRAFT text fixes (2026-10-04, NOT run)

Label: `author_self_tested; same_model_review_pending; Codex_review_pending`. Opus 5.5 author worker. Worktree
`codex-p35-timing-probe-1004`, base `0978558`. No push, no merge, no ssh, no GPU, no ports, no network, no real data,
no 2026 data, no fit. The probe has not been run.

## Commits

1. `12925d9` feat — `scripts/timing_probe_ple_residual.py` (sha `0ac1a098…f017`) + `tests/test_timing_probe_ple_residual.py`
   (34 tests; file later unchanged, sha `34c1102c…403a`).
2. `b1a2852` docs — DRAFT §10.1, §5, §2(c) N-a, §10.7 text exactly as in the wiring review table. DRAFT sha
   `82dd5b15…` → `9bdf9054…`.
3. `6fcb77f` test — `test_cuda_ple_wiring.py` design-note pin re-scoped (open question 1).
4. `71715c8` docs — `docs/AMENDMENT-P35-TIMING-PROBE-20261004.md`.
5. this report.

`git diff --quiet 0978558 HEAD -- pitchmdp/ run_cuda_direct.py cuda_direct_queue.py score_cuda_direct.py
timing_probe_tabm_residual.py configs/` → exit 0: frozen modules, runner, queue, scorer, the P34 probe and every
config are unchanged. Config `f0ae4e91…b9b2` and module `d3575019…175e` re-hashed: unchanged.

## What the script does (differences from the P34 probe at 9de203a)

- Same method: warm-up 1 epoch, measured 2 epochs, `max(wall, report)/2`, TRAIN stand-ins for early-stop and predict
  rows from manifest counts, startup + prepare + 60 × per_epoch + predict + 30 s, slowest GPU per arm, family =
  5 × Σ, worker timeout 1500 s + own alarm 1560 s, process-group stop before lock release, signal handling, `-timing`
  output never reused.
- P35-specific: refuses any experiment_id not starting `DRAFT-CUDA-PLE-RESIDUAL-` and any config file whose sha is not
  exactly `f0ae4e91…b9b2` (the pin itself is a constant, so a self-consistent pin of an edited file is refused);
  exact 5-arm order, seeds exactly 29–33, cap 60; probe seed 34, refused inside 29–33 (`check_seed` in
  `check_config` and `probe_arm`); GPU set must equal the two allow-listed UUIDs; N8 env OMP/MKL/OPENBLAS = 2 required
  in parent and worker; `host_pre`/`host_post` snapshots (nproc, loadavg, nvidia-smi GPUs + compute apps) in the
  JSON; output name must contain `probe` and end `-timing`.
- PLE prepare: `arm_inputs` + one `compute_bins` on the TRAIN inputs (the frozen fit's exact call), timed inside
  `prepare_seconds` and alone; records bins sha, ΣT_j, degenerate count, valid rows per past position, effective
  parameter count; refuses if the fit's bins hash differs from the prepare hash.
- Gates: `ple_arms_within`, `every_arm_within`, `family_within` (+ `complete`/exit codes checked by the operator);
  `marginal_arms` (within 10 % of 1800 s) informational.
- Forbidden output keys widened (`logit`, `probabilit`, `nll`, `brier`, `ece`, `score` added; case-insensitive).
- P34 helpers copied with attribution, not imported (importing would make the P34 file part of the P35 probe's
  identity and its constants are P34-specific).

## Commands run (cwd `experiments/pitchmdp` unless noted; pytest always
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 perl -e 'alarm 600; exec @ARGV' /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest … -q -p no:cacheprovider`)

1. Reads: `git log/show/diff/status/cat-file`, `sed -n`, `cat`, `grep`, `wc`, `shasum -a 256`, `/usr/bin/python3`
   JSON one-shots on the DRAFT config → exit 0. One `git show $c:experiments/…` loop failed under zsh (`$c:e` history
   modifier, exit 128, no effect) and was rerun with `${c}` → exit 0 (DRAFT blob at `5e8b1d0` and `0978558` both
   hash to `82dd5b15…`).
2. `tests/test_timing_probe_ple_residual.py` → **34 passed, exit 0** (5.2 s).
3. New file + `tests/test_cuda_ple_wiring.py tests/test_cuda_ple_wiring_review.py tests/test_timing_probe_tabm_residual.py`
   before the DRAFT edit → **191 passed, exit 0**.
4. Mutations `/tmp/pz-p35-timing-probe/mutate.py` (in place on the committed script, `git checkout` restore after
   each, `git status --short` empty after every restore), new test file only:

   | # | mutant | result |
   |---|---|---|
   | M1 | per_epoch = wall / 2 (report ignored) | killed: 1 failed (`test_per_epoch_is_max_of_wall_and_report_over_two[1000.0-0.0]`), exit 1 |
   | M2 | seed check disabled | killed: 5 failed (`test_probe_seed_inside_the_family_block_is_refused[29..33]`), exit 1 |
   | M3 | config sha pin check disabled | killed: 2 failed (`test_config_file_pin_is_the_exact_reviewed_draft`, `test_preflight_refuses_the_real_draft_unless_its_exact_bytes`), exit 1 |
   | M4 | worker also opens `earlystop` | killed: 1 failed (`test_worker_opens_train_only_and_passes_manifest_standins`), exit 1 |
   | M5 | bins seconds subtracted from prepare | killed: 2 failed (`test_probe_arm_reads_train_only_and_records_timing_only[ple_*]`), exit 1 |
   | M6 | 30 s allowance dropped | killed: 1 failed (`test_projection_and_gate_arithmetic`), exit 1 |
   | M7 | locks released before worker groups stopped | killed: 1 failed (`test_run_interrupt_stops_worker_groups_before_releasing_locks`), exit 1 |

5. DRAFT edit by `/usr/bin/python3 /tmp/pz-p35-timing-probe/draft_edit.py` (12 exact single-occurrence replacements,
   asserted) → exit 0.
6. Old `test_cuda_ple_wiring.py` (HEAD copy, temporary file, deleted after) `-k config` against the amended DRAFT →
   **1 failed** (`test_config_is_draft_exact_5_arm_family_on_seeds_29_33_with_cap_60`, design-note pin), 41 passed →
   this is why commit 3 exists.
7. After commits 2–3: the four files → **191 passed, exit 0** (13.7 s).
8. `git add <explicit paths>` + `git commit` ×5 → exit 0.

Not run: the full suite (by instruction), the probe itself, anything on newsrv.

## Open questions for the reviewer

1. **DRAFT pin vs DRAFT edits.** The frozen DRAFT config pins the DRAFT text (`source_configs.design_note`
   `82dd5b15…`). Applying the four required DRAFT fixes breaks that pin, so the wiring config test went red. I did not
   edit the config. Instead commit `6fcb77f` checks the design-note pin against the DRAFT blob at `5e8b1d0`, the
   commit that added the config. The other 13 pins are still checked against the working tree. Alternatives: (a)
   accept; (b) drop `6fcb77f` and accept a red test until the registered config re-pins the amended DRAFT; (c) commit
   a re-pinned DRAFT config, which would change the config sha `f0ae4e91…` the probe and the smoke are pinned to. I
   recommend (a): the registered `EXP-P35-001` config must pin `9bdf9054…` or later anyway. Runtime is unaffected
   (no runner, queue or scorer reads `design_note`).
2. **Binning counted twice.** The frozen PLE `fit` recomputes bins before its timer starts. So the measured wall time
   of each PLE fit already includes one binning, and `max(wall, report)/2 × 60` carries about 30 × the bins seconds on
   top of the binning counted in prepare. I kept the reviewed formula unchanged (conservative). The extra is recorded
   per row as `ple_binning_seconds_inside_projection_not_credited` and not subtracted. On real TRAIN it is not
   measured yet; it could be about 100–250 s if a binning takes 3–8 s, an unmeasured guess. Should a PLE arm come out
   marginal, this term is the first thing to look at. Changing the formula would need a new amendment before the run.
3. **Third gate wording.** P34's third gate was `tabm_residual_within` (the family's new arm). For P35 it is
   `ple_arms_within` (both PLE arms). That is implied by `every_arm_within`, so in practice the operator checks
   complete/exit 0 + 3 booleans, as in DRAFT §6 ("every arm ≤ 1800, family ≤ 36000, probe complete with exit 0").
   Please confirm the naming.
4. **Strictness added beyond P34.** The script refuses an env other than 2/2/2. It also refuses a GPU set other than
   exactly both config UUIDs (P34 allowed a subset), and an output name without `probe`. These are deliberate. A
   single-GPU retry would therefore need a new amendment.
5. **Probe seed 34.** It is outside 29–33, as required. If a later family registers seeds 34–38, nothing collides,
   because the probe keeps no artifact. Still, the reviewer may prefer the P34 style 1000.
6. **Forbidden-key list.** The `ece` / `score` substrings are broad. I checked the current output keys and the
   runtime/host keys: none match. A future key such as `recent…` would be refused (fail-closed).
7. **Frozen-files test window** `0978558..working tree`. Once the probe has run, re-scope its end to the archived
   probe commit (as T70c did for P34), or it goes red on the next legitimate change.
