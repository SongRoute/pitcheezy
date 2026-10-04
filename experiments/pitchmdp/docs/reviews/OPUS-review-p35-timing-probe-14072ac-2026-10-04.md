# Review — P35 PLE timing-only probe script, amendment and DRAFT text fixes (author 0978558..14072ac)

Label: `same_model_review; Codex_review_pending`. Opus 5.5 reviewer, separate from the author. Worktree
`codex-p35-timing-probe-review-1004`, branch `codex/p35-timing-probe-review-1004` at `14072ac`. No ssh, no GPU, no
network, no real data, no 2026 data, no fit, no probe run. No push, no merge. Frozen files not edited.

Reviewed blobs: `scripts/timing_probe_ple_residual.py` sha `0ac1a0985306f0e799b618147e0effbe900befc5fd181b1419e2a0b5171df017`;
DRAFT `9bdf9054f4e0b834a296aab7b6ea6c3cc36033940dc74d6d2e0768f728e9af8c` (was `82dd5b15…4786` at `5e8b1d0`/`0978558`);
config `f0ae4e91…b9b2` (rehashed: unchanged).

## Disposition

**Ready to run once at `14072ac`.** I found no BLOCKER and no MAJOR issue, so I added no fix commits.

- The script follows the P34 Amendment-2 method at `9de203a`. The only additions are PLE binning in prepare, a bins-hash check, stricter refusals and host records.
- It loads only TRAIN and writes no loss, logit or probability.
- Its safety code is the reviewed P34 code plus the exact two-GPU set and the 2/2/2 env check.

Recommended before the run, but text/tests only and the PM's choice: F1–F3 change the amendment text (pre-run; no method, gate or bound change), and F4 adds tests. If the PM adopts any of them, the probe runs from that later commit `<R>`. None of them changes the probe script.

## Findings (most severe first)

### F1 — MINOR — Binning is counted ≈ 31× in the PLE projection (author Q2): keep the formula, but pre-register how it is read

`scripts/timing_probe_ple_residual.py:181-209`, `:215-226`. Frozen:
- `pitchmdp/direct_ple_model.py:266-269` (single) and `:317-323` (TabM) compute the bins **before** calling the base `fit`.
- The base fit starts its timer at `direct_residual_model.py:102` / `direct_tabm_model.py:203`.

So `report['seconds']` excludes the binning and the probe's wall time includes it. `per_epoch = max(wall, report)/2` then carries `bins/2`, and the projection carries `60 × bins/2 = 30 × bins`. Prepare adds one more binning. The real task bins once. The over-count is **30 × bins seconds, on the two PLE arms only**. This is the same kind of overstatement P34 accepted for net build and TRAIN upload (×30), just larger.

Size (Mac measurement, not the server):
- `/tmp/pz-p35-review-bins-bench.py`: synthetic arrays of the real shapes (length 6, 38 token features, 56 context features, 41 continuous, 80 % valid past positions), single thread, M4.
- Measured `_no_current_physics` + `compute_bins`: 0.13 s / 0.28 s / 0.60 s at 62.5k / 125k / 250k rows. Linear in rows: ≈ 2.6–3.0 s at 1.25 M rows (P34: 1,224 updates × 1024).
- At Mac speed that is ≈ 80–90 s inside each PLE projection. On newsrv (load ≈ 28/64 during P34) it is 미측정 and could be a few times larger.

It matters only when a PLE arm's true cost is within `30 × bins` below 1,800 s. Changing the formula would change the DRAFT §6 decision ("the only projection method"), so I do **not** recommend changing it. I do recommend fixing now, before any number exists, how a false PLE gate is read. Replacement text, appended to the amendment §"Gates" after "…the gate value is not changed.":

> The per-row `ple_binning_seconds_inside_projection_not_credited` is reported next to any PLE projection but never
> subtracted and never rescues a false gate. If `ple_arms_within` is false, the record is reported as measured; a later
> redesign may move the binning cost out of the per-epoch measurement only through a new reviewed amendment, never by
> re-reading this probe.

Also NOTE (no change): `report['seconds']` is a sub-interval of the probe's `wall_seconds` (both `perf_counter`), so `max(wall, report)` is always `wall`. The N5 rule is effectively "use wall". The test with `report_seconds=1000` (`tests/test_timing_probe_ple_residual.py:228-241`) uses a fake model and only exercises the `max`.

### F2 — MINOR — The amendment undercounts binnings per worker ("2"; actually 6). The worker timeout is enough.

`docs/AMENDMENT-P35-TIMING-PROBE-20261004.md:102-103`. Each PLE arm bins once in prepare and once inside each of its two fits (warm-up and measured). That is 2 × 3 = 6 per worker. Replacement:

> - Expected cost: per worker ≈ startup (array hashing) + 5 × 3 TRAIN epochs + 6 TRAIN binnings (2 in prepare, 1 inside
>   each of the 4 PLE fits) + 5 predict passes; …

Timeout adequacy (author item 4):
- P34 probe evidence (`t50-timing-probe/evidence/summary.json`): startup 3.9/4.2 s, prepare 0.5–2.1 s, predict 0.26–0.80 s, worker totals 185.0/207.2 s.
- Every arm passes the 1,800 s gate only if per_epoch ≤ (1,800 − 30 − startup − prepare − predict)/60 < 29.5 s.
- Then a worker needs ≈ 4 + 5 × (3 × 29.5 + ≈ 10 s fit overhead/prepare/predict) + 6 binnings (≈ 3 s each at Mac speed, server 미측정) ≈ 560 s. That is about 2.7× below 1,500 s.
- So a 1,500 s timeout can only end a run whose arms would fail the gate anyway, or a hang or GPU contention. No change.
- Parent bound: preflight (≈ 4 s hashing) + `host_pre` (≤ 2 × 30 s) + 1,500 + stop grace 20 + `host_post` (≤ 60) ≈ 1,645 s < the `timeout --kill-after=30 1800` wrapper.

### F3 — MINOR — "At most 2 attempts" can be misread against DRAFT §6 "The probe runs once"

`docs/AMENDMENT-P35-TIMING-PROBE-20261004.md:127-129`; DRAFT `:354-356`. "A failed attempt" does not say whether a complete run with a false gate counts as failed. Replacement for the first sentence:

> - At most 2 attempts, and a second attempt only if attempt 1 is incomplete (`complete` false or a non-zero exit, e.g.
>   an infrastructure failure). A complete attempt is final whatever its gate values (DRAFT §6: the probe runs once).

### F4 — MINOR — Six of my eight mutants survive the author tests. A proposed test file kills all six.

The code is correct on every one of these paths; the tests just do not pin them down. The amendment's N9 gate-integrity rule depends on R1, R2 and R8.

| # | mutant (script line) | author tests (34) | proposed tests (6) |
|---|---|---|---|
| R1 | `complete = len(workers) >= 1` (`:390`), so a partial result yields gate fields | **survived** (exit 0) | killed (`test_partial_run_never_writes_gates`) |
| R2 | accept a worker JSON regardless of exit code (`:388`) | **survived** | killed (same test) |
| R3 | worker deadline reset per worker (`:370-373`), total wait 2 × 1,500 s > 1,800 s wrapper | **survived** | killed (`test_both_workers_share_one_deadline`) |
| R4 | `ple_bins` with `n_bins − 1` (`:152`) | killed (`test_probe_arm_reads_train_only…[ple_residual_counts]`) | — |
| R5 | no `sync()` after `fit` (`:187`) | killed (`…[control]`, sync count 6) | — |
| R6 | worker `signal.alarm` removed (`:423`) | **survived** | killed (`test_worker_sets_its_own_alarm`) |
| R7 | marginal band one-sided (`:246`) | **survived** | killed (`test_marginal_band_is_two_sided`) |
| R8 | `summary.json` written without `no_forbidden_keys` (`:395`) | **survived** | killed (`test_summary_is_checked_for_forbidden_keys`) |

The proposed tests are in Appendix A (verified as `/tmp/pz-p35-probe-review/test_reviewer_probe_paths.py` + `fake_worker.py`: 6 passed at `14072ac`). They also exercise the **complete** run path end to end, which no author test does: both workers exit 0 with JSON → gates written → `summary.json` with the real UUID keys passes the forbidden-key check.

I did not commit them: MINOR, PM's choice. To adopt them, add them to `tests/test_timing_probe_ple_residual.py`, move the fake worker into `tests/` and drop the absolute `TESTS` path. A new or edited test file does not change the probe's `source_sha256` (runner sources + probe + queue only).

### F5 — MINOR — Q1 re-scope `6fcb77f`: accept, but it no longer guards the working-tree DRAFT and it needs git object `5e8b1d0`

`tests/test_cuda_ple_wiring.py:120-130`.
- Correct: it checks the frozen config's `design_note` pin against the DRAFT as committed with the config.
- Side effect: further edits to the working-tree DRAFT (arms, gates, budgets) no longer turn any test red until the registered config re-pins.
- It also shells out to `git show 5e8b1d0:…`, which fails in a clone that does not have that object (squash merge, shallow clone). `5e8b1d0` is an ancestor of `14072ac` today, so it passes here.

Replacement (no git dependency, and the DRAFT stays guarded):

```python
DRAFT_AT_CONFIG = '82dd5b154997dec6c776ec8c350a2d35e494a7970efbc0b3495d4dd7a95c4786'  # DRAFT as pinned by RUN (5e8b1d0)
DRAFT_AMENDED = '9bdf9054f4e0b834a296aab7b6ea6c3cc36033940dc74d6d2e0768f728e9af8c'    # + wiring-review fixes (b1a2852)
...
    note = cfg['source_configs']['design_note']
    assert note['path'] == 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P35-001-ple-embedding-2026-10-04.md'
    assert note['sha256'] == DRAFT_AT_CONFIG
    # The DRAFT was amended before registration (§10.1, §5, §2 N-a, §10.7); any further DRAFT edit must update this
    # pin in its own reviewed commit, and the registered config must pin the amended DRAFT.
    assert rc.sha256_file(REPO / note['path']) == DRAFT_AMENDED
```

Either version is acceptable for the run (no runner, queue or scorer reads `design_note`). I recommend the replacement.

### F6 — MINOR — `preflight.json` is written without the forbidden-key check, and the output dir is created before its name is checked (both inherited from P34)

- `:350` `rc.write_json_atomic(args.output / 'preflight.json', identity)` is unchecked. The same dict is checked later only as part of `summary.json`. Replacement: `rc.write_json_atomic(args.output / 'preflight.json', no_forbidden_keys(identity))`.
- `:347` `args.output.mkdir(…, exist_ok=False)` runs before `preflight` checks the `probe`/`-timing` name (`:309`). A refused name still leaves an empty directory (e.g. a mistyped family `runs/` name).

Neither can happen with the pre-registered command. If touched, it is a code change, so it needs this amendment's script sha re-recorded and a re-review of the diff.

### F7 — NOTE — `summarize` accepts a single worker

`:238` checks `len(arms[n]) != len(workers)`, which is relative. `run` only calls it when `complete` (both GPUs), so this is defence in depth only. Possible hardening: `if len(workers) != 2 or len({w['gpu'] for w in workers}) != 2: raise ValueError(...)`.

### F8 — NOTE — An interrupted run writes no `summary.json`

`_interrupt` (`:339-342`) raises `KeyboardInterrupt`. The `finally` (`:376-384`) cleans up correctly, then the exception propagates past the summary write (`:395`). The amendment says a failed attempt keeps "worker error JSON + logs + summary"; for SIGTERM/SIGHUP/SIGINT only `preflight.json` and the logs remain. Text fix if wanted: "(… + summary when the parent was not interrupted)".

### F9 — NOTE — What the projection does not measure (all inside the 30 s allowance, as in P34)

`scripts/run_cuda_direct.py:709-792` per real task. These steps are not timed:
- load + `arm_inputs` of earlystop/temperature/blend/dev/mlb_dev (mmap; page cache warm after hashing)
- `fit_temperature`, `model.save`
- `np.savez_compressed` of two prediction archives, artifact sha256, rename
- `fit_record`/`effective_record`/`ple_bin_rows` (cheap, PLE only)

These are counted: CUDA init (`bind_gpu` → `get_device_properties`, inside `startup`) and the per-task `verify_export`.

Nothing PLE-specific is under-counted. Whether 30 s covers the rest on real data is 미측정 for P35. P34 has no actual-vs-projected comparison yet (its result doc says RUNNING). That comparison can be made once P34 is terminal, without touching this probe.

### F10 — NOTE — The forbidden-substring list (author Q6) is broad but no current key matches

Static audit:
- All 28 `source_hashes()` keys: 0 hits (one-shot, exit 0).
- `runtime_report` keys (`run_cuda_direct.py:686-700`), `host_snapshot` keys, `exit_codes` UUID keys (`GPU-ae803a24-…-3f233fcecf99` contains `cec`, not `ece`), arm names and record keys: 0 hits.

Future hazards: `ece` matches `piecewise`, `recent`, `precede`. Because the check is fail-closed at write time, such a key would refuse the **worker** JSON only after all GPU work. Keep the list; review any new output key against it.

### F11 — NOTE — The worker error JSON is not key-checked

`:426-428`. Its keys are fixed (`error`, `traceback`, `utc`). The messages that can reach it carry no label-derived value. I checked the probe's own raises and the frozen fit's `'Nonfinite early-stop loss'`, which has no number in it.

## Checks requested (1–7)

1. **Data isolation — OK.**
   - The worker loads only `load_split(…, 'train')` (`:288`). Test: `test_worker_opens_train_only…`, author M4.
   - Early-stop and predict stand-ins are TRAIN slices (`:173`, `:202`) sized from manifest `n` (`:132-139`).
   - Other splits are only byte-hashed by `verify_export` (`:286`, `:314`), as P34.
   - `bins_sha256`, ΣT_j, degenerate count and `past_valid_rows` come from TRAIN features and the valid mask; none uses a label.
   - `model.report` (history, best epoch, losses) is never copied out. Only `epochs_run`, `optimizer_updates` and `seconds` are read (`:191-198`).
   - Logits are discarded (`:205`).
   - Key check: worker JSON (`:299`) and summary (`:395`, which contains preflight and both host snapshots). Gaps: `preflight.json` on disk (F6) and the error JSON (F11). No label-derived number can leak.
2. **Method fidelity — OK.**
   - The diff against `9de203a` (`diff scripts/timing_probe_tabm_residual.py scripts/timing_probe_ple_residual.py`) shows the method unchanged: 1+2 epochs on fresh models (`:181-198`), `max(wall, report)/2` (`:209`), `startup + prepare + 60×per_epoch + predict + 30` (`:221-222`), slowest GPU per arm (`:240`), family = 5 × Σ (`:241`).
   - Frozen fit: bins are recomputed inside `fit` before the timer (F1). `model.net.config['ple_bins_sha256']` and `['ple_effective_parameter_count']` exist, set by `_install_ple` (`direct_ple_model.py:173-175`).
   - `ple_bins` (`:146-152`) is the frozen call: the same `compute_bins`, `_no_current_physics(train)`, `context_continuous` from the registry spec (which `build_ple` passes to the model, `direct_ple_factory.py:24-29`), and `n_bins` = registry `N_BINS` = `ARCHITECTURE.embedding.n_bins` = 48 (`direct_ple_registry.py:66,71`). The runtime hash equality at `:194` fails closed. Tests: `test_ple_bins_of_the_probe_are_the_bins_of_the_frozen_fit`, my R4.
   - Under-counting relative to P34: none (F9). Over-count: F1.
3. **Gates — arithmetic correct.**
   - `ple_arms_within`, `every_arm_within`, `family_within` at `:247-250` use `<=` (the 1,800 edge is tested).
   - Gate fields are added only when `complete`, i.e. both exit 0 **and** both JSONs exist (`:385-394`). `main` exits 1 when incomplete (`:418-419`). No partial path produces gates; now pinned by F4 R1/R2.
   - Q3: the naming `ple_arms_within` is fine. It is implied by `every_arm_within` and is the P35 counterpart of P34's `tabm_residual_within`. DRAFT §6 needs `every_arm_within`, `family_within`, `complete` and exit 0/0. The operator/pipeline should check all three booleans + `complete` + both exit codes == 0 (N9).
4. **Safety — OK.**
   - GPU set: exactly the two allow-listed UUIDs (`:311-313`).
   - Lock: the common `cuda_direct_queue.GpuLock` flock (`:346`, `:351-362`), non-blocking; a busy lock refuses. Lock fds are not inherited (Popen `close_fds` default).
   - Each worker runs in its own session (`:369`). Every group is stopped (SIGTERM, then SIGKILL after 20 s) in `finally` **before** any lock release (`:376-382`), on timeout, failure and SIGTERM/SIGHUP/SIGINT (`:353-357`).
   - Worker alarm 1,560 s (`:423`; now tested, F4 R6).
   - Existing output refused (`:347`), name check (`:309`, see F6).
   - Config: name prefix + constant sha pin (`:89`, `:101-106`), checked in parent (`:306`) and worker (`:283-285`).
   - Env 2/2/2 in parent (`:308`) and worker (`:276`, before `bind_gpu`).
   - Timeout 1,500 s is sufficient (F2).
5. **DRAFT edits — OK.**
   - `git diff --word-diff 0978558..14072ac -- docs/PREREG-DRAFT…` has four hunks: §2(c) N-a (i)(ii)(iii), §5 "changed (registry `CHANGED`)" column + sentence, §10.1 file name + sha pin + review/disposition commits, §10.7 new probe-script line. Each matches the wiring-review table (`OPUS-review-p35-wiring-6be28eb…:209-218`).
   - One addition beyond the four blocking items: the §5 "held (registry factors)" sentence, which the review classed as a *note*. It is the review's own proposed text and harmless.
   - The §10.7 rewrite drops "on GPU3/4 after smoke"; the amendment states GPU3/4.
   - No arm, seed, contrast, gate, budget or deadline text changed.
   - Q1: see F5. `6fcb77f` is acceptable; I recommend the git-free replacement.
6. **Tests.** 191 passed (exit 0). Mutants: see F4 (8 of mine, all different from author M1–M7). `git status --short` was empty after every restore.
7. **Remaining questions.**
   - **Q2** binning double count: F1. Keep the formula and pre-register the reading.
   - **Q4** extra strictness: agreed. Requiring exactly two GPUs matches the family's two concurrent fits. Requiring 2/2/2 makes the measurement environment the family queue's (N8). The `probe` name check protects family `runs/` names. A single-GPU retry needs a new amendment; that is intended.
   - **Q5** seed 34: acceptable. It is outside 29–33 and the family config seeds (refused at `:80-83` in both `check_config` and `probe_arm`), no artifact is kept, and timing does not depend on the seed. Switching to 1000 would only need an amendment edit; not worth it.
   - **Q6** forbidden substrings: F10. Keep.
   - **Q7** frozen-files window `0978558..working tree` (`tests/test_timing_probe_ple_residual.py:547-554`): agreed. After the run, re-scope the end to the archived probe commit `<R>` in its own commit (as T70c did for P34), before the next legitimate frozen-file change.

## Commands (cwd `experiments/pitchmdp` unless noted)

1. Reads: `git log/diff/show/merge-base --is-ancestor`, `cat -n`, `sed -n`, `grep`, `ls`, `shasum -a 256`, `/usr/bin/python3` JSON one-shots (config fields, P34 evidence `summary.json` timing fields only) → exit 0.
2. `git diff --quiet 0978558 HEAD -- pitchmdp scripts/run_cuda_direct.py scripts/cuda_direct_queue.py scripts/score_cuda_direct.py scripts/timing_probe_tabm_residual.py ../../configs` → exit 0 (frozen files unchanged). `git show 9de203a:…/timing_probe_tabm_residual.py` equals the working tree.
3. `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 perl -e 'alarm 600; exec @ARGV' /Users/song/Projects/pitcheezy/.venv/bin/python /tmp/pz-p35-review-bins-bench.py` → exit 0 (F1 numbers; synthetic, Mac).
4. Same wrapper, `python -c` forbidden-key audit of `source_hashes()` keys → 28 keys, 0 hits, exit 0.
5. `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 perl -e 'alarm 600; exec @ARGV' /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_timing_probe_ple_residual.py tests/test_cuda_ple_wiring.py tests/test_cuda_ple_wiring_review.py tests/test_timing_probe_tabm_residual.py -q -p no:cacheprovider` → **191 passed, exit 0** (13.7 s).
6. `/tmp/pz-p35-probe-review/mutate.py all` (in place, `git checkout -- scripts/timing_probe_ple_residual.py` + `git status --short` empty asserted after each) against `tests/test_timing_probe_ple_residual.py` → R4, R5 killed (exit 1); R1, R2, R3, R6, R7, R8 survived (exit 0).
7. Proposed tests: `pytest /tmp/pz-p35-probe-review/test_reviewer_probe_paths.py --rootdir=/tmp/pz-p35-probe-review` → 6 passed, exit 0. `mutate.py all` against them → R1, R2, R3, R6, R7, R8 killed (exit 1 each); R4, R5 pass (covered by the author tests).
8. The first `mutate.py` retarget via BSD `sed -i ''` did not match (exit 1, no effect; the script was outside the tree). Redone with a `/usr/bin/python3` exact replacement → exit 0.
9. `git add experiments/pitchmdp/docs/reviews/OPUS-review-p35-timing-probe-14072ac-2026-10-04.md` + `git commit` → this commit.

Not run: the full suite, the probe, anything on newsrv.

## Appendix A — proposed tests (F4; verified, not committed)

`tests/fake_probe_worker.py` (from `/tmp/pz-p35-probe-review/fake_worker.py`):

```python
"""Fake probe worker: writes worker-<uuid>.json with timing-only records; exits PZ_FAKE_EXIT_<uuid> (default 0)."""
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
out, uuid = Path(argv[argv.index('--output') + 1]), argv[-1]
arms = ['control', 'periodic_residual_counts', 'ple_residual_counts', 'tabm_residual_counts', 'ple_tabm_residual_counts']
records = []
for arm in arms:
    r = {'arm': arm, 'seed': 34, 'prepare_seconds': 2., 'predict_seconds': 3., 'measured_seconds_per_epoch': 5.}
    if arm.startswith('ple'):
        r['ple'] = {'bins_seconds': 1.}
    records.append(r)
(out / f'worker-{uuid}.json').write_text(json.dumps({'gpu': uuid, 'startup_seconds': 10., 'arms': records}))
sys.exit(int(os.environ.get('PZ_FAKE_EXIT_' + uuid.replace('-', '_'), '0')))
```

Tests (append to `tests/test_timing_probe_ple_residual.py`; `FAKE = Path(__file__).resolve().parent / 'fake_probe_worker.py'`, `import signal`):

```python
def fake_python(tmp_path):
    path = tmp_path / 'fake-python'
    path.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE} "$@"\n')
    path.chmod(0o755)
    return str(path)


def test_complete_run_writes_gates_and_passes_the_forbidden_key_check(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    args.python = fake_python(tmp_path)
    summary = tp.run(args, cfg)
    assert summary['complete'] is True and summary['exit_codes'] == {g: 0 for g in GPUS}
    assert set(summary['gate']) == {'task_seconds_max', 'family_seconds_max', 'ple_arms_within', 'every_arm_within',
                                    'family_within'}
    assert summary['slowest_task_seconds_projected']['control'] == pytest.approx(10 + 2 + 300 + 3 + 30)
    on_disk = json.loads((args.output / 'summary.json').read_text())
    assert on_disk['gate'] == summary['gate'] and tp.no_forbidden_keys(on_disk)


def test_partial_run_never_writes_gates(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    args.python = fake_python(tmp_path)
    monkeypatch.setenv('PZ_FAKE_EXIT_' + GPUS[1].replace('-', '_'), '4')  # JSON written, then a non-zero exit
    summary = tp.run(args, cfg)
    assert summary['complete'] is False and summary['exit_codes'] == {GPUS[0]: 0, GPUS[1]: 4}
    assert 'gate' not in summary and 'per_arm' not in summary
    assert 'gate' not in json.loads((args.output / 'summary.json').read_text())


def test_both_workers_share_one_deadline(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    fake = tmp_path / 'sleep-python'
    fake.write_text('#!/bin/sh\nexec sleep 60\n')
    fake.chmod(0o755)
    args.python = str(fake)
    monkeypatch.setattr(tp, 'WORKER_TIMEOUT_SECONDS', 2)
    waits, real_popen = [], tp.subprocess.Popen

    class Popen(real_popen):
        def wait(self, timeout=None):
            if timeout is not None and timeout < 10:  # the parent's worker waits (not the 20 s stop grace)
                waits.append(timeout)
            return super().wait(timeout)

    monkeypatch.setattr(tp.subprocess, 'Popen', Popen)
    summary = tp.run(args, cfg)
    assert summary['exit_codes'] == {g: 'timeout' for g in GPUS}
    assert len(waits) == 2 and sum(waits) <= 2 + 1 + .5  # one deadline: the second wait gets the 1 s floor


def test_worker_sets_its_own_alarm(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, _ = args_for(tmp_path, root, digest, monkeypatch)
    alarms = []
    monkeypatch.setattr(signal, 'alarm', lambda s: alarms.append(s))
    monkeypatch.setattr(tp, 'worker', lambda a, c: None)
    tp.main(['--config', str(args.config), '--config-sha256', args.config_sha256, '--export', str(root),
             '--export-manifest-sha256', digest, '--output', str(args.output), '--lock-dir', str(args.lock_dir),
             '--gpu', GPUS[0], '--gpu', GPUS[1], 'worker', '--gpu-uuid', GPUS[0]])
    assert alarms == [tp.WORKER_TIMEOUT_SECONDS + 60]


def test_marginal_band_is_two_sided():
    def workers(per_epoch):
        return [{'gpu': g, 'startup_seconds': 10., 'arms': [fake_record(a, per_epoch.get(a, 5.)) for a in ARMS]}
                for g in GPUS]
    # 45 s fixed + 60 x per_epoch: 1650 (150 below), 1950 (150 above), 1995 (195 above), 1575 (225 below)
    out = tp.summarize(workers({PLE[0]: 26.75, PLE[1]: 31.75, 'tabm_residual_counts': 32.5,
                                'periodic_residual_counts': 25.5}), 5)
    assert out['marginal_arms'] == [PLE[0], PLE[1]]


def test_summary_is_checked_for_forbidden_keys(export, tmp_path, monkeypatch):
    root, digest, _ = export
    args, cfg = args_for(tmp_path, root, digest, monkeypatch)
    args.python = fake_python(tmp_path)
    monkeypatch.setattr(tp, 'host_snapshot', lambda: {'loss': 1.})
    with pytest.raises(ValueError, match='loss'):
        tp.run(args, cfg)
    assert not (args.output / 'summary.json').exists()
```
