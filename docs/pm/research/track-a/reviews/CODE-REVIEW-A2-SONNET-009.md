VERDICT: PASS_WITH_NOTES

Reviewer: Claude Sonnet 5.5, a different model from the Opus 5.5 author (commit trailer). The gpt-6-astra reviewer was unavailable (quota), so this is the substitute independent read. Fixed commit 4e34a5d7, worktree /Users/song/Projects/pitcheezy-worktrees/track-a-a2-1027. Read-only. No real data was read. I ran only the synthetic tests and synthetic fault-injection subprocesses. The worktree was untouched afterwards (`git status --short --ignored` is empty and no file is newer than my start).

Summary: round-8 #1 is RESOLVED and round-8 note 3 is RESOLVED, with no regression in the earlier items. Tests: 155 passed, 0 failed, 169 subtests passed. I found 3 MINOR and 2 NOTE items; none blocks.

1. NOTE (a) Scope and protected bytes verified.
   - `git diff 7ee51b87 4e34a5d7 --stat` lists 6 files, all A2: `configs/EXP-OPE-RES-01.json`, `ope_diagnostics.py`, `ope_redesign_registry.py`, `scripts/run_ope_a2.py`, `tests/test_ope_a2_estimators.py`, `tests/test_ope_a2_runner.py`.
   - `git diff 11eeb6e2 4e34a5d7 --name-status` gives exactly 15 `A` entries and nothing else.
   - The six protected sources (`policy_estimator.py`, `policy_semisynthetic.py`, `policy_runtime.py`, `policy_requests.py`, `rollout_policy.py`, `scripts/run_policy_validation.py`) have identical blob ids at 11eeb6e2, at 4e34a5d7 and in the working tree.

2. NOTE (b) Round-8 #1 RESOLVED. Paths below are in `experiments/pitchmdp/scripts/run_ope_a2.py`.
   - **Seal bookkeeping.** `os.replace` (326) is followed at once by `self.committed.append(entry)` (330), memory only. `_seal_committed` (907-917) answers from that list (912-913) without reading the manifest. So the abort handler `stage()` 886-890 re-raises and never writes ABORTED or a `stage_failure` once the rename returned.
   - **Directory sync (334).** `bounded_io` (128-159) waits `min(10 s, io_budget())`. `io_budget` (664-667) is the time left minus `min(1.5 s, cleanup/2)`.
     - No time left: `HardDeadlineReached` is raised before any thread starts (139-140).
     - A step cut by the budget raises `HardDeadlineReached` (153-155).
     - A step cut by its own 10 s bound raises `HangGuardExceeded` (156).
     - The first post-commit step always has budget of at least cleanup/2, because the final check refuses to seal inside the last `cleanup` seconds (680).
   - **Record then warning (344-368).** Both go through `bounded_io` (357-362). `_write_postcommit_record` (182-192) picks its target inside the worker. `_emit` (195-197) writes and flushes stderr inside the worker. After `HardDeadlineReached`, `deadline` is set (365) and the loop breaks (359-360), so no later step runs. The raise (366-368) leaves `append()`. The lock is released by `_locked` / `writer()` (270-271), and deferral is restored.
   - **Exit.** `command_guard` disarms (803). `cli()` maps `HardDeadlineReached` to code 3 (2157). `finish()` (2162-2187) waits `min(1 s, remaining)`, and skips the report entirely when none is left. Then `os._exit` (2159).
   - **Real-clock probes.** Real SIGALRM, 3 s cap, synthetic seal committed at about 2.0 s, each run in a subprocess through the real `cli()`. Probe code is `scratchpad/r9/child.py` and `scratchpad/r9/probe.py` (under `/private/tmp/claude-501/-Users-song-Projects-pitcheezy/3ce806ea-24a8-40e7-a753-1cab7646c30e/`).
     - Record writer stalled: exit 3 at 2.992 s, seal committed, no ABORTED.json.
     - Stderr write stalled: record kept, exit 3 at 3.040 s.
     - Directory sync stalled: exit 3 at 3.024 s.
     - Seal staging stalled before the rename: exit 3 at 2.973 s. No seal, ABORTED.json present, no `stage_failure` receipt (no time left). The attempt reads `unfinished`, so it follows the reconcile rule. Consistent.
     - EIO only (control): warning and record written, exit 1, seal committed.
     - Final flush stalled on a full stdout pipe (30 s cap): exit 3 at 1.27 s, which is 0.21 s of work plus the 1.0 s report bound.
     - The same stderr-stall probe against 7ee51b87 had not returned after 19.5 s and was killed. That reproduces round 8 and shows the probe is sensitive.
   - **After every `os._exit`.** A fresh process took the manifest writer lock, verified the hash chain and `require_seal` accepted the committed seal (the pre-commit case was correctly unsealed).
     - No buffered manifest data is lost: all manifest I/O is fd-level `os.write` plus `fsync` (`_create_write` 165-176, `_append_synced` 200-209), and `dump()` closes its files.
     - `flock` is released by the kernel at exit. The lock file persists but carries no state.
     - Stale staging files are removed at the next `writer()` (265-266).
   - **Overshoot.** The 3.040 s exit is 0.04 s past a 3.0 s cap because the cleanup share is only 37 ms at that scale. At registered caps the share is 1.5 s, which covers the 1.0 s watchdog join plus the 0.5 s report.

3. NOTE (c) Round-8 note 3 RESOLVED.
   - `ope_diagnostics.py:107-111` `_kl` returns raw rows. The only callers are `agreement` (129: display clip of KL for `kl_candidate_reference`, plus `unit` at 131) and `tune_tau` (272: raw). `logging_mass` clips only the displayed mean and max (210), and `placebo_check` (286-342) never clips. `ope_synthetic_a2._tune_tau` (369-373) has its own raw KL.
   - `clip_display` (97-104) raises `IntegrityError` when a value is more than 1e-12 outside its bound (`ope_redesign_registry.py:103`).
   - The registry and config prose agree (`ope_redesign_registry.py:690-694`).
   - Tests `test_ope_a2_estimators.py:716-759` cover four points: tuning accepts a raw KL of about -4.5e-10; `clip_display` mocked to raise never fires inside `placebo_check`; the result is identical with display clipping disabled; the round-7 placement is rejected.

4. NOTE (d) No regression of the earlier items.
   - The final check (675-693) runs between the staging write and the rename (316-326).
   - The bounded create/write/fsync/close writer is intact (165-176, 321).
   - Staging names are still unique, with pid, counter, kind, stage and attempt (317-319).
   - `attempt_start` is still appended before the directory exists (864-870).
   - The fcntl sibling lock is unchanged (247-283).
   - `ope_variants`, `ope_bootstrap`, `ope_surrogate`, `ope_sealed`, `ope_synthetic_a2` and `ope_decision` are not in the 7ee51b87..4e34a5d7 diff, so the SN P-rule and the other statistics are byte-identical.
   - The round-7 fixes remain: diagnostic target selection inside the worker (186) and the outside-mask mass summed directly (`ope_diagnostics.py:208`).

5. NOTE (e) Tests.
   - Your command, run with `-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1` to keep the tree untouched, gave 155 passed, 0 failed, 169 subtests passed in 33.97 s.
   - Collection: registry 23, estimators 42, synthetic 20, runner 70. Round 8 had 149 tests, so there are 6 new ones.
   - Re-running the 10 new and related tests 3 times gave 10 passed each time.
   - The round-8 note 7 sandbox limit (53 temp-dir failures) did not occur here. The filesystem publication tests, including the four new runner tests (`test_ope_a2_runner.py:696-793`), executed.

6. MINOR cli() turns argparse exits into failures. `run_ope_a2.py:2155-2158` catches `SystemExit` as a failure.
   - Verified: `--help` now exits 1 and prints `run_ope_a2: SystemExit: 0`. An invalid command exits 1 with `SystemExit: 2`. At 7ee51b87 these exit 0 and 2.
   - Fix: add `except SystemExit as e` before the `BaseException` clause. Pass `e.code` through when it is an int or None (None becomes 0). Report any other value as a message. Add a test that `--help` returns 0 and a bad command returns 2.

7. MINOR cli() drops the traceback. `run_ope_a2.py:2158` reports only `Type: message`.
   - 7ee51b87 printed a full traceback. `ABORTED.json` (`stage()` 897-899) also stores only the type and message. For a run-once harness with two attempts, an unexpected exception now leaves no stack anywhere.
   - Fix: append `traceback.format_exc()[-4000:]` to `message`. The report is already bounded. Optionally add a `traceback` field to the ABORTED record.

8. MINOR Same class as round-8 #1, on the pre-commit side, pre-existing since round 4 (0a943be7). Some I/O still runs unbounded inside `deferred_interrupts()` (310).
   - `append()` 311 does `self.path.exists()` and `read_bytes()` outside any worker. The `_seal_committed` fallback `manifest.entries()` (915) reads the manifest in the abort handler. `os.replace` (326) and `_remove` (328) are also unbounded.
   - Probe: stalling `Path.read_bytes` on the manifest during the seal append, 3 s cap. The process was still blocked at 14.5 s and I killed it, because `_on_alarm` (695-700) only sets `pending`.
   - It needs a hung filesystem, and it is fail-safe (no seal can be published on that path), so it does not block.
   - Fix: `raw = bounded_io(lambda: self.path.read_bytes() if self.path.exists() else b'', 'the manifest read')`. Bound or drop the fallback read in `_seal_committed`. Leave `os.replace` as the commit point and say so in the prose.

9. NOTE The display path still narrows what the protected estimator accepts, as round 8 observed. This follows the pinned rule.
   - Verified synthetically: candidate `[0.4999999997, 0.4999999997]` against reference `[0.5, 0.5]` (sum off by 6e-10, accepted by the estimator) gives a raw KL of -6.0e-10. `agreement()` (`ope_diagnostics.py:129`) then raises `IntegrityError`.
   - In S3 that would be a deterministic-integrity failure that costs an attempt and needs a bug-fix addendum. No production occurrence is established, since ledger rows are float64-normalised.
   - Either keep it knowingly, or pin the display tolerance relative to the estimator's 1e-9.

10. NOTE Spec and test gaps, all non-blocking.
    - The cleanup share `min(WATCHDOG_JOIN_SECONDS + 0.5, cleanup/2)` (`run_ope_a2.py:664-667`) is not numerically pinned in the config prose (`ope_redesign_registry.py:764-784`). `REPORT_TIMEOUT_SECONDS` (`ope_redesign_registry.py:362`) is likewise prose-only. Add both to the `guards` text.
    - No test covers the directory-sync step reaching the hard deadline (my probe shows exit 3), the `cli()` exit mapping, or `--help`. Add them together with the fix for item 6.
    - The nine earlier open items are untouched by this commit.

Note for the PM: items 6-8 can be closed by the author without another full review. Items 6 and 7 are a few lines in `cli()`. If time is short, none of them blocks registration.
