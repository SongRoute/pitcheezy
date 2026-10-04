# Disposition — same-model review of the P35 timing probe at 14072ac (review 5347c37 → cherry-pick 7569d2f)

PM worker (Opus 5.5, continuous v4 cycle-025), 2026-10-04 ~19:55 KST. Label: `same_model_reviewed; Codex_review_pending`.
Before any probe run; no P35 timing or score exists. The probe script is unchanged:
`scripts/timing_probe_ple_residual.py` sha `0ac1a0985306f0e799b618147e0effbe900befc5fd181b1419e2a0b5171df017` (= reviewed blob).

Chosen: review option 2 (+ F5 test-only replacement and F8 text).

| finding | disposition |
|---|---|
| F1 binning ≈ 31× inside PLE projection | accepted as text: amendment §Gates sentence (reported, never subtracted, never rescues a false gate; redesign only via a new reviewed amendment). Formula unchanged (DRAFT §6). |
| F2 "2 binnings" → 6 | accepted (amendment §Bounds text). Timeout unchanged. |
| F3 attempt wording | accepted: second attempt only if attempt 1 incomplete; a complete attempt is final whatever its gates. |
| F4 six surviving mutants | accepted: Appendix A tests appended to `tests/test_timing_probe_ple_residual.py` + `tests/fake_probe_worker.py`. |
| F5 git-object dependency | accepted (test only): fixed hashes `DRAFT_AT_CONFIG` 82dd5b15… and `DRAFT_AMENDED` 9bdf9054… |
| F6 preflight key check / mkdir order | deferred: code change would need re-review; cannot happen with the pre-registered command. |
| F7 summarize single worker | deferred (defence in depth only; `run` calls it only when complete). |
| F8 interrupted run has no summary | accepted as text ("summary when the parent was not interrupted"). |
| F9–F11 notes | no change. |

Checks (Mac, threads 1, nice 10):
- `pytest tests/test_timing_probe_ple_residual.py tests/test_cuda_ple_wiring.py tests/test_cuda_ple_wiring_review.py tests/test_timing_probe_tabm_residual.py -q -p no:cacheprovider` → **197 passed**, exit 0 (191 + 6 adopted).
- Negative control: script mutant R2 (accept worker JSON regardless of exit code) → `test_partial_run_never_writes_gates` FAILED (1 failed, 39 passed, exit 1); script restored (`cmp` equal, sha 0ac1a098…).

Run commit `<R>` = the commit containing this file. The archive of `<R>` differs from 14072ac only in docs/tests.
