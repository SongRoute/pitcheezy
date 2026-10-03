# Disposition — TabM-mini integration review (805aadc → review 01714f9)

Label: `same_model_reviewed; Codex_review_pending`. Author and reviewer are both Opus 5.5
(separate sessions). No other model was called. Written 2026-10-04 08:16 KST (`date`, cycle-008),
branch `codex/tabm-mini-1004`. No server, GPU or score was touched; no P30 score was read.

Review: `docs/reviews/OPUS-review-tabm-integration-805aadc-2026-10-04.md` (cherry-picked as
`93bf8a9` from `01714f9`, with its registry/scorer hardening and 10 gap tests).

| # | Severity | Disposition |
|---|---|---|
| 1 | Low | **Accepted as written** (cherry-pick): `direct_tabm_registry.same()` JSON-exact equality for arms, baselines, workload, priority, seeds, fits, contrasts. P30/other registries unchanged (their value-equality looseness is noted, no effect on the running P30). |
| 2 | Low | **Accepted as written** (cherry-pick): Korean table scopes the compute note to `ensemble_head` rows; seed 13–15 / repeated-DEV / post-selection line. |
| 3 | Medium (registration) | **Accepted with a correction.** The failure scenario "up to 4 processes on 2 GPUs" holds only if a TabM queue is launched WITHOUT the common `--lock-dir`: `cuda_direct_queue.GpuLock` takes an exclusive non-blocking `flock` per GPU UUID in the lock dir, and P30 uses `/home/sds733/pitcheezy-autonomy/gpu-locks` (lock files for both UUIDs updated 08:12/08:13 KST server time). With the common dir the two queues interleave fits per GPU, so fit seconds are not inflated by a second fit on the same GPU, but the 19:00 no-new-fit cutoff is still shared and the families would compete for slots. Change: config `queue.start_after` (EXP-P30-001 queue terminal, EXP-P30-002 not running, never concurrent, common lock dir named, co-scheduled seconds labelled). It is a declared registration/launch precondition; the shared `cuda_direct_queue.py` is NOT modified to enforce it (it is P24–P30 infrastructure). Test `test_queue_declares_start_after_p30_and_common_lock_dir`. |
| 4 | Low | **Accepted**: design-note status line and step-2 commit time corrected (08:01:47 KST); steps 3–4 now state smoke-after-P30 and registration preconditions. |
| 5 | Low | **Partly accepted**: explicit refusal in `score()` when a TabM config also yields factorial or regularization contrasts (previously refused only indirectly). Failing-then-fixed: 2 parametrized tests fail on 93bf8a9 and pass after. The positional `score_family` return → dict refactor is **deferred** (refactor, not a defect). |
| 6 | Estimates | **Recorded as condition**: the review's time/memory numbers are ESTIMATES. They are replaced only by measured CUDA smoke seconds/peak memory per arm (next unit). No registration before that. |
| 7 | Tests | Missing scorer refusal test added (finding 5). The end-to-end P30-shaped family scored with the 14-key `5227ee1` source map remains a gap; generic version exists. Not needed for TabM smoke; noted for registration. |

Source changes in this disposition commit:
- `configs/DRAFT-CUDA-TABM-MINI-20261004.json`: `queue.start_after` only (still DRAFT, execution disabled).
- `experiments/pitchmdp/scripts/score_cuda_direct.py`: one guard line in the TabM branch.
- `experiments/pitchmdp/docs/DRAFT-TABM-MINI-DESIGN-20261004.md`: status/steps text.
- `experiments/pitchmdp/tests/test_cuda_tabm_mini_disposition.py`: 3 tests.

Tests (Mac, threads 1):
```
cd experiments/pitchmdp && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest \
  tests/test_cuda_tabm_mini.py tests/test_cuda_workload_dropout.py tests/test_cuda_frontier_run.py tests/test_cuda_direct.py \
  tests/test_cuda_dropout.py tests/test_cuda_data_combo.py tests/test_direct_tabm_model.py tests/test_direct_tabm_model_review.py \
  tests/test_cuda_tabm_mini_review.py tests/test_cuda_tabm_mini_disposition.py -q -p no:cacheprovider
→ 258 passed in 23.71s
```
Before the scorer guard: disposition file `2 failed, 1 passed`.

Status after this disposition: CUDA smoke allowed once EXP-P30-001 is terminal and GPU3/4 are
free (UUID check). Registration remains root's, after smoke identities and measured
seconds/memory. Nothing here is a scientific result.
