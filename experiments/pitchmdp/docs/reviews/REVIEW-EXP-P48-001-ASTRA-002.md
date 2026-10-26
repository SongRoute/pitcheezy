Reviewed `64d57aad..21a9537256ba5c125de4b44b1cec752ad4265814` on `research/direct-head-confirm-1026`. No files modified; worktree remains clean.

**A — PASS: frozen-member anchoring**

For every P11 seed **0–4**, the scorer now verifies:

- G15/G0 configuration hashes, then the P11 analysis-manifest hash against G0’s pinned `files.p11_manifest` record.
- `prediction_state.json` against that manifest’s `inputs`.
- Every state-listed artifact’s bytes against `artifact_hashes`. The actual P11 states list `predictions.npz`, `prediction_runtime.json`, `source_prediction_state.json`, and `source_calibration.json`.
- `predictions.npz` additionally requires its own manifest entry; every overlapping pinned listing must agree.
- Seed/cell identity and the recorded model hash against the frozen member record.

Missing entries or mismatches abort. See [scorer:199–279](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py:199) and [G0 manifest pin:320–322](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/configs/G0-RESEARCH-FROZEN-v1.json:320).

Anchoring is uniform: baselines and Cpanel states use G0; flatten states use G15; P11 states and archives use the pinned manifest. No `require_listed=False` bypass remains. **All frozen checks finish before frozen arrays load**, followed by metadata alignment and reproduction checks; candidate probabilities are requested afterward. See [loader:282–318](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py:282), [entry-point ordering:715–734](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py:715), and [gate-before-candidate:560–584](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py:560).

My independent metadata check passed for the pinned P11 manifest and **20 frozen member-state files**, including artifact-record agreement. I did not open or rehash prediction archives.

**B — PASS by inspection; execution NOTE**

Both required regressions exercise `score()` and expect the specific pinned-state mismatch:

- Replaced P11 archive with its adjacent state consistently resealed.
- Seed-1/seed-3 archive swaps with resealed identities, plus whole-member swaps.

Sentinels replace candidate loading, scoring, and alignment with assertion failures; tests also require no output directory. Thus an unrelated failure or premature candidate access cannot satisfy them. See [tests:565–607](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/tests/test_score_cuda_mix_vs_g15.py:565). Additional tests cover broken chain links and an in-memory seed swap caught by the new per-seed reproduction gate ([tests:610–680](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/tests/test_score_cuda_mix_vs_g15.py:610)).

The requested pytest command stopped before collection: **`No usable temporary directory found`**. With capture/cache writes disabled and temporary-file-dependent tests excluded, **77 passed, 22 deselected**. The filesystem regressions were inspected, not independently executed; no full-suite pass is claimed.

**C — PASS: single confirmatory claim**

Registry and config fix `tabm_residual_counts/MIX_FIT` as the sole primary, with exact equality enforced. Its own unadjusted one-sided p enters each split’s N decision; confirmation requires both splits to pass. Holm applies only across the three secondary rows per split.

Secondary combined status is unconditionally `exploratory`, including when both splits pass. Controls and unblended rows receive no N decision. Registry, config, scorer, tests and preregistration §6 agree:

- [Registry:135–146, equality check:208](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/pitchmdp/direct_adoption_confirm_registry.py:135)
- [Config:495–522](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/configs/EXP-P48-001.json:495)
- [Scorer decision:536–552](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py:536), [combined labels:613–646](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py:613)
- [Tests:428–435](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/tests/test_score_cuda_mix_vs_g15.py:428), [preregistration:157–166](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/docs/PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md:157)

**D — PASS: requested wording**

Both preregistration and scorer limits contain **“ceiling of this direct-head pipeline on the exposed 2025 DEV”**, explicitly disclose **P48 CUDA/cap 60 versus G15 MPS/cap 30**, and identify control `MIX_FIXED` as measuring that pipeline difference. See [preregistration:214–220](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/docs/PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md:214) and [scorer limits:763–768](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py:763).

**E — PASS: round-1 protections preserved**

- Dispatch remains P48-first; runner and Stage-A scorer are unchanged in this revision ([runner:318–369](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/run_cuda_direct.py:318), [Stage A:703](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/score_cuda_direct.py:703)).
- Export pin `49097109…1f578` and `last_year: 2025` remain unchanged ([config:61–63](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/configs/EXP-P48-001.json:61)).
- Training/early-stop/temperature boundaries remain intact; MIX_FIT still selects only on `blend` ([runner:942–979](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/run_cuda_direct.py:942), [scorer:510–520](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py:510)).
- Existing destinations and repository `results/` destinations remain refused ([scorer:146–166](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py:146)).
- Status remains `DRAFT`, execution disabled, and Stage B rejects that configuration ([config:4](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/configs/EXP-P48-001.json:4), [config:634–649](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/configs/EXP-P48-001.json:634), [scorer:709–710](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py:709)).

**F — NOTE: execution readiness**

No new execution-blocking implementation defect found. Real-data execution still requires the existing CUDA smoke, timing, source/config/preregistration hashes, queue-clock registration and explicit go. These remain open by design ([preregistration §10:225–233](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/docs/PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md:225)). Rerun the focused and required full suites in a writable test environment before execution.

Minor nonblocking correction: the test comment at [test:339](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/tests/test_score_cuda_mix_vs_g15.py:339) says four-test Holm would produce `.16`; this fixture produces `.06`. Its behavioral assertion remains valid.

**Required code fixes: none.**

VERDICT: PASS_WITH_NOTES