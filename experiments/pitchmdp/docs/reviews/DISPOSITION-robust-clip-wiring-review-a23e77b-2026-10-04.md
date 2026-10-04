# Disposition — same_model_review of the P36 robust-clip prereg + wiring a23e77b (T106b, v5 cycle-027, 2026-10-04 23:19–23:22 KST)

Review: `OPUS-review-robust-clip-wiring-a23e77b-2026-10-04.md` (T104b child, helper 20391, exit 0, permission_denials [],
52 turns; review commits 204929e tests + 77bd575 doc, cherry-picked here as 0fc93e6 + a1f0a64). Reviewer is another
Opus 5.5 instance: `same_model_reviewed; Codex_review_pending`. No P35 or P36 score exists; nothing here was run on a GPU.
Status after this disposition: NOT selected, NOT registered, NOT smoked.

| # | Level | Decision | Change |
|---|---|---|---|
| M1 | MAJOR | ACCEPT. Verified: smoke = 8,192 TRAIN rows × 2 epochs, so "smoke s/epoch × 60" measures warm-up only (same lesson as the P35 Amendment). | Prereg §4 gate replaced by the review's text: (i) smoke = wiring/identity only, (ii) timing probe seed 39 for `residual_counts` + robust-clip arm on each GPU, `compute_stats` and host RSS reported. DRAFT `queue.stop_rule` rewritten to match. The probe script is NEW code → own tests + same-model review before it runs (prereg §8 step 4). |
| m1 | MINOR | ACCEPT | Prereg §5 + DRAFT `date_note`: P35 pack rule fixed now (no_new = L + 5 h, kill + 1 h, aggregate + 30 min; latest L 16:30 on 10-05, else wait for a new dated go). Placeholders 22:00/22:30/23:00 → 21:30/22:30/23:00 (= the rule at L 16:30). |
| m2 | MINOR | ACCEPT | Prereg §3 + DRAFT `seed_rule`: seed 34 was the P35 timing-probe seed (TRAIN only, no artifact, no DEV row, never a score); P36 probe seed 39. |
| m3 | MINOR | ACCEPT | Prereg §2/§3 name the pinned module sha `a3d6cbce…52fa`; §8 steps 1–3 marked done. |
| m4 | MINOR | ACCEPT | "five flags" → "six flags (…)"; note on bounded proportions in the transformed set added. |
| m5 | MINOR | ACCEPT (registration step) | Prereg §8 step 5: add `source_configs.ple_registered` = `configs/EXP-P35-001.json` + test at registration (file exists only after P35 launch). |
| m6 | MINOR | ACCEPT (already fixed) | Review tests 204929e (15 tests) cherry-picked as 0fc93e6. |
| n1 | NOTE | ACCEPT | Prereg §7: the choice is recorded by the PM at registration citing the complete P35 report. |
| n2–n8 | NOTE | Noted, no change | n2 RSS only in the probe (no new fit.json field); n4 smoke cross-GPU hash read by the checker below. |

Re-pins: prereg sha256 `947e56c62444ed1cd4f8cf42ba2e6272cf31e30e9099b5ec03cd407fa52b4750` (was ac0d6cc7) in DRAFT
`source_configs.design_note`; DRAFT config sha256 `644239f3b2aaa4410391211ab1e4d972f13e1a6b025a026b058f0dff319c470a`
(was 7da55186). Module pin `a3d6cbce…52fa` unchanged.

Checks (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 .venv/bin/python -m pytest -q`, from `experiments/pitchmdp`):
8 files (test_cuda_robust_clip_wiring, test_cuda_robust_clip_wiring_review, test_direct_robust_clip_model,
test_cuda_ple_wiring, test_cuda_ple_wiring_review, test_direct_ple_model, test_cuda_tabm_residual,
test_cuda_embed_residual_registered) → 428 passed, 13.93 s. Mutants on the DRAFT (each alone, restored, sha re-checked
644239f3): old design_note sha → 1 failed; old "smoke s/epoch … within 10%" text → 1 failed.

Smoke evidence checker for gate (i): coordinator `reports/research-candidate3-smoke/check_p36_smoke.py` (f988f0a, 18
tests; P35 smoke evidence reproduces the t70d ad hoc numbers 27 / 650 / 25 / one bins hash). It checks one
`rc_stats_sha256` across seeds and both GPUs (review smoke list item 3), module sha = commit blob, sources = commit.
