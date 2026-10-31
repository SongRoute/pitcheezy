# A2b TAB-R3 stand-in sizing (EXP-OPE-RES-02 design evidence)

**What this is:** data-free, design-time computations for the EXP-OPE-RES-02 design
(`../A2b-ope-followup-design.md` §5.3 and appendix A). It is a **sizing, not a result**: it does not
measure anything about real rehearsal data, B2 or the policy, and it needs no registration.

- **Part 1 — registered world (primary, design v2.2):** the registered EXP-OPE-RES-01 TAB world rebuilt
  from the public sealed A2 stage outputs, requested by the PM after independent review round 2
  (`../reviews/REVIEW-A2B-SONNET-002.md`, NOTE R2-NOTE-6).
- **Part 2 — synthetic fixture worlds (robustness, design v2):** five toy TAB worlds from the A2 test
  fixture, requested after review round 1 (`../reviews/REVIEW-A2B-SONNET-001.md`, MAJOR-1). Files unchanged
  since v2; part 1 reproduces all 90 of its rows exactly.

**Data statement:** no rehearsal row, ledger, reward, model artifact, T7 path or 2026 file is read, and
nothing is fitted. Part 1 reads only `registered-world-inputs.json` (aggregate TRAIN marginals and frozen C0
situations already sealed in A2-S0, TRAIN 2023-05-15..2025-04-30, and the A2-S1 η_p and baseline summary).
Part 2 reads only the test fixture (toy WE, toy advancement, Dirichlet frequencies; "No real data, model or
file is read").

## Files

| file | sha256 | bytes |
|---|---|---|
| `registered-world-inputs.json` (PM extraction, not written by this worker) | `cdece800ffd98158a2d600fbdf062bc82427c7169b2556c136486dd4bc9f5d79` | 776,344 |
| `a2b_registered_world_sizing.py` (part 1) | `68fdbeae50d70fa056435ba603632764f4282b55073b56be99242ae2971d29a5` | 23,428 |
| `a2b_registered_world_sizing_output.json` (part 1) | `1e3b4b4a71df1a73fb06dcfadd591354fb549c7c274015a1d8fea4a9cbf30c96` | 34,224 |
| `a2b_tab_r3_sizing.py` (part 2) | `f1cf66c0b4f3d224cd0a821cea25cdac3fac9327219e123c8cb92521bacc938a` | 17,331 |
| `a2b_tab_r3_sizing_output.json` (part 2) | `924be5a622dff2f853d70344ba26fef6286b120f14875c967238b3b45bcdbe13` | 105,098 |

Code used read-only from the A2 worktree `/Users/song/Projects/pitcheezy-worktrees/track-a-a2-1027/experiments/pitchmdp`
(nothing written there; run with `python -B`; `git status` of the worktree stays empty):

| module | sha256 |
|---|---|
| `pitchmdp/ope_synthetic_a2.py` (TabWorld, `_expected_dr`, `_solve`, `_visitation`, `tab_integrity`) | `63836da8a7f3b0d9901dc5cc6bee8e5d5085f4daac0d7e52bd561748e5037413` |
| `pitchmdp/ope_redesign_registry.py` (imported by the module) | `166edab99ef113a8f41feb63d9e25e8cfb9162ba79aa6f077f4084dbbf5b1a34` |
| `scripts/run_ope_a2.py` (`situation_from_doc`/`marginals_from_doc` :1249–1262 mirrored, not imported) | `8205d27c5ad2444669a8a271035add9380bc4f038bd49a3dbbde9f8918ceea03` |
| `pitchmdp/game.py` (`GameState`) | `0e365c03c821e249f019b241dced8ac8e004c6ea9f95227c408cb2387389efc6` |
| `tests/ope_a2_fixtures.py` (synthetic marginals) | `d2ddb20f740d7511315a63d125367063739f20ce950a5cdca08127ad599c59a2` |

Environment: Python 3.12.14, numpy 2.5.3 (`/Users/song/Projects/pitcheezy/.venv`). Part 1 ran 2026-10-28 ~02:39 KST
(5.9 s); part 2 ran ~01:31 KST (2.9 s).

```
cd <scratchpad> && env PYTHONDONTWRITEBYTECODE=1 \
  A2_PITCHMDP=/Users/song/Projects/pitcheezy-worktrees/track-a-a2-1027/experiments/pitchmdp \
  PYTHONPATH=/Users/song/Projects/pitcheezy-worktrees/track-a-a2-1027/experiments/pitchmdp \
  DYLD_LIBRARY_PATH=/Users/song/Projects/pitcheezy/.venv/lib/python3.12/site-packages/torch/lib \
  /Users/song/Projects/pitcheezy/.venv/bin/python -B <this folder>/a2b_registered_world_sizing.py \
  --inputs <this folder>/registered-world-inputs.json \
  --toy-output <this folder>/a2b_tab_r3_sizing_output.json \
  --out <this folder>/a2b_registered_world_sizing_output.json
# part 2: the same environment, a2b_tab_r3_sizing.py --out <this folder>/a2b_tab_r3_sizing_output.json
```

## Definitions (shared by both parts)

- **D0** = TAB logging law `pi_b`; reference policy = D0; candidate = KL tilt of D0 (TabWorld, mean KL 0.0074).
- **xi** = sqrt(1 − ω²)·ε + ω·z̄; ε ~ N(0,1) per (count, archetype, group); z̄ = situation average of
  (Q_world − q̂) standardized over the support (q̂ = the DR control variate of TabWorld).
  ω sets how far the error field aligns with z̄; **ω = 1 is not the worst case** (see the worst-direction row).
- **D2** = D0·exp(θ·ξ) renormalized on the D0 support; θ by bisection so the D0-visitation-weighted mean
  KL(D2‖D0) = multiplier × 0.048863351237125756 (recorded A3 KL(B2‖B0) on S6).
- **Laws:** b_t ∝ D0^(1−t)·D2^t, t ∈ {0, 0.5, 1, 2}; V3 = D0^0.5 (the A2 T-A2 law).
- **β** = exact E[Δ̂_DR] − Δ_true by `_expected_dr` (κ = ζ = 0 unless stated, uniform starts), with the D0
  denominator (β₀, current baseline) and the D2 denominator (β₂, pure swap R1-0×R2-0×B2); `beta_lambda` = the same
  at a fixed shrinkage λ (descriptive).
- **ESS ratio** = (E[W]²/E[W²]) with the D2 denominator over the same with D0, W = PA-end product of ρ, exact
  recursion; null = infinite second moment in the two-strike foul loop.
- **Tails** = decision-level ρ quantile 0.99 and max, weighted by visits under the law × the law.
- **Pure-swap exact cells** (B2-tuple rule): absolute |β₂| ≤ B in C1 (t0), C3 (t0, 2κ_p), C4 (t0, 2ζ_p), C5 (t1),
  C7 (t1, 2κ_p), C8 (t1, 2ζ_p); relative |β₂| ≤ |β₀| + B in C2 (V3) and C6 (t2); B = 1.26e-5.
- **Adequacy band** (numbers fixed before the first run): reference PA-ESS ratio ∈ [0.5167, 0.8389] and
  candidate decision-ρ q99 ratio ∈ [1.1960, 1.5880], i.e. the recorded ESS loss 0.322 and q99 excess 0.392 each
  scaled by 0.5 to 1.5 (the v2 phrase "within a factor 1.5 either way" was imprecise). Registered evaluation point
  t = 1; t = 0 and t = 2 are reported too.

## Part 1 — registered world (primary)

**Inputs** (`registered-world-inputs.json`, extracted by the PM, 2026-10-28 02:26 KST): A2-S0-a1 `s0.json`
(sha `ae1edc95…`, digest manifest seq 7: `tab_marginals`, `s5x_layers`, `kappa_p`, `zeta_p`, `signs`,
`signs_prime`, `m`, `r1_table` + sha, `census`) and A2-S1-a1 `tab.json` (sha `53a7d19b…`, seq 66: `eta_p`, the
baseline acceptance summary). Only `tab_marginals`, `eta_p` and the baseline summary are used.

**Construction:** `TabMarginals` is rebuilt with a copy of the A2 runner's `situation_from_doc` /
`marginals_from_doc` (`run_ope_a2.py:1249–1262`; the runner itself is not imported), and every world is
`TabWorld(marginals, law, eta, kappa, zeta)` as in `stage_s1` (:1428–1433). The fixed draws are the registered
A2 draws `role_sequence('tab_world', 0, component)`. The registered ε field is `SeedSequence([20261028, 7])`
(design §9.4). D2 is built once in the C1 world (η_p, κ = ζ = 0) and kept fixed across the eight cells.

**Fidelity (before anything else):** η_p recomputed by the A2 rule = 0.16405664825560962 and the exact baseline
β₀ of T-A1..T-A4 are **bit-identical** to the sealed record (T-A2: 8.92541220275217e-06 = 0.708 B; T-A1/T-A3/T-A4
at 1e-18 noise level, also identical). The row function reproduces all 90 part-2 rows exactly.

**Grid:** η ∈ {0, η_p}; KL multiplier ∈ {0.5, 1, 2}; ω ∈ {0, 0.5, 1}; laws t0, t0.5, t1, t2, V3; λ ∈ {100, 10, 3}
at KL × 1 (18 rows, same fields as part 2). Additions: exact pure-swap cells C1–C8 for ω ∈ {0, 0.5, 1} at KL × 1,
η_p; 100 sizing-only alternative ε draws `SeedSequence([20261028, 7, 1000, j])`, j = 1..100 (not registered
streams) for ω ∈ {0, 0.5}; a first-order worst direction at the same KL (G = ∂β₂(C1)/∂ log D2 at D2 = D0 by
central differences, step 1e-4; ξ* = (G/D0 − Σ_g G)/v(c, a) on the support; θ calibrated; the sign maximizing
|β₂|) — a feasible D2, so its |β₂| is a lower bound on the worst case at that distance.

**Headline (registered stand-in: KL × 1, ω 0.5, η_p, registered ε; θ = 0.4908):**

- Pure swap, exact: **C1 1.278 B → EXACT_FAIL**; C2 2.467 B > 1.708 B FAIL; C3 1.111 B FAIL; C4 1.310 B FAIL;
  C5, C7, C8 0 (identities) PASS; C6 0.513 B ≤ 3.958 B PASS.
- Adequacy: t0 out (reference W second moment infinite, q99 ratio 1.980); **t1 out** (ESS ratio 0.637 inside,
  q99 ratio 1.762 > 1.588); t2 inside (0.727, 1.432). The record (0.678, 1.392) is matched between t1 and t2 (ESS)
  and near t2 (q99).
- Under decision 2 options 1/2, A2b-S0 would therefore set `DIAGNOSTIC_ONLY:STAND_IN_OUTSIDE_BAND` (and the pure
  swap is `SWAP_FAILS_C1`); everything it computes is fixed before registration, so it would reproduce these values.
- ω sensitivity (C1): 0.296 / 1.278 / 1.152 B at ω 0 / 0.5 / 1 (not monotone). Failing cells: ω 0 → C2; ω 0.5 →
  C1–C4; ω 1 → C1, C3, C4. None of ω 0, 0.5, 1 lets the pure swap pass all eight cells.
- KL sensitivity (ω 0.5, C1): 0.292 / 1.278 / 52.46 B at × 0.5 / 1 / 2. η = 0: 2.396 B.
- ε draws (100, ω 0.5): |β₂(C1)|/B median 1.305 (q10 0.254, q90 3.567, max 8.79); C1 pass 40/100; C1, C2 and C6
  together 32/100; t1 band 2/100 (ESS ratio alone 100/100). ω 0: 38/100, 29/100, 1/100. (ω 1 has no ε part.)
- Worst direction: cosine with z̄ −0.048; C1 |β₂| 12.5 B at KL × 0.5 (that D2 is inside the t1 band), 28.1 B at
  × 1, 1,823 B at × 2.
- Current baseline (B0) bias when B0 is wrong: t1 1.181 B, t2 2.958 B, V3 0.708 B (= the A2 T-A2 record).
- Fixed λ at t0: |β₂|/B 1.432 / 0.519 / 1.051 at λ 100 / 10 / 3 — no systematic reduction.

## Part 2 — synthetic fixture worlds (robustness; v2 evidence, unchanged)

World seeds 0–4 (`synthetic_tab_marginals(seed)`, 64 situations, 8 archetypes, 4 groups); η ∈ {0, η_p} with η_p
recalibrated per synthetic world by the A2 rule (ECE10 = 2 × 0.0087; 0.140–0.166, real TAB 0.164); the same grid;
ε from `SeedSequence([20261028, 7, world_seed])`.

**Shared noise (review R2-NOTE-1):** TabWorld draws the archetype perturbation, the η error field, the planning
noise and the q̂ noise from the A2-base `role_sequence('tab_world', 0, component)` whatever the marginals
(`ope_synthetic_a2.py:273–274`), so the five toy worlds share those arrays (and so does the registered world);
they differ in marginals, situations and the D2 ε field. The cross-world spread may therefore understate the
spread over noise draws; part 1's alternative ε draws add part of it.

**Headline (registered stand-in: KL × 1, ω = 0.5, η_p; ranges over 5 synthetic worlds):**

- Pure swap, B0 true (t = 0, C1): |β₂|/B **0.20–2.51**, within B in **3/5** (ω = 0: 0.20–2.44, 2/5; ω = 1: 0.63–2.48, 1/5).
  KL × 0.5: 0.13–1.54 (4/5); KL × 2: 0.65–9.31 (1/5).
- B2 true (t = 1): current-baseline |β₀|/B 0.32–2.49. Overshoot (t = 2): |β₀|/B 0.39–4.18 vs |β₂|/B 0.04–0.89 (C6 5/5).
- V3 baseline |β₀|/B 0.03–2.40 (the real TAB record 0.71 B, reproduced exactly in part 1, lies inside).
- Induced reference PA-ESS ratio (D2/D0): t0 0.26–0.54, **t1 0.68–0.70** (record 0.678), t2 0.75–0.78.
- Induced candidate ρ q99 ratio: t0 1.75–2.24, t1 1.48–1.92, t2 1.24–1.50 (record 1.392). Candidate ρ max under D2 5.1–9.5 (record B2 18.1).
- Fixed-λ shrinkage at t0: |β₂|/B 0.33–2.86 (λ 100), 0.24–2.00 (λ 10), 0.12–2.07 (λ 3) — no systematic reduction.
- Adequacy at t1: **3/5** inside (ESS ratio inside in 5/5; q99 ratio 1.7666 and 1.9175 above 1.588; two inside worlds
  within 0.3 % of the q99 edge, 1.5847 and 1.5833); inside at t0/t1/t2: 0/5, 3/5, 5/5; jointly with C1: 0/5, 1/5 (seed 0), 3/5.
- η = 0 (planning/q̂ noise only): |β₂(t0)|/B 0.08–2.00, C1 3/5.

## Revision notes (disclosed)

- **What v1 fixed and what changed after it (review R2-MINOR-3):** ω = 0.5, the KL target and the t family {0, 1, 2}
  were in design v1, before any sizing run. **z̄ was not:** v1's text defined it from "world Q − model Q"; it was changed
  to "world Q − q̂" (the quantity that multiplies the denominator error in the DR bias) when the script was first
  written (2026-10-28 01:25 KST), before the first trial run (01:26). Both the trial run and the final part-2 run (01:31)
  used q − q̂. The earlier heading "Definitions (fixed in design v1 before any sizing run)" overstated this.
- **Adequacy evaluation point:** the first trial run evaluated the band at t = 0 and found 0/5 worlds inside (the
  stand-in harsher than the record when B0 is the true law). Because the record's reference-policy mean ρ under B2
  (ratio 0.968 < 1) points to t > 1 in this family (E_{b_t}[B0/B2] ≥ 1 for t ≤ 1; design §2.5), the comparison point
  was moved to t = 1, the boundary of that region, before the final run. θ, ω, the KL target and the band numbers
  were not changed. The trial output was discarded (scratch, above the 100 KB scratch limit). The verdict depends on
  the evaluation point (toy 0/5, 3/5, 5/5; registered out, out, in at t = 0, 1, 2), so the design's push to decision 2
  option 3 rests on the C1 clause alone (review R2-MINOR-1).
- **Part 1 run:** a first run (02:36 KST) had the same numbers; the second run (02:39) only added the joint C1·C2·C6
  draw count and dropped the degenerate ω = 1 draw block. No real data was involved in any run.

## Limitations

- Four pitch groups and a Gaussian ξ field: this sizes a denominator family misspecified by the recorded distance, not
  B2's actual 18-code error structure; the recorded extreme weight (B2 max 18.1) is not reproduced (registered stand-in
  5.9 at t = 1).
- The registered ε field is one draw; the alternative draws show the spread but are sizing-only.
- Part 2 uses toy WE and toy advancement; its scale may differ from the registered world.
