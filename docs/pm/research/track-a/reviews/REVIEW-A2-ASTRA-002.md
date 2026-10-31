VERDICT: FAIL

## Findings

Findings 1–13 correspond to the numbered round-1 findings. Findings 14–18 cover remaining inconsistencies and regressions.

1. **note — Round 1 #1: RESOLVED. Signed-shift disclosure is removed.**  
   Signed shifts, their intervals, λ/M objectives and selected values now remain inside the sealed evaluator; public S-5 output is pass/fail. Selection is correctly described as outcome-dependent but operator-blinded. [Design:185–191](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:185), [JSON:491–507](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:491).  
   **Required fix:** None for the original leak. Resolve the output-schema ambiguity in finding 15.

2. **note — Round 1 #2: RESOLVED. Registration inputs and generated outputs are separated.**  
   R1 tables are sealed after registered S0. R3 is fixed at registration as diagnostic-only or `NOT_RUN:R3`, with `NOT_APPLICABLE` pins when absent; later addition requires another registration. [Design:163](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:163), [design:394–395](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:394), [JSON:629–657](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:629).  
   **Required fix:** None for the original circularity or fallback. Clarify the config digest separately—finding 16.

3. **note — Round 1 #3: RESOLVED. B matches D186 exactly.**  
   Both files use **`1.26e-5`** for acceptance and distinguish the explanatory arithmetic from the executable threshold. [Design:252–254](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:252), [JSON:411–418](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:411), [D186:325](/Users/song/Projects/pitcheezy/docs/decisions.md:325).  
   **Required fix:** None.

4. **minor — Round 1 #4: PARTIALLY RESOLVED. The matrix is complete, but one truth statement is wrong.**  
   The revision specifies 15 S5-X runs, 56 TAB combinations plus the drift cell, four acceptance cells, 23,900 repetitions, all 12 original-denominator pairs, explicit logging laws, and realized-WE total bias. R3 is explicitly outside certification. [Design:272–285](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:272), [JSON:386–435](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:386).  
   However, both files say the baseline expectation `E[Δ̂₀]` is zero under V2. Its **bias** is zero; its expectation equals `Δ_true(r)`. The deterministic checks elsewhere correctly use that equality. [Design:238](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:238), [JSON:367](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:367), [JSON:575](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:575).  
   **Required fix:** Replace the parenthetical with “`β₀=0` under V2, equivalently `E[Δ̂₀]=Δ_true(r)`.”

5. **major — Round 1 #5: PARTIALLY RESOLVED. S-2’s purported exact formula is incomplete for R1-a.**  
   The first-order approximation is now descriptive, and certification uses CRN estimates with uncertainty bounds and `INCONCLUSIVE`. But R1-a depends on the realized event **e**, whereas the new γ formula uses `r̃(x,s)` and averages only over **x**—the R1-b form. S-2 applies this formula to both rewards. [Design:117–118](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:117), [design:228](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:228), [design:264](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:264), [JSON:351](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:351).  
   **Required fix:** Define separate `γ_a` and `γ_b`. For R1-a, the surrogate term inside the x-expectation must be  
   `Σ_e Pκ(e | x,dp,a) × terminal_values(s)[e]`.  
   Include the deterministic invariant `γ_a=0` when `ζ=0`, including nonzero κ, under the declared matching advancement model. Otherwise S-2 can evaluate the wrong surrogate.

6. **note — Round 1 #6: RESOLVED. Statistical rejection no longer authorizes a retry.**  
   Statistical failures—including baseline shared-net failures—are preserved outcomes. Repairs require an independently reproducible deterministic defect, recorded fix, different-model review, unchanged settings/seeds and preserved attempts. [Design:381–385](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:381), [JSON:572–584](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:572).  
   **Required fix:** None.

7. **minor — Round 1 #7: PARTIALLY RESOLVED. Pair ordering is complete; numerical minimizer ties remain underspecified.**  
   All eleven eligible pairs now have a total order, R3 is excluded, and the 2% window uses one baseline reference. But “relative difference ≤1e-12” does not define its denominator, zero/zero handling, or comparison against the global minimum. [Design:308–314](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:308), [JSON:454–459](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:454).  
   **Required fix:** Define an explicit tolerance formula against the finite global minimum, including exact equality at zero, before applying the larger-m/λ/M preference.

8. **note — Round 1 #8: RESOLVED. Numerical validity and incomplete mandatory results fail closed.**  
   Minimum sample sizes, invalid bootstrap limits, empty P, invalid CIs, zero/nonfinite normalizers, invalid objectives and invalid synthetic repetitions are covered. Adoption requires every mandatory statistic and acceptance cell. [Design:318–321](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:318), [JSON:461–472](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:461).  
   **Required fix:** None.

9. **note — Round 1 #9: RESOLVED under the revised R3 eligibility rule.**  
   Complete artifact provenance now covers fitting, within-recipe selection, calibration and frozen feature inputs through **2025-05-31**, excluding June/DEV/S6/2026 rows and XF artifacts. A2’s R1 tables remain April-30 TRAIN-only; R3 remains diagnostic-only. [Design:160–163](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:160), [JSON:254–267](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:254).  
   **Required fix:** None. The actual artifact and provenance must still pass the stated registration gate.

10. **note — Round 1 #10: RESOLVED. The pitcher seed reference exists.**  
    `seeds.roles.pitcher_bootstrap` now specifies index 0 and value `3624346705`; the role mapping and derivation are explicit. [Design:168](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:168), [JSON:509–529](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:509).  
    **Required fix:** None for the broken reference. I did not execute seed derivation; its verification remains assigned to G-CODE.

11. **minor — Round 1 #11: PARTIALLY RESOLVED. Most estimand repairs are present; one overclaim remains.**  
    P is explicitly conditional, E1 includes game state, the decomposition is illustrative, and R5-f includes a common denominator and covariance. [Design:84](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:84), [design:180](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:180), [design:289–290](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:289).  
    Nevertheless, “otherwise differs by the bias measured in synthetic_bias_check” still equates the real-data estimand difference with synthetic-measured bias, contrary to the explicit certification limits. [Design:144](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:144), [JSON:215](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:215), [JSON:433](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:433).  
    **Required fix:** State that §4 quantifies the difference **only in the declared synthetic worlds**; the real-data difference remains uncertified.

12. **minor — Round 1 #12: PARTIALLY RESOLVED. Planning arithmetic is qualified, but the requested B/2 precision gate disappeared.**  
    R=2,000 is correctly described as planning arithmetic, and post-result repetition increases are prohibited. However, `U=|b̂|+1.96SE≤B` permits PASS when `SE>B/2`. For example, **derived arithmetic:** `b̂=0`, `SE=6.4e-6` gives `U=1.2544e-5≤B`, despite `B/2=6.3e-6`. [Design:239](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:239), [design:260–261](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:260), [JSON:424–425](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:424).  
    **Required fix:** Preserve the requested separate `SE≤B/2` requirement and specify `INCONCLUSIVE` for inadequate precision.

    Recalculation using `SE=CI width/3.92` and `2.8016×SE` verifies the recorded-summary arithmetic:

    | Derived quantity | S6 | 2026 |
    |---|---:|---:|
    | SE_lo | 0.0003584942 | 0.0002442886 |
    | SE_point | 0.0002900758 | 0.0000823537 |
    | Point MDE | 0.0008126765 | 0.0002307222 |
    | Current-rule MDE | 0.0025965991 | 0.0027410784 |
    | Current-rule MDE + b_V | 0.0053313286 | 0.0054758080 |
    | Illustrative variance share | 34.5275% | 88.6352% |

    Sources: [S6 summary:13](/Users/song/Projects/pitcheezy/results/ML-POLICY-VAL-v1-S6-v3.json:13), [S6 summary:26](/Users/song/Projects/pitcheezy/results/ML-POLICY-VAL-v1-S6-v3.json:26), [S6 summary:50](/Users/song/Projects/pitcheezy/results/ML-POLICY-VAL-v1-S6-v3.json:50), [2026 summary:29](/Users/song/Projects/pitcheezy/results/MLB-2026-OPE-v1.json:29), [2026 summary:64](/Users/song/Projects/pitcheezy/results/MLB-2026-OPE-v1.json:64), [2026 summary:77](/Users/song/Projects/pitcheezy/results/MLB-2026-OPE-v1.json:77), [2026 summary:102](/Users/song/Projects/pitcheezy/results/MLB-2026-OPE-v1.json:102).

    Further derived checks: `0.25×0.7×0.000290×√(150/2430)=1.260893887e-5`; `0.000290/√2000=6.484597e-6`. Cost scaling gives `2900×15/9=4833.333 s` and doubled guard `9666.667→9667 s`, consistent with the disclosed assumption and rounding. [Design:253–254](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:253), [JSON:545–556](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:545).

13. **note — Round 1 #13: PARTIALLY RESOLVED; PM-owned README only.**  
    B3 exclusion and registration-before-smoke are corrected. README still presents the 2027 quantity as a choice and calls G-CODE “code reading only,” whereas A2 includes synthetic tests. [README:9–12](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/README.md:9), [README:27](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/README.md:27), [design:392](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:392), [JSON:589](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:589).  
    **Required fix:** PM reconciliation only; excluded from A2’s defect count. A3→A1→A2 is correctly identified as a PM decision, not D186. [README:24–25](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/README.md:24).

14. **major — New: pooled S-1/S-4 standard errors omit cross-seed covariance.**  
    The revised gates use `sqrt(sum SE²)/3`, while preserving S5 log/truth streams. [Design:263](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:263), [JSON:427](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:427), [JSON:527](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:527).  
    The existing runner varies the planning seed but supplies the **same** `role_seed(config,'v2')` to each world run. Log and truth randomness derive from that shared seed. Consequently, independence of the three gaps is not established; independent added-layer streams do not remove shared action/delivery/nonterminal randomness. The pooled SE and stated ≤5% rejection guarantee are unsupported. [Runner:1545–1558](/Users/song/Projects/pitcheezy/experiments/pitchmdp/scripts/run_policy_validation.py:1545), [policy_semisynthetic.py:109](/Users/song/Projects/pitcheezy/experiments/pitchmdp/pitchmdp/policy_semisynthetic.py:109), [policy_semisynthetic.py:139–140](/Users/song/Projects/pitcheezy/experiments/pitchmdp/pitchmdp/policy_semisynthetic.py:139).  
    **Required fix:** Register independent complete MC streams per run, or compute the variance of the synchronized three-seed average including covariance.

15. **minor — New: the public allowlist and blanket prohibitions conflict.**  
    R5-g explicitly publishes reward-free weight CIs; R5-i includes signed TRAIN residuals. These enter the allowlist, but the prohibition excludes **any** nonbaseline CI endpoint, and the public-type rule does not classify signed TRAIN residuals. This is a schema contradiction, not an identified rehearsal-center leak. [Design:181–188](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:181), [JSON:285–295](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:285), [JSON:497–502](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:497).  
    **Required fix:** Scope the endpoint prohibition to effect/shift intervals and explicitly enumerate permitted TRAIN diagnostics.

16. **minor — New: the config hash target is undefined.**  
    `config_sha256` is required inside the config itself, while status changes and later seals/addenda also affect the registration record. No canonicalization, excluded-field or external-manifest rule specifies the hashed bytes. [Design:394–395](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:394), [JSON:19–25](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:19), [JSON:638–657](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:638).  
    **Required fix:** Define an immutable hash target and separate append-only receipts; an external digest manifest would avoid self-reference.

17. **minor — New: the closed-form cluster SE needs an executable definition.**  
    λ/M selection now depends on “sample variance of per-game sums,” without specifying centering, normalization or finite-cluster correction. Raw sums vary with game PA counts even when every PA contrast is identical. [Design:151](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:151), [JSON:225–226](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:225).  
    **Required fix:** Pin the exact cluster-SE equation, including treatment of unequal game sizes, and use it identically in rehearsal and TAB.

18. **note — New: smoke extrapolation omits work performed by full TAB.**  
    Full TAB repeats m selection and λ/M selection, while smoke prohibits those selections and the associated metrics. A simple repetition-ratio extrapolation therefore does not measure the complete per-repetition workload. [Design:238](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:238), [design:366](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:366), [JSON:534](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:534).  
    **Required fix:** Explicitly mark the excluded workload as unmeasured and account for it before treating the extrapolation as the G-SMOKE cost check. Keep the hard runtime guards unchanged.

## Sections

- **A — FAIL.** Arm sets, grids, matrix sizes and stage structure are substantially aligned. The R1-a certification formula, pooled SE and remaining numerical definitions prevent a fully executable specification. [Design:228](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:228), [JSON:427](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:427).

- **B — PASS.** D186’s ID, point-primary decision, separate censoring report, secondary lower-endpoint rule, WE, X/B and execution window are fixed. Deferring the 2027 numerical criterion/A2-C format is acceptable because A2 adoption means protocol eligibility only. [Design:325–330](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:325), [JSON:474–488](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:474).

- **C — NOTE.** The direct center leak is repaired; selection is sealed and unblinding requires a committed decision record. Clarify the allowlist scope; statistical outcomes cannot authorize retries. [JSON:491–507](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:491), [JSON:579–584](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:579).

- **D — PASS.** TRAIN-only R1 fitting, raw-file allowlisting, summary-only 2026 use, exposed DEV interpretation and the revised complete R3 provenance restriction are explicit. [JSON:159–175](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:159), [JSON:254–267](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:254).

- **E — PASS at specification level.** The five identity hashes match the cited preregistration; protected closure files and existing verdicts remain excluded from modification. Sealed artifact bytes were not reopened or verified. [JSON:43–62](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:43), [JSON:620–627](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:620), [preregistration:595–610](/Users/song/Projects/pitcheezy/configs/MLB-2026-OPE-PREREG-DRAFT-v1.json:595).

- **F — FAIL.** Registration-first order, G-INT, missing-R3 fallback, validity gates and operational guards are stated. The start-time-plus-hang-guard deadline is enforceable as written, but certification/precision defects remain. [Design:377–399](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:377), [JSON:424–430](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:424).

- **G — NOTE.** Sample arithmetic and recorded cost scaling verify within disclosed rounding. New costs remain unmeasured; smoke’s omitted workload needs explicit treatment. [Design:46–98](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:46), [JSON:534–562](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:534).

- **H — FAIL.** Conditional P, surrogate assumptions, planner-favoring bias and eligibility-only adoption are substantially clearer. The incorrect/incomplete certification definitions and remaining real-versus-synthetic bias wording still prevent accepting the stated certification. [Design:144–155](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:144), [design:228](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:228), [JSON:433–474](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:433).

Read-only review completed. No files changed, prohibited artifacts opened, or experiments/tests executed.