VERDICT: PASS_WITH_NOTES

## Findings

1. **note — Round-3 #1 (formerly major): RESOLVED.**  
   Each of the six repetitions uses log `r−1` and truth rollouts `2r−2, 2r−1` over the fixed starts, recomputing λ/M selection and SN normalizers. `SD(gap_r)/√6` now measures uncertainty of the complete repeated estimator; the inherited rowwise SE is descriptive only. The smaller 147-PA estimator and coarse-screen scope are disclosed. [Design:236](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:236), [JSON:359](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:359).  
   **Required fix:** None at design level. G-CODE must verify sample SD, draw membership and complete recomputation.

2. **note — Round-3 #2: RESOLVED.**  
   Test and review instructions now reference the typed allowlist: R5-g weight CIs and R5-i TRAIN residuals remain permitted; the prohibition specifically covers effect/shift endpoints. [Design:373](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:373), [JSON:627](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:627).  
   **Required fix:** None.

3. **note — Round-3 #3: RESOLVED.**  
   The config has no digest-value pin. Serialization specifies `ensure_ascii=False`, sorted keys, indent 2, UTF-8 and exactly one trailing newline; exact null paths are enumerated. `new_files_sha256` excludes the registration config, and receipts remain external. [Design:412](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:412), [Design:414](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:414), [JSON:26](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:26), [JSON:661](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:661).  
   **Required fix:** None.

4. **note — Round-3 #4: RESOLVED.**  
   Added-layer keys explicitly distinguish log and truth namespaces, while candidate/reference truth rollouts retain CRN. Distinct run seeds and disjoint repetition slices support the pooled SE calculation, conditional on fixed inputs. [Design:295](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:295), [JSON:410](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:410), [JSON:550](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:550).  
   **Required fix:** None; implementation wiring remains a G-CODE check.

5. **note — Round-3 #5: RESOLVED.**  
   The ≤5% statement now applies only to interval exclusion, conditional on the SE/coverage approximation. The magnitude and precision screens are explicitly separate. Six-repetition normal coverage is acknowledged as approximate. [Design:277](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:277), [JSON:446](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:446).  
   **Required fix:** None; preserve these qualifications in results reporting.

6. **note — Round-3 #6: RESOLVED.**  
   Both files distinguish removal of direct class dependence at κ=0 from full R1-b exclusion, which requires κ=ζ=0 in the declared world. [Design:226](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:226), [JSON:343](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:343).  
   **Required fix:** None.

7. **note — Round-3 #7: RESOLVED.**  
   L2 now includes completed E0 PAs plus `rho_zero` and `rho_zero_before_node` resolutions, matching the implementation. [Design:84](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:84), [JSON:133](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:133), [policy_estimator.py:250](/Users/song/Projects/pitcheezy/experiments/pitchmdp/pitchmdp/policy_estimator.py:250).  
   **Required fix:** None.

8. **minor — The introductory adoption summary overstates the absolute bias guarantee.**  
   The summary says combined synthetic bias has a 95% upper bound ≤B across the declared worlds. Detailed acceptance instead uses **U≤|β₀|+B for V3**, and S5-X S-1/S-4 are coarse screens. Those detailed rules agree between files; the summary is stronger than them. [Design:14](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:14), [Design:275](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:275), [Design:277](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:277), [JSON:448](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:448).  
   **Required fix:** Qualify the summary as an absolute B bound in the specified V2 acceptance cells, a baseline-relative bound in V3, plus coarse S5-X screens. Do not change the registered thresholds.

9. **note — The description of censoring share `s` remains imprecise.**  
   Both files call `s` the share of PAs left as bounds. Existing `s` counts worst-case residual resolutions only; it excludes shared-end bounds. The draft itself distinguishes six worst-case PAs and one shared-end PA, and R5-f already includes all contributions outside P. [Design:342](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:342), [JSON:499](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:499), [policy_estimator.py:404](/Users/song/Projects/pitcheezy/experiments/pitchmdp/pitchmdp/policy_estimator.py:404), [Design:180](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:180).  
   **Required clarification:** Say “worst-case residual share `s`, remaining shared-end bounds, and the censoring variance/covariance contribution.” This does not invalidate the A2 gate or require fixing A2-C’s final format here.

10. **note — No regression found in the previously resolved round-1/2 items.**

    | Earlier items | Current evidence |
    |---|---|
    | #1 blinding; #2 input/output pins and R3 fallback | Signed shifts/selection remain sealed; R3 is fixed at registration, and generated outputs remain external stage seals. [JSON:270](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:270), [JSON:511](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:511), [JSON:681](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:681). |
    | #3 exact B; #4 baseline expectation; #5 separate γ formulas | Exact B, `β₀=0`, separate γ_a/γ_b and the ζ=0 invariant remain. [JSON:429](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:429), [Design:239](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:239), [Design:252](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:252). |
    | #6 retries; #7 ties; #8 validity | Statistical failures cannot authorize retries; finite-minimum tolerance and mandatory-statistic rejection remain explicit. [JSON:478](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:478), [JSON:492](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:492), [JSON:600](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:600). |
    | #9 R3 provenance; #10 pitcher seed | Complete-artifact provenance, XF exclusion and diagnostic-only status remain; pitcher role index/value are specified. [JSON:265](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:265), [JSON:537](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:537). |
    | #11 estimand limits; #12 precision | Real-data surrogate differences remain uncertified; `SE≤B/2` precedes acceptance. [JSON:223](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:223), [JSON:443](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:443). |
    | #14–#18 streams, allowlist, digest, cluster SE, smoke projection | Preserved, including the exact unequal-cluster-size equation and explicitly unmeasured selection-cost projection. [JSON:234](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:234), [JSON:407](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:407), [JSON:557](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:557); findings 2–4 above. |

    **Required fix:** None for these resolved issues. README reconciliation is addressed separately below.

11. **note — Repetition budgets and sampled arithmetic verify.**  
    **Derived arithmetic:** each S5-X run still uses `147×6=882` logs and `147×12=1,764` paired truth-rollout indices. Six repetitions partition that budget; they do not multiply it by six. Additional estimation/selection cost is explicitly unmeasured. `2900×15/9=4833.333… s` supports the approximately 4,800-second projection; doubling gives `9666.666…→9667 s`. Thus the design’s approximately 9,700 seconds and JSON’s exact guard agree. [Design:236](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:236), [Design:387](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:387), [JSON:579](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:579).

    Recalculation from the committed summaries, using width/3.92 and `2.8016×SE`, gives:

    | Derived quantity | S6 | 2026 |
    |---|---:|---:|
    | SE_lo | 0.0003584942 | 0.0002442886 |
    | SE_point | 0.0002900758 | 0.0000823537 |
    | Point MDE | 0.0008126765 | 0.0002307222 |
    | Current-rule MDE | 0.0025965991 | 0.0027410784 |
    | Current-rule MDE + b_V | 0.0053313286 | 0.0054758080 |
    | Illustrative variance share | 34.5275% | 88.6352% |

    Sources: [S6 CI:13](/Users/song/Projects/pitcheezy/results/ML-POLICY-VAL-v1-S6-v3.json:13), [S6 L2:26](/Users/song/Projects/pitcheezy/results/ML-POLICY-VAL-v1-S6-v3.json:26), [S6 s:50](/Users/song/Projects/pitcheezy/results/ML-POLICY-VAL-v1-S6-v3.json:50), [2026 CI:64](/Users/song/Projects/pitcheezy/results/MLB-2026-OPE-v1.json:64), [2026 L2:77](/Users/song/Projects/pitcheezy/results/MLB-2026-OPE-v1.json:77), [2026 s:102](/Users/song/Projects/pitcheezy/results/MLB-2026-OPE-v1.json:102), [b_V:30](/Users/song/Projects/pitcheezy/results/MLB-2026-OPE-v1.json:30).

    Further **derived arithmetic**: `0.25×0.7×0.000290×√(150/2430)=1.260893887e−5`; `0.000290/√2000=6.484597e−6`, above `B/2=6.3e−6`. The draft correctly treats actual paired SD as unmeasured and inadequate precision as `INCONCLUSIVE`. [Design:253](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:253), [Design:266](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:266), [JSON:443](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:443).  
    **Required fix:** None; unchanged simulation counts must not be presented as measured unchanged runtime.

12. **note — The 95% bias bounds are pointwise.**  
    Bounds are defined per pair/cell, followed by selection among eleven candidates. This does not establish a simultaneous 95% guarantee for the selected candidate. The current detailed specification does not explicitly promise that guarantee. [Design:270](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:270), [Design:274](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:274), [Design:323](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:323).  
    **Required clarification:** Preserve “nominal per pair/cell” in reporting. Any stronger simultaneous claim requires a prospectively registered adjustment.

13. **note — README reconciliation remains PM-owned.**  
    Previous decision-quantity and code-reading-only discrepancies are corrected. README still says the registration block contains the receipt path and digest, whereas A2 keeps the digest external-only. [README:12](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/README.md:12), [README:34](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/README.md:34), [README:35](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/README.md:35), [JSON:27](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:27).  
    **Required fix:** PM should clarify A2’s external-only digest arrangement. This difference is excluded from the draft’s defect count.

## Sections

- **A — NOTE.** Arm sets, grids, R5 a–i, synthetic matrices, repetitions, seeds, labels, tie rules, stages and guards align. The remaining summary-versus-detailed-rule discrepancy is finding 8. [Design:288](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:288), [JSON:397](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:397).

- **B — PASS.** D186’s ID, WE, point-estimate primary quantity, separate censoring report, secondary lower-endpoint rule, X/B and run window are fixed. Deferring the 2027 numerical criterion/A2-C format is acceptable because A2 adoption means protocol eligibility only. [D186:325](/Users/song/Projects/pitcheezy/docs/decisions.md:325), [Design:340](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:340), [JSON:496](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:496).

- **C — PASS.** Selection is outcome-dependent but operator-blinded. Centers, shifts and selection details stay sealed; unblinding requires the committed decision record, with no outcome-driven retry or post-unblinding revision. [JSON:511](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:511), [JSON:602](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:602).

- **D — PASS under the previously reviewed R3 amendment.** R1 remains April-30 TRAIN-only; raw joins are allowlisted and 2026 use is summary-only. R3 permits the documented primary-artifact provenance through May 31—not strictly April-30 TRAIN-only—while excluding June/DEV/S6/2026 fitting and XF artifacts. [JSON:167](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:167), [JSON:265](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:265), [A3:218](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A3-behavior-policy-design.md:218).

- **E — PASS at specification level.** The five copied identity hashes match the preregistration. Protected closure files, sealed verdicts, service and G15 remain excluded from change. Sealed artifact bytes were not reopened or verified. [JSON:643](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:643), [Preregistration:595](/Users/song/Projects/pitcheezy/configs/MLB-2026-OPE-PREREG-DRAFT-v1.json:595), [Design:351](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:351).

- **F — PASS at specification level.** G-INT, validity/precision rejection, two-attempt limits and operational guards remain. Missing A3 input produces registration-fixed `NOT_RUN:R3`; the start-time-plus-hang-guard deadline is enforceable as written. Statistical baseline rejection prevents certification without authorizing repair. [JSON:270](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:270), [JSON:451](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:451), [JSON:591](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:591), [Design:418](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:418).

- **G — PASS.** Sampled arithmetic verifies within disclosed rounding. The six-repetition partition preserves simulation counts; added computation remains unmeasured, with unchanged hard guards. [Design:253](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:253), [Design:387](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:387).

- **H — NOTE.** Conditional-P interpretation, E1/E2, planner-favoring bias risk and eligibility-only adoption are explicit. Apply findings 8–9 and retain the pointwise-coverage limitation in finding 12. [Design:144](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:144), [Design:155](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:155), [Design:305](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:305).

This is the design-level G-REV review; G-CODE and G-REG remain required. [Design:407](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:407). No files were changed, prohibited artifacts opened, or experiments/tests executed.