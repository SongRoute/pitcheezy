VERDICT: FAIL

1. **MAJOR — New inconsistency: SW-A1 requires a confirmation label that current A1 does not emit.**

   RS-03 requires `STABLE_WITHIN_COST:<arm>` on both the original sample and confirmation slice. Current A1 emits `CONFIRMED` or `NOT_CONFIRMED` for the confirmation slice, leaving the original verdict unchanged. A literal implementation would therefore reject a successfully confirmed RB candidate. [RS-03 JSON:267](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:267), [design:34](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:34), [A1 JSON:117](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A1-planner-variance-registration-draft.json:117).

   **Fix:** Require original verdict `STABLE_WITHIN_COST:<arm>` **and** confirmation label `CONFIRMED` for that same arm. Map unsuccessful, missing or aborted confirmation to MC/no A_RB. Synchronize both RS-03 files without changing the numeric gates.

2. **MAJOR — Round-2 #11: PARTIALLY RESOLVED. Digest self-reference is fixed; sealing chronology still conflicts.**

   The seven nulled paths match the permitted transition paths, and external receipts remove the self-hash/future-review dependency. However, the terminal manifest requires the canonical seal timestamp and #37 comment URL, while step 3 appends that record **before posting the comment**. The design has the same ordering. [JSON:857](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:857), [JSON:866](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:866), [JSON:881](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:881), [design:119](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:119).

   **Fix:** Verify the sealed copy, publish the canonical seal event and decision record, then append the completion manifest using their actual identifiers and timestamp. Alternatively, explicitly distinguish preliminary and completion records. Execution must wait for completion.

3. **MINOR — New synchronization issue: ordered labels omit explicit K2/K4 routing.**

   The ordered predicates explicitly name K1/K6 and validity gates, while separate stop rules assign K2 → affected-arm `NOT_EVALUABLE` and K4 → ledger-analysis `INVALID`. Those outcomes should be part of the authoritative ordered rule. [JSON:418](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:418), [JSON:790](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:790), [JSON:800](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:800).

   **Fix:** Include applicable K4 failures in first-priority `INVALID`, and K2 failures in `NOT_EVALUABLE`, preserving their stated scope.

4. **NOTE — Round-2 #2: RESOLVED. SW-EST matches current A2.**

   The mapping covers all five current A2 outcomes plus unavailable/uncompleted A2. Adoption imports one complete original-denominator B0 tuple; R3 remains diagnostic-only unless separately registered certification is completed before sealing. The design agrees. [JSON:285](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:285), [design:47](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:47), [A2 JSON:460](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:460), [A2 JSON:473](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:473).

   **Fix:** None for this item.

5. **NOTE — Round-2 #5: RESOLVED. B1 restrictions and exposed-development selection are consistent.**

   DEV is prohibited from fitting, calibration and mixing-weight selection. Stage-B inclusion, A1 selection, A2 pair/λ/M selection and A3 candidate selection are explicitly disclosed as selection on exposed ≤2025 DEV. The complete 2026/2027 selection prohibition remains. [JSON:153](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:153), [JSON:548](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:548), [design:75](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:75), [P48 PREREG:147](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/docs/PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md:147).

   **Fix:** None; retain the distinction.

6. **NOTE — Round-2 #12: RESOLVED. Retry rules now agree.**

   Both layers share one atomic look. Two additional post-marker infrastructure attempts are permitted only before scientific exposure, logged append-only, and expressly exempted from K6. Scientific exposure prohibits subsequent attempts. [JSON:766](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:766), [JSON:777](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:777), [JSON:811](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:811), [design:112](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:112).

   **Suggested clarification:** Say explicitly that the first exposed **attempt may finish**, but no subsequent attempt is allowed. This avoids interpreting “열람은 거기서 완전히 끝난다” as interrupting the prescribed analysis immediately.

7. **NOTE — Round-2 #16: RESOLVED. Both Q2 options have an exhaustive statistical partition.**

   For valid finite intervals, the ordered rules cover positive, negative and zero-containing intervals, including exact-zero and exact-threshold boundaries. The new below-threshold labels close the θ > 0 gap. [JSON:416](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:416), [design:59](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:59).

   **Fix:** None for the Q2 partition; incorporate the operational routing correction in finding 3.

8. **NOTE — Round-2 #13: RESOLVED substantively; one editorial reference remains.**

   Numeric references now resolve to current source lines/semantic keys. Approved B is **exactly 1.26e-5**; 1.2612e-5 is explicitly explanatory arithmetic only. [JSON:1141](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1141), [JSON:1175](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1175), [A2 JSON:425](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:425).

   **Fix:** Change `sources_read_sha256_at_v1` in [JSON:1110](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1110) to the actual `sources_read_sha256_at_v2` block at [JSON:1089](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1089). Source line drift alone is a NOTE under the declared semantic-key rule.

9. **NOTE — Earlier resolved #1, #3 and #4: PRESERVED. Bias, population and censoring qualifications did not regress.**

   `b_total` remains **미측정**; A2’s B does not authorize confirmation. Claims remain conditional on common P, whereas historical L2 SE is only a planning proxy. The approximately 89% quantity is explicitly an illustrative variance residual across different populations/denominators; the asymptotic floors remain scenarios. [JSON:441](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:441), [JSON:610](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:610), [JSON:642](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:642), [design:95](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:95).

   **Fix:** None; preserve these qualifications.

10. **NOTE — Earlier resolved #6: PRESERVED. Stage-B inclusion predicates match the PREREG.**

    The primary candidate, both-split intersection-union, ΔNLL ≤ −0.003, strict CI upper < 0, unadjusted one-sided p ≤ .05, Brier upper ≤ .001, and four-of-five improving seeds are retained. [JSON:184](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:184), [P48 PREREG:157](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/docs/PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md:157).

    **Fix:** None. Preserve the authoritative predicates when replacing the draft config pin.

11. **NOTE — Earlier resolved #7 and #8: PRESERVED at draft stage.**

    Predictor scoring remains conditional on realized pitch type, with its incomplete scoring specification explicitly required before sealing or excluded. Ledger timing, historical snapshot replay and final-Statcast reconstruction retain distinct claims. These distinctions agree with the RS-02 source. [JSON:222](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:222), [JSON:564](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:564), [RS-02:833](/Users/song/Projects/pitcheezy-pm-jobs/20261020-rs02/REGISTRATION-RS02-SEALED.json:833), [RS-02:904](/Users/song/Projects/pitcheezy-pm-jobs/20261020-rs02/REGISTRATION-RS02-SEALED.json:904).

    **Required follow-through:** Complete and independently review the predictor specification, or exclude that layer. This review does not certify the unwritten specification.

12. **NOTE — Earlier resolved #9 and #10: PRESERVED.**

    Contrast operands and the Holm family freeze at sealing; interval-width selection, exact ties and p construction remain explicit. RS-03 now correctly identifies its wider-interval decision rule as its own choice. `NO_DIFFERENCE_ESTABLISHED` does not claim equivalence or exclude effects above MDE. [JSON:373](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:373), [JSON:507](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:507), [JSON:432](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:432).

    **Fix:** None for these items.

13. **NOTE — Earlier resolved #14: PRESERVED. Requested arithmetic reproduces.**

    These are **recalculations from recorded document inputs**, not new measurements or verification of original OPE outputs. Inputs and method: [A2 JSON:84](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:84), [A2 JSON:116](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:116).

    | Quantity | Recomputed |
    |---|---:|
    | Current-rule MDE | 0.002741074785 |
    | Including b_V | 0.005475804328 |
    | Point-estimate MDE | 0.000230720926 |
    | After 30% point-SE reduction | 0.000161504648 |
    | Approximate residual variance share | 88.63525% |
    | Current-rule MDE after reduction | 0.002720944893 |

    The printed 0.00023071 uses the source’s rounded SE; its tiny difference from the CI-derived value is source-recorded rounding. [A2 JSON:123](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:123).

    Using the **printed MDEs**, as RS-03’s formula specifies:

    | δ | Current-SE games | 30%-reduction games |
    |---|---:|---:|
    | 0.001 | 129.3 | 63.4 |
    | 0.0003 | 1,436.5 | 703.9 |
    | 0.0002 | 3,232.2 | 1,583.8 |
    | 0.0001 | 12,928.9 | 6,335.4 |

    These reproduce the registration table. Every listed δ is below b_V, so its bias-sensitivity “no solution under this formula” entries are correct. [JSON:650](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:650), [JSON:657](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:657).

    **Fix:** No arithmetic correction required; retain the planning-proxy limitations.

14. **NOTE — Earlier resolved #15: PRESERVED. D186, scientific boundaries and open decisions remain explicit.**

    C1 is the single named draft primary contrast; paired Δ on P, its interval, separate censoring report, secondary D108 rule and WE reward are represented. The 2026/2027 bans, postseason exclusion, conditional D155 reopening and prohibition on automatic service adoption remain. [JSON:354](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:354), [JSON:403](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:403), [JSON:558](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:558), [JSON:969](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:969).

    G0/G15 document hashes match their declared pins, and A retains `c13cc989…`. All eight questions remain open with one recommendation each; their answers and implementation must be finalized before exact-revision review and sealing. [JSON:74](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:74), [JSON:119](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:119), [design:150](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:150), [JSON:879](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:879).

    **Fix:** Preserve these boundaries while correcting findings 1–3. Draft review does not authorize execution.

Reviewed snapshots: design SHA256 `7b250fbe…0f7585`; JSON `f060d002…7746ac`, on `codex/ml-matrix-execution` at `11eeb6e2`. No files or git state were modified; no prohibited data, original OPE outputs, result directories or T7 contents were opened.