VERDICT: FAIL

The original bias-bound blocker is resolved, but four major inconsistencies remain. Items 1–15 below correspond to the round-1 findings; item 16 is an additional issue.

Reviewed RS-03 snapshots: design SHA256 `9b7398de…a2327f`; JSON `be3b889c…c46de`. Track-A references below reflect its current revised content.

1. **NOTE — Round 1 #1: RESOLVED. Total bias is no longer replaced by A2’s B.**

   `b_total` is explicitly **미측정**, confirmation requires a separately certified budget, and the invalid B-based confirmation power column is removed. The design agrees. [JSON:408](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:408), [JSON:639](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:639), [design:64](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:64).

   **Required follow-through:** Preserve this gate. Any eventual budget must cover the actual P estimand; current A2 certification expressly excludes real-data P-selection and 2027 drift. [A2 JSON:433](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:433).

2. **MAJOR — Round 1 #2: PARTIALLY RESOLVED. Coupled import is fixed, but the switch no longer matches current A2.**

   SW-EST now imports a complete tuple, eliminating independent A2/A3 substitution. However, it still permits an “A2-certified” A3/R3 denominator, while current A2 makes R3 **diagnostic-only and ineligible for adoption**. Its fallback list also omits A2’s new `NOT_EVALUABLE` outcome. [RS-03 JSON:278](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:278), [JSON:297](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:297), [A2 design:162](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:162), [A2 JSON:445](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:445).

   **Fix:** Under the current A2 registration, restrict primary adoption to its original-denominator tuples. Require separate registered certification for a future A3 primary combination. Make fallback handling exhaustive, including `NOT_EVALUABLE`.

3. **NOTE — Round 1 #3: RESOLVED. Population and power claims are separated.**

   Claims are restricted to common P with exclusions, censoring and coverage disclosed. Historical L2 SE is explicitly a planning proxy, and P-specific SE is **미측정**. [JSON:338](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:338), [JSON:566](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:566), [design:81](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:81).

   **Fix:** None beyond preserving these qualifications in the final report.

4. **NOTE — Round 1 #4: RESOLVED. Censoring floors are scenarios, not universal limits.**

   The approximately 89% quantity is identified as an inferred variance share. Constant-censoring-SE and zero-SE censoring-width scenarios are separated, with explicit rejection of a general sample-size limit. [JSON:598](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:598), [JSON:646](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:646), [design:92](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:92).

   **Suggested refinement:** Copy current A2’s additional caveat that the historical components use different populations and denominators, so independence alone does not establish a decomposition. [A2 JSON:125](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:125).

5. **MAJOR — Round 1 #5: PARTIALLY RESOLVED. B1 fitting roles are corrected, but DEV selection is inconsistently prohibited.**

   TRAIN, May calibration and June seed/ensemble MIX_FIT are now distinguished. Yet the design says DEV enters **no selection**, and JSON says it is “never a selection input,” while the same registration imports DEV-dependent Stage-B inclusion and rehearsal-selected λ/M. [JSON:146](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:146), [design:73](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:73), [JSON:511](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:511), [JSON:198](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:198), [JSON:285](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:285).

   **Fix:** Limit the B1 prohibition to fitting/calibration/mixing-weight selection, matching the PREREG’s “weight choice” restriction. Explicitly classify registered Stage-B inclusion and Track-A selection as **exposed-development selection**. Retain the complete 2026/2027 selection ban. [P48 PREREG:147](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/docs/PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md:147), [A2 JSON:488](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:488).

6. **NOTE — Round 1 #6: RESOLVED. Stage-B predicates now match.**

   The authoritative pinned rule and explicit operators include strict NLL CI upper `< 0`, unadjusted one-sided `p ≤ .05`, both-split intersection-union, Brier noninferiority and four-of-five improving seeds. They match the PREREG. [RS-03 JSON:173](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:173), [JSON:182](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:182), [P48 PREREG:157](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/docs/PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md:157).

   **Fix:** None; preserve the exact predicates when replacing the draft-config pin with its registered counterpart.

7. **NOTE — Round 1 #7: RESOLVED at draft stage. Predictor specification is explicitly gated.**

   Policy inputs and realized-pitch-type conditional scoring are separated. The incomplete mapping/revision, eligibility, pairing, clipping and coverage specification must be completed and independently reviewed before sealing; otherwise the predictor layer is excluded. [JSON:218](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:218), [JSON:553](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:553). This matches RS-02’s conditioning. [RS-02:904](/Users/song/Projects/pitcheezy-pm-jobs/20261020-rs02/REGISTRATION-RS02-SEALED.json:904).

   **Remaining work:** Complete and pin that specification, or record exclusion. This review does not certify an unwritten scoring contract.

8. **NOTE — Round 1 #8: RESOLVED. Reconstruction and historical as-of evidence are distinguished.**

   Immutable snapshot replay, first-receipt logging and final-Statcast reconstructed inputs now have separate definitions and claim limits. Reconstructed inputs no longer establish historical availability or physical pre-pitch timing. [JSON:521](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:521), [JSON:527](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:527), [JSON:531](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:531).

   **Fix:** None; preserve these distinctions after Q4 is answered.

9. **NOTE — Round 1 #9: RESOLVED. Contrast existence and statistical selection rules are explicit.**

   C5 requires its partner arm, the Holm family freezes after inclusion, and interval width selection, exact ties, p-value construction and sidedness are specified. [JSON:448](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:448), [JSON:473](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:473), [JSON:346](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:346), [JSON:491](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:491).

   **Follow-through:** The source attribution should acknowledge that current A2 requires both bootstrap tests to pass; RS-03’s wider-interval decision rule is its own choice. [A2 design:169](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-design.md:169).

10. **NOTE — Round 1 #10: RESOLVED. Failure to establish a difference is no longer effect exclusion.**

    `NO_DIFFERENCE_ESTABLISHED` explicitly means the registered analysis did not establish a difference, without asserting no effect or excluding effects above MDE. [JSON:399](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:399), [design:124](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:124).

    **Fix:** None for the default θ = 0 branch; see item 16 for the optional positive threshold.

11. **MAJOR — Round 1 #11: PARTIALLY RESOLVED. The sealing lifecycle still cannot satisfy its own requirements.**

    Finalization → exact-revision review → restricted sealing is now explicit, and late sealing has a cutoff, exposure attestations and partial-season claims. However, step 1 requires **every pin filled before review**, including the revision’s own SHA and the future step-2 review record. No self-hash exclusion rule exists. Step 3 permits a field named `review`, while the actual field is `independent_review`. [JSON:780](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:780), [JSON:805](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:805), [JSON:829](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:829), [JSON:852](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:852).

    **Fix:** Keep revision digests and review receipts in an external manifest, or define an exact non-self-referential hash payload. Specify permitted JSON paths and terminal values, including `registered`, `execution.enabled` and `independent_review`; mirror the contract in the design. [JSON:7](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:7), [JSON:30](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:30), [design:115](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:115).

12. **MAJOR — Round 1 #12: PARTIALLY RESOLVED. Both layers are protected, but retry rules conflict.**

    Any scientific exposure now consumes the shared look, with atomic marking and append-only access records. But creating `LOOK.done` also consumes it immediately; infrastructure retries are allowed before scientific exposure, while K6 invalidates **any rerun after consumption**. A failure after marker creation but before exposure therefore has contradictory recovery instructions. [JSON:722](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:722), [JSON:730](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:730), [JSON:762](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:762).

    **Fix:** Either restrict retries to pre-marker failures, or define bounded infrastructure attempts within the same consumed look and expressly exempt those attempts from K6. Make the design identical. [Design:108](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:108).

13. **MINOR — Round 1 #13: PARTIALLY RESOLVED. Provenance is substantially improved, but needs synchronization.**

    Document corroboration and original-output provenance are separated, with source digests and line-drift disclosure. Nevertheless, several current citations miss their values: the multiplier is now A2 JSON:78, point SE :117, and A1 confirmation games/salt :113. [RS-03 JSON:1021](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1021), [JSON:1074](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1074), [A2 JSON:78](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:78), [A2 JSON:117](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:117), [A1 JSON:113](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A1-planner-variance-registration-draft.json:113).

    There is also a substantive numeric distinction: RS-03 calls **1.2612×10⁻⁵** approved B, whereas current A2 uses exactly **1.26×10⁻⁵**; the former is explanatory arithmetic only. [RS-03 JSON:1108](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1108), [design:48](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:48), [A2 JSON:411](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:411).

    **Fix:** Refresh references against settled source revisions, retain semantic keys alongside lines, and distinguish the exact approved threshold from its derivation.

14. **NOTE — Round 1 #14: RESOLVED. Requested arithmetic reproduces.**

    The following are **arithmetic on recorded document inputs**, not new measurements. Inputs and formulas are recorded in [A2 JSON:78](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:78), [A2 JSON:110](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:110), and [RS-03 JSON:606](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:606).

    | Quantity | Recomputed |
    |---|---:|
    | Current-rule MDE | 0.002741074785 |
    | Including b_V | 0.005475804328 |
    | Point MDE | 0.000230720926 |
    | Point MDE after 30% SE reduction | 0.000161504648 |
    | Approximate residual variance share | 88.63525% |
    | Current-rule MDE after that reduction | 0.002720944893 |

    Using the printed MDEs, as the registration does:

    | δ | Current-SE games | 30%-reduction games |
    |---|---:|---:|
    | 0.001 | 129.3 | 63.4 |
    | 0.0003 | 1,436.5 | 703.9 |
    | 0.0002 | 3,232.2 | 1,583.8 |
    | 0.0001 | 12,928.9 | 6,335.4 |

    These match the table; every listed δ is below b_V, so the sensitivity column’s “no solution under this formula” is correct. [JSON:613](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:613). **Fix:** None beyond preserving the planning-proxy qualifications.

15. **NOTE — Round 1 #15: RESOLVED / PRESERVED. Core boundaries and open decisions remain explicit.**

    C1 is named; D186’s paired point estimate and interval, separate censoring report, secondary D108 row and WE reward are represented. The 2026/2027 restrictions, postseason exclusion, conditional D155 reopening and prohibition on automatic service adoption remain explicit. [JSON:336](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:336), [JSON:384](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:384), [JSON:42](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:42), [JSON:514](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:514), [JSON:899](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:899).

    G0/G15 document hashes match their declared pins; `c13cc989…` remains the fixed A identity. All eight decisions remain presented with one recommendation each. [JSON:72](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:72), [JSON:117](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:117), [design:145](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:145).

    **Fix:** Preserve these boundaries while resolving the remaining major findings.

16. **MINOR — Additional issue: Q2’s positive-threshold option has an incomplete label partition.**

    Improvement/harm predicates use θ, but `NO_DIFFERENCE_ESTABLISHED` requires the CI to contain zero. If Q2 chooses θ = MEI, an interval entirely above zero but below θ satisfies none of those predicates. The default θ = 0 is complete. [JSON:395](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:395), [JSON:931](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:931).

    **Fix:** Define an exhaustive, ordered label rule for either Q2 answer. If θ > 0, use an appropriate residual label for valid analyses establishing neither threshold-qualified improvement nor harm.

No files were modified, and no prohibited data, original result files, OPE outputs or T7 contents were opened.