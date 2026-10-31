VERDICT: PASS_WITH_NOTES

1. **MINOR — Completion time is required for game eligibility but is not recorded explicitly.**

   Eligibility requires scheduled first pitch after **both** the seal event and completion record. However, the completion-record fields contain only the #37 comment’s creation timestamp, not the completion record’s own timestamp. The prescribed records therefore cannot reconstruct the cutoff between steps 5 and 6. [RS-03 JSON:590](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:590), [JSON:867](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:867), [design:119](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:119).

   **Fix before sealing:** Record `completion_appended_at_utc`, with a defined clock and durable timing evidence. Define eligibility as `scheduled_first_pitch > max(seal_created_at, completion_appended_at)` in both files.

2. **MINOR — The historical L2 population description has not incorporated current A2’s correction.**

   RS-03 still describes L2 as completed-only and says P additionally includes ρ0 resolutions. Current A2 defines L2 as completed E0 PAs **plus `rho_zero` and `rho_zero_before_node` resolutions**. This is semantic source drift, beyond moved line numbers. [RS-03 design:82](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:82), [JSON:612](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:612), [JSON:643](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:643), [current A2 JSON:134](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:134).

   **Fix:** Synchronize every L2 description and refresh the A2 source references. Retain P-specific SE as **미측정**. The authoritative P classifier remains bound equality, so this correction does not change the registered estimand or arithmetic. [RS-03 JSON:357](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:357).

3. **NOTE — Round-3 #1, SW-A1 inclusion: RESOLVED.**

   Inclusion now requires original `STABLE_WITHIN_COST:<arm>` **AND** confirmation label `CONFIRMED` for that same arm. Negative, aborted, missing or unfinished confirmation yields MC/no A_RB. This matches current A1. [RS-03 JSON:268](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:268), [JSON:281](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:281), [design:46](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:46), [A1 JSON:117](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A1-planner-variance-registration-draft.json:117).

   **Fix:** None.

4. **NOTE — Round-3 #2, sealing chronology: RESOLVED for the reported ordering conflict.**

   The sequence is now review/preliminary record → digest-specific Song sign-off → written and reverified sealed copy → #37 canonical seal event → decisions.md seal line → completion record using actual identifiers → post-seal checks. Execution explicitly requires completion **and** successful post-seal checks. [RS-03 JSON:879](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:879), [JSON:888](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:888), [design:119](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:119).

   The seven nulled digest paths match the permitted transition paths; in-memory substitution preserved the digest, while an analysis-field change altered it. [JSON:857](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:857).

   **Fix:** Only the completion-clock refinement in finding 1 remains.

5. **NOTE — Round-3 #3, K2/K4 routing: RESOLVED.**

   Applicable K4 failures now receive first-priority `INVALID`; K2 failures route affected-arm comparisons to `NOT_EVALUABLE`; K3 retains comparison-specific `NOT_EVALUABLE`. The ordered labels and stop rules agree. The exhaustive partition for both Q2 thresholds is preserved. [RS-03 JSON:419](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:419), [JSON:429](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:429), [design:113](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:113).

   **Fix:** None.

6. **NOTE — Round-3 #8, editorial reference: RESOLVED.**

   `provenance_conventions.doc` now correctly names `sources_read_sha256_at_v2`. Approved B remains exactly **1.26e−5**; the longer derivation is explicitly explanatory. [RS-03 JSON:1115](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1115), [JSON:1180](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1180), [current A2 JSON:430](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:430).

   **Fix:** None for that typo; update substantive source drift under finding 2.

7. **NOTE — Earlier estimator, bias and population protections: PRESERVED.**

   SW-EST still maps every A2 outcome to either one complete certified B0 tuple or the default. A3 denominators remain sensitivity-only without separate preregistered certification. A2’s synthetic B does not become an RS-03 total bias allowance; `b_total` remains **미측정**, and claims remain conditional on common P. [RS-03 JSON:286](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:286), [JSON:306](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:306), [JSON:443](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:443).

   **Fix:** None beyond finding 2’s historical-population wording.

8. **NOTE — Earlier B1 inclusion and DEV-role fixes: PRESERVED.**

   The copied Stage-B rule matches the PREREG after formatting normalization: the designated primary candidate, both-split intersection-union, ΔNLL ≤ −0.003, strict CI upper < 0, unadjusted one-sided p ≤ .05, Brier upper ≤ .001, and ≥4/5 improving seeds. DEV fitting restrictions and exposed-development selection are distinguished correctly. [RS-03 JSON:185](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:185), [JSON:549](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:549), [P48 PREREG:157](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/docs/PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md:157).

   **Fix:** None; retain these predicates when finalizing the registration.

9. **NOTE — Earlier data, as-of and predictor-specification fixes: PRESERVED.**

   The 2026/2027 fitting and selection bans, postseason exclusion, and distinct ledger/snapshot/reconstructed-input claims remain explicit. Predictor scoring conditions on realized pitch type; policy recommendations exclude current-pitch realized information. The unfinished predictor specification must be completed and independently reviewed before sealing, or the layer is excluded. [RS-03 JSON:557](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:557), [JSON:565](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:565), [JSON:598](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:598), [JSON:957](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:957).

   **Required follow-through:** Complete that specification or record exclusion; draft review does not certify it.

10. **NOTE — Earlier multiplicity, interpretation and retry fixes: PRESERVED.**

    Contrast operands and the Holm family freeze at sealing; interval selection, exact ties and p construction remain explicit. `NO_DIFFERENCE_ESTABLISHED` does not assert equivalence. Both layers share one look; bounded infrastructure retries precede scientific exposure, and the first exposed attempt may finish but cannot be followed by another attempt. [RS-03 JSON:374](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:374), [JSON:433](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:433), [JSON:508](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:508), [JSON:775](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:775).

    **Fix:** None.

11. **NOTE — Requested arithmetic: VERIFIED; no arithmetic regression found.**

    These are **recalculations from documented inputs**, not new measurements or verification of original OPE outputs. Sources: [current A2 JSON:86](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:86), [JSON:119](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:119).

    | Quantity | Recomputed |
    |---|---:|
    | Current-rule MDE | 0.002741074785 |
    | Including b_V | 0.005475804328 |
    | Point MDE | 0.000230720926 |
    | After 30% point-SE reduction | 0.000161504648 |
    | Illustrative residual variance share | 88.63525% |
    | Current-rule MDE after reduction | 0.002720944893 |

    Using the **printed MDEs**, as the registration’s formula specifies:

    | δ | Current-SE games | Reduced-SE games |
    |---|---:|---:|
    | 0.001 | 129.3 | 63.4 |
    | 0.0003 | 1,436.5 | 703.9 |
    | 0.0002 | 3,232.2 | 1,583.8 |
    | 0.0001 | 12,928.9 | 6,335.4 |

    These reproduce the table. Every δ is below b_V, supporting its “no solution under this formula” entries. The small printed-versus-recomputed MDE difference is documented rounding. [RS-03 JSON:1237](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1237), [current A2 JSON:126](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:126).

    **Fix:** No numerical correction required. Preserve the planning-proxy, illustrative-decomposition and scenario qualifications; correct finding 2’s input description.

12. **NOTE — D186, sealed boundaries and eight open decisions: PRESERVED.**

    C1 remains the single named draft primary contrast; paired mean Δ with interval, separate censoring reporting, secondary D108 rule and WE reward are represented. D155 reopening remains an unanswered Song decision. D131, G15, G0 and `c13cc989…` remain protected, and service adoption remains separate. [RS-03 JSON:355](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:355), [JSON:404](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:404), [JSON:48](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:48), [JSON:974](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:974).

    All eight questions retain one recommendation each. Their answers, inclusion decisions and implementation must be finalized before exact-revision review and sealing. [Design:150](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:150), [JSON:880](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:880).

    **Required follow-through:** Address findings 1–2 and complete those existing gates. This verdict accepts the draft with notes; it does not authorize execution.

Reviewed SHA256: design `8a93d753…3d31474`; JSON `6efa5e47…26aa65b`. No files or Git state were modified; no prohibited data, original OPE outputs, result directories or T7 contents were opened.