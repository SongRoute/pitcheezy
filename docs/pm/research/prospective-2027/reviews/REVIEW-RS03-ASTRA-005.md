VERDICT: PASS_WITH_NOTES

1. **MINOR — Q4 overstates what snapshot replay proves.**

   Both ledger and snapshot-replay rows are described as proof of recording before first receipt at [RS-03 JSON:589](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:589), [JSON:1051](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1051) and [design:155](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:155). However, [RS-02:837](/Users/song/Projects/pitcheezy-pm-jobs/20261020-rs02/REGISTRATION-RS02-SEALED.json:837) explicitly disclaims prediction generation before receipt.

   **Fix:** Attribute pre-receipt recommendation-recording proof to **ledger rows only**. Snapshot replay supports historical input availability, agreement checks and secondary analysis. Synchronize the Q4 history text too. The underlying population definitions already distinguish these correctly at [JSON:568](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:568), so this is a reporting defect rather than an estimand change.

2. **NOTE — Decisions are settled; minor recommendation terminology remains.**

   Q1–Q9 each contain option 1, a decided value and D187 reference. Their substantive choices match [D187:327](/Users/song/Projects/pitcheezy/docs/decisions.md:327) and [design:152](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:152), subject to finding 1. Nevertheless, `why_recommended` remains at [JSON:1014](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1014), and `theta_mei_option` lacks a local historical label at [JSON:705](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:705).

   **Fix:** Rename these to `decision_rationale` and `theta_mei_history`, or explicitly mark them historical. Actual θ remains unambiguously zero at [JSON:417](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:417).

3. **NOTE — The A1 interpretation and Q9 addition are correct.**

   SW_A1 requires `STABLE_WITHIN_COST:<arm>` **AND** `CONFIRMED` for the same arm; unsuccessful, aborted or missing confirmation retains MC/no A_RB. [RS-03 JSON:270](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:270) agrees with registered A1’s [original labels:103](/Users/song/Projects/pitcheezy-worktrees/track-a-a1-1027/configs/EXP-STAB-02.yaml:103) and [confirmation rule:280](/Users/song/Projects/pitcheezy-worktrees/track-a-a1-1027/configs/EXP-STAB-02.yaml:280).

   Q9 correctly comes from D187, although absent from v4’s eight questions. P6 covers the total P-specific bias budget, belongs to research/A2 follow-up and remains 미착수. [JSON:1104](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1104), [JSON:962](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:962).

   **Fix:** None.

4. **NOTE — Both remaining round-4 findings are resolved.**

   Completion now has a named timestamp, defined clocks and durable public evidence; eligibility uses the maximum of the three recorded times. [JSON:869](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:869), [design:119](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:119).

   L2 now includes completed PAs plus `rho_zero` and `rho_zero_before_node`, consistent with [current A2:134](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:134). P remains distinct and its SE remains 미측정. [RS-03 JSON:614](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:614).

   **Fix:** None; preserve these distinctions during finalization.

5. **NOTE — Requested arithmetic reproduces; no numerical regression found.**

   Recalculation from the **recorded rounded SE inputs**, not original OPE outputs, gives current-rule MDE **0.00274108**, bias-adjusted **0.00547581**, and point MDE **0.000230711 → 0.000161497**. The illustrative residual variance share is **88.6364%**. These agree with the displayed rounding and approximation qualifications. [A2:125](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:125), [A2:134](/Users/song/Projects/pitcheezy/docs/pm/research/track-a/A2-ope-redesign-registration-draft.json:134), [RS-03 design:82](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-design.md:82).

   Using the printed MDEs in the registered sample-size formula reproduces [JSON:1276](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:1276):

   | δ | Current-SE games | Reduced-SE games |
   |---:|---:|---:|
   | 0.001 | 129.3 | 63.4 |
   | 0.0003 | 1,436.5 | 703.9 |
   | 0.0002 | 3,232.2 | 1,583.8 |
   | 0.0001 | 12,928.9 | 6,335.4 |

   **Fix:** None. Retain the planning-proxy and scenario labels; the 89% figure is not an identified variance decomposition.

6. **NOTE — Earlier inclusion, estimator and interpretation protections remain intact.**

   Arm C’s copied Stage-B predicates preserve the designated primary, both-split intersection, exact inequalities and unadjusted per-split p rule. [RS-03 JSON:187](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:187), [P48 PREREG:157](/Users/song/Projects/pitcheezy-worktrees/research-direct-confirm-1026/experiments/pitchmdp/docs/PREREG-EXP-P48-001-adoption-seed-confirmation-2026-10-26.md:157).

   SW_EST still imports one certified tuple or the default; A2’s synthetic allowance does not become RS-03’s total bias budget. C1 remains the single A−R primary contrast, with WE reward, separate censoring reporting and the D108 secondary row. [JSON:286](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:286), [JSON:357](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:357), [JSON:444](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:444).

   **Fix:** None beyond finding 1’s claim wording.

7. **NOTE — Data and sealed boundaries remain protected.**

   The 2026/2027 fitting and selection prohibitions, postseason exclusion and decided profile window remain explicit. [JSON:559](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:559), [JSON:605](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:605). The policy identifier, G0/G15 pins, D131 and conditional D155 boundary remain preserved; confirmation does not authorize service adoption. [JSON:50](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:50), [JSON:77](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:77), [JSON:122](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:122), [JSON:987](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:987).

   **Fix:** None.

8. **NOTE — This accepts the draft, not execution.**

   Status remains `DRAFT_UNREGISTERED`, with execution disabled. P1–P3 precede `FINAL_UNSEALED`; exact-revision review, digest-specific sign-off, completion evidence and post-seal checks remain required. No interim looks and the bounded infrastructure-retry rule remain explicit. [JSON:5](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:5), [JSON:753](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:753), [JSON:890](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:890), [JSON:898](/Users/song/Projects/pitcheezy/docs/pm/research/prospective-2027/RS-03-2027-prospective-multi-arm-registration-draft.json:898).

   **Required follow-through:** Correct finding 1 and complete the existing finalization gates.

Reviewed SHA256: design `36fefdff…2e3366`; JSON `10b652e3…cb5be`. No files or Git state were modified; no prohibited data, original OPE outputs, result directories or T7 contents were opened.