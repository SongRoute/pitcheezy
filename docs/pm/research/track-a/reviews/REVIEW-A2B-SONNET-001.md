VERDICT: FAIL

# REVIEW-A2B-SONNET-001 — independent design review, round 1: EXP-OPE-RES-02 (track A2b, A2 follow-up)

- **Reviewer:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), independent design reviewer, G-REV round 1. Read-only.
- **Author:** research-design worker (Claude Opus 5.5) under the Claude Fable 5.1 PM session (D191 (b)). Codex is unavailable until 2026-11-02 18:57 KST; the Codex re-confirmation remains a separate gate before G-CODE.
- **Date:** 2026-10-28.
- **Files under review** (main checkout `/Users/song/Projects/pitcheezy`, uncommitted drafts; sha256 at review time):
  - `docs/pm/research/track-a/A2b-ope-followup-design.md` — `38571874ad79a469caf60ba0e47f81498ce613178650538493904d0c0a9d5ede` (495 lines, 70,311 B)
  - `docs/pm/research/track-a/A2b-ope-followup-registration-draft.json` — `778bebf3eb68676a6021c953dd140caf23f93c8abf8e57479e337c556416a979` (2,143 lines, 107,599 B)
- **Read for context:** A2 design + draft; `results/EXP-OPE-RES-01.json` (summary, S0/S1/S2 summaries, S3_public, S4_decision_record, limits, registration, attempts); `docs/decisions.md:325–339` (D186, D187, D190, D191, D191 (b), D191 정정); A3 design §7.4/§9/§10 and `results/EXP-BC-CAL-01.json` (`final`, stage 04 manifest, stage 09 weights); RS-03 design + draft (SW-EST, SW-A3, bias_scope, sealing, P6); `docs/pm/contracts.md`; track A README; `configs/ML-POLICY-MATERIALIZATION-v1.json` (M-1/M-2, `le2025_validation_plan`, S6_V4); A2 worktree code (read-only).
- **Not done (by instruction):** no experiment, stage, fit or test run; no sealed ledger, `/Volumes/T7 Shield` path or 2026 file opened; no git state change. Computations done: `shasum`, seed derivation from the public integers (`.venv` numpy 2.5.3), arithmetic on recorded numbers.
- **Citation shorthand:** `D:n` = design line, `J:n` = draft line. A2 code paths are relative to `/Users/song/Projects/pitcheezy-worktrees/track-a-a2-1027/experiments/pitchmdp/`: `registry` = `pitchmdp/ope_redesign_registry.py`, `a2syn` = `pitchmdp/ope_synthetic_a2.py`, `sealed` = `pitchmdp/ope_sealed.py`, `decision` = `pitchmdp/ope_decision.py`, `runner` = `scripts/run_policy_validation.py`.

Counts: **0 BLOCKER, 2 MAJOR, 12 MINOR, 6 NOTE.** Both MAJORs are fixable in about a day (one data-free synthetic sizing job plus text and ordering changes); I expect PASS_WITH_NOTES at round 2.

## Hard constraints — all held

- **Draft state:** `DRAFT_UNREGISTERED`, `registered` false, `preregistration_complete` false, `execution.enabled` false (J:5–16). All 28 leaves of `registration_input_pins` and every `stage_output_seals` field are null (J:1888–1952); `registered_kst` and the three review references are null (J:21–24).
- **Fixed values:** B = 1.26e-5 exactly, SE ≤ B/2, X = 0.30 are carried unchanged (J:121–137). I found no threshold change hidden as a default. New judgment values are labelled and new: ω = 0.5, the KL target for θ, the t family, R = 2,000/300, N′, the C5–C8 relative rule for B0 tuples, the 2 % window order B0 < B2. N′ is sized with a B/4 target; that is a sizing choice, not an acceptance threshold.
- **Data and identity:** no 2026 row; date guards (J:773–777); sealed ledgers read-only; policy identity `c13cc989…` and protected paths untouched (J:97–116). One caveat on "imported unchanged" is MINOR-3.
- **Blinding and windows:** sealed evaluator, three extra reward-free allowlist types (J:1334–1357), operator-blind pooling of S-2′ runs; start ≥ 13:30 KST, hang guard ends before 06:30 KST, nothing inside 06:30–13:30 (J:1744–1755); RSS 8 GiB, free memory 4 GiB.
- **Gate order** matches the README PM decisions and D191 (b): G-REV (Sonnet, then Codex) → G-CODE → G-REG → G-SIGN → smoke → run.
- **ID:** `EXP-OPE-RES-02` has 0 hits in `docs/experiments.md`, `configs/`, `results/` (re-checked) and in any commit message (`git log -S`).

### Re-verified, no finding
- **Arithmetic:** 2κ_p replication 13.64→14 and 54.56→55; all PASS-probability tables (J:600–621 and J:634–653) reproduce; 18,400 = 8×2,000 + 8×300; 1,664.94 s = 18,400 / 22.103 × 2; 60 × 433.9164 = 26,034.98 s; 6 × 433.9164 = 2,603.5 s; totals 35,171.9 s (9.77 h), 15,645.7 s (4.35 h), 31,010.4 s (8.61 h); option 3 = 25,635.7 s with a 14.24 h guard, so 16:15 KST latest start; 87,759 / 75.67 ≈ 1,160 DEV games.
- **Records:** every number copied from `results/EXP-OPE-RES-01.json` (ratios, S2 cells, H_point, costs, κ_p/ζ_p calibration: 'other' n = 1,007, δ 0.01803, SE 0.01143 → 0.04043 against BR 0.00898; ζ 'other' n = 36 → 0.16996) and from `results/EXP-BC-CAL-01.json` (ESS 9,120→6,340 and 9,726→6,592; ρ q99 1.38→1.92, max 3.97→18.08; KL 0.048863 over 38,077 decisions; the three B2 artifact shas and sizes; inputs ≤ 2025-05-31) matches the current files.
- **Seeds:** all 9 role values and all 66 world seeds reproduce from `SeedSequence([20261028, …])`; no collision with the 15 A2 world seeds, the A2 role values or the ML-POLICY role values (including 4115122098).
- **MD ↔ JSON** agree on arms (1 + 11 + 12 = 24), cells (C1–C8 and 8 descriptive), thresholds, hang guards, stage list (SM, S0, L, S1, S2a, S2b, S3, S4, UB), labels, decisions 1–7 and costs.
- **Reused A2 pins** (`a76a48c2…`, `31e7bc3a…`, `8b7fa3aa…`, `d18949ac…`, decision input `ee7f14d7…`) equal `stages.*.manifest_sha256` and `S4_decision_record.inputs_sha256`. The identity block (`c13cc989…`, `b4f274eb…`, `152cdab4…`, `d0f1d452…`, `aa6c4e48…`) equals the A2 draft's.
- **Cited code lines hold:** a2syn:177, 263, 746, 874, 885, 1021; runner:80, 337–341, 783, 829, 835, 1230–1245, 1363, 1539; ML-POLICY config :69, :70, :428–444.

## Findings

### BLOCKER

None.

### MAJOR

**MAJOR-1 — TAB-R3: whether the B2 acceptance cells can be satisfied, and whether the stand-in means anything for B2, is unmeasured, judgment-driven and cheap to size.**

Evidence:
- **Cell rule** (D:204–207; J:1067–1098). In the t = 0 cells C1/C3/C4 the baseline bias is zero, so a B2 tuple (denominator D2, logging law D0) must meet U ≤ B with the whole D2 error inside its bias. The pure swap R1_0×R2_0×B2 is judged exactly (EXACT_PASS/EXACT_FAIL, SE = 0; D:136, J:1099). B0 tuples, in the mirror-image cells C5–C8 where B0 is wrong, only have to stay within |β₀| + B of the baseline. The incumbent is therefore favoured by construction. That is defensible, but it is the real decision rule of the B2 half, and decision 2 does not say so.
- **Size and alignment are two judgment values.** θ makes the mean KL(D2‖D0) equal the recorded 0.048863 (D:186). ω = 0.5 is "halfway" (D:185; J:963–977). ω ∈ {0, 1} appear only as descriptive cells with `decision_input` false (D:208; J:1100–1113), so worst-case alignment (ω = 1) is never an acceptance cell. The induced PA-ESS ratio is "reported next to" the recorded 0.695/0.678, not constrained (D:186). Mean-KL matching does not fix tails: B2's recorded ρ has q99 1.92 and max 18.08 against B0's 1.38 and 3.97 (`EXP-BC-CAL-01.json` stage 09 `weights.json`).
- **The deciding numbers are listed as unmeasured:** θ, the induced ESS ratio, β₀ and β₂ sizes (D:488; J:2101).
- **They are exact and cheap.** `TabWorld.__init__` already computes the baseline bias by exact dynamic programming (a2syn:331–336) with `_expected_dr` (a2syn:177), which takes the logging law and the denominator separately. The inputs are TRAIN marginals and the sealed A2-S0 outputs; no DEV, S6, reward or 2026 data is involved.
- **The only recorded measurement of what a wrong denominator costs in TAB** is A2 T-A2 (V3): baseline β₀ = 8.925e-6 = 0.71 B, and R1_b×R2_0 b̂ = 1.500e-5 > B (`results/EXP-OPE-RES-01.json` `S1_tab_summary.acceptance["T-A2"]`). That V3 error was not constructed to align with the model error; D2's is half-aligned.

Why it matters (inference, not a measurement): with a denominator error of mean KL ≈ 0.049 and half alignment, |β₂(t=0)| cannot be assumed below B. If it exceeds B, every B2 tuple fails C1/C3/C4 whatever is true of the real B2, and `DENOMINATOR_NOT_CERTIFIED` is decided by ω and θ. If instead it is far below B, the stand-in may be too mild to say anything about a model whose weights are about 30 % less efficient with an 18× tail. Song is asked (decisions 2, 6) to commit weeks of work before the one number that decides the B2 half is known.

Required fix:
1. PM-authorised, data-free sizing. Report |β₀| and |β₂| for t ∈ {0, 1, 2}, ω ∈ {0, 0.5, 1}, η ∈ {0, η_p}, κ = ζ = 0, plus the induced PA-ESS ratio and ρ q99/max. Fix ω = 0.5 and the KL target in the text first and mark the table "sizing, not a result". Do not tune ω afterwards. If the cells prove unsatisfiable, the response is to downgrade decision 2, not to change ω.
2. Register an adequacy condition for the stand-in (a band for the induced ESS ratio and ρ tail; outside it B2 tuples are not certifiable because the stand-in is too mild), or move ω = 1 into the acceptance family.
3. Reword the label meaning and decision 2 to "bias within B in the declared stand-in family (mean KL 0.0489, ω = 0.5)". Replace "the same level A2 required of its original pairs" (D:419; J:2013–2023): A2 had no t = 0 absolute cell for a wrong denominator, only the relative V3 cell, so the B2 standard is stricter.

**MAJOR-2 — The outcome space of the program is not stated. Six of the eleven candidates and most of the new machinery are reachable only if a B2 tuple first passes (ii), which the record argues against, and the stage order pays the expensive tail first.**

Evidence:
- **Only (ii) credits anything.** Selection is by H* (D:289–294). Bias reduction is never rewarded; the t ≥ 1 worlds only bound harm. S-5 asks a B2 tuple to agree with the B0 baseline within noise (D:226). The case for B2 is calibration (NLL −0.0949, ECE 0.035→0.009). The design says as much (D:21: accurate table ≠ finer scale) and at D:13 ("no reason to expect a smaller half-width, it may grow"), but decision 2 and the plan draw no consequence.
- **The record points to wider B2 intervals.** PA ESS falls to 0.695 (candidate) and 0.678 (reference) of B0's, mean ρ 0.959 against 0.991, mean W_T 0.876 against 0.968, tails heavier. By a 1/√ESS rule of thumb (arithmetic, not a measurement) intervals are about 1.20–1.21× wider, against an A2 best B0 ratio of 0.746/0.758.
- **Reachability.** `DENOMINATOR_NOT_CERTIFIED` (D:283; J:1232) and `ADOPT:<B2 tuple>` both need a B2 tuple with (ii) = PASS. S-2′ (7.2 h) matters only if an R1_b tuple passes (ii), and only R1_b×R2_1 is near; the pitcher-bootstrap ratio 0.7582 is the binding one, the next best R1_b×R2_0 is 0.828/0.836. Yet the plan runs S1, S2a, S2b (≈ 8.4 h of the 9.8 h; D:351, D:403) before the public, widths-only (ii) result of A2b-S3.
- **Reviewer's illustrative arithmetic** (SD of the ratio is unmeasured; the S6 pair was the best of twelve, so its ratio is selection-optimistic): with a true pitcher-bootstrap ratio of 0.758, P(ratio ≤ 0.70 on a fresh 150 games) is 0.2 % / 7 % / 17 % for a ratio SD of 0.02 / 0.04 / 0.06, and both bootstraps must pass.

Why it matters: Song is asked to approve a 9.8 h, new-code program (decisions 1, 2, 4, 6) whose most likely outcomes (`NO_PRECISION_GAIN` again, or `BIAS_NOT_CERTIFIED`) are never stated.

Required fix:
1. Add an "expected outcome space" paragraph to §0 and §11, labelled projection, and state that under this protocol B2 can only be adopted as a precision winner that also agrees with B0.
2. Gate or reorder, with all rules unchanged: run A2b-S3 (public widths only) before S2a/S2b and pre-register that S2a, S2b, the TAB-R3 B2 columns and the adapter work are executed only for tuples with (ii) = PASS. Base cost ≈ 1.4 h (L 3,808 s + S3 1,060.5 s) against 9.8 h. If the author prefers the current order, give the reason.
3. Decision 1 may add the variant "S6 pilot (option 1) first, option 2 only if a B2 tuple or R1_b×R2_1 looks within reach".

### MINOR

**MINOR-1 — Decision 3 (S-2′ size) is framed on a best case and a noise anchor.**
- "PASS ≈ 96 % if the true bias is 0" is the best case. The "A2 point estimate" scenario (9 %) anchors on b̂ = −1.06e-5 ± 2.33e-5 (z = −0.46, 95 % interval [−5.6e-5, +3.5e-5]); that is a noise draw, not an estimate of the true bias.
- The 2κ_p world tilts the hit-class weights of all five pitch groups by e^{±0.58} (signs per `S0_summary.kappa_calibration.S5X.signs`), although the large groups' TRAIN residuals are ≤ 0.7 pp (FB −0.0009, BR +0.0062); κ is set by the noisy 'other' group (n = 1,007). A true bias of order B is plausible.
- The only existing dose-response information, the A2 descriptive runs at κ_p, 5κ_p, ζ_p, 5ζ_p (k = 9–12; sealed `s5x.json`, sha `ee7f14d7…`, synthetic and not blinded), was not consulted.
- Fix: replace the two-scenario sentence with a table for N′ and b (from SE₁ = 2.3267e-5, same normal approximation):

  | N′ (hours at 433.9 s) | b = 0 | 0.25 B | 0.5 B | 0.75 B |
  |---|---|---|---|---|
  | 14 (1.7 h) | 0.05 | 0.05 | 0.03 | 0.02 |
  | 25 (3.0 h) | 0.55 | 0.45 | 0.25 | 0.10 |
  | 36 (4.3 h) | 0.80 | 0.67 | 0.37 | 0.13 |
  | 55 (6.6 h) | 0.96 | 0.85 | 0.52 | 0.17 |

  80 % needs 36 runs at b = 0, 48 at 0.25 B, 108 (13 h) at 0.5 B. Consult the dose-response runs before fixing N′. The 2ζ_p side is sound: z = 2.29 means a positive bias of about 7e-6 is likely real, N′ = 5 gives 96 % at that value and 45 % at 1.0e-5.

**MINOR-2 — `sources_read_sha256` is stale for 5 of 19 entries; the results file pin is no longer recoverable.**
- Recomputed with `shasum -a 256`. 14 match: A2 design, A2 draft, A3 design, `EXP-BC-CAL-01.json`, `contracts.md`, RS-03 design, ML-POLICY v1 and addenda 1/8, `results-all.md`, the poster review, and the three A2 code files.
- 5 differ, and all carry mtime 2026-10-28 00:36:47, one PM batch after the 00:32 draft: `README.md` f324e559… → ef9c1750…; `results/EXP-OPE-RES-01.json` 7c437ac4… → 5caa94ec…; `docs/decisions.md` e69e7754… → 436dff73… (D191 정정 at :339); RS-03 draft 3759008f… → 8ec7d1b0…; `docs/experiments.md` 1c941372… → 2d57bd18….
- I re-read the current results file; the quoted numbers still match (the change is the correction note in `summary.best_pair_descriptive.note`).
- `results/EXP-OPE-RES-01.json` is untracked (`??`), so the pinned bytes cannot be recovered from git, and `identities.exp_ope_res_01.results_file_sha256` (J:1883) is stale.
- Fix: refresh the five pins or label them "as read 00:32". Commit the results file before G-REG and re-pin. The RS-03 draft already carries the corrected 2ζ_p wording; RS-03 design `:47` still says "precision requirement not met" for that cell (PM).

**MINOR-3 — "A2 modules are imported unchanged" (J:114; D:315–318) cannot hold, and the new-code list is incomplete.**
- `sealed.rehearsal_evaluate` is wired to the 12 original-denominator pairs (sealed:27–29, 231, 275), A2's bootstrap seeds (sealed:100, 214–216) and the strings `EXP-OPE-RES-01`/`A2-S3` (sealed:228).
- `decision.decide` hard-codes `TOTAL_ORDER`, `TAB_ACCEPTANCE` and the five labels (decision:15, 85, 101, 107).
- `registry` hard-codes `A2_BASE` 20261027 (:245), `ROLES`/`ROLE_VALUES` (:246–247), `world_seed(k)` for k ≤ 15 (:271–274), `S5X_RUNS` (:278), `TAB_ACCEPTANCE` (:303), `TAB_DATASET_PAS` 10,131 (:310), `STAGES` (:350), `I_PAIR` and `PUBLIC_SCHEMA` (:436, :455). `TabWorld` draws its fixed fields from `role_sequence`, which uses `A2_BASE` (a2syn:273–274).
- The runtime assembly the design calls "reuse" (D:256) is three closures inside `dispatch` (`bind` runner:1211, `snapshot_for` :1219, `candidate` :1230): they must be re-implemented, together with `prerequisites` and the style-snapshot as-of rule.
- The new-file list (D:319–325) has no evaluator for 24 tuples with B2 ρ recomputation, no A2b decision/label module, no A2b registry or public schema.
- Fix: list these as new or copied modules, change the J:114 wording to "no existing file is modified; A2 logic is wrapped or copied", and restate the G-CODE scope. The schedule is not at risk: the README shows A2 went from design review to run in one day, and the plan allows 1–2 weeks.

**MINOR-4 — The new DEV selection is not fully specified, and `selection_salt` names two different values.**
- `role_seed` (runner:337–341) accepts only the five ML-POLICY roles (runner:80) on base 20260929. The A2b "new role `selection_salt`" is `SeedSequence([20261028, 6])` = 759,396,736. The ML-POLICY `selection_salt` is 2,021,242,497 (recomputed). D:255 and J:720 use the same name for both in one sentence: the S6 recomputation call uses the old role, the new selection the new value.
- `select_games` (runner:835–853) takes no exclusion set. "S6 excluded" means either filtering the frame first, so month quotas are computed on DEV∖S6∖straddlers, or post-filtering. The text does not say which.
- Fix: rename (for example `a2b_selection_salt` and `s6_selection_salt_recomputed`), register the exact call and the per-month quota table, and pin `selection_s6_game_list_sha256` and the new game list hash.

**MINOR-5 — TAB-R3 world identity and η_p.** C1–C4 are declared "A2 T-A1…T-A4 world" (D:194–197; J:998, 1007, 1016, 1025), and η_p, κ_p, ζ_p are pinned from A2 (J:889). But the streams rule (D:381; J:1494) draws the fixed components ([0, component]) under the new base, which changes situations, perturbations, the ε field and the planning/q̂ noise, and therefore the η_p that gives ECE = 2×0.0087 (η_p is calibrated per world, a2syn:406). Either keep A2's fixed draws (new base for replications only) or recalibrate η_p in A2b-S0 and drop the "same world" claim. Say which.

**MINOR-6 — The B2 adapter consistency check cannot test the new code.** At the 147 real E0 start states there is no within-PA history, count change or workload advance (D:218; J:1139), so only first-pitch features are checked. Replace it with a reward-free replay of the real pitch sequences of those PAs (all pitches, S5's two games) through the adapter against A3's batch path at 1e-12.

**MINOR-7 — A2b-L "no reward" and its hang guard.**
- The M-10 paired-identity run (J:732–738; D:259) consumes PA-end rewards internally (`paired_identity_run`, runner:736–750). State that A2b-L touches rewards internally with pass/fail the only public output, or move M-10 into the sealed evaluator. A reward-free ledger itself is implementable: `submit_pas(..., outcome=None)` exists and is what `run_v5` uses (runner:642, 678–680).
- The guard 7,616 s (J:1697) is 2× the 3,808 s record, but the registered S6_V4 guard was 14,400 s, and A2-S2 shows wall/CPU = 6,508.7/3,060.5 = 2.13 under load. Use 14,400 s, which still fits the window, or justify 7,616 s.

**MINOR-8 — `registration_input_pins` lacks what A2b-L needs (J:1888–1937).** Missing: the ML-POLICY D127 chain (config and addenda 1–8 shas, member-loader pin), the τ-freeze record and setting (samples, pitch_cap, `row_budget` 9,000,000, planning seed, evaluation role), the DEV style snapshot (path, sha, as-of), the census pin used by `census_pas`, the hand registry, the G0 bundle, and the lightgbm 4.6.0 / pyproject sha pin that A3 recorded for B2 inference.

**MINOR-9 — Labels and conditions are not defined per `r3_status`; S-5 is classed as non-denominator.**
- `conditions_by_tuple.B2_tuples` lists S1D/S4D unconditionally (J:1181–1188). Nothing says what changes for `ADOPTION_ELIGIBLE_TAB_ONLY` (drop S1D/S4D) or `DIAGNOSTIC_ONLY:*` (which label if a B2 tuple passes (ii)?). Label 4 counts FAIL/INCONCLUSIVE/invalid but not NOT_RUN (J:1233).
- For B2 tuples the S-5 shift contains the denominator swap (D:226), so the most likely B2 failure (a real shift) lands in `BIAS_NOT_CERTIFIED:REAL_SHIFT` rather than the new label. Either file S-5 under the denominator conditions for B2 tuples or say that the new label is reachable only when S-5 passes.

**MINOR-10 — Seeds and order details.**
- G-INT-A2 reproduces A2's resamples on S6 and must use A2's 3060201743/3624346705 even under options 2/3, but J:1495–1505 gives only the new seeds for those options.
- The registered order is S1 → S2a → S2b (J:19) while the schedule runs S2b before S2a (D:403). State that the order of S2a/S2b is free, or fix one.
- N′ means two counts (per cell 5/55 and total 60) in the cost gate (J:1740).

**MINOR-11 — Wording that over-reads the new sample.** "Confirmatory-style" (J:756) and "확인형" (D:268, D:413) next to "not independent validation" (D:87, D:457; contracts.md:7). Use "pre-registered re-test on unseen games". Also disclose that R2-2/R2-3 were demoted to descriptive because of S6 results, so the candidate set was narrowed after the unblinding.

**MINOR-12 — Decision-item text gaps.**
- Decision 1 ③ shows only the ledger (7.1 h); the TAB-R3 and rehearsal scale with data (about 7×: roughly 3.2 h and 2.1 h: my multiplication by 7 of the TAB-R3 projection 1,664.94 s (J:686–692) and the rehearsal projection 1,060.5 s (J:719–730); the draft only says "scale with the data size (about 7x), unmeasured", J:740–753), and RSS is unmeasured. There is no intermediate size such as 300 games.
- Decision 6 ③ omits the consequence: the certification must finish before the RS-03 seal (RS-03 `a3_rule`), otherwise SW-A3 stays OFF.
- Decision 7 ① omits that without a pre-seal budget certificate 2027 has no CONFIRMED labels (D187 ②; RS-03 design :142) and gives no start date for the "separate design". D187 ⑨ literally assigns P6 to "A2 follow-up".

### NOTE

- **N1 — selection optimism in (ii).** λ is chosen on the same ledger whose half-width enters (ii), and the bootstrap holds it fixed. On a fresh ledger, add a descriptive R1_b×R2_1×B0 row at λ = 100 chosen on S6, a selection-free ratio at no cost.
- **N2 — large-group descriptive cell.** The refusal to re-declare the κ_p/ζ_p calibration (decision 3 option 3) is right as a rule (it would shrink κ_p 4.5× and SE with it, making PASS near-certain). Consider a non-decisional descriptive 2κ cell calibrated on the large groups only, so Song sees the dose-response before any future registration.
- **N3 — `EXACT_PASS/EXACT_FAIL` is a rule change.** A2's `cell_outcome` turns SE = 0 into INCONCLUSIVE unless the registered invariant allows it (registry:169, 182–184). The change is justified (the exact DP value has no MC error) but is new; apply it symmetrically to B0 and B2 tuples and register it as such (D:136 presents it as part of the generalization).
- **N4 — cost caveats.** 433.9 s per run is a wall mean including logs, and the wall/CPU gap is 2.13. A truth-only run may be much cheaper (logs also call the planner via `runtime.submit`). The 2× factor for 24 tuples in TAB-R3 is an assumption. Both are flagged unmeasured; the cost gate covers S-2′.
- **N5 — M-2 reading is sound.** M-2 allows addenda with SHA pins only, no samples (ML-POLICY config :70), so the new sample must be registered inside EXP-OPE-RES-02, as the design says. S5's two games are the top-hash games of their months and therefore lie inside S6 (runner:1539, 835–853), as D:255 derives.
- **N6 — exposure ledger.** S4 v5-a1 processed all 87,759 DEV PAs reward-free (`results-all.md:1479`; `run_v5` takes no outcome), and S5 rewards are synthetic, so "no OPE reward on non-S6 DEV" is plausible. It stays unverified; check it, and the overlap with EXP-STAB-01/02 games, at G-REG as the design says (D:489).

## Answers to the five review questions

**1. Scientific validity.**
- **Well-posed with pre-fixed rules:** yes. The question (J:145), thresholds, cell rules, N′, label precedence, tie order, attempts rule and "no outcome-driven retry" are fixed before any run, and no hidden threshold change was found.
- **Exposure:** honest on the S6 side: the 12 seen pairs are exploratory, B2 tuples carry `EXPOSED_S6_NEW_ARM`, B2's DEV-NLL selection is flagged, and the new sample is "not independent validation" (D:87, D:457). Remaining issues are MINOR-11 and N1/N6.
- **B2 certification route:** sound as a protocol only in TAB-R3, where the pure swap bias of a stand-in is exact. S5-X-R3 cannot certify at B (gap SE 4.2–5.3e-4 is 66–85× B/2), and its B2 nets are exact-denominator worlds, so they test wiring, not misspecification. The real-ledger gates are an agreement test (S-5) and a trivial ESS gate (E5: ≥ 6,340 PA and 143 games against 100/30). Certification therefore rests on TAB-R3, whose meaning is conditional on ω, θ and the t family (MAJOR-1).
- **What the design assumes about B2 features in synthetic states:** the real PA-start context stays valid across the synthetic PA; synthetic count and within-PA history tokens, a workload advanced by the pitch count and the G0 context channels computed from synthetic PA states reproduce A3's features; no steals or wild pitches change runners. This is flagged unverified (D:218, D:484), and the check as written covers only the first pitch (MINOR-6).
- **R1 re-certification arithmetic:** correct (N′ 14 and 55; PASS tables). 2ζ_p: N′ = 5 gives 96 % only if b ≈ 7.26e-6 (z = 2.29, so a positive bias is likely real), 45 % at 1.0e-5, and a bias at or above B cannot pass. The 2ζ_p cell is correctly treated as interval-limited, not precision-limited.
- **'Other' driver:** correctly identified and numerically right, and the 2κ world is a stress far beyond the large groups' residuals. Not re-declaring is correct; see MINOR-1 and N2.

**2. Registration completeness and consistency.** The structure is complete and the MD and JSON agree. Gaps: MINOR-2 (stale pins), MINOR-4 and MINOR-8 (selection spec, pin list), MINOR-9 (per-`r3_status` rules, S-5 class), MINOR-10 (seeds under options 2/3, order). RS-03 mapping is given (D:297–303; J:1300–1327): `ADOPT:<B0 tuple>` → SW-EST ON, SW-A3 OFF; `ADOPT:<B2 tuple>` → both ON with `pins.pi_b_hat_sha256` = the three B2 artifacts; all other labels → default tuple. It matches RS-03's `a3_rule` and `mapping_exhaustive` once the PM adds the EXP-OPE-RES-02 rows.

**3. Feasibility and cost honesty.**
- **Hours:** 9.8 h, 7.2 h and 1.1 h are derived with explicit arithmetic from recorded costs (checked above). The projections are labelled and the unmeasured parts named.
- **New ledger stage:** registrable inside EXP-OPE-RES-02 without touching protected sources, and a reward-free ledger is implementable (MINOR-7); M-2 cannot carry it (N5). It needs MINOR-3 and MINOR-4 settled.
- **TAB-R3:** implementable (a2syn:177, 263, 331–336); the S5X-R3 adapter is unverified, as the design says.
- **Schedule:** respects Sonnet now, Codex before G-CODE, windows, and the RS-03 seal (opening day minus 14 days, opening day unknown). It is conservative: A2 went from design review to run within one day (README).
- **Understated:** the new-code scope (MINOR-3) and the cost of paying the tail before the first (ii) result (MAJOR-2).

**4. Decision items for Song.** Options are exclusive and broadly complete. The recommendations mostly rest on sourced facts. No recommendation claims B2 narrows the interval, but decision 2 ① does not face the record that says it probably will not.
- **1:** keep ② only with MAJOR-2's outcome space and the gated variant; drop "confirmatory"; complete ③'s cost.
- **2:** the main re-word. Show the sizing (MAJOR-1) and the outcome space (MAJOR-2); drop "the same level as A2 required". If the sizing shows |β₂(t=0)| near or above B, the rational recommendation becomes ③ (diagnostic only) or a gated ①.
- **3:** replace the two-scenario sentence with the table (MINOR-1); consider N′ = 36 (80 % at b = 0, 4.3 h) as an intermediate option.
- **4:** acceptable as written. Note ① adds six B2 tuples whose work is gated by MAJOR-2.
- **5:** acceptable; add that (ii) must hold under both bootstraps and that the pitcher bootstrap (0.758) binds.
- **6:** add ③'s consequence (MINOR-12).
- **7:** add the RS-03 consequence and a start date (MINOR-12).

**5. Clarity for the PM and Song.** §0 and §11 are readable. Fix before Song decides: the "A2 point estimate" wording; "OPE_FIRST_USE_DEV … adoptable" for B0 candidates narrowed after S6; the fact that B0 tuples now face new, stricter cells (the design says so at D:207 but §0 does not); the same name for two salts; and the missing "what outcome is likely" paragraph. For phone reading the option texts are long; keep one line per option plus a reason.

## Verdict

VERDICT: FAIL

Round 1 findings: 0 BLOCKER, 2 MAJOR, 12 MINOR, 6 NOTE. Must be fixed before Song is asked to decide:
- **MAJOR-1:** size the TAB-R3 acceptance cells data-free (β₀, β₂, induced ESS ratio and tail for ω ∈ {0, 0.5, 1}), register a stand-in adequacy condition or an ω = 1 acceptance cell, and reword decision 2 and the label meaning.
- **MAJOR-2:** state the outcome space (B2 can only win as a precision winner; ESS −30 %; most likely `NO_PRECISION_GAIN`), and gate or reorder the stages so the S2a/S2b/TAB-R3-B2/adapter tail is paid only for tuples with (ii) = PASS.

If the PM decides Song should be asked before round 2, decisions 2 and 3 should carry these two findings verbatim. The design is otherwise careful: honest projections, consistent arithmetic, correct seeds, and the hard constraints hold. Codex re-confirmation (after 2026-11-02 18:57 KST) and G-CODE remain required.
