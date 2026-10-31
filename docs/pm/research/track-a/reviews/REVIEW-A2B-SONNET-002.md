VERDICT: PASS_WITH_NOTES

# REVIEW-A2B-SONNET-002 — independent design review, round 2: EXP-OPE-RES-02 (track A2b) v2.1

- **Reviewer:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), independent design reviewer, G-REV round 2. Read-only.
- **Author:** research-design worker (Claude Opus 5.5) under the Claude Fable 5.1 PM session. Codex is unavailable until 2026-11-02 18:57 KST; the Codex re-confirmation remains a separate gate before G-CODE.
- **Date:** 2026-10-28. Round 1: [REVIEW-A2B-SONNET-001](REVIEW-A2B-SONNET-001.md) (FAIL: 0 BLOCKER, 2 MAJOR, 12 MINOR, 6 NOTE; sha256 `fef363df…`, unchanged since the author pinned it).
- **Files under review** (main checkout `/Users/song/Projects/pitcheezy`, uncommitted; sha256 at review time):
  - `docs/pm/research/track-a/A2b-ope-followup-design.md` — `1593d3d3ac203cea24682a9fdca08719b33bc7056d529b8e0ae0ab64b9d569dd` (530 lines, 82,894 B)
  - `docs/pm/research/track-a/A2b-ope-followup-registration-draft.json` — `24e796a03417b20db79b41fb3cce7ef566a2834eeffea01d089864a3d970301f` (2,846 lines, 149,911 B)
  - `docs/pm/research/track-a/A2b-sizing/` — `README.md` `fcd2e699c4513ddab95033d61a777527a33eba3b50901cec54c401719f6ee66d`; `a2b_tab_r3_sizing.py` `f1cf66c0b4f3d224cd0a821cea25cdac3fac9327219e123c8cb92521bacc938a` (315 lines); `a2b_tab_r3_sizing_output.json` `924be5a622dff2f853d70344ba26fef6286b120f14875c967238b3b45bcdbe13` (90 rows)
- **Also read:** `results/EXP-OPE-RES-01.json` (now `069bcd6b7337f24b5278004071086483beabc5bfdbb7ebfedf91cc47f52e2d7b`; `S2_s5x_summary.dose_response_s2_by_run`, `corrections_and_additions`, `S4_decision_record`, `digest_manifest`); `results/EXP-BC-CAL-01.json`; RS-03 design (now `f03d26b5…`, line 47 corrected) and draft (`8ec7d1b0…`); `docs/decisions.md:325–339`; ML-POLICY config and addenda 1–8; A2 worktree code `ope_synthetic_a2.py`, `ope_redesign_registry.py`, `ope_sealed.py`, `ope_decision.py`, `run_policy_validation.py`, `tests/ope_a2_fixtures.py` (read-only).
- **Not done (by instruction):** I did not re-run the sizing script or any experiment, stage, fit or test; no sealed ledger, `/Volumes/T7 Shield` path or 2026 file was opened; no git state change. Computations done: `shasum`, seed derivation from public integers (`.venv` numpy 2.5.3), arithmetic on recorded numbers, and re-aggregation of the sizing script's own output file. The A2 worktree is clean (`git status` empty, no `__pycache__`), so the author's sizing run left it untouched.
- **Citation shorthand:** `D:n` = design line, `J:n` = draft line, `S:` = sizing folder, `a2syn` = `pitchmdp/ope_synthetic_a2.py` in `/Users/song/Projects/pitcheezy-worktrees/track-a-a2-1027/experiments/pitchmdp/`.

Counts: **0 BLOCKER, 0 MAJOR left.** All 20 round-1 findings are CLOSED. New in round 2: 4 MINOR, 6 NOTE. The MINORs should be folded into a v2.2 before G-CODE; none changes a recommendation or what Song is asked to decide.

## Hard constraints — all held

- **Draft state:** `DRAFT_UNREGISTERED`, `registered` false, `preregistration_complete` false, `execution.enabled` false (J:5–16). All 46 pin leaves and every `stage_output_seals` field are null (J:2432–2529); `registered_kst` and the four review references (incl. `code_review_phase2_reference`) are null (J:21–24, J:58).
- **Fixed values:** B = 1.26e-5 exactly, SE ≤ B/2 and X = 0.30 unchanged (J:136–152). New judgment values are labelled: adequacy band, TAB reuse ±20 %, N′_ζ 10, hang guard 14,400 s, the κ_p′ descriptive cell, the push-to-③ rule.
- **Data and identity:** no 2026 row; sealed ledgers read-only; policy identity `c13cc989…` and the six protected paths untouched (J:113–131); the sizing reads no real row (S:README data statement; the fixture `tests/ope_a2_fixtures.py::synthetic_tab_marginals` is rng-generated Dirichlet/toy data).
- **Blinding and windows:** the gate A2b-G reads only the public file (D:292–298, J:1812–1818, J:1883); windows and caps unchanged (J:2276–2287, J:2216–2269).
- **ID:** `EXP-OPE-RES-02` still has 0 hits in `docs/experiments.md`, `configs/`, `results/`, and 0 commit messages.

## 1. Round-1 findings — status

**MAJOR-1 (TAB-R3 stand-in satisfiability and adequacy) — CLOSED.** Required fix 1 delivered: a data-free sizing over 5 synthetic worlds × KL ×0.5/1/2 × ω 0/0.5/1 × η 0/η_p × laws t0, t0.5, t1, t2, V3 with exact β₀/β₂, induced PA-ESS ratio, ρ q99 and max (D:202–211; S:). ω = 0.5 and the KL target were in v1 and are unchanged. Fix 2 delivered: a registered adequacy band and `r3_status` downgrades `STAND_IN_OUTSIDE_BAND` / `SWAP_FAILS_C1`, both settled at A2b-S0 before any rehearsal output (D:213–219; J:1577–1598). Fix 3 delivered: "B2 faces a stricter absolute t = 0 cell, which A2 never had" replaces "same level as A2" (D:224, D:454; J:1603, J:2592). The substance changed, not only the wording: the recommendation for decision 2 moved to ③ exactly as the sizing result warranted (D:457). Residual notes: R2-MINOR-1/2/3 below.

**MAJOR-2 (outcome space, stage order) — CLOSED.** §0.1 states the outcome space as a projection, including that bias reduction earns nothing and B2 can only win as a precision winner (D:19–25; J:921–932). The order is now phase 1 (SM, S0, L, S3) → gate record A2b-G → phase 2 only for tuples with (ii) ∧ S-5 (D:288–311; J:1812–1818, J:2061–2152). Phase 1 ≈ 1.35 h (3,808 + 1,060.5 = 4,868.5 s), maximum 9.89 h (J:952), most likely 1.35 h. The S6-pilot variant is listed (D:284).

**MINOR-1 (decision 3 framing) — CLOSED.** The "A2 point estimate" scenario is relabelled a noise draw (z = −0.46; D:56, J:250–251), the recorded dose cells are used (D:153–160), the N′ × bias table is present (D:164–173, J:673–724), and N′_ζ moved 5 → 10 on the recorded evidence. Arithmetic checked in §2(c).

**MINOR-2 (stale pins) — CLOSED for the author.** 29 of 30 `sources_read_sha256` entries match the working tree (§2(e)); the only difference is the PM's own `README.md` edit at 02:03:58, after the draft. The untracked results file now pins its current bytes (`069bcd6b…`, J:2426) and G-REG requires committing it first (D:426; J:2326). Carry-over for the PM, see R2-NOTE-3.

**MINOR-3 (code reuse) — CLOSED.** D:356–380 and J:158, J:2758–2799 now say that only part of the A2 code is importable and list the new modules (registry, 25-row sealed evaluator, decision module, denominator module, ledger, synthetic drivers, runner). I re-verified the cited hard-wiring: `tab_replication` binds `role_sequence('tab_world', 1, …)` to the A2 base at a2syn:606–613.

**MINOR-4 (selection spec, salt names) — CLOSED.** Two names (`s6_selection_salt_recomputed` 2,021,242,497 and `a2b_selection_salt` 759,396,736, both recomputed by me) and a four-step procedure: S6 list recomputed, frame filtered, `select_games(filtered, 'dev', 150, salt)` so quotas are over DEV∖S6∖straddlers, lists and quota table pinned at G-REG (D:263–272; J:983–1016). This matches the code of `select_games`.

**MINOR-5 (TAB world identity) — CLOSED.** Fixed draws stay on the A2 base, so C1–C4 are the A2 worlds and η_p 0.16406 is reused; the A2b base serves replication streams and the D2 ε field only (D:196, D:251; J:1261, J:2059).

**MINOR-6 (adapter check) — CLOSED.** Reward-free replay of every real pitch of the S5 PAs against A3's batch path at 1e-12 (D:230; J:1619); applies to decision 2 ① only.

**MINOR-7 (A2b-L reward use, guard) — CLOSED.** M-10 binds PA-end facts inside A2b-L with pass/fail and counts as the only public output; hang guard 14,400 s with start window 13:30–02:30 (D:384–387; J:2225–2229).

**MINOR-8 (pins) — CLOSED.** `ml_policy_d127_chain`, `runtime_setting`, `dev_style_snapshot`, `g0_bundle_sha256`, `b2_inference_environment`, `selection` added (J:2483–2510) with expected values (J:2800–2845). I checked them against the addenda: census `18ab73e6…` (addendum 1), bc/support/hands/materialize/style_dev (2), bind identity (3), profile (4), tau_freeze (5), v5 (6), v2 (7), s6_dr (8), style as-of 2025-07-01, row_budget 9,000,000, S6 guard 14,400.

**MINOR-9 (labels per `r3_status`, S-5 class) — CLOSED.** Conditions are listed per `r3_status` (D:239–247; J:1666–1677), S-5 is a denominator condition for B2 tuples (D:236, J:1646), `NOT_RUN` is counted (D:326, J:1742), and label 4 is stated unreachable under ③ (D:326, J:1741). One wording residue is R2-NOTE-2.

**MINOR-10 (seeds, order, N′) — CLOSED.** G-INT-A2 always uses 3060201743/3624346705 (D:401; J:2034–2039); phase-2 order is free (D:300); N′_ζ and N′_κ are separate (J:1251, J:2272).

**MINOR-11 (wording) — CLOSED.** "Confirmatory" appears only as a negation (D:94, J:961); the post-unblinding narrowing of the candidate set is disclosed (D:94; J:967).

**MINOR-12 (decision text gaps) — CLOSED.** Decision 1 ③ now carries ledger 7.1 h + rehearsal 2.1 h + TAB 0.7 h (D:279, D:449), the 300-game size is mentioned (D:282), decision 6 ③ states the consequence (D:480), decision 7 ① states the CONFIRMED consequence and a start date ≤ 2026-12-01 (D:484).

**NOTES N1–N6 — CLOSED.** N1 λ = 100 fixed row (D:96, D:119; J:1136); N2 descriptive κ_p′ cell, 5 runs (D:177; J:1253–1257); N3 `EXACT_PASS/EXACT_FAIL` registered as a new symmetric rule (D:139; J:1404); N4 cost caveats (D:69, D:526); N5 agreed; N6 exposure-ledger check at G-REG (D:429; J:2341).

## 2. The five checks requested

**(a) Closed, not merely reworded.** Yes for all 20. The two MAJORs changed the plan itself (new sizing evidence, a changed recommendation, a re-ordered stage list with a gate record). The MINORs changed registered content (pins, procedure, rules, seeds), not only prose. No round-1 item remains OPEN.

**(b) Sizing script and the disclosed move of the adequacy evaluation point.**
- *Correctness (static reading plus re-aggregation of the output).* The script builds D2 as defined (`tilt`, ξ = √(1−ω²)ε + ω z̄, θ by bisection on the D0-visitation-weighted mean KL; max relative KL error 1.3e-5, which is the 5-digit rounding of the reported value). It calls the A2 `_expected_dr` with logging law and denominator separately, the same call `TabWorld.__init__` uses for β₀ (a2syn:331–336). The ESS recursion mirrors the count-lattice transitions of `_solve`/`_visitation` (foul loop handled, divergence returns null). The tail uses visits under the law × the law. Strong evidence of correctness: in all 90 rows the identities hold to machine precision, |β(D0 denominator, t0)| ≤ 8.5e-18 and |β(D2 denominator, t1)| ≤ 1.6e-17. Every headline number in D:202–211, J:1431–1576 and S:README reproduces from the 90 output rows (|β₂(t0)|/B 0.197–2.507, 3/5 within B; ω 0: 0.195–2.444, 2/5; ω 1: 0.632–2.483, 1/5; KL ×0.5: 0.127–1.540, 4/5; KL ×2: 0.653–9.313, 1/5; t1 |β₀|/B 0.324–2.489; t2 0.392–4.18 and |β₂| 0.039–0.885, C6 5/5; V3 0.032–2.404, C2 4/5; ESS ratios t0 0.255–0.545, t1 0.682–0.702, t2 0.753–0.780; q99 ratios 1.754–2.240, 1.476–1.918, 1.241–1.496). Output provenance hashes equal the files. I did not re-run the script.
- *The move t = 0 → t = 1.* Disclosed (S:README, J:1591–1594, D:216). The band numbers come from recorded values only (loss 0.322, excess 0.392) and were not changed; the registered rule is evaluated on the registered world at A2b-S0 before any real-data output, so the toy-world trial cannot tune it. **It does not change what the sizing says or the recommendation**: C1 alone fails in 2 of 5 worlds (seeds 3 and 4: 2.507 B and 1.572 B), independent of adequacy. But the adequacy count is evaluation-point dependent (0/5, 3/5, 5/5 in band at t0, t1, t2; joint with C1 0/5, 1/5, 3/5), and the Song-facing reason leans on the move-dependent clause. See R2-MINOR-1.

**(c) Decision-3 arithmetic — verified.** From the six recorded cells in `dose_response_s2_by_run`: κ slope −1.2972e-6 ± 5.779e-6 per κ_p (χ² 0.242, df 2), so b(2κ_p) = −0.206 B ± 0.917 B; ζ slope +6.553e-7 ± 6.466e-7 (χ² 5.806, p 0.055), so the linear value is 0.104 B ± 0.103 B and the cell's own 0.576 B ± 0.252 B is the primary scenario. Predictive PASS for 2κ_p at N′ 14/25/36/48/55/108: 2.5/21.5/31.2/37.5/40.1/50.4 %; conditional at b = 0.21 B: 4.8/47.9/70.7/84.1/88.9/99.4 %; b = 0: 5.3/54.5/80.3/92.7/96.0/100 %. For 2ζ_p at N′ 1/2/5/10/20: at 0.58 B 39/66/96/100/100 %, at 0.79 B 13/21/45/74/96 %, predictive 42/60/77/85/89 %. Required SE for ζ is (B − 0.576 B)/1.96 = 2.72e-6, which takes 1.356 runs' worth of precision ((3.171e-6 / 2.723e-6)²). FAIL probabilities at b = 1.4e-5 (written "1.11 B"): 6.4/9.1/16.5/28.6 % for N′ 1/2/5/10, matching the stated 6/9/17/29 %. P(b > B) = 4.6 %. Joint 0.339/0.239/0.011. Costs: 65 runs 28,204.6 s (7.83 h), 41 runs 17,790.6 s (4.94 h), 15 runs 6,508.7 s (1.81 h), ζ 10 runs 4,339 s, maximum plan 1.352 + 7.834 + 0.603 + 0.1005 = 9.89 h. The k = 4 and k = 5 cells in the block equal the older recorded S2 cells exactly. I could not cross-check k = 9–12 against the sealed stage (T7), see R2-NOTE-4.

**(d) Outcome space under decision 2 ③, and blinding of the gated order.**
- *Outcome space.* Five B0 candidates on the new 150 games; reachable labels are `FAILED_INTEGRITY`, `NOT_EVALUABLE`, `ADOPT_FOR_2027_PROTOCOL:<B0 tuple>`, `BIAS_NOT_CERTIFIED`, `NO_PRECISION_GAIN`; label 4 is unreachable and B2 rows only raise the descriptive flag `B2_PRECISION_PASS_DIAGNOSTIC` (D:243, D:321–331; J:1726–1763). The precedence and selection rules are complete and consistent. Combination 1 ① + 2 ③ would leave zero candidates and is flagged as meaningless (D:261, J:981). The five candidates' recorded A2 status is as the design says: all pass S1, S4, shared nets and T-A1..T-A4; R1_0×R2_1 has no S-2 cells; R1_a×… lacks only 2ζ_p; R1_b×… lacks 2κ_p and 2ζ_p. The set is well-posed, and the design says honestly (§0.1) that adoption is unlikely.
- *Blinding.* The gate reads widths, S-5 pass/fail, E5 and validity, all A2 allowlist items 1–3 and 9 or the new reward-free items 10–12; no centre, shift endpoint or λ is read. The gate record is deterministic, its sha is appended to the external manifest and committed before any phase-2 run (G-GATE, J:2354–2358). Phase 2 is synthetic and cannot touch rehearsal data. Which phase-2 cells run reveals only that some tuple passed (ii) ∧ S-5, which is public anyway. N′ per cell is fixed at registration, so the data-dependent selection of cells adds no analyst freedom.

**(e) MD ↔ JSON and hashes.** Consistent on arms (5 + 6 B0 descriptive + 12 B2 + λ = 100 row + baseline = 25 rows), cells C1–C8 + 8 descriptive, thresholds, labels, gate, hang guards, N′ options and their hours, the seven decisions and recommendations (1②, 2③, 3①, 4①, 5①, 6①, 7①), costs (9.89 h), seeds. `sources_read_sha256`: I recomputed all 30; **29 match**, including the results JSON (`069bcd6b…`), the RS-03 design (`f03d26b5…`), the RS-03 draft (`8ec7d1b0…`), `decisions.md` (`436dff73…`), my round-1 file (`fef363df…`), ML-POLICY v1 and addenda 1–8, and the A2 code files. The sizing hashes in J:86–90 and J:1435–1440 equal the files. All 76 world seeds (k = 1..76) and the 9 role values reproduce from `SeedSequence([20261028, …])` with no collision against the 15 A2 world seeds or the A2/ML-POLICY roles. The one mismatch is `README.md` (see R2-NOTE-3).

## 3. New findings (round 2)

### BLOCKER / MAJOR

None.

### MINOR

**R2-MINOR-1 — The adequacy evaluation point and the Song-facing reason for decision 2.**
- *Evidence.* In the 5 worlds the in-band count is 0/5, 3/5, 5/5 at t0, t1, t2, and the joint adequacy ∧ C1 count is 0/5, 1/5, 3/5. At t1 two of the three in-band worlds sit within 0.3 % of the q99 upper edge (1.5847 and 1.5833 against 1.588); the other two exceed it (1.7666, 1.9175). The record argues for the one-parameter family "t > 1" (mean ρ 0.958, CI excludes 1; J:565–569 still says "t > 1"), while D:88 now says "t ≥ 1"; t = 1 is the boundary, chosen because it reproduces the ESS ratio (0.68–0.70 against 0.678), while the q99 ratio is matched near t = 2 (1.24–1.50 against 1.392). D:457 and J:2597 say "only one world also passes the adequacy band"; D:219 and J:1599–1602 push to ③ if "C1 fails … or adequacy fails in any".
- *Why it matters.* The recommendation is robust (C1 alone fails in 2/5), but the quoted joint count is not.
- *Fix.* Lead the reason with the robust clause, "the pure swap exceeds B in 2 of 5 worlds, and among the 3 worlds that match the record's ESS ratio at t = 1 it passes in 1", add "adequacy depends on the evaluation point (0/5, 3/5, 5/5 at t = 0, 1, 2)", make J:565–569 and D:88 agree, and say that the push rule is triggered by the C1 clause alone. If ①/② is ever chosen, consider an ESS-ratio-only adequacy rule (5/5 at t1) or a stated tolerance, because the q99 edge is hit at 0.2–0.3 %.

**R2-MINOR-2 — "ω = 1 is worst-case alignment" is not what the construction does.**
- *Evidence.* Per world, |β₂(t0)|/B at ω = 0 / 0.5 / 1: world 0: 0.195 / 0.455 / 1.036; world 1: 0.916 / 0.197 / 2.483; world 2: 1.306 / 0.280 / 0.632; world 3: 2.444 / 2.507 / 1.073; world 4: 2.178 / 1.572 / 2.236. There is no monotone ω effect. ω = 1 only sets ξ = z̄, the standardized situation-average of (Q_world − q̂); the bias-maximizing direction at a given KL is the gradient of β with respect to the log-ratio field weighted by visits × (π_c − π_r) × error, not z̄ itself. D:198 and J:1269 call ω = 1 "worst-case alignment".
- *Why it matters.* The sizing may understate the worst case. This only strengthens ③, but the registered justification should be accurate.
- *Fix.* Reword to "alignment with the situation-average model error", drop "worst case", and optionally compute the true worst direction as one more sizing row.

**R2-MINOR-3 — The sizing README overstates what was fixed in v1.**
- *Evidence.* S:README says all definitions were "fixed in design v1 before any sizing run". v1 defined z̄ from "world Q − model Q"; v2 and the script use "world Q − q̂" (D:198 says "corrected"). The text does not say whether this change preceded the first trial run. The band is described as "within a factor 1.5 either way" but the numbers are 0.5× to 1.5× of the recorded loss/excess, which is factor 2 on the low side.
- *Fix.* State when the z̄ definition changed relative to the trial run, and reword the band description to match the numbers. Neither affects ③ (z̄ enters only the ω part, which shows no systematic effect).

**R2-MINOR-4 — Real-data equivalence of the copied components is not checked.**
- *Evidence.* G-INT-A2 calls the original A2 `rehearsal_evaluate` on S6 (D:366, D:394; J:2797; cost 353.5 s), so it validates inputs and pins, not the new 25-row evaluator. The copy's A2-part is tested only on synthetic ledgers (D:380). The S-2′ driver (a copy of `run_s5x` with new seeds and truth-only rollouts) has an invariant check and seed tests but no check against recorded values. A2's own S3 attempt 1 failed on a real-data harness defect that 160 synthetic tests had not caught.
- *Fix.* Register two cheap integrity steps with zero new exposure: (i) in A2b-S3 also run the new evaluator on S6 with the A2 seeds and require its 12-pair part to match the unblinded A2 record at 1e-12 (about +707 s); (ii) run the new S-2′ driver at A2's world seeds for k = 4, 5 (and the descriptive k = 9–12, now recorded) and require b̂ and SE to match the recorded values at 1e-12, or, if planner randomness depends on call order, state the weaker statistical consistency check. Add both to `G-INT` and the hours.

### NOTE

- **R2-NOTE-1 — Common noise across the five sizing worlds.** `TabWorld` draws perturbation, ε, planning noise and q̂ noise from the A2-base `role_sequence('tab_world', 0, component)` (a2syn:273–274), so the five toy worlds share those arrays and differ in marginals, situations and the D2 field. The lottery spread is, if anything, understated. No effect on ③.
- **R2-NOTE-2 — Label-4 text under ①/②.** S-5, E5 and B2 validity are gate conditions, so a B2 tuple failing them never reaches S-2′ and cannot satisfy "passes (ii) and its R1 component's S-2′ cells"; they remain listed as failing denominator conditions (D:326; J:1742). Simplify. Irrelevant under ③.
- **R2-NOTE-3 — Pin carry-over.** `README.md` pin `f4b4a3d0…` is stale (current `f9f6b527…`; the PM edited it at 02:03:58, after the draft at 02:02:28). Refresh the pin or label it "as read". Commit `results/EXP-OPE-RES-01.json` before G-REG (already stated).
- **R2-NOTE-4 — Dose cells k = 9–12.** They come from the PM's extraction of the sealed A2-S2 stage (manifest seq 68, whose sha equals `8b7fa3aa…`); the k = 4, 5 values match earlier records exactly, but I cannot open T7 to cross-check the other four. The ζ response is non-monotone (5ζ_p is below 2ζ_p), so the design's choice of the cell's own 0.58 B as the primary scenario is the right conservative reading.
- **R2-NOTE-5 — Code freeze under ①/②.** Phase-2 B2 code is written after the gate opens (G-CODE-2; J:2359–2363), relaxing "all code frozen before registration". Acceptable because it is gated, the specification is fixed and the sizing script is the reference implementation, but state it as a deviation. Not applicable under ③.
- **R2-NOTE-6 — Registered-world value (optional).** The decision-2 argument rests on toy-fixture worlds; only the V3 baseline range (0.03–2.40 B, which brackets the real 0.71 B) anchors the scale. The registered-world exact C1 and adequacy are computable in seconds from the sealed A2-S0 TRAIN marginals (no DEV, no rewards). The PM may compute them before Song decides; ③ would stand unless both pass, and B2 would still have to pass (ii).

## 4. The seven recommendations

I would **not change any**: 1 ②, 2 ③, 3 ①, 4 ①, 5 ①, 6 ①, 7 ①.
- **2 ③** is supported by the sizing (C1 exceeds B in 2 of 5 worlds; ω gives no systematic handle), by the recorded ESS −30 %, and by the precision-only adoption logic. Only the wording of its reason needs R2-MINOR-1.
- **3 ①** (55 + 10 runs, 7.8 h) is acceptable because it runs only after the gate; the predictive PASS (40 % / 85 %) is honestly presented, and ② (36 + 5, 4.9 h, 31 % / 77 %) is a reasonable alternative if Song prefers a shorter night.
- **1 ②** remains the only meaningful population with 2 ③ (① leaves zero candidates, ③ spends the remaining DEV).

## Verdict

VERDICT: PASS_WITH_NOTES

0 BLOCKER, 0 MAJOR. All round-1 findings are CLOSED. The four MINORs (R2-MINOR-1 to 4) and the notes can be closed by the author in a v2.2 before G-CODE; R2-MINOR-4 is the one I would not skip. The design may go to Song with recommendations 1②, 2③, 3①, 4①, 5①, 6①, 7①. The Codex re-confirmation (after 2026-11-02 18:57 KST) and G-CODE remain required.
