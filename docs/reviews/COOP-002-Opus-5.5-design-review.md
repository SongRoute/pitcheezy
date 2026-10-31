# COOP-002: Claude Opus 5.5 independent design review

2026-09-27. Actual CLI response model `claude-opus-5-5`, source checkout `065a3a6`, attempt `COOP-002/attempt-003`. Read-only review; no fits or scoring. This is the original returned review, preserved before corrections.

```json
{
  "task": "COOP-002",
  "verdict": "APPROVE_WITH_REQUIRED_CHANGES — the scientific design is sound. Fix two things in the registration text before freeze. No fit is needed to resolve either.",
  "scope_note": "Read-only review of the draft at /Users/song/Projects/pitcheezy/docs/contracts/ML-G0-F1-CONFIRMATION-DRAFT-v1.md against worktree source at 065a3a6. That commit differs from the draft's base 7b52d0c only in docs/ and scripts/run_claude_task.py, so the cited line numbers hold. I made no edits, fits, scores, run-artifact reads or 2026 access.",

  "confirmed_correct": [
    "Fit count is right: 4 new, 6 reused. G0 needs only the 'global' unit (matrix_confirmation.py:11-12, 76-84). C1 therefore adds 1 unit × seeds {3,4} = 2 fits (run_ml_confirmation.py:298), and the masked arm adds 2 more.",
    "Per-seed calibration identity holds. summarize_cell fits each seed's June blend from that seed's members only (score_ml_matrix.py:73-77), so the seed_primary values for seeds 0-2 are the same at 3 and 5 members. The ensemble June blend is refit from the averaged members (lines 68-71), so the draft correctly calls the five-seed ensemble weight new for both arms.",
    "The three-seed reference is rebuilt and checked: C1 at score_ml_confirmation.py:97-103, F1 at run_ml_bridge.py:218-243. C1's check compares prediction arrays only; the bridge also compares June selection weights.",
    "The baseline-only route works as the draft says. matrix_confirmation.py:50-52 requires [G0] with no pairs. With no pairs, compare_family computes no R and no N, and finish_decisions pads four None slots, which holm_adjust accepts (matrix_metrics.py:124-136).",
    "The four-of-five rule matches prediction_decision(stage='final') → (5,4) at matrix_metrics.py:143. The draft correctly says the three-seed F1 scorer does not have it (score_ml_bridge.py:32, DECISION required_negative_seeds=2).",
    "Frozen G sources are unchanged since the G run. None of the run_ml_matrix/benchmark/sharing SOURCES files has a commit after the 2026-09-24 09:40 registration, so the parent source check at run_ml_confirmation.py:214-216 should pass. The native-environment and device match (lines 57-70; run_ml_bridge.py:416-421) is still an unverified precondition.",
    "The code gaps are stated accurately, not understated. The bridge hard-codes SEEDS=(0,1,2) (run_ml_bridge.py:48). Its fixed config fails on any change (lines 88-103). Its scorer requires three members and labels the result a screen (score_ml_bridge.py:43-65, 137-141, 225-236). C1's primary_comparisons accepts only mapped G pairs (matrix_confirmation.py:44-49).",
    "Cost inputs match the report. docs/reports/ML-matrix-execution-2026-09-24.md:478,483 shows 343.7 s per member projected, a 349.0 s ledger and a 3.83 s score. The actual parent G0 fits took 90-101 s.",
    "Independence wording is correct: 'same exposed 328 games', no independent_confirmation=true, and the three-seed pass is acknowledged as the adaptive trigger.",
    "R is correct. There are 24 slots (ROBUSTNESS, run_ml_bridge.py:63-65), Bonferroni 0.05/24, and volume_zero is kept as structural and unconfirmed (score_ml_bridge.py:85-90). This is correctly separated from C1's 48-slot candidate family (matrix_confirmation_metrics.py:12, 57)."
  ],

  "blocking_findings": [
    {
      "id": "B1",
      "title": "The C1 output is not named as a pinned parent of the F1 extension, and the pairing checks are not carried over to seeds 3-4",
      "evidence": "The draft (line 22) pins 'both parents' (G EXP-P4-001 and F1 EXP-P9-001-v2). The full arm for seeds 3-4, however, comes from the new C1 output (line 17). In the masked fit, the existing pairing checks load the matched G0 checkpoint from prep['full_reuse'] and compare device and network_signature (run_ml_bridge.py:416-421, 538-542, 569-586). For seeds 3-4 that checkpoint lives under the C1 output, which does not exist yet.",
      "required_change": "Register the C1 output as a third pinned parent of the F1 extension: its preparation SHA, fits/seed{3,4}/global state and model hashes, and members/G0-global/seed{3,4}/prediction_state hashes. Fix the order as: C1 prepare → profile → fit 3,4 → predict 3,4; then F1-ext prepare, which pins those hashes; then masked fit 3,4. Require device equality, network_signature equality and identical train/early row hashes between each masked seed-3/4 fit and its C1 counterpart. Require the F1-ext full-arm five-seed arrays (primary, calibrated, raw, seed_primary) to be byte-equal to the C1 analysis G0-global arrays whenever both exist."
    },
    {
      "id": "B2",
      "title": "The single 14,400 s ceiling that charges failures cannot be enforced for the full arm",
      "evidence": "The draft (line 44) sets one aggregate ceiling covering both arms, including failures and interrupted attempts. The C1 runner has no ledger: resource_gate is projection-only and says caller wall limits must be enforced externally (run_ml_confirmation.py:158-181). The C1 config also requires its own batch_wall_budget_seconds ≤ 28,800 (matrix_confirmation.py:68-72). Only the bridge's ledger_total charges interrupted attempts (run_ml_bridge.py:125-159).",
      "required_change": "Freeze an explicit split: C1 batch_wall_budget_seconds = X, and the F1-ext family budget = 14,400 − X or another stated total. Require an external start/end ledger wrapper for every C1 command, including prepare, profile, fit, predict and score, using the same charging rule as ledger_total. Otherwise the reported new spending omits failed full-arm attempts. Real overrun risk is low given the actual 90-101 s fits, but the accounting claim must be true."
    }
  ],

  "nonblocking_findings": [
    {
      "id": "N1",
      "title": "C1 scoring can open five-seed G0 results before the ten-member family exists",
      "evidence": "score_ml_confirmation.py:151 prints five-seed G0 NLL and writes results.json. The draft (line 46) says to open results only after all ten members exist, but it does not order the C1 score step.",
      "suggestion": "State that C1 `score` runs only after the F1-ext registration is frozen and masked seeds 3-4 are predicted, or at minimum after the F1-ext registration is frozen."
    },
    {
      "id": "N2",
      "title": "Status vocabulary is not frozen",
      "evidence": "The existing bridge_decision returns status 'predictive_improvement' and hard-codes stage 'three-seed bridge screen' (score_ml_bridge.py:53-65). The C1 scorer writes independent_confirmation: None, not false (score_ml_confirmation.py:130).",
      "suggestion": "Freeze the F1-ext status enum, for example development_stability_pass / inconclusive / worse_or_guardrail_failure / reverse_point_estimate. Require stage='five-seed development stability, exposed Cpanel' and independent_confirmation=false in both new outputs."
    },
    {
      "id": "N3",
      "title": "Profile projections for the two arms are not comparable",
      "evidence": "The C1 profile uses 8,192 TRAIN rows × 2 epochs (run_ml_confirmation.py:344). The F1 profile uses 65,536 rows (run_ml_bridge.py:56).",
      "suggestion": "The draft allows 'C1 G0 profile or matching full-arm profile'. Pick one before freeze, preferably the matching 65,536-row profile, so the two arms' projections are comparable."
    },
    {
      "id": "N4",
      "title": "The whole-MLB section does not address overlap with Cpanel",
      "evidence": "The draft (line 38) proposes whole-MLB G0/F1. The Cpanel DEV (12,334 pitches) is presumably a subset of the eligible whole-MLB DEV, so whole-MLB results would partly re-score exposed rows. The masked whole-MLB arm also depends on the Cpanel five-seed masked fits. No runner for G0 or masked whole-MLB inference is cited.",
      "suggestion": "When registering whole-MLB, pre-specify the estimand as the non-Cpanel complement, or report whole and complement both, with the overlap count. List the needed code (a whole-MLB predictor for G0 and masked members) as a gap. The separation and non-independence wording are otherwise adequate."
    },
    {
      "id": "N5",
      "title": "The multiplicity note could state the sequential look explicitly",
      "evidence": "The five-seed test is a second look at the same hypothesis on the same games after a passing three-seed look (lines 7, 11, 28).",
      "suggestion": "Add one sentence: the five-seed p-value is a stability diagnostic with no fresh α-level guarantee, and the three-seed and five-seed decisions are reported together, not in place of each other. Otherwise the one-hypothesis F1 family and the separate C1 four-slot Holm family are valid as written."
    }
  ],

  "suggested_next_step": "Revise the draft for B1 (third parent, fit order, seed-3/4 pairing checks, full-arm byte-equality) and B2 (budget split plus external C1 ledger). Add the N1 ordering and N2 status enum. Then freeze the F4 status and the C1/F1-ext configs, implement the F1 extension adapter and five-seed scorer with synthetic contract tests, and only then run prepare/profile. Leave whole-MLB as a separate later registration."
}
```

## Revision review, attempt-004

```json
{
  "task": "COOP-002 disposition check",
  "verdict": "RESOLVED. No blocker or circular dependency remains. The dependency order in the revised draft (line 24) runs one way only. The C1 third parent is declared at registration (line 22), and its hashes are materialized after the full-arm predictions and pinned before any masked fit (line 24). F1-extension prepare comes after C1 fits and predictions. Each score step comes after all ten members exist. No step waits on its own output.",
  "resolved_ids": {
    "B1": "Resolved at lines 22 and 24. The third parent is declared up front. The pinned manifest covers the C1 preparation SHA, the fits/seed{3,4}/global state and model.pt, and the members/G0-global/seed{3,4} prediction_state and predictions.npz. Each seed pair must have equal device and network_signature and identical TRAIN and early-stop row hashes. June and DEV keys and labels must match. The full-arm arrays must be byte-equal to the C1 analysis.",
    "B2": "Resolved at line 48. The budget splits 7,200 s to C1 and 7,200 s to the F1 extension, with no transfer between arms. One external ledger records every command. A start with no recorded end reserves the full stage cap. The F1 runner's internal ledger is a cross-check and is not added to the total. The early masked profile is charged to the F1 share.",
    "N1": "Resolved at line 24. C1 score waits until all ten member predictions exist.",
    "N2": "Resolved at lines 11 and 30. The status enum is frozen, reverse_point_estimate is a separate boolean, and the stage text is fixed. independent_confirmation stays null, matching the existing schema, and held_out_confirmation is set to false explicitly.",
    "N3": "Resolved at line 46. Both arms get matched profiles of 65,536 TRAIN rows, 2,048 early-stop rows and 64 May rows × 400 draws. The existing C1 8,192-row profile is kept as a separate gate.",
    "N4": "Resolved at line 40. Whole-MLB starts as a G0-only evaluation, reports both the whole population and the non-Cpanel complement with overlap counts, and states that neither is independent. The masked whole-MLB arm needs its own prospective registration. The whole-MLB adapter is listed as a code gap.",
    "N5": "Resolved at line 32. The five-seed p-value is a second-look diagnostic with no fresh alpha, and the three-seed result is reported alongside it."
  },
  "remaining_issues_nonblocking": [
    {
      "id": "R1",
      "issue": "The C1 budget cannot be chosen after C1 prepare. Line 24 puts 'freeze the two arm budgets' after C1 prepare and profiling. But batch_wall_budget_seconds sits inside the C1 config, which is validated at matrix_confirmation.py:68-72. That config's hash is bound into the preparation identity and rechecked by verify at run_ml_confirmation.py:146 and :186. C1 fit then enforces the value (run_ml_confirmation.py:395).",
      "fix": "Say that both budgets (7,200 + 7,200) are fixed in the configs before C1 prepare, and that the profiles only pass or stop the plan. A failed gate needs a new C1 output ID, with the old attempt preserved. Line 48 ('stop and revise the registration') already implies this."
    },
    {
      "id": "R2",
      "issue": "Nothing in code enforces the rule that C1 score waits for all ten members. The C1 scorer will run as soon as its five full members exist (score_ml_confirmation.py:69-70).",
      "fix": "Have the external wrapper refuse a C1 score job unless both F1-extension masked seed-3/4 prediction_state files exist and are hash-verified. Record that refusal rule in the ledger contract."
    },
    {
      "id": "R3",
      "issue": "Where the third-parent manifest is written, and the baseline match between the two analyses, are unspecified.",
      "fix": "Write the manifest in the F1-extension registration or output area, not inside the C1 output, so the C1 output stays immutable. Before comparing arrays byte-for-byte, require that the F1-extension baseline_predictions.npz has the same SHA as C1's parent_baseline_predictions.npz."
    }
  ],
  "suggested_next_step": "Add the three one-line clarifications R1–R3 to the draft. Then do the freeze checklist at line 50: F4 status, the two configs with their budgets, the third-parent declaration and the ledger contract. After that, implement the adapter, the external ledger and the matched profiles with synthetic contract tests."
}
```
