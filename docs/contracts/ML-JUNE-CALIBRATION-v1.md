# ML-JUNE-CALIBRATION-v1 — expanded June global and volume calibration

2026-09-28. **Scientific specification for preregistration; no execution is authorized by this document alone.** The current task covers design and preregistration only: no new label/probability values, runtime profile, inference, optimizer or quality calculation is run. Baseline research state is D81, repository `d5466e01bf3ad73262d6c4f4aacc1b43de6c7811`. The coordinator registers the scientific configs now; an independently reviewed executable source C, complete data/source/environment pins, registration D, command plan and explicit release remain prerequisites for future execution. Missing runners are not described as execution-ready.

Family ID is `ML-JUNE-CALIBRATION-v1`. `EXP-P12-001` changes the global blend's calibration population (B1 versus B0). `EXP-P12-002` changes the blending method on the same expanded population (B2 versus B1). B2 versus B0 is the shared family's net comparison, not an additional candidate or experiment axis. Future output is a fresh `RUNROOT/ML-JUNE-CALIBRATION-v1`, with separate B1/B2 artifacts; parents are read-only.

## 1. Question, minimal family, and interpretation

[D81 preparation](../reports/ML-June-eligibility-2026-09-27.md) established 104,970 eligible June pitches, compared with the old 4,821-pitch Cpanel. [P11](ML-G0-WHOLE-MLB-v1.md) found that an old-Cpanel class-bias correction worsened exposed DEV; the volume blend was inactive because the old calibration groups lacked support. The present family investigates the now-supported scalar network–frequency blend. It does not repeat class bias, add temperature fitting, neural experts, player-specific parameters, additional group definitions, or a combined correction.

| Predictor | Calibration data | Method | Role |
|---|---|---|---|
| B0 | Original June Cpanel 4,821 | Archived global network–frequency weight | Frozen G0 research reference; no parameter replacement |
| B1 | Full eligible June 104,970 | One newly fitted global scalar blend | Population-expansion axis |
| B2 | Same full eligible June 104,970 | Four group scalars shrunk toward B1's new global weight | Method axis at fixed data |

B1−B0 changes the calibration sample's size **and coverage/composition**. It is not a causal estimate of adding rows while holding the sampling distribution fixed. B2−B1 isolates the registered method change on identical fit/evaluation rows and components. No matched-size resampling arm or factorial interaction is estimated. Adding one later requires a new family. B2−B0 measures their combined net change.

Both B1 and B2 run if the registered input/cost gates pass; there is no new DEV-driven activation or replacement candidate. Choosing this family after P11 and the support audit is explicitly adaptive research. Local multiplicity correction does not undo earlier DEV exposure or provide a new study-wide false-positive guarantee.

## 2. Frozen data and predictor identity

June CAL is the exact ordered eligible inventory from `ML-JUNE-ELIGIBILITY-v1`: **104,970 pitches / 394 games / 549 pitchers**. Whole requested denominator is 115,816; exclusions 10,846 remain unchanged. Cpanel is exactly **4,821 / 110**, its eligible key complement **100,149 / 394**, with 110 shared games. Do not confuse this with the requested-key complement 110,995 or outside-panel requested rows 110,604.

| Frozen preparation artifact | SHA256 |
|---|---|
| manifest | `f564145deb6b6b58a079766d582573b95154c3515af39fb39b0dba56088bdc88` |
| result | `d00606b8f59d0244cbd8346a09f4b6bf9a2ac45f54be59e6e892891b1095671c` |
| eligible_keys.parquet | `ff9db8e9a6b2f32a9f4279b30ac6a303b0dbb9d9267746f4cb0dc45741d5f037` |
| eligible_metadata.parquet | `38507b7df491c210ccff46352c47ace4feb385e277c0b88dd3a96b62b5557c76` |
| eligible ordered KEY bytes | `ddb3a0859240821056ea687713b68ab4d2e2d0d8debb8dbd2d64871796e37f82` |

KEY is `(game_pk, at_bat_number, pitch_number)` in the sealed order. Every prediction/label/metadata array must have exactly this identity, with no duplication, missing row, reordered join or new eligibility decision. June labels will be derived using the already pinned P4 `outcome_labels`, not processed `pitch_outcome`; Cpanel labels must exactly equal the archived labels. Seal the full ordered label hash before any calibration fit. Its value cannot be invented at scientific registration: pin the source bytes, keys, mapping code and derivation first; the future preparation stage deterministically derives and seals the value under that registration.

Evaluation remains the **already exposed July–September 2025 whole-MLB DEV**, 311,721 eligible pitches / 1,161 games. Reuse frozen P11 member probabilities, frequency probabilities, labels and metadata. Cpanel DEV is 12,334 / 328; exact non-Cpanel DEV is 299,387 / 1,161, sharing 328 games. No new DEV sampling or inference is needed. All three comparisons use identical rows.

Use the five G0 members in fixed order `[0,1,2,3,4]`: P4 seeds 0–2 and P10/C1 seeds 3–4. Freeze actual checkpoint hashes, encoder/normalizer, feature ordering, TRAIN-only frequency model and its existing calibration, original May delivery temperatures, and the original **400 joint delivery draws and tier assignments**. Do not regenerate a merely same-sized random draw set. No May/global neural refit or temperature search is permitted.

For predictor index `a ∈ {ensemble, seed0,…,seed4}`, let `m_i^a` denote May-calibrated network probabilities. `m_i^ensemble = (1/5) sum_s m_i^s`. Let `f_i` be the same archived calibrated frequency predictor for all six indices. The ensemble is formed **before** fitting its scalar; it is not the average of five separately blended seeds. B0 uses archived five-seed C1 ensemble weight `w0^ensemble` and the five corresponding seed weights `w0^s`. A three-seed G weight is never substituted.

## 3. Future data-access and replay gates

The previous preparation only decoded 18 columns for eligibility and did not retain raw labels or perform inference. Future calibration execution needs more inputs and is a **new access event**. Its preparation may use the approved existing 2023–25 TRAIN/prior-context processed and feature/physical caches, archived frozen auxiliary estimators and June labels. Source datasets/caches and exact columns or serialized dependencies must be enumerated and hashed before release. No raw fetch or 2026 source is permitted. Do not silently invoke an old loader that fetches raw files or rewrites caches/parents.

June queries use only the original point-in-time features: previous-pitch history and prior-date player context under the frozen G0 rule; neither the current target nor future dates enter a feature. For prior-date aggregate features, same-day exclusion remains unchanged; legal earlier-pitch history within the current game is still allowed. Synthetic tests must distinguish these two clocks: perturbing same-day/later source rows cannot change a prior-date aggregate, and perturbing future pitches cannot change a query history. Full-date historical cache availability is not permission to include future observations. Current June realized location is marginalized by the same 400 TRAIN draws for delivery inference. Metadata-dependent role/volume groups are not newly learned from June or DEV. Existing DEV archives may be read for identity/replay and later scoring; no feature reconstruction needs future DEV outcomes.

At future execution, log when labels/probabilities were decoded or generated, source identity, date/column scope, purpose and output hashes. Training likelihood checks are optimizer diagnostics, not new held-out performance. No fit input may contain DEV keys/labels. Check exact fit/evaluation game-date separation, not only split strings.

Required identity gates, all before candidate DEV losses:

1. Validate research bundle, preparation and all actual member/dependency hashes. For every seed, check model, May calibration and original delivery identity; retain raw/calibrated/tier outputs separately.
2. Each full-June member's Cpanel intersection must exactly match archived keys/labels and delivery tiers, with calibrated and raw probability maximum absolute difference ≤`1e-6`, `rtol=0`. Frozen frequency must reproduce its old June Cpanel probabilities to the same tolerance. Check full intersection, not only a small probe.
3. Run six **identity-only scalar replay fits** on the original archived Cpanel component arrays/labels, using the optimizer in §4. Reproduced old weights must agree with archived C1 ensemble/seed weights to `1e-8`; original B0 predictions must replay to `1e-6`. Use original archived arrays for this optimizer check, so allowed streaming differences do not silently change B0. A mismatch stops execution; do not widen tolerance after seeing it.
4. Reconstruct B0 DEV from the frozen components and archived weights and compare with P11 `primary` and `seed_primary` (`1e-6`, `rtol=0`). The frequency authority is `EXP-P10-001/parent_baseline_predictions.npz`: `blend` for Cpanel replay and `mlb_dev` for DEV application. Its `mlb_dev_keys` must exactly equal P11 ordered DEV keys; its `dev_keys` defines the DEV Cpanel. Recheck whole/Cpanel/complement identity and source hashes. B1/B2 changes use those same components.

The original six weights were created by archived C1 `source/scripts/score_ml_matrix.py::summarize_cell`; its file hash and C1 `analysis/results.json`/manifest are explicitly pinned in config. Weight paths are `reports['G0-global'].selection.model_weight` and `reports['G0-global'].seeds[0..4].blend_selection.model_weight`. Ensemble construction is float64 `np.mean(np.stack(seed_calibrated[0..4]), axis=0)` and the optimizer argument is literally `objective='log_loss'`. C1 preparation records NumPy2.5.3 and SciPy1.18.1; preparation identity and this source are pinned. A version-related replay mismatch still stops the family rather than widening tolerance.

The six replay optimizations have computational cost and are counted separately. They are not additional candidates or an opportunity to update B0.

## 4. Exact algorithms and fixed optimizer

For any fit subset S and predictor a, define the scalar objective

`L_S^a(w) = mean_{i∈S}[-log(clip(w*m_i,y_i^a + (1-w)*f_i,y_i, 1e-12, 1))]`, `0 ≤ w ≤ 1`.

Use float64. The algorithm matches pinned `scripts/run_sequence_calibration.py::fit_blend`, SHA256 `951bada57ea33f34cc5440d9420943df7dc9cb782b27e9359c582e55b5cfa6bc`: SciPy `minimize_scalar(method='bounded', bounds=(0,1), options={'xatol':1e-5,'maxiter':500,'disp':0})`, followed by comparison of `[0, optimum.x, 1]`. Select minimum objective with ties resolved by that listed order, as Python `min(..., key=score)` does. Those explicit values match the currently pinned bounded-optimizer defaults. No grid, restarts, alternate objective, class weighting, sampling weights, early stopping on DEV or hyperparameter search is allowed.

An additive pure wrapper may expose optimizer diagnostics without importing the old experiment CLI. It must reproduce the pinned algorithm on synthetic cases and the old six replay fits. Freeze SciPy and numerical environment versions/source identity before execution. Inputs must be finite probabilities in `[0,1]`, sum to one with absolute tolerance `1e-6` and zero relative tolerance, with exactly ten fixed classes. Labels are integers 0–9. Do not silently renormalize malformed inputs.

Optimizer success is required. Chosen weight must be finite and in `[0,1]` (no silent clipping); objective must be finite, no worse than both endpoints within `1e-9`, and no worse than the relevant feasible anchor within `1e-9`. For B1 the anchor is `w0^a`; for each B2 local fit the anchor is B1's `w1^a`. These are validation conditions, not extra candidate points for changing the optimizer's tie rule. Convergence at `xatol=1e-5` does not guarantee the stricter absolute objective check; a converged result failing it is deliberately a conservative family stop. Keep the absolute `1e-9` rule. Before execution, synthetic high-curvature, boundary and tiny-probability stress cases must exercise this behavior; they cannot prove the absence of all numerical false failures. Do not substitute relative tolerance after seeing real data. If any activated optimization or replay fails, preserve its diagnostics and stop the entire family **before applying/scoring candidates**. Do not replace failures with endpoint/global fits or remove their comparison slots.

**B0:** `p_i,B0^a = w0^a*m_i^a + (1-w0^a)*f_i`.

**B1:** fit `w1^a = fit_blend(June_all, a)` independently for all six predictor indices. Then `p_i,B1^a = w1^a*m_i^a + (1-w1^a)*f_i`.

**B2:** for each registered TRAIN-volume group g and predictor a, let n_g be June eligible pitch count and G_g the distinct June game count. When `n_g≥500 AND G_g≥30`, fit `u_g^a = fit_blend(June_g, a)` and set

`rho_g = n_g/(n_g+1000)`

`w2,g^a = rho_g*u_g^a + (1-rho_g)*w1^a`.

For a valid but unsupported group, set `w2,g^a = w1^a` exactly and record `fallback=true`, the support counts and no group optimizer call. There is no fallback after optimizer failure. Predict `p_i,B2^a = w2,g(i)^a*m_i^a + (1-w2,g(i)^a)*f_i`.

Shrinkage is fixed at **1,000 pitches**. The anchor is the **new B1 global weight for the same predictor**, not the archived B0 weight used in P11's inactive I2 design. No class bias, extra temperature or second correction is applied. Within each supported group, the convex combination with its global anchor does not introduce a hyperparameter search. Its likelihood is still a fit diagnostic, not a validation estimate.

## 5. Fixed groups, support and fallback

Use original P4 D100 TRAIN pitcher-volume mapping; q25=170, q75=1514 and group labels are frozen. A pitcher absent from that TRAIN mapping is the valid **zero** group, not missing metadata. Unrecognized group strings, missing key/identity, conflicting mapping or invalid metadata fail closed. Do not silently route invalid metadata to B1 or call it an unseen-player fallback. There is no production-runtime policy in this protocol.

| g | June eligible pitches | Distinct games | Current fit support |
|---|---:|---:|---|
| zero | 4,788 | 127 | yes |
| low | 5,338 | 184 | yes |
| middle | 38,546 | 393 | yes |
| high | 56,298 | 386 | yes |

These counts must replay exactly; an unexpected change is an input failure, not grounds to drop a group. Therefore the expected new calibration optimizer calls are **6 global + 4×6 group = 30**, plus the **6 old-Cpanel replay calls**, total **36 scalar optimization calls**. Neural fits are zero. No cross-validation-fold fits are scheduled. Generic supported-group fallback remains specified/tested but is expected to be unused in this frozen June population; report actual fallback rows and calls even when zero.

The unsupported June intersections `starter×low` (24 games) and `L×zero` (29 games) are not calibration parameter groups. Do not create intersection calibrators, pool them differently, or infer their separate correction benefit from the supported marginal volume groups.

## 6. June reuse, cross-fitting and sealing

B0's archived blend was fitted on the old June Cpanel. Consequently a June score of B0 is not independent, and random K-fold evaluation of corrections on B0 would not remove the old blend's access to held-out June labels. This protocol **does not claim June out-of-fold validation**.

B1/B2 fit directly from the frozen **pre-June-blend** May-calibrated network and frequency components. They do not fit another correction on B0's June-fitted primary probabilities. Their method, shrinkage, groups and optimizer are fixed before fitting; there is no tuning/model selection from June fit losses, so cross-fitting is not needed to choose hyperparameters here. All eligible June labels are fitting data. No June performance leaderboard, bootstrap significance or generalization claim is produced. Fit objectives can be retained solely as finite/success/optimizer audit evidence, labeled in-sample.

Serialize all B1/B2 ensemble and seed parameters, inputs, per-group counts, objective/success flags and hashes in a fit manifest **before loading candidate DEV labels for scoring**. Cached DEV probabilities may be validated without loss computation; fit functions accept only the sealed June bundle. The apply command uses parameters plus component probabilities and groups, not labels. The score command is separate and requires the full fit/apply manifests for both candidates and all five seeds.

No choice among penalties, shrinkage strengths, temperatures or methods is made using DEV or June losses. Future tuning would require a new protocol with a genuinely nested or temporal June validation scheme that also refits the inner blend; it cannot be added after these results. Legal prior-day context updates and repeated use of already-exposed DEV do not create independent confirmation.

## 7. Exactly three N contrasts

Negative differences mean improvement. Primary estimand is **whole-MLB DEV pitch-weighted mean paired NLL**, not equal-game mean:

1. `C1 = B1 − B0` (population-expansion effect).
2. `C2 = B2 − B0` (net change).
3. `C3 = B2 − B1` (method effect with calibration data fixed).

For every contrast, report summed ten-class Brier alongside NLL, absolute B0/B1/B2 values, and five matched seed differences. Use the same predictor index on both sides of each seed comparison. The five directions are stability diagnostics, not five independent sample replicates.

Use **10,000 whole-game bootstrap draws**, fixed RNG seed `20260924`, sorted unique game IDs, sampling all games with replacement and recomputing sum(loss)/sum(pitches) per draw. Paired comparisons use the same game draws across methods. CI is percentile 2.5/97.5 using NumPy linear quantiles. The one-sided p is the existing null-centered plus-one statistic: `(1 + count(delta_star − observed_delta ≤ observed_delta))/(draws+1)`. Apply Holm at α=.05 to **all three NLL slots in the above fixed order**; do not reduce the family after seeing a result.

Each N slot passes only if all conditions hold: ΔNLL≤−`.003`; NLL 95% upper<0; Holm-adjusted p≤`.05`; Brier 95% upper≤+`.001`; at least **4/5** matched seed NLL differences<0. Classify `development_improvement`, `worse_or_guardrail_failure` (NLL 95% lower>0 or Brier 95% lower>+.001), or `inconclusive`. If the family aborts before scoring, mark all comparisons `unmeasured`, with no fabricated p or success claim; any summary requiring a p vector reserves unmeasured slots conservatively as 1 and retains denominator3.

B1 is a research improvement candidate only if C1 N and its R pass. B2 must pass **both C2 and C3 N and their R**, demonstrating net gain and gain over B1. It cannot be promoted by beating B0 while failing the method contrast. B1 need not pass for B2 to pass both of its own contrasts. The scorer must verify for NLL and Brier that `(B2−B0)=(B1−B0)+(B2−B1)` at the point estimate and every aligned bootstrap draw, absolute tolerance `1e-12`, relative tolerance0. Record `contrast_additivity_passed` in completion; invalid row/draw alignment stops scoring.

Report all slots regardless of these candidate labels; even a passing label does not authorize replacement of G0 or service promotion.

Report whole/Cpanel/non-Cpanel absolute NLL/Brier and descriptive paired CI for all three contrasts. The Cpanel/complement breakdown adds no N hypothesis. ECE10/reliability, class prevalence/ECE and month/seen–unseen summaries may be descriptive after scoring; they do not select the winning method or create new tests. Per-class claims require ≥30 observed events and retain insufficient-support classes without suppressing them.

## 8. Fixed R78 and uncertainty limits

For each of C1/C2/C3, use the same twelve whole-MLB groups as P11: starter, relief; L, R; TRAIN-volume zero/low/middle/high; two-strikes/less-than-two; runners-on/bases-empty. Add one whole non-Cpanel complement group. This gives **3 contrasts ×13 groups ×2 losses =78** one-sided simultaneous upper-bound slots.

Use **100,000 complete-game draws**, RNG seed `20260924`, Bonferroni `upper_alpha=.05/78`, linear quantile `1−upper_alpha`. Each group's sampling unit is its distinct game; include all of that group's pitches within each sampled game, preserving pitch-weighted ratios. Corresponding contrast/group comparisons share the same draws. NLL margin is +`.010`, Brier +`.002`. A group requires ≥30 games **AND** ≥500 pitches; empty/unsupported/invalid groups retain their slot as `unconfirmed`, never reduce78. An upper bound above the margin is noninferiority unestablished. Report positive ordinary two-sided lower bounds as separate deterioration evidence.

A contrast R passes only when all of its 26 bounds are measured and within margin. Candidate rules in §7 use the contrast-specific R while preserving the shared78 multiplicity. The remaining role×volume/hand×volume intersections, individual pitchers and complementary subgroup slices are descriptive only; do not attach simultaneous-R or improvement labels to them.

All intervals condition on the frozen base models, fitted calibration parameters and observed outputs. They omit model fitting, June calibration-estimation, adaptive candidate choice and repeated-DEV-selection uncertainty. Neither Holm3 nor R78 supplies fresh confirmatory error control after the earlier analyses. Keep `independent_confirmation:null`, `held_out_confirmation:false`, `policy_effect:null`, `service_adoption:null`.

## 9. Fixed compute budget and future profile gates

The **single family cap is 3,600 seconds**, including new June preparation, runtime profile/probes, five full-June member predictions, all36 scalar calls, apply, score, worker-side verification/I/O, and failed/interrupted work. The completed D81 eligibility preparation cost1.219837541s is disclosed separately and is not charged again; every new preparation command is charged. There is no separate free inference phase or transfer from earlier budgets. One shared `.heavy.lock`, one queue. No seed/draw/row reductions to pass a cost gate.

| Future command stage | Full-worker cap |
|---|---:|
| Prepare identities/June labels, full frozen-frequency probabilities and input store | 300 s |
| Profile seed0 and small old-Cpanel probe | 600 s |
| Predict each of seeds0–4 | 600 s each |
| Fit/replay all scalar parameters and seal | 300 s |
| Apply all B0/B1/B2 ensemble/seed outputs | 300 s |
| Score all fixed N/R and diagnostics | 300 s |

All command caps are additionally bounded by remaining family time. The fixed fit+apply+score reserve is **900 seconds**, retained throughout June inference. Individual maxima are ceilings, not additive reservations exceeding the family cap.

Profile uses the **first8,192 frozen June eligible keys**, seed0, exactly400draws, plus first64 old-Cpanel keys as an identity probe. It measures runtime only: no label-dependent sample choice, loss, class distribution or candidate decision. Discard profile probabilities after runtime/identity verification; the full seed0 prediction is still a complete registered member. Let `P_profile` be the official full-worker Popen→wait profile wall. Set **`T=P_profile*(104970/8192)`**. Profile must load/materialize the same complete frozen feature/history input store and production path as a full member; only the prediction query count differs. This scales interpreter/import, full input loading/construction, the separate probe and output/exit overhead as well as prediction, deliberately overcounting fixed work. Internal phase times are diagnostic only. The row-ratio model and factor2 are prospective margins, not a guarantee of linear scaling. Before seed0, require both `2*T≤600` and `2*(5*T)+900≤remaining_family_seconds` **after** charging prepare/profile. Remaining time uses the official worker ledger, not an internal timer.

After each completed full member, let M be the maximum observed full-process wall of completed members and k the number still missing. Before the next member require `2*k*M+900≤remaining_family_seconds` and `2*M≤600`; command timeout is at most `min(600, remaining−900)`. After all five are complete, require sufficient remaining time for the next registered fit/apply/score command's cap; no inference reserve is spent to bypass a gate. No unscored partial-seed ensemble may proceed.

The prior whole-MLB five-predict worker cost 1,777.683964s for311,721 rows gives an approximately599s linear **total for all five separately timed prediction workers** at104,970 rows; it is context only, not observed June inference or a budget guarantee. The new ×2 gates are prospective safety factors, not confidence bounds.

Authoritative cost is monotonic **immediately-before-Popen→wait/termination** for each worker, including interpreter/import/load/output/exit. Supervisor preflight/ledger work and independent post-run audits are reported separately; outer wall is a nonadditive crosscheck. Record all failures and exposure already incurred. A failed attempt never disappears from the family ledger; no automatic retry, alternate algorithm or tolerance expansion is permitted. Source/config changes require a new registered attempt with prior cost and exposure carried forward; the current budget does not silently reset.

## 10. Required artifact interfaces and release checklist

The implementation is additive; do not modify P4/P10/P11, June preparation, the research bundle or old calibration modules in place. Required inputs are:

| Role | Authority |
|---|---|
| June population/groups | D81 manifest/eligible keys+metadata and requested inventory; exact hashes above and preparation manifest |
| Original Cpanel labels/components | P4 seed0–2/P10 seed3–4 June member archives; canonical P10 `parent_baseline_predictions.npz` (`blend`); C1 ensemble/per-seed weight reports |
| Frozen model/delivery identity | G0 research reference bundle; each actual model/calibration/state; fixed auxiliary/feature/cache sources |
| Existing DEV components | P11 five member `dev`/`dev_raw`/keys/labels/tiers, archived calibrated frequency, metadata and analysis primary/seed-primary arrays |
| Pure algorithm/scoring | Registered scalar optimizer, loss/bootstrap/Holm helpers and dedicated explicit78-bound inventory; do not fake another helper's family size |

The300s prepare stage explicitly derives all104,970 labels using pinned P4 `model.py::outcome_labels`, predicts frozen frequency for those same keys from pinned P4 `aux.pkl['baseline']`, and applies the archived `preparation.json.baseline_temperature.temperature` with pinned frequency-temperature source. No frequency or temperature fit is performed. Both operations and feature/history preparation count against the family budget. P4/P10/P11 copies of the old frequency archive are byte-identical, but the registered canonical replay/application path is P10. New full-June frequency and ordered label hashes are sealed before candidate fitting.

June member archives use a registered stable schema: `seed`, ordered `june_keys`, `june_y`, `june_game_pk`, `june_pitcher`, `june_calibrated`, `june_raw`, `june_delivery_level`, with a state manifest of all hashes. Global fitting uses `calibrated[ensemble]` equal to the mean of ordered seeds. Frequency archive has identical keys/y and `june_frequency`. Labels are allowed in the sealed **fit** bundle under this new registration; the earlier eligibility inventory remains unchanged and label-free.

Fit manifest fixes predictors `['ensemble','seed0','seed1','seed2','seed3','seed4']`, B0 weights, B1 weights, B2 group MLE/shrunk weights/support/fallback, exact inputs/key/label hashes, optimizer settings/success/status/objective and source identities. Count30new versus6replay optimization calls separately; include failure calls. No DEV scores in this artifact.

Apply archive fixes keys and predictor order and stores `B0_primary`, `B1_primary`, `B2_primary` (N×10), `B0_seed_primary`, `B1_seed_primary`, `B2_seed_primary` (5×N×10), groups and parameter-manifest hash. Label-free application may preserve a label-archive reference hash without loading labels. The scorer separately joins pinned DEV labels and enforces all five seed outputs, both candidates and all3/78slots before emitting the complete result. Output state/manifest is written last and separately verified after process exit.

Scientific registration must freeze this contract, IDs, exact B definitions,30+6 fit count, unknown/fallback rules, N3/R78 and cost equations. Later execution registration additionally pins **actual** clean source C, config D bytes/path, all cache/parent/member/probability/label-mapping dependencies, numerical versions, canonical column/class/seed/key ordering, full command plan, supervisor bytes, environment, output paths and required completion fields. A config living in a registration checkout must be tracked/clean and hashed as actually passed; execution source must stay at C. Recheck before first new value access and before final score.

Meaningful synthetic/archived identity checks must reject shuffled or duplicate keys, seed-order swaps, old3-seed weight substitution, unsupported/unknown group confusion, fitting on DEV, B2 anchoring to B0 instead of B1, missing slots, malformed probabilities, failed optimizer success, parent writes, incomplete scoring, and cost ledger overruns. Review of this design and a committed config does not replace those executable checks or release authorization.
