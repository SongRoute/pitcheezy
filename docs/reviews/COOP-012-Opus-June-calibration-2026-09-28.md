# COOP-012 — Opus independent scientific preregistration review: ML-JUNE-CALIBRATION-v1

2026-09-28. Reviewer: Opus (hierarchy Astra → Opus → Sol; Fable excluded). Base `d5466e01bf3ad73262d6c4f4aacc1b43de6c7811`.

## Verdict

- **Scientific preregistration: CONDITIONAL PASS. Three blockers (B-1…B-3) must be fixed before registration.** All three are config/contract identity and provenance gaps with known fixes. They do not require a redesign. Once they are fixed, the scientific design is ready for registration. Math, B0/B1/B2 definitions, effect attribution, leakage boundary, Holm-3 / R78 multiplicity, decision gates and budget equations are internally consistent.
- **Runtime: NOT IMPLEMENTED.** No runner exists for June preparation/labels/features/frequency, full-June G0 inference, the scalar wrapper with diagnostics, B2 anchored on the new B1, apply, the R78 scorer, or the family supervisor. `execution.enabled=false` is correct. Nothing in this review authorizes profile, inference, fit or scoring.
- No metric was computed or read. All costs and effects of B1/B2 remain **unmeasured (미측정)**.

## Reviewed snapshot (review-input-001)

Snapshot dir: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260928-june-calibration/review-input-001`. All four SHA256 values were recomputed and match `manifest.json` (manifest SHA256 `77a0151ff71eb7095c7428876d138ab55d906d5ac37fc6973ffcbcfd8715d348`).

| Target path | SHA256 |
|---|---|
| `docs/contracts/ML-JUNE-CALIBRATION-v1.md` | `2a6a33e2f6d4b22c83f6b320f7bd2d5900f3907e2f62f87e4e470e75f2443703` |
| `configs/ML-JUNE-CALIBRATION-v1.json` | `7f89ef71b7b64009d1b3f749f06b65b228c232e0c6f161de17e08188112a2f50` |
| `docs/reports/ML-June-calibration-readiness-2026-09-28.md` | `ccff605af66a7a690d0d2a51c4f239929ed3b526b02652dfd783fb6a5da54147` |
| `docs/reports/ML-June-calibration-design-2026-09-28.md` | `01849020774fac9ba56b42d1f529d68e8739a747b78428bbec2ccdc2b24343c5` |

None of the four files are present in this worktree at base `d5466e0`. The review therefore covers the snapshot bytes above, not any later revision.

## Scope and access log (metadata only)

- Read: the four snapshot files; the source excerpts `run_sequence_calibration.py::fit_blend` (L31–51), `run_ml_g0_calibration.py::_june` (L260–318), `score_ml_matrix.py::summarize_cell` (L66–80), `matrix_g0_whole_metrics.py` family-size guard (L98), `run_ml_g0_whole.py` `PARTS` (L63), and `configs/EXP-P11-002.yaml` parent run paths. I also grepped count fields only in the sealed D81 `result.json`, which is label-free.
- File hashes were recomputed as opaque bytes and match the config/readiness pins: `run_sequence_calibration.py` `951bada5…`, `run_ml_g0_calibration.py` `67171381…`, `matrix_g0_calibration.py` `95beee1b…`, `run_ml_g0_whole.py` `5430bfbd…`, `ML-JUNE-ELIGIBILITY-v1.json` `afe19b81…`, `EXP-P11-001.yaml` `b6565f16…`, `EXP-P11-002.yaml` `75d51ecf…`, `G0-RESEARCH-FROZEN-v1.json` `c839dbf7…`, D81 `result.json` `d00606b8…`.
- Not done: no NPZ/parquet opened, no probability/label value read, no new field data, no aux/model deserialization, no inference/fit/profile, no web access, no tests, no implementation change.

## Arithmetic cross-checks (from registered numbers only)

| Item | Check | Result |
|---|---|---|
| June denominators | 115,816 − 104,970 = 10,846; 115,816 − 4,821 = 110,995; 104,970 − 4,821 = 100,149; 110,604 / 110,995 / 549 / 529 present as `actual==expected` in D81 `result.json` | consistent |
| DEV | 12,334 + 299,387 = 311,721 | consistent |
| Group support | 4,788 + 5,338 + 38,546 + 56,298 = 104,970; each ≥500 pitches AND ≥30 games | consistent; all 4 supported |
| Shrinkage ρ = n/(n+1000) | zero .8272, low .8422, middle .9747, high .9825 | convex, within [0,1] |
| Optimizer calls | 6 (B1) + 24 (B2) + 6 (replay) = 36 | consistent in contract/config/design |
| R family | 3 × 13 × 2 = 78; one-sided quantile 1 − .05/78 = 0.999359 | consistent; ≈64 tail draws of 100,000 |
| Cost proxy | 1,777.684 × 104,970/311,721 = 598.62 s **for all five members** (≈119.7 s/member) | arithmetic correct; see S-3 on wording |
| Budget worst case | prepare 300 + profile 600 ⇒ remaining 2,700 ⇒ gate `10T + 900 ≤ 2700` ⇒ T ≤ 180 s (binding over `2T ≤ 600`) | gate equations coherent; feasibility unmeasured |
| Tail reserve | fit + apply + score caps = 300 × 3 = 900 = reserve | consistent |

## Blockers for scientific registration

### B-1. Config does not pin the contract it registers
`config.contract.sha256` is `null` and `status` is `draft_pending_independent_review`. A scientific registration whose config does not hash its contract leaves B0/B1/B2 definitions and the N3/R78 gates unfrozen.
**Fix:** write the final contract SHA256 into `contract.sha256`, then commit the contract and config together. The contract does not embed the config hash, so there is no circularity. Record the registered commit and change `status` to a registered value. `execution.*` stays null/false.

### B-2. B0-defining and DEV-component parents are unpinned, and the frequency source is misnamed
B0 is defined as "archived five-seed C1 weights + archived frequency + archived members". The config's `parents` does not pin any artifact that carries those values. The readiness report already hashes most of them. The provenance is also inconsistent:
- The contract (§10) names a "P4 frequency archive". The pinned loader (`run_ml_g0_calibration.py::_june`, L263–273) and the C1 fitter (`score_ml_matrix.py::summarize_cell`, L69) both use `EXP-P10-001/parent_baseline_predictions.npz` (`blend`, and per readiness `mlb_dev`) as the frequency baseline. `EXP-P11-001/analysis/predictions.npz` has no frequency array (readiness header list), so DEV `f_i` for B1/B2 apply must also come from P10.
- The C1 weights come from `EXP-P10-001/analysis/results.json` → `reports['G0-global'].selection.model_weight` plus `seeds[*].blend_selection.model_weight` (loader L304–306). This file is not pinned.

**Fix:** add to `parents` (hashes already in readiness where available):
- `EXP-P10-001/parent_baseline_predictions.npz` (`bbf4f02a…`), declared as the sole frequency source for the Cpanel replay (`blend`) and for DEV apply (`mlb_dev`).
- `EXP-P10-001/analysis/results.json` and `analysis/manifest.json`, with the JSON paths of the six B0 weights.
- `EXP-P11-002/june_inputs.npz` (`0ea15b69…`).
- `EXP-P11-001/analysis/predictions.npz` (`6890f321…`).
- The five June Cpanel member archives (`6935d546…`, `40ea22c7…`, `32d04645…`, `749bab29…`, `c804aadf…`).
- The five P11 DEV member archives (`9bc36e1f…`, `2a7a1c70…`, `c3986ee9…`, `ea045bdc…`, `877a6c67…`).

Correct contract §10 from "P4 frequency archive" to the P10 file. Add a gate that the P10 `mlb_dev` keys equal the P11 ordered DEV keys. Gate 4 (B0 DEV replay to P11 `primary`/`seed_primary`) then proves that the frequency and weights are jointly the B0 ones.

### B-3. The replay-fit reference code is not identified
Gate 3 requires the six scalar replays to reproduce the archived C1 weights to `1e-8`. Those weights were produced by `score_ml_matrix.py::summarize_cell`. That function calls `fit_blend(cy, np.mean([m['blend'] …], axis=0), baseline['blend'], 'log_loss')` for the ensemble and `fit_blend(cy, member['blend'], …)` per seed. The contract pins only `fit_blend`. The `1e-8` weight replay also depends on the exact ensemble construction (`np.mean` over the stacked seeds 0..4 on axis 0, float64), the `objective='log_loss'` literal, and the SciPy version used at C1 time.
**Fix:** pin `score_ml_matrix.py` (path + SHA256 at the C1 commit) as the provenance of `w0`. State in the contract and config that the ensemble is `np.mean(np.stack(seed_calibrated[0..4]), axis=0)` in float64 and that `fit_blend` is called with `objective='log_loss'`. Where the C1 run manifest records the SciPy/NumPy versions, pin them. If it does not, state now that a replay mismatch under a different version is a family stop, not a tolerance change, as §3 already implies.

## Should fix before execution registration (not scientific blockers)

**S-1. Anchor-check tolerance versus the Brent tolerance.** `L_S(w)` is convex in `w` (−log of an affine function), so a feasible anchor beating the optimizer can only be optimizer imprecision. Bounded Brent with `xatol=1e-5` locates `w` to about 1e-5, which gives an objective gap of about ½·L''·(1e-5)². That gap can exceed `1e-9` when L'' ≳ 20, for example when the true-class probability is small for rare classes. With the current rule, this false failure aborts the whole family with no fallback. The endpoint check is automatically satisfied because `min([0,x,1])` already covers the endpoints. **Fix (decide now, not after seeing a failure):** keep the hard gate. Before execution registration, add a synthetic stress test with high-curvature cases showing the gate cannot false-fail at the pinned SciPy. Alternatively, preregister the anchor comparison as `L(chosen) ≤ L(anchor) + 1e-9·max(1, L(anchor))`. Either way, it must stay a check and never become a candidate choice.

**S-2. Full-June frequency probabilities and label derivation have no assigned stage or cost.** B1/B2 need `f_i` for all 104,970 June rows. The P10 archive holds only the 4,821 Cpanel rows, so a new frozen-frequency inference is required. Neither the contract stage table nor `budget.includes` names it. Label derivation (P4 `outcome_labels`) is also unpinned in the config.
**Fix:**
- Assign June frequency generation and label derivation explicitly to the 300 s prepare stage.
- Add both to `budget.includes`.
- Pin the frequency model state/source and `pitchmdp.model.outcome_labels` (source path + SHA256) in the config.
- Keep the existing "Cpanel frequency replay ≤1e-6" gate.

**S-3. The profile model can undercount row-proportional overhead, and the proxy wording is ambiguous.** `T = L + 104970·r` treats everything outside the timed 8,192-row interval `I` as constant. Feature/history-store construction (the `run_ml_sharing` loader builds a history store from the cached frame) likely scales with rows. If it runs outside `I`, the profile underestimates `T`. The later `M`-based gates self-correct only after seed0 has already run. Separately, the "≈599 s proxy for 104,970 rows" in contract §9 and the design doc is the **five-member total** (≈120 s/member). It comes from one P11 worker that amortized load across seeds, so per-worker load overhead is not in the proxy.
**Fix:**
- Either define `I` to include all row-proportional per-member work (feature assembly, delivery draws, output), or have prepare materialize and hash the June feature inputs once and forbid per-member rebuilding.
- Reword the proxy as "≈599 s total for five members in one process; ≈120 s/member before per-worker load; context only."

**S-4. Contrast identity audit.** The rows, denominators and bootstrap draws are identical, and the loss is a pitch-weighted ratio. So Δ(B2−B0) = Δ(B1−B0) + Δ(B2−B1) exactly, both at the point estimate and within every draw. Holm is valid under this dependence and the design choice is acceptable. Including C2 costs power relative to a Holm-2 family, and that choice is now fixed. **Fix:** add this additive identity (abs. tol ~1e-12) as a scorer self-check and a required completion field. It catches row or draw misalignment for free.

**S-5. Config does not yet cover every gate in the contract.** The following appear in the contract but not in the config:
- the p-value formula `(1+count(δ*−δ̂ ≤ δ̂))/(B+1)`
- the percentile method (`numpy` linear) for both CI and R quantiles
- the fixed Holm order
- the classification thresholds (`worse`: NLL lower > 0 or Brier lower > +.001)
- ordinary two-sided lower bounds as deterioration evidence in R
- per-class ≥30 events
- the `objective='log_loss'` literal
- the complement definition key source (`p4_panel` for the DEV Cpanel keys)

**Fix:** mirror these into the config so a scorer can be tested against the config alone. Executable checks should treat the config as authoritative for numbers and the contract for semantics.

**S-6. Point-in-time June features need an executable test, not only prose.** The G0 loader supports only `PARTS=('mlb_dev','dev')` (L63), so June needs a new path. **Fix:** in the execution registration, add a synthetic future-row injection test: perturbing any same-day-or-later row must not change a June feature. Also add a check that no DEV key or date enters any fit input. This is listed in §3 but has no named test.

## Items reviewed and accepted (no change requested)

- **Math/algorithms.** The B0/B1/B2 formulas, objective and clip `1e-12` on the true class match the pinned `fit_blend` exactly (L41–48). The `[0, x, 1]` tie order matches `min(..., key=score)`. Shrinkage toward the **new** `w1^a` is convex, and unsupported groups fall back to exactly `w1^a`. The ensemble is blended after averaging, not averaged after blending. The unknown-metadata fail-closed rule is separate from the valid `zero` group, matching `lookup.get(pid, 'zero')` in the existing loader.
- **Leakage / June reuse.** June is fit-only, and no June validation or generalization claim is made. B1/B2 refit from pre-blend components rather than stacking on June-fitted B0. There is no DEV-driven hyperparameter, group or shrinkage choice, and no 2026 access. Fit-manifest sealing comes before DEV labels, with separate label-free apply and separate score. The adaptive choice after P11 and repeated DEV exposure are disclosed (`fresh_alpha_claim:false`, `independent_confirmation:null`).
- **Effect attribution.** C1 is correctly labeled as a joint size + composition change, not a sample-size effect. C3 isolates the method at fixed rows. On DEV the components are identical, so each contrast reflects only the fitted weights. B2 requires both C2 and C3, which prevents promotion by beating B0 alone.
- **Multiplicity and decision gates.** The Holm-3 family is fixed with unmeasured slots at p=1 and the denominator retained. R78 uses Bonferroni with unsupported slots kept as `unconfirmed`, and per-contrast R requires all 26 measured. The 4/5 seed rule is labeled as a stability diagnostic, and promotion/service replacement is excluded.
- **Budget equations.** The gates are coherent with the stage caps and the 900 s reserve. Prediction timeout `min(600, remaining−900)` is consistent. No row/seed/draw reduction or automatic retry is allowed. Prior D81 cost is disclosed and not charged. The authoritative Popen→wait boundary is defined.
- **Audit language.** The readiness report states that only NPZ headers (names/shape/dtype) and file bytes were read, marks new cost as 미측정, and does not describe any runner as execution-ready. The design doc separates scientific and execution registration. No invented metric was found.

## Runtime components confirmed missing (execution registration prerequisites)

1. A June prepare/binding step covering labels, features/history, frequency and hashes. The existing `_june` loader hard-requires 4,821 rows (L265).
2. A full-June G0 inference path. `run_ml_g0_whole.py` `PARTS=('mlb_dev','dev')`.
3. An additive scalar wrapper that records `success/status/nfev/finite` (`fit_blend` returns none of these).
4. B2 fit anchored on the new B1 with an explicit dependency check.
5. Label-free apply to the P11 DEV components.
6. A Holm-3 + R78 scorer. `robust_bounds` rejects any family size other than 24/52 (`matrix_g0_whole_metrics.py` L98).
7. A family supervisor/ledger with the 3,600 s gates.

Each still requires reviewed source commit C, pinned environment, config D, a command plan and explicit release.

## Required actions summary

| ID | Severity | Where | Action |
|---|---|---|---|
| B-1 | blocker (scientific) | config `contract` | Pin contract SHA256; register |
| B-2 | blocker (scientific) | config `parents`, contract §10 | Pin P10 frequency / C1 results / P11-002 inputs / P11 analysis / 10 member archives; fix "P4 frequency" → P10 file; add P10 `mlb_dev` ↔ P11 DEV key gate |
| B-3 | blocker (scientific) | contract §3–4, config `optimizer` | Pin `score_ml_matrix.py::summarize_cell` as w0 provenance, ensemble construction, `objective='log_loss'`, C1-time numerics if recorded |
| S-1 | before execution | contract §4 | Stress-test or preregister a relative anchor tolerance |
| S-2 | before execution | contract §9, config `budget` | Assign/pin June frequency + label derivation to prepare |
| S-3 | before execution | contract §9, design | Row-proportional overhead in `I` or pre-materialized features; fix proxy wording |
| S-4 | before execution | scorer spec | C2 = C1 + C3 identity self-check |
| S-5 | before execution | config | Mirror remaining gate constants |
| S-6 | before execution | execution tests | Point-in-time and no-DEV-in-fit executable tests |

---

## Closure addendum — review-input-002 (2026-09-28)

The original findings above are preserved unchanged. This addendum reviews only the closure snapshot. Same access limits apply: no probability/label values, no new field data, no deserialization, no optimizer, inference or tests.

### Reviewed snapshot (review-input-002)

All SHA256 values were recomputed and match `manifest.json` (manifest SHA256 `88614d27f2de2eeb635ee924be44462d38c4a81cc13e35ebd41e97fb7f1e1b66`).

| Target path | SHA256 |
|---|---|
| `docs/contracts/ML-JUNE-CALIBRATION-v1.md` | `e155315dcf732c55009755443f4db8117f260851273d7c54c760e2f29c701d4c` |
| `configs/ML-JUNE-CALIBRATION-v1.json` | `49400f0ea8113bb0d5006d8abd09b0f048fb08881f21c2094fd88bac0458cc24` |
| `docs/reports/ML-June-calibration-readiness-2026-09-28.md` | `d01368345bec1beb8a95db3d100e47a1dbc4052413c2ac734037a4d27d706505` |
| `docs/reports/ML-June-calibration-design-2026-09-28.md` | `2fecc1b9a05027afc4c49718ec59769c7e0bd2f55fc1040542a4cf88f6158152` |

Method: a structural JSON diff of config 001→002 plus a text diff of contract 001→002. I also read cost fields only from the whole-MLB ledger `coordination/20260927-whole-mlb/execution/attempt-20260927T111115Z/status.json` (SHA256 `c784953daebb8445de26ce87fe00b6550c2f9dc90acf6ed25061df1d05aaed70`), to check the correction below.

### Correction to my original S-3

My original S-3 said the 1,777.684 s proxy came from "one P11 worker that amortized load across seeds, so per-worker load overhead is not in the proxy". **That was factually incorrect.** The ledger shows five separate timed prediction workers (`steps[2..6].worker_wall_seconds` = 354.287, 356.446, 354.834, 355.884, 356.233), which sum to 1,777.684 s. Each worker already includes its own interpreter/import/load/output overhead. The ≈599 s figure is therefore a linear row-ratio total for five separate workers at 104,970 rows, with no amortization. The contract (§9), design and readiness snapshots now state this correctly. The remaining parts of S-3 (profile overhead modeling and wording) are closed below.

### Closure status

| ID | Status | Evidence in 002 |
|---|---|---|
| B-1 | **closed** | `contract.sha256` = `e155315d…` equals the recomputed contract snapshot hash. `status` = `scientific_preregistration_pending_final_review`. `execution.enabled=false`, and C/D/plan are null. |
| B-2 | **closed** | 33 direct `parents`, including `p10_frequency` (`bbf4f02a…`), `p10_results`, `p10_manifest`, `p10_preparation`, `p11_cpanel_june_inputs`, `p11_predictions`, 10 member archives, `p4_auxiliary`, `p4_model_source` and `p4_frequency_temperature_source`. `B0_provenance` names P10 `blend`/`mlb_dev`/`mlb_dev_keys` and the weight JSON paths. Contract §3 gate 4 and §10 now name canonical P10 and require `mlb_dev_keys` = P11 DEV keys, with `dev_keys` defining the DEV Cpanel. The P4/P10/P11 byte-identity is asserted by the readiness report; I did not re-hash the non-P10 copies, and the P10 path is canonical regardless. |
| B-3 | **closed** | `p10_scalar_summarizer_source` (archived C1 `score_ml_matrix.py`, `379ab50b…`), `function: summarize_cell`, the ensemble expression, `objective_argument: log_loss`, and the recorded NumPy 2.5.3 / SciPy 1.18.1 / Python executable are pinned. A mismatch stops the family and never widens tolerance. |
| S-1 | **closed (accepted design choice)** | The absolute `1e-9` anchor check is kept as a deliberately conservative family stop, and the contract states that convergence does not guarantee it. Stress tests are required and are explicitly not proof that false fails cannot happen. I accept this: it can only cause false *stops*, never false *passes*. |
| S-2 | **closed** | Label derivation (P4 `model.py::outcome_labels`) and frozen-frequency prediction (`aux.pkl['baseline']` + archived temperature, no fit) are assigned to the 300 s prepare stage, and both appear in `budget.includes`. Label/frequency hashes are sealed before any fit. |
| S-3 | **closed** | `T = P_profile·(104970/8192)` uses the official full-worker wall. The profile must build the same complete input store and production path as a full member, and internal timings are diagnostic only. This is strictly more conservative than the earlier L+r model. Proxy wording is corrected. |
| S-4 | **closed** | `contrast_identity_check` covers NLL and Brier, at the point estimate and every aligned draw, atol `1e-12` / rtol 0, with completion field `contrast_additivity_passed`. A failure stops scoring. |
| S-5 | **closed** | The config now holds `p_value_formula`, `ci_quantiles`, `quantile_method` (primary and R), `holm_order`, `worse_rule`, R two-sided deterioration reporting, `diagnostics.minimum_class_events=30`, `objective_argument`, and `cpanel_key_reference`. |
| S-6 | **closed (as a requirement)** | `required_execution_tests` lists: the two-clock point-in-time test (same-day/later rows cannot change a prior-date aggregate; future pitches cannot change query history, while legal earlier-pitch history is allowed), no DEV keys/dates in fit inputs, unknown group vs zero group, additivity, abort-before-apply/score, profile/full input-store identity, and full Cpanel replay. These tests are specified but **not implemented**. |

### Non-blocking notes (no change required for scientific PASS)

1. **The conservative profile gate implies a hard feasibility ceiling.** From the registered equations, the gate before seed0 is `10·P·(104970/8192) + 900 ≤ 3600 − Pprep − P`, which gives **`P_profile ≤ (2700 − Pprep)/129.14`**: at most 20.91 s if prepare took 0 s, and lower as prepare time grows. The other condition, `2T ≤ 600`, gives P ≤ 23.41 s and is not binding. Because the profile must build the complete June input store, fixed load/build time is multiplied by about 12.8. The gate may therefore stop the family before seed0 even when a full run would fit in 3,600 s. That is a conservative false stop, not a validity risk. Actual June profile cost is **unmeasured**. Optionally, the future execution plan could run the profile before label derivation, so a profile-gate stop does not incur a June label-access event. Either way, the ledger must record the exposure.
2. **Final status change.** The registered config will differ from the reviewed bytes (`49400f0e…`). The registration step should record a byte diff showing that only `status` (and registration metadata, if any) changed relative to `49400f0e…`, with the contract hash unchanged at `e155315d…`. Any other difference needs re-review.

### Closure verdict

- **Scientific preregistration: PASS** for contract `e155315dcf732c55009755443f4db8117f260851273d7c54c760e2f29c701d4c` and config `49400f0ea8113bb0d5006d8abd09b0f048fb08881f21c2094fd88bac0458cc24`, conditional only on note 2 (the final registration may change `status` and nothing else). No remaining scientific blockers.
- **Runtime: still NOT IMPLEMENTED.** All seven runtime components listed in the original review, plus the `required_execution_tests`, still need reviewed source commit C, pinned environment, config D, a command plan and explicit release. This PASS authorizes no data-value access, profile, inference, fit or scoring. All B1/B2 effects and all June costs remain **unmeasured (미측정)**.
