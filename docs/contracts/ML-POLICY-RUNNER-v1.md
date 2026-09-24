# G → pitch-type policy runner implementation contract

2026-09-24. Additive implementation awaiting root review and actual execution registration. This document does not assert that preparation, resource profiling or real-data policy evaluation has run. The implementation subagent read frozen service JSON lineage/source metadata only; it did not load data, deserialize models or execute inference.

`run_ml_policy.py` implements `prepare`, `profile`, `p0-evaluate`, and `policy-run --split blend|dev --world control|candidate`. Every CLI stage acquires the common ML matrix heavy lock. Inputs remain read-only; outputs are a distinct run sibling. Existing or interrupted stages cannot be overwritten. Exceptions preserve partial files and create timestamped failure audits. No new WE, outcome-network or delivery-model fit is implemented. Only preparation fits the explicitly registered categorical TRAIN BC.

## Frozen dependencies and scientific interpretation

Candidate selection reuses `run_ml_transfer.chosen_comparison`: first G followup candidate, or lowest Cpanel NLL diagnostic candidate with dependency-fit-time tie break. Controls remain G1→G0, G2→G0, G3→G2, G4→G2. Preparation verifies parent preparation/analysis/member/dependency hashes and archives its own sources.

Candidate predictions plan Q^BC; the mapped control provides the **same common primary evaluation simulator for all policies**. Candidate-world evaluation is a sensitivity analysis using the same policies and tuned tau. These worlds share TRAIN, calibration and WE continuation; this is cross-model internal evaluation, not independent observed or causal policy validation.

Frozen G calibration is reused exactly: each seed's `calibration.json.delivery_temperature`, `preparation.json.baseline_temperature.temperature`, and `analysis/panel/results.json.reports[cell].selection.model_weight`. No policy split refits those outcome calibrators. For a selected physical draw, per-seed conditional probabilities are temperature-corrected, averaged, then blended with the temperature-corrected frequency model. Model-weight 1 means neural-only.

The safe context whitelist contains game/count/hand fields, IDs for joins/routing and prior-date batter profiles. It excludes current realized type, physics, outcome, current missingness and `supported_pa`. Candidate type is encoded as the current action; supplied normalized TRAIN delivery is the current physical token. Current outcome channels are zero. Past H5 type/physical/outcome tokens use strict same-PA earlier indices; generated tokens replace logged future history. Cached context updates balls/strikes at each simulated pitch; other game/profile context remains fixed through the supported PA.

Common action support is the intersection of positive TRAIN BC support, frozen TRAIN enriched-token vocabulary and an action-specific frozen delivery pool for **all 12 legal counts**. League type/hand fallback is allowed and recorded; global all-type physical fallback is rejected. This keeps one action repertoire across policies/counts. Minimum BC action count 1 is deliberately permissive; rare-action Q uncertainty and outcome-model extrapolation remain concerns, not verified causal overlap. Unsupported requests and unsupported observed action labels remain in the denominator, and full-request NLL is null if any label is unsupported.

## Verified WE lineage

`docs/contracts/model-v1.json` pins bundle `minimal-pitch-service-v1` and `game_values.pkl` SHA `aa6c4e486cc6b027758d966ab680566a4a4ce2afa4a918dd26f8fb65c8957677`. Its actual `bundle_manifest.json` and `source_hashes.json` metadata hashes match the contract. The matching builder source calls `assign_fold(raw, 2025)` and fits WE plus advancement on `frame[split == 'train']`. Matching fold source defines inclusive TRAIN **2023-05-15 through 2025-04-30**; June/July–September 2025 outcomes are excluded from those fits. Current `game.py` hash matches the pinned frozen implementation. The builder did expose 2025 DEV in an old Brier acceptance gate; that selection exposure remains a limitation.

During real preparation the runner rechecks the actual game-values artifact bytes before later deserialization and uses only its `we`/`advancement` objects. The stored RE24 object is ignored. Failure stops the run; no incompatible substitution/new fit occurs. Terminal utility remains absolute PA-end WE for the initial defender. Cap-tail approximation is frozen WE at the initial game state, independent of count; every capped trajectory also receives worst-case [0,1] bounds.

## Selection and resource profile

The runner constructs all requested Cpanel PA starts in May16–31 (resource profile), June (policy tuning), and July–September (DEV). It hashes game/PA IDs using the frozen selection seed, alternates the two frozen TRAIN role strata, and takes the registered prefix **before** support/outcome checks. A selected unsupported PA is not replaced. Complete request tables retain selection rank, selected flag, support flag and exclusion reason. Existing whole-PA support is retrospective and is labeled accordingly. P0 behavior-model scoring separately retains every requested Cpanel pitch, regardless of outcome-model eligibility.

Preparation registration fixes maximum requested prefixes; a separate execution registration fixes final smaller/equal prefixes after resource profiling. The May profile runs actual candidate/control callbacks, planning and MPS work, but publishes only elapsed time, memory and row counts—no NLL, policy WE/ranking or quality screen. Suggested initial profile: 2 requested May starts, 4 evaluation rollouts, 2 search rollouts/action, search/evaluation cap 8, maximum 1,000,000 conditional rows and 1,800 seconds. These are a proposal until root registers them. Empty supported profile fails.

Actual costs distinguish conditional rows, seed-wrapper rows, and routed neural subnetwork rows: a G composite predictor can call global plus cluster/personal networks, so three seed wrappers are **not** always just three neural forward rows. Final resource sizing must account for the entire tau-tuning family and repeated P1/P2/P3 evaluations, not just one search. Every stage shares one aggregate conditional-row/time budget. No hidden reduction occurs at a limit; preserve failure and explicitly register a fresh attempt.

## Registration schemas

Preparation JSON fields (root fills real experiment ID, paths and SHA256 values):

```json
{
  "protocol": "ml_policy_prepare_v1",
  "experiment_id": "ROOT_TO_ASSIGN",
  "parent_run": "FROZEN_G_RUN",
  "parent_preparation_sha256": "SHA256",
  "parent_analysis_sha256": "SHA256",
  "candidate": "G_CANDIDATE",
  "control": "MAPPED_G_CONTROL",
  "selection_status": "screen_promoted_OR_diagnostic_only_not_promoted",
  "seeds": [0, 1, 2],
  "draws": 400,
  "value_spec_version": "defense-we-pa-v1",
  "we_contract": "ABSOLUTE_PATH_TO_docs/contracts/model-v1.json",
  "we_contract_sha256": "SHA256",
  "evaluator": "mapped_control",
  "common_support": "bc_and_token_vocab_and_type_delivery_all_counts",
  "bc": {"prior_strength": 20.0, "minimum_action_count": 1},
  "selection_seed": 20260924,
  "maximum_requested_starts": {"temperature": 2, "blend": 64, "dev": 64},
  "profile": {
    "requested_starts": 2,
    "evaluation_rollouts": 4,
    "search_rollouts": 2,
    "search_cap": 8,
    "evaluation_cap": 8,
    "planning_seed": 701,
    "evaluation_seed": 1701,
    "row_budget": 1000000,
    "seconds_budget": 1800
  }
}
```

Final execution JSON contains `protocol="ml_policy_execution_v1"`, `preparation_sha256`, `profile_result_sha256`, `requested_starts={blend:Jt,dev:Jd}`, positive integers `search_rollouts`, `search_cap`, `evaluation_rollouts` (at least 2), `evaluation_cap`, `row_budget`, `seconds_budget`, `planning_seed`, `evaluation_seed`, `tau_grid=[0.001,0.003,0.01,0.03]`, `bootstrap_draws=10000`, `bootstrap_seed=20260924`. Freeze/commit it before June policy values are opened. Tune and DEV must use the identical execution hash. The June control-world results hash is additionally saved as a DEV dependency.

June selects the tau with the largest original unpenalized mean defensive WE; exact ties prefer larger tau (closer BC). Tau is in absolute WE units: .001/.003/.01/.03 equals .1/.3/1/3 percentage points. Paired simulation MC error is descriptive, not a reason to retune on DEV.

DEV primary family is P1−P0, P2−P1, P3−P2. The runner saves each PA/policy's trajectory WE, truncation flag and pitch count, plus game membership. It reports PA-weighted whole-game bootstrap intervals (10,000 shared resamples), one-sided null-centered p-values with Holm over these three tests, and separate stratified paired simulation MC SE. A model-internal preliminary P screen requires mean ΔWE≥.0001 (=.01 percentage points), bootstrap lower bound>0 and Holm≤.05. Truncation-difference worst-case bounds accompany this screen. An untruncated-PA model improvement is confirmed only if the worst-case mean delta lower bound also exceeds zero; otherwise even a passing imputed point/CI/Holm screen is explicitly tail-assumption-dependent. Final cap may be 24/32 if registered after profiling; existing primitive defaults are unchanged. Planning-seed uncertainty is not integrated, and three ensemble members do not provide independent policy seed confirmation. Bootstrap holds models/calibration/selection fixed and is start-state variation under a simulator, not observed policy-effect inference. Candidate-world success flags are sensitivity only, never a second chance at primary success. OPE, causal P and adoption remain null.

## Verification status

Synthetic-only tests cover frozen H5 token equivalence, count/history updates, poisoned-current-input invariance, exact400/type support, calibrated conditional blend, as-of history/date rejection, outcome-independent PA selection, and PA-weighted game bootstrap. Full real-data shape/provenance/performance validation remains the runner owner's next step under the heavy lock.
