# C — first event-analysis increment, 2026-09-23

## Follow-up: fixed replacement evidence, 2026-09-24

[C-EVIDENCE-003](C-evidence-v3.md) audits the six previously fixed pitching changes from stored sources. Its v2 evidence artifact adds bundle-metadata verification and explicit per-candidate comparison blockers while preserving the first output. It records 65 retrospective roster-screened pitchers with prior-seven-day workload, rest, and previous-day consecutive use; decision-time manager availability is unknown for all. Only one keep and one screened substitute have frozen pitcher/repertoire support, in different games. Therefore there is no valid same-state `inning_end` keep/substitute comparison, and replacement value remains null. The prior Wheeler calculation is a conditional keep-only bound. The separate evidence JSON changes no event-analysis or Observer API contract.

## Delivered

- C1 result contract `docs/contracts/event-analysis-v1.md` and synthetic `docs/contracts/examples/event-analysis-v1.json` (initial commit `7075730`). The C1 schema was sent to D before implementation.
- C2 `apps/observer/backend/observer_app/event_analysis.py`: versioned `analyze_event`, tagged `value_point`, frozen observed WE adapter, saved recommendation hash check, intent/pitch/frame validation, signed ordered contrasts, null shares under missing/tiny denominators, explicit unavailable/failed paths. It uses the frozen engine WE and does not train or add a new model.
- C3 `unavailable_replacement` requires keep/substitute identities, decision time, prior roster snapshot, eligible candidates, workload/rest evidence and a separate inning evaluator. It currently returns unavailable, even with those inputs, because an inning-end replacement evaluator has not been validated. PA and inning totals remain separate.
- Test `apps/observer/backend/tests/test_event_analysis.py` covers sum/sign/%p, zero denominator, absent intent, wrong pitch/clip/frame/model/defender, misuse of actual as intent, immutable recommendation/revision, frozen engine identity, and C3 timing. `scripts/c_real_event_smoke.py` exercises one real A packet through the frozen standalone WE.

## Calculation and limits

The same-pitch saved recommendation baseline is the reference in production. `total_pp=100×(observed post-PA WE−saved baseline)`, for the team defending before the event. The model-based path is reference → planned action → delivered action → observed outcome. Strategy and execution are model contrasts; the last component is an outcome residual associated with the batter event, **not causal batter skill**. Interactions/order effects remain unallocated. A complete synthetic fixture confirms arithmetic, not real attribution validity.

Real Statcast has no accepted intent. The A packet used for the smoke is strikeout `777227:17:6` in S0; recorded post state, fixed defender, and model input were present. The retrospective type-frequency reference was `0.6050701823`, observed WE `0.6259257186`, total `+2.085553628` pp (defense). Status `partial`; strategy/execution/outcome components and shares null; unallocated residual equals the total. This smoke rebuilds a same-pitch baseline retrospectively and explicitly marks `development_only=true`; it is **not** a saved live recommendation or policy effect estimate. Output: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/A-events-v1/c_smoke.json`. Its type-frequency baseline is distinct from Observer's joint type×location baseline; do not compare or combine them as the same policy.

## Integration boundary

For D, call `analyze_event(linkage, identity, initial_defender, values, evidence, provenance, stored_recommendation, ...)`. Obtain `identity.model_sha256` from the loaded frozen `bundle_manifest.json`; B exposes this and `baseline_policy_id='observer-repertoire-kernel-v1'` on its typed pre-pitch result. Tag the saved same-pitch recommendation `baseline_value` with `value_point(..., source='saved_same_pitch_recommendation_baseline')`. Build `GameState` for the actual post-PA state and call `frozen_observed_value(engine=Recommender.engine, initial_state=..., post_state=..., identity=...)`. If that state is missing or game-ending status cannot be represented faithfully, pass observed null. With no linked pre-release intent, pass plan/execution null. `recommendation_sha256(stored_recommendation)` must match linkage. `event_input_revision` starts at 1 and increments for late/corrected inputs; recommendation ID/hash stays fixed.

The D producer owns durable analysis version storage and asynchronous update semantics; C returns a pure result and does not mutate the service/store. B's typed candidate `Q` array/action mapping may supply compatible model contrasts after valid pre-release plan evidence arrives. No current accepted CV estimate supplies that evidence. Full C4 validation still requires real reviewed setup labels and sensitivity checks. Replacement evaluation additionally requires contemporaneous available candidates and a separately validated inning-end continuation.

## Verification

```sh
PYTHONPATH=apps/observer/backend:apps/observer/runtime_src \
  /Users/song/Projects/pitcheezy/.venv-observer-standalone/bin/python -m pytest \
  apps/observer/backend/tests/test_event_analysis.py -q
/Users/song/Projects/pitcheezy/.venv-observer-standalone/bin/python scripts/c_real_event_smoke.py
```

At handoff: 16 targeted tests passed; standalone real smoke completed. No 2026 data, retraining, or live CV ingestion was used.

Final boundary review: a `synthetic_contract_fixture` intent or `synthetic_fixture` plan action requires `evidence.development_only=true` and `use_for_performance_evaluation=false`; it cannot be promoted to production attribution. An `unavailable` IntentEstimate with the correct pitch ID can return its explicit reason without a release frame or clip mapping, because it supplies no plan action. A supplied mismatched pitch or clip ID still raises. The focused event-analysis suite now has 22 passing tests; the real model smoke was not rerun for this guard-only change.
