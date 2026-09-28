# Astra independent post-run review — 2026-09-28

**PASS for execution integrity and registered scientific decisions. G0/B0 remains the research baseline; neither B1 nor B2 qualifies as a research improvement candidate.** Ten stages completed within the registered cost cap. No sealed-artifact or analysis-decision blocker was found. This audit reads sealed JSON, checks byte identities, and independently evaluates registered rules against recorded scalar results; it does not independently recompute predictions, losses, bootstrap intervals or p-values from arrays.

## Identity and seal checks

Actual source C remained `1002f57ea09d14df279212c68100da755bc573dc` in the clean execution checkout. Root registration HEAD remained D `61b056caa5d752787d84c7ae74b2a627818c34b3`; execution config, scientific config and contract still equal their D Git blobs. Expected post-terminal report edits in the root checkout are not experimental mutations.

Rehashed 419 source/parent input path entries from the preparation manifest plus frozen G0 bundle, with every digest matching. The worker's full source map is covered after resolving the venv Python symlink to its registered binary. Verified all ten completion manifests against their official ledger identities and all **20 sealed output artifacts** against manifest hashes, without decoding NPZ/parquet/pickle data. Verified member/preparation/fit/apply/analysis links, exact seed order 0–4, and required completion fields. The full hash inventory is retained in `astra-execution-final-checks.json`.

| Sealed record | SHA-256 |
| --- | --- |
| analysis/results.json | `31f72b0202a2cc7c06a595094ab36be95d3edd6de2a9455f173da8bc15c034e9` |
| analysis/manifest.json | `a296c8ec4c3a379904706b4a88fc2f62e1bacb5c93c24d510345e333efa37fcc` |
| fit/parameters.json | `fab0aff37e3b381d7f9dcb53677a436a36357e7e8a50ad3ed3bd4292e1121e9f` |
| fit/manifest.json | `eb0dec3ccac973643ede9263ef4103665fae2f46f340db7e34aed9eb1eeae6ea` |
| queue status.json | `de0c5fe28ee9c53ec273eb1c4f6536e137c123bc47102ab5c4be7ed4b6959609` |

## Cost and stage gates

Exactly ten matching start/end ledger entries completed with exit 0, valid clocks and their registered stage caps. Independently summed official worker cost is **473.5211259601638 seconds / 3,600 seconds**. Five inference workers total 398.08364387601614 seconds. Prepare/profile/fit/apply/score cost 7.379271334, 8.753600334, 3.431040875, 4.751637416 and 51.121932125 seconds respectively. Every recorded profile/member budget gate agrees with the registered formula and remaining budget. No failed launch or retry appears in this single queue's ledger.

Supervisor postflight sums to 0.26420078962109983 seconds and is separate. Invocation-record UTC to terminal-status UTC spans 475.243574 seconds; this is a timestamp cross-check, **not** an independently timed full supervisor process wall. Internal overlapping timers are not added to official cost. This metadata audit and prior D81 preparation are not charged again to this experimental family.

## Preparation, inference and fit checks

Sealed preparation reports 104,970 June queries, 394 games and 549 pitchers, with all-query compact/full-store input equivalence. Profile uses the same store digest, first 8,192 queries, 400 draws and first 64 archived Cpanel probes; recorded probability/delivery probe checks pass, probabilities are discarded, and no quality is computed. Every full member reports 104,970 queries and 400 draws, shares the same store, and agrees with the frozen checkpoint hash and May temperature in pinned P11 provenance.

All five Cpanel replay reports pass exact 4,821-key/label/delivery-tier checks; maximum recorded probability difference is 8.46164993628662e-09 against the fixed 1e-06 tolerance. This is tolerance equivalence, not a claim that every recomputed member probability is bit-identical. Six archived scalar weights replay with maximum difference **0.0**. Application replays archived B0 ensemble and each seed with maximum probability difference **0.0**.

Fit ledger has exactly 36 unique calls: six archived 4,821-row replay optimizations and 30 candidate calls. All recorded SciPy success, finite feasibility, endpoint and fixed 1e-09 anchor checks pass. The six B1 fits use full June; 24 B2 fits use four supported TRAIN-volume groups. Every B2 weight satisfies recorded n/(n+1000) shrinkage toward the corresponding new B1 weight. All groups satisfy 500 pitches AND 30 games; fallback rows/groups are zero. Recorded DEV keys and games in fit are zero. June objectives remain in-sample optimizer diagnostics, not performance estimates. Fit was sealed before label-free apply, and both precede the explicit DEV label-archive scoring access.

## Independent decision interpretation

Whole DEV is 311,721 pitches / 1,161 games. The exact Cpanel overlap is 12,334 pitches; its complement is 299,387. These populations share 328 games and are not independent confirmations.

| Contrast | Recorded ΔNLL | Recorded 95% CI | Holm p | Negative seeds | Failed N gates |
| --- | ---: | --- | ---: | ---: | --- |
| B1−B0 | -0.0005790967765 | [-0.0007044085814, -0.0004564154316] | 0.000299970003 | 5/5 | ΔNLL ≤ -0.003 |
| B2−B0 | -0.0005675643175 | [-0.0007146727402, -0.0004225818170] | 0.000299970003 | 5/5 | ΔNLL ≤ -0.003 |
| B2−B1 | +0.0000115324590 | [-0.0000468903159, +0.0000706698701] | 0.6484351565 | 1/5 | Effect threshold, upper CI < 0, Holm ≤ .05, ≥4 negative seeds |

Independently evaluated all five N criteria and Holm-3 arithmetic from recorded summaries. All three registered N decisions are correctly **inconclusive**. Brier protection passes in each contrast. B1/B2 show small favorable exposed-DEV differences against B0 but fail the fixed minimum improvement magnitude; the added B2 volume method has no established advantage over B1. B1 requires its B0 N/R contrast; B2 requires both B0 and B1 N/R contrasts. Both candidate flags correctly remain false, with no service promotion.

R retains all 78 slots (3 contrasts × 13 groups × 2 losses), all measured, using 100,000 recorded game draws and one-sided .05/78. Every recorded upper bound is within its registered margin (NLL .010, Brier .002). Maximum upper bounds are .001282371331 NLL and .000302181798 Brier. R passes do not mean every slice improves: B2−B1 left-handed NLL/Brier and high-volume Brier have positive marginal 95% intervals. These remain descriptive observations rather than additional multiplicity-controlled discoveries.

N and all 13 R point/draw additivity diagnostics pass at 1e-12 (largest recorded N/R error 1.919037845299343e-17); the three descriptive population additivity records also pass. Independently checked point contrast arithmetic from JSON. The bootstrap streams themselves were not regenerated. Scientific claims remain conditional on fixed models/calibrations and already exposed DEV; no fresh alpha, independent confirmation, pure data-size causal effect, targeting-policy benefit or service adoption is established.

## Exposure and report review

The actual preparation exposure record discloses 2,145,111 approved regular-season rows from 2023-03-30 through 2025-09-28, joined 73 stored columns (processed 71 plus sidecar 5 with three overlapping keys). Later-than-June rows were decoded in memory and then excluded before feature construction. The full feature store contains 1,803,170 rows; the saved store contains 104,970 after the all-query equivalence check. This does not make DEV unexposed. No 2026/raw fetch/new network or temperature fit is reported.

Profile explicitly reports June label archive unopened; full prediction workers explicitly report opening it to copy sealed june_y into their archive, without quality selection or scoring. Prior-pitch outcome fields also exist in the shared history store. The six replay fits use archived Cpanel rows, distinct from the expanded June candidate fits.

Reviewed the root execution report and exposure report drafts. Their result tables, absolute NLL/Brier/ECE values, cost, maximum RSS, R interpretation, retained-baseline conclusion and proposed next-step limitations agree with the sealed evidence. Requested two precise exposure wording fixes before publication: distinguish profile's unopened June archive from prediction workers' archive-copy access, and distinguish 30 expanded/group candidate fits from six archived-4,821 replay fits. These are report clarifications, not defects in sealed execution or analysis. A newly partitioned already exposed population is correctly not described as fresh confirmation; future acquisition is a separate unregistered task.

## Audit boundary

Machine checks: `astra-execution-final-checks.json`, SHA-256 `5f1ed7deaa35336de6f440aff7b557dfc33003cbe27c455744f3ea63ae11ca6b`; reproducible metadata checker: `astra_postrun_audit.py`. Its successful check took 0.851727334 seconds, with no new numerical-data decoding, inference, fit or bootstrap. An initial reviewer-only path assertion was corrected to resolve the venv symlink before comparison; no experimental rerun occurred. No sealed artifact, registered source or root report was modified by this reviewer. The audit establishes consistency of sealed results and registered decisions; it does not independently establish their numerical values from raw arrays.

## Final report clarification recheck

**PASS; both requested wording clarifications are resolved.** The final exposure text separately states profile archive-unopened versus all-five prediction archive-copy access, and separates 30 candidate full/group fits from six archived-4,821 replay fits. All five member label-access JSON records and source hashes match their sealed runtime records. No report issue remains. Reviewed report snapshot hashes (later audit-link insertion may change documentation only):

- `docs/reports/ML-June-calibration-execution-2026-09-28.md`: `d0a216d3b0c0ba90dee7180ce1630772f424d866ead9acb5973a3a16d92fab8c`
- `docs/reports/ML-June-calibration-exposure-2026-09-28.md`: `4a3e2ab09199f04b5edab83e1b0b35d83c6936b2f0a85e6a39d9446fac2b3182`
- `results/ML-JUNE-CALIBRATION-EXPOSURE-v1.json`: `49a09cfe31f6330442fb1984c6c49ec1e21d4ee47637c861b90bb46dad4dc44b`
