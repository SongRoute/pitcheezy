# B prior-family posthoc diagnosis

2026-09-23. The diagnostic specification and script were committed as `fc377f9` before this diagnostic run. The B aggregate July result (+0.012605907986 NLL) was already known and stated in the config. This is an accounting exercise, not a new DEV selection.

The frozen A/B pitch keys, labels, A count predictions, previous-family sequence, prediction normalization, source and prediction hashes, and B S1 loss arithmetic all match exactly. There is no demonstrated row-alignment, label, prior-boundary, or score-computation error.

The 145 April TRAIN complete PAs contain 578 pitches. Splitting 24 count/hand cells by previous family produces 61 observed cells. Across those 61, median support is 4 pitches, 12 are singletons and 8 have two pitches. On the 563 July DEV pitches, eight extended cells are unseen while their count cells are seen; those eight use the original count prediction. The DEV median extended-cell TRAIN support is 17, compared with 31 for count cells; 188 DEV pitches have extended-cell TRAIN support of at most 10. These are observed sample-size facts, not proof that sparsity caused the loss increase.

`NONE` is exactly equivalent to first pitch in both splits. All 145 first TRAIN and 151 first DEV pitches have count 0–0; neither split has a later 0–0 pitch. The previous-family field therefore adds no information at first pitch. Even so, the candidate performs an additional fixed 50-pseudocount shrinkage of the `NONE` cell toward the count model's 0–0 distribution, which can change its probabilities. Its 151 first DEV pitches add +0.002285 to the overall +0.012606 NLL delta; the 412 later pitches add +0.010320. This identifies redundant smoothing arithmetic, not a software bug or a causal estimate of what removing that smoothing would achieve.

All contributions below use fixed stored predictions and sum to +0.012606 across the 563 pitches:

| Slice | Pitches | Mean NLL delta | Contribution to overall NLL delta |
|---|---:|---:|---:|
| Previous fastball | 267 | +0.016764 | +0.007950 |
| Previous breaking | 99 | +0.010314 | +0.001814 |
| Previous offspeed | 46 | +0.006814 | +0.000557 |
| First pitch / NONE | 151 | +0.008521 | +0.002285 |
| Ball outcome | 197 | −0.006314 | −0.002209 |
| Strike outcome | 166 | −0.019353 | −0.005706 |
| Foul outcome | 93 | +0.032846 | +0.005426 |
| Other terminal outcomes | 107 | — | +0.015095 |

Count slices are in `results.json`. The largest single DEV game, 777094, has 77 pitches and contributes +0.006254; game 777217 improves by −0.000853. Six-game bootstrap precision is limited. Rare outcomes can have large individual log losses, so these outcome rows describe where aggregate loss arose, not stable subgroup effects.

For an April-only generalization check, the two original fixed count estimators were refit six times, each holding out one whole TRAIN game. All 578 pitches were scored once. Pooled count NLL is 1.524461 and prior-family NLL is 1.534670, delta +0.010208; five games worsen and one improves. This rules out the simple story that the direction appears only after moving from April to July, but it does not identify whether cell sparsity, extra shrinkage, omitted state, or changing pitch strategies are responsible. The April check was selected after seeing the DEV result and remains descriptive.

The next experimental hypothesis should be fixed from TRAIN evidence: add prior-family history to A's known-current-pitch-type frequency hierarchy, return the parent prediction exactly for first-pitch `NONE`, and choose any history mixing weight (including zero as the null) using only TRAIN game holdouts. Freeze the rule before a separate evaluation; do not choose weights from this exposed July DEV set. A owns that experiment.

Artifacts: `configs/EXP-B-DIAG-001.json`, `scripts/b_history_diagnosis.py`, `results/EXP-B-DIAG-001/results.json`, and SSD copy `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/B-DIAG-001/results.json`. The result contains SHA256 hashes of both full-game parquets, A/B predictions, parent manifests/config/report and the diagnostic script/config. Run time recorded by the script was 0.749 seconds; the previous B fit time remains unmeasured. `tests/test_b_history_diagnosis.py` passed (1 test). No 2026 or final CV data, neural fit, policy value, service or UI changes were involved.
