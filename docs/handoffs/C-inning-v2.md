# C inning evaluator handoff — C-INNING-001

The standalone research evaluator in `scripts/c_inning_eval.py` now exercises a separate `inning_end` horizon with the frozen `minimal-pitch-service-v1` bundle. It uses `Engine.predict_counts` for the blend, `solve_pa.baseline_values` with one terminal basis at a time for an exact PA terminal distribution under fixed TRAIN repertoire frequencies, frozen `EmpiricalAdvancement` for state transitions, and frozen `WinExpectancy` at half-inning/game boundaries. The original defender remains fixed for valuation. Neither runtime, Observer event analysis, nor the PA recommendation policy was changed.

`experiments/pitchmdp/configs/EXP-C-INNING-001.json` and `docs/contracts/inning-eval-v1.md` froze the scope in commit `e1e4055` before real inference. The two-out selection amendment and implementation were committed at `6db5194` before that amendment's value run. The default caps are 9 PAs, 12 frozen model calls, 256 frontier states, and a .001 minimum state mass; any mass omitted under a cap stays unresolved. The result interval is `[absorbed WE, absorbed WE + unresolved mass]`. These are model-conditional bounds, not empirical confidence intervals.

The C0 replay pitch `776703:1:1` generated a valid terminal distribution with one frozen model call. At Top 1, zero outs, none of its first-PA terminals can end the half-inning, so its interval is `[0,1]` with all mass unresolved after the sole known hitter. That explicit abstention is in `results/EXP-C-INNING-001/real_state.json`.

The separately preregistered first supported two-out DEV PA is `777227:10:1` (2025-07-05, bottom 1, two outs, home 0–away 2). For the initial away defender, the interval is `[0.5266852522, 0.8099860070]` frozen final-game win probability. One PA absorbed `.7166992452` mass at the current half-inning boundary; `.2833007548` remains unresolved because no later hitter order was supplied. The values, full initial identity, and nine terminal probabilities are in `results/EXP-C-INNING-001/two_out_state.json`. This is an input-boundary demonstration, not a sampled substitution decision.

The roster screen in `results/EXP-C-INNING-001/roster_packet_screen.json` uses the packet from `C-ROSTER-001`. Its six historical observed-change states have one frozen-supported outgoing pitcher (`777063`, keep `554430`), for which one first PA was evaluated and the interval is `[0,1]`; five outgoing pitchers lack frozen support. The follow-up `lineup_supplement_v2.json` verifies prior batting order for five of six decisions, but `777063` remains null: a caught-stealing non-PA and a pinch hitter inserted after the observed pitching change prevent verification of the batter at the keep decision. Its first-observed-pitch batter and stance are therefore a conditional input, not a validated keep counterfactual. The packet also has no verified actually eligible substitute. Every actual replacement `value_pp` remains null. Retrospective roster listing and observed incoming pitcher are never converted into eligibility or a model-supported counterfactual.

`scripts/c_inning_source_audit.py` joins the three immutable inputs by all six fixed game and decision pitch IDs, verifies their SHA-256 hashes, and writes `results/EXP-C-INNING-001/source_admissibility_audit.json` by exclusive create. The audit records five verified prior batting orders, one supported keep pitcher, **zero games in their intersection**, and zero verified actually eligible substitutes. For `777063` it uses the stable reason `invalid_for_replacement_current_batter_after_pitcher_decision`; the earlier `[0,1]` numeric screen is preserved unchanged and excluded from actual replacement interpretation. The independent two-out demo is unaffected. The later structured pinch hitter is a source-admissibility finding, not an input to the earlier probability calculation.

`tests/test_c_inning_eval.py` covers exact fixed-policy terminal extraction, half-end absorption, partial mass bounds, missing lineup, walkoff, extra-inning abstention, prior-date profile/order checks, and same-evaluator comparison gating. Run with `/Users/song/Projects/pitcheezy/.venv-observer-standalone/bin/python -m pytest tests/test_c_inning_eval.py -q`.

Reproducibility identifiers (SHA-256):

| Item | SHA-256 |
|---|---|
| Frozen bundle manifest | `43ece920cb60c6c24ea9f1e720a7000b3b83038a3e24ac27d77ef17a2e8f0f1f` |
| Fixed DEV parquet | `89130facf5a8854780a33755241129240be6d43ea2a8584afcd65fe6111c1199` |
| Roster packet | `a327f1fecd7c9b97e4e2575328497520e373f9bcc325b28b341ae17303360371` |
| Lineup supplement v2 | `70ec1a13869b7a73f6a2dc6eb97fc5b813c2a03e0300f3b67205018ad6cd4605` |
| Main config | `50f54b3b1d98318a36061d694cb52bc8270a4ca7ec89dff36f802fa3c024d579` |
| Two-out amendment config | `f4e83e39dbb599f7fed51393792b9fbc1be24b54cc739fb6d66d554ad7fac333` |
| Final evaluator script | `d08323c9fc1127aacb5093192ffcb872a480241411f811262d873509e177c700` |
| Source audit script | `42b298184267375cc1b152e3e0890dd853b84aa0b2ad73fcde26d7d472b0d219` |
| C0 result | `98abe4ef0d856e26530c2f3d7832fc9ad9dc3623dacd03a0977c6af3083ca022` |
| Roster screen result | `e33ba90e5b10c2641d9dc41f85ff4b7c17590fcd0a2f12feab204608597b2ecb` |
| Two-out result | `f1ed2f18f10eac0024f44c7cf197e5c8890d359410d1ea119ba482917d6831ca` |
| Source audit result | `2da9176df04fa6c8aa4c1f0c1bae477e0c31b0c85e7919652bbafbe911643910` |

Each CLI result writer uses exclusive create and refuses an existing output path. New hypotheses require a new amendment and output name. The measured test run was 9 focused tests; the real outputs each used one frozen model call where supported. Wall-clock times are not a performance claim because they include local bundle loading and filesystem cache effects.

The checked-in source files for the frozen service, PA solver, game transition/WE, legality conditioning, and frequency baseline exactly match their five entries in the bundle's `source_hashes.json`. This source check used the current worktree at finalization. The six decision records are games, the one supported keep result is one first PA/model call, and its nine terminal probabilities sum to one; none of these counts is a number of validated substitutions.
