# A common frequency and model-internal policy comparison

2026-09-23. The contract/config were frozen in `7a27943`; the implementation/test were committed in `a4ee375` before any score. A row-order assertion stopped the initial run before scoring; `c49789b` sorted supported DEV keys to the frozen archive order and was committed before the successful run. Only the partial new checkpoint from that stopped run was removed. The scored run's source hash and Git commit are in the result.

All models use the same 145 April supported TRAIN PAs (578 pitches), 151 July DEV PAs (563 pitches), 10 outcomes and legal-transition conditioning. The restored 578-row count, known-type and pitcher-type tables reproduce their pre-serialization arrays exactly. Observed action indices from `Engine.predict_counts` match the saved A count/frequency/blend predictions array for array, including labels and pitch keys. The frozen frequency/blend had larger original TRAIN/CAL fits, so this comparison does not isolate model class.

| Model | All DEV NLL | Δ vs count NLL | Two-strike NLL, 159 pitches | All DEV Brier |
|---|---:|---:|---:|---:|
| April count+hand | 1.589641 | — | 1.678848 | 0.746602 |
| April known type | 1.595015 | +0.005373 | 1.692350 | 0.747412 |
| April pitcher+type | 1.598744 | +0.009102 | 1.698211 | 0.748140 |
| Frozen frequency | 1.555541 | −0.034101 | 1.635159 | 0.734695 |
| Frozen blend | 1.521566 | −0.068076 | 1.588874 | 0.726403 |

Six-game paired bootstrap 95% intervals for all-pitch NLL delta versus count are known type [−0.001907, +0.012303], pitcher type [−0.002229, +0.019444], frozen frequency [−0.063099, −0.012001], frozen blend [−0.107074, −0.033128]. For 2-strike pitches, the known-type and pitcher-type differences are +0.013502 [+.004777, +.021580] and +0.019362 [+.006794, +.030342]. The result also records NLL, Brier and top-label ECE for every model in both slices. These intervals resample only six fixed games and omit fitting uncertainty.

The known-type table has 120 observed cells; 539 DEV pitches use a type cell and 24 fall back to count. The pitcher-type table has 161 observed pitcher/type cells; 520 DEV pitches use them, 19 fall back to a type cell, and 24 to count. These are predictive table origins, not off-policy support or propensity evidence.

Each complete PA has the same 5- or 6-type action set, frozen WE/advancement, and fixed start (0,0). The repertoire reference uses the frozen bundle's TRAIN pitch-type counts. Count has action-independent outcomes, so its policy is exactly this reference. Every other policy is the deterministic full-count-MDP optimum under its own tensor, then evaluated under both shared frozen judges. Mean defensive WE change versus repertoire is in percentage points:

| Policy | Frozen frequency judge | Frozen blend judge |
|---|---:|---:|
| Count/reference | 0.000000 | 0.000000 |
| April known type | +0.040964 | −0.031598 |
| April pitcher+type | +0.031690 | −0.026924 |
| Frozen frequency | +0.449593 | +0.066093 |
| Frozen blend | +0.239067 | +0.306830 |

The frozen-frequency and frozen-blend diagonal entries are mechanically favored because each judge evaluates a policy optimized against itself. The off-diagonal cells share judges but are still model-internal. They are not causal policy value, independent evaluation, OPE, or evidence for service adoption. Full PA-level judge values and six-game bootstrap intervals are in the SSD `policy_pa.json` and `results.json`; causal WE and OPE ESS are null.

The successful run took 12.230 seconds with macOS peak RSS 391,233,536 bytes. SHA256 values for the config, script, contract, selected parquets, A predictions, bundle manifest/metadata, frozen source manifest and new outputs are in `results/EXP-A-COMMON-001/results.json`. SSD artifacts are `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/A-COMMON-001/` (`small_models.pkl`, `predictions.npz`, `policy_pa.json`, `results.json`). `tests/test_a_common_comparison.py` passed. No neural fitting, new calibration, 2026/final-set access, policy adoption, Observer UI or deployment occurred.
