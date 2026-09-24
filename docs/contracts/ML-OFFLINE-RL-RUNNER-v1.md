# Offline-RL runner v1

2026-09-24. Additive source implementation and synthetic validation; no real preparation, resource profile, training or policy evaluation has been executed by the implementation agent. Scientific decisions are fixed in [ML-OFFLINE-RL-FEASIBILITY-v1.md](ML-OFFLINE-RL-FEASIBILITY-v1.md). The sole heavy owner executes this runner after policy/G dependencies are frozen.

## Preparation registration

The JSON shape is exact below; replace paths and SHA placeholders with verified values. Hash strings are full SHA256. `fixed` must equal the implementation constant, not an independently selected parameter set. The inherited policy preparation must already satisfy its exact `POLICY_INFERENCE` registration.

```json
{
  "protocol": "ml_offline_rl_prepare_v1",
  "parent_policy_run": "/absolute/frozen-policy-run",
  "parent_policy_preparation_sha256": "<64 hex characters>",
  "parent_tuning_results_sha256": "<64 hex characters>",
  "parent_policy_execution_config": "/absolute/policy-execution.json",
  "fixed": {
    "width": 128,
    "hidden_layers": 2,
    "expectile": 0.7,
    "actor_temperature": 0.01,
    "actor_weight_clip": 100.0,
    "cql_alpha": 0.01,
    "learning_rate": 0.0003,
    "polyak": 0.005,
    "batch_size": 1024,
    "gradient_clip": 10.0,
    "gamma": 1.0
  },
  "seeds": [0, 1, 2],
  "profile_updates": 256,
  "prepare_seconds": 3600,
  "terminal_missing_next": "exclude_whole_pa_no_exceptions",
  "comparator_rule": "June_P3_if_mean_ge_P2_else_P2",
  "multiplicity": "joint_four_imputed_and_worst_case_Holm"
}
```

Preparation verifies the parent policy source/environment/config/artifact identity and sealed June family, independently recomputes tau and P2/P3 comparator, then freezes all dependencies. It audits raw chronological TRAIN PAs against complete frozen D100 TRAIN membership and common BC∩delivery∩vocabulary support. It writes every PA's eligibility/reasons, requested/complete-D100/retained counts by role and terminal class, retained keys, source copies and the lazy feature bank. There is no terminal missing-next exception. Unsupported or partial PAs are counted, not repaired. WE/advancement comes only from the parent's already verified `defense-we-pa-v1` lineage and original TRAIN dates.

The bank has one observed token per retained transition, H5 indices into that bank, safe pre-pitch context, common support, logged action, next-row index, terminal mask and reward. A row's current observed token is absent from its own state; a later state may legally include it in past history. Routing IDs are stripped. NNBC/IQL/CQL consume identical minibatch features and rows. Preparation uses in-memory transition records and token/context payload; the preparation report records peak RSS and bank bytes. Fits gather minibatches lazily without duplicated dense next-state tensors.

## Resource profile and final fit registration

Run actual TRAIN 256-update profiles for seed 0 of NNBC, IQL and CQL. Each profile has a 1,800-second cooperative cap. Profile weights are archived and never reused as completed fit members. Quality metrics, June values and DEV values do not select settings or update counts. The resource report records wall seconds, peak RSS, networks/parameters, optimizer steps, transition draws and the nine-job projection at 20,000 updates. Root freezes one common count from resource-only projections. Report preparation overhead separately.

```json
{
  "protocol": "ml_offline_rl_execution_v1",
  "preparation_sha256": "<64 hex characters>",
  "profile_result_sha256": "<64 hex characters>",
  "updates": 20000,
  "member_seconds": 3600,
  "family_seconds": 10800
}
```

`updates: 20000` illustrates the ceiling; it is **not yet the registered count**. Only an integer from 1 through 20,000 is accepted. The exact execution JSON is frozen on first fit. Each of nine method/seed jobs is immutable, chooses the final registered update, and uses the same uniform transition sampler budget. A failed or over-time job cannot silently become a completed shorter fit. All member attempts with elapsed runtime count against the 10,800-second family cap. Runtime is written before best-effort interrupted checkpoint saving; a failed checkpoint save preserves the original error and elapsed charge. An externally killed job lacking a runtime ledger blocks later fits. The caller must preserve a process elapsed ledger and use an external hard timeout because cooperative Python checks cannot interrupt a blocked device/filesystem call. No automatic retry, cap reduction, Q clipping, hyperparameter search or checkpoint selection is permitted.

## Commands and immutable outputs

Launch using the project's frozen Python and native-library environment, the same local config used by the parent, and an output directory inside the registered experiment root. The example uses shell variables set by the sole runner; no default location or count is inferred.

```sh
python experiments/pitchmdp/scripts/run_ml_offline_rl.py --config "$RL_CONFIG" --local-config "$LOCAL_CONFIG" --output "$RL_RUN" prepare
python experiments/pitchmdp/scripts/run_ml_offline_rl.py --config "$RL_CONFIG" --local-config "$LOCAL_CONFIG" --output "$RL_RUN" profile
python experiments/pitchmdp/scripts/run_ml_offline_rl.py --config "$RL_CONFIG" --local-config "$LOCAL_CONFIG" --output "$RL_RUN" fit --method NNBC --seed 0 --execution-config "$RL_EXECUTION"
python experiments/pitchmdp/scripts/run_ml_offline_rl.py --config "$RL_CONFIG" --local-config "$LOCAL_CONFIG" --output "$RL_RUN" evaluate --world control --execution-config "$RL_EXECUTION"
python experiments/pitchmdp/scripts/run_ml_offline_rl.py --config "$RL_CONFIG" --local-config "$LOCAL_CONFIG" --output "$RL_RUN" evaluate --world candidate --execution-config "$RL_EXECUTION"
```

Fit all combinations of `{NNBC,IQL,CQL}` and `{0,1,2}` sequentially under the exclusive heavy lock. `profile`, each member, and each evaluation world have exact sealed artifact families. Before any evaluation network is constructed, all nine members must pass preparation/execution/method/seed/completion identity, full artifact hashes, loaded checkpoint width/action count/final updates/fixed settings and finite weight checks. Evaluation records each member artifact hash and the sealed parent DEV dependency. Existing completed or failed stage directories are preserved; the runner does not overwrite them.

## Common evaluator and resource projection

Evaluation reuses exactly the parent's selected supported DEV keys, full requested/support denominators, initial states, calibrated conditional outcome models, exact 400 TRAIN delivery pools, frozen WE/cutoff, evaluation cap and paired uniforms. Current pitch physics is generated before its calibrated conditional outcome; legal generated history advances each pitch. The comparator's sealed per-PA P2/P3 arrays are reused, avoiding another expensive planner run. Primary evaluation is in the mapped-control world; candidate-world evaluation is a model sensitivity. Neither is independent empirical validation.

Each world evaluates three equal-probability seed ensembles and nine individual seed policies. CQL's ensemble is the mixture of three seed-greedy one-hot policies, not the greedy action of averaged Q. Seed values are descriptive; three fit seeds do not create three independent game samples. Batch actor/Q inference uses the same common rollout evaluator and random-number mapping as scalar policies.

For `N` supported PA starts, `R` rollout replications and cap `C`, evaluation conditional-row upper bound is `12*N*R*C` **per world**, or `24*N*R*C` for both. Each conditional row queries three frozen outcome ensemble members; report their actual network rows separately. Actor/Q inference is at most `18*N*R*C` rows per world: nine seed calls for the three ensembles plus nine individual-seed calls. Delivery pools remain exactly 400; a trajectory samples one full-pool index per pitch, not a 400-row integration. The parent policy row/time caps are reused as hard abort limits, not hidden downsampling permission. Profile the true full pipeline before registration if those evaluation caps may be insufficient; no outcome quality screening is allowed for that resource decision.

The joint four-test screen uses the shared paired whole-game bootstrap, 30-game/50-start minimum, .0001 absolute-WE point threshold, positive imputed and worst-case game CI lower bounds, and Holm .05 on max(imputed p, worst-case p). Monte Carlo SE is reported separately; planning seed, outcome models, calibration, selected comparator and training weights remain fixed in game uncertainty. A method needs both planner and NNBC passes for a broad RL improvement claim. Missing terminal next-state exclusions, role/event coverage, rare-action instability and observational confounding remain explicit. Causal effect, observational OPE and product adoption are null.
