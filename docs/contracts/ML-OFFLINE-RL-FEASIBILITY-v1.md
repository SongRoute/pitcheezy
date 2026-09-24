# P4/P5 bounded offline-RL feasibility proposal

2026-09-24. **Source-only proposal for root review; not an execution registration and no RL implementation/training has started.** The user requested reinforcement-learning experiments. A concrete motivation is to compare amortized policy learning with computationally expensive, potentially noisy Monte Carlo rollout action ranking. Neither cost nor quality improvement is assumed.

## Scope, gate and common baseline

Proceed only after the P2/P3 baseline, outcome candidate/control, TRAIN auxiliary hashes, action masks, WE continuation and policy evaluation protocol are frozen. Select the P2/P3 comparison baseline by a rule fixed before their DEV results: proposed choice is P3 if it exceeds P2 on June original-WE tuning, otherwise P2, with ties selecting P3 as the more BC-regularized comparator. Reusing DEV-selected policy ranking would be an additional exposure and must not be presented as independent validation. Root must finalize this baseline rule.

P4 is a discrete pitch-type IQL adaptation; P5 is discrete CQL(H) with a DQN-style target. Use only existing local PyTorch/MPS and the sole heavy-execution owner. No target-location action, additional data, paid compute, current realized physical input or RE24 reward is introduced.

Both RL methods receive the same legal pre-pitch state as the rollout outcome adapter: game/count/hand fields, frozen TRAIN pitcher representation, prior-date batter profile and H5 past physical/type/outcome/mask. The current token is absent, rather than filled with the logged current type or physics. Current action is the discrete Q-head index or action label. Routing IDs are not added as learned features; any categorical ID expansion requires a separate information change. Unknown representations use frozen fallbacks.

This encoder is richer than categorical P0 BC. Add an **information- and architecture-matched neural BC** trained on exactly the RL trajectory rows. All three amortized policy networks use the same pre-pitch encoder and two hidden width-128 ReLU layers; actor heads have the common TRAIN vocabulary and support mask. NNBC uses ordinary masked cross-entropy, IQL uses weighted masked cross-entropy, and CQL extracts a masked greedy action from the Q vector. NNBC versus categorical P0 is a bridge diagnosis; do not attribute that difference to RL.

## Transition/reward audit before fitting

Reconstruct full PAs from the verified chronological TRAIN frame, before filtering individual pitches. The first candidate pool is complete PAs whose pitches all belong to frozen eligible D100 TRAIN keys; report the reduction from both raw TRAIN and D100. This is a retrospective supported subset, not prospective MLB coverage. Require the same subset for NNBC/IQL/CQL. Never assemble a trajectory by shifting an already-filtered dataframe.

For each kept PA:

- Verify one game/PA identity, start count 0–0, pitch numbers strictly consecutive from 1, one initial pitcher/defender/context, no unsupported mid-PA runner/score/outs/substitution change, and exactly one final supported terminal event.
- A nonterminal transition appends the just-observed pitch's type/physical/outcome token to the previous history. Its count must agree with the next **pre-pitch** logged count and baseball transition rules. The next row's current action, physics, outcome and missingness never enter the next-state encoder. Two-strike fouls advance history but preserve strikes.
- Every observed action must be in the same frozen BC∩token-vocabulary∩type-specific-delivery support mask used in policy evaluation; it is not replaced by another action. A gap, unsupported action or malformed transition excludes the whole PA and receives a reason count. Missing terminal next-state fields are audited separately; any permitted game-ending exception needs an explicit observed-score/winner rule before use.
- Check observed terminal next-game-state fields for legal advancement/inning/team transitions and fixed initial-defender orientation. Do not bootstrap across the PA boundary or treat a censoring boundary as a terminal success.

Intermediate reward is zero. Terminal reward is **the same frozen `terminal_values(initial_game_state)[observed_terminal_event]` used by the rollout simulator**, with discount gamma=1 on these episodic PAs and terminal bootstrap multiplier zero. This keeps the existing TRAIN expected-advancement/WE objective identical. It is a model-derived terminal utility, not an observed policy effect or binary observed game win. Substituting WE at the actually observed terminal next state would alter the advancement/reward target and belongs in a separate registered sensitivity, not this primary comparison. No artificial training pitch cap or incomplete-PA tail reward is introduced.

## Loss definitions and adaptation choices

Let `qbar(s,a)=min(Q1_target,Q2_target)` and `u=qbar(s,a)-V(s)`. IQL minimizes `mean(abs(eta - 1[u<0])*u²)` for V, and squared TD error against `r+(1-done)*V(s_next)` for each Q. Actor extraction minimizes `-mean(w*log pi(a_logged|s))` with detached `w=min(exp((qbar-V)/T_actor),100)`. The upper-expectile V step, ordinary mean TD target and advantage-weighted policy extraction follow the primary IQL method; terminal masks, finite pitch actions and WE units are this proposal's adaptation. [IQL paper, §4.2–4.3](https://arxiv.org/html/2110.06169)

Proposed fixed first-screen settings: expectile eta=.7, T_actor=.01 absolute WE, target Polyak rate .005, two independent width-128 Q networks, one width-128 V and one matched actor. Use one V update, one update for each Q, then one actor update per minibatch; policy gradients do not enter critics. These are explicit representative settings, not values claimed optimal for baseball.

CQL minimizes half squared TD error plus `alpha*mean(logsumexp(Q(s,a), a in support)-Q(s,a_logged))`. Proposed target is `r+(1-done)*Q_target(s_next,argmax_supported Q_online(s_next))`; terminal rows never evaluate an invalid next support. Exact finite-action summation avoids action Monte Carlo in the conservative term. The log-sum-exp/data-Q penalty is CQL(H); discrete Double-DQN extraction and gamma=1 episodic handling are our declared adaptation. Its theoretical lower-bound results are not automatically valid for this confounded, finite-sample neural baseball setting. [CQL paper, Eq. 4 and §3.2](https://arxiv.org/html/2006.04779)

Proposed fixed CQL alpha=.01 in the stated absolute-WE loss scale, target Polyak .005, one width-128 Q network plus target. No reward rescaling, Lagrange coefficient search, hidden Q clipping, extra entropy temperature or DEV-selected alpha is allowed in the first screen. A later sensitivity would be a separately registered family. Q values may be conservative rather than calibrated [0,1] probabilities; they are never reported as outcome probabilities or used as the final policy-value estimator.

## Bounded compute and selection

Three methods (matched NNBC, IQL, CQL) × seeds 0/1/2 = **9 fit jobs**, not nine individual subnetworks. Log every learned network, target copy, parameter count, optimizer update and forward/backward cost. Do not reuse an NNBC fit that has different input/trajectory hashes.

Proposed common fixed budget is at most 20,000 minibatches per fit, batch 1,024, Adam learning rate 3e-4, gradient norm cap 10, no HPO, no DEV early stopping. Training samples the same transition pool uniformly; save the sampler seed and PA/game manifests. A fit chooses its final registered update, not its best DEV or June checkpoint. The 20,000 ceiling is a proposal, not permission to reduce steps after seeing results.

Before registration, the sole runner profiles **TRAIN only**, for up to 256 updates for one seed of each method on the actual MPS runtime. Do not expose predicted policy quality. Extrapolate updates/sec, feature-gather cost and memory; proposed safety ceilings are 3,600 seconds per fit and 10,800 seconds for the full nine-job screen, including preparation/calibration overhead separately reported. If forecast exceeds these limits, root freezes a smaller common update count or explicit larger local time budget before any policy screen, with reason recorded. No paid capacity. Preserve all failed/profile fits; do not count them as completed full-screen seeds.

Avoid storing duplicated dense next-state tensors: store legal history/next-row indices and batch-gather pre-pitch features. There is no 400-draw outcome integration during offline-RL gradient updates. The frozen 400-pool model remains in the **common evaluator**, where actor inference can batch states without the planner's nested search. Measure end-to-end latency in addition to training cost; a faster actor alone does not establish a faster full evaluation pipeline.

No additional June hyperparameter search is proposed for this first screen. June can report prespecified diagnostics after weights freeze; changing eta/T_actor/alpha or selecting a checkpoint after viewing those results creates a new registered tuning family. DEV is never used for parameter choice. Aggregate the three seed policies by equal probability averaging (CQL seeds contribute one-hot greedy policies); preserve per-seed probabilities/values rather than choose the best seed. The three fit seeds do not create three independent game samples.

## Evaluation and necessary audits

Use precisely the existing requested PA manifests, initial states, fixed masks, terminal/cutoff WE, evaluation cap and paired simulation random tensors. Evaluate NNBC/IQL/CQL and the frozen P2/P3 comparator under the same mapped-control world; candidate-world evaluation remains sensitivity. If coverage differs, keep a common supported subset **and** report full requested denominators, lost coverage and policy fallback, never silently change cases.

P4 and P5 versus the fixed planner form the matrix's two primary comparisons with Holm .05. Proposed additional information-matched NNBC comparisons form a separately prespecified two-test secondary family; any claim about the RL objective improving over imitation requires passing the relevant NNBC comparison too. If root wants one broad “best algorithm” claim, use all four comparisons in one Holm family instead. This multiplicity choice must precede results.

Retain the existing point threshold ΔWE≥.0001 (=.01 percentage points), paired whole-game CI lower>0, appropriate Holm gate and positive worst-case truncation mean-difference lower bound for an untruncated-PA internal improvement claim. Report simulation MC SE separately. Full game bootstrap resamples PA-start game membership, holding neural training/calibration/selection fixed.

Audit finite probabilities, exact unsupported zeros, action entropy, state/action sample support, unknown/fallback rates, CQL conservative penalty versus TD loss, supported/unsupported Q distributions, TD targets, IQL advantage/weight quantiles and clipping fraction, and terminal mask behavior. Nonfinite losses/parameters abort and preserve the fit; extreme finite Q/weights are reported and investigated against a threshold frozen before training, not automatically clipped until a desired policy emerges. Whole-PA exclusions, poor model support, hidden pitch intent, confounding and rare-action extrapolation remain explicit limitations. Learned Q, CQL's name, small training TD loss or agreement with logged pitches do not identify observational OPE. OPE/causal effects and product adoption remain null.

## Root decisions before implementation

Freeze the P2/P3 baseline-selection rule; approve exact common trajectory/reward handling and game-ending exceptions; approve the fixed algorithm settings and resource-only profile; choose the two-primary-plus-two-secondary or joint-four-test multiplicity scheme. After those choices, implementation can be additive with synthetic Bellman/expectile/mask/terminal/PA-link tests, followed by the sole runner's real preparation/profile and final registered screen.
