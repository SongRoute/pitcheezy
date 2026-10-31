# T2 entity exclusion and T3 fixed-input stress contract

Status: future protocol design, with an additive synthetic-tested T3 feature adapter.
No T2 fold, real-data T3 prediction, calibration, DEV score, or independent
confirmation has been executed under this document. The final run configuration,
candidate/control identities, source hashes, ordered samples and multiplicity
family must be registered before the corresponding evaluation is opened.

## Shared scope

- Select a single candidate and its registered control using Cpanel results only,
  before full-MLB, T2 or T3 results. Preserve the selection rule and evidence.
- Use the existing G0/G2/G3/G4 architectures, ten outcome classes, seeds 0–2,
  conditional 400-draw integration and explicit fallback coverage. G4 is fixed
  probability shrinkage, not a fitted hierarchical Bayesian posterior.
- The current pitch's realized physical vector and outcome are unavailable at
  prediction time. Supply frozen delivery draws and the candidate pitch type.
- Keep every requested evaluation key in the denominator; retain unavailable
  strata as unmeasured. Minimum reporting gates are 30 games and 500 pitches;
  class-specific interpretation requires at least 30 events.
- Scores remain development-time, retrospective-eligibility evaluations, not
  causal policy effects or independent final confirmation. Do not access 2026.

## T2: separate identity exclusion from observed-history adaptation

### Cohorts fixed without held-out outcomes

1. **Seen identities, new matchup:** both IDs are observed in allowed fitting
   data, but the pair is absent from every pre-DEV history, TRAIN, early-stop and
   CAL source. Evaluate their first observed DEV meeting; later meetings are a
   separate observed-matchup cohort. Existing checkpoints can be reused only
   after verifying these absence conditions. Missing coverage is not replaced.
2. **Held-out pitchers:** select the first deterministic SHA256-ranked member
   of each nonempty Cpanel role × hand × TRAIN-volume stratum, using ASCII
   `ml_t2_pitcher_v1|20260924|<integer ID>` and ascending hexadecimal digest.
   Cohort assignment
   may use TRAIN identity, role, handedness and volume metadata; it may not use
   outcomes or DEV coverage to choose replacements.
3. **Held-out batters:** hash ASCII `ml_t2_batter_v1|20260924|<integer ID>`;
   select IDs whose full SHA256 digest interpreted as a big-endian integer
   modulo five equals zero. Record TRAIN volume strata. Copy these exact hash
   conventions into the eventual run configuration.

The pitcher and batter exclusions are separate folds. They are not intersected
into a small joint exclusion. The candidate and control use identical exclusions,
examples, fitting budgets, seeds and auxiliary objects within each fold.

### Reconstruct features before fitting

Delete every held-out entity's observations from all label-bearing TRAIN,
early-stop, temperature and blend partitions. Purge entire affected PAs from
fitting examples when a held-out pitcher appears in a mid-PA substitution.
Preserve chronological PA keys and explicit history gaps; filtering must never
join neighboring pitches across a PA or disguise a missing observation as valid.

Build the fold from primitive dated observations, not inherited cumulative
features or a copied `aux.pkl`. Exclude held-out rows from:

- physical imputation/scaling, pitch-type vocabulary and frequency tables;
- pitcher physical/type profiles, counts, cluster scaling and centroids;
- pitcher-specific and league delivery pools, including their fallback;
- individual batter and **league** cumulative sufficient statistics;
- batting archetype scaling/centroids and all evidence/reliability features;
- all training, early-stop and CAL targets and their historical inputs.

In particular, `archetypes.add_batter_style_history` includes all players in
league fallback numerators and denominators. Merely zeroing a held-out batter's
own history leaks those outcomes through other batters. Reconstruct both levels
on permitted observations before fitting the frozen TRAIN encoder. Supervised
example eligibility and historically observable token outcomes remain distinct:
retrospective `supported_pa` must not mask prior outcome tokens.

Do not admit excluded pre-DEV entity observations into the later adaptation
prefix. Preserve exclusion counts, source hashes and a membership audit for
every feature-fitting and label-fitting path. Synthetic invariance tests must
change excluded outcomes/physical values and prove that allowed fitting objects
and strict-zero-shot features remain unchanged.

### Three observation regimes on paired evaluation rows

For each held-out entity, reserve its first two distinct DEV games as an exposure
prefix. Evaluate only dates strictly after the date of its second prefix game;
same-day doubleheaders therefore cannot cross the exposure/evaluation boundary.
Entities without subsequent eligible rows remain absent/unmeasured, with no
replacement. Prefix games never contribute evaluation losses. Record actual
prefix pitch counts because two games is not a fixed number of observations.

| Regime | Available held-out entity observations |
|---|---|
| Z: historical zero-shot | None. Mask all past H5 tokens and exclude all held-out history from cumulative feature sources. |
| W: within-PA observation | Observed prior pitches in the current PA only; cumulative entity histories remain excluded. |
| O: fixed observed-history adaptation | The fixed two-game prefix plus observed current-PA H5. No accumulation from scored outcomes in this first protocol. |

All three regimes retain legal pre-pitch game state, observed handedness and
candidate pitch type. Thus “zero-shot” describes absence of entity-specific
historical evidence, not absence of observable baseball state. W is explicitly
not zero-observation. The same keys are scored in Z/W/O; differing row sets must
not be used to estimate adaptation gains. A first-pitch-of-PA descriptive slice
can expose sensitivity to masking H5 at nonzero counts.

The frozen neural weights, feature transformations, delivery distributions and
calibration rules never adapt using held-out outcomes. For O, reconstruction may
insert permitted prefix sufficient statistics before the frozen context encoder;
other source histories remain constrained by the fold and chronological cutoff.

### Existing pitcher fallback degeneracy and optional routing adaptation

`SharingPredictor` sends an unknown pitcher to the same raw global model in
G0/G2/G3/G4. Under strict unknown routing, separate temperatures or blend weights
can create differences without any sharing effect. Report a common-fallback
capability audit with common calibration; do not claim architecture gains from
such calibration-only differences. Unseen batters facing known pitchers remain
a meaningful sharing comparison.

A nondegenerate pitcher O evaluation needs a separately registered adaptation:
derive its continuous physical/type/hand profile from the fixed prefix, transform
using sanitized-TRAIN scaling, and assign the nearest frozen centroid. Set its
evidence count to prefix observations, retain cluster TRAIN counts, and provide
no personal model. Ties use the smallest centroid index. Keep delivery pools at
the sanitized-TRAIN league fallback to isolate context/routing adaptation.
This rule is an additional feature/routing adaptation, not the original unknown
fallback and not gradient-based few-shot learning.

### Minimum fitting work and inference

- Natural new-matchup cohort: no new fits if all absence conditions verify.
- Strict pitcher common fallback: three global fits in its sanitized fold.
- Batter G2 versus G0: six fits (global and feature model × three seeds).
- Batter G3 versus G2: up to 18 fits (global fallback, feature control and four
  cluster experts × three seeds).
- Batter G4 versus G2: up to `3 × (6 + P)` fits, where `P` counts eligible personal
  models. Reuse byte-identical units within the same fold only.
- Prefix routing adaptation reuses fitted fold units; it does not fit new experts.

Rebuild calibration from allowed CAL entities. Freeze a complete T2 hypothesis
family, directional claims and margins before scoring. Use paired whole-game
resampling with pitch-weighted means on common rows. Natural new players whose
earlier observations already enter features must be labeled online-history
evaluation, not strict zero-shot.

## T3: immutable H5 sharing predictors

`pitchmdp.matrix_stress.protocol()` is the exact implemented feature contract,
version `ml_stress_h5_v1`, with seed 20260924 and these scenarios:

| Scenario | Transformation |
|---|---|
| clean | No feature changes; required equivalence control |
| H2 | Keep only the newest two prior H5 slots |
| H0 | Mask all five prior slots |
| mask20 / mask50 | Drop each prior observed pitch with probability 0.2 / 0.5, with nested masks |
| unknown_type_outcome | Replace prior type and prior outcome with their unknown one-hots; keep physics and validity |
| sensor01 | Independent normal noise SD 0.1 on six non-angle standardized physical fields; spin rotation SD 5 degrees |
| sensor03 | Same normal variates scaled to SD 0.3 and spin rotation SD 15 degrees |
| unknown_pitcher | Zero standardized pitcher profile/count, unknown cluster flag, cluster and personal routing IDs −1 |
| unknown_batter | Six declared initialization rates, zero evidence, and recomputed standardized rates and memberships |

All sensor modifications apply after the existing missing-value imputation. Spin
noise rotates the unscaled sine/cosine pair jointly and then restandardizes; it
preserves an imputed pair's radius instead of inventing a measured unit vector.
The six fixed batter rates are contact .76, swing .47, walk .085, strikeout .225,
isolated power .165 and groundball .43. Zero evidence yields uniform memberships
through the frozen encoder. Actual stand and game state are preserved.

Hash randomness uses **historical pitch KEY**, not query key, array offset, H5
position or draw index. It is invariant to batch boundaries, repeated queries and
the 400 current-delivery draws. Mask strengths share one uniform; sensor strengths
share normal variates. Do not require exactly 20%/50% missing per query; report
the realized affected-token fraction. SHA256 domains and Box–Muller conversion
are frozen by the implementation.

Padding remains zero and invalid. Current candidate type and supplied physical
draw are untouched, current outcome channels stay zero, and the current mask is
always valid. The API rejects a missing current physical override.

Unknown-pitcher stress is specifically an **encoder/routing outage**. Original
identity remains available to the frozen delivery sampler, so its 400 vectors
and fallback tiers remain identical to clean inference. This is intentionally
different from T2's truly unknown pitcher with league delivery fallback. Personal
routing ID must also become unknown; clearing only cluster flags leaves G4's
personal branch active.

### Additive adapter and future run requirements

`StressHistoryStore` and `StressSharingContext` wrap frozen G inputs without
mutating them. `predict_stress_streamed` returns temperature-calibrated and raw
integrated probabilities plus delivery tiers. It never fits, recalibrates,
blends, scores or writes artifacts. The future orchestration layer must:

1. Verify frozen source/environment/parent/member hashes and complete seeds;
   acquire the shared heavy-job lock before any real inference.
2. Register candidate/control, all ten scenarios, rows, labels, scenario contract
   hash and inference budget. Snapshot these additive source files.
3. Reuse archived per-seed temperatures and blend weights. Average/blend in the
   same previously registered order; never refit weights on stress predictions.
4. Archive ordered keys, labels, raw/calibrated/blended predictions, delivery
   tiers, affected-token counts, fallback coverage, timing and dependencies.
5. Require all registered candidate/control/seed/scenario predictions before
   scores. First verify clean equivalence against archived clean predictions.
6. Report candidate-minus-control within stress and stress-minus-clean within
   model, paired by pitch key and whole-game bootstrap samples. Predeclare the
   complete stress/metric/group multiplicity family and missing-group rules.

T3 does not validate an empirical error distribution. Stress-specific tuning,
calibration, scenario omission, key filtering and post-result intensity selection
are prohibited. Statistical scoring and an artifact-producing CLI remain future
work; the additive feature/prediction adapter is the implemented boundary.
