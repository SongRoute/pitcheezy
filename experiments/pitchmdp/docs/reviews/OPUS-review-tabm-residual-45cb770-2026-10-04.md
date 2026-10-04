# Same-model review — TabM-mini adapter on the periodic_residual backbone (45cb770)

Label: `same_model_review; Codex_review_pending`. Reviewer: Opus 5.5, separate from the author.
No other model was called and no further Claude process was started. 2026-10-04.

Target: `git diff 7313366 45cb770` (three new files):
- `experiments/pitchmdp/docs/DRAFT-TABM-RESIDUAL-DESIGN-20261004.md` (pre-registration)
- `experiments/pitchmdp/pitchmdp/direct_tabm_residual_model.py`
- `experiments/pitchmdp/tests/test_direct_tabm_residual_model.py`

Scope kept: no ssh, no GPU, no ports, no real-data fit, no 2026 data. CPU only with `OMP_NUM_THREADS=1
MKL_NUM_THREADS=1`, every command bounded by `perl -e 'alarm 600; exec @ARGV'`. The full suite was not run.

## Disposition

**Accept the module and tests (non-blocking findings only). The pre-registration should be amended
before wiring/smoke (findings 1–3); none of them invalidates the code.**

- Frozen modules unchanged: `git diff 7313366 45cb770 --stat` lists only the three new files.
- Pre-registration came first: `1894476` (13:42:20 +0900, design) is the parent of `45cb770`
  (13:44:22 +0900, code).
- The network matches the pre-registered formula exactly (checked by reading and by tests/mutations).
- All 8 mutations of the module were caught by the original 14 tests.
- I added one minimal guard fix (R1) and 7 test cases (R2). 21/21 pass.

## Commands run (outputs abridged)

```
$ git log --format='%h %ad %s' -3
45cb770 Sun Oct 4 13:44:22 2026 +0900 feat(pitchmdp): DRAFT additive TabM-mini adapter ...
1894476 Sun Oct 4 13:42:20 2026 +0900 docs(pitchmdp): DRAFT pre-registration TabM-mini ... (before any code or fit)
7313366 Sun Oct 4 13:15:29 2026 +0900 docs(pitchmdp): EXP-P33-001 EMA descriptive result ...

$ python -m pytest tests/test_direct_tabm_residual_model.py -q -p no:cacheprovider     # at 45cb770
14 passed in 1.44s

$ shasum -a 256 .../continuous-children/sources/tabm_official.py
fc654af6a16bac53...   (matches the prefix cited in the design)
```

Mutation run (`/tmp/mut_trm.py`: each mutation applied to `direct_tabm_residual_model.py`, the
focused file run, the source restored; afterwards `git checkout --` and `git status --short` empty):

| mutation | tests that failed |
|---|---|
| M1 no input scaling (`x` expanded, no `* r`) | `test_each_member_equals_the_frozen_network_...` |
| M2 skip the output LayerNorm | `test_k1_with_unit_scaling_...`, `test_each_member_equals_...` |
| M3 output init bound 1/sqrt(n_classes) | `test_scaling_is_chunked_like_tabm_and_output_init_bound_uses_width` |
| M4 per-scalar chunks (`[1] * d_in`) | `test_scaling_is_chunked_like_tabm_...` |
| M5 load skips `model.architecture != a` | `test_checkpoint_roundtrip_and_cross_format_refusals` |
| M6 load skips the network_config check | `test_checkpoint_roundtrip_and_cross_format_refusals` |
| M7 one extra RNG draw before `super().__init__` (RNG order) | `test_shared_parameters_equal_...`, `test_k1_with_unit_scaling_...` |
| M8 padding mask removed in forward | `test_members_differ_and_padded_...`, `test_k1_...`, `test_each_member_...` |

8/8 killed.

Load probe at 45cb770 (`/tmp/probe_trm.py`, synthetic arrays, 2-epoch CPU fit, k=3):

```
empty logits (0, 10) member (0, 3, 10)
arch missing k -> TypeError int() argument must be ... not 'NoneType'
arch missing scaling_init -> ValueError Invalid TabM-mini settings
arch missing backbone -> ValueError Archived network differs ...
arch missing ensemble -> ValueError Archived network differs ...
config missing kind -> KeyError 'kind'
config missing k / dropout -> ValueError Archived network differs ...
config n_classes 11 -> RuntimeError size mismatch (state_dict)
config context_continuous shortened -> RuntimeError size mismatch (state_dict)
arch k=True -> ValueError Invalid TabM-mini settings
eval members identical True / train members differ True   (r = 1, identical member heads)
```

After R1+R2: `21 passed in 2.25s`. With R2 and without R1 (module reverted to `HEAD`):
`4 failed, 17 passed` — the missing-key cases `architecture-k`, `architecture-scaling_init`,
`network_config-kind` and `network_config-context_continuous` fail (TypeError/KeyError/wrong message).

## Check list results

1. **Network vs pre-registration — correct.**
   - The scaling multiplies the full backbone input `x = cat(raw, e)` before the shared `self.input`
     (`direct_tabm_residual_model.py:78`). Input, blocks and output norm are shared. Only the final
     `Linear(128, 10)` is replaced: `self.output = Sequential(norm, act)` (:62) plus the einsum (:81).
   - Chunks come from `tm.feature_chunks(n_raw, n_embedded, 8)`. `n_raw + 8 * n_embedded == d_in` is tested.
   - The W/b bound is `1/sqrt(width) = 1/sqrt(128)`. This matches the official
     `LinearEnsemble` (`_init_rsqrt_uniform_(weight, d)`, `tabm_official.py:497-499`).
   - RNG order: the k validation is pure Python before `super().__init__`, and r, W, b are drawn
     after it. Mutation M7 confirms the test pins this.
   - Padding: same `masked_fill` and embed mask as the frozen forward (M8 caught).
   - 3D semantics: `LayerNorm(128)` normalises each (row, member) vector. `Dropout` on `[N, k, 128]`
     draws independent masks per member in training. The official MLP block does the same
     (`nn.Dropout` on the batched `[B, k, d]` tensor, `tabm_official.py:1185-1191`). The
     per-member-equivalence test covers eval mode. The new R2 test pins train-mode independence.
2. **Training/inference path — works unchanged.**
   - `DirectTabMModel.fit/logits/member_logits` only read `architecture['k']`, `self.n_classes` (10)
     and `self.network`. All are set by the subclass.
   - Empty-batch shapes are `(0, 10)` and `(0, k, 10)`.
   - The report holds `kind = periodic_residual_tabm_mini_mlp`, the nested architecture,
     `network.width = 128` and `network.dropout = .1`.
   - `self.width = 128` (`pr.BACKBONE`), the same value the frozen periodic model would set.
     `head_dropout` is absent from the architecture (tested). No stale value is reported. See N1 and N2
     for the two leftovers.
3. **Archive — refusals work.** A wrong kind/format is refused both ways; this is tested for
   TabM→ours, periodic_residual→ours, ours→TabM and ours→periodic_residual. A tampered k, dropout,
   scaling init or backbone is refused. Malformed archives raised TypeError/KeyError (fixed by R1).
4. **Tests — meaningful** (8/8 mutations). The gaps I found are added in R2.
5. **Pre-registration** — see findings 1–5.
   - Seeds 24–28: disjoint from every `seeds`/`seed_rule` in `configs/*.json|yaml` and
     `experiments/pitchmdp/configs/**` (scripted scan: no config contains 24–28; the registries hard-code
     13–15, 16–18 and 19–23).
   - No LR schedule exists in `DirectResidualModel.fit`, `DirectTabMModel.fit` or `MatrixModel.fit`
     (grep for scheduler/StepLR/Cosine/OneCycle: none).
   - The P31 RESULT says "단일 신경망 9회는 모두 patience 5로 멈췄다" with a maximum run epoch of 19
     (`periodic_counts_d10` 14/19). The P32 RESULT says "18회 모두 patience 5로 멈췄다" with a maximum of
     19. TabM at the 30 cap: 3/6, matching the design.
   - Budget: 5 × (85 + 220 + 750 + 1200) = 11,275 ≈ 11,300 worker-s. At 2 concurrent, ≈ 94 min:
     the arithmetic is correct. The 750 s figure is 60 × 12.5 s/epoch from P31 (350.9 s / ~28 epochs).
   - No winner, promotion or verdict language. The exposure wording is adequate.

## Findings

### 1. (non-blocking for code; fix in the design before smoke) The smoke-time stop rule cannot be measured as written
`DRAFT-TABM-RESIDUAL-DESIGN-20261004.md:96-98,103-104` says "smoke measures it" and "smoke time
per TabM-residual fit > 1800 s at the 60 cap → redesign". The runner smoke is
`SMOKE = {'train': 8192, ..., 'epochs': 2}` (`scripts/run_cuda_direct.py:111`): 8 updates per epoch
against ~1,224 on the full TRAIN split. A smoke task therefore never measures a full-fit time at a
60 cap. Whoever applies the rule would have to choose an extrapolation after seeing the number.
Suggested diff (design):
```
-smoke time per TabM-residual fit > 1800 s at the 60 cap → redesign before registration
+timing probe fixed now: projected fit seconds = (smoke train seconds / smoke optimizer updates)
+  × 1224 updates/epoch × 60 + measured non-train smoke seconds, per GPU, for the slowest of
+  the 5 smoke seeds; projected > 1800 s → redesign before registration (not silently shorten).
```
Alternatively, register one full-data timing epoch per arm as part of the smoke.

### 2. (non-blocking; must be explicit in wiring step 2) A 60-epoch cap needs a runner change, and the existing TabM path cannot carry this arm
- `scripts/run_cuda_direct.py:105,201` hard-codes `TRAINING = {'epochs': 30, ...}` and refuses any
  other `config['training']`. Cap 60 needs a family-scoped training constant. A global edit
  would change the runner source hash pinned by every earlier family.
- `scripts/run_cuda_direct.py:218,437-441,481-483` routes any arm with key `tabm` to
  `direct_tabm_registry.check_arm` / `direct_tabm_factory`. `load_tabm` reads
  `a['head_dropout']` (`direct_tabm_factory.py:26`), which raises KeyError on this archive. The new
  arm needs its own key, registry and factory.
- The factory must compare `context_continuous` with the registered map after load, as P31
  `load_tabm` does. `load` only catches a wrong-length map, through a state_dict size mismatch.
  A same-length permuted map loads silently.

Suggested diff (design, "Next steps" item 2): add "family-scoped TRAINING {epochs: 60, ...} checked
by the new registry (runner keeps 30 for every earlier family); new arm key `tabm_residual`
(not `tabm`); factory compares k, scaling init and continuous map after load."

### 3. (non-blocking; recommend adding before registration) Contrast 2 is not one-factor; one cheap arm closes most of the gap
`tabm_residual_counts − tabm_counts` (`design:22-24,87`) changes several things at once:
- backbone (flatten 256-wide MLP → 128-wide pre-LN ResNet with LayerNorm),
- per-member head input (256 → 128),
- dropout regime (shared internal .1 vs head dropout 0).

The dropout part is disclosed in the Question section (:23-24) but not on the contrast line (:87).
P31's `tabm_counts_d10` already exists (TabM-mini, head dropout .1, same module) and is a
frozen P31 arm. Adding it as arm 5 gives `tabm_residual_counts − tabm_counts_d10` with dropout .1
on both sides. The dropout placement still differs: after both head GELUs vs inside the residual
blocks. Cost: +5 fits × ≤ 750 s ≈ +3,750 worker-s → ≈ 15,000, inside the 36,000 cap.

Without this arm, at minimum change line 87 to "(backbone + width + dropout-regime bundle under
TabM-mini; not one-factor)". The result template should carry the same wording. P31 measured
d10 − d0 under TabM at about −0.0001 (dev) and −0.00015 (mlb_dev). That was a different seed block,
so it cannot be used to argue the bundle away.

### 4. (non-blocking) `load` raised TypeError/KeyError on malformed archives — fixed (R1)
`direct_tabm_residual_model.py:123-124` (45cb770):
- missing `architecture['k']` → `int(None)` TypeError inside `DirectTabMModel.__init__`;
- missing `network_config['kind']` → KeyError;
- missing `scaling_init` → ValueError with the settings message rather than "differs".

All of these were still refused, so no wrong model could load. Runners that catch only ValueError
would crash instead of recording a refusal. R1 adds a key-set guard.

### 5. (non-blocking, wording) The pre-registration omits the P31 caveats it inherits
For the result doc, state the following (all already in `direct_tabm_model.py:19-22,43-46`):
- `history.train_weighted_nll` holds the mean member CE, so its gap to
  `earlystop_conditional_nll` is not comparable with the single-network arms;
- the scaling r and the per-member heads get weight decay .01 (package default .0003);
- temperature acts on log p-bar.

Item 3 of "Next steps" says "CUDA smoke of all 20 tasks" — fine, but name the smoke budget/epochs.

### N1. (non-blocking) Stale inherited `backbone()` now returns features, not logits
`PeriodicResidualNetwork.backbone` (`direct_periodic_residual_model.py:76-80`, frozen) calls
`self.output(z)`. On this subclass that is `LayerNorm → GELU`, so `net.backbone(x)` returns `[N, 128]`
features (probe: shape `(16, 128)`). The subclass `forward` does not use it. The only caller in
`pitchmdp/` or `scripts/` is the frozen parent `forward` (`direct_periodic_residual_model.py:86`),
which the subclass overrides; `scripts/representation_adapters.py:156` is an unrelated attribute.
Optional diff:
```python
    def backbone(self, x):
        raise NotImplementedError('TabM-residual: use forward (member logits [N, k, 10])')
```

### N2. (non-blocking) `n_classes` is not among the load-checked keys
A tampered `network_config['n_classes']` fails with a state_dict RuntimeError rather than ValueError
(probe). The frozen loaders behave the same, so I left it. Optional: add `'n_classes': model.n_classes`
to `expected` in `load`.

## Reviewer changes (each separate)

- **R1 (fix, `pitchmdp/direct_tabm_residual_model.py`, 3 lines in `load`).** If `architecture` is
  not a dict with exactly the keys of `ARCHITECTURE`, or `network_config` lacks `kind` or
  `context_continuous`, raise `ValueError('Archived network differs from the archived architecture')`
  before building the model. It changes only the exception type and message on malformed archives:
  every archive it rejects was already rejected later. The module hash changes, so wiring must pin
  the post-review hash.
- **R2 (tests, appended to `tests/test_direct_tabm_residual_model.py`, 7 cases).**
  - `test_shared_dropout_draws_independent_masks_per_member_in_training_only`: with r = 1 and identical
    member heads, members are equal in eval mode and differ in train mode.
  - `test_empty_batch_shapes_and_report_width`: `(0, 10)`, `(0, k, 10)`, predict `(0, 10)`,
    width 128 in both model and report.
  - `test_load_refuses_archives_with_missing_keys_with_value_error` ×5 (architecture k / scaling_init /
    backbone; network_config kind / context_continuous). 4 of the 5 fail without R1.
