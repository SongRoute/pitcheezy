# same_model_review — robust scale + smooth clip residual module (5f1864b), T100b

- Reviewer: Opus 5.5 (claude-opus-5-5), a separate instance of the same model as the author. Label: **same_model_review**
  (not independent, not Codex reviewed). 2026-10-04 22:54 KST.
- Target: `5f1864b` on `codex/robust-clip-prep-review-1004` (parent `387acd7`): new
  `experiments/pitchmdp/pitchmdp/direct_robust_clip_model.py` and `experiments/pitchmdp/tests/test_direct_robust_clip_model.py`.
- Scope: source and synthetic CPU tests only. No GPU, no server, no network, no real data, no 2026, no scores read.
  Status of the code is unchanged by this review: author-only preparation of candidate 3 contrast A, **not selected, not
  registered**.

## Summary

- 0 BLOCKER, 2 MAJOR, 4 MINOR, 7 NOTE. The code is correct as far as checked; both MAJORs are **test gaps on the two
  claims contrast A rests on** (the forward really applies the transform; the training path is otherwise identical to
  `residual_counts`). In 5f1864b, deleting the transform from `forward`, or adding one torch RNG draw per forward,
  passed all 12 author tests.
- Fixes for M1, M2, m1–m4 are in the next commit (same two files only). After the fix: 14 tests pass, 24/24 of my
  mutants and 5/5 of the author's mutants are caught.

## Commands and exit codes

Run from `experiments/pitchmdp`, `PY=/Users/song/Projects/pitcheezy/.venv/bin/python` (Python 3.12.14).

| Command | Code | Result |
|---|---|---|
| `OMP_NUM_THREADS=1 $PY -m pytest -q -p no:cacheprovider tests/test_direct_robust_clip_model.py` at 5f1864b | 0 | 12 passed, 1.55 s |
| same, `tests/test_direct_ple_model.py` at 5f1864b | 0 | 27 passed, 1.64 s |
| mutation harness `/tmp/t100b-mut/mut.py` (22 mutants, 5f1864b) | — | 16 caught, **6 survived** (`/tmp/t100b-mut/run1.txt`) |
| robust-clip tests with the fix applied | 0 | 14 passed, 1.19 s |
| harness after the fix (24 mutants, incl. 3 new for the new report code) | — | 24 caught (`run2.txt`) |
| author's 5 mutants after the fix (`mut_author.py`) | — | 5 caught (`run3.txt`) |
| `tests/test_direct_ple_model.py` with the fix applied | 0 | 27 passed |

The harness rewrites the module in place, runs the test file with `-x`, and writes the original back in `finally`;
`git checkout -- <file>` and `git status` were run after every harness run (clean). Temp dir `/tmp/t100b-mut/` kept.

Mutants that **survived at 5f1864b**: `no_transform_in_forward` (forward without `self.transform`),
`rng_draw_in_transform` (`torch.rand(1)` in `transform`), `no_guard_in_logits`, `load_no_payload_hash_crosscheck`
(top-level `rc_stats_sha256` not compared), `load_arch_kwargs_check_off`, `report_preprocessing_missing`.
Caught: median→q25, context write-back reversed, past reshape channel-major, stats column order channel-major, stats
from early-stop, no fit guard, current token ×2, hard clip, no range rule, IQR `>=0`, `rc_scale` non-persistent,
fit-checkpoint re-save removed, RNG draw before `super().__init__`, architecture check off, rule counts wrong, center
+1e-3.

## Findings

### MAJOR

**M1. The forward-uses-the-transform claim is untested.** `direct_robust_clip_model.py:134-136`. Every author test
calls `net.transform` directly (`test_...:98, 141`) or compares a forward to the frozen network on inputs where the
transform is ~identity (`:178-190`) or only on padded positions (`:156-164`). Deleting line 135 leaves a plain
`ResidualMLPNetwork` and all 12 tests pass, i.e. the arm could silently equal `residual_counts`.
Fix (done): `test_forward_is_the_frozen_forward_of_the_transformed_inputs` — `net(...)` must equal
`ResidualMLPNetwork.forward(net, *transform(...))` bitwise and must differ from the forward on raw inputs.

**M2. The one-factor claim vs `residual_counts` is tested only at initialization.** Docstring `:30-32` says initial
parameters, RNG order and the training loop equal the `residual_counts` arm. `test_initial_parameters_...` checks the
parameters; nothing checks the loop. An extra RNG draw in `transform` shifts every dropout mask (a second factor) and
passed all tests. Fix (done): `test_fit_with_identity_transform_reproduces_the_residual_fit` monkeypatches c = 0,
s = 1, f = identity (then the inputs are bitwise the arm's inputs) and requires the same history, best epoch and
bitwise-equal logits as `DirectResidualModel.fit` for the same seed. It catches both RNG mutants.

I also confirmed by reading that the code itself satisfies the claim: `RobustClipResidualNetwork.__init__` does only
numpy/hash work before `super().__init__` and registers buffers after it (`:109-122`); `transform` draws no RNG;
`fit` calls the frozen `DirectResidualModel.fit` which seeds before building the network (`direct_residual_model.py:103-105`).

### MINOR

**m1. Fidelity wording: which columns the paper transforms.** The task brief says the paper leaves one-hot columns
alone. The LaTeX (`2407.04491v3/paper_arxiv_without_comments.tex:213`) says the opposite for RealMLP-TD: "all numerical
columns, including the one-hot encoded ones, are preprocessed independently"; only the preprocessing ablation table
(`:579`) "do[es] not apply the preprocessing methods to one-hot encoded categorical features". The module transforms only
the periodic/PLE continuous set and leaves the past type/outcome one-hots, the game-state ordinals and flags (context
0–10: balls, strikes, outs, inning, defensive_score_lead, five flags) and `left_share` (32) raw, but the docstring did
not list this among the deviations. It is a reasonable held-constant choice (same scalar set as the periodic/PLE arms),
not a defect. Fix (done): one docstring sentence naming the deviation and the ablation convention it follows.

**m2. `logits` guard untested** (`:178-179`). Fix (done): `test_current_physics_guard` now also calls `logits` on a
fitted model with nonzero current physics.

**m3. Load checks partially untested** (`:199-204`): the cross-check against the top-level `payload['rc_stats_sha256']`
and the `width/hidden/blocks/dropout/n_classes` check could be removed without a failing test. Fix (done): tamper tests
for both.

**m4. Double checkpoint write in `fit`** (`:170-175`). The frozen fit saves (`direct_residual_model.py:141-142`),
then the report is updated and the file is written again. Harmless for the CUDA runner (it calls `fit` without
`checkpoint` and saves once itself, `scripts/run_cuda_direct.py:736-742`), but a direct caller killed between the two
writes keeps an archive without the `rc_*` report fields, and the frozen save is wasted. Fix (done): override
`_base_report` (a `MatrixModel` hook the frozen fit calls before its own save, `direct_residual_model.py:137`) to add
`preprocessing`, `rc_stats_sha256`, `rc_rule_counts`, and new `rc_rule_columns` (labels of the range/constant columns,
for the disclosure T90b needs: the rare pitch-type columns are expected to take the range rule); `fit` returns the
frozen fit directly. `self.rc_rules` becomes `{label: rule}`. No frozen module touched (sha test still passes).

### NOTE

**n1. Paper rule — matches.** `compute_stats:75-81` = LaTeX `:214-220` exactly: center `q_{1/2}`; `s = 1/(q_{3/4} −
q_{1/4})` if they differ, else `2/(q_1 − q_0)` if they differ, else 0; `f(x) = x/sqrt(1 + (x/3)^2)` (`:104-105`). The
range fallback `2/(q_1 − q_0)` is correct (the paper's "MinMaxScaler" analogy, but still centered at the median). The
paper does not state the quantile interpolation; numpy `linear` is a choice (sklearn `RobustScaler`, the paper's named
analogue, also uses linear percentiles — from memory, not re-checked here).

**n2. Affine invariance — correct.** For `x' = a·x + b`, `a > 0`, linear-interpolated quantiles are equivariant, so
`s' = s/a`, `c' = a·c + b`, `s'(x' − c') = s(x − c)`, and the IQR/range zero tests are unchanged. Checked numerically
(float64, t(2) column and an 80%-zero IQR-0 column, `a = 3.7, b = −1.2`): max |diff| 8.9e-16 and 2.2e-16. All our
standardizations of these columns are one positive affine map per column (physics after median imputation, batter
style, pitcher profile), so the docstring claim holds up to float32 rounding. The rule is per (past position, channel),
as in PLE.

**n3. `CONTINUOUS['counts']` is the right set.** 41 indices = 11..51 without 32, plus 52 (`prior_records_per100`), for
the 53-wide context that `residual_counts` gets (`direct_combo_registry.py:45-47, 81`; `arm_inputs` appends the one
workload column, `run_cuda_direct.py:492-503`). Raw: 0..10 and 32. None of the 41 is binary/one-hot: 6 style priors
(standardized), 6 reliabilities and 5 **soft** memberships in [0, 1] (`archetypes.py:157-162`), 4 pitcher physics
means, 10 type proportions, the log count, and the workload column. 81 = 5×8 + 41 matches `SHAPE` in the test.

**n4. Guard does not change accepted inputs on the runner path.** `residual_counts` uses `DirectResidualModel` without
`_no_current_physics`, but `arm_inputs` already refuses nonzero current physics for every arm
(`run_cuda_direct.py:495-496`). Outside the runner the guard is strictly narrower (fail closed). It also costs one
extra float32 host copy of TRAIN + early-stop for the whole fit (same as `DirectPLEResidualModel.fit`; the TabM-PLE
fit avoided it after review c953bfd). Worth measuring in the CUDA smoke peak-RSS, not a change request.

**n5. Indexing, buffers, archive — correct.** `past.flatten(1)` is p-major (`p*8 + c`), the same order as `_columns`
and PLE `feature_labels`; the reshape back is consistent (mutants catch both transpositions). `index_copy` is
out-of-place (the caller's context tensor is not modified; checked by the raw-column equality test). `rc_center/
rc_scale` are float32 persistent buffers and move with `.to(device)`; `continuous` is non-persistent and rebuilt from
`config['context_continuous']` on load, so strict `load_state_dict` is fine. `index_copy` on MPS was **not run** (no
GPU use in this task); the default device in `__init__` is MPS when available, the runner passes `cuda`.

**n6. Leakage / contract — holds.** Statistics only from the `train` argument after the guard (`:166-168`); padded
past positions excluded from statistics (`_columns:58`) and zeroed by the frozen `masked_fill` after the transform;
current token untouched (`:131`); early-stop rows only reach the frozen loop. Hash covers c then s; a consistent
rewrite of buffers and both hashes is undetectable (integrity, not security; same as PLE).

**n7. Small items, no change.** `rc_rules` is not restored by `load` (report has it); `test_degenerate_columns:90-91`
sets the same column twice; `:111` is 125 characters.

## What registration would still need (list only, not implemented)

1. Pre-registration doc for contrast A: reference `residual_counts` (same seeds), one changed factor = input
   preprocessing; the m1 column-set deviation; exposure note (repeated DEV, fresh seeds are not new data).
2. Registry entry (torch-free), e.g. a `robust_clip_residual_counts` arm with its flag, `model:
   robust_clip_residual_mlp`, `workload_level: counts`, `context_continuous: CONTINUOUS['counts']`, plus the
   `residual_counts` baseline copied exactly and a contrast `{candidate, reference, isolates: 'input_preprocessing',
   held: {backbone: residual, embedding: none}}`; registry tests.
3. Factory (`build_*`) that refuses any other `context_continuous`, and `run_cuda_direct.py` dispatch in
   `build_model` and `load_model`; add the module to `MODEL_SOURCES` so `own_hashes` pins it; config guard / queue
   flag tuple; scorer support in `score_cuda_direct.py` for the new arm/flag (how it dispatches was not read in this
   review) and fit-record fields (`rc_stats_sha256`,
   `rc_rule_counts`, `rc_rule_columns`).
4. CUDA smoke on the real export: device placement of `rc_*` buffers, `index_copy` on CUDA, stats time over 1.25M
   TRAIN rows × 81 columns (unmeasured), peak host RSS with the extra guarded copy (n4), per-epoch time vs
   `residual_counts` (unmeasured), archive round trip on the server, and a bitwise check that a smoke fit with
   identity stats reproduces `residual_counts` on CUDA (or a documented tolerance if CUDA is not deterministic).
5. A different-model (Codex) review is still pending; this is a same-model review only.
