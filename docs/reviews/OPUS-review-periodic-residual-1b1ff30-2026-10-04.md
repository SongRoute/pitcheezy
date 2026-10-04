# Opus 5.5 review — periodic/linear embedding + residual backbone (1b1ff30)

Label: `same_model_review; Codex_review_pending`
Reviewer: Opus 5.5 (separate session from the author). No other model was called and no further Claude process was started.
Target: `8863904..1b1ff30` (additive only), worktree `codex-periodic-residual-review-1004`.
Scope respected: light CPU, single thread, focused tests only. No server, GPU, ssh or ports. No score was used or looked up. The only result file opened was `RESULT-EXP-P31-001-20261004.md` lines 10–20 (run info: timings, memory, epochs), to check the budget numbers.
Frozen references `direct_periodic_model.py`, `direct_residual_model.py` and `matrix_models.py` are unchanged (`git diff 8863904 -- <those>` is empty).

## Disposition

**Non-blocking** for the next step (registry + DRAFT config + runner dispatch + CUDA smoke).
- One real code bug (F1, low severity): the nested architecture dict was aliased to the module constant. Fixed here with a failing-then-passing test.
- Two wording fixes in the design note should land before the config is registered (F2 contrast 1 / dropout, F5 RNG limits). They are text changes, not new runs.
- One check is needed in the step-3 runner dispatch (F4: compare the loaded `context_continuous` with the registered arm).
- Optional arms (F2 `periodic_counts_d10`, F6 `flatten_counts`) need a PM decision before the config is frozen. I am not requiring them.

## Findings

### F1 — Low (FIXED here) — the nested architecture dict was aliased to `ARCHITECTURES`
`pitchmdp/direct_periodic_residual_model.py:135` (was :134)
- `dict(fixed)` is a shallow copy. `model.architecture['backbone']` and `['embedding']` were the very same objects as `ARCHITECTURES[...]['backbone'/'embedding']`. `report['architecture']` is the same object too.
- The frozen modules use flat dicts, so `dict()` was a real copy for them. The nesting added here (needed to fix the 'activation' key collision) made the shallow copy unsafe.
- Failure scenario (reproduced): `m = DirectPeriodicResidualModel(...)`, then `m.architecture['backbone']['blocks'] = 3` (or any later runner code that mutates `report['architecture']`). This rewrites `ARCHITECTURES['periodic']`. Every later `DirectPeriodicResidualModel(...)` in the same process then accepts blocks=3 as "the fixed registered network", so the fail-closed check is defeated without any error.
- Fix (applied):
  ```diff
  +import copy
  -        self.architecture = dict(fixed if architecture is None else architecture)
  +        self.architecture = copy.deepcopy(fixed if architecture is None else architecture)  # nested: no aliasing
  ```
- Test: `test_review_architecture_is_a_deep_copy_not_an_alias_of_the_fixed_constant`. It fails on 1b1ff30 (`assert 3 == 2`) and passes after the fix. On 1b1ff30 it also corrupted the constant for later tests in the same session: 4 other tests then failed in cascade.

### F2 — Medium (design note honesty; non-blocking) — contrast 1 "backbone" also changes dropout 0 → .1
`docs/DRAFT-PERIODIC-RESIDUAL-DESIGN-20261004.md:22` and `:41`
- The frozen `PeriodicDirectNetwork.head` has no dropout (`direct_periodic_model.py:91-92`). The residual blocks have `Dropout(.1)`.
- So contrast 1 (periodic_residual − periodic_counts) changes all of these together:
  - skip connections
  - LayerNorm
  - widths (256/256 flatten → 128 trunk / 256 hidden)
  - dropout 0 → .1
- Line 22 says "이번 가족은 드롭아웃을 바꾸지 않는다(동결 residual 블록 Dropout .1 그대로)". That is true against residual_counts but false against periodic_counts. P29/P31 are exactly about whether dropout .1 helps periodic_counts, so this bundled factor can be confused with "residual structure" when contrast 1 is read.
- The bundle is acceptable for a descriptive contrast ("backbone as a whole"), but the note must say it is a bundle.
- Suggested diff:
  ```diff
  -- P29: 헤드 드롭아웃 증분은 불확실. 이번 가족은 드롭아웃을 바꾸지 않는다(동결 residual 블록 Dropout .1 그대로).
  +- P29: 헤드 드롭아웃 증분은 불확실. 이번 가족은 동결 residual 블록의 Dropout .1을 그대로 쓴다. residual_counts 대비로는 드롭아웃이 같지만, periodic_counts(flatten 헤드, 드롭아웃 0) 대비로는 0 → .1이 함께 바뀐다(대비 1).
  -1. periodic_residual_counts − periodic_counts: 본체 (flatten 헤드 → residual), 임베딩 같음
  +1. periodic_residual_counts − periodic_counts: 본체 묶음 (flatten 256·256 → residual 128/256, skip·LayerNorm, 드롭아웃 0 → .1), 임베딩 같음. 드롭아웃과 residual 구조를 나누지 못한다.
  ```
- Optional (PM decision, not required): add a 6th arm `periodic_counts_d10` (frozen P31 head-dropout .1 code) at seeds 16–18. Contrast 1 would then split into dropout and structure (periodic_residual − periodic_counts_d10). Cost: about +3 × 176 s of training (P31 measured mean for that arm on GPU3/4).

### F3 — Medium (test adequacy; FIXED by an added test) — the raw column order of `forward` was not pinned against the frozen network
`pitchmdp/direct_periodic_residual_model.py:81-84`
- `forward` re-implements the frozen raw concatenation (`direct_periodic_model.py:105`) instead of calling it. The original 9 tests check `embed()` equality and the backbone blocks, but not the order of the raw part.
- Mutation check: swapping `tokens[:, -1]` and `valid.float()` in the new `forward` left **all 9 original tests passing**. The run would have fed a different (still same-width) input layout than the claimed "raw, e = as DirectPeriodicModel".
- Added `test_review_periodic_backbone_input_is_the_frozen_flatten_head_input`. It replaces the frozen head with `nn.Identity` and the new `backbone` with the identity, and checks that the two inputs are equal bit for bit. It catches the mutation (1 failed). The linear arm inherits the same `forward`.

### F4 — Low (docstring overclaim; action for the step-3 runner) — "load refuses another … continuous map" is not true for a same-length map
`pitchmdp/direct_periodic_residual_model.py:32-33`
- `continuous`/`raw` are non-persistent buffers, rebuilt from `network_config['context_continuous']`.
- Reproduced: an archive whose `context_continuous` is changed `[2,3,5] → [1,4,6]` loads silently for both kinds and embeds different columns. Length changes and invalid maps are refused (state-dict size mismatch / `Invalid continuous context map`).
- The module cannot know the registered map. The frozen periodic module has the same property, and `run_cuda_direct.py:450-451` does not compare it either (the combo/frontier branches check `context_names`).
- Suggested:
  - docstring:
    ```diff
    -external.  Frozen modules are not edited.  Own archive formats; load refuses
    -another kind, architecture or continuous map.
    +external.  Frozen modules are not edited.  Own archive formats; load refuses
    +another kind or architecture and an invalid or wrong-length continuous map; the
    +registered map itself is compared by the runner after load.
    ```
  - step-3 dispatch: after `load`, `if model.context_continuous != arm['context_continuous']: raise ValueError(...)`, plus a negative test using a same-length tampered archive.

### F5 — Low (design note "known limits" wording) — RNG and initialisation facts
`docs/DRAFT-PERIODIC-RESIDUAL-DESIGN-20261004.md:72`
- The linear arm also builds the 10-class flatten head (and frequencies/embedding) and then discards them. The note mentions only frequencies.
- Positive fact, verified and now pinned by `test_review_linear_and_periodic_share_backbone_initialisation_for_a_seed`: for the same seed, the periodic and linear arms have **bit-identical backbone initialisation**. Linear runs the same constructor path and only draws `weight`/`bias` afterwards. Contrast 3 is therefore init-paired on the backbone. Dropout masks during training still differ (extra RNG draws), and parameter order differs (`weight, bias` come first in the linear `named_parameters`). That order only changes the float summation order in `clip_grad_norm_`.
- Also verified: the periodic-residual constructor leaves the torch RNG in a different state from the frozen periodic constructor (expected, since residual modules are added after the head).
- Suggested diff:
  ```diff
  -- periodic 네트워크는 동결 periodic 생성자를 거쳐 만든 뒤 flatten 헤드를 버린다. 그래서 residual 부분 초기값은 같은 시드의 DirectResidualModel과 다르다(RNG 소비 순서가 다름). linear 팔도 frequencies를 만든 뒤 지운다.
  +- periodic 네트워크는 동결 periodic 생성자를 거쳐 만든 뒤 flatten 헤드를 버린다. 그래서 residual 부분 초기값은 같은 시드의 DirectResidualModel과 다르다(RNG 소비 순서가 다름). linear 팔도 같은 경로(frequencies·embedding·flatten 헤드 생성 후 삭제)를 거친 뒤 w·b를 뽑는다. 그 결과 같은 시드에서 periodic·linear 팔의 residual 본체 초기값은 비트 단위로 같다(테스트로 고정). 학습 중 드롭아웃 난수열은 다르다.
  ```

### F6 — Info (design; optional arm) — contrasts 2, 3, 4 have only two degrees of freedom, and there is no flatten_counts arm
- Contrast 2 = contrast 3 + contrast 4 exactly (same seeds, so per seed as well). Not misleading, but there are only two independent comparisons.
- The stated question "does the single combined network beat both parents" is answered by contrasts 1 and 2 as defined.
- But "does the embedding gain depend on the backbone" (the interaction) needs a raw-input flatten arm at n_context 53: (periodic_residual − residual_counts) − (periodic_counts − flatten_counts). The control here is flatten at 52 without workload, so it cannot serve.
- Optional 6th/7th arm `flatten_counts` (P27 workload code), PM decision. Without it, the note should not claim anything about interaction. It currently does not, so this is fine.

### F7 — Info (budget, verified) — the timing source checks out
- `RESULT-EXP-P31-001-20261004.md:16-17`: periodic_counts mean 160.8 s, periodic_counts_d10 175.9 s. These are training means (`fit.json`), GPU3/4, matching the note.
- The note's 3,000 worker-s estimate is labelled 미측정 for the new arms. P31's own 15 fits (heavier TabM arms included) used 3,572.5 worker-s, far below the 36,000 cap.
- Small precision fix: say "학습 시간 평균(fit.json)" so it is not read as per-task wall time.

### F8 — Info — the other load-path checks hold (all reproduced)
- Cross-loading is refused in every direction tested, all with `Unrecognized …`:
  - periodic-residual / linear-residual archives in `DirectResidualModel.load`, `DirectPeriodicModel.load` and each other's loader
  - frozen `DirectResidualModel` / `DirectPeriodicModel` archives in both new loaders
- Tampering is refused:
  - `network_config` `d_embedding`, `width`, `dropout`, `hidden` (and `n_frequencies`, `sigma` for periodic) → `Archived network differs`
  - payload `architecture` → `differs from the fixed`
- Tampered `n_classes` is refused as a `RuntimeError` (state-dict size mismatch), not a `ValueError`. An unknown extra key would raise `TypeError`. Both fail closed, same as the frozen modules.
- `del self.head` / `del self.frequencies, self.embedding` go through `nn.Module.__delattr__`. The exact `state_dict`/`named_parameters` key sets contain no leftover (now pinned by a test). The optimizer is built in `fit` after construction, so it holds no discarded tensors. The `weights_only=True` round trip is exact (existing test).
- The linear embedding is masked. Removing `* keep` makes the existing test fail (mutation check), because ReLU(b) ≠ 0 at x = 0.

### F9 — Info — LR initialisation
- My unverified recollection: `rtdl_num_embeddings.LinearEmbeddings` initialises both weight and bias as U(±d_embedding^-1/2), which would match U(±1/√8). I did not look it up (no external lookup in this review).
- Keep the note's "패키지와 대조하지 않았다" until someone checks it against commit a8fc250.

## Checks requested in the brief (summary)
1. Network correctness:
   - same raw columns (new test F3), same embedded columns, mask and `embed()` (existing test, bit-equal)
   - backbone block/output structure equal to `ResidualMLPNetwork` (existing test)
   - no leftover parameters (new test); the linear mask is correct
2. Architecture record:
   - the nested dicts are fine after F1
   - network/architecture mismatch is refused for every architecture key, except a same-length `context_continuous` (F4)
   - no cross-kind or cross-module load possible (F8)
3. Design note:
   - parameter counts verified by code: 244,898 / 244,634 / 288,802 / 170,762; differences: 264, −15.2 %, +43.4 %
   - seeds 16–18 are disjoint from frontier 0–2, workload-dropout 3–7 and 8–12 (`direct_workload_dropout_registry.py:49-50`) and TabM 13–15 (`direct_tabm_registry.py:37`)
   - exposure wording is honest
   - budget marked unmeasured (F7)
   - wording issues: F2, F5
4. Tests: 6 added (see below). They cover the gaps in raw ordering, exact parameter names, init pairing, frozen↔new cross-loads, `fit` refusal of current physics (train and early-stop) and tampered embedding keys/architecture.

## Changes made by this review (each separate)
1. `pitchmdp/direct_periodic_residual_model.py`: `import copy`; `copy.deepcopy` for `self.architecture` (F1). There are no other code changes.
2. `tests/test_direct_periodic_residual_model.py`: imports `DirectResidualModel`; adds 6 test functions (8 test cases) prefixed `test_review_`:
   - `..._architecture_is_a_deep_copy_not_an_alias_of_the_fixed_constant`: fails on 1b1ff30, passes after change 1
   - `..._exact_parameter_names_leave_no_discarded_module[periodic|linear]`
   - `..._periodic_backbone_input_is_the_frozen_flatten_head_input` (catches the F3 mutation)
   - `..._linear_and_periodic_share_backbone_initialisation_for_a_seed`
   - `..._frozen_and_new_archives_do_not_cross_load`
   - `..._fit_refuses_current_physics_and_tampered_embedding_archives[periodic|linear]`
3. This review document.

Not changed (left to the author): the design-note diffs in F2/F5/F7, the docstring diff in F4, and the runner check in F4 (step 3).

## Commands run (outputs trimmed)
```
$ cd experiments/pitchmdp && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
    tests/test_direct_periodic_residual_model.py tests/test_run_ml_direct_periodic.py tests/test_run_ml_direct_residual.py -q -p no:cacheprovider
# at 1b1ff30
31 passed in 2.02s
# after adding review tests, before fix (aliasing test corrupts the constant for later tests)
5 failed, 12 passed in 1.25s        (test_direct_periodic_residual_model.py only)
# original module restored, review tests except deep_copy / deep_copy alone
7 passed, 10 deselected  /  1 failed, 16 deselected
# after fix
39 passed in 1.96s

# mutation: swap tokens[:, -1] and valid.float() in forward
original 9 tests: 9 passed      review tests: 1 failed, 7 passed
# mutation: drop "* keep" in LinearResidualNetwork.embed
FAILED test_padded_past_physics_embed_to_zero_and_gradients_reach_the_embedding[linear]

# probes (/tmp scripts, CPU, synthetic data)
backbone init identical by name (periodic vs linear, seed 0): True
forward input equal to frozen: True
periodic|linear tampered continuous LOADED silently, ctx [1, 4, 6]
periodic|linear in DirectResidualModel/DirectPeriodicModel.load: refused Unrecognized
frozen r.pt / p.pt in both new loaders: refused Unrecognized
embedding aliased: True; module constant now: 3; new model accepted with blocks 3   (before fix)
```
