# REVIEW — EXP-P35-001 PLE module + CPU tests (c953bfd), 2026-10-04

Label: `same_model_review; Codex_review_pending`. Reviewer: one Opus 5.5 instance, separate from the author. No other
model was called, WebFetch/WebSearch were not used, and no Claude process was spawned. Worktree
`codex-p35-module-review-1004`, branch `codex/p35-module-review-1004`. Target: `git diff 490cd62 c953bfd`.
No real data, export, ssh, GPU/MPS, port, install or 2026 data was used. All Python ran CPU-only with 1 thread under
`nice`/`alarm`, on synthetic arrays of at most 500 rows. **No performance claim.**

## 요약 (Song용)

- 한 일: PLE 모듈을 패키지 원본(a8fc250)과 직접 대조했다. 패키지 코드를 /tmp 클론에서 불러와(설치 없이) 출력을 비교했고, float64 기준 구현을 따로 짜서 비교했으며, 변이 15개를 돌렸다.
- 결과: 인코딩, scale/shift, mask, 임베딩 출력이 패키지와 **비트 단위로 같다** (T_j ≥ 1인 열). 막는 문제(BLOCKER)는 없다. 고친 것 3가지(커밋 2개): ① TabM fit이 TRAIN 사본을 하나 더 들고 있던 문제(합성 예시에서 1.93 → 1.15 MB, 고정 P34 팔은 1.13 MB), ② 클래스 KIND/FORMAT 이름, ③ 테스트가 놓친 변이 2개(라벨 순서, early-stop 물리 검사)를 잡는 테스트. 출력은 수정 전후 비트 단위로 같다.
- 판정: **연결 작업(wiring) 진행 가능**. 단 `MODEL_SOURCES`에는 새 sha `d3575019…a175e`를 넣어야 하고, 아래 W1~W5를 같이 처리해야 한다.

## Reviewer commits (separate; no frozen module edited)

| sha | content | rerun |
|---|---|---|
| `a831f8d` | test: p-major degenerate labels (`past3.phys5`, `context52`) + early-stop physics guard before binning (single and TabM) | new file 26 passed, exit 0; kills X9 and X14 |
| `cac12ea` | fix: `DirectPLETabMResidualModel.fit` no longer keeps guarded copies; class-level `KIND`/`FORMAT` for both PLE classes; one test | focused 4 files **83 passed, exit 0**; the new test fails on the c953bfd module (1 failed / 26 passed, exit 1) |
| (this file) | docs: this review | — |

Module sha256 after `cac12ea`: `pitchmdp/direct_ple_model.py` `d3575019f971b06ba26060a1fa31e3894a070fd3b4b88191393743cb16ea175e`
(author `6a1541a3…7425`). The 5 frozen modules are unchanged: `18f0b96a…`, `ee1b3d3f…`, `7feb9cfd…`, `c968aec9…`,
`6a9daff6…` match the hashes asserted in the test. Synthetic logits and early-stop history before and after `cac12ea`
are bitwise equal (single seed 5 and TabM k=3 seed 5, 400 TRAIN / 120 early-stop rows, 2 epochs).

## Findings

No BLOCKER. No MAJOR. Three MINORs are fixed in the commits above. The NOTEs need no code change; W1–W5 belong to
the wiring task.

### Fixed

**F1 MINOR (fixed `cac12ea`) — the PLE TabM fit held one extra host copy of TRAIN + early-stop for the whole fit.**
`direct_ple_model.py:313-316` (c953bfd) rebinds `train, calibration` to the guarded copies (`_validate` →
`astype(np.float32)` always copies, `matrix_models.py:30`). It then calls the frozen `DirectTabMModel.fit`, which guards
and copies again (`direct_tabm_model.py:188`). The P34 `tabm_residual` arm has no such override, so it holds only one
copy. tracemalloc peak on 500 + 200 synthetic rows (arrays 0.79 MB) was: frozen tabm_residual 1.13 MB, PLE TabM
1.93 MB before the fix and 1.15 MB after. On real data the extra copy is about N_train × 1.1 KB
(6·38·4 + 6 + 53·4 bytes per row) of host RAM. The real N and the server host RAM were not checked here (미측정).
The single-head path is not affected: PLE 2.19 MB vs frozen periodic_residual 2.36 MB, because the frozen path also
holds its guarded argument. Fix: guard both splits, bin `_no_current_physics(train)` as a temporary, and pass the
caller's arrays to the frozen loop. Guard order, bins and outputs are unchanged (bitwise check above). Test:
`test_tabm_fit_hands_the_callers_arrays_to_the_frozen_loop_and_class_kinds`.

**F2 MINOR (fixed `cac12ea`) — class attributes `KIND`/`FORMAT` of `DirectPLETabMResidualModel` were the inherited
P31 values** (`periodic_tabm_mini_mlp`, `direct_tabm_mini_periodic_v1`; `direct_tabm_model.py:146`). The instance
`kind` was correct, and save/load use module constants, so behaviour was right. But the existing wiring tests assert
`reg.ARMS[...]['model'] == Model.KIND` (`tests/test_cuda_tabm_residual.py:119`), so a copied P35 wiring test would have
compared against the P31 kind. Both PLE classes now set `KIND, FORMAT = KINDS[...], FORMATS[...]`.

**F3 MINOR (tests added `a831f8d`) — two reviewer mutants survived the 23-test suite.**
- X9: labels built channel-major (`for c … for p …`, `direct_ple_model.py:73`) passed all 23 tests, because the only
  physics label tested is `past0.phys0` (j = 0), which both orders give. The real embedding order is p-major
  (`tokens[:, :-1, :8].flatten(1)`, `:184`; `compute_bins` columns `:85`). With the wrong order, a degenerate
  `past3.phys5` would have been recorded as `past0.phys3`, which is the wrong export column in `fit.json`. The new test
  checks the label of j = 29 and `context52`, and checks that perturbing `tokens[:, 3, 5]` changes only `e[:, 29]`.
- X14: dropping the early-stop physics guard from the single-head `fit` (`:265`) passed all tests.
  `DirectResidualModel.fit` itself has no physics guard, so this guard is the only one on that path. New test: both
  classes refuse current physics in the early-stop split, and `ple_bins` stays `None`.

### Verified (no change needed)

1. **Package fidelity (version B, a8fc250), checked by running the package code itself.** I imported
   `package/rtdl_num_embeddings.py` from `/tmp/pz-p35-module-src/rtdl` by `sys.path` (no install). I built
   `PiecewiseLinearEmbeddings(bins, 8, activation=False, version='B')` on the T_j ≥ 1 features of a random mixed case
   (81 features, T_j ∈ {0, 1, 2, 3, 5, 7, 23, 47, 48}; max T_j = 48, so the package width is 48) and copied our
   w0/c0/W into it. Results:
   - `impl.weight` / `impl.bias` equal our `ple_scale` / `ple_shift` **bitwise**, and `impl.mask` equals `ple_mask`;
   - the encodings are **bitwise** equal;
   - the full embedding max abs diff is **0.0**.
   This confirms: bias only in linear0 (`R:776-777`, `pkg.linear.bias is None`); W zero-init with no bias (`:170`);
   w0 then c0 U(−8^−½, 8^−½) (`:167-169`, package `R:83-86`; the package linear0 equals `LinearEmbeddings` for the
   same seed); last piece at index 47 and leading T−1 pieces at 0..T−2 (`:144-145`); first piece `clamp_max(1)`,
   middle pieces `clamp(0, 1)`, last piece `clamp_min(0)`, T = 1 unclamped (`:180-181`, package `R:656-681`). An
   independent float64 Eq. 1 reference (unpadded, written from 2022 Eq. 1, `embed2022.txt:155-169`) on the same case
   with padded past positions gives max abs error 6.9e-5 at |e| ≤ 829 (max relative 1.0e-5, i.e. float32 rounding).
   Padded past positions are exactly 0. Script: `/tmp/pz-p35-review/ref.py`.
2. **The author's source quotes match the clone:** R:772-778, 779-783, 79-86, 95, 348-350, 453-461, 404-410,
   313-339, 556-564, 656-681, 792-805, README 251-256, embed2022 "Trivial bins of zero size are removed" (:184) and
   "effectively equivalent to the scalar representation" (:165), TabM "slightly modified version" (:324) and
   "different parametrization and initialization" (:1116).
3. **Leakage.**
   - Bins come only from the `train` argument in both classes (`:266`, and `:313-316` after the fix). Nothing reaches
     the frozen `fit` before `compute_bins`; the guards only validate.
   - Physics columns use `valid[:, p]` rows (`:85`). Padded positions are zeroed before `embed` (frozen
     `forward` masked_fill) and multiplied by `keep` (`:187-189`).
   - `predict`/`logits`/`load` never call `compute_bins` (tested by monkeypatch). `load` rebuilds bins from the
     archived buffers and checks the hash against both the config and the top-level hash (`:235`).
   - Mutants are killed for: bins from all rows (X3), TabM bins pooled with early-stop (X7 at c953bfd, X7b after the
     fix), no `keep` (X8), and either hash check removed (X5, X6).
   - Early-stop exclusion for TabM was covered only indirectly before (the hash equals the single-head hash). The new
     F1 test asserts it directly.
4. **Frozen-backbone reuse.** On the PLE path, no frozen method reads `frequencies`, `embedding`, `n_frequencies` or
   `sigma`:
   - `forward` calls `self.embed`, which resolves to `_PLEEmbedding.embed` through the MRO for both networks;
   - AdamW and `clip_grad_norm_` use `parameters()`; there is no EMA on these paths;
   - the frozen `network()`/`load()` methods that read those keys (`direct_tabm_model.py:158-162, 252-267`,
     `direct_periodic_residual_model.py:145-177`) are overridden.
   Shared parameters equal the periodic counterpart for the same seed on the **factory path the runner uses**
   (`torch.manual_seed(seed)` → `model.network(...)`, `direct_residual_model.py:103-105`,
   `direct_tabm_model.py:204-206`) for both classes. Construction happens on CPU before `.to(device)`, so this init
   equality does not depend on the device. As expected and disclosed (N2), the CPU RNG stream after construction
   differs from the periodic arm. CUDA dropout-stream equality: unverified (not run).
5. **Parameter counts** (re-run): allocated 275,738 / 291,600; effective = 244,634 + 8ΣT_j (single) and
   260,496 + 8ΣT_j (TabM). Mixed case ΣT_j = 3,794 → 274,986 / 290,848. Frozen references are 244,898 / 260,760.
   Arithmetic check: 244,898 − 1,296 (frequencies) − 264 (Linear(32,8)) + 1,296 (w0, c0) + 31,104 (W) = 275,738.
   All of the report's numbers match.
6. **M5 (mask removed) is truly equivalent.** Inputs are validated finite (`matrix_models.py:26`). Padded encoding
   entries are `0·x + 0 = 0` and the T_j = 0 last slot is `clamp_min(0)` of 0, so padded W entries contribute 0 and
   receive zero gradient, and AdamW decay keeps 0 at 0. Rerun: M5 survives 27/27, as the author reported.

### NOTEs

- **N-a (package deviations not in the disclosure list).**
  - (i) The package `compute_bins` requires ≥ 2 rows and `n_bins < len(X)` (`R:398-413`). Ours has no row floor, so a
    column with a handful of valid rows still gets bins.
  - (ii) Package version B builds `_NLinear` with a uniform draw that is then zeroed (`R:192-197`, `779-783`). Ours
    draws nothing for W. This matters only for the RNG stream relative to a package-built model; the docstring says
    "the zero W draws nothing" but does not call it a package deviation.
  - (iii) The package warns on single-bin features; ours is silent.
  None changes outputs at a given state. Suggest adding (i) and (ii) to the DRAFT §2 / docstring disclosure at the
  next DRAFT edit.
- **N-b (B1 reachability on real shapes).** History is within the plate appearance: `sequence_data.py:187-190` pads
  indices before the PA start. Past position 0 is therefore valid only from the 6th pitch of a PA, and
  `compute_bins` raises `no rows` (`:90-91`) if no such row exists. This is not plausible on full TRAIN. In the smoke
  slice (first 8,192 TRAIN rows, `run_cuda_direct.py:123`) the count is 미측정 (no real data read). Constant physics
  or context columns become T_j = 0, not an abort. The subnormal-width abort (`:142-143`) needs two distinct float32
  values < 1.2e-38 apart, which is not plausible for standardized data. Suggest that wiring smoke print the
  per-position valid counts.
- **N-c (numerics).** At edges of widths 1e-3, 1e-5 and 1 ulp (2.4e-7) at 3.0, the float32 `addcmul` gives exactly
  0 / 1 at the bin's own edges in my probe (`/tmp/pz-p35-review/numerics.py`). The arithmetic is identical to the
  package (bitwise, item 1).
  - Extrapolation is unbounded below b_0 (first piece) and above b_T (last piece), as in Eq. 1 and the package.
  - The embedded inputs are: physics standardized by TRAIN mean/std (`sequence_data.py:127,156`); continuous context
    = standardized profile, log1p(count)/10, type proportions, and the unstandardized `prior_records_per100`.
  - Synthetic N(0,1), n = 500: last bin [1.98, 3.07], z_last = 1.92 at TRAIN max + 1 sd. Real DEV magnitudes are
    미측정.
- **N-d (encode memory, arithmetic only).** One [B, 81, 48] float32 tensor is 15.2 MiB at B = 1,024 and 60.8 MiB at
  4,096. Train steps hold a few of these (z, clamped pieces, cat, their grads); predict runs under `no_grad`. GPU peak
  and time are 미측정; the DRAFT §6 probe covers them.
- **N-e.** `ple_edges` entries past T_j + 1 (padding) are not covered by the hash. A tampered tail loads, but it is
  never used: scale/shift come from the sliced bins (`:234-237`). Optional hardening: require
  `edges[j, t+1:] == edges[j, t]` in `_network_from_archive`.
- **N-f.** No test compares bitwise against the package arithmetic. X13 (scale/shift in float64 then cast) was killed
  only by the subnormal fail-closed test. The package itself cannot be a test dependency (it lives in /tmp). A
  hard-coded golden vector would close this gap; optional.

### Wiring items (W, from this review; add to the author's open items 1–7)

- **W1** `MODEL_SOURCES`: pin `pitchmdp/direct_ple_model.py` at `d3575019…a175e` (after `cac12ea`), not `6a1541a3…`.
- **W2** `report['parameter_count']` is the **allocated** count (275,738 / 291,600 regardless of T_j). Record
  `ple_effective_parameter_count` next to it, and do not compare allocated counts to the frozen arms as if they were
  effective.
- **W3** Map labels `past{p}.phys{c}` (p-major, p = 0 oldest of the 5 past positions) and `context{i}` (export
  context index) to export column names. The order is now tested (F3).
- **W4** N1 dispatch-first in `config_check`/`training_budget`/scorer, as the author listed.
- **W5** Smoke must call `fit` with the smoke TRAIN slice only (bins from that slice; hash differs from the full run
  by design) and report per-position valid counts (N-b).

## Report / DRAFT accuracy

- All numbers in the author report match code and reruns: 23 tests, frozen 56 (79 total at c953bfd), the parameter
  counts, the 648 offset, and the M0–M5 outcomes as far as rerun (M5 rerun only).
- No wording implies performance, unseen validation or a completed Codex review. Status lines say `pending`.
- Missing from the report: the TabM extra host copy (F1, now fixed) and deviations N-a (i)–(ii).
- The report's "Memory: … Binning makes per-column copies only" is correct for binning, but missed F1.

## Mutation results (reviewer; module mutated in place, restored, sha checked after each run)

At c953bfd (23-test file): 12 / 14 reviewer mutants killed.
- X1 last piece at index 46: killed (`mixed_t…`)
- X2 leading pieces at 1..T−1: killed (`mixed_t…`)
- X3 physics bins from all rows: killed (`padded_past…`)
- X4 hash without T_j: killed (`quantile_edges…hash`)
- X5 load without top-level hash check: killed (`checkpoint_roundtrip[False]`)
- X6 load without bins-vs-config hash check: killed (`checkpoint_roundtrip[False]`)
- X7 TabM bins from TRAIN + early-stop: killed (`bins_and_hash_identical…`)
- X8 `keep` removed: killed (`zero_init…`)
- **X9 labels channel-major: SURVIVED → killed after `a831f8d`**
- X10 first piece clamped at 0: killed
- X11 last piece clamped at 1: killed
- X12 bias drawn before weight: killed (`linear0_init…`)
- X13 scale/shift in float64: killed (only by the subnormal check, N-f)
- **X14 no early-stop physics guard (single): SURVIVED → killed after `a831f8d`**

After `cac12ea` (27 tests): X1–X6 and X8–X14 killed; X7b (pooled bins in the fixed TabM fit) killed; M5 survives
(equivalent mutant). Restored module sha `d3575019…a175e` after every run.

## Commands (cwd `experiments/pitchmdp` unless noted; python = `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 perl -e 'alarm 600; exec @ARGV' /Users/song/Projects/pitcheezy/.venv/bin/python`)

1. `git log --oneline -6; git diff --stat 490cd62 c953bfd; ls /tmp/pz-p35-module-src/` (worktree root) → exit 0.
2. `cat -n` of `direct_ple_model.py`, the 5 frozen modules, the test file, the report, DRAFT §4 → exit 0. One
   `cd experiments/pitchmdp && git diff …` failed with exit 1 (the cwd was already there); re-ran without `cd` → exit 0.
3. `sed -n`/`grep -n` of `/tmp/pz-p35-module-src/rtdl/package/rtdl_num_embeddings.py`, `README.md`, `embed2022.txt`,
   `tabm.txt` (clone HEAD `a8fc2502`) → exit 0.
4. `grep` of the registry, `matrix_models._validate`, `sequence_data.build_history_indices`, `run_cuda_direct` SMOKE,
   and `.KIND`/`.FORMAT` uses → exit 0.
5. python `-m pytest -q -p no:cacheprovider tests/test_direct_ple_model.py tests/test_direct_periodic_residual_model.py
   tests/test_direct_tabm_residual_model.py tests/test_direct_tabm_model.py` at c953bfd → **79 passed** (exit 0).
   Versions: Python 3.12.14, torch 2.14.0, numpy 2.5.3.
6. python `/tmp/pz-p35-review/ref.py` (float64 reference + package comparison + RNG) → exit 0.
7. `.venv/bin/python /tmp/pz-p35-review/mutants.py` (14 mutants; each run uses the python prefix above) → exit 0
   (12 killed, 2 survived).
8. python `/tmp/pz-p35-review/mem.py` (tracemalloc, twice) → exit 0 both times.
9. python `/tmp/pz-p35-review/numerics.py` → exit 0.
10. python `/tmp/pz-p35-review/logits_snapshot.py before.npz` → exit 0.
11. Appended 2 tests; new file → 26 passed (exit 0). `mutants.py … X9 X14` → both killed (exit 0). `git commit` → `a831f8d`.
12. Module edit (TabM fit, class KIND/FORMAT) + 1 test; focused 4 files → **83 passed** (exit 0).
    `logits_snapshot.py after.npz` → exit 0, all 4 arrays bitwise equal. `mem.py` → exit 0 (PLE TabM 1.15 MB).
13. Swapped in the c953bfd module (`git show … >`), ran the new test file → 1 failed / 26 passed (exit 1, expected),
    restored the fixed module (sha `d3575019…`). `git commit` → `cac12ea`.
14. `mutants.py` with X1–X6, X8–X14 → exit 0, all killed; X7b → killed; M5 → survived (exit 0 harness).
15. `shasum -a 256` of the 5 frozen modules (worktree root) → unchanged; `git diff --stat 490cd62 HEAD -- pitchmdp/`
    → only `direct_ple_model.py`. Parameter-count script → exit 0.

Not done: the full test suite, CUDA, any real data or export, timing. `/tmp/pz-p35-review/` (scripts, npz) is left
in place. No push and no merge.

## Disposition

**Ready for wiring** at `cac12ea` (module `d3575019…a175e`), with W1–W5 carried into the wiring task. No fix is
required before wiring beyond what is committed here. Codex review is still pending.
