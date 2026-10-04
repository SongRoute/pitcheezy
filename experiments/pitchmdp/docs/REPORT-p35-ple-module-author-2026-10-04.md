# REPORT — EXP-P35-001 PLE module + focused CPU tests (author), 2026-10-04

Status: `author_self_tested; same_model_review_pending; Codex_review_pending`. Author: one Opus 5.5 worker
(coordinator v4 cycle-015). No other model was called. No WebFetch/WebSearch was used. No Claude process was spawned.
Branch `codex/p35-module-1004`, base `490cd62` (= `a5fe718` P34 register + P35 DRAFT/review/disposition).
No fit on real data exists. **This report makes no performance claim.**

## 요약 (Song용)

- 한 일: 원문(패키지 소스 a8fc250, TabM 논문 v3, 2022 PLE 논문 v4)을 직접 읽고 사실 5개를 확정한 뒤, PLE 모듈과 CPU 테스트 23개를 새 파일로만 추가했다.
- 결과: 새 테스트 23/23 통과, 기존 고정 모듈 테스트 56/56 통과. 고정 모듈 5개의 sha256은 전후 동일. 변이 6개 중 5개를 테스트가 잡았다(1개는 설계상 잡을 수 없음, 아래 설명).
- 원문 확인으로 바뀐 것: PLE 선형층에 **편향 없음**(초안의 "bias zero"는 틀림), 구간이 1개뿐인 열은 **양쪽 모두 자르지 않음**.
- 다음: 같은 모델 검토 → Codex 검토 → 연결 작업(registry/factory/runner/scorer, 이번 범위 밖).

## Commits

| sha | content |
|---|---|
| `3232f1b` | docs: DRAFT §2 B3 re-read (old WebFetch text kept struck through) + two §4 formula lines (T_j = 1 unclamped; no Linear_j bias; w0/c0 init) |
| `5412d11` | feat: `pitchmdp/direct_ple_model.py` + `tests/test_direct_ple_model.py` (22 tests) |
| `1894a0d` | test: padded-weight isolation check (23rd test); DRAFT §2 wording "table" → "list" |
| (this file) | docs: this report |

Paths are relative to `experiments/pitchmdp/`. File sha256 at `1894a0d`:
- `pitchmdp/direct_ple_model.py` `6a1541a33f36d02787ba33c9061ef55ab2c48030b00f3aa7992be0876c637425`
- `tests/test_direct_ple_model.py` `2fa3e5eefa4e158253c9dbfaa12002464bae2a40f65710e1798b91d618c77da5`

Name deviation: DRAFT §10.1 names the file `direct_ple_residual_model.py`. The task brief names it `direct_ple_model.py`, so
the brief's name is used. The wiring task should use `direct_ple_model` in `MODEL_SOURCES`.

## Frozen-module sha256 (before = `490cd62` checkout, after = `1894a0d`; identical, also asserted in a test)

- `direct_periodic_model.py` `18f0b96a9b8a0d9eb39af2de4b0cc1b7a10a0dd42ca92977f726042e1aec80d3`
- `direct_periodic_residual_model.py` `ee1b3d3f84b19f3564c853acf2bd12c640b1e5e597b8c62d485f839aeb8c992b`
- `direct_residual_model.py` `7feb9cfdcad85f4cb02c50a75e2ba2b528dd5546e600526b0334dcf0cf8e2d59`
- `direct_tabm_model.py` `c968aec94fd7278d30dbbc53cb81fe4fcfc23c0b33b912a48ee09122e216b319`
- `direct_tabm_residual_model.py` `6a9daff6c3de13a753357d8615bd9f0d5a7bfe796b350f7f7bd0a5e60c9287cf` (= DRAFT `6a9daff6…87cf`)

`git diff --stat 490cd62 1894a0d`: 3 files (DRAFT, new module, new test), 848 insertions, 22 deletions. No other file changed.

## Step 1 — B3 primary-source re-read (done before module code; committed `3232f1b` before `5412d11`)

Sources (in `/tmp/pz-p35-module-src/`, left in place):
- Package: `git clone --filter=blob:none https://github.com/yandex-research/rtdl-num-embeddings` → HEAD
  `a8fc25025c83f2321c63ff127a3bcef83bb1bfb5` (2025-04-16, "[package] Improve the introductory code snippet",
  `__version__ = '0.0.12'`). This is the `a8fc250` the periodic module already cites. Version B is present (added in
  `f2092094`), so no other commit was needed. `R` = `package/rtdl_num_embeddings.py` at that commit.
- TabM: `arxiv.org/pdf/2410.24210v3` (sha256 `a6988aa10d726e99c706dc92856c5a07f4e3f9fa854d08c8f854f06dc83ea34f`;
  page 1 contains "2410.24210v3"). PLE 2022: `arxiv.org/pdf/2203.05556` resolved to v4 (sha256
  `22b97677fec340fc9f95275c9501a8ad5e2fb01a82c0ed829aeb44d33163a0d8`; page 1 contains "2203.05556v4").
- Text: no PDF text extractor is installed (`pypdf`/`fitz`/`PyPDF2`/`pdfminer`/`pdftotext` all absent), so the text
  came from the arXiv HTML of the same versions (`/html/2410.24210v3`, `/html/2203.05556` → "arXiv:2203.05556v4"),
  converted with a stdlib `html.parser` script. Page numbers came from a 10-line macOS PDFKit Swift script that runs
  a substring search on the downloaded PDFs (`/usr/bin/swift`, already present).

Settled facts (exact quotes; the full block is in DRAFT §2):

- **(a) Linear_j bias in version B: none.** `R:772-778`: `self.linear = _NLinear(len(bins), self.impl.get_max_n_bins(),
  d_embedding, # For the version "B", the bias is already presented in self.linear0. bias=not is_version_B,)`.
  Zero init `R:779-783`: `if is_version_B: # Because of the following line, at initialization, # the whole embedding
  behaves like a linear embedding. … nn.init.zeros_(self.linear.weight)`. Forward `R:792-805`: `x_linear =
  self.linear0(x)`, `x_ple = self.linear(self.impl(x))`, ReLU only if `activation`, `return x_linear + x_ple`.
  → The DRAFT's "bias zero" / "+8 per feature" was wrong; the review's recollection (B3.1) was right.
- **(b) `LinearEmbeddings` init.** `R:79-86`: `self.weight = Parameter(torch.empty(n_features, d_embedding))`,
  `self.bias = Parameter(torch.empty(n_features, d_embedding))`, `d_rqsrt = self.weight.shape[1] ** -0.5`,
  `nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)`, `nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)`. Forward `R:95`
  `torch.addcmul(self.bias, self.weight, x[..., None])`. → w0, c0 ~ U(−1/√8, 1/√8), weight drawn first.
- **(c) `compute_bins`.** `R:348-350` `n_bins: int = 48`. `R:456-461` `bins = [q.unique() for q in torch.quantile(X,
  torch.linspace(0.0, 1.0, n_bins + 1).to(X), dim=0).T]`. `R:453-455`: "limiting the number of quantiles by the number
  of distinct values is NOT the same as removing identical quantiles after computing them". `R:404-410`: a non-finite X
  or "columns with just one distinct value" → `ValueError`. `_check_bins` `R:313-339`: ≥ 2 finite, strictly increasing
  edges; one bin only warns. 2022 §3.2.1 (PDF p. 4): "b_t = Q_{t/T}(…J_train)" … "Trivial bins of zero size are removed."
- **(d) Padded layout.** `R:556-564`: "x_ple_actual = [*x_ple[:-1], *zeros(max_n_bins - n_bins), x_ple[-1]]"; "The last
  encoding component is **always** located in the end, even if n_bins == 1"; "The leading n_bins - 1 components are
  located in the beginning"; "Everything in-between is always set to zeros". Per-bin `w = 1.0 / bin_width`,
  `b = -bin_edges[:-1] / bin_width` in zero buffers (`R:600-651`). Clamps `R:656-681`: `x[..., :1].clamp_max(1.0)`,
  `x[..., 1:-1].clamp(0.0, 1.0)`, last `clamp_min(0.0)`, except `torch.where(self.single_bin_mask[..., None],
  x[..., -1:], …)` for single-bin features (no clamp).
- **(e) PLE formula.** 2022 §3.2 Eq. 1 (PDF p. 4): "e_t = 0 if x < b_{t−1} AND t > 1; 1 if x ≥ b_t AND t < T;
  (x − b_{t−1})/(b_t − b_{t−1}) otherwise"; "For T = 1, the PLE-representation is effectively equivalent to the scalar
  representation"; "The cases (x < b_0) and (x ≥ b_T) are also covered by Equation 1 (which leads to (e_1 ≤ 0) and
  (e_T ≥ 1) respectively)". → The first piece extrapolates below and the last piece above the TRAIN range (N14). T = 1
  is unclamped on both sides. The DRAFT §4 formula clamped the T = 1 case; this was fixed in `3232f1b`.
- TabM paper: §3.4 (PDF p. 6) "we use a slightly modified version of the piecewise-linear embeddings (see subsection
  D.8 for details)"; §A.2 (PDF p. 14) "we use † to mark TabM variants with the piecewise-linear embeddings" and the
  first adapter R is initialised from "N(0, 1)"; §D.8 (PDF p. 20) "uses a different parametrization and
  initialization. See the source code for details." The paper does not spell out "version B". The link comes from
  the package README `README.md:251-256`: "use `version="B"` and `activation=False`", "\"B\" is the version used in
  the different paper about the TabM model", "starting points are `d_embedding=12` with `activation=False`".
  This is stated in DRAFT §2.
- Nothing stayed 미확인 among (a)–(e). Two things are still unverified: whether TabM's own experiments used exactly
  package version B (the paper only says "see the source code"), and real-TRAIN ΣT_j / degenerate-column count
  (미측정, no real data read).

## Step 2 — module (`pitchmdp/direct_ple_model.py`, new file)

- `compute_bins(arrays, context_continuous, n_bins=48)`: uses only the arrays passed in. Each physics scalar
  (past position p, channel c) uses the rows where `valid[:, p]` is true; each context scalar uses all rows. Steps:
  numpy float64 quantiles (`method='linear'`) at `linspace(0, 1, 49)` → cast to float32 → `np.unique` →
  `check_bins`. A column whose min equals its max (exactly one distinct value) keeps one edge, so T_j = 0 (B1).
  Fails closed with `ValueError` on: an empty column, a non-finite value, non-float32 or empty edges, T_j > 48,
  edges that are not strictly increasing, or a float32 scale `1/width` or shift that is not finite.
- Buffers: persistent `ple_edges` [81, 49] (padded by repeating the last edge) and `ple_counts` [81]. Derived and not
  persistent: `ple_scale`, `ple_shift`, `ple_mask` [81, 48], `ple_single` [81]. Parameters: `linear0_weight` and
  `linear0_bias` [81, 8], and `ple_weight` [81, 48, 8] (zero init, no bias).
  Forward: `addcmul(c0, w0, x) + einsum('nfb,fbd->nfd', PLE(x), W * mask)`, then `* keep`.
- `ple_bins_sha256`: sha256 over each feature in embedding order: T_j as `<i8`, then its edges as `<f4`.
  It is recorded in `net.config`, so it also appears in `report['network']`. `net.config` also records `ple_n_bins`,
  `ple_degenerate_columns` (labels `past{p}.phys{c}` / `context{i}`) and `ple_effective_parameter_count`.
  The archive stores the hash at the top level as well.
- `PLEResidualNetwork` / `PLEResidualTabMNetwork`: each calls the frozen `PeriodicResidualNetwork` /
  `PeriodicResidualTabMNetwork` constructor, then deletes `frequencies` / `embedding` and pops `n_frequencies` /
  `sigma` from the config. It then draws w0 and c0 last; the zero W draws nothing.
- `DirectPLEResidualModel(DirectResidualModel)` and `DirectPLETabMResidualModel(tm.DirectTabMModel)`: `fit` runs the
  zero-current-physics guard, computes bins from `train` only, then calls the frozen `fit` unchanged. `network()`
  raises if bins are absent. `logits` / `predict` / `load` never call `compute_bins` (tested). `load` rebuilds bins
  from the archived buffers and refuses: another format or kind, an architecture other than the fixed one (`n_bins`,
  `d_embedding`, backbone, k, scaling_init), a config key mismatch, or edges whose hash differs from the config or
  top-level hash.
- Memory: fit and predict encode only mini-batches (frozen loops: 1,024 train rows, 4,096 predict rows). Binning
  makes per-column copies only, never [N, 81, 48] over TRAIN.
- Parameter counts at the real shapes (n_context 53, n_token 38, length 6, `CONTINUOUS['counts']`) are tested:
  allocated `ple_residual` 275,738 and `ple_tabm_residual` 291,600. These equal the review N3 bounds minus 648
  (no Linear_j bias), and the allocated count does not depend on T_j. Effective = 244,634 + 8·ΣT_j (single) or
  260,496 + 8·ΣT_j (TabM). The frozen references are 244,898 and 260,760, both asserted.

## Step 3 — tests (`tests/test_direct_ple_model.py`, 23 tests, synthetic tensors ≤ 500 rows)

The tests cover the following:
- Bins come only from `train`: changing the early-stop rows leaves the hash and buffers unchanged; changing TRAIN
  changes them; pooled TRAIN + early-stop bins would differ.
- Padded past positions never enter binning: noise there leaves the hash unchanged; a past position with no valid rows
  raises `ValueError`.
- The quantile rule and the hash definition each match an independent recomputation.
- A constant column gets T_j = 0, is recorded as `['context40']`, encodes to 0, and its weight stays exactly 0 after a step.
- Fail-closed cases: NaN input, inf edge, 50 edges, repeated or decreasing edges, float64 edges, empty edges, wrong
  feature count, and a subnormal width whose scale overflows.
- float32 edge collapse: two adjacent float32 values with level 24/48 at sorted position 199.5. The test shows that
  unique-before-cast would produce a non-increasing edge, and the module returns `[lo, hi]` (T = 1).
- Mixed T_j (0, 1, 2, 48) in one padded batch match a float64 Eq. 1 reference at probes below, at and above the
  edges. The test checks the package layout (last piece at index 47, zeros between), the unclamped T = 1 piece,
  first/last extrapolation and the mask pattern.
- Zero init: at init the embedding equals `w0·x + c0` times keep. Padded past positions embed to exactly 0, and
  noise in them does not change the output.
- w0 and c0 equal the P32 `LinearResidualNetwork` weight and bias for the same seed. This confirms the RNG order and
  the U(−1/√8, 1/√8) bound.
- Backbone identity (single and TabM): every shared parameter equals the frozen periodic counterpart for seed 7. The
  removed set is exactly `{frequencies, embedding.weight, embedding.bias}` and the added set is exactly
  `{linear0_weight, linear0_bias, ple_weight}`.
- Parameter counts as listed above.
- After 5 AdamW steps (wd .01) the 93 padded rows of `ple_weight` are exactly 0, and every real piece has moved.
  The report shows `parameter_count` 275,738 and `ple_effective_parameter_count` 275,738 − 8·93.
- Bins and hash are identical across seeds 29 and 30 and the TabM model (seed 31); the parameters differ.
- The same seed gives identical logits.
- `predict` and `load` never recompute bins (`compute_bins` monkeypatched to fail).
- `network()` without fit bins raises.
- TabM: member shape, predict equals the mean member probability, architecture and report kind, physics guard, and
  the inherited `backbone` is disabled.
- Archive round trip (single and TabM) plus refusals: n_bins, d_embedding, dropout, config hash, top-level hash, kind,
  k, edge tampering ("hash"), architecture n_bins, cross-format and frozen-class loads.
- Architecture constants are not aliased.
- Frozen-module sha256 are hard-coded and asserted.
- Padded weights, even when set non-zero, cannot change the output.

## Commands (cwd = worktree unless noted; all exit 0 unless stated)

1. `git log/status`, `cat`/`sed -n` of the DRAFT, review, disposition and the 5 frozen modules plus the existing
   tests → read only.
2. `shasum -a 256` of the 5 frozen modules → `/tmp/pz-p35-module-frozen-before.txt`. The same command printed exit 2
   overall, because a chained `ls tests | grep` in the same line found nothing at the repo root. The hash output was complete.
3. `perl -e 'alarm 300' git clone --filter=blob:none https://github.com/yandex-research/rtdl-num-embeddings
   /tmp/pz-p35-module-src/rtdl` → exit 0; `git cat-file -t a8fc250` → commit (HEAD).
4. `curl -sSfL` of the two arXiv PDFs and the two arXiv HTML pages (alarm 200) → exit 0 each. The extractor probe
   (`python -c "import pypdf"` etc.) found none; that command exited 1 because `which pdftotext` failed.
5. `python3 /tmp/pz-p35-module-src/h2t.py` (stdlib HTML → text) → exit 0. Then `grep`/`sed` of the texts and the
   package source/README.
6. `swift /tmp/pz-p35-module-src/pages.swift <pdf> <phrases>` (PDFKit page search, alarm 300) → exit 0, pages 1, 4,
   6, 14 and 20 as cited.
7. Python edit of the DRAFT plus `sed -i ''` line-reference fixes, then `git commit` → `3232f1b`.
8. Tests were run each time as `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 perl -e 'alarm 600; exec @ARGV'
   /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest -q -p no:cacheprovider <files>` from `experiments/pitchmdp`:
   - First run of the new file: 5 failed / 17 passed (exit 1). Causes: a test regex `differs` against the message
     "differ from"; a test fixture whose collapse case had no interpolated level (n = 401); a subnormal width (1e-38)
     that does not overflow; a pieces-learn assertion using bins from another sample. All four were test-side fixes;
     no module code changed. A preceding `sed` multi-line edit was a no-op and was replaced by a Python edit.
   - Debug script (bounded, 1 thread): the zero pieces were the last piece of 3 features with no TRAIN row above b_47.
   - Re-run: 22 passed (exit 0).
   - Frozen focused tests `test_direct_periodic_residual_model.py test_direct_tabm_residual_model.py
     test_direct_tabm_model.py`: 56 passed (exit 0).
9. `shasum` diff against the before-list → identical. `git commit` → `5412d11`.
10. Mutation runs: `python3 /tmp/pz-p35-module-src/mutate.py <module> i`, then the new test file, then
    `git checkout -- <module>`, for i = 0..5. Module sha after each restore: `6a1541a3…7425`, unchanged.
11. Added the isolation test: 23 passed. Re-ran M5: 23 passed (survives, see below). Wording fix, then final runs:
    new file 23 passed (exit 0) and frozen focused 56 passed (exit 0). `git commit` → `1894a0d`.

## Mutation results (module mutated in place, restored by `git checkout`, 23- or 22-test file)

| # | mutant | result |
|---|---|---|
| M0 | single-head bins from TRAIN + early-stop rows | **killed**: 3 failed (`bins_come_from_the_train_argument_only`, `constant_column…`, `bins_and_hash_identical_across_seeds…`) |
| M1 | `np.unique` before the float32 cast | **killed**: `float32_edge_collapse…` |
| M2 | T = 1 last piece clamped at 0 | **killed**: `mixed_t_padded_batch…` |
| M3 | PLE weight random init instead of zeros | **killed**: 3 failed (zero-init, constant column, padded-zero) |
| M4 | B1 rule removed (one distinct value → ValueError) | **killed**: `constant_column…` |
| M5 | weight mask removed from the einsum | **survives (22/22 and 23/23 pass)** — by construction: padded encoding entries are exactly 0 (scale = shift = 0), so padded weights get zero gradient and contribute 0 with or without the mask. The mask is a second guard (N4 asked for it); no reachable state exists in which it changes the output, except non-finite padded weights, where `inf·0` is NaN either way. |

## Disclosures / deviations (also in the module docstring)

- Deviations from the package: numpy float64 quantiles cast to float32 (package: float32 `torch.quantile`, float32
  linspace); T_j = 0 for a one-valued column (package raises); physics columns binned on valid rows only; fixed padded
  width 48 (package: max T_j); d_embedding 8 (README start 12); unweighted quantiles (sample weights are ignored for
  binning).
- AdamW wd .01 now also applies to w0, c0 and W, and the global grad-norm clip 5 includes them (review §2 coupling).
- `linear0` behaves like the P32 LR embedding without ReLU; on CPU the extra draws shift the CPU dropout stream (N2,
  not tested on CUDA).
- DRAFT edits were made outside §2: the §4 formula comment lines for T_j = 1 and bias, and the §4 w0/c0 init bullet.
  Old text is struck through. These lines state the settled facts; no arm, seed, contrast, gate or budget changed.

## Open items for the wiring task (not done here)

1. `direct_ple_registry.py` / `direct_ple_factory.py`. Arm keys `ple_residual` / `ple_tabm_residual`, seeds exactly [29..33].
   The **P35 predicate must be checked first** in `run_cuda_direct.config_check`/`training_budget` and in
   `score_cuda_direct` (N1). Add a test that a P35 config never reaches the P34 `check_family`.
2. Runner `fit.json`: lift `report['network']['ple_bins_sha256']`, `ple_n_bins` and `ple_degenerate_columns` to top-level
   fields and map the labels `past{p}.phys{c}` / `context{i}` to export column names. Then check that all 10 PLE fits
   carry one hash (N12), and report the full-run degenerate count.
3. Factory post-load checks: the continuous map must equal `CONTINUOUS['counts']`, and the bins hash must equal the
   run's recorded hash.
4. `MODEL_SOURCES` must include `pitchmdp/direct_ple_model.py` (sha above, to be re-pinned after the review disposition).
5. Smoke (8,192 first TRAIN rows): bins come from the smoke subset, so its hash differs from the full run by design.
   The B1 rule makes constant-slice columns T_j = 0 rather than an abort.
6. Not measured: CPU/GPU time of the [N, 81, 48] encode and einsum, peak memory, CUDA dropout-stream equality with
   periodic_residual (N2), and real-TRAIN ΣT_j. The timing probe follows the DRAFT §6 Amendment-2 method only.
7. The pre-score extractor and the `결정 필요` go are unchanged from the DRAFT §10 checklist.

## Not done / rules kept

No ssh/newsrv, GPU/MPS, ports, 2026 data, real export, training on real data, full test suite, install, push, merge,
deploy, team message, deletion. The `/tmp/pz-p35-module-src` and `/tmp/pz-p35-module-frozen-before.txt` files are left
in place. No permission denials occurred.
