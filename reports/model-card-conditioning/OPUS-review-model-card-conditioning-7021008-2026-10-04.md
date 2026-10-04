# Opus 5.5 review — model evidence card: pitch-type conditioning + recommendation method (7021008)

`same_model_review; Codex_review_pending`

- Reviewer: Claude Opus 5.5 (`claude-opus-5-5`). Same model as the author, separate session. No other models or Claude processes were called.
- Target: `git diff 88976ab 7021008` on branch `codex/model-card-conditioning-review-1004`
- Primary sources: main repo `b38bea6` read with `git show` only; served API copy `t42-offline-v3/.../api__watch__849843.json` (read only); earlier review #7 (`OPUS-review-model-evidence-3a73f06-2026-10-04.md`).
- Date: 2026-10-04 KST. No ssh, no GPU, no network. Ports 8766/8770/8771 and other worktrees were not touched.

## Disposition

**PASS with minor fixes applied. No blocking findings.** All four new sentences match the code and contract at `b38bea6`. One sentence was slightly narrower than the code ("위치" when all 8 physical channels are averaged), so I fixed it. The smoke had a real gap: dropping a pin passed silently. I fixed that and checked the fix with mutations. Two omissions a professor might ask about (the frequency blend inside the "service model", and how small the P3 shift is) are left as proposals for the PM/author. They are not new claims added by 7021008.

## Verification of each new sentence (finding 1)

| # | Card sentence (`ModelEvidenceCard.tsx`) | Verdict | Primary source at b38bea6 |
|---|---|---|---|
| a1 | :23 "던질 구종이 정해졌을 때 다음 공 한 개의 결과 10종 … 확률" | **Correct** | G0 path = `run_ml_g0_whole.py:408` → `run_ml_sharing.sharing_load_data` → `MatrixHistoryStore.from_frame` (`run_ml_sharing.py:177-184`). The current-pitch type token is `matrix_features.py:61` (row's `pitch_type` unless `candidate_pitch_types`). G0 = 5 frozen `G0-global` members (`matrix_g0_whole.py:3-5,20`). |
| a2 | :24 "… 학습 기간 기록에서 뽑은 투구 400개로 평균합니다" | **Correct; scope was narrow → fixed (F1)** | `matrix_g0_whole.py:27` `'draws': 400`; `run_ml_g0_whole.py:409-410` refuses ≠400; `JointDelivery.fit` TRAIN-only (`sequence_delivery.py:15-16`). Pools: pitcher/type/hand/count → pitcher/type/hand → league/type/hand/count → league/type/hand → league-wide fallback of any type (`sequence_delivery.py:9-12,28,41-45`). Pools need ≥20/50 TRAIN rows, so 400 draws can repeat pitches (`replace=len<draws`). "학습 기간 기록에서 뽑은 투구 400개" is literally true. The averaged vector has 8 channels, not just location: `effective_speed, release_spin_rate, spin_axis(sin/cos), pfx_x, pfx_z, plate_x, plate_z` (`sequence_data.py:21-24`). |
| a3 | :24 "아래 점수는 평가 투구마다 실제로 던진 구종을 넣어 계산했습니다" | **Correct** | DEV scoring: `run_ml_g0_whole.py:481/566` → `matrix_benchmark.predict_streamed` → `JointDelivery.logits` → `store.gather(repeated, current=samples…)` with no candidate types (`sequence_delivery.py:58`). The delivery pool key is also the row's actual type (`sample(frame)`). `git grep candidate_pitch_types` in scripts: 0 hits (same as the author's check). |
| b1 | :33 "추천은 던질 구종만 고릅니다" | **Correct** | Contract §3 "ARM-B 식 (구종-only …)"; served API `policy.name` "ARM-B P3 (pitch type only)". |
| b2 | :33 "구종 후보마다 위 확률로 타석이 끝날 때까지의 수비 승리확률을 모의 계산하고" | **Correct (see N3 for the cap)** | Contract §3: Q̂^ref = MC with frozen simulator (delivery pool → frozen predictor → frozen WE terminal) under π_ref continuation. The predictor is the same 5-member G0 (`policy_artifacts.FrozenG0Ensemble`; `ML-POLICY-MATERIALIZATION-v1.json:136` binds `G0-RESEARCH-FROZEN-v1.json`; D145 "정책 식별자 c13cc989… G0 그대로"), so "위 확률" is the right referent. Each candidate pool is keyed by `pitch_type=action` (`matrix_policy.py:121,126`) with no league fallback (`:134-135` raises). "모의 계산" is a fair plain word for MC rollouts. |
| b3 | :33 "그 투수의 학습 기간 구종 비율에서 승리확률이 높은 구종 쪽으로 비중을 옮깁니다" | **Correct (simplified; see N2)** | π_cand ∝ π_ref^B·exp(Q̂^ref/τ) on M; π_ref^B = TRAIN `CategoricalBC` (pitcher, balls, strikes, batter side, previous type; prior 20), restricted to M (contract §3, `matrix_policy.fit_bc` TRAIN date guard 2023-05-15..2025-04-30). |
| b4 | :34 "관전 화면의 위치 표시는 그 투수의 2025년까지 실제 투구 분포에서 잡은 근사이며, 평가하지 않았습니다" | **Correct** | Served API `location`: `basis: realized_delivery_proxy`, `evaluated: false`, note "위치는 이 투수의 실제 투구 분포(2025년까지) 기반 근사이며 평가되지 않았습니다." |

The author's report left one item "미확인": whether the frequency baseline conditions on pitch type. **It does.** The P11 baseline archive comes from `ContextFrequencyBaseline(HierarchicalFrequencyBaseline())` (`run_ml_matrix.py:155,166-181`). Its levels are `TYPE_KEYS = ["pitch_type", balls, strikes, stand, p_throws]` (`run_sequence_frequency_baselines.py:27-29`), with outs/bases added in the context level (`run_sequence_context_frequency.py:27`). The DEV prediction uses each row's actual type. So the comparison is like-for-like: both sides are given the thrown type. I recorded this as a JSON fact (not shown on the card).

## Findings

### Blocking
None.

### Non-blocking — applied (each one separate, checks rerun)

**F1. "공이 들어갈 위치는" was narrower than what is averaged** — `ModelEvidenceCard.tsx:24`
- Problem: JointDelivery averages the whole 8-channel delivery (speed, spin, movement, location), not just location (`sequence_data.py:21-24`, `matrix_features.py:3-4`). A professor who knows the model has speed/spin inputs could ask why only location is unknown.
- Applied:
  ```diff
  -<span>공이 들어갈 위치는 던지기 전에 알 수 없어, 학습 기간 기록에서 뽑은 투구 {c.deliveryDraws}개로 평균합니다. …
  +<span>공의 위치·구속·회전 등은 던지기 전에 알 수 없어, 학습 기간 기록에서 뽑은 투구 {c.deliveryDraws}개로 평균합니다. …
  ```
  Added `sequence_data.py` pin + `PHYSICAL_COLUMNS` anchors. Updated JSON `facts[1]`. The smoke now requires the new phrase and rejects the old one (`'공이 들어갈 위치는'` added to the overclaim list).

**F2. Smoke source check passed when pins were removed** — `tests/model-evidence-smoke.mjs:49-56` (7021008)
- Problem: the loop only checked the entries that were present. Before the fix, these mutations exited 0:
  - M4: remove the `matrix_policy.py` pin and its anchors
  - M5: add an anchor list for a file that is not pinned
  - M6: empty `files`/`anchors`
  - M3: `sourceCommit` changed to `9e78911`. These files are byte-identical there, so the card's "레포 b38bea6" and the conditioning commit could drift apart unnoticed.
- Applied:
  - assert `c.sourceCommit === ev.source.repoCommit`
  - assert the exact sorted list of required source files
  - assert anchors keys == files keys
  - assert each file has ≥1 anchor

  All of these mutations now fail (table below).

**F3. The P11 400-draw and actual-type facts were pinned only through the policy path** — `model-evidence.json` conditioning
- Problem: the card states 400 draws in the *prediction* section (P11), but the only 400 anchor was `matrix_policy.py` ("Policy requires … 400-draw"). The actual-type claim was anchored only on the `gather` default.
- Applied: pinned `matrix_g0_whole.py` (anchor `'draws': 400, 'kind': 'flatten_mlp'`) and added a `sequence_delivery.py` anchor for the DEV path `store.gather(repeated, current=…)` (no candidate types). Also pinned `matrix_g0_whole_metrics.py` (blend formula) and `run_sequence_frequency_baselines.py` (`TYPE_KEYS`) for the Q&A fact in N1. That fact is in JSON only, not on the card.

### Non-blocking — proposed, not applied (PM/author decision)

**N1. "서비스 모델" probabilities include about 1/4 of the frequency baseline** — `ModelEvidenceCard.tsx:21-27` (comes before 7021008)
- Fact: P11 scores `primary = w·mean(5 members, May temperature) + (1−w)·frequency`, with w = C1 G0-global June ensemble weight **0.7468** (`matrix_g0_whole_metrics.py:28-31`, `results/EXP-P10-001.json` `reports.G0-global.selection.model_weight`). The policy uses the same blend (`FrozenG0Ensemble`, `calibrated_conditional`).
- Professor question: "Is 1.4904 the neural network alone? Your 'model' contains the baseline you compare against." The comparison is still valid (the blend was set on June, before DEV), but the card says nothing about it. 7021008 now explains *how* the probability is formed, so this gap is more visible.
- Proposed minimal wording (the number would need a pinned JSON field and a smoke equality against `results/EXP-P10-001.json`):
  ```diff
  -  <li><b>무엇을 예측하나</b><span>던질 구종이 정해졌을 때 … 나올 확률</span>
  +  <li><b>무엇을 예측하나</b><span>던질 구종이 정해졌을 때 … 나올 확률. 신경망 5개 평균에 상황 빈도 기준을 약 1/4 섞은 값입니다(비율은 2025년 6월 자료로 정함).</span>
  ```
  Optional companion sentence (fairness, now verified): "상황 빈도 기준도 같은 구종·카운트·주자 상황별 학습 기간 빈도입니다."

**N2. "비중을 옮깁니다" does not say how small the shift is** — `:33`
- τ = 0.1 in WE units. At ≤2025 S3b the mean KL to BC was 0.0074 (D107, restated in D108). D155 also records that 11.3% of top-1 picks change with the planning seed alone (6×12 search). A professor seeing a confident top-1 pick on the watch screen may think the shift is large.
- Optional: "… 쪽으로 비중을 **조금** 옮깁니다." and/or "그 투수의 학습 기간 **카운트별** 구종 비율" (BC conditions on count, batter side and previous type).

**N3. "타석이 끝날 때까지" ignores the rollout cap** — `:33`
- The registered search uses 6 rollouts × 12-pitch cap (D155 "6×12"). An unfinished tail is valued at the frozen initial-state WE (contract §1, "cap 꼬리"). PAs rarely run 12+ more pitches, so the plain wording is acceptable. Exact wording if wanted: "타석이 끝날 때까지(최대 12구)".

**N4. Served identity ≠ evaluated identity (outside this diff, for PM awareness)**
- Served `/api/watch/849843` `policy.identity_sha256` = `a6dffaea…` (D114). The 2026 OPE evaluated `c13cc989…` after D125 recertification on low-memory code: τ table identical, S6 outputs identical. The card shows no identity, so nothing is wrong on the card. If the professor asks "is the screen's policy the one you evaluated?", the answer is "same τ 0.1 P3 / same G0, recertified identity (D125)". I did not check byte equality of the two runtimes.

**N5. League fallback share in P11 is not on record**
- DEV rows with no pitcher/type pool use the league-wide (any-type) fallback (`sequence_delivery.py:28,40`). The committed `EXP-P11-001.json` has no tier counts, so the share is 미측정 here. The card does not claim "같은 구종의 투구", so this is not an error. Do not add "같은 구종" without the tier counts.

## Layout (finding 4)
- Smoke at 1440/390/360: no horizontal overflow, card inside the viewport, collapsed by default, `a, button, iframe, video` count 0 in the card. I viewed the 360 px open screenshot: the new 추천 방식 rows wrap cleanly with no clipping. Screenshots were written to `/tmp/rv-shots-*` (not committed; the author's 390/1440 PNGs already exist). After F1 the text gets about 4 characters longer, no layout change.

## Commands (real exit codes, run in `apps/observer/web`; `node_modules` symlinked to `codex-showcase-1006`, removed at the end)

On 7021008 as delivered:
| Command | Exit |
|---|---|
| `perl -e 'alarm 300; exec @ARGV' npx tsc -b` | 0 |
| `npx vite build --outDir dist-review --emptyOutDir` (→ `index-C16DW5kO.js`, same as the author's) | 0 |
| `DIST=dist-review MODEL_EVIDENCE_REPO=/Users/song/Projects/pitcheezy node tests/model-evidence-smoke.mjs` | 0 (`source identity: ok`) |
| `cp -R dist-review dist && node tests/showcase-smoke.mjs; rm -rf dist` | 0 |

After F1–F3:
| Command | Exit |
|---|---|
| `npx tsc -b` | 0 |
| `npx vite build --outDir dist-review --emptyOutDir` (→ `index-DgRtmNyT.js`, css unchanged `index-CbbIe769.css`) | 0 |
| `DIST=dist-review MODEL_EVIDENCE_REPO=… node tests/model-evidence-smoke.mjs` | 0 (`source identity: ok`, 1440/390/360 ok) |
| `showcase-smoke.mjs` (temporary `dist`, deleted after) | 0 |

## Mutations (finding 3)

Each mutation edited `src/model-evidence.json` (or the TSX + a mktemp build) and was restored afterwards: `git checkout` before the fix, `cp` of the saved fixed file after, confirmed with `cmp`/`git status`.

| Mutation | 7021008 smoke | Fixed smoke |
|---|---|---|
| M1 `matrix_policy.py` sha last hex digit changed | exit 1 (sha256 at b38bea6) | exit 1 (same) |
| M2 anchor → `pitch_type=candidate_mutant` | exit 1 (anchor) | exit 1 (same) |
| M3 `sourceCommit` → `9e78911` (files byte-identical there) | **exit 0** | exit 1 (commit the card cites) |
| M3b `sourceCommit` → bogus 40-hex | exit 1 (`git show` fatal) | exit 1 (commit the card cites) |
| M4 drop `matrix_policy.py` file + anchors | **exit 0** | exit 1 (conditioning source files) |
| M5 anchor list for an unpinned file | **exit 0** | exit 1 (one anchor list per pinned file) |
| M6 empty `files` and `anchors` | **exit 0** | exit 1 (conditioning source files) |
| M8 empty anchor list for a pinned file | not run | exit 1 (has anchors) |
| M7 card sentence "아래 점수는 … 계산했습니다" removed, rebuilt to mktemp | exit 1 (detail) | — |
| M7′ card reverted to "공이 들어갈 위치는", rebuilt to mktemp | — | exit 1 (detail: 공의 위치·구속·회전 등은) |

Conclusion: the `git show` identity check does fail on a changed file byte or anchor (M1/M2/M3b). Before F2 it did not fail when pins were *removed*.

## Files changed by this review
- `apps/observer/web/src/ModelEvidenceCard.tsx` (F1, one phrase)
- `apps/observer/web/src/model-evidence.json` (F1/F3: 4 new pins, 6 new anchors, facts[1] reworded, Q&A fact added)
- `apps/observer/web/tests/model-evidence-smoke.mjs` (F1/F2: commit equality, exact file list, anchor key equality, non-empty anchors, new card phrase, old phrase rejected)
- this report

Not committed: `dist-review/` (removed), the `node_modules` symlink (removed), screenshots in `/tmp`. Not pushed, merged or deployed. 8771 still serves 88976ab, so redeploying is the PM's decision.
