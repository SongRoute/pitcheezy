# Opus 리뷰 — 모델 근거 카드 섞기 문장(N1), 3a3690d

`same_model_review; Codex_review_pending`

- 리뷰어: Opus 5.5(작성자와 다른 세션). 다른 모델 호출·하위 Claude 프로세스는 쓰지 않았다.
- 대상: `git diff 80624ca 3a3690d`. 브랜치 `codex/model-card-blend-review-1004`.
- 1차 출처: 메인 레포 `b38bea6`(`git show`로만 읽음). 외부 SSD(T7)의 파일 2개를 읽기 전용으로 확인했다(아래 표시).
- 결론: **차단 사유 없음(ACCEPT)**. 문장은 카드가 보여 주는 숫자(P11 whole g0 NLL 1.4904)에 대해 사실이다. 관전 화면과 2026 OPE의 예측기도 같은 섞기(같은 가중치)를 쓴다. 비차단 권고는 4건이다. 그중 1건(F2, 서비스 경로 고정)은 리뷰어가 테스트·근거만 고쳤고, 화면 문구는 바꾸지 않았다.

## 1. 추적 결과(질문 1·2)

**P11 수치가 무엇인지**
- `matrix_g0_whole_metrics.evaluate_whole`에서 `estimands.whole.g0 = prediction_metrics(y, predictions['primary'])`이다. 결과 파일의 `results/EXP-P11-001.json` `estimands.whole.g0.log_loss = 1.4904081967248524`가 카드의 1.4904와 같다.
- `frozen_five_predictions`에서 `primary = weights[0]*calibrated + (1-weights[0])*frequency['dev']`이다.
  - `weights[0] = archived_report['selection']['model_weight']`이고, 시드별 가중치는 `seed_primary`에만 쓰인다.
  - `calibrated`는 멤버 5개의 `dev` 배열 평균이다.
- `score_ml_g0_whole.py`가 `parent_c1_analysis_results.json['reports']['G0-global']`을 넘긴다. P11 config는 이 파일을 `analysis_results_sha256 = 82768e20…`로 고정하고(`results/EXP-P10-001.json`와 같은 sha), `june_weights: reuse … unchanged`, `refit: false`로 둔다.
- 따라서 w = **0.7467754577062221**(EXP-P10-001 G0-global `selection.model_weight`, 시드별 값 아님)이다. 빈도 기준 비중은 `baseline_weight` 0.2532245422937779 → 반올림하면 "약 25%"이며, 정직한 반올림이다.

**"신경망 5개"가 맞는지**
- 맞다. 멤버는 `flatten_mlp`(width 128, P10 config `kind`)이고 시드 0–4로 다섯 개다. G에서 0–2, C1에서 3–4를 가져왔다(P11 config `"3": "c1", "4": "c1"`, `new_fits: none`).
- 다섯 멤버는 구조가 같고 시드만 다르다.

**보정(temperature)**
- 멤버마다 5월 하순(05-16~05-31, Cpanel 2,603구) temperature로 보정한 뒤 평균한다. P11 config `may_temperature: reuse each frozen member calibration.json unchanged`.
- 문장에는 이 단계가 없다. 바로 아래 줄의 "학습·보정 기간(~6월)"이 보정의 존재를 알리므로 생략해도 괜찮다고 본다(F4).

**섞는 비율을 6월 자료로만 정했는지**
- 맞다. `selection.calibration_objective = "log_loss"`로 blend 분할에서 골랐다.
- 리뷰어가 T7 `EXP-P4-001/blend_metadata.parquet`(sha `de4b784c…`, P10 preparation `external_hashes`의 값과 같음)를 읽어 확인했다. 4,821행 모두 `month == 2025-06`, `in_cpanel == True`, 투수 20명, 110경기였다.
- 기간 상수는 `matrix_policy.py:75`, `matrix_generalization.py:144`의 `'blend': ('2025-06-01', '2025-06-30')`이다.
- **단, 6월 MLB 전체가 아니라 개발 패널 투수 20명의 4,821구다**(F1).
- `run_temporal_blend.py`는 같은 기간 규칙을 쓰는 별도 실험(rolling-origin)이다. P10 가중치를 만든 실행기가 아니다. JSON `facts`가 기간의 출처로 ML_EXPERIMENT_MATRIX 문장을 든 것은 틀리지 않았다.

**서비스·정책 예측기가 같은 섞기를 쓰는지(질문 2)**
- 관전 화면 `/watch`·라이브의 ARM-B 경로는 같은 식을 쓴다.
  - 연결 순서: `scripts/demo_precompute.bind` → `policy_identity.bind_components` → `load_g0_ensemble` → `FrozenG0Ensemble` → `rollout_policy.calibrated_conditional`.
  - 식은 "멤버별 5월 temperature → 5개 확률 평균 → `neural_weight`로 빈도와 섞기"이다.
  - docstring에 "Per-seed June weights are NOT used for the ensemble."이 있다.
- `neural_weight`는 번들 `configs/G0-RESEARCH-FROZEN-v1.json`의 `p11_frozen_calibration`(sha `e3b96ef4…`)에서 읽는다.
  - 이 값은 `matrix_g0_whole.frozen_weights`가 `selection.model_weight`를 그대로 옮긴 것이다.
  - `policy_identity.py:412`이 `g0.neural_weight == calibration['june_ensemble_model_weight']`를 강제한다.
  - 리뷰어가 T7 `EXP-P11-001/frozen_calibration.json`을 읽었다. sha256이 `e3b96ef4…`로 번들 고정값과 같고, `june_ensemble_model_weight = 0.7467754577062221`이다. **카드 가중치와 비트 단위로 같다.**
- 2026 OPE(D131, 정책 `c13cc989…`)도 같은 G0 primary 경로를 쓴다. D147 단계 A는 "정책 경로의 G0 primary … 봉인 평가 예측과 1e-6 안"으로 확인했다.
- 예외: 예전 과거 기록 화면(`PITCHEEZY_OBSERVER_LEGACY=1`)은 다른 번들(`runs/minimal-pitch-service-v1`, seed 42–46 continuous, 자체 `neural_weight`)을 쓴다. `serve.sh:38` 기본값이 `0`이라 시연 화면에는 없다(F5, 정보).

## 2. 발견 사항

### F1 (비차단, 권고) — 6월 자료가 "일부 투수"라는 점이 빠짐
- 위치: `apps/observer/web/src/ModelEvidenceCard.tsx:25`
- 구체적 실패: 교수가 "6월 MLB 전체로 비율을 정했나?"라고 물으면 답은 "아니오, 개발 패널 투수 20명의 4,821구"다. 그 비율을 7–9월 MLB 전체(311,721구)에 그대로 옮겼다. 문장 자체는 거짓이 아니지만, 바로 아래 "전체 MLB" 문장과 붙어 있어 같은 모집단으로 읽히기 쉽다.
- 제안(숫자는 이미 고정된 P10 `coverage.blend.eligible_pitches`·P11 `expected_samples.blend`에서 온다. 리뷰어가 추가한 `blend.servicePath.fitSample.pitches`를 smoke가 대조한다):

```diff
-        <span>서비스 모델의 확률은 신경망 {b.members}개의 평균에 상황 빈도 기준을 약 {Math.round((1 - b.modelWeight) * 100)}% 섞은 값입니다. 섞는 비율은 평가 기간보다 앞선 {b.blendWindow[0].slice(0, 4)}년 {Number(b.blendWindow[0].slice(5, 7))}월 자료로 정했습니다.</span></li>
+        <span>서비스 모델의 확률은 신경망 {b.members}개의 평균에 상황 빈도 기준을 약 {Math.round((1 - b.modelWeight) * 100)}% 섞은 값입니다. 섞는 비율은 평가 기간보다 앞선 {b.blendWindow[0].slice(0, 4)}년 {Number(b.blendWindow[0].slice(5, 7))}월, 일부 투수들의 {n(b.servicePath.fitSample.pitches)}구로 정했습니다.</span></li>
```
  smoke 문구도 `'섞는 비율은 평가 기간보다 앞선 2025년 6월, 일부 투수들의 4,821구로 정했습니다'`로 바꾼다.
- 리뷰어는 이 문구를 적용하지 않았다. 배포된 미리보기 문구의 결정은 PM 몫이다.

### F2 (비차단, **리뷰어가 수정**) — "서비스 모델" 아래 문장인데 근거는 연구 채점기만 고정
- 위치: `apps/observer/web/src/model-evidence.json`(`blend`), `apps/observer/web/tests/model-evidence-smoke.mjs:66-84`
- 구체적 실패: 3a3690d의 고정 근거는 P10 결과·P11 config·P11 채점 스크립트뿐이었다. 정책 경로가 시드별 가중치나 다른 번들로 바뀌어도 smoke는 통과하고, "서비스 모델의 확률은 …" 문장만 틀리게 남는다.
- 수정: `blend.servicePath`를 추가했다.
  - `b38bea6`의 다섯 파일을 sha256과 anchor로 고정했다: `matrix_g0_whole_metrics.py`의 `weights = [archived_report['selection']['model_weight'],`, `matrix_g0_whole.py`의 `frozen_weights`, `policy_artifacts.py`의 `FrozenG0Ensemble` docstring·`june_ensemble_model_weight` 읽기, `policy_identity.py`의 `load_g0_ensemble` 호출·일치 검사, `configs/G0-RESEARCH-FROZEN-v1.json`.
  - 번들의 `p11_frozen_calibration` sha가 고정값과 같은지 검사한다.
  - 번들 멤버 5개의 `june_model_weight`가 P10 시드 가중치와 같은지 검사한다.
  - `frozenCalibration.juneEnsembleModelWeight == blend.modelWeight`, `fitSample.pitches == P10 coverage.blend.eligible_pitches`를 검사한다.
  - 선택 검사: `G0_FROZEN_CALIBRATION=<T7 경로>`를 주면 실제 파일의 sha와 가중치를 대조한다.
- 화면 출력은 바뀌지 않는다(TSX 변경 없음).

### F3 (비차단, 선택) — "섞은 값"의 비율 읽기
- 위치: `ModelEvidenceCard.tsx:25`
- "평균에 … 약 25% 섞은"을 "평균 위에 25%를 더했다"로 읽을 여지가 조금 있다.
- 선택 대안: "신경망 5개의 평균(약 75%)과 상황 빈도 기준(약 25%)을 섞은 값입니다."
- 지금 문구도 틀리지 않았다.

### F4 (비차단, 생략 허용) — 멤버별 5월 보정, 시드별 점수의 정체
- 위치: `ModelEvidenceCard.tsx:25`, `:28`
- 5월 temperature 보정은 문장에 없다. 아래 줄 "학습·보정 기간(~6월)"로 충분하다고 본다. 넣는다면 "신경망 5개(각각 5월 자료로 보정)의 평균"이다.
- `:28`의 "시드 5/5개 모두 빈도 기준보다 낮음"은 5개 평균이 아니다. 시드마다 **신경망 하나**를 자기 6월 가중치(0.666–0.736)로 섞은 결과다(`seed_primary`).
  - 이제 카드가 서비스 모델을 "5개 평균"으로 설명하므로 두 대상이 섞여 읽힐 수 있다.
  - 선택 대안: "시드별 신경망 하나씩도 5/5개 모두 빈도 기준보다 낮음"
  - 기존(80624ca 이전) 문구라 이번 diff 범위 밖이다.

### F5 (정보) — 예전 과거 기록 화면은 다른 번들
- `PITCHEEZY_OBSERVER_LEGACY=1`일 때만 켜지는 예전 화면(`recommender.py` → `minimal-pitch-service-v1`)은 다른 모델이다. 기본값이 꺼짐(`apps/observer/serve.sh:38`)이라 시연에는 영향이 없다.
- 미리보기를 `LEGACY=1`로 띄우면 그 화면에는 "서비스 모델" 설명이 맞지 않는다. 변경은 제안하지 않는다.

## 3. 과장·누락 점검(질문 3)
- 과장 없음. "평가 기간보다 앞선"은 사실이다(6월 < 7/1).
- "신경망 5개"·"약 25%"·"2025년 6월"은 모두 고정 파일과 일치한다.
- 누락 중 교수가 짚을 만한 것은 F1(6월 일부 투수)이 유일하다고 본다. F3·F4는 선택이다.

## 4. 실행한 명령과 종료 코드
작업 디렉터리 `apps/observer/web`, `node_modules`는 `codex-showcase-1006` 쪽 심볼릭 링크(끝나고 삭제), 모든 명령은 `perl -e 'alarm 300; exec @ARGV'`로 감쌌다.

수정 전(3a3690d 그대로):
- `npx tsc -b` → exit 0
- `npx vite build --outDir dist-review --emptyOutDir` → exit 0
- `DIST=dist-review MODEL_EVIDENCE_REPO=/Users/song/Projects/pitcheezy node tests/model-evidence-smoke.mjs /tmp/rv1004/shots` → exit 0 (`source identity: ok`)
- `DIST=dist-review CV_M3_RECEIPT=…/CV-M3-receipt-recalculation-20261004.json node tests/cv-evidence-smoke.mjs` → exit 0
- `cp -R dist-review dist && node tests/showcase-smoke.mjs; rm -rf dist` → showcase exit 0, 임시 `dist` 삭제 확인

수정 후(F2 적용):
- `npx tsc -b` → exit 0
- `npx vite build --outDir dist-review --emptyOutDir` → exit 0
- model-evidence smoke에 `G0_FROZEN_CALIBRATION=<T7 frozen_calibration.json>` 추가 → exit 0 (`frozen calibration: ok`, `source identity: ok`)
- model-evidence smoke, T7 없이 → exit 0
- model-evidence smoke, `MODEL_EVIDENCE_REPO` 없이 → exit 0 (`source identity: skipped`)
- cv-evidence smoke → exit 0
- showcase smoke(임시 `dist`) → exit 0, 임시 `dist` 삭제

## 5. 변형(mutation) 결과 — 모두 실패해야 정상, 모두 실패함
3a3690d smoke 그대로(각 변형 뒤 `git checkout -- <파일>`로 복구):

| # | 변형 | exit | 걸린 검사 |
|---|---|---|---|
| M1 | `modelWeight` 마지막 자리 1→2 | 1 | `archived June weight (exact)` |
| M2 | 채점 스크립트 anchor 목록 비움 | 1 | `… has anchors` |
| M3 | P10 파일 sha를 0으로 | 1 | `results/EXP-P10-001.json sha256 at b38bea6` |
| M4 | `members` 4 | 1 | `members = P11 seeds` |
| M5 | `blendWindow` 7월 | 1 | `deepEqual` 기간 |
| M6 | P11 config 고정 삭제 | 1 | `blend source files` |
| M7 | 조건화 쪽 섞기 식 anchor 교체 | 1 | `blend formula anchor kept` |
| M8 | TSX에서 문장 삭제(재빌드) | 1 | `detail: 서비스 모델의 확률은 …` |
| M9 | TSX가 신경망 비중(75%) 표시(재빌드) | 1 | `detail: … 약 25% …` |
| M10 | TSX "평가 기간보다 앞선" 삭제(재빌드) | 1 | `detail: 섞는 비율은 …` |

F2 추가 검사(수정본 기준, 각 변형 뒤 복구, `cmp`로 복구 확인):

| # | 변형 | exit | 걸린 검사 |
|---|---|---|---|
| S1 | 보정 파일 고정 sha 변경 | 1 | `service bundle pins the P11 frozen calibration` |
| S2 | `policy_identity.py` 거짓 anchor | 1 | `… anchor: neural_weight = per_seed` |
| S3 | `policy_artifacts.py` 고정 삭제 | 1 | `service path source files` |
| S4 | 서비스 가중치를 시드0 값으로 | 1 | `service ensemble weight = card weight` |
| S5 | 가짜 T7 보정 파일 | 1 | `frozen calibration sha256` |

변형이 끝난 뒤 작업 트리에는 F2 수정 파일 2개 외의 변경이 없었다(`git status`). `dist-review/`와 `node_modules` 링크는 커밋 전에 삭제했다.

## 6. 화면(질문 5)
- smoke가 1440/390/360에서 가로 넘침 없음과 카드 안 `a, button, iframe, video` 0개를 검사했고, 모두 통과했다.
- 360px 열린 카드 스크린숏을 눈으로 확인했다(`/tmp/rv1004/shots/model-card-open-360.png`, 커밋 안 함). 새 문장은 "무엇을 예측하나" 항목의 세 번째 줄로 들어가 줄바꿈이 정상이다.
- 이 항목이 세 문단으로 길어졌다. 나중에 "어떻게 계산하나"로 나누면 읽기 쉽겠지만 지금은 문제가 아니다.
- 새 링크·버튼은 없다.

## 7. 리뷰어가 적용한 변경
1. `apps/observer/web/src/model-evidence.json`: `blend.servicePath`(파일 5개 sha·anchor, `frozenCalibration`, `fitSample`) 추가. 화면 출력은 바뀌지 않는다.
2. `apps/observer/web/tests/model-evidence-smoke.mjs`: 서비스 경로 검사와 선택적 T7 검사 추가(F2).
3. 이 보고서.

화면 문구(F1·F3·F4)는 바꾸지 않았다. push·merge·배포는 하지 않았다.
