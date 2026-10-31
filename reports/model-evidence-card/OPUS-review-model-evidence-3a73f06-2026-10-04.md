# 검토: /showcase 예측 모델·추천 검증 카드 (`3a73f06`)

`same_model_review; Codex_review_pending`

- 검토자: Opus 5.5. 같은 모델이지만 작성자와 다른 인스턴스. 다른 모델 호출 없음. 하위 Claude 프로세스 없음.
- 대상: `git diff 7b268a3 3a73f06`. 워크트리 `codex-model-evidence-review-1004`.
- 원본(읽기 전용): 메인 레포 `b38bea6`. 커밋된 결과 JSON 세 개와 매트릭스 문서의 sha256을 다시 계산했고, 카드에 고정된 값과 일치했다.
- 한 일: 확인만 했다. ssh·GPU·서버·포트 사용 0. 2026 자료는 이미 커밋된 결과 JSON만 열었다. push·merge·8771 반영 없음.

## 처분

**수정 반영 후 승인.** 막는 문제(blocking) 1건은 이 커밋에서 고쳤다. 사소한 수정 4건과 테스트 보강도 같이 넣었다. 나머지 2건은 제안만 남긴다(7, 8). Codex 검토는 아직이다. 8771에는 반영하지 않았다.

## 숫자 대조 (원본 → 카드)

모두 원본과 정확히 같다(고정 복사본은 반올림하지 않음). 화면 반올림·부호·단위도 맞다.

- 예측 (`EXP-P11-001`, `estimands.whole`)
  - n 311,721, 경기 1,161
  - G0 NLL 1.490408 → 1.4904, 빈도 1.504081 → 1.5041
  - Δ −0.013673 → −0.0137, 95% CI [−0.014415, −0.012924] → −0.0144 ~ −0.0129
  - 재표본 10,000회(경기 단위), 음수 시드 5/5
  - ECE10: G0 0.008710, 빈도 0.001660
  - 결과 10종 (`class_reporting.eligible` 10개; `model.py:18` OUTCOMES 10개)
- DEV 기간 2025-07-01~09-30, TRAIN ~2025-04-30, 보정 05-01~06-30: `ML_EXPERIMENT_MATRIX.md:26`. "학습·보정 기간(~6월) 뒤"는 정확하다.
- 연구 혼합 (`EXP-P16-001`, `mlb_dev_blended.comparison["G15-direct-g0-mix"]`)
  - Δ −0.004062 → 0.0041, CI [−0.004383, −0.003752] → −0.0044 ~ −0.0038, 5/5
  - 대조군 G0가 P11과 같은 1.490408임은 D143에서 확인했다.
  - "서비스와 추천에는 아직 쓰지 않습니다"는 D145·D155와 맞다.
- 정책 (`MLB-2026-OPE-v1`)
  - 2,429경기, 184,288 PA, 시작 전 제외(미지 투수) 34,763
  - L1-R Δ 경계 [−0.001205, +0.000921] → −0.0012 ~ +0.0009
  - `NO_EVIDENCE_OF_IMPROVEMENT`, mei 0.001
  - 단위: 수비 WE/PA

## 발견 사항

### 1. [BLOCKING, 수정함] 2026 자료를 "학습·선택에 쓰지 않음, 1회만 평가"라고 쓴 과장
- 위치: `ModelEvidenceCard.tsx:33`(수정 전).
- 원래 문구: `사전에 등록한 방법으로 1회만 평가 (학습·선택에는 쓰지 않음)`.
- 원본과 충돌한다.
  - D87: 과거 P0 12건·P1 8건 OPE, 2026 propensity 교차 적합, 지원 집합, 평가 정책 적응, 반복 선택을 **노출 이력**으로 보존한다.
  - D131: "노출 이력이 있는 구간(D87)을 포함한 가정 조건부 regime 추정".
  - OPE JSON `interpretation`: "single registered 2026 OPE (exposed-window labels in the prereg)".
- 문제: 교수가 "2026은 한 번도 보지 않은 깨끗한 검증 자료"로 읽는다. 바로 D131이 막는 해석이다.
- 수정: 등록된 평가를 1회 실행한 것은 사실이므로 그대로 둔다. 다만 "이전 단계의 정책 평가·교차 적합·반복 선택에 2026 자료가 쓰인 노출 이력이 있어(D87), 처음 보는 자료에서의 평가는 아닙니다"를 추가했다.
  - JSON에 `policy.exposure`를 넣고 출처 결정 목록에 D87을 더했다.
  - 테스트는 OPE `interpretation`에 `exposed-window`가 있는지 확인한다.

### 2. [non-blocking, 수정함] 평가한 타석 수를 184,288로 읽게 됨
- 위치: `:32`, `:36`.
- 문제: 원래 "184,288타석 … 평가"가 먼저 나오고 "34,763타석은 평가 전에 제외"가 뒤에 따로 나와서 평가 모수가 헷갈린다.
- 원본:
  - D131: "184,288PA(E0 149,525, 미지 투수로 시작 전 제외 34,763)".
  - JSON status: COMPLETE 132,019 + CENSORED 17,506 = 149,525.
- 수정: "184,288타석 중 시작 전 제외를 뺀 149,525타석", "위에서 뺀 34,763타석은 …".
  - `evaluatedPlateAppearances`를 고정하고 테스트로 두 값과 대조한다: (전체 − 제외), 그리고 (COMPLETE + CENSORED).

### 3. [non-blocking, 수정함] "전체 MLB 311,721구"
- 위치: `:24`.
- 문제: 실제로는 적격 투구 311,721구다. 요청 341,941구 중 30,220구가 빠졌다.
  - 원본: P11 `population.requested_pitches` 341,941 / `eligible_pitches` 311,721.
  - `results-all.md:657`도 "DEV 적격 311,721구·1,161경기(요청 341,941구)"라고 쓴다.
- 수정: "전체 MLB 중 평가 조건을 만족한 311,721구(전체 341,941구 중)". `requestedPitches`를 고정하고 원본과 대조한다.

### 4. [non-blocking, 수정함] "예측은 좋아졌지만", "시드 5/5개 모두 개선"
- 위치: `:20`, `:26`.
- 문제: 시간에 따른 개선이나 인과적 개선처럼 읽힌다. 실제로는 같은 자료에서 빈도 기준과 비교한 것이다. 게다가 ECE는 빈도 쪽이 더 좋다.
- 수정: "상황 빈도 기준보다 로그손실이 낮았지만", "모두 빈도 기준보다 낮음".

### 5. [non-blocking, 수정함] 개선 문턱 문장의 비교 대상을 명시
- 위치: `:35`.
- 문제: 원래 "개선 문턱 +0.001에 미치지 못해"는 결론은 맞다. 하지만 무엇을 문턱과 비교했는지 빠져 있다.
- 원본의 실제 판정 규칙은 더 엄격하다(`labels.primary.passable_effect_approx`: Δ > 0.001 + b_V 0.00273 + s 0.00106 + 1.96 SE; D131 "편향 감안 문턱 0.0037347 모두 미달"). 그래서 카드 결론은 보수적인 방향으로 참이다.
- 수정: "범위의 위쪽 끝(+0.0009)도 개선 문턱 +0.001보다 작아 개선 근거 없음".

### 6. [non-blocking, 테스트 보강함] 스모크가 뜻이 뒤집힌 문장을 잡지 못함
- 변이 3: `화면에서도 이런 투수에게는 추천을 만들지 않습니다` → `…이런 투수에게도 추천을 만듭니다`로 바꾸고 다시 빌드했더니 **exit 0으로 통과**했다(`opus-review/mut3.log`).
- 또 `lower()`/`f4`가 `Math.abs`로 부호를 지운다(`:7`). 그래서 `MODEL_EVIDENCE_REPO` 없이 실행하면, JSON에서 Δ 부호가 뒤집혀도 화면에는 여전히 "낮음"이 나온다.
- 보강:
  - 방향 확인 assert: G0<빈도이고 CI 상한도 <0, 혼합<G0이고 CI 상한도 <0, 빈도 ECE < G0 ECE, 평가 PA = 전체 − 제외.
  - 카드 문장 고정: 화면 비추천, 노출 이력, 위쪽 끝 문장, "모든 면에서 나은 것은 아닙니다", 인과 한계, `결정 D87·D131`.
  - 지운 과장 문구가 다시 나오지 않는지 확인: `학습·선택에는 쓰지`, `예측은 좋아졌`, `모두 개선)`.
- 재변이 결과는 아래 표에 있다.

### 7. [non-blocking, 제안만] "다음 공 한 개의 결과 10종 … 이 나올 확률"
- 맞는 부분: 10종 구성은 정확하다(`model.py:18`: ball, strike, foul, out, single, double, triple, home_run, hbp, double_play). 문구도 `results-all.md:645`와 같다.
- 빠진 부분: G0는 조건부 모형이다. 입력에 `pitch_type`이 있고(`model.py:18–21`), 도달 위치는 투구 전 단계에서 학습 기간 분포로 적분한다(`model.py:1–5`, 매트릭스 "400 draw").
- 시연 위험: 교수가 "공을 던지기 전에 결과를 어떻게 아느냐"고 물을 수 있다. 카드에는 "추천은 구종 후보별 결과 확률을 비교한다"는 연결이 없다.
- 제안 diff(적용 안 함; P11 평가가 실제 구종을 조건으로 했다는 사실이 고정 원본 문서에 명시돼 있지 않아 PM 판단으로 남김):
  ```diff
  - 다음 공 한 개의 결과 {p.outcomeClasses}종(…)이 나올 확률
  + 던질 구종이 정해졌을 때 다음 공 한 개의 결과 {p.outcomeClasses}종(…)이 나올 확률. 추천은 구종 후보별로 이 확률을 비교해 만듭니다.
  ```

### 8. [non-blocking, 제안만] 인과 한계 문장
- 위치: `:42`.
- 현재 문구 "그 가정이 맞을 때만 인과 효과로 읽을 수 있습니다"는 OPE JSON `interpretation`과 같다.
- D131은 더 짧게 "인과 효과가 아니다"라고 쓰고, 추정 대상도 regime이다(추천을 계산 가능한 지점까지 따른 뒤 실제 투구로 돌아가는 경로; `censoring_rule.estimand`).
- 지금 문장도 틀리지는 않다. 시연을 단순하게 하려면 그대로 두는 것을 추천한다. 질문이 나오면 말로 "가상 경로 추정"이라고 보충한다.

### 9. [확인됨] 비지원 투수 문장
- 문장: "2025년까지 학습 기록이 없는 투수 … 화면에서도 추천을 만들지 않습니다".
- 서비스 코드 경로:
  - `experiments/pitchmdp/pitchmdp/policy_runtime.py:285–286`(시작 모집단 `UNKNOWN_PITCHER`), `:330–331`("no TRAIN history for this pitcher; league fallback refused")
  - → `scripts/demo_precompute.py:338` `STATUS_TEXT['UNSUPPORTED_UNKNOWN_PITCHER'] = '2025년까지 학습 기록이 없는 투수라 추천하지 않습니다.'`, `:708–710` `pre.status='unsupported'`, `recommendation=None`
  - → 화면 `WatchAlong`/`Game.tsx:342`이 이유 문구를 표시한다.
- live pins 경로도 같은 방향이다(`live_feed.py:368–369`). 다만 ARM-B 서비스 모드(`pins is None`)에서는 런타임이 판단한다.
- 정확하다. 차이 하나: OPE 제외는 "그 투수로 시작한 타석" 단위이고, 화면 거절은 공 단위다. "이런 투수에게는"이라는 표현은 두 경우 모두와 맞는다.

### 10. [확인됨] 코드
- CSS
  - `:is(.cv-card,.model-card)`의 명시도는 인자 중 최대인 0,1,0이다. 바꾸기 전 `.cv-card`와 같다. 규칙 순서도 그대로라 CV 카드 렌더링은 변하지 않는다.
  - 의도된 변화는 하나다: `.model-card + .cv-card{margin-top:12px}`로 두 카드 사이 간격이 24→12px로 줄었다.
  - `cv-evidence-smoke` exit 0.
- 카드 동작
  - 순서는 모델 카드 → CV 카드다(`Showcase.tsx:133–134`, 스모크에서 `+` 결합자로 확인).
  - 기본은 접힘이다. 링크·버튼·iframe·video는 0개다.
  - 360/390/1440 폭에서 가로 넘침이 없다. 새 화면은 `opus-review/model-card-open-{360,390,1440}.png`.
- JSON import는 `resolveJsonModule` + `strict` 설정으로 타입이 추론된다. tsc exit 0.
- 참고: 출처 확인은 `src/` JSON을, 브라우저 검사는 `dist/`를 본다. JSON만 고치고 다시 빌드하지 않으면 브라우저 검사는 통과한다. 반드시 빌드한 뒤 스모크를 돌려야 한다(문서화만 함).

### 11. 5분 시연에 빠진 것 (#4)
- 카드 자체는 충분하다. 고정되지 않은 P30–P33 수치는 넣지 않았다.
- 더 넣을 후보는 이미 검토된 문서에 있다. 월별 안정성 −0.0134/−0.0143/−0.0132, TRAIN에 없던 투수 28,746구 −0.010873(`results-all.md:43`). 다만 둘 다 카드의 고정 JSON에 없어서, 넣으려면 원본 대조를 추가해야 한다. 지금은 넣지 않기를 추천한다.
- 대신 20초 설명 문장을 제안한다:
  > "예측 모델은 2025년 7~9월 31만 구에서 상황 빈도보다 로그손실이 0.014 낮았습니다. 하지만 추천을 따랐을 때 승리확률이 좋아진다는 근거는 2026 정규시즌 평가에서 나오지 않았습니다. 해가 없다는 증명도 아닙니다. 화면의 추천은 서비스 모델 G0 기준이고, 더 나은 연구 모델은 아직 서비스에 쓰지 않습니다."

## 명령과 종료 코드 (모두 `perl -e 'alarm 300; exec @ARGV'`로 시간 제한, `apps/observer/web`에서)

**원본 상태(3a73f06)**
- `node_modules/.bin/tsc --noEmit -p .` → 0
- `node_modules/.bin/vite build` → 0. 결과물 `index-D7itW42y.js`, `index-CbbIe769.css`로 작성자 기록과 같다.
- `MODEL_EVIDENCE_REPO=… node tests/model-evidence-smoke.mjs` → 0 (`opus-review/model0.log`)
- `CV_M3_RECEIPT=… node tests/cv-evidence-smoke.mjs` → 0 (`cv0.log`)
- `node tests/showcase-smoke.mjs` → 0 (`sc0.log`)

**변이 시험 (각각 원복 뒤 `git status` 깨끗함 확인)**
- M1: JSON `excludedUnknownPitcher` 34763→34762 → **exit 1**. 출처 대조 `deepEqual`에서 실패(`mut1.log`).
- M2: TSX `해가 없다는 증명도 아닙니다`→`해가 없다는 뜻입니다`, 다시 빌드 → **exit 1**. `detail: 해가 없다는 증명도 아닙니다`(`mut2.log`).
- M3: TSX 비지원 투수 문장 뒤집기, 다시 빌드 → **exit 0 (빈틈)**(`mut3.log`). 그래서 발견 6을 고쳤다.
- M3b(보강 후): 같은 변이 → **exit 1**. `detail: 화면에서도 이런 투수에게는 추천을 만들지 않습니다`(`mut3b.log`).
- M4(보강 후, repo env 없이): JSON `deltaNll` 부호 뒤집기 → **exit 1**. `G0 below frequency, whole interval`(`mut4.log`).

**수정 후**
- tsc → 0
- build → 0. 결과물 `index-BpZyo5LN.js`, CSS는 그대로 `index-CbbIe769.css`.
- model smoke → 0 (`model1.log`, 원복 재확인 `model2.log`)
- cv smoke → 0 (`cv1.log`)
- showcase smoke → 0 (`sc1.log`)

## 이 커밋에서 바꾼 것
- `apps/observer/web/src/ModelEvidenceCard.tsx`: 발견 1–5 문구.
- `apps/observer/web/src/model-evidence.json`: `prediction.requestedPitches`, `policy.evaluatedPlateAppearances`, `policy.exposure`, 결정 목록에 D87 추가. 모두 원본 값이고 새 계산은 없다.
- `apps/observer/web/tests/model-evidence-smoke.mjs`: 발견 6 보강.
- `reports/model-evidence-card/opus-review/`: 로그와 수정 후 화면.

## 남은 것
- 작성자 보고서의 화면(`reports/model-evidence-card/model-card-*.png`)과 `REPORT-…-ko.md`의 문구 설명은 수정 전 기준이다. 통합할 때 갱신이 필요하다.
- 발견 7·8은 PM이 판단한다.
- Codex 검토, 8771 반영, 실제 아이폰/Safari 확인은 하지 않았다.
