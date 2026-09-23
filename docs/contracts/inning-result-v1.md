# C → D 조건부 이닝 결과 v1

이 계약은 기존 `inning-eval-v1.md` 계산을 D에 전달한다. 새 가치 모형이나 실제 교체 효과를 만들지 않는다. 버전은 `inning-result-v1`, 이번 지원 시나리오는 **투수 교체 직전 타순과 현 투수를 고정한 조건부 keep** 하나다. 기존 PA `event_analysis`/`replacement`와 별도 객체이며 합산하지 않는다. 기존 API 응답·DB·화면에 자동 연결하지 않는다.

## JSON 형식

모든 필드는 필수다. null은 미측정이며 0과 다르다. 알 수 없는 버전·잘못된 수치·부적합 원문은 오류로 거절한다.

```text
schema_version: "inning-result-v1"
kind: "conditional_keep"
status: "bounded" | "unavailable"
reason: nonempty string
linkage: {
  game_pk: positive integer,
  keep_pitcher_id: positive integer,
  official_game_date: YYYY-MM-DD,
  anchor_kind: "immediately_before_logged_pitching_substitution_action",
  anchor_time_utc: UTC ISO8601 string,
  anchor_action_index: nonnegative integer,
  first_observed_pitch_id: string  // reference only; NOT the decision timestamp
}
scope: {
  horizon: "inning_end", stopping_boundary: "current_half_inning_or_game_end",
  value_target: "final_game_win_probability",
  perspective: "initial_defense", initial_defender: "home" | "away",
  unit: "probability", additive_to_pa: false
}
estimate: {
  point: null, lower: number|null, upper: number|null,
  interval_kind: "unresolved_mass_bound"
}
coverage: {
  resolved_mass: number|null, unresolved_mass: number|null,
  unresolved_reasons: { reason_code: number }, model_calls: nonnegative integer|null
}
assumptions: ["keep_pitcher_fixed", "prechange_lineup_fixed",
              "frozen_train_repertoire_policy", "no_future_substitutions"]
profiles: { default_batter_ids: positive integer[] }
replacement: {
  status: "unavailable", value_pp: null, interval_pp: null,
  reason: "actual_eligible_substitutes_unverified"
}
provenance: {
  usage: "historical_research", source_result_sha256: SHA256,
  source_anchor_sha256: SHA256, model_bundle_sha256: SHA256,
  evaluation_identity: object  // original identity, including full initial state
}
```

`bounded`는 point=null, 0≤lower≤upper≤1, resolved+unresolved≈1, upper−lower≈unresolved, lower≤resolved, sum(unresolved_reasons)≈unresolved를 만족한다(절대 허용 오차 1e−8). 모든 수치는 유한해야 하며 bool을 숫자로 받지 않는다. `unavailable`은 estimate의 숫자와 coverage 질량/호출 수가 전부 null, reasons={}여야 한다. 둘 모두 replacement 숫자는 null이다. 수치를 지워 unavailable로 바꿀 때도 연결/출처/가정은 보존한다. malformed 입력을 조용히 unavailable로 변환하지 않는다.

evaluation_identity 필드는 기존 원문의 provider_identity, initial_state_and_count, lineup_sha256, evaluation_config_sha256, policy_id, horizon, initial_defender로 고정한다. policy_id는 `frozen_train_repertoire_frequency_v1`. state는 date, inning, topbot, outs, bases, home_score, away_score, balls, strikes이며 원문 topbot 규약은 `Top`/`Bot`이다. Top의 초기 수비팀은 home, Bot은 away. date는 linkage 공식 날짜와 같다. inning≥1, outs 0~2, bases 0~7, 점수≥0, balls 0~3, strikes 0~2이며 모두 정수다. 모든 정수는 JavaScript 안전 정수 범위 이내다. UTC 문자열은 실제로 존재하는 날짜/시각이어야 하고 Z 또는 +00:00을 받는다. 첫 관측 pitch_id 형식은 `game_pk:양의타석번호:1`이며 game_pk가 linkage와 일치해야 한다.

## 변환과 연결 검증

원문은 `results/EXP-C-INNING-001/conditional_keep_777063.json`, anchor는 별도 `C-ROSTER-001/decision_anchor_777063.json`이다. 실제 파일 bytes SHA를 계산하고 원문 source_anchor_sha256와 일치시킨다. 게임/투수/anchor 종류·초기 상태·수비 관점·원문 evaluation_identity의 bundle/horizon을 교차 확인한다. 원문 kind는 `conditional_fixed_prechange_lineup_keep_pitcher`, status는 bounded만 수용한다. 원문의 값·질량·호출 수·기본 프로필을 그대로 전달하며 중간값/95% 신뢰구간/교체 우위는 계산하지 않는다.

결정 전 타순 9개를 next_slot에서 회전한 ID 목록과 원문 lineup_batter_ids가 같아야 한다. 기본 프로필 ID는 해당 타순의 부분집합이며 원문 profile_sources와 일치해야 한다. anchor의 cutoff 검증이 true이고 이전 근거의 종료 시각은 anchor보다 엄격히 앞서야 한다. 이후 대타/감사 이벤트는 출력에 싣지 않는다. 공식 경기 날짜는 원문 identity에서 보존하고 UTC 날짜로 덮어쓰지 않는다.

프로덕션에 연결하는 다음 단계에서는 **게임·교체 이벤트·초기 상태가 같은 시점**에만 보여야 한다. 첫 관측 pitch_id가 일치한다는 이유만으로 교체 후 타석에 연결하면 안 된다. 이번 단계는 이 연결 준비용 fixture와 D 소비 함수까지이며 기존 세션 GET/PA 사건 카드의 의미를 바꾸지 않는다.

## D 표시 규칙과 예제

D는 런타임 검증 후 별도 표시 모델로 변환한다. 제목 `조건부 이닝 전망`, 범위 `현재 반이닝 종료까지 · 초기 수비팀 관점`, 핵심 라벨 `조건을 고정한 경기 승리 확률 범위`. 실제 예제는 **52.10%–53.47%**, 미해결 확률 **1.36%**, 기본 프로필 **2명**으로 표시한다. 이는 확률의 % 표시이며 %p 차이가 아니다. `Wheeler 유지 + 당시 타순 유지` 조건과 `실제 교체 효과 미측정`을 함께 보여준다. 팀 이름이 없으면 홈/원정 수비팀으로 표시하며 이름을 추측하지 않는다.

신뢰구간, 성공률, 기여 비중, 추천 우위로 이름을 바꾸지 않는다. unavailable에서는 범위를 숨기고 `계산 결과 없음`과 이유를 표시한다. 검증 실패는 오류이며 0%/빈 정상 카드로 폴백하지 않는다. 전체 미래 타순·이후 대타는 기본 표시 모델에 전달하지 않는다. source hash/상세 모델 ID는 제품 기본 문구에 넣지 않는다.

실제 bounded fixture와, 그 연결 정보를 보존해 숫자만 제거한 **개발용 unavailable fixture**를 제공한다. 개발용 fixture는 새 실제 평가 결과가 아니다. 실제 CV나 새 평가/학습 실행 없이 기존 산출물만 변환한다.
