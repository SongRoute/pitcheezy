# C → D 조건부 이닝 결과 전달

후속 저장/API는 [C→D API 인수인계](C-D-api-v1.md)에서 완료했다. 아래는 결과 형식 전달 시점의 기록이며, 최신 다음 단계는 별도 기록 선택과 UI 연결이다.

2026-09-23. 시작 `05becb5`, 계약 사전 커밋 **`1403211`**. B002 미채택 이후 다음 단계인 C 연구 결과 형식 고정과 D 소비 준비를 진행한다. CV는 제외하며 새 학습·추론·데이터 수집 없이 기존 결과만 변환한다.

## 경계와 의미

계약은 `docs/contracts/inning-result-v1.md`, 버전은 `inning-result-v1`이다. 모델은 기존 `minimal-pitch-service-v1`, bundle manifest SHA `43ece920cb60c6c24ea9f1e720a7000b3b83038a3e24ac27d77ef17a2e8f0f1f`를 그대로 식별한다. 확률은 초기 수비팀의 경기 승리 확률이며, 전파 종료는 현재 반이닝 또는 경기 종료다. 범위는 미해결 확률 질량으로 생긴 계산 상·하한이고 95% 신뢰구간이 아니다. PA 기여 값과 합산하지 않는다.

기존 실제 입력은 `results/EXP-C-INNING-001/conditional_keep_777063.json`(SHA `8c2d700508fd98c41bba04d94f09114f5f60e627259ab63f976a7147c4af56b2`)과 SSD `C-ROSTER-001/decision_anchor_777063.json`(SHA `e2a64575286160c6a30591a2a59b213b467393323a8884b0a802fb184611ac3c`)이다. 현 투수와 당시 타순을 유지한 조건부 계산을 실제 교체 우위로 바꾸지 않는다. 공식 경기 날짜 2025-07-21과 UTC 이벤트 날짜 2025-07-22를 구분한다. 이후 대타 감사 정보는 D payload에서 제외한다.

D의 기본 문구는 ‘조건부 이닝 전망’, ‘초기 수비팀 관점’, ‘현 투수 유지 + 당시 타순 유지’, ‘실제 교체 효과 미측정’이다. 선수 이름을 입력받지 않았으므로 ID만 보고 이름을 추측하지 않는다. 범위는 %로 표시하고 %p 차이 또는 기여 비중으로 표시하지 않는다. 미래 전체 타순이나 내부 hash는 기본 표시 모델로 넘기지 않는다.

## 구현 범위와 남은 의존성

- C는 독립 검증기/파일 변환기와 immutable JSON 예제를 제공한다. 원문 hash·게임/선수·결정 시점·초기 상태·수비 관점·회전 타순·프로필·질량을 검증한다.
- D는 같은 계약의 TypeScript 타입, 런타임 검사, 한국어 표시 모델을 받는다. invalid 입력은 오류이며 0%/성공 카드로 대신하지 않는다. unavailable 개발 사례는 새 실제 평가 결과가 아니다.
- 기존 API·세션 DB·PA 사건 결과·화면은 이 단계에서 바꾸지 않는다. 실제 화면 연결은 같은 게임의 **투수 교체 이벤트 직전 상태**를 식별하는 저장/API 경계가 먼저 필요하다. 첫 관측 투구 ID만으로 교체 후 타석에 붙이면 안 된다.

루트는 Astra 역할로 가치/시점/연결 계약과 핵심 diff를 검토한다(정확한 런타임 모델 ID 미노출). `c_result_contract`, `d_result_consumer`에 각각 `gpt-6-sol/medium`을 명시해 분리된 파일 범위를 위임했다. 전체 대화 상속·재귀 위임 없음. 토큰/비용은 미노출이며 추정하지 않는다.

## 실제 전달 결과와 검증

변환/소비 구현 기준 커밋 **`8979588`**(계약의 identity/시각 세부 규칙 포함). `results/C-D-INNING-001/`에 실제 변환 `bounded.json`, 개발 전용 `development_unavailable.json`, 출처/합성 여부 manifest, D가 실제 생성한 `presentation_examples.json`, 실행 hash/검사 기록 `execution_audit.json`을 저장했다. 같은 파일과 검사 로그는 SSD `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-D-INNING-001/`에 보존했다.

실제 D 소비 결과는 **52.10%–53.47%**, 미해결 확률 **1.36%**, 기본 프로필 **2명**, ‘현 투수 유지 + 당시 타순 유지’, ‘실제 교체 효과 미측정’이다. unavailable 예제는 숫자가 전부 null이고 ‘계산 결과 없음 · 평가 결과가 없습니다’로 변환된다. 원래 평가 결과와 anchor의 hash를 대조했고 기존 결과는 변경하지 않았다. 새 모델 호출·학습·자료 수집은 모두 0회다.

최종 **Python 관련 검사 80개 통과(0.91초)**: 새 변환/계약 49개와 기존 PA 사건/이닝 평가 검사를 함께 확인했다. **D의 실제 exported fixture 소비 검사 및 TypeScript noEmit 검사 통과.** 검증은 파일→C 검증/변환→JSON→D 검증/한국어 표시 모델까지이며 브라우저 화면 연결 테스트를 수행했다고 하지 않는다. 화면/서버는 이번에 시작하지 않았다.

핵심 거절 사례는 확률을 % 수치로 전달, 미해결 질량 불일치, 가짜 point/교체 가치, unavailable의 잔존 숫자, 다른 game/투수/초기 상태/타순, 교체 이후 근거, 상대 투수의 타격 자세 오용, 기본 프로필 불일치, 잘못된 날짜·수비 관점·평가 버전이다. 초/말은 기존 모형의 Top/Bot 규약을 보존하며 공식 경기 날짜와 UTC 이벤트 날짜를 강제로 같게 하지 않는다.

재현 명령(프로젝트 루트):

```sh
# 변환은 아직 없는 출력 폴더를 지정한다. 기존 파일은 덮어쓰지 않는다.
.venv-observer-standalone/bin/python scripts/export_inning_result.py --anchor '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-ROSTER-001/decision_anchor_777063.json' --output-dir /tmp/pitcheezy-inning-result-reproduction
PYTHONPATH=.:apps/observer/backend .venv-observer-standalone/bin/python -m pytest apps/observer/backend/tests/test_inning_result.py apps/observer/backend/tests/test_event_analysis.py tests/test_c_inning_eval.py -q
node apps/observer/web/tests/inning-result-contract.mjs results/C-D-INNING-001
apps/observer/web/node_modules/.bin/tsc --noEmit -p apps/observer/web/tsconfig.json
```

## D 시작 가능 범위와 다음 한 가지

**D는 이 계약·실제 JSON 예제·표시 함수를 사용한 별도 이닝 카드 연결 개발을 시작할 수 있다.** 새 C 결과 형식을 기다릴 필요는 없다. 다만 기존 세션의 PA 완료 이벤트에 자동으로 붙일 수 있는 결과는 아니다. 다음 한 가지는 **교체 이벤트 직전 상태를 식별하는 저장/API 연결을 구현하는 것**이며, 연결 검증 후 별도 카드에 전달한다. 실제 교체 후보 가용성 검증과 교체 우위 계산은 계속 미완료다. B002 미채택, 동결 서비스 유지, CV 인수 대기는 그대로다.

두 Sol 구현 작업은 완료했고 실행 중인 작업용 서버/계산 프로세스는 없다. 루트가 확률 단위·결정 전 정보 경계·Python/TypeScript 규약 일치·실제 예제 소비·원문 보존을 검토했다. API나 운영 화면 연결 완료를 뜻하지 않는다.
