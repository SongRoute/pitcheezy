# 교체 직전 결과의 저장/API 연결

2026-09-23. 시작 `a0ae5a4`, 계약 사전 커밋 `f821b55`. 이전 단계에서 검증한 실제 `inning-result-v1`을 독립 교체 직전 기록으로 저장하고 문맥 일치 조회를 구현한다. 새 추론·학습·CV·2026/최종 자료 열람 없이 같은 한 사례를 사용한다.

## 경계

기존 PA 세션은 교체 뒤 첫 투구에 등장한 타자/투수를 사용한다. 이 때문에 PA 세션의 pitch ID만 보고 조건부 keep 결과를 붙이지 않는다. 별도 `inning-decision-v1` 자원은 `historical_decision_review` 모드로, 검증된 과거 이벤트 직전 상태를 명시적으로 조회한다. 서버가 클라이언트의 실제 재생 cursor나 감독의 실제 가용 후보를 확인했다는 뜻은 아니다.

`GET /api/inning-decisions?game_id=777063`은 수치 없는 문맥 목록을 반환한다. `POST /api/inning-decisions/{id}/resolve`는 revision과 게임/이벤트/현 투수/초기 상태/phase를 대조한 뒤 결과를 반환한다. 잘못된 문맥은 409, 형식 오류는 400, 없는 ID는 404, 저장 훼손은 503이며 결과 수치를 포함하지 않는다. 상세 계약은 `docs/contracts/inning-decision-api-v1.md`다.

등록은 검증된 C 원문/anchor를 읽는 로컬 CLI에서만 한다. 일반 클라이언트가 임의의 계산 결과를 등록하는 API는 없다. 같은 이벤트의 동일 결과 재등록은 멱등, 다른 상태/결과는 충돌로 거절한다. 새 테이블의 UPDATE/DELETE는 거절되며 기존 session/recommendation/event row는 수정하지 않는다. 정정 revision은 이번 최소 연결 범위에 포함하지 않았다.

루트는 Astra 역할로 문맥/가치/저장 계약과 핵심 diff를 검토한다(정확한 런타임 모델 ID 미노출). `inning_decision_store`와 `inning_decision_api`에 `gpt-6-sol/medium`을 명시해 저장/CLI와 service/API를 분리 위임했다. 재귀 위임·전체 대화 상속 없음. 토큰/비용 미측정. 관련 코드·검사 외 범위와 기존 모델은 유지한다.

실제 실행·검증·재현·잔여 항목은 완료 후 아래에 기록한다.
