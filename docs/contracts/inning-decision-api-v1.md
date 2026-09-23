# 교체 직전 이닝 결과 저장/API v1

`inning-result-v1`의 실제 결과를 교체 이벤트 직전의 독립적인 기록 조회 자원으로 저장한다. 기존 PA 재생 세션은 교체 후 첫 투구의 타자/투수를 사용하므로 이 자원과 자동 연결하지 않는다. 이름은 `inning-decision-v1`, mode는 `historical_decision_review`다. 이는 서버에 저장된 과거 결정 문맥의 조회이며 실제 재생 cursor나 실제 불펜 가용성을 인증하지 않는다.

## 저장과 식별

등록은 로컬 CLI/저장소 메서드만 제공한다. 원래 C 평가 파일과 anchor 파일을 `convert_inning_result`로 다시 검증하여 등록한다. 개발용 unavailable fixture 또는 클라이언트가 만든 결과 JSON을 등록하는 HTTP endpoint는 없다. 새 추론/학습/외부 자료 조회를 하지 않는다.

context 형식:

```text
{
  phase: "before_pitching_change",
  linkage: <inning-result-v1의 linkage 전체>,
  initial_state: <evaluation_identity.initial_state_and_count 전체>
}
```

linkage UTC 시각은 UTC ISO의 microseconds + Z로 정규화한다. 숫자는 bool을 받지 않는 안전 정수이며 필수/추가 필드를 엄격히 검사한다. initial_state 날짜/수비 관점/초기 상태 규약은 기존 결과 계약을 따른다. 잘못된 UTC·날짜·투구 ID는 조회 전에 거절한다.

decision_id는 `inning-decision-` + 다음 객체의 canonical JSON(SHA256; sort_keys=True, separators=(',', ':'), ensure_ascii=True)이다: `{game_pk, anchor_kind, anchor_time_utc, anchor_action_index, first_observed_pitch_id}`. 시각은 위의 정규화 값을 사용한다. 상태/keep 투수/모형을 ID에 넣어 같은 이벤트의 충돌을 새 이벤트로 숨기지 않는다. revision은 이번 v1에서 1로 고정한다.

SQLite `inning_decisions` 별도 테이블에 ID·game_id·revision·canonical context·canonical result·result SHA·등록 시각을 저장한다. 기존 session/recommendation/event row는 수정하지 않는다. 같은 ID/같은 context/result의 재등록은 멱등이고 생성 시각도 유지한다. 같은 ID에 다른 결과/문맥을 등록하면 충돌로 거절한다. UPDATE/DELETE는 trigger로 거절한다. 정정 revision 구현은 이번 범위에 없다.

조회 시에도 JSON 계약·context/result 일치·ID·game_id·revision·SHA를 검증한다. 훼손된 row는 결과를 반환하지 않고 503을 낸다. 등록 실패 시 부분 row가 남지 않으며 동시 등록은 하나의 동일 결과만 남긴다.

## API

- `GET /api/inning-decisions?game_id=<positive integer>` → `{schema_version:"inning-decision-v1",mode:"historical_decision_review",decisions:[{decision_id,revision:1,context}]}`. 결과 수치·미래 타순·이후 대타를 목록에 넣지 않는다. 자료가 없는 game은 빈 목록.
- `POST /api/inning-decisions/{decision_id}/resolve` body `{revision:1,context:<expected context>}` → `{schema_version:"inning-decision-v1",mode:"historical_decision_review",decision_id,revision:1,context,result:<inning-result-v1>}`. body는 추가 필드 금지. D가 선택한 문맥과 저장 문맥을 모두 일치시켜야 수치를 반환한다. 동일 시각의 Z/+00:00 표기는 정규화 후 같다.
- missing/malformed query/body/ID/context는 400, 없는 유효 ID는 404, 유효하지만 다른 revision·게임·시점·선수·초기 상태·phase는 409, 저장 row 훼손/서비스 미준비는 503. 오류 응답은 기존 `{detail:string}` 형식이며 result/수치를 포함하지 않는다. 다른 phase는 형태가 맞는 문자열일 때 문맥 불일치 409로 처리한다.

Repository 인터페이스는 `InningDecisionRepository(store)`의 `import_result(source_path, anchor_path)`, `list_decisions(game_id)`, `resolve(decision_id, revision, context)`다. `DecisionNotFound`, `DecisionConflict`, `DecisionCorrupt`는 각각 404/409/503에 매핑한다. malformed 입력은 ValueError→400. `ObserverService`에서 소유하며 main의 기존 서비스 준비 상태/오류 처리 경로를 사용한다. inference와 session cursor 변경은 호출하지 않는다.

## 검증과 D 연결

실제 기존 C 파일→새 격리 SQLite→앱 재생성→목록→일치 문맥 resolve→기존 D parser까지 확인한다. 잘못된 게임/날짜/이벤트/선수/초기 상태/phase/revision, 멱등/충돌·손상·동시성, 등록 실패 rollback과 기존 세션의 미래 정보 비공개를 검증한다. 기본 운영 DB를 자동 변경하지 않는다. 검사 DB/응답/로그는 새 `C-D-API-001` 경로에 보존한다.

D의 다음 화면은 이 목록에서 **독립된 교체 직전 기록**을 명시적으로 선택하고 resolve 결과를 기존 `buildInningPresentation`에 넘긴다. 일반 PA 추천·기여도와 합산하지 않는다. 실제 교체 효과는 계속 null이며 CV가 없어도 이 기록 조회는 작동한다.
