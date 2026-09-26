# 교체 직전 결과의 저장/API 연결

후속 독립 기록 선택/이닝 카드 UI는 [C→D UI 인수인계](C-D-ui-v1.md)에서 완료했다. 아래는 저장/API 단계의 기록이다.

2026-09-23. 시작 `a0ae5a4`, 계약 사전 커밋 `f821b55`. 이전 단계에서 검증한 실제 `inning-result-v1`을 독립 교체 직전 기록으로 저장하고 문맥 일치 조회를 구현한다. 새 추론·학습·CV·2026/최종 자료 열람 없이 같은 한 사례를 사용한다.

## 경계

기존 PA 세션은 교체 뒤 첫 투구에 등장한 타자/투수를 사용한다. 이 때문에 PA 세션의 pitch ID만 보고 조건부 keep 결과를 붙이지 않는다. 별도 `inning-decision-v1` 자원은 `historical_decision_review` 모드로, 검증된 과거 이벤트 직전 상태를 명시적으로 조회한다. 서버가 클라이언트의 실제 재생 cursor나 감독의 실제 가용 후보를 확인했다는 뜻은 아니다.

`GET /api/inning-decisions?game_id=777063`은 수치 없는 문맥 목록을 반환한다. `POST /api/inning-decisions/{id}/resolve`는 revision과 게임/이벤트/현 투수/초기 상태/phase를 대조한 뒤 결과를 반환한다. 잘못된 문맥은 409, 형식 오류는 400, 없는 ID는 404, 저장 훼손은 503이며 결과 수치를 포함하지 않는다. 상세 계약은 `docs/contracts/inning-decision-api-v1.md`다.

등록은 검증된 C 원문/anchor를 읽는 로컬 CLI에서만 한다. 일반 클라이언트가 임의의 계산 결과를 등록하는 API는 없다. 같은 이벤트의 동일 결과 재등록은 멱등, 다른 상태/결과는 충돌로 거절한다. 새 테이블의 UPDATE/DELETE는 거절되며 기존 session/recommendation/event row는 수정하지 않는다. 정정 revision은 이번 최소 연결 범위에 포함하지 않았다.

루트는 Astra 역할로 문맥/가치/저장 계약과 핵심 diff를 검토한다(정확한 런타임 모델 ID 미노출). `inning_decision_store`와 `inning_decision_api`에 `gpt-6-sol/medium`을 명시해 저장/CLI와 service/API를 분리 위임했다. 재귀 위임·전체 대화 상속 없음. 토큰/비용 미측정. 관련 코드·검사 외 범위와 기존 모델은 유지한다.

## 실제 저장/API 검증

구현 기준 커밋 **`9c6d8f9`**. `scripts/import_inning_decision.py`로 기존 실제 C 결과와 anchor를 새 SSD `C-D-API-001/observer.sqlite3`에 등록했다. 기본 Observer DB는 변경하지 않았다. 등록 ID는 `inning-decision-2b1ead36958b4ee23cd8e4231a58c36a61a9ebbcd0072960891fe572abab6d4e`, revision 1이다. 문맥의 시각은 `2025-07-22T00:20:56.827000Z`, 공식 경기 날짜는 `2025-07-21`이다. 원래 result의 시각 문자열·확률 등은 그대로 보존한다.

실제 저장 row→FastAPI ASGI 목록→문맥 일치 resolve→기존 D parser/표시 함수를 통과했다. 앱/서비스를 다시 생성해 같은 DB를 열어도 같은 응답이었다. 게임·공식 날짜·이벤트 시각·현 투수·아웃·phase·revision을 바꾼 **7종 유효 형식의 불일치 요청을 409로 거절**했고, 오류 응답에는 수치가 없었다. 목록에도 결과 수치가 없다. D의 결과 범위는 그대로 **52.10%–53.47%**, 미해결 **1.36%**, 실제 교체 효과 미측정이다.

**관련 Python 검사 105개 통과(1.22초)**, D 계약 검사와 실제 API result 소비 검사 통과. 새 저장/HTTP 검사 외 기존 PA 미래 정보 비공개·추천 저장·사건 분석 검사도 포함했다. 첫 묶음에서는 테스트 실행 경로에 pitchmdp가 없어 기존 검사 1개가 import 오류를 냈고, PYTHONPATH에 기존 모듈 경로를 명시한 뒤 통과했다. 제품 코드 오류를 숨기거나 동결 소스를 수정하지 않았다. 첫 로그와 최종 로그를 모두 보존했다.

실제 DB에는 inning_decisions 1개, sessions/recommendations/event_results/jobs는 각각 0개다. API 검증은 모델/PA 데이터셋을 주입하지 않는 결정 전용 ASGI TestClient로 수행했다. 네트워크 포트의 HTTP 서버, 기본 서비스 시작 시 모델 로딩, 브라우저 화면까지 검증했다고 하지 않는다. 정상 앱은 기존 ObserverService에서 이 저장소와 두 route를 사용할 수 있으며, 데이터 등록은 CLI에서 지정한 DB에만 적용된다.

## 산출물과 재현

저장소 `apps/observer/backend/observer_app/inning_decision_store.py`, 서비스/route `service.py`·`main.py`, 등록 `scripts/import_inning_decision.py`, 실제 API 검증 `scripts/verify_inning_decision_api.py`. `results/C-D-API-001/`에 catalog/resolved/mismatches/d_presentation/audit/verification JSON을 커밋한다. 동일 사본·DB·import/API/Python/D 검사 로그는 SSD `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-D-API-001/`에 있다. 원래 평가/anchor hash 일치, DB와 코드/산출물/로그 hash, 역할과 미측정 사용량을 verification에 기록했다. 새 추론·학습·자료 조회는 0회다.

프로젝트 루트 재현(새 DB/출력 경로 사용):

```sh
.venv-observer-standalone/bin/python scripts/import_inning_decision.py --database /tmp/pitcheezy-decision-reproduction/observer.sqlite3 --source results/EXP-C-INNING-001/conditional_keep_777063.json --anchor '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-ROSTER-001/decision_anchor_777063.json'
.venv-observer-standalone/bin/python scripts/verify_inning_decision_api.py --database /tmp/pitcheezy-decision-reproduction/observer.sqlite3 --output-dir /tmp/pitcheezy-decision-reproduction/api
PYTHONPATH=.:apps/observer/backend:experiments/pitchmdp .venv-observer-standalone/bin/python -m pytest apps/observer/backend/tests/test_inning_decision_store.py apps/observer/backend/tests/test_inning_decision_api.py apps/observer/backend/tests/test_inning_result.py apps/observer/backend/tests/test_observer_service.py apps/observer/backend/tests/test_event_analysis.py -q
node apps/observer/web/tests/inning-result-contract.mjs results/C-D-INNING-001 results/C-D-API-001/resolved.json
```

등록은 같은 원문에 멱등이다. 다른 파일을 기존 이벤트에 덮어쓰지 않는다. API 검증 스크립트의 출력 폴더는 아직 없어야 한다. 조회 endpoint에는 write/import 경로가 없고 POST resolve도 읽기 전용이다. 앱 시작 시 별도 테이블/trigger만 준비하며 검증 자료를 기본 DB로 자동 복사하지 않는다.

## D 시작 가능 범위와 다음 한 가지

**D는 별도 ‘교체 직전 기록’ 선택 화면과 이닝 카드의 실제 API 연결을 시작할 수 있다.** 결과 형식과 저장/API가 준비됐다. 다음 한 가지는 **독립 기록을 명시적으로 선택해 resolve 후 기존 표시 함수로 렌더링하는 UI 연결**이다. 기본 DB에 해당 기록이 없을 때 빈 상태를 보여줘야 한다. 기존 PA 완료 카드에 자동 삽입하지 않는다. 실제 불펜 가용성/교체 우위는 여전히 미확인이고 CV는 인수 일정까지 제외한다.

두 Sol 작업은 완료했다. 작업용 실행 서버·학습 프로세스는 남기지 않았다. 루트는 이벤트 ID와 결과의 분리, 정규화 문맥 일치, immutable row/손상 거절, 실제 영속성/API→D 경계를 검토했다.
