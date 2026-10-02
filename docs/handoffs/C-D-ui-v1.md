# 교체 직전 기록 선택과 이닝 카드 UI

2026-09-23. 시작 `f50c54b`, 구현 커밋 **`62945c7`**. `/inning-decisions`에 독립 기록 선택과 조건부 이닝 전망 카드를 구현하고 기존 관전 화면 상단에서 연결했다. 기존 PA 카드에 자동 삽입하지 않는다. CV·새 학습·새 추론·외부 자료 수집은 제외했다.

## 구현과 의미

사용자는 등록된 경기와 교체 직전 시점을 고르고 **이닝 전망 보기**를 눌러 결과를 조회한다. 경기 목록은 새 `GET /api/inning-decision-games`로 실제 등록 자료에서만 만든다. 없는 팀/선수 이름은 만들지 않으며 현재는 날짜·경기/투수 식별자로 구분한다. 같은 game의 날짜 모순이나 저장 훼손은 503으로 거절한다.

API 클라이언트 `inningDecisionApi.ts`는 응답의 버전/ID/revision/context가 선택한 것과 맞는지 확인하고 result의 linkage/초기 상태를 한 번 더 대조한다. 그 뒤 기존 `buildInningPresentation`으로 **52.10%–53.47%**, 미해결 **1.36%**, 기본 프로필 **2명**, ‘현 투수 유지 + 당시 타순 유지’, ‘실제 교체 효과 미측정’을 표시한다. point·기여도·교체 효과를 새로 계산하지 않는다.

선택 변경과 재시도는 이전 결과를 지우고 AbortController/요청 세대 번호로 늦은 응답을 무시한다. 화면에 빈 목록, 연결 실패, 문맥 불일치, 잘못된 응답, 계산 불가 상태와 재시도를 제공한다. 관전 세션의 localStorage/조회/advance는 건드리지 않는다.

구현 파일: `web/src/InningDecisions.tsx`, `inning-decisions.css`, `inningDecisionApi.ts`, 라우팅 `main.tsx`, 홈 연결 `App.tsx`. backend는 등록 경기 목록을 위해 기존 repository/service/main만 최소 확장했다.

## 실제 검증

새 SSD `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-D-UI-001/observer.sqlite3`에 기존 실제 C 평가/anchor 한 건을 파일 검증 후 등록했다. 기본 DB는 변경하지 않았다. `scripts/inning_ui_review_server.py`로 실제 결정 DB를 쓰는 loopback HTTP 서버를 열었다. 이 검증 서버의 PA 목록은 비어 있고 추천 모델/worker는 로드하지 않는다. 따라서 새 이닝 화면의 실제 HTTP 연결을 검증한 것이며, 기본 관전 서비스의 전체 모델 시작을 재검증한 것은 아니다.

- 웹 production build 통과.
- 관련 backend 검사 **38개 통과(0.44초)**: 신규 경기 목록 4개와 기존 결정 저장/API/관전 서비스 회귀 검사.
- API 클라이언트 검사 통과: 실제 fixture와 문맥 7종 불일치, 다른 경기 목록, 중복 경기, 409/연결 실패.
- Playwright **1440px PC/390px 모바일** 실제 API→선택→resolve→카드 표시 통과. 선택/버튼 전 결과 요청 없음, PA 세션 저장값 유지, 가로 넘침 없음, 페이지 예외 없음.
- 오류 사례는 명시적 브라우저 응답 주입으로 검사했다: 빈 목록, 409 후 정상 재시도, 잘못된 200 응답, 개발용 unavailable의 숫자 숨김, 선택 후 늦은 결과 숨김, 연결 실패/복구. 합성 예외를 새로운 실제 평가로 사용하지 않는다.
- PC/모바일 스크린샷을 직접 확인했다. 초기 및 최종 브라우저 로그를 별도로 보존했다.

앱 내 Browser 스킬을 읽었으나 필요한 브라우저 제어 도구가 노출되지 않아 프로젝트에 설치된 Playwright를 사용했다. 루트는 Astra 역할로 계약/API 응답 경계·diff·실제 연결·화면을 검토했다(정확한 런타임 모델 ID 미노출). `inning_ui_catalog`, `inning_ui_page`는 각각 `gpt-6-sol/medium`, 분리 파일·재귀 위임 없음. 토큰/비용은 미노출이며 추정하지 않는다.

## 재현과 남은 것

프로젝트 루트에서 아래 명령을 사용한다. 검토 DB에는 이미 실제 한 건이 등록되어 있다. 별도 새 DB를 쓸 때는 이전 API 인수인계의 import 명령으로 같은 평가/anchor를 등록한다.

```sh
npm run build --prefix apps/observer/web
PITCHEEZY_INNING_REVIEW_DB='/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-D-UI-001/observer.sqlite3' .venv-observer-standalone/bin/python -m uvicorn --app-dir scripts inning_ui_review_server:create_review_app --factory --host 127.0.0.1 --port 8771
# 브라우저: http://127.0.0.1:8771/inning-decisions
# 다른 터미널에서 검사 출력은 새로운 경로로 지정한다.
OBSERVER_SMOKE_OUTPUT=/tmp/pitcheezy-inning-ui-reproduction node apps/observer/web/tests/inning-ui-smoke.mjs
node apps/observer/web/tests/inning-decision-client.mjs
PYTHONPATH=.:apps/observer/backend:experiments/pitchmdp .venv-observer-standalone/bin/python -m pytest apps/observer/backend/tests/test_inning_decision_games.py apps/observer/backend/tests/test_inning_decision_store.py apps/observer/backend/tests/test_inning_decision_api.py apps/observer/backend/tests/test_observer_service.py -q
```

기본 관전 서버에서도 같은 route와 화면을 제공한다. 단, DB에 등록 자료가 없으면 빈 목록이 정상이다. 사용자 세션이 있는 기본 DB로 검증 자료를 자동 복사하지 않았다. 다음 한 가지는 **기존 관전 서버의 모델/데이터/저장 세션과 함께 전체 실행 흐름을 리허설하는 것**이다. 실제 대체 선수 가용성·교체 효과, CV 인수와 최종 평가는 미완료다. 이번 연결은 그 과학적 검증을 대신하지 않는다.

구현 커밋/산출물 hash·브라우저 보고서는 `results/C-D-UI-001/verification.json` 및 SSD 로그에 기록했다. PC/모바일/빈 상태 스크린샷은 SSD `C-D-UI-001/browser-final/`에 보존했다. 검증용 서버 PID 93726은 SIGTERM 후 정상 종료 로그와 프로세스 부재를 확인했다. 두 Sol 작업도 완료했으며 남은 실행 프로세스는 없다.
