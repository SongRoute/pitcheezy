# 실제 관전 서버 리허설과 원격 확인 — 2026-09-23

교체 직전 UI 코드 `62945c7`·인수인계 `a5d3718`을 기준으로 기존 관전 앱의 실제 동결 모델, 기록 데이터, 저장 세션과 이닝 카드를 한 서버에 올렸다. 제품 모델/가치 계약은 바꾸지 않았다. CV는 제외하고 `unavailable:no_media`를 유지한다.

## 실행 공간과 보존

- 새 실행 공간: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-D-FULL-001`.
- 기존에 검증한 `ABCD-integration-v1`의 dataset/manifest와 추천 캐시를 복사하고 SQLite backup API로 DB를 복사했다. 이전 3개 세션, 6개 추천, 2개 사건 결과, 2개 메모 및 메타데이터를 그대로 보존했다. 이 공간에서 새 사용자/검사 세션을 만들 수 있다.
- 원 데이터 SHA256 `8ad2b369884265bb5197137499e002a32354930cce036253f92b91671dde479b`, 동결 모델 bundle manifest SHA256 `43ece920cb60c6c24ea9f1e720a7000b3b83038a3e24ac27d77ef17a2e8f0f1f`. 기존 2025 replay만 재사용했고 2026·최종 평가 자료는 추가 열람하지 않았다.
- 현재 모델 `observer-zone-v1`, 런타임 `standalone`. PA 가치 `defense-we-pa-v1`와 별도 이닝 결과 `inning-result-v1`/저장 `inning-decision-v1`을 유지한다. RE24와 혼합하지 않는다.
- 기존 `results/EXP-C-INNING-001/conditional_keep_777063.json`과 `C-ROSTER-001/decision_anchor_777063.json`을 명시적으로 등록했다. 이닝 결정 ID `inning-decision-2b1ead36958b4ee23cd8e4231a58c36a61a9ebbcd0072960891fe572abab6d4e`, revision 1. 실제 교체 효과는 null이며 현재 투수 유지에 대한 조건부 이닝 전망이다.
- 복사 전 목록/해시/테이블 보존 근거는 실행 공간의 `preparation.json`, 복사한 기존 세션의 실제 HTTP 복원·추천/사건 동일성·SQLite 무결성은 `persistence-verification.json`에 기록했다. 원 DB 파일 해시가 그대로임을 확인했다.

## 맥북에서 접속

사용자는 SSH와 Tailscale을 함께 사용한다. 맥미니의 Tailscale IPv4를 `tailscale ip -4`로 확인했다. **맥북의 로컬 터미널**에서 다음을 실행하고 창을 열어 둔다.

```sh
ssh -N -o ExitOnForwardFailure=yes -L 18766:127.0.0.1:8766 macmini
```

사용자가 평소 사용하는 SSH 별칭 `macmini`를 확인했다. 별칭이 없는 환경에서는 `song@100.108.252.111`을 사용할 수 있다. 맥북 브라우저 주소는 `http://localhost:18766/`, 이닝 화면은 `http://localhost:18766/inning-decisions`이다. 터널을 연 터미널에서 Ctrl-C를 누르면 맥북 연결만 닫힌다. 서버는 맥미니 `127.0.0.1:8766`에만 바인딩했다. 사용자가 맥북에서 접속 성공을 확인했다.

관전 화면에서는 경기/타석을 선택해 ‘타석 관전하기’ → ‘다음 실제 공 확인’으로 진행한다. 2025-08-17의 경기 776703/타석 3은 삼진, 경기 776710/타석 10은 홈런 연결 검사용 사례다. ‘교체 직전 기록’에서는 2025-07-21 경기 777063의 시점을 선택하고 ‘이닝 전망 보기’를 누른다. 52.10%–53.47%는 초기 수비팀 WE 범위이며 신뢰구간이나 교체 개선치가 아니다. 미해결 확률 1.36%와 실제 교체 효과 미측정 표시를 함께 확인한다.

## 재현과 프로세스

현재 서버 PID와 로그는 실행 공간의 `server.pid`, `server.log`; 영상 없는 상태 처리 워커 로그는 `worker.log`다. 사용자가 직접 확인할 수 있도록 검증 후에도 서버를 유지한다. 맥미니/SSD가 꺼지거나 잠들면 접속할 수 없으며 자동 로그인 시작 서비스는 설정하지 않았다.

서버가 종료된 경우 저장소 루트에서 재실행한다. 이미 떠 있으면 중복 실행하지 않는다.

```sh
PITCHEEZY_OBSERVER_RUN='/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-D-FULL-001' \
PITCHEEZY_OBSERVER_PORT=8766 sh apps/observer/run_standalone.sh
```

정지하려면 PID 파일의 프로세스가 위 실행 공간의 관전 서버인지 확인한 뒤 `kill -TERM <PID>`를 사용한다. 앱 종료 시 자식 워커도 종료한다.

브라우저 리허설은 기존 삼진/홈런·이닝 UI 검사를 재사용하고 실제 타석/페이지 이동 검사를 추가한다. 새 세션을 생성하므로 테스트/확인용 실행 공간에서 실행한다.

```sh
PLAYWRIGHT_BROWSERS_PATH='/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/observer-mvp-v1/browser-cache' \
OBSERVER_SMOKE_URL='http://127.0.0.1:8766/' \
OBSERVER_SMOKE_OUTPUT='/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-D-FULL-001/browser' \
node apps/observer/web/tests/full-rehearsal-smoke.mjs
```

검증 코드 커밋은 `a8cd5d5`다. B/C/D의 기존 구현은 모두 이 서버에서 확인 가능하지만 새 추천 후보의 채택, 실제 불펜 가용성/교체 효과, 실제 CV 품질을 완료했다고 해석하지 않는다.


## 완료 기록

- 실 서버 health: model/dataset ready, standalone, worker running, startup error 없음. 실제 CV는 계속 unavailable.
- 삼진(1440px)/홈런(390px)의 공개 전 정보 숨김, 매 공 저장 추천 동일성, 사건/수동 메모 분리, 새로고침 복원 통과. 전체 차이 +1.202167/−22.586870 %p는 기존 결과와 같다.
- 인접 타석의 새 세션/이어보기, 이닝 페이지 왕복 후 기존 타석 유지, 실제 이닝 결정의 명시적 선택/범위/미해결 표시, PC/모바일 너비 검사 통과. 오류·빈 목록·불일치·지연·offline은 브라우저 모의 응답으로 연결 동작만 검사했다.
- 기존 catalog 검사에서 확률 상세 패널 2개를 단일 locator로 선택하는 실패를 확인했다. 모델 내부 비교 패널을 명시하도록 고친 뒤 검색/날짜 필터/미래 결과 검색 제외/인접 타석/처음부터/복원/390px 검사 모두 통과했다. 새 wrapper의 이닝 URL 끝 슬래시 처리도 수정했다. 최초 실패 로그는 SSD `browser/diagnostics`에 보존했다.
- 최종 브라우저 console/page 오류와 가로 넘침 없음. 제품/모델 코드 변경은 없어 기존 backend 검사/빌드를 반복하지 않았다. 실제 제공된 기존 production build에서 검증했다.
- 기계 판독 기록은 저장소 `results/C-D-FULL-001/{verification,preparation,persistence-verification,browser-report,catalog-report}.json`. 스크린샷과 자세한 로그는 SSD 실행 공간의 `browser/` 및 `catalog-fixed.log`에 있다.
- 재현 명령의 통합 wrapper 외 catalog 검사는 같은 환경변수에서 `node apps/observer/web/tests/observer-browse-smoke.mjs`로 실행한다. 출력 폴더를 별도로 지정하면 기존 증거를 보존할 수 있다.
- 주 에이전트는 Astra 역할로 계약/보존·실행 공간·접속·핵심 diff/증거를 검토하고 기존 catalog 테스트를 수정했다. 브라우저 리허설은 `gpt-6-sol/medium`에 필요한 파일/범위만 전달해 위임했다. 재귀 위임 없음. 주 에이전트 정확한 모델 ID와 토큰/비용은 미노출·미측정이다.
- 완료 시 서버 PID **94628**, 워커 PID **94642**를 사용자 확인용으로 유지했다. 다른 서비스·학습 작업은 변경하지 않았다. 사용자 맥북에서 `ssh macmini` 별칭의 SSH 터널 접속을 확인했다.

**다음 한 가지:** 사용자가 직접 확인한 관전 흐름에서 혼동되거나 불편한 점을 모아 D의 작은 사용성 수정 범위를 정한다. B 후보는 미채택 상태이며 다음 학습을 자동 확대하지 않는다. 실제 불펜 가용성과 이닝 지원 범위 검증은 별도 연구로 남고, CV는 인수 일정까지 제외한다.
