# 라이브 UI 연결 (2026-10-13)

현재 제공 가능한 정보로 연결한 개발 화면이다. 운영 교체나 새 실자료 수집을 수행하지 않는다.

- 모바일: 상황 → 추천 구종·대략 위치 → 이번 타석. 최근 3구를 실제 좌표에 표시하며 전체 보기로 확대한다. 동일한 좌표의 공은 번호만 묶고 좌표를 옮기지 않는다. 직전 타석 기록은 접을 수 있다.
- 760–1199px: 추천·경기 2열, 선수는 상세. 1200px 이상: 선수·추천·경기 3열; 상세가 열리면 라이브 2열 옆에 420px 비모달 패널. 모바일 상세만 focus trap과 배경 잠금을 사용한다.
- 헤더 LIVE 점은 연결이 끊기면 숨기되 폭을 유지한다. 320px에서 연결 상태만 바뀌어도 날짜/구장 줄바꿈과 전체 화면 위치가 움직이던 문제를 방지한다.
- 과거 투구 상세는 stable `key`로 선택한다. 라이브 polling과 메인 추천은 계속 갱신되며 정정은 같은 선택에 반영한다. delta의 key/index 충돌은 full refetch한다. 사라진 선택은 상세를 닫는다.
- 포수 시점: source `plate_x`와 drawing 방향 일치. 좌우 문구를 반전하지 않으며 우타자는 왼쪽. actual/미트 데이터 자체는 바꾸지 않는다. 미트는 x-only 선이다.
- 기록 패널: `current.context.lineups/boxscore`, 선수: `pregame.players`, 투구: `actual.tracking`. 자료가 없으면 —/정보 없음. 시즌별 통계 전환·RE24 책임분해·225행동 위치가치·LI/WPA/승률시계열은 생성하지 않는다.
- 구종 비율은 수신·확인된 구종만의 관측 비율이며 경기 전체 사용률로 부르지 않는다. 추천 막대는 정책 선택 비중이다.
- 승률: top-level `win_expectancy`의 available/home 확률만 표시. 현재 추천 상태·feed·공개파일 검증 게이트를 모두 통과해야 하며 지연·연결실패에는 숨긴다. 카운트 미반영 모델임을 상세에 쓴다.
- 경기 목록은 기존 한국날짜 loader와 지금/예정/기록을 재사용한다. 실제 제공된 시리즈 설명·차전·선발만 표시하고 WS 매치업이나 시리즈전적을 추측하지 않는다.

검증 명령:

```sh
npm run build
npm run build:public
node tests/ws-responsive-smoke.mjs /tmp/pitcheezy-ws-ui-1013
PZ_STATIC_CORE_ONLY=1 node tests/static-publish-smoke.mjs
node tests/zone-geometry-check.mjs
node tests/display-state-smoke.mjs
```

`ws-responsive-smoke`는 완전 합성 payload로 390×844, 320×568, 1024×768, 1440×1000에서 폭/모달성/라이브갱신/키정정/미확인 masking/WE 연결실패/10종 display state를 검사한다. 기존 geometry/display/static 검사에서 사용하는 `service-849843.json`은 기존 경기 기반 fixture이며 합성 자료라고 부르지 않는다. 브라우저 요청은 로컬 route mock으로 한정한다.

## 기존 경기 fixture 기반 회귀 이관

`display-state-smoke.mjs`와 `zone-geometry-check.mjs`는 저장된 `service-849843.json`을 사용한다. 상태·영수증 등은 검사별로 구성하지만 전체를 완전 합성 검사라고 부르지 않는다. 모든 브라우저 요청을 route mock으로 처리하고 `http://pz.test` 이외 origin은 차단한다. 외부 수집·실서비스 호출·모델 실행·운영 변경은 없다. 아래 이관에서는 테스트와 이 문서만 수정했다.

구형 고정 높이/1줄 이름/2열 요약/투수 시점/과거 선택 시 메인 화면 이동 기대를 승인된 모바일 스크롤·이름 2줄·세로 요약·포수 시점·상세만 변경으로 옮겼다. 390×844 핵심영역 노출, 가로 overflow, 10종 상태, recommendation masking, 원장 정정 full refetch, 추천 교체 알림, 연결 회복, polling, 실제 좌표 오차 0.03ft 이내, 존 밖 공의 edge marker 검증은 유지했다. PC 상세의 `aria-modal`은 속성 부재, 모바일은 `true`이며 PC에서도 메인 LIVE가 갱신되는지 검사한다.

전체 회귀의 첫 빌드는 `index-qR24OstV.js`였다. 320px 헤더 결함을 수정한 최종 빌드 `index-z3BXXaqs.js`에서 section 1의 320×568 전체 조합과 section 3 연결 회복을 재검사했다. `PZ_TEST_START`/`PZ_TEST_END`는 변경 구간 재실행용이며 미지정 시 1–14 전체를 실행한다. `PZ_TEST_VIEWPORT`는 section 1의 특정 viewport만 선택하며 미지정 시 다섯 크기를 모두 검사한다. 최종 수용은 각 구간의 마지막 실행을 합친 결과이며 단일 전체 성공 실행으로 표현하지 않는다.

재현 명령(작업 디렉터리 `apps/observer/web`):

```sh
node tests/zone-geometry-check.mjs
PZ_TEST_START=1 PZ_TEST_END=4 node tests/display-state-smoke.mjs
PZ_TEST_START=5 PZ_TEST_END=14 node tests/display-state-smoke.mjs
# 실패했던 구 UI 기대를 수정한 후 해당 구간만 재실행
PZ_TEST_START=7 PZ_TEST_END=7 node tests/display-state-smoke.mjs
PZ_TEST_START=13 PZ_TEST_END=14 node tests/display-state-smoke.mjs
PZ_TEST_START=13 PZ_TEST_END=13 node tests/display-state-smoke.mjs
# 프런트의 320px 헤더 수정 및 양 빌드 후
PZ_TEST_START=1 PZ_TEST_END=1 PZ_TEST_VIEWPORT=320x568 node tests/display-state-smoke.mjs
PZ_TEST_START=3 PZ_TEST_END=3 node tests/display-state-smoke.mjs
```

초기 실행에는 구 UI 높이·이름 1줄 기대 실패가 있어 해당 실행을 중단했다. 이후 5–14 실행의 6실패는 summary 2열 기대 2건, 숨겨진 팀칸 `innerText` 공백 기대 3건, PC `aria-modal="false"` 기대 1건이었다. 이를 승인된 동작에 맞췄다. 13–14 재실행은 팀칸 공백 수정 전에 시작되어 13번 3실패가 남았으므로 13번을 다시 실행했다. 단독 수정 명령의 `python` 실행 파일 부재도 있었으며 `python3`로 재실행했다. 실패·중단 로그를 삭제하지 않았고 최종 수용 구간과 분리 보존했다. 추가로 320×568 연결 끊김 전환에서 헤더 높이가 11px 줄어 아래 영역이 이동하는 동일한 제품 결함이 드러난 3조합을 찾아 프런트 작업자에게 전달했다. 원인은 연결 끊김 때 LIVE 장식점 DOM이 사라져 버튼 폭이 줄고 중앙 날짜/구장 문구가 3줄에서 2줄로 바뀌는 것이었다. 프런트 작업자가 장식점 공간을 유지하고 비활성 때 숨기는 방식으로 고쳤으며, 테스트는 DOM 부재 대신 실제 보이는 점이 없음을 검사한다.

로그: `/Users/song/Projects/pitcheezy-pm-jobs/20261013-ws-ui-connect-001/`. `display-state-passed-*.log`는 원본의 통과 구간 발췌이며 첫 줄에 원본 파일과 행 범위를 표시한다. 원본 전체도 함께 보존한다. 브라우저 기반 회귀이므로 실제 기기·실시간 공급자/운영 연결 검증을 대신하지 않는다.

최종 수용 결과(중복 재실행 제외):

| 범위 | 통과 assertion | 남은 실패 |
| --- | ---: | ---: |
| display 1: 390×844·390×664·375×553·360×640 | 1239 | 0 |
| display 1: 320×568, 헤더 수정 후 | 168 | 0 |
| display 2: 추천 교체 guard | 15 | 0 |
| display 3: 연결·timeout, 헤더 수정 후 | 16 | 0 |
| display 4: 초기 실패 재시도 | 10 | 0 |
| display 5–6: receipt·정정·경기 전/후 | 27 | 0 |
| display 7: fail-closed·숨긴 탭 | 123 | 0 |
| display 8–12: masking·긴 문구·경기 시작·polling | 124 | 0 |
| display 13: 공수 교대 | 51 | 0 |
| display 14: 과거 추천·비모달 상세·실시간 유지 | 81 | 0 |
| **display 합계** | **1854** | **0** |
| geometry: 메인·전체 상세, 좌표·존 밖·중첩 | 342 | 0 |

각 통과 구간의 원본/발췌 로그 매핑은 `legacy-final-summary.txt`에 보존했다. `node --check tests/display-state-smoke.mjs` 및 변경 파일 `git diff --check`도 통과했다. Root는 최종 소스에 `graphify update .`를 AST-only로 실행했다(10855 nodes, 27837 edges).
