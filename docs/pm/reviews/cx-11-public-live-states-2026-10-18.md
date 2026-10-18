# CX-11 — 합성 live 추천 표시 상태

상태: 실행 전 독립 경계 PASS/root 수용 → 합성 suite1회 PASS → 독립 결과 검토 PASS/root 최종 수용. 작성자 `coordination_docs`, 검토자 `coordination_review`.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249075784 . base `8f7c257f2971baa63f04e7718e8458505d844cde`.

## 대상·고정 기대

기존 CX-05/CX-08 실행기와 pin/경계 코드를 재사용하고 `--live-states`만 추가했다. 기존6인자/default3사례와 `--host-errors`5사례는 보존한다. 제품 코드 수정은 없으며, 작성자와 실행기는 기존 fixture·모델·실경기·2026자료를 읽지 않았다. 검토자 탐색 중 경계 일탈은 아래에 별도 기록한다. 새 라이브러리/빌드는 없다.

대상은 v4.3.1 준비 후보 source `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`, manifest SHA `7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93`, 동일 public 자산4개다. 최종 출시/운영 UI가 아니다. 다음 근거는 모두 이 commit의 `git show`로 읽었다.

| 고정 소스 | 계약과 검사에 사용한 의미 |
|---|---|
| `apps/observer/web/src/service/api.ts:215–231,275–310,387–405` | candidate/ready 정규화; live source의 미확인·display 누락/미지원은 shown=false와 후보 제거 |
| `apps/observer/web/src/service/GameScreen.tsx:420–455,500–513,590–607` | healthy 추천 노출; health feedfalse는 disconnected/held·마지막 추천 유지·LIVE off; goLatest 버튼은 강제 load(false,true) |
| `apps/observer/web/src/service/publish.ts:96–115,129–138` | host Date/monotonic clock과 health check 유효성; file_sequence≤sequence_checked |
| `apps/observer/publish_contract/v1/schema/{game,health}.schema.json` | current/next candidate 및 health game_checks 필드 형태. 전체 JSON Schema 적합성 검사는 하지 않음 |

모든 JSON은 기존 순수 합성5JSON을 복제해 메모리에서 생성한다. game990001/2030-01-01, 합성 타자·투수, pre_pitch 상황, ready 후보1개뿐이며 관측 투구는 없다. 후보 probability=1은 합성 표시값이지 모델 출력/정확도/목표치가 아니다. unconfirmed에도 recommendation_shown=true와 후보1개를 넣어 state gate를 검사하고, display 누락·미지원 문자열도 후보1개를 유지한다.

| 별도 context 사례(390×900) | 실행 전 고정 기대 | 결과 |
|---|---|---|
| transition: 정상→feedfalse→정상 | 같은 context/game; 추천1개 실제 visible → disconnected.held/마지막 추천/연결 끊김/LIVE off → recommended/held 없음/LIVE on | PASS,8 GET(6+1+1) |
| unconfirmed | payload 후보1개여도 화면 추천0, unconfirmed state | PASS,6 GET |
| display_state 누락 | payload 후보1개여도 화면 추천0, fallback unconfirmed | PASS,6 GET |
| 미지원 display state 문자열 | payload 후보1개여도 화면 추천0, fallback unconfirmed | PASS,6 GET |

transition은 health publish/sequence_checked1→2→3, file_sequence는1 유지, feed_ok true→false→true다. health.publish는 별도 객체로 분리하여 game JSON은 같은 bytes로 유지된다. 순수 생성 함수 두 개만 Node VM에서1회 확인해 game sequence1/동일 SHA `d495b0d2503eee2bad5333198c0b8ac4646f29b33e7d59662215e70f29b33544`, health sequence1/2/3와 feed true/false/true가 일치했다(서버/브라우저 실행0). 각 전환은 `.sv-head .sv-live` 클릭으로 강제 갱신하며 자동 폴링 회복·실제 시간 지연을 입증하지 않는다. fixture published_at/checked_at 및 HTTP Date를 모두2030-01-01T00:00:00Z로 맞춘다. 이 합성 시계 설정은 실제 자료 freshness 증거가 아니다.

`live-phases.json`에4사례/6단계의 전체5JSON 생성값을 보존하고 receipt에 각 단계의 fixture SHA·실제 응답 SHA·기대 state/추천수/held·트리거·PNG를 기록한다. 첫 단계 game와 각 단계 health 소비를 확인하며 후보1개의 실제 가시성도 검사한다.

## 실행 경계·명령

기존 exact4asset/source/manifest 선검사, symlink/탈출/변조 거부, 배타적 새 output, exact origin GET/SW·WebSocket 차단,60GET/80초 실행·90초 정리 포함 판정, 본인 ephemeral loopback/browser1개 정리를 보존한다. 네트워크 오류나 요청 예산 초과 시 실패를 남기며 상한을 늘리지 않는다. 실제 host/private/모델/자격/운영 프로세스 접근은 없다. root 실행 수용 뒤 선택 suite1회만 실행하고 기본/host-errors는 재실행하지 않는다.

```sh
node tests/coordination/public_candidate_smoke.mjs \
  /Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/pack-v4.3.1/contents/MANIFEST.json \
  7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93 \
  6282c63bd699d9cdf5d7110fff2ce530b7221c4f \
  /Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/pack-v4.3.1/contents/public-build \
  /Users/song/Projects/pitcheezy/apps/observer/web/node_modules/playwright/index.mjs \
  /Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX11/run-001 \
  --live-states
```

실행기 SHA `4b0373a6c84a14aab9a6bebbf1d028bac30ea7ba9e17ec9d9037158cfc4e5e2b`. 위 명령1회 exit0, `PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP`.4사례/6단계, totalCalls=routedAttempts=26, elapsedMs=1308, blocked=[]/pageErrors=[], ownedBrowserClosed=true/ownedServerClosed=true. 기본/host-errors 재실행0. 같은 모델의 별도 에이전트 `coordination_review`가 실행기/receipt/live-phases SHA와 PNG6개를 직접 대조하여 기능 결과 PASS했다. 검토자 추가 브라우저 실행0회다.

기능 표시 검증이며390×844 레이아웃·실기기·진짜 모델/추천 품질·실시간 시계 정합·자동 재연결·TLS·배포 GO는 검증하지 않는다. COORD-04가 정확2파일만 선별 커밋한다.


## 결과 증거와 인계

출력 root: `/Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX11/run-001`.
- receipt.json SHA `cf9aff4e5ea39ddfd752a0a5751b231a0892252ff58e26952f89929115ad2e40`.
- live-phases.json SHA `b545203791ef0bb2c70743717a496e1070eb1071599c0ad8bd762a9ce081155c`.
- synthetic-files.json SHA `278dd95a0738d18e7b0b040eeeaf02755267798b6f3add1c18c14b0ab51eab46`(기존 기본 생성값, live 입력은 별도 phases).
- PNG6개: `transition-{recommended,disconnected_held,recovered}-390.png`, `unconfirmed-unconfirmed-390.png`, `missing_display-missing_display-390.png`, `unknown_display-unknown_display-390.png`. 작성자가 모두 직접 확인했다.

전환의 두 수동 갱신은 health만 각1회 읽고 같은 game seq1/추천을 유지했다. 실제 추천1행의 가시성, held/마지막 추천 문구/LIVE off, 회복 후 held 없음/LIVE on을 확인했다. 나머지3사례는 후보가1개인 합성 game bytes를 읽었으나 화면 추천 행0이었다. HTTP Date만 합성시각이며 화면의 수신 시각은 브라우저 실제 시계를 사용한다. 이 조합을 실제 운영 clock 정합으로 해석하지 않는다.

경계 일탈 기록(검토자 보고, root 기록 지시): 독립 검토자의 후속 위험 탐색에서 고정 commit의 web/tests 디렉터리 전체에 git grep를 적용해 기존 경기 fixture JSON 일부가 검색 출력에 포함됐다. 이 내용은 분석·수치 판정·실행·산출물 입력에 사용하거나 재출력/저장하지 않았고 추가 탐색을 중지했다. CX11의 생성 fixture와 실행 입력은 이 탐색과 분리돼 있다.

graphify: `tests/coordination`에서 `graphify update .` 1회 exit0, AST111 nodes/167 edges/8 communities. 기존4source만 대상이며 ignored `graphify-out/`은 비커밋이다. 전체 저장소/의미 추출은 실행하지 않았다.
