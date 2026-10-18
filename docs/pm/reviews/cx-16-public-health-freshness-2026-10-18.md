# CX-16 — 공개 후보의 health freshness 경계

상태: schema 대조1회 PASS. 최초 브라우저의 chip 기대 오류를 수정한 재시도에서2사례 PASS, 독립 결과 검토 PASS·Root 최종 수용 완료, 커밋 대기.
CLAIM https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249533283 ; base `29888c291fe3523c7b19732c9702ef57ba758a7d`. 작성자 coordination_docs, 검토자 coordination_review.

## 기대와 최소 변경

제품 source `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`: service/publish.ts:158–165는 checked_at이 health.published_at 이후이면 Infinity, 아니면 health/check 중 더 오래된 나이를 사용한다. GameScreen.tsx:37,453–458,504–508은 이를 PUBLISH_LATE로 표시한다. 표시 state는 disconnected지만 feed 장애의 CLIENT_LOST와 달리 문구는 **갱신 지연**이다. 후보는 마지막 수신값으로 흐리게 남고 LIVE 표시가 꺼진다. 즉 추천행0을 기대하지 않는다.

| 새 사례 | checked_at | 공통 입력 | 사전 기대 |
|---|---|---|---|
| stale_check | 2029-12-31T00:00:00.000Z | health/HTTP Date 2030-01-01, feed_ok=true, 후보1 | disconnected.held, 추천1, 마지막 수신 라벨, LIVE off, 갱신 지연 |
| future_check | 2030-01-02T00:00:00.000Z | 위와 동일 | 위와 동일; 시간 순서 모순은 구조 형식과 별개 |

기존 syntheticFiles/liveFixtures 함수 bytes는 base와 동일하다. 새 healthFixtures는 기존 정상 transition 첫 단계를 복제해 checked_at만 바꾸고 검사 기대를 붙인다. 두 사례 game 본문은 동일하다. 새 선택 인자 `--health-freshness`만 추가했으며 기존 default/host-errors/live-states 의미는 보존한다. 공통 held 검사는 선택적 publish-late 문구/data-kind/chip을 검사하고 기존 사례에서는 기존 연결 끊김 기대를 유지한다.

최초 실행기 SHA `99ca9bcd4939941a0949b56da815ddb715bd74d29c4de40d002e001570386e27`, 기대 한 줄 수정 후 SHA `23ba485cf75e6a822830f217ddd51a9a69fa7a8bb96875857be68296a17a74e0`. 출력 root `/Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX16`는 새로 생성했다. source의 순수 생성 함수만 Node VM에서 실행한 `candidate-live-phases.json` SHA `1b9ae18e56ecb7fc1efacbd1bf09ee056ac14234efb2aa41474d5873b00eaf62`.

## 오프라인 구조 대조 계획

기존 CX14/tool-env의 고정 jsonschema4.25.1·referencing0.37.0 및 고정5schema export/hash를 재사용한다. 설치·프로젝트 환경 변경0. 임시 `CX16/schema-check-001.py`는 CX14/003의 import 전 socket 생성/DNS/connection 차단, Registry deny, ref/dynamic/recursive ref 부재·hash·format checker 확인을 유지한다. 지정 candidate1개와 schema5만 읽고 새 결과파일에 배타 저장한다.
명령: `<CX14>/tool-env/bin/python -I -B <CX16>/schema-check-001.py`. 검토 후1회, 총10 body-schema 대조(중복 bytes 포함). Draft202012Validator + FormatChecker지만 schema format 선언0이므로 시간의 의미·교차파일 정합 검증이라고 하지 않는다. 두 사례는 client-semantic-negative이며 구조 거부를 기대하지 않는다.

실제 결과: exit0, 10본문 대조/부적합0/오류0/format0/네트워크 시도0. `schema-check-001.json` SHA `822be327c30633b496e473625f46a9def59497a1a7fcd686db446fa9b928e188`. 구조 적합성이 시간 의미의 정상성을 뜻하지 않는다.

## 브라우저 계획

독립 source/경계 검토 및 Root 수용 뒤 아래 선택 suite1회만 실행한다. 기존4live/default/host suite를 다시 세지 않는다. 기존4asset manifest pin·symlink/경로 allowlist·외부/WS/SW 차단·60GET/90초·본인 ephemeral server/browser cleanup을 유지한다. 예상12GET은 결과가 아니며 실제 계수는 실행 후 기록한다.

```sh
node tests/coordination/public_candidate_smoke.mjs \
  /Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/pack-v4.3.1/contents/MANIFEST.json \
  7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93 \
  6282c63bd699d9cdf5d7110fff2ce530b7221c4f \
  /Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/pack-v4.3.1/contents/public-build \
  /Users/song/Projects/pitcheezy/apps/observer/web/node_modules/playwright/index.mjs \
  /Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX16/browser-001 \
  --health-freshness
```

최초 browser-001은 exit1/INCOMPLETE, 6GET/완료사례0/731ms/외부0/소유 자원 정리true였다. receipt SHA `39f21581607e54c5f5276c58e1a600aeadb0d59b36f6eb0ec698dc416461dfc9`. 실제 chip은 `⚠︎` 아이콘과 줄바꿈 뒤 `갱신 지연`이었고 문구만 비교한 새 assertion이 실패했다. 고정 GameScreen.tsx:30–32,613의 아이콘 렌더링과 일치하는 하네스 기대 오류다. 아이콘+공백+정확한 문구를 요구하도록 한 줄만 수정했다. 첫 증거는 보존하며 browser-002 새 출력 재시도는 재검토·Root 수용 뒤 같은 명령의 출력 폴더만 바꿔 실행한다. fixture bytes는 그대로여서 추가 schema 대조는 필요 없다.

## 실제 재시도 결과

한 줄 수정의 독립 검토·Root 수용 후 같은 명령의 출력만 `browser-002`로 바꿔 1회 실행했다. exit0/PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP, 2사례 각각6GET/총12GET, 517ms, page error0/외부0/소유 browser·server 종료true. 두 사례 모두 disconnected.held·보이는 추천1·마지막 수신 라벨·LIVE off·갱신 지연 문구/data-kind/chip을 만족했다. 최초001 실패와002 성공을 별도로 보존하며 고유 사례를4개로 합산하지 않는다.

receipt SHA `3d3efed4c1e4d596bcb0702e60d83ac78fc155efe2c663319c4041297e58bb86`; browser002의 live-phases SHA는 offline candidate `1b9ae18e56ecb7fc1efacbd1bf09ee056ac14234efb2aa41474d5873b00eaf62`와 같다. PNG 화면을 직접 확인했고 두 PNG는 SHA `a0991e5e0649663d1a9462d82f671ce4deb115c068f7a8bc4ab803b37adadf4d`로 byte 동일하다. 이전 suite는 재실행하지 않았다.

현재 고정 v4.3.1 준비 후보의 합성 HTTP/390×900 기능 검사다. 초기 응답의 오래된 추천을 마지막 수신값으로 표시하는 검사이며 실제 시간 흐름·자동 polling·실제 freshness·TLS·모델·성능·운영/배포를 증명하지 않는다. 제품 시계·threshold 정책·실자료는 수정/사용하지 않았다. 같은 모델 별도 에이전트 coordination_review가 source/receipt/phase SHA·PNG를 독립 대조해 PASS했고 추가 실행0이다. Root가 최종 수용했다. scoped graphify update .를 tests/coordination에서1회 실행했고 기존4 source만 AST 갱신했다(산출물 ignored).
