# CX-24 — 부분 JSON 응답의 본문 timeout과 수동 복구

작성자 `implementation_1019`, 독립 검토자 `queue_1019`. 상태: 독립 경계·결과·PNG 검토 PASS 및 Root 수용. 브라우저 1회 통과, 검토자의 추가 재실행 없음. Base `68451af77cf891a665eba2701c77b4e958212d64`.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6254526164 .

## 목적과 고정 경계

공개 source `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`의 `api.ts:getPublished`는 response.json 처리까지 timeout을 유지한다. 같은 source의 `.env.public`에 timeout15000ms/retry4000ms가 있고 `GameScreen.tsx`는 최초 status0 오류에 CADENCE.retry를 사용한다. 헤더를 받은 뒤 본문이 끝나지 않아도 사용자 화면이 무기한 멈추지 않는지 검사했다.

하네스에 optional `--partial-body` 1case를 추가했다. 완전히 생성한 기존 final game 경로만 HTTP200/JSON header와 10bytes `{"schema":`를 보내고 **server가 response.end를 호출하지 않는다**. 소유 response의 close를 관측한다. 다른 경로는 기존 정상 응답이다. 앱·bundle·시계·timeout 설정을 바꾸지 않았다.

신규 case의 alert 대기는25초, 부분 응답 시작부터 alert/response close까지 각각14–25초를 검사 허용범위로 사전 고정했다. 이것은 제품 SLA나 실서비스 측정 기준이 아니다. `performance.now()`로 실제 구간을 기록했다. alert 후2초 안에 flag를 해제하고 수동 다시 시도하며,4초 안에 회복을 확인해 자동 재시도가 먼저 복구한 것을 수동 성공으로 오인하지 않도록 했다. 요청 기록에 최초/수동 trigger와 monotonic 시작 시점을 추가하고 최종 완전 body SHA를 대조한다.

기존 exact4asset pin·manifest/source pin·소유 loopback·protected port 거부·GET allowlist·외부/WS/SW 차단·60GET/90초 및 소유 자원 정리는 유지했다. partial response가 열려 있으면 cleanup 실패로 판정한다. 기존 4개 fixture 생성 함수와 pin 함수는 바이트 동일하다. 기존 suite는 재실행하지 않았다.

## 실행과 실제 결과

하네스 SHA `a5cbf8e3400537cab281c793bd7c3c7ed9ddf96c2349fd3f1c0fb0de5820e975`. `queue_1019`가 diff·경계·고정 timeout/retry source를 읽기 검토한 뒤1회 실행했다. manifest `7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93`와 기존 v4.3.1 public-build/Playwright 입력을 그대로 사용하고 출력은 CX24/browser-001 새 폴더다.

```sh
node tests/coordination/public_candidate_smoke.mjs \
  PINNED_MANIFEST 7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93 \
  6282c63bd699d9cdf5d7110fff2ce530b7221c4f PINNED_PUBLIC_BUILD \
  INSTALLED_PLAYWRIGHT_INDEX NEW_CX24_BROWSER_OUTPUT --partial-body
```

실제 `browser-001`: exit0, `PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP`,1case/2화면,8GET,총16244ms. 부분 응답1개/serverEnded=false/client close1개. 소유 browser/server 및 partial response 종료 true, page error0/금지 요청0. 하네스·manifest·4asset 전후 SHA가 같다.

| 관측 | 결과 |
|---|---|
| 부분 body 시작→response close | 15006.914083ms |
| 부분 body 시작→alert 관측 | 15421.655042ms |
| alert→수동 retry 표시 시점 | 41.883082999998805ms |
| 최초 오류 DOM | `서버 응답을 읽지 못했어요.` / 추천0 |
| 수동 회복 | health200 뒤 game200, final 화면/종료 표시, alert0·추천0 |
| 완전 game body SHA | `13692f0be6ea6bb97590f688ac3ce9b2fd9037cb887bcb239f77358823afeff9` |
| 문서 이동 | navigation1 / reload0 |

두 recovery 요청은 수동 trigger 표시 이후 시작했다. client close가 먼저 확인됐고 server는 부분 응답을 정상 종료하지 않았다. 두 PNG를 직접 열어 오류 안내와 정상 final 화면을 확인했다. timeout과 복구를 두 개의 독립 case로 합산하지 않는다.

| 산출물 | SHA-256 |
|---|---|
| receipt.json | `2085b22b2a439ce238b726b612a1b468ae36922a6c59fb8e1cfbac70f3e3a684` |
| synthetic-files.json | `278dd95a0738d18e7b0b040eeeaf02755267798b6f3add1c18c14b0ab51eab46` |
| partial-timeout-390.png | `91e821b8e7ae66a1fe5de2a07139990d42a11096a3d35ffcd68f9f67ffb06a89` |
| partial-recovered-390.png | `54cbc2166f9a62f009e7f32a5fd82fe247ecee6dc0a8a09380d5b92c83a3dffb` |

정상 synthetic-files bytes는 CX05/CX14 검증 입력과 동일해서 schema를 반복 실행하지 않았다. 의도적으로 미완성인 stream은 schema PASS라고 하지 않는다. `node --check`/`git diff --check` 통과, `tests/coordination` scoped AST-only update1회(140nodes/241edges/7communities), ignored graph만 갱신했다.

## 한계와 인계

고정 공개 빌드·390×900·소유 HTTP loopback의 한 번의 본문 정지와 수동 복구를 확인했다. TLS·실호스트·실서비스 지연·자동 재시도 성공·모델·운영/배포를 검증하지 않았다. 실제 자료·자격·다른 프로세스·운영 자원을 사용하지 않았다. 제품 source 수정이나 dependency 설치도 없다.

독립 결과·PNG 검토 후 COORD-05 단일 index 소유자가 정확한 하네스와 이 보고서2파일만 선별 커밋한다. 작성자는 stage/commit하지 않았다.
