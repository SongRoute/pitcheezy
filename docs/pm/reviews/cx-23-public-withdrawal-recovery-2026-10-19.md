# CX-23 — 같은 페이지에서 철회 기억과 명시 복구

작성자 `implementation_1019`, 독립 검토자 `queue_1019`. 상태: 독립 경계·schema·결과·PNG 검토 PASS 및 Root 수용. 브라우저 1회 통과, 검토자의 추가 실행 없음. Base `4d0327e490f509efd484308a930e6bf06f257182`.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6254463354 .

## 근거와 변경

공개 후보 source `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`의 `GameScreen.tsx` load 경로는 health의 `file_sequence=null`로 철회된 경기의 detail을 지우고, 그 뒤 유효한 명시 check가 없으면 과거 game 파일을 다시 읽지 않는다. `api.ts`는 이미 적용한 health보다 낮은 publish sequence를 버린다. 새 명시 check/file이 준비되면 detail을 복구한다. 이 동작을 페이지의 ref/cache가 유지된 상태로 확인한다.

하네스에 optional `--withdrawal-recovery`와 `withdrawalFixtures`만 추가하고 공통 live 분기에 unavailable DOM·버튼·navigation 검사를 최소 확장했다. 기존 `syntheticFiles`, `liveFixtures`, `healthFixtures`의 함수 바이트는 기준 커밋과 동일하다. 제품 파일·빌드·dependency·기존 suite 기대는 변경하지 않았다.

1개 case에 5개 phase를 둔다. game990001/2030-01-01/인명·추천은 기존 전적 생성 fixture를 재사용하며 실자료·캡처된 행을 읽지 않는다.

| phase | health publish/check | game 파일 | 기대 |
|---|---|---|---|
| recommended | seq1 / file1 | seq1 추천1 | recommended·추천1·LIVE on |
| withdrawn | seq2 / file null | 이전 seq1 추천1 그대로 서버에 존재 | unavailable·추천0·LIVE0, game GET0 |
| old_health | seq1 / file1 | 이전 seq1 그대로 | 과거 health로 철회 해제하지 않음, unavailable·game GET0 |
| missing_check | seq3 / game_checks={} | 이전 seq1 그대로 | 명시 복구 근거 없으므로 unavailable·game GET0 |
| recovered | seq4 / file4 | seq4 추천1 | 새 game 읽고 recommended·추천1·LIVE on |

최초 `goto`는 1번뿐이다. 철회 진입은 LIVE 버튼, 이후는 `다시 확인` 버튼을 누른다. 각 health 응답이 끝난 후 두 animation frame을 기다려 렌더 기회를 주고 DOM을 검사한다. 고정 시간 sleep이나 reload는 없다. 각 단계 navigation count1을 확인하며 단계별 body SHA·요청·PNG를 남긴다. 철회 중에는 서버에 후보1이 남아 있어도 추천0이며 game 요청 자체가 없어야 한다.

## 고정 입력과 구조 검증

- manifest SHA `7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93`, 지정 v4.3.1 pack의 public-build 4개 asset은 기존 정확 pin을 유지했다.
- harness SHA `b686e1df3bd2cf49fa1a2fe054ba9100547cbab14933f012943cf577bc1ea9a5`.
- Node VM에서 하네스 순수 생성 함수만 추출해 만든 candidate SHA `29185f6b8567dffc2eda2e5cd74daca8510d80bc570a8af2343396b89eb80f2a`. 이 단계에서는 서버·브라우저를 실행하지 않았다.
- CX14의 고정 jsonschema4.25.1 환경과 5개 schema SHA를 재사용했다. [CX14](cx-14-synthetic-schema-conformance-2026-10-18.md)의 핀을 임시 schema checker에 정확히 assert하며 import 전 socket/DNS 차단, Registry 외부 reference 거부를 유지했다. 새 설치는 없다.

`queue_1019`가 하네스·입력·schema 명령을 독립 경계 검토한 뒤 CX23/schema-check-001.py를 1회 실행했다. exit0, 25 body–schema 대조, 부적합0/오류0/네트워크 시도0. 동일 bytes를 반복 대조한 항목을 포함하며 25개 고유 fixture가 아니다. format 선언0이므로 시간 의미·교차파일 관계의 schema 검증이라고 하지 않는다. schema 결과 SHA `a22dd32025d33c542e3f50f6b57482cdfddca1cb2d43963ef82bf131f729c41c`.

## 브라우저 결과

schema 결과 독립 대조 PASS 뒤 아래 기존 인자 방식에 새 suite를 선택해 **1회** 실행했다. 정확한 manifest/build/Playwright 위치는 CX16의 고정 인계를 재사용하고 출력은 CX23/browser-001 새 폴더다.

```sh
node tests/coordination/public_candidate_smoke.mjs \
  PINNED_MANIFEST 7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93 \
  6282c63bd699d9cdf5d7110fff2ce530b7221c4f PINNED_PUBLIC_BUILD \
  INSTALLED_PLAYWRIGHT_INDEX NEW_CX23_BROWSER_OUTPUT --withdrawal-recovery
```

실제 결과: exit0, `PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP`, 1case/5phase, 총11GET(6/1/1/1/2), 1249ms, page error0/외부·금지 요청0. 소유 browser/server 종료 모두 true. 60GET/90초 상한·protected port 거부·GET allowlist·외부/WS/SW 차단은 그대로다. 하네스·manifest·4asset의 실행 전후 SHA가 모두 같았다.

각 단계의 DOM·game GET0·navigation1 및 최종 file4 SHA 대조가 통과했다. 브라우저가 기록한 live-phases SHA는 오프라인 candidate와 같다. receipt SHA `7b31a3092cae9631c7acf0c9e9b5276b440a913382848b4d541548dcb77c4b4c`.

서로 다른 화면2개를 직접 확인했다. 최초 추천과 최종 복구 PNG는 SHA `19ef00efcb5b040212ba15744a775f3df9cbbc99806e071f94095ac09d5025bc`로 동일하며 철회/과거 health/누락 check PNG3개는 SHA `5fc75d37eea62fc3ee553d75a889b8469c9b007e170fd570bd2d7106df70bed0`로 동일하다. 철회 화면에는 안내·다시 확인·경기 목록만 있고 추천이 없다. 복구 화면에는 추천1과 LIVE 표시가 있다.

`node --check`, `git diff --check` 통과. `tests/coordination`에서 AST-only `graphify update .` 1회 수행했고 하위 ignored graph만 갱신했다.

## 한계와 인계

동일 문서의 수동 재확인과 유한한 합성 순서를 검증했다. 실제 publisher 재시작/epoch, 실운영 지연, 자동 polling 시간 경과, 503 경로, TLS·배포·모델 성능은 검증하지 않았다. viewport390×900이고 HTTP Date는2030 합성 기준이다. 실자료·운영 프로세스·GPU·기록기·실호스트에 접근하지 않았다. 이번 case를 5개의 독립 사례로 합산하지 않는다.

작성자는 Git index를 조작하지 않았으며 독립 결과·PNG 검토 후 COORD-05가 정확한 2파일을 선별 커밋한다.
