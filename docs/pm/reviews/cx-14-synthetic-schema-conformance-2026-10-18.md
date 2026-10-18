# CX-14 — 합성 JSON의 고정 공개 스키마 적합성

상태: 오프라인 구조 대조·최소 fixture 보완 및 영향받은 live 브라우저 1회 PASS. 독립 결과 검토 PASS·Root 최종 수용 완료, 커밋 대기.
작성자 `coordination_docs`, 검토자 `coordination_review`. CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249351495 . base `4022acb1588a526b2946792c67f3a7b232d3bf98`.

## 범위와 도구

프로젝트 `.venv` 및 기존 기본 Python에는 jsonschema/referencing이 없었다. 새 task 전용 `CX14/tool-env`에 wheel만 설치했고 프로젝트 환경·dependency 파일은 수정하지 않았다. PyPI 취득만 네트워크를 사용했으며 검증 단계는 socket 생성·DNS·connection 및 외부 reference 취득을 차단했다. 모델·실경기 fixture·실자료·private/운영 파일은 읽지 않았다.

출력 root는 `/Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX14`다. 설치 명령은 `python3 -I -B -m venv <root>/tool-env`, 이어서 `<root>/tool-env/bin/python -I -B -m pip --isolated --disable-pip-version-check --no-cache-dir install --only-binary=:all: --index-url https://pypi.org/simple jsonschema==4.25.1`이다. Python3.14.3, jsonschema4.25.1, referencing0.37.0, jsonschema-specifications2025.9.1, attrs26.1.0, rpds-py2026.9.1이며 `requirements-frozen.txt`에 보존했다. 재설치/빌드/새 compiler는 없었다.

지정 source `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`의 `apps/observer/publish_contract/v1/schema/` 아래5파일만 `git show`로 `<root>/schemas`에 배타 저장했다. 모든 스키마는 draft2020-12이며 `$ref`/`$dynamicRef`/`$recursiveRef` 키가 없다. Draft202012Validator.check_schema와 iter_errors, FormatChecker를 사용했으나 실제 schema의 format 선언은0개다. 따라서 날짜 의미·자료 진실성·교차파일 관계 검증을 주장하지 않는다.

| schema | SHA-256 |
|---|---|
| dates | dabbfe4e99526c419238a555a849a7751b7183cf4248f1b5ab1bad69d7e9a669 |
| game | 0fc2bc10d723ddc05391b5aa3b9e8674624660b42722b08523bca12ebde35251 |
| games | 73fbc1867f8e48def9de0f7fe009b3c0ef6209e48be9b0b175303e24763edc08 |
| health | 7b75305c59bcdd4b4707213b96743f8c4f0b1b6b033138858614897f3fb8fa66 |
| manifest | f932e253e1ba422fe857ccdcf02fe818eaec283a339981d86e931b322b0b9583 |

## 실제 대조와 수정

| 단계 | 결과 | 의미 |
|---|---|---|
| schema-check-001 | guard/import TypeError, 실제 대조0 | socket.socket을 함수로 바꿔 SSL 클래스 로딩이 실패한 도구 문제; schema/제품 결함 아님 |
| schema-check-002 | 35건 중11본문 부적합,82개 오류 | 기존 CX05 기본5개는 적합. CX11 live display/feed 필수14키×5game와 목록 inning/half null형식2키×6단계 불일치 |
| schema-check-003 | 35건 대조, 부적합0/오류0/네트워크 시도0 | 보완한 생성 JSON 구조 적합. 중복 bytes 포함 본문–스키마 대조이며35고유 fixture/시험 아님 |

001 이후 socket.socket 타입을 유지하는 DeniedSocket.__new__ 거부로 고쳤고 독립 경계 재검토/root 수용 뒤002를 실행했다. jsonschema import 전에 이 guard와 DNS 차단을 설치하고, Registry(retrieve=deny)도 사용했다. 설치/고정 schema export는 재실행하지 않았다.

원래 입력은 CX05 `synthetic-files.json` SHA `278dd95a0738d18e7b0b040eeeaf02755267798b6f3add1c18c14b0ab51eab46` 및 CX11 `live-phases.json` SHA `b545203791ef0bb2c70743717a496e1070eb1071599c0ad8bd762a9ce081155c`로 확인했다. 기존 파일/증거는 그대로다. 결과 JSON은 error path와 schema path를 분리해 기록한다.

수정은 `public_candidate_smoke.mjs`의 liveFixtures 안에만 있다. schema가 요구하는 display/feed/ledger 중립 합성 필드를 추가하고 live 목록 card의 inning/half를 null로 맞췄다. game의 실제 합성 상황은 그대로다. label 빈 문자열/message null은 기존 로컬 문구 fallback을 보존한다. threshold0·ledger synthetic_only 등은 **시험용 형식값**이며 운영 정책·실제 기록/시계 보장을 뜻하지 않는다. 이 작업은 제품 버그 수정이 아니다.

`liveFixtures` 밖 모든 bytes가 base와 동일함을 확인했다. 기존 기본 JSON SHA도 동일하다. 보완 live 생성값은 `candidate-live-phases.json` SHA `d310ae448f822396dbbe42c73e68b2b9178deed7f10d06544ded7efe9a1a79c7`이며 같은 game의 전환3단계 bytes/seq1 보존도 확인했다.

**missing_display는 schema상 optional이고 unknown state는 enum 제한 없는 string이므로 둘 다 구조검사를 통과할 수 있다.** 둘은 클라이언트 의미상 negative다. unconfirmed/누락/미지원에서 추천을 숨기는지는 브라우저 판정과 별개다. 일반 필수키 누락을 negative 성공으로 셈하지 않았다. 이전12개 브라우저 사례를 소급 schema PASS로 바꾸거나 새 재실행과 고유 사례 수를 합산하지 않는다.

## 재현과 후속 실행

`<root>/schema-check-002.py`와 `schema-check-003.py`는 로컬 입력·hash·경로 및 차단을 고정한 이번 작업의 임시 재현 명령이다(새 repo helper/검증 framework 아님). 실행 명령: `<root>/tool-env/bin/python -I -B <root>/schema-check-003.py`. 기존 결과 파일은 배타 생성으로 보호하므로 재현 시 새 결과 파일명을 사용해야 한다. 003은 Node VM에서 현재 source의 순수 생성 함수2개만 실행하여 배타 저장한 candidate JSON을 검사했다. 이 오프라인 단계에서는 전체 harness import/서버/브라우저를 실행하지 않았다.

실행기 SHA `1c555b52c4b8e4fa1002e818cb6e2695903260ab317b8622ebab0a3953ed1906`.
독립 source 검토/root 수용 후 보완 영향을 받는 live suite1회만 아래 명령으로 실행했다. default/host-errors는 bytes가 보존돼 재실행하지 않는다. 기존 exact4asset pin/60GET/90초/소유 loopback·browser 정리를 유지한다.

```sh
node tests/coordination/public_candidate_smoke.mjs \
  /Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/pack-v4.3.1/contents/MANIFEST.json \
  7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93 \
  6282c63bd699d9cdf5d7110fff2ce530b7221c4f \
  /Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/I-partner-pack/pack-v4.3.1/contents/public-build \
  /Users/song/Projects/pitcheezy/apps/observer/web/node_modules/playwright/index.mjs \
  /Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX14/browser-live-001 \
  --live-states
```

003 결과 SHA `2334440d2ad64d4dc9ca2b32e3dc5e95a5d98abfb2a154e0bb7fd72de5458d08`;002 결과 SHA `c60742b584e90c5a34ac014d757bfbb1d4f7265f28051a5b9a5140812374f7c6`.

## 브라우저 결과

`browser-live-001/receipt.json`: exit0, `PASS_PINNED_CANDIDATE_SYNTHETIC_HTTP`; 4사례/6단계, 26GET(라우팅26), 1278ms, 외부 차단 시도0/page error0, 소유 browser/server 종료 모두 true. receipt SHA `5464bf8311b2a28058b9583d79a1bcc336e8badfd7ca4e05a3ad6864965e404a`.

| 사례 | 관측 | GET |
|---|---|---|
| 같은 game 전환 | 추천1 → 흐린 마지막 추천1/LIVE off → 추천1/LIVE on | 8 |
| unconfirmed | payload 후보1, 보이는 추천0 | 6 |
| display 누락 | payload 후보1, 보이는 추천0 | 6 |
| 미지원 display 문자열 | payload 후보1, 보이는 추천0 | 6 |

브라우저가 저장한 `live-phases.json` SHA가 오프라인003의 candidate SHA `d310ae448f822396dbbe42c73e68b2b9178deed7f10d06544ded7efe9a1a79c7`와 일치했다. 같은 입력의 구조 적합성과 클라이언트 의미상 fail-closed를 각각 확인한 것이다. 기존 CX11 증거는 변경하지 않았다.

PNG6개 중 서로 다른 화면3개(정상/연결 끊김/확인 중)를 직접 확인했다. 정상과 복구 PNG는 byte 동일하며, unconfirmed/누락/미지원 PNG3개도 byte 동일함을 SHA로 확인했다. 수동 LIVE 버튼으로 health 재조회했고 자동 polling 복구·실제 freshness·TLS·모델 성능·390×844 레이아웃·배포를 검증하지 않았다. viewport390×900, HTTP Date는2030 합성 기준이다.

`tests/coordination`에서 scoped `graphify update .` 1회 exit0: 기존4개 source만, AST111nodes/167edges/8communities. 산출물은 해당 하위 graphify-out에만 생성했다. 같은 모델 별도 에이전트 `coordination_review`가 source·오프라인 기록·브라우저 receipt·PNG를 대조해 최종 PASS했다. 검토자의 추가 실행은0회이며 Root가 최종 수용했다. 운영/실자료/성능/배포 판단으로 확대하지 않는다.
