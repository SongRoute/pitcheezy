# CX39 — 참고 카드 합성 계약 예제의 gateway 원문 보존

- 작성: Codex implementation_1019, 2026-10-19 KST. 검토: queue_1019 / Root.
- 상태: 독립 실행 경계·결과 검토 PASS 및 Root 수용. 합성1 lifecycle 통과, 별도 검토자의 재실행 없음.
- 선점: [CLAIM6258020314](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6258020314), base `8afedb118b7ddc4890867114e0df55e382283b2b`.

## 목적과 고정 입력

우리 v4.3.1의 `next.reference` 예제 원문을 새로 수신한 팀원 gateway handler에 전달하여 생성→읽기→삭제표식→비공개 HEAD 확인→CAS 복구→읽기 한 lifecycle을 검사한다. 기존 [CX37 정적 반환팩 검증](cx-37-partner-return-pack-review-2026-10-19.md)을 바탕으로 실제 모듈 함수를 메모리 저장소에 연결한다. Cloudflare 런타임·Miniflare·실제 HTTPS·브라우저 화면·배포 검증이 아니다.

고정 팩 `pack-v4.3.1/contents/README.md`는 포함된 여섯 예제를 **synthetic examples**라고 명시한다. MANIFEST의 source는 `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`다. 선택 예제는 game900200, 가명 B/P 인물이 들어 있는 계약 합성 입력이다. 실제 경기·모델 출력·실자료를 열거나 원생성기를 실행하지 않았다. 원생성기 독립 재현 주장은 하지 않는다. 예제 안의 통계 설명/숫자는 이 합성 입력의 일부이며 실측 성능·실경기 통계로 인용하지 않는다.

| 입력 | SHA-256 |
|---|---|
| `MANIFEST.json` | `7c8ef7e8df38507e2ee9adfdb9e9d3637370065ea4ed67171080d3846d0d2d93` |
| `README.md` | `0cb5492b71866008fd74e1f0add60088960c6cba6d89eac58d5589533acff80c` |
| `apps/observer/publish_contract/v1/examples/game-live-unsupported-reference.json` (38454 bytes) | `9e34b37959b1221bbbc2a7d55b30a053981a6d73743183b695ed56ba3b7604f2` |
| `apps/observer/publish_contract/v1/schema/game.schema.json` (41721 bytes) | `0fc2bc10d723ddc05391b5aa3b9e8674624660b42722b08523bca12ebde35251` |

예제/schema는 manifest의 정확 키·크기·SHA와도 대조한다. schema의 참고 카드 정의와 선택 필드를 정적으로 확인하며 전체 JSON Schema validator를 실행한 것으로 표현하지 않는다. gateway는 야구 schema 검증기를 포함하지 않으며 PUT 응답의 `schema_validated:false`도 확인한다.

## 실행 경계와 검사

[테스트](../../../tests/coordination/test_partner_reference_transport.mjs)는 runtime5파일을 모두 고정 SHA·일반 파일·심볼릭 링크 없음·0400 mode로 확인한 뒤 gateway만 import한다. 실행 import closure는 `gateway.mjs`와 직접 의존 `notice.mjs`/`access.mjs`/`telemetry.mjs`의4파일이다. `worker.mjs`는 배포 entry 참고 pin만 확인하며 import하지 않는다. 이5개는 CX37에서5eff 반환 소스 및48ec runtime 정본과 대조한 바이트다.

모듈을 load하기 전에 global fetch/WebSocket, HTTP·socket·서버·child process API를 차단하고 현재 테스트 프로세스의 환경을 비운다. token은 실행 중 새로 만든 합성 값만 사용하며 출력하지 않는다. 외부 HTTP·포트·설치·Miniflare·브라우저·실호스트·운영 자격을 사용하지 않는다. partner의 전체 테스트나 Miniflare 모듈은 import하지 않는다.

메모리 Bucket은 고정 partner `tests/worker.test.mjs`의 CAS/ETag 패턴을 최소 재사용한다. 조건부 put, 원문 SHA, 임의 UUID가 포함된 내부 record의 ETag를 다룬다. limiter는 같은 publisher key를 계수하는 성공 mock이다. 실제 R2의 동시성/원자성이나 Cloudflare rate limiter 동작을 입증하지 않는다. viewer gate는 이 합성 fixture에서 비활성이고 관람 인증/회수는 검사 대상이 아니다.

| 순서 | 직접 handler 요청 | 기대 |
|---|---|---|
| 1 | 비공개 HEAD | 미사용404 |
| 2 | PUT If-None-Match:* | 생성200, schema_validated=false |
| 3 | 공개 GET | 200, 입력 전체 바이트/SHA·생성 ETag 일치 |
| 4 | DELETE If-Match 생성 ETag | 200, 새 ETag의 tombstone |
| 5 | 공개 GET | 404 not_found, 이전 payload 미노출 |
| 6 | 비공개 HEAD | tombstone ETag·deleted=true |
| 7 | PUT If-Match tombstone ETag, 동일 본문 | 복구200, 생성/철회와 다른 ETag |
| 8 | 공개 GET | 원래 바이트/SHA 동일·복구 ETag 일치 |

합계는8회 직접 호출(HEAD2/PUT2/GET3/DELETE1), 모의 limiter5회, 성공 저장3회로 고정한다. 공개 GET2회에 전체 원문·파싱 결과와 다음 의미를 대조한다: `next.rec.status=unsupported`, 후보0, provenance absent, policy/recorded_at null; `next.reference`의 참고 카드 표기·추천/예측이 아니라는 문구;10개 투구의 `rec.pairing.status=none` 및 no_recorded_recommendation. reference를 추천으로 변환하거나 과거 투구에 추천을 붙이지 않는다. timestamps를 바꾸지 않으며 freshness 평가는 하지 않는다.

## 결과와 보존

독립 경계 검토 후 `env -i PATH=/opt/homebrew/bin:/usr/bin:/bin /opt/homebrew/bin/node tests/coordination/test_partner_reference_transport.mjs <fixed-pack-root> <fixed-host-root>`를 정확 한 번 실행했다. Node **v25.9.0**, exit **0**, stderr **0 bytes**, 결과 **PASS_SYNTHETIC_GATEWAY_HANDLER**다. **in-process gateway handler 8 calls / wire0 / 1 lifecycle**이며8개 고유 테스트나 HTTP 실측으로 합산하지 않는다. 모의 limiter5회·성공 저장3회·차단 API 시도0이다.

생성 ETag는 `5081f3c004f069a2312f3264fe9187fa`, 철회는 `31f2335986cb13146083e8c4feaa4b15`, 복구는 `74db4fd16a92e8f70f54f4cf7cc187d9`였다(응답에는 strong ETag의 큰따옴표 포함). 공개 응답은200→404→200이고 생성/복구200의 본문38454 bytes와 SHA는 모두 원본 `9e34b379…7604f2`와 같았다. unsupported·빈 후보·참고 표기·10개 기록의 no_recorded_recommendation/none 대조도 통과했다. 입력4개·runtime5개는 실행 전/완료 시 정확 whitelist로 다시 읽어 모두 같은 pin임을 확인했다.

전용 출력은 `20261019-codex-return-pack/CX39`이며 원본 입력과 별도로 보존했다.

| 증거 | SHA-256 |
|---|---|
| 실행 테스트 | `82d26e9aeed2ec1e79e3a459b48ebca0dfef45acccb0018209393569d1d3b772` |
| `stdout.json` | `38653cf8ea90526b0fae88861b9b5f09482daa617c6f489d9f6f627507ac1929` |
| `stderr.txt` (빈 파일) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `exit.txt` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `run-metadata.json` | `c5dfb7fabee06c54c7ca0ebf10f65212e87ee130648b78c10083b70e3c9cc618` |

이 결과는 고정 입력의 메모리 handler 원문 보존 근거다. 전체 schema 유효성·실제 서버의 인증/회수·동시성·화면 표시·freshness·추천 성능을 새로 확인한 것이 아니다. 추가 회귀·브라우저 실행은 하지 않았다. 코드 변경 후 `tests/coordination`에서 `graphify update .`를1회 실행해 해당 폴더의 ignored graph만 AST 방식으로 갱신했다(exit0,173 nodes/296 edges). 제품/root graph와 실자료는 대상이 아니다.

repo 변경은 이 테스트와 보고서2파일뿐이며 제품·원본 첨부·기존 예제·기존 테스트는 변경하지 않는다. 작성자는 stage/commit하지 않는다. 외부 회신 발신도 없다.
