# CX37 — 팀원 v4.2 반환팩 정적 수신 검증

- 작성: Codex implementation_1019, 2026-10-19 KST. 검토: queue_1019.
- 상태: 실제 첨부·manifest·고정 소스 99blob 대조 및 독립 정적 검토 PASS, Root 수용. 실제 공동시험이나 운영 GO가 아니다.
- 선점: [CLAIM 6257253130](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6257253130). 기존 stale-alert 후보는 미착수 보류이며 이번 CX37은 사용자 반환팩 검토다.

## 수신한 바이트와 안전한 추출

입력은 `pitcheezy-host-v42-compat-20261018.zip`, **354208 bytes**, SHA-256 **`ec24db14ba36610618b307cb2840bf9080c45d94fe6e3a68064878dc74b13164`**다. 사용자 전달 기대값과 실제 첨부가 일치했다.

전용 CX37 출력에 새로 추출하기 전에 절대/부모경로·역슬래시·제어문자·중복 및 대소문자 충돌·심볼릭 링크·특수파일·암호화 항목을 거부하는 검사를 적용했다. 104개 항목은 모두 일반 파일이고 총 비압축 크기는 **1017471 bytes**, 개별 최대 압축비는 약 **6.54**다. 검사 상한은 항목256개·전체4MiB·개별1MiB·개별비율100이었다. 항목별 CRC를 확인하며 배타 생성했고 파일은 읽기 전용 0400으로 보존했다. 첨부의 실행 비트나 명령을 실행하지 않았다.

`MANIFEST.json`은 자신을 제외한 **103항목**을 담는다. 정확한 파일 집합, 각 크기와 SHA-256이 모두 일치한다. **source로 기재한99 + 생성한 인계 metadata4 + manifest자체1 = ZIP104**이며 숫자들은 서로 모순되지 않는다. manifest SHA는 `747cef0eacc57673fcdd5af12809e741ab00144070ccdcd15b90f8ce1b592c09`다.

## 소스 정본 대조와 식별자의 의미

README/MANIFEST의 source는 `5effc7a4deef21ee37d07f8574b43b23935d2189`다. README 자체에는 저장소 이름이 없으나 RECEIPT의 팀원 이슈 출처, [기존 CX01](cx-01-onekey-audit-2026-10-18.md)의 같은 후보·팀원 문서 연결과 Root의 명시한 저장소 범위로 `Pitcheezy/transition-models`를 확인했다.

초기 로컬 Git 객체 조회에는 해당 commit이 없어 처음에는 manifest 내부 일치까지만 기록했다. 이후 허용된 GitHub의 **고정 5eff tree API 한 번**을 읽어 저장했고 `truncated=false`를 확인했다. 첨부99파일 각각에 `SHA1("blob " + byte_length + NUL + exact_bytes)`를 계산하여 해당 경로의 Git tree blob ID·일반 파일 mode와 대조했다. **99개 모두 일치, 누락·불일치0**이다. 따라서 source99에 대해서는 manifest 자기 진술을 넘어 고정 저장소 tree와 바이트를 대조했다. fetch·checkout·첨부 코드 import/실행은 하지 않았다.

| 식별자 | 의미 / 이번 확인 범위 |
|---|---|
| ZIP `ec24…3164`, 354208 bytes | 이번 팀원 반환팩 실제 수신 바이트 |
| source `5effc7a4…2189` | 이번 반환 소스99파일. 고정 GitHub tree와 대조 완료 |
| RECEIPT의 `source_commit=57f5d3ff…ebf`, ZIP `37c0d071…693b`, 181953 bytes | 팀원이 과거 받은 **우리 v4.2 팩**의 검증 대상 식별자. 이번 ZIP의 source/크기 불일치가 아님 |
| runtime reference `48ec4065…6c0` | 기존 PR42 런타임 참조. 첨부 runtime5개를 독립 검토자가 고정48ec GitHub tree와 Git blob ID로 대조해 5/5 일치 확인 |

작성자의 최초 runtime 대조는 RECEIPT 기재 해시와의 일치였다. 이후 queue_1019가 `48ec406589982547e298ff2ee4bc2d64ed0216c0`의 tree를 별도로 한 번 읽어 worker/gateway/access/notice/telemetry 5파일의 Git blob ID를 독립 대조했다. 모두 일치했고 `truncated=false`였다. 그 응답 SHA-256은 `1796966b6211cde3dda905fae8aa561798b2a0202cae4d9febc12bd64455d1f8`이며 검토자의 tool 출력에 남아 있다. 별도 raw JSON 파일을 보존했다고 주장하지 않는다.

생성 metadata4개는 manifest와 일치하지만 Git commit의 source99와 같은 근거로 인증된 것은 아니다. 반환 소스가 고정 commit과 같아도 실제 배포 버전·설정·시험 자격을 증명하지 않는다.

## 제공자가 보고한 결과와 이번 재검증의 구분

| 제공 문서의 보고 | 이번 판정 |
|---|---|
| Node322/실패0, Python21, 별도 모델87검사 및 검토 PASS | RECEIPT/VALIDATION의 보고값. 이번에 테스트를 실행하거나 원로그를 재검증하지 않음 |
| fresh staging Node24.1.0, offline npm 설치56개, syntax 통과 | VALIDATION의 보고. 설치·syntax 실행0. 기재된 install/check/test/dry-run 로그 SHA는 있으나 ZIP에 `.log` 파일 자체가 없음 |
| 우리 과거 v4.2 내부21파일, 예제5·schema2·빌드4핀, gateway 원문5개 PUT/GET10요청 | 과거 수신품에 대한 팀원 보고. 해당 비공개 입력·화면은 이번 반환팩에 없음 |
| Windows static mock15항목 PASS | 별도 합성 화면 보고. 실제 Worker와 화면의 합친 실호스트 검증으로 바꾸지 않음 |
| native Windows offline 실패 `fcntl` 부재, WSL 미가동 | 실패/미실행 구분 유지. 정적 검사 통과를 Unix runner 실행 성공으로 표현하지 않음 |
| 사용자 초안의 실제 Worker+화면 loopback22항목·105요청 | 이번 ZIP의 top-level receipt에는 해당 통합 실행 receipt가 없음. 전달된 초안의 별도 주장으로 두며 독립 재검증하지 않음. 여러 키/큰 예제의 loopback을 한 키 실연결 예산 증거로 대체하지 않음 |

README의 재현 명령은 검토 대상 텍스트로만 읽었다. VALIDATION은 publisher dry-run을 시도하지 않았다고 기록하므로 dry-run 로그 해시가 있다는 이유로 실행 성공을 추정하지 않는다. 이 ZIP에는 Python 소스도 없으며 제공자의 Python21 보고를 새 반환 소스의 전체 실행 증거로 합산하지 않는다.

## 기존 감사 대비 해결·유지·새 증거

**새로 확보한 증거는 실제 반환 바이트와99개 source 정본 일치**다. source5eff나 요청·회수 쟁점 자체를 신규 발견으로 세지 않는다.

- **건강 계약 반영 확인:** 첨부 `MONITORING.md`와 `monitor.mjs`는 선택 `health_contract` 및 운영자가 명시한 `active_game_ids`를 사용한다. 최상위 `published_at`과 활성 행의 `checked_at/feed_ok`를 검사하고 `file_sequence <= sequence_checked <= health.sequence`를 허용한다. 파일이 안 바뀌어 `file_sequence`가 작은 것은 정상이다. 종료 경기는 운영자가 목록에서 제외해야 하며 자동 탐색은 없다. 누락·철회(null)는 unconfirmed/partial로 유지하고 기존 장애를 자동 종료하지 않는다. 설정 변경은 기존 장애 이력을 보존한 새 state를 요구한다. 이는 소스·문서의 정적 일치이며 실제 운영 감시가 작동했다는 검증은 아니다.
- **추천 의미 보존 확인:** `tests/receipt-reason.test.mjs`는 `held_last_accepted`와 미래 reason의 원문을 유지하고 `rec.pairing.status=none`을 보존하도록 작성됐다. 테스트 실행은 하지 않았다. 투구/관측 존재를 새 추천 존재로 바꾸지 않으며 `next.reference` 후속팩까지 검증한 것으로 확장하지 않는다.
- **우리 v4의 과거 수리와 구분:** [CX15](cx-15-current-v4-snapshot-regressions-2026-10-18.md)는 당시 고정 v4 snapshot에서 필수 증거 누락의 INCOMPLETE, 입력 후 창 재확인, HTML403 거부를 합성 회귀로 확인했다. 반환팩을 읽었다고 이 이미 다룬 CX01 F2–F4를 새 결함으로 반복하지 않는다. [CX19](cx-19-onekey-isolated-candidate-2026-10-19.md)의 NaN/후속 target 수정 후보14PASS도 후보만의 결과이며 이번에 원본 적용 상태나 정식 registration/review 쌍을 재검증하지 않았다.

### 여전히 공동 실행표에서 정할 핵심

| 항목 | 실제 첨부와 기존 CX01을 함께 읽은 판단 |
|---|---|
| 18/120과 전체 요청 원장 | README/RESULTS의 runner17+login1+발행회수HEAD1=19는 **관람회수GET1을 뺀 부분합**이다. CX01의20은 그것까지 넣은 부분합이며 전체 총량 합의가 아니다. scoped probe 포함21, expiry22, upload_off23, 사전/관리/추가 확인 요청은 별도다. 우리 등록의 본창18+후속최대3도 공동 전체 요청 수가 아니다. 18→20 자동 승인으로 바꾸지 않음 |
| 공유 rate limiter | `UPLOAD_LIMITS.md`와 gateway는 인증된 PUT/DELETE/비공개HEAD가 같은 `publisher` key를 쓴다. 정상 업로드·health 갱신·조건 실패412 등이 공유 한도를 소비한다. 공개 GET/HEAD는 이 발행 limiter를 쓰지 않으며 초기 인증·범위·만료 거부는 limiter 전단이다. 다만 limiter 대기 후 만료 재검사에서403이 나면 이미 한도를 소비했을 수 있다. 모든 HTTP 요청 원장과 limiter 소비를 구별해야 함 |
| 영구 자격 회수 | `PUBLISH_SCOPE.md`는 OFF→digest 제거/교체를 안내한다. 만료403이나 upload_off503만으로 영구 회수 증명이 되지 않으며 같은 digest로 범위를 풀면 옛 bearer가 되살아날 수 있다고 명시한다. gateway는 OFF를 인증 전503으로 반환하므로 CX01의 정상401 확인 순서와 다른 증거 기준 문제는 해소됐다고 볼 수 없음 |
| 실제 시험 준비 | 창·단일 writer·배포/source/scope·새 자격·정상/긴급 회수 순서·발행 및 관람 회수 담당·비공개 연락 경로·전체 서비스/관리 요청 원장·후속 시간의 양쪽 확정이 남음. 수치 초안만으로 실제 HTTPS/실경기/운영 GO를 선언하지 않음 |

우리와 팀원의 종료 순서 차이는 기존 CX01 F1/F5의 후속 합의 항목이다. 401을 얻기 위해 긴급 OFF를 임의로 되돌리라는 제안은 하지 않는다. 같은 토큰 거부, 실제 설정/version 증거, 관람 세션 회수는 서로 다른 확인 대상으로 남는다.

## 보존 증거와 종료 범위

전용 출력 식별자: `20261019-codex-return-pack/CX37`. 안전 추출된 파일과 다음 정적 evidence를 보존했다.

| 파일 | SHA-256 |
|---|---|
| `extraction-receipt.json` | `374af2d1f2d58a5ef7f4b6d4222c34d0526d0e7595c02a0c5ec8084248d4dc9e` |
| `static-verification.json` | `1fec68647ea231f2aa8b218248a1a697d54b61be1f55c7efcdbf4b05cea325be` |
| `source-tree-5eff.json` | `8f352c0e3f37652b46a64512766f6d8840f1a79774bab6fc187ea83bc061a582` |
| `git-blob-verification.json` | `7829a5ed7679b3ba5626f8d2232f9a7c1c8fedbb647759dd3685f02f03ae3b99` |

`static-verification.json`은 로컬 객체가 없던 초기 단계이고, 후속 `git-blob-verification.json`이 허용된 원격 고정 tree 대조를 보완한다. 초기 기록을 덮어쓰지 않았다.

repo 변경은 이 보고서1개뿐이다. 첨부 코드 실행/import·설치·시험·브라우저·실호스트 통신·운영/제품 변경·자격 열람은 모두0이다. 첨부 검증 과정의 알림·회신 발신은0이며 공동큐 점유·완료 기록은 별도다. 네트워크는 허용된 GitHub 고정 tree 읽기만 수행했다(작성자5eff 1회, 독립 검토자48ec 1회). 사용자 답장 초안은 발신하지 않았고 작성자는 stage/commit하지 않았다.
