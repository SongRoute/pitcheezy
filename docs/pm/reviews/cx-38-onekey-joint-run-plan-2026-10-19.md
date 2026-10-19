# CX38 — 한 키 공동 실행표와 미발신 회신

- 작성: Codex implementation_1019, 2026-10-19 KST. 검토: queue_1019 / Root.
- 상태: 작성·독립 정적 검토 완료, Root 수용. **양측 합의 전 미발신 초안이며 실행 승인·원본 수정·발신은 하지 않았다.**
- 선점: [CX38 CLAIM6257870258](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6257870258). 기준 commit `beca1163978bf1d6a4078a57827d970e9071b867`.

## 권장안과 현재 준비 상태

**scoped 한 키 시험을 기준으로 정상 종료는 업로드 ON 유지 → 유효한 다른 토큰 digest로 교체 → 같은 옛 bearer의 gateway401 확인 → OFF를 제안한다.** 팀원의 OFF 우선안과 아직 합의되지 않았다. 현재 wrapper는 회수401을 필수로 요구하므로 OFF 우선503을 같은 성공으로 바꾸지 않는다. 다른 판정을 채택하려면 CL02 소유자의 등록·판정 변경과 독립 검토가 먼저 필요하다. 긴급 중단은 즉시 OFF가 우선이며 증거를 얻으려고 다시 켜지 않는다.

이 문서는 기존 GATE-06 공동 원장·회수 순서를 채울 초안이다. 새 출시 게이트를 추가하지 않는다. [CX01 감사](cx-01-onekey-audit-2026-10-18.md), [CX37 실제 반환팩 검증](cx-37-partner-return-pack-review-2026-10-19.md)의 상세는 반복하지 않는다.

2026-10-19 읽기 전용 재확인 기준은 다음과 같다. `L-onekey`는 기존 전용 준비 폴더이며 비공개 preconditions·자격·실행 로그는 열지 않았다.

| 현재 파일 | SHA-256 / 확인 |
|---|---|
| `L-onekey/REGISTRATION-v4.json` | `91a872902e63ed49b74d7a7d27fd5424c4fdcb4cd1134e1842037fb139f081be`, `execute:false` |
| `L-onekey/onekey_wrapper.py` | `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af`, 등록의 wrapper pin과 일치 |
| `L-onekey/OPERATOR-CHECKLIST.md` | `6e07e2d53ff40be8aadb0d9c69f13c02323fbaf7fe19ccdc42359eaf5b8af4a6` |

[CX19 격리 후보](cx-19-onekey-isolated-candidate-2026-10-19.md)의 NaN 거부·후속 target 결합 두 수정은 합성14검사 PASS였으나 원본에 적용되지 않았다. 후보를 정식 실행본으로 간주하지 않는다. 실제 사용할 source/등록/독립 검토 쌍의 확정 증거는 이번에 새로 확보하지 않았으며 대기한다. 실제 preconditions는 그 확정본과 양측 합의가 일치할 때만 채운다.

## 실행 단계와 시간표 — 모두 제안, 빈칸 확정 후 사용

| 단계 | 담당 제안 | 시간 / 시작 조건 | 완료 근거와 다음 행동 |
|---|---|---|---|
| 0. 공동 확정 | Song + 팀원 운영자(TBD), CL02 등록 소유자 | 시험 날짜·KST 및 UTC(ms) 시작/끝 TBD; 사람 응답 기한 TBD | 아래 원장·정상/긴급 회수·배포5파일 raw/LF pin·단일 writer·scope/expiry·연락 경로·정식 검토 쌍 확인. 해당 창 Song GO 전 실행 대기 |
| 1. 준비 / 로그인 | 팀원 설정 담당(TBD), Song | 본창 밖 준비 마감 TBD | 새 미사용 합성 키/범위 밖 확인용 인접 키, 전용 bearer·viewer cookie, uploads/data ON·notice OFF 확인. 비밀은 사람의 로컬 터미널/비공개 경로로만 취급 |
| 2. 본 시험 | Song, 팀원 감시·차단 담당(TBD) | 승인 시작/끝 창 안에서 runner 경과<120초; 첫 요청 및 runner 직전 창 재확인. 실행 시작 시 승인창 끝·scope 만료까지 각각 최소180초 | scope HEAD1 후 고정 runner17. 본 시험 PASS는 후속 완료와 다름 |
| 3. 만료 확인(선택) | Song + 팀원 | expiry+10초, digest 교체 전. 자동 기다림은 계산된 대기≤900초일 때만; 그 밖 후속 시각 TBD | gateway403/JSON·비차단 응답. 선택해 필수 증거로 등록했으면 미확보 상태는 INCOMPLETE |
| 4. 발행 회수 | 팀원 digest 담당(TBD), Song 확인 | 본창 밖; 요청·ACK 기한 TBD | ON 상태에서 유효한 다른 digest로 교체한 ACK → 같은 옛 bearer HEAD401 → OFF ACK. 응답 기한/예산 초과면 추가 요청 없이 중단 처리 |
| 5. 관람 회수 / 종료 | 팀원 access 담당(TBD), Song 확인 | 후속 종료 시각 TBD | 시험용 access digest 회수 및 같은 옛 cookie 검증, 설정/version 근거·요청 원장·잔여 tombstone·미확인 항목을 함께 인계 |

runner 경과120초 미만 기준은 승인 시작/끝 창 자체를120초로 정하라는 뜻이 아니며 전체 공동 작업의 종료 시간 보장도 아니다. 현행 `check_window`는 승인창 끝과 scoped 만료까지 각각180초 이상 남아야 한다. 등록에는 runner110초, supervisor114초, wrapper runner 대기125초, 작업별 timeout5초가 따로 있다. 만료 대기·사람 ACK·후속 확인은 별도이며 합의 창/후속 마감이 부족하면 시작하지 않는다. 이 시간들을 더해 임의의 전체 SLA를 만들지 않는다.

## 전체 요청 원장

아래는 **예정 시도 수**다. 실제 원장에는 시각·주체·method·결과/미응답·요청 식별자·누적 본창/후속/외부 수를 따로 채운다. 모든 실패·timeout도 시도로 센다. 비밀·호스트·키 값은 공개 문서에 쓰지 않는다. P는 발행 limiter 공유 key, 외부는 wrapper 예산 밖이라는 뜻이다.

| 단계 | 담당 제안 | method / 예정 횟수 | 구분 | limiter | 기대 결과 | 보존 증거 | 중단 조건 |
|---|---|---|---|---|---|---|---|
| 전용 관람 로그인 | Song / 팀원 access | POST1 | 외부 | P 아님; access 정책 별도 | 합의된 로그인 성공·유효 cookie | 비밀 제외 상태/시각 | 로그인 실패·추가 redirect/요청 미계상 |
| 범위 밖 확인 | Song wrapper | HEAD1 | 본창 | 초기 scope 거부는 P 전단 | gateway403, JSON, cf-mitigated 없음 | wrapper probe | HTML/WAF403 포함 기대 불일치면 쓰기 전 중단 |
| 최초 확인·생성·읽기 | Song runner/자식 worker | HEAD3 + PUT1 + GET1 =5 | 본창 | 비공개 HEAD/PUT P, 공개 GET 제외 | 최초/생성 전404 → 저장200·E1·SHA 일치 → GET200 | runner receipt + 가능하면 팀원 상태 로그 | 기존 객체/ETag·불일치·오류 |
| 첫 삭제표식 | 같은 단일 writer | HEAD2 + DELETE1 =3 | 본창 | P | E2, deleted=true | 같은 receipt | 조건/응답 불일치 |
| 복구·읽기 | 같은 단일 writer | HEAD2 + PUT1 + GET1 =4 | 본창 | 비공개 HEAD/PUT P, 공개 GET 제외 | E3≠E1, 동일 합성 본문·GET200 | 같은 receipt | ETag/SHA 불일치 |
| 의도한 낡은 삭제 | Song runner | DELETE1 + HEAD1 =2 | 본창 | P, 기대412도 소비 | 이 DELETE만412, E3 보존 | 같은 receipt | 다른412 또는 상태 변경 |
| 최종 삭제표식 | 같은 단일 writer | HEAD2 + DELETE1 =3 | 본창 | P | E4·deleted=true, worker 종료·pending0 | 같은 receipt/정리 상태 | 불명확한 결과·잔여 worker/pending |
| 만료 확인(선택) | Song wrapper | HEAD1 또는0 | 후속 | 초기 만료 거부 P 전단; limiter 대기 후 만료 재검사는 소비 가능 | gateway403 upload_expired | probe + 만료 시각 | 미응답/불일치, 회수 후 검사하려는 경우 |
| 옛 발행 자격 회수 | Song wrapper | HEAD1 필수 | 후속 | 초기 인증 거부 P 전단 | 같은 옛 bearer gateway401 | same-token 근거·probe·교체 ACK | 503/403/2xx 등은 회수 증거 아님 |
| OFF 기록(선택) | Song wrapper | HEAD1 또는0 | 후속 | OFF503은 P 전단 | 503 기록, 회수 증거 아님 | probe + OFF ACK | 추가 시도 예산 없음 |
| 옛 관람 세션 회수 확인 | Song | GET1 | 외부 | P 아님 | 같은 옛 cookie·같은 보호 JSON의 접근 거부; 아래 판정 적용 | 상태·설정/version·전용 digest 제거 근거 | 다른 cookie/경로, 가림 응답만으로 판정 |
| 팀원 측 옛 발행 토큰 거부 확인 | 팀원 운영자(TBD) | HEAD 등 method/횟수 TBD | 외부 | 인증 결과/진입 지점에 따름 | 등록의 팀원 확인 항목; 우리 HEAD와 중복 여부 합의 | 주체·방법·상태 또는 공유 관측 근거 | 별도 전송 여부·상한 미확정 |
| 배포/설정 조회·변경, digest/access 회수, scope 정리 | 팀원 운영자(TBD) | 관리 API/콘솔 method·횟수 TBD | 외부 관리 | P와 별도 관리 제한 | ACK·실제 적용 version/설정 | 비밀 제외 변경 기록 | 미확정 호출/반영 불명 |
| notice/data/긴급 스위치 확인·조작 | 팀원 운영자(TBD) | 관리 요청·서비스 확인 method/횟수 TBD | 외부 | 진입 경로별 구분 | 계획 상태 또는 긴급 차단 | 스위치/version·시각 | 안전 상태 미확인 |
| 화면·asset·자동 polling/monitor·추가 상태 조회 | 양측 담당 TBD | GET/HEAD 등 횟수·상한 TBD | 외부 | 공개 읽기는 P 제외; 발행 요청이면 P 공유 | 불필요 화면/폴링은 시험 중 사용하지 않는 안 제안 | 사용 여부·요청별 원장 | 미계상 병행 트래픽 |
| 실패 정리·추가 확인·물리 삭제 | 양측 담당 TBD | 기본 추가 실행0; 필요 method/상한 TBD | 외부 또는 승인된 후속 잔여분 | 요청 종류별 구분 | 멈춘 상태 먼저 대조; 자동 재시도/DELETE 없음 | 불명확 상태·미사용 잔여 예산 | 새 합의 없이 재전송·키 재사용 |

runner는 **HEAD10+PUT2+DELETE3+GET2=17**, scope를 포함하면 본창18이다. 최초 HEAD와 최종 HEAD는 위17에 포함돼 별도로 더하지 않는다. 정상 runner의 발행 limiter 대상은 비공개 HEAD10+PUT2+DELETE3=15이며 HTTP 총량18과 다르다. 일반 발행·health 업로드 등 같은 limiter를 쓰는 다른 작업도 양측이 확인해야 한다. 전역 단일 정확 상한으로 취급하지 않는다.

**scoped 본창18 + 필수 발행 회수1 + 로그인1 + 관람 회수1 =21은 알려진 부분합**이다. expiry를 더하면22, OFF 기록까지 더하면23이다. 등록의 본창18+후속최대3=21은 wrapper 범위만의 계획값이며 위 외부/관리 요청은 포함하지 않는다. 팀원 확인·관리·asset·polling·실패 처리의 방법/수/상한이 TBD인 동안 **공동 전체 상한도 TBD**다. 이를18→20 또는21 승인으로 읽지 않는다.

후속 최대3은 종류 수가 아니라 시도 수다. deferred out-of-scope나 실패 후 재확인도 후속 슬롯을 소모할 수 있어 선택한 expiry/revoked/off와 별도로 무제한 반복할 수 없다. 이번 안은 자동 재시도0이며 추가 시도는 남은 슬롯·시각·양측 합의를 확인한 뒤에만 별도 기록한다. 필수 회수 슬롯을 남기지 못하면 완료를 선언하지 않는다.

runner 영수증이 없으면 실제 전송 수는 **UNKNOWN**이다. wrapper의 보수 가산18을 전송0 또는 측정18로 쓰지 않는다. scope1 후 timeout/영수증 누락 또는 runner18번째 슬롯 사용이면 계상19가 될 수 있으므로 wrapper가 모든 전송의18회 hard cap을 자동 보장한다고 주장하지 않는다. 429·예상 밖412·5xx·timeout·미확인 원격 상태에서 멈추고 요청/정리 상태를 대조한다. 위 의도적412와 등록된 부정 probe 응답만 예외다.

## 정상 회수와 긴급 회수의 판정

- **정상 발행 회수:** uploads ON에서 `PUBLISH_TOKEN_SHA256`을 **유효한 다른 digest**로 교체한다. 빈 값/삭제는 gateway503이 될 수 있어401안과 같지 않다. 같은 옛 bearer의 gateway401과 교체 ACK를 남긴 뒤 OFF한다. 만료403·OFF503은 영구 회수 대체 증거가 아니다. scope/expiry를 해제할 때 옛 digest가 남아 다시 유효해지지 않도록 설정/version을 함께 확인한다.
- **관람 회수:** 시험 전용 access digest 제거 후 **로그아웃·재로그인으로 바꾸지 않은 같은 옛 cookie**와 앞서 접근한 같은 보호 JSON을 사용한다. cookie의 만료 시각을 기록하고 유효기간 안에 검사한다. 자연 만료로 인한401과 구분할 수 없으면 회수 응답 증거는 미확인이다. 다른 유효 코드가 남고 gate ON이면 옛 세션401을 기대할 수 있다. 모든 코드가 없어503인 경우503만으로 완료 판정하지 않고 전용 digest 제거·설정/version을 함께 대조한다. notice/data OFF 등 앞단 응답이 인증을 가리면 미확인으로 남긴다. 새 전용 코드나 추가 계정을 이 문서만으로 만들지 않는다.
- **긴급/실패:** 팀원은 즉시 OFF, Song은 기존 운영자 중단 절차를 따른다.401을 얻기 위해 재ON하지 않는다. 원래 로그·소진 키·마지막 확인 상태를 보존하고 자동 재시도/추가 DELETE를 하지 않는다. 최종 tombstone은 물리 삭제가 아니며 별도 합의 없는 정리는 하지 않는다. 발행 회수와 관람 회수 미확인을 각각 기록한다.
- **완료:** wrapper MAIN_PASS와 공동 완료를 구별한다. 필수 증거가 없으면 INCOMPLETE/FAIL을 유지한다. 원장 대조·관람 회수·팀원 설정 확인은 wrapper PASS만으로 대신하지 않는다. 독립 checker의 일관성 검사도 실제 대상/배포 검증을 대신하지 않는다.

## 양측이 채울 결정 칸

| 결정 | 값 / 현재 상태 |
|---|---|
| 시험 run ID·KST 및 UTC(ms) 시작/끝·만료·후속 종료·각 ACK 제한 | TBD, 모든 사람 일정은 KST와 UTC(ms)를 병기해 비공개 실행표에 기입 |
| Song 실행 / 팀원 배포·감시·digest·access·긴급 차단 담당 | Song 제안, 팀원 실명/역할 TBD |
| 단일 writer·scoped 모드·expiry/OFF 옵션·외부 트래픽 중지/포함 | TBD; hostwide bearer 자동 수용 없음 |
| 전체 서비스/관리 요청 수·별도 실패 예산·공유 limiter 여유 | TBD; 미지 항목을0으로 간주하지 않음 |
| 정상 ON→다른 digest→옛401→OFF 동의 | 양측 ACK 대기; OFF 우선 유지 시 CL02 등록/판정 변경·독립 검토 전 실행 대기 |
| 관람 회수 방식·503/가림 때 근거와 미확인 처리·tombstone 유지 | TBD |
| 실제 배포5핀·정식 실행 source/등록/검토 쌍·preconditions·해당 창 GO | 새 완료 증거 없음, 현행 절차대로 대기 |
| 비공개 자격 전달·연락/중단 채널 | TBD; 이 문서/공개 이슈에 값 기재하지 않음 |

## Song이 팀원에게 보낼 회신 초안 — 미발신

반환팩 확인했습니다. 이번 ZIP은354208바이트이고 SHA-256 `ec24db14ba36610618b307cb2840bf9080c45d94fe6e3a68064878dc74b13164`와 일치했습니다. 소스99파일은5effc7a4 고정 Git tree와, 런타임5파일은48ec4065 고정 tree와 각각 Git blob으로 대조해 모두 일치했습니다. 앞서 받으신 우리 팩181953바이트와 이번 반환팩은 별개로 구분했습니다. 실제 배포나 공동 실호스트 시험까지 확인된 것은 아닙니다.

다음은 한 키 공동시험의 실행표를 맞추면 좋겠습니다. runner17 안에 최초/마지막 HEAD가 이미 들어 있고 scoped 확인까지 본창18입니다. 필수 발행 회수1·로그인1·관람 회수1을 더한21도 부분합이며, 선택 만료/OFF 확인과 팀원 측 확인·관리/화면/폴링 요청은 따로 계상해야 합니다. runner 경과120초 미만과 승인 시간대, 만료 대기·사람 응답·후속 종료 시간도 나누겠습니다. 시작할 때 승인창 끝과 scope 만료까지 각각180초 이상 남아야 하므로 승인창을 단순히120초로 잡지는 않겠습니다. 날짜·시간은 KST와 UTC(ms)를 함께 적고, 양측 담당·전체 요청 상한·실패 때 추가 요청 여부도 확정 부탁드립니다.

정상 종료는 업로드를 켠 상태에서 유효한 다른 발행 토큰 digest로 교체하고, 우리가 같은 옛 토큰으로401을 확인한 뒤 OFF하는 안을 제안드립니다. 현재 우리 wrapper는401을 필수 회수 증거로 봅니다. OFF 우선안을 유지한다면 해당 등록·판정의 변경과 검토를 먼저 맞춰야 합니다. 긴급 상황에서는 즉시 OFF하고401 확인 때문에 다시 켜지 않겠습니다. 관람 자격도 별도로 회수한 뒤 같은 옛 쿠키/보호 JSON으로 확인하고,503이나 앞단 공지에 가려지면 설정 근거와 함께 미확인 여부를 남기겠습니다.

결과 설명에서 두 가지만 확인 부탁드립니다. VALIDATION의 `publisher_dry_run_attempted=false`와 dry-run 로그 해시가 서로 다른 검사 대상을 뜻하는지 구분해 주시면 좋겠습니다. 별도로 말씀하신 Worker+화면22항목/105요청은 이번 ZIP에 해당 실행 receipt가 없어, 비밀을 제외한 그 실행의 고정 receipt와 source/input pin을 주시면 보고 범위를 정확히 연결하겠습니다. Windows의 fcntl 실패/WSL 미가동과는 분리해서 기록하고 있습니다. 여러 키/큰 예제의 loopback 결과를 한 키 실연결 결과로 바꾸어 적지는 않겠습니다.

우리 쪽 두 수정 후보는 격리 합성14검사를 통과했지만 원본 적용이나 정식 실행본 확정은 아직 아닙니다. 정식 source·등록·독립 검토 쌍과 위 실행표를 맞춘 뒤 해당 시간대 진행 여부를 확정하겠습니다. 지금은 실행표와 회신을 준비한 단계이며 운영·일반 공개 실행은 시작하지 않았습니다.

---

검사 범위: 기존 문서/현재 비밀 없는 등록·wrapper·체크리스트의 정적 읽기, 문서 링크·diff 확인만 수행했다. 새 시험·브라우저·첨부 실행/import·키/비공개 로그 열람·실호스트·원본/제품 수정·외부 회신 발신0. 공동큐 점유/검토 기록은 별도이며 작성자는 stage/commit하지 않는다.
