# CX-02 — 출시 후보 검증 근거표

2026-10-18 KST · 작성: Codex native `queue_setup` · [CLAIM](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6248452476).
**정적 근거 대조 완료. 실제 시험 GO·최종 통합 PASS가 아니다.** 아래 PASS/숫자는 기존 검토 보고이며 이번 재실행은 0회다. 공개 E2E의 과거 자산 고정과 최종 후보 검증 공백을 먼저 해소해야 한다.
읽은 범위는 고정 소스·테스트 설정·handoff/runbook·지정 REPORT/REVIEW/build 메타데이터다. 모델/raw/실 fixture payload·자격·private 파일은 열지 않았고 테스트·호스트·운영 실행 및 광범위 해시는 하지 않았다. graphify query는 탐색 보조만 사용했다.

## 기준과 버전 연결

소스 기준 `dccde711748d40565a78c50356cf8cb5c4f43d3e`(이하 D), 감사 브랜치 `codex/ws-preflight-audits-1018`. 아래 저장소 줄은 D 기준이다. `J/`는 PM job `20261017-ws-launch-prep/`의 상대 식별자이고 공개 접속/개인 경로가 아니다.

| 계층 | 고정 근거 | 확인 범위·남은 차이 |
|---|---|---|
| 출시 소스 D | `docs/pm/integration/handoffs/2026-10-17-ws-launch-prep.md:61` | B/C/D/E 뒤 참고 카드·운영 실행기·알림을 포함한 후보. 앞부분의 `a9d52c4`는 이전 시점이며 최종 D와 구분 |
| v4.3 / v4.3.1 팩 소스 | `J/I-partner-pack/REVIEW-v4.3-001.json:10,15,18` | `6282c63bd699d9cdf5d7110fff2ce530b7221c4f`의 source13/13·public4/4 및 검토자 public rebuild 동치 보고. D와 이 커밋의 `git diff --name-only`는 handoff 문서 1개만 반환: 소스/테스트/빌드 설정 변화 없음. 빌드 재실행 증거는 이 차이 확인과 별개 |
| v4.3.1 내용 | `J/I-partner-pack/REVIEW-v4.3.1-001.json:3,8,14–20,32` | 문서 delta PASS, v4.3 대비 source/public 자산 동일 보고, offline22/public4. ZIP `df2cf89420c45f019657a6d1ed3977bda5b4106c2874cddd9a78f047a7ed3014`; 외부 전송 false. 따라서 재포장 준비 완료 근거는 있으나 팀원 수신/적용은 미확인 |
| 스키마 | `J/I-partner-pack/REVIEW-v4.3-001.json:27–36` | 공개 v1의 `next.reference` 선택 필드·키 whitelist·엄격 이전 cutoff·기본 broad OFF 검토 보고. `reference_card_integrity`는 실시간 write gate가 아니라 재생 감사임(v4.3.1 검토:22) |
| 팀원 현재 후보 | [Stage5@724888c](https://github.com/Pitcheezy/transition-models/blob/724888caaa65414db030260ffdcc0cae2aff2135/docs/WS_STAGE5_READINESS_20261018.md#L18) | host `5effc7a4deef21ee37d07f8574b43b23935d2189` + 받은 v4.2 화면 출처 `57f5d3f` 보고. `724888c`는 보고/main pin이며 host runtime pin과 다름 |
| 후보 vs 배포 | 같은 Stage5:21–30,51,103 | 화면 출처는 공급자 보고·manifest 동치이며 팀원의 소스 rebuild 아님. deployable=false·합성 설정·실제 배포/version 미확인. v4.3.1과 최종 UI 변경 뒤 자산을 다시 연결해야 함 |

v4.2 수신·해시 일치는 [팀원 보고](https://github.com/Pitcheezy/transition-models/issues/3#issuecomment-6247347962)와 Claude [정정](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6248393581)으로 확인한 **보고 상태**다. v4.3.1 수신·실배포 완료로 넓히지 않는다.

## 기존 검사 근거 — 서로 합산하지 않음

| 대상·실제 기준 | 기존 기록 | 해석 한계 |
|---|---|---|
| B/C 재생, 각각 작업 커밋 | handoff:21–28 | 한 경기 오프라인 전후 비교·불변식 PASS 보고. 최종 전체 테스트·실경기/실호스트 근거가 아님 |
| E 웹, base383cd31 + 0fbbfc7/5cdeeff | `J/E-static/REVIEW-001.json:6–26` | public/default build, static385/0·viewer105/0·boxscore6/0·responsive4/4·host-errors28 보고. compact4+studio3 실패를 남긴 PASS이며 “모든 UI 통과” 아님 |
| J 참고 카드, 0c2ef3c | `J/J-reference/REVIEW-002.json:7–17,31,38` | subset164 PASS+기존 baseline 실패1 보고, extras A2 layout22 실패 이력 유지. handoff:63의 재생47/0·audit0은 별도 검증이며 테스트 개수에 합산하지 않음 |
| 운영 delta, c8b849a/base d0fff79 | `J/O-ops/REVIEW-005.json:5–10,36–46` | 가짜/격리 운영 조건의 3 suites108 PASS 2회 보고. 두 번을216으로 더하지 않으며 실경기·폰 전송·launchd 설치 검증 아님. 잔여 비차단 지적 존재 |
| 최신 운영 묶음 보고 | handoff:61 | “110 tests pass”는 범위가 다른 각 부분 검토 이후 보고. D에서 전체 backend/web를 한 번에 실행한 증거로 사용하지 않음; 위108·344 등과 합산 금지 |
| 전달팩 subset, 6282c63 | `J/I-partner-pack/REVIEW-v4.3-001.json:18–24` | archive snapshot 회귀344/0, PR42 loopback4 및 public rebuild 보고. ops 도구는 팩에서 제외(:17); 전체 통합/실배포 근거가 아님 |
| 팀원 Stage5, 고정host+v4.2 | Stage5:32–44,105; [완료 보고](https://github.com/Pitcheezy/transition-models/issues/3#issuecomment-6247601845) | Windows/Edge loopback22항목·105요청, 관련 Python209 보고. 여러 키·큰 합성자료이고 한키2KiB 시험·실기기·실클라우드와 별개 |

## 현재 공백과 근거

1. **최종 후보 전체 검사 미확인.** handoff:33,39–42는 이전 통합 끝의 전체 재실행·공개 E2E·실호스트/실기기를 미확인으로 남긴다. D까지의 지정 검토 메타데이터에서도 이를 닫는 하나의 최종 후보 영수증은 찾지 못했다. “없다”가 아니라 이번 확인 범위에서 미확인이다.
2. **공개 E2E를 그대로 실행해도 최종 화면 검증이 아닐 수 있다.** `apps/observer/backend/tests/test_public_viewer_e2e.py:11,55`의 BUILD는 과거 10/16 job 자산을 고정한다. `test_public_viewer_gateway_e2e.py:11,14–29`는 이를 가져오고 과거 gateway fixture/PINS를 사용한다. 최신 D/최종 public-build/실제 배포 후보와의 pin 연결을 먼저 고쳐야 한다. 해당 private fixture는 열지 않았으며 테스트 실행도 하지 않았다.
3. **기본 pytest 명령 범위 주의.** `pyproject.toml:33–34`는 `testpaths=["tests"]`; observer backend/ops가 자동 포함됐다고 볼 수 없다. `apps/observer/README.md:249–250,334`에 backend와 ops의 별도 경로가 있다. `apps/observer/web/package.json:8–9`의 일반/public build도 별개다. 정확한 대상·명령·종료코드·수집 테스트 수를 최종 영수증에 고정해야 한다.
4. **과거 7개 레이아웃 실패의 현행 수정 근거 미확인.** E 검토:17–18,26,58 및 handoff:32,51에서 390×844 투구 띠가 탭바에 가려지는 7건을 유지한다. 이후 참고 카드가 추가돼 같은 문제로 충분하다고 단정할 수 없다. J의 별도22건 이력(:38)도 동일 실패로 임의 합치거나 해소됐다고 쓰지 않는다. SONG-03의 최종 화면·같은 assertion 결과가 필요하다.
5. **실클라우드/철회·회수·복구 공백.** Stage5:43은 새로고침 후 철회만 검증하고 자동철회 지연·관리자 회수/만료·CDN/TLS·Song runner·실기기를 미측정으로 적는다. stage5 loopback PASS를 실제 배포 결과로 옮기지 않는다. CX-01의 요청·회수 합의 조건도 GATE-01에 남는다.
6. **운영 근거와 안내 시점 불일치.** runbook:3은 a9d52c4, :22는 운영 실행기 “검증 전”인 반면 :225,228–242는 가짜 통합 시험·실경기 미검증을 기록한다. 안내는 부분적으로 갱신됐고 최종 release pin·실제 프로필/알림/실행 환경까지 검증됐다는 증거는 아니다. 실제 담당자가 최종 runbook을 고정해야 한다.

## 작은 후속과 완료 조건 — PLAN 제안, 직접 실행 안 함

| 우선 | 작업·기존 소유자 | 완료 조건 |
|---|---|---|
| P1 | 공개 E2E 자산 pin 갱신 — Claude CL-03 통합 소유자 | 최종 고정 public-build와 gateway source/설정 pin을 명시하고 과거 절대경로 자산으로 잘못 통과하지 않는 실행 입력 확인. 변경 범위 등록·별도 검토 후 생성자료 E2E 결과 연결 |
| P1 | 최종 검사 영수증 — Claude 통합/CX-02 후속 | 고정 commit·source diff·runtime·backend/ops/web 대상·정확한 명령·결과/잔여 실패와 검토자를 한 표에 기록. 필요한 검사만 재실행하고 실자료 CLI는 별도 게이트로 분리 |
| P1 | 전달팩→뷰어 배포 바인딩 — CL-02/GATE-04 | v4.3.1 수신 영수증 후 SONG-03 최종 UI의 재빌드/팩·스키마·실제 배포 version을 연결. 배포 권한·담당 별도 확정 |
| P1 | 실제 연결·실기기·철회/복구 — GATE-01/02/05 | CX-01 회수/요청 합의 후 허용된 실제 경로의 근거, 최신 폰 Safari의 로그인/갱신/끊김/철회/복구 확인. 새로고침과 자동 반영을 분리 |
| P2 | 390×844/참고 카드·알림 확인 — SONG-03/02·CL-03 | 기존7 및 별도22 이력의 케이스를 구분해 최종 UI 판정, 프로필·폰 알림·운영 검증 범위와 미측정 항목 명시 |

**검토 상태:** 작성자와 다른 native `coordination_review` 정적 검토 PASS·root 대표 수용. 동일 모델의 별도 에이전트 검토이며 테스트 재실행은 0회다. 본문은 운영 GO나 다른 소유자의 코드 변경 승인이 아니다. 등록·독립 검토·기존 lock/실자료/운영 게이트와 CL-01~03·RS-01 점유를 유지한다.
