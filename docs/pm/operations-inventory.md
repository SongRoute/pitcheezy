# 운영 배치 인벤토리와 정리 대장

소유: 통합 문서 담당(관측/쓰기 권한은 해당 자원 소유자) · 갱신: 새 배포·자원 인수·보존/정리 결정 시 · 마지막 확인: 2026-10-06 로컬 metadata; 원격/GitHub 17:53–17:58 KST 읽기 전용 감사.

[자원 사용 규칙](resources.md) · [논리 프로젝트 지도](project-map.md) · [업무 절차](workflow.md). 아래 절대 경로는 Song의 Mac 환경 위치이며 신규 팀원에게 동일 경로 생성이나 자료 접근 권한을 부여하지 않는다.

## 확인된 로컬 배치

| 위치 | 목적/소유 | 관측·재확인 | 쓰기·보존 조건 |
|---|---|---|---|
| `/Users/song/Projects/pitcheezy` | 공통 문서 정본; 통합 배정 단일 작성자 | 10/6 HEAD `f7e8f475`, branch `codex/ml-matrix-execution`; `git status --short`, `git rev-parse HEAD` | 기존 dirty/untracked 보존; 다른 작업 diff/index 포함 금지 |
| Git 등록 worktree 전체 | 작업별 source/실험 고정/서비스 소유 혼재 | 10/6 `git worktree list --porcelain` 296건 | 오래됨/clean만으로 제거 불가; owner·runtime/source핀·미통합 commit 확인 필요 |
| `/Users/song/Projects/pitcheezy-worktrees` | 격리 작업 경로 모음, 작업별 소유 | 10/6 직속 항목 272개; 등록 worktree 총수와 분모 다름 | 읽기 목록만 확인, 개별 작성자·보존 상태 미확인 |
| `/Users/song/Projects/pitcheezy-pm-jobs` | 요청/검토/실행 증거 및 현장 전용 자료, 작업 소유자 | 10/6 디렉터리 존재 확인; [정리 증거](/Users/song/Projects/pitcheezy-pm-jobs/20261006-project-consolidation/docs/) | 봉인 report/config/hash/log 보존, private 이미지 공개 금지 |
| `/Users/song/Projects/pitcheezy-demo`, `pitcheezy-demo-sessions` | 별도 작업/시연 관련 위치 | 존재 확인; 현재 실행 소유는 별도 프로세스 감사 필요 | 이름만 보고 stale/삭제 가능 판단 금지 |
| `/Volumes/T7 Shield/pitcheezy` 및 repo `runs/` | 외부 데이터/모델/실행 산출물, 분야별 소유 | 외장 경로 존재, `runs/` symlink 확인; `readlink runs`로 대상 재확인 | 데이터 payload/봉인 OPE 이번 감사에서 열람하지 않음; 원본·sealed 보존 |

현재 관측 근거: [local-inventory.json](/Users/song/Projects/pitcheezy-pm-jobs/20261006-project-consolidation/docs/local-inventory.json), [worktree 목록](/Users/song/Projects/pitcheezy-pm-jobs/20261006-project-consolidation/docs/worktrees-readonly.txt). 개수는 용량·활성 작업 수가 아니다. GitHub/서버 관측은 다음 절의 시각별 결과로 갱신하며 수치는 예약/운영 소유권 승인을 뜻하지 않는다. 과거 8775 롤백은 [통합 상태](integration/state.md)의 당시 receipt이며 현재 재관측을 대신하지 않는다.

## GitHub·서버 1차 관측 — 17:53–17:58 KST

독립 인프라 작업자 research_audit의 [감사 REPORT](/Users/song/Projects/pitcheezy-pm-jobs/20261006-project-consolidation/infra-audit/REPORT.md)와 [observations.json](/Users/song/Projects/pitcheezy-pm-jobs/20261006-project-consolidation/infra-audit/observations.json)을 대조했다. 이 절은 현장 운영 기록이며 공개 GitHub push 대상 검토를 아직 받지 않았다.

| 대상 | 확인 사실 | 재확인·unknown·쓰기 경계 |
|---|---|---|
| 공개 `SongRoute/pitcheezy` | default main `70aa5499`(9/23), open PR #33/#34(MERGEABLE/CLEAN이나 reviewDecision 빈 값·checks[]) | GitHub repo/PR API로 재확인; 로컬 최신 개발과 자동 일치하지 않음 |
| 로컬 공통 checkout | `f7e8f47`; tracking `origin/codex/ml-matrix-execution`=`813ce61` 대비 ahead71+dirty | `git rev-list --left-right --count`의 비교 대상 명시; main 대비71이라 쓰지 않음; 자동 push/merge0 |
| GitHub 관리 | 조회 identity SongRoute ADMIN, 협업자 목록1, Actions workflow/run0, protection404/rulesets0 | 현 조회 결과 범위, CI 통과나 보호 존재를 가정하지 않음; 정책 변경0 |
| Mac 8766 | Tailnet PID3277 `pitcheezy-demo` HEAD09e12e9와 loopback PID94628 main f7e8f47 별개 | health200, Tailnet ready false/false·loopback true/true; 주소별 구별, readiness만으로 live E2E 보장 안 됨 |
| Mac 8771 | PID99202 HEAD5488c93 | health200·model/dataset false/false; runtime/served asset 구분 및 서비스 소유 인수 필요 |
| Mac 8772/8775 | 현재 listener 없음 | 8775 과거 rollback 뒤 종료 주체·시각 unknown; 시연 종료 보고와 인과 단정/자동 재시작 금지 |
| Mac 기타 | 8770 PID149,8767 PID6713,8787은 tokken 별도 | owner 미확인/타 프로젝트 보호; kill/restart0 |
| newsrv | 5×A5000 UUID 기존 문서와 일치, 각2MiB/0%, compute apps0 | 순간 관측, 예약/배정 아님; 등록·lock·새 소유 재확인 필요 |
| T7·SQLite | T7 mount; main Observer 설정 SQLite 경로348160B metadata | payload 읽기0; 실제 데이터 쓰기 소유권은 미증명 |

GitHub 정본 수렴의 다음 checkpoint는 ①PR33/34 목적·base/변경/기존 리뷰 확인 ②공개해도 되는 코드/문서와 현장 private자료 분리 ③선택 기준 commit의 깨끗한 clone에서 최소 smoke 재현 ④CI·branch 보호 제안 검토 ⑤정확한 integration diff 승인이다. 지금 어떤 branch를 정본으로 merge할지 자동 결정하지 않는다.

## 정리 대장

| 항목 | 상태 | 완료 증거 또는 결정 조건 |
|---|---|---|
| 수정 전 dirty/HEAD/대상 원문 보존 | done | docs job `baseline.json`, `baseline-status.txt`, `baseline/` |
| README START HERE·온보딩·문서 역할 연결 | done / 독립 PASS | [독립 검토](/Users/song/Projects/pitcheezy-pm-jobs/20261006-project-consolidation/docs-review/REPORT.md), 로컬 신규환경10PASS·공개main 최소 archive10PASS; 전체 앱 재현과 구분 |
| 지난 시연 일정·역사적 실행 안내 구별 | done / 독립 PASS | overview/앱 README/거버넌스 진입 안내; 원 역사 기록 삭제0 |
| 중복/낡은 worktree 물리 정리 | needs decision | 개별 owner·미통합 diff·실행/봉인 source 의존·backup 확인 후 별도 허가 |
| GitHub branch/PR/원격 정리 | needs decision | 위 공개 정본 수렴 checkpoint; push/merge/delete 이번 작업0 |
| 서버·포트·DB·GPU 정리 | observed / needs decision | 1차 관측 완료; owner 미증명 항목 보존, restart/kill/DB쓰기/fit0 |
| 현재 봉인 자료·현장 이미지·원본 데이터 | preserve | 현재 삭제/이동 범위 아님; 별도 보존 정책 없이는 그대로 유지 |

정리는 삭제 건수로 평가하지 않는다. 신규 팀원이 정본과 작업 위치를 찾고, 소유자와 검증 방법을 알며, 과거 실행을 현재 사실로 착각하지 않는지가 수용 기준이다. unknown은 확인할 항목으로 유지한다.


## 다음 정리 checkpoint — 실행 배정 전

| 항목 | 책임/실행자 | 다음 점검 산출물·완료 조건 |
|---|---|---|
| PR34와 로컬71commit 차이 | 통합 PM / 작업자 미배정 | 정확한 base/head 목록·미통합 변경 분류·공개 export/권한 검사·CI 제안; 검토 후에만 통합 방안 결정 |
| PR33의 별도 base/역사 문서 | 통합+서비스 / 작업자 미배정 | 대상 base와 현재 계약 충돌·필요한 변경만 식별, 공개할 diff와 보존할 역사 구분 |
| worktree 보존/제거 목록 | 분야 PM / 개별 소유자 확인 필요 | source/evidence/미통합diff/실행의존/보존 사유·소유자 disposition; 296개 자동 삭제0 |
| 주소별8766·preview·DB 소유 | 서비스 PM / 작업자 미배정 | runtime source·DB 설정·작성권·backup/복구 담당 연결, 종료/통합은 별도 검토 |
| 공개 배포 온보딩 재현 | 통합+서비스 / 전체 배포 검증 작업자 미배정 | 공개 main SHA일치 object의 src/최소검사 git archive는 독립10PASS([결과](/Users/song/Projects/pitcheezy-pm-jobs/20261006-project-consolidation/docs/public-main-probe-001/RESULT.json)). 향후 공개 승인 exact ref의 실제 clean clone에서 신규 문서·전체 설치 경로 검증 필요 |

다음 2~3일 진척 점검에서 배정 여부·막힘을 확인한다. 이 표는 후속 정리 준비 대장이며 push/merge/삭제/재시작을 시작하는 명령이 아니다.
