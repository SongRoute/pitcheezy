# 2026-10-06 운영 방식 전환 기준점

2026-10-06 10:35 KST 문서 작업 전 관측. 소스·서비스·자료는 이동/삭제하지 않는다. 전환 작성자 `/root/pm_transition`은 관리 문서만 수정한다.

| 항목 | 확인/보존 기준 |
|---|---|
| 메인 | `/Users/song/Projects/pitcheezy`, `codex/ml-matrix-execution`, HEAD `a8aa091169d936cf46dcecde5598e3416ccf9da7` |
| 기존 dirty | ` M runs`만 관측. 원복·stage·commit하지 않음 |
| 오래된 cwd | `/Users/song/.codex/worktrees/aa0a/pitcheezy` detached70aa549; 최신 작업 기준 아님 |
| 현행 시연 소스 | `/Users/song/Projects/pitcheezy-worktrees/codex-ui-live-preview-1005`, `codex/demo-pa-anchors-1006`, HEAD2005702, status clean |
| 시연 제품 | 566ade5, 백엔드8a4a633. 8771 PID99202가 Tailscale100.108.252.111에서 LISTEN 직접 확인 |
| 복구 자료 | [DELIVERY](/Users/song/Projects/pitcheezy-pm-jobs/20261006-ui-astra-delivery/integrate-001/DELIVERY.md)의 source529eca2+dist+privateexactlaunch와 rollback.py; 복구 실행 자체는 미수행 |
| 수용 기준 | [ROOT-ACCEPTANCE](/Users/song/Projects/pitcheezy-pm-jobs/20261006-ui-astra-delivery/ROOT-ACCEPTANCE.md), 최초30.357초 공개/이후seek검증; 실제 두 기기 리허설 미확인 |
| 시연 묶음 | [zip](/Users/song/Projects/pitcheezy-pm-jobs/20261006-ui-astra-delivery/stage-001/pitcheezy-demo-pack.zip), SHA256 `7b4663d22b6a3f6cfc84eb0d092e3a6a1f106f6417151155da2a8489720f580d` (기존 수용 기록, 이번 전환에서 재해시 아님) |

09:46–09:48 전체 감사의 서버·GPU·DB·CV 상태는 [기존 PM-state](../handoffs/PM-state.md)에 보존한다. 과거 시점의 health/idle을 현재 실시간 관측이라고 쓰지 않는다. 이번 문서 전환에서 새fit·서버 재시작·DB변경·배포·외부전송은 수행하지 않는다.

전환 되돌리기는 해당 문서 커밋만 새 revert 커밋으로 처리할 수 있다. 작업트리 reset·dirty 삭제·제품/DB 롤백을 문서 복구와 혼동하지 않는다. 문서 전환은 전체 데이터 백업 완료를 의미하지 않는다.

## 도구 도입 단계

이번 단계: 공통 문서·큐·역할·인계 절차와 kickoff. 실제 설치된 스킬/MCP는 없음. 다음: 기존 실행기/lock 재사용 감사 → 연구/서비스/인계 절차를 실제 작업에서 시험 → 필요하면 skill-creator로 설치 가능한 스킬 작성 → 반복되는 상태 조회·인수 기록부터 MCP 범위 검토. MCP 구축을 연구/서비스 착수의 선행 조건으로 만들지 않는다.
