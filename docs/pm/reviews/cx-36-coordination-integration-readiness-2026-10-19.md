# CX36 — 연속 작업과 협업 문서의 통합 적합성

- 작성: Codex implementation_1019, 2026-10-19 KST.
- 상태: 독립 metadata·본문 검토 PASS, Root 수용. 실제 통합은 0회다.
- 공동 작업큐 #37: 단계1 CLAIM 6255617320, 보고서 1파일 범위의 단계2 CLAIM 6255633562.
- 이 문서는 [CX30의 당시 인계표](../integration/handoffs/2026-10-19-codex-continuous-integration.md)를 보완하는 별도 관측이다. CX30이나 원래 6문서의 이력을 다시 쓰지 않는다.

## 결론

출시 브랜치의 고정 tip `dccde711748d40565a78c50356cf8cb5c4f43d3e`는 현재 연속 작업 tip `1d2e14d8c6ca184a7833e8180b745622cfd77d72`의 조상이다. `git merge-base --is-ancestor`가 exit 0을 반환했다. 따라서 이 두 **고정 커밋의 계보만 보면 fast-forward가 가능**하며 같은 변경을 다시 복제·구현할 필요가 없다. 이는 현재 작업트리 상태나 제품 적용·통합 수용·push 승인을 뜻하지 않으며 실제 branch 이동은 하지 않았다.

반면 startup의 우리 문서 3커밋은 다른 기반 위에 있다. continuous에는 수정 대상 문서 일부가 아예 없고 운영 규칙 문맥도 다르므로, 세 커밋만 가져와 동일한 협업 문서 상태를 만든다고 볼 수 없다. **직접 cherry-pick을 실행하지 않았으며 아래 충돌은 파일 존재와 문맥 비교에 따른 예상**이다. 무리한 충돌 해결이나 PM 문서 기반 전체 복사를 하지 않는다.

## 비교한 고정 계보

| 역할 | 커밋 |
|---|---|
| continuous 관측 tip / CX35 | `1d2e14d8c6ca184a7833e8180b745622cfd77d72` |
| startup의 3커밋 이전 base | `35b3001debd95d10c070cc2dd02a68294eada17e` |
| 공동 시작 문서·작업큐 | `947ae53d014a078f7c5bab27d6ae25afbc2af163` |
| 연구 state 현재 포인터 | `02fc88ec10b818b48b51869cb556e32948e91c53` |
| 통합 state 현재 포인터 / startup tip | `c274a01c562a71ab5366c4b30ae9ef051cc94e5c` |
| continuous와 startup base의 merge-base | `81fbea78cf2badf5cb3f30865d15b35f791b1636` |

내용 읽기는 아래 6문서의 startup base/tip와 continuous tip에 한정했다. 관련 커밋 이력·링크 대상의 존재 여부는 Git metadata만 확인했으며 대상 문서·결과 payload로 탐색을 확장하지 않았다.

## 6문서 차이와 보존 조건

| 경로 | continuous tip | startup 3커밋의 실제 변경 | 통합 시 주의 |
|---|---|---|---|
| `AGENTS.md` | 존재, 이전 규칙 | 공동 작업 선점 절 9줄 추가 | startup base에는 continuous 대비 선행 51줄 추가·1줄 교체가 있다. 상단의 10/11 운영 문맥도 다름 |
| `CLAUDE.md` | 존재, 이전 규칙 | 같은 선점 절 9줄 추가 | startup base에는 선행 44줄 추가가 있다. 협업 절만 붙여 최신 운영 문구까지 반영됐다고 할 수 없음 |
| `docs/pm/kickoff.md` | 없음 | 기존 27줄 PM 시작 문서를 공통 시작 안내로 개정 | 수정 대상 부재에 따른 modify/delete 충돌 예상 |
| `docs/pm/work-queue.md` | 없음 | 88줄 신규 문서 | 파일 추가 자체는 가능하지만 연결된 PM 규약이 continuous에 없음 |
| `docs/pm/research/state.md` | 없음 | 기존 이력 앞에 현재 진입점 12줄 추가 | 수정 대상 부재. 원래 노출 장부·한계 이력을 함께 보존해야 함 |
| `docs/pm/integration/state.md` | 없음 | 기존 이력 앞에 현재 진입점 11줄 추가 | 수정 대상 부재. 원래 통합 체크포인트를 대체하거나 버리는 작업이 아님 |

startup base→tip의 총 변경은 6파일 **168줄 추가·17줄 삭제**다. AGENTS/CLAUDE의 기존 본문과 두 state의 기존 이력은 보존됐으며 kickoff만 오래된 역할별 복사 프롬프트를 공통 시작 절차로 개정했다. continuous→startup tip의 차이인 489줄 추가·1줄 삭제를 전부 우리 3커밋의 작업량으로 해석하면 안 된다. 상당 부분은 먼저 존재한 PM 문서 기반이다.

## 필요한 기반과 링크 의존

정확 경로의 최초 도입 metadata는 다음과 같다. **최초 도입 커밋만 적용하면 현재 기반이 완성된다는 뜻이 아니다.** 이후 문서 개정까지 포함한 고정 base `35b3001d`와 대조해야 한다.

| 도입 커밋 | 관련 선행 경로 |
|---|---|
| `86b3a0e0c411802d81a097277e09e98f9c53db23` | `docs/pm/kickoff.md`, `docs/pm/research/state.md`, `docs/pm/integration/state.md` |
| 같은 `86b3a0e0` | 직접 연결된 `docs/pm/overview.md`, `docs/pm/contracts.md`, `docs/pm/resources.md`, `docs/pm/workflow.md`, `docs/pm/service/state.md` |
| `1c1ef9b2b8ab7e367be9df818b0ed6a00d2346c2` | `docs/pm/collab-protocol.md`, `docs/pm/collaboration-charter-draft.md` |

위 경로들은 startup base에는 있지만 continuous tip에는 없다. AGENTS/CLAUDE의 선행 변경에는 `59fd20159cc63c17cee28ba67fecf7b93fd12ad0`의 10/11 운영 전환과 `bfb265c81590807ba65e0ee1ad2a038e8b6746b3`의 두 사람 협업 규약 등이 포함된다. 두 state에는 그 뒤의 체크포인트·노출 장부도 있다. 관련 이력은 문서만의 단일 커밋들로 구성됐다고 가정하지 않으며, 이 목록을 통째 cherry-pick할 실행 목록으로 제안하지 않는다.

상대 링크 존재 검사에서도 기존 이력의 일부 링크는 startup tip 자체에 없다. 이번 작업은 그 오래된 링크를 수정하거나 연결된 payload를 검증하지 않았다. 새로 사용할 통합 진입점은 선택한 최종 기반에서 별도로 링크를 검토해야 하며, `git show <고정 commit>:<경로>` 포인터를 checkout 내 파일 존재와 혼동하지 않는다.

## 안전한 통합 선택지와 다음 단계

이 의존은 협업 문서를 조합할 때의 조건이며 새로운 출시 필수 게이트를 추가하지 않는다.

1. **코드·합성 검증 계보:** 출시 담당자가 실제 대상 tip·작업트리·소유권을 재확인한 뒤 continuous 계보의 변경과 테스트 증거를 수용할 수 있다. 고정 ancestry가 있으므로 동일 변경의 재구현이나 복사 브랜치는 필요하지 않다. fast-forward 가능성은 적용 결정과 별개다.
2. **협업 문서 계보:** 최신 PM 기반을 가진 정본에서 해당 문서 소유자가 startup 3커밋의 의미와 preimage를 확인해 결합하는 편이 안전하다. 또는 다른 통합 대상을 택한다면 필요한 기반 문서의 정확 버전·범위를 먼저 별도 계획·선점해야 한다. 현재 continuous에 3커밋만 강제로 붙이고 최신 협업 체계가 완성됐다고 표시하지 않는다.
3. 최종 조합 시 이 6문서의 기존 이력·운영 게이트·과학적 한계를 보존하고, 선택한 대상에서 정확 diff와 링크를 독립 검토한다. 통합 대상이나 제품 코드가 새로 바뀌는 경우의 검증은 해당 통합 계획에서 정한다.

이번에는 이 보고서만 작성했다. root/ws/startup/continuous의 기존 내용이나 공유 index, branch tip을 옮기지 않았고 새 worktree·cherry-pick·merge·push·제품 적용·브라우저·기존 테스트 재실행은 모두 0이다. 이후 이 보고서를 저장하는 문서 커밋은 queue의 단일 index 담당자가 수행한다.
