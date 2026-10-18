# Codex 연속 작업 선택 통합 인계표 — 2026-10-19

작성자 `implementation_1019`, 독립 검토자 `queue_1019`. 상태: metadata·정확 경로·문서 독립 검토 PASS 및 Root 수용. **As-of 2026-10-19 05:37 KST**이며 세션 종료 보고나 실제 통합 지시가 아니다.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6254848474 . 최신 소유권은 [공동 큐 #37](https://github.com/SongRoute/pitcheezy/issues/37), 목표·미해결 요약은 [상황판 #35](https://github.com/SongRoute/pitcheezy/issues/35)를 따른다.

## 결론과 고정 범위

현재 root HEAD는 `35b3001debd95d10c070cc2dd02a68294eada17e`다. 아래 continuous 대상 **18개 보고서와 테스트4파일은 root HEAD/index/worktree에 모두 없다**. 반면 startup의6개 문서는 root worktree bytes가 이미 해당 tip과 같지만 HEAD/index에는 반영되지 않았다. 이것은 작업 사본 선택 반영이며 커밋 통합 완료가 아니다. 대상28경로만 비교했고 다른 dirty/untracked 전체 목록이나 실제 payload는 열람하지 않았다.

| 계보 | 고정 끝점 | 이 표의 범위 |
|---|---|---|
| `codex/continuous-1018` | `f821ea2b168c04c1a79f79faf28ccabf581324f6` | `dd0209c0`부터11 commits |
| `codex/continuous-1019` | `18f5ecfff22ef8d73bcbd517497c2175ea27c735` | `d5ec9034`부터7 commits |
| `codex/coordination-startup-1018` | `c274a01c562a71ab5366c4b30ae9ef051cc94e5c` | `947ae53d`, `02fc88ec`, `c274a01c`의3 commits |

이11+7+3은 **선택 통합 계보 범위**다. 작업 완료 건수나 한 세션의 커밋 수와 같다고 해석하지 않는다. 특히 continuous 앞의 준비7 commits를 이번 목표 실적 또는 통합 대상18에 더하지 않는다. CX27은 실행 증거만 있고 별도 commit이 없다.

## Continuous 커밋과 정확 파일

코드 별칭은 B=`tests/coordination/public_candidate_smoke.mjs`, R=`tests/coordination/check_onekey_receipt.py`, T=`tests/coordination/test_onekey_receipt.py`, A=`tests/coordination/test_onekey_acceptance.py`다. 각 행의 보고서 경로와 코드 별칭이 해당 commit의 전체 변경 파일이다.

| 범위 / 큐 작업 | 고정 commit | 변경 파일 |
|---|---|---|
| 이전11 / CX05 | `dd0209c060226db27a7139b6af2bd4335a0fa43e` | B + [cx-05-current-candidate-e2e-2026-10-18.md](../../reviews/cx-05-current-candidate-e2e-2026-10-18.md) |
| 이전11 / CX06 | `2bac32489c2e2237d016154b2d3f6b027515f36f` | R+T + [cx-06-onekey-receipt-checker-2026-10-18.md](../../reviews/cx-06-onekey-receipt-checker-2026-10-18.md) |
| 이전11 / CX08 | `e57fa64d34041f6638b3dda8c0cd398ea352e1bd` | B + [cx-08-public-host-errors-2026-10-18.md](../../reviews/cx-08-public-host-errors-2026-10-18.md) |
| 이전11 / CX10 | `8f7c257f2971baa63f04e7718e8458505d844cde` | A + [cx-10-onekey-token-fixture-2026-10-18.md](../../reviews/cx-10-onekey-token-fixture-2026-10-18.md) |
| 이전11 / CX11 | `4022acb1588a526b2946792c67f3a7b232d3bf98` | B + [cx-11-public-live-states-2026-10-18.md](../../reviews/cx-11-public-live-states-2026-10-18.md) |
| 이전11 / CX14 | `9fbccd33191becafbb07d5dee8de8b69d5008cb0` | B + [cx-14-synthetic-schema-conformance-2026-10-18.md](../../reviews/cx-14-synthetic-schema-conformance-2026-10-18.md) |
| 이전11 / CX15 | `29888c291fe3523c7b19732c9702ef57ba758a7d` | 보고서만 + [cx-15-current-v4-snapshot-regressions-2026-10-18.md](../../reviews/cx-15-current-v4-snapshot-regressions-2026-10-18.md) |
| 이전11 / CX16 | `359e0c657269209278c386551c24b948428d8cc0` | B + [cx-16-public-health-freshness-2026-10-18.md](../../reviews/cx-16-public-health-freshness-2026-10-18.md) |
| 이전11 / CX17 | `f322bf22a0eac0f26940b2d71205801c356d1bd7` | A + [cx-17-onekey-clock-finiteness-2026-10-18.md](../../reviews/cx-17-onekey-clock-finiteness-2026-10-18.md) |
| 이전11 / CX18 | `645632ade8c91a7902d1a551ba5626aab6f7993b` | A + [cx-18-onekey-deferred-target-binding-2026-10-18.md](../../reviews/cx-18-onekey-deferred-target-binding-2026-10-18.md) |
| 이전11 / CX20 | `f821ea2b168c04c1a79f79faf28ccabf581324f6` | A + [cx-20-onekey-deferred-retry-budget-2026-10-18.md](../../reviews/cx-20-onekey-deferred-retry-budget-2026-10-18.md) |
| 현재7 / CX21 | `d5ec90342c0d497b84a034bdb41d9bba79ee0896` | R+T + [cx-21-onekey-v4-receipt-checker-2026-10-19.md](../../reviews/cx-21-onekey-v4-receipt-checker-2026-10-19.md) |
| 현재7 / CX22 | `4d0327e490f509efd484308a930e6bf06f257182` | A + [cx-22-onekey-generated-log-integration-2026-10-19.md](../../reviews/cx-22-onekey-generated-log-integration-2026-10-19.md) |
| 현재7 / CX23 | `68451af77cf891a665eba2701c77b4e958212d64` | B + [cx-23-public-withdrawal-recovery-2026-10-19.md](../../reviews/cx-23-public-withdrawal-recovery-2026-10-19.md) |
| 현재7 / CX24 | `2b52902c45c906e0e63871b08a92822508889120` | B + [cx-24-public-partial-body-timeout-2026-10-19.md](../../reviews/cx-24-public-partial-body-timeout-2026-10-19.md) |
| 현재7 / CX26 | `3b2b2e789eb54f3cd6e7834f9f22078cb98b7d48` | B + [cx-26-public-cache-propagation-2026-10-19.md](../../reviews/cx-26-public-cache-propagation-2026-10-19.md) |
| 현재7 / CX28 | `e4a435f3648db6849749110ef05b46c3b9828977` | R+T + [cx-28-onekey-initial-scope-history-2026-10-19.md](../../reviews/cx-28-onekey-initial-scope-history-2026-10-19.md) |
| 현재7 / CX29 | `18f5ecfff22ef8d73bcbd517497c2175ea27c735` | B + [cx-29-public-sheet-keyboard-2026-10-19.md](../../reviews/cx-29-public-sheet-keyboard-2026-10-19.md) |

## 의존성과 최소 선택 단위

| 단위 | 필요한 계보 / 의미 | 통합 시 함께 보존할 것 |
|---|---|---|
| B 공개 합성 브라우저 검사 | dd0209c0→e57fa64d→4022acb1→9fbccd33→359e0c65→68451af7→2b52902c→3b2b2e78→18f5ecff | 최종 B와 이9개 보고서. body/schema 수정과 304 대기 수리 이력을 잃지 않음 |
| R+T 독립 영수증 검사 | 2bac3248→d5ec9034→e4a435f3 | R/T를 같은 최종 상태로 선택하고 CX06/21/28 보고서 동반. v1 보존과 v4 초기 scope 선행성 수리를 함께 유지 |
| A wrapper 합성 회귀 | 준비 도입6c234009→8f7c257f→f322bf22→645632ad→f821ea2b→4d0327e4 | 최종 A, 관련5개 대상 보고서와 준비 CX04 근거. CX22의3method는 R도 필요 |
| 문서만인 CX15 | 29888c29 | 기존 고정 snapshot 진단 근거이며 코드 수리/PASS 선언이 아님 |

최종 source bytes를 선택하는 경우에는 위 파일 집합과 보고서·실행 경계를 함께 보존한다. 이력 cherry-pick 방식을 선택한다면 A의 최초 도입이 대상18보다 앞선 `6c234009ba8619058f6bead2fb496adef94e71a0`에 있고, 그 보고서 보완은 `26dee81579409496621a72af6c2ea7263cd08de9`에 있다는 선행조건을 해결해야 한다. 중간 수정 commit만 적용할 수 있다고 가정하지 않는다. 어느 방식도 이 작업에서 실행하지 않았다.

준비 계보7개는 `0ac37ba9`, `d817ef77`, `446c407f`, `41352074`, `d10d6348`, `6c234009`, `26dee815`다(`dccde711748d40565a78c50356cf8cb5c4f43d3e` 이후). 앞5개는 감사·측정/포스터 초안이며 이 표의 통합 대상에 포함하지 않는다. 뒤2개만 A의 필수 선행 source/문서 근거로 표시했다.

| 최종 코드 | Git blob @18f5ecff | SHA-256 |
|---|---|---|
| B | `fbc78c9e2634c84f947c77d48444ccdc0a5638b5` | `9c85730fc6d7e72ca1bdaaaf9c4d18a579cad923f48b01219c0b8d9985c2a05d` |
| R | `e5e5442cb06c985dccc8553d03ea4e01c10033cb` | `e8dcd6e18acbfde8cc52ff00e27ad3bc7ee2e77d2aa5a16c28e2f50621726e82` |
| T | `98986addba59755ab4134726e47aaf65846eb1d9` | `5a1d004272d068e49aaaad1e0d4a4b013c2f51b5d5fccd69d5cec5d72ef5be7e` |
| A | `a3199a474fc04d1b72a7dd0af393ee6d1dff424a` | `c395c7d42872961361e0923bb0a63f7532927e881e607383d772cf5ce5597d2c` |

## Startup의 이미 선택 반영된 경로

세 commit은 같은 startup 계보에서 순서대로 이어진다. 아래 root index는 모두 HEAD와 같고 staged 변경이 없다. 기존 선택 반영 파일을 새 파일처럼 덮어쓰거나, dirty worktree에 commit을 그대로 다시 적용해서는 안 된다. 실제 통합 담당자가 그 시점의 소유권과 정확 경로 preimage를 다시 확인한다.

| commit / 대상 | root HEAD=index blob | root worktree=startup tip blob | 관측 |
|---|---|---|---|
| 947ae53d014a078f7c5bab27d6ae25afbc2af163 / AGENTS.md | `60b73f1b0ce0d49f9330bb015bedfa42928578ec` | `1cebb3d300bdaaeffa884dc6d8fc89e3a2b40575` | tracked unstaged |
| 같은 commit / CLAUDE.md | `87f0bcbb08dede9783743337fd6fb2d7cb5bbdf1` | `cd89473a08367e9a04e7c41012540458f5eeeff4` | tracked unstaged |
| 같은 commit / docs/pm/kickoff.md | `7dffe7cde61baa6320fad62ebe7d4d420465a57a` | `8df10c7042f7e658b1a6129aa28fa4a3273ec0f4` | tracked unstaged |
| 같은 commit / docs/pm/work-queue.md | 없음 | `ceb129157cdbb82d130b1a80beaae00e13f25375` | 정확 경로 untracked |
| 02fc88ec10b818b48b51869cb556e32948e91c53 / docs/pm/research/state.md | `7c95708d94696c9353730ac6209c022a84689198` | `3ccc2fe90f8df1414a3aab613cc22273c2b5a67e` | tracked unstaged |
| c274a01c562a71ab5366c4b30ae9ef051cc94e5c / docs/pm/integration/state.md | `c2bd922ce3b10d1ed707c618787f7614933a22fe` | `a00c186475dc6aab1b747f2fb555aa0ff64ce2cc` | tracked unstaged |

따라서 현재 정확 경로 비교에서 startup 내용 충돌은 발견하지 않았지만, 미커밋 상태를 건너뛰는 안전한 merge를 증명한 것은 아니다. 다른 root/Claude 변경은 이 비교 범위 밖이다. 이 작업은 root/index를 수정하지 않았다.

## 검증 증거와 남은 경계

- B: CX27에서 `3b2b2e78`의7suite/17case를 각1회 순차 PASS했다. 뒤 CX29는 `18f5ecff`의 새 키보드2case만1회 PASS하고 기존7suite 본문 보존을 대조했다. 최종 B에서8suite 전체를 다시 실행했다고 쓰지 않는다. viewport 검사는 실기기·스크린리더 전체 검증이 아니다.
- R/T: CX28에서 우리 checker의 잘못된 초기 scope 이력 수용을 신규1method/2실패로 재현하고5줄 수정 뒤 전체17method PASS했다. 이는 wrapper 제품 수리와 별개다.
- A: 현재14method 중 고정 원본의 NaN clock/target binding에 대한2개 RED가 있다. CX22의 새 생성로그 통합3method는 당시 checker에서 PASS했으며 CX28 이후 재대조는 후속 CX31 PLAN이다. 전체14 PASS로 합산하지 않는다.
- 원본 wrapper `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af`의 [NaN clock 요청](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249663784)과 [deferred target 요청](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249730071)은 미수리다. [정적 최소 patch 제안](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249776984)은 미적용/미실행이며 CL02 owner ACK와 CX09 정식 source/등록/검토 쌍은 별도 대기다.
- 실제 host/인증/시험 창/회수·단일 writer·파트너 연결과 운영 GO는 이 통합 인계가 해결하지 않는다. 파일이 root에 없다는 것은 운영/과학 게이트 통과나 통합 허가가 아니다.

완료한 검사는 Git commit/file metadata, 대상28경로의 HEAD/index/worktree blob·SHA 대조와 문서 링크 존재/diff 검사다. 재현 실행·제품 테스트·build·cherry-pick·merge·push는0이다. 전용 `CX30/inventory-001.json` SHA `6b650f02bed47ca65f18b51c0f36f07827a8e544bffc499b1d6bb3e6d6399209`에 정확 경로 근거를 보존했다. Root index SHA `87b7a225d1fdbe25ce5e9221d0d77e5c5b705faa3ef99c11daadbdeada082e48`도 읽기 대조한다. 독립 문서 검토 후 이 새 인계표1파일만 COORD-05 index 소유자에게 커밋 인계한다.
