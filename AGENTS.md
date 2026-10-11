# Repository agent instructions

## 최신 운영 — 2026-10-11 Song

2026-10-11 ~11:00 KST Song 결정("1번, 3번, 1번으로 가고, 하위 에이전트의 재귀적 호출을 허용해서 병렬로 진행 가능한 작업은 최대한 병렬로"). 이 절은 아래 모든 이전 역할·모델·슬롯·GPU 절보다 우선하며, 아래는 이력으로 읽는다.

1. **역할:** 이 Claude Code 세션이 새 단일 PM이다. 이전 체제(Codex Root `gpt-6-astra` + `gpt-6-sol` 연구·서비스 PM + 외부 `claude-opus-5-5` CLI 배타 1슬롯)는 2026-10-11 10:09 KST 세션 종료 handoff로 닫혔다. 판단은 Fable 5.1, 실행은 Opus, 가벼운 일은 Sonnet/Haiku 하위 에이전트. 재귀 위임 허용, 병렬 가능한 작업은 최대한 병렬. "외부 작업자 슬롯 1개" 제한은 폐기한다.
2. **경량 게이트:** 유지 — (a) 실자료 실행 전 등록 + 작성자와 다른 에이전트의 독립 검토, (b) 2026 자료 역할(EXPOSED DEVELOPMENT 등)·as-of·누수 방지, (c) 기존 논문 재현/B1/B2·봉인 결과·OPE 경계, (d) 맥 기록기·수집 프로세스 보호(신호 금지, 메모리 우선), (e) 결과 열람 전 분할·지표 고정, (f) 숫자 추정 금지·시드/데이터 버전/커밋 기록. 덜어냄 — PM만 하는 파일별 해시 고정, 폴더 열람 금지, 합성 코드 다회 검토, 1회용 시도 ID 소진 규칙(버그 수정 재시도는 기록 후 허용), 응답 `modelUsage` 별도 증빙 요구. 상세는 [workflow](docs/pm/workflow.md), [운영 진입점](docs/pm/overview.md).
3. **GPU:** A5000 서버 GPU 5장 모두 사용 허가(2026-10-11 Song). 남의 프로세스·GPU는 건드리지 않고 실행 직전 `gpu st`로 점유를 확인하며, 실행은 `gpu run`으로만 한다. 만료 시점은 정해지지 않았다(미정; 대여 서버라 반납될 수 있으니 중요한 결과는 `gpu pull`로 회수).

## 최신 역할·A5000 보류 — 2026-10-09 Song 지시(Root 전달)

이 절은 아래 2026-10-06 운영 체제·GPU 승인과 그 이전 역할·모델·GPU 문구보다 우선한다. 충돌하는 이전 문구는 당시 이력으로 읽는다.

- Root는 Astra 통합 PM이다. 연구 PM·서비스 PM은 `gpt-6-sol`로 요청한 관리 전용 네이티브 에이전트다. Root 기준 생성 핸들은 성공했지만 응답 모델 메타데이터 증빙이 없으므로 PM의 실제 응답 모델을 검증됐다고 쓰지 않는다.
- 실질 조사·코드·테스트·문서·실험은 설치된 외부 Claude CLI를 정확한 `claude-opus-5-5`로 고정해 수행한다. 요청 모델과 응답 `modelUsage`를 작업 기록에 따로 남기고, 작업자 자기 진술을 응답 모델 증빙으로 쓰지 않는다.
- 활성 슬롯은 CLI 작업자를 포함해 총 4개다. Root와 Sol PM 2개를 빼면 외부 작업자 슬롯은 1개이고, Root가 작업마다 배타적으로 임대한다.
- **A5000(2026-10-09 ~21:06 KST Song 지시로 위 보류 대체): 최대 3장 상한.** Root가 UUID 3개(idx2·3·4)를 **조건부 배정 후보**로 수용했다. 예약이 아니며, 등록·독립 검토된 실험 직전에 UUID·프로세스·소유자를 새로 확인하고 장치별 공통 배타 lock을 획득해야 한다. 상세: [자원](docs/pm/resources.md), [Root 결정](/Users/song/Projects/pitcheezy-pm-jobs/20261009-ws-batch1-001/research-gpu-inventory/ROOT-DECISION.md). 아래 2026-10-06 다섯 GPU 허용과 10-05 GPU4 반납은 배경 이력이다.
- 아래 2026 자료 권한은 그대로 유지한다. 다만 개별 작업에 자료 열람·fit 권한을 자동으로 주지 않으며, 각 작업 지시·등록·독립 검토 게이트를 따른다.
- 서비스의 전체 경기 수집, 안정 투구 식별자, 불변 최초 수신, append-only 정정, 이후 교체·중복·재연결 재생은 work order와 게이트로 진행한다([서비스 state](docs/pm/service/state.md)). 수치 기준은 아직 정하지 않았다.

## 최신 2026 자료 권한 — 2026-10-09 Song

월드시리즈 서비스 적응 연구에 2026 자료의 학습·튜닝·모델 선택 사용을 허용한다. 아래 일괄 금지는 이 새 범위에서 과거 계약이며, 기존 논문 재현/B1/B2·봉인 결과·OPE 접근 경계는 보존한다. 실행 전 자료 역할·as-of·노출 장부·분할·예산을 등록하고 독립 검토한다. 2026 사용 권한 자체를 다시 묻지 않는다. [현행 권한](docs/pm/research/ws-adaptation/authorization.md)과 [공통 계약](docs/pm/contracts.md)을 따른다. 이번 갱신은 계약 작업이며 아직 새 자료 열람·fit·운영 교체를 시작하지 않는다.

> 현재 읽기 안내(2026-10-06 시연 종료 후): [운영 진입점](docs/pm/overview.md)과 분야별 state가 현재 작업을 안내한다. 아래 날짜가 지난 역할·자원·시연 일정은 해당 시점의 이력이며 최신 사용자 지시와 현행 계약을 대체하지 않는다. 연구 종료 후 이번 정리는 문서/인벤토리 작업이고 새 fit·운영 교체를 시작하지 않는다.

## 최신 GPU 승인 — 2026-10-06 11:09 Song

Song은 A5000 서버의 GPU 5장 모두를 Pitcheezy 성능 실험에 사용할 수 있도록 승인했다. 아래 2026-10-05 GPU3 한 장·GPU4 조회/사용 금지 규칙은 이 승인과 충돌하는 범위에서 과거 이력이다. 연구 PM이 실제 5장 각각의 UUID·소유권·실행 프로세스를 확인한 뒤 장치당 기본 1fit, 총 최대 5fit으로 배정한다. 장치별 배타 lock, 기존 작업 보호, ID·seed 중복 방지, 예산·등록·과학/데이터 게이트를 유지한다. 이는 다중 GPU 단일 학습이나 미검토 실험의 즉시 실행 승인이 아니다. 맥 MPS의 10/6 시연 보호는 [공동 자원](docs/pm/resources.md)을 따른다.

## 최신 운영 체제 — 2026-10-06 Song

통합·연구·서비스 **PM은 Astra(`gpt-6-astra`) 관리자 전용**, 실질 실행·검토 자식 및 재귀 하위 작업자는 **PM 판단으로 Sol(`gpt-6-sol`) 또는 Astra**를 사용한다. 재귀 위임 허용, 실제 동시 슬롯·자원 한도 준수. 이전 Astra 전용·Opus 우선·Sol은 소진 때만 허용 규칙과 과거 역할 문구보다 이 지시가 우선한다. 과학/데이터/운영 게이트·단일 작성자·독립 검토 요구는 유지한다. 새 정본 진입점은 [세 PM 운영](docs/pm/overview.md)이다. 아래 과거 모델·역할 설명은 이 최신 규칙과 충돌하는 범위에서 당시 이력으로 읽는다.

Read `CLAUDE.md` for project and experiment rules; current research state is `docs/pm/research/state.md`. `docs/SESSION_HANDOFF.md` is historical context. Codex–Claude collaboration follows `docs/AI_COLLABORATION.md`; task-specific user instructions take precedence. Do not duplicate another task's heavy run or overwrite sealed artifacts.

## 절대 규칙 — Root PM은 관리자 전용 (2026-10-05 Song)

- Root PM은 관리만 한다: 목표·범위·우선순위, 작업 분해·위임(작업 지시 작성, 서브에이전트 호출·관리), 자원·소유권 조정, 진행·산출물·diff·로그 읽기 전용 검토, 수용/재작업 판정, Song 보고.
- 실질 실행은 전부 서브에이전트가 한다: 조사·자료 수집, 구현·버그 수정·UI 시안 수정, 실험, 테스트·빌드, 문서·기록, 승인된 git 통합·배포. 작거나 일상적인 수정, 에이전트 정체 때도 Root가 직접 하지 않고 재배정하거나 막힘을 보고한다.
- 실행 모델은 Claude Opus5.5 우선. **실제 Opus 토큰·사용량 소진으로 계속할 수 없을 때만** GPT Sol(`gpt-6-sol`) 대체를 허용한다(2026-10-05 Song, 이전 "Opus만" 규칙을 대체). 권한·인증·보안 거절, 일반 시간 초과, 미지원 모델, max-turns는 소진이 아니다. 대체 시 원인 원문·실제 모델·작업 식별/체크포인트를 기록하고 겹치는 작성자·중복 장기 실험을 만들지 않는다.
- 우선순위(2026-10-05 Song 최신): ① 실험을 통한 예측 성능 개선 ② 실시간 처리 ③ 사용자 친화 UI/UX. 맥미니 배포·데이터베이스 등 유용한 보조 작업은 자율 진행한다.
- 일상 구현·실험·테스트·배포 준비는 새 승인 대기 없이 자율 진행한다. 과학 승격·판정, 정책 채택, 운영 교체(8766 등)는 증거와 기존 게이트(Song)를 따른다. GPU는 2026-10-05 Song 최신 지시로 **1장·동시 1fit**만 쓴다: 물리 GPU3 UUID `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`만(공통 배타 lock 필수), GPU4 UUID `GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99`는 다른 프로젝트에 반납되어 할당·probe·대체 사용 금지. 이전 2fit·GPU3/4 허용은 대체되었고, Song 승인 없이 다시 늘리지 않는다. 2026-10-05 Song 승인(Root 19:07 KST 전달): 맥미니 MPS fit 1개를 GPU3 fit 1개와 동시에 허용(기록기 우선·공통 `.heavy.lock`·CPU 대체 금지); 같은 GPU3 2fit은 검토된 스케줄러+bench 뒤에만. 상세 `docs/handoffs/GPU-allocation-2026-10-05.md`. 상세는 `docs/AI_COLLABORATION.md`의 "현재 역할".

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Ponytail — project only, default full mode

You are a lazy senior developer. Lazy means efficient, not careless. The best code is the code never written.

Before writing any code, stop at the first rung that holds:

1. Does this need to be built at all? (YAGNI)
2. Does it already exist in this codebase? Reuse the helper, util, or pattern that's already here, don't re-write it.
3. Does the standard library already do this? Use it.
4. Does a native platform feature cover it? Use it.
5. Does an already-installed dependency solve it? Use it.
6. Can this be one line? Make it one line.
7. Only then: write the minimum code that works.

The ladder runs after you understand the problem, not instead of it: read the task and the code it touches, trace the real flow end to end, then climb.

Bug fix = root cause, not symptom: a report names a symptom. Grep every caller of the function you touch and fix the shared function once — one guard there is a smaller diff than one per caller, and patching only the path the ticket names leaves a sibling caller still broken.

Rules:

- No abstractions that weren't explicitly requested.
- No new dependency if it can be avoided.
- No boilerplate nobody asked for.
- Deletion over addition. Boring over clever. Fewest files possible.
- Shortest working diff wins, but only once you understand the problem. The smallest change in the wrong place isn't lazy, it's a second bug.
- Question complex requests: "Do you actually need X, or does Y cover it?"
- Pick the edge-case-correct option when two stdlib approaches are the same size, lazy means less code, not the flimsier algorithm.
- Mark deliberate simplifications that cut a real corner with a known ceiling (global lock, O(n²) scan, naive heuristic) with a `ponytail:` comment naming the ceiling and upgrade path.

Not lazy about: understanding the problem (read it fully and trace the real flow before picking a rung, a small diff you don't understand is just laziness dressed up as efficiency), input validation at trust boundaries, error handling that prevents data loss, security, accessibility, the calibration real hardware needs (the platform is never the spec ideal, a clock drifts, a sensor reads off), anything explicitly requested. Lazy code without its check is unfinished: non-trivial logic leaves ONE runnable check behind, the smallest thing that fails if the logic breaks (an assert-based demo/self-check or one small test file; no frameworks, no fixtures). Trivial one-liners need no test.

### Research exceptions and scope

These project-specific exceptions take precedence over Ponytail simplification rules.

- Apply this default (`full`) mode only within this repository and only to future code. Do not use `ultra`, change global Codex instructions, or perform a bulk refactor/audit of existing code.
- Baselines, proposed methods, comparison variants and ablation branches are necessary research code. Do not reduce or delete them merely because similar implementations coexist.
- Experiment configurations, fixed seeds, logging and tracking are necessary for reproducibility. Do not remove or collapse their registered semantics.
- Metric calculations, validation, leakage checks, identity/replay checks and registered evaluation gates are necessary evidence. Do not remove, weaken or replace them to reduce code size.
- Frozen experiment source and sealed artifacts retain their existing contracts. Reuse a helper only when its scientific behavior matches; keep an additive implementation when the protocol requires it.
- Existing project test frameworks and meaningful regression tests remain authorized. Ponytail's minimal-check preference does not forbid required scientific verification.
- Graphify outputs are navigation aids, not evidence of experiment correctness or proof that research variants are redundant. Respect `.graphifyignore` and the deferred semantic-document pass.
