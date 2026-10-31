<!-- 메모리 교체안(초안). 대상: ~/.claude/projects/-Users-song-Projects-pitcheezy/memory/feedback-opus-alone.md 와 MEMORY.md의 해당 줄. Song이 N1을 확정한 뒤 PM이 적용한다. -->
---
name: feedback-opus-alone
description: "운영 방식 변천 — 10/3부터 Fable 5.1 PM이 Song과 대화·결정 기록, Opus 5.5 작업자가 작성(재귀 하위 에이전트 허용), 다른 모델 검토만 독립 검토. D93 'Opus 단독·Fable 금지'는 끝남"
metadata:
  type: feedback
---

2026-10-02 밤 Song이 운영 방식을 바꾸자고 제안했다. (이 문장은 확정된 뒤 적용할 때 "<날짜>에 확정했다(D번호)"로 채운다. 확정 전에는 이 교체안을 적용하지 않는다.) Claude Fable 5.1이 PM을 맡아 Song과 대화하고 결정을 기록한다. 작업은 흐름마다 Opus 5.5 작업자에게 맡기고, 작업자가 하위 에이전트를 재귀로 부르는 것을 허용한다. Song은 PM과만 대화한다. 규칙은 docs/PM-charter.md, 상태는 docs/handoffs/PM-state.md.

**Why:** 여러 세션이 각자 Song과 대화해 지시와 결정 번호가 갈라졌다. 10/2 커밋은 거의 전부 Fable이 작성해서 D100의 "Opus 작성·Fable 검토" 분리도 깨졌다. Song은 폰으로 한 곳에서 결정하고 싶어 한다.

**How to apply:**
- 작업자는 Agent 도구에 `model: "opus"`(또는 `.claude/agents` 정의)로 부른다. 첫 보고에서 실제 모델을 확인한다.
- 독립 검토는 작성과 다른 모델이 한다. Opus가 쓴 것은 Fable이, Fable이 쓴 것은 Opus가 검토한다. 같은 모델 검토는 자기 검토로만 적는다.
- 결정 줄은 Song이 PM 세션에서 답한 직후 PM이 쓴다. 분류기가 막으면 Song에게 돌려보내고 다른 세션으로 우회하지 않는다.
- 결정은 번호 선택지와 추천 하나를 먼저 보여 준다. 사소한 구현 선택은 묻지 않는다.

**이력 (지우지 않는다):**
- D93(9/29): Opus 단독, Fable·Sol·Astra·Codex 금지
- D100(9/30): 핵심 검토만 Fable. stage 승인은 조건부 연속 실행, 수정은 모아서 한 번에 승인
- D103: 테스트 파일만 바뀌면 같은 모델 검토 뒤 source_commit 이동
- 기존 메모리에 적혀 있던 운영 메모(결정 기록 본문에는 없음): 게이트 PASS 기록은 테스트 통과 뒤에 커밋, 실행기는 `status == 'PASS'` 정확 일치만 받는다
- D106: 추천대로 자율 진행 / D116: 10시간 자율 — 새 방식에서 유지할지는 Song 결정(N9)
- 그대로 유지: 절대 규칙, 2026 OPE 1회, 봉인 보존, PR merge 금지

관련: [[project-pm-oversight]], [[feedback-parallel-sessions]], [[user-song-prefers-plain-language]]

MEMORY.md 줄 교체안:
- [운영 방식(Fable PM·Opus 작업자)](feedback-opus-alone.md) — 10/3부터 PM(Fable)이 Song과 대화·결정 기록, 작업은 Opus 5.5 작업자와 재귀 하위 에이전트, 다른 모델 검토만 독립 검토, D93 "Opus 단독"은 이력
