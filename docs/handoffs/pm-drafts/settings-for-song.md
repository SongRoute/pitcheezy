# Song이 직접 넣을 설정 (초안)

PM과 작업자는 설정·권한 파일을 고치지 않는다. 아래는 Song이 고르고 직접 넣는다. 넣은 항목은 이 파일에서 지운다.
조사 근거는 [도구 조사](../PM-tooling-research-2026-10-03.md)에 있다. "시험 안 함"이라고 적은 것은 넣은 뒤 한 번 확인한다.

## 지금 넣어도 되는 것 (동결과 무관)

### 1. 작업자 정의 적용
N1을 확정한 뒤에 한다. 초안 4개를 Claude Code가 읽는 자리로 복사한다.

```bash
mkdir -p /Users/song/Projects/pitcheezy/.claude/agents && cp /Users/song/Projects/pitcheezy/docs/handoffs/pm-drafts/agents/*.md /Users/song/Projects/pitcheezy/.claude/agents/
```

- 작업자 셋은 `model: claude-opus-5-5`, 검토자는 `model: fable`로 고정돼 있다.
- 정의 파일로 부른 작업자가 실제로 Opus 5.5로 뜨는지는 시험 안 함. 새 PM의 첫 위임에서 확인한다. (정의 없이 `model: "opus"`를 넘긴 호출은 10/3 00:40에 Opus 5.5로 뜨는 것을 확인했다.)

### 2. 권한 규칙 (`~/.claude/settings.json`의 `permissions`)
- `ask`에 추가: `"Edit(//Users/song/Projects/pitcheezy/docs/decisions.md)"`, `"Bash(git push *)"`
  - PM이 결정 줄을 쓰거나 push할 때마다 승인 창이 뜬다. 창이 폰으로 오는지는 시험 안 함. 안 오면 PM이 줄을 보여 주고 Song이 직접 붙이는 방식으로 간다.
  - 결정 줄을 `allow`로 두면 모든 세션과 하위 에이전트가 승인 없이 쓸 수 있게 되므로 권하지 않는다.
- `allow`에 추가: `"Bash(git ls-remote *)"`, `"Bash(gh pr view *)"`
  - 10/2에 push 뒤 원격 확인 명령이 막혔던 것을 푼다.
- `deny`에 추가: `"Bash(git push --force *)"`, `"Bash(git push -f *)"`, `"Bash(git push --force-with-lease *)"`, `"Bash(git push origin demo/*)"`, `"Bash(gh pr merge *)"`, `"Bash(git add -A *)"`, `"Bash(git add --all *)"`, `"Bash(git add .)"`
  - 규칙은 명령 글자에 맞춰 보므로 `git -C . push` 같은 변형은 못 막는다. 보조 장치로만 본다.
- 이 규칙들이 권한 분류기보다 먼저 적용되는지는 문서상 그렇지만 시험 안 함.

### 3. `gpu` 이름 충돌 (지금 고치는 것을 권함)
- 이 맥의 셸에서 `gpu`는 zsh git 플러그인의 `git push upstream` 별칭이 먼저 잡힌다. 준비 조사 중에 `gpu --help`가 실제로 `git push upstream --help`로 실행됐다(upstream 원격이 없어 아무것도 올라가지 않았다).
- 사용자 설정의 허용 규칙 `Bash(gpu st *)` 등 8개와 `ask` 1개도 같은 이름을 쓴다.
- 고치는 방법 둘 중 하나:
  - `~/.zshrc` 끝에 `unalias gpu 2>/dev/null` 한 줄을 넣는다.
  - 허용 규칙을 `/Users/song/.local/bin/gpu` 전체 경로로 바꾸고, 지시문에서도 전체 경로를 쓴다.

### 4. 확인만
- 모델 화면에서 Fable 사용분이 별도 크레딧으로 청구되는지 본다. 청구된다면 동의 창이 폰으로 오지 않아 PM의 턴이 끊길 수 있다(조사 결과, 확인 못함).
- 옛 Desktop 세션(모델, 실시간, 화면, 브리핑): 하던 일이 끝난 것을 확인하고 보관한다. 결정 줄을 쓰는 세션은 새 PM 하나만 남긴다.

## 동결이 끝난 뒤에 넣을 것

### 5. 하위 에이전트 기본 모델과 동시 실행 수 (`/Users/song/Projects/pitcheezy/.claude/settings.local.json`)
```json
"env": { "CLAUDE_CODE_SUBAGENT_MODEL": "claude-opus-5-5", "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS": "6" }
```
- 첫 줄: 모델을 적지 않은 하위 에이전트와 워크플로 에이전트가 Fable 대신 Opus 5.5로 돈다. 검토자 정의의 `model: fable`은 이 값보다 우선한다. (둘 다 문서 기준, 시험 안 함)
- `CLAUDE_CODE_SUBAGENT_MODEL_FORCE`는 넣지 않는다. 넣으면 검토자까지 Opus가 된다.
- 동시 실행 6은 제안값이며 측정값이 아니다. 기본은 20이다.

### 6. 격리 작업트리의 기준 브랜치 (같은 파일)
```json
"worktree": { "baseRef": "head" }
```
- 이 레포는 원격 기본 브랜치가 `main`이라, 이 줄이 없으면 격리 작업트리를 쓰는 에이전트가 9월의 `main`에서 갈라진다. 격리 작업트리를 쓰기 시작할 때만 필요하다.

### 7. 편집 뒤 테스트 훅 (`/Users/song/Projects/pitcheezy/.claude/settings.json`)
- 지금 훅은 `.py`·`.yaml`·`.yml`을 고칠 때마다 `pytest tests -q -x`(120초 제한)를 돌린다. 문서에 따르면 하위 에이전트 안에서도 실행되므로 작업자가 여럿이면 겹쳐 돈다.
- 바꿀 방향: 하위 에이전트 안에서는 건너뛰고 작업자가 지시된 테스트를 직접 돌린다. 훅 입력으로 하위 에이전트를 구분하는 방법은 시험 안 함이므로, 동결 뒤 PM이 초안을 내고 한 번 시험한 다음 넣는다.

### 8. 분류기에 환경 알려 주기 (선택)
- `/permissions`의 Auto mode 탭이나 `/auto-mode-setup`에서 "이 레포는 공개 GitHub 레포"라는 사실과 "사용자가 이 대화에서 정한 결정을 decisions.md에 한 줄로 붙이는 것은 허용"을 문장으로 넣을 수 있다. 사용자 설정에서만 읽히고 프로젝트 설정에 넣으면 무시된다. 2번의 `ask` 규칙으로 충분하면 넣지 않아도 된다.

## CLAUDE.md와 메모리
- [CLAUDE.md 개정안](CLAUDE.proposed.md): 네 군데만 바꿨다. N1 확정 뒤 PM이 결정 줄을 쓰고, Song이 "CLAUDE.md를 개정안으로 바꿔"라고 말하면 PM이 바꾼다.
- [메모리 교체안](memory-feedback-opus-alone.proposed.md): "Opus 단독·Fable 금지" 메모리를 새 방식으로 고친 것. 같은 절차로 적용한다.
