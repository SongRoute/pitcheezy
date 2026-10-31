# Codex 독립 검토 — main d04afab1

검토 기준: `git diff 61be7a86...d04afab1 -- apps/observer`, `REVIEW-PROMPT-svc-ui-1026-main-2026-10-31.md`. 수정 확인: `fba6b2ff846dcfa9afad9ea5df4204007e0604c1`. 코드 작성에 참여하지 않은 Codex와 읽기 전용 UI 하위 검토자가 검토했다. 판정 대상은 지정 전시 경기 849843의 고정 번들 및 모니터 재생 경로이며 WS fixture의 부가 재생을 구분한다. 제품 소스 수정 없음.

**판정: PASS_WITH_NOTES.** Opus의 전시 차단 P1 두 건은 수정됐다. 다만 경로별 안내의 최초 감지는 부분 미해결이고, MLB 원문 결과명을 쓰는 WS 재생에는 잘못된 투구 표식이 남는다. 실기기·팀원 호스트·849843 전체 재생을 통과했다는 판정은 아니다.

## 1. 발견 사항

### P2-1 — 결과명이 MLB 원문이면 볼·인플레이를 스트라이크로 표시
- 위치: `apps/observer/web/src/service/model.ts:199`, `api.ts:243`; 호출: `GameScreen.tsx:523,573,670,696`, `Sheets.tsx:159,190`.
- 원인: `resultKind`는 소문자 Statcast 코드 집합/`hit_into_play` 접두사만 인정하고 나머지 비어 있지 않은 값은 모두 `strike`로 처리한다. API adapter는 description 원문을 유지한다.
- 재현: `node --experimental-strip-types --input-type=module -e "import {resultKind} from './src/service/model.ts'; console.log(['Ball','Ball In Dirt','In play, out(s)','future_code'].map(resultKind))"`를 web에서 실행하면 모두 strike다. WS 849803 fixture의 index 2는 `Ball`, index 5는 `In play, out(s)`다. 해당 파일에서 볼 원문 127개·인플레이 원문 64개를 확인했다. 공개 WS 재생에서 해당 공을 고르면 볼/타격 설명과 G1 표식이 모순된다.
- 범위: 실제 849843 `out4/v1/game/849843.json`의 262구는 알려진 소문자 코드만 사용한다. 지정 고정 전시본에서 이 오류는 재현되지 않으므로 P2. 라이브/WS 원문 경로는 영향 있다.
- 최소 수정: 공통 `resultKind`에서 두 어휘 체계를 명시적으로 매핑하고 미등록 값은 `unknown`으로 둔다. 볼·인플레이·스트라이크·미등록 값의 작은 회귀 검사 하나를 둔다. 데이터 fixture를 바꾸거나 결과명에서 임의 추정하지 않는다.

### P2-2 — 게임 파일에만 최초 안내가 생기고 health sequence가 같으면 안내를 발견하지 못함
- 위치: `apps/observer/web/src/service/GameScreen.tsx:236,246`.
- 재현: 정적 200 게임을 연 뒤 health는 같은 `file_sequence`의 200, 게임 파일만 503 service_notice로 전환한다. 이미 안내를 본 적이 없어 `holdRef.current === false`이므로 246행에서 매번 반환하며 게임 파일을 읽지 않는다. 이전 점수·추천 화면이 남는다. 원래 Opus P2-2의 두 번째 경우다.
- 최소 수정: 호스트가 안내를 health에도 반드시 싣는 계약이면 그 계약과 시험을 명시한다. 경로별 안내가 요구되면 sequence가 같아도 게임 파일 안내를 확인하는 제한된 재검증 경로를 둔다. 요청 예산/호스트 계약은 미확정이므로 이번 검토에서 임의 채택하지 않는다.
- 판정: **PARTIAL**. 이미 안내를 감지한 뒤 health만 성공하는 경우는 수정됨. 최초 감지는 여전히 조건부 미해결. 호스트가 경로별 안내를 제공하는지는 미측정.

### P3 — 기존 잔여 사항
| 위치 | 재현·영향 | 최소 수정 / 판정 |
|---|---|---|
| `GameScreen.tsx:579,749-750`, `LivePanels.tsx:11,63` | 선수 투수 시트의 오늘 행은 `context_before`, 구종 표는 선택 공 포함. 첫 공에 0구/1구가 공존 | 오늘 행에 ‘이 공 전’ 기준을 명시하거나 표를 같은 시점으로 맞춤. 전체 경기 누수와 구분, 미해결 |
| `LivePanels.tsx:12` | 같은 이닝 여러 타석의 ‘2구’가 모두 같은 기준 문구 | 타자 이름을 기준 문구에 포함. 미해결 |
| `LivePanels.tsx:50` | corrected=true, unconfirmed 또는 held가 아닌 reason도 corrected-tagged에 포함 | ‘표시하지 않은 정정’과 실제 held 집계를 분리한 문구. 미해결 |
| `LivePanels.tsx:25` | first는 숫자, last=null인 PA가 선택 공 이후 결과를 그대로 가짐 | 열린 PA의 완료/결과를 선택 시점으로 제한. 전시 fixture 해당 사례 미확인, 미해결 |
| `Zone.tsx:14`, `api.ts:406` | 미사용 좌표 export, 승리확률 ‘미측정’ 주석과 화면 문구 차이 | 사용처 없는 export 정리 및 주석 정합. 비차단·미해결 |

API final 경로는 pollMs=0이며 안내→404 후 firstLoadRetry도 꺼진다(`GameScreen.tsx:342-350`). 과거 화면은 계속 숨기지만 자동 복구는 보장하지 않는다. 기존 ‘404는 거절 응답’ 정책이므로 새 회귀로 단정하지 않는다. 일시 404까지 자동 복구해야 하는지는 계약 선택 사항이다. 정적/진행 중 경기는 별도 polling이 유지된다.

## 2. fba6b2ff의 Opus 지적 처리 확인

| Opus 항목 | 코드 근거 | 판정 |
|---|---|---|
| P1-1 stylesheet 병합 사고 | `service.css:1`의 두 import와 설명만 유지. alert keep-all은 `service-layout.css` 마지막 규칙으로 이동 | CLOSED; zone 397 및 layout 검사 통과 |
| P1-2 선수 MixTable 미래 전체 경기 누수 | `GameScreen.tsx:750`에 asOfIndex 전달, `LivePanels.tsx:11`에서 <= 선택 공 필터 | CLOSED; 1구 차이는 위 P3로 유지 |
| P2-1 안내→일반 오류 후 과거 내용 노출 | hold/holdRef, 성공 적용 경로에서만 해제, noticeHold 조기 반환 | CLOSED; public/API 안내→404→정상 재현 검사 128 ok |
| P2-2 health 성공만으로 안내 해제 | hold 동안 setError(null)/sequence 조기반환 금지 | PARTIAL; 감지 후 복구는 닫힘, 최초 game-only 안내는 위 P2-2 |
| P2-3 backoff 거짓 복구/재알림 억제 | `ws_alert.py:548`: running이며 armb 필드 존재할 때만 평가. backoff에서 key가 없어 engine closed | CLOSED; off→backoff→off 테스트 포함 ops 128 passed |
| P3-1 미사용 CSS | service.css 중복 본문 삭제 | CSS는 CLOSED, Zone export는 남음 |
| P3-7 상태 팝업 자동 재개 | `GameScreen.tsx:348`에서 setPop(false) | CLOSED (코드 확인) |

경보 한계: status가 running인데 publisher.running=false이며 armb 필드를 유지하는 합성 조합은 inactive condition이 되어 recover를 낼 수 있다. 현재 ws_operate 작성 경로는 중간 상태에서 armb 필드를 생략하므로 실제 backoff 경로 회귀로 판정하지 않았다.

## 3. 운영 차선의 변경 범위

`ws_operate.py` 변경 없음. `ws_alert.py` diff는 `ops_armb_off` 규칙 추가뿐이며 기존 임계값·알림 전달·설정 파일을 바꾸지 않는다. 의도적으로 ARM-B를 끈 실행도 critical 대상이고 경기 시간 밖에도 감시한다. 발행기 자체의 자동 활성화/재시작을 추가하지 않는다. 재생 도구는 ws_operate를 통하지 않는다. 경보 전송은 FakeNotifier 시험만 수행했다.

## 4. 실행한 검사와 한계

별도 worktree `codex-review-main-1101`, 기준 d04afab1. 새 데이터 수집·학습·튜닝·모델 선택·운영 접속·외부 발신 없음. graphify 기존 그래프는 UI 최신 관계를 충분히 찾지 못해 고정 소스를 직접 대조했다.

| 명령 (해당 worktree 기준) | 실제 결과 |
|---|---|
| web: `npm run build && npm run build:public` | 성공, TypeScript 오류 없음 |
| `PYTHONPATH=src:.:apps/observer/backend /Users/song/Projects/pitcheezy/.venv/bin/pytest apps/observer/ops/tests -q` | 128 passed |
| observer Python에서 pytest.main으로 `apps/observer/backend/tests/test_ledger_policy.py`, `test_publisher_replay.py` 실행 | 73 passed, 의존성 deprecation 경고 2개 |
| web: `node tests/notice-hide-smoke.mjs` | 267 ok / 0 FAIL |
| web: `node tests/zone-geometry-check.mjs` | 397 ok / 0 FAIL |
| web: `node tests/ws-launch-layout-check.mjs` | 360 checks passed, 종료 0 |
| web: `node tests/publish-merge-check.mjs` | 성공 (9 files) |
| `/tmp/pitcheezy-codex-1101/notice-error.mjs` | 기존 smoke에서 390px public/API만 실행하고 안내→404 12초→정상 단계를 추가, 128 ok / 0 FAIL |

환경 시도도 기록한다: 처음 잘못 지정한 루트 `tests/test_publisher_replay.py`는 파일 없음으로 수집 0; 루트 venv는 fastapi가 없어 backend 2개 수집 실패. observer venv에는 pytest 실행기가 없어 observer Python에 기존 루트 venv의 site-packages를 append해 pytest만 재사용했다. 설치·설정 변경 없음. 정확한 명령:

```sh
PYTHONPATH=src:.:apps/observer/backend /Users/song/Projects/pitcheezy/.venv-observer/bin/python -c "import sys;sys.path.append('/Users/song/Projects/pitcheezy/.venv/lib/python3.12/site-packages');import pytest;raise SystemExit(pytest.main(['apps/observer/backend/tests/test_ledger_policy.py','apps/observer/backend/tests/test_publisher_replay.py','-q']))"
```

빌드 기록 원본 대조: 보존 ZIP 867,682 bytes, SHA-256 `af10293d5134219b7f750d00786366b93b0134c895519b939fd99a356d77a4bc` 일치. out4 849843의 투구 262·ready 246·unsupported 16을 재집계했다. 이 대조는 ZIP 내부 모든 파일과 out4의 바이트 동일성 또는 운영 재생 성공을 새로 증명하지 않는다.

미측정: iPhone Safari, 팀원 Cloudflare/QR, 실모니터 동기화, 849843 전체 live-replay, 경로별 안내 호스트 계약. 포팅 전 ws-gaps/ws-moments/static 4절 이후 통과를 주장하지 않는다. 제품 변경이 없는 검토 문서이므로 테스트 고치기나 별도 P2 구현은 다음 작업 범위다.
