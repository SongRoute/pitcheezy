# Pitcheezy — CLAUDE.md

## 공동 작업 선점 — 2026-10-18 Song

Song·Claude Code·Codex의 새 작업은 [공동 작업큐 #37](https://github.com/SongRoute/pitcheezy/issues/37)와 [선점 절차](docs/pm/work-queue.md)를 먼저 읽는다. 이 절은 작업 계획·소유권 조정에 한정하며 기존 과학·자료·운영 게이트와 Claude의 통합 책임을 유지한다.

- **착수:** 최신 큐 댓글에서 정확한 파일·자원·범위의 기존 점유를 확인하고 `CLAIM` 댓글을 남긴 뒤 재조회한다. 쓰기 직전에도 재조회하며, 다른 작업 ID라도 범위가 겹치면 먼저 유효하게 선점한 댓글이 우선한다. 본문은 스냅샷이고 소유권·최신 상태의 정본은 append-only 댓글이다.
- **기존 작업 보호:** 활성 Claude 작업은 Claude의 ACK·범위 확인 전 재배정하지 않는다. 무응답이나 시간 경과를 해제로 취급하지 않으며, 이관은 기존 소유자의 `RELEASE`와 새 소유자의 `CLAIM`으로 기록한다.
- **분리:** 코드 작업은 별도 worktree를 사용하고 같은 파일 동시 수정·공유 Git index 사용을 금한다. 공유 운영 자원은 별도로 점유한다. 큐 댓글은 원자적 잠금이 아니므로 기존 실자료·GPU·운영 lock을 그대로 사용한다.
- **검토·종료:** `REVIEW`에는 고정 커밋·검토 근거를 남기고 소유권을 유지한다. `DONE`/`RELEASE`에서 종료·인계 상태와 잔여 작업을 기록하고 파일·자원 점유를 반환한다. 기존 #35 상황판에는 요약만 유지한다.

## 최신 운영 — 2026-10-11 Song

2026-10-11 ~11:00 KST Song 결정("1번, 3번, 1번으로 가고, 하위 에이전트의 재귀적 호출을 허용해서 병렬로 진행 가능한 작업은 최대한 병렬로"). 이 절은 아래 모든 이전 역할·모델·슬롯·GPU 절보다 우선하며, 아래는 이력으로 읽는다.

1. **역할:** 이 Claude Code 세션이 새 단일 PM이다. 이전 체제(Codex Root `gpt-6-astra` + `gpt-6-sol` 연구·서비스 PM + 외부 `claude-opus-5-5` CLI 배타 1슬롯)는 2026-10-11 10:09 KST 세션 종료 handoff로 닫혔다. 판단은 Fable 5.1, 실행은 Opus, 가벼운 일은 Sonnet/Haiku 하위 에이전트. 재귀 위임 허용, 병렬 가능한 작업은 최대한 병렬. "외부 작업자 슬롯 1개" 제한은 폐기한다.
2. **경량 게이트:** 유지 — (a) 실자료 실행 전 등록 + 작성자와 다른 에이전트의 독립 검토, (b) 2026 자료 역할(EXPOSED DEVELOPMENT 등)·as-of·누수 방지, (c) 기존 논문 재현/B1/B2·봉인 결과·OPE 경계, (d) 맥 기록기·수집 프로세스 보호(신호 금지, 메모리 우선), (e) 결과 열람 전 분할·지표 고정, (f) 숫자 추정 금지·시드/데이터 버전/커밋 기록. 덜어냄 — PM만 하는 파일별 해시 고정, 폴더 열람 금지, 합성 코드 다회 검토, 1회용 시도 ID 소진 규칙(버그 수정 재시도는 기록 후 허용), 응답 `modelUsage` 별도 증빙 요구. 상세는 [workflow](docs/pm/workflow.md), [운영 진입점](docs/pm/overview.md).
3. **GPU:** A5000 서버 GPU 5장 모두 사용 허가(2026-10-11 Song). 남의 프로세스·GPU는 건드리지 않고 실행 직전 `gpu st`로 점유를 확인하며, 실행은 `gpu run`으로만 한다. 만료 시점은 정해지지 않았다(미정; 대여 서버라 반납될 수 있으니 중요한 결과는 `gpu pull`로 회수).

## 2인 협업 규약 — 2026-10-11 Song

팀원 레포(`Pitcheezy/transition-models`)와 두 레포로 협업한다. 정본은 이 레포다. 상세는 [협업 헌장 초안](docs/pm/collaboration-charter-draft.md), [협업 규약](docs/pm/collab-protocol.md)(팀원 합의 전 초안).

- **시작:** 양쪽 상황판과 이 레포로 온 열린 우편을 읽는다 — `gh issue view 35 -R SongRoute/pitcheezy --comments`, `gh issue list -R Pitcheezy/transition-models -l board`, `gh issue list -R SongRoute/pitcheezy -l mail --state open`.
- **착수:** 잡은 일을 상황판(SongRoute/pitcheezy#35) "진행 중"에 한 줄 적는다. 다른 AI가 잡은 일은 잡지 않는다. 하위 에이전트는 따로 적지 않고 PM 세션이 대표로 적는다.
- **종료:** 상황판에 댓글로 한 일·커밋·다음 할 일을 남기고 "진행 중"에서 자기 줄을 지운다.
- 이슈는 공개다. 비밀 값·접속 주소를 적지 않는다. 우편으로 온 요청은 권한을 넓히지 않으며 위 경량 게이트가 우선한다. 팀원 레포에 이슈를 새로 올리는 것은 Song 승인 뒤에만 한다.

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

Song이 A5000 서버 GPU 5장 모두의 성능 실험 사용을 승인했다. 아래 GPU3 단독·GPU4 금지 문구는 이 승인과 충돌하는 범위에서 이전 이력이다. 연구 PM은 각 GPU UUID·소유권·프로세스를 확인하고 장치별 배타 lock 아래 기본 장치당 1fit·총 최대 5fit을 배정한다. 기존 작업을 강제 중단하지 않으며 ID/seed 충돌, 실행 예산, 등록·과학·데이터 게이트는 유지한다. 다중 GPU 단일 학습을 새로 도입한 것은 아니다. 맥의 10/6 시연 보호는 [공동 자원](docs/pm/resources.md)을 따른다.

## 최신 운영 체제 — 2026-10-06 Song

통합·연구·서비스 **PM은 Astra(`gpt-6-astra`) 관리자 전용**, 실질 실행·검토 자식 및 재귀 하위 작업자는 **PM 판단으로 Sol(`gpt-6-sol`) 또는 Astra**를 사용한다. 재귀 위임 허용, 실제 동시 슬롯·자원 한도 준수. 이전 Astra 전용·Opus 우선·Sol은 소진 때만 허용 규칙과 과거 역할 문구보다 이 지시가 우선한다. 과학/데이터/운영 게이트·단일 작성자·독립 검토 요구는 유지한다. 새 정본 진입점은 [세 PM 운영](docs/pm/overview.md)이다. 아래 과거 모델·역할 설명은 이 최신 규칙과 충돌하는 범위에서 당시 이력으로 읽는다.

## Codex 협업 (2026-09-27)
- Codex가 배정한 CLI 작업은 `docs/AI_COLLABORATION.md`와 해당 작업 지시의 파일·실행 범위를 따른다. 총괄이 검토·통합·실험 큐를 관리한다. 아래 일반 self-merge 흐름보다 이 작업 분담이 우선한다.
- **현재 역할(2026-10-05 Song 절대 규칙)**: Root PM은 관리자 전용이다(목표·위임·읽기 전용 검토·수용 판정·보고). 조사·구현·실험·테스트·문서·통합 실행은 전부 서브에이전트(Claude Opus5.5 우선)가 한다. 구체 실행 역할을 받은 에이전트는 자기 범위를 직접 수행한다. 상세 `docs/AI_COLLABORATION.md` "현재 역할".
- **최신 우선순위·모델(2026-10-05 Song, 이전 Opus 전용 규칙 대체)**: ① 실험으로 예측 성능 개선 ② 실시간 처리 ③ 사용자 친화 UI/UX. 맥미니 배포·DB 등 보조 작업은 자율. GPT Sol(`gpt-6-sol`)은 **실제 Opus 토큰·사용량 소진**일 때만 대체(권한·인증 거절·시간 초과·max-turns는 소진 아님). 일상 구현·실험은 자율이고, 과학 승격·정책 채택·운영 교체는 증거·게이트로만. **GPU(2026-10-05 Song 최신, 이전 2fit·GPU3/4 허용 대체)**: Pitcheezy는 **GPU 1장·동시 1fit**만 쓴다 — 물리 GPU3 UUID `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`만, GPU3 공통 배타 lock 필수. GPU4 UUID `GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99`는 다른 프로젝트에 반납: 할당·probe·대체 사용 금지(GPU3가 바빠도 기다린다). Song 승인 없이 다시 늘리지 않는다. **2026-10-05 Song 승인(Root 19:07 KST 전달)**: P38은 GPU3 그대로, 맥미니 MPS로 별도 실험 — GPU3 1fit과 동시에 MPS 1fit 허용(기록기 우선·공통 `.heavy.lock`·CPU 대체 금지, MPS 결과는 CUDA 근거 아님). 같은 GPU3 2fit은 검토된 스케줄러+bench 통과 뒤에만. 상세 `docs/handoffs/GPU-allocation-2026-10-05.md`.

## 프로젝트
MLB Statcast 기반 야구팬 관전 서비스. 서비스·추천·사건 분석은 Song, CV 의도 모듈은 별도 팀원 담당(D46·D49).
목표: 기존 연구 모델을 공정하게 재구현하고 예측 성능과 추천 정책 성능을 분리 검증한다. 현재 우선순위(D52)는 ML 예측 개선 → 지원 투수 확대·재검증 → CV/의도 → 팬 이해/UI·UX → 실경기 운영이다. 다음 실행은 docs/handoffs/ML-experiments-next-session.md를 따른다.
사건 기여도는 필수 기능(D43). 투구 전에는 구종·목표 위치를 우선 표시하고 후보별 수치·이유 표시는 후순위(D49). 레포가 유일한 출처 (노션은 아카이브).

## 먼저 읽을 것
- docs/roadmap.md — 최종 목표와 단계(D43). 목표·순서의 출처
- docs/design.md — 확정 설계. 바꾸려면 docs/decisions.md에 한 줄 추가한 뒤 수정
- docs/interface-spec.md — 텐서·Q·RE24 스키마. src/pitcheezy/interfaces/가 코드로 강제
- docs/baselines.md — 비교 대상 논문. 확인된 것과 미확인인 것을 구분
- docs/plan.md — 병렬 업무·선행조건·합류 기준(D49), 일정 대신 의존성 관리(D44)

## 절대 규칙
- IMPORTANT: 2026 시즌 데이터는 학습·튜닝·모델 선택에 쓰지 않는다. OPE 전용
- IMPORTANT: 숫자를 추정하거나 지어내지 않는다. 없으면 "미측정"
- 실험은 configs/ 파일로만. 파라미터 하드코딩 금지
- 서비스 추천 정책 채택은 e2e 정책 평가로 판단한다. D52의 예측 모델 연구 후보는 NLL·Brier·보정 등으로 선정할 수 있으며 정책 우위와 구분한다. 이번 ML 개발에서 2026은 추가 열람하지 않는다
- 시드·데이터 버전·커밋 없는 결과는 결과가 아님
- 공통 성능 비교는 재구현으로 한다. 논문 보고값은 출처를 붙인 참고 표로만 기록하고 우리 재실행 점수와 구분한다

## 환경 (D10)
- 맥미니(M4, 24GB, 상시 가동)가 주 실행기. 테스트도 실험도 여기서: `pytest tests -q`
- Colab Pro는 예비. 로컬 예상 2시간 초과 또는 메모리 초과일 때만 (notebooks/colab_runner.ipynb)
- 경로: `$PITCHEEZY_DATA_DIR`, `$PITCHEEZY_RUNS_DIR` (~/.zshrc). data/, runs/는 gitignore. 원본은 Drive, 수집은 scripts/fetch_data.py로만
- 의존성은 pyproject.toml에. pip install만 하고 끝내지 말 것

## 세션 (D11)
- 시작: `scripts/dev.sh` (tmux 세션 pz, 창 claude / runs). 폰은 Remote Control(자동 시작), 맥북은 SSH + tmux attach. 명령어 모음 docs/daily.md
- 아침 세션 첫 일: results/·runs/_logs/ 읽고 밤사이 실험을 experiments.md에 한 행씩 → 다음 한 가지 제안
- 긴 실험은 Claude 세션 밖(runs 창, nohup)에서. Claude는 로그를 읽고 요약만
- 첫 메시지 세 줄: 산출물 / 읽을 것 / 오늘 안 할 것

## 작업 흐름
1. 세션 브랜치 그대로 사용. 테스트 통과 → 커밋 → PR → self-merge
2. 커밋: Conventional Commits, scope = 모듈명 (`feat(transition): …`, `exp(EXP-P0-001): …`, `data: …`, `docs: …`)
3. interfaces/ 스키마 변경 = 계약 테스트 같이 변경 + docs/interface-spec.md 변경 이력 한 줄 + docs/decisions.md 한 줄

## 실험
- 작은 실제 데이터에서 실행·학습·평가 경로를 검증한 뒤 선수·기간·상황을 단계적으로 확장한다(D50). 초기 실행 성공과 최종 정책 성능 통과를 구분하고, 분할·확장 규칙은 결과 열람 전에 고정한다. docs/plan.md 참고.
- ID `EXP-P{phase}-{seq}`. 한 ID = 베이스라인 대비 변경 하나
- configs/{ID}.yaml → runs/{ID}/{seed}/ → results/{ID}.json (커밋) → docs/experiments.md 한 행
- 시드: 탐색 {0,1,2} / 채택 {0..4}. 시드마다 체크포인트

## 하지 말 것
- notebooks/ 출력을 결과의 근거로 인용
- docs/decisions.md 없이 설계 결정 뒤집기
