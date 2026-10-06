# Pitcheezy — CLAUDE.md

## 현재 작업 모델 우선 적용 — 2026-10-06 Song

Song 최신 지시: “지금부터 opus 사용량이 부족해서 astra를 활용해서 작업하자.” **현재 작업은 Astra(`gpt-6-astra`)로 실행한다.** 아래 2026-10-05 Opus 우선·소진 시 Sol만 허용 규칙보다 이 사용자 지시가 우선한다. 사용량 부족은 사용자 보고이며 별도 계량·잔여량 확인 결과가 아니다. Root PM 관리자 전용과 서브에이전트 실행 분담, 단일 작성자·검토·과학/운영 게이트는 유지한다. 타자 그림을 제외한 [합의 UI 개선 계획](docs/handoffs/UI-refinement-2026-10-06.md)은 구현 승인 상태이며, 구현·검증·배포 완료를 뜻하지 않는다.

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
