# Pitcheezy — CLAUDE.md

## 프로젝트
MLB Statcast 투구 추천 모델. 개인 개발 (Song).
목표: 기존 연구 모델(SmartPitch, Takamido & Nakamoto)을 같은 데이터·같은 저울(2026 OPE) 위에서 재구현하고, 우리 모델이 그보다 나은지 보인다.
4자 책임 분해는 Phase 2 후순위. 레포가 유일한 출처 (노션은 아카이브).

## 먼저 읽을 것
- docs/design.md — 확정 설계. 바꾸려면 docs/decisions.md에 한 줄 추가한 뒤 수정
- docs/interface-spec.md — 텐서·Q·RE24 스키마. src/pitcheezy/interfaces/가 코드로 강제
- docs/baselines.md — 비교 대상 논문. 확인된 것과 미확인인 것을 구분
- docs/plan.md — 앵커와 이번 주 할 일

## 절대 규칙
- IMPORTANT: 2026 시즌 데이터는 학습·튜닝·모델 선택에 쓰지 않는다. OPE 전용
- IMPORTANT: 숫자를 추정하거나 지어내지 않는다. 없으면 "미측정"
- 실험은 configs/ 파일로만. 파라미터 하드코딩 금지
- 채택 판단은 e2e OPE로만. NLL·ECE는 스크리닝용
- 시드·데이터 버전·커밋 없는 결과는 결과가 아님
- 비교는 재구현으로만. 논문이 보고한 숫자를 우리 표에 옮기지 않는다

## 환경 (D10)
- 맥미니(M4, 24GB, 상시 가동)가 주 실행기. 테스트도 실험도 여기서: `pytest tests -q`
- Colab Pro는 예비. 로컬 예상 2시간 초과 또는 메모리 초과일 때만 (notebooks/colab_runner.ipynb)
- 경로: `$PITCHEEZY_DATA_DIR`, `$PITCHEEZY_RUNS_DIR` (~/.zshrc). data/, runs/는 gitignore. 원본은 Drive, 수집은 scripts/fetch_data.py로만
- 의존성은 pyproject.toml에. pip install만 하고 끝내지 말 것

## 세션 (D11)
- 시작: `scripts/dev.sh` (tmux 세션 pitcheezy, 창 claude / runs). 폰은 Remote Control, 맥북은 SSH + tmux attach
- 아침 세션 첫 일: results/·runs/_logs/ 읽고 밤사이 실험을 experiments.md에 한 행씩 → 다음 한 가지 제안
- 긴 실험은 Claude 세션 밖(runs 창, nohup)에서. Claude는 로그를 읽고 요약만
- 첫 메시지 세 줄: 산출물 / 읽을 것 / 오늘 안 할 것

## 작업 흐름
1. 세션 브랜치 그대로 사용. 테스트 통과 → 커밋 → PR → self-merge
2. 커밋: Conventional Commits, scope = 모듈명 (`feat(transition): …`, `exp(EXP-P0-001): …`, `data: …`, `docs: …`)
3. interfaces/ 스키마 변경 = 계약 테스트 같이 변경 + docs/interface-spec.md 변경 이력 한 줄 + docs/decisions.md 한 줄

## 실험
- ID `EXP-P{phase}-{seq}`. 한 ID = 베이스라인 대비 변경 하나
- configs/{ID}.yaml → runs/{ID}/{seed}/ → results/{ID}.json (커밋) → docs/experiments.md 한 행
- 시드: 탐색 {0,1,2} / 채택 {0..4}. 시드마다 체크포인트

## 하지 말 것
- notebooks/ 출력을 결과의 근거로 인용
- docs/decisions.md 없이 설계 결정 뒤집기
