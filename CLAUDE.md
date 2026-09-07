# Pitcheezy — CLAUDE.md

## 프로젝트
MLB Statcast 투구 추천 + 4자 책임 분해. 3인 졸업 캡스톤.
트랙 A: data/, transition/ | B: policy/, ope/ | C: decomp/, cv/
설계 SSOT는 노션. docs/는 스냅샷 — 편집하지 말 것. 규약 전체는 docs/conventions.md.

## 먼저 읽을 것
- docs/design-note.md — 확정 설계. 재논의·수정 금지
- docs/interface-spec.md — 트랙 간 계약. src/pitcheezy/interfaces/가 이걸 코드로 강제

## 규칙
- IMPORTANT: 2026 시즌 데이터는 전이 모델 학습·튜닝·모델 선택에 쓰지 않는다. OPE·분해 전용
- IMPORTANT: 숫자를 추정하거나 지어내지 않는다. 없으면 "미측정"
- interfaces/ 스키마 변경 = 스펙 버전 업 + ADR. 코드만 바꾸지 말 것
- 실험은 configs/ 파일로만. 파라미터 하드코딩 금지
- 채택 판단은 e2e OPE로만. NLL·ECE는 스크리닝용
- 시드·데이터 버전·커밋 없는 결과는 결과가 아님

## 환경
- 로컬은 테스트만: `pytest tests/interfaces -q` (tests/fixtures/ 소형 데이터). 풀 실행은 Colab 러너 (notebooks/colab_runner.ipynb)
- data/, runs/는 gitignore. Drive에 있음. 수집은 scripts/fetch_data.py로만
- 의존성은 pyproject.toml에. pip install만 하고 끝내지 말 것

## 작업 흐름
1. 세션 브랜치 그대로 사용. PR 제목에 `[A]` `[B]` `[C]` 접두
2. 커밋: Conventional Commits, scope = 모듈명 (`feat(transition): …`, `exp(EXP-P0-A-001): …`)
3. PR 전 `pytest tests/interfaces -q` 통과. 출력을 보여줄 것
4. interfaces/ 변경은 리뷰 필수

## 실험
- ID EXP-P{phase}-{track}-{seq}. 한 ID = 변경 하나
- configs/p{phase}/{ID}.yaml → runs/{ID}/{seed}/ → results/{ID}.json (커밋)
- 시드: 탐색 {0,1,2} / 채택 {0..4}. 시드마다 체크포인트

## 하지 말 것
- docs/ 편집, notebooks/ 결과 인용, 설계 노트 결정 우회
