# Pitcheezy 규약 (conventions)

기준일 2026-09-07. 이 문서의 변경은 [운영] 대화에서만. CLAUDE.md는 이 문서의 요약본이며, 둘이 어긋나면 이 문서가 기준.

## 1. 레포 구조 (모노레포)

```
pitcheezy/
├── CLAUDE.md
├── README.md                  # 한 줄 + 노션 링크
├── pyproject.toml
├── docs/
│   ├── design-note.md         # 노션 확정본 스냅샷. ADR 채택 시만 갱신
│   ├── interface-spec.md      # 스펙 버전 = git 태그 spec-vN
│   └── conventions.md         # 이 문서
├── configs/
│   └── p0/ p1/ p2/            # {실험ID}.yaml — 실험은 여기서만 정의
├── src/pitcheezy/
│   ├── interfaces/            # 공통: RE24·텐서·Q 스키마 + 검증 (스펙의 코드화)
│   ├── data/                  # A: 수집·전처리·RE24 테이블
│   ├── transition/            # A: 전이 모델
│   ├── policy/                # B: VI
│   ├── ope/                   # B: OPE
│   ├── decomp/                # C: 분해·기준점
│   ├── cv/                    # C: 파일럿
│   └── common/
├── scripts/
│   ├── fetch_data.py          # 데이터 수집. 파라미터 고정 + 해시 검증
│   └── run_experiment.py      # --config configs/p0/EXP-P0-A-001.yaml --seed 0
├── tests/
│   ├── interfaces/            # 계약 테스트. PR 필수
│   └── fixtures/              # 소형 parquet (투수 2명 × ~100구). 로컬 테스트용
├── results/                   # 커밋 대상: {실험ID}.json 요약만
├── notebooks/
│   └── colab_runner.ipynb     # 유일한 실행 진입점 노트북. 로직 없음
├── data/                      # gitignore. Drive. versions.md만 커밋
└── runs/                      # gitignore. Drive. {실험ID}/{seed}/

```

* `docs/`는 노션 확정본 스냅샷. ADR 채택 시 Song이 동기화하고 태그를 붙인다. 레포에서 직접 편집하지 않는다.
* `notebooks/`는 탐색용. 결과의 근거로 인용하지 않는다. `colab_runner.ipynb`만 실행 진입점.

## 2. 실행 환경 — Colab Pro

* 코드는 레포, Colab은 실행기. `colab_runner.ipynb` 하나: clone → `pip install -e .` → Drive 마운트 → `python scripts/run_experiment.py --config … --seed …`
* 저장: Google Drive `pitcheezy/data/`, `pitcheezy/runs/`. 레포의 `data/`, `runs/`는 gitignore
* 세션 만료 대비: 시드마다 체크포인트 저장·재개. 5시드 채택 판정은 시드 단위로 나눠 실행 (Pro는 백그라운드 실행이 없어 탭이 닫히면 끊길 수 있음)
* 로컬(Mac Mini)은 `tests/fixtures/`로 테스트만. 풀 실행 없음
* W&B 로그인은 Colab Secrets. 키를 노트북·레포에 쓰지 않는다

## 3. 브랜치·커밋

* `main` 보호. 작업 브랜치는 Claude Code Code 탭이 세션마다 자동 생성하는 브랜치 그대로 (`claude/…`). 직접 만들 때는 `<track>/<topic>` (`a/re24-table`)
* 트랙 식별은 PR 제목 접두 `[A]` `[B]` `[C]` `[공통]`과 커밋 scope로
* PR 리뷰어 1명. `src/pitcheezy/interfaces/` 변경은 양쪽 트랙 리뷰 필수. 팀 합류 전에는 Code 탭 Reviewer 세션이 대신
* 커밋: Conventional Commits, scope = 모듈명 — `feat(transition): …` `fix(ope): …` `exp(EXP-P0-A-001): …` `data: …` `docs: …`
* 태그: `spec-vN`, `p0-baseline`, `p1-adopted-<축>`. 노션 ADR·실험 로그의 커밋 칸에는 태그 우선

## 4. 실험

* ID `EXP-P{phase}-{track}-{seq}` → `EXP-P0-A-001`. 설정 파일명 = ID. 한 ID = 베이스라인 대비 변경 하나
* 시드: 탐색 `{0,1,2}` / 채택 판정 `{0,1,2,3,4}`. 채택은 5시드 없이 하지 않는다
* 채택은 e2e OPE로만. 모듈 지표(NLL, ECE, 셋업 일치도)는 OPE로 볼 후보의 순서를 정하는 스크리닝용
* ② 축 비교의 NLL·ECE는 학습 창 내 홀드아웃. 2026은 OPE·분해 전용 (ADR-3)
* 실험 로그 항목의 데이터 버전·커밋·시드가 비면 결과가 아니다. e2e 칸이 비면 "완료" 불가

## 5. 데이터 버전

* ID `d{YYYYMMDD}-{tag}` — 예 `d20260908-s2326`
* `data/versions.md`에 pybaseball 쿼리 파라미터, 시즌, 행 수, parquet sha256, 수집 스크립트 커밋
* 수집은 `scripts/fetch_data.py`로만. 파라미터 고정 + 해시 검증으로 누가 받아도 같은 파일
* 2026 검증셋: 지금 스냅샷으로 구축, 정규시즌 종료 직후 1회 갱신 후 고정. 고정본은 versions.md에 `frozen` 표시. 포스트시즌 제외 (ADR-2)
* "누가 어떤 데이터 썼는지" 헷갈리는 일이 실제로 생기면 그때 DVC 검토

## 6. 결과 기록

* 원천 수치: W&B 프로젝트 `pitcheezy`. run 이름 `{실험ID}/s{seed}`
* 요약: `results/{실험ID}.json` 커밋 — 시드별 지표 + 평균·CI, 설정 해시, 데이터 버전, 커밋, W&B run URL
* 해석·채택 판단: 노션 04 실험 로그. 여기서만
* 아티팩트(텐서, Q, 가중치): Drive `runs/{실험ID}/{seed}/`

## 7. 문서 동기화

* 노션 = 설계·결정·로그의 SSOT. 레포 `docs/` = 스냅샷
* ADR 채택 → 같은 날 Song이 `docs/design-note.md`, `docs/interface-spec.md` 갱신 + 태그 + Claude 프로젝트 지식 파일 교체
* 노션 로그 DB(02~05)는 스냅샷하지 않는다. Claude 프로젝트는 커넥터, Claude Code는 Notion MCP로 읽는다

## 8. Claude Code / Cowork

* `CLAUDE.md`는 팀 공유(git). 이 문서의 요약본
* Code 탭: 코드·테스트·실험 config·데이터 스크립트. 세션 = 작업 하나 = 브랜치 하나
* Cowork: 노션↔docs 동기화, 회의록 정리, results → 실험 로그 요약
* 2주차부터 `.claude/`에 추가: hooks(`docs/design-note.md` 쓰기 차단, 편집 후 `pytest tests/interfaces -q`), skills(`/new-experiment`, `/log-results`), Notion MCP
