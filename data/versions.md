# 데이터 버전

- ID 형식 `d{YYYYMMDD}-{tag}` (예 `d20260908-s2326`)
- 수집은 `scripts/fetch_data.py`로만. 파라미터 고정 + sha256 검증으로 누가 받아도 같은 파일
- 데이터 파일은 Google Drive `pitcheezy/data/`. 레포의 `data/`에는 이 파일만 커밋
- 2026 검증셋: 정규시즌 종료 직후 1회 갱신 후 고정. 고정본은 `frozen` 표시. 포스트시즌 제외 (ADR-2)

| 버전 ID | 시즌 | pybaseball 쿼리 파라미터 | 행 수 | parquet sha256 | 수집 스크립트 커밋 | frozen |
|---|---|---|---|---|---|---|
