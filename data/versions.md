# 데이터 버전

- ID 형식 `d{YYYYMMDD}-{tag}` (예 `d20260910-s2325`). `scripts/fetch_data.py` 가 실행일과 `--tag` 로 만든다
- 수집은 `scripts/fetch_data.py`로만. 시즌 날짜·월 청크·정렬 키·dtype 은 코드 상수로 고정, `--verify {버전}` 으로 sha256·행 수 대조
- 데이터 파일은 Google Drive `pitcheezy/data/raw/{버전}/`. 레포의 `data/`에는 이 파일만 커밋
- 시즌(파일)마다 한 행. 홀드아웃 시즌은 `holdout_{시즌}/` 아래 별도 저장, 정규시즌 종료 전에는 `frozen` false
- 2026 검증셋: 정규시즌 종료 직후 1회 갱신(`--freeze`) 후 고정. 고정본은 `frozen` true. 포스트시즌 제외 (ADR-2)
- 기존 행은 수정하지 않는다. 같은 버전 ID 는 스크립트가 거부하므로 재수집은 새 버전으로

| 버전 ID | 시즌 | 파일 | 날짜 범위 | 행 수 | parquet sha256 | pybaseball 버전 | 수집 커밋 | frozen |
|---|---|---|---|---|---|---|---|---|
| d20260911-s2325 | 2023 | statcast_2023.parquet | 2023-03-30~2023-10-02 | 720684 | 46c354beae196ce2d501fe989c491725f2ecc47601f58d4b8b0234b982984730 | 2.2.7 | 9ebc6bb | true |
| d20260911-s2325 | 2024 | statcast_2024.parquet | 2024-03-20~2024-09-30 | 711899 | 3cf05edddd22a2eac3d55be11cc56e00fe56f7fcf75823a8ecc8fa89e7e9a274 | 2.2.7 | 9ebc6bb | true |
| d20260911-s2325 | 2025 | statcast_2025.parquet | 2025-03-18~2025-09-28 | 712528 | 4c785756d8901f8c46c70891954a4c54b8ad1c6909db67f4a000221691dc0249 | 2.2.7 | 9ebc6bb | true |
| d20260911-s2325 | 2026 | holdout_2026/statcast_2026.parquet | 2026-03-25~2026-09-09 | 647896 | 374e8b9539ae7c42cf1f5a8bb38194d3ccfcc1f3e34746fd5b0d46b98ffd55c5 | 2.2.7 | 9ebc6bb | false |
