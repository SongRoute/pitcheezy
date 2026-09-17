# tests/fixtures

소형 parquet(투수 2명 × 한 경기 ≈ 100구). 맥미니 로컬 테스트 전용.
`scripts/make_fixture.py`가 버전 있는 데이터셋(`data/raw/{버전}/statcast_{시즌}.parquet`)에서 투수별 시즌 첫 경기 한 판을
통째로 잘라 컬럼 전부(119열)·정렬 키 그대로 저장한다. 재생성:

```bash
.venv/bin/python scripts/make_fixture.py --version d20260911-s2325 --season 2024 --pitchers 657277 666142
```

| 파일 | 출처 데이터 버전 | 투수 | 경기 | 행 수 |
|---|---|---|---|---|
| statcast_2024_d20260911-s2325_p2.parquet | d20260911-s2325 | Webb, Logan (657277, R) | game_pk 745445 (2024-03-28) | 98 |
| statcast_2024_d20260911-s2325_p2.parquet | d20260911-s2325 | Ragans, Cole (666142, L) | game_pk 746335 (2024-03-28) | 98 |

파일 sha256 `caf63732a7b22f471b9025a95fa63c7598a533d0a0cd2f30b4569cdf71f87114` (196행, 2026-09-17 생성).
