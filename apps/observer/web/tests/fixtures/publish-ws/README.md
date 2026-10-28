# publish-ws — 월드시리즈 실발행본 재생 fixture (DEMO-WS-REPLAY-1028)

맥미니 발행기가 경기 중 로컬 정적 JSON으로 쓴 공개 발행 폴더(`out/v1/`)의 마지막 상태를 2026-10-28에 바이트 그대로 복사했다. 파일은 고치지 않는다.

- `2026-10-24/v1/` — WS 1차전 849805, TB 2 @ MIL 3, 263구(추천 짝 180). 원본 `pm-jobs/ws-ops/runs/2026-10-24/out/v1`, 발행 커밋 df2f41fa, 마지막 발행 seq 1492 · 2026-10-24T03:41:52Z.
- `2026-10-27/v1/` — WS 3차전 849804, MIL 10 @ TB 1, 294구(추천 짝 274). 원본 `pm-jobs/ws-ops/runs/2026-10-27/out/v1`, 발행 커밋 e53b89a7, 마지막 발행 seq 1397 · 2026-10-27T03:25:49Z.

띄우기: `npm run build:public && node tests/ui-preview.mjs --ws` (두 폴더를 겹쳐 제공, fixture clock).

알고 쓸 것:
- 두 경기 모두 final이다. `next`가 null이라 '다음 공 추천' 단계와 미지원 투수 참고 카드(`next.reference`)는 이 파일로 재현되지 않는다. 투구마다 `rec`(투구 전에 기록된 추천)·`actual`·`situation_before`·`linescore_before`가 있다.
- 두 run의 `dates.json`은 `dates: []`다. 날짜 목록은 manifest `live_dates`에서 온다.
- 2026 자료다. 화면 표시·데모 전용이며 학습·튜닝·모델 선택에 쓰지 않는다.

sha256 (각 폴더 기준 상대경로, 원본과 동일):

```
590cde5a02606673bff989476aa6c12fbd10a4d29bbc30c9403678c630d39094  2026-10-24/v1/dates.json
4035b84afad791d003f43dd8f8dfe614c2825c0b3854c04ed5fbcbdd105f2793  2026-10-24/v1/game/849805.json
d3aa805ddb13425278ab6a5c8f1e9a53cfa405116b7e6e1e026e95a512e6cca9  2026-10-24/v1/games/2026-10-24.json
1484205b79ba9bbc1c6160ce1676bc7d2b435e33c3a1167c41e7fa66bb7c6258  2026-10-24/v1/health.json
6f694958be5e6c56d64ed554ca2147fe6c850db9df73c8db94bbc401a39ab57f  2026-10-24/v1/manifest.json
0e6c95b319822cf4015800743620832dcfa72590f1fdf8f7dd48f852a7374dcf  2026-10-27/v1/dates.json
40a5cadd54ac486c203c8a4afed8c1072efe1eb790b7bc691b450a391163795e  2026-10-27/v1/game/849804.json
540a5dc8d546d702052cc3fe75debbb91a5c824e2b3798e295f5d93e3f76ef8a  2026-10-27/v1/games/2026-10-27.json
08cd5dc1b76a1e2dfdaaa9fc504aef3bc496b69f71d4153a800fd265e0910159  2026-10-27/v1/health.json
152816b6cafa26c430a6fd0108778dfc52d340a4a313ab3cd381be5d9ece6525  2026-10-27/v1/manifest.json
```
