# publish-ws — 월드시리즈 실발행본 재생 fixture (DEMO-WS-REPLAY-1028 · 4차전 추가 DEMO-WS-GAPS-1028)

맥미니 발행기가 경기 중 로컬 정적 JSON으로 쓴 공개 발행 폴더(`out/v1/`)의 마지막 상태를 2026-10-28에 바이트 그대로 복사했다. 파일은 고치지 않는다.

- `2026-10-24/v1/` — WS 1차전 849805, TB 2 @ MIL 3, 263구(추천 짝 180). 원본 `pm-jobs/ws-ops/runs/2026-10-24/out/v1`, 발행 커밋 df2f41fa, 마지막 발행 seq 1492 · 2026-10-24T03:41:52Z.
- `2026-10-27/v1/` — WS 3차전 849804, MIL 10 @ TB 1, 294구(추천 짝 274). 원본 `pm-jobs/ws-ops/runs/2026-10-27/out/v1`, 발행 커밋 e53b89a7, 마지막 발행 seq 1397 · 2026-10-27T03:25:49Z.
- `2026-10-28/v1/` — WS 4차전 849803, MIL 3 @ TB 4 (12회), 367구(추천 짝 301). 원본 `pm-jobs/ws-ops/runs/2026-10-28/out/v1`, 발행 커밋 e53b89a7, 마지막 발행 seq 1864 · 2026-10-28T04:44:40Z(경기 파일 seq 1835 · 04:39:46Z). 라인업 9+9명·박스 수치 있음.

띄우기: `npm run build:public && node tests/ui-preview.mjs --ws` (세 폴더를 겹쳐 제공, fixture clock). 정적 호스트용 한 폴더: `node tests/publish-merge.mjs --ws --out <새 폴더>` (경기 파일은 바이트 그대로, 색인 3파일은 합쳐서 씀; 검사 `node tests/publish-merge-check.mjs`).

알고 쓸 것:
- 세 경기 모두 final이다. `next`가 null이라 '다음 공 추천' 단계와 미지원 투수 참고 카드(`next.reference`)는 이 파일로 재현되지 않는다. 투구마다 `rec`(투구 전에 기록된 추천)·`actual`·`situation_before`·`linescore_before`가 있다.
- 세 run의 `dates.json`은 `dates: []`다. 날짜 목록은 manifest `live_dates`에서 온다.
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
b23bbf200f0e09c67a8762dd89df5e3cad7162115f9e34e3ddc54cbbc3b34f57  2026-10-28/v1/dates.json
a742f8973758dcfc3ca76223f6dff05934a9e75d0fd5fe0a465f940aebcfc766  2026-10-28/v1/game/849803.json
923e3b60ac2dbc89a065b6071d7a78bf202acc784e793908091b8047ad1ae4d8  2026-10-28/v1/games/2026-10-28.json
9bf0eb70dc6c566d3a4b4c4aec898ce3ab68d83d49fe9e43c4192a68afb037f0  2026-10-28/v1/health.json
3e18c72c51c5615760a1d40fd2f9edab3c26cdee51a7449db209a9a54c535320  2026-10-28/v1/manifest.json
```
