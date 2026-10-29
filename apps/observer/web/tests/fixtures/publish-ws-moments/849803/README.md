# 월드시리즈 4차전 849803 — 경기 중 시점 fixture (재생 본문)

**실발행본이 아니다.** 10/28 KST 4차전을 발행한 맥의 저장소(`store.sqlite3`: 발행기가 받은 MLB 피드 스냅샷 + 발행기가
경기 중 각 상황에 기록한 추천 장부 `served_recommendations`) **사본**을 오프라인 재생기(`scripts/publisher_replay.py --policy ledger
--keep-moments`)로 다시 흘려 되살린 **경기 중 발행 본문**이다. 추천은 다시 계산하지 않았다. 각 시점의 `next.rec`는 **경기 중 장부에
그 상황(state_sha256)으로 기록된 추천 그대로**다(`observer_app/ledger_policy.py`). 발행 도장(`publish.sequence/published_at`)과
영수증 시각은 재생 시계의 것이고(지연 0초; 경기 중에는 15초 지연으로 발행됐다), `ledger.decision_clock`은 `unverified`다.
DEMO-WS-MOMENTS-1029 (Song 결정 2026-10-29 "장부 값으로 진행"), 재생 코드 커밋 8678ba07(이 README와 fixture는 검토 반영 커밋에 들어갔다; 그 사이 코드 차이는
검토 지적 수정뿐이고 이 실행에서는 해당 경로가 0번 쓰여 본문은 같다), 실행 `pitcheezy-pm-jobs/20261029-demo-ws-moments/run-849803-ledger-001`
(등록 `REGISTRATION.json`, 독립 검토 `reviews/SONNET-REVIEW-001.md` PASS_WITH_NOTES, 지표 `METRICS.json` PASS). 선택 기록은 `SELECTION.json`.

| 폴더 | 규칙(실행 전 고정) | 시점 | 화면에 보이는 것 |
|---|---|---|---|
| `first-rec` | 추천이 처음 보인 주기 | 1회초 0:0, Griffin Jax ↔ Jackson Chourio, `849803:1:1` (주기 125; 첫 투구 전 Warmup 본문, 예정 시작 09:00 KST의 약 10분 전) | '다음 공 추천 · 1구' 후보 3개(스위퍼·싱커·포심) |
| `mid-rec` | 추천 주기 중 투구 수가 최종(367)의 절반에 가장 가까운 것 | 7회초 2:2, Nick Martinez ↔ Jackson Chourio, `849803:50:5` (주기 1001, 183구) | '다음 공 추천 · 5구' |
| `reference-card` | `next.reference`(A2 참고 카드)가 처음 붙은 주기 | 12회초 2:2, Ian Seymour ↔ Cooper Pratt, `849803:83:1` (주기 1614, 317구) | '추천 없음'(2025년까지 기록 없는 투수) + 참고 카드(포심 29%·스위퍼 24%·체인지업 20%·싱커 15%, 기록 기준 2026-10-25) |
| `last-rec` | 추천이 마지막으로 보인 주기 | 12회말 3:2, Antonio Senzatela ↔ Liam Hicks, `849803:89:1` (주기 1726, 352구) | '다음 공 추천 · 1구' |

보는 법 (공개 번들 `npm run build:public` 뒤):

```bash
node tests/ui-preview.mjs --ws-moment first-rec   # 이 시점만(단독), 시계 now(갱신 지연 안 뜸); --ws·--fixture와 같이 쓰면 거부
node tests/ws-moments-smoke.mjs                     # 네 시점 화면 검사
```

한 서버는 한 시점만 낸다. `--ws`(종료 경기 3개)와 겹치면 시점의 빈 날짜 파일이 다른 경기를 가리고, 합쳐진 `health.json`이 최종 파일의 순번을 담아
화면이 순번 낮은 시점 파일을 적용하지 않는다(확인: 4차전 페이지가 비고 목록이 빈 10/29로 열림). 종료 경기 화면은 `--ws`로 따로 띄운다.

정직하게 적어 둘 것:
- 장부는 "경기 중 어느 초에 어느 팬의 화면에 무엇이 떠 있었는지"까지는 말하지 않는다. 여기 추천은 "그 상황으로 장부에 기록된 추천"이다.
- 화면의 'LIVE'와 '실시간으로 받는 중 · HH:MM 수신'은 미리보기 서버의 now 시계 때문이다(재생이지 생중계가 아니다). 머리의 점수는 원정 : 홈 순서다.
- 장부에 행이 없는 상황(정책 호출 기준 27상태, 화면 시점으로는 9개)은 '추천 없음 · 경기 중 이 상황에 기록된 추천이 없어요.'로 나온다(`display_state.reason=policy_refused`,
  `detail=LEDGER_NO_ROW` — 정책이 거절한 것이 아니라 행이 없다는 뜻). 네 시점에는 해당 없음.
- 미지원 투수 2명(Seymour·Drohan)의 '추천 없음' 사유(unknown_pitcher)는 등록된 지원 맵(10/17 커버리지 재집계 `armb_static`)에서 온다.
- `pregame`은 null이라 타자·투수의 시즌 줄(ERA·타율·OPS)은 '—'다(재생기가 선수 캐시 env를 지움). 승리확률 칸은 unavailable → '미측정'.
- `games/2026-10-27.json`·`2026-10-29.json`(빈 날)도 재생기가 쓴 그대로 둔다.
- 2026 자료는 표시 전용. 공개 JSON v1 스키마는 바뀌지 않았다. 최종 경기 파일은 `../publish-ws/2026-10-28`(실발행본)을 그대로 쓴다.
