# 전시 고정 빌드 기록 — 849843 · svc/ui-1026-main fba6b2ff (2026-10-31)

## 1. 무엇
- 경기: 849843, 2026 NL 와일드카드 1차전 CHC @ SD (현지 2026-09-29, KST 2026-09-30 11:00 시작). 262구, 추천 246구(ready)·미지원 16구(Ryan Rolison, 2025년까지 학습 기록 없음).
- 화면: UI-1026(d0d25eda) + 안내 숨김 수정(ui/notice-hide-1021 e3e3cbe0) + ARM-B off 경보 + DEMO-WS-GAPS-1028 포팅 + 독립 검토 수정. 브랜치 `svc/ui-1026-main`, 커밋 `fba6b2ff`, 작업 트리 clean, PR #43 → main.
- 발행 파일: `v1/` (dates, manifest(commit fba6b2ff, dirty false), health, games/ 5개 날짜, game/*.json 15경기, setup/849843.json). 발행기 오프라인 실행(보관 피드 archive 모드), 동결 ARM-B 정책, 추천 장부 켬, 업로드 OFF, 참고 카드 프로필 포인터 연결(컷오프 2026-09-28, usable).
- 묶음: 맥미니 `~/Projects/pitcheezy-pm-jobs/exhibition-849843-1031/bundle-849843-fba6b2ff/` (site/ = dist-public + v1, 28파일 11,992,694 bytes), `MANIFEST.tsv`(path, bytes, sha256_raw, sha256_lf_normalized), `TOOLS.txt`(node v25.9.0, npm, python).
- ZIP: `bundle-849843-fba6b2ff.zip` **867,682 bytes, SHA-256 `af10293d5134219b7f750d00786366b93b0134c895519b939fd99a356d77a4bc`** (원본 바이트, zip -X).

## 2. 어떻게 (재현)
```sh
# 맥미니, 워크트리 ~/Projects/pitcheezy-wt/integrate-main-1031 @ fba6b2ff
cd apps/observer/web && npm ci && npm run build:public
# 발행 (보관 피드 → archive, 1회). <job> = ~/Projects/pitcheezy-pm-jobs/exhibition-849843-1031
env -i HOME=$HOME USER=$USER LANG=ko_KR.UTF-8 LC_ALL=ko_KR.UTF-8 TMPDIR=$TMPDIR PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  PYTHONPATH=apps/observer/backend PYTHONUNBUFFERED=1 \
  PITCHEEZY_OBSERVER_LIVE_STORE=<job>/store4.sqlite3 PITCHEEZY_OBSERVER_LIVE_ARMB=1 PITCHEEZY_OBSERVER_LIVE_DELAY_S=0 \
  PITCHEEZY_OBSERVER_REC_LEDGER=1 PITCHEEZY_OBSERVER_LEGACY=0 PITCHEEZY_OBSERVER_VIEWER_DELAY=0 PITCHEEZY_OBSERVER_STEP5_EXPECTATION=0 \
  PITCHEEZY_FEED_DIR="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/feeds" \
  PITCHEEZY_WATCH_DIR="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/watch" \
  PITCHEEZY_INTENT_DIR="/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/intent" \
  PITCHEEZY_DEMO_DIR=<job>/empty/demo PITCHEEZY_RUNS_DIR=~/Projects/pitcheezy/runs PITCHEEZY_DATA_DIR=~/Projects/pitcheezy/data \
  PITCHEEZY_PLAYER_SNAPSHOT_DIR=~/Projects/pitcheezy-pm-jobs/20261020-ws-pregame-cache-001/cache \
  PITCHEEZY_OBSERVER_PROFILE_SOURCES=<job>/profiles/LATEST.json \
  .venv-observer/bin/python -B -m observer_app.publisher --out <job>/out4 --once --date 2026-09-30 --config apps/observer/publish_config.json
```
- 프로필(참고 카드): `refresh_profile.py run --target 2026-09-29 --base d20260930-h2026f`(incremental 없음, 코드 codex-profile-parent-guard-1013 6f50798) → `<job>/profiles/target-2026-09-29`, profile_sha256 `f64abb8511fa071ebcd3561e7634a76ead8fbaf8e3db4cda2f5f78178016b7f3`, 컷오프 2026-09-28(엄격 선행, stale 아님). 최종 경기 파일에는 `next`가 없어 카드가 실리지 않는다(재생 경로에서만, 6절).

## 3. 검증 (커밋 fba6b2ff)
| 검사 | 결과 |
|---|---|
| tsc -b | 오류 0 |
| notice-hide-smoke | 267 ok, 0 FAIL |
| ws-launch-layout-check | 360 passed |
| zone-geometry-check | 397 ok |
| zone-label-clip-check | 160 ok |
| static-publish-smoke | 1~3절 64 ok (4절 이후 UI-1026 미포팅) — 32e79c2f 기준 |
| apps/observer/ops tests | 128 passed (ARM-B off 규칙·세그먼트 경계 테스트 포함) |
| ledger_policy / replay·publisher 관련 | 9 + 27 passed — 32e79c2f 기준(이후 변경 없음) |
| 정적 서버 스크린샷(폰 390×844·데스크톱) | 849843 1구 화면, 순위 사각형 UI-1026 디자인, 페이지 오류 0 |
| 팀원 #38 긴급 안내 3항목 | notice-hide-smoke 측정: 열린 화면에서 안내 ON 시 추천 행·존 표시·타석 목록 숨김 ✓, 정상 응답 뒤 새로고침 없이 복원 ✓, 안내 뒤 일반 오류에서 과거 추천 미노출 ✓(sticky hold, 코드 검토 기준) |

## 4. 독립 검토
- 작성: Claude Fable 5.1 (Claude Code). 검토: **Claude Opus 5.5**, 정적 검토(테스트 미실행), 대상 32e79c2f, 결과 `review/OPUS-REVIEW-svc-ui-1026-main-32e79c2f.md` → **BLOCK**.
  - P1-1 service.css 병합 사고(옛 스타일시트 702줄이 UI-1026 @import 아래 재등장, 순위 사각형·스테이지 디자인 되돌림) → fba6b2ff에서 3줄 복원, `.sv-alert p` 규칙은 service-layout.css로. zone-geometry 397 ok·스크린샷으로 확인.
  - P1-2 선수 시트 MixTable asOfIndex 누락 → 전달.
  - P2-1 안내 뒤 비안내 오류가 이전 화면을 다시 그림 → sticky hold(정상 응답 적용 시만 해제).
  - P2-2 정적 모드에서 health만 200·게임 파일만 안내일 때 → 보류 중 health 단독 해제·조기 반환 금지(호스트가 경로별 안내를 주는지는 레포 밖이라 미확인).
  - P2-3 경보 규칙이 세그먼트 사이 거짓 복구·300초 침묵 → running 세그먼트에서만 평가, 테스트 확장.
  - P3-7 안내 시 상태 팝업 → 닫음. 나머지 P3(미사용 CSS/Zone export, api.ts 주석, asOfWords 문구, 보류 집계 표현, pasAsOf last=null)는 미수정으로 남김.
  - fba6b2ff에 대한 재검토는 하지 않았다(수정은 지적 항목에 한정, 검사는 3절).
- Codex gpt-6-astra(CLI 0.166.0): 두 번 모두 "timed out negotiating with the code-mode host"로 명령 실행 불가 → 검토 불성립(`review/ASTRA-attempt1-failed.md`, `codex2.err`). 같은 프롬프트 `review/PROMPT.md`를 Codex 앱에서 돌릴 수 있다.

## 5. 남은 조건·미측정
- 실기기(iPhone Safari) 검수 미실시. 데스크톱은 폰 레이아웃 가운데 정렬(UI-1026 설계).
- 헤더 시리즈 문구·demo PitchScatter·ws-gaps/ws-moments/static-publish 4절+ 스모크는 UI-1026 미포팅.
- 발행 계약 v1·업로드 API 변경 없음. 정적 호스팅만 전제(팀원 Cloudflare 호스트).

## 6. 모니터 1 재생(포스터 흐름 전체) — 확인 완료
- 발행기 재생 모드로 공마다 `recommended`(reason `ledger_recorded`)가 발행되고, 미지원 투수 Rolison 구간(5회말 138~145구)에는 `next.reference` 참고 카드(as_of 2026-09-28, stale false)가 실린다. 300초 관찰: recommended 69회, 참고 카드 27회.
- 방법: ① `demo_precompute.py live-replay --game-pk 849843 --start 0 --count 262 --output <dir>`(T7) ② `<job>/stamp_snapshots.py <dir> <stamped> 25` — demo 스냅샷의 `metaData.timeStamp`가 `rehearsal_NNNN`이라 발행기의 제공자 순서 게이트가 `unorderable_provider_stamp`로 추천을 보류하므로 단조 증가 도장을 부여(demo_precompute에 넣을 수정 후보) ③ 2절 env에 `PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR=<stamped> PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S=20 PITCHEEZY_FEED_DIR=<빈 폴더>`, `--interval 10`(--once 없이) `--date 2026-09-30`; 공개 빌드를 `out/v1` 옆에 두고 `./?game=849843`. 한 구 약 10초.
- 라이브 경로 `/live?game=849843&replay=<start>&speed=<x>`(t0 전달)도 공마다 바뀌지만 참고 카드는 없다.
