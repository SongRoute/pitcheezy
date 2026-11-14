# pz-skin — 팀원 색 테마와 3D 타자를 우리 공개 화면 빌드에 얹는 소스

출처: 팀원 저장소 `Pitcheezy/transition-models` 가지 `claude/exhibit-records-1112` 커밋 `10988ce65f5e3ecd33363b3f960fccbdb5abd9fc`
(PR 90, `web/pitch-studio-home-mobile/relay/tools/songskin/`와 `relay/batter3d/`). 통합 기록: `docs/pm/service/front-3d-integration-2026-11-15.md`.

## 빌드

`npm run build:public`(와 `npm run build`)이 `pz-skin/plugin.mjs`로 팀원이 돌려 본 배치를 그대로 만든다: 빌드 결과물 옆 `pz/` 폴더 + index.html 태그 세 가지
(import map, `pz/pz-theme.css` 링크, `pz/pz-boot.js` 모듈 스크립트).

| 환경 변수 | 값 | 결과 |
|---|---|---|
| `PZ_SKIN` | `3d`(기본) | 색 + 3D 타자. `pz/pz-theme.css`, `pz/pz-boot.js`, `pz/pz-batter.js`, `pz/batter3d/batter3d.js`, `pz/batter3d/vendor/**`(three.js 0.187.0) |
| | `theme` | 색만(팀원 `theme-only/`). `pz/pz-theme.css` 하나와 CSS 링크 하나 |
| | `off` | 손대지 않은 빌드(index.html·자산이 원래 빌드와 바이트까지 같다) |
| `PZ_BATTER_MODEL` | `batter-ctl`(사람형) / `batter-indexed`(마네킹) | `pz-batter.js`의 `__MODEL__`을 이 이름으로 바꾼다. 비우면 `__MODEL__` 그대로(404 → 실루엣 유지) |
| `PZ_BATTER_ASSETS_DIR` | 저장소 밖 폴더 | `<모델>.glb`(+`<모델>.meta.json`)을 `pz/batter3d/assets/`로 복사. **모델·설명 파일은 저장소에 넣지 않는다** |

`vite` 개발 서버에는 얹지 않는다(빌드 전용).

## 파일

| 파일 | 팀원 원본 SHA-256 | 바꾼 것 |
|---|---|---|
| `pz-theme.css` | `94b79728…6fdc463` | 수정 ② 타격 주황 대비, 수정 ④ 3D 상자 자리 |
| `theme-only/pz-theme.css` | `4badfb14…74208fd3` | 수정 ② |
| `pz-batter.js` | `728aa0a4…9889b91cb` | 수정 ① 탭 복귀 재생 |
| `pz-boot.js` | (새 파일) | 수정 ③ 경기 화면에서만 3D 코드 불러오기 |
| `batter3d/batter3d.js` | `66a3664c…8b75d0` | 바꾸지 않음 |
| three.js 0.187.0 | npm `three@0.187.0`(package-lock) | 빌드 때 `node_modules/three`에서 복사, 팀원 표의 SHA-256과 대조(다르면 빌드 실패) |

바뀐 뒤 해시는 통합 기록 문서에 있다.
