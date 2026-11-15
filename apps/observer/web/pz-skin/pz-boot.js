/* pz-boot.js — 3D 타자 코드를 경기 화면에서만 불러오는 작은 시작 스크립트(2026-11-15, 수정 ③).
 * 팀원 index.html은 pz-batter.js를 정적 모듈 태그로 넣어 batter3d.js와 three.js(압축 전 약 2.35 MB)를 목록 화면에서도 받았다.
 * 이 파일은 그 자리에 들어가며, 경기 화면의 존 행(.sv-game .sv-zrow)이 처음 나타날 때 한 번만 pz-batter.js를 동적 import한다.
 * 불러오기에 실패하면 다시 시도하지 않는다(실루엣이 그대로 남는다). 외부 주소·저장소·eval을 쓰지 않는다.
 * 검토 P2-3(2026-11-15): 빌드에서 모델 이름(PZ_BATTER_MODEL)을 받지 않았으면(자리표시 그대로) 3D 코드를 아예 부르지 않는다 — 요청 0, 실루엣 유지.
 * 맞춤(pz-fit.js)은 맨 위에서 정적으로 불러온다(목록 화면에서는 아무것도 하지 않는다).
 * 확인용: window.PZSongSkinBoot.status() → { requested, loaded, error, skipped } */
import "./pz-fit.js"; // 세로 폰 화면 맞춤(2026-11-15, demo/front-fit-1115): 모델 유무와 무관하게 경기 화면 존 높이를 보이는 높이에 맞춘다
const MODEL = "__MODEL__"; // 빌드(pz-skin/plugin.mjs)가 pz-batter.js와 같은 이름으로 바꾼다
const SEL = ".sv-game .sv-zrow";
const HAS_MODEL = /^[a-z0-9][a-z0-9-]*$/.test(MODEL); // 자리표시(밑줄)는 통과하지 못한다
const B = { requested: false, loaded: false, error: null, skipped: HAS_MODEL ? null : "모델 이름 없음" };
function go() {
  if (B.requested || !HAS_MODEL) return;
  B.requested = true; mo.disconnect();
  import("./pz-batter.js").then(() => { B.loaded = true; }, (e) => { B.error = String(e && e.message ? e.message : e); });
}
const mo = new MutationObserver(() => { if (document.querySelector(SEL)) go(); });
if (HAS_MODEL) mo.observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
if (document.querySelector(SEL)) go();
window.PZSongSkinBoot = Object.freeze({ status: () => ({ ...B }) });
