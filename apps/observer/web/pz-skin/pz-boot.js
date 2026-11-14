/* pz-boot.js — 3D 타자 코드를 경기 화면에서만 불러오는 작은 시작 스크립트(2026-11-15, 수정 ③).
 * 팀원 index.html은 pz-batter.js를 정적 모듈 태그로 넣어 batter3d.js와 three.js(압축 전 약 2.35 MB)를 목록 화면에서도 받았다.
 * 이 파일은 그 자리에 들어가며, 경기 화면의 존 행(.sv-game .sv-zrow)이 처음 나타날 때 한 번만 pz-batter.js를 동적 import한다.
 * 불러오기에 실패하면 다시 시도하지 않는다(실루엣이 그대로 남는다). 외부 주소·저장소·eval을 쓰지 않는다.
 * 확인용: window.PZSongSkinBoot.status() → { requested, loaded, error } */
const SEL = ".sv-game .sv-zrow";
const B = { requested: false, loaded: false, error: null };
function go() {
  if (B.requested) return;
  B.requested = true; mo.disconnect();
  import("./pz-batter.js").then(() => { B.loaded = true; }, (e) => { B.error = String(e && e.message ? e.message : e); });
}
const mo = new MutationObserver(() => { if (document.querySelector(SEL)) go(); });
mo.observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
if (document.querySelector(SEL)) go();
window.PZSongSkinBoot = Object.freeze({ status: () => ({ ...B }) });
