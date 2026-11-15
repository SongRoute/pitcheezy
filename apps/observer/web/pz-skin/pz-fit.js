/* pz-fit.js — 세로 폰 화면에서 경기 화면 첫 화면이 "보이는 높이" 안에 들어가게 존 높이를 맞춘다(2026-11-15, demo/front-fit-1115).
 * 기준 높이는 작은 보이는 높이(100svh: Safari 주소창·아래 막대가 보이는 상태; 모르면 100dvh, 그것도 모르면 100vh)이다.
 * 존 밖의 모든 줄(머리·점수판·상황·추천·참고 카드·실제 투구·이번 타석·하단 메뉴 자리)은 화면에 그려진 높이를 그대로 재서 뺀다.
 * 그래서 상태에 따라 줄이 늘거나 줄어도(참고 카드·공수 교대 줄 등) 상수 없이 존만 따라 줄어든다.
 *   존 높이 = clamp(LOW, 보이는 높이 − 존 밖 높이, 216px)   (216px = 지금 폰 화면의 존 높이, 상한이라 큰 화면은 지금과 같다)
 * 존이 216px보다 작아져야 할 때는 먼저 여백·간격을 줄이고(<html>.pz-fit-tight, pz-theme.css) 그 뒤 존 밖 높이를 다시 잰다.
 * LOW(150px) 아래로도 안 들어가는 아주 낮은 화면에서만 지금처럼 세로 스크롤이 생긴다.
 * 폭 760px 이상(존 330px)과 가로 화면은 건드리지 않는다. 글자·정보·문구·순서는 바꾸지 않는다(CSS 변수 하나만 쓴다).
 * 값은 <html>의 --pz-zone-fit에 쓰고, pz-theme.css가 그 값을 .sv-page.sv-game의 --sv-zone-height로 쓴다(3D 상자도 이 값을 따른다).
 * 스크립트가 없거나 실패하면 변수가 없어 지금 화면(216px) 그대로다. 외부 주소·저장소·eval을 쓰지 않는다.
 * 확인용: window.PZFit.status() */
const LOW = 150, HIGH = 216;
const MQ = typeof matchMedia === "function" ? matchMedia("(max-width: 759px) and (orientation: portrait)") : null;
const F = { applied: null, avail: null, rest: null, restLoose: null, tight: false, delta: null, runs: 0, reason: "대기" };
let probe = null;
function availHeight() {
  if (!probe) {
    probe = document.createElement("div");
    probe.setAttribute("aria-hidden", "true");
    // 뒤 선언이 앞을 덮는다: 100svh를 모르면 100dvh, 그것도 모르면 100vh
    probe.style.cssText = "position:fixed;left:0;top:0;width:0;height:100vh;height:100dvh;height:100svh;visibility:hidden;pointer-events:none;";
    document.body.appendChild(probe);
  }
  return probe.getBoundingClientRect().height;
}
const TIGHT = "pz-fit-tight"; // pz-theme.css: 이 클래스가 <html>에 있으면 경기 화면의 여백·간격만 줄인다(글자·정보 그대로)
const root = document.documentElement;
function clear(reason) {
  if (F.applied !== null) root.style.removeProperty("--pz-zone-fit");
  if (root.classList.contains(TIGHT)) root.classList.remove(TIGHT);
  F.applied = null; F.tight = false; F.delta = null; F.reason = reason;
}
/** 존 밖 높이: 문서 맨 위에서 페이지 내용 끝(아래 여백 = 하단 메뉴 자리 포함)까지에서 지금 존 높이를 뺀다.
 *  페이지의 min-height(100dvh)가 아니라 자식들의 아래 끝을 쓴다 — 내용이 짧아도 정확한 존 밖 높이가 나온다. 존 높이와 무관하다.
 *  검토 R4 P2-2(PM 결정): 관람자가 "이번 타석 펼치기"로 연 목록은 첫 화면이 아니므로 빼고 잰다 — 펼쳐도 존·3D는 그대로이고
 *  c47처럼 페이지만 길어져 스크롤한다. 열린 시트(자세히 등)는 position: fixed라 원래 측정에 들어가지 않는다. */
function restOf(page, zrow) {
  const padB = parseFloat(getComputedStyle(page).paddingBottom) || 0;
  let bottom = 0;
  for (const c of page.children) { const cs = getComputedStyle(c); if (cs.position === "fixed" || cs.position === "absolute" || cs.display === "none") continue; const b = c.getBoundingClientRect().bottom + (parseFloat(cs.marginBottom) || 0); if (b > bottom) bottom = b; }
  let opened = 0;
  const list = page.querySelector(".sv-atbat-list"), head = list && list.parentElement && list.parentElement.querySelector(".sv-atbat-head");
  if (list && head) opened = Math.max(0, list.getBoundingClientRect().bottom + (parseFloat(getComputedStyle(list).marginBottom) || 0) - head.getBoundingClientRect().bottom);
  return bottom + window.scrollY + padB - zrow.getBoundingClientRect().height - opened;
}
function setTight(on) { if (root.classList.contains(TIGHT) !== on) root.classList.toggle(TIGHT, on); }
/* 순서: (1) 여백 그대로 존 216px이 들어가면 지금 화면 그대로. (2) 아니면 여백·간격을 먼저 줄이고(TIGHT), (3) 그래도 모자라면 존을 하한(LOW)까지 줄인다.
   (4) 하한에서도 넘치면 그 만큼만 세로 스크롤. 다시 재는 것은 같은 마이크로태스크 안이라 그리기 전에 끝난다.
   검토 R4 P2-1: 클래스·변수는 값이 바뀔 때만 쓴다. 이미 여백을 줄인 상태면 줄인 채로 재고, 줄이기 전 높이는 지난번에 잰 차이(delta)로 어림해
   "풀어도 216이 들어갈 수 있을 때만" 풀어서 다시 잰다 — 그래서 가만히 있는 화면에서는 <html>을 건드리지 않는다. */
function fit() {
  F.runs++;
  const page = document.querySelector(".sv-page.sv-game"), zrow = page && page.querySelector(".sv-zrow");
  if (!page || !zrow) return clear("경기 화면 아님");
  if (MQ && !MQ.matches) return clear("넓은 화면·가로 화면");
  const avail = availHeight();
  if (!(avail >= 200 && avail <= 4000)) return clear(`보이는 높이 비정상(${avail})`); // 검토 R4 P3-2: 탐침이 0 등 이상하면 아무것도 바꾸지 않는다
  const wasTight = root.classList.contains(TIGHT);
  let loose, rest, tight;
  try {
    if (wasTight && F.delta !== null) {
      rest = restOf(page, zrow); loose = rest + F.delta; tight = true;
      if (avail - loose >= HIGH) { setTight(false); loose = restOf(page, zrow); rest = loose; tight = false; if (avail - loose < HIGH) { setTight(true); rest = restOf(page, zrow); F.delta = loose - rest; tight = true; } }
    } else {
      setTight(false); loose = restOf(page, zrow); rest = loose; tight = false;
      if (avail - loose < HIGH) { setTight(true); rest = restOf(page, zrow); F.delta = loose - rest; tight = true; }
    }
  } catch (e) { setTight(wasTight); throw e; } // 검토 R4 P3-5: 재는 도중 예외면 클래스를 원래대로
  const want = Math.max(LOW, Math.min(HIGH, Math.floor(avail - rest)));
  F.avail = Math.round(avail * 10) / 10; F.rest = Math.round(rest * 10) / 10; F.restLoose = Math.round(loose * 10) / 10; F.tight = tight;
  F.reason = avail - rest < LOW ? "하한(스크롤 남음)" : tight ? "여백 줄임 + 맞춤" : "그대로";
  if (F.applied === want) return;
  root.style.setProperty("--pz-zone-fit", want + "px");
  F.applied = want;
}
let queued = false;
const run = () => { queued = false; try { fit(); } catch (e) { F.reason = "오류: " + (e && e.message ? e.message : e); } };
// 마이크로태스크로 모아서 한 번: MutationObserver 콜백 안이라 그리기 전에 끝난다(첫 화면이 216px로 한 번 그려졌다 줄어드는 깜빡임 방지)
const later = () => { if (!queued) { queued = true; queueMicrotask(run); } };
// 3D 상자(.pz-bat3d, position:absolute) 안·자체의 변화는 높이와 무관하므로 무시한다(검토 R4 P2-1: 반응·hidden 쓰기로 다시 재지 않게)
const in3d = (n) => { const e = n && (n.nodeType === 1 ? n : n.parentElement); return !!(e && e.closest && e.closest(".pz-bat3d")); };
new MutationObserver((recs) => { if (recs.some((r) => !in3d(r.target))) later(); })
  .observe(document.body || document.documentElement, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ["class", "open", "aria-expanded"] });
addEventListener("resize", later);
if (MQ && MQ.addEventListener) MQ.addEventListener("change", later);
if (document.fonts && document.fonts.ready) document.fonts.ready.then(later, () => {});
later();
window.PZFit = Object.freeze({ status: () => ({ ...F }) });
