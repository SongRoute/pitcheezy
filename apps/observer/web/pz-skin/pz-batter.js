/* pz-batter.js — Song 고정 빌드(c47fe6cb)의 경기 화면에 우리 3D 타자(batter3d 모듈)를 "덧붙이는" 호스트(2026-11-09, V3).
 * Song 코드는 한 줄도 바꾸지 않는다. 이 모듈은 Song React 앱이 그린 DOM을 관찰해서
 *   (1) 경기 화면의 존 행(.sv-zrow)이 나타나면 그 안에 상자(.pz-bat3d)를 넣고 3D 타자를 띄운다(실루엣은 CSS로 숨김, 틀 불변),
 *   (2) 타석 방향은 .sv-zrow의 bat-l / bat-r 클래스로 따라가고,
 *   (3) '실제 투구' 줄(.sv-actual-line)의 글이 바뀌면(다음 공으로 넘어가면) 결과 글에 따라 반응을 재생한다:
 *       볼·루킹 스트라이크·몸에 맞는 공 → take(한 발 물러남), 헛스윙 → miss, 파울·파울팁·아웃·실책 → contact, 1루타·2루타·3루타·홈런(Song 표기; '안타'도 받음) → hit(치고 달려 나감).
 *       Song 화면에는 공 비행 연출이 없으므로 고정 길이(LEAD_MS)의 와인드업 뒤 접촉 시점이 온다. 끝나면 대기 동작으로 섞여 돌아간다.
 *   (4) 존 행이 사라지면(목록 화면 등) 3D를 해제한다. 로딩·WebGL 실패 시에는 아무것도 바꾸지 않는다(실루엣 그대로).
 * 결과 판정은 Song이 이미 화면에 쓴 글을 읽을 뿐이고(자료·추천·위치는 건드리지 않음), 감속 모션 설정이면 정지 자세만 보여 준다.
 * 검증용: window.PZSongSkin.status()
 */
import { createBatter3D } from "./batter3d/batter3d.js";

const MODEL = "__MODEL__"; // build-songskin.mjs가 batter-ctl | batter-indexed 로 바꾼다
const ASSETS = new URL("./batter3d/assets/", import.meta.url).href; // 문서 경로(/, /?game=…)와 무관하게 이 모듈 옆
const VIEW = { zoom: 0.8 }; // relay 호스트와 같은 카메라 보정(140×210 틀)
const LEAD_MS = 520; // 와인드업(공 비행 대신) 길이
const REACT_DELAY_MS = 250; // 안 치는 공: 이만큼 뒤에 반응 시작
const HIT_FADE_FROM = 0.86; // hit 끝부분 투명 처리 시작 진행률
const FALLBACK_PLAY = { contact: { clip: "contact_hit1", cue: 1.456042, end: 2.1 }, miss: { clip: "miss", cue: 1.198889, end: 2.45 } };

const S = { b: null, box: null, row: null, side: null, play: null, lastKey: null, raf: 0, busy: false, fails: 0, last: "대기", played: 0, ro: null };
const reduceMotion = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

function resolvePlay(meta) {
  const mp = meta && meta.play; if (!mp || typeof mp !== "object") return { ...FALLBACK_PLAY };
  const ok = (p) => p && typeof p.clip === "string" && Number.isFinite(p.cue) && Number.isFinite(p.end) && p.cue < p.end;
  const out = {};
  for (const k of ["contact", "miss"]) { if (!ok(mp[k])) return { ...FALLBACK_PLAY }; out[k] = { clip: mp[k].clip, cue: mp[k].cue, end: mp[k].end }; }
  for (const k of ["take", "hit"]) if (ok(mp[k])) out[k] = { clip: mp[k].clip, cue: mp[k].cue, end: mp[k].end };
  return out;
}
function checkPlay(play, clips) {
  for (const k of Object.keys(play)) { const p = play[k]; if (!(clips[p.clip] > 0) || !(p.end <= clips[p.clip] + 1e-6)) { if (k === "take" || k === "hit") delete play[k]; else return false; } }
  return true;
}

/** Song이 쓴 '실제 투구' 줄에서 반응 종류를 읽는다. 표식 원(.sv-g1)의 분류 + 결과 글의 낱말 */
export function kindFromActual(line) {
  if (!line) return null;
  const g1 = line.querySelector(".sv-g1"), cls = g1 ? g1.className : "", text = line.textContent || "";
  if (/\bball\b/.test(cls)) return "take";
  if (/\bstrike\b/.test(cls)) {
    if (/헛스윙/.test(text)) return "miss";
    if (/파울/.test(text)) return "contact"; // 파울·파울팁
    return "take"; // 루킹 스트라이크·루킹 삼진 등
  }
  if (/\binplay\b/.test(cls)) return /홈런|안타|1루타|2루타|3루타/.test(text) && !/아웃|실책|병살|선택/.test(text) ? "hit" : "contact"; // Song은 단타를 '1루타'로 쓴다(검토 지적)
  return null; // unknown·확인 중·자료 없음
}
/** 투구 식별 열쇠: 라벨("실제 투구 · N구 · 이름") + 표식 원 분류·번호 + 구종 + 결과 글. 속도(<small>, km/h↔mph 단위 전환)와 '정정됨' 태그는 뺀다 — 같은 공을 다시 재생하지 않게(검토 지적) */
const keyOf = (line) => {
  if (!line) return "";
  const t = (sel) => { const e = line.querySelector(sel); return e ? (e.textContent || "").replace(/\s+/g, " ").trim() : ""; };
  const g1 = line.querySelector(".sv-g1");
  const res = [...line.querySelectorAll(":scope > span")].filter((s) => !s.classList.contains("sv-al-label") && !s.classList.contains("sv-tag")).map((s) => (s.textContent || "").replace(/\s+/g, " ").trim()).join(" ");
  return [t(".sv-al-label"), g1 ? g1.className : "", g1 ? (g1.textContent || "").trim() : "", t(".sv-actual-name"), res].join("|");
};

function stopRaf() { if (S.raf) cancelAnimationFrame(S.raf); S.raf = 0; if (S.box) { S.box.style.transition = "none"; S.box.style.opacity = ""; } }
function run(ms, frame, done) {
  const t0 = performance.now();
  const step = (now) => { const k = document.hidden ? 1 : Math.min(1, (now - t0) / ms); frame(k); if (k < 1) S.raf = requestAnimationFrame(step); else { S.raf = 0; done(); } };
  S.raf = requestAnimationFrame(step);
}
function setOpacity(v, soft) { if (!S.box) return; S.box.style.transition = soft ? "opacity .35s ease" : "none"; S.box.style.opacity = v >= 1 ? "" : String(Math.max(0, Math.min(1, v))); }

/** 한 공의 반응 재생. relay 호스트(batter3d-host.js swing/react)와 같은 시간 규칙: 앞은 LEAD_MS에 맞춰 cue까지, 뒤는 원속도로 end까지 */
function playKind(kind) {
  const b = S.b; if (!b || !b.ok || !kind) return;
  stopRaf();
  if (reduceMotion()) { b.reset({ still: true }); return; }
  const clips = b.getState().clips, P = S.play;
  if (kind === "take") {
    const p = P.take; if (!p) { b.reset(); return; }
    const d = clips[p.clip], startMs = LEAD_MS + REACT_DELAY_MS, playMs = (p.end - p.cue) * 1000, ms = startMs + playMs; let primed = false;
    run(ms, (u) => { const tau = u * ms; if (tau < startMs) { if (!primed) { primed = true; b.reset(); } return; } b.setSwing(p.clip, Math.min(1, (p.cue + (tau - startMs) / 1000) / d)); }, () => b.reset());
    S.played++; S.last = "take"; return;
  }
  const key = kind === "miss" ? "miss" : kind === "hit" && P.hit ? "hit" : "contact";
  const p = P[key], d = clips[p.clip], followMs = (p.end - p.cue) * 1000, ms = LEAD_MS + followMs, fade = key === "hit";
  run(ms, (u) => {
    const tau = u * ms, t = tau <= LEAD_MS ? p.cue * (tau / LEAD_MS) : p.cue + (tau - LEAD_MS) / 1000;
    b.setSwing(p.clip, Math.min(1, t / d));
    if (fade) setOpacity(u < HIT_FADE_FROM ? 1 : (1 - u) / (1 - HIT_FADE_FROM), false);
  }, () => { b.reset(); if (fade) setOpacity(1, true); });
  S.played++; S.last = key;
}

async function mount(row) {
  if (S.busy || S.fails >= 3) return;
  S.busy = true;
  const box = document.createElement("div"); box.className = "pz-bat3d"; box.setAttribute("aria-hidden", "true");
  const side = row.classList.contains("bat-r") ? "R" : "L";
  if (side === "R") box.classList.add("side-r");
  row.append(box);
  let b = null;
  try { b = await createBatter3D(box, { assetsBase: ASSETS, modelName: MODEL, side, theme: "light", view: VIEW }); }
  catch (e) { box.remove(); S.busy = false; S.fails++; S.last = `로딩 실패: ${e && e.message ? e.message : e}`; return; }
  const st = b.getState(), play = resolvePlay(b.meta);
  if (!b.ok || b.mode !== "webgl" || !checkPlay(play, st.clips)) { try { b.dispose(); } catch (e) { /* 무시 */ } box.remove(); S.busy = false; S.fails++; S.last = `초기화 실패: ${st.mode}${st.error ? " " + st.error : ""}`; return; }
  if (!row.isConnected || !box.isConnected) { try { b.dispose(); } catch (e) { /* 무시 */ } box.remove(); S.busy = false; schedule(); return; } // 로딩 사이에 화면이 바뀜 → 다음 틱에서 지금 행에 다시(검토 지적: 재시도 예약 누락)
  S.b = b; S.box = box; S.row = row; S.side = side; S.play = play; S.lastKey = keyOf(row.closest("main") ? row.closest("main").querySelector(".sv-actual-line") : null);
  row.classList.add("pz-has3d"); hideSilhouette(row, true);
  S.ro = typeof ResizeObserver === "function" ? new ResizeObserver(() => { if (S.b === b) b.resize(); }) : null; if (S.ro) S.ro.observe(box);
  const canvas = box.querySelector("canvas"); if (canvas) canvas.addEventListener("webglcontextlost", () => { if (S.b === b) { S.fails++; unmount("WebGL 문맥 상실"); } }, false); // 문맥 상실도 실패로 세어 무한 재시도(재다운로드) 방지(검토 지적)
  if (reduceMotion()) b.reset({ still: true });
  S.busy = false; S.last = "연결됨";
}
/** 실루엣 svg 숨김/복원. 클래스(pz-has3d) 규칙과 겹치지만, React가 클래스를 지우는 한 프레임 동안도 비치지 않게 인라인으로도 둔다 */
function hideSilhouette(row, hide) {
  for (const svg of row.querySelectorAll(".sv-batband svg")) { if (hide) { if (svg.style.visibility !== "hidden") svg.style.visibility = "hidden"; } else if (svg.style.visibility === "hidden") svg.style.visibility = ""; }
}
function unmount(reason) {
  stopRaf();
  if (S.ro) { try { S.ro.disconnect(); } catch (e) { /* 무시 */ } S.ro = null; }
  if (S.row) { S.row.classList.remove("pz-has3d"); hideSilhouette(S.row, false); }
  if (S.b) { try { S.b.dispose(); } catch (e) { /* 무시 */ } }
  if (S.box && S.box.parentNode) S.box.remove();
  S.b = null; S.box = null; S.row = null; S.side = null; S.play = null; S.lastKey = null; S.last = reason || "해제";
}

let scheduled = 0;
function tick() {
  scheduled = 0;
  const row = document.querySelector(".sv-game .sv-zrow");
  if (!row) { if (S.b) unmount("존 행 없음(다른 화면)"); return; }
  if (S.b && (S.row !== row || !S.box.isConnected)) unmount("존 행 교체");
  // Song이 타자 방향을 모를 때(경기 전·타자 교체 사이)는 실루엣 띠(.sv-batband) 자체를 그리지 않는다 → 우리도 3D를 숨기고 반응하지 않는다(검토 지적)
  const band = row.querySelector(".sv-batband");
  if (!band) { if (S.box) S.box.hidden = true; return; }
  if (!S.b) { mount(row); return; }
  if (S.box.hidden) { S.box.hidden = false; S.b.resize(); }
  // React가 .sv-zrow의 className을 통째로 다시 쓰면(bat-l ↔ bat-r) 우리가 붙인 pz-has3d가 지워진다 → 틱마다 되살리고, 실루엣 svg는 인라인 style로도 숨긴다(11/9 확인: 우타자로 바뀐 뒤 실루엣이 비쳐 보였음)
  if (!row.classList.contains("pz-has3d")) row.classList.add("pz-has3d");
  hideSilhouette(row, true);
  const side = row.classList.contains("bat-r") ? "R" : "L";
  if (side !== S.side) { S.side = side; S.box.classList.toggle("side-r", side === "R"); stopRaf(); S.b.setSide(side); S.b.reset(reduceMotion() ? { still: true } : undefined); }
  const line = document.querySelector(".sv-game .sv-actual-line"), key = keyOf(line);
  if (key !== S.lastKey) { const first = S.lastKey === null; S.lastKey = key; if (!first && key) playKind(kindFromActual(line)); }
}
const schedule = () => { if (!scheduled) scheduled = requestAnimationFrame(tick); };
const mo = new MutationObserver(schedule);
mo.observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"], characterData: true });
// 탭이 가려지면 진행 중 반응을 끊고 바로 대기 자세로(끊기만 하면 setSwing 자세에서 멈춘 채 남는다; 검토 지적)
document.addEventListener("visibilitychange", () => { if (document.hidden) { stopRaf(); if (S.b && S.b.ok) S.b.reset(reduceMotion() ? { still: true } : undefined); } });
schedule();

window.PZSongSkin = Object.freeze({
  status: () => ({ attached: !!S.b, model: MODEL, side: S.side, last: S.last, played: S.played, lastKey: S.lastKey, play: S.play, fails: S.fails, state: S.b ? S.b.getState() : null }),
  kindFromActual,
});
