/* batter3d.js — 3D 타자 표시 모듈 (로컬 시안, 2026-11-08)
 * 담당: 대기(idle) / 타격 스윙(contact) / 헛스윙(miss) / 안 치는 공 반응(take) / 안타·홈런 타격(hit, 치고 달려 나감) 자세를 "진행률 k(0..1)"로 그리고, 좌우 타석·테마·reset을 받는다.
 * 하지 않는 것: 공, 존, 비행 시간, 결과 공개 판단. 스윙·반응의 시계는 호스트 하나다(호스트가 setSwing(kind, k)를 매 프레임 부르고, 끝나면 reset()).
 * 예외(11/9): 대기(idle) 반복만 reset() 뒤 자체 rAF로 돈다(30 fps 상한; setSwing·dispose·문맥 상실·reset({still:true})·감속 모션 설정에서 멈춤). 그 밖의 rAF·setTimeout·Promise 체인은 로딩(createBatter3D) 밖에서 쓰지 않는다.
 * 모든 공개 메서드는 동기이며 예외를 던지지 않는다(내부에서 잡고 ok=false). 렌더는 호출이 있을 때와 대기 반복 중에만 한다.
 * 섞기(11/9): 대기↔클립 전환은 모듈이 직접 가중치를 선형으로 옮긴다(startBlend/stepBlend). three.js fadeIn/fadeOut은 항상 0/1에서 다시 시작해 섞기 도중 끊기면 자세가 튄다(독립 검토 지적).
 * 단위 2(11/8): WebGL 초기화 실패 뒤에도 getState·sample이 예외 없이 답한다. 배트 재생성은 이전 geometry와 공유 material을 한 번씩만 해제한다.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const SWING_KINDS = new Set(["contact", "miss", "contact_hit", "contact_hit1", "contact_hit2", "take", "hit", "run_r", "run_l", "strafe_r", "strafe_l"]);
const EXIT_KINDS = new Set(["run_r", "run_l", "strafe_r", "strafe_l"]); // 나가기 동작(11/9 V4): 출루 = 달리기, 아웃 = 옆걸음. 골반이 틀 밖으로 움직이므로 그림자가 따라가고, 대기로 돌아갈 때는 컷 // take(11/9): 안 치는 공 뒤 한 발 물러나는 반응(Mixamo Step Out을 ctl 뼈대로 옮긴 클립). hit(11/9): 안타·홈런 — 치고 1루 쪽으로 달려 나감(Mixamo Baseball Hit 5)
const BLEND_IN = 0.3; // 대기 반복 → 호스트 진행 클립으로 넘어갈 때 섞는 시간(s). take의 첫 자세는 다른 캐릭터의 서 있는 자세라 컷으로 넘기면 손·발이 15~19 cm 튄다(11/9 Node 측정; hit도 18~19 cm)
const FADE_BACK = 0.35, FADE_BACK_TAKE = 0.6, FADE_BACK_HIT = 0; // 클립 끝 → 대기로 돌아가는 섞기(s). take는 끝 자세가 25 cm쯤 물러나 있어 조금 길게. hit는 끝에 타자가 틀 밖(골반 -134 cm)이라 컷(0): 섞으면 틀을 가로질러 미끄러져 돌아온다
const prefersReducedMotion = () => { try { return typeof window !== "undefined" && !!window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) { return false; } }; // 감속 모션이면 대기 반복 없이 정지 자세(11/9 독립 검토)
const TARGET_HEIGHT = 1.8; // 모델을 이 높이(미터 단위 가정)로 정규화한다. 카메라 거리 계산용

const THEMES = {
  light: { hemiSky: 0xffffff, hemiGround: 0xb8c4b0, hemi: 1.15, dir: 1.6, dirColor: 0xffffff, rim: 0.0, shadow: 0.16 },
  dark: { hemiSky: 0xdfe8ff, hemiGround: 0x2a3a2e, hemi: 1.35, dir: 1.25, dirColor: 0xfff2dc, rim: 0.9, shadow: 0.34 },
};

/* 유니폼 색(11/9 V4). 사람형 batter-ctl의 재질 이름 → 부위. 텍스처 실측(1024², 밝기 0..1): 상의·바지 흰 바탕 0.85~0.90(77 %), 세로 줄무늬 0.75~0.85,
 * 언더셔츠 소매·양말·벨트·신발 0.10~0.15, UV 빈 곳 0~0.05, 채도 있는 작은 부분(단추 등) 0.2 %. 헬멧: 0.05~0.10과 0.30~0.40 두 층 */
const UNIFORM_PARTS = { M_Clothes: "body", M_Helmet: "helmet" };
const UNI_WHITE = 0.875, UNI_DARK = 0.12, UNI_HELMET = 0.33;
const hexRgb = (hex) => { const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || "")); if (!m) return null; const n = parseInt(m[1], 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
const smooth = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };
/** RGBA 픽셀 배열을 제자리에서 바꾼다(순수 계산, DOM 없음 → node에서도 검사 가능).
 *  body: 밝은 천은 base 색 × (원래 밝기 / 흰 기준), 줄무늬(흰 기준보다 조금 어두운 곳)는 stripe 색 쪽으로 stripeAmt만큼, 어두운 부분은 accent × (원래 밝기 / 어두운 기준).
 *  둘 사이(0.22~0.42)는 부드럽게 섞어 경계가 각지지 않는다. 채도 있는 작은 부분은 그대로.
 *  helmet: 전체를 color × (lift + (1 − lift) × 원래 밝기 / 헬멧 기준)으로(원래 명암을 줄여 남김) */
export function recolorPixels(px, part, colors) {
  if (part === "helmet") {
    const c = hexRgb(colors && colors.color); if (!c) return px;
    const lift = Number.isFinite(colors.lift) ? colors.lift : 0.55;
    for (let i = 0; i < px.length; i += 4) {
      const l = (0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]) / 255;
      const f = Math.min(1.35, lift + (1 - lift) * (l / UNI_HELMET));
      px[i] = Math.min(255, c[0] * f); px[i + 1] = Math.min(255, c[1] * f); px[i + 2] = Math.min(255, c[2] * f);
    }
    return px;
  }
  const base = hexRgb(colors && colors.base), accent = hexRgb(colors && colors.accent), stripe = hexRgb(colors && colors.stripe) || base;
  if (!base || !accent) return px;
  const amt = Number.isFinite(colors.stripeAmt) ? Math.min(1, Math.max(0, colors.stripeAmt)) : 0;
  for (let i = 0; i < px.length; i += 4) {
    const r = px[i], g = px[i + 1], b = px[i + 2], mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    if (mx > 40 && (mx - mn) / mx > 0.28) continue; // 채도 있는 작은 부분(단추·장식)은 그대로
    const l = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    const fl = Math.min(1.08, l / UNI_WHITE), s = amt * smooth(UNI_WHITE - 0.03, UNI_WHITE - 0.12, l); // 줄무늬일수록 s → amt
    const fd = Math.min(1.6, Math.max(0.4, l / UNI_DARK)), td = smooth(0.42, 0.22, l); // 어두울수록 td → 1
    for (let k = 0; k < 3; k++) {
      const light = base[k] * fl * (1 - s) + stripe[k] * s;
      px[i + k] = Math.min(255, light * (1 - td) + accent[k] * fd * td);
    }
  }
  return px;
}

/** 컨테이너 안에 캔버스를 만들고 glb와 메타를 읽는다. 읽기 실패는 reject(호스트는 SVG 타자를 유지).
 *  WebGL 초기화 실패는 reject하지 않고 ok=false, mode="init-failed"인 객체를 돌려준다(메서드는 전부 안전하게 무동작). */
export async function createBatter3D(container, opts = {}) {
  if (!container || !container.appendChild) throw new Error("container 필요");
  const modelUrl = opts.modelUrl || (opts.assetsBase || "./assets/") + (opts.modelName || "batter") + ".glb";
  const metaUrl = opts.metaUrl || (opts.assetsBase || "./assets/") + (opts.modelName || "batter") + ".meta.json";
  let meta = {};
  try { const r = await fetch(metaUrl, { cache: "no-store" }); if (r.ok) meta = await r.json(); } catch (e) { /* 메타 없이도 동작(기본값) */ }
  const gltf = await new GLTFLoader().loadAsync(modelUrl);
  const b = new Batter3D(container, gltf, meta, opts);
  return b;
}

class Batter3D {
  constructor(container, gltf, meta, opts) {
    this.ok = true; this.mode = "webgl"; this.meta = meta || {}; this.container = container;
    this.side = opts.side === "L" ? "L" : "R"; this.theme = opts.theme === "dark" ? "dark" : "light";
    this.kind = null; this.k = 0; this.disposed = false; this.error = null;
    // 초기화가 중간에 실패해도 getState·sample이 쓰는 필드는 항상 존재한다
    this.clips = {}; this.actions = {}; this.bones = {}; this.unitHeight = null; this.bat = null; this.batTip = null; this.batKnob = null; this.active = null;
    this.blend = null; this.swingLast = 0; // 진행 중인 섞기 {t, dur, items:[{a, from, to}]}, setSwing 사이 시간 측정용
    this.idleRaf = 0; this.idleLast = 0; this.idleOn = false; this.idleRendered = 0; // 대기 동작 반복(11/9): idle 클립이 충분히 길 때만 자체 rAF로 돌린다(스윙 중·해제·문맥 상실 때는 끈다)
    this.renderer = null; this.scene = null; this.camera = null; this.mixer = null; this.model = null; this.skinned = []; this.soleCandidates = null;
    this.view = Object.assign({ yawDeg: 0, azimuthDeg: 28, elevationDeg: 9, zoom: 1.0, targetY: 0.5 }, this.meta.view || {}, opts.view || {});
    // 배트는 아래 손(우타자의 왼손)에 붙인다. Mixamo 타격·헛스윙 클립은 팔로스루에서 위 손(오른손)을 놓는다(손 간격 표본 0.13 → 0.95 m)
    this.batParams = Object.assign({ enabled: true, bone: "LeftHand", lengthRatio: 0.48, radiusRatio: 0.016, position: [0, 0, 0], rotationDeg: [0, 0, 0], handleRatio: 0.1 }, this.meta.bat || {}, opts.bat || {});
    try {
      // 렌더러: 투명 배경, 픽셀 비율 2 제한, 상시 루프 없음. WebGL이 없으면 여기서 예외 → ok=false
      this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "low-power" });
      this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      this.renderer.setClearColor(0x000000, 0);
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      const canvas = this.renderer.domElement; canvas.style.display = "block"; canvas.style.width = "100%"; canvas.style.height = "100%"; canvas.setAttribute("aria-hidden", "true");
      container.appendChild(canvas);
      this.onContextLost = (e) => { e.preventDefault(); this.ok = false; this.mode = "lost"; this.stopIdle(); };
      canvas.addEventListener("webglcontextlost", this.onContextLost, false);

      this.scene = new THREE.Scene();
      this.hemi = new THREE.HemisphereLight(0xffffff, 0x888888, 1); this.scene.add(this.hemi);
      this.dir = new THREE.DirectionalLight(0xffffff, 1); this.dir.position.set(1.5, 3, 2.5); this.scene.add(this.dir);
      this.rim = new THREE.DirectionalLight(0xffffff, 0); this.rim.position.set(-2, 2, -2); this.scene.add(this.rim);
      this.camera = new THREE.PerspectiveCamera(28, 1, 0.05, 50);

      // 모델: 바인딩 자세의 크기로 정규화(키 TARGET_HEIGHT), 발바닥(바인딩 자세 메시 최저점)을 y=0에
      this.model = gltf.scene; this.pivot = new THREE.Group(); this.yawGroup = new THREE.Group();
      const box = Batter3D.skinnedBounds(this.model); const size = new THREE.Vector3(); box.getSize(size); // 스킨 메시만(캐릭터 몸). glb에 든 배트 메시(사람형 캐릭터, 11/9)는 키 정규화에서 뺀다
      this.unitHeight = size.y || 1; const s = TARGET_HEIGHT / this.unitHeight;
      this.model.scale.setScalar(s); this.model.position.set(-(box.min.x + size.x / 2) * s, -box.min.y * s, -(box.min.z + size.z / 2) * s);
      this.model.traverse((o) => { if (o.isMesh || o.isSkinnedMesh) { o.frustumCulled = false; o.castShadow = false; o.receiveShadow = false; } if (o.isSkinnedMesh) this.skinned.push(o); });
      this.yawGroup.add(this.model); this.pivot.add(this.yawGroup); this.scene.add(this.pivot);

      // 바닥 접지 원(그림자 대용)
      this.shadow = new THREE.Mesh(new THREE.CircleGeometry(0.38, 40), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.16, depthWrite: false }));
      this.shadow.rotation.x = -Math.PI / 2; this.shadow.position.y = 0.002; this.pivot.add(this.shadow);

      // 뼈 찾기(이름 끝으로: 변환 과정에서 접두어 표기가 달라질 수 있다)
      this.model.traverse((o) => { if (o.isBone) for (const key of ["RightHand", "LeftHand", "RightHandIndex1", "RightHandPinky1", "LeftHandIndex1", "LeftHandPinky1", "Hips", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase", "Head"]) if (o.name.endsWith(key) && !this.bones[key]) this.bones[key] = o; });

      // 애니메이션: 클립 이름으로 액션을 만들고 전부 정지 상태로 둔다
      this.mixer = new THREE.AnimationMixer(this.model);
      for (const c of gltf.animations) { this.clips[c.name] = c; const a = this.mixer.clipAction(c); a.setLoop(THREE.LoopOnce, 1); a.clampWhenFinished = true; a.enabled = true; a.weight = 1; this.actions[c.name] = a; }

      this.buildBat();
      this.applyTheme(this.theme); this.applySide(this.side); this.applyView();
      this.reset(); // idle 자세 + 첫 렌더(크기는 resize에서)
      this.resize();
    } catch (e) { this.ok = false; this.mode = "init-failed"; this.error = e; try { this.disposeInternal(); } catch (e2) { /* 무시 */ } }
  }

  /* ---------- 공개 메서드: 전부 동기, 예외 없음 ---------- */
  setSide(side) { return this.guard(() => { this.side = side === "L" ? "L" : "R"; this.applySide(this.side); this.render(); }); }
  setTheme(theme) { return this.guard(() => { this.theme = theme === "dark" ? "dark" : "light"; this.applyTheme(this.theme); this.render(); }); }
  /** 유니폼 색(11/9 V4). spec = { body: { base, accent, stripe, stripeAmt }, helmet: { color, lift } } 또는 null(원래 텍스처).
   *  재질 이름이 UNIFORM_PARTS에 있는 것(사람형 batter-ctl의 M_Clothes·M_Helmet)에만 적용한다. 그런 재질이 없으면(마네킹) 아무것도 하지 않는다.
   *  원래 텍스처의 밝기(음영·주름·줄무늬)를 그대로 살리고 색만 옮긴다(recolorPixels). 같은 spec은 캐시를 다시 쓴다. glb 파일은 바꾸지 않는다 */
  setUniform(spec) {
    return this.guard(() => {
      const key = spec ? JSON.stringify(spec) : "";
      if (key === this.uniformKey) return;
      this.uniformKey = key;
      if (!this.uniformMats) {
        this.uniformMats = []; this.uniformCache = new Map();
        this.model.traverse((o) => { if (!o.isMesh) return; for (const m of Array.isArray(o.material) ? o.material : [o.material]) if (m && m.map && UNIFORM_PARTS[m.name] && !this.uniformMats.includes(m)) { m.userData.pzOrigMap = m.map; this.uniformMats.push(m); } });
      }
      for (const m of this.uniformMats) {
        const part = UNIFORM_PARTS[m.name], orig = m.userData.pzOrigMap, colors = spec && spec[part];
        let tex = orig;
        if (colors && orig && orig.image) {
          const ck = m.name + "|" + JSON.stringify(colors);
          tex = this.uniformCache.get(ck);
          if (!tex) { tex = Batter3D.recolorTexture(orig, part, colors); this.uniformCache.set(ck, tex); }
        }
        if (m.map !== tex) { m.map = tex; m.needsUpdate = true; }
      }
      this.render();
    });
  }
  /** kind: "contact" | "miss" | "take" | "hit" (또는 contact_hit, contact_hit1, contact_hit2), k: 0..1 진행률. 그 프레임만 그린다. 모델에 없는 종류는 contact로 대체. */
  setSwing(kind, k, opts) {
    return this.guard(() => {
      const name = SWING_KINDS.has(kind) && this.actions[kind] ? kind : this.actions.contact ? "contact" : null;
      if (!name) return;
      const kk = Number.isFinite(k) ? Math.min(1, Math.max(0, k)) : 0;
      const now = typeof performance !== "undefined" ? performance.now() : Date.now(); const dt = this.swingLast ? Math.min(0.1, Math.max(0, (now - this.swingLast) / 1000)) : 0; this.swingLast = now;
      const a = this.actions[name];
      if (this.active !== a) { // 전환. 대기 반복 중이었거나 앞 섞기가 아직 진행 중이면 지금 가중치에서 이어서 BLEND_IN 동안 섞는다(가중치는 아래 stepBlend(dt); 재생 시각은 paused라 안 움직인다)
        const fromIdle = this.idleOn && this.active === this.actions.idle && BLEND_IN > 0;
        this.stopIdle(); // 호스트가 진행률을 주는 동안은 자체 시계를 끈다(idle은 paused로 자세 고정)
        if (!a.isScheduled()) a.reset(); // play()는 startBlend 안에서(가중치를 읽은 뒤). 처음 쓰는 액션은 _effectiveWeight 초기값이 1이라 먼저 play하면 섞기가 1에서 시작해 튄다(11/9 측정)
        a.enabled = true; a.paused = true;
        const viaOpt = opts && Number.isFinite(opts.blendIn) && opts.blendIn > 0 ? opts.blendIn : 0; // 클립 → 클립(11/9 V4: 스윙 → 나가기)은 호스트가 섞는 시간을 준다. 없으면 전과 같음
        this.startBlend(a, fromIdle || this.blend ? BLEND_IN : viaOpt); // 0 = 즉시(마네킹·짧은 idle: 다른 액션 정지, a 가중치 1 — 전과 같음)
        this.active = a;
        this.exitHips0 = EXIT_KINDS.has(name) ? this.hipsLocal() : null; // 나가기 시작 때 골반 자리(그림자는 여기서부터 움직인 만큼만 따라간다; 첫 프레임에 튀지 않게)
      }
      const clip = this.clips[name]; this.active.time = kk * Math.max(0, clip.duration - 1e-4); this.kind = name; this.k = kk;
      this.stepBlend(dt); this.mixer.update(dt); this.followShadow(EXIT_KINDS.has(name)); this.render();
    });
  }
  /** 나가기 동작 중에는 접지 원이 골반 바로 아래를 따라간다(타자가 걸어 나가는데 그림자만 남지 않게). 그 밖에는 원래 자리(0) */
  followShadow(on) {
    if (!this.shadow) return;
    const v = on && this.exitHips0 ? this.hipsLocal() : null;
    if (v) { this.shadow.position.x = v.x - this.exitHips0.x; this.shadow.position.z = v.z - this.exitHips0.z; }
    else { this.shadow.position.x = 0; this.shadow.position.z = 0; }
  }
  /** 골반 뼈의 pivot 기준 위치(거울·yaw 포함 좌표계에서 그림자와 같은 공간). 뼈가 없으면 null */
  hipsLocal() { if (!this.bones.Hips) return null; const v = new THREE.Vector3(); this.bones.Hips.getWorldPosition(v); return this.pivot.worldToLocal(v); }
  /** 대기 자세. idle 클립이 0.4 s 이상이면 반복 재생(자체 rAF, 30 fps 상한)으로 들어가고, 스윙 중이었다면 0.35 s 동안 섞어 넘어간다(끝 자세 컷 없음). 짧은 idle(마네킹 2프레임)은 전처럼 시간 0 정지 자세 */
  reset(opts) {
    return this.guard(() => {
      const name = this.actions.idle ? "idle" : this.actions.contact ? "contact" : Object.keys(this.actions)[0]; if (!name) return;
      const still = !!(opts && opts.still) || prefersReducedMotion(); // 호스트가 정지 자세를 원하거나(애니메이션 끔·다른 화면) OS 감속 모션이면 반복 없이 시각 0 정지
      const loopable = !still && name === "idle" && this.clips.idle && this.clips.idle.duration >= 0.4;
      const a = this.actions[name];
      if (!loopable) { // 정지 자세(마네킹·짧은 idle·still): 전과 같이 시각 0 한 프레임
        this.stopIdle(); if (!a.isScheduled()) a.reset(); a.enabled = true; a.paused = true; a.time = 0;
        this.startBlend(a, 0); this.active = a; this.kind = null; this.k = 0; this.swingLast = 0; this.mixer.update(0); this.followShadow(false); this.render(); return;
      }
      if (this.idleOn && this.active === a) return; // 이미 대기 반복 중
      const idle = a, prev = this.active;
      idle.setLoop(THREE.LoopRepeat, Infinity); idle.clampWhenFinished = false; idle.enabled = true;
      if (!(this.weightOf(idle) > 0)) idle.time = 0; // 보이지 않던 대기는 처음부터. 섞기 도중 끊겨 아직 보이는 중이면 시각을 유지해 튀지 않게
      idle.paused = false; // play()는 startBlend 안에서(가중치를 읽은 뒤)
      // 스윙·반응 끝 자세 → 대기로 섞기. 가중치가 남은 모든 액션(앞 섞기가 끊긴 세 번째 액션 포함)을 지금 값에서 0으로, 대기는 지금 값에서 1로(합 1 유지)
      const visiblePrev = prev && prev !== idle && this.weightOf(prev) > 0;
      // hit는 달려 나간 뒤(k ≥ 0.65, 골반 이동 시작 전후)에만 컷. 그 전에 끊기면(다음 투구) 아직 틀 안의 타격 자세라 다른 클립처럼 0.35 s 섞는다(독립 검토 지적)
      const exitOut = visiblePrev && EXIT_KINDS.has(this.kind); // 나가기 동작은 첫 프레임부터 골반이 움직이므로(독립 검토) 끊기면 언제든 컷: 섞으면 틀을 가로질러 미끄러져 돌아온다(호스트가 상자를 투명하게 해 두거나 새 공에서 바로 다시 그린다)
      this.startBlend(idle, visiblePrev ? (exitOut ? 0 : prev === this.actions.take ? FADE_BACK_TAKE : prev === this.actions.hit && this.k >= 0.65 ? FADE_BACK_HIT : FADE_BACK) : (this.blend ? FADE_BACK : 0));
      this.followShadow(false);
      this.active = idle; this.kind = null; this.k = 0; this.swingLast = 0;
      this.startIdle();
    });
  }
  /** 액션의 지금 표시 가중치. 재생 중이 아니면 0(아직 play 전인 액션은 _effectiveWeight 초기값이 1이라 isScheduled로 거른다) */
  weightOf(a) { return a && a.enabled && a.isScheduled() ? a.getEffectiveWeight() : 0; }
  /** 섞기 시작: incoming은 지금 가중치→1, 가중치가 남은 다른 액션은 지금 값→0(합이 1로 유지되어 원래 자세가 비치지 않는다). dur 0이면 즉시. 끝나면 0이 된 액션은 stop */
  startBlend(incoming, dur) {
    const items = [{ a: incoming, from: this.weightOf(incoming), to: 1 }];
    for (const a of Object.values(this.actions)) { if (a === incoming) continue; const w = this.weightOf(a); if (w > 0) items.push({ a, from: w, to: 0 }); else if (a.isScheduled()) a.stop(); }
    incoming.enabled = true; if (!incoming.isScheduled()) incoming.play();
    this.blend = { t: 0, dur: Math.max(0, dur), items };
    this.stepBlend(0);
  }
  /** mixer.update 직전에 부른다(setSwing·대기 반복). 진행 중인 섞기가 없으면 아무것도 안 함 */
  stepBlend(dt) {
    const b = this.blend; if (!b) return;
    b.t += Math.max(0, dt); const u = b.dur > 0 ? Math.min(1, b.t / b.dur) : 1;
    for (const it of b.items) it.a.setEffectiveWeight(it.from + (it.to - it.from) * u);
    if (u >= 1) { this.blend = null; for (const it of b.items) if (it.to === 0) it.a.stop(); }
  }
  startIdle() {
    if (this.idleOn || this.disposed || !this.ok) return;
    this.idleOn = true; this.idleLast = 0;
    const tick = (now) => {
      if (!this.idleOn || this.disposed || !this.ok) { this.idleRaf = 0; this.idleOn = false; return; }
      const dt = this.idleLast ? Math.min(0.1, (now - this.idleLast) / 1000) : 0; // 탭 복귀 등 긴 공백은 0.1 s로 잘라 튀지 않게
      if (dt >= 1 / 32 || !this.idleLast) { this.idleLast = now; this.stepBlend(dt); this.mixer.update(dt); this.render(); this.idleRendered++; } // 30 fps 상한
      this.idleRaf = requestAnimationFrame(tick);
    };
    this.idleRaf = requestAnimationFrame(tick);
  }
  stopIdle() { if (this.idleRaf) { cancelAnimationFrame(this.idleRaf); this.idleRaf = 0; } this.idleOn = false; if (this.actions.idle) this.actions.idle.paused = true; }
  resize() {
    return this.guard(() => {
      const w = this.container.clientWidth, h = this.container.clientHeight;
      if (!(w > 0 && h > 0)) return; // 숨겨진 상태(0×0)면 다음 resize까지 기다린다
      this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.applyView(); this.render();
    });
  }
  dispose() { if (this.disposed) return; this.disposed = true; try { this.stopIdle(); } catch (e) { /* 무시 */ } try { this.disposeInternal(); } catch (e) { /* 무시 */ } this.ok = false; this.mode = "disposed"; }
  /* 데모 조절용(호스트는 쓰지 않음) */
  setView(v) { return this.guard(() => { Object.assign(this.view, v || {}); this.applyView(); this.render(); }); }
  setBat(p) { return this.guard(() => { Object.assign(this.batParams, p || {}); this.buildBat(); this.render(); }); }
  /** 상태 조회. 초기화 실패·dispose 뒤에도 예외 없이 답한다(gpu는 렌더러가 살아 있을 때만) */
  getState() {
    const safe = { ok: this.ok, mode: this.mode, side: this.side, theme: this.theme, kind: this.kind, k: this.k, error: this.error ? String(this.error.message || this.error) : null };
    try {
      safe.view = { ...this.view };
      safe.bat = JSON.parse(JSON.stringify(this.batParams));
      safe.clips = Object.fromEntries(Object.entries(this.clips || {}).map(([n, c]) => [n, c.duration]));
      safe.bones = Object.keys(this.bones || {});
      safe.unitHeight = this.unitHeight;
      safe.gpu = this.renderer && !this.disposed ? { geometries: this.renderer.info.memory.geometries, textures: this.renderer.info.memory.textures, programs: this.renderer.info.programs ? this.renderer.info.programs.length : null } : null;
      safe.batParts = this.bat ? this.bat.children.length : 0;
      safe.idleLoop = this.idleOn; safe.idleRendered = this.idleRendered; // 대기 반복 상태·그린 프레임 수(검증용)
      safe.weights = { active: this.active ? +this.weightOf(this.active).toFixed(3) : null, idle: this.actions.idle ? +this.weightOf(this.actions.idle).toFixed(3) : null }; safe.blending = !!this.blend; // 섞기 진행 확인용
    } catch (e) { safe.stateError = String(e && e.message || e); }
    return safe;
  }
  /** 스윙 전 구간 검사용: 발·손 위치 표본(월드 좌표, 미터). opts.sole=true면 스킨 적용 메시의 최저점(실제 발바닥)도 계산한다.
   *  발 뼈(관절) 위치와 메시 최저점은 다른 것이다: 관절은 발목·발가락 뿌리의 위치, 메시 최저점이 바닥에 닿는 살(발바닥)이다. */
  sample(kind, k, opts = {}) {
    const out = { ok: false };
    this.guard(() => {
      this.setSwing(kind, k); if (this.blend) { this.stepBlend(Infinity); this.mixer.update(0); } // 측정용 표본은 섞기 없이 요청 클립만의 자세(11/9 독립 검토: 대기에서 넘어오는 첫 표본들이 섞여 cue 추정이 흔들림)
      this.model.updateMatrixWorld(true);
      const p = (b) => { if (!this.bones[b]) return null; const v = new THREE.Vector3(); this.bones[b].getWorldPosition(v); return [v.x, v.y, v.z].map((x) => Math.round(x * 1000) / 1000); };
      out.ok = true; out.leftFoot = p("LeftFoot"); out.rightFoot = p("RightFoot"); out.leftToe = p("LeftToeBase"); out.rightToe = p("RightToeBase"); out.leftHand = p("LeftHand"); out.rightHand = p("RightHand"); out.head = p("Head");
      if (this.batTip && this.batKnob) { const v = new THREE.Vector3(); this.batTip.getWorldPosition(v); out.batTip = [v.x, v.y, v.z].map((x) => Math.round(x * 1000) / 1000); this.batKnob.getWorldPosition(v); out.batKnob = [v.x, v.y, v.z].map((x) => Math.round(x * 1000) / 1000); }
      if (out.leftHand && out.rightHand) out.handGap = Math.round(Math.hypot(out.leftHand[0] - out.rightHand[0], out.leftHand[1] - out.rightHand[1], out.leftHand[2] - out.rightHand[2]) * 1000) / 1000;
      if (opts.sole) out.sole = this.soleMinY(!!opts.soleFull);
    });
    return out;
  }

  /* ---------- 내부 ---------- */
  guard(fn) { if (this.disposed || !this.ok) return false; try { fn(); return true; } catch (e) { this.ok = false; this.error = e; return false; } }
  render() { if (this.disposed || !this.ok) return; this.renderer.render(this.scene, this.camera); }
  applySide(side) { this.pivot.scale.x = side === "L" ? -1 : 1; } // 거울 반전은 메시와 뼈대의 공통 부모(pivot)에만 건다
  applyTheme(theme) {
    const t = THEMES[theme] || THEMES.light;
    this.hemi.color.setHex(t.hemiSky); this.hemi.groundColor.setHex(t.hemiGround); this.hemi.intensity = t.hemi;
    this.dir.intensity = t.dir; this.dir.color.setHex(t.dirColor); this.rim.intensity = t.rim; this.shadow.material.opacity = t.shadow;
  }
  applyView() {
    const v = this.view, az = THREE.MathUtils.degToRad(v.azimuthDeg), el = THREE.MathUtils.degToRad(v.elevationDeg);
    this.yawGroup.rotation.y = THREE.MathUtils.degToRad(v.yawDeg);
    const halfH = TARGET_HEIGHT * 0.56, fov = THREE.MathUtils.degToRad(this.camera.fov);
    const aspect = this.camera.aspect || 1;
    const distH = halfH / Math.tan(fov / 2), distW = (TARGET_HEIGHT * 0.42) / Math.tan(fov / 2) / aspect; // 세로 기준과 가로(좁은 열) 기준 중 먼 쪽
    const dist = Math.max(distH, distW) / (v.zoom || 1);
    const target = new THREE.Vector3(0, TARGET_HEIGHT * v.targetY, 0);
    this.camera.position.set(target.x + dist * Math.cos(el) * Math.sin(az), target.y + dist * Math.sin(el), target.z + dist * Math.cos(el) * Math.cos(az));
    this.camera.lookAt(target); this.camera.updateProjectionMatrix();
  }
  /** 배트(원통 3개)를 손 뼈에 붙인다. 다시 부를 때 이전 배트의 geometry와 material(손잡이·손잡이끝이 공유)을 각각 한 번씩만 해제한다 */
  buildBat() {
    this.disposeBat();
    const p = this.batParams, handKey = /RightHand$/.test(String(p.bone || "")) ? "RightHand" : "LeftHand", hand = this.bones[handKey];
    if (!p.enabled || !hand) { this.pickEmbeddedMarkers(); return; } // 원통 배트를 안 쓸 때는 glb에 든 배트 표지(BatTip/BatKnob)를 쓴다. setBat으로 켰다 꺼도 다시 잡힌다(11/9 독립 검토 지적)
    const L = p.lengthRatio * this.unitHeight, R = p.radiusRatio * this.unitHeight; // 뼈 공간 단위(모델 원래 단위)
    const bat = new THREE.Group();
    // 손잡이 쪽이 가늘고 배럴 쪽이 굵은 원통. 로컬 +Y가 배럴 방향. 원점은 아래 손의 그립 위치
    const barrelMat = new THREE.MeshStandardMaterial({ color: 0x8a5a2b, roughness: 0.6, metalness: 0.0 });
    const handleMat = new THREE.MeshStandardMaterial({ color: 0x3b2a1a, roughness: 0.8 }); // 손잡이와 손잡이끝(knob)이 공유
    const barrel = new THREE.Mesh(new THREE.CylinderGeometry(R, R * 0.55, L * (1 - p.handleRatio), 14), barrelMat);
    barrel.position.y = L * (1 - p.handleRatio) / 2;
    const handle = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.55, R * 0.5, L * p.handleRatio, 12), handleMat);
    handle.position.y = -L * p.handleRatio / 2;
    const knob = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.9, R * 0.9, R * 0.6, 12), handleMat); knob.position.y = -L * p.handleRatio;
    bat.add(barrel, handle, knob);
    this.batTip = new THREE.Object3D(); this.batTip.position.y = L * (1 - p.handleRatio); bat.add(this.batTip);
    this.batKnob = new THREE.Object3D(); this.batKnob.position.y = -L * p.handleRatio; bat.add(this.batKnob);
    // 기본 방향: 주먹을 가로지르는 방향(새끼손가락 → 집게손가락 뿌리)을 배럴 방향으로 둔다. 그 뒤 메타의 회전·위치 보정
    const idx = this.bones[handKey + "Index1"], pky = this.bones[handKey + "Pinky1"];
    if (idx && pky) {
      const a = idx.position.clone(), b = pky.position.clone(); const axis = a.clone().sub(b).normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis); bat.quaternion.copy(q);
      bat.position.copy(a.add(b).multiplyScalar(0.5));
    }
    const e = new THREE.Euler(...p.rotationDeg.map((d) => THREE.MathUtils.degToRad(d)), "XYZ"); bat.quaternion.multiply(new THREE.Quaternion().setFromEuler(e));
    bat.position.add(new THREE.Vector3(...p.position));
    bat.traverse((o) => { if (o.isMesh) o.frustumCulled = false; });
    hand.add(bat); this.bat = bat;
  }
  /** glb 안의 배트 표지(변환기가 넣은 BatTip/BatKnob) → sample()의 배트 끝 속도 측정용. 없으면 null */
  pickEmbeddedMarkers() { const tip = this.model.getObjectByName("BatTip"), knob = this.model.getObjectByName("BatKnob"); this.batTip = tip && knob ? tip : null; this.batKnob = tip && knob ? knob : null; }
  disposeBat() {
    if (!this.bat) return;
    if (this.bat.parent) this.bat.parent.remove(this.bat);
    Batter3D.disposeTree(this.bat);
    this.bat = null; this.batTip = null; this.batKnob = null;
  }
  /** 트리 안의 geometry·material을 중복 없이(공유분은 한 번만) 해제한다 */
  /** 스킨 메시(캐릭터 몸)만의 경계 상자. 일반 Mesh(글러브·배트 같은 소품)는 넣지 않는다. 스킨 메시가 없으면 전체 */
  static skinnedBounds(model) {
    model.updateMatrixWorld(true);
    const box = new THREE.Box3(); let n = 0;
    model.traverse((o) => { if (o.isSkinnedMesh) { n++; box.expandByObject(o); } });
    return n ? box : new THREE.Box3().setFromObject(model);
  }
  /** 원래 텍스처 그림을 캔버스에 그려 픽셀 색을 옮긴 새 텍스처(설정·변환은 원래 것 복제). 결과는 원래와 같은 크기 */
  static recolorTexture(orig, part, colors) {
    const img = orig.image, w = img.width, h = img.height;
    const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
    const cx = cv.getContext("2d", { willReadFrequently: true });
    cx.drawImage(img, 0, 0);
    const data = cx.getImageData(0, 0, w, h);
    recolorPixels(data.data, part, colors);
    cx.putImageData(data, 0, 0);
    const t = orig.clone(); t.source = new (THREE.TextureSource || THREE.Source)(cv); t.needsUpdate = true; // r187: Source → TextureSource 이름 바뀜
    return t;
  }
  /** 유니폼 텍스처를 원래 것으로 되돌리고 만든 텍스처를 해제한다(dispose 전에; 원래 텍스처는 disposeTree가 해제) */
  disposeUniform() {
    if (this.uniformMats) for (const m of this.uniformMats) if (m.userData.pzOrigMap && m.map !== m.userData.pzOrigMap) m.map = m.userData.pzOrigMap;
    if (this.uniformCache) { for (const t of this.uniformCache.values()) { try { t.dispose(); } catch (e) { /* 무시 */ } } this.uniformCache.clear(); }
    this.uniformKey = "";
  }
  static disposeTree(root) {
    const geos = new Set(), mats = new Set();
    root.traverse((o) => { if (o.geometry) geos.add(o.geometry); if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => mats.add(m)); });
    for (const g of geos) { try { g.dispose(); } catch (e) { /* 무시 */ } }
    for (const m of mats) { try { if (m.map) m.map.dispose(); m.dispose(); } catch (e) { /* 무시 */ } }
    return { geometries: geos.size, materials: mats.size };
  }
  /** 스킨 적용 메시의 월드 최저 y(미터). full=false면 바인딩 자세에서 낮은 쪽 8% 높이 안의 정점(발 부위)만 본다 */
  soleMinY(full) {
    if (!this.skinned.length) return null;
    if (!this.soleCandidates) { // 후보 = 첫 호출 시점 자세(대개 대기 자세)에서 월드 높이가 낮은 8 %의 정점. geometry 로컬 y로 고르면 Z-up 출처(3ds Max/Unity 에셋)에서 엉뚱한 정점이 뽑힌다(11/9 발견: 사람형 캐릭터에서 발바닥이 0.7~1.4 m로 나옴)
      const v0 = new THREE.Vector3();
      this.soleCandidates = this.skinned.map((m) => {
        if (m.skeleton) m.skeleton.update();
        const count = m.geometry.attributes.position.count, ys = new Float32Array(count); let minY = Infinity, maxY = -Infinity;
        for (let i = 0; i < count; i++) { m.getVertexPosition(i, v0); v0.applyMatrix4(m.matrixWorld); ys[i] = v0.y; if (v0.y < minY) minY = v0.y; if (v0.y > maxY) maxY = v0.y; }
        const lim = minY + (maxY - minY) * 0.08, idx = [];
        for (let i = 0; i < count; i++) if (ys[i] <= lim) idx.push(i);
        return idx;
      });
    }
    const v = new THREE.Vector3(); let minY = Infinity, n = 0;
    this.skinned.forEach((m, mi) => {
      if (m.skeleton) m.skeleton.update();
      const count = m.geometry.attributes.position.count;
      const run = (i) => { m.getVertexPosition(i, v); v.applyMatrix4(m.matrixWorld); if (v.y < minY) minY = v.y; n++; };
      if (full) for (let i = 0; i < count; i++) run(i); else for (const i of this.soleCandidates[mi]) run(i);
    });
    return { minY: Math.round(minY * 10000) / 10000, vertices: n, full: !!full };
  }
  disposeInternal() {
    // 단위 3(11/8 독립 검토): 스켈레톤의 뼈 텍스처를 해제하고, renderer.dispose() 뒤 문맥도 즉시 잃게 해(forceContextLoss) 떼어낸 캔버스가 GC 전까지 WebGL 문맥을 붙들지 않게 한다(반복 해제·재연결 시 "Too many active WebGL contexts" 방지)
    for (const m of this.skinned) { if (m.skeleton) { try { m.skeleton.dispose(); } catch (e) { /* 무시 */ } } }
    if (this.renderer) { const c = this.renderer.domElement; if (this.onContextLost) c.removeEventListener("webglcontextlost", this.onContextLost, false); try { this.renderer.dispose(); } catch (e) { /* 무시 */ } try { const gl = this.renderer.getContext(); if (!(gl && gl.isContextLost && gl.isContextLost())) this.renderer.forceContextLoss(); } catch (e) { /* 무시 */ } if (c.parentNode) c.parentNode.removeChild(c); }
    if (this.mixer) { try { this.mixer.stopAllAction(); } catch (e) { /* 무시 */ } }
    this.disposeBat();
    this.disposeUniform();
    if (this.scene) Batter3D.disposeTree(this.scene);
    this.renderer = null;
  }
}
