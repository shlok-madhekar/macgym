// MacGym three.js tiled renderer.
// Pixel-space orthographic camera (y up). All capsules = ONE InstancedMesh with an SDF shader
// (perfect round caps at any aspect, AA edges, soft glow). Tiles (bg + ground + ticks) = one
// InstancedMesh; targets = one InstancedMesh.
import * as THREE from "three";

const DEFAULT_PALETTE = ["#4FD1FF", "#A3FF6B", "#7CF2C8", "#A3FF6B", "#7CF2C8", "#FFD36B", "#FFD36B", "#B39CFF", "#FF9F6B"];

const CAPSULE_VS = /* glsl */ `
attribute vec4 aSeg; attribute float aRad; attribute vec3 aCol; attribute vec4 aClip;
varying vec2 vP; varying vec4 vSeg; varying float vRad; varying vec3 vCol; varying vec4 vClip;
void main() {
  vec2 a = aSeg.xy, b = aSeg.zw; vec2 d = b - a; float L = length(d);
  vec2 dir = L > 1e-4 ? d / L : vec2(1.0, 0.0); vec2 n = vec2(-dir.y, dir.x);
  float pad = aRad + 7.0;
  vec2 p = (a + b) * 0.5 + dir * position.x * (L + 2.0 * pad) + n * position.y * (2.0 * pad);
  vP = p; vSeg = aSeg; vRad = aRad; vCol = aCol; vClip = aClip;
  gl_Position = projectionMatrix * viewMatrix * vec4(p, 0.0, 1.0);
}`;
const CAPSULE_FS = /* glsl */ `
precision highp float;
varying vec2 vP; varying vec4 vSeg; varying float vRad; varying vec3 vCol; varying vec4 vClip;
uniform float uGlow;
void main() {
  if (vP.x < vClip.x || vP.y < vClip.y || vP.x > vClip.z || vP.y > vClip.w) discard;
  vec2 a = vSeg.xy, b = vSeg.zw; vec2 pa = vP - a, ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  float dist = length(pa - ba * h);
  float d = dist - vRad;
  float fill = 1.0 - smoothstep(-0.7, 0.7, d);
  float t = clamp(dist / max(vRad, 0.5), 0.0, 1.0);
  vec3 body = vCol * mix(1.18, 0.62, t * t) + vec3(0.18) * (1.0 - smoothstep(0.0, 0.55, t));
  body *= mix(1.0, 0.55, smoothstep(-1.6, -0.2, d) * step(2.2, vRad));
  float glow = exp(-max(d, 0.0) / 3.2) * uGlow * (1.0 - fill);
  float alpha = max(fill, glow);
  vec3 col = mix(vCol, body, fill);
  gl_FragColor = vec4(col, alpha);
}`;

const TILE_VS = /* glsl */ `
attribute vec4 aRect; attribute vec4 aView; attribute float aHover;
varying vec2 vP; varying vec4 vRect; varying vec4 vView; varying float vHover;
void main() {
  vec2 p = aRect.xy + (position.xy + 0.5) * aRect.zw;
  vP = p; vRect = aRect; vView = aView; vHover = aHover;
  gl_Position = projectionMatrix * viewMatrix * vec4(p, 0.0, 1.0);
}`;
const TILE_FS = /* glsl */ `
precision highp float;
varying vec2 vP; varying vec4 vRect; varying vec4 vView; varying float vHover;
uniform vec3 uAccent; uniform float uTime; uniform float uTick;
float sdRound(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
void main() {
  vec2 c = vRect.xy + vRect.zw * 0.5; vec2 hs = vRect.zw * 0.5;
  float rad = min(10.0, min(hs.x, hs.y) * 0.25);
  float d = sdRound(vP - c, hs, rad);
  if (d > 0.5) discard;
  float inside = 1.0 - smoothstep(-0.5, 0.5, d);
  float border = 1.0 - smoothstep(0.0, 1.2, abs(d + 0.6));
  float yN = (vP.y - vRect.y) / vRect.w;
  vec3 col = mix(vec3(0.050, 0.060, 0.082), vec3(0.075, 0.088, 0.118), yN);
  float alpha = 0.92 * inside;
  // ground
  float gy = vView.w;
  if (gy > -1e5) {
    float s = vView.y;
    float below = step(vP.y, gy);
    // hatch below ground
    float hatch = step(0.82, fract((vP.x + vP.y) / 7.0));
    col = mix(col, vec3(0.035, 0.04, 0.055), below * 0.9);
    col += below * hatch * 0.022 * smoothstep(gy - 40.0, gy, vP.y);
    float line = 1.0 - smoothstep(0.4, 1.4, abs(vP.y - gy));
    col = mix(col, vec3(0.42, 0.95, 0.55), line * 0.55);
    // scrolling distance ticks every uTick meters
    float wx = (vP.x - vView.x) / s + vView.z;
    float m = abs(fract(wx / uTick + 0.5) - 0.5) * uTick * s;
    float tick = (1.0 - smoothstep(0.3, 1.1, m)) * step(gy - 6.0, vP.y) * below;
    col = mix(col, vec3(0.45, 0.55, 0.65), tick * 0.5);
  }
  vec3 bcol = mix(vec3(0.16, 0.19, 0.25), uAccent, vHover);
  col = mix(col, bcol, border * (0.55 + 0.45 * vHover));
  alpha = max(alpha, border * (0.7 + 0.3 * vHover));
  gl_FragColor = vec4(col, alpha);
}`;

const TARGET_VS = /* glsl */ `
attribute vec3 aT; varying vec2 vUv; varying float vS;
void main() {
  vUv = position.xy * 2.0; vS = aT.z;
  vec2 p = aT.xy + position.xy * aT.z * 2.0;
  gl_Position = projectionMatrix * viewMatrix * vec4(p, 0.0, 1.0);
}`;
const TARGET_FS = /* glsl */ `
precision highp float;
varying vec2 vUv; varying float vS; uniform float uTime; uniform vec3 uCol;
void main() {
  float r = length(vUv);
  float px = 1.0 / max(vS, 1.0);
  float core = 1.0 - smoothstep(0.22 - px, 0.22 + px, r);
  float ph = fract(uTime * 0.9);
  float ring = (1.0 - smoothstep(0.0, 2.5 * px, abs(r - (0.3 + ph * 0.65)))) * (1.0 - ph);
  float glow = exp(-r * 5.0) * 0.55;
  float a = max(core, max(ring * 0.8, glow));
  if (a < 0.01) discard;
  vec3 col = mix(uCol, vec3(1.0), core * 0.55);
  gl_FragColor = vec4(col, a);
}`;

function hexToRgb(hex) {
  const c = new THREE.Color(hex);
  return [c.r, c.g, c.b];
}

export class TileRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setClearColor(0x000000, 0);
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(0, 1, 1, 0, -10, 10);
    this.W = 1; this.H = 1;
    this.hover = -1;
    this.focus = -1;       // focused tile index (accent border)
    this.insetRight = 0;   // px reserved on the right of the stage (focus panel)
    this.fitTop = null;    // smoothed robot height above ground (auto-zoom)
    this.fitEnv = null;
    this.accent = new THREE.Color("#4FD1FF");
    this.tiles = []; // {x,y,w,h,cx,cy,s}
    this.R = 0; this.K = 0;
    this.layoutKey = "";
    this.colorCache = new Map();
    this._buildMeshes(4096, 512, 1024);
  }

  _mesh(maxCount, vs, fs, attrs, uniforms, order) {
    const geo = new THREE.InstancedBufferGeometry();
    const plane = new THREE.PlaneGeometry(1, 1);
    geo.index = plane.index;
    geo.setAttribute("position", plane.getAttribute("position"));
    const buffers = {};
    for (const [name, size] of Object.entries(attrs)) {
      const arr = new Float32Array(maxCount * size);
      const attr = new THREE.InstancedBufferAttribute(arr, size);
      attr.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute(name, attr);
      buffers[name] = attr;
    }
    geo.instanceCount = 0;
    const mat = new THREE.ShaderMaterial({ vertexShader: vs, fragmentShader: fs, uniforms, transparent: true, depthTest: false, depthWrite: false });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.frustumCulled = false;
    mesh.renderOrder = order;
    this.scene.add(mesh);
    return { mesh, geo, mat, buffers, max: maxCount };
  }

  _buildMeshes(maxCaps, maxTiles, maxTargets) {
    this.uTime = { value: 0 };
    this.tileM = this._mesh(maxTiles, TILE_VS, TILE_FS, { aRect: 4, aView: 4, aHover: 1 },
      { uAccent: { value: this.accent }, uTime: this.uTime, uTick: { value: 0.5 } }, 0);
    this.capM = this._mesh(maxCaps, CAPSULE_VS, CAPSULE_FS, { aSeg: 4, aRad: 1, aCol: 3, aClip: 4 },
      { uGlow: { value: 0.32 } }, 1);
    this.tgtM = this._mesh(maxTargets, TARGET_VS, TARGET_FS, { aT: 3 },
      { uTime: this.uTime, uCol: { value: new THREE.Color("#FF4FD8") } }, 2);
  }

  setAccent(hex) { this.accent.set(hex); }

  resize(w, h) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.W = w; this.H = h;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, h, false);
    this.camera.left = 0; this.camera.right = w; this.camera.bottom = 0; this.camera.top = h;
    this.camera.updateProjectionMatrix();
    this.layoutKey = "";
  }

  _layout(R, aspect) {
    const key = `${R}|${aspect.toFixed(3)}|${this.W}|${this.H}|${this.insetRight}`;
    if (key === this.layoutKey) return;
    this.layoutKey = key;
    const W = this.W, H = this.H;
    const marginX = 18, marginTop = 54, marginBottom = 116;
    const aw = W - 2 * marginX - this.insetRight, ah = H - marginTop - marginBottom;
    const gap = R > 100 ? 4 : R > 36 ? 6 : 10;
    let best = { size: 0, cols: 1, rows: R };
    for (let cols = 1; cols <= R; cols++) {
      const rows = Math.ceil(R / cols);
      const tw = (aw - gap * (cols - 1)) / cols;
      const th = (ah - gap * (rows - 1)) / rows;
      const w = Math.min(tw, th * aspect);
      if (w > best.size) best = { size: w, cols, rows };
    }
    const tw = best.size, th = tw / aspect;
    const gw = best.cols * tw + (best.cols - 1) * gap, gh = best.rows * th + (best.rows - 1) * gap;
    const ox = marginX + (aw - gw) / 2, oyTop = marginTop + (ah - gh) / 2;
    this.grid = { cols: best.cols, rows: best.rows, tw, th, gap, ox, oyTop };
    this.tiles = [];
    for (let i = 0; i < R; i++) {
      const c = i % best.cols, r = Math.floor(i / best.cols);
      const x = ox + c * (tw + gap);
      const yTop = oyTop + r * (th + gap);
      const y = H - yTop - th; // to y-up
      this.tiles.push({ x, y, w: tw, h: th, cx: x + tw / 2, cy: y + th / 2 });
    }
  }

  tileAt(px, py) { // px,py in CSS pixels from top-left
    const y = this.H - py;
    for (let i = 0; i < this.tiles.length; i++) {
      const t = this.tiles[i];
      if (px >= t.x && px <= t.x + t.w && y >= t.y && y <= t.y + t.h) return i;
    }
    return -1;
  }

  // inverse transform: CSS pixel -> env world coords for tile i
  toWorld(i, px, py) {
    const t = this.tiles[i]; const v = this.view;
    if (!t || !v) return null;
    const y = this.H - py;
    const cam = this.camX ? this.camX[i] || 0 : 0;
    return { x: (px - t.cx) / v.s + cam + v.mx, y: (y - t.cy) / v.s + v.my };
  }

  // Per-env view box (relative to camera for follow envs). Ground envs: auto-zoom so the robot
  // fills ~60% of tile height in a ~16:9 tile. Arm (no ground): world bounds, widened slightly.
  fitView(frame, aspectOverride) {
    const wb = frame.world_bounds || [-2, 2, -0.5, 2];
    const ww = wb[1] - wb[0], wh = wb[3] - wb[2];
    if (frame.ground == null) {
      const aspect = aspectOverride || Math.max(1.2, ww / wh);
      const h = Math.max(wh, ww / aspect);
      const cx = (wb[0] + wb[1]) / 2, cy = (wb[2] + wb[3]) / 2;
      return { x0: cx - (h * aspect) / 2, x1: cx + (h * aspect) / 2, y0: cy - h / 2, y1: cy + h / 2, aspect };
    }
    const aspect = aspectOverride || 16 / 9;
    const g = frame.ground;
    if (this.fitEnv !== frame.env) { this.fitEnv = frame.env; this.fitTop = null; }
    // robust robot height: 85th percentile of per-env max(y + r) over the rendered envs
    const caps = frame.capsules, K = frame.K, R = frame.R;
    const tops = [];
    for (let i = 0; i < R; i++) {
      let t = -Infinity;
      for (let k = 0; k < K; k++) {
        const o = (i * K + k) * 5, r = caps[o + 4];
        if (!(r > 0)) continue;
        t = Math.max(t, caps[o + 1] + r, caps[o + 3] + r);
      }
      if (isFinite(t)) tops.push(t - g);
    }
    if (tops.length) {
      tops.sort((a, b) => a - b);
      const top = Math.max(0.3, Math.min(wb[3] - g, tops[Math.floor(tops.length * 0.85)]));
      this.fitTop = this.fitTop == null ? top : this.fitTop + (Math.max(top, this.fitTop * 0.97) - this.fitTop) * 0.05;
    }
    const top = this.fitTop || wh * 0.6;
    let h = top / 0.6;
    const follow = !!frame.camera_x;
    if (!follow) h = Math.max(h, (ww * 0.8) / aspect); // static envs (cartpole): keep most of the track
    const y0 = g - h * 0.13;
    const cx = follow ? 0 : (wb[0] + wb[1]) / 2;
    return { x0: cx - (h * aspect) / 2, x1: cx + (h * aspect) / 2, y0, y1: y0 + h, aspect };
  }

  _rgb(hex) {
    let c = this.colorCache.get(hex);
    if (!c) { c = hexToRgb(hex); this.colorCache.set(hex, c); }
    return c;
  }

  // frame: {R,K,capsules(Float32Array R*K*5), targets|null, colors|null, ground|null, world_bounds, camera_x|null}
  draw(frame, time) {
    this.uTime.value = time;
    if (!frame) { this.renderer.render(this.scene, this.camera); return; }
    const R = Math.min(frame.R, this.tileM.max);
    const K = frame.K;
    const vb = this.fitView(frame);
    const wb = [vb.x0, vb.x1, vb.y0, vb.y1];
    const ww = wb[1] - wb[0], wh = wb[3] - wb[2];
    this._layout(R, Math.max(0.3, Math.min(4, vb.aspect)));
    const pad = 1;
    const s = Math.min((this.grid.tw - 2 * pad) / ww, (this.grid.th - 2 * pad) / wh);
    const mx = (wb[0] + wb[1]) / 2, my = (wb[2] + wb[3]) / 2;
    this.view = { s, mx, my };
    this.camX = frame.camera_x;
    const ground = frame.ground;

    // tiles
    const tb = this.tileM.buffers;
    for (let i = 0; i < R; i++) {
      const t = this.tiles[i];
      tb.aRect.array.set([t.x, t.y, t.w, t.h], i * 4);
      const cam = frame.camera_x ? frame.camera_x[i] || 0 : 0;
      const gy = ground == null ? -1e6 : t.cy + (ground - my) * s;
      tb.aView.array.set([t.cx, s, cam + mx, gy], i * 4);
      const cur = tb.aHover.array[i];
      const target = i === this.hover || i === this.focus ? 1 : 0;
      tb.aHover.array[i] = cur + (target - cur) * 0.25;
    }
    for (const k in tb) tb[k].needsUpdate = true;
    this.tileM.geo.instanceCount = R;

    // capsules
    const cb = this.capM.buffers;
    const caps = frame.capsules;
    const colors = frame.colors;
    let n = 0;
    const minR = 1.6;
    for (let i = 0; i < R; i++) {
      const t = this.tiles[i];
      const cam = frame.camera_x ? frame.camera_x[i] || 0 : 0;
      const ox = t.cx - (cam + mx) * s, oy = t.cy - my * s;
      const clip0 = t.x + 1, clip1 = t.y + 1, clip2 = t.x + t.w - 1, clip3 = t.y + t.h - 1;
      for (let k = 0; k < K; k++) {
        if (n >= this.capM.max) break;
        const o = (i * K + k) * 5;
        const r = caps[o + 4];
        if (!(r > 0)) continue;
        cb.aSeg.array[n * 4] = ox + caps[o] * s;
        cb.aSeg.array[n * 4 + 1] = oy + caps[o + 1] * s;
        cb.aSeg.array[n * 4 + 2] = ox + caps[o + 2] * s;
        cb.aSeg.array[n * 4 + 3] = oy + caps[o + 3] * s;
        cb.aRad.array[n] = Math.max(minR, r * s * 1.15);
        const rgb = this._rgb((colors && colors[k]) || DEFAULT_PALETTE[k % DEFAULT_PALETTE.length]);
        cb.aCol.array.set(rgb, n * 3);
        cb.aClip.array[n * 4] = clip0; cb.aClip.array[n * 4 + 1] = clip1;
        cb.aClip.array[n * 4 + 2] = clip2; cb.aClip.array[n * 4 + 3] = clip3;
        n++;
      }
    }
    for (const k in cb) cb[k].needsUpdate = true;
    this.capM.geo.instanceCount = n;
    this.capM.mat.uniforms.uGlow.value = this.grid.tw > 120 ? 0.32 : 0.22;

    // targets
    const tg = frame.targets;
    let m = 0;
    if (tg && tg.length) {
      const T = Math.floor(tg.length / (R * 2)) || 0;
      const size = Math.max(5, Math.min(16, this.grid.tw * 0.07));
      const ab = this.tgtM.buffers.aT.array;
      for (let i = 0; i < R && T; i++) {
        const t = this.tiles[i];
        const cam = frame.camera_x ? frame.camera_x[i] || 0 : 0;
        const ox = t.cx - (cam + mx) * s, oy = t.cy - my * s;
        for (let j = 0; j < T; j++) {
          if (m >= this.tgtM.max) break;
          const o = (i * T + j) * 2;
          const x = ox + tg[o] * s, y = oy + tg[o + 1] * s;
          if (x < t.x || x > t.x + t.w || y < t.y || y > t.y + t.h) continue;
          ab[m * 3] = x; ab[m * 3 + 1] = y; ab[m * 3 + 2] = size;
          m++;
        }
      }
      this.tgtM.buffers.aT.needsUpdate = true;
    }
    this.tgtM.geo.instanceCount = m;
    this.renderer.render(this.scene, this.camera);
  }
}
