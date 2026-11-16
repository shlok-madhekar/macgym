// engine.js : one WebGL THREE.Points cloud, GPU morph between per-state point sets, camera rig,
// time-driven HTML text reveals. Everything is a pure function of (slide, local time), so the
// deck can be stepped deterministically for video export via window.__deck.renderAt(t).
import * as THREE from 'three';
import { Cloud, hilbert, hash, clamp, easeIO } from './shapes3d.js';

export const NP = 64000;
const PALETTE = ['#111111', '#8f8f8b', '#cfcfc9', '#ff4d00', '#d23a24', '#f3b08f', '#e6e6e1'];
const PAPER = '#fafaf7';

const VERT = /* glsl */ `
attribute vec4 aP0; attribute vec4 aN0; attribute vec4 aP1; attribute vec4 aN1; attribute vec2 aR;
uniform float uProg, uTime, uPx, uMode, uSeed, uSpread, uWin, uShrink, uCap;
uniform vec3 uFocus, uLight, uCamPos;
uniform vec3 uCol[7];
varying vec3 vCol;
vec3 flow(vec3 p){
  return vec3(sin(p.y*0.37+p.z*0.21+uSeed)+0.5*sin(p.z*0.71+uSeed*1.3),
              sin(p.z*0.33+p.x*0.27+1.7+uSeed)+0.5*sin(p.x*0.63+2.1),
              sin(p.x*0.29+p.y*0.23+3.1+uSeed)+0.5*sin(p.y*0.59+0.7));
}
float ease(float x){ return x<0.5 ? 4.0*x*x*x : 1.0-pow(-2.0*x+2.0,3.0)/2.0; }
void main(){
  float s = clamp((uProg - aR.x*(1.0-uWin))/uWin, 0.0, 1.0);
  s = ease(s);
  vec3 p0 = aP0.xyz, p1 = aP1.xyz;
  vec3 mid = (p0+p1)*0.5;
  float L = length(p1-p0);
  vec3 ctrl = mid + flow(mid*0.09)*min(L,40.0)*uSpread;
  if (uMode > 0.01) ctrl = mix(ctrl, uFocus + flow(mid*0.25)*0.9, uMode);
  float a = 1.0-s;
  vec3 p = a*a*p0 + 2.0*a*s*ctrl + s*s*p1;
  float ph = aR.y*6.2831;
  p += 0.04*vec3(sin(uTime*1.1+p.y*0.9+p.z*0.4), sin(uTime*1.3+p.x*0.7+p.z*0.5), sin(uTime*0.9+p.x*0.5+p.y*0.6));
  float size = mix(aP0.w, aP1.w, s);
  vec3 n = mix(aN0.xyz, aN1.xyz, s);
  int t0 = int(aN0.w+0.5), t1 = int(aN1.w+0.5);
  vec3 c = mix(uCol[t0], uCol[t1], s);
  float ln = length(n);
  if (ln > 0.25) {
    n /= ln;
    float lam = max(dot(n, uLight), 0.0);
    vec3 V = normalize(uCamPos - p);
    float f = dot(n, V);
    size *= 0.22 + 1.08*(1.0-lam)*(1.0-lam*0.2);
    if (f < 0.0) { size *= 0.6; c = mix(c, uCol[2], 0.55*(1.0-sin(3.14159*s))); }
    else size *= 0.75 + 0.25*smoothstep(0.0, 0.35, f);
  }
  size *= 1.0 - uShrink*sin(3.14159*s);
  vec4 mv = modelViewMatrix*vec4(p,1.0);
  gl_Position = projectionMatrix*mv;
  gl_PointSize = clamp(size*uPx/max(-mv.z,0.1), 0.0, mix(uCap, 44.0, (s<=0.0||s>=1.0)?1.0:0.0));
  vCol = c;
}`;
const FRAG = /* glsl */ `
varying vec3 vCol;
void main(){
  vec2 q = gl_PointCoord-0.5; float r = length(q);
  float a = 1.0 - smoothstep(0.40, 0.5, r);
  if (a < 0.02) discard;
  gl_FragColor = vec4(vCol, a);
}`;

export class Engine {
  constructor(stage, canvas, slides) {
    this.stage = stage; this.canvas = canvas; this.slides = slides;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    this.renderer.setClearColor(PAPER, 1);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 16 / 9, 0.1, 600);
    const g = new THREE.BufferGeometry();
    const mk = (k) => { const a = new THREE.BufferAttribute(new Float32Array(NP * k), k); a.setUsage(THREE.DynamicDrawUsage); return a; };
    this.A = { aP0: mk(4), aN0: mk(4), aP1: mk(4), aN1: mk(4), aR: mk(2) };
    for (const k in this.A) g.setAttribute(k, this.A[k]);
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NP * 3), 3));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
    this.uni = {
      uProg: { value: 1 }, uTime: { value: 0 }, uPx: { value: 1000 }, uMode: { value: 0 }, uSeed: { value: 0 }, uSpread: { value: 0.3 }, uWin: { value: 0.58 }, uShrink: { value: 0.5 }, uCap: { value: 44 },
      uFocus: { value: new THREE.Vector3() }, uLight: { value: new THREE.Vector3(-0.5, 0.75, 0.45).normalize() }, uCamPos: { value: new THREE.Vector3() },
      uCol: { value: PALETTE.map((c) => new THREE.Color(c)) },
    };
    this.mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uni, transparent: true, depthWrite: true, depthTest: true, alphaToCoverage: true });
    this.points = new THREE.Points(g, this.mat); this.points.frustumCulled = false; this.scene.add(this.points);
    this.clA = new Cloud(140000); this.clB = new Cloud(140000);
    this.toState = null; this.fps = { last: 0, acc: [], value: 0 };
    this.layers = slides.map((s, i) => { const d = document.createElement('div'); d.className = 'layer ' + (s.cls || ''); d.innerHTML = s.html || ''; d.style.visibility = 'hidden'; stage.querySelector('#text').appendChild(d); return d; });
    this.resize(); window.addEventListener('resize', () => this.resize());
  }
  resize() {
    const w = window.innerWidth, h = window.innerHeight, sc = Math.min(w / 1920, h / 1080);
    const L = (w - 1920 * sc) / 2, Tp = (h - 1080 * sc) / 2;
    this.stage.style.transform = `translate(${L}px,${Tp}px) scale(${sc})`;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(dpr); this.renderer.setSize(1920 * sc, 1080 * sc, false);
    Object.assign(this.canvas.style, { left: L + 'px', top: Tp + 'px', width: 1920 * sc + 'px', height: 1080 * sc + 'px' });
    this.pxH = 1080 * sc * dpr;
  }
  // ---------------------------------------------------------------- state resolution
  statesAt(si, tl) { const st = this.slides[si].states; let k = 0; for (let i = 0; i < st.length; i++) if (tl >= st[i].at) k = i; return k; }
  camOf(state, t) { const c = state.cam(t); return { p: c.p, l: c.l, fov: c.fov || 32 }; }
  mapping(state, t) {
    if (state._map) return state._map;
    const cl = this.clA; cl.reset(); state.shape.frame(t, cl); const n = cl.n;
    const cam = this.camOf(state, t); const C = new THREE.PerspectiveCamera(cam.fov, 16 / 9, 0.1, 600);
    C.position.set(...cam.p); C.lookAt(...cam.l); C.updateMatrixWorld(); C.updateProjectionMatrix();
    const v = new THREE.Vector3(), sx = new Float32Array(n), sy = new Float32Array(n);
    for (let i = 0; i < n; i++) { v.set(cl.P[i * 3], cl.P[i * 3 + 1], cl.P[i * 3 + 2]).project(C); sx[i] = clamp(v.x, -1.5, 1.5); sy[i] = clamp(v.y, -1.5, 1.5); }
    const H = 1024, key = new Float64Array(n), idx = new Int32Array(n);
    for (let i = 0; i < n; i++) { key[i] = hilbert(H, Math.floor(((sx[i] + 1.5) / 3) * (H - 1)), Math.floor(((sy[i] + 1.5) / 3) * (H - 1))) + i * 1e-7; idx[i] = i; }
    idx.sort((a, b) => key[a] - key[b]);
    const map = new Int32Array(NP), delay = new Float32Array(NP);
    let fx = 0, fy = 0;
    if (state.focus) { v.set(...state.focus).project(C); fx = v.x; fy = v.y; }
    for (let i = 0; i < NP; i++) {
      const j = idx[Math.floor((i * n) / NP)]; map[i] = j;
      let k;
      if (state.sweep === 'radial') k = clamp(Math.hypot(sx[j] - fx, (sy[j] - fy) * 0.56) / 1.3, 0, 1);
      else if (state.sweep === 'y') k = clamp((1 - sy[j]) / 2, 0, 1);
      else if (state.sweep === 'none') k = 0;
      else k = clamp((sx[j] + 1) / 2, 0, 1);
      const jit = state.style === 'wipe' ? 0.03 : 0.2; delay[i] = clamp(k * (1 - jit) + hash(i) * jit, 0, 1);
    }
    return (state._map = { map, delay, n });
  }
  fill(cl, m, P, Nn) {
    const pa = P.array, na = Nn.array, map = m.map, CP = cl.P, CN = cl.N, CS = cl.S, CC = cl.C;
    for (let i = 0; i < NP; i++) {
      const j = map[i], i4 = i * 4, j3 = j * 3;
      pa[i4] = CP[j3]; pa[i4 + 1] = CP[j3 + 1]; pa[i4 + 2] = CP[j3 + 2]; pa[i4 + 3] = CS[j];
      na[i4] = CN[j3]; na[i4 + 1] = CN[j3 + 1]; na[i4 + 2] = CN[j3 + 2]; na[i4 + 3] = CC[j];
    }
    P.needsUpdate = true; Nn.needsUpdate = true;
  }
  // ---------------------------------------------------------------- frame
  // cur: {si, tl}; from: {si, tl0} | null ; T: global seconds for drift
  draw(cur, from, T) {
    const slide = this.slides[cur.si], k = this.statesAt(cur.si, cur.tl), to = slide.states[k];
    let fromState = null, fromT = 0, prog = 1, tr = to.style === 'cut' ? 0.14 : (to.trans ?? 2.2);
    if (k > 0) { fromState = slide.states[k - 1]; fromT = cur.tl; prog = (cur.tl - to.at) / tr; }
    else if (from) { const fs = this.slides[from.si]; fromState = fs.states[this.statesAt(from.si, from.tl0)]; fromT = from.tl0 + cur.tl; prog = cur.tl / tr; }
    prog = clamp(prog, 0, 1);
    const mTo = this.mapping(to, cur.tl);
    if (this.toState !== to) {
      this.toState = to; this.A.aR.array.set(this.interleave(mTo.delay)); this.A.aR.needsUpdate = true;
      this.uni.uMode.value = to.mode === 'focus' ? (to.focusMix ?? 0.9) : 0; if (to.focus) this.uni.uFocus.value.set(...to.focus);
      this.uni.uSeed.value = (this.slides.indexOf(slide) * 1.7 + k * 0.9) % 6.28; this.uni.uSpread.value = to.spread ?? 0.22; const sty = to.style || 'flow'; this.uni.uWin.value = sty === 'wipe' ? 0.22 : sty === 'cut' ? 1.0 : 0.58; this.uni.uShrink.value = sty === 'focus' ? 0.75 : 0.5; this.uni.uCap.value = sty === 'focus' ? 16 : 44; if (sty === 'wipe' || sty === 'cut') this.uni.uSpread.value = to.spread ?? 0.04;
    }
    const cB = this.clB; cB.reset(); to.shape.frame(cur.tl, cB); this.fill(cB, mTo, this.A.aP1, this.A.aN1);
    if (prog < 1 && fromState) {
      const mF = this.mapping(fromState, fromT); const cA = this.clA; cA.reset(); fromState.shape.frame(fromT, cA); this.fill(cA, mF, this.A.aP0, this.A.aN0);
    }
    this.uni.uProg.value = fromState ? prog : 1;
    // camera
    let cam = this.camOf(to, cur.tl);
    if (prog < 1 && fromState) {
      const cf = this.camOf(fromState, fromT);
      if (to.camTrans) cam = to.camTrans(prog, cf, cam);
      else { const e = easeIO(clamp(prog * 1.05, 0, 1)); cam = { p: lerp3(cf.p, cam.p, e), l: lerp3(cf.l, cam.l, e), fov: cf.fov + (cam.fov - cf.fov) * e }; }
    }
    const C = this.camera, d = 0.35;
    C.position.set(cam.p[0] + Math.sin(T * 0.21) * d, cam.p[1] + Math.sin(T * 0.17 + 1) * d * 0.6, cam.p[2] + Math.sin(T * 0.13 + 2) * d);
    C.fov = cam.fov; C.aspect = 16 / 9; C.updateProjectionMatrix(); C.lookAt(cam.l[0], cam.l[1], cam.l[2]); C.updateMatrixWorld();
    this.uni.uCamPos.value.copy(C.position); this.uni.uTime.value = T;
    this.uni.uPx.value = this.pxH / (2 * Math.tan((C.fov * Math.PI) / 360));
    this.renderer.render(this.scene, this.camera);
    this.text(cur, from);
    const now = performance.now(); if (this.fps.last) { this.fps.acc.push(now - this.fps.last); if (this.fps.acc.length > 120) this.fps.acc.shift(); } this.fps.last = now;
  }
  interleave(delay) { const a = new Float32Array(NP * 2); for (let i = 0; i < NP; i++) { a[i * 2] = delay[i]; a[i * 2 + 1] = hash(i * 3.7 + 1); } return a; }
  project(p) { const v = new THREE.Vector3(...p).project(this.camera); return [((v.x + 1) / 2) * 1920, ((1 - v.y) / 2) * 1080, v.z]; }
  // ---------------------------------------------------------------- text
  text(cur, from) {
    this.layers.forEach((L, i) => {
      if (i === cur.si) { L.style.visibility = 'visible'; L.style.opacity = 1; L.style.transform = ''; animLayer(L, cur.tl); }
      else if (from && i === from.si && cur.tl < 0.6) { const e = clamp(cur.tl / 0.5, 0, 1); L.style.visibility = 'visible'; L.style.opacity = 1 - e; L.style.transform = `translateY(${-40 * e}px)`; animLayer(L, from.tl0 + cur.tl); }
      else L.style.visibility = 'hidden';
    });
    const s = this.slides[cur.si]; if (s.onFrame) s.onFrame(cur.tl, this, this.layers[cur.si]);
  }
}
const lerp3 = (a, b, e) => [a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e, a[2] + (b[2] - a[2]) * e];
const eo = (x) => 1 - Math.pow(1 - clamp(x, 0, 1), 4);
function fmt(v, dec, sep) { let s = v.toFixed(dec); if (sep) { const [a, b] = s.split('.'); s = a.replace(/\B(?=(\d{3})+(?!\d))/g, ',') + (b ? '.' + b : ''); } return s; }
function animLayer(L, t) {
  if (!L._els) L._els = { ln: [...L.querySelectorAll('.ln')], fx: [...L.querySelectorAll('.fx')], cnt: [...L.querySelectorAll('.cnt')], win: [...L.querySelectorAll('[data-win]')], strike: [...L.querySelectorAll('.strike')], type: [...L.querySelectorAll('.type')], slam: [...L.querySelectorAll('.slam')], grow: [...L.querySelectorAll('.grow')] };
  const E = L._els;
  for (const el of E.ln) { const d = +el.dataset.d || 0, e = eo((t - d) / 0.75); el.firstElementChild.style.transform = `translateY(${(1 - e) * 112}%)`; }
  for (const el of E.fx) { const d = +el.dataset.d || 0, e = eo((t - d) / 0.6); el.style.opacity = e; el.style.transform = `translateY(${(1 - e) * 26}px)`; }
  for (const el of E.slam) { const d = +el.dataset.d || 0, x = clamp((t - d) / 0.55, 0, 1), e = eo(x); el.style.opacity = clamp(x * 3, 0, 1); el.style.transform = `scale(${1.6 - 0.6 * e})`; el.style.letterSpacing = `${(1 - e) * 0.25}em`; }
  for (const el of E.cnt) {
    const d = +el.dataset.d || 0, dur = +el.dataset.dur || 1.6, a = +el.dataset.from || 0, b = +el.dataset.to;
    const x = clamp((t - d) / dur, 0, 1), e = el.dataset.lin ? x : el.dataset.ease === 'expo' ? (x >= 1 ? 1 : 1 - Math.pow(2, -10 * x)) : 1 - Math.pow(1 - x, 3);
    el.textContent = (el.dataset.pre || '') + fmt(a + (b - a) * e, +el.dataset.dec || 0, el.dataset.sep) + (el.dataset.suf || '');
  }
  for (const el of E.win) { const [a, b] = el.dataset.win.split(',').map(Number); const e = Math.min(eo((t - a) / 0.45), 1 - eo((t - b) / 0.35)); el.style.opacity = e; el.style.transform = `translateY(${(1 - Math.min(1, eo((t - a) / 0.45))) * 40 + (t > b ? -30 * eo((t - b) / 0.35) : 0)}px)`; }
  for (const el of E.grow) { const d = +el.dataset.d || 0, k = eo((t - d) / 0.55); el.style.maxWidth = (k * 900).toFixed(1) + 'px'; el.style.opacity = Math.min(1, k * 1.5); }
  for (const el of E.strike) { const d = +el.dataset.d || 0; el.style.setProperty('--k', eo((t - d) / 0.35)); }
  for (const el of E.type) {
    if (!el._full) el._full = el.textContent; const d = +el.dataset.d || 0, cps = +el.dataset.cps || 40;
    const n = Math.floor(clamp((t - d) * cps, 0, el._full.length)); const s = el._full.slice(0, n);
    if (el._last !== n) { el.textContent = s || '\u00a0'; el._last = n; } el.style.opacity = t >= d ? 1 : 0;
    el.classList.toggle('caret', t >= d && n < el._full.length);
  }
}
