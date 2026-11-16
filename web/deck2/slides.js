// slides.js : one object per slide. states[] = point-cloud beats (shape + camera), html = text overlay.
// All numbers are copied from web/deck/NOTES.md and web/deck/data/*.json.
import {
  T, R, clamp, smooth, easeIO, easeOut, hash, loopFrame, genFrame, makeRobotSampler, pushRobot, pushStick,
  pushSphere, textPoints, resample, buildLaptop,
} from './shapes3d.js';

const v3 = (a, b, s = 1) => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
// orbit camera around target; shift > 0 moves the target to the RIGHT of frame by `shift` world units
function orbit(tgt, dist, yaw, pitch, shift = 0, fov = 32, lift = 0) {
  const dir = [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
  const right = [Math.cos(yaw), 0, -Math.sin(yaw)];
  const l = v3(v3(tgt, right, -shift), [0, 1, 0], lift);
  return { p: v3(l, dir, dist), l, fov };
}
const F0 = new Float32Array(35), F1 = new Float32Array(35), F2 = new Float32Array(35);

// ---------------------------------------------------------------- shape helpers
function ground(cl, t, speed, o) {
  const sp = o.sp || 0.6, off = ((t * speed) % sp + sp) % sp;
  const nx = Math.round((o.x1 - o.x0) / sp), nz = Math.round((o.z1 - o.z0) / sp);
  for (let j = 0; j <= nz; j++) for (let i = 0; i <= nx; i++) {
    const x = o.x0 + i * sp - off, z = o.z0 + j * sp;
    const f = smooth(0, o.fade || 6, x - o.x0) * smooth(0, o.fade || 6, o.x1 - x) * smooth(0, 3, z - o.z0) * smooth(0, 3, o.z1 - z);
    cl.push(x, o.y || 0, z, 0, 0, 0, (o.dot || 0.1) * f, o.tone ?? T.GRAY);
  }
}
const SMP = {};
const smp = (key, S, sp, inf) => SMP[key] || (SMP[key] = makeRobotSampler(S, sp, inf));
function heroRobot(cl, t, S, sp, o = {}) {
  const fr = o.fr || loopFrame(t * (o.rate || 1), F0);
  pushRobot(cl, smp('h' + S + sp, S, sp, o.inflate || 1.9), fr, { x: o.x || 0, y: o.y || 0, z: o.z || 0, yaw: o.yaw || 0, tone: o.tone ?? T.INK, far: o.far ?? T.GRAY, foot: o.foot, dot: o.dot || 0.95, size: o.size });
}
let LAP = null;
const lap = () => LAP || (LAP = buildLaptop());
function laptop(cl, t, o) {
  const L = lap(), c = Math.cos(o.yaw), s = Math.sin(o.yaw), sc = o.scale || 1;
  for (const p of L.pts) {
    let [x, y, z, nx, ny, nz, d, tone, kind] = p;
    if (kind === 1) { tone = o.chipTone ?? T.DIM; d *= o.chip ?? 1; }
    if (kind === 2 && o.screen) { const r = o.screen(x, y, z); d = r[0]; tone = r[1]; }
    cl.push((x * c + z * s) * sc + (o.x || 0), y * sc + (o.y || 0), (-x * s + z * c) * sc + (o.z || 0), nx * c + nz * s, ny, -nx * s + nz * c, d * sc, tone);
  }
}
function lapChip(yaw) { const x = 0, z = -1.95, c = Math.cos(yaw), s = Math.sin(yaw); return [x * c + z * s, 0.6, -x * s + z * c]; }
function sphereShape(x, y, z, Rr, n, dot, tone, spin = 0.12) { return { frame(t, cl) { pushSphere(cl, x, y, z, Rr, n, t * spin, dot, tone); } }; }

// glyph sets for the particle odometer
let GLYPH = null;
function glyphs() {
  if (GLYPH) return GLYPH; GLYPH = {};
  for (const ch of '0123456789.') {
    const { pts } = textPoints([ch], '400 760px "Bebas Neue"', 820, 11, 'center');
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    GLYPH[ch] = resample(pts.map((p) => [p[0] - cx, p[1]]), ch === '.' ? 300 : 2400);
  }
  return GLYPH;
}
function pushGlyphMorph(cl, a, b, e, x0, y0, sc, depth) {
  // odometer roll: old glyph flicks up and shrinks, new glyph rises in from below
  const out = e < 0.5, k = out ? easeIO(e * 2) : easeIO(e * 2 - 1);
  const G = out ? GLYPH[a] : GLYPH[b], dy = out ? k * 5 : (1 - k) * -5, sz = out ? 1 - k : k;
  const s0 = e <= 0 || e >= 1 ? 1 : sz;
  for (let i = 0; i < G.length; i++) {
    const wx = x0 + G[i][0] * sc, wy = y0 - (G[i][1] - 430) * sc + (e <= 0 || e >= 1 ? 0 : dy);
    cl.push(wx, wy, 0, 0, 0, 0, 0.21 * s0, T.INK);
    cl.push(wx + 0.12, wy - 0.12, -depth, 0, 0, 0, 0.17 * s0, T.LIGHT);
  }
}
// code -> dots
let CODE = null;
const CODE_LINES = ['robot.joint(torso, leg, motor=80)', 'env = mg.make(robot, task, num_envs=4096)', 'mg.train(env)'];
function codePts() {
  if (CODE) return CODE;
  const r = textPoints(CODE_LINES, '500 66px "JetBrains Mono"', 110, 4);
  CODE = r; return r;
}

// QR code
function pushQR(cl, t, cx, cy, cz, cell, yaw) {
  const M = window.QR_MATRIX, n = M.length, c = Math.cos(yaw), s = Math.sin(yaw);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const on = M[j][i]; const finder = (i < 7 && j < 7) || (i >= n - 7 && j < 7) || (i < 7 && j >= n - 7);
    for (let b = 0; b < 2; b++) for (let a = 0; a < 2; a++) {
      const lx = (i - n / 2 + (a + 0.5) / 2) * cell, ly = (n / 2 - j - (b + 0.5) / 2) * cell;
      const wave = 0.5 + 0.5 * Math.sin(t * 2.2 - (i + j) * 0.35);
      const lz = on ? 0.12 * wave : 0;
      cl.push(cx + lx * c + lz * s, cy + ly, cz - lx * s + lz * c, 0, 0, 0, on ? cell * 0.7 : 0, T.INK);
    }
  }
}

// ---------------------------------------------------------------- slides
export function buildSlides() {
  const S = [];
  const loopSpd = () => R.loop.speed;

  // 01 COLD OPEN -----------------------------------------------------
  {
    const Sx = 6.2, cam = (t) => orbit([5.2, 4.9, 0], 25 - 1.5 * easeOut(t / 6), 0.7 - 0.0095 * t, 0.1, 5.6);
    S.push({
      id: 'open', dur: 5.5,
      states: [
        { at: 0, trans: 0.01, cam, shape: sphereShape(5.2, 4.8, 0, 6.2, 16000, 0.165, T.INK, 0.25) },
        { at: 0.2, trans: 2.0, cam, sweep: 'y', spread: 0.22, shape: { frame(t, cl) { heroRobot(cl, t, Sx, 0.08, { x: 5.2 }); ground(cl, t, loopSpd() * Sx, { x0: -22, x1: 34, z0: -9, z1: 9, sp: 0.62, dot: 0.1 }); } } },
      ],
      html: `<div class="kicker fx" data-d="0.3" style="left:110px;top:120px">Bay-Valley Hacks · Elevate the Ordinary</div>
        <div class="h xxl slam" data-d="1.3" style="left:104px;top:160px">MACGYM</div>
        <div class="ln sub big" data-d="2.5" style="left:110px;top:470px;width:720px"><span>An ordinary MacBook.</span></div>
        <div class="ln sub big gray" data-d="2.65" style="left:110px;top:522px;width:720px"><span>A robot-learning lab.</span></div>`,
    });
  }
  // 02 THE ORDINARY THING -------------------------------------------
  {
    const yaw = (t) => -0.42 + 0.1 * Math.sin(t * 0.35) + 0.015 * t;
    const cam = (t) => orbit([1.5, 3.6, 0], 38 - 2 * easeOut(t / 7), 0.18, 0.3, 5.4);
    S.push({
      id: 'ordinary', dur: 8,
      states: [{ at: 0, trans: 2.4, cam, spread: 0.16, shape: { frame(t, cl) { laptop(cl, t, { yaw: yaw(t), x: 1.5, chip: 0.85 + 0.25 * Math.sin(t * 2.4) + smooth(6, 8, t) * 0.6, chipTone: t > 6.6 ? T.ACC : T.DIM }); } } }],
      html: `<div class="kicker fx" data-d="0.6" style="left:110px;top:120px">The ordinary thing</div>
        <div class="h m" style="left:104px;top:160px">
          <div class="ln" data-d="0.8"><span>Every laptop here</span></div><div class="ln" data-d="0.92"><span>has a GPU.</span></div>
          <div class="ln gray" data-d="1.8"><span>For robots,</span></div><div class="ln gray" data-d="1.92"><span>it sits idle.</span></div></div>
        <div class="fx sub" data-d="2.6" style="left:110px;top:780px;width:760px">Fast robot simulators need NVIDIA. Apple's GPU has no maintained robot-RL backend.</div>`,
    });
  }
  // 03 SO I BUILT ----------------------------------------------------
  {
    const G = 64, sp = 1.9, Ss = 1.0;
    const yawL = -0.42 + 0.1 * Math.sin(8 * 0.35) + 0.015 * 8;
    const chip = v3(lapChip(yawL), [1.5, 0, 0]);
    const cam = (t) => { const e = easeOut((t - 1.6) / 6.5); return orbit([0, 0, 6], 32 + 38 * e, -0.55 + 0.12 * e, 0.34 + 0.1 * e, 17 + 7 * e, 34); };
    const close = { p: v3(chip, [2.5, 12, 14]), l: chip, fov: 34 };
    S.push({
      id: 'built', dur: 9,
      states: [{
        at: 0, trans: 2.0, style: 'focus', cam, mode: 'focus', focusMix: 0.7, focus: chip, sweep: 'radial', spread: 0.15,
        camTrans: (p, cf, ct) => { if (p < 0.45) { const e = easeIO(p / 0.45); return { p: lerp(cf.p, close.p, e), l: lerp(cf.l, close.l, e), fov: cf.fov + (close.fov - cf.fov) * e }; } const e = easeIO((p - 0.45) / 0.55); return { p: lerp(close.p, ct.p, e), l: lerp(close.l, ct.l, e), fov: close.fov + (ct.fov - close.fov) * e }; },
        shape: { frame(t, cl) {
          for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
            const k = j * G + i, fr = loopFrame(t + hash(k) * 3.0, F1);
            const x = (i - G / 2) * sp + (j % 2) * sp * 0.5, z = (j - G / 2) * sp * 1.15 + 6;
            const hero = false;
            pushStick(cl, fr, hero ? 1.6 : Ss, x, 0, z, hero ? 0.34 : 0.24, hero ? T.ACC : T.INK, hero ? T.ACC : T.GRAY);
          }
        } },
      }],
      cls: 'veil',
      html: `<div class="kicker fx" data-d="1.5" style="left:110px;top:120px">So I built</div>
        <div class="h xl" style="left:104px;top:160px"><div class="ln" data-d="1.55"><span>MACGYM.</span></div></div>
        <div class="fx sub big" data-d="2.1" style="left:110px;top:400px;width:700px">Write a robot in Python.<br>Train <b><span class="cnt" data-to="4096" data-sep="1" data-d="2.1" data-dur="2.0">0</span> copies</b> on your Mac.<br><span class="gray">About a minute.</span></div>
        <div class="fx sub" data-d="3.0" style="left:110px;top:600px;width:640px">Robots that learn to walk, grasp and balance: for RL classes, games and real robot arms.</div>
        <div class="fx tag" data-d="3.4" style="left:110px;top:740px">PLANAR 2D, BY CHOICE</div>
        <div class="fx spec" data-d="3.6" style="left:110px;top:1000px">Apple M3 Pro, 18 GB</div>`,
    });
  }
  // 03b THE THEME ---------------------------------------------------
  {
    const camA = (t) => orbit([1.5, 3.6, 0], 40, -0.1 + 0.05 * Math.sin(t * 0.3), 0.22, 7.0);
    const camB = (t) => orbit([1.0, 6.0, 0], 50 - 2 * easeOut(t / 6), -0.18 + 0.04 * Math.sin(t * 0.3), 0.14, 12.5);
    const tiers = (cl, t) => {
      const steps = [[-4.8, 2.6, 2.8, T.LIGHT, 0.0], [0.8, 6.6, 3.5, T.GRAY, 0.37], [6.8, 11.0, 4.3, T.INK, 0.71]];
      for (const [x, y, sc, tone, ph] of steps) {
        heroRobot(cl, t + ph, sc, 0.09, { x, y, z: 0, tone, far: tone === T.INK ? T.GRAY : T.LIGHT });
        for (let i = -12; i <= 12; i++) for (let k = -3; k <= 3; k++) cl.push(x + i * 0.16, y - 0.05, k * 0.3, 0, 1, 0, 0.1, T.GRAY);
      }
    };
    S.push({
      id: 'theme', dur: 10,
      states: [
        { at: 0, trans: 2.0, style: 'wipe', cam: camA, sweep: 'x', shape: { frame(t, cl) { laptop(cl, t, { yaw: -0.3 + 0.05 * Math.sin(t * 0.4), x: 1.5, chip: 0.8, chipTone: T.DIM }); } } },
        { at: 3.0, trans: 2.4, cam: camB, spread: 0.2, shape: { frame(t, cl) { laptop(cl, t, { yaw: -0.3 + 0.05 * Math.sin(t * 0.4), x: -2.5, y: -4.2, scale: 0.75, chip: 1.2, chipTone: T.ACC }); tiers(cl, t); } } },
      ],
      html: `<div class="kicker fx" data-d="0.3" style="left:110px;top:120px">The theme</div>
        <div class="h m" style="left:104px;top:160px"><div class="ln" data-d="0.4"><span>Elevate</span></div><div class="ln gray" data-d="0.55"><span>the ordinary.</span></div></div>
        <div class="theme fx" data-d="1.2" style="left:110px;top:480px">
          <div class="th"><span>ordinary</span><span>elevated</span></div>
          <div class="tr fx" data-d="1.4"><span>the laptop already on your desk</span><span>a robot-learning lab</span></div>
          <div class="tr fx" data-d="1.9"><span>a GPU that sits idle</span><span>4,096 robots training at once</span></div>
          <div class="tr fx" data-d="2.4"><span>one minute of your time</span><span>7.8 days of robot practice</span></div>
          <div class="tr fx" data-d="2.9"><span>15 lines of Python</span><span>a robot that learns to walk</span></div>
        </div>`,
    });
  }
  // 04 HOW -----------------------------------------------------------
  {
    const camCode = (t) => orbit([0, 0, 0], 31, -0.12 + 0.03 * Math.sin(t * 0.5), 0.05, 0, 32, -1.2);
    const camLat = (t) => orbit([0, -2, 0], 46, -0.35 + 0.04 * t, 0.5, -1.5, 32, 3.2);
    const camBot = (t) => orbit([0, 5.2, 0], 31, -0.5 + 0.03 * t, 0.16, 5.5, 32, 0);
    const sc = 0.0175;
    const lattice = (cl, t, dim) => {
      const G = 64, w = 26 / G;
      for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
        const wave = 0.5 + 0.5 * Math.sin(t * 3.0 - (i + j) * 0.22);
        for (let q = 0; q < 4; q++) {
          const x = (i - G / 2 + 0.5) * w + (q % 2 - 0.5) * w * 0.38, z = (j - G / 2 + 0.5) * w + ((q >> 1) - 0.5) * w * 0.38;
          cl.push(x, dim ? -0.02 : -2 + wave * 0.35, z, 0, 0, 0, dim ? 0.07 : 0.1 + 0.07 * wave, dim ? T.LIGHT : (wave > 0.6 ? T.ACC : T.DIM));
        }
      }
    };
    S.push({
      id: 'how', dur: 9.5,
      states: [
        { at: 0, trans: 1.6, style: 'wipe', cam: camCode, sweep: 'x', shape: { frame(t, cl) {
          const { pts, W } = codePts();
          for (const p of pts) {
            const line = p[2], typed = (t - 0.9 - line * 0.55) * 1500;
            const on = p[0] < typed ? 1 : 0;
            cl.push((p[0] - W / 2) * sc, -(p[1] - 185) * sc, Math.sin(p[0] * 0.004 + t) * 0.3, 0, 0, 0, on * 0.085, line === 1 && p[0] > 900 * (66 / 66) ? T.ACC : T.INK);
          }
        } } },
        { at: 3.3, trans: 1.4, style: 'wipe', cam: camLat, sweep: 'x', shape: { frame(t, cl) { lattice(cl, t, false); } } },
        { at: 6.3, trans: 2.2, cam: camBot, sweep: 'y', mode: 'focus', focus: [0, 1.5, 0], shape: { frame(t, cl) { heroRobot(cl, t, 5.4, 0.08, {}); lattice(cl, t, true); } } },
      ],
      html: `<div class="kicker fx" data-d="0.4" style="left:110px;top:120px">How it works</div>
        <div class="h l" data-win="0.5,3.2" style="left:104px;top:160px">1 · You describe it.</div>
        <div class="h l" data-win="3.5,6.2" style="left:104px;top:160px">2 · One GPU thread <span class="acc">per robot.</span></div>
        <div class="h l" data-win="6.5,99" style="left:104px;top:160px">3 · It learns <span class="gray">to walk.</span></div>
        <div class="steps fx" data-d="0.5" style="left:110px;top:980px"><i data-win="0,3.3">Python: a joint with motor 80</i><i data-win="3.3,6.3">4,096 threads, one compiled Metal kernel</i><i data-win="6.3,99">PPO in MLX, same memory</i></div>`,
    });
  }
  // 05 IT LEARNS -----------------------------------------------------
  {
    const Sx = 4.3, zs = [-4.2, -2.8, -1.4, 0], tones = [T.FAINT, T.FAINT, T.LIGHT, T.INK], start = 1.6, show = [false, true, true, true];
    const camX = (t) => { const f = genFrame(3, Math.max(0, t - start), F2); return ((f[0] + f[2]) / 2) * Sx; };
    const cam = (t) => { const x = camX(t); return { p: [x - 3.5, 6.8, 25], l: [x + 2.2, 4.0, -1.2], fov: 34 }; };
    S.push({
      id: 'learns', dur: 10.5,
      states: [{ at: 0, style: 'cut', cam, sweep: 'y', shape: { frame(t, cl) {
        const cx = camX(t);
        for (let gi = 1; gi < 4; gi++) {
          const fr = genFrame(gi, Math.max(0, t - start), F0);
          pushRobot(cl, smp('lane', Sx, 0.075, 1.9), fr, { x: 0, y: 0, z: zs[gi], tone: tones[gi], far: gi === 3 ? T.GRAY : T.FAINT, dot: 0.95 });
        }
        const sp = 0.55, base = Math.floor((cx - 22) / sp) * sp;
        for (let i = 0; i < 100; i++) for (let r = 0; r < 14; r++) {
          const x = base + i * sp, f = smooth(cx - 22, cx - 14, x) * smooth(cx + 33, cx + 25, x);
          const meter = Math.abs(x / Sx - Math.round(x / Sx)) < sp / Sx / 2;
          cl.push(x, 0, -6.5 + r * 0.6, 0, 0, 0, (meter ? 0.16 : 0.09) * f, meter ? T.GRAY : T.LIGHT);
        }
      } } }],
      html: `<div class="kicker fx" data-d="0.3" style="left:110px;top:120px">It learns · real replays from one run</div>
        <div class="h m2" style="left:104px;top:160px"><div class="ln" data-d="0.4"><span>From falling to running.</span></div><div class="ln gray" data-d="0.6"><span>In one minute.</span></div></div>
        <div class="meter fx" data-d="1.4" style="right:110px;top:150px"><span class="m-num">0.0</span><small>m · trained 1:00</small></div>
        <div class="genrow" style="left:110px;top:930px"><span class="fx" data-d="1.55"><b>0:00</b> falls</span><span class="fx" data-d="1.7"><b>0:10</b> falls</span><span class="fx" data-d="1.85"><b>0:30</b> 15.1 m</span><span class="fx" data-d="2.0"><b>1:00</b> 21.5 m</span></div>
        <div class="fx spec" data-d="2.6" style="left:110px;top:1000px">5 of 5 seeds learned · 8 s replays</div>`,
      onFrame(tl, eng, L) {
        const f0 = R.gens[3].frames[0], best = genFrame(3, Math.max(0, tl - start), F1);
        L.querySelector('.m-num').textContent = Math.max(0, (best[0] + best[2]) / 2 - (f0[0] + f0[2]) / 2).toFixed(1);
      },
    });
  }
  // 06 THE NUMBER ----------------------------------------------------
  {
    const cam = (t) => orbit([-4.5, 0, 0], 40, 0.16 * Math.sin(t * 0.3) - 0.06, 0.06, 7.2, 32, 0.5);
    const val = (t) => { const x = clamp((t - 1.3) / 4.4, 0, 1); return 3.2 * (x >= 1 ? 1 : 1 - Math.pow(2, -10 * x)); };
    const Sx = 3.0, NR = 6;
    S.push({
      id: 'number', dur: 8.5,
      states: [{ at: 0, trans: 2.2, cam, sweep: 'x', spread: 0.28, shape: { frame(t, cl) {
        glyphs(); const v = val(t), f = v * 10, a = Math.floor(f + 1e-6), fr = f - a;
        const u = Math.floor(v + 1e-6), fu = v - u;
        const sc = 0.019, y0 = 1.2;
        pushGlyphMorph(cl, String(u % 10), String((u + 1) % 10), smooth(0.93, 1, fu), -10, y0, sc, 1.0);
        pushGlyphMorph(cl, '.', '.', 0, -4.8, y0, sc, 1.0);
        pushGlyphMorph(cl, String(a % 10), String((a + 1) % 10), v >= 3.2 ? 0 : smooth(0.4, 1, fr), 0.4, y0, sc, 1.0);
        for (let i = 0; i < NR; i++) {
          const W = 34, x = ((i * (W / NR) + t * loopSpd() * Sx) % W) - 6;
          const fade = smooth(-6, -2, x) * smooth(28, 24, x);
          heroRobot(cl, t + i * 0.37, Sx, 0.085, { x: x - 4.5, y: -10.0, z: 1, size: fade, dot: 0.95 });
        }
      } } }],
      html: `<div class="kicker fx" data-d="0.4" style="left:110px;top:120px">The number · simulated robot time</div>
        <div class="h m" style="left:104px;top:160px"><div class="ln" data-d="0.5"><span>Days of</span></div><div class="ln" data-d="0.62"><span>simulated</span></div><div class="ln" data-d="0.74"><span>practice.</span></div></div>
        <div class="h l acc fx" data-d="1.3" style="left:104px;top:560px"><span class="cnt tab2" data-to="28" data-d="1.3" data-dur="4.4" data-ease="expo">0</span> seconds.</div>
        <div class="fx sub" data-d="2.4" style="left:110px;top:745px;width:820px">Days of practice, squeezed into seconds.</div>
        <div class="fx spec" data-d="3.0" style="left:110px;top:800px">4,096 robots. Live counter from the playground. Apple M3 Pro, 18 GB.</div>`,
    });
  }
  // 08 IT'S REAL -----------------------------------------------------
  {
    const PIV = [7.5, 13.0, 0], Lr = 7.4, per = 1.8049728;
    const th = (t) => (60 * Math.PI / 180) * Math.cos((2 * Math.PI * t) / per);
    const cam = (t) => orbit([0, 6.0, -1], 37, -0.22 + 0.06 * Math.sin(t * 0.3), 0.06, 0, 32, 0);
    const pend = (cl, t, z, tone, dotS) => {
      const a = th(t), dx = Math.sin(a), dy = -Math.cos(a);
      const bob = [PIV[0] + dx * Lr, PIV[1] + dy * Lr, z];
      for (let u = 0; u <= Lr - 1.2; u += 0.16) for (let k = 0; k < 8; k++) { const ang = (k / 8) * Math.PI * 2; const r = 0.2; const nx = Math.cos(ang) * -dy, ny = Math.cos(ang) * dx, nz = Math.sin(ang); cl.push(PIV[0] + dx * u + nx * r, PIV[1] + dy * u + ny * r, z + nz * r, nx, ny, nz, 0.15 * dotS, tone); }
      pushSphere(cl, bob[0], bob[1], bob[2], 2.0, 3600, t * 0.4, 0.125 * dotS, tone);
      pushSphere(cl, PIV[0], PIV[1], z, 0.45, 160, 0, 0.16 * dotS, tone);
    };
    S.push({
      id: 'real', dur: 9.5,
      states: [{ at: 0, trans: 2.2, cam, sweep: 'y', spread: 0.3, shape: { frame(t, cl) { pend(cl, t, -3.2, T.LIGHT, 0.95); pend(cl, t, 0, T.INK, 1); for (let i = -5; i <= 5; i++) cl.push(PIV[0] + i * 0.25, PIV[1] + 0.55, 0, 0, 0, 0, 0.14, T.INK); } } }],
      html: `<div class="kicker fx" data-d="0.3" style="left:110px;top:120px">Is it real?</div>
        <div class="h m" style="left:104px;top:160px"><div class="ln" data-d="0.4"><span>Checked against</span></div><div class="ln gray" data-d="0.55"><span>the pro simulator.</span></div></div>
        <div class="trio">
          <div class="fx" data-d="1.2"><b>0.01%</b><span class="lab">matches the textbook</span><span class="det">pendulum period vs the exact analytic answer (MuJoCo matches too)</span></div>
          <div class="fx" data-d="1.5"><b class="acc">22×</b><span class="lab">faster than the industry standard</span><span class="det">same planar 2D walker · M3 Pro · 4,096 robots · vs MuJoCo C rollout on 11 CPU threads</span></div>
          <div class="fx" data-d="1.8"><b>90%</b><span class="lab">still walks in the pro simulator</span><span class="det">19.7 m vs 22.0 m, sim-to-sim (MuJoCo), 20 runs</span></div>
          <div class="fx" data-d="2.1"><b>2.74<small> m/s</small></b><span class="lab">honest gait, cheat removed</span><span class="det">first robots cheated at 6.16 m/s (unlimited motor power, skating feet); fixed with real motor curves + sticky contacts</span></div>
        </div>
        <div class="fx foot" data-d="2.6">n=20 · MuJoCo 4.0.1 · sim-to-sim, not hardware · Apple M3 Pro, 18 GB</div>`,
    });
  }
  // 09 IT'S A FRAMEWORK ---------------------------------------------
  {
    const Sx = 5.0;
    const cam = (t) => orbit([4.5, 4, 0], 30, 0.45 - 0.02 * t, 0.14, 4.0);
    S.push({
      id: 'framework', dur: 9.5,
      states: [
        { at: 0, style: 'cut', cam, sweep: 'y', shape: { frame(t, cl) { const fr = genFrame(0, Math.max(0, (t - 1.2) * 0.8), F0); const rx = (fr[0] + fr[2]) / 2; for (let k = 0; k < 35; k += 5) { fr[k] -= rx; fr[k + 2] -= rx; } heroRobot(cl, t, Sx, 0.08, { x: 5.5, fr, tone: T.GRAY, far: T.LIGHT }); ground(cl, 0, 0, { x0: -14, x1: 30, z0: -7, z1: 7, sp: 0.62, dot: 0.1 }); } } },
        { at: 5.0, trans: 2.0, cam, sweep: 'y', spread: 0.15, shape: { frame(t, cl) { heroRobot(cl, t, Sx, 0.08, { x: 5.5 }); ground(cl, t, loopSpd() * Sx, { x0: -14, x1: 30, z0: -7, z1: 7, sp: 0.62, dot: 0.1 }); } } },
      ],
      html: `<div class="kicker fx" data-d="0.3" style="left:110px;top:120px">It's a framework · fresh install on this laptop</div>
        <div class="h m" style="left:104px;top:160px"><div class="ln" data-d="0.4"><span>One install.</span></div><div class="ln gray" data-d="0.55"><span>A real library.</span></div></div>
        <div class="term fx" data-d="0.9" style="left:110px;top:460px"><div class="type" data-d="1.1" data-cps="55">$ git clone … &amp;&amp; pip install -e .</div><div class="type dim" data-d="1.9" data-cps="80">Successfully installed macgym-0.3.0  (10.6 s)</div><div class="type" data-d="2.6" data-cps="55">$ macgym train walker</div><div class="type ok" data-d="3.6" data-cps="90">done in 1m13s | best distance 54.26 m</div></div>
        <div class="bigsec fx" data-d="4.6" style="left:110px;top:760px"><b>87 s</b><span>from clone to a trained robot</span></div>
        <div class="fx spec" data-d="5.0" style="left:110px;top:950px"><b>Early access:</b> 2 people have committed to pay.</div>
        <div class="fx spec" data-d="5.4" style="left:110px;top:1000px">Built today with AI coding agents. I designed the API and directed and tested it.</div>`,
    });
  }
  // 10 CLOSE ---------------------------------------------------------
  {
    const camL = (t) => orbit([0, 3.4, 0], 31 - 2 * easeOut(t / 5), 0.22 + 0.05 * Math.sin(t * 0.4), 0.2, 0, 32, 0.6);
    const camQ = (t) => orbit([0, 1, 0], 34, 0.1 * Math.sin(t * 0.35), 0.04, 0);
    const L = lap();
    const screenFn = (t) => (x, y, z) => [0.13, T.LIGHT];
    S.push({
      id: 'close', dur: 11,
      states: [
        { at: 0, trans: 2.6, cam: camL, spread: 0.3, shape: { frame(t, cl) {
          laptop(cl, t, { yaw: 0.04 * Math.sin(t * 0.5), chip: 1.1, chipTone: T.ACC, screen: screenFn(t) });
          // running robots on the screen plane
          const s = L.scr, nrm = s.n;
          for (let j = 0; j < 3; j++) for (let i = 0; i < 6; i++) {
            const fr = loopFrame(t + (i * 3 + j) * 0.29, F1), sc = 1.45;
            const ox = 0.09 + i * 0.152, oy = 0.12 + j * 0.29;
            for (let k = 0; k < 7; k++) { const b = k * 5; const m = k === 0 ? 7 : 5; for (let q = 0; q < m; q++) { const u = q / (m - 1); const px = (fr[b] + (fr[b + 2] - fr[b]) * u) * sc, py = (fr[b + 1] + (fr[b + 3] - fr[b + 1]) * u) * sc; const a = ox + px / s.u[0], bb = oy + py / 10.6; cl.push(s.o[0] + s.u[0] * a + s.v[0] * bb + nrm[0] * 0.08, s.o[1] + s.v[1] * bb + nrm[1] * 0.08, s.o[2] + s.v[2] * bb + nrm[2] * 0.08, 0, 0, 0, k >= 4 ? 0.13 : 0.17, k >= 4 ? T.GRAY : T.INK); } }
          }
        } } },
        { at: 5.2, trans: 1.6, style: 'wipe', cam: camQ, sweep: 'x', shape: { frame(t, cl) { pushQR(cl, t, 8.8, 0.6, 0, 0.42, -0.1 + 0.04 * Math.sin(t * 0.5)); } } },
      ],
      html: `<div class="h l" data-win="0.6,5.0" style="left:104px;top:160px">The same laptop.<br><span class="gray">Now a robot lab.</span></div>
        <div class="h l2" style="left:104px;top:300px"><div class="ln" data-d="5.6"><span>Elevate</span></div><div class="ln acc" data-d="5.8"><span>the ordinary.</span></div></div>
        <div class="fx sub" data-d="6.6" style="left:110px;top:700px">Scan for the code. Live demo: <kbd>L</kbd></div>
        <div class="fx spec" data-d="6.8" style="left:110px;top:1000px">github.com/shlok-madhekar/macgym</div>`,
    });
  }

  // APPENDIX ---------------------------------------------------------
  const apx = (id, kick, title, body) => ({
    id, dur: 8, appendix: true, cls: 'apx',
    states: [{ at: 0, trans: 2, cam: (t) => orbit([0, 0, 0], 40, 0.2 * Math.sin(t * 0.2), 0.1, -9), sweep: 'x', shape: sphereShape(0, 0, 0, 7.5, 9000, 0, T.LIGHT, 0.1) }],
    html: `<div class="kicker fx" data-d="0.2" style="left:110px;top:120px">Appendix · ${kick}</div><div class="h s fx" data-d="0.3" style="left:104px;top:160px">${title}</div><div class="apxbody fx" data-d="0.45">${body}</div>`,
  });
  S.push(apx('a-valid', 'validation vs MuJoCo 4.0.1', 'Textbook physics matches.', `<table>
    <tr><th>Test</th><th>MacGym</th><th>MuJoCo</th></tr>
    <tr><td>Pendulum period (analytic 1.80497 s)</td><td>1.80511 s · 0.008% err</td><td>1.80497 s (Euler) · 1.80497 s (RK4)</td></tr>
    <tr><td>Pendulum energy drift, 5 s</td><td>0.038%</td><td>−0.121% (Euler)</td></tr>
    <tr><td>Walker contact (collapse, tilt, open-loop torques)</td><td colspan="2">within 5 cm for the first 0.6 to 0.8 s, then the contact models differ (hard impulses vs soft constraints); same end pose</td></tr>
    </table>`));
  S.push(apx('a-speed', 'where the speed comes from', 'Speed, step by step.', `<table>
    <tr><th>Same walker algorithm, 16,384 robots</th><th>env-steps/s</th><th></th></tr>
    <tr><td>CPU array ops</td><td>135K</td><td></td></tr>
    <tr><td>Same ops on the GPU</td><td>498K</td><td>×3.7</td></tr>
    <tr><td>One compiled Metal kernel per robot</td><td>16.3M</td><td>×32.8 → 121× total</td></tr>
    <tr><th>vs MuJoCo C rollout (11 threads, best 165K)</th><th></th><th></th></tr>
    <tr><td>MacGym at 4,096 robots</td><td>3.64M</td><td>22×</td></tr>
    <tr><td>MacGym at 16,384 robots</td><td>6.75M</td><td>40×</td></tr>
    </table><p>Physics is simpler on our side: planar 2D, impulse contacts, ground only. Training is PPO-bound: update 72%, physics 10%.</p>`));
  S.push(apx('a-seeds', 'not one lucky run', '5 of 5 seeds learned.', `<table>
    <tr><th>Setup</th><td>mg.train(walker, N=4096) defaults, PPO 256×256, 75 s wall per seed</td></tr>
    <tr><th>Criterion</th><td>distance &gt; 10 m at the last iteration with t ≤ 60 s</td></tr>
    <tr><th>Result</th><td>5 / 5 seeds learned to walk within a minute</td></tr>
    <tr><th>Replays</th><td>0:00 falls (0.8 m) · 0:10 falls (1.5 m) · 0:30 15.1 m · 1:00 21.5 m in 8 s</td></tr>
    </table>`));
  S.push(apx('a-robust', 'sim-to-sim transfer, n=20 each', 'It transfers to MuJoCo.', `<table>
    <tr><th>Condition (in MuJoCo)</th><th>Distance in 8 s</th><th>Falls</th></tr>
    <tr><td>nominal</td><td>19.7 ± 0.9 m</td><td>1/20</td></tr>
    <tr><td>friction ×0.8</td><td>19.7 ± 1.7 m</td><td>1/20</td></tr>
    <tr><td>friction ×1.2</td><td>19.9 ± 0.6 m</td><td>0/20</td></tr>
    <tr><td>mass ×0.8</td><td>19.9 ± 1.4 m</td><td>1/20</td></tr>
    <tr><td>mass ×1.2 (weak case)</td><td>15.6 ± 5.2 m</td><td>3/20</td></tr>
    <tr><td>random init</td><td>20.1 ± 0.5 m</td><td>0/20</td></tr>
    </table><p>MacGym itself: 22.0 m. This is MuJoCo, not real hardware.</p>`));
  S.push(apx('a-who', 'who it is for', 'RL classes on MacBooks.', `<table>
    <tr><th>vs free Colab / MJX</th><td>Runs offline on the laptop you own. No GPU quota, no disconnects, a live viewer and a code playground, nothing to rent.</td></tr>
    <tr><th>First users</th><td>DVHS CS Club (I'm an officer), my hackathons (Cove Hacks, HackPilot). 2 people have committed to pay.</td></tr>
    <tr><th>Next</th><td>MJCF import and 3D. Idea, not built: the same compiler emitting CUDA or ROCm.</td></tr>
    </table>`));
  S.push(apx('a-built', 'built today', 'One day. One laptop.', `<table>
    <tr><th>11:38</th><td>contract + skeleton (first commit)</td></tr>
    <tr><th>11:51</th><td>MLX envs, Metal 2D multibody physics, PPO, viewer</td></tr>
    <tr><th>13:55</th><td>v3 framework: Robot / Task / make / train, CLI, CoreML export, playground</td></tr>
    <tr><th>14:25</th><td>physics realism: warm-start contacts, motor torque-speed curve</td></tr>
    <tr><th>Mine:</th><td>the idea (after rejecting ~20), framework over demo, scrapping the first UI, every experiment. Agents wrote most of the code; I reviewed, tested and directed.</td></tr>
    </table>`));
  return S;
}
function lerp(a, b, e) { return [a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e, a[2] + (b[2] - a[2]) * e]; }
