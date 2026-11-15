// shapes.js : procedural + replay-driven halftone dot generators.
// Every generator writes into a Writer (w.push(x, y, r, tone)). Coordinates are stage px (1920x1080).
// Tones: 0 light, 1 gray, 2 ink, 3 bad (muted red), 4 accent. Drawn in that order.
(function () {
  const T = { LIGHT: 0, GRAY: 1, INK: 2, BAD: 3, ACC: 4 };
  const TONE_COLORS = ['#D8D8D2', '#B3B3AE', '#111111', '#B8432F', '#FF4D00'];
  const LIGHT = norm3([-0.55, -0.62, 0.56]); // light from top-left-front (screen y down)

  function norm3(v) { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; }
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
  // deterministic hash noise
  function hash(i) { let x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }

  class Writer {
    constructor(cap) { this.X = new Float32Array(cap); this.Y = new Float32Array(cap); this.R = new Float32Array(cap); this.C = new Uint8Array(cap); this.n = 0; this.cap = cap; }
    reset() { this.n = 0; }
    push(x, y, r, c) { if (this.n >= this.cap) return; const i = this.n++; this.X[i] = x; this.Y[i] = y; this.R[i] = r; this.C[i] = c; }
  }

  // ---------- halftone sphere ----------
  const fibCache = {};
  function fib(n) {
    if (fibCache[n]) return fibCache[n];
    const a = new Float32Array(n * 3), g = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < n; i++) {
      const y = 1 - (i / (n - 1)) * 2, rr = Math.sqrt(1 - y * y), th = g * i;
      a[i * 3] = Math.cos(th) * rr; a[i * 3 + 1] = y; a[i * 3 + 2] = Math.sin(th) * rr;
    }
    return (fibCache[n] = a);
  }
  // o: {x, y, R, n, rot, tilt, dot, tone, backTone?}
  function sphere(w, o) {
    const P = fib(o.n), cr = Math.cos(o.rot || 0), sr = Math.sin(o.rot || 0);
    const tl = o.tilt == null ? 0.35 : o.tilt, ct = Math.cos(tl), st = Math.sin(tl);
    const dot = o.dot || 3.2;
    for (let i = 0; i < o.n; i++) {
      let x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
      let x1 = x * cr + z * sr, z1 = -x * sr + z * cr; // rotate about y
      let y2 = y * ct - z1 * st, z2 = y * st + z1 * ct; // tilt about x
      const sx = o.x + x1 * o.R, sy = o.y - y2 * o.R;
      if (z2 < 0.02) { w.push(sx, sy, 0, o.tone); continue; }
      // normal in screen space: (x1, -y2, z2) ; light dot
      const lam = clamp(x1 * LIGHT[0] + (-y2) * LIGHT[1] + z2 * LIGHT[2], 0, 1);
      const shade = 1 - lam; // halftone: darker = bigger dot
      const edge = smooth(0.02, 0.25, z2);
      w.push(sx, sy, dot * (0.22 + 0.95 * shade) * (0.55 + 0.45 * edge), o.tone);
    }
  }

  // ---------- capsule / line primitives ----------
  // dotted segment with fixed count n
  function segment(w, x1, y1, x2, y2, n, r, tone) {
    for (let i = 0; i < n; i++) { const s = n === 1 ? 0.5 : i / (n - 1); w.push(x1 + (x2 - x1) * s, y1 + (y2 - y1) * s, r, tone); }
  }
  // filled, cylinder-shaded capsule: along x across counts fixed
  function capsule(w, x1, y1, x2, y2, rad, along, across, rMax, tone) {
    let dx = x2 - x1, dy = y2 - y1; const L = Math.hypot(dx, dy) || 1e-6; dx /= L; dy /= L;
    const nx = -dy, ny = dx; // perpendicular
    const ndl = nx * -0.6 + ny * -0.8; // perpendicular facing light (up-left)
    for (let a = 0; a < along; a++) {
      const s = along === 1 ? 0.5 : a / (along - 1);
      // extend slightly beyond endpoints for rounded look
      const ext = (s - 0.5) * (L + rad * 0.9);
      const cx = (x1 + x2) / 2 + dx * ext, cy = (y1 + y2) / 2 + dy * ext;
      const endT = Math.abs(s - 0.5) * 2; // 0 middle, 1 ends
      for (let b = 0; b < across; b++) {
        const off = across === 1 ? 0 : (b / (across - 1)) * 2 - 1;
        const taper = across === 1 ? 1 : Math.sqrt(Math.max(0, 1 - Math.pow(endT, 6) * off * off));
        const px = cx + nx * off * rad * 0.82 * taper, py = cy + ny * off * rad * 0.82 * taper;
        const shade = 0.5 - 0.5 * off * ndl; // side away from light gets bigger dots
        w.push(px, py, rMax * (0.5 + 0.62 * shade), tone);
      }
    }
  }

  // ---------- replay robots ----------
  const REPLAY = { gens: null, dt: 0.04, K: 7 };
  const FRAME = new Float32Array(200);
  // returns interpolated frame (shared buffer!) ; mode 'loop' | 'hold'
  function frameAt(gen, t, mode) {
    const F = gen.frames, n = F.length; let fi = t / (gen.dt || REPLAY.dt);
    if (mode === 'hold') fi = clamp(fi, 0, n - 1); else fi = ((fi % (n - 1)) + (n - 1)) % (n - 1);
    const i0 = Math.floor(fi), i1 = Math.min(n - 1, i0 + 1), f = fi - i0, A = F[i0], B = F[i1];
    for (let k = 0; k < A.length; k++) FRAME[k] = A[k] + (B[k] - A[k]) * f;
    return FRAME;
  }
  const layoutCache = {};
  function robotLayout(gen, scale, spacing) {
    const key = (gen.robot || 'walker') + ':' + gen.frames[0].length + ':' + scale.toFixed(2) + ':' + spacing;
    if (layoutCache[key]) return layoutCache[key];
    const f0 = gen.frames[0], lay = [], K = Math.floor(f0.length / 5);
    for (let k = 0; k < K; k++) {
      const o = k * 5, L = Math.hypot(f0[o + 2] - f0[o], f0[o + 3] - f0[o + 1]), rad = f0[o + 4];
      const along = Math.max(2, Math.round(((L + rad) * scale) / spacing) + 1);
      const across = Math.max(1, Math.round((2 * rad * scale * 0.82) / spacing) + 1);
      lay.push([along, across]);
    }
    return (layoutCache[key] = lay);
  }
  function robotCount(gen, scale, spacing) { return robotLayout(gen, scale, spacing).reduce((s, l) => s + l[0] * l[1], 0); }
  // o: {fr, ox, oy, scale, rootWX (world x placed at ox), spacing, rMax, near, far, fade}
  function robot(w, gen, o) {
    const lay = robotLayout(gen, o.scale, o.spacing), F = o.fr, s = o.scale;
    const K = lay.length, walker = K === 7 && (gen.robot || 'walker') === 'walker';
    const order = walker ? [4, 5, 6, 0, 1, 2, 3] : [...Array(K).keys()]; // walker: far leg first
    for (const k of order) {
      const b = k * 5, tone = walker && k >= 4 ? o.far : o.near;
      const x1 = o.ox + (F[b] - o.rootWX) * s, y1 = o.oy - F[b + 1] * s;
      const x2 = o.ox + (F[b + 2] - o.rootWX) * s, y2 = o.oy - F[b + 3] * s;
      capsule(w, x1, y1, x2, y2, F[b + 4] * s, lay[k][0], lay[k][1], o.rMax * (o.fade == null ? 1 : o.fade), tone);
    }
  }

  // ---------- raster text -> halftone dots ----------
  function textDots(text, font, cx, cy, spacing, opts = {}) {
    const c = document.createElement('canvas'), W = 1400, H = 700; c.width = W; c.height = H;
    const g = c.getContext('2d'); g.fillStyle = '#000'; g.font = font; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(text, W / 2, H / 2);
    const d = g.getImageData(0, 0, W, H).data, pts = [];
    for (let y = 0; y < H; y += spacing) for (let x = 0; x < W; x += spacing) {
      // coverage within cell
      let cov = 0, cnt = 0;
      for (let yy = 0; yy < spacing; yy += 2) for (let xx = 0; xx < spacing; xx += 2) { cov += d[((y + yy) * W + (x + xx)) * 4 + 3]; cnt++; }
      cov /= cnt * 255;
      if (cov > 0.08) pts.push([cx + x - W / 2 + spacing / 2, cy + y - H / 2 + spacing / 2, cov]);
    }
    return pts;
  }

  window.Shapes = { T, TONE_COLORS, Writer, sphere, segment, capsule, robot, robotCount, frameAt, REPLAY, textDots, hash, clamp, smooth };
})();
