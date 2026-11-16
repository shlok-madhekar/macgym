// shapes3d.js : 3D point-cloud generators for the MacGym v2 deck.
// Every shape is an object { frame(t, cloud) } that pushes a CONSTANT number of points per call
// (fixed order), so the engine can build a stable slot mapping once and animate it every frame.
// World units: hero robot ~9 units tall. Normals drive halftone shading in the shader
// (zero normal = flat, unshaded dot). Tones index the palette in engine.js.

export const T = { INK: 0, GRAY: 1, LIGHT: 2, ACC: 3, BAD: 4, DIM: 5, FAINT: 6 };

export class Cloud {
  constructor(cap) {
    this.cap = cap; this.P = new Float32Array(cap * 3); this.N = new Float32Array(cap * 3);
    this.S = new Float32Array(cap); this.C = new Float32Array(cap); this.n = 0;
  }
  reset() { this.n = 0; }
  push(x, y, z, nx, ny, nz, s, c) {
    if (this.n >= this.cap) return; const i = this.n++, j = i * 3;
    this.P[j] = x; this.P[j + 1] = y; this.P[j + 2] = z; this.N[j] = nx; this.N[j + 1] = ny; this.N[j + 2] = nz;
    this.S[i] = s; this.C[i] = c;
  }
}

export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
export const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
export const easeIO = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
export const easeOut = (x) => 1 - Math.pow(1 - clamp(x, 0, 1), 3);
export function hash(i) { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }

// ---------------------------------------------------------------- replay data
export const R = { gens: null, dt: 0.04, loop: null };
export async function loadReplay(url) {
  const j = await (await fetch(url)).json();
  R.gens = j.generations; R.dt = j.dt || 0.04; R.loop = findLoop(R.gens[R.gens.length - 1]);
  return R;
}
const rootX = (f) => (f[0] + f[2]) / 2;
function findLoop(g) {
  const F = g.frames; let best = [1e9, 60, 90];
  for (let i = 60; i < 130; i++) for (let j = i + 12; j < Math.min(F.length, i + 60); j++) {
    let d = 0; const ri = rootX(F[i]), rj = rootX(F[j]);
    for (let k = 0; k < 35; k++) { if (k % 5 === 4) continue; const a = F[i][k] - (k % 5 === 0 || k % 5 === 2 ? ri : 0), b = F[j][k] - (k % 5 === 0 || k % 5 === 2 ? rj : 0); d += (a - b) * (a - b); }
    if (d < best[0]) best = [d, i, j];
  }
  const [, i0, i1] = best, P = i1 - i0;
  return { i0, P, speed: (rootX(F[i1]) - rootX(F[i0])) / (P * R.dt) };
}
const FR = new Float32Array(35);
// root-relative looping run frame. phase in seconds.
export function loopFrame(t, out = FR) {
  const g = R.gens[R.gens.length - 1], L = R.loop, F = g.frames;
  let fi = ((t / R.dt) % L.P + L.P) % L.P; const i = Math.floor(fi), f = fi - i;
  const A = F[L.i0 + i], B = F[L.i0 + Math.min(i + 1, L.P)]; const ra = rootX(A), rb = rootX(B);
  for (let k = 0; k < 35; k++) {
    const m = k % 5; let a = A[k], b = B[k];
    if (m === 0 || m === 2) { a -= ra; b -= rb; }
    out[k] = a + (b - a) * f;
  }
  return out;
}
// absolute replay frame of generation gi, holding the last frame
export function genFrame(gi, t, out = FR) {
  const F = R.gens[gi].frames; const fi = clamp(t / R.dt, 0, F.length - 1), i = Math.floor(fi), i1 = Math.min(F.length - 1, i + 1), f = fi - i;
  for (let k = 0; k < 35; k++) out[k] = F[i][k] + (F[i1][k] - F[i][k]) * f;
  return out;
}

// ---------------------------------------------------------------- robots
// Precomputed surface samples per capsule (u along axis in [0,1] incl. caps, ang, rho, nu, nr)
const ZSIDE = [0, 1, 1, 1, -1, -1, -1]; // walker: torso center, near leg +z, far leg -z
export function makeRobotSampler(scale, sp, inflate = 1.8, zsep = 0.11) {
  const f0 = R.gens[R.gens.length - 1].frames[0];
  const caps = [];
  for (let k = 0; k < 7; k++) {
    const o = k * 5, L = Math.hypot(f0[o + 2] - f0[o], f0[o + 3] - f0[o + 1]) * scale, r = f0[o + 4] * inflate * scale * (k === 0 ? 1.35 : 1);
    const S = [];
    for (let u = -r; u <= L + r + 1e-6; u += sp * 0.9) {
      let rho = r, nu = 0, nr = 1;
      if (u < 0) { rho = Math.sqrt(Math.max(0, r * r - u * u)); nu = u / r; nr = rho / r; }
      if (u > L) { const d = u - L; rho = Math.sqrt(Math.max(0, r * r - d * d)); nu = d / r; nr = rho / r; }
      const m = Math.max(1, Math.round((2 * Math.PI * rho) / sp));
      for (let a = 0; a < m; a++) { const ang = (a / m) * Math.PI * 2 + (Math.round(u / sp) % 2) * (Math.PI / m); S.push(u / Math.max(L, 1e-6), u, ang, rho, nu, nr); }
    }
    caps.push({ S: new Float32Array(S), L0: L, n: S.length / 6 });
  }
  const count = caps.reduce((s, c) => s + c.n, 0);
  return { caps, count, scale, sp, zsep: zsep * scale };
}
// push one robot. fr: 35-float frame in meters. o: {x,y,z offsets (world), rot (yaw), tone, far tone, size mult, skip}
export function pushRobot(cl, smp, fr, o) {
  const s = smp.scale, dot = smp.sp * (o.dot || 1.25);
  const cy = Math.cos(o.yaw || 0), sy = Math.sin(o.yaw || 0);
  for (let k = 0; k < 7; k++) {
    const b = k * 5, c = smp.caps[k];
    const x1 = fr[b] * s, y1 = fr[b + 1] * s, x2 = fr[b + 2] * s, y2 = fr[b + 3] * s;
    let ax = x2 - x1, ay = y2 - y1; const L = Math.hypot(ax, ay) || 1e-6; ax /= L; ay /= L;
    const bx = -ay, by = ax; const zo = ZSIDE[k] * smp.zsep;
    const tone = k >= 4 ? (o.far ?? T.GRAY) : (k >= 1 && o.leg != null ? o.leg : o.tone ?? T.INK);
    const footTone = (k === 3 || k === 6) && o.foot != null ? o.foot : tone;
    const S = c.S;
    for (let i = 0; i < c.n; i++) {
      const q = i * 6; const ang = S[q + 2], rho = S[q + 3], nu = S[q + 4], nr = S[q + 5];
      const u = S[q] * L, ca = Math.cos(ang), sa = Math.sin(ang);
      // local: along axis (ax,ay,0), perp (bx,by,0), depth (0,0,1)
      let px = x1 + ax * u + bx * ca * rho, py = y1 + ay * u + by * ca * rho, pz = sa * rho + zo;
      let nx = ax * nu + bx * ca * nr, ny = ay * nu + by * ca * nr, nz = sa * nr;
      // yaw about y
      const wx = px * cy + pz * sy, wz = -px * sy + pz * cy, mx = nx * cy + nz * sy, mz = -nx * sy + nz * cy;
      cl.push(wx + o.x, py + o.y, wz + o.z, mx, ny, mz, dot * (o.size ?? 1), footTone);
    }
  }
}
// cheap stick robot (points along axes) for crowds: returns per-robot count
export function stickCount() { return 13; }
const STICK = [[0, 4], [1, 2], [2, 2], [3, 1], [4, 2], [5, 1], [6, 1]];
export function pushStick(cl, fr, s, x, y, z, size, tone, far) {
  for (const [k, m] of STICK) {
    const b = k * 5;
    for (let i = 0; i < m; i++) {
      const u = m === 1 ? 0.5 : i / (m - 1);
      const px = (fr[b] + (fr[b + 2] - fr[b]) * u) * s, py = (fr[b + 1] + (fr[b + 3] - fr[b + 1]) * u) * s;
      cl.push(x + px, y + py, z + ZSIDE[k] * 0.12 * s, 0, 0, 0, size * (k === 0 ? 1.25 : 1), k >= 4 ? far : tone);
    }
  }
}

// ---------------------------------------------------------------- primitives
const fibCache = {};
export function fib(n) {
  if (fibCache[n]) return fibCache[n];
  const a = new Float32Array(n * 3), g = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) { const y = 1 - (i / (n - 1)) * 2, rr = Math.sqrt(1 - y * y), th = g * i; a[i * 3] = Math.cos(th) * rr; a[i * 3 + 1] = y; a[i * 3 + 2] = Math.sin(th) * rr; }
  return (fibCache[n] = a);
}
export function pushSphere(cl, cx, cy, cz, R0, n, rot, dot, tone) {
  const P = fib(n), c = Math.cos(rot), s = Math.sin(rot);
  for (let i = 0; i < n; i++) {
    const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2]; const x1 = x * c + z * s, z1 = -x * s + z * c;
    cl.push(cx + x1 * R0, cy + y * R0, cz + z1 * R0, x1, y, z1, dot, tone);
  }
}
// filled rectangle of dots in a plane given origin o, axes u,v (vectors), counts, normal
export function pushPlane(cl, o, u, v, nu, nv, nrm, dot, tone, fn) {
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const a = nu === 1 ? 0.5 : i / (nu - 1), b = nv === 1 ? 0.5 : j / (nv - 1);
    let d = dot, t = tone;
    if (fn) { const r = fn(a, b, i, j); if (r) { d = r[0]; t = r[1]; } }
    cl.push(o[0] + u[0] * a + v[0] * b, o[1] + u[1] * a + v[1] * b, o[2] + u[2] * a + v[2] * b, nrm[0], nrm[1], nrm[2], d, t);
  }
}

// ---------------------------------------------------------------- text -> dots (2D raster)
export function textPoints(lines, font, lineH, sp, align = 'left') {
  const c = document.createElement('canvas'), W = 2400, H = Math.ceil(lineH * lines.length + 40);
  c.width = W; c.height = H; const g = c.getContext('2d');
  g.fillStyle = '#000'; g.font = font; g.textBaseline = 'middle'; g.textAlign = align;
  lines.forEach((l, i) => g.fillText(l, align === 'center' ? W / 2 : 10, 20 + lineH * (i + 0.5)));
  const d = g.getImageData(0, 0, W, H).data, pts = []; let maxX = 0;
  for (let y = 0; y < H; y += sp) for (let x = 0; x < W; x += sp) {
    const a = d[((y | 0) * W + (x | 0)) * 4 + 3]; if (a > 110) { pts.push([x, y, Math.floor((y - 20) / lineH)]); if (x > maxX) maxX = x; }
  }
  return { pts, W: maxX, H };
}
// resample a point list to exactly m points by rank along a stable order (sorted by x then y)
export function resample(pts, m) {
  const s = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]); const out = [];
  for (let i = 0; i < m; i++) out.push(s[Math.floor((i * s.length) / m)]);
  return out;
}

// ---------------------------------------------------------------- laptop
// Built once into static arrays, transformed per frame (yaw). Returns {pts:[x,y,z,nx,ny,nz,dot,tone,kind]} and screen frame.
export function buildLaptop(W = 17, D = 11.5, sp = 0.2) {
  const pts = []; const add = (x, y, z, nx, ny, nz, d, t, k = 0) => pts.push([x, y, z, nx, ny, nz, d, t, k]);
  const H = 0.55, x0 = -W / 2, z0 = -D / 2;
  const dot = sp * 0.9;
  // base top: keyboard + trackpad + chip
  const nu = Math.round(W / sp), nv = Math.round(D / sp);
  for (let j = 0; j <= nv; j++) for (let i = 0; i <= nu; i++) {
    const x = x0 + (i / nu) * W, z = z0 + (j / nv) * D;
    const kx = (x - x0) / W, kz = (z - z0) / D; let d = dot, t = T.INK, k = 0;
    const inKeys = kx > 0.09 && kx < 0.91 && kz > 0.1 && kz < 0.56;
    const inPad = kx > 0.33 && kx < 0.67 && kz > 0.64 && kz < 0.92;
    const chip = Math.abs(kx - 0.5) < 0.085 && Math.abs(kz - 0.33) < 0.13;
    if (chip) { t = T.ACC; k = 1; d = dot * 1.25; }
    else if (inKeys) { const gx = ((kx - 0.09) / 0.82) * 14, gz = ((kz - 0.1) / 0.46) * 6; const fx = gx - Math.floor(gx), fz = gz - Math.floor(gz); d = fx > 0.12 && fx < 0.88 && fz > 0.12 && fz < 0.88 ? dot * 1.35 : dot * 0.35; }
    else if (inPad) d = dot * 0.55;
    add(x, H, z, 0, 1, 0, d, t, k);
  }
  // front + sides of base
  for (let j = 0; j <= 3; j++) for (let i = 0; i <= nu; i++) add(x0 + (i / nu) * W, (j / 3) * H, -z0, 0, 0, 1, dot, T.INK);
  for (const sx of [-1, 1]) for (let j = 0; j <= 3; j++) for (let i = 0; i <= nv; i++) add(sx * W / 2, (j / 3) * H, z0 + (i / nv) * D, sx, 0, 0, dot, T.INK);
  // lid: hinged at back edge (z0, y=H), opened by angle; screen face normal toward +z when upright
  const LH = 10.6, open = (108 * Math.PI) / 180; // angle between base and lid
  const tilt = Math.PI - open; // lid leaning back from vertical
  const up = [0, Math.cos(tilt), -Math.sin(tilt)], nrm = [0, Math.sin(tilt), Math.cos(tilt)];
  const nh = Math.round(LH / sp);
  const scr = { o: [x0, H, z0], u: [W, 0, 0], v: [up[0] * LH, up[1] * LH, up[2] * LH], n: nrm };
  for (let j = 0; j <= nh; j++) for (let i = 0; i <= nu; i++) {
    const a = i / nu, b = j / nh; const inScreen = a > 0.04 && a < 0.96 && b > 0.06 && b < 0.95;
    const x = x0 + a * W, y = H + up[1] * LH * b, z = z0 + up[2] * LH * b;
    // screen face (front)
    add(x + nrm[0] * 0.02, y + nrm[1] * 0.02, z + nrm[2] * 0.02, inScreen ? 0 : nrm[0], inScreen ? 0 : nrm[1], inScreen ? 0 : nrm[2], inScreen ? dot * 0.62 : dot, inScreen ? T.GRAY : T.INK, inScreen ? 2 : 0);
    // back of lid (only sparse every other)
    if ((i + j) % 2 === 0) add(x - nrm[0] * 0.3, y - nrm[1] * 0.3, z - nrm[2] * 0.3, -nrm[0], -nrm[1], -nrm[2], dot * 1.2, T.INK, 0);
  }
  return { pts, scr, dot };
}
// Hilbert curve index (n must be power of 2)
export function hilbert(n, x, y) {
  let d = 0;
  for (let s = n >> 1; s > 0; s >>= 1) {
    const rx = (x & s) > 0 ? 1 : 0, ry = (y & s) > 0 ? 1 : 0; d += s * s * ((3 * rx) ^ ry);
    if (ry === 0) { if (rx === 1) { x = n - 1 - x; y = n - 1 - y; } const t = x; x = y; y = t; }
  }
  return d;
}
