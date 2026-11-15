// Fake server for ?demo=1. Emits the exact same message shapes as macgym/server.py (CONTRACT §3).

const ENVS = [
  { name: "walker", obs_dim: 17, act_dim: 6, description: "Planar biped learns to walk" },
  { name: "cheetah", obs_dim: 17, act_dim: 6, description: "Planar quadruped sprints" },
  { name: "arm", obs_dim: 10, act_dim: 2, description: "2-link arm reaches target" },
  { name: "cartpole", obs_dim: 4, act_dim: 1, description: "Balance a pole on a cart" },
];

const COLORS = {
  walker: ["#4FD1FF", "#A3FF6B", "#A3FF6B", "#7CF2C8", "#7CF2C8", "#FFD36B", "#FFD36B"],
  cheetah: ["#4FD1FF", "#A3FF6B", "#7CF2C8", "#FFD36B", "#A3FF6B", "#7CF2C8", "#FFD36B"],
  arm: ["#5A6378", "#4FD1FF", "#A3FF6B"],
  cartpole: ["#4FD1FF", "#A3FF6B"],
};
const MAX_RET = { walker: 3200, cheetah: 4800, arm: -2, cartpole: 500 };
const MIN_RET = { walker: -120, cheetah: -300, arm: -95, cartpole: 12 };

function rand(seed) { const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }

export class DemoServer {
  constructor(onMessage) {
    this.emit = onMessage;
    this.state = "idle";
    this.env = "walker";
    this.N = 4096; this.R = 64;
    this.iter = 0; this.steps = 0; this.t0 = 0;
    this.target = [0.55, 0.65];
    this.sim = 0; // sim time
    this.envT = new Float32Array(1024);
    this.envX = new Float32Array(1024);
    this.timers = [];
    this.lastFrame = performance.now();
    setTimeout(() => this.emit({ type: "status", state: "idle", env: this.env }), 50);
    this._frameLoop();
  }

  send(msg) {
    switch (msg.cmd) {
      case "list": setTimeout(() => this.emit({ type: "envs", envs: ENVS }), 30); break;
      case "bench": setTimeout(() => this.emit({ type: "bench", data: null }), 30); break;
      case "start": this.start(msg); break;
      case "stop": this.stop(); break;
      case "showcase": if (this.state !== "idle") { this.state = msg.on ? "showcase" : "training"; this._status(); } break;
      case "set_target": this.target = [msg.x, msg.y]; break;
      case "save": setTimeout(() => this.emit({ type: "saved", path: `checkpoints/${this.env}_${this.iter}.npz` }), 200); break;
    }
  }

  _status() { this.emit({ type: "status", state: this.state, env: this.env }); }

  start(msg) {
    this.env = msg.env || this.env; this.N = msg.num_envs || this.N; this.R = Math.min(msg.render_envs || 64, 1024);
    this.iter = 0; this.steps = 0; this.t0 = performance.now(); this.sim = 0;
    for (let i = 0; i < this.R; i++) { this.envT[i] = rand(i) * 4; this.envX[i] = 0; }
    this.state = "training"; this._status();
    clearInterval(this.metricTimer);
    this.metricTimer = setInterval(() => this._metrics(), 280);
  }

  stop() { clearInterval(this.metricTimer); this.state = "idle"; this._status(); }

  progress() { return 1 - Math.exp(-this.iter / 90); }

  _metrics() {
    if (this.state === "idle") return;
    this.iter++;
    const sps = this.N * 449 * (0.96 + 0.08 * Math.random()) * (this.env === "cartpole" ? 6 : this.env === "arm" ? 2.2 : 1);
    this.steps += this.N * 32;
    const p = this.progress();
    const lo = MIN_RET[this.env], hi = MAX_RET[this.env];
    const noise = (Math.random() - 0.5) * Math.abs(hi - lo) * 0.06 * (1 - p * 0.6);
    this.emit({
      type: "metrics", iter: this.iter, env_steps: this.steps, sps,
      mean_reward: (lo + (hi - lo) * p) / 1000, mean_ep_return: lo + (hi - lo) * p + noise,
      mean_ep_len: 1000 * p, policy_loss: 0.01, value_loss: 0.1, entropy: 1 - p,
      time: (performance.now() - this.t0) / 1000, num_envs: this.N,
    });
  }

  _frameLoop() {
    setInterval(() => {
      const now = performance.now();
      const dt = Math.min(0.1, (now - this.lastFrame) / 1000);
      this.lastFrame = now;
      if (this.state === "idle" && this.iter === 0) { this._frame(dt, 0.02); return; }
      this._frame(this.state === "idle" ? 0 : dt, this.state === "showcase" ? Math.max(0.9, this.progress()) : this.progress());
    }, 33);
  }

  _frame(dt, p) {
    const R = this.R, env = this.env;
    this.sim += dt;
    const gen = { walker: walkerPose, cheetah: cheetahPose, arm: armPose, cartpole: cartPose }[env];
    const K = { walker: 7, cheetah: 7, arm: 3, cartpole: 2 }[env];
    const caps = new Array(R * K * 5);
    const cam = env === "walker" || env === "cheetah" ? new Array(R) : null;
    const targets = env === "arm" ? new Array(R * 2) : null;
    for (let i = 0; i < R; i++) {
      const pi = Math.min(1, Math.max(0, p * (0.75 + 0.5 * rand(i + 9))));
      this.envT[i] += dt;
      const out = gen(this, i, this.envT[i], pi, dt);
      for (let k = 0; k < K * 5; k++) caps[i * K * 5 + k] = Math.round(out.caps[k] * 1000) / 1000;
      if (cam) cam[i] = out.cam;
      if (targets) { targets[i * 2] = out.target[0]; targets[i * 2 + 1] = out.target[1]; }
    }
    const bounds = {
      walker: [-1.6, 1.6, -0.25, 1.75], cheetah: [-1.8, 1.8, -0.25, 1.35],
      arm: [-1.1, 1.1, -1.1, 1.1], cartpole: [-2.6, 2.6, -0.4, 1.4],
    }[env];
    this.emit({
      type: "frame", env, R, K, capsules: caps, targets, colors: COLORS[env],
      ground: env === "arm" ? null : 0, world_bounds: bounds, camera_x: cam,
    });
  }
}

function seg(out, x1, y1, x2, y2, r) { out.push(x1, y1, x2, y2, r); }

function leg(out, hx, hy, a1, a2, a3, L1, L2, L3, r) {
  const kx = hx + Math.sin(a1) * L1, ky = hy - Math.cos(a1) * L1;
  const fx = kx + Math.sin(a1 + a2) * L2, fy = ky - Math.cos(a1 + a2) * L2;
  return { kx, ky, fx, fy, toe: [fx + Math.cos(a1 + a2 + a3) * L3, fy + Math.sin(a1 + a2 + a3) * L3] };
}

function walkerPose(sv, i, t, p, dt) {
  const ep = 5 + 6 * p; // episode length seconds
  if (t > ep) { sv.envT[i] = 0; sv.envX[i] = 0; t = 0; }
  const speed = 1.6 * p * p;
  sv.envX[i] += speed * dt;
  const x = sv.envX[i];
  const ph = t * (4 + 3 * p) + rand(i) * 6.28;
  const chaos = 1 - p;
  const jitter = (k) => Math.sin(t * (7 + k) + i * (1.3 + k)) * chaos;
  // falling for untrained policies
  const fall = Math.min(1, Math.max(0, (t - 1.2 - 3 * p) / 1.4)) * chaos;
  const tilt = jitter(1) * 0.5 + fall * 1.3 * (rand(i) > 0.5 ? 1 : -1) + 0.12 * p;
  const hipY = 1.0 - 0.06 * Math.abs(Math.sin(ph)) * p - fall * 0.75;
  const out = [];
  const sx = x - Math.sin(tilt) * 0.0, shx = x + Math.sin(tilt) * 0.55, shy = hipY + Math.cos(tilt) * 0.55;
  seg(out, sx, hipY, shx, shy, 0.07);
  const amp = 0.25 + 0.4 * p;
  const legs = [];
  for (const side of [0, 1]) {
    const q = ph + side * Math.PI;
    const a1 = Math.sin(q) * amp + jitter(2 + side) * 0.7 + fall * 0.6;
    const a2 = -Math.max(0, Math.sin(q + 1.2)) * (0.3 + 0.7 * p) - jitter(4 + side) * 0.5 - 0.1;
    const a3 = -Math.PI / 2 + Math.PI / 2 + 0.1 * Math.sin(q);
    legs.push(leg([], x, hipY, a1, a2, a3, 0.45, 0.45, 0.18, 0.05));
  }
  // thighs (back leg first so front draws over)
  for (const l of legs) seg(out, x, hipY, l.kx, l.ky, 0.055);
  for (const l of legs) seg(out, l.kx, l.ky, l.fx, l.fy, 0.045);
  for (const l of legs) seg(out, l.fx, l.fy, l.toe[0] + 0.0, l.toe[1], 0.04);
  // keep above ground roughly
  let minY = 1e9; for (let k = 1; k < out.length; k += 5) minY = Math.min(minY, out[k] - out[k + 3] * 0, out[k + 2] - 0);
  for (let k = 0; k < out.length; k += 5) minY = Math.min(minY, out[k + 1] - out[k + 4], out[k + 3] - out[k + 4]);
  const lift = minY < 0 ? -minY : 0;
  for (let k = 0; k < out.length; k += 5) { out[k + 1] += lift; out[k + 3] += lift; }
  return { caps: out, cam: x };
}

function cheetahPose(sv, i, t, p, dt) {
  const ep = 6;
  if (t > ep) { sv.envT[i] = 0; sv.envX[i] = 0; t = 0; }
  const speed = 3.2 * p * p;
  sv.envX[i] += speed * dt;
  const x = sv.envX[i];
  const chaos = 1 - p;
  const ph = t * (6 + 6 * p) + rand(i) * 6.28;
  const flip = Math.min(1, Math.max(0, (t - 1.5) / 2)) * chaos * (rand(i + 3) > 0.6 ? 1 : 0);
  const pitch = Math.sin(ph) * 0.12 * p + Math.sin(t * 5 + i) * 0.3 * chaos + flip * 2.6;
  const cy = 0.62 + 0.05 * Math.sin(ph * 2) * p;
  const hl = 0.5;
  const bx = x - Math.cos(pitch) * hl, by = cy - Math.sin(pitch) * hl;
  const fx = x + Math.cos(pitch) * hl, fy = cy + Math.sin(pitch) * hl;
  const out = [];
  seg(out, bx, by, fx, fy, 0.065);
  const amp = 0.3 + 0.5 * p;
  const jit = (k) => Math.sin(t * (8 + k) + i * 2.1 + k) * chaos * 0.8;
  const back = leg([], bx, by, Math.sin(ph) * amp + jit(1), -0.6 - 0.4 * Math.sin(ph + 1) * p + jit(2), 0, 0.3, 0.28, 0.16, 0.045);
  const front = leg([], fx, fy, Math.sin(ph + 2.2) * amp + jit(3), 0.5 + 0.4 * Math.sin(ph + 3) * p + jit(4), 0, 0.27, 0.25, 0.13, 0.045);
  seg(out, bx, by, back.kx, back.ky, 0.05); seg(out, back.kx, back.ky, back.fx, back.fy, 0.042); seg(out, back.fx, back.fy, back.fx + 0.12, back.fy - 0.04, 0.035);
  seg(out, fx, fy, front.kx, front.ky, 0.05); seg(out, front.kx, front.ky, front.fx, front.fy, 0.042); seg(out, front.fx, front.fy, front.fx + 0.1, front.fy - 0.05, 0.035);
  let minY = 1e9;
  for (let k = 0; k < out.length; k += 5) minY = Math.min(minY, out[k + 1] - out[k + 4], out[k + 3] - out[k + 4]);
  const lift = -minY;
  for (let k = 0; k < out.length; k += 5) { out[k + 1] += lift; out[k + 3] += lift; }
  return { caps: out, cam: x };
}

function armPose(sv, i, t, p) {
  const [tx0, ty0] = sv.target;
  // per-env target jitter when untrained so the grid feels alive; trained arms all lock onto the shared target
  const tx = tx0, ty = ty0;
  const L1 = 0.5, L2 = 0.42;
  const d = Math.min(L1 + L2 - 0.01, Math.max(0.1, Math.hypot(tx, ty)));
  const c2 = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2);
  const q2 = Math.acos(Math.max(-1, Math.min(1, c2)));
  const q1 = Math.atan2(ty, tx) - Math.atan2(L2 * Math.sin(q2), L1 + L2 * Math.cos(q2));
  const chaos = 1 - p;
  const w1 = Math.sin(t * 1.7 + i * 1.9) * 2.4 * chaos + Math.sin(t * 9 + i) * 0.08 * chaos;
  const w2 = Math.sin(t * 2.3 + i * 2.7) * 2.0 * chaos;
  const k = sv._armSmooth || (sv._armSmooth = new Float32Array(2048));
  const goal1 = q1 * p + w1, goal2 = q2 * p + w2;
  k[i * 2] += (goal1 - k[i * 2]) * 0.18; k[i * 2 + 1] += (goal2 - k[i * 2 + 1]) * 0.18;
  const a1 = k[i * 2], a2 = k[i * 2 + 1];
  const ex = Math.cos(a1) * L1, ey = Math.sin(a1) * L1;
  const hx = ex + Math.cos(a1 + a2) * L2, hy = ey + Math.sin(a1 + a2) * L2;
  return { caps: [-0.12, 0, 0.12, 0, 0.09, 0, 0, ex, ey, 0.055, ex, ey, hx, hy, 0.045], target: [tx, ty] };
}

function cartPose(sv, i, t, p) {
  const ep = 2 + 10 * p;
  if (t > ep) { sv.envT[i] = 0; t = 0; }
  const chaos = 1 - p;
  const x = Math.sin(t * 0.9 + i) * (0.4 + 0.8 * chaos);
  const dir = rand(i + 1) > 0.5 ? 1 : -1;
  const fall = chaos * Math.min(1.45, 0.05 * Math.exp(t * 1.4)) * dir;
  const th = fall + Math.sin(t * 3 + i) * 0.05 * p;
  const cy = 0.12, L = 1.0;
  return { caps: [x - 0.22, cy, x + 0.22, cy, 0.11, x, cy, x + Math.sin(th) * L, cy + Math.cos(th) * L, 0.035] };
}

export const DEMO_BENCH = {
  machine: "Apple M3 Pro · 18 GB (demo data)",
  results: [
    { task: "cartpole", backend: "MLX GPU (N=16384)", sps: 222000000 },
    { task: "cartpole", backend: "MuJoCo CPU ×1", sps: 1150000 },
    { task: "cartpole", backend: "MuJoCo CPU ×11 proc", sps: 9800000 },
    { task: "walker", backend: "MLX GPU (N=16384)", sps: 18400000 },
    { task: "walker", backend: "MuJoCo CPU ×1", sps: 92000 },
    { task: "walker", backend: "MuJoCo CPU ×11 proc", sps: 760000 },
  ],
};
