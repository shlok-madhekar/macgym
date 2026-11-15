// MacGym demo mode (?demo=1): a fake server that speaks the same websocket protocol
// (frame / metrics / status / info / replay) with fake but plausible learning.
(function () {
  "use strict";
  const TIME_SCALE = 2;              // demo "training seconds" per real second
  const REPLAY_AT = [0, 10, 30, 60, 120, 240];
  const R = 256;

  const rand = mulberry(7);
  function mulberry(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-9)) * Math.cos(2 * Math.PI * rand());
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const label = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const skillAt = (ts) => 1 - Math.exp(-ts / 75);

  // ---------- poses ----------
  function rot(px, py, cx, cy, a) { const c = Math.cos(a), s = Math.sin(a); const dx = px - cx, dy = py - cy; return [cx + dx * c - dy * s, cy + dx * s + dy * c]; }
  function walkerPose(out, o, x, phase, gaitAmp, fall) {
    const hip = [x, 0.98 - 0.04 * Math.abs(Math.sin(phase)) * gaitAmp];
    const sh = [x + 0.06 * gaitAmp, hip[1] + 0.5];
    const segs = [[hip[0], hip[1], sh[0], sh[1], 0.055]];
    for (const side of [0, Math.PI]) {
      const ph = phase + side;
      const th = -0.55 * gaitAmp * Math.sin(ph) + 0.05;
      const kn = [hip[0] + 0.45 * Math.sin(th), hip[1] - 0.45 * Math.cos(th)];
      const knee = 0.9 * gaitAmp * Math.max(0, Math.sin(ph + 1.2)) + 0.05;
      const sa = th - knee;
      const an = [kn[0] + 0.43 * Math.sin(sa), kn[1] - 0.43 * Math.cos(sa)];
      segs.push([hip[0], hip[1], kn[0], kn[1], 0.042]);
      segs.push([kn[0], kn[1], an[0], an[1], 0.036]);
      segs.push([an[0] - 0.03, an[1], an[0] + 0.16, an[1], 0.028]);
    }
    // order: torso, thighL, shinL, footL, thighR, shinR, footR
    const pivot = [x + 0.05, 0];
    for (let k = 0; k < 7; k++) {
      let [x1, y1, x2, y2, r] = segs[k];
      if (fall) { [x1, y1] = rot(x1, y1, pivot[0], pivot[1], -fall); [x2, y2] = rot(x2, y2, pivot[0], pivot[1], -fall); }
      out[o + k * 5] = x1; out[o + k * 5 + 1] = Math.max(r, y1); out[o + k * 5 + 2] = x2; out[o + k * 5 + 3] = Math.max(r, y2); out[o + k * 5 + 4] = r;
    }
  }
  function cheetahPose(out, o, x, phase, gaitAmp, fall) {
    const y = 0.62 + 0.03 * Math.sin(phase * 2) * gaitAmp;
    const pitch = 0.08 * Math.sin(phase) * gaitAmp - (fall ? fall * 0.9 : 0);
    const b = rot(x - 0.5, y, x, y, pitch), f = rot(x + 0.5, y, x, y, pitch);
    const head = [f[0] + 0.18, f[1] + 0.16];
    const segs = [[b[0], b[1], f[0], f[1], 0.06], [f[0], f[1], head[0], head[1], 0.05]];
    for (const [hp, ph0] of [[b, 0], [f, Math.PI * 0.8]]) {
      const ph = phase + ph0;
      const th = 0.6 * gaitAmp * Math.sin(ph);
      const kn = [hp[0] + 0.3 * Math.sin(th), hp[1] - 0.3 * Math.cos(th)];
      const sa = th - 0.5 - 0.6 * gaitAmp * Math.max(0, Math.sin(ph + 1));
      const ft = [kn[0] + 0.3 * Math.sin(sa), kn[1] - 0.3 * Math.cos(sa)];
      segs.push([hp[0], hp[1], kn[0], kn[1], 0.045], [kn[0], kn[1], ft[0], ft[1], 0.038]);
    }
    segs.push([b[0], b[1], b[0] - 0.22, b[1] + 0.12, 0.03]);
    for (let k = 0; k < 7; k++) {
      const s = segs[k];
      out[o + k * 5] = s[0]; out[o + k * 5 + 1] = Math.max(s[4], s[1]); out[o + k * 5 + 2] = s[2]; out[o + k * 5 + 3] = Math.max(s[4], s[3]); out[o + k * 5 + 4] = s[4];
    }
  }
  const L = [0.42, 0.34, 0.24];
  function armPose(out, o, angles) {
    let x = 0, y = 0, a = 0;
    for (let k = 0; k < 3; k++) {
      a += angles[k];
      const nx = x + L[k] * Math.cos(a), ny = y + L[k] * Math.sin(a);
      out[o + k * 5] = x; out[o + k * 5 + 1] = y; out[o + k * 5 + 2] = nx; out[o + k * 5 + 3] = ny; out[o + k * 5 + 4] = [0.045, 0.036, 0.028][k];
      x = nx; y = ny;
    }
    return [x, y];
  }
  function armIK(tx, ty) {
    const d = Math.hypot(tx, ty), dir = Math.atan2(ty, tx);
    const wx = tx - L[2] * Math.cos(dir), wy = ty - L[2] * Math.sin(dir);
    const r = clamp(Math.hypot(wx, wy), 0.09, L[0] + L[1] - 1e-3);
    const c2 = clamp((r * r - L[0] * L[0] - L[1] * L[1]) / (2 * L[0] * L[1]), -1, 1);
    const a2 = Math.acos(c2);
    const a1 = Math.atan2(wy, wx) - Math.atan2(L[1] * Math.sin(a2), L[0] + L[1] * Math.cos(a2));
    return [a1, a2, dir - a1 - a2, d];
  }
  function cartPose(out, o, x, theta) {
    out[o] = x - 0.22; out[o + 1] = 0.08; out[o + 2] = x + 0.22; out[o + 3] = 0.08; out[o + 4] = 0.08;
    out[o + 5] = x; out[o + 6] = 0.08; out[o + 7] = x + Math.sin(theta) * 1.0; out[o + 8] = 0.08 + Math.cos(theta) * 1.0; out[o + 9] = 0.025;
  }

  // ---------- episode models ----------
  function newLocoEp(r, skill, env) {
    const top = env === "cheetah" ? 2.6 : 1.7;
    r.x = 0; r.t = 0; r.phase = rand() * 6.28;
    r.speed = Math.max(0, top * skill * (0.55 + 0.45 * rand()) + 0.08 * gauss());
    r.fallAt = clamp(0.5 + 0.6 * rand() + 18 * Math.pow(skill, 1.4) * Math.sqrt(rand()), 0.4, 20);
    r.gait = clamp(0.25 + 0.8 * skill + 0.15 * gauss(), 0.1, 1.1);
    r.fall = 0; r.done = false;
  }
  function newArmEp(r, skill) {
    const ang = rand() * Math.PI * 1.6 - 0.3, rad = 0.35 + 0.55 * rand();
    r.tx = rad * Math.cos(ang); r.ty = rad * Math.sin(ang);
    r.noise = [0, 1, 2].map(() => gauss() * (1.3 * (1 - skill) + 0.03));
    if (!r.ang) r.ang = [Math.PI / 2 + gauss() * 0.3, gauss() * 0.3, gauss() * 0.3];
    r.t = 0; r.dur = 3;
  }
  function newCartEp(r, skill) {
    r.x = gauss() * 0.1; r.v = 0; r.t = 0; r.th = gauss() * 0.05;
    r.fallAt = clamp(0.4 + 0.4 * rand() + 11 * Math.pow(skill, 1.6) * Math.sqrt(rand()), 0.3, 10);
    r.dir = rand() < 0.5 ? -1 : 1; r.wob = 0.25 * (1 - skill) + 0.02;
  }

  function Demo(onMessage) {
    this.emit = (m) => setTimeout(() => onMessage(JSON.parse(JSON.stringify(m))), 0);
    this.state = "idle"; this.env = "walker"; this.num = 4096;
    this.timer = null; this.robots = []; this.done = [];
    this.replays = []; this.nextReplay = 0;
  }
  Demo.prototype.send = function (m) {
    if (m.cmd === "list") this.emit({ type: "envs", envs: [
      { name: "walker", obs_dim: 17, act_dim: 6, description: "planar biped" },
      { name: "cheetah", obs_dim: 17, act_dim: 6, description: "planar quadruped" },
      { name: "arm", obs_dim: 10, act_dim: 3, description: "3-link reacher" },
      { name: "cartpole", obs_dim: 4, act_dim: 1, description: "balance a pole" }] });
    if (m.cmd === "status") this.emit(this.status());
    if (m.cmd === "start") this.start(m.env || "walker", m.num_envs || 4096, false);
    if (m.cmd === "load") this.start(m.env || this.env, m.num_envs || this.num, true);
    if (m.cmd === "stop") this.stop();
  };
  Demo.prototype.status = function () { return { type: "status", state: this.state, env: this.env, num_envs: this.num, render_envs: R, checkpoint: true }; };
  Demo.prototype.stop = function () {
    if (this.state !== "idle") this.pushReplay("now");
    clearInterval(this.timer); this.timer = null; this.state = "idle"; this.emit(this.status());
  };
  Demo.prototype.start = function (env, num, trained) {
    clearInterval(this.timer);
    this.env = env; this.num = num; this.state = "training";
    this.offset = trained ? 300 : 0;
    this.t0 = performance.now(); this.lastTick = this.t0; this.lastMetric = 0; this.iter = 0;
    this.done = []; this.best = 0; this.replays = []; this.nextReplay = 0;
    this.robots = [];
    for (let i = 0; i < R; i++) {
      const r = {};
      if (env === "arm") newArmEp(r, this.skill());
      else if (env === "cartpole") newCartEp(r, this.skill());
      else newLocoEp(r, this.skill(), env);
      this.robots.push(r);
    }
    this.emit(this.status());
    this.emit({ type: "info", sim_sps: { walker: 3.05e6, cheetah: 2.9e6, arm: 5.1e6, cartpole: 6.1e6 }[env] * num / 4096, env, num_envs: num });
    this.timer = setInterval(() => this.tick(), 1000 / 30);
    this.tick();
  };
  Demo.prototype.ts = function () { return this.offset + (performance.now() - this.t0) / 1000 * TIME_SCALE; };
  Demo.prototype.skill = function () { return skillAt(this.ts()); };

  Demo.prototype.tick = function () {
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.lastTick) / 1000); this.lastTick = now;
    const env = this.env, skill = this.skill();
    const K = env === "arm" ? 3 : env === "cartpole" ? 2 : 7;
    const caps = new Float32Array(R * K * 5);
    const dist = new Float32Array(R);
    const targets = env === "arm" ? new Float32Array(R * 2) : null;
    let bestI = 0, bestV = -Infinity;
    for (let i = 0; i < R; i++) {
      const r = this.robots[i];
      if (env === "arm") {
        r.t += dt;
        if (r.t > r.dur) { this.done.push(Math.hypot(r.ex - r.tx, r.ey - r.ty) * 100); newArmEp(r, skill); }
        const ik = armIK(r.tx, r.ty);
        for (let k = 0; k < 3; k++) {
          const goal = ik[k] + r.noise[k] + 0.04 * (1 - skill) * Math.sin(now / 300 + i + k);
          r.ang[k] += (goal - r.ang[k]) * Math.min(1, dt * 4);
        }
        const [ex, ey] = armPose(caps, i * K * 5, r.ang); r.ex = ex; r.ey = ey;
        targets[i * 2] = r.tx; targets[i * 2 + 1] = r.ty;
        const err = Math.hypot(ex - r.tx, ey - r.ty);
        dist[i] = err;
        if (-err > bestV) { bestV = -err; bestI = i; }
      } else if (env === "cartpole") {
        r.t += dt;
        if (r.t < r.fallAt) r.th = r.wob * Math.sin(r.t * 3 + i) * 0.5;
        else r.th += r.dir * (0.4 + Math.abs(r.th) * 6) * dt;
        r.x += Math.sin(r.t * 0.8 + i) * 0.3 * dt;
        if (Math.abs(r.th) > 1.45 || r.t >= 10) { this.done.push(Math.min(r.t, r.fallAt)); newCartEp(r, skill); }
        cartPose(caps, i * K * 5, r.x, r.th);
        dist[i] = r.t;
        const v = r.t - Math.abs(r.th) * 5;
        if (v > bestV) { bestV = v; bestI = i; }
      } else {
        r.t += dt;
        if (!r.done) {
          r.x += r.speed * dt; r.phase += dt * (3 + r.speed * 3.2);
          if (r.t >= r.fallAt) {
            if (r.fallAt >= 20) { this.done.push(r.x); newLocoEp(r, skill, env); }
            else r.done = true;
          }
        } else {
          r.fall = Math.min(1.45, r.fall + dt * (1 + r.fall * 4));
          r.x += r.speed * dt * 0.3;
          if (r.fall >= 1.45) { r.dead = (r.dead || 0) + dt; if (r.dead > 0.5) { r.dead = 0; this.done.push(r.x); newLocoEp(r, skill, env); } }
        }
        (env === "cheetah" ? cheetahPose : walkerPose)(caps, i * K * 5, r.x, r.phase, r.gait, r.fall);
        dist[i] = r.x;
        if (r.x > bestV) { bestV = r.x; bestI = i; }
      }
    }
    if (this.done.length > 400) this.done.splice(0, this.done.length - 400);
    const wb = env === "arm" ? [-1.1, 1.1, -1.1, 1.1] : env === "cartpole" ? [-2.4, 2.4, -0.3, 1.3] : [-1, 4, -0.2, 1.8];
    this.emit({ type: "frame", env, R, K, capsules: Array.from(caps, (v) => Math.round(v * 1000) / 1000),
      targets: targets ? Array.from(targets) : null, colors: null, ground: env === "arm" ? null : 0,
      world_bounds: wb, camera_x: null, dist: Array.from(dist), best_index: bestI });

    const ts = this.ts();
    if (now - this.lastMetric > 450) { this.lastMetric = now; this.metrics(ts); }
    if (this.nextReplay < REPLAY_AT.length && ts - this.offset >= REPLAY_AT[this.nextReplay]) {
      this.pushReplay(label(REPLAY_AT[this.nextReplay])); this.nextReplay++;
    }
  };

  Demo.prototype.metrics = function (ts) {
    this.iter++;
    const env = this.env, w = this.done.slice(-100);
    const mean = w.length ? w.reduce((a, b) => a + b, 0) / w.length : (env === "arm" ? 60 : 0);
    const best = w.length ? (env === "arm" ? Math.min(...w) : Math.max(...w)) : mean;
    const meta = env === "arm" ? { human_unit: "cm", human_label: "average miss distance", higher_is_better: false, err_mean: mean / 100, unit: "cm" }
      : env === "cartpole" ? { human_unit: "s", human_label: "average time balanced", higher_is_better: true, balance_mean: mean, unit: "s" }
      : { human_unit: "m", human_label: "average distance walked", higher_is_better: true, dist_mean: mean, dist_best: best, unit: "m", headline_label: "distance walked" };
    const sps = (env === "cartpole" ? 4.2e6 : env === "arm" ? 1.9e6 : 0.284e6 * 4) * this.num / 16384 * (0.95 + 0.1 * rand());
    this.emit(Object.assign({ type: "metrics", iter: this.iter, env_steps: Math.round(this.iter * this.num * 32), sps,
      mean_ep_return: mean * 10, num_envs: this.num, train_seconds: ts - this.offset, time: ts - this.offset,
      human_value: mean, human_best: best }, meta));
  };

  Demo.prototype.pushReplay = function (lab) {
    const env = this.env, ts = this.ts(), skill = skillAt(ts);
    const K = env === "arm" ? 3 : env === "cartpole" ? 2 : 7;
    const frames = [], tg = [];
    const dtr = 0.04; let result = 0, text = "";
    if (env === "arm") {
      const targetsSeq = [[0.55, 0.5], [-0.4, 0.62], [0.7, -0.1]];
      const ang = [Math.PI / 2, 0, 0], noise = [0, 1, 2].map((k) => (1.2 * (1 - skill) + 0.02) * [0.8, -0.6, 0.9][k]);
      let err = 0;
      for (let f = 0; f < 200; f++) {
        const [tx, ty] = targetsSeq[Math.floor(f / 67)];
        const ik = armIK(tx, ty);
        for (let k = 0; k < 3; k++) ang[k] += (ik[k] + noise[k] - ang[k]) * 0.12;
        const c = new Float32Array(K * 5); const [ex, ey] = armPose(c, 0, ang);
        frames.push(Array.from(c)); tg.push([tx, ty]); err = Math.hypot(ex - tx, ey - ty) * 100;
      }
      result = err; text = err < 3 ? `touches target · ${err.toFixed(1)} cm off` : `misses by ${err.toFixed(1)} cm`;
    } else if (env === "cartpole") {
      const fallAt = clamp(0.5 + 10 * Math.pow(skill, 1.6), 0.4, 8.5);
      let th = 0, x = 0, t = 0, end = 200;
      for (let f = 0; f < 200; f++) {
        t = f * dtr;
        if (t < fallAt) th = (0.25 * (1 - skill) + 0.02) * Math.sin(t * 3); else th += (0.4 + Math.abs(th) * 6) * dtr;
        x = 0.4 * Math.sin(t * 0.7);
        const c = new Float32Array(K * 5); cartPose(c, 0, x, th); frames.push(Array.from(c));
        if (Math.abs(th) > 1.45) { end = f; break; }
      }
      result = Math.min(fallAt, 8); text = fallAt >= 8 ? `balanced the full 8 s` : `pole fell after ${fallAt.toFixed(1)} s`;
    } else {
      const top = env === "cheetah" ? 2.6 : 1.7;
      const speed = top * Math.max(0, skill - 0.03) * 0.95;
      const fallAt = clamp(0.6 + 9 * Math.pow(skill, 1.5), 0.5, 9);
      const gait = clamp(0.25 + 0.8 * skill, 0.1, 1.1);
      let x = 0, phase = 0, fall = 0, t = 0;
      for (let f = 0; f < 200; f++) {
        t = f * dtr;
        if (t < fallAt) { x += speed * dtr; phase += dtr * (3 + speed * 3.2); }
        else { fall = Math.min(1.45, fall + dtr * (1 + fall * 4)); x += speed * dtr * 0.3; }
        const c = new Float32Array(K * 5); (env === "cheetah" ? cheetahPose : walkerPose)(c, 0, x, phase, gait, fall);
        frames.push(Array.from(c, (v) => Math.round(v * 1000) / 1000));
        if (fall >= 1.45 && t > fallAt + 0.6) break;
      }
      result = x;
      text = fallAt >= 8 ? `${env === "cheetah" ? "ran" : "walked"} ${x.toFixed(1)} m in 8 s` : `fell after ${fallAt.toFixed(1)} s · ${x.toFixed(1)} m`;
    }
    const msg = { type: "replay", env, label: lab, train_seconds: ts - this.offset, iter: this.iter, K, frames, targets: env === "arm" ? tg : null, result, result_text: text, dt: dtr };
    this.replays.push(msg); this.emit(msg);
  };

  window.MacGymDemo = { create: (onMessage) => new Demo(onMessage) };
})();
