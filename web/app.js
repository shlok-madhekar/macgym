// MacGym viewer v2. Talks to ws://<host>/ws (CONTRACT.md §3 + REDESIGN.md), or a fake server with ?demo=1.
(function () {
  "use strict";
  const C = { paper: "#F3F0E8", ink: "#121212", rule: "#D9D3C7", muted: "#6B665C", accent: "#FF4D00", old: "#8A94A6", faint: "#B4AD9F" };
  const SANS = '"Inter Tight", "Helvetica Neue", Arial, sans-serif';
  const MONO = '"JetBrains Mono", ui-monospace, Menlo, monospace';
  const REPLAY_AT = [0, 10, 30, 60, 120, 240];
  const LOCO = new Set(["walker", "cheetah"]);
  const META = {
    walker: { unit: "m", label: "average distance walked", hib: true },
    cheetah: { unit: "m", label: "average distance run", hib: true },
    arm: { unit: "cm", label: "average miss distance", hib: false },
    cartpole: { unit: "s", label: "average time balanced", hib: true },
  };
  const params = new URLSearchParams(location.search);
  const DEMO = params.get("demo") === "1";
  const $ = (id) => document.getElementById(id);
  const fmtInt = (n) => Math.round(n).toLocaleString("en-US");
  const fmtSI = (n) => n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 1 : 2) + "M" : n >= 1e3 ? Math.round(n / 1e3) + "K" : String(Math.round(n));
  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const S = {
    connected: false, env: "walker", numEnvs: 4096, state: "idle",
    frame: null, prev: null, frameT: 0, frameDt: 33,
    hist: [], humanValue: null, humanBest: null, unit: "m", label: META.walker.label, hib: true, usingReward: false,
    iter: 0, sps: 0, simSps: null, trainSeconds: 0, localStart: 0,
    replays: [], avgMarks: [], liveMean: 0, machine: "Apple M3 Pro GPU", record: 0,
    cam: { range: 6, targetRange: 6 },
  };

  // ---------------- transport ----------------
  let ws = null, demo = null, retry = 500;
  function send(o) {
    if (demo) return demo.send(o);
    if (ws && ws.readyState === 1) ws.send(JSON.stringify(o));
  }
  function connect() {
    if (DEMO) { demo = window.MacGymDemo.create(onMessage); S.connected = true; onOpen(); return; }
    const url = params.get("ws") || `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
    try { ws = new WebSocket(url); } catch (e) { return later(); }
    ws.onopen = () => { S.connected = true; retry = 500; onOpen(); renderFoot(); };
    ws.onmessage = (ev) => { try { onMessage(JSON.parse(ev.data)); } catch (e) { console.warn("bad message", e); } };
    ws.onclose = () => { S.connected = false; renderFoot(); later(); };
    ws.onerror = () => {};
  }
  function later() { setTimeout(connect, retry); retry = Math.min(5000, retry * 1.6); }
  function onOpen() { send({ cmd: "list" }); send({ cmd: "status" }); }

  fetch("bench.json").then((r) => r.ok ? r.json() : null).then((j) => {
    const m = j && (j.machine || "");
    const chip = /Apple M\d+( Pro| Max| Ultra)?/.exec(m);
    if (chip) S.machine = chip[0] + " GPU";
    renderFoot();
  }).catch(() => {});

  function onMessage(m) {
    switch (m.type) {
      case "frame": onFrame(m); break;
      case "metrics": onMetrics(m); break;
      case "status": onStatus(m); break;
      case "replay": onReplay(m); break;
      case "info": if ("sim_sps" in m) { S.simSps = m.sim_sps; renderFoot(); } break;
      case "envs": break;
      case "error": console.warn("server:", m.message); break;
    }
  }

  // ---------------- messages ----------------
  function onStatus(m) {
    const prevState = S.state;
    S.state = m.state || "idle";
    if (m.num_envs) { S.numEnvs = m.num_envs; $("count").value = String(m.num_envs); }
    if (m.env && m.env !== S.env) setEnv(m.env, false);
    if (S.state !== "idle" && prevState === "idle" && !S.localStart) S.localStart = performance.now();
    if (S.state === "idle") S.sps = 0;
    renderControls(); renderFoot(); renderReady();
  }

  function onFrame(m) {
    if (m.env && m.env !== S.env) setEnv(m.env, false);
    const R = m.R | 0, K = m.K | 0;
    const caps = Float32Array.from(m.capsules || []);
    let dist = m.dist ? Float32Array.from(m.dist) : null;
    let shift = null;
    if (LOCO.has(S.env) && !dist) {
      // Server without start-relative frames: fall back to camera_x (torso x) or torso capsule x.
      dist = new Float32Array(R);
      for (let i = 0; i < R; i++) dist[i] = m.camera_x ? m.camera_x[i] : caps[i * K * 5];
    }
    const f = { R, K, caps, dist, targets: m.targets ? Float32Array.from(m.targets) : null, best: m.best_index, ground: m.ground, wb: m.world_bounds, shift };
    if (f.best == null || f.best < 0 || f.best >= R) f.best = computeBest(f);
    const now = performance.now();
    if (S.frame && S.frame.R === R && S.frame.K === K) {
      S.prev = S.frame;
      S.frameDt = clamp(S.frameDt * 0.85 + (now - S.frameT) * 0.15, 16, 150);
    } else S.prev = null;
    S.frame = f; S.frameT = now;
    if (S.state === "idle" && !DEMO) { /* frames after stop are fine to show */ }
    renderReady();
  }

  function computeBest(f) {
    const { R, K, caps } = f;
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < R; i++) {
      let v;
      if (S.env === "arm") {
        const o = (i * K + K - 1) * 5;
        v = f.targets ? -Math.hypot(caps[o + 2] - f.targets[i * 2], caps[o + 3] - f.targets[i * 2 + 1]) : caps[o + 3];
      } else if (S.env === "cartpole") {
        const o = (i * K + K - 1) * 5;
        v = caps[o + 3] - caps[o + 1];
      } else v = f.dist ? f.dist[i] : caps[i * K * 5];
      if (v > bv) { bv = v; bi = i; }
    }
    return bi;
  }

  function onMetrics(m) {
    if (m.iter != null) S.iter = m.iter;
    if (m.sps != null) S.sps = m.sps;
    if (m.num_envs) S.numEnvs = m.num_envs;
    let ts = m.train_seconds;
    if (ts == null) ts = S.localStart ? (performance.now() - S.localStart) / 1000 : (typeof m.time === "number" ? m.time : 0);
    S.trainSeconds = ts;
    const meta = META[S.env] || META.walker;
    let v = m.human_value, best = m.human_best, unit = m.human_unit, label = m.human_label, hib = m.higher_is_better;
    if (v == null) {
      if (m.dist_mean != null) { v = m.dist_mean; best = m.dist_best; unit = "m"; }
      else if (m.err_mean != null) { v = m.err_mean * 100; unit = "cm"; hib = false; }
      else if (m.balance_mean != null) { v = m.balance_mean; unit = "s"; }
    }
    S.usingReward = false;
    if (v == null && m.mean_ep_return != null && isFinite(m.mean_ep_return)) { v = m.mean_ep_return; unit = ""; label = "average episode reward"; hib = true; S.usingReward = true; }
    if (v == null || !isFinite(v)) { renderFoot(); return; }
    S.humanValue = v; S.humanBest = best;
    S.unit = unit != null ? unit : meta.unit; S.label = label || meta.label; S.hib = hib != null ? hib : meta.hib;
    const last = S.hist[S.hist.length - 1];
    if (last && ts < last[0] - 1) S.hist = [];
    S.hist.push([ts, v]);
    if (S.hist.length > 2000) S.hist = S.hist.filter((_, i) => i % 2 === 0);
    renderHeadline(); renderFoot();
  }

  function onReplay(m) {
    if (m.env && m.env !== S.env) return;
    const idx = S.replays.findIndex((r) => r.label === m.label);
    const rep = { ...m, born: performance.now() };
    if (m.label === "now") S.replays = S.replays.filter((r) => r.label !== "now");
    if (idx >= 0) S.replays[idx] = rep; else S.replays.push(rep);
    if (LOCO.has(S.env) && m.label !== "now" && idx < 0 && S.frame) S.avgMarks.push({ label: m.label, x: S.liveMean || 0 });
    S.replays.sort((a, b) => (a.train_seconds ?? 0) - (b.train_seconds ?? 0));
    buildGens();
  }

  // ---------------- controls ----------------
  for (const b of document.querySelectorAll("#tabs button")) {
    b.addEventListener("click", () => {
      const env = b.dataset.env;
      if (env === S.env) return;
      setEnv(env, true);
      if (S.state !== "idle") startRun();
    });
  }
  $("count").addEventListener("change", (e) => { S.numEnvs = +e.target.value; renderReady(); renderFoot(); });
  $("train").addEventListener("click", () => { if (S.state === "idle") startRun(); else send({ cmd: "stop" }); });
  $("load").addEventListener("click", () => { resetRun(); send({ cmd: "load", env: S.env, num_envs: S.numEnvs }); send({ cmd: "render", render_envs: 256 }); });

  function resetRun() {
    S.hist = []; S.replays = []; S.avgMarks = []; S.humanValue = null; S.humanBest = null; S.record = 0; S.iter = 0; S.sps = 0;
    S.frame = null; S.prev = null; S.localStart = performance.now(); S.trainSeconds = 0;
    S.cam.range = S.cam.targetRange = 6;
    buildGens(); renderHeadline();
  }
  function startRun() {
    resetRun();
    send({ cmd: "start", env: S.env, num_envs: S.numEnvs, render_envs: 256 });
  }
  function setEnv(env, user) {
    if (!META[env]) return;
    const changed = env !== S.env;
    S.env = env;
    for (const b of document.querySelectorAll("#tabs button")) b.classList.toggle("on", b.dataset.env === env);
    if (changed) {
      S.frame = null; S.prev = null; S.hist = []; S.replays = []; S.avgMarks = []; S.humanValue = null; S.record = 0;
      S.unit = META[env].unit; S.label = META[env].label; S.hib = META[env].hib;
      S.cam.range = S.cam.targetRange = 6;
      buildGens(); renderHeadline();
    }
    renderReady();
  }

  function renderControls() {
    const running = S.state !== "idle";
    const btn = $("train");
    btn.textContent = running ? "Stop" : "Train";
    btn.classList.toggle("stop", running);
  }
  function renderReady() {
    const show = S.state === "idle" && !S.frame;
    $("ready").classList.toggle("hide", !show);
    $("readyMain").textContent = S.state === "measuring" ? `Measuring raw simulation speed…` : `${fmtInt(S.numEnvs)} robots ready. Press Train.`;
    $("readySub").textContent = {
      walker: "Every robot starts at 0 m. As they learn, the crowd moves right.",
      cheetah: "Every robot starts at 0 m. As they learn, the crowd moves right.",
      arm: "Every arm shares one base. As they learn, the tips close in on their targets.",
      cartpole: "Every cart starts upright. As they learn, the poles stay up longer.",
    }[S.env];
    const R = S.frame ? S.frame.R : 256;
    $("heroLabel").textContent = "Track";
    $("heroNote").textContent = S.frame
      ? `${fmtInt(R)} of ${fmtInt(S.numEnvs)} robots shown, overlaid${LOCO.has(S.env) ? " · each from its own start" : ""}`
      : "";
  }
  function renderHeadline() {
    const v = S.humanValue;
    const num = $("hlNum");
    if (v == null) { num.textContent = "–"; $("hlUnit").textContent = S.usingReward ? "" : S.unit; }
    else {
      const d = S.unit === "cm" ? (v < 10 ? 1 : 0) : S.usingReward ? 0 : 1;
      num.textContent = v.toFixed(d);
      $("hlUnit").textContent = S.unit ? (S.unit === "s" || S.unit === "cm" ? " " + S.unit : S.unit) : "";
    }
    $("hlLabel").textContent = S.label;
    let sub = "no run yet";
    if (S.hist.length > 1) {
      const first = S.hist[0][1];
                  const fmt = (x) => S.unit === "cm" ? x.toFixed(1) + " cm" : S.unit ? x.toFixed(1) + " " + S.unit : x.toFixed(0);
      sub = `was ${fmt(first)} at ${mmss(S.hist[0][0])}${S.humanBest != null && S.unit === "m" ? ` · best ${fmt(S.humanBest)}` : ""}`;
    } else if (S.state !== "idle") sub = "first episodes finishing…";
    $("hlSub").textContent = sub;
  }
  function renderFoot() {
    const parts = [];
    parts.push(`<b>${fmtInt(S.numEnvs)}</b> robots in parallel`);
    parts.push(`${S.machine} (MLX)`);
    parts.push(`sim ${S.simSps ? `<b>${fmtSI(S.simSps)}</b> steps/s` : "–"}`);
    parts.push(`training ${S.sps ? `<b>${fmtSI(S.sps)}</b> steps/s` : "–"}`);
    parts.push(`iteration ${S.iter ? `<b>${fmtInt(S.iter)}</b>` : "–"}`);
    if (DEMO) parts.push("demo data");
    else if (!S.connected) parts.push("server offline, reconnecting");
    $("foot").innerHTML = parts.map((x) => `<span>${x}</span>`).join('<span class="sep">·</span>');
  }

  // ---------------- canvases ----------------
  function fit(cv) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }
  const trackCv = $("track"), chartCv = $("chart");

  // Interpolated capsule buffer for the current render time.
  let lerpBuf = new Float32Array(0), lerpDist = new Float32Array(0);
  function interpolated(now) {
    const f = S.frame; if (!f) return null;
    const n = f.caps.length;
    if (lerpBuf.length !== n) lerpBuf = new Float32Array(n);
    if (lerpDist.length !== f.R) lerpDist = new Float32Array(f.R);
    const p = S.prev;
    const t = clamp((now - S.frameT) / S.frameDt, 0, 1);
    if (!p) { lerpBuf.set(f.caps); if (f.dist) lerpDist.set(f.dist); return f; }
    const per = f.K * 5;
    for (let i = 0; i < f.R; i++) {
      const o = i * per;
      const jump = Math.abs(f.caps[o] - p.caps[o]) > 0.6 || Math.abs(f.caps[o + 1] - p.caps[o + 1]) > 0.6;
      for (let j = 0; j < per; j++) lerpBuf[o + j] = jump ? f.caps[o + j] : p.caps[o + j] + (f.caps[o + j] - p.caps[o + j]) * t;
      if (f.dist) lerpDist[i] = jump || !p.dist ? f.dist[i] : p.dist[i] + (f.dist[i] - p.dist[i]) * t;
    }
    return f;
  }

  function strokeRobots(ctx, f, buf, X, Y, s, alpha, color, skip) {
    const per = f.K * 5;
    ctx.lineCap = "round"; ctx.strokeStyle = color; ctx.globalAlpha = alpha;
    for (let i = 0; i < f.R; i++) {
      if (i === skip) continue;
      const o = i * per;
      for (let k = 0; k < f.K; k++) {
        const q = o + k * 5;
        ctx.lineWidth = clamp(buf[q + 4] * 2 * s * 0.45, 1, 3.2);
        ctx.beginPath(); ctx.moveTo(X(buf[q]), Y(buf[q + 1])); ctx.lineTo(X(buf[q + 2]), Y(buf[q + 3])); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }
  function strokeOne(ctx, buf, o, K, X, Y, s, color, joints) {
    ctx.lineCap = "round"; ctx.strokeStyle = color;
    for (let k = 0; k < K; k++) {
      const q = o + k * 5;
      ctx.lineWidth = clamp(buf[q + 4] * 2 * s * 0.6, 1.6, 4.5);
      ctx.beginPath(); ctx.moveTo(X(buf[q]), Y(buf[q + 1])); ctx.lineTo(X(buf[q + 2]), Y(buf[q + 3])); ctx.stroke();
    }
    if (joints) {
      for (let k = 0; k < K; k++) {
        const q = o + k * 5;
        for (const [x, y] of [[buf[q], buf[q + 1]], [buf[q + 2], buf[q + 3]]]) {
          ctx.fillStyle = C.paper; ctx.beginPath(); ctx.arc(X(x), Y(y), 2.6, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(X(x), Y(y), 1.5, 0, Math.PI * 2); ctx.fill();
        }
      }
    }
  }
  function mono(ctx, size, color) { ctx.font = `500 ${size}px ${MONO}`; ctx.fillStyle = color; }
  function tracked(ctx, text, x, y, align, spacing = 0.8) {
    ctx.letterSpacing = spacing + "px";
    ctx.textAlign = align || "left"; ctx.fillText(text, x, y); ctx.letterSpacing = "0px";
  }
  function dashed(ctx, x, y1, y2, color, dash) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash(dash || [4, 4]);
    ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, y1); ctx.lineTo(Math.round(x) + 0.5, y2); ctx.stroke(); ctx.restore();
  }
  function label(ctx, text, x, y, color, align, weight, size) {
    ctx.font = `${weight || 600} ${size || 14}px ${SANS}`; ctx.fillStyle = color; ctx.textAlign = align || "left"; ctx.fillText(text, x, y);
  }

  function drawTrack(now, dt) {
    const { ctx, w, h } = fit(trackCv);
    ctx.clearRect(0, 0, w, h);
    const pad = 32;
    const f = interpolated(now);
    if (LOCO.has(S.env)) drawLoco(ctx, w, h, pad, f, dt);
    else if (S.env === "arm") drawArm(ctx, w, h, pad, f);
    else drawCart(ctx, w, h, pad, f);
  }

  function drawLoco(ctx, w, h, pad, f, dt) {
    const groundY = Math.round(h - 64) + 0.5;
    // distances
    let mean = 0, max = 0, bi = f ? f.best : -1;
    if (f && f.dist) {
      for (let i = 0; i < f.R; i++) { mean += lerpDist[i]; if (lerpDist[i] > max) max = lerpDist[i]; }
      mean /= f.R;
      if (bi >= 0) max = Math.max(max, lerpDist[bi]);
      S.record = Math.max(S.record, max);
    }
    // camera: 0 m at left; furthest robot + 15% fits. Grows quickly, shrinks slowly.
    S.cam.targetRange = Math.max(8, max * 1.15 + 1);
    const k = S.cam.targetRange > S.cam.range ? 2.2 : 0.35;
    S.cam.range += (S.cam.targetRange - S.cam.range) * (1 - Math.exp(-k * dt));
    const usableW = w - pad * 2;
    const topSpace = 92; // room for marker labels
    const robotH = S.env === "cheetah" ? 1.1 : 1.65;
    const s = Math.min(usableW / (S.cam.range + 0.7), (groundY - topSpace) / robotH * 0.62);
    const x0 = pad + 0.55 * s;
    const X = (x) => x0 + x * s, Y = (y) => groundY - y * s;
    const visibleM = (w - pad - x0) / s;

    // ground + ruler
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, groundY); ctx.lineTo(w - pad, groundY); ctx.stroke();
    const labelEvery = s > 60 ? 1 : s > 22 ? 5 : s > 9 ? 10 : 20;
    const minorEvery = s > 6 ? 1 : 5;
    for (let m = 0; m <= visibleM + 1e-6; m += minorEvery) {
      const x = Math.round(X(m)) + 0.5;
      const major = m % labelEvery === 0;
      ctx.strokeStyle = major ? C.ink : C.muted;
      ctx.beginPath(); ctx.moveTo(x, groundY); ctx.lineTo(x, groundY + (major ? 9 : 4)); ctx.stroke();
      if (major) { mono(ctx, 11, m === 0 ? C.ink : C.muted); tracked(ctx, `${m} m`, x, groundY + 25, "center", 0.4); }
    }
    if (!S.avgMarks.length) { mono(ctx, 10.5, C.muted); tracked(ctx, "START", X(0), groundY + 43, "center"); }

    if (!f) return;
    // Figures keep a legible size even when the runway is 100 m long: scale each body about its own torso.
    const sb = Math.max(s, (groundY - topSpace) / robotH * 0.42);
    if (sb > s * 1.01) {
      const g = sb / s, per = f.K * 5;
      for (let i = 0; i < f.R; i++) {
        const o = i * per, ax = lerpBuf[o];
        for (let q = o; q < o + per; q += 5) {
          lerpBuf[q] = ax + (lerpBuf[q] - ax) * g; lerpBuf[q + 1] *= g;
          lerpBuf[q + 2] = ax + (lerpBuf[q + 2] - ax) * g; lerpBuf[q + 3] *= g; lerpBuf[q + 4] *= g;
        }
      }
    }
    // crowd
    strokeRobots(ctx, f, lerpBuf, X, Y, s, S.frame.R > 128 ? 0.13 : 0.22, C.ink, bi);

    // markers
    const mTop = Math.max(58, groundY - robotH * Math.max(s, (groundY - topSpace) / robotH * 0.42) - 64);
    // where the crowd's average stood at each generation moment, along the ruler
    for (const mk of S.avgMarks) {
      const x = Math.round(X(mk.x)) + 0.5;
      ctx.strokeStyle = C.old; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, groundY + 34); ctx.lineTo(x, groundY + 46); ctx.stroke();
    }
    S.avgMarks.forEach((mk, i) => {
      const next = S.avgMarks[i + 1];
      if (next && X(next.x) - X(mk.x) < 34) return;
      mono(ctx, 9.5, C.old); tracked(ctx, mk.label, X(mk.x) + 4, groundY + 46, "left", 0.2);
    });
    S.liveMean = mean;
    const bestNow = f.dist && bi >= 0 ? lerpDist[bi] : max;
    if (S.record > 0.3 && S.record - bestNow > 0.4) {
      const xr = X(S.record);
      dashed(ctx, xr, mTop + 30, groundY, C.old, [1, 3]);
      mono(ctx, 10.5, C.old); tracked(ctx, `RECORD ${S.record.toFixed(1)} M`, xr + 6, mTop + 40, "left", 0.6);
    }
    if (f.dist) {
      const xa = X(mean), xb = X(lerpDist[bi] ?? max);
      dashed(ctx, xa, mTop + 8, groundY, C.ink, [4, 4]);
      dashed(ctx, xb, mTop - 14, groundY, C.accent, []);
      const avgTxt = `avg ${mean.toFixed(1)} m`, bestTxt = `best ${(lerpDist[bi] ?? max).toFixed(1)} m`;
      // avoid overlap: avg label left of its line if lines are close
      ctx.font = `600 14px ${SANS}`;
      const close = Math.abs(xb - xa) < ctx.measureText(avgTxt).width + 20;
      label(ctx, avgTxt, close ? xa - 6 : xa + 6, mTop + 18, C.ink, close ? "right" : "left");
      label(ctx, bestTxt, xb + 6, mTop - 4, C.accent, "left");
    }
    // best robot on top
    if (bi >= 0) strokeOne(ctx, lerpBuf, bi * f.K * 5, f.K, X, Y, s, C.accent, true);
  }

  function drawArm(ctx, w, h, pad, f) {
    const wb = (f && f.wb) || [-1.1, 1.1, -1.1, 1.1];
    const cx = w / 2, cy = h / 2 + 10;
    const span = Math.max(wb[1] - wb[0], wb[3] - wb[2]);
    const s = (h - 110) / span;
    const X = (x) => cx + x * s, Y = (y) => cy - y * s;
    const reach = Math.max(Math.abs(wb[1]), Math.abs(wb[3])) * 0.9;
    // reference rings
    ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
    for (const r of [reach / 3, reach * 2 / 3, reach]) { ctx.beginPath(); ctx.arc(cx, cy, r * s, 0, Math.PI * 2); ctx.stroke(); }
    ctx.beginPath(); ctx.moveTo(cx - reach * s - 12, cy + 0.5); ctx.lineTo(cx + reach * s + 12, cy + 0.5); ctx.moveTo(cx + 0.5, cy - reach * s - 12); ctx.lineTo(cx + 0.5, cy + reach * s + 12); ctx.stroke();
    mono(ctx, 10.5, C.muted); tracked(ctx, `${Math.round(reach * 100)} CM`, cx + reach * s + 6, cy - 6, "left");
    if (!f) return;
    let errSum = 0;
    const per = f.K * 5;
    if (f.targets) {
      ctx.strokeStyle = C.ink; ctx.globalAlpha = 0.25; ctx.lineWidth = 1;
      for (let i = 0; i < f.R; i++) { ctx.beginPath(); ctx.arc(X(f.targets[i * 2]), Y(f.targets[i * 2 + 1]), 3, 0, Math.PI * 2); ctx.stroke(); }
      ctx.globalAlpha = 1;
      for (let i = 0; i < f.R; i++) {
        const o = i * per + (f.K - 1) * 5;
        errSum += Math.hypot(lerpBuf[o + 2] - f.targets[i * 2], lerpBuf[o + 3] - f.targets[i * 2 + 1]);
      }
    }
    strokeRobots(ctx, f, lerpBuf, X, Y, s, 0.12, C.ink, f.best);
    strokeOne(ctx, lerpBuf, f.best * per, f.K, X, Y, s, C.accent, true);
    if (f.targets) {
      const tx = X(f.targets[f.best * 2]), ty = Y(f.targets[f.best * 2 + 1]);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(tx, ty, 7, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx - 12, ty); ctx.lineTo(tx - 4, ty); ctx.moveTo(tx + 4, ty); ctx.lineTo(tx + 12, ty); ctx.moveTo(tx, ty - 12); ctx.lineTo(tx, ty - 4); ctx.moveTo(tx, ty + 4); ctx.lineTo(tx, ty + 12); ctx.stroke();
    }
    // base
    ctx.fillStyle = C.ink; ctx.fillRect(cx - 14, cy - 2, 28, 4);
    const avg = S.humanValue != null && S.unit === "cm" ? S.humanValue : f.targets ? errSum / f.R * 100 : null;
    if (avg != null) {
      label(ctx, `avg miss ${avg.toFixed(1)} cm`, pad, 70, C.ink);
      mono(ctx, 10.5, C.muted); tracked(ctx, "TIP TO TARGET, LAST 20 STEPS", pad, 90, "left");
    }
    label(ctx, "best arm", w - pad, 70, C.accent, "right");
    mono(ctx, 10.5, C.muted); tracked(ctx, "◯ TARGETS", w - pad, 90, "right");
  }

  function drawCart(ctx, w, h, pad, f) {
    const wb = (f && f.wb) || [-2.4, 2.4, -0.3, 1.3];
    const groundY = Math.round(h - 64) + 0.5;
    const s = Math.min((w - pad * 2) / (wb[1] - wb[0]), (groundY - 110) / 1.25);
    const cx = w / 2;
    const X = (x) => cx + x * s, Y = (y) => groundY - y * s;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(wb[0]), groundY); ctx.lineTo(X(wb[1]), groundY); ctx.stroke();
    for (let m = Math.ceil(wb[0]); m <= wb[1]; m++) {
      const x = Math.round(X(m)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, groundY); ctx.lineTo(x, groundY + (m === 0 ? 9 : 5)); ctx.stroke();
      mono(ctx, 11, m === 0 ? C.ink : C.muted); tracked(ctx, `${m} m`, x, groundY + 25, "center", 0.4);
    }
    // upright reference
    dashed(ctx, X(0), Y(1.15), groundY, C.rule, [2, 4]);
    for (const xs of [wb[0], wb[1]]) { ctx.fillStyle = C.ink; ctx.fillRect(Math.round(X(xs)) - 1, groundY - 10, 2, 10); }
    if (!f) return;
    strokeRobots(ctx, f, lerpBuf, X, Y, s, 0.13, C.ink, f.best);
    strokeOne(ctx, lerpBuf, f.best * f.K * 5, f.K, X, Y, s, C.accent, true);
    let up = 0;
    const per = f.K * 5;
    for (let i = 0; i < f.R; i++) { const o = i * per + (f.K - 1) * 5; if (lerpBuf[o + 3] - lerpBuf[o + 1] > 0.85 * Math.hypot(lerpBuf[o + 2] - lerpBuf[o], lerpBuf[o + 3] - lerpBuf[o + 1])) up++; }
    if (S.humanValue != null && S.unit === "s") label(ctx, `avg balanced ${S.humanValue.toFixed(1)} s`, pad, 70, C.ink);
    mono(ctx, 10.5, C.muted); tracked(ctx, `${up} OF ${f.R} POLES UP RIGHT NOW`, pad, 90, "left");
  }

  // ---------------- progress chart ----------------
  function niceStep(range, target) {
    const raw = range / target, p = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / p;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * p;
  }
  function drawChart(now) {
    const { ctx, w, h } = fit(chartCv);
    ctx.clearRect(0, 0, w, h);
    const L = 44, R = 18, T = 14, B = 26;
    const pw = w - L - R, ph = h - T - B;
    const hist = S.hist;
    const tMax = Math.max(60, hist.length ? hist[hist.length - 1][0] * 1.08 : 60);
    let vMin = 0, vMax = 1;
    for (const [, v] of hist) { vMax = Math.max(vMax, v); vMin = Math.min(vMin, v); }
    vMax *= 1.12;
    const xs = (t) => L + (t / tMax) * pw, ys = (v) => T + ph - ((v - vMin) / (vMax - vMin)) * ph;
    // grid
    const vStep = niceStep(vMax - vMin, 4);
    ctx.lineWidth = 1;
    mono(ctx, 10.5, C.muted);
    for (let v = Math.ceil(vMin / vStep) * vStep; v <= vMax + 1e-9; v += vStep) {
      const y = Math.round(ys(v)) + 0.5;
      ctx.strokeStyle = v === 0 ? C.ink : C.rule;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
      const txt = (vStep < 1 ? v.toFixed(1) : String(Math.round(v))) + (S.unit && v !== 0 ? "" : "");
      mono(ctx, 10.5, C.muted); tracked(ctx, txt, L - 8, y + 3.5, "right", 0.2);
    }
    mono(ctx, 10.5, C.muted); tracked(ctx, (S.unit || "").toUpperCase(), L - 8, T - 2, "right");
    const tStep = [10, 15, 30, 60, 120, 300, 600].find((st) => tMax / st <= 7) || 1200;
    for (let t = 0; t <= tMax; t += tStep) {
      const x = Math.round(xs(t)) + 0.5;
      ctx.strokeStyle = C.ink;
      ctx.beginPath(); ctx.moveTo(x, T + ph); ctx.lineTo(x, T + ph + 4); ctx.stroke();
      mono(ctx, 10.5, C.muted); tracked(ctx, mmss(t), x, T + ph + 18, "center", 0.2);
    }
    ctx.strokeStyle = C.ink; ctx.beginPath(); ctx.moveTo(L + 0.5, T); ctx.lineTo(L + 0.5, T + ph + 0.5); ctx.lineTo(L + pw, T + ph + 0.5); ctx.stroke();
    // replay moments as faint vertical ticks
    for (const r of S.replays) {
      if (r.train_seconds == null || r.label === "now") continue;
      const x = Math.round(xs(r.train_seconds)) + 0.5;
      ctx.strokeStyle = C.rule; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + ph); ctx.stroke(); ctx.setLineDash([]);
    }
    if (hist.length < 1) {
      mono(ctx, 11, C.faint); tracked(ctx, S.state === "idle" ? "PROGRESS APPEARS HERE ONCE TRAINING STARTS" : "WAITING FOR FIRST FINISHED EPISODES", L + pw / 2, T + ph / 2, "center");
      return;
    }
    // light smoothing for a calm line
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.75; ctx.lineJoin = "round";
    ctx.beginPath();
    let sm = hist[0][1];
    const pts = [];
    for (let i = 0; i < hist.length; i++) { sm = i ? sm * 0.6 + hist[i][1] * 0.4 : sm; pts.push([xs(hist[i][0]), ys(sm)]); }
    pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.stroke();
    const [lx, ly] = pts[pts.length - 1];
    ctx.strokeStyle = C.accent; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(L, Math.round(ly) + 0.5); ctx.lineTo(lx, Math.round(ly) + 0.5); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(lx, ly, 4.5, 0, Math.PI * 2); ctx.fill();
  }

  // ---------------- generations ----------------
  let genCells = [];
  function pickReplays() {
    const r = S.replays;
    if (r.length <= 5) return r;
    return [r[0], ...r.slice(r.length - 4)];
  }
  function buildGens() {
    const row = $("gens"); row.innerHTML = ""; genCells = [];
    const shown = pickReplays();
    const lastT = shown.length ? (shown[shown.length - 1].train_seconds ?? 0) : -1;
    const upcoming = REPLAY_AT.filter((t) => t > lastT + 0.5);
    for (let i = 0; i < 5; i++) {
      const rep = shown[i];
      const cell = document.createElement("div");
      cell.className = "gen";
      if (rep) {
        if (i === shown.length - 1) cell.classList.add("newest");
        const resTxt = rep.result != null ? fmtResult(rep.result) : "";
        cell.innerHTML = `<div class="gen-top"><span class="t">${rep.label === "now" ? "NOW" : rep.label}</span><span class="r">${resTxt}</span></div><div class="gen-box"><canvas></canvas></div><div class="gen-cap"></div>`;
        cell.querySelector(".gen-cap").textContent = rep.result_text || "";
        genCells.push({ rep, cv: cell.querySelector("canvas"), start: performance.now() + i * 180 });
      } else {
        const t = upcoming[i - shown.length];
        const lab = t != null ? `after ${mmss(t)}` : "";
        cell.classList.add("empty");
        cell.innerHTML = `<div class="gen-top"><span class="t">${t != null ? mmss(t) : "–"}</span><span class="r"></span></div><div class="gen-box"><span class="mono">${lab}</span></div><div class="gen-cap"></div>`;
      }
      row.appendChild(cell);
    }
  }
  function fmtResult(v) {
    const u = (META[S.env] || META.walker).unit;
    return u === "cm" ? `${v.toFixed(1)} cm` : u === "s" ? `${v.toFixed(1)} s` : `${v.toFixed(1)} m`;
  }
  function drawGens(now) {
    for (const g of genCells) {
      const { ctx, w, h } = fit(g.cv);
      ctx.clearRect(0, 0, w, h);
      const rep = g.rep, frames = rep.frames || [];
      if (!frames.length) continue;
      const K = rep.K || frames[0].length / 5;
      const dtr = rep.dt || 0.04;
      const hold = 1.0;
      const dur = frames.length * dtr + hold;
      const t = Math.max(0, (now - g.start) / 1000) % dur;
      const fi = Math.min(frames.length - 1, Math.floor(t / dtr));
      const fr = frames[fi];
      const isNew = g.cv.closest(".gen").classList.contains("newest");
      const col = isNew ? C.accent : C.ink;
      if (LOCO.has(rep.env || S.env)) {
        const gy = Math.round(h - 22) + 0.5;
        const robotH = (rep.env || S.env) === "cheetah" ? 1.05 : 1.6;
        const s = Math.min((gy - 22) / robotH, w / 2.6);
        const camX = fr[0]; // follow torso
        const X = (x) => w * 0.42 + (x - camX) * s, Y = (y) => gy - y * s;
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
        const m0 = Math.floor(camX - w * 0.42 / s), m1 = Math.ceil(camX + w * 0.58 / s);
        for (let m = Math.max(0, m0); m <= m1; m++) {
          const x = Math.round(X(m)) + 0.5;
          ctx.beginPath(); ctx.moveTo(x, gy); ctx.lineTo(x, gy + (m % 5 === 0 ? 6 : 3)); ctx.stroke();
          if (m % 5 === 0) { mono(ctx, 9.5, C.muted); tracked(ctx, `${m}`, x, gy + 16, "center", 0); }
        }
        if (m0 <= 0) { ctx.fillStyle = C.muted; ctx.fillRect(Math.round(X(0)), gy - 6, 1, 6); }
        strokeOne(ctx, fr, 0, K, X, Y, s, col, false);
        mono(ctx, 9.5, C.muted); tracked(ctx, `${Math.max(0, camX).toFixed(1)} M`, w - 6, 14, "right", 0.4);
      } else if ((rep.env || S.env) === "arm") {
        const s = Math.min(w, h) / 2.4;
        const cx = w / 2, cy = h / 2 + 4;
        const X = (x) => cx + x * s, Y = (y) => cy - y * s;
        ctx.strokeStyle = C.rule; ctx.beginPath(); ctx.arc(cx, cy, s, 0, Math.PI * 2); ctx.stroke();
        const tg = rep.targets && (Array.isArray(rep.targets[fi]) ? rep.targets[fi] : rep.targets[0]);
        if (tg) { ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(X(tg[0]), Y(tg[1]), 4, 0, Math.PI * 2); ctx.stroke(); }
        ctx.fillStyle = C.ink; ctx.fillRect(cx - 8, cy - 1.5, 16, 3);
        strokeOne(ctx, fr, 0, K, X, Y, s, col, false);
      } else {
        const gy = Math.round(h - 22) + 0.5;
        const s = Math.min((gy - 24) / 1.15, w / 2.6);
        const X = (x) => w / 2 + x * s, Y = (y) => gy - y * s;
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
        dashed(ctx, X(0), Y(1.1), gy, C.rule, [2, 3]);
        strokeOne(ctx, fr, 0, K, X, Y, s, col, false);
        mono(ctx, 9.5, C.muted); tracked(ctx, `${(fi * dtr).toFixed(1)} S`, w - 6, 14, "right", 0.4);
      }
    }
  }

  // ---------------- loop ----------------
  let lastNow = performance.now(), chartTick = 0;
  function loop(now) {
    const dt = Math.min(0.1, (now - lastNow) / 1000); lastNow = now;
    drawTrack(now, dt);
    if (now - chartTick > 100) { drawChart(now); chartTick = now; }
    drawGens(now);
    requestAnimationFrame(loop);
  }

  window.addEventListener("resize", () => { chartTick = 0; });
  buildGens(); renderHeadline(); renderControls(); renderReady(); renderFoot();
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(() => requestAnimationFrame(loop));
  connect();
  if (DEMO && params.get("autostart") !== "0") setTimeout(() => { if (S.state === "idle" && params.get("autostart") === "1") startRun(); }, 400);
  window.__macgym = { S, startRun };
})();
