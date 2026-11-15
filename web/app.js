// MacGym viewer v2. Talks to ws://<host>/ws (CONTRACT.md §3 + REDESIGN.md), or a fake server with ?demo=1.
(function () {
  "use strict";
  const T = window.MacGymTrack; // shared drawing (track.js)
  const { C, fit, strokeRobots, strokeOne, mono, tracked, dashed, label } = T;
  const REPLAY_AT = [0, 10, 30, 60, 120, 240];
  const LOCO = new Set(["walker", "cheetah", "hopper", "tripod", "worm"]);
  // Framework built-ins in display order; tabs show whichever of these the server lists (+ any others it adds).
  const TAB_ORDER = ["walker", "cheetah", "hopper", "tripod", "worm", "arm", "cartpole"];
  const LOCO_LABEL = "average distance per finished episode";
  const LIVE_LABEL = "average distance so far (no episode finished yet)";
  const META = {
    walker: { unit: "m", label: LOCO_LABEL, hib: true },
    cheetah: { unit: "m", label: LOCO_LABEL, hib: true },
    hopper: { unit: "m", label: LOCO_LABEL, hib: true },
    tripod: { unit: "m", label: LOCO_LABEL, hib: true },
    worm: { unit: "m", label: LOCO_LABEL, hib: true },
    arm: { unit: "cm", label: "average miss distance", hib: false },
    cartpole: { unit: "s", label: "average time balanced", hib: true },
  };
  // Any env not in META (e.g. "playground" frames from /playground, or a custom robot) is a
  // planar locomotion robot: label it by its own name instead of pretending it is the walker.
  function metaFor(env, robot) {
    if (META[env]) return META[env];
    const name = robot || (env === "playground" ? "playground robot" : env);
    return { unit: "m", label: `${LOCO_LABEL} · ${name}`, hib: true, custom: true };
  }
  const isLoco = (env) => LOCO.has(env) || (!!env && !META[env]);
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
      case "envs": buildTabs(m.envs || []); break;
      case "error": console.warn("server:", m.message); break;
    }
  }

  // ---------------- messages ----------------
  function onStatus(m) {
    const prevState = S.state;
    S.state = m.state || "idle";
    if (m.num_envs) { S.numEnvs = m.num_envs; $("count").value = String(m.num_envs); }
    // a finished Playground run is not a viewer env: keep the current tab so Train starts a built-in
    const adopt = m.env && m.env !== S.env && !(m.env === "playground" && S.state === "idle");
    if (adopt) setEnv(m.env, false);
    if (S.state !== "idle" && prevState === "idle" && !S.localStart) S.localStart = performance.now();
    if (S.state === "idle") S.sps = 0;
    renderControls(); renderFoot(); renderReady();
  }

  function onFrame(m) {
    if (m.env && m.env !== S.env) setEnv(m.env, false, m.robot_name || m.robot);
    const R = m.R | 0, K = m.K | 0;
    const caps = Float32Array.from(m.capsules || []);
    let dist = m.dist ? Float32Array.from(m.dist) : null;
    let shift = null;
    if (isLoco(S.env) && !dist) {
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

  function xpText(steps, ctrlDt, secs, noun) {
    if (!steps || !ctrlDt) return null;
    const h = steps * ctrlDt / 3600;
    const v = h >= 48 ? `${(h / 24).toFixed(h >= 240 ? 0 : 1)} days` : h >= 1 ? `${h.toFixed(h >= 10 ? 0 : 1)} hours` : `${Math.round(h * 60)} minutes`;
    return `<b>${v}</b> of ${noun} experience in ${mmss(secs)} of wall time`;
  }
  function renderXp() {
    const t = S.frame && xpText(S.envSteps, S.ctrlDt, S.trainSeconds, S.env);
    if (t) $("heroLabel").innerHTML = t; else $("heroLabel").textContent = "Track";
  }
  function onMetrics(m) {
    if (m.env_steps != null) S.envSteps = m.env_steps;
    if (m.iter != null) S.iter = m.iter;
    if (m.sps != null) S.sps = m.sps;
    if (m.num_envs) S.numEnvs = m.num_envs;
    let ts = m.train_seconds;
    if (ts == null) ts = S.localStart ? (performance.now() - S.localStart) / 1000 : (typeof m.time === "number" ? m.time : 0);
    S.trainSeconds = ts;
    const meta = metaFor(S.env);
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
    // one headline number per screen: locomotion always reads "average distance per finished episode"
    if (isLoco(S.env) && S.unit === "m") S.label = m.human_source === "live" ? LIVE_LABEL : meta.label;
    const last = S.hist[S.hist.length - 1];
    if (last && ts < last[0] - 1) S.hist = [];
    S.hist.push([ts, v]);
    if (S.hist.length > 2000) S.hist = S.hist.filter((_, i) => i % 2 === 0);
    renderHeadline(); renderFoot(); renderXp();
  }

  function onReplay(m) {
    if (m.env && m.env !== S.env) return;
    if (m.dt) S.ctrlDt = m.dt / 2;
    const idx = S.replays.findIndex((r) => r.label === m.label);
    const rep = { ...m, born: performance.now() };
    if (m.label === "now") S.replays = S.replays.filter((r) => r.label !== "now");
    if (idx >= 0) S.replays[idx] = rep; else S.replays.push(rep);
    if (isLoco(S.env) && m.label !== "now" && idx < 0 && S.frame) S.avgMarks.push({ label: m.label, x: S.liveMean || 0 });
    S.replays.sort((a, b) => (a.train_seconds ?? 0) - (b.train_seconds ?? 0));
    buildGens();
  }

  // ---------------- controls ----------------
  $("tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-env]");
    if (!b) return;
    const env = b.dataset.env;
    if (env === S.env) return;
    setEnv(env, true);
    if (S.state !== "idle") startRun();
  });
  const titleCase = (n) => n.charAt(0).toUpperCase() + n.slice(1);
  function buildTabs(envs) {
    const names = envs.map((e) => e.name).filter((n) => n && n !== "stub" && n !== "playground");
    if (!names.length) return;
    names.sort((a, b) => {
      const ia = TAB_ORDER.indexOf(a), ib = TAB_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
    });
    const nav = $("tabs");
    nav.innerHTML = "";
    for (const n of names) {
      const b = document.createElement("button");
      b.dataset.env = n; b.textContent = titleCase(n);
      const e = envs.find((x) => x.name === n);
      if (e && e.description) b.title = e.description;
      b.classList.toggle("on", n === S.env);
      nav.appendChild(b);
    }
  }
  $("count").addEventListener("change", (e) => { S.numEnvs = +e.target.value; renderReady(); renderFoot(); });
  $("train").addEventListener("click", () => { if (S.state === "idle") startRun(); else send({ cmd: "stop" }); });
  $("load").addEventListener("click", () => { resetRun(); send({ cmd: "load", env: S.env, num_envs: S.numEnvs }); send({ cmd: "render", render_envs: 256 }); });

  function resetRun() {
    S.hist = []; S.replays = []; S.avgMarks = []; S.humanValue = null; S.humanBest = null; S.record = 0; S.iter = 0; S.sps = 0; S.envSteps = 0; S.trail = null; S.dens = null;
    S.frame = null; S.prev = null; S.localStart = performance.now(); S.trainSeconds = 0;
    S.cam.range = S.cam.targetRange = 6;
    buildGens(); renderHeadline();
  }
  function startRun() {
    resetRun();
    send({ cmd: "start", env: S.env, num_envs: S.numEnvs, render_envs: 256 });
  }
  function setEnv(env, user, robot) {
    if (!env) return;
    const meta = metaFor(env, robot);
    const changed = env !== S.env;
    S.env = env;
    for (const b of document.querySelectorAll("#tabs button")) b.classList.toggle("on", b.dataset.env === env);
    if (changed) {
      S.frame = null; S.prev = null; S.hist = []; S.replays = []; S.avgMarks = []; S.humanValue = null; S.record = 0; S.trail = null; S.dens = null;
      S.unit = meta.unit; S.label = meta.label; S.hib = meta.hib;
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
    renderXp();
    $("heroNote").textContent = S.frame
      ? `${fmtInt(R)} of ${fmtInt(S.numEnvs)} robots shown, overlaid${isLoco(S.env) ? " · each from its own start" : ""}`
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
    parts.push(`sim only <b>${S.simSps ? fmtSI(S.simSps) : "–"}</b> steps/s`);
    parts.push(`training (sim + PPO) <b>${S.sps ? fmtSI(S.sps) : "–"}</b> steps/s`);
    parts.push(`iteration ${S.iter ? `<b>${fmtInt(S.iter)}</b>` : "–"}`);
    if (DEMO) parts.push("demo data");
    else if (!S.connected) parts.push("server offline, reconnecting");
    $("foot").innerHTML = parts.map((x) => `<span>${x}</span>`).join('<span class="sep">·</span>');
  }

  // ---------------- canvases ----------------
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

  function drawTrack(now, dt) {
    const { ctx, w, h } = fit(trackCv);
    ctx.clearRect(0, 0, w, h);
    const pad = 32;
    const f = interpolated(now);
    if (isLoco(S.env)) drawLoco(ctx, w, h, pad, f, dt);
    else if (S.env === "arm") drawArm(ctx, w, h, pad, f);
    else drawCart(ctx, w, h, pad, f);
  }

  function drawLoco(ctx, w, h, pad, f, dt) {
    T.drawLoco(ctx, w, h, pad, f, lerpBuf, lerpDist, S, dt, { robotH: S.env === "cheetah" ? 1.1 : 1.65 });
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
  function drawChart(now) { T.drawChart(chartCv, S); }

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
    const longest = shown.reduce((a, r) => Math.max(a, (r.frames || []).length * (r.dt || 0.04)), 0);
    $("gensNote").textContent = "saved policy replayed for up to 8 s, ends early on a fall";
    void longest;
    for (let i = 0; i < 5; i++) {
      const rep = shown[i];
      const cell = document.createElement("div");
      cell.className = "gen";
      if (rep) {
        if (i === shown.length - 1) cell.classList.add("newest");
        const loco = isLoco(rep.env || S.env);
        const resTxt = rep.result != null ? fmtResult(rep.result) : "";
        cell.innerHTML = `<div class="gen-top"><span class="t">${rep.label === "now" ? "NOW" : rep.label}</span><span class="r"></span></div><div class="gen-box"><canvas></canvas></div><div class="gen-cap"></div>`;
        cell.querySelector(".gen-top .r").textContent = loco ? "" : resTxt;
        cell.querySelector(".gen-cap").textContent = loco
          ? `${replaySecs(rep)} s replay · walked ${resTxt || "–"}${/fell/.test(rep.result_text || "") ? " · fell" : ""}`
          : rep.result_text || "";
        cell.title = rep.result_text || "";
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
  function replaySecs(rep) {
    const sec = (rep.frames || []).length * (rep.dt || 0.04);
    return sec >= 2 ? String(Math.round(sec)) : sec.toFixed(1);
  }
  function fmtResult(v) {
    const u = metaFor(S.env).unit;
    return u === "cm" ? `${v.toFixed(1)} cm` : u === "s" ? `${v.toFixed(1)} s` : `${v.toFixed(1)} m`;
  }
  function drawGens(now) {
    for (const g of genCells) {
      const env = g.rep.env || S.env;
      const kind = isLoco(env) ? "loco" : env === "arm" ? "arm" : "cart";
      const isNew = g.cv.closest(".gen").classList.contains("newest");
      T.drawReplay(g.cv, g.rep, now, g.start, isNew, kind, { robotH: env === "cheetah" ? 1.05 : 1.6 });
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
