// MacGym web viewer. Talks CONTRACT §3 over ws://<host>/ws, or runs a fake server with ?demo=1.
import { TileRenderer } from "./scene.js";
import { DemoServer, DEMO_BENCH } from "./demo.js";

const params = new URLSearchParams(location.search);
const DEMO = params.has("demo") && params.get("demo") !== "0";
const RENDER_ENVS = Math.max(1, Math.min(1024, parseInt(params.get("r") || "64", 10)));
const N_STOPS = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384];

const $ = (id) => document.getElementById(id);
const el = {
  conn: $("conn"), connText: $("connText"), runState: $("runState"), runStateText: $("runStateText"),
  envName: $("envName"), viewInfo: $("viewInfo"), modeChip: $("modeChip"), robotsNum: $("robotsNum"),
  hint: $("hint"), offline: $("offline"), toast: $("toast"),
  sps: $("spsNum"), total: $("totalSteps"), iter: $("iterNum"), ret: $("meanRet"), elapsed: $("elapsed"),
  chart: $("chart"), chartDelta: $("chartDelta"), spark: $("spark"),
  envPicker: $("envPicker"), numEnvs: $("numEnvs"), numEnvsVal: $("numEnvsVal"),
  btnStart: $("btnStart"), btnStartText: $("btnStartText"), btnShowcase: $("btnShowcase"), btnSave: $("btnSave"),
  benchPanel: $("benchPanel"), benchBody: $("benchBody"), benchMachine: $("benchMachine"), stage: $("stage"), gl: $("gl"),
};

// ---------------- formatting ----------------
function fmtSI(n) {
  if (!isFinite(n)) return "–";
  const a = Math.abs(n);
  if (a < 1e3) return Math.round(n).toString();
  if (a < 1e6) return (n / 1e3).toFixed(a < 1e4 ? 2 : a < 1e5 ? 1 : 0) + "K";
  if (a < 1e9) return (n / 1e6).toFixed(a < 1e7 ? 2 : a < 1e8 ? 1 : 0) + "M";
  return (n / 1e9).toFixed(a < 1e10 ? 2 : 1) + "B";
}
const fmtInt = (n) => Math.round(n).toLocaleString("en-US");
function fmtRet(v) {
  if (v == null || !isFinite(v)) return "–";
  const a = Math.abs(v);
  return a >= 1000 ? fmtInt(v) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
}
function fmtTime(s) {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}` : `${m}:${String(ss).padStart(2, "0")}`;
}

// ---------------- state ----------------
const S = {
  connected: false, runState: "idle", env: "walker", envs: [], numEnvs: 4096,
  sps: 0, spsShown: 0, totalSteps: 0, totalShown: 0, iter: 0, meanRet: null,
  startedAt: null, elapsedBase: 0,
  robotsShown: 4096, retHistory: [], spsHistory: [],
  frame: null, prevFrame: null, frameT: 0, frameDt: 33,
};

// ---------------- transport ----------------
let ws = null, demo = null, reconnectDelay = 800;
function send(obj) {
  if (demo) return demo.send(obj);
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
}

function setConn(kind, text) {
  el.conn.className = "conn " + kind;
  el.connText.textContent = text;
}

function connect() {
  if (DEMO) {
    demo = new DemoServer(onMessage);
    setConn("demo", "DEMO MODE");
    onOpen();
    return;
  }
  const url = params.get("ws") || `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  setConn("connecting", "CONNECTING");
  try { ws = new WebSocket(url); } catch (e) { scheduleReconnect(); return; }
  let offlineTimer = setTimeout(() => el.offline.classList.remove("hidden"), 2500);
  ws.onopen = () => { clearTimeout(offlineTimer); reconnectDelay = 800; S.connected = true; setConn("live", "LIVE"); el.offline.classList.add("hidden"); onOpen(); };
  ws.onmessage = (ev) => { try { onMessage(JSON.parse(ev.data)); } catch (e) { console.warn("bad message", e); } };
  ws.onclose = () => { clearTimeout(offlineTimer); S.connected = false; setConn("down", "RECONNECTING"); scheduleReconnect(); };
  ws.onerror = () => {};
}
function scheduleReconnect() {
  setTimeout(connect, reconnectDelay);
  reconnectDelay = Math.min(5000, reconnectDelay * 1.6);
  if (!DEMO) setTimeout(() => { if (!S.connected) el.offline.classList.remove("hidden"); }, 2500);
}
function onOpen() { send({ cmd: "list" }); send({ cmd: "bench" }); }

function onMessage(m) {
  switch (m.type) {
    case "frame": onFrame(m); break;
    case "metrics": onMetrics(m); break;
    case "status": onStatus(m); break;
    case "envs": onEnvs(m.envs || []); break;
    case "bench": if (m.data || m.bench || m.results) renderBench(m.data || m.bench || m); else if (DEMO && !benchLoaded) renderBench(DEMO_BENCH); break;
    case "saved": toast(`Saved policy${m.path ? " → " + m.path : ""}`); break;
    case "error": toast(m.message || m.error || "Server error", true); break;
    case "info": case "toast": toast(m.message || ""); break;
  }
}

// ---------------- messages ----------------
function onFrame(m) {
  const R = m.R | 0, K = m.K | 0;
  const caps = Float32Array.from(m.capsules || []);
  const f = {
    R, K, capsules: caps, targets: m.targets ? Float32Array.from(m.targets) : null,
    colors: m.colors || null, ground: m.ground ?? null, world_bounds: m.world_bounds || [-2, 2, -0.5, 2],
    camera_x: m.camera_x ? Float32Array.from(m.camera_x) : null, env: m.env,
  };
  const now = performance.now();
  if (S.frame && S.frame.R === R && S.frame.K === K && S.frame.env === f.env) {
    S.prevFrame = S.frame;
    S.frameDt = Math.max(16, Math.min(120, S.frameDt * 0.8 + (now - S.frameT) * 0.2));
  } else {
    S.prevFrame = null;
  }
  S.frame = f; S.frameT = now;
  if (m.env && m.env !== S.env && S.runState !== "idle") setEnv(m.env, false);
  el.viewInfo.textContent = `${fmtInt(R)} of ${fmtInt(S.numEnvs)}`;
}

function onMetrics(m) {
  if (m.sps != null) { S.sps = m.sps; S.spsHistory.push(m.sps); if (S.spsHistory.length > 120) S.spsHistory.shift(); }
  if (m.env_steps != null) S.totalSteps = m.env_steps;
  if (m.iter != null) S.iter = m.iter;
  if (m.num_envs) { S.numEnvs = m.num_envs; syncSlider(); }
  if (m.mean_ep_return != null && isFinite(m.mean_ep_return)) {
    S.meanRet = m.mean_ep_return;
    S.retHistory.push([S.iter, m.mean_ep_return]);
    if (S.retHistory.length > 1200) S.retHistory = S.retHistory.filter((_, i) => i % 2 === 0);
    chart.dirty = true;
  }
  if (typeof m.time === "number" && m.time > 1 && S.startedAt) {
    // trust server's wall clock if it looks cumulative
    const local = (performance.now() - S.startedAt) / 1000;
    if (Math.abs(m.time - local) < 30) S.startedAt = performance.now() - m.time * 1000;
  }
  el.iter.textContent = fmtInt(S.iter);
  el.ret.textContent = fmtRet(S.meanRet);
  if (!S.startedAt && S.runState !== "idle") S.startedAt = performance.now();
}

function onStatus(m) {
  const prev = S.runState;
  S.runState = m.state || "idle";
  if (m.env) setEnv(m.env, false);
  el.runState.className = "run-state " + S.runState;
  el.runStateText.textContent = S.runState.toUpperCase();
  const running = S.runState !== "idle";
  el.btnStart.classList.toggle("stop", running);
  el.btnStartText.textContent = running ? "Stop" : "Start";
  el.btnShowcase.classList.toggle("on", S.runState === "showcase");
  el.modeChip.classList.toggle("hidden", S.runState !== "showcase");
  renderer.setAccent(S.runState === "showcase" ? "#FF4FD8" : "#4FD1FF");
  if (running && !S.startedAt) S.startedAt = performance.now();
  if (!running) S.sps = 0;
  if (!running && prev !== "idle" && S.startedAt) { S.elapsedBase = (performance.now() - S.startedAt) / 1000; S.startedAt = null; }
}

function onEnvs(envs) {
  S.envs = envs;
  el.envPicker.innerHTML = "";
  for (const e of envs) {
    const b = document.createElement("button");
    b.className = "env-btn"; b.dataset.env = e.name;
    b.innerHTML = `<div class="n"></div><div class="d"></div>`;
    b.querySelector(".n").textContent = e.name;
    b.querySelector(".d").textContent = e.description || `obs ${e.obs_dim} · act ${e.act_dim}`;
    b.title = e.description || "";
    b.onclick = () => { setEnv(e.name, true); };
    el.envPicker.appendChild(b);
  }
  if (!envs.find((e) => e.name === S.env) && envs.length) S.env = (envs.find((e) => e.name === "walker") || envs[0]).name;
  setEnv(S.env, false);
}

function isArmEnv() {
  return /arm|reach/i.test(S.env) || !!(S.frame && S.frame.targets && S.frame.ground == null);
}

function setEnv(name, user) {
  const changed = name !== S.env;
  S.env = name;
  el.envName.textContent = name;
  for (const b of el.envPicker.children) b.classList.toggle("active", b.dataset.env === name);
  el.hint.classList.toggle("hidden", !isArmEnv());
  if (user && changed && S.runState !== "idle") start();
}

// ---------------- controls ----------------
function start() {
  S.retHistory = []; S.spsHistory = []; S.iter = 0; S.totalSteps = 0; S.totalShown = 0; S.meanRet = null;
  S.startedAt = performance.now(); S.elapsedBase = 0; chart.dirty = true;
  el.iter.textContent = "0"; el.ret.textContent = "–";
  send({ cmd: "start", env: S.env, num_envs: S.numEnvs, render_envs: RENDER_ENVS });
}
function stop() { send({ cmd: "stop" }); }
function toggleRun() { S.runState === "idle" ? start() : stop(); }
function toggleShowcase() {
  if (S.runState === "idle") { toast("Start training first", true); return; }
  send({ cmd: "showcase", on: S.runState !== "showcase" });
}

el.btnStart.onclick = toggleRun;
el.btnShowcase.onclick = toggleShowcase;
el.btnSave.onclick = () => { if (!S.connected && !DEMO) return toast("Not connected", true); send({ cmd: "save" }); toast("Saving policy…"); };

function syncSlider() {
  let idx = 0, best = Infinity;
  N_STOPS.forEach((n, i) => { const d = Math.abs(Math.log(n / S.numEnvs)); if (d < best) { best = d; idx = i; } });
  el.numEnvs.value = idx;
  el.numEnvsVal.textContent = fmtInt(S.numEnvs);
  el.numEnvs.style.setProperty("--p", `${(idx / (N_STOPS.length - 1)) * 100}%`);
}
el.numEnvs.addEventListener("input", () => {
  S.numEnvs = N_STOPS[+el.numEnvs.value];
  syncSlider();
});
el.numEnvs.addEventListener("change", () => { if (S.runState !== "idle") start(); });
syncSlider();

window.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" && e.target.type !== "range")) return;
  if (e.code === "Space") { e.preventDefault(); toggleRun(); }
  else if (e.key === "s" || e.key === "S") { if (e.metaKey || e.ctrlKey) { e.preventDefault(); send({ cmd: "save" }); toast("Saving policy…"); } else toggleShowcase(); }
});

let toastTimer = null;
function toast(msg, err = false) {
  el.toast.textContent = msg;
  el.toast.classList.toggle("err", err);
  el.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.remove("show"), 2600);
}

// ---------------- renderer + interaction ----------------
const renderer = new TileRenderer(el.gl);
function resize() {
  const r = el.stage.getBoundingClientRect();
  renderer.resize(r.width, r.height);
  chart.resize(); spark.resize();
}

let dragging = false, lastTargetSend = 0;
function pointerPos(e) { const r = el.gl.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
function sendTargetAt(e, force) {
  const [px, py] = pointerPos(e);
  const i = renderer.tileAt(px, py);
  if (i < 0) return;
  const w = renderer.toWorld(i, px, py);
  if (!w) return;
  const now = performance.now();
  if (!force && now - lastTargetSend < 33) return;
  lastTargetSend = now;
  send({ cmd: "set_target", x: Math.round(w.x * 1000) / 1000, y: Math.round(w.y * 1000) / 1000 });
}
el.gl.addEventListener("pointermove", (e) => {
  const [px, py] = pointerPos(e);
  renderer.hover = renderer.tileAt(px, py);
  el.gl.style.cursor = isArmEnv() && renderer.hover >= 0 ? "crosshair" : "default";
  if (dragging && isArmEnv()) sendTargetAt(e, false);
});
el.gl.addEventListener("pointerleave", () => { renderer.hover = -1; });
el.gl.addEventListener("pointerdown", (e) => {
  if (!isArmEnv()) return;
  dragging = true; el.gl.setPointerCapture(e.pointerId); sendTargetAt(e, true);
});
window.addEventListener("pointerup", () => { dragging = false; });

// ---------------- frame interpolation ----------------
let interp = null;
function interpolatedFrame(now) {
  const f = S.frame;
  if (!f) return null;
  const p = S.prevFrame;
  if (!p) return f;
  const t = Math.min(1, (now - S.frameT) / S.frameDt);
  if (t >= 1) return f;
  const n = f.capsules.length;
  if (!interp || interp.capsules.length !== n) interp = { capsules: new Float32Array(n), camera_x: null };
  const a = p.capsules, b = f.capsules, out = interp.capsules;
  const K5 = f.K * 5;
  // per-env: skip lerp if the env teleported (auto-reset)
  for (let i = 0; i < f.R; i++) {
    const o = i * K5;
    const jump = Math.abs(a[o] - b[o]) + Math.abs(a[o + 1] - b[o + 1]);
    const tt = jump > 0.6 ? 1 : t;
    for (let k = o; k < o + K5; k++) out[k] = a[k] + (b[k] - a[k]) * tt;
  }
  let cam = null;
  if (f.camera_x && p.camera_x && p.camera_x.length === f.camera_x.length) {
    if (!interp.camera_x || interp.camera_x.length !== f.camera_x.length) interp.camera_x = new Float32Array(f.camera_x.length);
    cam = interp.camera_x;
    for (let i = 0; i < cam.length; i++) {
      const d = f.camera_x[i] - p.camera_x[i];
      cam[i] = Math.abs(d) > 0.6 ? f.camera_x[i] : p.camera_x[i] + d * t;
    }
  }
  return { ...f, capsules: out, camera_x: cam || f.camera_x };
}

// ---------------- charts ----------------
class LineChart {
  constructor(canvas, opts) {
    this.c = canvas; this.ctx = canvas.getContext("2d"); this.o = opts;
    this.lo = 0; this.hi = 1; this.dirty = true; this.w = 0; this.h = 0;
  }
  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = this.c.getBoundingClientRect();
    this.w = r.width; this.h = r.height;
    this.c.width = Math.max(1, Math.round(r.width * dpr)); this.c.height = Math.max(1, Math.round(r.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.dirty = true;
  }
  draw(values) {
    const { ctx, w, h, o } = this;
    // smooth auto-scaling
    let tlo = Infinity, thi = -Infinity;
    for (const v of values) { if (v < tlo) tlo = v; if (v > thi) thi = v; }
    if (!values.length) { tlo = 0; thi = 1; }
    if (o.zeroBase) tlo = Math.min(0, tlo);
    if (thi - tlo < 1e-6) { thi += 1; tlo -= 1; }
    const padv = (thi - tlo) * 0.12; tlo -= o.zeroBase && tlo === 0 ? 0 : padv; thi += padv;
    const k = 0.12;
    const moving = Math.abs(this.lo - tlo) + Math.abs(this.hi - thi) > (thi - tlo) * 0.002;
    if (!this.dirty && !moving) return;
    this.lo += (tlo - this.lo) * k; this.hi += (thi - this.hi) * k;
    if (!moving) { this.lo = tlo; this.hi = thi; }
    this.dirty = false;
    ctx.clearRect(0, 0, w, h);
    const L = o.left ?? 0, Rr = o.right ?? 0, T = o.top ?? 4, B = o.bottom ?? 4;
    const pw = w - L - Rr, ph = h - T - B;
    const yOf = (v) => T + ph - ((v - this.lo) / (this.hi - this.lo)) * ph;
    if (o.grid && values.length >= 2) {
      ctx.font = "500 10px JetBrains Mono, ui-monospace, monospace";
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (let g = 0; g <= 3; g++) {
        const v = this.lo + ((this.hi - this.lo) * g) / 3;
        const y = yOf(v);
        ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(L, Math.round(y) + 0.5); ctx.lineTo(L + pw, Math.round(y) + 0.5); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(124,133,153,0.75)";
        ctx.fillText(fmtRet(v), w - 2, Math.min(h - 6, Math.max(6, y)));
      }
    }
    if (values.length < 2) {
      if (o.empty) { ctx.fillStyle = "rgba(124,133,153,0.6)"; ctx.font = "500 12px Inter, system-ui"; ctx.textAlign = "center"; ctx.fillText(o.empty, L + pw / 2, T + ph / 2); }
      return;
    }
    const n = values.length;
    const pts = values.map((v, i) => [L + (i / (n - 1)) * pw, yOf(v)]);
    const path = () => {
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < n - 1; i++) {
        const mx = (pts[i][0] + pts[i + 1][0]) / 2, my = (pts[i][1] + pts[i + 1][1]) / 2;
        ctx.quadraticCurveTo(pts[i][0], pts[i][1], mx, my);
      }
      ctx.lineTo(pts[n - 1][0], pts[n - 1][1]);
    };
    // fill
    path(); ctx.lineTo(pts[n - 1][0], T + ph); ctx.lineTo(pts[0][0], T + ph); ctx.closePath();
    const fg = ctx.createLinearGradient(0, T, 0, T + ph);
    fg.addColorStop(0, o.fill0); fg.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = fg; ctx.fill();
    // line with glow
    const lg = ctx.createLinearGradient(L, 0, L + pw, 0);
    lg.addColorStop(0, o.c0); lg.addColorStop(1, o.c1);
    path(); ctx.strokeStyle = lg; ctx.lineWidth = o.lw || 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.shadowColor = o.glow; ctx.shadowBlur = 12; ctx.stroke(); ctx.shadowBlur = 0;
    // head dot
    const [hx, hy] = pts[n - 1];
    ctx.fillStyle = o.c1; ctx.shadowColor = o.c1; ctx.shadowBlur = 14;
    ctx.beginPath(); ctx.arc(hx, hy, o.dot || 3.5, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
  }
}
const chart = new LineChart(el.chart, {
  c0: "#4FD1FF", c1: "#A3FF6B", glow: "rgba(163,255,107,0.55)", fill0: "rgba(120,240,180,0.16)",
  grid: true, right: 46, top: 8, bottom: 8, lw: 2.4, empty: "Press Start to train",
});
const spark = new LineChart(el.spark, { c0: "rgba(79,209,255,0.2)", c1: "#4FD1FF", glow: "rgba(79,209,255,0.6)", fill0: "rgba(79,209,255,0.14)", lw: 1.6, dot: 2.5, top: 4, bottom: 2 });

// ---------------- bench ----------------
let benchLoaded = false;
function normalizeBench(j) {
  const groups = new Map();
  const add = (task, label, sps) => {
    if (!(sps > 0)) return;
    task = String(task || "benchmark");
    if (!groups.has(task)) groups.set(task, []);
    groups.get(task).push({ label: String(label), sps: +sps, gpu: /mlx|gpu|metal/i.test(label) });
  };
  const spsOf = (o) => {
    if (typeof o === "number") return o;
    if (!o || typeof o !== "object") return null;
    for (const k of ["sps", "env_steps_per_sec", "steps_per_sec", "steps_per_second", "env_steps_per_s", "throughput"]) if (typeof o[k] === "number") return o[k];
    for (const k in o) if (/sps|per_s|per_sec|throughput/i.test(k) && typeof o[k] === "number") return o[k];
    return null;
  };
  const labelOf = (o, fallback) => {
    const base = o.backend || o.name || o.method || o.engine || o.label || fallback;
    const n = o.num_envs || o.N || o.n_envs;
    const procs = o.processes || o.workers || o.num_procs;
    const nTxt = n >= 64 && !/N=/.test(base) ? ` · ${n % 1024 === 0 ? n / 1024 + "K" : fmtSI(n)}` : "";
    return `${base}${nTxt}${procs && !/proc/i.test(base) ? ` ×${procs}` : ""}`;
  };
  const arr = Array.isArray(j) ? j : Array.isArray(j.results) ? j.results : Array.isArray(j.benchmarks) ? j.benchmarks : null;
  if (arr) {
    for (const r of arr) add(r.task || r.env || r.benchmark || "benchmark", labelOf(r, "run"), spsOf(r));
  } else {
    const src = j.results && typeof j.results === "object" ? j.results : j;
    for (const [task, v] of Object.entries(src)) {
      if (/machine|info|hardware|meta|date|notes?|system/i.test(task) || !v || typeof v !== "object") continue;
      if (Array.isArray(v)) { for (const r of v) add(task, labelOf(r, "run"), spsOf(r)); continue; }
      const direct = spsOf(v);
      if (direct != null && (v.backend || v.name)) { add(v.task || task, labelOf(v, task), direct); continue; }
      for (const [label, val] of Object.entries(v)) {
        const s = spsOf(val);
        if (s != null && s > 0) add(task, val && typeof val === "object" ? labelOf(val, label) : label.replace(/_/g, " "), s);
      }
    }
  }
  let machine = j.machine || j.machine_info || j.hardware || j.system || "";
  if (machine && typeof machine === "object") machine = machine.chip || machine.cpu || machine.processor || machine.name || Object.values(machine).filter((x) => typeof x === "string").slice(0, 2).join(" · ");
  return { groups: [...groups.entries()], machine: String(machine || ""), notes: j.notes || j.note || "" };
}

function renderBench(j) {
  let b;
  try { b = normalizeBench(j); } catch (e) { console.warn("bench parse failed", e); return; }
  if (!b.groups.length) return;
  benchLoaded = true;
  el.benchPanel.classList.remove("hidden");
  el.benchMachine.textContent = b.machine;
  el.benchMachine.title = b.machine;
  el.benchBody.innerHTML = "";
  for (const [task, rows] of b.groups) {
    rows.sort((a, c) => c.sps - a.sps);
    const max = rows[0].sps;
    const gpuBest = Math.max(0, ...rows.filter((r) => r.gpu).map((r) => r.sps));
    const cpuBest = Math.max(0, ...rows.filter((r) => !r.gpu).map((r) => r.sps));
    const g = document.createElement("div");
    g.className = "bench-group";
    const title = document.createElement("div");
    title.className = "bench-title";
    title.innerHTML = `<span></span>`;
    title.firstChild.textContent = task;
    if (gpuBest && cpuBest) {
      const sp = document.createElement("span"); sp.className = "speedup";
      const x = gpuBest / cpuBest; sp.textContent = `${x >= 10 ? Math.round(x) : x.toFixed(1)}× faster`;
      title.appendChild(sp);
    }
    g.appendChild(title);
    for (const r of rows) {
      const row = document.createElement("div");
      row.className = "bar-row" + (r.gpu ? " gpu" : "");
      row.innerHTML = `<div class="bar-label"></div><div class="bar-track"><div class="bar ${r.gpu ? "gpu" : "cpu"}"></div></div><div class="bar-val"></div>`;
      row.querySelector(".bar-label").textContent = r.label; row.querySelector(".bar-label").title = r.label;
      row.querySelector(".bar-val").textContent = fmtSI(r.sps);
      const bar = row.querySelector(".bar");
      requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = `${Math.max(0.6, (r.sps / max) * 100)}%`; }));
      g.appendChild(row);
    }
    el.benchBody.appendChild(g);
  }
  if (b.notes && typeof b.notes === "string") {
    const n = document.createElement("div"); n.className = "bench-note"; n.textContent = b.notes; el.benchBody.appendChild(n);
  }
}
fetch("bench.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((j) => { if (j) renderBench(j); }).catch(() => {});

// ---------------- main loop ----------------
let lastNow = performance.now();
function tick(now) {
  const dt = Math.min(0.1, (now - lastNow) / 1000); lastNow = now;
  const ease = 1 - Math.exp(-dt * 6);
  // counters
  S.spsShown += (S.sps - S.spsShown) * ease;
  el.sps.textContent = fmtSI(S.spsShown);
  S.totalShown += (S.totalSteps - S.totalShown) * Math.min(1, ease * 1.5);
  el.total.textContent = fmtSI(S.totalShown);
  S.robotsShown += (S.numEnvs - S.robotsShown) * Math.min(1, ease * 1.5);
  if (Math.abs(S.robotsShown - S.numEnvs) < 0.5) S.robotsShown = S.numEnvs;
  el.robotsNum.textContent = fmtInt(S.robotsShown);
  const el_s = S.startedAt ? (now - S.startedAt) / 1000 : S.elapsedBase;
  el.elapsed.textContent = fmtTime(el_s);
  // chart delta
  const h = S.retHistory;
  if (h.length > 1) {
    const d = h[h.length - 1][1] - h[0][1];
    el.chartDelta.textContent = `${d >= 0 ? "▲" : "▼"} ${fmtRet(Math.abs(d))} · it ${fmtInt(h[h.length - 1][0])}`;
    el.chartDelta.style.color = d >= 0 ? "#A3FF6B" : "#FF5C7A";
  } else el.chartDelta.textContent = "";
  chart.draw(h.map((p) => p[1]));
  spark.dirty = true; spark.draw(S.spsHistory);
  renderer.draw(interpolatedFrame(now), now / 1000);
  requestAnimationFrame(tick);
}

window.addEventListener("resize", resize);
resize();
requestAnimationFrame(tick);
connect();
if (DEMO && !params.has("idle")) setTimeout(() => { if (S.runState === "idle") start(); }, 400);

// expose for integration/debug
window.macgym = { send, state: S, start, stop };
