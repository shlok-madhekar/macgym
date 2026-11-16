// MacGym playground: edit a robot in Python, press Run, watch thousands of copies learn.
// Talks to ws://<host>/ws: {"cmd":"preview_code"} -> preview, {"cmd":"run_code"} -> v2 training stream.
const T = window.MacGymTrack;
const { C, SANS, MONO, clamp, mmss, fit, mono, tracked, label } = T;
const $ = (id) => document.getElementById(id);
const fmtInt = (n) => Math.round(n).toLocaleString("en-US");
const fmtSI = (n) => n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 1 : 2) + "M" : n >= 1e3 ? Math.round(n / 1e3) + "K" : String(Math.round(n));
const REPLAY_AT = [0, 10, 30, 60, 120, 240];
const fmtM = (v) => `${v.toFixed(Math.abs(v) < 10 ? 2 : Math.abs(v) < 100 ? 1 : 0)} m`;
const params = new URLSearchParams(location.search);

const S = {
  connected: false, state: "idle", numEnvs: 4096, preset: params.get("robot") || "tripod",
  code: "", ranCode: null, dirty: false,
  preview: null, previewPrev: null, previewT: 0, previewErr: null, seq: 0,
  interp: new T.Interp(), hasFrame: false,
  hist: [], humanValue: null, humanBest: null, unit: "m", label: "average distance per finished episode", humanSource: null, robotName: null,
  iter: 0, sps: 0, simSps: null, trainSeconds: 0, runStart: 0, lastProgLog: -99,
  replays: [], avgMarks: [], liveMean: 0, record: 0, cam: { range: 6, targetRange: 6 },
  machine: "Apple Silicon GPU", robotH: 1.0, robotW: 1.0, runClock: 0,
};


// Simulated experience = env steps x control dt (dt from the replay message: replay dt = 2 x control dt).
// m:ss under an hour, h:mm:ss above
function fmtWall(secs) {
  secs = Math.max(0, Math.floor(secs));
  const h = Math.floor(secs / 3600), m = Math.floor((secs % 3600) / 60), s2 = String(secs % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${s2}` : `${m}:${s2}`;
}
function xpText(steps, ctrlDt, secs, noun) {
  if (!steps || !ctrlDt) return null;
  const h = steps * ctrlDt / 3600;
  const v = h >= 48 ? `${(h / 24).toFixed(h >= 240 ? 0 : 1)} days` : h >= 1 ? `${h.toFixed(h >= 10 ? 0 : 1)} hours` : `${Math.round(h * 60)} minutes`;
  return `<span class="sv">${v}</span><span class="sl">of practice in ${fmtWall(secs)}</span>`;
}

// ======================================================================
// Editor (CodeMirror 6 via esm.sh, textarea fallback)
// ======================================================================
let editor = null; // { get(), set(text), setError(line, msg), clearError(), focus() }

async function makeCodeMirror(host, onChange, onRun) {
  const V = "https://esm.sh/";
  const [cm, py, view, state, lang, hl] = await Promise.all([
    import(V + "codemirror@6.0.1"), import(V + "@codemirror/lang-python@6"),
    import(V + "@codemirror/view@6"), import(V + "@codemirror/state@6"),
    import(V + "@codemirror/language@6"), import(V + "@lezer/highlight@1"),
  ]);
  const { EditorView, basicSetup } = cm;
  const { Decoration, WidgetType, keymap } = view;
  const { StateEffect, StateField, Prec } = state;
  const { tags: t } = hl;

  class ErrWidget extends WidgetType {
    constructor(msg) { super(); this.msg = msg; }
    eq(o) { return o.msg === this.msg; }
    toDOM() { const s = document.createElement("div"); s.className = "cm-errmsg"; s.textContent = this.msg; return s; }
  }
  const setErr = StateEffect.define();
  const errField = StateField.define({
    create: () => Decoration.none,
    update(deco, tr) {
      deco = deco.map(tr.changes);
      for (const e of tr.effects) {
        if (!e.is(setErr)) continue;
        if (!e.value) { deco = Decoration.none; continue; }
        const n = clamp(e.value.line, 1, tr.state.doc.lines);
        const line = tr.state.doc.line(n);
        deco = Decoration.set([
          Decoration.line({ class: "cm-errline" }).range(line.from),
          Decoration.widget({ widget: new ErrWidget(e.value.msg), side: 1, block: true }).range(line.to),
        ]);
      }
      return deco;
    },
    provide: (f) => EditorView.decorations.from(f),
  });

  const theme = EditorView.theme({
    "&": { height: "100%", fontSize: "12px", backgroundColor: "var(--window-bg)", color: "var(--label-primary)" },
    ".cm-scroller": { fontFamily: "var(--font-mono)", lineHeight: "18px", paddingTop: "4px" },
    ".cm-content": { caretColor: "var(--label-primary)", padding: "8px 0 40px" },
    ".cm-line": { padding: "0 18px 0 10px" },
    ".cm-gutters": { backgroundColor: "var(--window-bg)", color: "var(--label-tertiary)", border: "none", paddingLeft: "8px" },
    ".cm-lineNumbers .cm-gutterElement": { padding: "0 8px 0 4px", minWidth: "24px" },
    ".cm-activeLine": { backgroundColor: "var(--fill-quaternary)" },
    ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--label-secondary)" },
    ".cm-cursor, .cm-dropCursor": { borderLeft: "1.5px solid var(--label-primary)" },
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": { backgroundColor: "var(--text-selection-bg) !important" },
    ".cm-matchingBracket": { backgroundColor: "var(--fill-tertiary)", outline: "none" },
    ".cm-foldGutter .cm-gutterElement": { color: "var(--label-quaternary)" },
    ".cm-tooltip": { border: "1px solid var(--separator)", backgroundColor: "var(--window-bg)", fontFamily: "var(--font-mono)", fontSize: "12px" },
    ".cm-tooltip-autocomplete ul li[aria-selected]": { backgroundColor: "var(--selection-fill)", color: "var(--label-primary)" },
    ".cm-panels": { backgroundColor: "var(--window-bg)", color: "var(--label-primary)", borderTop: "1px solid var(--separator)" },
    ".cm-searchMatch": { backgroundColor: "var(--fill-tertiary)" },
  }, { dark: true });
  const highlight = lang.HighlightStyle.define([
    { tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword, t.operatorKeyword], color: "var(--label-primary)", fontWeight: "600" },
    { tag: [t.string, t.special(t.string), t.docString], color: "var(--label-secondary)" },
    { tag: [t.number, t.bool, t.null], color: "var(--label-secondary)" },
    { tag: [t.comment, t.lineComment], color: "var(--label-tertiary)" },
    { tag: [t.function(t.variableName), t.function(t.propertyName)], color: "var(--label-primary)" },
    { tag: [t.definition(t.variableName), t.definition(t.function(t.variableName))], color: "var(--label-primary)", fontWeight: "500" },
    { tag: [t.propertyName, t.variableName], color: "var(--label-primary)" },
    { tag: [t.operator, t.punctuation, t.bracket, t.separator], color: "var(--label-tertiary)" },
    { tag: [t.self, t.className], color: "var(--label-primary)", fontWeight: "500" },
  ]);
  const ed = new EditorView({
    parent: host,
    doc: "",
    extensions: [
      Prec.highest(keymap.of([{ key: "Mod-Enter", preventDefault: true, run: () => { onRun(); return true; } }])),
      basicSetup, py.python(), theme, lang.syntaxHighlighting(highlight), errField,
      EditorView.updateListener.of((u) => {
        if (u.docChanged) onChange(ed.state.doc.toString());
        if (u.selectionSet || u.docChanged) {
          const pos = u.state.selection.main.head, line = u.state.doc.lineAt(pos);
          $("edPos").textContent = `Ln ${line.number}, Col ${pos - line.from + 1}`;
        }
      }),
    ],
  });
  return {
    kind: "codemirror",
    get: () => ed.state.doc.toString(),
    set: (text) => ed.dispatch({ changes: { from: 0, to: ed.state.doc.length, insert: text }, selection: { anchor: 0 }, scrollIntoView: true }),
    setError: (line, msg) => {
      ed.dispatch({ effects: setErr.of({ line, msg }) });
      const ln = ed.state.doc.line(clamp(line, 1, ed.state.doc.lines));
      ed.dispatch({ effects: EditorView.scrollIntoView(ln.from, { y: "nearest" }) });
    },
    clearError: () => ed.dispatch({ effects: setErr.of(null) }),
    focus: () => ed.focus(),
    view: ed,
  };
}

function makeTextarea(host, onChange, onRun) {
  host.innerHTML = '<div class="fallback"><pre></pre><textarea spellcheck="false"></textarea></div>';
  const ta = host.querySelector("textarea"), gut = host.querySelector("pre");
  let errLine = null;
  const nums = () => { const n = ta.value.split("\n").length; gut.innerHTML = Array.from({ length: n }, (_, i) => i + 1 === errLine ? `<b style="color:${C.accent}">${i + 1}</b>` : i + 1).join("\n"); };
  ta.addEventListener("input", () => { nums(); onChange(ta.value); });
  ta.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); onRun(); }
    if (e.key === "Tab") { e.preventDefault(); document.execCommand("insertText", false, "    "); }
  });
  return {
    kind: "textarea",
    get: () => ta.value,
    set: (text) => { ta.value = text; nums(); },
    setError: (line) => { errLine = line; nums(); },
    clearError: () => { errLine = null; nums(); },
    focus: () => ta.focus(),
  };
}

// ======================================================================
// Transport
// ======================================================================
let ws = null, retry = 500;
function send(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
function connect() {
  const url = params.get("ws") || `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  try { ws = new WebSocket(url); } catch (e) { return later(); }
  ws.onopen = () => {
    S.connected = true; retry = 500; renderConn(); renderFoot();
    send({ cmd: "status" });
    if (S.code) requestPreview();
  };
  ws.onmessage = (ev) => { try { onMessage(JSON.parse(ev.data)); } catch (e) { console.warn("bad message", e); } };
  ws.onclose = () => { S.connected = false; renderConn(); renderFoot(); later(); };
  ws.onerror = () => {};
}
function later() { setTimeout(connect, retry); retry = Math.min(5000, retry * 1.6); }

// While a new run is building, ignore the tail of the previous run's stream.
function live(m) { return m.env === "playground" && S.state !== "building"; }
function onMessage(m) {
  switch (m.type) {
    case "preview": onPreview(m); break;
    case "status": onStatus(m); break;
    case "frame": if (live(m)) onFrame(m); break;
    case "metrics": if (live(m)) onMetrics(m); break;
    case "replay": if (live(m)) onReplay(m); break;
    case "info": if ("sim_sps" in m && m.env === "playground") { S.simSps = m.sim_sps; renderFoot(); } break;
    case "log": log(m.line, m.level || "info"); break;
    case "error": onError(m); break;
  }
}

// ======================================================================
// Preview
// ======================================================================
let previewTimer = null;
function onCodeChange(text) {
  S.code = text;
  S.dirty = true;
  $("dirty").classList.add("on");
  clearTimeout(previewTimer);
  previewTimer = setTimeout(requestPreview, 400);
  renderControls();
}
function requestPreview() {
  S.seq += 1;
  send({ cmd: "preview_code", code: S.code, seq: S.seq });
}
function onPreview(m) {
  if (m.seq != null && m.seq !== S.seq) return;
  S.previewErr = null;
  editor && editor.clearError();
  const caps = m.capsules || [];
  if (S.preview && S.preview.capsules.length === caps.length) { S.previewPrev = S.preview; S.previewT = performance.now(); }
  else S.previewPrev = null;
  S.preview = m;
  let top = 0;
  for (const c of caps) top = Math.max(top, c[1] + c[4], c[3] + c[4]);
  let x0 = Infinity, x1 = -Infinity;
  for (const c of caps) { x0 = Math.min(x0, c[0] - c[4], c[2] - c[4]); x1 = Math.max(x1, c[0] + c[4], c[2] + c[4]); }
  S.robotH = Math.max(0.8, top);
  S.robotW = caps.length ? x1 - x0 : 1;
  $("meta").textContent = `${m.bodies} bodies, ${m.joints} motors`;
  if (typeof postStatus === "function") postStatus();
  setStatusMsg(`${m.name || "robot"} · ${m.bodies} bodies · ${m.joints} joints · preview ${m.ms ?? "–"} ms`, false);
  renderStageText();
}
function onError(m) {
  if (m.seq != null && m.seq !== S.seq) return;
  const isPreview = m.phase === "preview";
  if (m.source === "code" || m.line != null) {
    const msg = m.message + (m.line ? "" : "");
    if (m.line && editor) editor.setError(m.line, msg);
    S.previewErr = m;
    setStatusMsg(m.line ? `line ${m.line}: ${m.message}` : m.message, true);
    renderStageText();
    if (!isPreview) { /* console line arrives via server log */ }
  } else if (!isPreview) {
    log(m.message, "error");
  }
}
function setStatusMsg(text, err) {
  $("edMsg").textContent = text;
  $("edStatus").classList.toggle("err", !!err);
}

// ======================================================================
// Training stream
// ======================================================================
function onStatus(m) {
  const prev = S.state;
  S.state = m.state || "idle";
  if (S.state === "training" && prev !== "training" && m.env === "playground") {
    S.runStart = performance.now();
  }
  if (S.state === "idle" && prev === "training") {
    log(`stopped after ${mmss(S.trainSeconds)} · ${fmtInt(S.iter)} iterations`, "info");
    S.sps = 0;
  }
  renderControls(); renderStageText(); renderFoot(); renderHeadline(); postStatus();
}
function onFrame(m) {
  const R = m.R | 0, K = m.K | 0;
  const caps = Float32Array.from(m.capsules || []);
  const dist = m.dist ? Float32Array.from(m.dist) : new Float32Array(R);
  const f = { R, K, caps, dist, best: m.best_index ?? 0 };
  if (f.best < 0 || f.best >= R) f.best = 0;
  S.interp.push(f);
  if (m.robot_name && m.robot_name !== S.robotName) { S.robotName = m.robot_name; renderStageText(); }
  if (!S.hasFrame) { S.hasFrame = true; renderStageText(); }
}
function onMetrics(m) {
  if (m.iter != null) S.iter = m.iter;
  if (m.sps != null) S.sps = m.sps;
  if (m.env_steps != null) S.envSteps = m.env_steps;
  const ts = m.train_seconds ?? (performance.now() - S.runStart) / 1000;
  S.trainSeconds = ts;
  const v = m.human_value;
  if (v != null && isFinite(v)) {
    S.humanValue = v; S.humanBest = m.human_best; S.humanSource = m.human_source || "episodes";
    S.unit = m.human_unit || "m";
    const last = S.hist[S.hist.length - 1];
    if (last && ts < last[0] - 1) S.hist = [];
    S.hist.push([ts, v]);
    if (S.hist.length > 2000) S.hist = S.hist.filter((_, i) => i % 2 === 0);
  }
  if (ts - S.lastProgLog >= 5 || S.lastProgLog < 0) {
    S.lastProgLog = ts;
    const parts = [`iter <b>${fmtInt(S.iter)}</b>`];
    if (v != null && isFinite(v)) parts.push(m.human_source === "live" ? `avg so far <b>${fmtM(v)}</b> (none finished)` : `avg/finished ep <b>${fmtM(v)}</b>`);
    if (m.human_best != null) parts.push(`best ${fmtM(m.human_best)}`);
    if (m.sps) parts.push(`training (sim + PPO) ${fmtSI(m.sps)} steps/s`);
    log(parts.join(" · "), "prog");
  }
  renderHeadline(); renderFoot(); renderXp();
}
function renderXp() {
  const t = S.hasFrame && xpText(S.envSteps, S.ctrlDt, S.trainSeconds, S.robotName || "robot");
  const el = $("stageLabel");
  if (t) el.innerHTML = t; else el.innerHTML = "";
}
function onReplay(m) {
  if (S.ranCode == null && S.state !== "training") return; // stale replays of a finished run
  const idx = S.replays.findIndex((r) => r.label === m.label);
  const rep = { ...m };
  if (m.dt) S.ctrlDt = m.dt / 2;
  if (m.label === "now") S.replays = S.replays.filter((r) => r.label !== "now");
  if (idx >= 0) S.replays[idx] = rep; else S.replays.push(rep);
  if (m.label !== "now" && idx < 0 && S.hasFrame) S.avgMarks.push({ label: m.label, x: S.liveMean || 0 });
  S.replays.sort((a, b) => (a.train_seconds ?? 0) - (b.train_seconds ?? 0));
  if (m.label !== "0:00" && m.label !== "now") log(`generation ${m.label}: ${m.result_text}`, "info");
  buildGens();
}

// ======================================================================
// Controls
// ======================================================================
function run() {
  if (!editor) return;
  const code = editor.get();
  S.code = code; S.ranCode = code; S.runClock = performance.now();
  S.interp.clear(); S.hasFrame = false;
  S.hist = []; S.replays = []; S.avgMarks = []; S.humanValue = null; S.humanBest = null; S.record = 0;
  S.iter = 0; S.sps = 0; S.simSps = null; S.trainSeconds = 0; S.lastProgLog = -99; S.envSteps = 0; S.trail = null; S.dens = null;
  S.cam.range = S.cam.targetRange = 6;
  editor.clearError();
  buildGens(); renderHeadline();
  logRule();
  log(`run ${S.preset}.py${S.dirty ? " (edited)" : ""} · ${fmtInt(S.numEnvs)} copies`, "info");
  S.state = "building";
  renderControls(); renderStageText(); postStatus();
  send({ cmd: "run_code", code, num_envs: S.numEnvs, render_envs: 256 });
}
function stop() { send({ cmd: "stop" }); }

$("run").addEventListener("click", () => {
  if (S.state === "training" && S.ranCode === S.code) stop(); else run();
});
$("count").addEventListener("change", (e) => { S.numEnvs = +e.target.value; renderStageText(); renderFoot(); });
$("preset").addEventListener("change", (e) => loadPreset(e.target.value));
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !(e.target.closest && e.target.closest(".editor"))) { e.preventDefault(); run(); }
});

async function loadPreset(name) {
  S.preset = name;
  $("preset").value = name;
  $("jumpFile").textContent = `${name}.py`;
  let text;
  try {
    const r = await fetch(`/presets/${name}`);
    if (!r.ok) throw new Error(r.status);
    text = await r.text();
  } catch (e) {
    log(`could not load preset ${name}: ${e}`, "error");
    return;
  }
  editor.set(text);
  editor.clearError();
  S.code = text; S.dirty = false;
  $("dirty").classList.remove("on");
  const u = new URL(location.href); u.searchParams.set("robot", name); history.replaceState(null, "", u);
  clearTimeout(previewTimer);
  requestPreview();
  renderControls();
}

function renderControls() {
  const b = $("run");
  const busy = S.state === "building" || S.state === "measuring" || S.state === "starting";
  const training = S.state === "training";
  const stale = training && S.ranCode !== S.code;
  b.classList.toggle("busy", busy);
  b.classList.toggle("stop", training && !stale);
  b.querySelector(".run-label").textContent = busy ? "Building…" : training ? (stale ? "Rerun" : "Stop") : "Run";
  b.querySelector("kbd").style.display = training && !stale ? "none" : "";
}
function renderConn() { const c = $("conn"); c.textContent = ""; c.classList.toggle("on", S.connected); c.title = S.connected ? "connected" : "offline"; }

function renderStageText() {
  const training = S.hasFrame;
  renderXp();
  const p = S.preview;
  if (training) {
    const R = S.interp.frame ? S.interp.frame.R : 256;
    $("stageNote").textContent = `${fmtInt(R)} of ${fmtInt(S.numEnvs)}${S.robotName ? " " + S.robotName : ""} copies shown, overlaid · each from its own start`;
  } else {
    $("stageNote").textContent = p ? `${p.bodies} bodies · ${p.joints} joints · ${p.joints} motors` : "";
  }
  let hint = "";
  if (S.state === "building") hint = `Building…`;
  else if (S.state === "measuring") hint = `Measuring speed…`;
  else if (!training && S.previewErr) hint = `Error on line ${S.previewErr.line ?? "?"}`;
  else if (training && S.previewErr) hint = `Error on line ${S.previewErr.line ?? "?"}`;
  $("stageHint").innerHTML = hint;
}
function renderHeadline() {
  const v = S.humanValue;
  $("hlNum").textContent = v == null ? "–" : v.toFixed(v < 10 ? 2 : 1);
  $("hlNum").parentElement.classList.toggle("empty", v == null);
  $("hlUnit").textContent = S.unit;
  $("hlLabel").textContent = "average distance";
  let sub = "press Run to train";
  if (S.hist.length > 1 && S.ranCode == null) sub = `joined a run in progress at ${mmss(S.hist[0][0])}${S.humanBest != null ? ` · best ${fmtM(S.humanBest)}` : ""}`;
  else if (S.hist.length > 1) sub = `was ${fmtM(S.hist[0][1])} at ${mmss(S.hist[0][0])}${S.humanBest != null ? ` · best ${fmtM(S.humanBest)}` : ""}`;
  else if (S.state === "training" || S.state === "starting") sub = "first episodes finishing…";
  $("hlSub").textContent = sub;
}
function renderFoot() {
  const parts = [`<b>${fmtInt(S.numEnvs)}</b> copies in parallel`, `${S.machine} (MLX)`,
    S.simSps ? `sim only <b>${fmtSI(S.simSps)}</b> steps/s` : null,
    S.sps ? `training (sim + PPO) <b>${fmtSI(S.sps)}</b> steps/s` : null,
    S.iter ? `PPO iteration <b>${fmtInt(S.iter)}</b>` : "press Run to train"].filter(Boolean);
  if (!S.connected) parts.push("server offline");
  $("foot").innerHTML = parts.map((x) => `<span>${x}</span>`).join('<span class="sep">·</span>');
}

// ======================================================================
// Console
// ======================================================================
const consoleEl = $("console");
const t0 = performance.now();
function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]); }
function log(line, level, ts) {
  const el = document.createElement("div");
  el.className = `ln ${level}`;
  const secs = (performance.now() - (S.runClock || t0)) / 1000;
  const mark = { ok: "✓", error: "×", warn: "!", prog: "·", info: "›" }[level] || "›";
  let text = level === "prog" ? line : esc(line);
  // make "line N" clickable: jumps the editor to that line
  text = text.replace(/\(line (\d+)\)|line (\d+)/g, (mm, a, b) => `<a data-line="${a || b}">${mm}</a>`);
  el.innerHTML = `<span class="tx">${text}</span>`;
  consoleEl.appendChild(el);
  while (consoleEl.children.length > 400) consoleEl.firstChild.remove();
  consoleEl.scrollTop = consoleEl.scrollHeight;
}
function logRule() {
  if (!consoleEl.children.length) return;
  const el = document.createElement("div");
  el.style.cssText = `border-top:1px solid ${C.rule};margin:6px 0`;
  consoleEl.appendChild(el);
}
$("consoleClear").addEventListener("click", () => { consoleEl.innerHTML = ""; });
consoleEl.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-line]");
  if (!a || !editor || !editor.view) return;
  const doc = editor.view.state.doc, n = clamp(+a.dataset.line, 1, doc.lines);
  editor.view.dispatch({ selection: { anchor: doc.line(n).from }, scrollIntoView: true });
  editor.focus();
});

// ======================================================================
// Stage drawing
// ======================================================================
const stageCv = $("stage"), chartCv = $("chart");

function previewCaps(now) {
  const p = S.preview; if (!p) return null;
  const cur = p.capsules;
  if (!S.previewPrev) return cur;
  const t = clamp((now - S.previewT) / 220, 0, 1), e = 1 - Math.pow(1 - t, 3);
  if (t >= 1) { S.previewPrev = null; return cur; }
  return cur.map((c, i) => c.map((v, j) => S.previewPrev.capsules[i][j] + (v - S.previewPrev.capsules[i][j]) * e));
}

function dimLine(ctx, x1, y1, x2, y2, text, vertical) {
  ctx.strokeStyle = C.muted; ctx.fillStyle = C.muted; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  const tick = (x, y) => { ctx.beginPath(); if (vertical) { ctx.moveTo(x - 5, y); ctx.lineTo(x + 5, y); } else { ctx.moveTo(x, y - 5); ctx.lineTo(x, y + 5); } ctx.stroke(); };
  tick(x1, y1); tick(x2, y2);
  mono(ctx, 10.5, C.muted);
  if (vertical) { ctx.save(); ctx.translate(x1 + 14, (y1 + y2) / 2); ctx.rotate(-Math.PI / 2); tracked(ctx, text, 0, 0, "center", 0.6); ctx.restore(); }
  else {
    const mx = (x1 + x2) / 2, tw = ctx.measureText(text).width + 12;
    ctx.fillStyle = C.paper; ctx.fillRect(mx - tw / 2, y1 - 7, tw, 14);
    mono(ctx, 10.5, C.muted); tracked(ctx, text, mx, y1 + 3.5, "center", 0.6);
  }
}

function drawPreview(ctx, w, h, now) {
  const caps = previewCaps(now);
  const pad = 28;
  const groundY = Math.round(h - 78) + 0.5;
  ctx.strokeStyle = C.ground; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, groundY); ctx.lineTo(w - pad, groundY); ctx.stroke();
  if (!caps || !caps.length) {
    mono(ctx, 13, C.faint); tracked(ctx, S.connected ? "" : "Server offline", w / 2, groundY - 80, "center");
    return;
  }
  let x0 = Infinity, x1 = -Infinity, top = 0, bot = Infinity;
  for (const c of caps) {
    x0 = Math.min(x0, c[0] - c[4], c[2] - c[4]); x1 = Math.max(x1, c[0] + c[4], c[2] + c[4]);
    top = Math.max(top, c[1] + c[4], c[3] + c[4]); bot = Math.min(bot, c[1] - c[4], c[3] - c[4]);
  }
  const bw = Math.max(0.2, x1 - x0), bh = Math.max(0.2, top);
  const availH = groundY - 96, availW = (w - pad * 2) * 0.62;
  const s = Math.min(availH / bh, availW / bw, 520);
  const cx = w / 2 - ((x0 + x1) / 2) * s;
  const X = (x) => cx + x * s, Y = (y) => groundY - y * s;
  const dim = !!S.previewErr;

  ctx.globalAlpha = dim ? 0.35 : 1;
  ctx.lineCap = "round"; ctx.strokeStyle = C.ink;
  for (const c of caps) {
    ctx.lineWidth = Math.max(3, c[4] * 2 * s * 0.6);
    ctx.beginPath(); ctx.moveTo(X(c[0]), Y(c[1])); ctx.lineTo(X(c[2]), Y(c[3])); ctx.stroke();
  }
  const jp = (S.preview && S.preview.joint_points) || [];
  ctx.fillStyle = C.paper;
  for (const [jx, jy] of jp) { ctx.beginPath(); ctx.arc(X(jx), Y(jy), 2.5, 0, Math.PI * 2); ctx.fill(); }
  ctx.globalAlpha = 1;
}

function drawStage(now, dt) {
  const { ctx, w, h } = fit(stageCv);
  ctx.clearRect(0, 0, w, h);
  // full-bleed canvas behind the glass panels; the track lives right of the editor, above the card
  const RX = 0, RH = h;
  ctx.save(); ctx.translate(RX, 0);
  if (!S.hasFrame) { drawPreview(ctx, w - RX, RH, now); ctx.restore(); return; }
  const f = S.interp.at(now);
  // many small robots spread over a long runway: lighter strokes, smaller figures than the viewer
  const alpha = clamp(0.5 / Math.sqrt(f.R) * Math.sqrt(6 / Math.max(6, f.K)) * 1.6, 0.05, 0.2);
  T.drawLoco(ctx, w - RX, RH, 28, f, S.interp.buf, S.interp.dist, S, dt, { robotH: S.robotH, minRange: 6, alpha: alpha * 1.2, figScale: 0.36, groundFrom: 18 });
  ctx.restore();
}

// ======================================================================
// Generations
// ======================================================================
let genCells = [], laneT0 = performance.now();
const GEN_SLOTS = 4;
function pickReplays() { const r = S.replays; return r.length <= GEN_SLOTS ? r : [r[0], ...r.slice(r.length - (GEN_SLOTS - 1))]; }
function buildGens() {
  const row = $("gens"); row.innerHTML = ""; genCells = [];
  const shown = pickReplays();
  const lastT = shown.length ? (shown[shown.length - 1].train_seconds ?? 0) : -1;
  const upcoming = REPLAY_AT.filter((t) => t > lastT + 0.5);
  laneT0 = performance.now();
  const longest = shown.reduce((a, r) => ((r.frames || []).length * (r.dt || 0.04) > (a.frames || []).length * (a.dt || 0.04) ? r : a), {});
  void longest;
  $("gensNote").textContent = "one robot, fresh start, no exploration noise";
  for (let i = 0; i < GEN_SLOTS; i++) {
    const rep = shown[i];
    const cell = document.createElement("div");
    cell.className = "lane";
    if (rep) {
      if (i === shown.length - 1) cell.classList.add("newest");
      const res = rep.result != null ? `walked ${fmtM(rep.result)}` : "";
      cell.innerHTML = `<span class="lt mono">${rep.label === "now" ? "NOW" : rep.label}</span><div class="lbox"><canvas></canvas></div><div class="lr"><b></b><span></span></div>`;
      cell.querySelector(".lr b").textContent = res;
      cell.querySelector(".lr span").textContent = "";
      cell.title = rep.result_text || "";
      genCells.push({ rep, cv: cell.querySelector("canvas"), isNew: i === shown.length - 1 });
    } else {
      const t = upcoming[i - shown.length];
      cell.classList.add("empty");
      cell.innerHTML = `<span class="lt mono">${t != null ? mmss(t) : "–"}</span><div class="lbox"><span class="mono">${t != null ? (t === 0 ? "untrained policy" : `after ${mmss(t)} of training`) : ""}</span></div><div class="lr"></div>`;
    }
    row.appendChild(cell);
  }
}
// length of a recorded replay in simulated seconds (frames are captured every rep.dt)
function replaySecs(rep) {
  const n = (rep.frames || []).length;
  const sec = n * (rep.dt || 0.04);
  return sec >= 2 ? String(Math.round(sec)) : sec.toFixed(1);
}
function drawGens(now) {
  if (!genCells.length) return;
  let laneMax = 2, dur = 1;
  for (const g of genCells) {
    const fr = g.rep.frames || [];
    for (const f of fr) laneMax = Math.max(laneMax, f[0] + S.robotW * 0.6);
    dur = Math.max(dur, fr.length * (g.rep.dt || 0.04) + 1.2);
  }
  for (const g of genCells) T.drawLane(g.cv, g.rep, now, g.isNew, { robotH: S.robotH * 1.15, laneMax, t0: laneT0, dur });
}

// ======================================================================
// Loop + boot
// ======================================================================
let lastNow = performance.now(), chartTick = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - lastNow) / 1000); lastNow = now;
  drawStage(now, dt);
  if (now - chartTick > 100) { T.drawChart(chartCv, S, { tMin: 60, emptyText: " " }); chartTick = now; }
  drawGens(now);
  requestAnimationFrame(loop);
}
window.addEventListener("resize", () => { chartTick = 0; });
// Redraw canvases at their new size immediately (fit() re-reads clientWidth/devicePixelRatio every draw).
if (window.ResizeObserver) {
  const ro = new ResizeObserver(() => { chartTick = 0; const n = performance.now(); drawStage(n, 0); T.drawChart(chartCv, S, { tMin: 60, emptyText: " " }); drawGens(n); });
  for (const el of [stageCv, chartCv, $("gens")]) ro.observe(el);
}

// ======================================================================
// Shell contract (XCODE.md): status out, commands in
// ======================================================================
const EMBED = params.get("embed") === "1";
function shellState() { return S.state === "training" ? "training" : (S.state === "building" || S.state === "measuring" || S.state === "starting") ? "building" : "idle"; }
function postStatus() {
  if (!EMBED || window.parent === window) return;
  const xp = S.hasFrame && S.envSteps && S.ctrlDt ? S.envSteps * S.ctrlDt / 3600 : 0;
  const v = !xp ? "" : xp >= 48 ? `${(xp / 24).toFixed(xp >= 240 ? 0 : 1)} days` : xp >= 1 ? `${xp.toFixed(xp >= 10 ? 0 : 1)} hours` : `${Math.round(xp * 60)} minutes`;
  const p = S.preview;
  try {
    window.parent.postMessage({
      type: "macgym:status", view: "playground", state: shellState(), robot: S.preset,
      detail: p ? `${p.bodies} bodies, ${p.joints} motors` : "", copies: S.numEnvs,
      headline: v ? `${v} of practice` : "", sub: v ? `in ${fmtWall(S.trainSeconds)}` : "",
      best: S.record > 0 ? fmtM(S.record) : "",
    }, "*");
  } catch (e) { /* parent gone */ }
}
window.addEventListener("message", (e) => {
  const m = e.data;
  if (!m || m.type !== "macgym:cmd") return;
  if (m.cmd === "run") run();
  else if (m.cmd === "stop") stop();
  else if (m.cmd === "setCopies" && m.value) {
    const sel = $("count"), v = String(m.value);
    if (![...sel.options].some((o) => o.value === v)) sel.add(new Option(Number(v).toLocaleString("en-US"), v));
    sel.value = v; sel.dispatchEvent(new Event("change"));
  }
});
setInterval(() => { if (S.state === "training") postStatus(); }, 1000);
$("count").addEventListener("change", postStatus);
$("preset").addEventListener("change", () => setTimeout(postStatus, 300));

fetch("bench.json").then((r) => r.ok ? r.json() : null).then((j) => {
  const chip = j && /Apple M\d+( Pro| Max| Ultra)?/.exec(j.machine || "");
  if (chip) { S.machine = chip[0] + " GPU"; $("machine").textContent = `${chip[0]} · MLX · Metal`; }
  renderFoot();
}).catch(() => {});

async function boot() {
  log("MacGym playground · edit the robot, then press ⌘↵ to train it", "info");
  const host = $("editor");
  try {
    editor = await makeCodeMirror(host, onCodeChange, run);
  } catch (e) {
    console.warn("CodeMirror unavailable, using textarea", e);
    host.innerHTML = "";
    editor = makeTextarea(host, onCodeChange, run);
  }
  buildGens(); renderHeadline(); renderControls(); renderStageText(); renderFoot(); renderConn();
  await loadPreset(S.preset);
  editor.focus();
}
(document.fonts ? document.fonts.ready : Promise.resolve()).then(() => requestAnimationFrame(loop));
boot();
connect();
window.__pg = { S, run, loadPreset, get editor() { return editor; } };
