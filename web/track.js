// MacGym shared canvas drawing: overlaid track, progress chart, replay cells.
// Used by the viewer (app.js) and the playground (playground.js). Classic script: exposes window.MacGymTrack.
(function () {
  "use strict";
  // Colours and fonts come from the Probe0 design tokens (web/probe0/tokens.css), resolved at runtime.
  // Custom properties built with light-dark() are not resolved by getPropertyValue, so each colour token
  // is resolved by painting it onto a probe element and reading the used colour back.
  const C = { paper: "#1e1e1e", ink: "#ffffff", rule: "rgba(255,255,255,0.08)", muted: "rgba(255,255,255,0.55)", accent: "#ffffff", old: "rgba(255,255,255,0.25)", faint: "rgba(255,255,255,0.25)", ground: "rgba(255,255,255,0.1)" };
  let SANS = '"Geist", -apple-system, system-ui, sans-serif';
  let MONO = '"Geist Mono", ui-monospace, Menlo, monospace';
  const TOKEN_MAP = { paper: "--window-bg", ink: "--label-primary", rule: "--chart-grid", muted: "--label-secondary", accent: "--accent", old: "--label-tertiary", faint: "--label-tertiary", ground: "--separator" };
  function refreshTokens() {
    const root = document.documentElement, cs = getComputedStyle(root);
    if (!cs.getPropertyValue("--label-primary")) return;
    const probe = document.createElement("span");
    probe.style.cssText = "position:absolute;visibility:hidden;pointer-events:none";
    (document.body || root).appendChild(probe);
    for (const [k, v] of Object.entries(TOKEN_MAP)) { probe.style.color = `var(${v})`; const c = getComputedStyle(probe).color; if (c) C[k] = c; }
    probe.remove();
    const fu = cs.getPropertyValue("--font-ui").trim(), fm = cs.getPropertyValue("--font-mono").trim();
    if (fu) SANS = fu; if (fm) MONO = fm;
  }
  refreshTokens();
  if (document.readyState !== "complete") window.addEventListener("load", refreshTokens);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  // ---------------- primitives ----------------
  function fit(cv) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
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
  function strokeOne(ctx, buf, o, K, X, Y, s, color, joints, widthScale) {
    ctx.lineCap = "round"; ctx.strokeStyle = color;
    const ws = widthScale || 0.6;
    for (let k = 0; k < K; k++) {
      const q = o + k * 5;
      ctx.lineWidth = clamp(buf[q + 4] * 2 * s * ws, 1.6, ws > 0.6 ? 14 : 4.5);
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
  function mono(ctx, size, color) { ctx.font = `400 ${size + 1}px ${SANS}`; ctx.fillStyle = color; } // canvas labels: Geist, not mono
  function tracked(ctx, text, x, y, align, spacing = 0.8) {
    ctx.letterSpacing = "0px";
    ctx.textAlign = align || "left"; ctx.fillText(text, x, y); ctx.letterSpacing = "0px";
  }
  function dashed(ctx, x, y1, y2, color, dash) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash(dash || [4, 4]);
    ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, y1); ctx.lineTo(Math.round(x) + 0.5, y2); ctx.stroke(); ctx.restore();
  }
  function label(ctx, text, x, y, color, align, weight, size) {
    ctx.font = `${weight || 600} ${size || 14}px ${SANS}`; ctx.fillStyle = color; ctx.textAlign = align || "left"; ctx.fillText(text, x, y);
  }
  function median(a, n) {
    if (!n) return 0;
    const b = Array.prototype.slice.call(a, 0, n).sort((x, y) => x - y);
    return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2;
  }
  // y-axis title per display unit (spelled out so "M" can never be misread as minutes)
  const Y_TITLE = { m: "METRES", cm: "CENTIMETRES", s: "SECONDS", "": "REWARD" };
  function niceStep(range, target) {
    const raw = range / target, p = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / p;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * p;
  }

  // ---------------- frame interpolation ----------------
  // Holds the last two frames and produces a smoothly interpolated capsule buffer per render.
  function Interp() {
    this.frame = null; this.prev = null; this.frameT = 0; this.frameDt = 33;
    this.buf = new Float32Array(0); this.dist = new Float32Array(0);
  }
  Interp.prototype.push = function (f) {
    const now = performance.now();
    if (this.frame && this.frame.R === f.R && this.frame.K === f.K) {
      this.prev = this.frame;
      this.frameDt = clamp(this.frameDt * 0.85 + (now - this.frameT) * 0.15, 16, 150);
    } else this.prev = null;
    this.frame = f; this.frameT = now;
  };
  Interp.prototype.clear = function () { this.frame = null; this.prev = null; };
  Interp.prototype.at = function (now) {
    const f = this.frame; if (!f) return null;
    const n = f.caps.length;
    if (this.buf.length !== n) this.buf = new Float32Array(n);
    if (this.dist.length !== f.R) this.dist = new Float32Array(f.R);
    const p = this.prev, buf = this.buf, dist = this.dist;
    const t = clamp((now - this.frameT) / this.frameDt, 0, 1);
    if (!p) { buf.set(f.caps); if (f.dist) dist.set(f.dist); return f; }
    const per = f.K * 5;
    for (let i = 0; i < f.R; i++) {
      const o = i * per;
      const jump = Math.abs(f.caps[o] - p.caps[o]) > 0.6 || Math.abs(f.caps[o + 1] - p.caps[o + 1]) > 0.6;
      for (let j = 0; j < per; j++) buf[o + j] = jump ? f.caps[o + j] : p.caps[o + j] + (f.caps[o + j] - p.caps[o + j]) * t;
      if (f.dist) dist[i] = jump || !p.dist ? f.dist[i] : p.dist[i] + (f.dist[i] - p.dist[i]) * t;
    }
    return f;
  };

  // ---------------- locomotion track ----------------
  // st: mutable view state { cam:{range,targetRange}, record, avgMarks:[{label,x}], liveMean }.
  // opts: { robotH (m, figure height used for scale), unitVerb }
  function drawLoco(ctx, w, h, pad, f, buf, dist, st, dt, opts) {
    opts = opts || {};
    const groundY = Math.round(h - 64) + 0.5;
    let mean = 0, max = 0, bi = f ? f.best : -1;
    if (f && f.dist) {
      // "pack median now": where the middle robot of the shown pack is right now (in-progress episodes).
      // Deliberately a median, not a mean, so it never reads as the headline "average per finished episode".
      for (let i = 0; i < f.R; i++) if (dist[i] > max) max = dist[i];
      mean = median(dist, f.R);
      if (bi >= 0) max = Math.max(max, dist[bi]);
      st.record = Math.max(st.record, max);
    }
    // camera: 0 m at left; furthest robot + 15% fits. Grows quickly, shrinks slowly.
    st.cam.targetRange = Math.max(opts.minRange || 8, max * 1.15 + 1);
    const k = st.cam.targetRange > st.cam.range ? 2.2 : 0.35;
    st.cam.range += (st.cam.targetRange - st.cam.range) * (1 - Math.exp(-k * dt));
    const usableW = w - pad * 2;
    const topSpace = 92; // room for marker labels
    const robotH = opts.robotH || 1.65;
    const s = Math.min(usableW / (st.cam.range + 0.7), (groundY - topSpace) / robotH * 0.62);
    const x0 = pad + 0.55 * s;
    const X = (x) => x0 + x * s, Y = (y) => groundY - y * s;
    const visibleM = (w - pad - x0) / s;

    ctx.strokeStyle = C.ground; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(opts.groundFrom ?? pad, groundY); ctx.lineTo(w - pad, groundY); ctx.stroke();
    const labelEvery = s > 60 ? 1 : s > 22 ? 5 : s > 9 ? 10 : 20;
    const minorEvery = s > 6 ? 1 : 5;
    for (let m = 0; m <= visibleM + 1e-6; m += minorEvery) {
      const x = Math.round(X(m)) + 0.5;
      const major = m % labelEvery === 0;
      ctx.strokeStyle = major ? C.ground : C.rule;
      ctx.beginPath(); ctx.moveTo(x, groundY); ctx.lineTo(x, groundY + (major ? 9 : 4)); ctx.stroke();
      if (major) { mono(ctx, 11, C.faint); tracked(ctx, `${m} m`, x, groundY + 25, "center", 0.4); }
    }

    if (!f) return;
    // Figures keep a legible size even when the runway is 100 m long: scale each body about its own torso.
    const sb = Math.max(s, (groundY - topSpace) / robotH * (opts.figScale || 0.42));
    if (sb > s * 1.01) {
      const g = sb / s, per = f.K * 5;
      for (let i = 0; i < f.R; i++) {
        const o = i * per, ax = buf[o];
        for (let q = o; q < o + per; q += 5) {
          buf[q] = ax + (buf[q] - ax) * g; buf[q + 1] *= g;
          buf[q + 2] = ax + (buf[q + 2] - ax) * g; buf[q + 3] *= g; buf[q + 4] *= g;
        }
      }
    }
    const mTop = Math.max(58, groundY - robotH * sb - 64);

    // --- where the whole pack is: smoothed distance histogram over the runway (real per-robot dist) ---
    if (false && f.dist) {
      const bandBot = Math.round(mTop - 34), bandTop = Math.max(52, bandBot - 54);
      if (bandBot - bandTop >= 22) {
        const nb = Math.max(24, Math.min(160, Math.floor((w - pad - x0) / 7)));
        const span = visibleM / nb;
        if (!st.dens || st.dens.length !== nb) st.dens = new Float32Array(nb);
        const raw = new Float32Array(nb);
        for (let i = 0; i < f.R; i++) {
          const b = Math.floor(clamp(dist[i], 0, visibleM - 1e-6) / span);
          raw[b] += 1;
        }
        let peak = 1e-6;
        for (let b = 0; b < nb; b++) {
          const v = (raw[b] * 2 + (raw[b - 1] || 0) + (raw[b + 1] || 0)) / 4;
          st.dens[b] += (v - st.dens[b]) * Math.min(1, dt * 6);
          peak = Math.max(peak, st.dens[b]);
        }
        const hh = bandBot - bandTop;
        ctx.beginPath(); ctx.moveTo(X(0), bandBot);
        for (let b = 0; b < nb; b++) ctx.lineTo(X((b + 0.5) * span), bandBot - (st.dens[b] / peak) * hh);
        ctx.lineTo(X(visibleM), bandBot); ctx.closePath();
        ctx.fillStyle = C.rule; ctx.fill();
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1.1; ctx.lineJoin = "round"; ctx.stroke();
        ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad, bandBot + 0.5); ctx.lineTo(w - pad, bandBot + 0.5); ctx.stroke();
        ctx.font = `500 12px ${SANS}`; ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText(`where all ${f.R} shown robots are right now`, pad, bandTop - 6);
      }
    }

    // --- chronophotograph of the leading robot: its last ~1.5 s, fading ---
    if (bi >= 0) {
      const per = f.K * 5, o = bi * per, bd = f.dist ? dist[bi] : 0;
      const tr = st.trail || (st.trail = { bi: -1, last: 0, poses: [], d: 0 });
      const nowMs = performance.now();
      if (tr.bi !== bi || bd < tr.d - 0.5) { tr.poses = []; tr.bi = bi; }
      tr.d = bd;
      if (nowMs - tr.last > 200) { tr.poses.push(buf.slice(o, o + per)); tr.last = nowMs; if (tr.poses.length > 7) tr.poses.shift(); }
      tr.poses.forEach((pz, i) => {
        ctx.globalAlpha = 0.07 + 0.22 * (i / tr.poses.length);
        strokeOne(ctx, pz, 0, f.K, X, Y, s, C.accent, false, 0.45);
      });
      ctx.globalAlpha = 1;
    }

    strokeRobots(ctx, f, buf, X, Y, s, opts.alpha || (f.R > 128 ? 0.13 : 0.22), C.ink, bi);
    for (const mk of []) {
      const x = Math.round(X(mk.x)) + 0.5;
      ctx.strokeStyle = C.old; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, groundY + 34); ctx.lineTo(x, groundY + 46); ctx.stroke();
    }
    [].forEach((mk, i) => {
      const next = null;
      if (next && X(next.x) - X(mk.x) < 34) return;
      mono(ctx, 9.5, C.old); tracked(ctx, mk.label, X(mk.x) + 4, groundY + 46, "left", 0.2);
    });
    st.liveMean = mean;
    const bestNow = f.dist && bi >= 0 ? dist[bi] : max;
    if (false && st.record > 0.3 && st.record - bestNow > 0.4) {
      const xr = X(st.record);
      dashed(ctx, xr, mTop + 30, groundY, C.old, [1, 3]);
      mono(ctx, 10.5, C.old); tracked(ctx, `RECORD ${st.record.toFixed(1)} M`, xr + 6, mTop + 40, "left", 0.6);
    }
    if (f.dist) {
      const xa = X(mean), xb = X(dist[bi] ?? max);
      dashed(ctx, xb, mTop - 14, groundY, C.accent, []);
      const avgTxt = `pack median now ${mean.toFixed(1)} m`, bestTxt = `best ${(dist[bi] ?? max).toFixed(1)} m`;
      ctx.font = `600 14px ${SANS}`;
      const aw = ctx.measureText(avgTxt).width;
      let close = Math.abs(xb - xa) < aw + 20;
      if (close && xa - 6 - aw < 4) close = false; // no room on the left: keep it on screen
      if (false) label(ctx, avgTxt, close ? xa - 6 : xa + 6, close ? mTop + 18 : (Math.abs(xb - xa) < aw + 20 ? mTop + 40 : mTop + 18), C.ink, close ? "right" : "left");
      ctx.font = `600 14px ${SANS}`;
      const bestFlip = xb + 6 + ctx.measureText(bestTxt).width > w - 4; // keep the label on screen
      label(ctx, bestTxt, bestFlip ? xb - 6 : xb + 6, mTop - 4, C.accent, bestFlip ? "right" : "left");
    }
    if (bi >= 0) {
      ctx.save(); ctx.lineCap = "round"; ctx.strokeStyle = C.paper;
      const o = bi * f.K * 5;
      for (let k = 0; k < f.K; k++) { const q = o + k * 5; ctx.lineWidth = clamp(buf[q + 4] * 2 * s * 0.6, 1.6, 4.5) + 2.5; ctx.beginPath(); ctx.moveTo(X(buf[q]), Y(buf[q + 1])); ctx.lineTo(X(buf[q + 2]), Y(buf[q + 3])); ctx.stroke(); }
      ctx.restore();
      strokeOne(ctx, buf, o, f.K, X, Y, s, C.accent, true);
    }
  }

  // ---------------- progress chart ----------------
  // st: { hist:[[t,v]], unit, replays, state }
  function drawChart(cv, st, opts) {
    opts = opts || {};
    const { ctx, w, h } = fit(cv);
    ctx.clearRect(0, 0, w, h);
    const L = opts.left || 44, R = 18, T = 26, B = 40;
    const pw = w - L - R, ph = h - T - B;
    const hist = st.hist;
    const tMin = opts.tMin || 60;
    // x-range snaps to 1:00, 2:00, 4:00, 8:00 … and auto-extends as training continues
    const tLast = hist.length ? hist[hist.length - 1][0] : 0;
    let tMax = tMin;
    while (tLast > tMax * 0.96) tMax *= 2;
    let vMin = 0, vMax = opts.vMin || 1;
    for (const [, v] of hist) { vMax = Math.max(vMax, v); vMin = Math.min(vMin, v); }
    vMax *= 1.12;
    const xs = (t) => L + (t / tMax) * pw, ys = (v) => T + ph - ((v - vMin) / (vMax - vMin)) * ph;
    const vStep = niceStep(vMax - vMin, opts.yTicks || 4);
    ctx.lineWidth = 1;
    for (let v = Math.ceil(vMin / vStep) * vStep; v <= vMax + 1e-9; v += vStep) {
      const y = Math.round(ys(v)) + 0.5;
      ctx.strokeStyle = Math.abs(v) < 1e-9 ? C.ground : C.rule;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
      const txt = vStep < 1 ? v.toFixed(1) : String(Math.round(v));
      mono(ctx, 10.5, C.muted); tracked(ctx, txt, L - 8, y + 3.5, "right", 0.2);
    }
    // y title sits above the plot, left-aligned, clear of every tick label
    const yTitle = opts.yTitle || Y_TITLE[st.unit || ""] || String(st.unit || "").toUpperCase();

    // tick spacing is chosen in pixels so labels never collide, even after hours of training
    const long = tMax >= 3600;
    const fmtT = long ? (s) => `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}` : mmss;
    const minPx = long ? 50 : 46;
    const tStep = [10, 15, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200, 14400, 28800].find((sp) => pw / (tMax / sp) >= minPx) || 28800;
    for (let t = 0; t <= tMax + 1e-6; t += tStep) {
      const x = Math.round(xs(t)) + 0.5;
      ctx.strokeStyle = C.ink;
      ctx.beginPath(); ctx.moveTo(x, T + ph); ctx.lineTo(x, T + ph + 4); ctx.stroke();
      mono(ctx, 10.5, C.muted); tracked(ctx, fmtT(t), x, T + ph + 18, "center", 0.2);
    }

    ctx.strokeStyle = C.ground; ctx.beginPath(); ctx.moveTo(L + 0.5, T); ctx.lineTo(L + 0.5, T + ph + 0.5); ctx.lineTo(L + pw, T + ph + 0.5); ctx.stroke();
    for (const r of st.replays || []) {
      if (r.train_seconds == null || r.label === "now") continue;
      const x = Math.round(xs(r.train_seconds)) + 0.5;
      ctx.strokeStyle = C.rule; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + ph); ctx.stroke(); ctx.setLineDash([]);
    }
    if (hist.length < 1) {
      ctx.font = `500 13px ${SANS}`; ctx.fillStyle = C.faint; ctx.textAlign = "center";
      ctx.fillText(opts.emptyText || (st.state === "idle" ? "The learning curve draws here once training starts" : "Waiting for the first robots to finish an episode"), L + pw / 2, T + ph / 2);
      return;
    }
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

  // ---------------- replay cell (one looping generation) ----------------
  // kind: "loco" | "arm" | "cart". opts: { robotH }
  function drawReplay(cv, rep, now, start, isNew, kind, opts) {
    opts = opts || {};
    const { ctx, w, h } = fit(cv);
    ctx.clearRect(0, 0, w, h);
    const frames = rep.frames || [];
    if (!frames.length) return;
    const K = rep.K || frames[0].length / 5;
    const dtr = rep.dt || 0.04;
    const dur = frames.length * dtr + 1.0;
    const t = Math.max(0, (now - start) / 1000) % dur;
    const fi = Math.min(frames.length - 1, Math.floor(t / dtr));
    const fr = frames[fi];
    const col = isNew ? C.accent : C.ink;
    if (kind === "loco") {
      const robotH = opts.robotH || 1.6;
      const s = Math.min((h - 44) / robotH, w / (opts.widthM || 2.6));
      // width-limited (small robots in narrow cells): sit the ground just below the middle
      const gy = Math.round(opts.center ? Math.min(h - 22, h * 0.5 + robotH * s * 0.5) : h - 22) + 0.5;
      const camX = fr[0];
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
    } else if (kind === "arm") {
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

  // ---------------- race lane (one generation per row, shared x scale) ----------------
  // All lanes share t0 and laneMax so generations race each other side by side.
  // opts: { robotH, laneMax (m), t0 (ms), dur (s, common loop length) }
  function drawLane(cv, rep, now, isNew, opts) {
    const { ctx, w, h } = fit(cv);
    ctx.clearRect(0, 0, w, h);
    laneBody(ctx, w, h, rep, now, isNew, opts);
  }
  // same lane, drawn into a sub-rectangle of an existing canvas (viewer idle state)
  function drawLaneRect(ctx, x, y, w, h, rep, now, isNew, opts) {
    ctx.save(); ctx.translate(x, y); laneBody(ctx, w, h, rep, now, isNew, opts); ctx.restore();
  }
  function laneBody(ctx, w, h, rep, now, isNew, opts) {
    const frames = rep.frames || [];
    if (!frames.length) return;
    const K = rep.K || frames[0].length / 5;
    const dtr = rep.dt || 0.04;
    const t = Math.max(0, (now - opts.t0) / 1000) % opts.dur;
    const fi = Math.min(frames.length - 1, Math.floor(t / dtr));
    const gy = Math.round(h - 7) + 0.5;
    const padL = 6, padR = 10;
    const laneMax = Math.max(2, opts.laneMax);
    const sx = (w - padL - padR) / (laneMax + 0.6);
    const sf = Math.max(sx, Math.min((gy - 4) / (opts.robotH || 1), 140)); // figure scale (legible at any lane length)
    const X0 = padL + 0.3 * sx;
    ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    const every = sx > 40 ? 1 : sx > 12 ? 5 : sx > 5 ? 10 : 25;
    ctx.strokeStyle = C.faint;
    for (let m = 0; m <= laneMax + 0.6; m += every) {
      const x = Math.round(X0 + m * sx) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, gy); ctx.lineTo(x, gy + 4); ctx.stroke();
    }
    // figure at its true distance, drawn about its own torso at figure scale
    const draw = (fr, color, alpha, joints) => {
      const ax = fr[0];
      const bx = X0 + ax * sx;
      const X = (x) => bx + (x - ax) * sf, Y = (y) => gy - y * sf;
      ctx.globalAlpha = alpha;
      strokeOne(ctx, fr, 0, K, X, Y, sf, color, joints);
      ctx.globalAlpha = 1;
    };
    // chronophotograph: faint poses once per simulated second
    const step = Math.max(1, Math.round(1 / dtr));
    for (let i = 0; i < fi; i += step) draw(frames[i], C.ink, 0.1, false);
    const fr = frames[fi];
    draw(fr, isNew ? C.accent : C.ink, 1, false);
    const xNow = X0 + Math.max(0, fr[0]) * sx;
    ctx.fillStyle = isNew ? C.accent : C.ink;
    ctx.fillRect(Math.round(xNow) - 0.5, gy - 1, 2, 3);
  }

  window.MacGymTrack = { C, get SANS() { return SANS; }, get MONO() { return MONO; }, refreshTokens, clamp, mmss, fit, strokeRobots, strokeOne, mono, tracked, dashed, label, niceStep, median, Interp, drawLoco, drawChart, drawReplay, drawLane, drawLaneRect };
})();
