// MacGym docs: syntax highlighting, copy buttons, on-this-page TOC, hero animation.
(function () {
  "use strict";

  // ---------- tiny Python / shell highlighter ----------
  const KW = new Set(("False None True and as assert async await break class continue def del elif else except " +
    "finally for from global if import in is lambda nonlocal not or pass raise return try while with yield").split(" "));
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function hlPython(src) {
    const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[rbf]?"(?:\\.|[^"\\\n])*"|[rbf]?'(?:\\.|[^'\\\n])*')|(\b\d[\d_]*(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|([^\sA-Za-z0-9_])/g;
    let out = "", m;
    while ((m = re.exec(src))) {
      const [tok, com, str, num, id] = m;
      if (com) out += `<span class="t-com">${esc(com)}</span>`;
      else if (str) out += `<span class="t-str">${esc(str)}</span>`;
      else if (num) out += `<span class="t-num">${num}</span>`;
      else if (id) {
        const next = src.slice(re.lastIndex).match(/^\s*\(/);
        if (id === "mg" || id === "macgym") out += `<span class="t-mg">${id}</span>`;
        else if (KW.has(id)) out += `<span class="t-kw">${id}</span>`;
        else if (id === "self") out += `<span class="t-self">${id}</span>`;
        else if (next) out += `<span class="t-fn">${id}</span>`;
        else out += id;
      } else out += esc(tok);
    }
    return out;
  }

  function hlShell(src) {
    return src.split("\n").map((line) => {
      if (/^\s*#/.test(line)) return `<span class="t-com">${esc(line)}</span>`;
      const m = line.match(/^(\$ )(.*)$/);
      let body = m ? m[2] : line;
      const pre = m ? `<span class="t-prompt">$ </span>` : "";
      if (!m) return `<span class="t-com">${esc(line)}</span>`;
      body = esc(body)
        .replace(/(^|\s)(--?[a-z][\w-]*)/g, '$1<span class="t-flag">$2</span>')
        .replace(/^(macgym|pip|python3?|git|cd)\b/, '<span class="t-fn">$1</span>');
      return pre + body;
    }).join("\n");
  }

  document.querySelectorAll("pre code[data-lang]").forEach((el) => {
    const lang = el.dataset.lang;
    const src = el.textContent.replace(/^\n/, "").replace(/\s+$/, "");
    el.dataset.src = lang === "sh" ? src.split("\n").filter((l) => l.startsWith("$ ")).map((l) => l.slice(2)).join("\n") : src;
    let html = lang === "py" ? hlPython(src) : lang === "sh" ? hlShell(src) : esc(src);
    if (el.hasAttribute("data-lines")) {
      html = html.split("\n").map((l, i) => `<span class="ln">${String(i + 1).padStart(2, " ")}</span>${l}`).join("\n");
    }
    el.innerHTML = html;
  });

  // ---------- copy buttons ----------
  function flash(btn, text) {
    const old = btn.textContent;
    btn.textContent = text;
    setTimeout(() => (btn.textContent = old), 1300);
  }
  function copy(text, btn) {
    (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject())
      .then(() => flash(btn, "Copied"))
      .catch(() => flash(btn, "Select + copy"));
  }
  document.querySelectorAll("pre").forEach((pre) => {
    if (pre.closest(".codeblock") && pre.closest(".codeblock").querySelector(".cap .copy")) return;
    const b = document.createElement("button");
    b.className = "copy"; b.type = "button"; b.textContent = "Copy";
    b.addEventListener("click", () => {
      const c = pre.querySelector("code");
      copy(c ? c.dataset.src || c.textContent : pre.textContent, b);
    });
    pre.appendChild(b);
  });
  document.querySelectorAll("[data-copy]").forEach((b) => {
    b.addEventListener("click", () => {
      const tgt = b.dataset.copy;
      const el = tgt ? document.querySelector(tgt) : null;
      const code = el ? (el.dataset.src || el.textContent) : "";
      copy(code, b);
    });
  });

  // ---------- on this page ----------
  const toc = document.querySelector(".toc");
  if (toc) {
    const hs = document.querySelectorAll(".content h2[id]");
    if (hs.length < 2) toc.style.visibility = "hidden";
    hs.forEach((h) => {
      const a = document.createElement("a");
      a.href = "#" + h.id; a.textContent = h.textContent;
      toc.appendChild(a);
    });
  }

  // ---------- hero: overlaid walkers across generations ----------
  const cv = document.getElementById("swarm");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const label = document.getElementById("swarm-gen");
  const best = document.getElementById("swarm-best");
  const N = 72;
  const GENS = [
    { name: "iteration 0", it: 0, speed: 0.0, skill: 0.0 },
    { name: "iteration 25", it: 25, speed: 0.55, skill: 0.35 },
    { name: "iteration 80", it: 80, speed: 1.25, skill: 0.75 },
    { name: "iteration 200", it: 200, speed: 1.9, skill: 1.0 },
  ];
  const GEN_SECONDS = 4.2;
  let rnd = 7;
  const rand = () => ((rnd = (rnd * 16807) % 2147483647) / 2147483647);
  const bots = Array.from({ length: N }, () => ({
    phase: rand() * 6.28, spd: 0.65 + rand() * 0.55, freq: 0.85 + rand() * 0.3,
    lean: (rand() - 0.5), fallAt: 0.4 + rand() * 1.8, fallDir: rand() < 0.5 ? -1 : 1, stride: 0.75 + rand() * 0.4,
  }));
  let W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = cv.getBoundingClientRect();
    W = r.width; H = r.height;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  }
  window.addEventListener("resize", resize);
  resize();

  // pose of one walker in metres, ground at y=0, start at x=0
  function pose(b, g, t) {
    const skill = g.skill;
    const s = g.speed * b.spd * (0.55 + 0.45 * skill);
    const fallT = b.fallAt * (1 + skill * 6);
    const fall = Math.max(0, Math.min(1, (t - fallT) / 0.7));
    const fallen = skill < 0.98 && fall > 0;
    const moveT = fallen ? Math.min(t, fallT) : t;
    const x = s * moveT + (skill < 0.2 ? Math.sin(t * 2 + b.phase) * 0.04 : 0);
    const w = (t * 6.2 * b.freq * (0.35 + 0.65 * skill) + b.phase);
    const amp = 0.25 + 0.45 * skill * b.stride + (1 - skill) * 0.25 * Math.abs(Math.sin(b.phase + t * 3));
    const hipH = 0.92 - (fallen ? fall * 0.72 : 0) - Math.abs(Math.sin(w)) * 0.03;
    const tilt = (fallen ? b.fallDir * fall * 1.35 : 0) + b.lean * 0.18 * (1 - skill) + 0.12 * skill;
    const hip = [x, hipH];
    const neck = [x + Math.sin(tilt) * 0.62, hipH + Math.cos(tilt) * 0.62];
    const leg = (ph) => {
      const th = Math.sin(ph) * amp;
      const kneeBend = Math.max(0, -Math.cos(ph)) * 0.7 * (0.3 + skill);
      const k = [hip[0] + Math.sin(th) * 0.46, hip[1] - Math.cos(th) * 0.46];
      const th2 = th - kneeBend;
      let f = [k[0] + Math.sin(th2) * 0.46, k[1] - Math.cos(th2) * 0.46];
      if (f[1] < 0.02) f = [f[0], 0.02];
      return [k, f];
    };
    const [k1, f1] = leg(w), [k2, f2] = leg(w + Math.PI);
    return { x, hip, neck, legs: [[k1, f1], [k2, f2]], dist: x };
  }

  let camMax = 4, t0 = performance.now();
  function frame(now) {
    const T = Math.max(0, (now - t0) / 1000);
    const total = GEN_SECONDS * GENS.length;
    const loopT = T % total;
    const gi = Math.floor(loopT / GEN_SECONDS);
    const g = GENS[gi];
    const t = loopT - gi * GEN_SECONDS;
    const poses = bots.map((b) => pose(b, g, t));
    let bi = 0;
    poses.forEach((p, i) => { if (p.dist > poses[bi].dist) bi = i; });
    const target = Math.max(8, poses[bi].dist * 1.12 + 1.5);
    camMax += (target - camMax) * (t < 0.05 ? 1 : 0.06);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const padL = 56, padR = 56, groundY = H - 46;
    const sc = Math.min((W - padL - padR) / (camMax + 0.4), (groundY - 30) / 1.7);
    const X = (m) => padL + (m + 0.4) * sc;
    const Y = (m) => groundY - Math.max(0.015, m) * sc;

    // ground + ticks
    ctx.strokeStyle = "#121212"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, groundY + 0.5); ctx.lineTo(W, groundY + 0.5); ctx.stroke();
    ctx.font = "500 10px 'JetBrains Mono', monospace"; ctx.fillStyle = "#6B665C"; ctx.textAlign = "center";
    const step = camMax > 30 ? 5 : camMax > 14 ? 2 : 1;
    for (let m = 0; m <= camMax + 0.4; m += step) {
      const px = Math.round(X(m)) + 0.5;
      ctx.strokeStyle = "#C9C2B4";
      ctx.beginPath(); ctx.moveTo(px, groundY); ctx.lineTo(px, groundY + (m % (step * 5) === 0 ? 8 : 5)); ctx.stroke();
      if (m % step === 0) ctx.fillText(m + " m", px, groundY + 22);
    }

    // avg marker
    const avg = poses.reduce((a, p) => a + p.dist, 0) / N;
    ctx.setLineDash([3, 4]); ctx.strokeStyle = "#121212"; ctx.globalAlpha = 0.5;
    ctx.beginPath(); ctx.moveTo(X(avg) + 0.5, 14); ctx.lineTo(X(avg) + 0.5, groundY); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;

    const draw = (p, solid) => {
      const seg = (a, b) => { ctx.moveTo(X(a[0]), Y(a[1])); ctx.lineTo(X(b[0]), Y(b[1])); };
      ctx.beginPath();
      seg(p.hip, p.neck);
      p.legs.forEach(([k, f]) => { seg(p.hip, k); seg(k, f); });
      ctx.stroke();
      if (solid) {
        ctx.fillStyle = "#FF4D00";
        [p.hip, p.legs[0][0], p.legs[1][0]].forEach((j) => { ctx.beginPath(); ctx.arc(X(j[0]), Y(j[1]), 2.4, 0, 6.29); ctx.fill(); });
      }
    };
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(18,18,18,0.16)"; ctx.lineWidth = Math.min(3, Math.max(1.2, sc * 0.035));
    poses.forEach((p, i) => { if (i !== bi) draw(p, false); });
    ctx.strokeStyle = "#FF4D00"; ctx.lineWidth = Math.min(4, Math.max(2, sc * 0.05));
    draw(poses[bi], true);

    // labels
    ctx.textAlign = "left"; ctx.font = "500 10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#6B665C"; ctx.fillText("AVG " + avg.toFixed(1) + " M", X(avg) + 6, 22);
    if (label) label.textContent = g.name + "  ·  " + N + " of 4,096 shown";
    if (best) best.textContent = "best " + poses[bi].dist.toFixed(1) + " m";
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
