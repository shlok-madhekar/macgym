// deck.js : MacGym pitch deck. One persistent halftone particle field morphing between slides.
// Structure: DATA loading -> SLIDES (one object per slide) -> ENGINE (particles, text, controls).
(function () {
  const S = window.Shapes, T = S.T;
  const W = 1920, H = 1080, N = 4400, DUR = 1.25;
  const qs = new URLSearchParams(location.search);
  const STATIC = qs.get('static') === '1';

  // ======================= DATA =======================
  const DATA = { gens: null, bench: null };
  const fmt = (v) => (v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? Math.round(v / 1e3) + 'K' : String(Math.round(v)));
  function summarizeBench(b) {
    const out = { machine: b.machine, tasks: {} };
    const sim = b.results.filter((r) => r.backend === 'MLX GPU (Metal)');
    const cpu = (task) => b.results.find((r) => r.task === task && /11 procs/.test(r.backend));
    const train = b.results.find((r) => /training/.test(r.backend));
    for (const task of ['walker', 'cheetah', 'cartpole']) {
      const rows = sim.filter((r) => r.task === task);
      if (!rows.length) continue;
      const best = rows.reduce((a, r) => (r.steps_per_sec > a.steps_per_sec ? r : a));
      const c = cpu(task === 'walker' ? 'cheetah' : task);
      if (!c) continue;
      const x = (b.speedup_best_gpu_vs_best_cpu && b.speedup_best_gpu_vs_best_cpu[task]) || best.steps_per_sec / c.steps_per_sec;
      out.tasks[task] = { gpu: best.steps_per_sec, n: best.num_envs, cpu: c.steps_per_sec, x };
    }
    out.train = train ? train.steps_per_sec : null;
    return out;
  }
  const xfmt = (x) => (x >= 10 ? Math.round(x) : x.toFixed(1)) + '×';

  // ======================= HELPERS =======================
  // headline lines: "~text" gray, "^text" accent
  function hl(lines, cls = '') {
    return `<div class="hl ${cls}">` + lines.map((l, i) => {
      const c = l[0] === '~' ? 'g' : l[0] === '^' ? 'a' : '';
      const t = c ? l.slice(1) : l;
      return `<span class="ln rv ${c}" style="transition-delay:${0.08 + i * 0.07}s">${t}</span>`;
    }).join('') + '</div>';
  }
  const kick = (t) => `<div class="kicker rv" style="transition-delay:.02s">${t}</div>`;
  function facts(rows, d0 = 0.42, extra = '') {
    return `<ul class="facts ${extra}">` + rows.map((r, i) => {
      const cls = r[0] === '^' ? 'acc' : r[0] === '!' ? 'bad' : '';
      const t = cls ? r.slice(1) : r;
      return `<li class="rv ${cls}" style="transition-delay:${d0 + i * 0.07}s">${t}</li>`;
    }).join('') + '</ul>';
  }
  const lede = (t, d = 0.4, cls = '') => `<p class="lede rv ${cls}" style="transition-delay:${d}s">${t}</p>`;
  const hash = S.hash;
  const sm = S.smooth;

  function ground(w, x0, x1, y, camWX, scale, spacingM, tickEveryM, tone, r = 1.6) {
    // world-anchored dotted ground, constant count
    const n = Math.ceil((x1 - x0) / (spacingM * scale)) + 2;
    const startWX = Math.floor((camWX) / spacingM) * spacingM;
    for (let k = 0; k < n; k++) {
      const wx = startWX + k * spacingM;
      const x = x0 + (wx - camWX) * scale;
      const fade = sm(x0, x0 + 70, x) * sm(x1, x1 - 70, x);
      w.push(x, y, r * fade, tone);
      const isTick = tickEveryM && Math.abs(wx / tickEveryM - Math.round(wx / tickEveryM)) < 1e-3;
      w.push(x, y + 9, isTick ? r * 1.1 * fade : 0, tone);
    }
  }
  function rectOutline(w, x, y, ww, hh, sp, r, tone) {
    const nx = Math.round(ww / sp), ny = Math.round(hh / sp);
    for (let i = 0; i < nx; i++) { w.push(x + i * sp, y, r, tone); w.push(x + ww - i * sp, y + hh, r, tone); }
    for (let j = 0; j < ny; j++) { w.push(x + ww, y + j * sp, r, tone); w.push(x, y + hh - j * sp, r, tone); }
  }
  function gen(i) { return DATA.gens[i]; }

  // ---- evidence files in deck/data/ (other agents drop them in; slides adapt to what exists) ----
  const F = () => DATA.f || {};
  const has = (n) => !!F()[n];
  const V = () => DATA.ver || 0;
  const imgTag = (n, style, d = 0.3) => (has(n) ? `<img class="panel rv" style="${style};transition-delay:${d}s" src="data/${n}?v=${V()}">` : '');
  const esc = (t) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const pct = (x, d = 2) => (Math.abs(x) < 0.01 ? x.toFixed(3) : x.toFixed(d));
  function fair() { // same-solver CPU vs GPU, adapter for fair_bench.json (schema tolerant)
    const j = F()['fair_bench.json']; if (!j || !j.rows) return null;
    const at = (re, n) => j.rows.find((r) => re.test(r.backend) && r.num_envs === n);
    const g = at(/^GPU Metal/, 4096), c = at(/^CPU same solver/, 4096);
    if (!g || !c) return null;
    return { x: g.env_steps_per_s / c.env_steps_per_s, gpu: g.env_steps_per_s, cpu: c.env_steps_per_s, n: 4096 };
  }
  function numberReading() {
    const r = F()['recordings.json'], t30 = r && r['playground_tripod.mp4'] && r['playground_tripod.mp4'].readings && r['playground_tripod.mp4'].readings.t30;
    const m = t30 && /([\d.]+) days of tripod experience in (\d+):(\d+)/.exec(t30.experience);
    if (m) return { days: +m[1], secs: (+m[2]) * 60 + (+m[3]), clock: m[2] + ':' + m[3], avg: t30.avg_distance_m.toFixed(2), from: '-0.06', src: true };
    return { days: 3.2, secs: 43, clock: '0:43', avg: '7.20', from: '-0.06', src: false };
  }
  const QUIET = { walker: 6.27e6, mujoco: 294e3 }; // earlier quiet bench run (walker vs MuJoCo HalfCheetah, 11 procs)
  function tripodGen() { const j = F()['tripod_replay.json']; if (!j) return null; const gs = j.generations || (j.frames ? [j] : null); if (!gs) return null; const g = gs[gs.length - 1]; g.robot = j.robot || 'tripod'; g.dt = g.dt || j.dt; return g; }

  // ======================= SLIDES =======================
  // Each slide: { id, cls, html(), notes, steps, staticT, shape(w, t, step), draw(ctx, t, a, step) }
  const SLIDES = [
    // ---------------------------------------------------------------- title
    {
      id: 'title', cls: 's-title',
      html: () => kick('Bay-Valley Hacks <b>·</b> Elevate the Ordinary') + hl(['MacGym']) + lede('Robot learning on the Mac you already own.', 0.35),
      notes: 'This robot taught itself to run, on this laptop, in about a minute. Nobody animated it. And the first versions of it cheated, which I will show you. I am Shlok, and this is MacGym.',
      staticT: 2.6,
      shape(w, t) {
        const g = gen(3), Fr = S.frameAt(g, t + 2.0, 'loop'), scale = 300, root = Fr[0];
        ground(w, 980, 1860, 770, root - 1.2, scale, 0.1, 0.5, T.GRAY, 1.7);
        S.robot(w, g, { fr: Fr, ox: 1360, oy: 770, scale, rootWX: root, spacing: 7, rMax: 3.3, near: T.INK, far: T.GRAY });
      },
      draw(ctx, t, a) { label(ctx, 'REAL REPLAY · WALKER AFTER 1:00 OF TRAINING · ' + gen(3).result_text.toUpperCase(), 1000, 830, a, '#9A9A95'); },
    },
    // ---------------------------------------------------------------- problem
    {
      id: 'problem',
      html: () => kick('The problem') + hl(['Every serious GPU', 'robot-RL sim is', 'NVIDIA-only.', '~Your Mac isn’t invited.'], 'sm') +
        facts(['<b>Isaac Lab</b> · NVIDIA GPU required <span class="src">Isaac Lab docs</span>', '<b>MuJoCo Warp</b> · NVIDIA GPU for the fast path <span class="src">MuJoCo Warp README</span>', '!<b>jax-metal</b> · “consider it dead” <span class="src">a JAX maintainer, Jan 2026</span>', '^best community Mac port · <b>~5.2K steps/s</b> on an M3 Max <span class="src">community benchmark</span>']),
      notes: 'Every serious GPU simulator for robot learning assumes an NVIDIA card. Isaac Lab, MuJoCo Warp. The Mac path for JAX got called dead by a maintainer this January, and the best community port I found does about five thousand steps a second.',
      shape(w, t) { chip(w, t); },
      draw(ctx, t, a) { label(ctx, 'CUDA REQUIRED', 1390, 862, a, '#9A9A95', 'center'); },
    },
    // ---------------------------------------------------------------- laptop (theme)
    {
      id: 'laptop',
      html: () => kick('The ordinary laptop') + hl(['The laptop you', 'already own', 'has a GPU.', '~Nobody uses it', '~for this.'], 'sm') +
        facts(['<b>Apple M3 Pro</b> · the laptop this deck is running on', '<b>18 GB unified memory</b>, CPU and GPU share it', '^MacGym turns it into a robot gym']),
      notes: 'Meanwhile the most ordinary computer in this room is a MacBook, and it has a real GPU sitting idle. That is the ordinary thing I wanted to elevate.',
      shape(w, t) { laptop(w, t, 1390, 230, 620, 390, true); },
      draw(ctx, t, a) { label(ctx, 'GPU · IDLE', 1390, 518, a, '#FF4D00', 'center'); },
    },
    // ---------------------------------------------------------------- so I built
    {
      id: 'built', steps: 2,
      html: () => kick('So I built') + hl(['MacGym'], 's-big-hl') +
        lede('Define a robot in Python. Train <b>4,096 copies</b> of it on your Mac’s GPU in about a minute. Export the policy to <b>CoreML</b>.', 0.3) +
        `<div class="s1">${facts(['<b>one</b> robot definition', '^<b>4,096</b> copies, one GPU thread each'], 0.15)}</div>`,
      notes: 'So I built MacGym. You describe a robot in Python, MacGym generates a GPU kernel for it, and then it is not one robot, it is four thousand ninety six, all learning at once. Let me show you it running.',
      staticT: 2.4,
      shape(w, t, step) {
        const g = gen(3);
        if (step === 0) {
          const Fr = S.frameAt(g, t + 3.1, 'loop'), root = Fr[0];
          ground(w, 1040, 1760, 790, root - 1.2, 270, 0.1, 0.5, T.GRAY, 1.6);
          S.robot(w, g, { fr: Fr, ox: 1370, oy: 790, scale: 270, rootWX: root, spacing: 7, rMax: 3.2, near: T.INK, far: T.GRAY });
          return;
        }
        for (let r = 0; r < 5; r++) for (let c = 0; c < 7; c++) {
          const k = r * 7 + c, Fr = S.frameAt(g, t + 3.1 + hash(k) * 7.5, 'loop'), root = Fr[0];
          const ox = 1040 + c * 118, oy = 330 + r * 128, acc = r === 2 && c === 3;
          S.robot(w, g, { fr: Fr, ox, oy, scale: 70, rootWX: root, spacing: 5, rMax: 1.55, near: acc ? T.ACC : T.INK, far: acc ? T.ACC : T.GRAY });
          for (let q = -3; q <= 3; q++) w.push(ox + q * 12, oy + 5, 1.0, T.LIGHT);
        }
      },
      draw(ctx, t, a, step) { if (step === 1) label(ctx, '35 DRAWN HERE · 4,096 TRAIN AT ONCE', 1390, 930, a, '#9A9A95', 'center'); },
    },
    // ---------------------------------------------------------------- live proof
    {
      id: 'live', cls: 's-live',
      html: () => {
        const vids = ['playground_edit.mp4', 'playground_tripod_4x.mp4'].filter(has);
        const media = vids.length
          ? `<video class="rv" style="transition-delay:.7s" id="pgvid" muted playsinline autoplay data-list="${vids.join(',')}" src="data/${vids[0]}?v=${V()}"></video>`
          : (Object.keys(F()).filter((n) => /^pg_t.*\.png$/.test(n)).sort().slice(-1).map((n) => `<img class="rv" src="data/${n}?v=${V()}">`)[0] || '');
        return kick('Live proof') + hl(['It runs.', '~Right here.'], 'sm') +
          facts(['real screen recording of the <b>MacGym playground</b> on this M3 Pro', 'add a leg in the browser: Metal kernel recompiled for 9 bodies / 8 joints in <b>0.24 s</b>, then <b>4,096 copies</b> train', '^press <b>L</b> and I will do it live'], 0.35) +
          `<div class="screen">${media}</div><div class="screencap rv" style="transition-delay:.5s">${vids.length ? 'REAL SCREEN RECORDING · EDIT THE ROBOT, THEN TRAINING AT 4× SPEED' : media ? 'PLAYGROUND SCREENSHOT' : 'LIVE DEMO · PRESS L'}</div>`;
      },
      notes: 'This is a real screen recording, not an animation. I add a fourth leg right in the browser, MacGym recompiles the Metal kernel in a quarter of a second, and four thousand copies start learning on the laptop GPU. If you want, I will press L and do it live right now.',
      staticT: 2.0,
      shape(w, t) {
        laptop(w, t, 1390, 214, 840, 500, false);
        if (!document.querySelector('#pgvid, .screen img')) { // no recording yet: mini robots in the screen
          const g = gen(3);
          for (let r = 0; r < 3; r++) for (let c = 0; c < 6; c++) {
            const k = r * 6 + c, Fr = S.frameAt(g, t + hash(k) * 7, 'loop');
            S.robot(w, g, { fr: Fr, ox: 1100 + c * 116, oy: 380 + r * 130, scale: 62, rootWX: Fr[0], spacing: 5.5, rMax: 1.4, near: T.INK, far: T.GRAY });
          }
        }
      },
    },
    // ---------------------------------------------------------------- how it works
    {
      id: 'how',
      html: () => kick('How it works') + hl(['Python in.', '~Metal kernel out.'], 'sm') +
        facts(['<b>1 · You write ~15 lines of Python</b><span class="pl">describe bodies, joints, motors and a reward</span>',
          '<b>2 · MacGym writes a Metal kernel</b><span class="pl">turns that robot into code your Mac’s graphics chip runs, one GPU thread per robot</span>',
          '<b>3 · PPO in MLX learns the policy</b><span class="pl">on the same unified memory, no copying robots back and forth</span>',
          '^<b>4 · Export to CoreML / ONNX</b><span class="pl">the trained brain runs on an iPhone</span>']) +
        `<div class="codecap rv" style="transition-delay:.25s">robot.py · you write this</div><pre class="code code-py rv" style="transition-delay:.3s">${PY_SHORT}</pre>` +
        `<div class="codecap kcap rv" style="transition-delay:.45s">${has('kernel_excerpt.txt') ? 'generated Metal · MacGym writes this' : 'generated Metal kernel · excerpt pending'}</div><pre class="code code-metal rv" style="transition-delay:.5s">${kernelHTML()}</pre>`,
      notes: 'Here is the pipeline. You write the robot in Python, top right. MacGym generates the Metal kernel underneath it, that is real generated code, one GPU thread per robot copy. PPO runs in MLX on the same memory. Out comes a CoreML or ONNX file.',
      shape(w, t) {
        // dot stream from the Python block down into the kernel block + thread grid ripple
        for (let k = 0; k < 160; k++) {
          const u = ((k / 160) + t * 0.22) % 1, y = 420 + u * 130, x = 1300 + (hash(k) - 0.5) * 22;
          w.push(x, y, 2.0 * Math.sin(u * Math.PI), T.ACC);
        }
        for (let j = 0; j < 4; j++) for (let i = 0; i < 64; i++) {
          const ph = Math.sin(t * 4 - (i * 0.35 + j * 0.8));
          w.push(1040 + i * 12, 1000 + j * 12, 1.3 + 0.8 * ph, i % 16 === 0 ? T.ACC : T.GRAY);
        }
      },
      draw(ctx, t, a) { label(ctx, 'ONE GPU THREAD PER ROBOT · 256 OF 4,096 SHOWN', 1040, 985, a, '#9A9A95'); },
    },
    // ---------------------------------------------------------------- the number
    {
      id: 'number',
      html: () => {
        const R = numberReading(), steps = Math.round((R.days * 86400) / (4096 * 0.02)), tot = R.days * 86400 / 0.02;
        return kick('The number') + hl([`${R.days} days of`, 'experience.', `^${R.secs} seconds.`]) +
          facts([`<b>4,096</b> tripods in parallel, live playground run on one M3 Pro${R.src ? ' (screen recording, timer at ' + R.clock + ')' : ''}`, `<span class="m">4,096 robots × ${steps.toLocaleString()} steps × 0.02 s ≈ ${Math.round(R.days * 86400).toLocaleString()} s ≈ ${R.days} days</span>`, `that is ${(tot / 1e6).toFixed(1)}M steps in ${R.secs} s, about <b>${fmt(tot / R.secs)} steps/s</b> including PPO`, `^average distance per episode: <b>${R.avg} m</b> at that same moment`]) + imgTag(R.src ? 'pg_t30.png' : '__none', 'left:1010px;top:150px;width:520px;border:1px solid #E3E3DE', 0.5);
      },
      notes: 'That recording, frozen at twenty eight seconds: 4,096 tripods have already racked up three point two days of experience. The math is on the slide: robots times steps times the 0.02 second timestep. That is roughly half a million steps a second, with learning included.',
      staticT: 3.0,
      shape(w, t) {
        const tg = tripodGen(), g = tg || gen(3), shot = has('pg_t30.png'), rows = shot ? 4 : 5, per = tg ? 5 : 7, X0 = 960, Wd = 900, v = 1.3;
        for (let r = 0; r < rows; r++) {
          const scale = (tg ? 30 : 34) + r * 8.5, oy = shot ? 560 + r * 118 : 300 + r * 132;
          for (let c = 0; c < per; c++) {
            const k = r * per + c, Fr = S.frameAt(g, t + hash(k + 9) * 6, 'loop'), root = Fr[0];
            const x = X0 + (((c / per) * Wd + hash(k) * 60 + t * v * scale) % Wd);
            const fade = sm(X0, X0 + 90, x) * sm(X0 + Wd, X0 + Wd - 90, x), acc = r === 4 && c === 2, tone = r < 2 ? T.GRAY : T.INK;
            S.robot(w, g, { fr: Fr, ox: x, oy, scale, rootWX: root, spacing: 5.2, rMax: 1.1 + r * 0.2, near: acc ? T.ACC : tone, far: acc ? T.ACC : (r < 2 ? T.LIGHT : T.GRAY), fade });
          }
        }
      },
      draw(ctx, t, a) { if (has('pg_t30.png')) label(ctx, 'PLAYGROUND AT 0:28 · REAL SCREENSHOT', 1550, 170, a, '#9A9A95');
        label(ctx, tripodGen() ? 'REAL TRIPOD REPLAYS' : 'WALKER REPLAYS (ILLUSTRATION; THE STAT IS THE TRIPOD)', 1410, 1000, a, '#9A9A95', 'center'); },
    },
    // ---------------------------------------------------------------- it learns
    {
      id: 'learns',
      html: () => {
        const sj = F()['seeds.json'];
        return kick('It learns') + hl(['From falling', 'to running.', '~In one minute.'], 'sm') +
          facts(DATA.gens.map((g, i) => (i === 3 ? '^' : '') + `<b>${g.label}</b> of training · ${g.result_text}`)) +
          `<p class="caveat rv" style="transition-delay:.75s">Lanes are real replays from one from-scratch walker run: one 8 s evaluation episode per checkpoint, distances as recorded.${has('seeds.png') ? ' Chart: the same training repeated over 5 seeds (mean and spread).' : ''}</p>` +
          imgTag('seeds.png', 'left:1010px;top:120px;width:800px', 0.4);
      },
      notes: 'These are real replays from one run. At zero seconds it falls. Ten seconds, it lunges and falls. Thirty seconds, fifteen metres. One minute, twenty one and a half metres in eight seconds. And so you know it is not one lucky run, the chart is five seeds.',
      staticT: 9.0,
      shape(w, t) {
        const withChart = has('seeds.png'), period = 9.0, tt = Math.min(t, 8.6) % period, scale = withChart ? 52 : 82, x0 = 1020, x1 = 1820;
        const top = withChart ? 640 : 300, gap = withChart ? 108 : 178;
        for (let i = 0; i < 4; i++) {
          const g = gen(i), oy = top + i * gap, Fr = S.frameAt(g, tt, 'hold'), root = Fr[0], start = g.frames[0][0];
          const prog = Math.min(1, Math.max(0, (root - start) / 22)), sx = 1110 + prog * 560, cam = root - (sx - x0) / scale, acc = i === 3;
          ground(w, x0, x1, oy, cam, scale, withChart ? 0.2 : 0.125, 1, T.GRAY, 1.3);
          S.robot(w, g, { fr: Fr, ox: sx, oy, scale, rootWX: root, spacing: withChart ? 4.6 : 5.5, rMax: withChart ? 1.35 : 1.75, near: acc ? T.ACC : T.INK, far: acc ? T.ACC : T.GRAY });
        }
      },
      draw(ctx, t, a) {
        const withChart = has('seeds.png'), top = withChart ? 640 : 300, gap = withChart ? 108 : 178, lift = withChart ? 78 : 118;
        for (let i = 0; i < 4; i++) {
          const g = gen(i), oy = top + i * gap;
          label(ctx, g.label, 1020, oy - lift, a, i === 3 ? '#FF4D00' : '#111', 'left', withChart ? 'mono' : 'big');
          label(ctx, g.result_text, 1820, oy - lift, a, i === 3 ? '#FF4D00' : '#6A6A66', 'right', 'mono');
        }
      },
    },
    // ---------------------------------------------------------------- cheated
    {
      id: 'cheated',
      html: () => kick('The twist') + hl(['My first robots', 'cheated.', '~Here’s how', '~I caught it.'], 'sm') +
        facts(['!v1 walker ran <b>6.16 m/s</b>, tripod <b>5.23 m/s</b>. Too fast to be real', '!why: motors had unlimited power and the feet <b>skated</b> along the ground', 'caught it with a foot-slip metric · fixed with warm-started contact impulses + a <b>DC-motor torque-speed curve</b>', '^now walker <b>2.74</b>, tripod <b>1.32</b>, cheetah <b>3.76 m/s</b> · cheetah foot slip <b>0.735 → 0.053 m/s</b>']),
      notes: 'This is my favorite part. My first robots were running six metres a second. That is RL cheating: the motors had infinite power and the feet skated. I measured foot slip, found it, and fixed the physics with warm-started contacts and a real DC motor curve. Speeds dropped to believable numbers and slip dropped about fourteen times.',
      shape(w, t) { slipPanels(w, t); },
      draw(ctx, t, a) {
        label(ctx, 'BEFORE · CHEETAH STANCE FOOT SLIPS 0.735 M/S', 1010, 205, a, '#B8432F');
        label(ctx, 'AFTER · 0.053 M/S', 1010, 615, a, '#FF4D00');
      },
    },
    // ---------------------------------------------------------------- validation
    {
      id: 'valid',
      html: () => {
        const v = F()['validation.json'];
        if (!v) return kick('Validation') + hl(['It’s real', '~physics.'], 'sm') + facts(['sim-to-sim check vs MuJoCo · results pending']);
        const p = v.pendulum, s = v.slide, d = v.drop;
        return kick('Is it real physics?') + hl(['Checked it', 'against MuJoCo.', '~It matches.'], 'sm') +
          facts([`<b>pendulum period</b> ${p.period_s.ours.toFixed(4)} s vs analytic ${p.analytic_period_s.toFixed(4)} s · error <b>${pct(p.period_err_pct_vs_analytic.ours)}%</b>`,
            `<b>swing angle</b> vs MuJoCo RK4 · RMS error <b>${p.rms_angle_err_deg_vs_mujoco_rk4.ours.toFixed(2)}°</b> (MuJoCo’s own Euler: ${p.rms_angle_err_deg_vs_mujoco_rk4.mujoco_euler.toFixed(2)}°)`,
            `<b>energy drift</b> over 5 s · <b>${pct(p.energy_drift_pct_of_swing_energy_5s.ours)}%</b> of swing energy`,
            `<b>friction stop</b> at 2 m/s, μ 0.5 · ours ${s.ours_m.toFixed(3)} m, MuJoCo ${s.mujoco_pyramidal_m.toFixed(3)} m, analytic ${s.analytic_m.toFixed(3)} m`,
            `^<b>drop test</b> · impact at ${d.ours.impact_time_s.toFixed(2)} s, same as MuJoCo · penetration ${d.ours.penetration_mm.toFixed(4)} mm`]) +
          `<p class="caveat rv" style="transition-delay:.8s">MuJoCo ${esc(v.mujoco_version)} with matching geometry, mass and timestep. Dots on the right are the recorded trajectories.</p>`;
      },
      notes: 'The obvious question: is this real physics or a toy? So I ran the same setups in MuJoCo. A pendulum, a friction slide, a drop. The period is within a hundredth of a percent of the textbook answer, the swing tracks MuJoCo to about a sixth of a degree, and the friction stop lands within a few millimetres.',
      staticT: 2.8,
      shape(w, t) { pendulum(w, t); },
      draw(ctx, t, a) {
        if (!F()['validation.json']) return;
        label(ctx, 'PENDULUM ANGLE · RECORDED · THE TWO CURVES OVERLAP', 1010, 650, a, '#9A9A95');
        label(ctx, '● MACGYM', 1590, 650, a, '#FF4D00');
        label(ctx, '● MUJOCO RK4', 1700, 650, a, '#8C8C88');
      },
    },
    // ---------------------------------------------------------------- fast
    {
      id: 'fast', cls: 's-big',
      html: () => {
        const fb = fair(), x = fb && fb.x;
        const hero = x ? `${x >= 10 ? Math.round(x) : x.toFixed(1)}×` : '21×';
        return kick(x ? 'Fast · same solver, CPU vs GPU' : 'Fast') + hl(['^' + hero]) +
          lede(x ? `faster on the Mac’s GPU than <b>the same solver on its CPU</b>, both at 4,096 walkers (${fmt(fb.gpu)} vs ${fmt(fb.cpu)} steps/s). Same physics, so that gain is the GPU.` : 'faster than <b>MuJoCo on all 11 CPU cores</b> of this laptop, quiet machine.', 0.3) +
          facts([
            `<b>vs MuJoCo:</b> about <b>21× faster</b> than MuJoCo on all 11 cores · walker 6.27M vs MuJoCo HalfCheetah 294K steps/s (quiet run)`,
            (() => { const p = F()['profile.json']; return p ? `<b>raw sim vs training:</b> training runs at ${fmt(p.plain_env_steps_per_s)} steps/s end-to-end because the <b>PPO update takes ${Math.round(p.pieces_pct.ppo_update_backprop_adam)}%</b> of each iteration; physics is ${Math.round(p.pieces_pct.env_step_physics_reward_reset)}%` : '<b>raw sim vs training:</b> ~300K steps/s end-to-end, the PPO update is the bottleneck'; })()], 0.42) +
          `<p class="caveat big rv" style="transition-delay:.7s"><b>Not the same physics.</b> MacGym is a 2D maximal-coordinate impulse solver; MuJoCo uses generalized coordinates and a convex contact solver. Read it as “robot experience per second on this laptop”, not “a faster MuJoCo”.</p>` +
          profileHTML();
      },
      notes: 'Speed, honestly. The fair number is the same solver on the CPU versus the GPU, that isolates what the GPU buys you. Against MuJoCo on all eleven cores we measured about twenty one times on a quiet machine, but that is not the same physics, ours is simpler 2D. And training is slower than raw simulation because the PPO update is the bottleneck now, not physics.',
      shape(w, t) {
        const fb = fair(), x = fb && fb.x;
        const bars = x ? [[1, T.ACC], [1 / x, T.INK]] : [[1, T.ACC], [QUIET.mujoco / QUIET.walker, T.GRAY]];
        const base = 700, rowsMax = 50, sp = 9.5, cols = 14;
        bars.forEach((b, gi) => {
          const gx = 1130 + gi * 330, rows = Math.max(1, Math.round(rowsMax * b[0]));
          for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) w.push(gx + c * sp, base - r * sp, 2.9 + (r === rows - 1 ? 0.4 * Math.sin(t * 3 + c) : 0), b[1]);
        });
      },
      draw(ctx, t, a) {
        const fb = fair(), x = fb && fb.x;
        const bars = x ? [['GPU · METAL KERNEL', '#FF4D00', fmt(fb.gpu)], ['CPU · SAME SOLVER', '#111', fmt(fb.cpu)]] : [['MACGYM GPU', '#FF4D00', '6.3M'], ['MUJOCO, 11 CORES', '#8C8C88', '294K']];
        bars.forEach((b, gi) => { label(ctx, b[2], 1130 + gi * 330 + 62, 742, a, b[1], 'center', 'big'); label(ctx, b[0], 1130 + gi * 330 + 62, 766, a, '#9A9A95', 'center'); });
        if (x) label(ctx, 'WALKER · 4,096 ENVS · STEPS/S', 1130, 200, a, '#9A9A95');
      },
    },
    // ---------------------------------------------------------------- framework
    {
      id: 'framework',
      html: () => kick('It’s a framework') + hl(['Not a demo.', '~A library.'], 'sm') +
        facts(['<b>Python API</b> · <span class="m">mg.Robot · mg.Task · mg.make · mg.train</span>', '<b>CLI</b> · <span class="m">macgym new · train · view · export · bench</span>', '<b>Gymnasium</b> wrapper, docs site, browser playground', '^CoreML export checked against MLX: max diff <b>3.6e-7</b>']) +
        `<div class="codecap rv" style="transition-delay:.25s">terminal · real output, M3 Pro</div><pre class="code term rv" style="transition-delay:.3s">${termHTML()}</pre>` + imgTag('docs.png', 'left:1030px;top:790px;width:360px;border:1px solid #E3E3DE', 0.6),
      notes: 'And it is a real library. This is real terminal output: macgym train hopper, four thousand robots, forty iterations, seventeen seconds, distance climbing from zero to three metres. Then one command exports it to CoreML, checked against the original to within four ten-millionths.',
      shape(w, t) { cliCurve(w, t); },
      draw(ctx, t, a) { label(ctx, 'HOPPER DISTANCE PER ITERATION · FROM THAT RUN', has('docs.png') ? 1430 : 1040, 790, a, '#9A9A95'); },
    },
    // ---------------------------------------------------------------- built today
    {
      id: 'today',
      html: () => {
        const tj = F()['tests.json'], tl = F()['timeline.json'];
        return kick('Built today') + hl(['Started 11:38 AM.', '~Tested, not just typed.'], 'sm') +
          facts([tl ? `<b>${tl.count} commits</b> since <b>${tl.commits[0].hhmm} AM</b>, timeline from git log` : 'git history starts at <b>11:38 AM</b> today',
            tj ? `<b>${tj.total_passed} checks passing, ${tj.total_failed} failing</b> · ${tj.pytest_cli_export.summary} + ${tj.test_api_quick.total_passed} API checks` : '7 pytest tests + 6 test scripts',
            tj ? `legacy parity: obs max err <b>0.0</b> · walker ${tj.test_api_quick.parity.walker}` : '5 examples, 7 built-in robots',
            '^Built solo today with AI coding agents. I designed it, directed it and caught the bugs.']);
      },
      notes: 'All of this was built today. The git log starts at eleven thirty eight. Eighteen checks pass, zero fail. And to be straight with you: I built it solo with AI coding agents. I designed the architecture, directed it, and caught the bugs, like the cheating robots.',
      staticT: 3.0,
      shape(w, t) { timeline(w, t); },
      draw(ctx, t, a) { timelineLabels(ctx, a); },
    },
    // ---------------------------------------------------------------- who
    {
      id: 'who',
      html: () => kick('Who it’s for') + hl(['RL classes', 'on MacBooks.', '~No cloud GPU', '~required.'], 'sm') +
        lede('A student can go from a fresh clone to a trained walking robot in about a minute, on the laptop they already carry.', 0.35) +
        `<pre class="quick rv" style="transition-delay:.5s"><span class="n">1</span>pip install -e .\n<span class="n">2</span>macgym train walker\n<span class="n">3</span>macgym view walker</pre>`,
      notes: 'The wedge is students. RL classes and robotics clubs where everyone has a MacBook and nobody has a cluster. Three commands: install, train, view. A trained walking robot in about a minute, no cloud bill.',
      shape(w, t) { classroom(w, t); },
      draw(ctx, t, a) { label(ctx, 'A CLASSROOM OF MACBOOKS · EACH ONE A ROBOT GYM', 1390, 905, a, '#9A9A95', 'center'); },
    },
    // ---------------------------------------------------------------- next
    {
      id: 'next',
      html: () => {
        const v = F()['validation.json'];
        return kick('Done · next') + hl(['Sim-to-sim: done.', '~Real arm: next.'], 'sm') +
          facts([v ? `<b>done</b> · physics checked against MuJoCo: pendulum within <b>${v.pendulum.rms_angle_err_deg_vs_mujoco_rk4.ours.toFixed(2)}°</b>, friction stop within <b>${(Math.abs(v.slide.ours_m - v.slide.mujoco_pyramidal_m) * 1000).toFixed(1)} mm</b>` : '<b>done</b> · CoreML / ONNX export verified',
            '<b>next</b> · 3D physics with the same API', '<b>next</b> · policy-transfer check: train in MacGym, run in MuJoCo', '^<b>next</b> · CoreML policy on an iPhone driving a real SO-101 arm']) +
          `<p class="caveat rv" style="transition-delay:.8s">Today it is 2D planar physics only, and there is no sim-to-real result yet.</p>`;
      },
      notes: 'What is done: the physics is validated against MuJoCo. What is next: 3D with the same API, a check that policies trained here still work in MuJoCo, and then a CoreML policy on an iPhone driving a real SO-101 arm. Today it is 2D only, and I want to be upfront about that.',
      shape(w, t) { arm3d(w, t); },
      draw(ctx, t, a) { label(ctx, 'SO-101 ARM · 3D · NEXT', 1250, 930, a, '#9A9A95', 'center'); label(ctx, 'IPHONE · COREML', 1730, 690, a, '#FF4D00', 'center'); },
    },
    // ---------------------------------------------------------------- close
    {
      id: 'close', cls: 's-close',
      html: () => hl(['Elevate the ordinary.']) + `<pre class="quick close rv" style="transition-delay:.35s">git clone github.com/shlokmadhekar/macgym && pip install -e .</pre>` + lede('MacGym · robot learning on the Mac you already own · scan for the repo', 0.45),
      notes: 'Your laptop is the most ordinary computer you own. MacGym makes it a robot gym. Scan the code, it is all on GitHub. Thank you.',
      shape(w, t) { closeMark(w, t); },
      draw(ctx, t, a) { label(ctx, 'GITHUB.COM/SHLOKMADHEKAR/MACGYM', 1180, 555, a, '#9A9A95', 'center'); },
    },
  ];

  // ======================= SHAPE BUILDERS =======================
  function chip(w, t) {
    const cx = 1390, cy = 520, sub = 500, sp = 15;
    for (let j = 0; j <= sub / sp; j++) for (let i = 0; i <= sub / sp; i++) {
      const x = cx - sub / 2 + i * sp, y = cy - sub / 2 + j * sp, inDie = Math.abs(x - cx) < 140 && Math.abs(y - cy) < 140;
      w.push(x, y, inDie ? 0 : 1.25, T.LIGHT);
    }
    const d = 270, ds = 9, phase = ((t * 0.35) % 1.6) - 0.3;
    for (let j = 0; j < d / ds; j++) for (let i = 0; i < d / ds; i++) {
      const u = i / (d / ds - 1), v = j / (d / ds - 1), shade = 0.25 + 0.75 * (u * 0.55 + v * 0.45);
      const pulse = Math.exp(-Math.pow(((u + v) / 2 - phase) * 9, 2));
      w.push(cx - d / 2 + i * ds + ds / 2, cy - d / 2 + j * ds + ds / 2, 1.1 + 2.3 * shade + 1.0 * pulse, T.INK);
    }
    for (let p = 0; p < 13; p++) {
      const o = -sub / 2 + 40 + (p * (sub - 80)) / 12;
      for (let k = 0; k < 5; k++) { const e = sub / 2 + 14 + k * 8; w.push(cx + o, cy - e, 2.1, T.INK); w.push(cx + o, cy + e, 2.1, T.INK); w.push(cx - e, cy + o, 2.1, T.INK); w.push(cx + e, cy + o, 2.1, T.INK); }
    }
    rectOutline(w, cx - sub / 2, cy - sub / 2, sub, sub, 8, 1.6, T.INK);
  }
  function laptop(w, t, cx, top, sw, sh, fill) {
    rectOutline(w, cx - sw / 2, top, sw, sh, 7, 2.6, T.INK);
    rectOutline(w, cx - sw / 2 + 8, top + 8, sw - 16, sh - 16, 7, 2.2, T.INK);
    if (fill) {
      rectOutline(w, cx - sw / 2 + 18, top + 18, sw - 36, sh - 36, 9, 1.2, T.GRAY);
      for (let y = top + 34; y < top + sh - 26; y += 11) for (let x = cx - sw / 2 + 34; x < cx + sw / 2 - 26; x += 11) {
        const dd = Math.hypot((x - cx) / (sw / 2), (y - (top + sh / 2)) / (sh / 2)); w.push(x, y, 0.5 + 1.1 * Math.max(0, dd - 0.35), T.LIGHT);
      }
      const b = 1 + 0.06 * Math.sin(t * 2.2), cy = top + sh / 2;
      for (let j = 0; j < 9; j++) for (let i = 0; i < 9; i++) w.push(cx - 44 + i * 11, cy - 44 + j * 11, (1.5 + 1.6 * (i / 16 + j / 16)) * b, T.ACC);
      for (let p = 0; p < 6; p++) for (let k = 0; k < 2; k++) { const o = -33 + p * 13.2, e = 62 + k * 8; w.push(cx + o, cy - e, 1.6, T.ACC); w.push(cx + o, cy + e, 1.6, T.ACC); w.push(cx - e, cy + o, 1.6, T.ACC); w.push(cx + e, cy + o, 1.6, T.ACC); }
    }
    const by = top + sh + 8, bw0 = sw + 40, bw1 = sw + 150, bh = 70;
    for (let j = 0; j <= 7; j++) {
      const v = j / 7, ww = bw0 + (bw1 - bw0) * v, y = by + v * bh, n = Math.round(ww / 12);
      for (let i = 0; i <= n; i++) {
        const x = cx - ww / 2 + (i * ww) / n, edge = j === 0 || j === 7 || i === 0 || i === n, pad = Math.abs(x - cx) < 90 && j >= 5 && j <= 6;
        w.push(x, y, edge ? 2.1 : pad ? 0.6 : 1.25, edge ? T.INK : T.GRAY);
      }
    }
  }
  function slipPanels(w, t) {
    const ph = t * 1.7, g1 = 430, hx = 1390, hy = 250;
    ground(w, 1010, 1790, g1 + 8, 0, 100, 0.1, 0, T.LIGHT, 1.3);
    for (let k = 70; k >= 1; k--) for (let row = 0; row < 3; row++) {
      const fx = hx + 170 * Math.sin(ph - k * 0.03) + (hash(k * 3 + row) - 0.5) * 10;
      w.push(fx, g1 - 10 + row * 7 + (hash(k + row * 91) - 0.5) * 4, 2.8 * Math.pow(1 - k / 71, 0.8), T.BAD);
    }
    const fx = hx + 170 * Math.sin(ph);
    S.capsule(w, hx, hy, fx, g1 - 16, 15, 26, 5, 2.4, T.INK);
    S.capsule(w, fx - 30, g1 - 7, fx + 30, g1 - 7, 9, 11, 3, 2.4, T.INK);
    S.sphere(w, { x: hx, y: hy, R: 28, n: 160, rot: t, tilt: 0.3, dot: 2.4, tone: T.INK });
    const g2 = 840, ax = 1390 + 70 * Math.sin(ph), ay = 660;
    ground(w, 1010, 1790, g2 + 8, 0, 100, 0.1, 0, T.LIGHT, 1.3);
    S.capsule(w, ax, ay, 1390, g2 - 16, 15, 26, 5, 2.4, T.INK);
    S.capsule(w, 1360, g2 - 7, 1420, g2 - 7, 9, 11, 3, 2.4, T.INK);
    S.sphere(w, { x: ax, y: ay, R: 28, n: 160, rot: t, tilt: 0.3, dot: 2.4, tone: T.INK });
    for (let i = 0; i < 11; i++) for (let r = 0; r < 2; r++) w.push(1350 + i * 8, g2 + 6 + r * 7, 2.4 - r * 0.8, T.ACC);
  }
  function pendulum(w, t) {
    const v = F()['validation.json'];
    const P = v && v.pendulum, ts = P ? P.t : null;
    // swinging pendulum driven by the recorded MacGym angle
    let phi = 60 * Math.cos(t * 2 * Math.PI / 1.805);
    if (P) { const dtS = ts[1] - ts[0], span = ts[ts.length - 1], tt = t % span, i = Math.min(ts.length - 2, Math.floor(tt / dtS)), f = tt / dtS - i; phi = P.phi_deg.ours[i] + (P.phi_deg.ours[i + 1] - P.phi_deg.ours[i]) * f; }
    const px = 1400, py = 150, L = 330, a = (phi * Math.PI) / 180, ex = px + Math.sin(a) * L, ey = py + Math.cos(a) * L;
    for (let i = -10; i <= 10; i++) w.push(px + i * 9, py - 8, 1.6, T.GRAY);
    S.capsule(w, px, py, ex, ey, 14, 34, 4, 2.4, T.INK);
    S.sphere(w, { x: px, y: py, R: 16, n: 90, rot: t, tilt: 0.3, dot: 2.0, tone: T.ACC });
    // arc trace of the swing
    for (let k = 0; k < 41; k++) { const aa = ((-60 + k * 3) * Math.PI) / 180; w.push(px + Math.sin(aa) * (L + 26), py + Math.cos(aa) * (L + 26), 1.1, T.LIGHT); }
    if (!P) return;
    // recorded trajectories: MuJoCo RK4 (gray, big) under MacGym (accent, small)
    const x0 = 1010, x1 = 1810, yc = 820, amp = 120, n = ts.length, tMax = ts[n - 1];
    for (let i = 0; i < n; i++) {
      const x = x0 + (ts[i] / tMax) * (x1 - x0);
      w.push(x, yc - (P.phi_deg.mujoco_rk4[i] / 60) * amp, 3.4, T.GRAY);
    }
    for (let i = 0; i < n; i++) {
      const x = x0 + (ts[i] / tMax) * (x1 - x0), on = (ts[i] / tMax) <= ((t % tMax) / tMax) + 0.002 || STATIC;
      w.push(x, yc - (P.phi_deg.ours[i] / 60) * amp, on ? 1.7 : 0, T.ACC);
    }
    for (let i = 0; i < 60; i++) w.push(x0 + i * ((x1 - x0) / 59), yc, 0.9, T.LIGHT);
  }
  function cliCurve(w, t) {
    const txt = F()['cli_train.txt']; if (!txt) return;
    const pts = [...txt.matchAll(/iter\s+(\d+)\/(\d+)\s+\|\s+distance\s+(-?[\d.]+)/g)].map((m) => [+m[1], +m[2], +m[3]]);
    if (!pts.length) return;
    const x0 = has('docs.png') ? 1430 : 1040, x1 = 1800, yb = 990, hgt = 150, maxD = Math.max(...pts.map((p) => p[2])), N = pts[0][1];
    for (let i = 0; i <= 60; i++) w.push(x0 + (i / 60) * (x1 - x0), yb, 0.9, T.LIGHT);
    const reveal = STATIC ? 1 : Math.min(1, (t % 8) / 5);
    pts.forEach((p, k) => {
      const x = x0 + (p[0] / N) * (x1 - x0), y = yb - (Math.max(0, p[2]) / maxD) * hgt, on = p[0] / N <= reveal;
      for (let j = 0; j < 4; j++) w.push(x, y + j * 6, on ? (j === 0 ? 3.2 : 1.4 - j * 0.3) : 0, k === pts.length - 1 ? T.ACC : T.INK);
    });
  }
  function timeline(w, t) {
    const tl = F()['timeline.json'];
    const toMin = (hhmm) => { const [h, m] = hhmm.split(':').map(Number); return h * 60 + m; };
    const start = 11 * 60 + 38, end = Math.max(toMin(new Date().toTimeString().slice(0, 5)) || 0, tl ? toMin(tl.commits[tl.commits.length - 1].hhmm) + 20 : 15 * 60);
    const x0 = 1010, x1 = 1800, y = 560, X = (m) => x0 + ((m - start) / (end - start)) * (x1 - x0);
    for (let i = 0; i <= 140; i++) w.push(x0 + (i / 140) * (x1 - x0), y, 1.2, T.GRAY);
    for (let hr = 12; hr * 60 <= end; hr++) for (let k = 0; k < 4; k++) w.push(X(hr * 60), y + 12 + k * 6, 1.2, T.GRAY);
    const cs = tl ? tl.commits : [{ hhmm: '11:38' }];
    cs.forEach((c, i) => {
      const x = X(toMin(c.hhmm)), big = i === cs.length - 1, up = i % 2 === 0;
      S.sphere(w, { x, y, R: big ? 22 : 15, n: big ? 160 : 90, rot: t * 0.6 + i, tilt: 0.3, dot: big ? 2.2 : 1.8, tone: i === 0 || big ? T.ACC : T.INK });
      const tier = [0, 0, 0, 0, 0, 1, 1, 0][i] || 0; for (let k = 0; k < 7 + tier * 6; k++) w.push(x, up ? y - 30 - k * 9 : y + 30 + k * 9, 1.0, T.LIGHT);
    });
    // "now" pulse
    const nx = X(end); for (let k = 0; k < 8; k++) w.push(nx, y - 24 + k * 7, 1.3 + 0.6 * Math.sin(t * 4 + k), T.ACC);
  }
  function timelineLabels(ctx, a) {
    const tl = F()['timeline.json']; if (!tl) return;
    const toMin = (hhmm) => { const [h, m] = hhmm.split(':').map(Number); return h * 60 + m; };
    const start = 11 * 60 + 38, end = Math.max(toMin(new Date().toTimeString().slice(0, 5)) || 0, toMin(tl.commits[tl.commits.length - 1].hhmm) + 20);
    const x0 = 1010, x1 = 1800, y = 560, X = (m) => x0 + ((m - start) / (end - start)) * (x1 - x0);
    const short = ['skeleton', 'physics + PPO', 'gait fix', 'replays + UI', 'framework spec', 'public API + CLI', 'docs', 'real physics'];
    const dy = [0, 1, 0, 1, 0, 1, 0, 1], idx = {};
    tl.commits.forEach((c, i) => {
      const tier = [0, 0, 0, 0, 0, 1, 1, 0][i] || 0, x = X(toMin(c.hhmm)), up = i % 2 === 0, yy = up ? y - 104 - tier * 52 : y + 112 + tier * 52;
      label(ctx, c.hhmm, x, yy, a, i === 0 ? '#FF4D00' : '#111', 'center', 'mono');
      label(ctx, (short[i] || c.message.slice(0, 18)).toUpperCase(), x, yy + 18, a, '#9A9A95', 'center');
    });
    label(ctx, 'NOW', X(end), y - 40, a, '#FF4D00', 'center');
    label(ctx, 'GIT LOG · ' + tl.count + ' COMMITS', x0, 330, a, '#9A9A95');
  }
  function classroom(w, t) {
    for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) {
      const cx = 1080 + c * 205, top = 250 + r * 205, sw = 150, sh = 92, k = r * 4 + c;
      rectOutline(w, cx - sw / 2, top, sw, sh, 6.5, 1.7, T.INK);
      for (let i = 0; i <= 26; i++) w.push(cx - sw / 2 - 12 + i * ((sw + 24) / 26), top + sh + 12, 1.7, T.INK);
      const g = gen(3), Fr = S.frameAt(g, t + hash(k) * 7, 'loop');
      S.robot(w, g, { fr: Fr, ox: cx, oy: top + sh - 10, scale: 46, rootWX: Fr[0], spacing: 5, rMax: 1.2, near: k === 5 ? T.ACC : T.INK, far: k === 5 ? T.ACC : T.GRAY });
    }
  }
  function closeMark(w, t) {
    // MacGym mark (left) + repo QR code made of dots (right)
    const cx = 760, cy = 360, size = 330, u = size / 20, x0 = cx - size / 2, y0 = cy - size / 2, br = 1 + 0.012 * Math.sin(t * 1.4);
    const P = (x, y) => [cx + (x0 + x * u - cx) * br, cy + (y0 + y * u - cy) * br];
    for (const inset of [0.5, 1.1]) for (let i = 0; i < 46; i++) {
      const s = i / 46, L = 20 - 2 * inset;
      [[inset + s * L, inset], [20 - inset, inset + s * L], [20 - inset - s * L, 20 - inset], [inset, 20 - inset - s * L]].forEach(([x, y]) => { const q = P(x, y); w.push(q[0], q[1], inset === 0.5 ? 2.4 : 1.0, inset === 0.5 ? T.INK : T.GRAY); });
    }
    const path = [[4, 15], [8, 6], [10, 11], [12, 6], [16, 15]];
    for (let s = 0; s < 4; s++) {
      const A = path[s], B = path[s + 1], len = Math.hypot(B[0] - A[0], B[1] - A[1]), n = Math.round(len * 3.6), nx = -(B[1] - A[1]) / len, ny = (B[0] - A[0]) / len;
      for (let i = 0; i <= n; i++) for (let b = -2; b <= 2; b++) {
        const f = i / n, wave = 0.5 + 0.5 * Math.sin((s + f) * 2.6 - t * 3), q = P(A[0] + (B[0] - A[0]) * f + nx * b * 0.27, A[1] + (B[1] - A[1]) * f + ny * b * 0.27);
        w.push(q[0], q[1], (2.3 + 1.0 * wave) * (1 - Math.abs(b) * 0.12), T.INK);
      }
    }
    for (let j = 0; j < 6; j++) for (let i = 0; i < 6; i++) { const q = P(14.25 + i * 0.5, 3.25 + j * 0.5); w.push(q[0], q[1], 2.6, T.ACC); }
    const M = window.QR_MATRIX; if (!M) return;
    const n = M.length, cell = 330 / n, qx = 1180 - (n * cell) / 2, qy = cy - (n * cell) / 2;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      if (!M[j][i]) continue;
      const finder = (i < 7 && j < 7) || (i >= n - 7 && j < 7) || (i < 7 && j >= n - 7);
      const wave = finder ? 0 : 0.06 * Math.sin(t * 2.5 - (i + j) * 0.3);
      w.push(qx + i * cell + cell / 2, qy + j * cell + cell / 2, cell * (0.5 + wave), finder ? T.ACC : T.INK);
    }
  }

  // ======================= TEXT ASSETS =======================
  var PY_SHORT = [
    'robot = mg.Robot("strider")',
    'torso = robot.body("torso", start=(0, 1), end=(0.6, 1), radius=0.06)',
    'leg   = robot.body("front", start=(0.6, 1), end=(0.6, 0.5), radius=0.04)',
    'robot.joint(torso, leg, at=(0.6, 1), limits=(-1, 1), motor=80)',
    'task  = mg.Task(reward=lambda s: s.root.vx)',
    'env   = mg.make(robot, task, num_envs=4096)',
  ].map((l) => esc(l).replace(/(".*?")/g, '<span class="s">$1</span>').replace(/\b(mg)\./g, '<span class="k">mg</span>.').replace(/\b(lambda)\b/g, '<span class="c">$1</span>')).join('\n');
  function kernelHTML() {
    const k = F()['kernel_excerpt.txt'];
    if (!k) return '<span class="c">// waiting for the generated kernel excerpt (data/kernel_excerpt.txt)</span>';
    return k.split('\n').filter((l) => l.trim() !== '').slice(0, 11).map((l) => esc(l.length > 84 ? l.slice(0, 83) + '…' : l)
      .replace(/\b(kernel|device|const|float|float2|uint|thread_position_in_grid|for|if|return|constant)\b/g, '<span class="k">$1</span>')
      .replace(/(\/\/.*)$/, '<span class="c">$1</span>')).join('\n');
  }
  function termHTML() {
    const txt = F()['cli_train.txt'] || '', ex = F()['cli_export.txt'] || '';
    const L = txt.split('\n').filter(Boolean);
    const keep = L.filter((l, i) => i < 3 || /iter\s+(1|10|20|30|40)\//.test(l) || /^done/.test(l));
    const exl = ex.split('\n').filter((l) => /^\$|exported/.test(l));
    const fmtL = (l) => esc(l.replace(/\s{2,}/g, ' ').replace(/ \| /g, ' · ').replace('/tmp/macgym_cli/', ''))
      .replace(/^(\$.*)$/, '<span class="k">$1</span>').replace(/(distance [-\d.]+ m)/, '<b>$1</b>');
    return [...keep, '', ...exl].map(fmtL).join('\n');
  }
  function profileHTML() {
    const p = F()['profile.json'];
    if (p) {
      const seg = p.pieces_pct || p.segments || p.breakdown || p;
      const entries = Array.isArray(seg) ? seg.map((s) => [s.name || s.stage, +(s.seconds ?? s.ms ?? s.frac ?? s.value)]) : Object.entries(seg).filter(([, v]) => typeof v === 'number');
      const tot = entries.reduce((s, e) => s + e[1], 0);
      if (entries.length && tot > 0) {
        const NAMES = { policy_forward_rollout: 'policy forward', env_step_physics_reward_reset: 'physics (Metal)', rollout_buffer_writes: 'buffer writes', reward_norm_gae_advnorm: 'GAE', ppo_update_backprop_adam: 'PPO update', stats_sync_obsnorm_host_copy: 'sync' };
        const col = (n) => (/ppo_update/.test(n) ? '#FF4D00' : /physics/.test(n) ? '#111' : /policy/.test(n) ? '#8C8C88' : '#CFCFCA');
        entries.sort((a, b) => b[1] - a[1]);
        return `<div class="prof rv" style="transition-delay:.8s"><div class="pcap">ONE TRAINING ITERATION (4,096 WALKERS) · WHERE THE TIME GOES</div><div class="pbar">${entries.map((e) => `<i style="width:${(100 * e[1]) / tot}%;background:${col(e[0])}"></i>`).join('')}</div><div class="plab">${entries.filter((e) => (100 * e[1]) / tot >= 1.5).map((e) => `<span><b style="background:${col(e[0])}"></b>${esc(NAMES[e[0]] || String(e[0]).replace(/_/g, ' '))} ${Math.round((100 * e[1]) / tot)}%</span>`).join('')}</div></div>`;
      }
    }
    return has('profile.png') ? `<img class="panel rv" style="left:1010px;top:800px;width:800px;transition-delay:.8s" src="data/profile.png?v=${V()}">` : '';
  }

  // ---- 3D arm (slide 13) ----
  function arm3d(w, t) {
    const yaw = -0.6 + t * 0.35, tilt = 0.32, s = 520, ox = 1250, oy = 840;
    const sh = 0.55 + 0.22 * Math.sin(t * 0.9), el = -1.25 + 0.3 * Math.sin(t * 1.15 + 1);
    const cy = Math.cos(yaw), sy = Math.sin(yaw), ct = Math.cos(tilt), st = Math.sin(tilt);
    const L = Shapes_LIGHT;
    const proj = (x, y, z) => { // y up
      const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
      const y2 = y * ct - z1 * st, z2 = y * st + z1 * ct;
      const persp = 1 / (1 + z2 * 0.35);
      return [ox + x1 * s * persp, oy - y2 * s * persp, z2, persp, x1, y2];
    };
    const rot = (v) => { const x1 = v[0] * cy + v[2] * sy, z1 = -v[0] * sy + v[2] * cy; const y2 = v[1] * ct - z1 * st, z2 = v[1] * st + z1 * ct; return [x1, y2, z2]; };
    // floor grid
    for (let i = -6; i <= 6; i++) for (let j = -6; j <= 6; j++) { const p = proj(i * 0.06, 0, j * 0.06); w.push(p[0], p[1], 1.1 * p[3], T.LIGHT); }
    function link(a, b, rad, nAlong, nRing, tone) {
      const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], Ln = Math.hypot(...d), dn = d.map((v) => v / Ln);
      let up = Math.abs(dn[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      let u = [dn[1] * up[2] - dn[2] * up[1], dn[2] * up[0] - dn[0] * up[2], dn[0] * up[1] - dn[1] * up[0]]; const ul = Math.hypot(...u); u = u.map((v) => v / ul);
      const v2 = [dn[1] * u[2] - dn[2] * u[1], dn[2] * u[0] - dn[0] * u[2], dn[0] * u[1] - dn[1] * u[0]];
      for (let i = 0; i < nAlong; i++) {
        const f = i / (nAlong - 1);
        for (let k = 0; k < nRing; k++) {
          const th = (k / nRing) * Math.PI * 2, c = Math.cos(th), sn = Math.sin(th);
          const nrm = [u[0] * c + v2[0] * sn, u[1] * c + v2[1] * sn, u[2] * c + v2[2] * sn];
          const P = [a[0] + d[0] * f + nrm[0] * rad, a[1] + d[1] * f + nrm[1] * rad, a[2] + d[2] * f + nrm[2] * rad];
          const p = proj(P[0], P[1], P[2]), n2 = rot(nrm);
          if (n2[2] < -0.05) { w.push(p[0], p[1], 0, tone); continue; }
          const lam = Math.max(0, n2[0] * L[0] + -n2[1] * L[1] + n2[2] * L[2]);
          w.push(p[0], p[1], (0.5 + 2.6 * (1 - lam)) * p[3], tone);
        }
      }
    }
    const base = [0, 0, 0], shoulder = [0, 0.22, 0];
    const elbow = [Math.cos(sh) * 0.42 * 1, 0.22 + Math.sin(sh) * 0.42, 0];
    const a2 = sh + el, wrist = [elbow[0] + Math.cos(a2) * 0.36, elbow[1] + Math.sin(a2) * 0.36, 0];
    const tip = [wrist[0] + Math.cos(a2 - 0.6) * 0.1, wrist[1] + Math.sin(a2 - 0.6) * 0.1, 0];
    link(base, shoulder, 0.09, 8, 22, T.INK);
    link(shoulder, elbow, 0.045, 18, 14, T.INK);
    link(elbow, wrist, 0.038, 15, 12, T.INK);
    link(wrist, [tip[0], tip[1], 0.03], 0.016, 5, 6, T.ACC);
    link(wrist, [tip[0], tip[1], -0.03], 0.016, 5, 6, T.ACC);
    // phone + signal
    const px = 1730, py = 520, pw = 96, ph = 190;
    rectOutline(w, px - pw / 2, py - ph / 2, pw, ph, 7, 2.0, T.ACC);
    for (let j = 0; j < 6; j++) for (let i = 0; i < 4; i++) w.push(px - 18 + i * 12, py - 30 + j * 12, 1.4 + 0.8 * Math.sin(t * 3 + i + j), T.ACC);
    const wp = proj(wrist[0], wrist[1], wrist[2]);
    for (let k = 0; k < 26; k++) {
      const f = ((k / 26) + t * 0.25) % 1, x = px - pw / 2 - 20 + (wp[0] + 30 - (px - pw / 2 - 20)) * f, y = py + (wp[1] - py) * f - Math.sin(f * Math.PI) * 60;
      w.push(x, y, 1.6 * Math.sin(f * Math.PI), T.ACC);
    }
  }
  const Shapes_LIGHT = (() => { const v = [-0.55, -0.62, 0.56], l = Math.hypot(...v); return v.map((x) => x / l); })();

  // ---- canvas labels ----
  let CTX_SCALE = 1;
  function label(ctx, text, x, y, a, color, align = 'left', kind = 'small') {
    if (a <= 0.01) return;
    ctx.globalAlpha = a; ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = 'alphabetic';
    if (kind === 'big') ctx.font = '400 40px "Bebas Neue"';
    else if (kind === 'mono') ctx.font = '500 17px "JetBrains Mono"';
    else ctx.font = '700 12.5px "DM Sans"';
    if (kind === 'small' && 'letterSpacing' in ctx) ctx.letterSpacing = '2.5px'; else if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
    ctx.fillText(text, x, y);
    ctx.globalAlpha = 1;
  }

  let DIGITS = null;

  // ======================= ENGINE =======================
  const stage = document.getElementById('stage'), canvas = document.getElementById('field'), ctx = canvas.getContext('2d');
  const textLayer = document.getElementById('text'), pager = document.getElementById('pager'), notesEl = document.getElementById('notes');
  const Wr = new S.Writer(9000);
  const px = new Float32Array(N), py = new Float32Array(N), pr = new Float32Array(N), pc = new Uint8Array(N);
  const fx = new Float32Array(N), fy = new Float32Array(N), fr = new Float32Array(N), fc = new Uint8Array(N);
  const dl = new Float32Array(N), bend = new Float32Array(N);
  let cur = 0, step = 0, slideT0 = 0, transT0 = 0, needDelays = true, curEl = null;

  for (let i = 0; i < N; i++) { // initial scatter: soft cloud
    const a = hash(i) * Math.PI * 2, rr = Math.sqrt(hash(i + 7)) * 700;
    px[i] = 1390 + Math.cos(a) * rr; py[i] = 540 + Math.sin(a) * rr * 0.7; pr[i] = 0; pc[i] = T.LIGHT;
    bend[i] = (hash(i + 3) - 0.5) * 140;
  }

  function fit() {
    const vw = innerWidth, vh = innerHeight, k = Math.min(vw / W, vh / H);
    stage.style.transform = `translate(${-W * k / 2}px, ${-H * k / 2}px) scale(${k})`;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    CTX_SCALE = k * dpr;
    canvas.width = Math.round(W * k * dpr); canvas.height = Math.round(H * k * dpr);
  }

  function buildText(i) {
    const sl = SLIDES[i];
    const el = document.createElement('div');
    el.className = 'slide ' + (sl.cls || '') + ' step' + step;
    el.innerHTML = `<div class="col">${sl.html()}</div>`;
    // pre-col elements that must be positioned absolutely (code block)
    el.querySelectorAll('.codecap, .code, .panel, .prof, .screen, .screencap').forEach((n) => el.appendChild(n));
    const vid = el.querySelector('#pgvid'); if (vid) setupVideo(vid);
    return el;
  }

  function go(i, st = 0, instant = false) {
    i = Math.max(0, Math.min(SLIDES.length - 1, i));
    const slideChanged = i !== cur || !curEl;
    // snapshot current particle state as morph origin
    fx.set(px); fy.set(py); fr.set(pr); fc.set(pc);
    const now = performance.now() / 1000;
    transT0 = now; needDelays = true;
    if (slideChanged) {
      cur = i; step = st; slideT0 = now;
      if (curEl) { const old = curEl; old.classList.add('off'); old.classList.remove('on'); setTimeout(() => old.remove(), 320); }
      curEl = buildText(i); textLayer.appendChild(curEl);
      if (instant) curEl.classList.add('on'); else requestAnimationFrame(() => requestAnimationFrame(() => curEl.classList.add('on')));
      if (!STATIC && location.hash !== '#' + (i + 1)) history.replaceState(null, '', '#' + (i + 1));
    } else {
      step = st; curEl.classList.remove('step0', 'step1'); curEl.classList.add('step' + st);
    }
    pager.textContent = String(cur + 1).padStart(2, '0') + ' / ' + SLIDES.length;
    notesEl.innerHTML = `<div class="t">Notes · slide ${cur + 1}</div>${SLIDES[cur].notes}`;
  }
  function next() { const s = SLIDES[cur]; if (step < (s.steps || 1) - 1) go(cur, step + 1); else if (cur < SLIDES.length - 1) go(cur + 1, 0); }
  function prev() { if (step > 0) go(cur, step - 1); else if (cur > 0) go(cur - 1, (SLIDES[cur - 1].steps || 1) - 1); }

  const spring = (u) => (u >= 1 ? 1 : 1 - Math.exp(-6.5 * u) * Math.cos(9.5 * u) * (1 - u * u * u));

  function frame(nowMs, forcedT) {
    const now = nowMs / 1000;
    const sl = SLIDES[cur];
    const t = forcedT != null ? forcedT : now - slideT0;
    Wr.reset(); sl.shape(Wr, t, step);
    const n = Math.max(1, Wr.n), X = Wr.X, Y = Wr.Y, R = Wr.R, C = Wr.C;
    const tt = forcedT != null ? 99 : now - transT0;
    if (needDelays) {
      for (let i = 0; i < N; i++) {
        const j = ((i * n) / N) | 0;
        dl[i] = 0.34 * S.clamp((X[j] - 760) / 1150, 0, 1) + 0.26 * hash(i * 1.37 + cur);
      }
      needDelays = false;
    }
    for (let i = 0; i < N; i++) {
      const j = ((i * n) / N) | 0;
      const u = (tt - dl[i]) / DUR;
      if (u >= 1) { px[i] = X[j]; py[i] = Y[j]; pr[i] = R[j]; pc[i] = C[j]; continue; }
      if (u <= 0) { px[i] = fx[i]; py[i] = fy[i]; pr[i] = fr[i]; pc[i] = fc[i]; continue; }
      const s = spring(u), dx = X[j] - fx[i], dy = Y[j] - fy[i], len = Math.hypot(dx, dy) || 1;
      const arc = Math.sin(Math.PI * Math.min(1, u * 1.3)) * bend[i] * Math.min(1, len / 400);
      px[i] = fx[i] + dx * s + (-dy / len) * arc;
      py[i] = fy[i] + dy * s + (dx / len) * arc;
      const rs = Math.min(1, u * 1.6);
      pr[i] = (fr[i] + (R[j] - fr[i]) * rs) * (1 - 0.35 * Math.sin(Math.PI * Math.min(1, u * 1.4)));
      pc[i] = u > 0.42 ? C[j] : fc[i];
    }
    // draw
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(CTX_SCALE, 0, 0, CTX_SCALE, 0, 0);
    for (let tone = 0; tone < 5; tone++) {
      ctx.fillStyle = S.TONE_COLORS[tone]; ctx.beginPath();
      for (let i = 0; i < N; i++) {
        if (pc[i] !== tone) continue; const r = pr[i]; if (r < 0.2) continue;
        ctx.moveTo(px[i] + r, py[i]); ctx.arc(px[i], py[i], r, 0, 6.2832);
      }
      ctx.fill();
    }
    if (sl.draw) { const a = forcedT != null ? 1 : S.clamp((tt - 0.55) / 0.5, 0, 1); sl.draw(ctx, t, a, step); }
  }

  function loop(ms) { frame(ms); requestAnimationFrame(loop); }

  // ---- controls ----
  addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown' || e.key === 'Enter') { e.preventDefault(); next(); }
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp' || e.key === 'Backspace') { e.preventDefault(); prev(); }
    else if (e.key === 'f' || e.key === 'F') { if (!document.fullscreenElement) document.documentElement.requestFullscreen(); else document.exitFullscreen(); }
    else if (e.key === 'p' || e.key === 'P') notesEl.classList.toggle('show');
    else if (e.key === 'l' || e.key === 'L') window.open('/playground', '_blank');
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(SLIDES.length - 1);
  });
  addEventListener('click', (e) => { if (e.clientX < innerWidth * 0.15) prev(); else next(); });
  addEventListener('hashchange', () => { const k = parseInt(location.hash.slice(1), 10); if (k && k - 1 !== cur) go(k - 1); });
  addEventListener('resize', fit);

  // ---- evidence loader: polls deck/data/ so slides fill in as files land ----
  const DATA_FILES = ['validation.json', 'validation.png', 'fair_bench.json', 'profile.json', 'profile.png', 'seeds.json', 'seeds.png', 'tripod_replay.json', 'kernel_excerpt.txt', 'timeline.json', 'tests.json', 'tests.txt', 'cli_train.txt', 'cli_export.txt', 'playground_tripod.mp4', 'playground_tripod_4x.mp4', 'playground_edit.mp4', 'docs.png', 'pg_t00.png', 'pg_t15.png', 'pg_t30.png', 'pg_t50.png', 'recordings.json', 'viewer.png'];
  async function loadData() {
    const f = {}, v = Date.now();
    await Promise.all(DATA_FILES.map(async (n) => {
      try {
        if (/\.(png|mp4)$/.test(n)) { const r = await fetch('data/' + n + '?v=' + v, { method: 'HEAD' }); if (r.ok) f[n] = true; return; }
        const r = await fetch('data/' + n + '?v=' + v); if (!r.ok) return;
        const t = await r.text(); f[n] = /\.json$/.test(n) ? JSON.parse(t) : t;
      } catch (e) { /* not there yet / half-written */ }
    }));
    const sig = Object.keys(f).sort().join(',') + JSON.stringify(Object.values(f).map((x) => (typeof x === 'string' ? x.length : typeof x === 'object' ? JSON.stringify(x).length : 1)));
    const changed = sig !== DATA.sig; DATA.f = f; DATA.sig = sig; if (changed) DATA.ver = v;
    return changed;
  }
  function setupVideo(vid) {
    const list = vid.dataset.list.split(','); let i = 0;
    vid.addEventListener('ended', () => { i = (i + 1) % list.length; vid.src = 'data/' + list[i] + '?v=' + DATA.ver; vid.play().catch(() => {}); });
    if (STATIC) { vid.removeAttribute('autoplay'); vid.addEventListener('loadedmetadata', () => { vid.currentTime = Math.min(vid.duration * 0.45, 6); }, { once: true }); }
  }
  // ---- boot ----
  async function boot() {
    fit();
    const [rep, bench] = await Promise.all([
      fetch('/docs/replay.json').then((r) => r.json()),
      fetch('/bench.json?' + Date.now()).then((r) => r.json()),
    ]);
    DATA.gens = rep.generations; S.REPLAY.dt = rep.dt;
    DATA.bench = summarizeBench(bench); DATA.benchRaw = bench;
    await loadData();
    await Promise.all([document.fonts.load('400 100px "Bebas Neue"'), document.fonts.load('700 12px "DM Sans"'), document.fonts.load('500 16px "JetBrains Mono"')]);
    await document.fonts.ready;

    if (STATIC) {
      document.body.classList.add('static');
      const k = parseInt(qs.get('slide') || '1', 10) - 1;
      const sl = SLIDES[k];
      go(k, (sl.steps || 1) - 1, true);
      const vid = document.querySelector('#pgvid');
      if (vid) await new Promise((res) => { vid.addEventListener('seeked', res, { once: true }); setTimeout(res, 4000); });
      await Promise.all([...document.querySelectorAll('.slide img')].map((im) => (im.complete ? 0 : new Promise((r) => { im.onload = im.onerror = r; }))));
      frame(performance.now(), sl.staticT || 3.0);
      window.__ready = true;
      return;
    }
    const k = parseInt(location.hash.slice(1), 10);
    go(k ? k - 1 : 0, 0);
    window.__ready = true;
    requestAnimationFrame(loop);
    setInterval(async () => { if (await loadData()) { const st = step; curEl && curEl.remove(); curEl = null; go(cur, st, true); } }, 12000);
  }
  window.__deck = { go, next, prev, SLIDES, DATA, count(k, st = 0, t = 2) { Wr.reset(); SLIDES[k].shape(Wr, t, st); return Wr.n; }, get cur() { return cur; }, get step() { return step; } };
  boot();
})();
