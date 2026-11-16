// main.js : boot, controls, autoplay, deterministic render API.
import { Engine } from './engine.js';
import { loadReplay } from './shapes3d.js';
import { buildSlides } from './slides.js';

const Q = new URLSearchParams(location.search);
await Promise.all([document.fonts.load('600 92px "Geist"'), document.fonts.load('500 60px "Geist Mono"')]).catch(() => {}); await document.fonts.ready.catch(() => {});
await loadReplay('data/replay.json');
const slides = buildSlides();
const main = slides.filter((s) => !s.appendix).map((s) => slides.indexOf(s));
const apx = slides.filter((s) => s.appendix).map((s) => slides.indexOf(s));
const eng = new Engine(document.getElementById('stage'), document.getElementById('gl'), slides);

// warm up: build every state's slot mapping (Hilbert sort) up front so no frame hitches later
for (const s of slides) for (const st of s.states) eng.mapping(st, st.at + 0.01);
const offsets = []; let acc = 0; for (const i of main) { offsets.push(acc); acc += slides[i].dur; }
const TOTAL = acc;
let cur = 0, start = performance.now(), from = null, auto = Q.has('autoplay'), T0 = performance.now();
const pager = document.getElementById('pager');

function go(i) {
  i = Math.max(0, Math.min(slides.length - 1, i)); if (i === cur) return;
  const tl = (performance.now() - start) / 1000; from = { si: cur, tl0: tl }; cur = i; start = performance.now();
  history.replaceState(null, '', '#' + (cur + 1));
}
function next() { const order = slides[cur].appendix ? apx : main; const k = order.indexOf(cur); if (k < order.length - 1) go(order[k + 1]); else if (!slides[cur].appendix) go(apx[0]); }
function prev() { const order = slides[cur].appendix ? apx : main; const k = order.indexOf(cur); if (k > 0) go(order[k - 1]); else if (slides[cur].appendix) go(main[main.length - 1]); }
function setPager() { const s = slides[cur]; pager.textContent = s.appendix ? `A${apx.indexOf(cur) + 1}` : `${String(main.indexOf(cur) + 1).padStart(2, '0')} / ${String(main.length).padStart(2, '0')}`; }

window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { e.preventDefault(); next(); }
  else if (e.key === 'ArrowLeft' || e.key === 'PageUp') prev();
  else if (e.key === 'f' || e.key === 'F') { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
  else if (e.key === 'l' || e.key === 'L') window.open(/^(localhost|127\.0\.0\.1)$/.test(location.hostname) ? '/playground.html' : 'https://github.com/shlok-madhekar/macgym', '_blank');
  else if (e.key === 'b' || e.key === 'B') go(slides[cur].appendix ? main[main.length - 1] : apx[0]);
  else if (e.key === 'a' || e.key === 'A') { auto = !auto; if (auto) { go(main[0]); } }
  else if (e.key === 'Home') go(main[0]); else if (e.key === 'End') go(main[main.length - 1]);
});
window.addEventListener('click', () => next());

const h = parseInt(location.hash.slice(1)); if (h > 0 && h <= slides.length) { cur = h - 1; }

// deterministic: global autoplay time T -> frame
function renderAt(T) {
  T = Math.max(0, Math.min(TOTAL - 1e-4, T)); let k = 0; for (let i = 0; i < main.length; i++) if (T >= offsets[i]) k = i;
  const si = main[k], tl = T - offsets[k]; const fr = k > 0 ? { si: main[k - 1], tl0: slides[main[k - 1]].dur } : null;
  cur = si; setPager(); eng.draw({ si, tl }, fr, T);
}
function renderSlide(n, tl) { const si = n - 1, k = main.indexOf(si); const fr = k > 0 ? { si: main[k - 1], tl0: slides[main[k - 1]].dur } : null; cur = si; setPager(); eng.draw({ si, tl }, fr, (k >= 0 ? offsets[k] : 0) + tl); }
window.__deck = { renderAt, renderSlide, TOTAL, offsets, main, slides, fps: () => { const a = eng.fps.acc; if (!a.length) return 0; const m = a.reduce((s, x) => s + x, 0) / a.length; return 1000 / m; }, frameTimes: () => eng.fps.acc.slice(), ready: true };

if (Q.has('render')) { document.body.classList.add('render'); renderAt(+Q.get('t') || 0); }
else if (Q.has('static')) { const n = +Q.get('slide') || 1; renderSlide(n, Q.has('t') ? +Q.get('t') : slides[n - 1].dur - 0.3); }
else {
  const loop = () => {
    const now = performance.now(); let tl = (now - start) / 1000;
    if (auto && !slides[cur].appendix && tl > slides[cur].dur) { const k = main.indexOf(cur); if (k < main.length - 1) { go(main[k + 1]); tl = 0; } }
    setPager(); eng.draw({ si: cur, tl: (performance.now() - start) / 1000 }, from, (now - T0) / 1000);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
