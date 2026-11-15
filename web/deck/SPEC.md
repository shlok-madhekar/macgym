# MacGym pitch deck spec

Purpose: the pitch deck for MacGym at Bay-Valley Hacks (theme "Elevate the Ordinary"; judged on Creativity & Originality, Technical Difficulty, Functionality, UI/UX, Theme Relevance, Presentation & Pitch). Used live in judging AND recorded for the demo video. Goal: first overall.

## Style: inspired by Shlok's Syro deck (see /Users/shlok/.aside/u/0/sessions/2026-11-15_S93FPPu5Bqvr1dNz/tmp/syro_grid.png and syro0..12.png — LOOK AT THEM)
- White/paper background (#FAFAF7), near-black ink #111. Headline in a tall condensed display face (Bebas Neue, Google Fonts), huge, ALL CAPS, 2-4 short lines, the punchline line in gray (#BDBDBD) or in the accent. Tiny tracked uppercase kicker label above the headline ("THE PROBLEM", "SO I BUILT"). Body facts as 3-4 hairline-separated short rows in small DM Sans/Inter, key word bold, last row in accent.
- ONE accent: MacGym orange #FF4D00 (matches product UI). Optional muted red for "bad" items.
- Left half text, right half a HALFTONE DOT visual (thousands of small dots) — the signature of Syro. Dots must ANIMATE ("hella sick animations"): one persistent canvas particle field (~3,000-6,000 dots) that MORPHS between shapes on every slide transition (spring/eased interpolation, staggered), with subtle idle motion (breathing, drift, rotation). Shapes are generated procedurally or from real data (see below). Dot size varies for halftone shading (3D sphere shading like Syro).
- Restraint: no gradients, no glass, no emoji, no icons-in-circles, no stock imagery, no fake numbers. Every number on a slide must be real (listed below).

## Real data available (USE IT; don't invent numbers)
- web/../site/replay.json (served at /docs/replay.json): real recorded walker replays from a from-scratch training run: generations 0:00 "0.8 m in 1.0 s (fell)", 0:10 "1.5 m in 1.3 s (fell)", 0:30 "15.1 m in 8.0 s", 1:00 "21.5 m in 8.0 s". frames = arrays of K=7 capsules [x1,y1,x2,y2,r] in metres, dt 0.04 s. Draw robots AS DOTS (sample points along each capsule) so they live in the particle field.
- Live playground observation (screenshot, real): tripod preset, 4,096 copies, "3.2 days of tripod experience in 0:43 of wall time", average distance per finished episode went from -0.06 m to 7.20 m within ~45 s.
- Verified training (real, this afternoon, new physics): walker average episode distance -0.16 m -> 31.6 m in 57 s of training; tripod -0.13 m -> 21.7 m in 44 s. 4,096 robots in parallel. Training throughput (sim + PPO) ~280-315K env-steps/s; sim-only ~2-5M env-steps/s at N=4096.
- Physics honesty: first version gaits were too fast (walker 6.16 m/s, tripod 5.23 m/s) because motors had unlimited power and feet skated; fixed with warm-started contact impulses + DC-motor torque-speed curve. After: walker 2.74 m/s, tripod 1.32 m/s, cheetah 3.76 m/s; cheetah stance-foot slip median 0.735 -> 0.053 m/s. 
- Benchmark: will be in web/bench.json (re-running now; read it at the end and use its numbers + caveat: our engine is a 2D maximal-coordinate impulse solver vs MuJoCo's generalized-coordinate solver; walker compared against MuJoCo HalfCheetah on all 11 CPU cores). Earlier clean run: walker 21.4x, cheetah 22.4x, cartpole 37.4x faster than MuJoCo on all 11 cores.
- Ecosystem gap (from research): Isaac Lab, MuJoCo Warp, ManiSkill GPU are NVIDIA-only; jax-metal declared dead by a JAX maintainer (Jan 2026); best community Metal port ~5.2k env-steps/s on M3 Max.
- Framework: Python API (mg.Robot, mg.Task, mg.make, mg.train, mg.evaluate, mg.load_policy), auto-generated Metal kernel per robot ("compiled Metal kernel for 7 bodies / 6 joints in 0.26 s"), PPO in MLX, CLI (macgym train/view/export/new/bench), CoreML + ONNX export verified (max error 3.6e-7 CoreML, 4.8e-7 ONNX), Gymnasium wrapper, code Playground, docs site. Built-in robots: walker, cheetah, hopper, tripod, worm (+ arm, cartpole). Built today: ~8,100 lines (4,200 in the Python package), 7 pytest tests + 6 test scripts, 5 examples, git history starts 11:38 AM today.
- The 15-line snippet from site/index.html (strider robot).

## Slides (12-14). Kicker / headline / facts / dot visual
1. Cold open: dots assemble into a walker robot (real replay frames, 1:00 generation) running across; title "MACGYM" + "Robot learning on the Mac you already own."
2. THE PROBLEM — "EVERY ROBOT SIM NEEDS AN NVIDIA GPU." gray: "YOUR MAC ISN'T INVITED." facts: Isaac Lab: NVIDIA only / MuJoCo Warp: NVIDIA only / jax-metal: "consider it dead" (JAX maintainer, Jan 2026) / best Mac port: ~5.2K steps/s. Visual: dots form a GPU chip / or a locked grid.
3. "THE ORDINARY LAPTOP" — theme slide: "THE LAPTOP YOU ALREADY OWN HAS A GPU. NOBODY USES IT FOR THIS." Visual: dots form a laptop.
4. SO I BUILT — "MACGYM" one-liner: define a robot in Python, train 4,096 copies on your Mac's GPU in about a minute, export to CoreML. Visual: one dot robot splits into a grid of many.
5. HOW IT WORKS — 4 rows: 15 lines of Python -> auto-generated Metal kernel (one GPU thread per robot) -> PPO in MLX, unified memory, zero copies -> CoreML / ONNX export. Visual: dots flow through 4 stations.
6. THE NUMBER — "3.2 DAYS OF EXPERIENCE. 43 SECONDS." facts: 4,096 copies / one M3 Pro / -0.06 m -> 7.2 m. Visual: crowd of dot robots streaming right.
7. IT LEARNS — generations 0:00 fell / 0:10 fell / 0:30 15.1 m / 1:00 21.5 m drawn from replay.json as dot robots in 4 lanes, animated, newest in accent.
8. IT CAN'T CHEAT — physics honesty: before/after speeds and foot slip; "real motors slow down. ours do too." Visual: dot foot planted vs skating trail.
9. FAST — benchmark headline (from bench.json) with dot bars and a tiny honest caveat line.
10. IT'S A FRAMEWORK — pip install / CLI / Gymnasium / CoreML export verified 3.6e-7; visual: the 15-line code drawn as dots or shown crisp in mono next to dots.
11. BUILT TODAY — big dot-number (like Syro's 116): 8,100 lines; facts rows (robots, tests, examples, start time 11:38 AM).
12. WHO IT'S FOR — students, hackers, researchers on MacBooks; RL education; iterate morphology/reward before scaling to cloud.
13. WHAT'S NEXT — 3D, MuJoCo sim-to-sim check, CoreML policy on iPhone driving a real SO-101 arm.
14. Close: dots form the MacGym mark / a walker; "ELEVATE THE ORDINARY." small: macgym · github link placeholder.

## Tech
- Location: /Users/shlok/macgym/web/deck/ (served by the running MacGym server at http://localhost:8765/deck/index.html). No build step: index.html + deck.js + style.css (+ shapes.js). Canvas 2D at devicePixelRatio, 60fps.
- Controls: → / space / click next, ← back, F fullscreen, P presenter notes overlay. Hash routing #1..#14. Builds within a slide allowed (step-reveal) but keep each slide to <= 2 steps.
- `?static=1&slide=N` renders slide N in its settled final state instantly (for PNG export).
- 16:9 letterboxed stage that scales to any window (1920x1080 design size).
- Speaker notes per slide (casual, confident, 16yo founder voice, no em dashes) in a notes object; also write web/deck/NOTES.md.
- Export: write a script web/deck/export.mjs is NOT possible (no node playwright guaranteed). Instead, use your repl browser tool to capture each slide at 1920x1080 with ?static=1&slide=N into web/deck/export/slide-NN.png, plus 3 mid-animation frames for slides 1, 6, 7 (export/anim-NN-a.png...). Also build web/deck/export/contact.png (grid of all slides) with python cv2.
