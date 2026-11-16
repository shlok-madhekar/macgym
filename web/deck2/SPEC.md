# MacGym deck v2: cinematic rebuild

User verdict on v1 (web/deck): "not animated and looks bad. make it better and further." Live capture of v1 shows the real problems:
- Visuals are small, stuck in a box on the right, with huge empty paper around them. Body text is tiny and dense.
- Transitions turn into random sprays of noise mid-flight (dots scatter everywhere) instead of a coherent, satisfying morph.
- Robots are thin, 1-2 dots wide, and read as fragile scribbles.
- Rounds of judge feedback stuffed slides with charts, tables and screenshots (matplotlib-looking PNGs), which killed the Syro look.
- He also looked at a static export / PDF, so "not animated" partly came from that. v2 must be undeniably alive in every frame, and must also be delivered as a rendered MP4 so it's animated wherever he opens it.

Reference he loves: his Syro deck (/Users/shlok/.aside/u/0/sessions/2026-11-15_S93FPPu5Bqvr1dNz/tmp/syro_grid.png, syro0..12.png): white, huge black condensed caps with a gray punchline, big bold 3D halftone dot spheres. Keep that DNA but go further: cinematic, Apple-keynote-grade motion.

## Engine: WebGL, not canvas 2D
- three.js (importmap from unpkg three@0.170.0) with ONE THREE.Points cloud of 40,000-80,000 particles, custom ShaderMaterial: per-particle attributes for source and target positions (vec3), size, color, and stagger delay. Morph happens on the GPU (uniform uProgress; per-particle eased progress with delay; curved paths via a mid-point control offset along a smooth noise field so dots FLOW in streams/ribbons, never random spray). Size attenuation and depth shading give real 3D halftone spheres (bigger/darker dots on the shadow side).
- Real 3D: shapes are 3D point clouds (spheres, robots extruded/sampled as 3D capsules with shading, laptop, grids of thousands of robots in perspective, terrain). Slow continuous camera drift and orbit on every slide, plus per-slide camera moves (dolly in, pan along a running robot, pull back to reveal 4,096 robots).
- Text: HTML overlay. Headlines in Bebas Neue, huge (120-200 px at 1920 design), with masked line-by-line slide-up reveals; numbers COUNT UP; the punchline line in gray or orange. Max ~12 words of body per slide, at 28px+. Nothing smaller than 18px except one tiny kicker label.
- 60fps at 1920x1080 on the M3 Pro. Stage fills the window (cover/letterbox correctly, no gray margins at 16:9).
- Everything is either moving or breathing at all times. No dead frames.

## Story (10 main slides + appendix). Every number is real (copy from web/deck/NOTES.md canonical line and web/deck/data/*.json; never invent)
1. COLD OPEN: black-on-paper void; particles assemble into one big 3D walker robot (use REAL replay frames from /site/replay.json, generation 1:00, scaled huge, capsules as thick shaded 3D tubes of dots) running toward camera / across; camera tracks it. Title "MACGYM" slams in, sub "An ordinary MacBook → a robot-learning lab."
2. THE ORDINARY THING: particles become a huge 3D MacBook (lid open, rotating slowly); the GPU chip inside glows orange but dim; "THE LAPTOP ON EVERY DESK HERE HAS A GPU." gray "FOR ROBOTS, IT SITS IDLE." one line: "fast robot-RL simulators need NVIDIA. Apple's GPU has no maintained backend."
3. SO I BUILT: the laptop screen flares, camera dives into the chip, and the chip explodes outward into a sea of 4,096 tiny robots in perspective (instanced from replay poses, staggered gait phases), camera pulls back to reveal the scale. "MACGYM." sub "Write a robot in Python. Train 4,096 copies on your Mac's GPU. In about a minute." tag "planar 2D, on purpose".
4. HOW: three beats on one slide with step reveals: code (3 big lines, typed on) → dots stream from the code into a grid of 4,096 GPU threads (a glowing lattice) → out comes a walking robot. Caption words: "describe it" → "one GPU thread per robot" → "it learns to walk".
5. IT LEARNS: the money shot. 4 lanes become one: real replays 0:00 (falls) → 0:10 (falls) → 0:30 → 1:00 (runs 21.5 m in 8 s) played as big dot robots side by side or overlaid as ghost generations, a live counter of meters; headline "FROM FALLING TO RUNNING. IN ONE MINUTE." small "5 of 5 seeds learned".
6. THE NUMBER: "3.2 DAYS OF SIMULATED PRACTICE." orange "28 SECONDS." A giant particle clock/odometer whose digits are made of dots counting up 0 → 3.2 days while the wall clock counts 0 → 28 s; robots streaming past underneath.
7. THE TWIST: "MY FIRST ROBOTS CHEATED." big dot robot sprints absurdly (6.16 m/s) with feet SKATING (red smear trails) → freeze frame, red "6.16 m/s" struck through → physics fix → planted feet, honest gait "2.74 m/s". One line: "infinite motor power + skating feet. Fixed with a real motor curve and contacts that hold."
8. IT'S REAL (merge validation + speed + transfer into ONE clean slide of three huge numbers that count up, each with a 4-word label, and a tiny honest caption): "0.01%" pendulum timing vs MuJoCo · "22×" faster than MuJoCo's fastest CPU mode (same 2D walker, 4,096 robots) · "90%" of its distance when the trained robot is dropped into MuJoCo (19.7 m vs 22.0 m). Visual: a swinging 3D dot pendulum ghosted against a second one, synchronized.
9. IT'S A FRAMEWORK: "NOT A DEMO. A LIBRARY." the terminal types itself (`pip install -e .` / `macgym train walker` / real output lines from web/deck/data/fresh_install.json) while dots form the walker being trained; big "87 s" from clone to trained robot.
10. CLOSE: all particles converge into the 3D laptop again, now its screen is full of running robots (payoff of slide 2), then morph into the QR code (dots) beside "ELEVATE THE ORDINARY." Live demo cue: "press L".
APPENDIX (hidden unless B is pressed, or after the close): clean, readable backup slides with the evidence for Q&A: validation chart, speed breakdown (135K → 498K → 16.3M, ×3.7 × ×32.8 = 121×; 22×/40× vs MuJoCo C rollout 165K), seeds curve, robustness table, who it's for (RL classes on MacBooks, DVHS CS Club, Colab comparison), built today (11:38 start, agents wrote most code, what Shlok decided). Same typographic system, still with subtle particle motion behind.

## Modes
- Keys: → / space / click next, ← back, F fullscreen, L opens /playground in a new tab, B appendix, A autoplay (cinematic auto-advance with per-slide durations ~6-9 s, for recording).
- Speaker notes: reuse/adapt web/deck/NOTES.md into web/deck2/NOTES.md (casual confident voice, no em dashes).
- ?static=1&slide=N for settled-frame PNG export.

## Deliverables
- web/deck2/index.html, engine.js (WebGL particles + camera + morph), shapes3d.js (point-cloud generators, robot-from-replay sampler), slides.js (one object per slide: text, target shape fn, camera path, idle motion, step beats), style.css, NOTES.md.
- A rendered video of the autoplay run: web/deck2/export/macgym-deck.mp4 (1920x1080, 30 or 60 fps, H.264 yuv420p, +faststart). Render deterministically by stepping time via a ?render=1&t=<sec> or window.__deck.renderAt(t) API and capturing canvas frames (preferred: canvas.toDataURL per frame from page.evaluate, or screenshots), assemble with /opt/homebrew/bin/ffmpeg. Also export/slide-NN.png and export/contact.png, plus mid-transition frames.
- The MacGym server already serves web/ at http://localhost:8765/ so the deck is at http://localhost:8765/deck2/index.html.
