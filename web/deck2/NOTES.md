# MacGym deck v2: speaker notes

Open http://localhost:8765/deck2/index.html. Keys: → / space / click = next, ← = back, F = fullscreen, **L = live playground in a new tab (localhost) or the GitHub repo (hosted)**, B = appendix (B again to come back), A = autoplay (auto-advances, 91 s total), Home / End. Hash routing #1..#16.
Static frame: `?static=1&slide=N` (optionally `&t=seconds`). Deterministic render: `?render=1&t=seconds` or `window.__deck.renderAt(t)`.
Video: `export/macgym-deck.mp4` is the autoplay run. Evidence lives in `../deck/data/`.

One number per fact: cheat speed 6.16 m/s; honest gait 2.74 m/s; pendulum period +0.008% vs the exact analytic answer (say 0.01% from the textbook, not vs MuJoCo); walker transfer 22.0 m MacGym vs 19.7 m MuJoCo (n=20, about 90%); speed 22× at 4,096 robots (40× at 16,384) vs MuJoCo's C rollout on 11 CPU threads, same planar 2D walker, Apple M3 Pro 18 GB; 3.2 days of practice at 0:28; 87 s from clone to trained walker.

## 01. Cold open: MACGYM (5.5 s)

Every one of you has an ordinary laptop with you. I turned mine into a robot-learning lab. This robot taught itself to run on this MacBook in about a minute. I am Shlok, and this is MacGym.

## 02. Every laptop here has a GPU

The most ordinary thing at this hackathon is the laptop on every desk. It has a real GPU, and for robots it sits idle: the fast robot simulators need NVIDIA, and Apple's GPU has no maintained robot-RL backend.

## 03. So I built MacGym

So I built MacGym. You write a robot in Python, and MacGym trains 4,096 copies of it at once on your Mac, in about a minute. Robots that learn to walk, grasp and balance, for RL classes, games and real robot arms. Planar 2D, by choice. Everything you see ran on this Apple M3 Pro with 18 GB.

## 04. How it works (three beats)

One: you describe the robot, here a joint with motor 80. Two: MacGym compiles it into one Metal kernel, and every robot copy is one GPU thread running the whole step, contacts included. Three: PPO in MLX learns on the same memory, and the robot learns to walk.

## 05. It learns

Real replays from one run. After zero and ten seconds of training it falls. After thirty seconds, 15.1 metres. After one minute, 21.5 metres in eight seconds. And it is not one lucky run: five seeds from scratch, all five learned.

## 06. The number

Three point two days of simulated practice in twenty eight seconds. Days of practice, squeezed into seconds. That is the live counter from the playground, 4,096 robots on this M3 Pro.


## 08. Checked against the pro simulator

Three numbers. The pendulum's period is within a hundredth of a percent of the exact textbook answer, and MuJoCo matches too. At 4,096 robots MacGym is 22 times faster than MuJoCo's C rollout on 11 CPU threads, same planar 2D walker, same M3 Pro. And the trained walker, dropped unchanged into MuJoCo, keeps about 90 percent of its distance: 19.7 metres versus 22.0, over 20 runs. Sim-to-sim, not hardware.

## 09. One install. A real library.

A fresh install on this laptop: clone, pip install, macgym train walker. 87 seconds from clone to a trained walker. I built it today with AI coding agents: I designed the API, and I directed and tested it.

## 10. Close

The laptop on your desk is the most ordinary computer you own. MacGym turns it into a robot-learning lab. Scan the code, it is all at github.com/shlok-madhekar/macgym. Want to see it live? Press L (opens the playground on localhost, the GitHub repo when hosted).

## Appendix (B)

- A1 Validation: pendulum period 1.80511 s vs analytic 1.80497 s (0.008%), energy drift 0.038% vs MuJoCo Euler −0.121%; walker contact within 5 cm for the first 0.6 to 0.8 s, then the contact models differ.
- A2 Speed: 135K (CPU array ops) → 498K (same ops on GPU, ×3.7) → 16.3M (one Metal kernel per robot, ×32.8) = 121×. vs MuJoCo C rollout best 165K: 3.64M at 4,096 (22×), 6.75M at 16,384 (40×). PPO update 72%, physics 10%.
- A3 Seeds: 5 of 5 learned (> 10 m by 60 s).
- A4 Transfer robustness (n=20 in MuJoCo): nominal 19.7 ± 0.9 m; friction ×0.8 19.7 ± 1.7; ×1.2 19.9 ± 0.6; mass ×0.8 19.9 ± 1.4; mass ×1.2 15.6 ± 5.2 (3/20 falls, the weak case); random init 20.1 ± 0.5.
- A5 Who it's for: vs Colab/MJX it runs offline on the laptop you own. First users: DVHS CS Club (I'm an officer), my hackathons (Cove Hacks, HackPilot).
- A6 Built today: 11:38 first commit. Mine: the idea, framework over demo, scrapping the first UI, every experiment. AI coding agents wrote most of the code; I reviewed, tested and directed.

Q&A detail (fairness of 22×, MJX/Warp, cheating ablation) is unchanged from `../deck/NOTES.md`.
