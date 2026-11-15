# MacGym demo video (about 100 s)

Format: 1080p screen recordings on the actual M3 Pro, plus one real camera shot of the laptop. Voiceover recorded separately: casual, confident, a 16-year-old founder talking to a friend, not reading. Low electronic music bed that ducks under the voice. Burned-in captions.

Prep before recording:
- Local server running, Playground open at `/playground`, browser full screen, notifications off.
- A walker training run already going for about a minute in a second tab, so shot 1 opens mid-training.
- Terminal with a large font, in the repo folder, venv active.
- A trained checkpoint ready (`runs/strider.npz`) so the export beat works on the first take.
- `web/bench.json` up to date. Every number on screen comes from it. Don't round up, and don't put gait speeds on screen as hard numbers (the physics is still being tuned, so they'll change).

| # | Time | Shot | Exact on-screen action | Voiceover |
|---|------|------|------------------------|-----------|
| 1 | 0:00 to 0:05 | Full-width training track, mid-run. | Thousands of faint robots streaming right, the orange best robot out front, the progress chart in the corner shooting up. Caption, big and centered: **"4,096 robots learning to walk. On a MacBook. No NVIDIA."** | (music only, or) "That's four thousand robots learning to walk. On my MacBook." |
| 2 | 0:05 to 0:14 | Hard cut to the Playground. Editor on the left with the tripod preset, rest-pose drawing on the right. | Zoom on `HIPS = [-0.35, 0.0, 0.35]`. Type `, 0.7` so it reads `HIPS = [-0.35, 0.0, 0.35, 0.7]`. Press **Cmd+Enter**. The rest-pose drawing redraws with a 4th leg. | "This is a robot. It's like fifteen lines of Python. I'm gonna give it a fourth leg. One number. Cmd Enter." |
| 3 | 0:14 to 0:30 | Same screen, track fills. Time-lapse with a "4x" badge in the corner. | Thousands of four-legged robots flop at 0 m, then spread down the runway. Best robot pulls ahead, chart climbs, generation replays fill in. | "And now there's four thousand of them, training on the GPU in my laptop. PPO runs on the same chip, so nothing leaves the machine. A minute ago these things couldn't stand up." |
| 4 | 0:30 to 0:35 | **Real shot.** MacBook on a desk, Playground running on screen. | Caption: "a normal MacBook." | "No NVIDIA card. No cloud. Just the laptop I already had." |
| 5 | 0:35 to 0:50 | Terminal, then editor. | Type `pip install -e .`, install finishes. Cut to an editor showing the 15-line strider snippet from the homepage (`mg.Robot("strider")`, three `robot.body` lines, two `robot.joint` lines, `mg.Task(...)`, `mg.make`, `mg.train`). Slow scroll top to bottom. | "So I made it a real framework. It's called MacGym. Pip install it, and this is a whole robot plus training. Bodies, joints, a reward, make, train. That's it." |
| 6 | 0:50 to 1:02 | Terminal. | Type `macgym train strider.py --envs 4096 --iters 200`. Progress lines tick by with distance and steps/s. Then `macgym view strider.py`, browser pops up with the robot walking. | "There's a CLI too. Train runs headless and tells you how far your robot gets in actual metres. View opens it in the browser." |
| 7 | 1:02 to 1:12 | Terminal. | Type `macgym export runs/strider.npz --format coreml`. Highlight the printed `max_abs_err` line (under 1e-6). Cut to Finder showing `strider.mlpackage`. | "When you like it, export it. CoreML for a Mac or iPhone app, or ONNX. And it checks the exported model against the original, so you know it's the same policy." |
| 8 | 1:12 to 1:27 | Docs site Benchmarks page. | Slow scroll down the table. Pause on the framework-path rows, then the training row, then the Limitations box. On-screen text: "3.19M env-steps/s · 4,096 walkers · public API" and "272K env-steps/s training (sim + PPO)". Small text: "simplified planar physics, not the same solver as MuJoCo". | "Honest benchmark. About 3.2 million steps a second for four thousand walkers through the normal API, and 270 thousand a second when you include the actual PPO training. Our physics is simpler than MuJoCo's, it's in the fine print. But the GPU was just sitting there." |
| 9 | 1:27 to 1:37 | Quick montage: a classroom-style shot of laptops (or the docs "Who it's for" section), then the Playground with the worm preset. | Caption: "students · hackers · researchers". Press Run on worm, a field of worms starts wiggling forward. | "This is for everyone with a MacBook and no NVIDIA GPU. Students learning RL, hackers, researchers testing a robot idea before they pay for cloud." |
| 10 | 1:37 to 1:45 | Title card. | "MacGym. Robot learning, on the Mac you already own." Repo URL underneath. Theme line: "Elevate the Ordinary." | "Your laptop's already a robot-learning rig. MacGym just turns it on." |

## Shot notes

- Shot 1 must already be moving when the video starts. Record 20 seconds of a real run and use the best 5. No fade in.
- Shot 2 is the hook. Zoom the editor so `HIPS` is readable on a phone, and cut the instant the drawing updates.
- Shot 3: one long real take, sped up. Keep the "4x" badge visible the whole time it's sped up.
- Shots 6 to 8: numbers come straight from the terminal and `web/bench.json`. If a run comes out lower on the day, use the real number.
- Keep every screen shot moving with a slow zoom or pan. No static frame over 3 seconds.
- Never fake training footage. If Run takes a beat to start, cut on the keypress.
