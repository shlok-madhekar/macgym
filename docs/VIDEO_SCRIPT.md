# MacGym demo video (about 110 s)

Format: 1080p screen recordings on the actual M3 Pro, plus one real camera shot at the open. Voiceover recorded separately: casual, confident, not rushed. Low electronic music bed that ducks under the voice. Burned-in captions.

Prep before recording:
- Local server running with the Playground open at `/playground`, browser full screen, notifications off.
- Terminal with a large font, in the repo folder, venv active.
- A trained walker checkpoint ready (`runs/walker.npz`) so the export beat works on the first take.
- `web/bench.json` up to date. Every number on screen comes from it. Don't round up.

| # | Time | Shot | On screen | Voiceover |
|---|------|------|-----------|-----------|
| 1 | 0:00 to 0:10 | Screen: Playground. Code editor on the left with the tripod robot file, empty track on the right. Cursor changes a leg's `end=` coordinate, the robot preview on the right updates. | Caption: "this is a robot. it's 15 lines of python." | "This is a robot. The whole thing is like fifteen lines of Python. I'm gonna make its back leg longer." |
| 2 | 0:10 to 0:18 | Press **Run**. The track fills with thousands of faint tripods, all flopping at 0 m. | Footer: "4,096 robots in parallel · Apple M3 Pro GPU" | "Hit run. That's not one robot. That's four thousand and ninety six copies of it, on my laptop's GPU." |
| 3 | 0:18 to 0:36 | Time-lapse of training: robots spread down the runway, orange best robot pulls ahead, avg and best markers move right, progress chart climbs, generation replays fill in (0:00, 0:10, 0:30, 1:00). | Small "4x" badge in the corner while sped up. | "Every one of them is trying something slightly different. PPO, also on the GPU, keeps what works. A minute ago these things couldn't move. Now look at them." |
| 4 | 0:36 to 0:44 | **Real shot.** Close-up of the MacBook on a desk, Playground still running on the screen. | Caption: "a normal MacBook." | "And this is just a normal MacBook. No NVIDIA card, no cloud. The laptop I already had." |
| 5 | 0:44 to 0:56 | Screen: terminal. Type `git clone`, `cd macgym`, `pip install -e .`. Install finishes. | Caption: "pip install -e ." | "So I turned it into a framework. It's called MacGym. Clone it, pip install, done." |
| 6 | 0:56 to 1:10 | Terminal: `macgym new tripod`, then `macgym train tripod.py --envs 4096`. Progress lines tick by with distance in metres and steps per second. | Highlight the steps/s column. | "There's a CLI. New scaffolds a robot, train trains it. It tells you how far your robot walks in actual metres, not some random reward number." |
| 7 | 1:10 to 1:20 | Terminal: `macgym export runs/walker.npz --format coreml`. Cut to Finder showing `walker.mlpackage`. | Caption: "CoreML. ONNX too." | "When you like it, export it. CoreML so it runs in a real Mac or iPhone app, or ONNX for anything else." |
| 8 | 1:20 to 1:36 | Screen: docs site Benchmarks page. Slow scroll down the table, pause on the caveat box. | "3.0M env-steps/s · 4,096 walkers" and "21x vs MuJoCo on all 11 CPU cores", with small text "2D simplified physics vs MuJoCo full 3D" | "Here's the honest benchmark. Three million steps a second with four thousand walkers, about twenty one times what MuJoCo gets on every CPU core of the same laptop. Different physics, ours is 2D, so read the fine print. But that's the point. The GPU was just sitting there." |
| 9 | 1:36 to 1:50 | Screen: back to the Playground, swap the preset to worm, press Run, a field of worms starts wiggling forward. Fade to title card. | Title card: "MacGym. Robot learning, on the Mac you already own." Repo URL underneath. Theme line: "Elevate the Ordinary." | "Your laptop is already a robot-learning rig. MacGym just turns it on." |

## Shot notes

- Keep every screen shot moving with a slow zoom or pan. No static frame over 3 seconds.
- Shot 3: record one long real take and speed it up. Keep the "4x" badge visible the whole time it's sped up so it stays honest.
- Shots 6 and 8: numbers come straight from the terminal and `web/bench.json`. If a run comes out lower on the day, use the real number.
- Shot 1: zoom the editor so the code is readable on a phone screen.
- If the Run in shot 2 takes a beat to start, cut on the button press so it feels instant, but never fake the training footage.
