# MacGym demo video (90 s)

Format: 1080p, screen recordings captured with the server running on the actual M3 Pro, plus two real camera shots. Voiceover recorded separately, casual and confident, not rushed. Music: low, punchy electronic bed, ducks under voice. Burn in captions.

Prep before recording:
- `python3 -m macgym.server --port 8765`, browser full screen on `http://localhost:8765`, dark mode, notifications off.
- Have a pre-trained walker checkpoint ready for the showcase beat so it looks good on the first take.
- Charger plugged in at the start (needed for shot 2).

| # | Time | Shot | On screen | Voiceover |
|---|------|------|-----------|-----------|
| 1 | 0:00 to 0:05 | **Real shot.** Close-up, MacBook closed on a desk. Hands open the lid, screen lights up. | Caption: "this is a normal MacBook." | "This is my laptop. Totally normal MacBook. M3 Pro, 18 gigs." |
| 2 | 0:05 to 0:12 | Screen: quick scroll past the Isaac Lab, MuJoCo Warp and ManiSkill install pages, each with the "NVIDIA GPU required" line highlighted. | Red underline on each "NVIDIA" / "CUDA". | "If you want to train robots with RL right now, basically every simulator wants an NVIDIA GPU. Isaac Lab, MuJoCo Warp, ManiSkill. All of them." |
| 3 | 0:12 to 0:18 | Screen: text card. | "jax-metal: dead (Jan 2026)" / "best Mac port: ~5.2k steps/s" | "The Mac options are kind of dead. Best port I found does about five thousand steps a second. That's not enough to train anything." |
| 4 | 0:18 to 0:24 | Screen: terminal, type `python3 -m macgym.server --port 8765`, hit enter, cut to browser loading. | MacGym logo appears in the viewer. | "So I built MacGym. Robot physics and RL that run entirely on the Mac's GPU." |
| 5 | 0:24 to 0:34 | Screen: viewer. Pick "walker", drag the slider to 4,096, press Start. Grid of walkers spawns and starts flailing. | Big env-steps/sec counter climbs. | "Pick a robot, pick how many. Four thousand walkers. Hit start." |
| 6 | 0:34 to 0:44 | Screen: slow push in on the counter, then on the "N robots" label. | Counter reads 6.3M at 16,384. | "That number is real. Millions of physics steps a second, on a laptop, with no NVIDIA anything." |
| 7 | 0:44 to 0:52 | **Real shot.** Hand unplugs the MagSafe charger from the MacBook. Cut straight back to the screen, still training. | Caption: "on battery." | "And yeah, it's not plugged into anything. Still training." |
| 8 | 0:52 to 1:04 | Screen: time-lapse of the reward chart going up, walkers going from falling over to actually walking. Toggle Showcase. | Reward curve highlighted. | "This is PPO, also running on the GPU. Watch them figure out how to walk. Few minutes ago these things couldn't stand up." |
| 9 | 1:04 to 1:12 | Screen: switch to the arm env, drag the target around, thousands of arms chase the cursor. | Cursor trail visible. | "You can mess with them live. Drag the target and every arm goes after it." |
| 10 | 1:12 to 1:22 | Screen: benchmark table from `bench.json` (viewer bench panel). Animate the bars. | "24.8x vs MuJoCo on every CPU core" plus small caveat text: "2D simplified dynamics vs MuJoCo full 3D" | "Here's the honest benchmark. Way more experience per second than MuJoCo on every core of the same laptop. Different physics, so read the fine print, but that's the point. The GPU was just sitting there." |
| 11 | 1:22 to 1:30 | Screen: wide shot of the full grid in showcase mode, then fade to title card. | "MacGym. Elevate the Ordinary." + repo URL | "Your laptop is already a robotics supercomputer. MacGym just turns it on." |

## Shot notes

- Keep every screen shot moving (slow zoom or pan). No static frames over 3 seconds.
- Shot 6 and 10 numbers come straight from `web/bench.json`. Don't round up.
- Shot 7: film the unplug in one take, then cut on the motion so it feels continuous. Make sure the menu bar battery icon is visible in the screen shot right after.
- If walker training isn't impressive in real time, record a long take and speed it up in shot 8, with a small "4x" badge in the corner so it stays honest.
