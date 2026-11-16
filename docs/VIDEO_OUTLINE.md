# MacGym demo video outline (target 2:00, hard max 3:00)

Who it's for: Bay-Valley Hacks judges watching on Devpost. They decide in the first 10 seconds, so open on the payoff, not the setup.

Rule: every shot is real. Real screen recordings, real numbers from today's runs. If a take comes out different, use the real number.

## What you film (you on camera)

| # | Where | What you say (rough, say it your way) |
|---|-------|---------------------------------------|
| A | Desk, MacBook open, camera at chest height | "This is a normal MacBook. In the next minute it's going to teach four thousand robots to walk." |
| B | Same, holding the laptop up or tapping the lid | "Every serious robot sim needs an NVIDIA GPU. The one in here just sits there. So I built MacGym." |
| C | Close-up talking, after the cheat clip | "My first robots cheated. They figured out the motors had infinite power and slid their feet. So we caught it and fixed the physics." |
| D | Ending, looking at camera | "MacGym. Robot learning on the laptop you already own. Link's below." |

Film each 3 to 4 times. Short lines, one breath each. Clip-on mic if you have one.

## Full timeline

| Time | Shot | On screen | Voice |
|------|------|-----------|-------|
| 0:00 to 0:06 | Screen: Playground mid-training, thousands of tripods streaming right | Big text: "4,096 robots. One MacBook. No NVIDIA." | none, or your line A over it |
| 0:06 to 0:12 | You on camera (A/B) | Small corner label: "Bay-Valley Hacks · Elevate the Ordinary" | Line A, then B |
| 0:12 to 0:30 | Screen: Playground, tripod preset. Type a 4th hip into `HIPS`, rest-pose redraws, press Cmd+Enter | Console line visible: "compiled Metal kernel for 9 bodies / 8 joints in 0.24 s" | "This is a robot, about fifteen lines of Python. I'll give it a fourth leg. Cmd Enter. It just compiled into one GPU program, one thread per robot." |
| 0:30 to 0:50 | Screen: same run sped up with a "4x" badge | Counter: "3.2 days of simulated practice in 28 s". Generations panel: 0:00 falls, 1:00 walks | "That's four thousand copies practicing at once. Thirty seconds in, they've done three days of practice. A minute ago they couldn't stand." |
| 0:50 to 1:05 | Deck B-roll (`web/deck2/export/macgym-deck.mp4`, slide 7 "The twist") then you on camera (C) | Red "6.16 m/s" struck through, then "2.74 m/s" | Line C |
| 1:05 to 1:20 | Deck slide 8 (three numbers count up) | 0.01% · 22× · 90% | "Is it real physics? Same tests as MuJoCo, the pro simulator: textbook cases match to a hundredth of a percent. It's 22 times faster than MuJoCo's fastest CPU mode on the same robot. And the trained robot still walks when you drop it into MuJoCo." |
| 1:20 to 1:40 | Terminal, big font | `pip install -e .` then `macgym train walker` ticking, then `macgym export ... --format coreml` | "And it's a real library. Pip install, train from the command line, export to CoreML. Clone to a trained robot in 87 seconds." |
| 1:40 to 1:50 | Docs site scroll (`/docs`), then the code playground | "Open source · github.com/shlok-madhekar/macgym" | "It's for anyone learning RL on a Mac. Which is basically everyone here." |
| 1:50 to 2:00 | You on camera (D), then deck close slide (QR) | "Elevate the ordinary." | Line D |

## Before you record

- Server running: `cd ~/macgym && /Users/shlok/.aside/runtime/bin/python3 -m macgym.server --port 8765`, Playground at http://localhost:8765/playground, browser full screen, notifications off (Do Not Disturb).
- Start one tripod run a minute early in a second tab so the 0:00 shot is already moving.
- Terminal font size 20+, in `~/macgym`.
- Deck video for B-roll: `~/macgym/web/deck2/export/macgym-deck.mp4` (slide 7 starts around 0:56, slide 8 around 1:05; scrub to confirm).
- Screen record with Screen Studio (you have it) at 1080p or higher.

## Numbers you can say (all measured today)

- 4,096 robots train at once on the M3 Pro
- 3.2 days of simulated practice in 28 s (7.8 days in 1:06)
- 5 of 5 training runs learned to walk within a minute
- 22× faster than MuJoCo's fastest CPU mode, same 2D walker, 4,096 robots (40× at 16,384)
- Pendulum timing matches MuJoCo within 0.01%
- Trained walker in MuJoCo: 19.7 m vs 22.0 m in MacGym (about 90%)
- 87 s from git clone to a trained walker
- Cheat: 6.16 m/s before, 2.74 m/s after the fix

Say "planar 2D" once if a judge might ask (the 1:05 beat is a good spot: "it's 2D physics on purpose, fast to teach").
