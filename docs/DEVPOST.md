# MacGym

**Robot learning, on the Mac you already own.**

## Inspiration

The theme was "Elevate the Ordinary", and the most ordinary piece of hardware I own is my laptop. It's a normal M3 Pro MacBook. If you want to train robots with reinforcement learning right now, basically every serious GPU simulator (Isaac Lab, MuJoCo Warp, ManiSkill) needs an NVIDIA card. The Mac options I found were either dead (jax-metal) or far too slow to actually train anything.

But the chip in a MacBook has a real GPU and unified memory sitting right next to it. So the question was: can the laptop you already own become a robot-learning rig, not just for one demo but as a framework anyone can build on?

## What it does

MacGym is a robot-learning framework for Apple Silicon.

- **Define a robot in about 15 lines of Python.** Capsule bodies, motorized hinge joints, limits. A task is a reward lambda and an optional termination over a batched state.
- **Train 4,096 copies of it on the Mac GPU.** `mg.make` compiles the body plan into its own Metal kernel with one GPU thread per robot, and `mg.train` runs PPO in MLX on the same chip.
- **Export the policy** to CoreML (for native apps) or ONNX.
- **Playground.** Edit a robot's definition in a browser code editor, press Run, and watch thousands of copies learn on one shared runway, with replays of earlier generations so you can see the gait improve without reading a number.
- **CLI.** `macgym new`, `train`, `view`, `export`, `bench`.
- **Gymnasium wrapper.** `mg.to_gymnasium(env)` so existing RL libraries can use MacGym envs.
- **Five built-in robots** (walker, cheetah, hopper, tripod, worm), each written with the same public API a user would use.

## How we built it

- **Physics.** A generic planar multibody engine: capsules, hinge joints, penalty ground contact, a few substeps per control step, in maximal coordinates. For each robot definition MacGym generates a Metal kernel specialized to that body plan, so there are no Python loops over environments. Auto-reset and per-episode metrics run on the GPU too.
- **RL.** PPO written in MLX: rollouts, GAE, observation normalization and minibatch updates all on the GPU. Because Apple Silicon has unified memory, the simulator writes observations and the policy reads them in place. Nothing gets copied between host and device.
- **Framework layer.** A small public API (`Robot`, `Task`, `make`, `train`, `load_policy`, `Policy.export`, `view`, `to_gymnasium`) plus a `macgym` CLI, packaged with `pip install -e .`.
- **Viewer and Playground.** An aiohttp WebSocket server streams a sample of robots to the browser. The Playground runs user code in a restricted namespace, builds the env, and starts training, with errors reported inline.
- **Docs.** A static docs site (quickstart, API, CLI, examples, benchmarks, how it works) with no build step.

## Benchmark, with the honest caveat

Measured on an Apple M3 Pro (18 GB, 11 CPU cores) with `macgym bench`:

- Walker: **3.05M env-steps/s** at 4,096 robots, **6.27M** at 16,384.
- Cheetah: **6.59M env-steps/s** at 16,384.
- Cartpole: **19.3M env-steps/s** at 16,384.
- CPU MuJoCo, one process per core (11 processes): 294K (half-cheetah), 515K (cartpole).

Best GPU vs best CPU: walker **21.4x**, cheetah **22.4x**, cartpole **37.4x**.

This is not the same physics. MacGym's tasks use our own simplified 2D planar dynamics with penalty contacts. MuJoCo solves full 3D generalized-coordinate dynamics with a constraint-based contact solver, and our MuJoCo baseline runs from a Python `mj_step` loop, so it includes Python overhead. The MLX walker is compared against MuJoCo's half-cheetah as the closest planar locomotion model. GPU numbers moved roughly ±30% run to run because other training jobs were sharing the GPU during the hackathon. Read the ratios as "how much RL experience per second this laptop can generate for these tasks", not "MLX is N times faster than MuJoCo at identical simulation". Full methodology lives in `web/bench.json` and `scripts/bench.py`.

## Challenges

- **Going from demo to framework.** The first version had four hard-coded environments. Turning that into "write any planar robot in Python and get a fast kernel for it" meant building a generic multibody model and a code generator instead of hand-tuning each env.
- **Making learning visible.** An early viewer looked like a random number generator. The fix was showing every robot on one shared runway in metres, with the best one highlighted and replays of past generations, so you can see the improvement instead of trusting a reward number.
- **Gaits that cheat.** RL is very good at finding exploits. Joint limits and reward terms had to be tuned so the walker actually walks instead of scooting.
- **Being honest about benchmarks.** It would be easy to post one huge number. We put the 2D vs 3D caveat next to every comparison instead.

## Accomplishments

- A real framework, not a single demo: public API, CLI, Gymnasium wrapper, export, docs, and a Playground, all running locally on a laptop.
- Millions of env-steps per second on a MacBook GPU, with PPO on the same chip.
- Define a new robot, press Run, and watch 4,096 copies of it learn.

## What we learned

Unified memory changes the shape of RL code. When the simulator and the network share memory, the whole loop can stay on one chip, and the laptop's GPU stops being idle hardware.

## What's next

- 3D bodies and richer contact.
- More export targets and a small on-device runtime for exported policies.
- A shared gallery of community robot files for the Playground.

## Built with

Python, MLX, Metal, Apple Silicon (unified memory), NumPy, aiohttp, WebSockets, JavaScript, HTML/CSS, ONNX, coremltools, Gymnasium, MuJoCo (CPU baseline benchmark only)
