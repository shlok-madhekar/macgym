# MacGym

**Robot learning, on the Mac you already own.**

## Inspiration

The theme was "Elevate the Ordinary", and the most ordinary piece of hardware I own is my laptop: a normal M3 Pro MacBook. If you want to train robots with reinforcement learning right now, basically every serious GPU simulator (Isaac Lab, MuJoCo Warp, ManiSkill) needs an NVIDIA card. The Mac options I found were either dead (jax-metal) or way too slow to actually train anything.

But a MacBook has a real GPU with unified memory sitting right next to it. So the question was: can the laptop you already own become a robot-learning rig, not just for one demo but as a framework anyone can build on?

## What it does

MacGym is a robot-learning framework for Apple Silicon.

- **Define a robot in about 15 lines of Python.** Capsule bodies, motorized hinge joints, limits. A task is a reward lambda and an optional termination over a batched state.
- **Train 4,096 copies of it on the Mac GPU.** `mg.make` compiles the body plan into its own Metal kernel with one GPU thread per robot, and `mg.train` runs PPO in MLX on the same chip.
- **Export the policy** to CoreML (for native apps) or ONNX. Every export is checked against the original network: max error was 3.6e-7 (CoreML) and 4.8e-7 (ONNX) on our tripod policy.
- **Playground.** Edit a robot in a browser code editor, press Run, and watch thousands of copies learn on one shared runway, with replays of earlier generations so you can see the gait improve without reading a number.
- **CLI.** `macgym new`, `train`, `view`, `export`, `bench`.
- **Gymnasium wrapper.** `mg.to_gymnasium(env)` so existing RL libraries can use MacGym envs.
- **Five built-in robots** (walker, cheetah, hopper, tripod, worm), each written with the same public API a user would use.

## Who it's for

- **Students and hackers on MacBooks.** The MacBook is the most common laptop in CS classrooms and at hackathons, and most of those people can't get an NVIDIA GPU. MacGym lets them do real RL on the machine they already have.
- **RL education.** You can see 4,096 robots learn in a minute, change one line, and see the result. That is a much better way to learn PPO than staring at a reward curve from a cloud job.
- **Researchers iterating on morphology and rewards.** Try ten body plans or reward shapes locally in a few minutes each, then scale the one that works on a cloud cluster.

## How we built it

- **Physics.** A generic planar multibody engine in maximal coordinates (each body has its own position and angle, joints are constraints between them), solved with sequential impulses over a few substeps per control step, with Coulomb friction at the ground. For each robot definition MacGym generates a Metal kernel specialized to that body plan, so there are no Python loops over environments. Auto-reset and per-episode metrics run on the GPU too.
- **RL.** PPO in MLX: rollouts, GAE, observation normalization and minibatch updates all on the GPU. Because of unified memory, the simulator writes observations and the policy reads them in place. Nothing gets copied between host and device.
- **Framework layer.** A small public API (`Robot`, `Task`, `make`, `train`, `load_policy`, `Policy.export`, `view`, `to_gymnasium`) plus a `macgym` CLI, packaged with `pip install -e .`.
- **Viewer and Playground.** An aiohttp WebSocket server streams a sample of robots to the browser. The Playground runs user code in a restricted namespace, builds the env, and starts training, with errors shown inline.
- **Docs.** A static docs site (quickstart, API, CLI, examples, benchmarks, how it works) with no build step.

## Benchmark

Measured on an Apple M3 Pro (18 GB, 11 CPU cores) with `macgym bench`. Env-steps per second, summed over all robots, random actions, auto-reset included.

Through the public API, exactly the way a user builds an env (`mg.make(mg.robots.walker())`):

| Robot | 1,024 envs | 4,096 envs | 16,384 envs |
|---|---:|---:|---:|
| walker | 892K | 3.19M | 3.00M |
| tripod | 726K | 2.76M | 8.11M |
| hopper | 1.09M | 2.85M | 10.22M |

- **Training throughput (sim + PPO, `mg.train`, walker, 4,096 envs): 272K env-steps/s.** This is the number that decides how long training actually takes, since it includes the network forward pass and the gradient updates.
- Hand-written sim-only envs at 16,384: walker 13.20M, cheetah 11.13M, cartpole 42.86M.
- CPU MuJoCo, one process per core (11 processes): 136K (HalfCheetah-like), 238K (cartpole).

Why small batches don't track robot complexity (in an earlier run walker and cheetah came out nearly identical at 1,024 robots, 752K vs 746K): at that size the GPU isn't the bottleneck. Every step pays a fixed cost for Python and MLX dispatch no matter how heavy the robot is. With more robots that cost gets spread out, the GPU saturates, and the numbers start to reflect the actual physics work.

## Limitations

- **The physics is simplified and planar.** Robots live in a 2D plane. MuJoCo HalfCheetah is also planar, so the benchmark isn't 2D vs 3D. The real difference is the solver: MuJoCo uses generalized coordinates and a convex, constraint-based contact solver; we use maximal coordinates and a simpler impulse solver. So the speed comparison means "how much RL experience per second this laptop can generate for these tasks", not "MLX beats MuJoCo at the same physics". Full method and raw numbers are in `web/bench.json` and `scripts/bench.py`.
- **Noisy numbers.** The laptop wasn't idle during the hackathon. GPU numbers move about ±30% run to run, and the CPU MuJoCo rows drop when other jobs are using the cores. `bench.json` records the load average for that reason. Best GPU vs best CPU on the final run: walker 97.2x, cheetah 81.9x, cartpole 179.9x. Those are inflated by a busy CPU; a quieter run earlier the same day gave walker 21.4x, cheetah 22.4x, cartpole 37.4x. Treat the ratio as rough and the absolute numbers as the real story.
- **No sim-to-real yet.** Nothing has run on a physical robot.

## Questions we expect

**Is that a real gait or is it exploiting the simulator?** It's simplified planar physics, so we're careful here. RL loves exploits, so we measure stance-foot slip (how far a foot slides while it's supposed to be planted) and tuned the contact friction so gaits don't skate along the ground. Joint limits and torque caps keep motions physically plausible. It's still a 2D sim, not a claim about real hardware.

**Does it transfer?** The policy transfers out of MacGym: CoreML and ONNX exports match the original network to under 1e-6. Transfer to a real robot is future work. The next step is 3D bodies plus a sim-to-sim check: train in MacGym, run the policy in MuJoCo, and see how much performance survives.

**Who actually uses this?** Students, hackers and researchers on MacBooks who can't get an NVIDIA GPU. It's good for learning RL, and for fast morphology and reward iteration before you pay for cloud GPUs.

## Challenges

- **Going from demo to framework.** The first version had four hard-coded environments. Turning that into "write any planar robot in Python and get a fast kernel for it" meant building a generic multibody model and a code generator instead of hand-tuning each env.
- **Making learning visible.** An early viewer looked like a random number generator. The fix was putting every robot on one shared runway in metres, with the best one highlighted and replays of past generations.
- **Exploits.** Early walkers scooted and slid their feet instead of stepping. Joint limits, reward terms and contact friction all had to be tuned until they actually stepped.
- **Benchmarking honestly.** It would be easy to post one huge number. We measure through the same API users call, include an end-to-end training number, and keep the caveats next to the table.

## Accomplishments

- A real framework, not a single demo: public API, CLI, Gymnasium wrapper, verified export, docs, and a Playground, all running locally on a laptop.
- Millions of env-steps per second on a MacBook GPU, with PPO on the same chip.
- Change one line of a robot, press Run, and watch 4,096 copies of it learn.

## What we learned

Unified memory changes the shape of RL code. When the simulator and the network share memory, the whole loop stays on one chip, and the laptop's GPU stops being idle hardware.

## What's next

- 3D bodies and richer contact.
- Sim-to-sim validation against MuJoCo.
- A small on-device runtime for exported policies, and a shared gallery of community robots for the Playground.

## Built with

Python, MLX, Metal, Apple Silicon (unified memory), NumPy, aiohttp, WebSockets, JavaScript, HTML/CSS, ONNX, coremltools, Gymnasium, MuJoCo (CPU baseline benchmark only)
