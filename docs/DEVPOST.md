# MacGym

**Elevate the Ordinary: the laptop you already own becomes a robotics supercomputer.**

MacGym is a massively parallel robot-learning gym that runs entirely on an Apple Silicon Mac. It simulates thousands of robots at once on the Mac's GPU with MLX, trains them with PPO on that same GPU, and streams every one of them live into your browser so you can watch a swarm of robots learn to walk in minutes. No NVIDIA card, no cloud bill, no CUDA.

---

## Inspiration

Every serious robot reinforcement-learning simulator right now assumes you own an NVIDIA GPU. Isaac Lab is CUDA. MuJoCo Warp is CUDA. ManiSkill's GPU sim is CUDA. The modern recipe for robot learning is "simulate 4,096 robots in parallel on a GPU, run PPO, get a policy in minutes", and that recipe is locked behind hardware most students do not have.

Macs were supposed to be the escape hatch, and they keep getting closed off. jax-metal, the Apple backend for JAX (which is what MuJoCo's own GPU stack, MJX, runs on), was declared dead by a JAX maintainer in January 2026. The best community port of a GPU robotics sim to the Mac we could find (warp-metal / mjlab) reports about **5.2k env-steps/s on an M3 Max**. For comparison, a single CPU core running plain MuJoCo on our M3 Pro does about 52.4k env-steps/s on a half-cheetah.

So there are tens of millions of Apple Silicon laptops with a strong GPU and up to 128 GB of unified memory sitting there, and none of them can do the thing robotics people actually need. That felt like the most "ordinary" piece of hardware we could elevate: the MacBook already in your backpack.

## What it does

- **Massively parallel physics on the Mac GPU.** MacGym steps thousands of independent robots in a single batched MLX kernel. On an M3 Pro (18 GB):
  - CartPole: **19.3M env-steps/s** at 16,384 envs
  - Planar Walker: **6.3M env-steps/s** at 16,384 envs
  - Planar Cheetah: **6.6M env-steps/s** at 16,384 envs
  - That is **37.4x** (cartpole) and **21.4x** (walker vs MuJoCo half-cheetah) the experience throughput of CPU MuJoCo using every core on the same laptop.
- **PPO training on the GPU, end to end.** Rollouts, GAE, advantage normalization, observation normalization and the policy update all stay on the GPU in MLX. Robots go from flailing to walking in {{TRAIN_TIME_WALKER}}.
- **A live browser viewer of the whole swarm.** Open `localhost:8765` and you see a grid of robots learning in real time, rendered with three.js instanced meshes, with a giant env-steps/sec counter, a reward curve, an env picker, and a slider from 256 to 16,384 robots.
- **Interactive.** Drag in the arm environment to move the target and watch thousands of arms chase it. Flip on showcase mode to run the deterministic policy.
- **Honest benchmark built in.** `scripts/bench.py` measures MLX against CPU MuJoCo (single thread and one process per core) on this machine and writes `web/bench.json`, which the viewer shows.

## How we built it

**MLX as the physics engine.** We wrote the simulator directly in MLX, Apple's array framework, instead of porting a CUDA engine. Every environment's `step()` is pure vectorized `mx` ops over a batch dimension `N`, wrapped in `mx.compile` so the whole control step (several physics substeps, rewards, terminations and auto-reset) fuses into a small number of Metal kernels. There are no Python loops over environments anywhere.

**Batched maximal-coordinate 2D multibody physics.** Each robot is a set of capsule bodies, each with its own position, angle and velocities (maximal coordinates), connected by joint constraints and pushed off the ground with contact forces. Maximal coordinates are a great fit for GPUs: every body updates with the same arithmetic and there is no tree traversal, so a batch of 16,384 walkers is just a few big tensors. Joint constraints are solved iteratively inside the compiled substep loop.

**PPO in MLX.** A Gaussian policy with a state-independent log-std and a separate value network, running observation normalization, GAE (gamma 0.99, lambda 0.95), clipped surrogate objective, minibatched epochs. The update is compiled too.

**Unified memory means zero copy.** On a discrete GPU, simulation output has to cross PCIe to reach the learner and the renderer. On Apple Silicon the CPU and GPU share one pool of memory, so the physics state, the PPO rollout buffer and the frames we stream to the browser all live in the same place. Pulling the first 64 robots out for rendering is a slice, not a transfer.

**aiohttp + WebSocket server.** `python3 -m macgym.server` serves the viewer and a `/ws` endpoint. Training runs in a background thread; after each PPO iteration the server samples `render_state()` for a subset of robots and pushes compact capsule lists (x1, y1, x2, y2, radius) to the browser at about 30 Hz, along with metrics.

**three.js instanced rendering.** The viewer uses an orthographic camera and one `InstancedMesh` for every capsule in the scene, so drawing thousands of limbs is a handful of draw calls. No build step, just an importmap.

## Challenges we ran into

- **No reference implementation to copy.** Every GPU robotics sim we wanted to learn from is written for CUDA or JAX on NVIDIA. We had to design the physics around what MLX does well (big fused elementwise ops) rather than translate someone else's kernels.
- **Stability vs speed.** Stiff contacts and joint constraints blow up with large timesteps. Getting walkers that are stable at a useful control rate meant tuning substeps, constraint iterations and contact stiffness while keeping everything inside one compiled function.
- **Auto-reset without branching.** Thousands of episodes end at different times. We reset finished environments with masked `where` ops inside the compiled step, so there is no Python-side bookkeeping and no GPU stall.
- **Keeping the viewer smooth while training.** The browser can't receive 16,384 robots at 30 Hz, so we stream a representative subset and keep the full batch on the GPU for learning.
- **Being honest about benchmarks.** It's easy to post a huge number. Our MLX tasks are simplified 2D dynamics and MuJoCo solves full 3D dynamics with a real contact solver, so we say that next to every comparison (see the caveat below).

## Accomplishments that we're proud of

- 19.3M env-steps/s of CartPole and 6.3M env-steps/s of a planar walker on a base-level M3 Pro laptop, versus about 5.2k env-steps/s reported by the best community Mac port we found.
- Robot policies trained from scratch on a laptop GPU, with no NVIDIA hardware involved at any step.
- A live viewer where you can literally watch thousands of robots learn at once, which makes RL feel real instead of a loss curve in a terminal.
- A benchmark script anyone with a Mac can run to check our numbers.

## What we learned

- Apple Silicon is a genuinely good robot-learning machine once you write the simulator for it. Unified memory removes an entire category of engineering (host/device copies) that GPU RL frameworks spend a lot of effort on.
- `mx.compile` matters enormously: fusing the physics step turned hundreds of tiny kernel launches per step into a few.
- Maximal-coordinate physics trades some accuracy and constraint drift for extremely regular, parallel math, and for RL training that trade is usually worth it.
- The bottleneck in RL on a laptop is rarely raw FLOPs. It's kernel launch overhead, synchronization points and Python in the loop.

## What's next for MacGym

- **3D.** Extend the maximal-coordinate solver to 3D rigid bodies so we can run quadrupeds and humanoids.
- **Sim-to-real on a real arm.** Export trained policies to an SO-101 arm through LeRobot, so a policy trained on a MacBook moves a physical robot.
- **A native Mac app.** One download, pick a robot, press train, no Python setup.
- **MJCF import** so existing robot models load directly, and more tasks (manipulation, terrain, domain randomization).

## Benchmark caveat (read this)

MacGym's MLX tasks use our own simplified 2D planar dynamics (capsule bodies, maximal coordinates, penalty contacts, a few substeps per control step). The CPU baseline is MuJoCo solving full 3D generalized-coordinate dynamics with a constraint-based contact solver, driven from a Python `mj_step` loop. These are not the same physics, so the speedups describe how much RL experience per second this laptop can generate for these tasks, not "MLX is N times faster than MuJoCo at identical physics." Full numbers and methodology are in `web/bench.json` and `scripts/bench.py`.

## Built with

MLX, Python, Apple Silicon (Metal GPU, unified memory), NumPy, aiohttp, WebSockets, three.js, JavaScript, HTML/CSS, MuJoCo (CPU baseline benchmark only)
