# MacGym

**Massively parallel robot RL on the Mac you already own.**

MacGym simulates thousands of robots at once on an Apple Silicon GPU with [MLX](https://github.com/ml-explore/mlx), trains them with PPO on that same GPU, and streams the whole swarm live into your browser. No NVIDIA card, no CUDA, no cloud.

Every mainstream GPU robotics sim (Isaac Lab, MuJoCo Warp, ManiSkill) needs NVIDIA. MacGym is written from scratch for Metal and unified memory instead.

- Batched 2D multibody physics in pure MLX, `mx.compile`d, no Python loops over envs
- PPO fully on the GPU (rollouts, GAE, normalization, updates)
- Live three.js viewer: thousands of agents, env-steps/sec counter, reward curve, drag-to-set-target
- Honest benchmark vs CPU MuJoCo included (`scripts/bench.py`)

Environments: `cartpole`, `arm` (reach a draggable target), `walker` (planar biped), `cheetah` (planar quadruped-style runner).

## Quickstart

Requirements: Apple Silicon Mac, Python 3.10+, `mlx`, `numpy`, `aiohttp` (plus `mujoco` only if you want to run the CPU baseline benchmark).

```bash
pip install mlx numpy aiohttp mujoco
python3 -m macgym.server --port 8765
```

Then open **http://localhost:8765**, pick an env, set the number of robots (256 to 16,384) and press **Start**.

Viewer without a backend (fake data, for UI work): http://localhost:8765/?demo=1

## Training from the CLI

Headless PPO training (prints iteration, env steps, env-steps/sec and returns):

```bash
python3 -m macgym.train --env walker --num-envs 4096 --iters 300 --save walker.npz
python3 -m macgym.train --env cartpole --num-envs 4096 --iters 100
```

Flags: `--env` (cartpole | arm | walker | cheetah), `--num-envs`, `--iters`, `--save PATH.npz`, `--load PATH.npz`, `--hidden 128,128`, `--lr`, `--rollout-len`, `--epochs`, `--minibatches`, `--gamma`, `--ent-coef`, `--lr-decay`, `--seed`.

Or start the live server and kick off a run immediately (the browser can attach any time):

```bash
python3 -m macgym.server --port 8765 --env walker --num-envs 4096 --autostart
```

Server flags: `--host` (default `0.0.0.0`), `--port` (8765), `--env` (walker), `--num-envs` (4096), `--render-envs` (how many robots to stream to the viewer), `--autostart`.

Or drive PPO directly from Python:

```python
from macgym.envs import make
from macgym.rl.ppo import PPO

env = make("walker", 4096)
ppo = PPO(env, hidden=(256, 256))
for i in range(300):
    m = ppo.train_iter()
    print(f"iter {m['iter']:4d}  sps {m['sps']:,.0f}  ep_return {m['mean_ep_return']:.1f}")
ppo.save("walker.npz")
```

WebSocket protocol (`/ws`): `{"cmd":"list"}`, `{"cmd":"start","env":"walker","num_envs":4096,"render_envs":64}`, `{"cmd":"stop"}`, `{"cmd":"showcase","on":true}`, `{"cmd":"set_target","x":0.5,"y":0.8}`, `{"cmd":"save"}`, `{"cmd":"bench"}`. See `CONTRACT.md` for the full spec.

## Architecture

```
                        Apple Silicon  (one pool of unified memory, zero copy)
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  GPU (Metal, via MLX)                                                     │
 │                                                                           │
 │   ┌────────────────────────────┐  obs, reward, done   ┌────────────────┐  │
 │   │ BatchedEnv.step()          │ ───────────────────▶ │ PPO            │  │
 │   │  N envs x K capsule bodies │                      │  policy + value│  │
 │   │  maximal-coord 2D physics  │ ◀─────────────────── │  GAE, clip obj │  │
 │   │  joints, contacts, substeps│      actions (N,A)   │  obs norm      │  │
 │   │  auto-reset   (mx.compile) │                      │  (mx.compile)  │  │
 │   └─────────────┬──────────────┘                      └───────┬────────┘  │
 │                 │ render_state(R): slice first R envs         │ metrics   │
 └─────────────────┼─────────────────────────────────────────────┼───────────┘
                   ▼                                             ▼
          ┌───────────────────────────────────────────────────────────┐
          │ macgym.server  (aiohttp, trainer thread)                  │
          │   static web/   +   WebSocket /ws  (frames ~30 Hz, metrics)│
          └───────────────────────────┬───────────────────────────────┘
                                      │ JSON: capsules [x1,y1,x2,y2,r]
                                      ▼
          ┌───────────────────────────────────────────────────────────┐
          │ Browser: three.js, orthographic camera, InstancedMesh     │
          │   grid of R robots, env-steps/s HUD, reward chart,        │
          │   env picker, num_envs slider, drag-to-set-target         │
          └───────────────────────────────────────────────────────────┘
```

```
macgym/
  envs/       base.py (BatchedEnv), cartpole.py, arm.py, walker.py, cheetah.py, physics2d.py, registry in __init__.py
  rl/ppo.py   PPO in MLX
  server.py   aiohttp + WebSocket trainer server
  train.py    headless training CLI
web/          index.html, app.js, style.css, bench.json
scripts/      bench.py
docs/         DEVPOST.md, VIDEO_SCRIPT.md
```

## Benchmark

Run it yourself: `python3 scripts/bench.py` (add `--quick` for shorter timing windows). Results go to `web/bench.json` and show up in the viewer.

<!-- BENCH:START -->
Machine: Apple M3 Pro, 18 GB unified memory, 11 CPU cores (5P+6E), macOS 26.5.2

| Task | Backend | Envs | Env-steps/s |
|---|---|---:|---:|
| cartpole | MLX GPU (Metal) | 1,024 | 1,027,297 |
| cartpole | MLX GPU (Metal) | 4,096 | 3,868,791 |
| cartpole | MLX GPU (Metal) | 16,384 | 13,629,563 |
| walker | MLX GPU (Metal) | 1,024 | 704,170 |
| walker | MLX GPU (Metal) | 4,096 | 2,862,325 |
| walker | MLX GPU (Metal) | 16,384 | 6,312,861 |
| cheetah | MLX GPU (Metal) | 1,024 | 761,179 |
| cheetah | MLX GPU (Metal) | 4,096 | 3,042,999 |
| cheetah | MLX GPU (Metal) | 16,384 | 6,705,159 |
| cartpole | CPU MuJoCo 1 thread | 1 | 89,971 |
| cartpole | CPU MuJoCo 11 procs | 11 | 550,048 |
| cheetah | CPU MuJoCo 1 thread | 1 | 51,546 |
| cheetah | CPU MuJoCo 11 procs | 11 | 274,702 |

Best MLX GPU vs best CPU MuJoCo (all cores): cartpole **24.8x**, cheetah **24.4x**, walker **23x** (walker compared against MuJoCo half-cheetah)
<!-- BENCH:END -->

**Methodology.** One env-step = one control step of one environment, physics substeps / frame-skip included. MLX: random actions generated on the GPU, full `mx.eval` every step, auto-reset included. MuJoCo: Gymnasium-style InvertedPendulum-like cartpole (timestep 0.02, frame_skip 2) and HalfCheetah-like planar model (timestep 0.01, frame_skip 5), Python `mj_step` loop with random ctrl, single thread and one process per core.

**Caveat, honestly.** MacGym's MLX tasks use our own simplified 2D planar dynamics (capsule bodies, maximal coordinates, penalty contacts). MuJoCo solves full 3D generalized-coordinate dynamics with a constraint-based contact solver. This is not identical physics, so read the speedup as "RL experience per second this laptop can generate for these tasks", not "MLX beats MuJoCo at the same simulation". The MLX walker is compared against the MuJoCo half-cheetah as the nearest planar locomotion model.

## Roadmap

- 3D rigid bodies (quadrupeds, humanoids)
- Export policies to a real SO-101 arm via LeRobot
- Native Mac app
- MJCF import, domain randomization, terrain

Built for the "Elevate the Ordinary" hackathon: the laptop you already own becomes a robotics supercomputer.
