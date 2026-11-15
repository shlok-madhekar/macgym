# MacGym

**Robot learning, on the Mac you already own.**

MacGym is a robot-learning framework for Apple Silicon. Define a planar (2D) robot in about 15 lines of Python, train 4,096 copies of it on your Mac's GPU, and export the policy to CoreML. No NVIDIA card, no CUDA, no cloud.

```python
import macgym as mg

robot = mg.Robot("strider")   # planar (2D): x forward, y up
torso = robot.body("torso", start=(0, 1.0), end=(0.6, 1.0), radius=0.06)
front = robot.body("front", start=(0.6, 1.0), end=(0.6, 0.5), radius=0.04)
back = robot.body("back", start=(0.0, 1.0), end=(0.0, 0.5), radius=0.04)
robot.joint(torso, front, at=(0.6, 1.0), limits=(-1.0, 1.0), motor=80)
robot.joint(torso, back, at=(0.0, 1.0), limits=(-1.0, 1.0), motor=80)

task = mg.Task(
    reward=lambda s: s.root.vx - 0.001 * s.action_sq,
    done=lambda s: s.root.y < 0.4,
)

if __name__ == "__main__":   # `macgym train strider.py` only reads robot + task
    env = mg.make(robot, task, num_envs=4096)
    result = mg.train(env, iters=200, save="runs/strider.npz")
    print(mg.evaluate(env, result.policy))   # {'dist': ..., 'return': ..., 'survived': ...}
```

`mg.make` turns that body plan into its own Metal kernel (one GPU thread per robot), and `mg.train` runs PPO in MLX on the same chip, sharing unified memory with the simulator.

## Install

Requires an Apple Silicon Mac (M1 or newer) and Python 3.10+.

```bash
git clone <repo-url> macgym
cd macgym
pip install -e .
pip install -e ".[gym]"      # optional: gymnasium wrapper
pip install -e ".[export]"   # optional: onnx + coremltools
```

## CLI

```bash
macgym new strider                                   # scaffold a robot file
macgym train strider.py --envs 4096 --iters 200      # headless PPO, progress in metres and steps/s
macgym view strider.py                               # live viewer in the browser
macgym export runs/strider.npz --format coreml       # or --format onnx
macgym bench                                        # GPU vs CPU MuJoCo on this machine
```

Anywhere a file is accepted you can pass a built-in name instead: `walker`, `cheetah`, `hopper`, `tripod`, `worm`. A robot file defines `robot` and `task` at module level, or a `make_env(num_envs)` function.

## Python API

| Call | What it does |
|------|--------------|
| `mg.Robot(name="robot", friction=0.9, dt=0.02, substeps=8, iters=4)` | Planar robot builder. `.body(name, start, end, radius=0.05, density=1000.0, color=None)` adds a capsule; `.joint(parent, child, at, limits=(-1.0, 1.0), motor=50.0, damping=0.1, stiffness=0.0, name=None)` adds a motorized hinge; `.set_root(body)` picks the body used for position and distance. |
| `mg.Task(reward, done=None, episode_steps=1000, obs=None, name=None)` | Reward and termination as lambdas over a batched `State` (pure `mx` ops; they run inside `mx.compile`). Observations come from `mg.default_obs(s)` unless you pass `obs`. |
| `mg.make(robot, task, num_envs=4096, seed=0, backend=None)` | Compiles the robot and returns a `RobotEnv` with `reset()`, `step(a) -> (obs, reward, done, info)`, `obs_dim`, `act_dim`, `render_state` and `metrics`. Auto-resets. Also accepts a `(robot, task)` pair from `mg.robots.*`. |
| `mg.train(env, iters=200, hidden=(256, 256), on_iter=None, save=None, lr=3e-4, seed=0, verbose=True, log_every=None, **ppo_kwargs)` | PPO on the GPU. Returns `TrainResult(policy, history, ppo)`; `result.final` is the last metrics dict. `on_iter(metrics)` returning `False` stops early. |
| `mg.evaluate(env, policy=None, steps=None)` | Deterministic rollout from a fresh reset. Returns `{"dist", "return", "survived"}` (mean metres, mean return, fraction still standing). |
| `mg.load_policy(path, env=None)` | Loads a checkpoint. `policy.act(obs)` returns actions in [-1, 1] (MLX or NumPy in, same out); also `.save(path)`, `.weights()`. |
| `policy.export(path, format=None, verify=True, tol=1e-4)` | CoreML (`.mlpackage`), ONNX (`.onnx`) or `.npz`, chosen by extension; verified against MLX and returns `{path, max_abs_err, ...}`. Needs the `[export]` extra. |
| `mg.view(target=None, policy=None, port=8765, num_envs=4096, open_browser=True)` | Local server plus browser viewer. `target` is a built-in name or a robot `.py` path (not an env object). Blocks. |
| `mg.to_gymnasium(env)` | `gymnasium.vector.VectorEnv` wrapper (NumPy in and out) for Stable-Baselines3, CleanRL, or your own loop. |
| `mg.robots.walker()`, `cheetah()`, `hopper()`, `tripod()`, `worm()` | Built-in `(robot, task)` pairs, written with the same public API (see table below). |

Inside `reward`, `done` and `obs`, every field of `s` is an MLX array with one entry per robot: `s.root.{x, y, vx, vy, angle, omega, height}` (x/y are world metres, angle is relative to the rest pose), `s.body("name")`, `s.joint("name" or index).{q, qd}`, `s.q` / `s.qd` (N, J), `s.action`, `s.action_sq`, `s.contact("name")`, `s.contacts` (N, B), `s.t`, `s.dt`.

### Built-in robots

All robots are planar (2D: x forward, y up). MacGym does not simulate 3D bodies.

| Call | Body plan |
|------|-----------|
| `mg.robots.walker()` | Planar biped: torso + 2 x (thigh, shin, foot), 6 motors. Same 24-dim observation as the legacy `walker` env, so `runs/walker.npz` loads. |
| `mg.robots.cheetah()` | HalfCheetah-like runner: torso + back/front (thigh, shin, foot), 6 motors. 17-dim observation. |
| `mg.robots.hopper()` | One-legged hopper (MuJoCo Hopper proportions): torso, thigh, leg, foot; 3 motors. |
| `mg.robots.tripod()` | Horizontal torso on three 2-segment legs (front, middle, back); 6 motors. |
| `mg.robots.worm()` | 5-segment chain lying on the ground that has to crawl forward; 4 motors; 500-step episodes. |

## Playground

`/playground` on the local server puts a robot file in a code editor next to the live track. Pick a preset or start blank, change the body, press **Run**, and 4,096 copies start learning. Errors show inline in the editor.

## Examples

| File | Shows |
|------|-------|
| `examples/01_quickstart.py` | Train a built-in robot in five lines |
| `examples/02_custom_robot.py` | Define a body plan from scratch |
| `examples/03_custom_reward.py` | Rewards and terminations over the batched `State` |
| `examples/04_gymnasium.py` | Use MacGym envs from Gymnasium-based libraries |
| `examples/05_export_coreml.py` | Export a trained policy to CoreML and ONNX |

## Benchmarks

Measured with `macgym bench` (`scripts/bench.py`) on an Apple M3 Pro, 18 GB unified memory, 11 CPU cores (5P+6E). Env-steps per second, summed over all environments, physics substeps and auto-reset included.

| Task | Backend | Envs | Env-steps/s |
|------|---------|-----:|------------:|
| walker | MLX GPU (Metal) | 4,096 | 3,048,904 |
| walker | MLX GPU (Metal) | 16,384 | 6,274,760 |
| cheetah | MLX GPU (Metal) | 16,384 | 6,589,763 |
| cartpole | MLX GPU (Metal) | 16,384 | 19,256,854 |
| cheetah | CPU MuJoCo, 1 thread | 1 | 52,376 |
| cheetah | CPU MuJoCo, 11 processes | 11 | 293,810 |
| cartpole | CPU MuJoCo, 11 processes | 11 | 514,545 |

Best GPU vs best CPU: walker 21.4x, cheetah 22.4x, cartpole 37.4x.

**Read the caveat.** MacGym uses its own simplified 2D planar dynamics (capsule bodies, maximal coordinates, penalty ground contact, a few substeps). MuJoCo solves full 3D generalized-coordinate dynamics with a constraint-based contact solver, driven here from a Python `mj_step` loop. That is not the same physics, so read the ratios as "how much RL experience per second this laptop can generate for these tasks", not "MLX beats MuJoCo at identical simulation". The walker is compared against MuJoCo's half-cheetah as the nearest planar locomotion model. GPU numbers varied by roughly ±30% run to run because other jobs shared the GPU. Full methodology is in `web/bench.json`.

## How it works

1. **Define.** A `Robot` is capsules and hinges in Python. A `Task` is array math over a batched state.
2. **Compile.** `mg.make` generates a Metal kernel specialized to the body plan. Each GPU thread steps one robot, so there is no Python loop over environments.
3. **Train.** PPO in MLX: rollouts, advantages, normalization and updates all on the GPU. Simulator state and network weights live in the same unified memory, so nothing is copied between devices.
4. **Ship.** Watch it with `macgym view`, export with `policy.export`, or hand the env to Gymnasium.

## Docs

The documentation site is plain static HTML in `site/` (no build step). The local server serves it at `/docs`, or open it directly:

```bash
python3 -m http.server 8797 --directory site
```

## Layout

```
macgym/        package: envs/ (physics2d, built-in envs), rl/ (PPO), server.py
web/           viewer and playground
site/          documentation site
examples/      runnable examples
scripts/       bench.py
docs/          DEVPOST.md, VIDEO_SCRIPT.md
```
