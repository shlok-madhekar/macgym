# MacGym

**Robot learning, on the Mac you already own.**

MacGym is a robot-learning framework for Apple Silicon. Define a planar (2D) robot in about 15 lines of Python, train 4,096 copies of it on your Mac's GPU, and export the policy to CoreML. No NVIDIA card, no CUDA, no cloud.

Built at **Bay-Valley Hacks 2026**. Honest scope, up front:

- **Planar (2D) only.** Robots are capsules and hinges in the x/y plane. There is no 3D physics yet.
- **Apple Silicon only.** The simulator is a generated Metal kernel and training runs in MLX, so it needs an M1 or newer Mac.
- **Simple solver.** Maximal coordinates with an impulse-based contact solver. It is fast and good for RL experiments, not a replacement for MuJoCo-grade accuracy (see Benchmarks below).
- **Hackathon-age code.** It works end to end (define, train, view, export), but expect rough edges and API changes.

Docs: [`site/`](site/) (static HTML) · Source: https://github.com/shlok-madhekar/macgym · License: [MIT](LICENSE)

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
git clone https://github.com/shlok-madhekar/macgym macgym
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

Measured with `macgym bench` (`scripts/bench.py`) on an Apple M3 Pro, 18 GB unified memory, 11 CPU cores (5P+6E), macOS 26.5.2. Env-steps per second, summed over all environments, random actions, physics substeps and auto-reset included. Generated 2026-11-15 14:04:50 PST.

| Task | Backend | Envs | Env-steps/s |
|------|---------|-----:|------------:|
| walker | MLX GPU, public API (`mg.make`) | 1,024 | 892,199 |
| walker | MLX GPU, public API (`mg.make`) | 4,096 | 3,193,891 |
| walker | MLX GPU, public API (`mg.make`) | 16,384 | 2,995,792 |
| tripod | MLX GPU, public API (`mg.make`) | 1,024 | 725,503 |
| tripod | MLX GPU, public API (`mg.make`) | 4,096 | 2,764,401 |
| tripod | MLX GPU, public API (`mg.make`) | 16,384 | 8,106,502 |
| hopper | MLX GPU, public API (`mg.make`) | 1,024 | 1,085,328 |
| hopper | MLX GPU, public API (`mg.make`) | 4,096 | 2,854,852 |
| hopper | MLX GPU, public API (`mg.make`) | 16,384 | 10,217,217 |
| walker | MLX GPU, **training: sim + PPO** (`mg.train`) | 4,096 | 271,589 |
| walker | MLX GPU, hand-written env (sim only) | 1,024 | 1,224,240 |
| walker | MLX GPU, hand-written env (sim only) | 4,096 | 4,610,621 |
| walker | MLX GPU, hand-written env (sim only) | 16,384 | 13,202,073 |
| cheetah | MLX GPU, hand-written env (sim only) | 1,024 | 2,110,874 |
| cheetah | MLX GPU, hand-written env (sim only) | 4,096 | 2,037,532 |
| cheetah | MLX GPU, hand-written env (sim only) | 16,384 | 11,128,162 |
| cartpole | MLX GPU, hand-written env (sim only) | 1,024 | 3,617,662 |
| cartpole | MLX GPU, hand-written env (sim only) | 4,096 | 14,066,870 |
| cartpole | MLX GPU, hand-written env (sim only) | 16,384 | 42,858,271 |
| cheetah | CPU MuJoCo, 1 thread | 1 | 35,601 |
| cheetah | CPU MuJoCo, 11 processes | 11 | 135,887 |
| cartpole | CPU MuJoCo, 1 thread | 1 | 53,244 |
| cartpole | CPU MuJoCo, 11 processes | 11 | 238,231 |

The public-API rows build the env exactly like a user does (`mg.make(mg.robots.walker())`). The training row is end-to-end `mg.train` (rollout + GAE + PPO updates), which is what actually decides how long training takes.

At 1,024 envs the numbers don't track how heavy the robot is (an earlier run had walker and cheetah at 752K vs 746K): the step is bound by fixed Python and MLX dispatch overhead, not the GPU. With more envs that overhead is amortized, the GPU saturates, and throughput reflects the real physics cost.

**Limitations.** Both sides are planar for locomotion (MuJoCo HalfCheetah is constrained to the x-z plane too). The difference is the solver: MuJoCo uses generalized coordinates and a convex, constraint-based contact solver; MacGym uses maximal coordinates and a simpler impulse solver, and the MuJoCo baseline runs from a Python `mj_step` loop. Best GPU vs best CPU on this run: walker 97.2x, cheetah 81.9x, cartpole 179.9x (a quieter run earlier the same day gave 21.4x, 22.4x, 37.4x, so the CPU side is very load-sensitive), but read that as "RL experience per second this laptop generates for these tasks", not "MLX beats MuJoCo at the same physics". The laptop was not idle (1-minute load average 16.4 at the end of the run), so GPU numbers move about ±30% and CPU rows can drop up to 2x. Full method in `web/bench.json`.

## Who it's for

- **Students and hackers on MacBooks.** The MacBook is the most common laptop in CS classrooms and at hackathons, and most of those people can't get an NVIDIA GPU.
- **RL education.** Watch 4,096 robots learn in a minute, change one line, see what happens.
- **Quick morphology and reward iteration.** Try body plans and reward shapes locally in minutes, then scale the winner on cloud GPUs.

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

## Contributing

Issues and pull requests are welcome.

1. Fork, then `pip install -e ".[dev]"` on an Apple Silicon Mac.
2. Run the tests: `python3 -m pytest tests`.
3. Keep pull requests small and focused, and describe what you measured if you touch the physics or PPO (steps/s, distance, or return before and after).
4. Do not commit run artifacts, logs, or videos; `.gitignore` covers the usual ones.

Good first areas: new built-in robots in `macgym/presets/`, docs fixes in `site/`, and more Gymnasium compatibility tests.

## License

MIT. See [LICENSE](LICENSE).
