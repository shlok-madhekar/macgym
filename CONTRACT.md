# MacGym shared contract (READ FIRST, do not change without telling the integrator)

Goal: massively parallel rigid-body physics + RL that trains robot policies on an Apple Silicon Mac GPU in minutes (MLX), with a live browser viewer of thousands of agents learning. Hackathon "Elevate the Ordinary", deadline 8 PM PT. Machine: M3 Pro, 18 GB. Python: /Users/shlok/.aside/runtime/bin/python3 (has mlx, numpy, aiohttp, mujoco 3.x). Repo: /Users/shlok/macgym. Run modules from repo root: `python3 -m macgym.xxx`.

Measured: batched cartpole in MLX with mx.compile = 222M env-steps/s at N=16384.

## Units / coords
2D planar, meters, seconds, radians. x right, y up. Ground at y=0 for locomotion envs.

## 1. Env interface — macgym/envs/base.py (already written)
class BatchedEnv:
    name: str; obs_dim: int; act_dim: int; num_envs: int
    max_episode_steps: int; dt: float  (control dt)
    def reset(self) -> mx.array (N, obs_dim)            # resets ALL envs
    def step(self, action: mx.array (N, act_dim) in [-1,1]) -> (obs, reward (N,), done (N,) float32 0/1, info: dict)
        # MUST auto-reset envs where done==1 internally (obs returned for those = fresh reset obs). Track timeouts internally (done=1 at max_episode_steps). info may include "episode_return" / "episode_len" arrays (N,) valid where done.
    def render_state(self, R: int) -> dict   # numpy float32, first R envs
        {"capsules": np.ndarray (R, K, 5)  [x1,y1,x2,y2,radius] world coords per env,
         "targets":  np.ndarray (R, T, 2) or None,
         "colors":   list of K hex strings (per capsule index) or None,
         "ground": float or None (y of ground line), "world_bounds": [xmin,xmax,ymin,ymax] (per-env view window, may follow agent via "camera_x": np.ndarray (R,) optional)}
    def set_target(self, x: float, y: float) -> None   # optional; arm env uses it (world coords relative to env origin)
Everything in step() must be pure mx ops, vectorized over N, and use mx.compile on the inner physics function. No Python loops over envs. Physics substeps allowed (loop over a few substeps inside compiled fn is fine).

Registry: macgym/envs/__init__.py exposes ENVS = {"cartpole": CartPole, "arm": ArmReach, "walker": Walker2D, "cheetah": Cheetah2D, ...} and make(name, num_envs) -> env. Each env module must be importable standalone.

## 2. PPO — macgym/rl/ppo.py
class PPO:
    def __init__(self, env, hidden=(128,128) or (256,256), lr=3e-4, rollout_len=32, epochs=4, minibatches=4, gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.0, vf_coef=0.5, seed=0)
    def train_iter(self) -> dict  # one rollout+update; returns {"iter","env_steps","sps","mean_reward","mean_ep_return","mean_ep_len","policy_loss","value_loss","entropy","time"}
    def act(self, obs, deterministic=False) -> mx.array (N, act_dim) in [-1,1]
    def save(self, path) / load(self, path)   # npz
    Gaussian policy with state-independent log_std, tanh or clip to [-1,1]; running obs normalization; GAE; advantage norm; all MLX on GPU; mx.compile the update where possible.
    Should also be safe to call env.render_state between iterations (server does that from same thread).

## 3. Server — macgym/server.py  (python3 -m macgym.server --port 8765)
aiohttp. Serves web/ as static at "/". WebSocket at "/ws". Trainer runs in a background thread; one active run at a time.
Client -> server JSON:
  {"cmd":"list"}                                          -> {"type":"envs","envs":[{"name","obs_dim","act_dim","description"}]}
  {"cmd":"start","env":"walker","num_envs":4096,"render_envs":64}  -> starts fresh run (stops previous)
  {"cmd":"stop"} ; {"cmd":"showcase","on":true}  (deterministic policy) ; {"cmd":"set_target","x":0.5,"y":0.8}
  {"cmd":"save"} ; {"cmd":"bench"} (returns contents of web/bench.json if present)
Server -> client:
  {"type":"frame","env":str,"R":int,"K":int,"capsules":[flat R*K*5 floats],"targets":[flat]|null,"colors":[...]|null,"ground":y|null,"world_bounds":[...],"camera_x":[R]|null}   ~30 Hz
  {"type":"metrics", ...train_iter dict..., "num_envs":N}  every iteration
  {"type":"status","state":"idle|training|showcase","env":...}
Frames are produced by calling env.render_state(R) after each train_iter (and between, in showcase mode by stepping with ppo.act deterministic). Round floats to 3 decimals.

## 4. Web viewer — web/index.html + web/app.js (+ web/style.css)
No build step. three.js via importmap from https://unpkg.com/three@0.170.0/build/three.module.js . Orthographic camera. Tiles R envs in a grid; each capsule = instanced mesh (InstancedMesh of a unit capsule/rounded box scaled+rotated per instance), ground lines, target dots. HUD: big env-steps/sec counter, total steps, reward chart (canvas), env picker, num_envs slider (256..16384), start/stop/showcase buttons, drag-to-set-target (sends set_target), "N robots" label. Dark, polished (#0B0D12 bg). Must work with a fake/demo mode if websocket not connected (?demo=1 animates fake data) so it can be developed independently.

## 5. Bench — scripts/bench.py -> web/bench.json
Compare MLX GPU env-steps/s vs CPU MuJoCo (single-thread and multi-process if easy) for comparable tasks (cartpole and a planar walker/cheetah). Honest numbers, include machine info.

## 6. Story — docs/DEVPOST.md, docs/VIDEO_SCRIPT.md, README.md
