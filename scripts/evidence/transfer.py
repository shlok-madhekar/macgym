"""Sim-to-sim policy transfer: runs/walker.npz (trained in MacGym) rolled out in MacGym and in
the MuJoCo mirror (scripts/evidence/mj_walker.py). 10 episodes x 8 s, deterministic actions,
same 24-d observation layout, same reset noise distribution (velocity noise U(+/-0.1), +/-0.2 ang)."""
import json, math, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mj_walker import ROOT, MjWalker, DT, TORSO_Y0
import numpy as np
import mlx.core as mx

OUT = os.path.join(ROOT, "web/deck/data")
P = np.load(os.path.join(ROOT, "runs/walker.npz"))
W = [(P[f"actor.layers.{i}.weight"].astype(np.float64), P[f"actor.layers.{i}.bias"].astype(np.float64)) for i in range(3)]
mean, var = P["__obs_mean"].astype(np.float64), P["__obs_var"].astype(np.float64)


def act(obs):  # == PPO.act(deterministic=True)
    x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10)
    for i, (w, b) in enumerate(W):
        x = x @ w.T + b
        if i < 2: x = np.tanh(x)
    return np.clip(x, -1, 1)


EPS, STEPS = 10, int(8.0 / DT)

# ---------------- MacGym (legacy walker env physics + obs, N=10 batch, no auto-reset)
from macgym.envs.walker import Walker2D
env = Walker2D(EPS, seed=123)
M = env.model; B = M.B
st = env._fresh(mx.random.key(7), EPS)
x0 = np.array(st[:, 0]); alive = np.ones(EPS, bool); tfall = np.full(EPS, np.nan); dist = np.zeros(EPS)
sim = mx.compile(M.step_metal)
# sanity: our numpy policy == macgym PPO deterministic act
from macgym.rl.ppo import PPO
ppo = PPO(env, hidden=tuple(int(h) for h in P["__hidden"])); ppo.load(os.path.join(ROOT, "runs/walker.npz"))
o0 = env.observe(st)
pol_err = float(np.max(np.abs(np.array(ppo.act(o0, deterministic=True)) - act(np.array(o0, np.float64)))))
ours_x = []
for k in range(STEPS):
    o = np.array(env.observe(st), np.float64)
    st = sim(st, mx.array(act(o).astype(np.float32))); mx.eval(st)
    s = np.array(st)
    y, ang = s[:, B], s[:, 2 * B] - math.pi / 2
    fell = alive & ((y < 0.7 * 1.25) | (np.abs(ang) > 1.0))
    tfall[fell] = (k + 1) * DT
    dist[alive] = s[alive, 0] - x0[alive]
    alive &= ~fell
    ours_x.append(s[0, 0] - x0[0])
ours = {"distance_m": dist.tolist(), "mean_distance_m": float(dist.mean()), "falls": int((~alive).sum()),
        "fall_rate": float((~alive).mean()), "fall_times_s": [None if np.isnan(t) else float(t) for t in tfall],
        "mean_speed_mps_over_survival": float(np.mean(dist / np.where(np.isnan(tfall), 8.0, tfall)))}
print("ours", ours)

# ---------------- MuJoCo mirror
rng = np.random.default_rng(123)
mj = {"distance_m": [], "fall_times_s": []}
frames = []
w = MjWalker(speed_curve=True)
mj_x = []
for ep in range(EPS):
    qv = np.concatenate([rng.uniform(-0.1, 0.1, 2), rng.uniform(-0.2, 0.2, 1), rng.uniform(-0.2, 0.2, 6)])
    w.reset(qv)
    x0m = w.torso()["x"]; tf = None; d = 0.0
    for k in range(STEPS):
        w.step(act(w.obs().astype(np.float64)))
        t = w.torso()
        if ep == 0: mj_x.append(t["x"] - x0m)
        if tf is None:
            d = t["x"] - x0m
            if t["y"] < 0.7 * 1.25 or abs(t["angle"]) > 1.0:
                tf = (k + 1) * DT
        if ep == 0 and k % 2 == 0:
            frames.append(w.data.qpos.copy())
    mj["distance_m"].append(float(d)); mj["fall_times_s"].append(tf)
dm = np.array(mj["distance_m"]); ft = np.array([8.0 if t is None else t for t in mj["fall_times_s"]])
mj.update(mean_distance_m=float(dm.mean()), falls=int(sum(t is not None for t in mj["fall_times_s"])),
          mean_speed_mps_over_survival=float(np.mean(dm / ft)))
mj["fall_rate"] = mj["falls"] / EPS
print("mujoco", mj)

# deterministic zero-noise episode in both (cleanest comparison)
import mujoco
up = subprocess.check_output(["uptime"], text=True).strip()
out = {"policy": "runs/walker.npz (PPO, MLP 256x256 tanh, trained in MacGym), deterministic mean action",
       "policy_numpy_vs_macgym_max_abs_err": pol_err,
       "episodes": EPS, "episode_seconds": 8.0, "control_dt": DT,
       "fall_definition": "torso height < 0.875 m or |torso angle| > 1.0 rad (MacGym's training termination); distance = torso x at fall or 8 s",
       "macgym": ours, "mujoco": mj, "mujoco_version": mujoco.__version__,
       "observation": "24-d legacy walker layout rebuilt from MuJoCo qpos/qvel/geom_xpos: [y, sin a, cos a, vx, vy, 0.1w, q(6), 0.1qd(6), contact R/L (lowest point <2 cm), foot dx/y R, L]",
       "uptime": up, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
json.dump(out, open(os.path.join(OUT, "transfer.json"), "w"), indent=1)
np.save(os.path.join(HERE, "..", "..", "runs", "_mj_transfer_frames.npy"), np.array(frames))
print(up)
