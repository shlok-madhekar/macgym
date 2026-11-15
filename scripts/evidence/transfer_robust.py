"""Robust sim-to-sim transfer (extends transfer.py): runs/walker.npz (trained in MacGym) rolled out in the
MuJoCo mirror (mj_walker.py) under perturbed physics. 20 episodes x 8 s per condition, deterministic actions.

Every condition uses the same base reset distribution as transfer.py (root vel U(+/-0.1), torso ang vel
U(+/-0.2), joint vel U(+/-0.2)) with common random numbers (same seed per condition), so differences between
conditions come from the perturbation, not from different reset draws.
  nominal          - mirror as-is (friction 0.9, capsule masses at density 1000)
  friction x0.8/1.2 - sliding friction of every geom (incl. floor) scaled
  mass x0.8/1.2    - every link mass scaled, with its rotational inertia scaled by the same factor (density scale)
  init_random      - base reset + joint angles U(+/-0.05 rad) + extra joint velocities U(+/-0.1 rad/s) on all 6
                     actuated joints (angles clipped to joint ranges; the knee's range ends at its 0 rest angle)
MacGym nominal (20 episodes, its own reset distribution) is included for reference.
Writes web/deck/data/transfer_robust.json."""
import json, math, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mj_walker import ROOT, MjWalker, DT
import numpy as np

OUT = os.path.join(ROOT, "web/deck/data")
P = np.load(os.path.join(ROOT, "runs/walker.npz"))
W = [(P[f"actor.layers.{i}.weight"].astype(np.float64), P[f"actor.layers.{i}.bias"].astype(np.float64)) for i in range(3)]
mean, var = P["__obs_mean"].astype(np.float64), P["__obs_var"].astype(np.float64)


def act(obs):  # == PPO.act(deterministic=True)  (checked against MacGym in transfer.py)
    x = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10)
    for i, (w, b) in enumerate(W):
        x = x @ w.T + b
        if i < 2: x = np.tanh(x)
    return np.clip(x, -1, 1)


EPS, STEPS = 20, int(8.0 / DT)
FALL = "torso height < 0.875 m or |torso angle| > 1.0 rad (MacGym training termination); distance = torso x at fall or at 8 s"


def summarize(d, tf):
    d = np.asarray(d, float)
    return {"mean_distance_m": float(d.mean()), "std_distance_m": float(d.std(ddof=0)),
            "min_distance_m": float(d.min()), "max_distance_m": float(d.max()),
            "falls": int(sum(t is not None for t in tf)), "episodes": len(d),
            "distance_m": [round(float(x), 4) for x in d], "fall_times_s": tf}


# ---------------- MacGym nominal reference (legacy walker env, no auto-reset)
def macgym_nominal():
    import mlx.core as mx
    from macgym.envs.walker import Walker2D
    env = Walker2D(EPS, seed=123); M = env.model; B = M.B
    st = env._fresh(mx.random.key(7), EPS)
    x0 = np.array(st[:, 0]); alive = np.ones(EPS, bool); tfall = [None] * EPS; dist = np.zeros(EPS)
    sim = mx.compile(M.step_metal)
    for k in range(STEPS):
        o = np.array(env.observe(st), np.float64)
        st = sim(st, mx.array(act(o).astype(np.float32))); mx.eval(st)
        s = np.array(st)
        y, ang = s[:, B], s[:, 2 * B] - math.pi / 2
        fell = alive & ((y < 0.7 * 1.25) | (np.abs(ang) > 1.0))
        for i in np.where(fell)[0]: tfall[i] = (k + 1) * DT
        dist[alive] = s[alive, 0] - x0[alive]
        alive &= ~fell
    return summarize(dist, tfall)


# ---------------- MuJoCo conditions
def mujoco_condition(friction=1.0, mass=1.0, init_random=False, seed=123):
    w = MjWalker(speed_curve=True)
    m = w.model
    m.geom_friction[:, 0] *= friction
    m.body_mass[1:] *= mass
    m.body_inertia[1:] *= mass
    jr = m.jnt_range[3:9].copy()
    rng = np.random.default_rng(seed); rng2 = np.random.default_rng(seed + 1000)
    D, TF = [], []
    for ep in range(EPS):
        qv = np.concatenate([rng.uniform(-0.1, 0.1, 2), rng.uniform(-0.2, 0.2, 1), rng.uniform(-0.2, 0.2, 6)])
        dq, dqd = rng2.uniform(-0.05, 0.05, 6), rng2.uniform(-0.1, 0.1, 6)  # drawn always -> CRN
        w.reset(qv)
        if init_random:
            w.data.qpos[3:9] = np.clip(w.data.qpos[3:9] + dq, jr[:, 0], jr[:, 1])
            w.data.qvel[3:9] += dqd
            w.mj.mj_forward(m, w.data)
        x0 = w.torso()["x"]; tf = None; d = 0.0
        for k in range(STEPS):
            w.step(act(w.obs().astype(np.float64)))
            t = w.torso()
            if tf is None:
                d = t["x"] - x0
                if t["y"] < 0.7 * 1.25 or abs(t["angle"]) > 1.0:
                    tf = round((k + 1) * DT, 3)
                    break
        D.append(float(d)); TF.append(tf)
    total_mass = float(m.body_mass.sum())
    r = summarize(D, TF); r["total_mass_kg"] = total_mass
    r["floor_friction"] = float(m.geom_friction[0, 0])
    return r


if __name__ == "__main__":
    import mujoco
    t0 = time.time()
    conds = {
        "nominal": {}, "friction_x0.8": {"friction": 0.8}, "friction_x1.2": {"friction": 1.2},
        "mass_x0.8": {"mass": 0.8}, "mass_x1.2": {"mass": 1.2}, "init_random": {"init_random": True},
    }
    res = {}
    for name, kw in conds.items():
        res[name] = mujoco_condition(**kw); res[name]["params"] = kw
        r = res[name]
        print(f"{name:15s} {r['mean_distance_m']:7.2f} +/- {r['std_distance_m']:5.2f} m  falls {r['falls']}/{EPS}", flush=True)
    mg = macgym_nominal()
    print(f"{'macgym_nominal':15s} {mg['mean_distance_m']:7.2f} +/- {mg['std_distance_m']:5.2f} m  falls {mg['falls']}/{EPS}")
    out = {"policy": "runs/walker.npz (PPO, MLP 256x256 tanh, trained in MacGym), deterministic mean action",
           "episodes_per_condition": EPS, "episode_seconds": 8.0, "control_dt": DT, "fall_definition": FALL,
           "method": __doc__, "mujoco": res, "macgym_nominal_reference": mg,
           "mujoco_version": mujoco.__version__, "wall_seconds": round(time.time() - t0, 1),
           "uptime": subprocess.check_output(["uptime"], text=True).strip(),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    json.dump(out, open(os.path.join(OUT, "transfer_robust.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "transfer_robust.json"))
