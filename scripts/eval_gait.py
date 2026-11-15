"""Gait plausibility metrics for any macgym robot (legacy env, mg.robots built-in, or preset file).

    python3 scripts/eval_gait.py walker --ckpt runs/walker.npz
    python3 scripts/eval_gait.py preset:tripod --ckpt runs/api_tripod.npz
    python3 scripts/eval_gait.py robots:tripod --random

Deterministic policy rollout (8 s by default). Reports, over envs that did not fall:
  speed (m/s), body-lengths/s, stance-foot slip speed (tangential velocity of every capsule
  end sphere touching the ground, median / p90), flight fraction (steps with no ground
  contact), penetration depth (mean / max, cm), fall fraction."""
import argparse
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macgym as mg  # noqa: E402
from macgym.rl.ppo import PPO  # noqa: E402

CONTACT_TOL = 0.004  # sphere bottom within 4 mm of the ground counts as touching


def make_env(spec, n):
    if spec.startswith("preset:"):
        from macgym.presets import DIR
        ns = {}
        path = os.path.join(DIR, spec.split(":", 1)[1] + ".py")
        exec(compile(open(path).read(), path, "exec"), ns)
        return mg.make(ns["robot"], ns["task"], num_envs=n)
    if spec.startswith("robots:"):
        from macgym import robots
        return mg.make(robots.get(spec.split(":", 1)[1]), num_envs=n)
    from macgym.envs import make
    return make(spec, n)


def body_length(model):
    caps = model.capsules_np(model.init_state[None])[0]
    xs = np.concatenate([caps[:, 0] - caps[:, 4], caps[:, 2] + caps[:, 4]])
    ys = np.concatenate([caps[:, 1] - caps[:, 4], caps[:, 3] + caps[:, 4], caps[:, 1] + caps[:, 4]])
    return float(max(xs.max() - xs.min(), ys.max() - min(ys.min(), 0.0)))  # largest extent


def rollout(env, ppo=None, steps=400, seed=0):
    """Returns metrics dict. ppo=None -> uniform random actions."""
    model = env.model
    B = model.B
    half = np.asarray(model.half)[None]
    rad = np.asarray(model.rad)[None]
    obs = env.reset()
    n = env.num_envs
    x0 = np.array(env.state[:, env.torso])
    alive = np.ones(n, bool)
    x_last = x0.copy()
    slips, pens, contact_steps, total_steps = [], [], 0, 0
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if ppo is None:
            a = mx.array(rng.uniform(-1, 1, (n, env.act_dim)).astype(np.float32))
        else:
            a = ppo.act(obs, deterministic=True)
        obs, rew, done, info = env.step(a)
        s = np.array(env.state).reshape(n, 6, B)
        d = np.array(done) > 0
        alive &= ~d
        x_last = np.where(alive, s[:, 0, env.torso], x_last)
        if not alive.any():
            break
        st = s[alive]
        th, vx, w = st[:, 2], st[:, 3], st[:, 5]
        any_c = np.zeros(len(st), bool)
        for sg in (-1.0, 1.0):
            ox = sg * half * np.cos(th)
            oy = sg * half * np.sin(th) - rad
            bottom = st[:, 1] + oy
            vt = vx - w * oy
            c = bottom < CONTACT_TOL
            slips.append(np.abs(vt[c]))
            pens.append(np.maximum(-bottom[c], 0.0))
            any_c |= c.any(1)
        contact_steps += int(any_c.sum())
        total_steps += len(st)
    T = steps * env.dt
    slips = np.concatenate(slips) if slips else np.zeros(1)
    pens = np.concatenate(pens) if pens else np.zeros(1)
    L = body_length(model)
    sp = (x_last - x0)[alive] / T if alive.any() else np.zeros(1)
    return {
        "speed": float(np.median(sp)), "bl_per_s": float(np.median(sp)) / L, "length": L,
        "slip_med": float(np.median(slips)), "slip_p90": float(np.percentile(slips, 90)),
        "flight": 1.0 - contact_steps / max(total_steps, 1),
        "pen_mean_cm": 100 * float(pens.mean()), "pen_max_cm": 100 * float(pens.max()),
        "fell": float(1.0 - alive.mean()), "seconds": T,
    }


def load_ppo(env, ckpt):
    h = tuple(int(v) for v in np.load(ckpt)["__hidden"])
    ppo = PPO(env, hidden=h)
    ppo.load(ckpt)
    return ppo


def fmt(name, m):
    return (f"{name:<22s} speed {m['speed']:5.2f} m/s ({m['bl_per_s']:4.2f} BL/s, L={m['length']:.2f} m) | "
            f"slip med {m['slip_med']:.3f} p90 {m['slip_p90']:.3f} m/s | flight {m['flight']:.2f} | "
            f"pen {m['pen_mean_cm']:.2f}/{m['pen_max_cm']:.2f} cm | fell {m['fell']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("robot", help="walker | cheetah | preset:<name> | robots:<name>")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--random", action="store_true")
    a = ap.parse_args()
    env = make_env(a.robot, a.n)
    ppo = None if (a.random or not a.ckpt) else load_ppo(env, a.ckpt)
    m = rollout(env, ppo, steps=int(round(a.seconds / env.dt)))
    print(fmt(a.robot, m))


if __name__ == "__main__":
    main()
