"""Gait diagnostics: python3 scripts/eval_gait.py --env walker --ckpt runs/walker.npz --hidden 256,256
Rolls out the deterministic policy, prints torso/velocity/contact/energy stats and writes a
contact-sheet PNG (one env, every 10 steps) to runs/<env>_gait.png."""
import argparse
import math
import os
import sys

import cv2
import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macgym.envs import make  # noqa: E402
from macgym.envs.physics2d import GRAVITY  # noqa: E402
from macgym.rl.ppo import PPO  # noqa: E402


def energy(model, s):
    """s (N, 6B) numpy -> total mechanical energy (N,)."""
    B = model.B
    s = s.reshape(-1, 6, B)
    m = np.asarray(model.masses)[None]
    I = 1.0 / np.asarray(model.inv_i)[None]
    ke = 0.5 * m * (s[:, 3] ** 2 + s[:, 4] ** 2) + 0.5 * I * s[:, 5] ** 2
    pe = -GRAVITY * m * s[:, 1]
    return (ke + pe).sum(1)


def hexcol(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def draw_frame(caps, colors, cam_x, step, W=220, Hh=180, scale=80.0, ground_px=150):
    img = np.full((Hh, W, 3), (18, 13, 11), np.uint8)
    cv2.line(img, (0, ground_px), (W, ground_px), (90, 90, 90), 1)
    for gx in np.arange(math.floor(cam_x - 2), cam_x + 2, 0.5):  # ground ticks show motion
        u = int(W / 2 + (gx - cam_x) * scale)
        cv2.line(img, (u, ground_px), (u, ground_px + 5), (90, 90, 90), 1)
    for k, (x1, y1, x2, y2, r) in enumerate(caps):
        p1 = (int(W / 2 + (x1 - cam_x) * scale), int(ground_px - y1 * scale))
        p2 = (int(W / 2 + (x2 - cam_x) * scale), int(ground_px - y2 * scale))
        cv2.line(img, p1, p2, hexcol(colors[k]), max(1, int(2 * r * scale)), cv2.LINE_AA)
    cv2.putText(img, f"t={step}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="walker")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--random", action="store_true", help="random actions instead of policy")
    a = ap.parse_args()

    env = make(a.env, a.n)
    model = env.model
    B, J = model.B, model.J
    ppo = PPO(env, hidden=tuple(int(h) for h in a.hidden.split(",")))
    if a.ckpt:
        ppo.load(a.ckpt)
    obs = env.reset()
    gears = np.asarray(model.gears)
    feet = getattr(env, "feet", ())
    rng = np.random.default_rng(0)

    rows, frames, done_count, ep_rets = [], [], 0, []
    alive = np.ones(a.n, bool)  # envs that have not yet reset (energy bookkeeping valid)
    E_prev = energy(model, np.array(env.state))
    inj = np.zeros(a.n)
    work = np.zeros(a.n)
    for t in range(a.steps):
        s0 = np.array(env.state)
        if a.random:
            act = mx.array(rng.uniform(-1, 1, (a.n, env.act_dim)).astype(np.float32))
        else:
            act = ppo.act(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        mx.eval(obs, rew, done)
        s1 = np.array(env.state)
        d = np.array(done) > 0
        A = np.clip(np.array(act), -1, 1)
        w0, w1 = s0.reshape(-1, 6, B)[:, 5], s1.reshape(-1, 6, B)[:, 5]
        qd0 = np.stack([w0[:, j.b] - w0[:, j.a] for j in model.joints], 1)
        qd1 = np.stack([w1[:, j.b] - w1[:, j.a] for j in model.joints], 1)
        th0, th1 = s0.reshape(-1, 6, B)[:, 2], s1.reshape(-1, 6, B)[:, 2]
        dq = np.stack([(th1[:, j.b] - th1[:, j.a]) - (th0[:, j.b] - th0[:, j.a]) for j in model.joints], 1)
        W_act = (gears * A * dq).sum(1)  # exact: torque constant over the control step
        # foot slip: lowest-point x speed of each foot while in contact
        slip = []
        for f in feet:
            st0, st1 = s0.reshape(-1, 6, B), s1.reshape(-1, 6, B)
            hx1 = model.half[f] * np.cos(st1[:, 2, f]); hy1 = model.half[f] * np.sin(st1[:, 2, f])
            hx0 = model.half[f] * np.cos(st0[:, 2, f]); hy0 = model.half[f] * np.sin(st0[:, 2, f])
            sg1 = np.where(hy1 > 0, -1, 1); sg0 = sg1
            x1 = st1[:, 0, f] + sg1 * hx1; x0 = st0[:, 0, f] + sg0 * hx0
            y1 = st1[:, 1, f] + sg1 * hy1 - model.rad[f]
            slip.append(np.where((y1 < 0.01) & ~d, np.abs(x1 - x0) / env.dt, np.nan))
        E1 = energy(model, s1)
        dE = E1 - E_prev
        ok = alive & ~d
        inj[ok] += np.maximum(dE - W_act, 0)[ok]
        work[ok] += np.abs(W_act)[ok]
        alive &= ~d
        E_prev = E1
        if d.any():
            done_count += int(d.sum())
            ep_rets += list(np.array(info["episode_return"])[d])
        st = s1.reshape(-1, 6, B)
        vx_t = (s1[:, 0] - s0[:, 0]) / env.dt
        vx_t[d] = np.nan
        contacts = [np.array(c) for c in env.foot_contacts(env.state, feet)] if feet else []
        rows.append(dict(y=st[:, 1, 0], ang=st[:, 2, 0] - model.init_state[2 * B], vx=vx_t,
                         c=np.stack(contacts, 1) if contacts else None,
                         qd=np.abs(qd1).max(1), r=np.array(rew), dE=dE - W_act, slip=np.stack(slip, 1) if slip else None))
        if t % 10 == 0:
            caps = model.capsules_np(s1[:1])[0]
            frames.append(draw_frame(caps, env.colors, float(st[0, 0, 0]), t))

    vx = np.stack([r["vx"] for r in rows])
    y = np.stack([r["y"] for r in rows])
    ang = np.stack([r["ang"] for r in rows])
    qd = np.stack([r["qd"] for r in rows])
    R = np.stack([r["r"] for r in rows])
    print(f"env={a.env} ckpt={a.ckpt} N={a.n} steps={a.steps}")
    print(f"  reward/step mean {R.mean():.3f} | x-vel mean {np.nanmean(vx):.2f} m/s (median {np.nanmedian(vx):.2f}, max {np.nanmax(vx):.2f})")
    print(f"  torso y mean {y.mean():.3f} min {y.min():.3f} max {y.max():.3f} | torso angle mean {ang.mean():.3f} std {ang.std():.3f}")
    print(f"  max joint vel (rad/s) median {np.median(qd):.1f} p99 {np.percentile(qd, 99):.1f}")
    if rows[0]["c"] is not None:
        C = np.stack([r["c"] for r in rows])  # (T, N, F)
        print(f"  foot contact fraction {C.mean((0, 1)).round(2)} | both-air {np.mean(C.sum(2) == 0):.2f} | both-ground {np.mean(C.sum(2) == len(feet)):.2f}")
        pat = "".join("".join("X" if C[t, 0, f] else "." for f in range(len(feet))) + " " for t in range(0, min(120, a.steps), 2))
        print(f"  env0 contact pattern (every 2 steps, feet R/L): {pat}")
    if rows[0]["slip"] is not None:
        SL = np.stack([r["slip"] for r in rows])
        print(f"  stance foot slip speed: median {np.nanmedian(SL):.2f} m/s p90 {np.nanpercentile(SL, 90):.2f} m/s")
    print(f"  dones {done_count} ep_rets {np.round(ep_rets[:8], 1)}")
    print(f"  energy injected (sum max(dE-W_act,0)) per env mean {inj.mean():.1f} J vs |actuator work| {work.mean():.1f} J (ratio {inj.mean() / max(work.mean(), 1e-6):.2f})")
    dEs = np.stack([r["dE"] for r in rows])
    print(f"  per-step dE-W_act: mean {dEs.mean():+.3f} J p99 {np.percentile(dEs, 99):+.2f} J max {dEs.max():+.2f} J")

    cols = 8
    while len(frames) % cols:
        frames.append(np.zeros_like(frames[0]))
    sheet = np.vstack([np.hstack(frames[i:i + cols]) for i in range(0, len(frames), cols)])
    out = a.out or f"runs/{a.env}_gait.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, sheet)
    print("  wrote", out)


if __name__ == "__main__":
    main()
