"""Stability + throughput test for the MLX 2D locomotion envs.

Run from repo root:  python3 scripts/test_locomotion.py  (or python3 -m scripts.test_locomotion)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from macgym.envs.cheetah import Cheetah2D  # noqa: E402
from macgym.envs.walker import Walker2D  # noqa: E402

N = 4096
STEPS = 500
ok = True


def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg)
    ok = ok and bool(cond)


def joint_error(model, state):
    """max world-space separation between joint anchors (m)."""
    s = np.array(state).reshape(-1, 6, model.B)
    px, py, th = s[:, 0], s[:, 1], s[:, 2]
    err = 0.0
    for jn, j in enumerate(model.joints):
        pts = []
        for body, (lx, ly) in ((j.a, model.la[jn]), (j.b, model.lb[jn])):
            c, sn = np.cos(th[:, body]), np.sin(th[:, body])
            pts.append((px[:, body] + c * lx - sn * ly, py[:, body] + sn * lx + c * ly))
        err = max(err, float(np.max(np.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]))))
    return err


for Env in (Walker2D, Cheetah2D):
    print(f"\n== {Env.name} (N={N}) ==")
    env = Env(N, seed=1)
    obs = env.reset()
    check(obs.shape == (N, env.obs_dim), f"obs shape {obs.shape}, act_dim {env.act_dim}")
    key = mx.random.key(0)
    # warmup / compile
    o, r, d, info = env.step(mx.zeros((N, env.act_dim)))
    mx.eval(o, r, d)
    acts = [mx.random.uniform(-1, 1, (N, env.act_dim), key=k) for k in mx.random.split(key, 8)]
    mx.eval(acts)
    max_pos, nan_any, dones, rew = 0.0, False, 0.0, 0.0
    t0 = time.time()
    for i in range(STEPS):
        o, r, d, info = env.step(acts[i % 8])
        if i % 50 == 49:
            mx.eval(o, r, d)
            s = np.array(env.state)
            B = env.model.B
            max_pos = max(max_pos, float(np.abs(s[:, B:2 * B]).max()))  # heights
            nan_any |= bool(np.isnan(s).any() or np.isnan(np.array(o)).any())
        dones += float(d.sum()) if i % 50 == 49 else 0.0
        rew += float(r.mean()) if i % 50 == 49 else 0.0
    mx.eval(o, r, d)
    el = time.time() - t0
    sps = N * STEPS / el
    print(f"  env-steps/sec: {sps:,.0f}  ({el:.2f}s for {STEPS} steps)")
    s = np.array(env.state)
    B = env.model.B
    xs = s[:, :B] - s[:, :1]
    check(not nan_any and np.isfinite(s).all(), "no NaNs in state/obs")
    check(max_pos < 5.0, f"max |y| sane: {max_pos:.2f} m")
    check(np.abs(xs).max() < 3.0, f"bodies stay attached (max |x - torso x| {np.abs(xs).max():.2f} m)")
    je = joint_error(env.model, env.state)
    check(je < 0.05, f"max joint separation under random torques: {je*1000:.1f} mm")
    rs = env.render_state(16)
    check(rs["capsules"].shape == (16, B, 5) and len(rs["colors"]) == B and rs["camera_x"].shape == (16,),
          "render_state shapes")

    # zero-action drop test (raw physics, no auto-reset)
    sim = env._sim_fn
    st = env._fresh(mx.random.key(3), 256)
    zero = mx.zeros((256, env.act_dim))
    y0 = float(np.array(st)[:, B].mean())
    mins = []
    for i in range(300):  # 6 s
        st = sim(st, zero)
        if i % 10 == 9:
            caps = env.model.capsules_np(np.array(st))
            mins.append(float((np.minimum(caps[..., 1], caps[..., 3]) - caps[..., 4]).min()))
    sn = np.array(st)
    y1 = float(sn[:, B].mean())
    v = np.abs(sn[:, 3 * B:5 * B]).max()
    print(f"  zero-action: torso y {y0:.2f} -> {y1:.2f}, min body bottom {min(mins):+.3f} m, "
          f"final max |v| {v:.3f} m/s, joint err {joint_error(env.model, st)*1000:.1f} mm")
    check(np.isfinite(sn).all(), "drop test finite")
    if Env is Walker2D:
        check(y1 < 0.7 * y0, "passive walker falls over")
    check(min(mins) > -0.05, "no deep ground penetration (> -5 cm)")
    check(v < 0.5, "comes to rest after landing")

    # backend agreement (metal kernel vs pure-mx reference)
    st = env._fresh(mx.random.key(5), 64)
    a = mx.random.uniform(-1, 1, (64, env.act_dim), key=mx.random.key(6))
    s_metal = env.model.step_metal(st, a)
    s_mx = env.model.step(st, a)
    diff = float(mx.abs(s_metal - s_mx).max())
    check(diff < 1e-2, f"metal vs mx backend max diff after 1 step: {diff:.2e}")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
