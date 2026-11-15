"""Public API test: builds every built-in robot + a custom inline robot, steps them at
N=4096 (no NaNs, steps/s), checks legacy obs parity, then trains hopper and tripod.

    python3 tests/test_api.py            # full (includes ~60-iter training runs)
    python3 tests/test_api.py --quick    # skip training
"""
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macgym as mg  # noqa: E402

N = 4096
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def custom_biped():
    robot = mg.Robot("biped2")
    hip = robot.body("hip", start=(-0.2, 0.9), end=(0.2, 0.9), radius=0.06)
    for side, x in (("l", -0.15), ("r", 0.15)):
        thigh = robot.body(side + "_thigh", start=(x, 0.9), end=(x, 0.5), radius=0.04)
        shin = robot.body(side + "_shin", start=(x, 0.5), end=(x, 0.06), radius=0.035)
        robot.joint(hip, thigh, at=(x, 0.9), limits=(-1.0, 1.0), motor=80, name=side + "_hip")
        robot.joint(thigh, shin, at=(x, 0.5), limits=(-1.5, 0.0), motor=60, name=side + "_knee")
    robot.set_root(hip)
    task = mg.Task(
        reward=lambda s: s.root.vx + 0.5 * s.contact("l_shin") * 0 - 0.001 * s.action_sq
        + 0.01 * s.joint("l_knee").q * 0 + 0.0 * s.body("r_thigh").omega + 0.0 * s.t,
        done=lambda s: s.root.y < 0.4,
        episode_steps=500,
    )
    return robot, task


def step_check(name, robot, task, steps=200):
    env = mg.make(robot, task, num_envs=N, seed=0)
    assert env.obs.shape == (N, env.obs_dim), env.obs.shape
    key = mx.random.key(1)
    acts = mx.random.uniform(-1, 1, (steps + 5, N, env.act_dim), key=key)
    mx.eval(acts)
    for i in range(5):  # warmup / compile
        o, r, d, info = env.step(acts[i])
    mx.eval(o, r, d)
    bad = False
    t0 = time.perf_counter()
    for i in range(steps):
        o, r, d, info = env.step(acts[5 + i])
        bad_i = mx.any(mx.isnan(o)) | mx.any(mx.isnan(r))
        mx.eval(o, r, d, bad_i)
        bad = bad or bool(bad_i)
    dt = time.perf_counter() - t0
    sps = steps * N / dt
    rs = env.render_state(8)
    for k in ("capsules", "colors", "camera_x", "start_x", "dist", "world_bounds"):
        assert k in rs, k
    assert set(info) >= {"episode_return", "episode_len", "episode_dist"}
    print(f"  {name:8s} bodies={len(robot.bodies)} joints={len(robot.joints)} obs={env.obs_dim:3d} "
          f"act={env.act_dim} | {sps / 1e6:5.2f}M steps/s | nan={bad}", flush=True)
    assert not bad, f"{name}: NaN"
    assert sps > 1e6, f"{name}: too slow ({sps:.0f} steps/s)"
    return env


def legacy_parity():
    """Built-in walker/cheetah (public API) vs legacy envs: same obs on the same state, and the
    old checkpoints load and run."""
    from macgym.envs import make as make_legacy
    for name in ("walker", "cheetah"):
        new = mg.make(mg.robots.get(name), num_envs=64)
        old = make_legacy(name, 64)
        assert new.obs_dim == old.obs_dim and new.act_dim == old.act_dim
        st = old.state
        err = float(mx.max(mx.abs(new.observe(st) - old.observe(st))))
        ckpt = os.path.join(ROOT, "runs", f"{name}.npz")
        msg = ""
        if os.path.exists(ckpt):
            pol = mg.load_policy(ckpt, new)
            new_e = mg.make(mg.robots.get(name), num_envs=256)
            ev = mg.evaluate(new_e, pol, steps=300)
            msg = f"| runs/{name}.npz on public-API env: {ev['dist']:.2f} m in 6 s"
        print(f"  {name}: obs parity max err {err:.2e} {msg}", flush=True)
        assert err < 1e-4


def train_check(name, iters=60):
    env = mg.make(mg.robots.get(name), num_envs=N, seed=0)
    eval_env = mg.make(mg.robots.get(name), num_envs=1024, seed=123)
    before = mg.evaluate(eval_env, None, steps=250)
    seen = []
    res = mg.train(env, iters=iters, hidden=(256, 256), on_iter=lambda m: seen.append(m["iter"]),
                   log_every=10, save=os.path.join(ROOT, "runs", f"api_{name}.npz"))
    assert len(seen) == iters and len(res.history) == iters
    after = mg.evaluate(eval_env, res.policy, steps=250)
    pol = mg.load_policy(os.path.join(ROOT, "runs", f"api_{name}.npz"), env)
    after2 = mg.evaluate(eval_env, pol, steps=250)
    print(f"  {name}: distance in 5 s  before {before['dist']:+.2f} m -> after {after['dist']:+.2f} m "
          f"(reloaded {after2['dist']:+.2f} m) | return {before['return']:.1f} -> {after['return']:.1f} | "
          f"survived {before['survived']:.0%} -> {after['survived']:.0%}", flush=True)
    assert after["dist"] > before["dist"] + 0.3, f"{name} did not learn to move forward"
    assert abs(after2["dist"] - after["dist"]) < 0.05
    return before, after


def main():
    quick = "--quick" in sys.argv
    print("build + step 200 @ N=4096:")
    for name, fn in mg.robots.BUILTINS.items():
        step_check(name, *fn())
    step_check("biped2", *custom_biped())

    print("errors:")
    r = mg.Robot("bad")
    a = r.body("a", start=(0, 1), end=(0, 0.5))
    for fn, exc in ((lambda: r.body("a", start=(0, 0), end=(1, 0)), ValueError),
                    (lambda: r.joint(a, a, at=(0, 1)), ValueError),
                    (lambda: mg.make(r, mg.Task(reward=lambda s: s.root.vx)), ValueError)):
        try:
            fn()
        except exc as e:
            print(f"  ok: {e}")
        else:
            raise AssertionError("expected error")

    print("legacy parity:")
    legacy_parity()
    if quick:
        return
    print("training:")
    for name in ("hopper", "tripod"):
        train_check(name)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
