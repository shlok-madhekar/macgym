"""Basic smoke + throughput test for cartpole and arm envs.
Run: python3 scripts/test_envs_basic.py  (from repo root)"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

from macgym.envs import ENVS, IMPORT_ERRORS, make

N = 4096
STEPS = 1000


def test_env(name):
    env = make(name, N, seed=0)
    obs = env.reset()
    assert obs.shape == (N, env.obs_dim), obs.shape
    key = mx.random.key(1)
    # warmup (compile)
    for _ in range(5):
        key, k = mx.random.split(key)
        obs, r, d, info = env.step(mx.random.uniform(-1, 1, (N, env.act_dim), key=k))
    mx.eval(obs, r, d)
    total_r = mx.zeros((N,))
    n_done = 0.0
    ret_sum = 0.0
    t0 = time.perf_counter()
    for i in range(STEPS):
        key, k = mx.random.split(key)
        a = mx.random.uniform(-1, 1, (N, env.act_dim), key=k)
        obs, r, d, info = env.step(a)
        total_r = total_r + r
        if i % 50 == 49:
            mx.eval(obs, r, d, total_r)
    mx.eval(obs, r, d, total_r, info["episode_return"], info["episode_len"])
    dt = time.perf_counter() - t0
    sps = N * STEPS / dt
    assert obs.shape == (N, env.obs_dim)
    assert r.shape == (N,) and d.shape == (N,)
    for nm, arr in [("obs", obs), ("reward", total_r)]:
        assert not np.isnan(np.array(arr)).any(), f"{name}: NaN in {nm}"
    rs = env.render_state(16)
    assert not np.isnan(rs["capsules"]).any()
    shapes = {k: (v.shape if isinstance(v, np.ndarray) else v) for k, v in rs.items()}
    print(f"[{name}] {sps/1e6:.2f}M env-steps/s  ({dt:.2f}s for {STEPS} steps x {N})")
    print(f"   mean reward/step {float(total_r.mean())/STEPS:.3f}, obs absmax {float(mx.abs(obs).max()):.2f}")
    print(f"   render_state: {shapes}")
    # episode stats sanity
    d_np = np.array(d)
    if d_np.any():
        print(f"   last-step done: {int(d_np.sum())}, ep_len there: {np.array(info['episode_len'])[d_np > 0][:3]}")
    if name == "arm":
        env.set_target(0.5, 0.8)
        o, *_ = env.step(mx.zeros((N, env.act_dim)))
        tg = np.array(env.target)
        assert np.allclose(tg, tg[0]), "set_target not applied"
        print(f"   set_target(0.5,0.8) -> clamped {tg[0]}")
        env.set_target(None)
    return sps


if __name__ == "__main__":
    if IMPORT_ERRORS:
        print("import errors (other envs):", list(IMPORT_ERRORS))
    for name in ["cartpole", "arm"]:
        test_env(name)
    print("OK")
