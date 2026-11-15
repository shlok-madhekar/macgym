"""Smoke test: PPO learns an inline batched Pendulum (and cartpole if registered).

Run: python3 scripts/test_ppo.py [--seconds 60] [--num-envs 4096]
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from macgym.envs.base import BatchedEnv  # noqa: E402
from macgym.rl.ppo import PPO  # noqa: E402

MAX_T = 200


def _obs(th, thd):
    return mx.stack([mx.cos(th), mx.sin(th), thd], axis=1)


def _pend_step(th, thd, t, ret, action, key):
    u = 2.0 * mx.clip(action[:, 0], -1.0, 1.0)
    ang = ((th + math.pi) % (2 * math.pi)) - math.pi
    reward = -(ang * ang + 0.1 * thd * thd + 0.001 * u * u)
    thd = mx.clip(thd + (3 * 10.0 / 2.0 * mx.sin(th) + 3.0 * u) * 0.05, -8.0, 8.0)
    th = th + thd * 0.05
    t = t + 1.0
    done = (t >= MAX_T).astype(mx.float32)
    ret = ret + reward
    ep_ret, ep_len = ret * done, t * done
    k1, k2 = mx.random.split(key)
    n = th.shape[0]
    d = done > 0.5
    th = mx.where(d, mx.random.uniform(-math.pi, math.pi, (n,), key=k1), th)
    thd = mx.where(d, mx.random.uniform(-1.0, 1.0, (n,), key=k2), thd)
    t = mx.where(d, 0.0, t)
    ret = mx.where(d, 0.0, ret)
    return th, thd, t, ret, _obs(th, thd), reward, done, ep_ret, ep_len


class Pendulum(BatchedEnv):
    name = "pendulum"
    obs_dim, act_dim, max_episode_steps, dt = 3, 1, MAX_T, 0.05

    def __init__(self, num_envs, seed=0):
        super().__init__(num_envs, seed)
        self._fn = mx.compile(_pend_step)
        self.reset()

    def reset(self):
        n = self.num_envs
        self.th = mx.random.uniform(-math.pi, math.pi, (n,), key=self._split())
        self.thd = mx.random.uniform(-1.0, 1.0, (n,), key=self._split())
        # stagger episode clocks so timeouts are spread out
        self.t = mx.floor(mx.random.uniform(0, MAX_T, (n,), key=self._split()))
        self.ret = mx.zeros((n,))
        return _obs(self.th, self.thd)

    def step(self, action):
        (self.th, self.thd, self.t, self.ret, obs, rew, done, er, el) = self._fn(
            self.th, self.thd, self.t, self.ret, action, self._split())
        return obs, rew, done, {"episode_return": er, "episode_len": el}


def run(env, seconds, label, max_iters=10_000, **kw):
    ppo = PPO(env, **kw)
    t0 = time.perf_counter()
    hist = []
    while time.perf_counter() - t0 < seconds and ppo.iter < max_iters:
        m = ppo.train_iter()
        hist.append(m)
        if m["iter"] % 10 == 0 or m["iter"] == 1:
            print(f"[{label}] it {m['iter']:4d} steps {m['env_steps']/1e6:7.2f}M sps {m['sps']/1e6:5.2f}M "
                  f"ep_ret {m['mean_ep_return']:8.1f} ep_len {m['mean_ep_len']:6.1f} "
                  f"rew {m['mean_reward']:7.3f} pl {m['policy_loss']:+.4f} vl {m['value_loss']:.4f} "
                  f"ent {m['entropy']:.3f}", flush=True)
    sps = np.median([h["sps"] for h in hist[2:]]) if len(hist) > 3 else hist[-1]["sps"]
    first = hist[min(9, len(hist) - 1)]["mean_ep_return"]  # iter 10 (iter 1 has partial episodes)
    print(f"[{label}] DONE {ppo.iter} iters in {time.perf_counter()-t0:.1f}s, "
          f"median sps {sps/1e6:.2f}M, ep_return {first:.1f} -> {hist[-1]['mean_ep_return']:.1f}")
    # deterministic eval + save/load roundtrip
    path = "/tmp/macgym_test_ppo.npz"
    ppo.save(path)
    ppo2 = PPO(env, **kw)
    ppo2.load(path)
    o = env.reset()
    assert mx.allclose(ppo.act(o, True), ppo2.act(o, True)).item(), "save/load mismatch"
    return ppo, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--num-envs", type=int, default=4096)
    a = ap.parse_args()
    _, hist = run(Pendulum(a.num_envs), a.seconds, "pendulum")
    assert hist[-1]["mean_ep_return"] > -400, "pendulum did not learn"
    try:
        from macgym.envs import make
        env = make("cartpole", a.num_envs)
    except Exception as e:  # registry may not exist yet
        print("cartpole skipped:", repr(e))
    else:
        run(env, 30, "cartpole")
