"""Batched classic CartPole (continuous force) in MLX. See CONTRACT.md."""
import math

import mlx.core as mx
import numpy as np

from macgym.envs.base import BatchedEnv

GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
HALF_LEN = 0.5  # half pole length (m)
POLE_ML = MASS_POLE * HALF_LEN
FORCE_MAG = 10.0
DT = 0.02
THETA_LIMIT = 12.0 * math.pi / 180.0
X_LIMIT = 2.4
MAX_STEPS = 500
CART_Y = 0.15


def _reset_states(key, n):
    return mx.random.uniform(-0.05, 0.05, (n, 4), key=key)


def _physics(state, t, ret, action, key):
    """One control step for all envs + auto-reset. Pure mx ops (compiled)."""
    n = state.shape[0]
    x, x_dot, theta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    force = FORCE_MAG * mx.clip(action[:, 0], -1.0, 1.0)
    cos_t = mx.cos(theta)
    sin_t = mx.sin(theta)
    temp = (force + POLE_ML * theta_dot * theta_dot * sin_t) / TOTAL_MASS
    theta_acc = (GRAVITY * sin_t - cos_t * temp) / (
        HALF_LEN * (4.0 / 3.0 - MASS_POLE * cos_t * cos_t / TOTAL_MASS)
    )
    x_acc = temp - POLE_ML * theta_acc * cos_t / TOTAL_MASS
    # semi-implicit Euler
    x_dot = x_dot + DT * x_acc
    x = x + DT * x_dot
    theta_dot = theta_dot + DT * theta_acc
    theta = theta + DT * theta_dot
    new_state = mx.stack([x, x_dot, theta, theta_dot], axis=1)

    t = t + 1.0
    reward = mx.ones((n,), dtype=mx.float32)
    fell = (mx.abs(theta) > THETA_LIMIT) | (mx.abs(x) > X_LIMIT)
    timeout = t >= MAX_STEPS
    done = (fell | timeout).astype(mx.float32)
    ret = ret + reward
    ep_return = ret * done
    ep_len = t * done

    fresh = _reset_states(key, n)
    d = done[:, None] > 0.5
    new_state = mx.where(d, fresh, new_state)
    t = mx.where(done > 0.5, 0.0, t)
    ret = mx.where(done > 0.5, 0.0, ret)
    return new_state, t, ret, reward, done, ep_return, ep_len


class CartPole(BatchedEnv):
    name = "cartpole"
    description = "Classic cart-pole balance, continuous force. Reward +1 per step upright."
    obs_dim = 4
    act_dim = 1
    max_episode_steps = MAX_STEPS
    dt = DT

    def __init__(self, num_envs: int, seed: int = 0):
        super().__init__(num_envs, seed)
        self._step_fn = mx.compile(_physics)
        self.reset()

    def reset(self) -> mx.array:
        n = self.num_envs
        self.state = _reset_states(self._split(), n)
        self.t = mx.zeros((n,), dtype=mx.float32)
        self.ret = mx.zeros((n,), dtype=mx.float32)
        mx.eval(self.state, self.t, self.ret)
        return self.state

    def step(self, action: mx.array):
        action = action.astype(mx.float32).reshape(self.num_envs, self.act_dim)
        (self.state, self.t, self.ret, reward, done, ep_ret, ep_len) = self._step_fn(
            self.state, self.t, self.ret, action, self._split()
        )
        info = {"episode_return": ep_ret, "episode_len": ep_len}
        return self.state, reward, done, info

    def render_state(self, R: int) -> dict:
        R = min(R, self.num_envs)
        s = np.array(self.state[:R], dtype=np.float32)
        x, theta = s[:, 0], s[:, 2]
        caps = np.zeros((R, 2, 5), dtype=np.float32)
        # cart: fat horizontal capsule
        caps[:, 0] = np.stack(
            [x - 0.2, np.full(R, CART_Y), x + 0.2, np.full(R, CART_Y), np.full(R, 0.12)], axis=1
        )
        # pole
        tip_x = x + 2 * HALF_LEN * np.sin(theta)
        tip_y = CART_Y + 2 * HALF_LEN * np.cos(theta)
        caps[:, 1] = np.stack([x, np.full(R, CART_Y), tip_x, tip_y, np.full(R, 0.03)], axis=1)
        return {
            "capsules": caps,
            "targets": None,
            "colors": ["#4F8CFF", "#FFB547"],
            "ground": 0.0,
            "world_bounds": [-2.8, 2.8, -0.3, 1.5],
            "dist": np.array(self.t[:R], dtype=np.float32) * DT,  # seconds balanced so far
        }

    def live_balance_mean(self) -> float:
        return float(mx.mean(self.t)) * DT
