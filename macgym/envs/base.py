"""Base class for batched MLX environments. See CONTRACT.md."""
import mlx.core as mx
import numpy as np


class BatchedEnv:
    name = "base"
    description = ""
    obs_dim = 0
    act_dim = 0
    max_episode_steps = 1000
    dt = 0.02

    def __init__(self, num_envs: int, seed: int = 0):
        self.num_envs = num_envs
        self.key = mx.random.key(seed)

    def _split(self):
        self.key, sub = mx.random.split(self.key)
        return sub

    def reset(self) -> mx.array:
        raise NotImplementedError

    def step(self, action: mx.array):
        raise NotImplementedError

    def render_state(self, R: int) -> dict:
        raise NotImplementedError

    def set_target(self, x: float, y: float) -> None:
        pass
