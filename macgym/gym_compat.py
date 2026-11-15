"""Gymnasium adapter: wrap any MacGym batched env as a ``gymnasium.vector.VectorEnv``.

    import macgym as mg
    venv = mg.to_gymnasium(mg.make(*mg.robots.walker(), num_envs=64))
    obs, info = venv.reset(seed=0)
    obs, rew, terminated, truncated, info = venv.step(venv.action_space.sample())

Data: numpy float32 in / out. Observation space Box(-inf, inf, (N, obs_dim)); action space
Box(-1, 1, (N, act_dim)) (actions are clipped to [-1, 1]; MacGym motors scale them).

Autoreset semantics (``metadata["autoreset_mode"] = SAME_STEP``): the simulation runs entirely
on the GPU and resets finished sub-envs inside the same ``step`` call. So when
``terminated[i] or truncated[i]`` is True, ``obs[i]`` is already the FIRST observation of the next
episode. The terminal observation itself is not kept (``info["final_obs"]`` is not provided);
bootstrapping on truncation should use the value of the reset obs or ignore it, which is what
most PPO implementations (CleanRL, SB3 VecEnv) already do. The reward returned on a done step
belongs to the episode that just ended.

Truncation vs termination: a done env whose episode length reached ``max_episode_steps`` is
reported as ``truncated``; any other done (fell over, ``Task(done=...)``) is ``terminated``.
Episode stats for finished envs are in ``info["episode_return"|"episode_len"|"episode_dist"]``
(numpy, zero for envs that did not finish this step) with mask ``info["_episode"]``.

Seeding: MacGym envs are seeded at construction; ``reset(seed=...)`` reseeds the env's MLX key
when the env exposes ``key`` (BatchedEnv does).
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium.vector import VectorEnv
    from gymnasium.vector.utils import batch_space
except ImportError as e:  # pragma: no cover
    raise ImportError("to_gymnasium needs `pip install gymnasium` (or macgym[gym])") from e

try:
    from gymnasium.vector import AutoresetMode
    _SAME_STEP = AutoresetMode.SAME_STEP
except ImportError:  # gymnasium < 1.1
    _SAME_STEP = "SameStep"


class MacGymVectorEnv(VectorEnv):
    metadata = {"autoreset_mode": _SAME_STEP, "render_modes": []}

    def __init__(self, env):
        self.env = env
        self.num_envs = int(env.num_envs)
        self.obs_dim, self.act_dim = int(env.obs_dim), int(env.act_dim)
        self.max_episode_steps = getattr(env, "max_episode_steps", None) or getattr(env, "episode_steps", None)
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.render_mode = None
        self.spec = None
        self.closed = False

    def reset(self, *, seed=None, options=None):
        import mlx.core as mx
        if seed is not None:
            self.action_space.seed(seed)
            if hasattr(self.env, "key"):
                self.env.key = mx.random.key(int(seed))
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, actions):
        import mlx.core as mx
        a = np.clip(np.asarray(actions, dtype=np.float32).reshape(self.num_envs, self.act_dim), -1.0, 1.0)
        obs, rew, done, info = self.env.step(mx.array(a))
        done = np.array(done).astype(bool)
        np_info = {k: np.array(v) for k, v in (info or {}).items() if hasattr(v, "shape")}
        ep_len = np_info.get("episode_len")
        if self.max_episode_steps and ep_len is not None:
            truncated = done & (ep_len >= self.max_episode_steps)
        else:
            truncated = np.zeros_like(done)
        terminated = done & ~truncated
        np_info["_episode"] = done
        return (np.array(obs, dtype=np.float32), np.array(rew, dtype=np.float32).reshape(self.num_envs),
                terminated, truncated, np_info)

    def close_extras(self, **kwargs):
        pass


def to_gymnasium(env):
    """Wrap a MacGym batched env (mg.make / macgym.envs.make) as a gymnasium VectorEnv."""
    return MacGymVectorEnv(env)
