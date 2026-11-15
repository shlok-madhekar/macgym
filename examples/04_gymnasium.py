"""Use a MacGym env from any Gymnasium-based library (SB3, CleanRL, ...).

to_gymnasium returns a gymnasium.vector.VectorEnv with numpy float32 in/out.
Autoreset is SAME_STEP: when an env finishes, the obs returned for it is already the
first obs of its next episode (see macgym/gym_compat.py).

    python examples/04_gymnasium.py
"""
import os

import numpy as np

import macgym as mg

N = int(os.environ.get("MACGYM_ENVS", 256))
STEPS = int(os.environ.get("MACGYM_STEPS", 500))

venv = mg.to_gymnasium(mg.make(*mg.robots.walker(), num_envs=N))
print("obs space:", venv.single_observation_space, " action space:", venv.single_action_space)

obs, info = venv.reset(seed=0)
episodes, returns = 0, []
for _ in range(STEPS):
    actions = venv.action_space.sample()                     # random policy
    obs, reward, terminated, truncated, info = venv.step(actions)
    done = terminated | truncated
    episodes += int(done.sum())
    returns += list(info["episode_return"][done])
assert obs.shape == (N, venv.single_observation_space.shape[0]) and obs.dtype == np.float32
print(f"{STEPS} steps x {N} envs, {episodes} episodes finished, "
      f"mean random-policy return {np.mean(returns) if returns else 0:.1f}")
venv.close()
