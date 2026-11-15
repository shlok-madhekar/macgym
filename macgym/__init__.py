"""MacGym: robot-learning framework for Apple Silicon (MLX).

    import macgym as mg
    robot, task = mg.robots.hopper()
    env = mg.make(robot, task, num_envs=4096)
    result = mg.train(env, iters=200, save="runs/hopper.npz")
"""
import importlib

from macgym.api import (Body, JointHandle, Robot, RobotEnv, State, Task, TrainResult, default_obs,
                        evaluate, make, train)
from macgym.policy import Policy, load_policy
from macgym import robots

__version__ = "0.3.0"


def _lazy(candidates, attr, what):
    for mod in candidates:
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name != mod:
                raise
            continue
        if hasattr(m, attr):
            return getattr(m, attr)
    raise NotImplementedError(f"mg.{what} is not available (looked for {attr} in {candidates})")


def view(env=None, policy=None, port=8765, **kwargs):
    """Launch the live server + viewer for env (trains live unless a policy is given)."""
    fn = _lazy(["macgym.viewer", "macgym.view", "macgym.server"], "view", "view")
    return fn(env, policy=policy, port=port, **kwargs)


def to_gymnasium(env, **kwargs):
    """gymnasium.vector.VectorEnv wrapper (numpy in/out) for SB3 / CleanRL."""
    fn = _lazy(["macgym.gym_compat"], "to_gymnasium", "to_gymnasium")
    return fn(env, **kwargs)


__all__ = ["Robot", "Body", "JointHandle", "Task", "State", "RobotEnv", "TrainResult", "Policy",
           "make", "train", "evaluate", "load_policy", "view", "to_gymnasium", "robots",
           "default_obs"]
