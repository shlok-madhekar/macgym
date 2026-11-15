"""Env registry. Each import is guarded so a broken sibling env doesn't break the others."""
import traceback

ENVS = {}
IMPORT_ERRORS = {}

try:
    from macgym.envs.cartpole import CartPole
    ENVS["cartpole"] = CartPole
except Exception:
    IMPORT_ERRORS["cartpole"] = traceback.format_exc()

try:
    from macgym.envs.arm import ArmReach
    ENVS["arm"] = ArmReach
except Exception:
    IMPORT_ERRORS["arm"] = traceback.format_exc()

try:
    from macgym.envs.walker import Walker2D
    ENVS["walker"] = Walker2D
except Exception:
    IMPORT_ERRORS["walker"] = traceback.format_exc()

try:
    from macgym.envs.cheetah import Cheetah2D
    ENVS["cheetah"] = Cheetah2D
except Exception:
    IMPORT_ERRORS["cheetah"] = traceback.format_exc()


def make(name, num_envs, seed=0):
    if name not in ENVS:
        err = IMPORT_ERRORS.get(name)
        raise KeyError(f"unknown env {name!r}; available: {list(ENVS)}" + (f"\nimport error:\n{err}" if err else ""))
    return ENVS[name](num_envs, seed=seed)
