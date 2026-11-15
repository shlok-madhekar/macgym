"""Train the built-in walker for 100 iterations, then report how far it gets.

    python examples/01_quickstart.py
"""
import os

import macgym as mg

ITERS = int(os.environ.get("MACGYM_ITERS", 100))
ENVS = int(os.environ.get("MACGYM_ENVS", 4096))

env = mg.make(*mg.robots.walker(), num_envs=ENVS)
result = mg.train(env, iters=ITERS, save="runs/quickstart_walker.npz")
print(f"final distance: {result.final.get('dist') or 0:.2f} m   policy: {result.policy}")
