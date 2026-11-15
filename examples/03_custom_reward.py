"""Same cheetah body, new goal: hop BACKWARDS. Only the Task changes.

    python examples/03_custom_reward.py
"""
import os

import macgym as mg

robot, _ = mg.robots.cheetah()          # reuse the built-in body, throw away its task


def hop_backwards(s):
    airborne = (1 - s.contact("bfoot")) * (1 - s.contact("ffoot"))  # both feet off the ground
    return -s.root.vx + 0.5 * airborne - 0.1 * s.action_sq


task = mg.Task(reward=hop_backwards, episode_steps=1000)

if __name__ == "__main__":
    env = mg.make(robot, task, num_envs=int(os.environ.get("MACGYM_ENVS", 4096)))
    result = mg.train(env, iters=int(os.environ.get("MACGYM_ITERS", 150)), save="runs/cheetah_backflip.npz")
    print(f"distance (negative = backwards): {result.final.get('dist') or 0:.2f} m")
