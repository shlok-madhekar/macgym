"""A brand-new three-legged creature and its reward, in about 20 lines.

    python examples/02_custom_robot.py
"""
import os

import macgym as mg

robot = mg.Robot("tripod3")
torso = robot.body("torso", start=(-0.5, 0.7), end=(0.5, 0.7), radius=0.07)
for name, x in (("back", -0.5), ("mid", 0.0), ("front", 0.5)):
    thigh = robot.body(f"{name}_thigh", start=(x, 0.7), end=(x, 0.35), radius=0.04)
    shin = robot.body(f"{name}_shin", start=(x, 0.35), end=(x + 0.1, 0.05), radius=0.035)
    robot.joint(torso, thigh, at=(x, 0.7), limits=(-1.0, 1.0), motor=70, name=f"{name}_hip")
    robot.joint(thigh, shin, at=(x, 0.35), limits=(-1.5, 0.3), motor=50, name=f"{name}_knee")
robot.set_root(torso)

task = mg.Task(
    reward=lambda s: s.root.vx + 0.5 * (s.root.y > 0.45) - 0.002 * s.action_sq,  # go forward, stay up
    done=lambda s: s.root.y < 0.25,
    episode_steps=1000,
)

if __name__ == "__main__":
    env = mg.make(robot, task, num_envs=int(os.environ.get("MACGYM_ENVS", 4096)))
    mg.train(env, iters=int(os.environ.get("MACGYM_ITERS", 150)), save="runs/tripod3.npz")
