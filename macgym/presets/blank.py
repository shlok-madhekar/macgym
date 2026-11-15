"""A blank robot to start from.

    robot.body(name, start=(x, y), end=(x, y), radius=r)   # a capsule
    robot.joint(parent, child, at=(x, y), limits=(lo, hi), motor=gear)

Units are metres and radians; y = 0 is the ground.
"""
import macgym as mg

robot = mg.Robot("my_robot")

torso = robot.body("torso", start=(-0.3, 0.4), end=(0.3, 0.4), radius=0.05)
leg = robot.body("leg", start=(0.3, 0.4), end=(0.3, 0.05), radius=0.04)
robot.joint(torso, leg, at=(0.3, 0.4), limits=(-1.0, 1.0), motor=60)

task = mg.Task(
    reward=lambda s: s.root.vx,  # metres per second to the right
    episode_steps=1000,
)
