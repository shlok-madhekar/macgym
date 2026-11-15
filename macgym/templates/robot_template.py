"""{{name}}: a MacGym robot.

    macgym train {{name}}.py --iters 200     # train 4096 copies on the GPU
    macgym view {{name}}.py                  # watch them learn

Coordinates are metres in a vertical plane (x forward, y up). Every body is a capsule
from `start` to `end`. Joints connect two bodies at a world-space point `at`.
"""
import macgym as mg

robot = mg.Robot("{{name}}")
torso = robot.body("torso", start=(-0.4, 0.8), end=(0.4, 0.8), radius=0.06)
back = robot.body("back_leg", start=(-0.4, 0.8), end=(-0.4, 0.3), radius=0.04)
front = robot.body("front_leg", start=(0.4, 0.8), end=(0.4, 0.3), radius=0.04)
back_foot = robot.body("back_foot", start=(-0.4, 0.3), end=(-0.4, 0.05), radius=0.04)
front_foot = robot.body("front_foot", start=(0.4, 0.3), end=(0.4, 0.05), radius=0.04)
robot.joint(torso, back, at=(-0.4, 0.8), limits=(-1.0, 1.0), motor=80)
robot.joint(torso, front, at=(0.4, 0.8), limits=(-1.0, 1.0), motor=80)
robot.joint(back, back_foot, at=(-0.4, 0.3), limits=(-1.2, 0.2), motor=60)
robot.joint(front, front_foot, at=(0.4, 0.3), limits=(-1.2, 0.2), motor=60)
robot.set_root(torso)

task = mg.Task(
    reward=lambda s: s.root.vx - 0.001 * s.action_sq,   # run forward, gently penalise effort
    done=lambda s: s.root.y < 0.3,                       # fell over
    episode_steps=1000,
)
