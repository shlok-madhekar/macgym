# Robots are planar (2D): bodies move in the x/y plane only (x forward, y up), no 3D.
"""Hopper: one leg, four bodies. It has to learn to hop without falling."""
import macgym as mg

robot = mg.Robot("hopper")

hip, knee, ankle = (0, 1.05), (0, 0.60), (0, 0.10)
torso = robot.body("torso", start=(0, 1.45), end=hip, radius=0.05)
thigh = robot.body("thigh", start=hip, end=knee, radius=0.05)
shin = robot.body("shin", start=knee, end=ankle, radius=0.04)
foot = robot.body("foot", start=(-0.13, 0.06), end=(0.26, 0.06), radius=0.06)

robot.joint(torso, thigh, at=hip, limits=(-2.6, 0.0), motor=200)
robot.joint(thigh, shin, at=knee, limits=(-2.6, 0.0), motor=200)
robot.joint(shin, foot, at=ankle, limits=(-0.8, 0.8), motor=200)

task = mg.Task(
    reward=lambda s: s.root.vx + 1.0 - 0.001 * s.action_sq,
    # sat down, or tipped more than ~23 degrees
    done=lambda s: (s.root.y < 0.7) | (abs(s.root.angle) > 0.4),
    episode_steps=1000,
)
