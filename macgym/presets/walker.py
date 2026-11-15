"""Walker: a planar biped. Torso, two thighs, two shins, two feet.

The classic locomotion benchmark shape, in about 20 lines.
"""
import macgym as mg

robot = mg.Robot("walker")

# Units are metres; y = 0 is the ground. The torso is the root body.
torso = robot.body("torso", start=(0, 1.45), end=(0, 1.05), radius=0.05)

for side in ("left", "right"):
    hip, knee, ankle = (0, 1.05), (0, 0.60), (0, 0.10)
    thigh = robot.body(f"{side}_thigh", start=hip, end=knee, radius=0.05)
    shin = robot.body(f"{side}_shin", start=knee, end=ankle, radius=0.04)
    foot = robot.body(f"{side}_foot", start=(-0.05, 0.06), end=(0.15, 0.06),
                      radius=0.06)
    robot.joint(torso, thigh, at=hip, limits=(-1.5, 0.3), motor=100)
    robot.joint(thigh, shin, at=knee, limits=(-1.5, 0.0), motor=100)
    robot.joint(shin, foot, at=ankle, limits=(-0.8, 0.8), motor=60)

task = mg.Task(
    # forward speed, a bonus for staying upright, minus wasted effort
    reward=lambda s: s.root.vx + 1.0 - 0.001 * s.action_sq,
    done=lambda s: (s.root.y < 0.8) | (abs(s.root.angle) > 1.0),
    episode_steps=1000,
)
