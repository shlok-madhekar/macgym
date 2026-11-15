"""Tripod: a flat body on three two-segment legs.

Try it: add a fourth hip position to HIPS, then press Run.
"""
import macgym as mg

robot = mg.Robot("tripod")

# A 0.8 m plank at hip height. Units are metres; y = 0 is the ground.
torso = robot.body("torso", start=(-0.4, 0.55), end=(0.4, 0.55), radius=0.05)

# One leg per hip, x measured from the middle of the plank.
HIPS = [-0.35, 0.0, 0.35]
for i, x in enumerate(HIPS):
    hip, knee, foot = (x, 0.55), (x + 0.12, 0.30), (x + 0.05, 0.03)
    thigh = robot.body(f"thigh{i}", start=hip, end=knee, radius=0.035)
    shin = robot.body(f"shin{i}", start=knee, end=foot, radius=0.03)
    robot.joint(torso, thigh, at=hip, limits=(-0.9, 0.9), motor=30)
    robot.joint(thigh, shin, at=knee, limits=(-1.2, 0.3), motor=20)

task = mg.Task(
    # metres per second forward, a small bonus for staying up, minus effort
    reward=lambda s: s.root.vx + 0.5 - 0.002 * s.action_sq,
    # episode ends if the plank sags to the ground or tips past ~45 degrees
    done=lambda s: (s.root.y < 0.25) | (abs(s.root.angle) > 0.8),
    episode_steps=1000,
)
