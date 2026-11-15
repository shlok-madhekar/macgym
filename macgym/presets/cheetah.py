"""Cheetah: a long body on a back leg and a front leg, three segments each.

It cannot fall over, so the only thing left to learn is speed.
"""
import macgym as mg

robot = mg.Robot("cheetah")

torso = robot.body("torso", start=(-0.5, 0.70), end=(0.5, 0.70), radius=0.046)

# Each leg is a chain hip -> knee -> ankle -> toe (points in metres).
LEGS = {
    "back": [(-0.5, 0.70), (-0.6, 0.45), (-0.45, 0.22), (-0.55, 0.05)],
    "front": [(0.5, 0.70), (0.42, 0.45), (0.55, 0.24), (0.62, 0.05)],
}
for leg, points in LEGS.items():
    parent = torso
    for seg, a, b in zip(("thigh", "shin", "foot"), points, points[1:]):
        link = robot.body(f"{leg}_{seg}", start=a, end=b, radius=0.046)
        robot.joint(parent, link, at=a, limits=(-1.0, 0.8), motor=90)
        parent = link

task = mg.Task(
    reward=lambda s: s.root.vx - 0.1 * s.action_sq,  # pure speed
    episode_steps=1000,
)
