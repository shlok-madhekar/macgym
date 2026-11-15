"""Worm: five segments lying on the ground, hinged end to end.

No legs at all. It has to discover an inchworm wave on its own.
"""
import macgym as mg

robot = mg.Robot("worm")

SEGMENTS = 5
LENGTH = 0.3   # metres per segment
R = 0.05       # capsule radius, so the worm rests on the ground at y = R

segments = []
for i in range(SEGMENTS):
    a, b = (i * LENGTH, R), ((i + 1) * LENGTH, R)
    seg = robot.body(f"seg{i}", start=a, end=b, radius=R)
    if segments:
        robot.joint(segments[-1], seg, at=a, limits=(-1.2, 1.2), motor=25)
    segments.append(seg)

robot.set_root(segments[SEGMENTS // 2])  # measure from the middle segment

task = mg.Task(
    reward=lambda s: s.root.vx - 0.01 * s.action_sq,
    episode_steps=1000,
)
