"""Built-in robots, each written with the public API. Every function returns (robot, task):

    robot, task = mg.robots.hopper()
    env = mg.make(robot, task, num_envs=4096)
"""
import math

import mlx.core as mx

from macgym.api import Robot, Task

INK, BLUE, PINK, GREEN, AMBER, VIOLET = "#E8ECF4", "#4F8CFF", "#FF6B9A", "#3DDC97", "#FFB547", "#B48CFF"


def walker():
    """Planar biped: torso + 2 x (thigh, shin, foot), 6 motors. Same morphology and
    observation layout (24) as the legacy `walker` env, so runs/walker.npz loads."""
    robot = Robot("walker", friction=0.9)
    torso = robot.body("torso", start=(0, 1.05), end=(0, 1.45), radius=0.05, color=INK)
    for side, col in (("r", BLUE), ("l", PINK)):
        thigh = robot.body(side + "_thigh", start=(0, 1.05), end=(0, 0.6), radius=0.05, color=col)
        shin = robot.body(side + "_shin", start=(0, 0.6), end=(0, 0.1), radius=0.04, color=col)
        foot = robot.body(side + "_foot", start=(0, 0.1), end=(0.2, 0.1), radius=0.06, color=col)
        robot.joint(torso, thigh, at=(0, 1.05), limits=(-0.3, 2.0), motor=100, name=side + "_hip")
        robot.joint(thigh, shin, at=(0, 0.6), limits=(-2.6, 0.0), motor=100, name=side + "_knee")
        robot.joint(shin, foot, at=(0, 0.1), limits=(-0.8, 0.8), motor=100, name=side + "_ankle")

    def obs(s):  # legacy Walker2D layout
        r = s.root
        feet = []
        for f in ("r_foot", "l_foot"):
            b = s.body(f)
            feet += [b.x - r.x, b.y]
        return [r.y, mx.sin(r.angle), mx.cos(r.angle), r.vx, r.vy, r.omega * 0.1,
                s.q, s.qd * 0.1, s.contact("r_foot"), s.contact("l_foot"), *feet]

    task = Task(
        reward=lambda s: s.root.vx + 1.0 - 1e-3 * s.action_sq,
        done=lambda s: (s.root.y < 0.7 * 1.25) | (mx.abs(s.root.angle) > 1.0),
        episode_steps=1000, obs=obs, name="walk forward without falling")
    return robot, task


def cheetah():
    """HalfCheetah-like runner: torso + back/front (thigh, shin, foot), 6 motors.
    Same morphology and observation layout (17) as the legacy `cheetah` env."""
    H, r = 0.7, 0.046
    robot = Robot("cheetah", friction=0.4)
    hip_b, knee_b, ank_b, toe_b = (-0.5, H), (-0.34, H - 0.25), (-0.62, H - 0.39), (-0.56, H - 0.58)
    hip_f, knee_f, ank_f, toe_f = (0.5, H), (0.36, H - 0.24), (0.49, H - 0.42), (0.58, H - 0.56)
    torso = robot.body("torso", start=(-0.5, H), end=(0.5, H), radius=r, color=INK)
    bthigh = robot.body("bthigh", start=hip_b, end=knee_b, radius=r, color=AMBER)
    bshin = robot.body("bshin", start=knee_b, end=ank_b, radius=r, color=AMBER)
    bfoot = robot.body("bfoot", start=ank_b, end=toe_b, radius=r, color=AMBER)
    fthigh = robot.body("fthigh", start=hip_f, end=knee_f, radius=r, color=GREEN)
    fshin = robot.body("fshin", start=knee_f, end=ank_f, radius=r, color=GREEN)
    ffoot = robot.body("ffoot", start=ank_f, end=toe_f, radius=r, color=GREEN)
    robot.joint(torso, bthigh, at=hip_b, limits=(-1.05, 0.52), motor=120, damping=6.0, stiffness=240)
    robot.joint(bthigh, bshin, at=knee_b, limits=(-0.785, 0.785), motor=90, damping=4.5, stiffness=180)
    robot.joint(bshin, bfoot, at=ank_b, limits=(-0.785, 0.4), motor=60, damping=3.0, stiffness=120)
    robot.joint(torso, fthigh, at=hip_f, limits=(-0.7, 1.0), motor=120, damping=4.5, stiffness=180)
    robot.joint(fthigh, fshin, at=knee_f, limits=(-0.87, 1.2), motor=60, damping=3.0, stiffness=120)
    robot.joint(fshin, ffoot, at=ank_f, limits=(-0.5, 0.5), motor=30, damping=1.5, stiffness=60)

    def obs(s):  # legacy Cheetah2D layout
        r = s.root
        return [r.y, r.angle, s.q, r.vx, r.vy, r.omega * 0.1, s.qd * 0.1]

    task = Task(reward=lambda s: s.root.vx - 0.1 * s.action_sq, episode_steps=1000, obs=obs,
                name="run forward (never terminates)")
    return robot, task


def hopper():
    """One-legged hopper (MuJoCo Hopper proportions): torso, thigh, leg, foot; 3 motors."""
    robot = Robot("hopper", friction=0.9)
    torso = robot.body("torso", start=(0, 1.05), end=(0, 1.45), radius=0.05, color=INK)
    thigh = robot.body("thigh", start=(0, 1.05), end=(0, 0.6), radius=0.05, color=BLUE)
    leg = robot.body("leg", start=(0, 0.6), end=(0, 0.1), radius=0.04, color=BLUE)
    foot = robot.body("foot", start=(-0.13, 0.1), end=(0.26, 0.1), radius=0.06, color=PINK)
    robot.joint(torso, thigh, at=(0, 1.05), limits=(-0.3, 2.6), motor=200, name="hip")
    robot.joint(thigh, leg, at=(0, 0.6), limits=(-2.6, 0.0), motor=200, name="knee")
    robot.joint(leg, foot, at=(0, 0.1), limits=(-0.8, 0.8), motor=200, name="ankle")
    task = Task(
        reward=lambda s: s.root.vx + 1.0 - 1e-3 * s.action_sq,
        done=lambda s: (s.root.y < 0.8) | (mx.abs(s.root.angle) > 0.6),
        episode_steps=1000, name="hop forward without falling")
    return robot, task


def tripod():
    """Horizontal torso on three 2-segment legs (front, middle, back); 6 motors."""
    robot = Robot("tripod", friction=0.9)
    y = 0.85
    torso = robot.body("torso", start=(-0.35, y), end=(0.35, y), radius=0.06, color=INK)
    for name, x, col in (("back", -0.35, AMBER), ("mid", 0.0, VIOLET), ("front", 0.35, GREEN)):
        thigh = robot.body(name + "_thigh", start=(x, y), end=(x, y - 0.4), radius=0.04, color=col)
        shin = robot.body(name + "_shin", start=(x, y - 0.4), end=(x, 0.05), radius=0.035, color=col)
        robot.joint(torso, thigh, at=(x, y), limits=(-1.0, 1.0), motor=80, damping=1.0, name=name + "_hip")
        robot.joint(thigh, shin, at=(x, y - 0.4), limits=(-1.2, 1.2), motor=60, damping=1.0, name=name + "_knee")
    task = Task(
        reward=lambda s: s.root.vx + 0.5 - 0.01 * s.action_sq,
        done=lambda s: (s.root.y < 0.3) | (mx.abs(s.root.angle) > 1.2),
        episode_steps=1000, name="walk forward on three legs")
    return robot, task


def worm():
    """5-segment chain lying on the ground that has to crawl forward; 4 motors."""
    robot = Robot("worm", friction=0.9)
    seg, r, y = 0.3, 0.05, 0.05
    cols = [GREEN, GREEN, INK, GREEN, GREEN]
    bodies = [robot.body(f"seg{i}", start=(i * seg, y), end=((i + 1) * seg, y), radius=r, color=cols[i])
              for i in range(5)]
    for i in range(4):
        robot.joint(bodies[i], bodies[i + 1], at=((i + 1) * seg, y), limits=(-1.2, 1.2), motor=15,
                    damping=0.2, name=f"j{i}")
    robot.set_root(bodies[2])
    task = Task(reward=lambda s: s.root.vx - 0.01 * s.action_sq, episode_steps=500,
                name="crawl forward")
    return robot, task


BUILTINS = {"walker": walker, "cheetah": cheetah, "hopper": hopper, "tripod": tripod, "worm": worm}


def get(name):
    if name not in BUILTINS:
        raise KeyError(f"unknown built-in robot {name!r}; available: {list(BUILTINS)}")
    return BUILTINS[name]()
