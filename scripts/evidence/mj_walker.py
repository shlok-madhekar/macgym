"""MuJoCo MJCF that mirrors macgym.robots.walker() (planar, x-z plane).

Mapping: our 2D (x, y) -> MuJoCo (x, z). Hinge axes are (0,-1,0) so positive = CCW in the
x-z view, matching our CCW-positive angles; joint q = th_child - th_parent - rest, same as ours.
Masses use the same capsule-volume formula (density 1000) and the in-plane inertia is set
explicitly to ours: I = m*(Ltot^2/12 + r^2/4). Only feet/body-vs-ground contacts (no
self-collision), friction 0.9, joint damping 0.1, motor gear 100, timestep 0.0025 x 8.
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BODIES = [  # name, start, end, radius
    ("torso", (0, 1.05), (0, 1.45), 0.05),
    ("r_thigh", (0, 1.05), (0, 0.6), 0.05), ("r_shin", (0, 0.6), (0, 0.1), 0.04), ("r_foot", (0, 0.1), (0.2, 0.1), 0.06),
    ("l_thigh", (0, 1.05), (0, 0.6), 0.05), ("l_shin", (0, 0.6), (0, 0.1), 0.04), ("l_foot", (0, 0.1), (0.2, 0.1), 0.06),
]
JOINTS = [("hip", (0, 1.05), (-0.3, 2.0)), ("knee", (0, 0.6), (-2.6, 0.0)), ("ankle", (0, 0.1), (-0.8, 0.8))]
DENSITY, GEAR, DAMPING, MU = 1000.0, 100.0, 0.1, 0.9
DT, FRAME_SKIP = 0.02, 8
TORSO_Y0 = 1.25


def mass_inertia(p1, p2, r):
    L = math.dist(p1, p2)
    m = DENSITY * (math.pi * r * r * L + 4.0 / 3.0 * math.pi * r ** 3)
    Lt = L + 2 * r
    return m, m * (Lt * Lt / 12.0 + r * r / 4.0)


def _body_xml(name, indent):
    b = {n: (s, e, r) for n, s, e, r in BODIES}[name]
    s, e, r = b
    m, I = mass_inertia(s, e, r)
    mid = ((s[0] + e[0]) / 2, (s[1] + e[1]) / 2)
    pad = " " * indent
    return (f'{pad}<inertial pos="{mid[0]} 0 {mid[1]}" mass="{m:.9g}" diaginertia="{I:.9g} {I:.9g} {I:.9g}"/>\n'
            f'{pad}<geom name="{name}_geom" type="capsule" fromto="{s[0]} 0 {s[1]} {e[0]} 0 {e[1]}" size="{r}"/>\n')


def mjcf(timestep=DT / FRAME_SKIP):
    x = f'''<mujoco model="macgym_walker_mirror">
  <compiler angle="radian"/>
  <option timestep="{timestep}" gravity="0 0 -9.81"/>
  <default>
    <geom contype="1" conaffinity="0" friction="{MU} 0.005 0.0001" condim="3" rgba="0.8 0.8 0.8 1"/>
    <joint damping="{DAMPING}" limited="true" armature="0"/>
    <motor gear="{GEAR}" ctrlrange="-1 1" ctrllimited="true"/>
  </default>
  <visual><global offwidth="960" offheight="540"/></visual>
  <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.98 0.98 0.97" rgb2="0.9 0.9 0.88" width="300" height="300"/>
  <material name="grid" texture="grid" texrepeat="40 40"/></asset>
  <worldbody>
    <light pos="0 -3 4" dir="0 1 -1"/>
    <geom name="floor" type="plane" size="200 2 0.1" contype="0" conaffinity="1" material="grid" friction="{MU} 0.005 0.0001"/>
    <body name="torso" pos="0 0 0">
      <joint name="rootx" type="slide" axis="1 0 0" limited="false" damping="0"/>
      <joint name="rootz" type="slide" axis="0 0 1" limited="false" damping="0"/>
      <joint name="rooty" type="hinge" axis="0 -1 0" pos="0 0 {TORSO_Y0}" limited="false" damping="0"/>
{_body_xml("torso", 6)}'''
    for side in ("r", "l"):
        parts = [f"{side}_thigh", f"{side}_shin", f"{side}_foot"]
        ind = 6
        for (jn, at, lim), bn in zip(JOINTS, parts):
            x += " " * ind + f'<body name="{bn}" pos="0 0 0">\n'
            ind += 2
            x += " " * ind + f'<joint name="{side}_{jn}" type="hinge" axis="0 -1 0" pos="{at[0]} 0 {at[1]}" range="{lim[0]} {lim[1]}"/>\n'
            x += _body_xml(bn, ind)
        for _ in parts:
            ind -= 2
            x += " " * ind + "</body>\n"
    x += "    </body>\n  </worldbody>\n  <actuator>\n"
    for side in ("r", "l"):
        for jn, _, _ in JOINTS:
            x += f'    <motor name="{side}_{jn}_m" joint="{side}_{jn}"/>\n'
    x += "  </actuator>\n</mujoco>\n"
    return x


GEOMS = [n + "_geom" for n, *_ in BODIES]


class MjWalker:
    """Single MuJoCo walker with the same control semantics as ours: per control step,
    ctrl = a * clip(1 - sign(a)*qd/10, 0, 1) (our DC-motor torque-speed curve, qd at step start),
    held for 8 substeps of 0.0025 s."""

    def __init__(self, speed_curve=True):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(mjcf())
        self.data = mujoco.MjData(self.model)
        self.speed_curve = speed_curve
        self.gid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in GEOMS]
        self.half = self.model.geom_size[self.gid, 1].copy()
        self.rad = self.model.geom_size[self.gid, 0].copy()

    def reset(self, qvel=None):
        self.mj.mj_resetData(self.model, self.data)
        if qvel is not None:
            self.data.qvel[:] = qvel
        self.mj.mj_forward(self.model, self.data)

    def step(self, a):
        import numpy as np
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        if self.speed_curve:
            qd = self.data.qvel[3:9]
            a = a * np.clip(1.0 - np.where(a >= 0, qd, -qd) / 10.0, 0.0, 1.0)
        self.data.ctrl[:] = a
        for _ in range(FRAME_SKIP):
            self.mj.mj_step(self.model, self.data)

    # ---- state readouts in our conventions
    def torso(self):
        d = self.data
        return dict(x=d.qpos[0], y=d.qpos[1] + TORSO_Y0, angle=d.qpos[2], vx=d.qvel[0], vy=d.qvel[1], omega=d.qvel[2])

    def lowest(self):
        """(7,) lowest point (z) of every capsule."""
        import numpy as np
        d = self.data
        axz = d.geom_xmat[self.gid].reshape(-1, 3, 3)[:, 2, 2]  # z-component of capsule axis
        return d.geom_xpos[self.gid, 2] - self.half * np.abs(axz) - self.rad

    def obs(self):
        import numpy as np
        d = self.data
        t = self.torso()
        low = self.lowest()
        cR = float(low[3] < 0.02); cL = float(low[6] < 0.02)
        fr = d.geom_xpos[self.gid[3]]; fl = d.geom_xpos[self.gid[6]]
        o = [t["y"], math.sin(t["angle"]), math.cos(t["angle"]), t["vx"], t["vy"], t["omega"] * 0.1,
             *d.qpos[3:9], *(d.qvel[3:9] * 0.1), cR, cL, fr[0] - d.qpos[0], fr[2], fl[0] - d.qpos[0], fl[2]]
        return np.clip(np.asarray(o, dtype=np.float32), -50, 50)


if __name__ == "__main__":
    import mujoco
    m = mujoco.MjModel.from_xml_string(mjcf())
    print(mjcf())
    print("nq", m.nq, "nv", m.nv, "nu", m.nu, "masses", m.body_mass)
    from macgym import robots
    r, _ = robots.walker()
    mm = r.build_model()
    print("ours masses", mm.masses, "joint dampings", [j.damping for j in mm.joints], "mu", mm.mu)
