"""Batched planar 3-link torque-controlled robot arm reaching task in MLX. See CONTRACT.md.

Vertical plane, gravity acts in -y, base at (0, 0). Joint angles are relative;
absolute link angle a_i = q_1 + ... + q_i measured from +x.
Dynamics: per-joint simplified articulated model. Each joint's effective inertia is the
configuration-dependent inertia of all distal links about that joint; gravity torque is
the exact static torque of distal link masses; plus viscous damping, joint limits
(soft spring) on joints 2-3 and torque saturation.
"""
import math

import mlx.core as mx
import numpy as np

from macgym.envs.base import BatchedEnv

LINK_LEN = (0.4, 0.35, 0.25)
LINK_MASS = (1.0, 0.8, 0.5)
REACH = sum(LINK_LEN)
R_MIN, R_MAX = 0.25, 0.95
GRAVITY = 9.81
TORQUE_GAIN = (16.0, 8.0, 3.0)  # N*m at |a|=1
DAMPING = (0.8, 0.5, 0.15)
ARMATURE = 0.02
JOINT_LIMIT = 2.7
LIMIT_K = 60.0
LIMIT_D = 2.0
MAX_QD = 25.0
DT = 0.02
SUBSTEPS = 4
H = DT / SUBSTEPS
MAX_STEPS = 200
SUCCESS_DIST = 0.05
SUCCESS_BONUS = 1.0
ERR_TAIL = 20  # steps at episode end averaged for the reported miss distance


def _sample_targets(key, n):
    k1, k2 = mx.random.split(key)
    u = mx.random.uniform(R_MIN * R_MIN, R_MAX * R_MAX, (n,), key=k1)
    r = mx.sqrt(u)
    ang = mx.random.uniform(-math.pi, math.pi, (n,), key=k2)
    return mx.stack([r * mx.cos(ang), r * mx.sin(ang)], axis=1)


def _sample_joints(key, n):
    k1, k2, k3 = mx.random.split(key, 3)
    q1 = mx.random.uniform(-math.pi, math.pi, (n, 1), key=k1)
    q23 = mx.random.uniform(-1.2, 1.2, (n, 2), key=k2)
    q = mx.concatenate([q1, q23], axis=1)
    qd = mx.random.uniform(-0.1, 0.1, (n, 3), key=k3)
    return q, qd


def _fk_lists(q1, q2, q3):
    """Forward kinematics on per-joint (n,) arrays (keeps the compiled graph fusable).
    Returns joint xs/ys [p0..p3] and COM xs/ys [c1..c3]."""
    a1 = q1
    a2 = a1 + q2
    a3 = a2 + q3
    px, py = [0.0], [0.0]
    cx, cy = [], []
    for a, L in zip((a1, a2, a3), LINK_LEN):
        dx, dy = mx.cos(a), mx.sin(a)
        cx.append(px[-1] + 0.5 * L * dx)
        cy.append(py[-1] + 0.5 * L * dy)
        px.append(px[-1] + L * dx)
        py.append(py[-1] + L * dy)
    return px, py, cx, cy


def _kinematics(q):
    """Joint positions p0..p3 as (n,2) arrays (used for obs / rendering)."""
    px, py, _, _ = _fk_lists(q[:, 0], q[:, 1], q[:, 2])
    z = mx.zeros_like(q[:, 0])
    return [mx.stack([z + x, z + y], axis=1) for x, y in zip(px, py)]


def _dynamics_substep(q, qd, tau):
    """q, qd, tau: lists of 3 (n,) arrays."""
    px, py, cx, cy = _fk_lists(*q)
    new_q, new_qd = [], []
    for i in range(3):
        inertia = ARMATURE
        g_tau = 0.0
        for j in range(i, 3):
            rx = cx[j] - px[i]
            ry = cy[j] - py[i]
            m = LINK_MASS[j]
            inertia = inertia + m * (rx * rx + ry * ry) + m * LINK_LEN[j] ** 2 / 12.0
            g_tau = g_tau - m * GRAVITY * rx
        t_i = tau[i] + g_tau - DAMPING[i] * qd[i]
        if i > 0:  # soft joint limits on elbow / wrist
            pen = mx.maximum(q[i] - JOINT_LIMIT, 0.0) + mx.minimum(q[i] + JOINT_LIMIT, 0.0)
            t_i = t_i - LIMIT_K * pen - LIMIT_D * qd[i] * (pen != 0)
        v = mx.clip(qd[i] + H * t_i / inertia, -MAX_QD, MAX_QD)
        new_qd.append(v)
        new_q.append(q[i] + H * v)
    return new_q, new_qd


def _obs(q, qd, target):
    ee = _kinematics(q)[3]
    return mx.concatenate([mx.cos(q), mx.sin(q), 0.1 * qd, target, ee, target - ee], axis=1)


def _physics(q, qd, target, t, ret, err_acc, action, key, fixed_flag, fixed_xy):
    n = q.shape[0]
    a = mx.clip(action, -1.0, 1.0)
    tau = [a[:, i] * TORQUE_GAIN[i] for i in range(3)]
    ql = [q[:, i] for i in range(3)]
    qdl = [qd[:, i] for i in range(3)]
    for _ in range(SUBSTEPS):
        ql, qdl = _dynamics_substep(ql, qdl, tau)
    ql[0] = mx.remainder(ql[0] + math.pi, 2 * math.pi) - math.pi
    q = mx.stack(ql, axis=1)
    qd = mx.stack(qdl, axis=1)

    px, py, _, _ = _fk_lists(*ql)
    dxe = target[:, 0] - px[3]
    dye = target[:, 1] - py[3]
    dist = mx.sqrt(dxe * dxe + dye * dye)
    reward = -dist - 0.01 * mx.sum(a * a, axis=1) + SUCCESS_BONUS * (dist < SUCCESS_DIST)

    t = t + 1.0
    done = (t >= MAX_STEPS).astype(mx.float32)
    ret = ret + reward
    ep_return = ret * done
    ep_len = t * done
    err_acc = err_acc + dist * (t > MAX_STEPS - ERR_TAIL)
    ep_err = err_acc * (1.0 / ERR_TAIL) * done

    k1, k2 = mx.random.split(key)
    fq, fqd = _sample_joints(k1, n)
    ftarget = _sample_targets(k2, n)
    ftarget = mx.where(fixed_flag > 0.5, mx.broadcast_to(fixed_xy, ftarget.shape), ftarget)
    d1 = done > 0.5
    d2 = d1[:, None]
    q = mx.where(d2, fq, q)
    qd = mx.where(d2, fqd, qd)
    target = mx.where(d2, ftarget, target)
    # a fixed (user-set) target applies to every env immediately
    target = mx.where(fixed_flag > 0.5, mx.broadcast_to(fixed_xy, target.shape), target)
    t = mx.where(d1, 0.0, t)
    ret = mx.where(d1, 0.0, ret)
    err_acc = mx.where(d1, 0.0, err_acc)
    obs = _obs(q, qd, target)
    return q, qd, target, t, ret, err_acc, obs, reward, done, ep_return, ep_len, ep_err


class ArmReach(BatchedEnv):
    name = "arm"
    description = "Planar 3-link torque-controlled robot arm reaching random targets under gravity. Drag to set target."
    obs_dim = 15
    act_dim = 3
    max_episode_steps = MAX_STEPS
    dt = DT

    def __init__(self, num_envs: int, seed: int = 0):
        super().__init__(num_envs, seed)
        self._step_fn = mx.compile(_physics)
        self.reset()

    def reset(self) -> mx.array:
        n = self.num_envs
        self.fixed_flag = mx.array(0.0)
        self.fixed_xy = mx.zeros((2,), dtype=mx.float32)
        self.q, self.qd = _sample_joints(self._split(), n)
        self.target = _sample_targets(self._split(), n)
        self.t = mx.zeros((n,), dtype=mx.float32)
        self.ret = mx.zeros((n,), dtype=mx.float32)
        self.err_acc = mx.zeros((n,), dtype=mx.float32)
        self.obs = _obs(self.q, self.qd, self.target)
        mx.eval(self.q, self.qd, self.target, self.t, self.ret, self.err_acc, self.obs)
        return self.obs

    def step(self, action: mx.array):
        action = action.astype(mx.float32).reshape(self.num_envs, self.act_dim)
        (self.q, self.qd, self.target, self.t, self.ret, self.err_acc, self.obs,
         reward, done, ep_ret, ep_len, ep_err) = self._step_fn(
            self.q, self.qd, self.target, self.t, self.ret, self.err_acc, action, self._split(),
            self.fixed_flag, self.fixed_xy,
        )
        # episode_err: mean |ee - target| (m) over the last ERR_TAIL steps, valid where done
        info = {"episode_return": ep_ret, "episode_len": ep_len, "episode_err": ep_err}
        return self.obs, reward, done, info

    def set_target(self, x=None, y=None) -> None:
        """Force every env's target to (x, y) (clamped to reachable annulus).
        set_target(None) returns to random per-episode targets."""
        if x is None or y is None:
            self.fixed_flag = mx.array(0.0)
            return
        x, y = float(x), float(y)
        r = math.hypot(x, y)
        if r < 1e-6:
            x, y, r = R_MIN, 0.0, R_MIN
        rc = min(max(r, R_MIN), R_MAX)
        x, y = x * rc / r, y * rc / r
        self.fixed_flag = mx.array(1.0)
        self.fixed_xy = mx.array([x, y], dtype=mx.float32)
        self.target = mx.broadcast_to(self.fixed_xy, (self.num_envs, 2))
        self.obs = _obs(self.q, self.qd, self.target)
        mx.eval(self.target, self.obs)

    def render_state(self, R: int) -> dict:
        R = min(R, self.num_envs)
        pts = _kinematics(self.q[:R])
        p = np.stack([np.array(pt, dtype=np.float32) for pt in pts], axis=1)  # (R,4,2)
        caps = np.zeros((R, 3, 5), dtype=np.float32)
        caps[:, :, 0:2] = p[:, :3]
        caps[:, :, 2:4] = p[:, 1:]
        caps[:, :, 4] = np.array([0.045, 0.038, 0.03], dtype=np.float32)
        targets = np.array(self.target[:R], dtype=np.float32)[:, None, :]
        err = np.linalg.norm(p[:, 3] - targets[:, 0], axis=1).astype(np.float32)
        return {
            "dist": err,  # current end-effector miss distance (m), lower is better
            "higher_is_better": False,
            "capsules": caps,
            "targets": targets,
            "colors": ["#4F8CFF", "#7C5CFF", "#22D3A6"],
            "ground": None,
            "world_bounds": [-1.1, 1.1, -1.1, 1.1],
        }

    def live_err_mean(self) -> float:
        ee = _kinematics(self.q)[3]
        return float(mx.mean(mx.sqrt(mx.sum(mx.square(ee - self.target), axis=1))))
