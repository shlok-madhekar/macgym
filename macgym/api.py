"""MacGym public API: Robot builder, Task, State view, make(), train().

Everything here sits on top of the existing planar engine (macgym.envs.physics2d):
a Robot compiles to Capsule/Joint lists -> Model (auto-generated Metal kernel), and
make() returns a RobotEnv, a LocoEnv subclass whose reward/done/obs come from the
Task callables. Those callables run INSIDE the mx.compile'd env step, so they must be
pure mx ops on the batched State view (no Python branching on array values).
"""
import math
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from macgym.envs.physics2d import Capsule, Joint, LocoEnv, Model, state_bad

PALETTE = ["#E8ECF4", "#4F8CFF", "#FF6B9A", "#3DDC97", "#FFB547", "#B48CFF", "#5FD4E8", "#FF8A5B"]
CONTACT_EPS = 0.02  # body counts as touching ground when its lowest point is within 2 cm


# --------------------------------------------------------------------------- builder
class Body:
    """Handle returned by Robot.body(). `index` is the capsule index in the model."""

    def __init__(self, robot, index, name):
        self.robot, self.index, self.name = robot, index, name

    def __repr__(self):
        return f"Body({self.name!r}, index={self.index})"


class JointHandle:
    """Handle returned by Robot.joint(). `index` is the actuator/action index."""

    def __init__(self, robot, index, name, parent, child):
        self.robot, self.index, self.name = robot, index, name
        self.parent, self.child = parent, child

    def __repr__(self):
        return f"Joint({self.name!r}, index={self.index}, {self.parent.name}->{self.child.name})"


class Robot:
    """Planar robot builder: capsule bodies connected by motorized revolute joints.

    robot = mg.Robot("tripod")
    torso = robot.body("torso", start=(0, 1.0), end=(0.6, 1.0), radius=0.06)
    leg = robot.body("leg", start=(0, 1.0), end=(0, 0.5), radius=0.04)
    robot.joint(torso, leg, at=(0, 1.0), limits=(-1, 1), motor=80)
    """

    def __init__(self, name="robot", friction=0.9, dt=0.02, substeps=8, iters=4):
        self.name = str(name)
        self.friction = float(friction)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.iters = int(iters)
        self.bodies = []   # dicts: name, start, end, radius, density, color
        self.joints = []   # dicts: name, parent, child, at, limits, motor, damping, stiffness
        self._body_handles = []
        self._joint_handles = []
        self.root = None

    # -- construction
    def body(self, name, start, end, radius=0.05, density=1000.0, color=None):
        name = str(name)
        if any(b.name == name for b in self.bodies):
            raise ValueError(f"duplicate body name {name!r}")
        start, end = _vec2(start, "start"), _vec2(end, "end")
        radius, density = float(radius), float(density)
        if not radius > 0:
            raise ValueError(f"body {name!r}: radius must be > 0")
        if not density > 0:
            raise ValueError(f"body {name!r}: density must be > 0")
        idx = len(self.bodies)
        self.bodies.append(SimpleNamespace(name=name, start=start, end=end, radius=radius, density=density,
                                color=color or PALETTE[idx % len(PALETTE)]))
        h = Body(self, idx, name)
        self._body_handles.append(h)
        if self.root is None:
            self.root = h
        return h

    def joint(self, parent, child, at, limits=(-1.0, 1.0), motor=50.0, damping=0.1,
              stiffness=0.0, name=None):
        parent, child = self._resolve(parent), self._resolve(child)
        if parent.index == child.index:
            raise ValueError("joint parent and child must be different bodies")
        lo, hi = (float(v) for v in limits)
        if not lo < hi:
            raise ValueError(f"joint limits must satisfy lo < hi, got {limits}")
        name = str(name) if name is not None else child.name
        if any(j.name == name for j in self.joints):
            name = f"{parent.name}_{child.name}"
        if any(j.name == name for j in self.joints):
            raise ValueError(f"duplicate joint name {name!r}")
        idx = len(self.joints)
        self.joints.append(SimpleNamespace(name=name, parent=parent.index, child=child.index,
                                at=_vec2(at, "at"), limits=(lo, hi), motor=float(motor),
                                damping=float(damping), stiffness=float(stiffness)))
        h = JointHandle(self, idx, name, parent, child)
        self._joint_handles.append(h)
        return h

    def set_root(self, body):
        self.root = self._resolve(body)
        return self.root

    # -- queries
    def body_index(self, name):
        return self._resolve(name).index

    def joint_index(self, key):
        if isinstance(key, JointHandle):
            return key.index
        if isinstance(key, (int, np.integer)):
            if not -len(self.joints) <= key < len(self.joints):
                raise IndexError(f"joint index {key} out of range ({len(self.joints)} joints)")
            return int(key) % len(self.joints)
        for i, j in enumerate(self.joints):
            if j.name == key:
                return i
        raise KeyError(f"no joint named {key!r}; joints: {[j['name'] for j in self.joints]}")

    @property
    def body_names(self):
        return [b.name for b in self.bodies]

    @property
    def joint_names(self):
        return [j.name for j in self.joints]

    def validate(self):
        if not self.bodies:
            raise ValueError(f"robot {self.name!r} has no bodies")
        if not self.joints:
            raise ValueError(f"robot {self.name!r} has no joints (nothing to actuate)")

    def build_model(self):
        """Compile to the physics engine's Model (Capsule/Joint lists)."""
        self.validate()
        caps = [Capsule(b.start, b.end, b.radius, density=b.density,
                        color=b.color, name=b.name) for b in self.bodies]
        joints = [Joint(j.parent, j.child, j.at, j.limits[0], j.limits[1],
                        gear=j.motor, damping=j.damping, stiffness=j.stiffness)
                  for j in self.joints]
        return Model(caps, joints, dt=self.dt, substeps=self.substeps, iters=self.iters,
                     mu=self.friction)

    def rest_capsules(self):
        """(B, 5) float32 [x1, y1, x2, y2, r] of the rest pose (for previews)."""
        return np.array([[*b.start, *b.end, b.radius] for b in self.bodies], np.float32).reshape(-1, 5)

    @property
    def colors(self):
        return [b.color for b in self.bodies]

    def _resolve(self, b):
        if isinstance(b, Body):
            if b.robot is not self:
                raise ValueError(f"body {b.name!r} belongs to a different robot")
            return b
        if isinstance(b, str):
            for h in self._body_handles:
                if h.name == b:
                    return h
            raise KeyError(f"no body named {b!r}; bodies: {self.body_names}")
        if isinstance(b, (int, np.integer)):
            return self._body_handles[int(b)]
        raise TypeError(f"expected a Body handle or body name, got {type(b).__name__}")

    def __repr__(self):
        return f"Robot({self.name!r}, bodies={len(self.bodies)}, joints={len(self.joints)}, root={self.root.name if self.root else None!r})"


def _vec2(v, what):
    try:
        x, y = (float(c) for c in v)
    except Exception:
        raise ValueError(f"{what} must be an (x, y) pair, got {v!r}") from None
    return (x, y)


# --------------------------------------------------------------------------- task
@dataclass
class Task:
    """What the robot should do. Callables get a batched State view `s` and must use
    pure mx ops (they are traced by mx.compile inside the env step).

    reward(s) -> (N,) float     done(s) -> (N,) bool (optional)     obs(s) -> (N, D) (optional)
    """
    reward: object
    done: object = None
    episode_steps: int = 1000
    obs: object = None
    name: str = None


# --------------------------------------------------------------------------- state view
class BodyState:
    """Per-body batched view. x/y = capsule center (m), vx/vy (m/s), angle = rotation
    relative to the rest pose (rad, CCW+), omega (rad/s), height = y."""

    def __init__(self, x, y, angle, vx, vy, omega):
        self.x, self.y, self.angle = x, y, angle
        self.vx, self.vy, self.omega = vx, vy, omega

    @property
    def height(self):
        return self.y


class JointState:
    def __init__(self, q, qd):
        self.q, self.qd = q, qd


class State:
    """Batched view handed to reward/done/obs. All fields are mx arrays of shape (N,)
    unless noted: s.root, s.body(name), s.joint(name|index), s.q / s.qd (N, J),
    s.action (N, A), s.action_sq, s.contact(name), s.t (steps into the episode), s.dt."""

    def __init__(self, env, state, action=None, t=None):
        self._env = env
        self._robot = env.robot
        self._model = env.model
        B = self._model.B
        self._cols = [state[:, i] for i in range(6 * B)]
        self._raw = state
        n = state.shape[0]
        self.action = action if action is not None else mx.zeros((n, self._model.J))
        self.t = t if t is not None else mx.zeros((n,))
        self.dt = env.dt
        self.num_envs = n
        self._cache = {}

    # bodies
    def body(self, name):
        k = self._robot.body_index(name) if not isinstance(name, (int, np.integer)) else int(name)
        if k not in self._cache:
            B, c, th0 = self._model.B, self._cols, self._env.rest_angle[k]
            self._cache[k] = BodyState(c[k], c[B + k], c[2 * B + k] - th0,
                                       c[3 * B + k], c[4 * B + k], c[5 * B + k])
        return self._cache[k]

    @property
    def root(self):
        return self.body(self._env.torso)

    # joints
    def _joint_arrays(self):
        if "qqd" not in self._cache:
            self._cache["qqd"] = self._model.joint_angles(self._raw)
        return self._cache["qqd"]

    @property
    def q(self):
        return self._joint_arrays()[0]

    @property
    def qd(self):
        return self._joint_arrays()[1]

    def joint(self, key):
        j = self._robot.joint_index(key)
        q, qd = self._joint_arrays()
        return JointState(q[:, j], qd[:, j])

    # actions
    @property
    def action_sq(self):
        return mx.sum(self.action * self.action, axis=1)

    # contacts
    def contact(self, name):
        k = self._robot.body_index(name) if not isinstance(name, (int, np.integer)) else int(name)
        key = ("c", k)
        if key not in self._cache:
            B, c, m = self._model.B, self._cols, self._model
            hy = m.half[k] * mx.abs(mx.sin(c[2 * B + k]))
            low = c[B + k] - hy - m.rad[k]
            self._cache[key] = (low < CONTACT_EPS).astype(mx.float32)
        return self._cache[key]

    @property
    def contacts(self):
        """(N, B) ground contact flags for every body."""
        return mx.stack([self.contact(k) for k in range(self._model.B)], axis=1)


def default_obs(s):
    """Auto observation: root height, sin/cos of root tilt, root vx/vy/omega, joint q/qd,
    ground contact of every body."""
    r = s.root
    cols = mx.stack([r.y, mx.sin(r.angle), mx.cos(r.angle), r.vx, r.vy, r.omega * 0.1], axis=1)
    return mx.concatenate([cols, s.q, s.qd * 0.1, s.contacts], axis=1)


def _as_batch(v, n, dtype, what):
    """Coerce a user callable's result to an (N,) array of dtype."""
    if v is None:
        return mx.zeros((n,), dtype=dtype)
    if not isinstance(v, mx.array):
        v = mx.array(v)
    if v.ndim == 2 and v.shape[1] == 1:
        v = v[:, 0]
    if v.ndim == 0:
        v = mx.broadcast_to(v, (n,))
    if v.shape != (n,):
        raise ValueError(f"task.{what} must return shape ({n},), got {tuple(v.shape)}")
    return v.astype(dtype)


def _as_obs(v, n):
    if isinstance(v, (list, tuple)):
        v = mx.concatenate([x[:, None] if x.ndim == 1 else x for x in v], axis=1)
    if not isinstance(v, mx.array) or v.ndim != 2 or v.shape[0] != n:
        raise ValueError(f"task.obs must return an (N, D) array (or list of (N,)/(N,k) arrays), got "
                         f"{getattr(v, 'shape', type(v).__name__)}")
    return v.astype(mx.float32)


# --------------------------------------------------------------------------- env
class RobotEnv(LocoEnv):
    """LocoEnv driven by a Robot + Task. Same interface as the built-in envs:
    reset() -> obs, step(a) -> (obs, reward, done, info) with info episode_return /
    episode_len / episode_dist, render_state(R), live_dist_mean(), metrics via PPO."""

    def __init__(self, robot, task, num_envs=4096, seed=0, backend=None):
        if not isinstance(robot, Robot):
            raise TypeError("mg.make(robot, task): robot must be an mg.Robot")
        if not isinstance(task, Task):
            raise TypeError("mg.make(robot, task): task must be an mg.Task")
        if not callable(task.reward):
            raise TypeError("task.reward must be callable: reward=lambda s: ...")
        robot.validate()
        self.robot, self.task = robot, task
        self.name = robot.name
        self.description = task.name or f"custom robot: {len(robot.bodies)} bodies / {len(robot.joints)} joints"
        self.torso = robot.root.index
        self.act_dim = len(robot.joints)
        self.max_episode_steps = int(task.episode_steps)
        self.dt = robot.dt
        self.rest_angle = [math.atan2(b.end[1] - b.start[1], b.end[0] - b.start[0])
                           for b in robot.bodies]
        top = max(max(b.start[1], b.end[1]) + b.radius for b in robot.bodies)
        span = max(max(b.start[0], b.end[0]) for b in robot.bodies) - \
            min(min(b.start[0], b.end[0]) for b in robot.bodies)
        self.view_y = (-0.3, max(1.6, top + 0.6))
        self.view_half_width = max(2.5, span * 1.5)
        self._obs_fn = task.obs if task.obs is not None else default_obs
        self.obs_dim = 1  # placeholder; fixed after the first observe()
        super().__init__(num_envs, seed=seed, backend=backend)
        self.obs_dim = int(self.obs.shape[1])
        self.metrics = {}

    def build_model(self):
        return self.robot.build_model()

    def state_view(self, state, action=None, t=None):
        return State(self, state, action, t)

    def observe(self, state):
        s = State(self, state)
        return mx.clip(_as_obs(self._obs_fn(s), state.shape[0]), -50.0, 50.0)

    def reward_done(self, prev, state, action, t=None):
        n = state.shape[0]
        s = State(self, state, action, t)
        reward = _as_batch(self.task.reward(s), n, mx.float32, "reward")
        done = _as_batch(self.task.done(s), n, mx.bool_, "done") if self.task.done else \
            mx.zeros((n,), dtype=mx.bool_)
        return reward, done

    def _env_step(self, state, t, ret, action, key):
        # same as LocoEnv._env_step, but the State view also sees t (steps into episode)
        n = state.shape[0]
        new_state = (self.model.step_metal if self.backend == "metal" else self.model.step)(state, action)
        t = t + 1.0
        reward, done = self.reward_done(state, new_state, action, t)
        bad = state_bad(new_state)
        reward = mx.where(bad | mx.isnan(reward), 0.0, reward)
        done = done | bad | (t >= self.max_episode_steps)
        donef = done.astype(mx.float32)
        ret = ret + reward
        ep_ret = ret * donef
        ep_len = t * donef
        ep_dist = mx.where(bad, 0.0, new_state[:, self.torso] - self.start_x0) * donef
        new_state = mx.where(done[:, None], self._fresh(key, n), new_state)
        t = mx.where(done, 0.0, t)
        ret = mx.where(done, 0.0, ret)
        obs = self.observe(new_state)
        return new_state, t, ret, obs, reward, donef, ep_ret, ep_len, ep_dist

    def __repr__(self):
        return (f"RobotEnv({self.name!r}, num_envs={self.num_envs}, obs_dim={self.obs_dim}, "
                f"act_dim={self.act_dim})")


def make(robot, task=None, num_envs=4096, seed=0, backend=None):
    """mg.make(robot, task, num_envs=4096) -> RobotEnv. Also accepts a (robot, task) pair
    (what mg.robots.* return) as the first argument, or a built-in env name string
    ("walker", "cheetah", "arm", "cartpole") for the legacy envs."""
    if isinstance(robot, str) and task is None:
        from macgym.envs import make as make_legacy
        return make_legacy(robot, num_envs, seed=seed)
    if isinstance(robot, tuple) and task is None:
        robot, task = robot
    if task is None:
        raise TypeError("mg.make(robot, task, ...): task is required")
    return RobotEnv(robot, task, num_envs=num_envs, seed=seed, backend=backend)


# --------------------------------------------------------------------------- training
@dataclass
class TrainResult:
    policy: object
    history: list = field(default_factory=list)
    ppo: object = None

    @property
    def final(self):
        return self.history[-1] if self.history else {}


def _fmt_steps(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def train(env, iters=200, hidden=(256, 256), on_iter=None, save=None, lr=3e-4, seed=0,
          verbose=True, log_every=None, **ppo_kwargs):
    """Train a PPO policy on env. on_iter(metrics) is called after every iteration with the
    PPO metrics dict plus `dist` (m, mean start-relative root x: finished episodes if any,
    else live). Return False from on_iter to stop early. Returns TrainResult(policy, history)."""
    from macgym.policy import Policy
    from macgym.rl.ppo import PPO

    ppo = PPO(env, hidden=tuple(hidden), lr=lr, seed=seed, **ppo_kwargs)
    history = []
    log_every = log_every or max(1, iters // 20)
    t0 = time.perf_counter()
    has_dist = hasattr(env, "live_dist_mean")
    if verbose:
        print(f"[macgym] training {getattr(env, 'name', 'env')}: {env.num_envs} envs, "
              f"obs {env.obs_dim}, act {env.act_dim}, {iters} iters", flush=True)
    dt = getattr(env, "dt", 0.02)
    for i in range(iters):
        m = ppo.train_iter()
        m["elapsed"] = time.perf_counter() - t0
        live = env.live_dist_mean() if has_dist else None
        m["live_dist"] = live
        ep_d = m.get("mean_episode_dist")
        m["dist"] = ep_d if ep_d is not None else live
        m["ep_seconds"] = m["mean_ep_len"] * dt
        history.append(m)
        if verbose and ((i + 1) % log_every == 0 or i == 0 or i + 1 == iters):
            d = f"dist {m['dist']:6.2f} m | " if m["dist"] is not None else ""
            print(f"  it {m['iter']:4d}/{iters} | {d}return {m['mean_ep_return']:8.1f} | "
                  f"episode {m['ep_seconds']:5.1f} s | {m['sps'] / 1e6:4.1f}M steps/s | "
                  f"{_fmt_steps(m['env_steps'])} steps | {m['elapsed']:5.1f}s", flush=True)
        if on_iter is not None and on_iter(m) is False:
            break
    policy = Policy.from_ppo(ppo, env)
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        ppo.save(save)
        if verbose:
            print(f"[macgym] saved {save}", flush=True)
    return TrainResult(policy=policy, history=history, ppo=ppo)


def evaluate(env, policy=None, steps=None):
    """Roll out `policy` (deterministic; zero actions if None) from a fresh reset for `steps`
    control steps (default: episode length, capped at 500). Returns mean root distance (m)
    reached before the first episode end, plus mean return."""
    steps = steps or min(env.max_episode_steps, 500)
    obs = env.reset()
    n = env.num_envs
    alive = mx.ones((n,))
    ret = mx.zeros((n,))
    dist = mx.zeros((n,))
    for _ in range(steps):
        a = policy.act(obs) if policy is not None else mx.zeros((n, env.act_dim))
        obs, rew, done, info = env.step(a)
        # where an episode ends this step, its distance is in info; otherwise track live x
        d_now = mx.where(done > 0, info["episode_dist"], env.state[:, env.torso] - env.start_x0)
        dist = mx.where(alive > 0, d_now, dist)
        ret = ret + alive * rew
        alive = alive * (1.0 - done)
        mx.eval(dist, ret, alive)
    env.reset()
    d = np.nan_to_num(np.array(dist))
    return {"dist": float(d.mean()), "return": float(np.array(ret).mean()),
            "survived": float(np.array(alive).mean())}
