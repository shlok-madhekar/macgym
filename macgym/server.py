"""MacGym live training server. See CONTRACT.md section 3.

    python3 -m macgym.server --port 8765 [--env walker --num-envs 4096 --autostart]

aiohttp app: static web/ at "/", WebSocket at "/ws" (broadcast to all clients).
All env / PPO / MLX work happens on a single trainer thread; the asyncio thread
only enqueues commands and forwards messages.
"""
import argparse
import asyncio
import importlib
import json
import math
import os
import queue
import threading
import time
import traceback

import numpy as np
from aiohttp import WSMsgType, web

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")
RUNS_DIR = os.path.join(ROOT, "runs")

FRAME_HZ = 30.0
DEFAULT_RENDER_ENVS = 64
MAX_RENDER_ENVS = 256

# Per-env PPO hyperparameters (merged over DEFAULT_HPARAMS).
DEFAULT_HPARAMS = {"hidden": (128, 128), "lr": 3e-4, "rollout_len": 32, "epochs": 4, "minibatches": 4}
HPARAMS = {
    "cartpole": {"hidden": (64, 64), "rollout_len": 32},
    "arm": {"hidden": (128, 128), "rollout_len": 32},
    "walker": {"hidden": (256, 256), "rollout_len": 24},
    "cheetah": {"hidden": (256, 256), "rollout_len": 24},
    "hopper": {"hidden": (256, 256), "rollout_len": 24},
    "stub": {"hidden": (64, 64), "rollout_len": 16},
}


def hparams_for(name):
    hp = dict(DEFAULT_HPARAMS)
    hp.update(HPARAMS.get(name, {}))
    return hp


# ---------------------------------------------------------------------------
# Stub fallbacks (used only when real modules are not importable)
# ---------------------------------------------------------------------------
class StubEnv:
    """Numpy stub: a 2-link swinging arm per env, so the viewer has motion."""

    name = "stub"
    description = "Stub env (real envs not importable): swinging two-link arm."
    obs_dim = 4
    act_dim = 2
    max_episode_steps = 200
    dt = 0.033

    def __init__(self, num_envs, seed=0):
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.phase = self.rng.uniform(0, 2 * np.pi, num_envs).astype(np.float32)
        self.target = np.array([0.5, 0.8], np.float32)

    def reset(self):
        return np.zeros((self.num_envs, self.obs_dim), np.float32)

    def step(self, action):
        self.t += self.dt
        a = np.asarray(action, np.float32).reshape(self.num_envs, -1)
        self.phase += self.dt * (1.0 + 0.5 * a[:, 0])
        obs = np.stack([np.sin(self.phase), np.cos(self.phase), a[:, 0], a[:, 1]], 1).astype(np.float32)
        return obs, np.zeros(self.num_envs, np.float32), np.zeros(self.num_envs, np.float32), {}

    def render_state(self, R):
        R = min(R, self.num_envs)
        th1 = np.sin(self.phase[:R] + self.t) * 1.2 + np.pi / 2
        th2 = th1 + np.sin(2.0 * self.t + self.phase[:R]) * 1.5
        x0 = np.zeros(R, np.float32)
        y0 = np.full(R, 0.1, np.float32)
        x1, y1 = x0 + 0.5 * np.cos(th1), y0 + 0.5 * np.sin(th1)
        x2, y2 = x1 + 0.4 * np.cos(th2), y1 + 0.4 * np.sin(th2)
        caps = np.zeros((R, 2, 5), np.float32)
        caps[:, 0] = np.stack([x0, y0, x1, y1, np.full(R, 0.05)], 1)
        caps[:, 1] = np.stack([x1, y1, x2, y2, np.full(R, 0.04)], 1)
        targets = np.broadcast_to(self.target, (R, 1, 2)).astype(np.float32)
        return {"capsules": caps, "targets": targets, "colors": ["#5B8CFF", "#36D399"],
                "ground": 0.0, "world_bounds": [-1.1, 1.1, -0.2, 1.2]}

    def set_target(self, x, y):
        self.target = np.array([x, y], np.float32)


class StubPPO:
    """Fake trainer: emits plausible improving metrics, random actions."""

    def __init__(self, env, seed=0, **kw):
        self.env = env
        self.rollout_len = kw.get("rollout_len", 16)
        self.it = 0
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        self.obs = env.reset()

    def train_iter(self):
        t0 = time.time()
        for _ in range(4):
            self.obs, *_ = self.env.step(self.act(self.obs))
        time.sleep(0.05)
        self.it += 1
        self.steps += self.env.num_envs * self.rollout_len
        dt = time.time() - t0
        prog = 1.0 - math.exp(-self.it / 50.0)
        return {"iter": self.it, "env_steps": self.steps, "sps": self.env.num_envs * self.rollout_len / dt,
                "mean_reward": prog, "mean_ep_return": 200 * prog + self.rng.normal(0, 3),
                "mean_ep_len": 200.0, "policy_loss": -0.01, "value_loss": 1.0 / (1 + self.it),
                "entropy": 1.0 - 0.5 * prog, "time": dt}

    def act(self, obs, deterministic=False):
        n = self.env.num_envs
        if deterministic:
            return np.zeros((n, self.env.act_dim), np.float32)
        return self.rng.uniform(-1, 1, (n, self.env.act_dim)).astype(np.float32)

    def save(self, path):
        np.savez(path, it=self.it)

    def load(self, path):
        self.it = int(np.load(path)["it"])


# ---------------------------------------------------------------------------
# Module resolution (prefer real modules)
# ---------------------------------------------------------------------------
KNOWN_ENV_MODULES = {
    "cartpole": ("macgym.envs.cartpole", "CartPole"),
    "arm": ("macgym.envs.arm", "ArmReach"),
    "walker": ("macgym.envs.walker", "Walker2D"),
    "cheetah": ("macgym.envs.cheetah", "Cheetah2D"),
}


def env_registry():
    """name -> env class. Real registry if present, else known modules, plus stub."""
    reg = {}
    try:
        envs_pkg = importlib.import_module("macgym.envs")
        reg.update(getattr(envs_pkg, "ENVS", {}) or {})
    except Exception:
        traceback.print_exc()
    for name, (mod, cls) in KNOWN_ENV_MODULES.items():
        if name in reg:
            continue
        try:
            reg[name] = getattr(importlib.import_module(mod), cls)
        except Exception:
            pass
    reg["stub"] = StubEnv
    return reg


def make_env(name, num_envs):
    reg = env_registry()
    if name not in reg:
        raise ValueError(f"unknown env {name!r}; available: {sorted(reg)}")
    return reg[name](num_envs)


def checkpoint_path(name):
    return os.path.join(RUNS_DIR, f"{name}.npz") if name else None


def has_checkpoint(name):
    p = checkpoint_path(name)
    return bool(p and os.path.exists(p))


def measure_sim_sps(env, steps=200, max_seconds=3.0):
    """Sim-only throughput: random actions, no policy / no PPO update. Leaves env state dirty."""
    n, act_dim = int(env.num_envs), int(env.act_dim)
    if isinstance(env, StubEnv):
        rand = lambda: np.random.uniform(-1, 1, (n, act_dim)).astype(np.float32)
        sync = lambda out: None
    else:
        import mlx.core as mx
        rand = lambda: mx.random.uniform(-1.0, 1.0, (n, act_dim))
        sync = lambda out: mx.eval(out[0], out[1])
    env.reset()
    for _ in range(5):  # warm-up / compile
        sync(env.step(rand()))
    t0 = time.perf_counter()
    done = 0
    for _ in range(steps):
        sync(env.step(rand()))
        done += 1
        if time.perf_counter() - t0 > max_seconds:
            break
    dt = max(time.perf_counter() - t0, 1e-9)
    return done * n / dt, done


def make_trainer(env, name):
    hp = hparams_for(name)
    if isinstance(env, StubEnv):
        return StubPPO(env, **hp)
    try:
        from macgym.rl.ppo import PPO
    except Exception:
        traceback.print_exc()
        print("[server] macgym.rl.ppo not importable; using StubPPO", flush=True)
        return StubPPO(env, **hp)
    return PPO(env, **hp)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _flat(a):
    if a is None:
        return None
    return np.round(np.asarray(a, np.float64).ravel(), 3).tolist()


def _num(v):
    if isinstance(v, (np.generic,)):
        v = v.item()
    if hasattr(v, "item") and not isinstance(v, (int, float, str)):
        try:
            v = v.item()
        except Exception:
            return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return round(v, 5)
    if isinstance(v, (tuple, list)):
        return [_num(x) for x in v]
    return v


LOCO_ENVS = ("walker", "cheetah", "hopper")
REPLAY_SCHEDULE = (10.0, 30.0, 60.0, 120.0, 240.0)  # training wall seconds
REPLAY_MAX_STEPS = 400
REPLAY_EVERY = 2
ARM_REPLAY_TARGETS = ((0.55, 0.45), (-0.45, 0.55))  # fixed target sequence, 200 steps each


def fmt_clock(sec):
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def start_relative(rs):
    """Shift capsule / camera x so each robot's episode start is x=0 (locomotion)."""
    caps = np.array(rs["capsules"], np.float32)
    sx = rs.get("start_x")
    cam = rs.get("camera_x")
    if sx is not None:
        sx = np.asarray(sx, np.float32)[:, None]
        caps[:, :, 0] -= sx
        caps[:, :, 2] -= sx
        if cam is not None:
            cam = np.asarray(cam, np.float32) - sx[:, 0]
    return caps, cam


def frame_message(env_name, rs):
    caps, cam = start_relative(rs)
    caps = np.nan_to_num(caps, nan=0.0, posinf=0.0, neginf=0.0)
    R, K = int(caps.shape[0]), int(caps.shape[1])
    wb = rs.get("world_bounds")
    msg = {
        "type": "frame", "env": env_name, "R": R, "K": K,
        "capsules": _flat(caps),
        "targets": _flat(rs.get("targets")),
        "colors": list(rs["colors"]) if rs.get("colors") is not None else None,
        "ground": float(rs["ground"]) if rs.get("ground") is not None else None,
        "world_bounds": [round(float(v), 3) for v in wb] if wb is not None else None,
        "camera_x": _flat(cam),
        "start_relative": rs.get("start_x") is not None,
    }
    d = rs.get("dist")
    if d is not None:
        d = np.nan_to_num(np.asarray(d, np.float32), nan=0.0)
        hib = bool(rs.get("higher_is_better", True))
        msg["dist"] = _flat(d)
        msg["best_index"] = int(np.argmax(d) if hib else np.argmin(d))
        msg["dist_higher_is_better"] = hib
    else:
        msg["dist"] = None
        msg["best_index"] = 0
    return msg


def human_fields(env_name, env, m, train_seconds):
    """Human-unit headline metrics (REDESIGN.md section 1). Window = PPO's last ~100 episodes;
    falls back to the live mean of in-progress episodes before any episode finished."""
    out = {"train_seconds": round(train_seconds, 2)}
    dt = float(getattr(env, "dt", 0.02))
    n_eps = m.get("episodes_in_window") or 0
    try:
        if env_name in LOCO_ENVS:
            verb = "run" if env_name == "cheetah" else "walked"
            mean, best = m.get("mean_episode_dist"), m.get("max_episode_dist")
            src = "episodes"
            if mean is None and hasattr(env, "live_dist_mean"):
                mean, best, src = env.live_dist_mean(), None, "live"
            out.update({"dist_mean": mean, "dist_best": best, "unit": "m",
                        "headline_label": f"distance {verb}",
                        "human_value": mean, "human_best": best, "human_unit": "m",
                        "human_label": f"average distance {verb}", "higher_is_better": True,
                        "human_source": src})
        elif env_name == "arm":
            mean, best = m.get("mean_episode_err"), m.get("min_episode_err")
            src = "episodes"
            if mean is None and hasattr(env, "live_err_mean"):
                mean, best, src = env.live_err_mean(), None, "live"
            out.update({"err_mean": mean, "err_best": best, "unit": "cm",
                        "headline_label": "miss distance",
                        "human_value": None if mean is None else mean * 100.0,
                        "human_best": None if best is None else best * 100.0, "human_unit": "cm",
                        "human_label": "average miss distance", "higher_is_better": False,
                        "human_source": src})
        elif env_name == "cartpole":
            if n_eps:
                mean, best, src = m.get("mean_ep_len", 0.0) * dt, (m.get("max_ep_len") or 0.0) * dt, "episodes"
            else:
                mean, best, src = (env.live_balance_mean() if hasattr(env, "live_balance_mean") else None), None, "live"
            out.update({"balance_mean": mean, "balance_best": best, "unit": "s",
                        "headline_label": "time balanced",
                        "human_value": mean, "human_best": best, "human_unit": "s",
                        "human_label": "average time balanced", "higher_is_better": True,
                        "human_source": src})
        else:
            v = m.get("mean_ep_return")
            out.update({"human_value": v, "human_best": None, "human_unit": "",
                        "human_label": "average episode return", "higher_is_better": True,
                        "human_source": "episodes"})
    except Exception:
        traceback.print_exc()
    return {k: _num(v) for k, v in out.items()}


def run_replay(env_name, renv, ppo):
    """Run the current policy deterministically on a separate small env; returns replay payload.
    States are captured lazily (mx references) and converted in one sync at the end, so the
    rollout costs ~one GPU dispatch per step instead of a CPU round trip per frame."""
    import mlx.core as mx
    renv.key = mx.random.key(1234)  # same start state for every replay of this env
    is_arm = env_name == "arm" and hasattr(renv, "q")
    is_loco = env_name in LOCO_ENVS and hasattr(renv, "model")
    if is_arm:
        renv.reset()
        renv.set_target(*ARM_REPLAY_TARGETS[0])
        obs = renv.obs
    else:
        obs = renv.reset()
    dt = float(renv.dt)
    lazy = []   # (step, payload)
    fell, steps, result = False, 0, None

    def capture(step):
        if is_loco:
            lazy.append((step, renv.state[0]))
        elif is_arm:
            lazy.append((step, mx.concatenate([renv.q[0], renv.target[0]])))
        else:
            lazy.append((step, renv.render_state(1)))

    capture(0)
    max_ep = getattr(renv, "max_episode_steps", 10 ** 9)
    pending = []  # (step, done, ep_dist) not yet synced
    for i in range(REPLAY_MAX_STEPS):
        if is_arm and i == REPLAY_MAX_STEPS // 2 and len(ARM_REPLAY_TARGETS) > 1:
            renv.set_target(*ARM_REPLAY_TARGETS[1])
            obs = renv.obs
        act = ppo.act(obs, deterministic=True)
        obs, rew, done, info = renv.step(act)
        steps = i + 1
        if not is_arm:
            pending.append((steps, done, info.get("episode_dist")))
        if steps % REPLAY_EVERY == 0:
            capture(steps)
        if pending and (steps % 20 == 0 or steps == REPLAY_MAX_STEPS):
            dn = np.asarray(mx.stack([p[1].reshape(-1)[0] for p in pending]))
            hit = np.nonzero(dn > 0.5)[0]
            if len(hit):
                st, _, ed = pending[int(hit[0])]
                steps = st
                fell = st < max_ep
                result = float(np.asarray(ed).reshape(-1)[0]) if (is_loco and ed is not None) else st * dt
                break
            pending = []
    lazy = [(st, p) for st, p in lazy if st < steps or (st == steps and not fell)]
    targets = None
    if is_loco:
        S = np.array(mx.stack([p for _, p in lazy]), dtype=np.float32)
        caps = renv.model.capsules_np(S)
        sx = getattr(renv, "start_x0", 0.0)
        caps[:, :, 0] -= sx
        caps[:, :, 2] -= sx
        last_rs = {"colors": list(renv.colors), "ground": 0.0,
                   "dist": np.array([S[-1, renv.torso] - sx], np.float32)}
    elif is_arm:
        from macgym.envs import arm as arm_mod
        Q = np.array(mx.stack([p for _, p in lazy]), dtype=np.float64)
        q, tgt = Q[:, :3], Q[:, 3:5]
        ang = np.cumsum(q, axis=1)
        pts = [np.zeros((len(Q), 2))]
        for j, L in enumerate(arm_mod.LINK_LEN):
            pts.append(pts[-1] + L * np.stack([np.cos(ang[:, j]), np.sin(ang[:, j])], 1))
        caps = np.zeros((len(Q), 3, 5), np.float32)
        for j, rad in enumerate((0.045, 0.038, 0.03)):
            caps[:, j, 0:2], caps[:, j, 2:4], caps[:, j, 4] = pts[j], pts[j + 1], rad
        targets = np.round(tgt, 3).tolist()
        last_rs = {"colors": ["#4F8CFF", "#7C5CFF", "#22D3A6"], "ground": None,
                   "dist": np.array([np.linalg.norm(pts[3][-1] - tgt[-1])], np.float32)}
    else:
        caps = np.stack([start_relative(p)[0][0] for _, p in lazy])
        last_rs = lazy[-1][1]
        if last_rs.get("targets") is not None:
            targets = [np.round(np.asarray(p["targets"][0][0], np.float64), 3).tolist() for _, p in lazy]
    caps = np.nan_to_num(caps, nan=0.0)
    K = int(caps.shape[1])
    frames = [np.round(c.ravel().astype(np.float64), 3).tolist() for c in caps]
    T = steps * dt
    if is_loco:
        if result is None:
            result = float(np.asarray(last_rs["dist"]).reshape(-1)[0])
        result = float(np.nan_to_num(result))
        verb = "ran" if env_name == "cheetah" else "walked"
        text = f"fell after {T:.1f} s · {result:.1f} m" if fell else f"{verb} {result:.1f} m in {T:.0f} s"
    elif env_name == "cartpole":
        result = T
        text = f"fell after {T:.1f} s" if fell else f"balanced {T:.0f} s, never fell"
    elif is_arm:
        err = float(np.asarray(last_rs["dist"]).reshape(-1)[0])
        result = err * 100.0
        text = f"missed by {result:.1f} cm"
    else:
        result = None
        text = f"{T:.1f} s"
    return {"K": K, "frames": frames, "targets": targets,
            "colors": list(last_rs["colors"]) if last_rs and last_rs.get("colors") is not None else None,
            "ground": last_rs.get("ground") if last_rs else None,
            "result": _num(result), "result_text": text, "dt": dt * REPLAY_EVERY,
            "steps": steps, "fell": fell}


# ---------------------------------------------------------------------------
# Trainer thread
# ---------------------------------------------------------------------------
class Trainer:
    """Owns env + PPO. All MLX work runs on its own thread; commands via queue."""

    def __init__(self, emit):
        self.emit = emit  # thread-safe callable(dict)
        self.cmds = queue.Queue()
        self.env = None
        self.ppo = None
        self.env_name = None
        self.num_envs = 0
        self.render_envs = DEFAULT_RENDER_ENVS
        self.sim_info = None
        self.state = "idle"
        self.obs = None
        self.run_id = 0
        self.run_t0 = None          # wall time training started
        self.replays = []           # replay messages of the current run (re-sent on connect)
        self.replay_env = None
        self.replay_next = 0        # index into REPLAY_SCHEDULE
        self.last_frame_t = 0.0
        self.thread = threading.Thread(target=self._run, name="trainer", daemon=True)
        self._alive = True

    def start_thread(self):
        self.thread.start()

    def shutdown(self):
        self._alive = False
        self.cmds.put(("quit", {}))

    def submit(self, cmd, msg):
        self.cmds.put((cmd, msg))

    def status(self):
        return {"type": "status", "state": self.state, "env": self.env_name, "num_envs": self.num_envs,
                "render_envs": self.render_envs, "checkpoint": has_checkpoint(self.env_name)}

    # -- command handlers (trainer thread) --
    def _handle(self, cmd, msg):
        if cmd == "start":
            name = msg.get("env", "cartpole")
            n = int(msg.get("num_envs", 4096))
            n = max(1, min(n, 65536))
            self.render_envs = max(1, min(int(msg.get("render_envs", DEFAULT_RENDER_ENVS)), MAX_RENDER_ENVS, n))
            self.env = self.ppo = None
            self.state = "idle"
            env = make_env(name, n)
            self.emit({"type": "status", "state": "measuring", "env": name, "num_envs": n,
                       "render_envs": self.render_envs, "checkpoint": has_checkpoint(name)})
            try:
                sim_sps, nsteps = measure_sim_sps(env)
                self.sim_info = {"type": "info", "sim_sps": round(sim_sps, 1), "env": name, "num_envs": n,
                                 "steps": nsteps, "note": "sim-only: random actions, no policy, no PPO update"}
                self.emit(self.sim_info)
            except Exception as e:
                traceback.print_exc()
                self.emit({"type": "info", "sim_sps": None, "message": f"sim benchmark skipped: {e}"})
            env.reset()
            self.ppo = make_trainer(env, name)
            self.env, self.env_name, self.num_envs = env, name, n
            self.obs = None
            self._new_run()
            self._push_frame(force=True)
            self._replay("0:00", 0.0)  # untrained policy, BEFORE any update
            self.run_t0 = time.time()
            self.state = "training"
            self.emit(self.status())
        elif cmd == "stop":
            was = self.state
            self.state = "idle"
            self.emit(self.status())
            if self.ppo is not None and was != "idle":
                self._replay("now", self._train_seconds())
        elif cmd == "showcase":
            if self.env is None:
                self.emit({"type": "error", "message": "no active run"})
                return
            self.state = "showcase" if msg.get("on", True) else "training"
            if self.state == "showcase":
                self.obs = self.env.reset()
            self.emit(self.status())
            if self.state == "showcase":
                self._replay("now", self._train_seconds())
            if self.state == "showcase" and has_checkpoint(self.env_name):
                self.emit({"type": "info", "checkpoint": True, "env": self.env_name,
                           "message": f"Saved policy runs/{self.env_name}.npz available: press Load trained"})
        elif cmd == "render":
            n = int(msg.get("render_envs", msg.get("r", DEFAULT_RENDER_ENVS)))
            self.render_envs = max(1, min(n, MAX_RENDER_ENVS))
            if self.env is not None:
                self._push_frame(force=True)
            self.emit(self.status())
        elif cmd == "set_target":
            if self.env is not None and hasattr(self.env, "set_target"):
                self.env.set_target(float(msg["x"]), float(msg["y"]))
        elif cmd == "save":
            if self.ppo is None:
                self.emit({"type": "error", "message": "no active run"})
                return
            os.makedirs(RUNS_DIR, exist_ok=True)
            path = os.path.join(RUNS_DIR, f"{self.env_name}.npz")
            self.ppo.save(path)
            self.emit({"type": "saved", "path": os.path.relpath(path, ROOT)})
            self.emit(self.status())
        elif cmd == "load":
            name = msg.get("env") or self.env_name
            path = os.path.join(RUNS_DIR, f"{name}.npz")
            if not name or not os.path.exists(path):
                self.emit({"type": "error", "message": f"no checkpoint at runs/{name}.npz"})
                return
            if self.env is None or self.env_name != name:
                n = int(msg.get("num_envs", self.num_envs or 1024))
                self.env = make_env(name, n)
                self.ppo = make_trainer(self.env, name)
                self.env_name, self.num_envs = name, n
                self._new_run()
            self.ppo.load(path)
            self.obs = self.env.reset()
            if hasattr(self.ppo, "obs"):
                self.ppo.obs = None
            self.state = "showcase"
            self.emit({"type": "loaded", "path": os.path.relpath(path, ROOT)})
            self.emit(self.status())
            self._replay("now", self._train_seconds())

    def _new_run(self):
        self.run_id += 1
        self.replays = []
        self.replay_next = 0
        self.run_t0 = None
        self.replay_env = None
        try:
            if not isinstance(self.env, StubEnv):
                self.replay_env = make_env(self.env_name, 1)
        except Exception:
            traceback.print_exc()

    def _train_seconds(self):
        return (time.time() - self.run_t0) if self.run_t0 else 0.0

    def _replay(self, label, train_seconds):
        if self.replay_env is None or self.ppo is None:
            return
        t0 = time.perf_counter()
        try:
            r = run_replay(self.env_name, self.replay_env, self.ppo)
        except Exception as e:
            traceback.print_exc()
            self.emit({"type": "error", "message": f"replay failed: {e}"})
            return
        msg = {"type": "replay", "env": self.env_name, "run_id": self.run_id, "label": label,
               "train_seconds": round(train_seconds, 2), "iter": int(getattr(self.ppo, "iter", 0)),
               **r, "cost_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
        if label == "now":
            self.replays = [x for x in self.replays if x["label"] != "now"]
        self.replays.append(msg)
        self.emit(msg)

    def _maybe_scheduled_replay(self):
        if self.replay_next >= len(REPLAY_SCHEDULE) or self.run_t0 is None:
            return
        ts = self._train_seconds()
        if ts >= REPLAY_SCHEDULE[self.replay_next]:
            while self.replay_next < len(REPLAY_SCHEDULE) and ts >= REPLAY_SCHEDULE[self.replay_next]:
                label = fmt_clock(REPLAY_SCHEDULE[self.replay_next])
                self.replay_next += 1
            self._replay(label, ts)

    def _push_frame(self, force=False):
        now = time.time()
        if not force and now - self.last_frame_t < 1.0 / FRAME_HZ:
            return
        self.last_frame_t = now
        try:
            rs = self.env.render_state(min(self.render_envs, self.num_envs))
            self.emit(frame_message(self.env_name, rs))
        except Exception as e:
            traceback.print_exc()
            self.emit({"type": "error", "message": f"render_state failed: {e}"})

    def _train_step(self):
        m = self.ppo.train_iter()
        out = {k: _num(v) for k, v in dict(m).items()}
        out["type"] = "metrics"
        out["env"] = self.env_name
        out["num_envs"] = self.num_envs
        out.update(human_fields(self.env_name, self.env, dict(m), self._train_seconds()))
        self.emit(out)
        self._push_frame()
        self._maybe_scheduled_replay()

    def _showcase_step(self):
        t0 = time.time()
        if self.obs is None:
            self.obs = self.env.reset()
        act = self.ppo.act(self.obs, deterministic=True)
        self.obs, *_ = self.env.step(act)
        self._push_frame()
        rem = 1.0 / FRAME_HZ - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)

    def _run(self):
        while self._alive:
            block = self.state == "idle"
            try:
                while True:
                    if block:
                        cmd, msg = self.cmds.get(timeout=0.5)
                    else:
                        cmd, msg = self.cmds.get_nowait()
                    if cmd == "quit":
                        return
                    try:
                        self._handle(cmd, msg)
                    except Exception as e:
                        traceback.print_exc()
                        self.state = "idle" if self.env is None else self.state
                        self.emit({"type": "error", "message": f"{cmd} failed: {e}"})
                        self.emit(self.status())
                    block = False
            except queue.Empty:
                pass
            try:
                if self.state == "training":
                    self._train_step()
                elif self.state == "showcase":
                    self._showcase_step()
            except Exception as e:
                traceback.print_exc()
                self.state = "idle"
                self.emit({"type": "error", "message": f"trainer crashed: {e}"})
                self.emit(self.status())


# ---------------------------------------------------------------------------
# aiohttp app
# ---------------------------------------------------------------------------
class Server:
    def __init__(self):
        self.clients = set()
        self.loop = None
        self.trainer = Trainer(self.emit_threadsafe)

    def emit_threadsafe(self, msg):
        if self.loop is None or self.loop.is_closed():
            return
        try:
            data = json.dumps(msg, separators=(",", ":"))
        except Exception as e:
            data = json.dumps({"type": "error", "message": f"serialize: {e}"})
        self.loop.call_soon_threadsafe(self._schedule_broadcast, data, msg.get("type"))

    def _schedule_broadcast(self, data, mtype):
        for ws in list(self.clients):
            if ws.closed:
                self.clients.discard(ws)
                continue
            # drop frames for slow clients so they don't pile up
            if mtype == "frame" and getattr(ws, "_macgym_pending", 0) > 4:
                continue
            ws._macgym_pending = getattr(ws, "_macgym_pending", 0) + 1
            asyncio.ensure_future(self._send(ws, data))

    async def _send(self, ws, data):
        try:
            await ws.send_str(data)
        except Exception:
            self.clients.discard(ws)
        finally:
            ws._macgym_pending = max(0, getattr(ws, "_macgym_pending", 1) - 1)

    def list_envs(self):
        out = []
        reg = env_registry()
        if len(reg) > 1:
            reg.pop("stub", None)
        for name, cls in reg.items():
            out.append({"name": name, "obs_dim": int(getattr(cls, "obs_dim", 0)),
                        "act_dim": int(getattr(cls, "act_dim", 0)),
                        "description": getattr(cls, "description", "") or "",
                        "checkpoint": has_checkpoint(name)})
        return {"type": "envs", "envs": out}

    async def ws_handler(self, request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 22)
        await ws.prepare(request)
        self.clients.add(ws)
        try:
            await ws.send_str(json.dumps(self.trainer.status()))
            if self.trainer.sim_info:
                await ws.send_str(json.dumps(self.trainer.sim_info))
            for rp in list(self.trainer.replays):
                await ws.send_str(json.dumps(rp, separators=(",", ":")))
            async for m in ws:
                if m.type != WSMsgType.TEXT:
                    if m.type == WSMsgType.ERROR:
                        break
                    continue
                try:
                    msg = json.loads(m.data)
                    cmd = msg.get("cmd")
                except Exception:
                    await ws.send_str(json.dumps({"type": "error", "message": "bad json"}))
                    continue
                if cmd == "list":
                    await ws.send_str(json.dumps(await asyncio.get_running_loop().run_in_executor(None, self.list_envs)))
                elif cmd == "bench":
                    p = os.path.join(WEB_DIR, "bench.json")
                    data = None
                    if os.path.exists(p):
                        try:
                            with open(p) as f:
                                data = json.load(f)
                        except Exception:
                            data = None
                    await ws.send_str(json.dumps({"type": "bench", "data": data}))
                elif cmd == "status":
                    await ws.send_str(json.dumps(self.trainer.status()))
                elif cmd in ("start", "stop", "showcase", "set_target", "save", "load", "render"):
                    self.trainer.submit(cmd, msg)
                else:
                    await ws.send_str(json.dumps({"type": "error", "message": f"unknown cmd {cmd!r}"}))
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.clients.discard(ws)
        return ws

    async def index(self, request):
        p = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(p):
            return web.FileResponse(p)
        return web.Response(text="MacGym server running. web/index.html not found.", content_type="text/html")

    async def on_startup(self, app):
        self.loop = asyncio.get_running_loop()
        self.trainer.start_thread()

    async def on_cleanup(self, app):
        self.trainer.shutdown()
        for ws in list(self.clients):
            await ws.close()

    def make_app(self):
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.ws_handler)
        os.makedirs(WEB_DIR, exist_ok=True)
        app.router.add_static("/", WEB_DIR, show_index=False)
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        return app


def main():
    ap = argparse.ArgumentParser(description="MacGym live training server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--env", default="walker")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--render-envs", type=int, default=DEFAULT_RENDER_ENVS)
    ap.add_argument("--autostart", action="store_true")
    args = ap.parse_args()
    srv = Server()
    if args.autostart:
        srv.trainer.submit("start", {"env": args.env, "num_envs": args.num_envs, "render_envs": args.render_envs})
    print(f"[server] MacGym on http://localhost:{args.port}  (ws /ws)", flush=True)
    web.run_app(srv.make_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
