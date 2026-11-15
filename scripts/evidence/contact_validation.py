"""Contact-rich validation: MacGym walker vs a MuJoCo MJCF mirror of the same robot.
(1) passive collapse, zero torque, 3 s (rest pose + small rigid-body tilt rate);
(2) open-loop sinusoidal hip/knee torques, 2 s. Writes web/deck/data/contact_validation.{json,png}."""
import json, math, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mj_walker import ROOT, MjWalker, DT, TORSO_Y0
import numpy as np
import mlx.core as mx
from macgym import robots

OUT = os.path.join(ROOT, "web/deck/data")
robot, task = robots.walker()
M = robot.build_model()
B = M.B


def ours_init(omega):
    s = M.init_state.reshape(6, B).astype(np.float64).copy()
    cx, cy = s[0, 0], s[1, 0]
    s[5, :] = omega                      # rigid rotation about torso centre
    s[3, :] = -omega * (s[1] - cy)
    s[4, :] = omega * (s[0] - cx)
    return s.reshape(-1).astype(np.float32)


def ours_lowest(st):
    caps = M.capsules_np(st[None])[0]
    return np.minimum(caps[:, 1], caps[:, 3]) - caps[:, 4]


def run_ours(omega, actfn, T):
    st = mx.array(ours_init(omega)[None])
    rows = []
    step = mx.compile(M.step_metal)
    for k in range(int(round(T / DT)) + 1):
        s = np.array(st[0]).astype(np.float64)
        r = s.reshape(6, B)
        rows.append([k * DT, r[0, 0], r[1, 0], r[2, 0] - math.pi / 2, *ours_lowest(s.astype(np.float32))])
        a = mx.array(np.asarray(actfn(k * DT), np.float32)[None])
        st = step(st, a); mx.eval(st)
    return np.array(rows)


def run_mj(omega, actfn, T):
    w = MjWalker(speed_curve=True)
    qv = np.zeros(9); qv[2] = omega
    w.reset(qv)
    rows = []
    for k in range(int(round(T / DT)) + 1):
        t = w.torso()
        rows.append([k * DT, t["x"], t["y"], t["angle"], *w.lowest()])
        w.step(actfn(k * DT))
    return np.array(rows)


def first_time(rows, mask_fn):
    idx = np.where(mask_fn(rows))[0]
    return float(rows[idx[0], 0]) if len(idx) else None


def agree_window(a, b, col, tol):
    d = np.abs(a[:, col] - b[:, col])
    bad = np.where(d > tol)[0]
    return float(a[bad[0], 0]) if len(bad) else float(a[-1, 0])


def compare(name, ours, mj):
    res = {}
    for col, key in ((1, "torso_x"), (2, "torso_height"), (3, "torso_angle")):
        d = ours[:, col] - mj[:, col]
        res[key] = {"rms_full": float(np.sqrt(np.mean(d ** 2))), "max_abs": float(np.max(np.abs(d))),
                    "ours_final": float(ours[-1, col]), "mujoco_final": float(mj[-1, col])}
        for w in (0.5, 1.0):
            m = ours[:, 0] <= w + 1e-9
            res[key][f"rms_first_{w}s"] = float(np.sqrt(np.mean(d[m] ** 2)))
    res["agree_until_s"] = {"height_within_5cm": agree_window(ours, mj, 2, 0.05),
                            "angle_within_0.1rad": agree_window(ours, mj, 3, 0.1)}
    nonfoot = [0, 1, 4]   # torso, thighs (shin tips sit on the foot, so excluded) (lowest point < 2 cm = ground contact)
    for lab, rows in (("ours", ours), ("mujoco", mj)):
        res[f"{lab}_t_first_nonfoot_contact_s"] = first_time(rows, lambda r: np.min(r[:, 4 + np.array(nonfoot)], axis=1) < 0.02)
        res[f"{lab}_t_torso_contact_s"] = first_time(rows, lambda r: r[:, 4] < 0.02)
        res[f"{lab}_t_fall_s"] = first_time(rows, lambda r: (r[:, 2] < 0.7 * TORSO_Y0) | (np.abs(r[:, 3]) > 1.0))
        res[f"{lab}_t_feet_contact_s"] = first_time(rows, lambda r: np.minimum(r[:, 7], r[:, 10]) < 0.02)
    return res


zero = lambda t: np.zeros(6)
def sine(t):
    s = math.sin(2 * math.pi * 1.5 * t)
    return np.array([0.6 * s, -0.4 * max(s, 0), 0.2 * s, -0.6 * s, -0.4 * max(-s, 0), -0.2 * s])

cases = {
    "collapse_rest": dict(omega=0.0, act=zero, T=3.0, desc="rest pose, zero velocity, zero torque, 3 s"),
    "collapse_tilt": dict(omega=0.3, act=zero, T=3.0, desc="rest pose + 0.3 rad/s rigid-body tilt rate, zero torque, 3 s"),
    "open_loop_sine": dict(omega=0.0, act=sine, T=2.0,
                           desc="open-loop torques 1.5 Hz: hips +/-0.6 antiphase, knees -0.4*max(+/-s,0), ankles +/-0.2 (x gear 100 Nm), 2 s"),
}
results, traj = {}, {}
for k, c in cases.items():
    o = run_ours(c["omega"], c["act"], c["T"]); m = run_mj(c["omega"], c["act"], c["T"])
    results[k] = {"description": c["desc"], **compare(k, o, m)}
    traj[k] = (o, m)
    print(k, json.dumps(results[k], indent=1))

up = subprocess.check_output(["uptime"], text=True).strip()
import mujoco
out = {"what": "MacGym 2D walker (sequential-impulse, Metal kernel) vs MuJoCo MJCF mirror of the same robot",
       "mujoco_version": mujoco.__version__, "mujoco_model": "scripts/evidence/mj_walker.py (timestep 0.0025, frame_skip 8, Euler)",
       "matched": "capsule endpoints/radii/density->mass, in-plane inertia, joint positions/limits, motor gear 100 + our torque-speed curve, damping 0.1, friction 0.9, ground-only contact",
       "not_matched": "contact model (ours: hard velocity-level impulses + Baumgarte, 4 GS iters; MuJoCo: soft convex constraints solref 0.02), our velocity clamps (30 m/s, 60 rad/s)",
       "cases": results, "uptime": up, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
json.dump(out, open(os.path.join(OUT, "contact_validation.json"), "w"), indent=1)

# ---- plot (paper/ink)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.edgecolor": "#111", "axes.labelcolor": "#111",
                     "xtick.color": "#111", "ytick.color": "#111", "axes.linewidth": 0.6})
fig, axs = plt.subplots(3, 2, figsize=(10, 7.2), facecolor="#FAFAF7")
rowsdef = [("collapse_rest", "Passive collapse (zero torque)"), ("collapse_tilt", "Passive collapse, 0.3 rad/s tilt"),
           ("open_loop_sine", "Open-loop sinusoidal torques")]
for i, (k, title) in enumerate(rowsdef):
    o, m = traj[k]
    cols = [(2, "torso height (m)"), (3, "torso angle (rad)")] if k != "open_loop_sine" else [(2, "torso height (m)"), (1, "torso x (m)")]
    for j, (c, lab) in enumerate(cols):
        ax = axs[i, j]; ax.set_facecolor("#FAFAF7")
        ax.plot(o[:, 0], o[:, c], color="#FF4D00", lw=1.3, label="MacGym (Metal)")
        ax.plot(m[:, 0], m[:, c], color="#111", lw=1.0, ls="--", label="MuJoCo (same robot)")
        rms = np.sqrt(np.mean((o[:, c] - m[:, c]) ** 2))
        msk = o[:, 0] <= 0.5
        rms5 = np.sqrt(np.mean((o[msk, c] - m[msk, c]) ** 2))
        ax.set_title(f"{title}  |  RMS {rms:.3f}  (first 0.5 s: {rms5:.3f})", fontsize=8.5, color="#111", loc="left")
        ax.set_ylabel(lab); ax.spines[["top", "right"]].set_visible(False)
        ax.axvspan(0, 0.5, color="#FF4D00", alpha=0.06, lw=0)
        if i == 2: ax.set_xlabel("time (s)")
        if i == 0 and j == 0: ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "contact_validation.png"), dpi=160, facecolor="#FAFAF7")
print("saved", up)
