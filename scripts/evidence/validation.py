"""Sim-to-sim validation of macgym physics2d vs MuJoCo. Writes web/deck/data/validation.json."""
import json, math, os, subprocess, time
import numpy as np
import mlx.core as mx
import mujoco
from macgym.envs.physics2d import Capsule, Joint, Model, GRAVITY

OUT = os.path.join(os.path.dirname(__file__), "../../web/deck/data")
G = -GRAVITY  # 9.81
DT, SUB = 0.02, 8
H = DT / SUB


def load():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def run_model(model, state0, steps, metal=False):
    f = mx.compile(model.step_metal if (metal and model.J > 0) else model.step)
    s = mx.array(state0[None].astype(np.float32))
    a = mx.zeros((1, max(model.J, 1)))[:, :model.J] if model.J else mx.zeros((1, 0))
    out = [np.array(s[0])]
    for _ in range(steps):
        s = f(s, a)
        mx.eval(s)
        out.append(np.array(s[0]))
    return np.array(out)


def period_from_zero_crossings(t, phi):
    # upward zero crossings (interpolated), mean spacing
    idx = np.nonzero((phi[:-1] < 0) & (phi[1:] >= 0))[0]
    tc = [t[i] - phi[i] * (t[i + 1] - t[i]) / (phi[i + 1] - phi[i]) for i in idx]
    return float(np.mean(np.diff(tc))) if len(tc) >= 2 else None


# ------------------------------------------------------------------ a) pendulum
def pendulum():
    L, r, rho = 1.0, 0.05, 1000.0
    piv = np.array([0.0, 2.0])
    phi0 = math.radians(60)
    tip = piv + L * np.array([math.sin(phi0), -math.cos(phi0)])
    base = Capsule((-0.5, 0.1), (0.5, 0.1), 0.1, density=1e7, name="base")  # ~3e5 kg, rests on ground
    pend = Capsule(piv, tip, r, density=rho, name="pend")
    model = Model([base, pend], [Joint(0, 1, piv, -100, 100, gear=0.0)], dt=DT, substeps=SUB)
    m, I = model.masses[1], 1.0 / model.inv_i[1]
    T_sim = 5.0
    steps = int(round(T_sim / DT))
    S = run_model(model, model.init_state, steps, metal=True)
    B = 2
    px, py, th, vx, vy, w = (S[:, q * B + 1] for q in range(6))
    bpx, bpy = S[:, 0], S[:, B]
    t = np.arange(steps + 1) * DT
    # pivot = base pos + local anchor (base barely moves; measure relative to actual pivot point on base)
    phi_ours = np.arctan2(px - (bpx + model.la[0][0]), -(py - (bpy + model.la[0][1])))
    E_ours = 0.5 * m * (vx ** 2 + vy ** 2) + 0.5 * I * w ** 2 + m * G * py
    base_drift = float(np.hypot(bpx[-1] - bpx[0], bpy[-1] - bpy[0]))
    jgap = np.hypot(px - L / 2 * np.cos(th) - (bpx + model.la[0][0]), py - L / 2 * np.sin(th) - (bpy + model.la[0][1]))

    # MuJoCo: hinge at pivot, explicit inertial with same m, I about COM. x-z plane, gravity -z.
    xml = f"""<mujoco><option timestep="{H}" gravity="0 0 -{G}" integrator="Euler"/>
    <worldbody><body pos="0 0 {piv[1]}"><joint name="h" type="hinge" axis="0 -1 0" damping="0" armature="0"/>
    <inertial pos="0 0 {-L/2}" mass="{m}" diaginertia="{I} {I} 1e-6"/>
    <geom type="capsule" fromto="0 0 0 0 0 {-L}" size="{r}" contype="0" conaffinity="0" mass="0"/>
    </body></worldbody></mujoco>"""
    res = {}
    for integ in ("Euler", "RK4"):
        mm = mujoco.MjModel.from_xml_string(xml.replace('integrator="Euler"', f'integrator="{integ}"'))
        dd = mujoco.MjData(mm)
        dd.qpos[0] = phi0
        mujoco.mj_forward(mm, dd)
        phis, Es = [], []
        sub = SUB
        for k in range(steps + 1):
            com = dd.xipos[1]
            phis.append(math.atan2(com[0], -(com[2] - piv[1])))
            v = dd.qvel[0]
            Es.append(0.5 * (I + m * (L / 2) ** 2) * v * v + m * G * com[2])
            if k < steps:
                for _ in range(sub):
                    mujoco.mj_step(mm, dd)
        res[integ] = (np.array(phis), np.array(Es))
    phi_mj, E_mj = res["Euler"]
    phi_rk, E_rk = res["RK4"]
    # small-amplitude-corrected analytic period for a compound pendulum at 60 deg
    Ip = I + m * (L / 2) ** 2
    k = math.sin(phi0 / 2)
    from math import pi
    # complete elliptic integral K(k) via AGM
    a_, b_ = 1.0, math.sqrt(1 - k * k)
    for _ in range(30):
        a_, b_ = (a_ + b_) / 2, math.sqrt(a_ * b_)
    T_exact = 4 * math.sqrt(Ip / (m * G * L / 2)) * (pi / (2 * a_))
    P_ours, P_mj, P_rk = (period_from_zero_crossings(t, p) for p in (phi_ours, phi_mj, phi_rk))
    drift = lambda E: float((E[-1] - E[0]) / abs(E[0] - m * G * (piv[1] - L / 2)) * 100) if True else 0
    # energy drift relative to the swing's mechanical energy (E0 - E at bottom rest)
    Eswing = m * G * (L / 2) * (1 - math.cos(phi0))
    out = {
        "setup": f"capsule L={L} m r={r} m, m={m:.4f} kg, I_com={I:.5f} kg m^2, released at 60 deg, no damping, "
                 f"dt={DT} s x {SUB} substeps (h={H} s) for ours; MuJoCo timestep h={H} s; 5 s",
        "ours_pin": "hinge Joint to a 1e7 kg/m^3 base capsule resting on the ground (no world-fixed joint in physics2d)",
        "analytic_period_s": T_exact,
        "period_s": {"ours": P_ours, "mujoco_euler": P_mj, "mujoco_rk4": P_rk},
        "period_err_pct_vs_analytic": {kk: (v - T_exact) / T_exact * 100 for kk, v in
                                       (("ours", P_ours), ("mujoco_euler", P_mj), ("mujoco_rk4", P_rk))},
        "rms_angle_err_deg_vs_mujoco_rk4": {
            "ours": float(np.degrees(np.sqrt(np.mean((phi_ours - phi_rk) ** 2)))),
            "mujoco_euler": float(np.degrees(np.sqrt(np.mean((phi_mj - phi_rk) ** 2))))},
        "rms_angle_err_deg_ours_vs_mujoco_euler": float(np.degrees(np.sqrt(np.mean((phi_ours - phi_mj) ** 2)))),
        "energy_drift_pct_of_swing_energy_5s": {
            "ours": float((E_ours[-1] - E_ours[0]) / Eswing * 100),
            "mujoco_euler": float((E_mj[-1] - E_mj[0]) / Eswing * 100),
            "mujoco_rk4": float((E_rk[-1] - E_rk[0]) / Eswing * 100)},
        "max_abs_energy_dev_pct_of_swing_energy": {
            "ours": float(np.max(np.abs(E_ours - E_ours[0])) / Eswing * 100),
            "mujoco_euler": float(np.max(np.abs(E_mj - E_mj[0])) / Eswing * 100),
            "mujoco_rk4": float(np.max(np.abs(E_rk - E_rk[0])) / Eswing * 100)},
        "amplitude_last_second_deg": {
            "ours": float(np.degrees(np.max(np.abs(phi_ours[-50:])))),
            "mujoco_euler": float(np.degrees(np.max(np.abs(phi_mj[-50:])))),
            "mujoco_rk4": float(np.degrees(np.max(np.abs(phi_rk[-50:]))))},
        "ours_base_drift_m": base_drift,
        "ours_max_joint_gap_m": float(np.max(jgap)),
        "t": t[::2].round(3).tolist(),
        "phi_deg": {"ours": np.degrees(phi_ours[::2]).round(3).tolist(),
                    "mujoco_euler": np.degrees(phi_mj[::2]).round(3).tolist(),
                    "mujoco_rk4": np.degrees(phi_rk[::2]).round(3).tolist()},
    }
    return out


# ------------------------------------------------------------------ b) friction slide
def slide():
    mu, v0, r, L = 0.5, 2.0, 0.1, 0.4
    analytic = v0 ** 2 / (2 * mu * G)
    cap = Capsule((-L / 2, r), (L / 2, r), r, name="block")
    model = Model([cap], [], dt=DT, substeps=SUB, mu=mu)
    s0 = model.init_state.copy()
    s0[3] = v0  # vx of body 0
    S = run_model(model, s0, 150)
    x = S[:, 0]
    vx = S[:, 3]
    stop_i = int(np.argmax(np.abs(vx) < 1e-3)) if np.any(np.abs(vx) < 1e-3) else None
    ours = float(x[-1] - x[0])
    # MuJoCo: free capsule on a plane, friction 0.5 on both geoms (MuJoCo takes the max)
    res = {}
    for cone in ("pyramidal", "elliptic"):
        xml = f"""<mujoco><option timestep="{H}" gravity="0 0 -{G}" cone="{cone}"/>
        <worldbody><geom type="plane" size="10 10 0.1" friction="{mu} 0 0"/>
        <body pos="0 0 {r}"><freejoint/><geom type="capsule" fromto="{-L/2} 0 0 {L/2} 0 0" size="{r}" friction="{mu} 0 0"/></body>
        </worldbody></mujoco>"""
        mm = mujoco.MjModel.from_xml_string(xml)
        dd = mujoco.MjData(mm)
        # settle 0.5 s first so contact is established
        for _ in range(int(0.5 / H)):
            mujoco.mj_step(mm, dd)
        x0 = float(dd.qpos[0])
        dd.qvel[0] = v0
        for _ in range(int(3.0 / H)):
            mujoco.mj_step(mm, dd)
        res[cone] = float(dd.qpos[0] - x0)
    return {"setup": f"capsule L={L} m r={r} m lying on ground, v0={v0} m/s, mu={mu}",
            "analytic_m": analytic, "ours_m": ours, "mujoco_pyramidal_m": res["pyramidal"],
            "mujoco_elliptic_m": res["elliptic"],
            "err_pct_vs_analytic": {"ours": (ours - analytic) / analytic * 100,
                                    "mujoco_pyramidal": (res["pyramidal"] - analytic) / analytic * 100,
                                    "mujoco_elliptic": (res["elliptic"] - analytic) / analytic * 100},
            "ours_stop_time_s": (stop_i * DT if stop_i else None), "analytic_stop_time_s": v0 / (mu * G),
            "ours_x_traj": (x[:60] - x[0]).round(4).tolist()}


# ------------------------------------------------------------------ c) drop test
def drop():
    r, L, h0 = 0.1, 0.4, 1.0
    cap = Capsule((-L / 2, h0), (L / 2, h0), r, name="block")
    model = Model([cap], [], dt=DT, substeps=SUB)
    S = run_model(model, model.init_state, 100)
    y = S[:, 1]
    imp = int(np.argmax(y < r + 0.01))
    bounce = float(np.max(y[imp:]) - r)
    xml = f"""<mujoco><option timestep="{H}" gravity="0 0 -{G}"/>
    <worldbody><geom type="plane" size="10 10 0.1"/>
    <body pos="0 0 {h0}"><freejoint/><geom type="capsule" fromto="{-L/2} 0 0 {L/2} 0 0" size="{r}"/></body>
    </worldbody></mujoco>"""
    mm = mujoco.MjModel.from_xml_string(xml)
    dd = mujoco.MjData(mm)
    zs = []
    for k in range(101):
        zs.append(float(dd.qpos[2]))
        for _ in range(SUB):
            mujoco.mj_step(mm, dd)
    zs = np.array(zs)
    impm = int(np.argmax(zs < r + 0.01))
    t_an = math.sqrt(2 * (h0 - r) / G)
    return {"setup": f"capsule r={r} dropped from centre height {h0} m (restitution 0 in ours)",
            "analytic_impact_time_s": t_an,
            "ours": {"impact_time_s": imp * DT, "rest_height_m": float(y[-1]), "penetration_mm": float((r - y[-1]) * 1000),
                     "max_bounce_after_impact_mm": bounce * 1000},
            "mujoco": {"impact_time_s": impm * DT, "rest_height_m": float(zs[-1]), "penetration_mm": float((r - zs[-1]) * 1000),
                       "max_bounce_after_impact_mm": float((np.max(zs[impm:]) - r) * 1000)}}


if __name__ == "__main__":
    t0 = time.time()
    res = {"uptime_before": load(), "mujoco_version": mujoco.__version__}
    res["pendulum"] = pendulum()
    res["slide"] = slide()
    res["drop"] = drop()
    res["uptime_after"] = load()
    res["wall_s"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "validation.json"), "w"), indent=1)
    p = res["pendulum"]
    print(json.dumps({k: v for k, v in p.items() if k not in ("t", "phi_deg")}, indent=1))
    print(json.dumps({k: v for k, v in res["slide"].items() if k != "ours_x_traj"}, indent=1))
    print(json.dumps(res["drop"], indent=1))
