"""Paper/ink plots for the deck from web/deck/data/*.json. usage: plots.py [validation|profile|seeds|fair]..."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "../../web/deck/data")
BG, INK, ACC, GREY = "#FAFAF7", "#111111", "#FF4D00", "#9A9A94"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
J = lambda n: json.load(open(os.path.join(D, n)))


def validation():
    v = J("validation.json")
    p = v["pendulum"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 2.8), gridspec_kw={"width_ratios": [2.2, 1]})
    t = np.array(p["t"])
    a.plot(t, p["phi_deg"]["mujoco_rk4"], color=INK, lw=2.6, label="MuJoCo (RK4)")
    a.plot(t, p["phi_deg"]["ours"], color=ACC, lw=1.2, label="MacGym physics2d")
    a.set_xlabel("time (s)"); a.set_ylabel("pendulum angle (deg)")
    a.set_title(f"Pendulum from 60°: RMS diff {p['rms_angle_err_deg_vs_mujoco_rk4']['ours']:.2f}°, "
                f"period err {p['period_err_pct_vs_analytic']['ours']:.3f}%", fontsize=9, loc="left")
    a.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.06), ncol=2)
    s = v["slide"]
    names = ["analytic", "MacGym", "MuJoCo"]
    vals = [s["analytic_m"], s["ours_m"], s["mujoco_pyramidal_m"]]
    b.bar(names, [x * 100 for x in vals], color=[GREY, ACC, INK], width=0.6)
    for i, x in enumerate(vals):
        b.text(i, x * 100 + 0.6, f"{x*100:.1f}", ha="center", fontsize=8)
    b.set_ylabel("stopping distance (cm)"); b.set_ylim(0, max(vals) * 100 * 1.2)
    b.set_title("Slide, v0=2 m/s, μ=0.5", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(D, "validation.png"), dpi=160)


def profile():
    p = J("profile.json")
    pieces = p["pieces_s"]
    order = [("env_step_physics_reward_reset", "physics + reward + reset", ACC),
             ("policy_forward_rollout", "policy forward (rollout)", INK),
             ("rollout_buffer_writes", "buffer writes", "#5A5A55"),
             ("reward_norm_gae_advnorm", "GAE + norm", GREY),
             ("ppo_update_backprop_adam", "PPO update (backprop+Adam)", "#3A3A36"),
             ("stats_sync_obsnorm_host_copy", "stats / sync", "#C8C8C0")]
    fig, ax = plt.subplots(figsize=(8, 1.8))
    left = 0
    tot = sum(pieces.values())
    for k, lab, c in order:
        w = pieces[k] * 1000
        ax.barh([0], [w], left=left, color=c, height=0.5, edgecolor=BG, lw=1)
        pct = pieces[k] / tot * 100
        if pct > 6:
            ax.text(left + w / 2, 0, f"{pct:.0f}%", ha="center", va="center", color=BG if c not in (GREY, "#C8C8C0") else INK, fontsize=8)
        left += w
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in order]
    ax.legend(handles, [f"{l} {pieces[k]*1000:.0f} ms" for k, l, _ in order], frameon=False, fontsize=7,
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_title("One PPO iteration, walker, 4096 envs x 24 steps (ms)", fontsize=9, loc="left")
    ax.set_xlim(0, left)
    fig.tight_layout(); fig.savefig(os.path.join(D, "profile.png"), dpi=160, bbox_inches="tight", pad_inches=0.15)


def seeds():
    s = J("seeds.json")
    grid = np.linspace(0, 75, 151)
    ys = []
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for r in s["runs"]:
        t = [0.0] + [c["t"] for c in r["curve"]]
        d = [0.0] + [c["dist"] or 0.0 for c in r["curve"]]
        ax.plot(t, d, color=INK, lw=0.6, alpha=0.45)
        ys.append(np.interp(grid, t, d, right=np.nan))
    Y = np.array(ys)
    ax.fill_between(grid, np.nanmin(Y, 0), np.nanmax(Y, 0), color=ACC, alpha=0.15, lw=0)
    ax.plot(grid, np.nanmean(Y, 0), color=ACC, lw=2.2, label="mean of %d seeds" % len(Y))
    ax.axvline(60, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("wall-clock training time (s)"); ax.set_ylabel("avg episode distance (m)")
    ax.set_title(f"Walker from scratch, 4096 envs: {s['n_learned']}/{s['n_seeds']} seeds > 10 m by 60 s", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(0, 75)
    fig.tight_layout(); fig.savefig(os.path.join(D, "seeds.png"), dpi=160)




def fair():
    f = J("fair_bench.json")
    fig, ax = plt.subplots(figsize=(6.5, 3))
    sty = {"GPU Metal kernel": (ACC, 2.2, "GPU, generated Metal kernel"),
           "GPU same solver": (INK, 1.4, "GPU, same solver as mx ops"),
           "CPU same solver": (GREY, 1.4, "CPU, same solver as mx ops")}
    for key, (c, lw, lab) in sty.items():
        rows = [r for r in f["rows"] if r["backend"].startswith(key) and "env_steps_per_s" in r]
        ax.plot([r["num_envs"] for r in rows], [r["env_steps_per_s"] for r in rows], marker="o", ms=3, color=c, lw=lw, label=lab)
    for r in f["rows"]:
        if r["backend"].startswith("MuJoCo"):
            ax.axhline(r["env_steps_per_s"], color=INK, lw=0.6, ls=":")
            ax.text(1, r["env_steps_per_s"] * 1.12, r["backend"], fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("parallel envs"); ax.set_ylabel("env-steps / s (physics only)")
    ax.set_title("Walker physics, same algorithm on CPU vs GPU (" + f["machine"] + ")", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(D, "fair_bench.png"), dpi=160)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        globals()[n]()
        print("ok", n)
