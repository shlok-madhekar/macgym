"""(4) multi-seed walker learning curves and (5) tripod deterministic replays.
usage: seeds_tripod.py seeds | tripod"""
import json, os, subprocess, sys, time
import numpy as np
import mlx.core as mx
import macgym as mg

OUT = os.path.join(os.path.dirname(__file__), "../../web/deck/data")
up = lambda: subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
N = 4096


def seeds(wall=75.0, n_seeds=5):
    runs = []
    for s in range(n_seeds):
        env = mg.make(mg.robots.walker(), num_envs=N, seed=s)
        curve = []
        u0 = up()

        def cb(m):
            curve.append({"t": round(m["elapsed"], 3), "dist": m["dist"], "episode_dist": m.get("mean_episode_dist"),
                          "live_dist": m["live_dist"], "env_steps": m["env_steps"], "ep_seconds": m["ep_seconds"],
                          "return": m["mean_ep_return"]})
            return m["elapsed"] < wall

        mg.train(env, iters=100000, seed=s, on_iter=cb, verbose=False)
        d60 = [c["dist"] for c in curve if c["t"] <= 60.0]
        at60 = d60[-1] if d60 else None
        runs.append({"seed": s, "uptime_start": u0, "uptime_end": up(), "curve": curve,
                     "dist_at_60s": at60, "max_dist_by_60s": max(d60) if d60 else None,
                     "final_dist": curve[-1]["dist"], "final_t": curve[-1]["t"], "iters": len(curve),
                     "env_steps": curve[-1]["env_steps"], "learned_10m_by_60s": bool(at60 is not None and at60 > 10)})
        print(f"seed {s}: dist@60s {at60:.2f} final {curve[-1]['dist']:.2f} at {curve[-1]['t']:.1f}s, "
              f"{len(curve)} iters, {curve[-1]['env_steps']/1e6:.1f}M steps", flush=True)
        del env
    out = {"setup": f"mg.train(mg.robots.walker(), N={N}) defaults (PPO 256x256, T=24), {wall:.0f} s wall per seed "
                    "from train() start (includes first-iteration compile). dist = mean start-relative torso x of "
                    "episodes finished in the last 100-episode window (live mean before any finish).",
           "learned_criterion": "dist > 10 m at the last iteration with t <= 60 s",
           "n_learned": sum(r["learned_10m_by_60s"] for r in runs), "n_seeds": n_seeds, "runs": runs}
    json.dump(out, open(os.path.join(OUT, "seeds.json"), "w"), indent=1)


def tripod(marks=(0.0, 15.0, 30.0, 60.0)):
    from macgym.server import run_replay
    ns = {}
    src = open(os.path.join(os.path.dirname(mg.__file__), "presets", "tripod.py")).read()
    exec(compile(src, "tripod.py", "exec"), ns)
    robot, task = ns["robot"], ns["task"]
    env = mg.make(robot, task, num_envs=N, seed=0)
    renv = mg.make(robot, task, num_envs=1, seed=0)
    from macgym.rl.ppo import PPO
    ppo = PPO(env, hidden=(256, 256), seed=0)
    gens, u0 = [], up()
    train_s = 0.0
    nxt = 0

    def rec(label):
        t0 = time.perf_counter()
        r = run_replay("playground", renv, ppo)
        r.update({"label": label, "train_seconds": round(train_s, 2), "iter": ppo.iter,
                  "env_steps": ppo.env_steps, "replay_cost_s": round(time.perf_counter() - t0, 3)})
        gens.append(r)
        print(label, r["result_text"], "iter", ppo.iter, flush=True)

    rec("0:00"); nxt = 1
    while nxt < len(marks):
        t0 = time.perf_counter()
        ppo.train_iter()
        train_s += time.perf_counter() - t0
        if train_s >= marks[nxt]:
            m = int(marks[nxt])
            rec(f"{m // 60}:{m % 60:02d}")
            nxt += 1
    out = {"robot": "tripod", "source": "macgym/presets/tripod.py", "dt": gens[0]["dt"], "K": gens[0]["K"],
           "colors": gens[0]["colors"], "uptime_start": u0, "uptime_end": up(),
           "note": "deterministic policy (mean action) on a separate 1-env copy, same start state each replay "
                   "(key 1234), frames every 2 control steps, up to 400 steps (8 s), x start-relative. "
                   "train_seconds excludes replay time.",
           "generations": [{k: g[k] for k in ("label", "train_seconds", "iter", "env_steps", "result_text", "result",
                                               "fell", "steps", "frames")} for g in gens]}
    json.dump(out, open(os.path.join(OUT, "tripod_replay.json"), "w"))


if __name__ == "__main__":
    {"seeds": seeds, "tripod": tripod}[sys.argv[1]]()
