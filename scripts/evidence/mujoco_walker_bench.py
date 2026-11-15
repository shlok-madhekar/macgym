"""Same-robot speed: MacGym walker (Metal kernel, bench.py timing method) vs MuJoCo MJCF mirror
(scripts/evidence/mj_walker.py, timestep 0.0025 x frame_skip 8 = one 0.02 s control step).
MuJoCo: random ctrl (pre-generated), mj_step loop, reset on fall (torso z<0.875 or |angle|>1) or
1000 steps; 1 process, then 11 processes (spawn pool, warm). Median of 3 timing windows each."""
import json, multiprocessing as mp, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mj_walker import ROOT, mjcf, FRAME_SKIP
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np

WIN = 3.0


def mj_worker(args):
    seconds, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import mujoco
    m = mujoco.MjModel.from_xml_string(mjcf()); d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    acts = rng.uniform(-1, 1, size=(4096, m.nu))
    mujoco.mj_resetData(m, d)
    state = {"t": 0}

    def run(n):
        t = state["t"]
        for i in range(n):
            d.ctrl[:] = acts[i & 4095]
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(m, d)
            t += 1
            if d.qpos[1] < -0.375 or abs(d.qpos[2]) > 1.0 or t >= 1000:
                mujoco.mj_resetData(m, d); t = 0
        state["t"] = t

    run(300)
    steps, t0 = 0, time.perf_counter()
    while True:
        run(500); steps += 500
        el = time.perf_counter() - t0
        if el >= seconds:
            return steps, el


def main():
    import bench
    up0 = subprocess.check_output(["uptime"], text=True).strip()
    print(up0, flush=True)
    res = {}
    # ---- ours
    for n in (4096, 16384):
        rate, src = bench.bench_mlx("walker", n, seconds=10.0)
        res[f"macgym_metal_N{n}"] = rate
        print("ours", n, f"{rate:,.0f}", src, flush=True)
    up1 = subprocess.check_output(["uptime"], text=True).strip()
    # ---- MuJoCo single
    singles = []
    for i in range(3):
        s, el = mj_worker((WIN, i)); singles.append(s / el)
    res["mujoco_1proc_windows"] = singles
    res["mujoco_1proc"] = float(np.median(singles))
    print("mj single", singles, flush=True)
    # ---- MuJoCo all cores
    procs = 11
    ctx = mp.get_context("spawn")
    multis = []
    with ctx.Pool(procs) as pool:
        pool.map(mj_worker, [(0.3, i) for i in range(procs)])
        for r in range(3):
            t0 = time.perf_counter()
            out = pool.map(mj_worker, [(WIN, 100 * r + i) for i in range(procs)])
            wall = time.perf_counter() - t0
            multis.append(sum(s for s, _ in out) / wall)
    res["mujoco_11proc_windows"] = multis
    res["mujoco_11proc"] = float(np.median(multis))
    print("mj multi", multis, flush=True)
    up2 = subprocess.check_output(["uptime"], text=True).strip()
    ours_best = max(res["macgym_metal_N4096"], res["macgym_metal_N16384"])
    import mujoco
    out = {
        "what": "env-steps/s (one env-step = one 0.02 s control step incl. 8 physics substeps) on the SAME walker robot",
        "machine": bench.machine_info(), "mujoco_version": mujoco.__version__,
        "macgym": {"N4096": res["macgym_metal_N4096"], "N16384": res["macgym_metal_N16384"],
                   "method": "scripts/bench.py bench_mlx: random actions generated on GPU each step, full mx.eval of obs/reward/done every step, auto-reset, 20 warmup steps, median of 5 timing windows (10 s total) per N",
                   "env": "macgym.envs.make('walker') (Metal kernel, 7 capsules / 6 motors, dt 0.02, 8 substeps, 4 iters)"},
        "mujoco": {"1proc": res["mujoco_1proc"], "1proc_windows": singles,
                   "11proc": res["mujoco_11proc"], "11proc_windows": multis,
                   "method": f"MJCF mirror (scripts/evidence/mj_walker.py), timestep 0.0025 x frame_skip 8, Euler, random ctrl pre-generated, reset on fall/1000 steps, 300 warmup steps; median of 3 windows of {WIN} s; 11 procs = spawn Pool warm, aggregate steps / wall time",
                   "note": "MuJoCo loop is the pure-Python mj_step loop as in Gymnasium (no obs/reward computation counted against it)"},
        "speedup_ours_best_vs_mujoco_allcores": ours_best / res["mujoco_11proc"],
        "speedup_ours_best_vs_mujoco_1proc": ours_best / res["mujoco_1proc"],
        "ours_best": ours_best,
        "uptime_start": up0, "uptime_after_mlx": up1, "uptime_end": up2,
        "load_note": "machine not idle (desktop + other apps running); load averages recorded above",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    json.dump(out, open(os.path.join(ROOT, "web/deck/data/mujoco_walker_bench.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k.startswith("speedup") or k.startswith("uptime")}, indent=1))


if __name__ == "__main__":
    main()
