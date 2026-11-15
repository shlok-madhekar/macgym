"""Same solver, CPU vs GPU: walker physics (macgym mg.robots.walker model).
CPU = Model.step (pure mx ops) under mx.compile with default device mx.cpu.
GPU = Model.step_metal (generated Metal kernel) and Model.step (mx ops) on mx.gpu.
Writes web/deck/data/fair_bench.json."""
import json, os, subprocess, sys, time, multiprocessing as mp
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import macgym as mg

OUT = os.path.join(os.path.dirname(__file__), "../../web/deck/data")
up = lambda: subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
NS = [1, 64, 1024, 4096, 16384]
BUDGET = 1.5


def bench(model, device, fn_name, n):
    mx.set_default_device(device)
    fn = mx.compile(getattr(model, fn_name))
    s0 = np.broadcast_to(model.init_state[None], (n, 6 * model.B)).copy()
    s = mx.array(s0)
    rng = np.random.default_rng(0)
    acts = [mx.array(rng.uniform(-1, 1, (n, model.J)).astype(np.float32)) for _ in range(8)]
    tc = time.perf_counter()
    s = fn(s, acts[0]); mx.eval(s)  # compile + warm
    s = fn(s, acts[1]); mx.eval(s)
    compile_s = time.perf_counter() - tc
    k = 0
    t0 = time.perf_counter()
    while True:
        for _ in range(4):
            s = fn(s, acts[k % 8]); k += 1
        mx.eval(s)
        el = time.perf_counter() - t0
        if el > BUDGET:
            break
    finite = bool(np.isfinite(np.array(s)).all())
    return {"steps": k, "seconds": el, "env_steps_per_s": k * n / el, "warmup_s": compile_s, "finite": finite}


if __name__ == "__main__":
    robot, task = mg.robots.walker()
    model = robot.build_model()
    rows = []
    configs = [("GPU Metal kernel (step_metal)", mx.gpu, "step_metal"),
               ("CPU same solver, mx ops (Model.step on mx.cpu)", mx.cpu, "step"),
               ("GPU same solver, mx ops (Model.step on mx.gpu)", mx.gpu, "step")]
    for label, dev, fn in configs:
        for n in NS:
            u = up()
            try:
                r = bench(model, dev, fn, n)
            except Exception as e:
                r = {"error": repr(e)}
            r.update({"backend": label, "num_envs": n, "uptime": u})
            rows.append(r)
            print(label, n, {k: r.get(k) for k in ("env_steps_per_s", "warmup_s", "finite", "error")}, flush=True)
    mx.set_default_device(mx.gpu)
    # MuJoCo cheetah reference, measured now (scripts/bench.py harness, includes auto-reset)
    from bench import bench_mujoco_single, bench_mujoco_multi
    procs = os.cpu_count()
    u = up()
    s1 = bench_mujoco_single("cheetah", 3.0)
    rows.append({"backend": "MuJoCo cheetah, 1 thread", "num_envs": 1, "env_steps_per_s": s1, "uptime": u})
    print("mujoco 1", s1, flush=True)
    u = up()
    sm = bench_mujoco_multi("cheetah", 3.0, procs)
    rows.append({"backend": f"MuJoCo cheetah, {procs} processes", "num_envs": procs, "env_steps_per_s": sm, "uptime": u})
    print("mujoco multi", sm, flush=True)
    out = {"what": "env-steps/s of one control step (dt=0.02 s, 8 substeps x 4 solver iters) of the walker model, "
                   "physics only (no policy). Same algorithm on CPU and GPU. MuJoCo cheetah via scripts/bench.py "
                   "(frame_skip 5, includes resets).",
           "machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
           "cpu_count": procs, "rows": rows}
    json.dump(out, open(os.path.join(OUT, "fair_bench.json"), "w"), indent=1)
