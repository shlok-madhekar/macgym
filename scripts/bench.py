#!/usr/bin/env python3
"""MacGym honest benchmark: MLX GPU batched envs vs CPU MuJoCo.

Run from repo root:
    python3 scripts/bench.py            # full run, writes web/bench.json
    python3 scripts/bench.py --quick    # shorter timing windows

Definitions (kept identical across backends so numbers are comparable):
  * one "env-step" = one control step of one environment (the unit an RL
    algorithm consumes), including any physics substeps / frame-skip.
  * MLX: random actions in [-1, 1] are generated on the GPU each step and the
    env is fully evaluated (mx.eval on obs/reward/done) every step, so this is
    what a PPO rollout actually sees. Auto-reset is included.
  * MuJoCo: Gymnasium conventions, InvertedPendulum-like cartpole
    (timestep 0.02, frame_skip 2) and a planar HalfCheetah-like model
    (timestep 0.01, frame_skip 5). Random ctrl each control step, mj_step loop,
    reset on fall / every 1000 steps. Single process, then one process per core.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

# --------------------------------------------------------------------------
# Machine info
# --------------------------------------------------------------------------

def _sysctl(key):
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return ""


def machine_info():
    chip = _sysctl("machdep.cpu.brand_string") or platform.processor()
    mem = _sysctl("hw.memsize")
    mem_gb = f"{int(mem) / 2**30:.0f} GB" if mem.isdigit() else "?"
    p = _sysctl("hw.perflevel0.physicalcpu")
    e = _sysctl("hw.perflevel1.physicalcpu")
    cores = f"{os.cpu_count()} CPU cores" + (f" ({p}P+{e}E)" if p and e else "")
    return f"{chip}, {mem_gb} unified memory, {cores}, macOS {platform.mac_ver()[0]}"


# --------------------------------------------------------------------------
# MLX: self-contained batched cartpole (fallback when macgym.envs.make is absent)
# --------------------------------------------------------------------------

def _fallback_cartpole(num_envs):
    import mlx.core as mx

    G, MC, MP, L, FMAX, DT, MAXT = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02, 500

    def phys(s, t, a, key):
        x, xd, th, thd = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        f = mx.clip(a[:, 0], -1.0, 1.0) * FMAX
        ct, st = mx.cos(th), mx.sin(th)
        tmp = (f + MP * L * thd * thd * st) / (MC + MP)
        thacc = (G * st - ct * tmp) / (L * (4.0 / 3.0 - MP * ct * ct / (MC + MP)))
        xacc = tmp - MP * L * thacc * ct / (MC + MP)
        xd = xd + DT * xacc
        x = x + DT * xd
        thd = thd + DT * thacc
        th = th + DT * thd
        s = mx.stack([x, xd, th, thd], axis=1)
        t = t + 1
        done = (mx.abs(x) > 2.4) | (mx.abs(th) > 0.21) | (t >= MAXT)
        fresh = mx.random.uniform(-0.05, 0.05, s.shape, key=key)
        s = mx.where(done[:, None], fresh, s)
        t = mx.where(done, 0, t)
        return s, t, mx.ones_like(t), done.astype(mx.float32)

    class _CP:
        name = "cartpole"
        act_dim = 1

        def __init__(self, n):
            self.num_envs = n
            self.key = mx.random.key(0)
            self.fn = mx.compile(phys)

        def reset(self):
            self.s = mx.random.uniform(-0.05, 0.05, (self.num_envs, 4))
            self.t = mx.zeros((self.num_envs,))
            return self.s

        def step(self, a):
            self.key, k = mx.random.split(self.key)
            self.s, self.t, r, d = self.fn(self.s, self.t, a, k)
            return self.s, r, d, {}

    return _CP(num_envs)


def _get_make():
    try:
        from macgym.envs import make  # noqa
        return make
    except Exception:
        return None


def _framework_env(robot_name, num_envs):
    import macgym as mg
    return mg.make(getattr(mg.robots, robot_name)(), num_envs=num_envs)


def bench_mlx(task, num_envs, seconds, framework=False):
    import mlx.core as mx

    make = _get_make()
    source = "macgym.envs"
    env = None
    if framework:
        env = _framework_env(task, num_envs)
        source = f"mg.make(mg.robots.{task}())"
    elif make is not None:
        try:
            env = make(task, num_envs)
        except Exception as ex:  # env not written yet
            env = None
            err = repr(ex)
    if env is None:
        if task != "cartpole":
            return None, None
        env = _fallback_cartpole(num_envs)
        source = "bench fallback"

    obs = env.reset()
    mx.eval(obs)
    key = mx.random.key(1)
    act_dim = env.act_dim

    def one_step(k):
        a = mx.random.uniform(-1.0, 1.0, (num_envs, act_dim), key=k)
        obs, rew, done, _ = env.step(a)
        mx.eval(obs, rew, done)

    # warmup (compile + allocator)
    for _ in range(20):
        key, k = mx.random.split(key)
        one_step(k)

    # several timing windows, report the median (GPU is shared with the desktop, so single windows are noisy)
    rates = []
    for _ in range(5):
        steps = 0
        t0 = time.perf_counter()
        while True:
            for _ in range(10):
                key, k = mx.random.split(key)
                one_step(k)
            steps += 10
            el = time.perf_counter() - t0
            if el >= seconds / 5:
                break
        rates.append(steps * num_envs / el)
    rates.sort()
    return rates[len(rates) // 2], source


def bench_train(robot_name, num_envs, iters):
    """End-to-end PPO throughput: env-steps consumed per wall-clock second including rollout,
    GAE and gradient updates. First iteration (compile/warmup) is excluded."""
    import macgym as mg
    env = _framework_env(robot_name, num_envs)
    res = mg.train(env, iters=iters, verbose=False)
    h = res.history
    return (h[-1]["env_steps"] - h[0]["env_steps"]) / (h[-1]["elapsed"] - h[0]["elapsed"])


# --------------------------------------------------------------------------
# CPU MuJoCo
# --------------------------------------------------------------------------

CARTPOLE_XML = """
<mujoco model="cartpole">
  <compiler angle="degree" inertiafromgeom="true"/>
  <default>
    <joint damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option integrator="RK4" timestep="0.02"/>
  <worldbody>
    <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule"/>
    <body name="cart" pos="0 0 0">
      <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
      <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
      <body name="pole" pos="0 0 0">
        <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-90 90" type="hinge"/>
        <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
  </actuator>
</mujoco>
"""

CHEETAH_XML = """
<mujoco model="cheetah">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <size nstack="300000" nuser_geom="1"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 .7">
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="-.5 0 0 .5 0 0" name="torso" size="0.046" type="capsule"/>
      <geom axisangle="0 1 0 .87" name="head" pos=".6 0 .1" size="0.046 .15" type="capsule"/>
      <body name="bthigh" pos="-.5 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom axisangle="0 1 0 -3.8" name="bthigh" pos=".1 0 -.13" size="0.046 .145" type="capsule"/>
        <body name="bshin" pos=".16 0 -.25">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom axisangle="0 1 0 -2.03" name="bshin" pos="-.14 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .15" type="capsule"/>
          <body name="bfoot" pos="-.28 0 -.14">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom axisangle="0 1 0 -.27" name="bfoot" pos=".03 0 -.097" rgba="0.9 0.6 0.6 1" size="0.046 .094" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos=".5 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="fthigh" pos="-.07 0 -.12" size="0.046 .133" type="capsule"/>
        <body name="fshin" pos="-.14 0 -.24">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="fshin" pos=".065 0 -.09" rgba="0.9 0.6 0.6 1" size="0.046 .106" type="capsule"/>
          <body name="ffoot" pos=".13 0 -.18">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="ffoot" pos=".045 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .07" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco>
"""

MJ_TASKS = {
    # task: (xml, frame_skip, max_episode_steps)
    "cartpole": (CARTPOLE_XML, 2, 1000),
    "cheetah": (CHEETAH_XML, 5, 1000),
}


def _mj_worker(args):
    task, seconds, seed = args
    import mujoco

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    xml, frame_skip, max_steps = MJ_TASKS[task]
    model = mujoco.MjModel.from_xml_string(xml)
    if task == "cheetah":
        mujoco.mj_setTotalmass(model, 14.0)  # Gymnasium HalfCheetah total mass
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    nu = model.nu
    # pre-generate random actions so RNG cost is not counted against MuJoCo
    acts = rng.uniform(lo, hi, size=(4096, nu))
    mujoco.mj_resetData(model, data)

    def run(n):
        t = 0
        for i in range(n):
            data.ctrl[:] = acts[i & 4095]
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
            t += 1
            if task == "cartpole":
                if abs(data.qpos[1]) > 0.2 or t >= max_steps:
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = rng.uniform(-0.01, 0.01, model.nq)
                    t = 0
            elif t >= max_steps:
                mujoco.mj_resetData(model, data)
                t = 0

    run(200)  # warmup
    steps = 0
    t0 = time.perf_counter()
    while True:
        run(500)
        steps += 500
        el = time.perf_counter() - t0
        if el >= seconds:
            break
    return steps, el


def bench_mujoco_single(task, seconds):
    steps, el = _mj_worker((task, seconds, 0))
    return steps / el


def bench_mujoco_multi(task, seconds, procs):
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs) as pool:
        # throwaway round so process spawn + mujoco import is not timed
        pool.map(_mj_worker, [(task, 0.2, i) for i in range(procs)])
        t0 = time.perf_counter()
        res = pool.map(_mj_worker, [(task, seconds, i) for i in range(procs)])
        wall = time.perf_counter() - t0
    total = sum(s for s, _ in res)
    return total / wall


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}k"
    return f"{n:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "bench.json"))
    ap.add_argument("--sizes", default="1024,4096,16384")
    args = ap.parse_args()

    sec = 1.5 if args.quick else 4.0
    sizes = [int(s) for s in args.sizes.split(",")]
    procs = os.cpu_count() or 1
    results = []
    mlx_sources = {}

    print(f"Machine: {machine_info()}")
    print("== MLX GPU ==", flush=True)
    for task in ["cartpole", "walker", "cheetah"]:
        for n in sizes:
            sps, src = bench_mlx(task, n, sec)
            if sps is None:
                print(f"  {task:9s} N={n:6d}  (env not available yet, skipped)", flush=True)
                break
            mlx_sources[task] = src
            results.append({"task": task, "backend": "MLX GPU (Metal)", "num_envs": n,
                            "steps_per_sec": round(sps, 1)})
            print(f"  {task:9s} N={n:6d}  {fmt(sps):>8s} env-steps/s  [{src}]", flush=True)

    print("== MLX GPU, public framework path: mg.make(mg.robots.X()) ==", flush=True)
    for robot in ["walker", "tripod", "hopper"]:
        for n in sizes:
            sps, src = bench_mlx(robot, n, sec, framework=True)
            results.append({"task": robot, "backend": "MLX GPU (Metal), framework path", "num_envs": n,
                            "steps_per_sec": round(sps, 1), "path": src})
            print(f"  {robot:9s} N={n:6d}  {fmt(sps):>8s} env-steps/s  [{src}]", flush=True)

    print("== Training throughput: sim + PPO (mg.train) ==", flush=True)
    train_n = 4096
    train_iters = 6 if args.quick else 12
    tps = bench_train("walker", train_n, train_iters)
    results.append({"task": "walker", "backend": "MLX GPU (Metal), training (sim + PPO)", "num_envs": train_n,
                    "steps_per_sec": round(tps, 1),
                    "path": f"mg.train(mg.make(mg.robots.walker()), iters={train_iters})"})
    print(f"  walker    N={train_n:6d}  {fmt(tps):>8s} env-steps/s  [sim + PPO update]", flush=True)

    print("== CPU MuJoCo ==", flush=True)
    import mujoco
    for task in ["cartpole", "cheetah"]:
        s1 = bench_mujoco_single(task, sec)
        results.append({"task": task, "backend": "CPU MuJoCo 1 thread", "num_envs": 1,
                        "steps_per_sec": round(s1, 1)})
        print(f"  {task:9s} 1 thread     {fmt(s1):>8s} env-steps/s", flush=True)
        sm = bench_mujoco_multi(task, sec, procs)
        results.append({"task": task, "backend": f"CPU MuJoCo {procs} procs", "num_envs": procs,
                        "steps_per_sec": round(sm, 1)})
        print(f"  {task:9s} {procs} procs     {fmt(sm):>8s} env-steps/s", flush=True)

    # speedups: best MLX vs best CPU per task (cheetah MLX vs cheetah MuJoCo; walker vs cheetah MuJoCo as nearest planar locomotion model)
    def best(task, prefix):
        v = [r["steps_per_sec"] for r in results if r["task"] == task and r["backend"].startswith(prefix)]
        return max(v) if v else None

    speedups = {}
    for task, ref in [("cartpole", "cartpole"), ("cheetah", "cheetah"), ("walker", "cheetah")]:
        g = max([r["steps_per_sec"] for r in results
                 if r["task"] == task and r["backend"] == "MLX GPU (Metal)"] or [0]) or None
        c = best(ref, "CPU MuJoCo")
        if g and c:
            speedups[task] = round(g / c, 1)

    import mlx.core as mx
    notes = (
        "env-steps/s = control steps per second summed over all environments (physics substeps / frame-skip included). "
        "MLX sim rows: median of 5 timing windows, random actions generated on GPU, full mx.eval every step, auto-reset "
        "included, mx.compile on the physics step. Rows marked 'framework path' build the env exactly the way a user does "
        "(mg.make(mg.robots.walker()), mg.robots.tripod(), mg.robots.hopper()), so they include the generic Robot/Task "
        "layer, reward and obs functions. The 'training (sim + PPO)' row is end-to-end mg.train: rollout, GAE and gradient "
        "updates, first (compile) iteration excluded; this is the number that decides how long training actually takes. "
        "Why walker is about the same speed as cheetah at small N: at 1024 envs the GPU is not the bottleneck. Each step "
        "pays a roughly fixed cost for Python and MLX graph dispatch, so per-step time barely depends on how heavy the "
        "robot is. As N grows that fixed cost is amortized, the GPU saturates, and throughput starts to reflect the actual "
        "physics work (more bodies and contacts means fewer steps/s). "
        "MuJoCo: Gymnasium-style models written inline (InvertedPendulum-like cartpole, timestep 0.02 frame_skip 2; "
        "HalfCheetah-like model, timestep 0.01 frame_skip 5), plain Python mj_step loop with random ctrl, "
        f"then one process per core ({procs} processes, spawn, warm). "
        "CAVEAT: not an apples-to-apples physics comparison. Both sides are planar for locomotion (MuJoCo HalfCheetah is "
        "also constrained to the x-z plane). The real difference is the solver: MuJoCo integrates generalized "
        "coordinates with a convex, constraint-based contact solver, while MacGym uses maximal coordinates (each body "
        "has its own position and angle, joints held together by constraints) with a simpler impulse-based contact "
        "solver and few substeps. MuJoCo numbers also include Python loop overhead per step. Walker is compared against "
        "MuJoCo HalfCheetah as the nearest planar locomotion model. Read speedups as 'how much RL experience per second "
        "this laptop can generate for these tasks', not 'MLX is X times faster than MuJoCo at the same physics'. "
        "Numbers vary run to run (roughly +/-30% on the GPU side, and the CPU MuJoCo rows drop by up to 2x when other "
        "jobs are using the cores) because the laptop was not idle during the hackathon; load_avg_1m_at_end records how "
        "busy it was. "
        f"MLX sim rows measured via: {mlx_sources}. mlx {mx.__version__}, mujoco {mujoco.__version__}, "
        f"Python {platform.python_version()}."
    )
    out = {
        "machine": machine_info(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "load_avg_1m_at_end": round(os.getloadavg()[0], 1),
        "results": results,
        "speedup_best_gpu_vs_best_cpu": speedups,
        "notes": notes,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"{'task':10s} {'backend':24s} {'num_envs':>9s} {'env-steps/s':>14s}")
    print("-" * 60)
    for r in results:
        print(f"{r['task']:10s} {r['backend']:24s} {r['num_envs']:>9d} {r['steps_per_sec']:>14,.0f}")
    print("-" * 60)
    for k, v in speedups.items():
        print(f"speedup {k}: {v}x (best MLX GPU vs best CPU MuJoCo)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
