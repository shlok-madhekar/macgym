"""`macgym` command line.

    macgym train walker --iters 200            # built-in robot
    macgym train my_robot.py --envs 4096       # file defining `robot` + `task` (or make_env(num_envs))
    macgym view [walker | my_robot.py]         # live viewer in the browser
    macgym bench [--quick]                     # throughput benchmark (scripts/bench.py)
    macgym export runs/walker.npz --format coreml
    macgym new crab                            # scaffold crab.py from the robot template
"""
import argparse
import os
import runpy
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILTINS = ("walker", "cheetah", "hopper", "tripod", "worm")
LEGACY = ("walker", "cheetah", "arm", "cartpole")


# --------------------------------------------------------------------------- formatting
def _clock(sec):
    sec = int(sec)
    return f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s"


def _rate(x):
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def _err(msg):
    print(f"macgym: error: {msg}", file=sys.stderr)
    return 2


# --------------------------------------------------------------------------- env loading
def _is_file(target):
    return target.endswith(".py") or os.path.sep in target


def _target_name(target):
    return os.path.splitext(os.path.basename(target))[0] if _is_file(target) else target


def load_env(target, num_envs, seed=0):
    """Build a batched env from a built-in name or a robot file."""
    if _is_file(target):
        if not os.path.exists(target):
            raise FileNotFoundError(f"no such file: {target}")
        ns = runpy.run_path(target, run_name="__macgym__")
        if callable(ns.get("make_env")):
            return ns["make_env"](num_envs)
        if "robot" in ns and "task" in ns:
            return _mg_make(ns["robot"], ns["task"], num_envs, seed)
        raise ValueError(f"{target} must define `robot` and `task`, or `def make_env(num_envs)`")
    # built-in name: prefer the public-API robot, fall back to the legacy env registry
    try:
        import macgym.robots as robots
        fn = getattr(robots, target, None)
        if callable(fn):
            robot, task = fn()
            return _mg_make(robot, task, num_envs, seed)
    except ImportError:
        pass
    from macgym.envs import make as make_legacy
    return make_legacy(target, num_envs, seed=seed)


def _mg_make(robot, task, num_envs, seed):
    from macgym.api import make
    return make(robot, task, num_envs=num_envs, seed=seed)


# --------------------------------------------------------------------------- commands
def cmd_train(a):
    name = _target_name(a.target)
    save = a.save or os.path.join("runs", f"{name}.npz")
    print(f"macgym train  {name}  |  {a.envs} envs  |  {a.iters} iters", flush=True)
    t_build = time.perf_counter()
    env = load_env(a.target, a.envs, seed=a.seed)
    print(f"  robot ready in {time.perf_counter() - t_build:.1f}s  |  obs {env.obs_dim}  act {env.act_dim}"
          f"  |  dt {getattr(env, 'dt', 0.02):.3f}s", flush=True)
    every = a.log_every or max(1, a.iters // 20)
    hidden = tuple(int(h) for h in a.hidden.split(","))
    t0 = time.perf_counter()
    best = [None]

    def report(m):
        i = m["iter"]
        d = m.get("dist")
        if d is None and hasattr(env, "live_dist_mean"):
            d = env.live_dist_mean()
        if d is not None and (best[0] is None or d > best[0]):
            best[0] = d
        if i == 1 or i % every == 0 or i == a.iters:
            elapsed = time.perf_counter() - t0
            dist = f"{d:7.2f} m" if d is not None else "      - "
            print(f"  iter {i:5d}/{a.iters}  |  distance {dist}  |  return {m['mean_ep_return']:9.1f}  |  "
                  f"{_rate(m['sps'])} steps/s  |  {m['env_steps'] / 1e6:7.2f}M steps  |  {_clock(elapsed):>6}",
                  flush=True)

    try:
        from macgym.api import train as mg_train
    except ImportError:
        mg_train = None
    if mg_train is not None:
        mg_train(env, iters=a.iters, hidden=hidden, on_iter=report, save=save, seed=a.seed, verbose=False)
    else:  # API not available: drive PPO directly
        from macgym.rl.ppo import PPO
        ppo = PPO(env, hidden=hidden, seed=a.seed)
        for _ in range(a.iters):
            report(ppo.train_iter())
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        ppo.save(save)
    total = time.perf_counter() - t0
    tail = f"  |  best distance {best[0]:.2f} m" if best[0] is not None else ""
    print(f"done in {_clock(total)}{tail}  |  saved {save}", flush=True)
    return 0


def _server_help():
    try:
        return subprocess.run([sys.executable, "-m", "macgym.server", "--help"],
                              capture_output=True, text=True, timeout=30, cwd=ROOT).stdout
    except Exception:
        return ""


def cmd_view(a):
    url = f"http://localhost:{a.port}"
    target = a.target
    if target and _is_file(target):
        if not os.path.exists(target):
            return _err(f"no such file: {target}")
        if "--file" in _server_help():
            args = ["--file", os.path.abspath(target)]
        else:
            # server cannot load files itself: build the env here and hand it to mg.view
            import macgym as mg
            if not hasattr(mg, "view"):
                return _err("this server build cannot load robot files yet; pass a built-in name "
                            f"({', '.join(LEGACY)}) or open {url}/playground and paste the file")
            env = load_env(target, a.envs)
            print(f"macgym view  {target}  |  {url}", flush=True)
            if not a.no_browser:
                _open_later(url)
            mg.view(env, port=a.port)
            return 0
    else:
        args = ["--env", target] if target else []
    cmd = [sys.executable, "-m", "macgym.server", "--port", str(a.port), "--num-envs", str(a.envs), *args]
    if target:
        cmd.append("--autostart")
    print(f"macgym view  {target or '(pick a robot in the browser)'}  |  {url}", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT)
    if not a.no_browser:
        _wait_port(a.port, proc)
        subprocess.run(["open", url], check=False)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


def _wait_port(port, proc, timeout=20.0):
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout and proc.poll() is None:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def _open_later(url):
    import threading
    threading.Timer(2.0, lambda: subprocess.run(["open", url], check=False)).start()


def cmd_bench(a):
    script = os.path.join(ROOT, "scripts", "bench.py")
    if not os.path.exists(script):
        return _err(f"benchmark script not found at {script}")
    args = [sys.executable, script]
    if a.quick:
        args.append("--quick")
    if a.sizes:
        args += ["--sizes", a.sizes]
    if a.out:
        args += ["--out", a.out]
    return subprocess.call(args, cwd=ROOT)


def cmd_export(a):
    from macgym.export import export_policy
    if not os.path.exists(a.checkpoint):
        return _err(f"no such checkpoint: {a.checkpoint}")
    ext = {"onnx": ".onnx", "coreml": ".mlpackage", "npz": "_actor.npz"}[a.format]
    out = a.out or os.path.splitext(a.checkpoint)[0] + ext
    t0 = time.perf_counter()
    r = export_policy(a.checkpoint, out, format=a.format)
    err = r["max_abs_err"]
    check = f"max |diff| vs MLX {err:.1e}" if err is not None else "not verified on this machine"
    print(f"exported {a.format}  {r['path']}  |  {check}  |  {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


def cmd_new(a):
    name = a.name
    if not name.isidentifier():
        return _err(f"robot name must be a Python identifier, got {name!r}")
    path = a.out or f"{name}.py"
    if os.path.exists(path) and not a.force:
        return _err(f"{path} already exists (use --force to overwrite)")
    tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "robot_template.py")
    with open(tpl) as f:
        src = f.read().replace("{{name}}", name)
    with open(path, "w") as f:
        f.write(src)
    print(f"created {path}\n  train it:  macgym train {path} --iters 200\n  watch it:  macgym view {path}")
    return 0


# --------------------------------------------------------------------------- entry
def build_parser():
    p = argparse.ArgumentParser(prog="macgym", description="MacGym: robot learning on Apple Silicon.")
    sub = p.add_subparsers(dest="cmd", metavar="<command>")

    t = sub.add_parser("train", help="train a robot (built-in name or robot .py file)")
    t.add_argument("target", help=f"built-in ({', '.join(BUILTINS)}) or path/to/robot.py")
    t.add_argument("--envs", type=int, default=4096, help="parallel copies (default 4096)")
    t.add_argument("--iters", type=int, default=200, help="PPO iterations (default 200)")
    t.add_argument("--save", default=None, help="checkpoint path (default runs/<name>.npz)")
    t.add_argument("--hidden", default="256,256", help="MLP hidden sizes (default 256,256)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=None, help="print every N iters")
    t.set_defaults(fn=cmd_train)

    v = sub.add_parser("view", help="start the live viewer and open the browser")
    v.add_argument("target", nargs="?", default=None)
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--envs", type=int, default=4096)
    v.add_argument("--no-browser", action="store_true")
    v.set_defaults(fn=cmd_view)

    b = sub.add_parser("bench", help="run the throughput benchmark (scripts/bench.py)")
    b.add_argument("--quick", action="store_true")
    b.add_argument("--sizes", default=None, help="e.g. 1024,4096,16384")
    b.add_argument("--out", default=None, help="results json (default web/bench.json)")
    b.set_defaults(fn=cmd_bench)

    e = sub.add_parser("export", help="export a checkpoint to ONNX / CoreML / npz")
    e.add_argument("checkpoint")
    e.add_argument("--format", choices=("onnx", "coreml", "npz"), default="onnx")
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_export)

    n = sub.add_parser("new", help="scaffold a new robot file from the template")
    n.add_argument("name")
    n.add_argument("--out", default=None)
    n.add_argument("--force", action="store_true")
    n.set_defaults(fn=cmd_new)
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        p.print_help()
        return 0
    try:
        return a.fn(a) or 0
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, KeyError) as ex:
        return _err(ex)


if __name__ == "__main__":
    sys.exit(main())
