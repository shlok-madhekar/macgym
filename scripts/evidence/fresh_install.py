"""Fresh-install timing: clone the repo into a brand-new temp dir, make a venv, pip install -e ., train walker.
Writes web/deck/data/fresh_install.json. Deletes the temp dir afterwards."""
import json, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "web/deck/data/fresh_install.json")
RUNTIME_PY = "/Users/shlok/.aside/runtime/bin/python3"
BREW_PY = "/opt/homebrew/bin/python3"
tmp = tempfile.mkdtemp(prefix="macgym_fresh_")
steps = []
env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "PIP_", "VIRTUAL_ENV", "__PYVENV"))}  # strip the agent runtime's PYTHONHOME etc.


def run(cmd, cwd, timeout=1800):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    dt = time.time() - t0
    out = (p.stdout + p.stderr).strip().splitlines()
    rec = {"cwd": cwd, "command": cmd, "returncode": p.returncode, "wall_seconds": round(dt, 2), "last_lines": out[-25:]}
    steps.append(rec)
    print(f"[{p.returncode}] {dt:7.1f}s  {cmd}", flush=True)
    return p, dt, out


def save(extra):
    head = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    json.dump({"temp_dir": tmp, "source_commit": head,
               "note": "git clone copies committed files only (HEAD); uncommitted working-tree changes are not included.",
               "machine": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
               "uptime_at_end": subprocess.check_output(["uptime"], text=True).strip(),
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"), **extra, "steps": steps},
              open(OUT, "w"), indent=1)


extra = {}
try:
    run(f"git clone {ROOT} fresh", tmp)
    d = os.path.join(tmp, "fresh")
    p, _, _ = run(f"{RUNTIME_PY} -m venv --system-site-packages=false .venv", d)
    if p.returncode != 0 or not os.path.exists(os.path.join(d, ".venv/bin/python")):
        shutil.rmtree(os.path.join(d, ".venv"), ignore_errors=True)
        p, _, _ = run(f"{BREW_PY} -m venv .venv", d)
    vpy = os.path.join(d, ".venv/bin/python")
    vp, _, vo = run(f"{vpy} --version", d)
    extra["venv_python"] = vo[-1] if vo else None
    p, t_inst, out = run(f"{vpy} -m pip install -e .", d)
    extra["install_seconds"] = round(t_inst, 2); extra["install_ok"] = p.returncode == 0
    if p.returncode == 0:
        run(f"{vpy} -m pip freeze", d)
        mg = os.path.join(d, ".venv/bin/macgym")
        p, t_tr, out = run(f"{mg} train walker --iters 150 --envs 4096", d, timeout=3000)
        extra["train_seconds"] = round(t_tr, 2); extra["train_ok"] = p.returncode == 0
        dl = [l for l in out if "distance" in l]
        extra["final_distance_line"] = dl[-1] if dl else None
        # import location check: macgym imported from the fresh clone, not the dev tree
        run(f"{vpy} -c \"import macgym,sys;print(macgym.__file__)\"", "/tmp")
    else:
        extra["install_failure"] = "\n".join(out[-25:])
except Exception as e:
    extra["exception"] = repr(e)
finally:
    save(extra)
    shutil.rmtree(tmp, ignore_errors=True)
    print("deleted", tmp, "exists:", os.path.exists(tmp))
    print(json.dumps(extra, indent=1))
