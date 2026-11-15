"""Export round-trips (ONNX / CoreML / npz vs MLX) and `macgym train walker --iters 3`.

    python -m pytest tests/test_cli_export.py -q
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from macgym.export import export_policy, load_actor, mlx_forward, numpy_forward  # noqa: E402

TOL = 1e-4


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """A small freshly trained walker checkpoint (2 PPO iters, 256 envs)."""
    from macgym.envs import make
    from macgym.rl.ppo import PPO
    env = make("walker", 256)
    ppo = PPO(env, hidden=(64, 64))
    for _ in range(2):
        ppo.train_iter()
    path = str(tmp_path_factory.mktemp("ck") / "walker.npz")
    ppo.save(path)
    obs_mx = env.reset()
    for _ in range(30):  # move off the rest pose so observations are varied
        obs_mx, _, _, _ = env.step(ppo.act(obs_mx, deterministic=True))
    obs = np.array(obs_mx, dtype=np.float32)
    ref = np.array(ppo.act(obs_mx, deterministic=True), dtype=np.float32)
    return path, obs, ref


def test_numpy_matches_ppo(ckpt):
    path, obs, ref = ckpt
    actor = load_actor(path)
    assert np.abs(numpy_forward(actor, obs) - ref).max() < TOL
    assert np.abs(mlx_forward(path, obs) - ref).max() < TOL


def test_onnx_roundtrip(ckpt, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    path, obs, ref = ckpt
    out = export_policy(path, tmp_path / "p.onnx")
    assert out["max_abs_err"] is not None and out["max_abs_err"] < TOL
    got = ort.InferenceSession(out["path"]).run(["action"], {"obs": obs})[0]
    assert got.shape == ref.shape and np.abs(got - ref).max() < TOL


def test_coreml_roundtrip(ckpt, tmp_path):
    ct = pytest.importorskip("coremltools")
    path, obs, ref = ckpt
    out = export_policy(path, tmp_path / "p.mlpackage")
    assert os.path.isdir(out["path"])
    model = ct.models.MLModel(out["path"])
    got = np.asarray(model.predict({"obs": obs[:8]})["action"], np.float32)
    assert np.abs(got - ref[:8]).max() < TOL


def test_npz_roundtrip(ckpt, tmp_path):
    path, obs, ref = ckpt
    out = export_policy(path, tmp_path / "p.npz")
    d = np.load(out["path"])
    actor = {"layers": [(d[f"W{i}"], d[f"b{i}"]) for i in range(int(d["n_layers"]))],
             "mean": d["obs_mean"], "var": d["obs_var"]}
    assert np.abs(numpy_forward(actor, obs) - ref).max() < TOL


def test_gymnasium_wrapper():
    pytest.importorskip("gymnasium")
    from macgym.envs import make
    from macgym.gym_compat import to_gymnasium
    v = to_gymnasium(make("walker", 16))
    obs, _ = v.reset(seed=0)
    assert obs.shape == (16, 24) and obs.dtype == np.float32
    for _ in range(20):
        obs, r, te, tr, info = v.step(v.action_space.sample())
    assert r.shape == (16,) and te.dtype == bool and tr.dtype == bool


def _cli(*args, cwd):
    return subprocess.run([sys.executable, "-m", "macgym.cli", *args], cwd=cwd, capture_output=True,
                          text=True, timeout=300, env={**os.environ, "PYTHONPATH": ROOT})


def test_cli_train_walker(tmp_path):
    save = tmp_path / "w.npz"
    r = _cli("train", "walker", "--iters", "3", "--envs", "256", "--save", str(save), cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "iter     3/3" in r.stdout and "steps/s" in r.stdout and " m " in r.stdout
    assert save.exists()
    r = _cli("export", str(save), "--format", "onnx", "--out", str(tmp_path / "w.onnx"), cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "w.onnx").exists()


def test_cli_new_and_train_file(tmp_path):
    r = _cli("new", "crab", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    r = _cli("train", "crab.py", "--iters", "2", "--envs", "128", "--save", str(tmp_path / "c.npz"), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "c.npz").exists()
