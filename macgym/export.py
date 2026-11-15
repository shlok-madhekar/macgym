"""Export a trained MacGym policy (deterministic actor) to ONNX, CoreML or plain .npz.

The exported graph is the full inference path used by ``PPO.act(obs, deterministic=True)``:

    x   = clip((obs - obs_mean) / sqrt(obs_var + 1e-8), -10, 10)
    h_i = tanh(W_i h_{i-1} + b_i)            (hidden layers)
    a   = clip(W_out h + b_out, -1, 1)       (action in [-1, 1])

Input name ``obs`` (float32, shape (batch, obs_dim)); output name ``action`` (batch, act_dim).

    from macgym.export import export_policy
    export_policy("runs/walker.npz", "walker.onnx")
    export_policy(policy, "walker.mlpackage")
    export_policy(policy, "walker_actor.npz")   # actor weights + normalization only
"""
import os
import tempfile

import numpy as np

EPS = 1e-8
OBS_CLIP = 10.0


# ---------------------------------------------------------------------------
# weight extraction
# ---------------------------------------------------------------------------
def load_actor(policy_or_ckpt):
    """Return dict(layers=[(W(out,in), b(out,)), ...], mean, var, meta) as float32 numpy.

    Accepts: a checkpoint path (.npz written by PPO.save / mg.train), a PPO instance,
    or any object exposing ``.ppo`` / ``.trainer`` (a PPO) or a ``save(path)`` method.
    """
    obj = policy_or_ckpt
    if isinstance(obj, (str, os.PathLike)):
        data = np.load(os.fspath(obj), allow_pickle=False)
        return _actor_from_npz(data)
    for attr in ("ppo", "trainer", "_ppo"):
        inner = getattr(obj, attr, None)
        if inner is not None and hasattr(inner, "save"):
            obj = inner
            break
    if not hasattr(obj, "save"):
        raise TypeError(f"cannot export {type(policy_or_ckpt).__name__}: pass a checkpoint path, "
                        "a PPO, or a Policy with .save(path)")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "policy.npz")
        obj.save(p)
        if not os.path.exists(p) and os.path.exists(p + ".npz"):
            p = p + ".npz"
        data = dict(np.load(p, allow_pickle=False))
    return _actor_from_npz(data)


def _actor_from_npz(data):
    files = list(data.keys()) if isinstance(data, dict) else list(data.files)
    if "__obs_mean" not in files:
        raise ValueError("checkpoint has no __obs_mean; not a MacGym PPO checkpoint")
    layers = []
    i = 0
    while f"actor.layers.{i}.weight" in files:
        layers.append((np.asarray(data[f"actor.layers.{i}.weight"], np.float32),
                       np.asarray(data[f"actor.layers.{i}.bias"], np.float32)))
        i += 1
    if not layers:
        raise ValueError("checkpoint has no actor.layers.* weights")
    meta = {}
    if "__env" in files:
        meta["env"] = str(data["__env"])
    if "__hidden" in files:
        meta["hidden"] = [int(h) for h in np.asarray(data["__hidden"]).ravel()]
    return {
        "layers": layers,
        "mean": np.asarray(data["__obs_mean"], np.float32),
        "var": np.asarray(data["__obs_var"], np.float32),
        "meta": meta,
    }


def numpy_forward(actor, obs):
    """Reference float32 numpy implementation of the exported graph."""
    x = np.asarray(obs, np.float32)
    x = np.clip((x - actor["mean"]) / np.sqrt(actor["var"] + EPS), -OBS_CLIP, OBS_CLIP)
    layers = actor["layers"]
    for W, b in layers[:-1]:
        x = np.tanh(x @ W.T + b)
    W, b = layers[-1]
    return np.clip(x @ W.T + b, -1.0, 1.0).astype(np.float32)


def mlx_forward(policy_or_ckpt, obs):
    """Run the actor in MLX exactly like PPO._act_det (for verification)."""
    import mlx.core as mx
    a = load_actor(policy_or_ckpt)
    x = mx.array(np.asarray(obs, np.float32))
    x = mx.clip((x - mx.array(a["mean"])) * mx.rsqrt(mx.array(a["var"]) + EPS), -OBS_CLIP, OBS_CLIP)
    for W, b in a["layers"][:-1]:
        x = mx.tanh(x @ mx.array(W).T + mx.array(b))
    W, b = a["layers"][-1]
    out = mx.clip(x @ mx.array(W).T + mx.array(b), -1.0, 1.0)
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def export_npz(actor, path):
    out = {"obs_mean": actor["mean"], "obs_var": actor["var"], "n_layers": np.array(len(actor["layers"]))}
    for i, (W, b) in enumerate(actor["layers"]):
        out[f"W{i}"], out[f"b{i}"] = W, b
    np.savez(path, **out)
    return path


def build_onnx(actor, opset=17):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    obs_dim = actor["mean"].shape[0]
    act_dim = actor["layers"][-1][0].shape[0]
    inits = [
        numpy_helper.from_array(actor["mean"], "obs_mean"),
        numpy_helper.from_array((1.0 / np.sqrt(actor["var"].astype(np.float64) + EPS)).astype(np.float32), "obs_inv_std"),
        numpy_helper.from_array(np.array(-OBS_CLIP, np.float32), "obs_lo"),
        numpy_helper.from_array(np.array(OBS_CLIP, np.float32), "obs_hi"),
        numpy_helper.from_array(np.array(-1.0, np.float32), "act_lo"),
        numpy_helper.from_array(np.array(1.0, np.float32), "act_hi"),
    ]
    nodes = [
        helper.make_node("Sub", ["obs", "obs_mean"], ["x_c"]),
        helper.make_node("Mul", ["x_c", "obs_inv_std"], ["x_n"]),
        helper.make_node("Clip", ["x_n", "obs_lo", "obs_hi"], ["h0"]),
    ]
    prev = "h0"
    n = len(actor["layers"])
    for i, (W, b) in enumerate(actor["layers"]):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(W.T), f"W{i}"))
        inits.append(numpy_helper.from_array(b, f"b{i}"))
        lin = f"lin{i}"
        nodes.append(helper.make_node("Gemm", [prev, f"W{i}", f"b{i}"], [lin]))
        if i < n - 1:
            nodes.append(helper.make_node("Tanh", [lin], [f"h{i + 1}"]))
            prev = f"h{i + 1}"
        else:
            nodes.append(helper.make_node("Clip", [lin, "act_lo", "act_hi"], ["action"]))
    graph = helper.make_graph(
        nodes, "macgym_actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", obs_dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch", act_dim])],
        initializer=inits,
    )
    model = helper.make_model(graph, producer_name="macgym", opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    for k, v in actor["meta"].items():
        helper.set_model_props(model, {**{p.key: p.value for p in model.metadata_props}, k: str(v)})
    onnx.checker.check_model(model)
    return model


def export_onnx(actor, path):
    import onnx
    onnx.save(build_onnx(actor), path)
    return path


def build_coreml(actor, batch=1):
    """CoreML ML Program built directly with the MIL builder (no torch/onnx needed)."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb

    obs_dim = actor["mean"].shape[0]
    mean = actor["mean"]
    inv_std = (1.0 / np.sqrt(actor["var"].astype(np.float64) + EPS)).astype(np.float32)
    layers = actor["layers"]
    from coremltools.converters.mil.mil import get_new_symbol

    @mb.program(input_specs=[mb.TensorSpec(shape=(get_new_symbol(), obs_dim))])
    def prog(obs):
        x = mb.mul(x=mb.sub(x=obs, y=mean), y=inv_std)
        x = mb.clip(x=x, alpha=-OBS_CLIP, beta=OBS_CLIP)
        for i, (W, b) in enumerate(layers):
            x = mb.linear(x=x, weight=W, bias=b)
            if i < len(layers) - 1:
                x = mb.tanh(x=x)
        return mb.clip(x=x, alpha=-1.0, beta=1.0, name="action")

    bdim = ct.RangeDim(lower_bound=1, upper_bound=65536, default=batch)
    model = ct.convert(prog, convert_to="mlprogram",
                       inputs=[ct.TensorType(name="obs", shape=(bdim, obs_dim), dtype=np.float32)], compute_precision=ct.precision.FLOAT32,
                       minimum_deployment_target=ct.target.macOS13)
    model.short_description = "MacGym deterministic policy: obs -> action in [-1, 1]"
    for k, v in actor["meta"].items():
        model.user_defined_metadata[k] = str(v)
    return model


def export_coreml(actor, path):
    model = build_coreml(actor)
    model.save(path)
    return path


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def _sample_obs(actor, n=64, seed=0):
    rng = np.random.default_rng(seed)
    std = np.sqrt(actor["var"] + EPS)
    return (actor["mean"] + std * rng.normal(size=(n, actor["mean"].shape[0])) * 2.0).astype(np.float32)


def verify_onnx(path, actor, obs=None):
    import onnxruntime as ort
    obs = _sample_obs(actor) if obs is None else obs
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(["action"], {"obs": obs})[0]
    return float(np.max(np.abs(got - _reference(actor, obs))))


def verify_coreml(path, actor, obs=None):
    import coremltools as ct
    obs = _sample_obs(actor, n=16) if obs is None else obs
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    got = model.predict({"obs": obs})
    got = np.asarray(got["action"], np.float32)
    return float(np.max(np.abs(got - _reference(actor, obs))))


def _reference(actor, obs):
    try:
        import mlx.core as mx
        x = mx.array(obs)
        x = mx.clip((x - mx.array(actor["mean"])) * mx.rsqrt(mx.array(actor["var"]) + EPS), -OBS_CLIP, OBS_CLIP)
        for W, b in actor["layers"][:-1]:
            x = mx.tanh(x @ mx.array(W).T + mx.array(b))
        W, b = actor["layers"][-1]
        return np.array(mx.clip(x @ mx.array(W).T + mx.array(b), -1.0, 1.0), dtype=np.float32)
    except ImportError:
        return numpy_forward(actor, obs)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def export_policy(policy_or_ckpt, path, format=None, verify=True, tol=1e-4):
    """Export to ``path``. Format inferred from extension (.onnx / .mlpackage / .npz) or ``format``
    ('onnx' | 'coreml' | 'npz'). Returns dict(path, format, max_abs_err)."""
    actor = load_actor(policy_or_ckpt)
    fmt = (format or "").lower()
    if not fmt:
        ext = os.path.splitext(os.fspath(path))[1].lower()
        fmt = {".onnx": "onnx", ".mlpackage": "coreml", ".mlmodel": "coreml", ".npz": "npz"}.get(ext)
        if fmt is None:
            raise ValueError(f"cannot infer export format from {path!r}; use .onnx, .mlpackage or .npz")
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    err = None
    if fmt == "onnx":
        try:
            import onnx  # noqa: F401
        except ImportError as e:
            raise ImportError("ONNX export needs `pip install onnx onnxruntime` (or macgym[export])") from e
        export_onnx(actor, path)
        if verify:
            try:
                err = verify_onnx(path, actor)
            except ImportError:
                err = None
    elif fmt == "coreml":
        try:
            import coremltools  # noqa: F401
        except ImportError as e:
            raise ImportError("CoreML export needs `pip install coremltools` (or macgym[export])") from e
        if not path.endswith((".mlpackage", ".mlmodel")):
            path += ".mlpackage"
        export_coreml(actor, path)
        if verify:
            try:
                err = verify_coreml(path, actor)
            except Exception:  # prediction needs macOS + CoreML framework; export itself succeeded
                err = None
    elif fmt == "npz":
        export_npz(actor, path)
        if verify:
            back = dict(np.load(path))
            err = float(max(np.max(np.abs(back[f"W{i}"] - W)) for i, (W, _) in enumerate(actor["layers"])))
    else:
        raise ValueError(f"unknown export format {fmt!r}")
    if err is not None and err > tol:
        raise RuntimeError(f"exported {fmt} model differs from MLX policy: max abs err {err:.2e} > {tol:.0e}")
    return {"path": path, "format": fmt, "max_abs_err": err}
