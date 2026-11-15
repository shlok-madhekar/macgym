"""Train briefly, then export the policy to CoreML (.mlpackage) and ONNX and run it outside MLX.

    python examples/05_export_coreml.py
"""
import os

import numpy as np

import macgym as mg

env = mg.make(*mg.robots.walker(), num_envs=int(os.environ.get("MACGYM_ENVS", 4096)))
result = mg.train(env, iters=int(os.environ.get("MACGYM_ITERS", 50)), save="runs/export_demo.npz")
os.makedirs("exports", exist_ok=True)

info = result.policy.export("exports/walker.onnx")
print(f"ONNX   -> {info['path']}  max |diff| vs MLX: {info['max_abs_err']:.1e}")
try:
    info = result.policy.export("exports/walker.mlpackage")
    err = info["max_abs_err"]
    print(f"CoreML -> {info['path']}  max |diff| vs MLX: " + (f"{err:.1e}" if err is not None else "n/a"))
    import coremltools as ct

    model = ct.models.MLModel(info["path"])
    obs = np.array(env.reset(), dtype=np.float32)[:1]
    print("CoreML action for one observation:", np.round(model.predict({"obs": obs})["action"][0], 3))
except ImportError as e:
    print("CoreML export skipped:", e)
