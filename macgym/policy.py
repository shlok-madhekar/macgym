"""Trained policy handle: mg.load_policy(path, env) -> Policy (.act, .export, .weights)."""
import numpy as np
import mlx.core as mx

from macgym.rl.ppo import ActorCritic


class Policy:
    """Deterministic actor: a = clip(actor(clip((obs - obs_mean) / sqrt(obs_var + 1e-8), -10, 10)), -1, 1),
    actor = MLP with tanh hidden activations and linear output.

    Attributes for exporters: obs_dim, act_dim, hidden, obs_mean / obs_var (np float32 (obs_dim,)),
    model (ppo.ActorCritic), weights() -> [(W (out, in), b (out,)), ...] numpy float32, env_name."""

    def __init__(self, model, obs_mean, obs_var, hidden, env_name=None):
        self.model = model
        self.obs_mean = mx.array(obs_mean)
        self.obs_var = mx.array(obs_var)
        self.hidden = tuple(int(h) for h in hidden)
        self.obs_dim = int(self.obs_mean.shape[0])
        self.act_dim = int(model.log_std.shape[0])
        self.env_name = env_name
        self._act = mx.compile(self._act_impl, inputs=model.state)

    @classmethod
    def from_ppo(cls, ppo, env=None):
        p = cls(ppo.model, ppo.obs_rms.mean, ppo.obs_rms.var, ppo.hidden,
                env_name=getattr(env, "name", None))
        p.ppo = ppo  # live: obs stats keep updating if training continues
        return p

    def _act_impl(self, obs, mean, var):
        x = mx.clip((obs - mean) * mx.rsqrt(var + 1e-8), -10.0, 10.0)
        return mx.clip(self.model.actor(x), -1.0, 1.0)

    def act(self, obs):
        """obs (N, obs_dim) or (obs_dim,) as mx or numpy -> actions of the same kind, in [-1, 1]."""
        is_np = not isinstance(obs, mx.array)
        o = mx.array(np.asarray(obs, np.float32)) if is_np else obs.astype(mx.float32)
        single = o.ndim == 1
        if single:
            o = o[None]
        if o.shape[1] != self.obs_dim:
            raise ValueError(f"policy expects obs_dim {self.obs_dim}, got {o.shape[1]}")
        mean, var = self.obs_mean, self.obs_var
        if getattr(self, "ppo", None) is not None:
            mean, var = self.ppo.obs_rms.mean, self.ppo.obs_rms.var
        a = self._act(o, mean, var)
        if single:
            a = a[0]
        return np.array(a) if is_np else a

    __call__ = act

    def weights(self):
        out = []
        for layer in self.model.actor.layers:
            out.append((np.array(layer.weight, np.float32), np.array(layer.bias, np.float32)))
        return out

    def save(self, path):
        """Write a PPO-compatible checkpoint (.npz) loadable by mg.load_policy / PPO.load."""
        if getattr(self, "ppo", None) is not None:
            return self.ppo.save(path)
        from mlx.utils import tree_flatten
        flat = {k: np.array(v) for k, v in tree_flatten(self.model.parameters())}
        np.savez(path, **flat, __obs_mean=np.array(self.obs_mean), __obs_var=np.array(self.obs_var),
                 __obs_count=np.array(1.0, np.float32), __ret_var=np.array(1.0, np.float32),
                 __hidden=np.array(self.hidden), __env=np.array(self.env_name or ""))

    def export(self, path, **kwargs):
        from macgym.export import export_policy  # owned by the export module
        return export_policy(self, path, **kwargs)

    def __repr__(self):
        return f"Policy(obs={self.obs_dim}, act={self.act_dim}, hidden={self.hidden}, env={self.env_name!r})"


def load_policy(path, env=None):
    """Load a PPO checkpoint (.npz written by mg.train / PPO.save / macgym.train)."""
    data = np.load(path)
    hidden = tuple(int(h) for h in np.atleast_1d(data["__hidden"]))
    obs_dim = int(data["actor.layers.0.weight"].shape[1])
    act_dim = int(data["log_std"].shape[0])
    if env is not None and (env.obs_dim != obs_dim or env.act_dim != act_dim):
        raise ValueError(f"checkpoint {path} is obs {obs_dim} / act {act_dim}, env "
                         f"{getattr(env, 'name', '?')} is obs {env.obs_dim} / act {env.act_dim}")
    model = ActorCritic(obs_dim, act_dim, hidden, 0)
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten([(k, mx.array(data[k])) for k in data.files if not k.startswith("__")]))
    mx.eval(model.parameters())
    env_name = str(data["__env"]) if "__env" in data.files else None
    return Policy(model, data["__obs_mean"], data["__obs_var"], hidden, env_name=env_name)
