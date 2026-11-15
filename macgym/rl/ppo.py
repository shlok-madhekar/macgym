"""Batched PPO in MLX (GPU). See CONTRACT.md section 2."""
import math
import time
from collections import deque
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

LOG2PI = math.log(2.0 * math.pi)


def _orthogonal(shape, gain, rng):
    """Orthogonal init for an MLX Linear weight of shape (out, in)."""
    rows, cols = shape
    a = rng.standard_normal((max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    return mx.array((gain * q[:rows, :cols]).astype(np.float32))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, out_gain, rng):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        for i, layer in enumerate(self.layers):
            gain = out_gain if i == len(self.layers) - 1 else math.sqrt(2.0)
            layer.weight = _orthogonal(layer.weight.shape, gain, rng)
            layer.bias = mx.zeros_like(layer.bias)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = mx.tanh(layer(x))
        return self.layers[-1](x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.actor = MLP(obs_dim, hidden, act_dim, 0.01, rng)
        self.critic = MLP(obs_dim, hidden, 1, 1.0, rng)
        self.log_std = mx.full((act_dim,), -0.5)


class RunningMeanStd:
    """Running mean/var (Chan parallel update), kept as mx arrays."""

    def __init__(self, shape):
        self.mean = mx.zeros(shape)
        self.var = mx.ones(shape)
        self.count = 1e-4

    def update(self, x):
        x = x.reshape(-1, *self.mean.shape)
        b_mean = mx.mean(x, axis=0)
        b_var = mx.var(x, axis=0)
        b_count = x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * (b_count / tot)
        m2 = self.var * self.count + b_var * b_count + delta * delta * (self.count * b_count / tot)
        self.var = m2 / tot
        self.count = tot


def _gaussian_logp(x, mu, log_std):
    return mx.sum(-0.5 * mx.square((x - mu) / mx.exp(log_std)) - log_std - 0.5 * LOG2PI, axis=-1)


class PPO:
    def __init__(self, env, hidden=(128, 128), lr=3e-4, rollout_len=24, epochs=4, minibatches=4,
                 gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.0, vf_coef=0.5, seed=0,
                 max_grad_norm=1.0, clip_value=True, norm_reward=True, lr_decay_iters=None,
                 window=100):
        self.env = env
        self.N = env.num_envs
        self.T = rollout_len
        self.obs_dim, self.act_dim = env.obs_dim, env.act_dim
        self.epochs, self.minibatches = epochs, minibatches
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.max_grad_norm, self.clip_value = max_grad_norm, clip_value
        self.norm_reward = norm_reward
        self.lr0, self.lr_decay_iters = lr, lr_decay_iters
        self.hidden = tuple(hidden)
        mx.random.seed(seed)

        self.model = ActorCritic(self.obs_dim, self.act_dim, self.hidden, seed)
        mx.eval(self.model.parameters())
        self.opt = optim.Adam(learning_rate=lr)
        self.obs_rms = RunningMeanStd((self.obs_dim,))
        self.ret_rms = RunningMeanStd(())
        self.ret_acc = mx.zeros((self.N,))  # per-env discounted return for reward scaling

        T, N = self.T, self.N
        self.buf_obs = mx.zeros((T, N, self.obs_dim))
        self.buf_act = mx.zeros((T, N, self.act_dim))
        self.buf_logp = mx.zeros((T, N))
        self.buf_val = mx.zeros((T, N))
        self.buf_rew = mx.zeros((T, N))
        self.buf_done = mx.zeros((T, N))
        self.buf_epret = mx.zeros((T, N))
        self.buf_eplen = mx.zeros((T, N))

        self.ep_returns = deque(maxlen=window)
        self.ep_lens = deque(maxlen=window)
        self.iter = 0
        self.env_steps = 0
        self.obs = None
        self._build_compiled()

    # ---------------------------------------------------------------- compiled fns
    def _build_compiled(self):
        model, opt = self.model, self.opt
        clip, vf_coef, ent_coef, clip_value = self.clip, self.vf_coef, self.ent_coef, self.clip_value
        max_norm = self.max_grad_norm
        gamma, lam = self.gamma, self.lam

        @partial(mx.compile, inputs=model.state)
        def policy_step(obs, mean, var, noise):
            x = mx.clip((obs - mean) * mx.rsqrt(var + 1e-8), -10.0, 10.0)
            mu = model.actor(x)
            log_std = model.log_std
            raw = mu + mx.exp(log_std) * noise
            logp = _gaussian_logp(raw, mu, log_std)
            value = model.critic(x)[:, 0]
            return x, raw, mx.clip(raw, -1.0, 1.0), logp, value

        @partial(mx.compile, inputs=model.state)
        def act_det(obs, mean, var):
            x = mx.clip((obs - mean) * mx.rsqrt(var + 1e-8), -10.0, 10.0)
            return mx.clip(model.actor(x), -1.0, 1.0)

        @partial(mx.compile, inputs=model.state)
        def value_fn(obs, mean, var):
            x = mx.clip((obs - mean) * mx.rsqrt(var + 1e-8), -10.0, 10.0)
            return model.critic(x)[:, 0]

        def loss_fn(m, x, act, logp_old, adv, ret, val_old):
            mu = m.actor(x)
            logp = _gaussian_logp(act, mu, m.log_std)
            ratio = mx.exp(logp - logp_old)
            pg = -mx.mean(mx.minimum(ratio * adv, mx.clip(ratio, 1.0 - clip, 1.0 + clip) * adv))
            v = m.critic(x)[:, 0]
            if clip_value:
                v_clip = val_old + mx.clip(v - val_old, -clip, clip)
                vl = 0.5 * mx.mean(mx.maximum(mx.square(v - ret), mx.square(v_clip - ret)))
            else:
                vl = 0.5 * mx.mean(mx.square(v - ret))
            ent = mx.sum(m.log_std + 0.5 * (1.0 + LOG2PI))
            loss = pg + vf_coef * vl - ent_coef * ent
            return loss, (pg, vl, ent)

        grad_fn = nn.value_and_grad(model, loss_fn)
        state = [model.state, opt.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def update_step(x, act, logp_old, adv, ret, val_old):
            (loss, (pg, vl, ent)), grads = grad_fn(model, x, act, logp_old, adv, ret, val_old)
            grads, _ = optim.clip_grad_norm(grads, max_norm)
            opt.update(model, grads)
            return pg, vl, ent

        @mx.compile
        def gae(rew, val, done, last_val):
            T = rew.shape[0]
            adv = []
            last = mx.zeros_like(last_val)
            for t in range(T - 1, -1, -1):
                nxt = last_val if t == T - 1 else val[t + 1]
                nonterm = 1.0 - done[t]
                delta = rew[t] + gamma * nxt * nonterm - val[t]
                last = delta + gamma * lam * nonterm * last
                adv.append(last)
            adv = mx.stack(adv[::-1])
            return adv, adv + val

        self._policy_step, self._act_det, self._value_fn = policy_step, act_det, value_fn
        self._update_step, self._gae = update_step, gae

    # ---------------------------------------------------------------- public API
    def act(self, obs, deterministic=False):
        if deterministic:
            return self._act_det(obs, self.obs_rms.mean, self.obs_rms.var)
        noise = mx.random.normal((obs.shape[0], self.act_dim))
        return self._policy_step(obs, self.obs_rms.mean, self.obs_rms.var, noise)[2]

    def train_iter(self) -> dict:
        t0 = time.perf_counter()
        if self.obs is None:
            self.obs = self.env.reset()
        T, N = self.T, self.N
        if self.lr_decay_iters:
            frac = max(0.0, 1.0 - self.iter / self.lr_decay_iters)
            self.opt.learning_rate = self.lr0 * max(frac, 0.05)

        mean, var = self.obs_rms.mean, self.obs_rms.var
        raw_obs = []
        obs = self.obs
        for t in range(T):
            noise = mx.random.normal((N, self.act_dim))
            x, raw, action, logp, value = self._policy_step(obs, mean, var, noise)
            raw_obs.append(obs)
            obs, rew, done, info = self.env.step(action)
            self.buf_obs[t] = x
            self.buf_act[t] = raw
            self.buf_logp[t] = logp
            self.buf_val[t] = value
            self.buf_rew[t] = rew
            self.buf_done[t] = done
            if "episode_return" in info:
                self.buf_epret[t] = info["episode_return"]
            if "episode_len" in info:
                self.buf_eplen[t] = info["episode_len"]
            mx.eval(obs, rew, done, logp, value)
        self.obs = obs
        last_val = self._value_fn(obs, mean, var)

        rew = self.buf_rew
        if self.norm_reward:
            accs = []
            acc = self.ret_acc
            for t in range(T):
                acc = acc * self.gamma + rew[t]
                accs.append(acc)
                acc = acc * (1.0 - self.buf_done[t])
            self.ret_acc = acc
            self.ret_rms.update(mx.stack(accs))
            rew = rew * mx.rsqrt(self.ret_rms.var + 1e-8)
        adv, ret = self._gae(rew, self.buf_val, self.buf_done, last_val)

        B = T * N
        b_obs = self.buf_obs.reshape(B, self.obs_dim)
        b_act = self.buf_act.reshape(B, self.act_dim)
        b_logp = self.buf_logp.reshape(B)
        b_val = self.buf_val.reshape(B)
        b_ret = ret.reshape(B)
        b_adv = adv.reshape(B)
        b_adv = (b_adv - mx.mean(b_adv)) / (mx.std(b_adv) + 1e-8)
        mx.eval(b_obs, b_act, b_logp, b_val, b_ret, b_adv)

        mb = B // self.minibatches
        stats = []
        for _ in range(self.epochs):
            perm = mx.random.permutation(B)
            for i in range(self.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                out = self._update_step(b_obs[idx], b_act[idx], b_logp[idx], b_adv[idx],
                                        b_ret[idx], b_val[idx])
                stats.append(out)
                mx.eval(self.model.state, self.opt.state)
        last = stats[-1]

        # obs normalization uses this rollout's raw obs (affects next rollout)
        self.obs_rms.update(mx.stack(raw_obs))

        done_np = np.array(self.buf_done)
        rew_mean = float(mx.mean(self.buf_rew))
        if done_np.any():
            mask = done_np > 0.5
            er = np.array(self.buf_epret)[mask]
            el = np.array(self.buf_eplen)[mask]
            self.ep_returns.extend(er.tolist())
            self.ep_lens.extend(el.tolist())
        mx.eval(self.obs_rms.mean, self.obs_rms.var, self.ret_rms.var, self.ret_acc)

        self.iter += 1
        self.env_steps += B
        dt = time.perf_counter() - t0
        return {
            "iter": self.iter,
            "env_steps": self.env_steps,
            "sps": B / dt,
            "mean_reward": rew_mean,
            "mean_ep_return": float(np.mean(self.ep_returns)) if self.ep_returns else 0.0,
            "mean_ep_len": float(np.mean(self.ep_lens)) if self.ep_lens else 0.0,
            "policy_loss": float(last[0]),
            "value_loss": float(last[1]),
            "entropy": float(last[2]),
            "time": dt,
        }

    def save(self, path):
        flat = dict(tree_flatten(self.model.parameters()))
        flat["__obs_mean"] = self.obs_rms.mean
        flat["__obs_var"] = self.obs_rms.var
        flat["__obs_count"] = mx.array(self.obs_rms.count)
        flat["__ret_var"] = self.ret_rms.var
        np.savez(path, **{k: np.array(v) for k, v in flat.items()},
                 __hidden=np.array(self.hidden), __env=np.array(self.env.name))

    def load(self, path):
        data = np.load(path)
        params = [(k, mx.array(data[k])) for k in data.files if not k.startswith("__")]
        self.model.update(tree_unflatten(params))
        self.obs_rms.mean = mx.array(data["__obs_mean"])
        self.obs_rms.var = mx.array(data["__obs_var"])
        self.obs_rms.count = float(data["__obs_count"])
        self.ret_rms.var = mx.array(data["__ret_var"])
        mx.eval(self.model.parameters())
