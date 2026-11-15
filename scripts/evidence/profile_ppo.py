"""Time breakdown of one PPO iteration, walker N=4096 (mg.train defaults: hidden 256x256, T=24,
4 epochs x 4 minibatches). Mirrors macgym/rl/ppo.py PPO.train_iter with mx.eval barriers
between pieces. Writes web/deck/data/profile.json."""
import json, os, subprocess, time
from collections import defaultdict
import numpy as np
import mlx.core as mx
import macgym as mg
from macgym.rl.ppo import PPO, EXTRA_EPISODE_KEYS

OUT = os.path.join(os.path.dirname(__file__), "../../web/deck/data")
up = lambda: subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
N = 4096


def timed_iter(p, acc):
    tic = time.perf_counter
    T, N = p.T, p.N
    t_all = tic()
    if p.obs is None:
        p.obs = p.env.reset()
    mean, var = p.obs_rms.mean, p.obs_rms.var
    raw_obs = []
    obs = p.obs
    for t in range(T):
        t0 = tic()
        noise = mx.random.normal((N, p.act_dim))
        x, raw, action, logp, value = p._policy_step(obs, mean, var, noise)
        mx.eval(x, raw, action, logp, value)
        t1 = tic(); acc["policy_forward_rollout"] += t1 - t0
        raw_obs.append(obs)
        obs, rew, done, info = p.env.step(action)
        mx.eval(obs, rew, done, info["episode_return"], info["episode_len"], info["episode_dist"])
        t2 = tic(); acc["env_step_physics_reward_reset"] += t2 - t1
        p.buf_obs[t] = x; p.buf_act[t] = raw; p.buf_logp[t] = logp; p.buf_val[t] = value
        p.buf_rew[t] = rew; p.buf_done[t] = done
        p.buf_epret[t] = info["episode_return"]; p.buf_eplen[t] = info["episode_len"]
        for k in EXTRA_EPISODE_KEYS:
            if k in info:
                if k not in p.buf_extra:
                    from collections import deque
                    p.buf_extra[k] = mx.zeros((T, N)); p.ep_extra[k] = deque(maxlen=p.window)
                p.buf_extra[k][t] = info[k]
        mx.eval(p.buf_obs, p.buf_act, p.buf_logp, p.buf_val, p.buf_rew, p.buf_done, p.buf_epret, p.buf_eplen,
                *p.buf_extra.values())
        acc["rollout_buffer_writes"] += tic() - t2
    p.obs = obs
    t0 = tic()
    last_val = p._value_fn(obs, mean, var); mx.eval(last_val)
    acc["policy_forward_rollout"] += tic() - t0
    t0 = tic()
    rew = p.buf_rew
    accs = []; a = p.ret_acc
    for t in range(T):
        a = a * p.gamma + rew[t]; accs.append(a); a = a * (1.0 - p.buf_done[t])
    p.ret_acc = a
    p.ret_rms.update(mx.stack(accs))
    rew = rew * mx.rsqrt(p.ret_rms.var + 1e-8)
    adv, ret = p._gae(rew, p.buf_val, p.buf_done, last_val)
    B = T * N
    b_obs = p.buf_obs.reshape(B, p.obs_dim); b_act = p.buf_act.reshape(B, p.act_dim)
    b_logp = p.buf_logp.reshape(B); b_val = p.buf_val.reshape(B); b_ret = ret.reshape(B)
    b_adv = adv.reshape(B); b_adv = (b_adv - mx.mean(b_adv)) / (mx.std(b_adv) + 1e-8)
    mx.eval(b_obs, b_act, b_logp, b_val, b_ret, b_adv, p.ret_acc, p.ret_rms.var)
    acc["reward_norm_gae_advnorm"] += tic() - t0
    t0 = tic()
    mb = B // p.minibatches
    for _ in range(p.epochs):
        perm = mx.random.permutation(B)
        for i in range(p.minibatches):
            idx = perm[i * mb:(i + 1) * mb]
            out = p._update_step(b_obs[idx], b_act[idx], b_logp[idx], b_adv[idx], b_ret[idx], b_val[idx])
            mx.eval(p.model.state, p.opt.state)
    acc["ppo_update_backprop_adam"] += tic() - t0
    t0 = tic()
    p.obs_rms.update(mx.stack(raw_obs))
    done_np = np.array(p.buf_done)
    if done_np.any():
        mask = done_np > 0.5
        p.ep_returns.extend(np.array(p.buf_epret)[mask].tolist())
        p.ep_lens.extend(np.array(p.buf_eplen)[mask].tolist())
        for k, buf in p.buf_extra.items():
            p.ep_extra[k].extend(np.array(buf)[mask].tolist())
    mx.eval(p.obs_rms.mean, p.obs_rms.var)
    _ = float(out[0]); _ = p.env.live_dist_mean()
    acc["stats_sync_obsnorm_host_copy"] += tic() - t0
    p.iter += 1; p.env_steps += B
    acc["total_instrumented"] += tic() - t_all


if __name__ == "__main__":
    env = mg.make(mg.robots.walker(), num_envs=N, seed=0)
    p = PPO(env, hidden=(256, 256), seed=0)
    for _ in range(3):
        p.train_iter()  # warm/compile
    u0 = up()
    plain = []
    for _ in range(5):
        t0 = time.perf_counter(); p.train_iter(); plain.append(time.perf_counter() - t0)
    acc = defaultdict(float)
    K = 8
    for _ in range(K):
        timed_iter(p, acc)
    u1 = up()
    per = {k: v / K for k, v in acc.items()}
    tot = per["total_instrumented"]
    parts = {k: v for k, v in per.items() if k != "total_instrumented"}
    out = {"setup": f"walker (mg.robots.walker), N={N}, rollout T={p.T}, PPO 4 epochs x 4 minibatches, MLP 256x256, "
                    f"{p.T * N} env-steps/iter; mx.eval barrier after every piece; mean of {K} iterations after 8 warm iters",
           "uptime_start": u0, "uptime_end": u1,
           "plain_train_iter_s_mean": float(np.mean(plain)), "plain_train_iter_s_all": plain,
           "plain_env_steps_per_s": p.T * N / float(np.mean(plain)),
           "instrumented_iter_s": tot,
           "pieces_s": parts,
           "pieces_pct": {k: v / tot * 100 for k, v in parts.items()},
           "unaccounted_s": tot - sum(parts.values())}
    json.dump(out, open(os.path.join(OUT, "profile.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
