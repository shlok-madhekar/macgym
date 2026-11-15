"""CLI trainer: python3 -m macgym.train --env arm --num-envs 4096 --iters 300 --save runs/arm.npz"""
import argparse
import os
import time

from macgym.envs import make
from macgym.rl.ppo import PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cartpole")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--save", default=None)
    ap.add_argument("--load", default=None)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rollout-len", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--lr-decay", action="store_true", help="linear lr decay over --iters")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    env = make(a.env, a.num_envs)
    ppo = PPO(env, hidden=tuple(int(h) for h in a.hidden.split(",")), lr=a.lr,
              rollout_len=a.rollout_len, epochs=a.epochs, minibatches=a.minibatches,
              gamma=a.gamma, ent_coef=a.ent_coef, seed=a.seed,
              lr_decay_iters=a.iters if a.lr_decay else None)
    if a.load:
        ppo.load(a.load)
    print(f"env={a.env} N={a.num_envs} obs={env.obs_dim} act={env.act_dim}", flush=True)
    t0 = time.perf_counter()
    for _ in range(a.iters):
        m = ppo.train_iter()
        print(f"it {m['iter']:4d} | steps {m['env_steps']/1e6:8.2f}M | sps {m['sps']/1e6:5.2f}M | "
              f"ep_ret {m['mean_ep_return']:9.2f} | ep_len {m['mean_ep_len']:6.1f} | "
              f"rew {m['mean_reward']:8.4f} | pl {m['policy_loss']:+.4f} | vl {m['value_loss']:.4f} | "
              f"ent {m['entropy']:.3f} | {time.perf_counter()-t0:6.1f}s", flush=True)
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        ppo.save(a.save)
        print("saved", a.save)


if __name__ == "__main__":
    main()
