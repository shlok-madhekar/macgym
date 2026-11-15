"""End-to-end test of the v2 server messages (REDESIGN.md server sections 1-4).

    python3 scripts/test_server_v2.py        # starts its own server on port 8790
"""
import asyncio
import json
import os
import subprocess
import sys
import time

import aiohttp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import socket


def _port_free(p):
    with socket.socket() as so:
        return so.connect_ex(("127.0.0.1", p)) != 0


# 8790 by default; on this machine the Probe0 desktop app may already own localhost:8790
PORT = int(os.environ.get("MACGYM_TEST_PORT", 8790))
while not _port_free(PORT):
    print(f"port {PORT} busy, trying {PORT + 1}", flush=True)
    PORT += 1
URL = f"http://127.0.0.1:{PORT}/ws"
HUMAN_KEYS = ("human_value", "human_best", "human_unit", "human_label", "higher_is_better", "train_seconds")
FAILS = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def short(msg, n=260):
    s = json.dumps(msg)
    return s if len(s) <= n else s[:n] + "...(%d bytes)" % len(s)


async def recv_until(ws, pred, timeout, sink=None):
    end = time.time() + timeout
    while time.time() < end:
        try:
            m = await ws.receive(timeout=max(0.05, end - time.time()))
        except asyncio.TimeoutError:
            break
        if m.type != aiohttp.WSMsgType.TEXT:
            if m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
            continue
        msg = json.loads(m.data)
        if sink is not None:
            sink.append((time.time(), msg))
        if msg.get("type") == "error":
            print("  server error:", msg.get("message"), flush=True)
        if pred(msg):
            return msg
    return None


async def test_walker(session):
    print("== walker N=4096 render 256", flush=True)
    log = []
    async with session.ws_connect(URL, max_msg_size=1 << 24) as ws:
        await ws.send_json({"cmd": "start", "env": "walker", "num_envs": 4096})
        await ws.send_json({"cmd": "render", "render_envs": 256})
        t_start = time.time()
        r0 = await recv_until(ws, lambda m: m.get("type") == "replay", 60, log)
        check(r0 is not None and r0["label"] == "0:00", "replay 0:00 received")
        if r0:
            print("  replay 0:00:", short({k: v for k, v in r0.items() if k != "frames"}), flush=True)
            check(r0["iter"] == 0, "replay 0:00 is before any update (iter 0)")
            check(bool(r0.get("result_text")), f"result_text: {r0.get('result_text')!r}")
            check(len(r0["frames"]) >= 2 and len(r0["frames"][0]) == r0["K"] * 5, "replay frames are flat K*5")
            print(f"  replay 0:00 cost {r0['cost_ms']} ms (includes first-time compile)", flush=True)
        t_train0 = time.time()
        r10 = await recv_until(ws, lambda m: m.get("type") == "replay" and m.get("label") == "0:10", 40, log)
        check(r10 is not None, "replay 0:10 received")
        if r10:
            print(f"  replay 0:10 arrived {time.time() - t_train0:.1f}s after 0:00:",
                  short({k: v for k, v in r10.items() if k != "frames"}), flush=True)
            check(bool(r10.get("result_text")) and r10["iter"] > 0, "0:10 has result_text and iter>0")
            check(r10["cost_ms"] < 300, f"replay 0:10 cost {r10['cost_ms']} ms < 300")
        # let it train a bit more past the replay to measure throughput
        await recv_until(ws, lambda m: False, 4, log)

        metrics = [m for _, m in log if m.get("type") == "metrics"]
        frames = [(t, m) for t, m in log if m.get("type") == "frame"]
        infos = [m for _, m in log if m.get("type") == "info"]
        check(len(metrics) > 5, f"{len(metrics)} metrics messages")
        if metrics:
            m = metrics[-1]
            check(all(k in m for k in HUMAN_KEYS), "metrics have generic human fields")
            check(all(k in m for k in ("dist_mean", "dist_best", "unit", "headline_label")), "metrics have locomotion fields")
            print("  last metrics:", short({k: m.get(k) for k in ("iter", "sps", "mean_ep_return") + HUMAN_KEYS +
                                             ("dist_mean", "dist_best", "unit", "headline_label", "human_source")}, 600), flush=True)
            hv = [x.get("human_value") for x in metrics]
            print("  human_value series (first, mid, last):", hv[0], hv[len(hv) // 2], hv[-1], flush=True)
        if infos:
            print("  info:", short(infos[-1]), flush=True)
        big = [m for _, m in frames if m.get("R") == 256]
        check(len(big) > 3, f"{len(big)} frames with R=256")
        if big:
            f = big[-1]
            caps = np.array(f["capsules"]).reshape(f["R"], f["K"], 5)
            tx = caps[:, 0, [0, 2]].mean(1)
            frac = float(np.mean((tx > -1) & (tx < 30)))
            check("dist" in f and len(f["dist"]) == 256 and "best_index" in f, "frame has dist[256] and best_index")
            check(frac > 0.9, f"start-relative torso x in [-1,30] for {frac:.0%} of robots (min {tx.min():.2f}, max {tx.max():.2f})")
            check(int(np.argmax(f["dist"])) == f["best_index"], "best_index = argmax(dist)")
            nbytes = len(json.dumps(f))
            check(nbytes < 120000, f"frame size {nbytes} bytes")
            span = frames[-1][0] - frames[0][0]
            print(f"  frame rate {len(frames) / max(span, 1e-6):.1f} Hz", flush=True)
            print("  frame:", short({k: v for k, v in f.items()}, 400), flush=True)
        # throughput: PPO's own sps around the replay vs. effective env-steps/wall
        if r10 and len(metrics) > 6:
            sps = [x["sps"] for x in metrics[2:]]
            print(f"  train sps median {np.median(sps):,.0f}", flush=True)
            mt = [(t, m) for t, m in log if m.get("type") == "metrics"]
            t_r = next(t for t, m in log if m.get("type") == "replay" and m.get("label") == "0:10")
            def eff(a, b):
                seg = [(t, m) for t, m in mt if a <= t <= b]
                if len(seg) < 2:
                    return None
                return (seg[-1][1]["env_steps"] - seg[0][1]["env_steps"]) / (seg[-1][0] - seg[0][0])
            pre = [m["sps"] for t, m in mt if t_r - 4 <= t < t_r]
            post = [m["sps"] for t, m in mt if t_r < t <= t_r + 4]
            gaps = np.diff([t for t, _ in mt])
            gap_r = max((t2 - t1) for (t1, _), (t2, _) in zip(mt, mt[1:]) if t1 <= t_r <= t2 + 0.05)
            print(f"  PPO sps median 4s before replay {np.median(pre):,.0f}, 4s after {np.median(post):,.0f}; "
                  f"metrics gap across replay {gap_r * 1000:.0f} ms vs median gap {np.median(gaps) * 1000:.0f} ms", flush=True)
            overhead = (r10["cost_ms"] / 1000.0) / 10.0  # replay cost / replay interval
            print(f"  replay overhead {overhead:.1%} of wall time over the 0:00->0:10 interval (less later)", flush=True)
            check(overhead < 0.10 and np.median(post) > 0.9 * np.median(pre), "training sps drop from replays < 10%")
        await ws.send_json({"cmd": "stop"})
        rn = await recv_until(ws, lambda m: m.get("type") == "replay" and m.get("label") == "now", 20)
        check(rn is not None and bool(rn.get("result_text")), "stop -> replay 'now'")
        if rn:
            print("  replay now:", short({k: v for k, v in rn.items() if k != "frames"}), flush=True)
    # reconnect: replays of the current run are re-sent
    async with session.ws_connect(URL, max_msg_size=1 << 24) as ws2:
        got = []
        await recv_until(ws2, lambda m: False, 2, got)
        labels = [m["label"] for _, m in got if m.get("type") == "replay"]
        check(labels[:3] == ["0:00", "0:10", "now"], f"replays re-sent on connect: {labels}")


async def test_small(session, env, extra_keys):
    print(f"== {env} N=1024", flush=True)
    log = []
    async with session.ws_connect(URL, max_msg_size=1 << 24) as ws:
        await ws.send_json({"cmd": "start", "env": env, "num_envs": 1024, "render_envs": 256})
        r0 = await recv_until(ws, lambda m: m.get("type") == "replay" and m.get("env") == env, 60, log)
        check(r0 is not None and r0["label"] == "0:00", f"{env} replay 0:00")
        if r0:
            print("  replay:", short({k: v for k, v in r0.items() if k != "frames"}), flush=True)
        await recv_until(ws, lambda m: m.get("type") == "metrics" and m.get("iter", 0) >= 15, 40, log)
        metrics = [m for _, m in log if m.get("type") == "metrics"]
        frames = [m for _, m in log if m.get("type") == "frame"]
        if metrics:
            m = metrics[-1]
            check(all(k in m for k in HUMAN_KEYS + extra_keys), f"{env} metrics human fields")
            print("  metrics:", short({k: m.get(k) for k in HUMAN_KEYS + extra_keys + ("human_source",)}, 500), flush=True)
        else:
            check(False, f"{env} metrics received")
        if frames:
            f = frames[-1]
            check(f.get("dist") is not None and "best_index" in f, f"{env} frame dist/best_index (R={f['R']})")
        await ws.send_json({"cmd": "stop"})
        await recv_until(ws, lambda m: m.get("type") == "replay" and m.get("label") == "now", 15)


async def main():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as session:
        for _ in range(60):  # wait for the server to accept websockets
            try:
                async with session.ws_connect(URL) as ws:
                    await ws.receive(timeout=5)
                break
            except Exception:
                await asyncio.sleep(0.5)
        await test_walker(session)
        await test_small(session, "arm", ("err_mean", "unit"))
        await test_small(session, "cartpole", ("balance_mean", "unit"))


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    srv = subprocess.Popen([sys.executable, "-m", "macgym.server", "--port", str(PORT)], cwd=ROOT,
                           stdout=open(os.path.join(ROOT, "logs", "test_server_v2.log"), "w"),
                           stderr=subprocess.STDOUT)
    try:
        time.sleep(3.0)
        asyncio.run(main())
    finally:
        srv.terminate()
        srv.wait(timeout=10)
    print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)}): {FAILS}", flush=True)
    sys.exit(1 if FAILS else 0)
