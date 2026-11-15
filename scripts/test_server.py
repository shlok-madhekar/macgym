"""Integration test for macgym.server: subprocess + websocket round trip.

    python3 scripts/test_server.py [--env cartpole] [--port 8799]
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import aiohttp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def recv_until(ws, pred, timeout):
    """Collect messages until pred(msg) is True; returns (match, all_msgs)."""
    seen = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            m = await ws.receive(timeout=max(0.1, deadline - time.time()))
        except asyncio.TimeoutError:
            break
        if m.type != aiohttp.WSMsgType.TEXT:
            raise RuntimeError(f"ws closed/err: {m.type}")
        msg = json.loads(m.data)
        seen.append(msg)
        if msg.get("type") == "error":
            print("  server error:", msg.get("message"))
        if pred(msg):
            return msg, seen
    raise TimeoutError(f"timed out; last types: {[s.get('type') for s in seen[-10:]]}")


def check_frame(f, R_expected):
    R, K = f["R"], f["K"]
    assert R == R_expected, (R, R_expected)
    assert K >= 1
    assert len(f["capsules"]) == R * K * 5, (len(f["capsules"]), R, K)
    assert all(isinstance(v, (int, float)) for v in f["capsules"])
    if f["targets"] is not None:
        assert len(f["targets"]) % (R * 2) == 0
    if f["colors"] is not None:
        assert len(f["colors"]) == K
    assert f["world_bounds"] is None or len(f["world_bounds"]) == 4
    if f.get("camera_x") is not None:
        assert len(f["camera_x"]) == R


async def run(port, env_name, n, R):
    url = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession() as s:
        for _ in range(100):
            try:
                async with s.get(url + "/") as r:
                    assert r.status == 200
                    break
            except aiohttp.ClientConnectorError:
                await asyncio.sleep(0.2)
        else:
            raise RuntimeError("server did not come up")

        async with s.ws_connect(url + "/ws") as ws:
            st, _ = await recv_until(ws, lambda m: m["type"] == "status", 5)
            print("initial status:", st)

            await ws.send_json({"cmd": "list"})
            envs, _ = await recv_until(ws, lambda m: m["type"] == "envs", 30)
            names = [e["name"] for e in envs["envs"]]
            print("envs:", names)
            for e in envs["envs"]:
                assert {"name", "obs_dim", "act_dim", "description"} <= set(e)
            if env_name not in names:
                print(f"{env_name} unavailable, using stub")
                env_name = "stub"

            await ws.send_json({"cmd": "bench"})
            b, _ = await recv_until(ws, lambda m: m["type"] == "bench", 5)
            print("bench present:", b["data"] is not None)

            await ws.send_json({"cmd": "start", "env": env_name, "num_envs": n, "render_envs": R})
            await recv_until(ws, lambda m: m["type"] == "status" and m["state"] == "training", 60)

            metrics, frames = [], []
            deadline = time.time() + 90
            while (len(metrics) < 3 or len(frames) < 3) and time.time() < deadline:
                m, _ = await recv_until(ws, lambda m: m["type"] in ("metrics", "frame"), 90)
                (metrics if m["type"] == "metrics" else frames).append(m)
            assert len(metrics) >= 3 and len(frames) >= 3, (len(metrics), len(frames))
            for mm in metrics:
                for k in ("iter", "env_steps", "sps", "num_envs"):
                    assert k in mm, (k, mm)
                assert mm["num_envs"] == n
            assert metrics[-1]["env_steps"] > metrics[0]["env_steps"]
            for f in frames:
                assert f["env"] == env_name
                check_frame(f, R)
            print("metrics sample:", json.dumps(metrics[-1]))
            f = frames[-1]
            print("frame sample: R=%d K=%d capsules[:10]=%s colors=%s ground=%s wb=%s" % (
                f["R"], f["K"], f["capsules"][:10], f["colors"], f["ground"], f["world_bounds"]))

            # showcase: frames should keep coming at ~30 Hz with no metrics
            await ws.send_json({"cmd": "showcase", "on": True})
            await recv_until(ws, lambda m: m["type"] == "status" and m["state"] == "showcase", 30)
            t0 = time.time()
            sc = 0
            while sc < 15:
                m, _ = await recv_until(ws, lambda m: m["type"] == "frame", 10)
                check_frame(m, R)
                sc += 1
            print(f"showcase: 15 frames in {time.time() - t0:.2f}s")

            await ws.send_json({"cmd": "set_target", "x": 0.3, "y": 0.7})
            await ws.send_json({"cmd": "save"})
            sv, _ = await recv_until(ws, lambda m: m["type"] in ("saved", "error") and m.get("path", "x") is not None, 30)
            print("save ->", sv)

            await ws.send_json({"cmd": "stop"})
            await recv_until(ws, lambda m: m["type"] == "status" and m["state"] == "idle", 30)
            print("stopped OK")

        # second client connects + disconnects abruptly; server must survive
        ws2 = await s.ws_connect(url + "/ws")
        await ws2.close()
        async with s.ws_connect(url + "/ws") as ws3:
            await ws3.send_json({"cmd": "list"})
            await recv_until(ws3, lambda m: m["type"] == "envs", 10)
    print("PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cartpole")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--render-envs", type=int, default=16)
    a = ap.parse_args()
    proc = subprocess.Popen([sys.executable, "-m", "macgym.server", "--port", str(a.port), "--host", "127.0.0.1"],
                            cwd=ROOT)
    try:
        asyncio.run(run(a.port, a.env, a.num_envs, a.render_envs))
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
