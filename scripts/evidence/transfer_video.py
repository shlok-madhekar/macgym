"""Render episode 0 of the MuJoCo transfer rollout (saved qpos frames, 25 fps) to data/transfer.mp4."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mj_walker import ROOT, mjcf
import numpy as np, mujoco
frames = np.load(os.path.join(ROOT, "runs/_mj_transfer_frames.npy"))
m = mujoco.MjModel.from_xml_string(mjcf()); d = mujoco.MjData(m)
r = mujoco.Renderer(m, 540, 960)
cam = mujoco.MjvCamera(); cam.distance = 4.0; cam.elevation = -5; cam.azimuth = 90
ff = subprocess.Popen(["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", "960x540", "-r", "25", "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264",
                       os.path.join(ROOT, "web/deck/data/transfer.mp4")], stdin=subprocess.PIPE)
for q in frames:
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    cam.lookat[:] = [d.qpos[0], 0, 0.8]
    r.update_scene(d, cam); ff.stdin.write(r.render().tobytes())
ff.stdin.close(); ff.wait(); print("frames", len(frames))
