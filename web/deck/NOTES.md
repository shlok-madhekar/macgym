# MacGym deck: speaker notes

Open: http://localhost:8765/deck/index.html (served by the MacGym server). Keys: → / space / click = next, ← = back, F = fullscreen, P = notes overlay, Home/End. Hash routing #1..#14. Slide 4 has 2 steps (one robot, then the grid).
Static export view: `?static=1&slide=N`. Exports live in `export/` (slide-NN.png, anim-NN-[abc].png, contact.png).

Benchmark numbers on slide 9 are read live from /bench.json (generated 2026-11-15 14:40:43 PST, speedups {'cartpole': 241.2, 'cheetah': 64.2, 'walker': 66.0}).

Every robot on slides 1, 4, 6 and 7 is drawn from real frames in site/replay.json (walker, from-scratch run). Slide 6 illustrates the tripod stat with walker replays (we have no tripod replay). Slide 10's strider legs are procedurally swung (illustration of the snippet's geometry), slide 8's legs are a schematic.

## 01. Bay-Valley Hacks · Elevate the Ordinary: MacGym

This robot taught itself to run. Nobody animated it. It is a real replay from a training run on this laptop, about a minute in. I am Shlok, and this is MacGym.

## 02. The problem: Every robot sim / needs an NVIDIA GPU. / Your Mac isn’t / invited.

If you want to train robots today, every serious simulator assumes an NVIDIA card. Isaac Lab, MuJoCo Warp, all CUDA. The Mac path for JAX got called dead by a maintainer this January. The best community port does about five thousand steps a second. That is nothing.

## 03. The ordinary laptop: The laptop you / already own / has a GPU. / Nobody uses it / for this.

Meanwhile, the most ordinary computer at this hackathon is a MacBook, and it has a real GPU sitting there idle. Unified memory, so the CPU and GPU share the same RAM. Nobody uses it for robot learning. That felt like the most ordinary thing in the room to elevate.

## 04. So I built: MacGym

So I built MacGym. You describe a robot in a few lines of Python. MacGym compiles a GPU kernel for it, and then it is not one robot, it is four thousand ninety six of them, all learning at once on the laptop. When it is done you export the policy to CoreML.

## 05. How it works: Python in. / Metal kernel out.

Here is the pipeline. Fifteen lines of Python. MacGym writes a Metal kernel for that exact robot, one GPU thread per robot copy. PPO runs in MLX right on the same memory, so nothing gets copied between CPU and GPU. Out the other end you get a CoreML or ONNX file.

## 06. The number: 3.2 days of / experience. / 43 seconds.

This is the number I care about. In the live playground, 4,096 tripods racked up three point two days of robot experience in forty three seconds of wall time. Average distance went from slightly backwards to over seven metres in under a minute. On a laptop.

## 07. It learns: From falling / to running. / In one minute.

These are real replays from one run, not animations. At zero seconds it face plants. Ten seconds in, it lunges and falls. Thirty seconds, it is running fifteen metres. One minute, twenty one and a half metres in eight seconds. Every dot is a real recorded frame.

## 08. It can’t cheat: Real motors / slow down. / Ours do too.

My first robots were way too fast, like six metres a second. That is the policy exploiting the physics: infinite motor power and feet that skate. RL will cheat if you let it. So I added warm-started contact impulses and a real DC motor torque-speed curve. Now speeds are believable and foot slip dropped about fourteen times.

## 09. Fast: 66×

Speed. Against MuJoCo running on all eleven CPU cores of this same laptop, MacGym generates way more robot experience per second. I want to be honest: it is not the same solver. Ours is simpler, maximal coordinates with impulses. But for learning to walk on a laptop, this is the number that matters.

## 10. It’s a framework: Not a demo. / A library.

And it is a real framework, not a one off demo. This is the whole robot definition: bodies, joints, a reward, train. There is a CLI, a Gymnasium wrapper, docs, and CoreML export that matches the original policy to within about four ten-millionths.

## 11. Built today: One day. / One laptop.

All of this was built today. About eighty one hundred lines, forty two hundred in the Python package. Seven robots, tests, five examples. The git history starts at eleven thirty eight this morning.

## 12. Who it’s for: Anyone with / a MacBook. / Which is / everyone here.

Who is this for? Students learning reinforcement learning without paying for cloud GPUs. Hackers who want a walking robot by Sunday. Researchers who want to iterate on body shape and reward locally before they rent a cluster.

## 13. What’s next: From the sim / to a real arm.

Next: 3D physics with the same API, a sim-to-sim check against MuJoCo so you can trust the policies, and the fun one: a CoreML policy running on an iPhone driving a real SO-101 robot arm.

## 14. Close: Elevate the ordinary.

Your laptop is the most ordinary computer you own. MacGym makes it a robot gym. Thank you.
