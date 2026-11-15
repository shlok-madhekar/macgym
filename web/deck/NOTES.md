# MacGym deck: speaker notes (final)

Open http://localhost:8765/deck/index.html. Keys: → / space / click = next, ← = back, F = fullscreen, P = notes overlay, **L = live playground in a new tab**, Home/End. Hash routing #1..#15. Slide 3 has 2 steps.
Static export: `?static=1&slide=N`. Exports in `export/`. Evidence in `data/` (re-polled every 12 s); `assets/pg_t30_counter.png` is a crop of data/pg_t30.png.

Spine: the ordinary laptop on every desk (slide 2, GPU idle) becomes a robot-learning lab (slide 15, same laptop full of robots, 7.8 days of practice in 1:06).

## Slides

### 01. Bay-Valley Hacks · Elevate the Ordinary: MacGym

Every one of you has an ordinary laptop with you. I turned mine into a robot-learning lab. This robot taught itself to run on this MacBook in about a minute, and its first versions cheated, which I will show you. I am Shlok, and this is MacGym.

### 02. The ordinary thing: The laptop on / every desk here / has a GPU. / It sits idle.

The most ordinary thing at this hackathon is the laptop on every desk. It has a real GPU, and for robot learning it sits idle, because the fast simulators are NVIDIA only and the Mac path for JAX was called dead this January. MuJoCo on the CPU exists, but there is no fast GPU path. MacGym changes that: this laptop did almost eight days of robot practice in about a minute.

### 03. So I built: MacGym

So I built MacGym. You write a robot in Python, and MacGym trains four thousand ninety six copies of it at once on the Mac GPU. Planar 2D physics, so think robot learning class, not Isaac Lab. Let me show you it running.

### 04. Live proof: It runs. / Right here.

This is a real screen recording, not an animation. I add a fourth leg right in the browser, MacGym recompiles the Metal kernel in a quarter of a second, and four thousand copies start learning on the laptop GPU. If you want, I will press L and do it live right now.

### 05. How it works: You describe it. / The laptop practices / 4,096 times at once.

You describe the robot, top right. The joint line with motor 80 becomes these lines of generated Metal: a torque that falls off with speed, like a real DC motor. Every robot copy is one GPU thread running the whole step, contacts included. PPO in MLX learns on the same memory, and you export to CoreML.

### 06. The number · simulated robot time: 3.2 days of / practice. / 28 seconds.

This is the real playground counter, frozen at twenty eight seconds: 4,096 tripods have already done three point two days of practice. Same run hits seven point eight days at one minute six. It is simulated robot time: robots times steps times the 0.02 second timestep.

### 07. It learns: From falling / to running. / In one minute.

These are real replays from one run. At zero seconds it falls. Ten seconds, it lunges and falls. Thirty seconds, fifteen metres. One minute, twenty one and a half metres in eight seconds. And so you know it is not one lucky run: I retrained from scratch with five seeds, and all five learned to walk within a minute.

### 08. The twist: My first robots / cheated. / Here’s how / we caught it.

My favorite part. The first walkers ran six metres a second. That was RL cheating: infinite motor power and feet that skated. We added a foot-slip metric to catch it, then fixed the physics: contacts that remember last step’s push, and a real DC motor curve. Now the walker does 2.7 metres a second and slip dropped fourteen times.

### 09. Checked against MuJoCo: Same tests as / the pro simulator. / Same answers.

Is it real physics? Same tests as the pro simulator, MuJoCo. The textbook cases match: pendulum period off by less than a hundredth of a percent, friction stop within millimetres, the drop lands on the same frame. The hard one is the walker hitting the ground: it tracks MuJoCo within five centimetres for the first six to eight tenths of a second, then the contact models differ, and both end in the same pose.

### 10. Fast · same 2D walker, MuJoCo’s fastest CPU path: 40×

One number. Same 2D walker model, against MuJoCo’s fastest CPU path, the C rollout module on all eleven threads: about forty times more practice per second, twenty two times at the 4,096 we train with. Where it comes from: the GPU alone is about 3.7 times, compiling each robot into one Metal kernel is another 33 times. Training is limited by the PPO update, not physics. And our physics is simpler, planar 2D.

### 11. It’s a framework: Not a demo. / A library.

And it is a real library. This is real terminal output: macgym train hopper, four thousand robots, forty iterations, seventeen seconds, distance climbing from zero to three metres. Then one command exports it to CoreML, checked against the original to within four ten-millionths.

### 12. Built today: One day. / One laptop.

Built today, since eleven thirty eight. What I decided: I picked this idea after rejecting about twenty others, made it a framework instead of a demo, and scrapped the first UI because you could not tell the robots were learning. The agents wrote most of the code; I reviewed, tested and directed it. The eighteen checks cover export round trips, the CLI, physics stepping for six robots, and parity with the old envs.

### 13. Who it’s for: RL classes / on MacBooks.

Who it is for: RL classes and clubs where everyone has a MacBook. First users are the DVHS Computer Science Club, where I am an officer, and my hackathons; next is a club workshop. Versus free Colab or MJX: it runs offline on the laptop you own, no GPU quota, no disconnects, a live viewer and a code playground, nothing to rent.

### 14. Done · next: Sim-to-sim: / transfers.

Sim-to-sim: the walker policy trained in MacGym, run unchanged in MuJoCo, keeps about ninety percent of its distance with zero falls, and the robustness rows show it under changed friction, mass and starts. This is MuJoCo, not real hardware. Next: MJCF import, 3D, and since the approach is compile a robot into one GPU kernel, it is backend agnostic: Metal today, CUDA or ROCm codegen is possible.

### 15. Close: Elevate the ordinary.

The laptop on your desk is the most ordinary computer you own. MacGym turns it into a robot-learning lab: almost eight days of robot practice in a minute. Scan the code, it is all on GitHub. Thank you.

## Q&A appendix

**Is 40× a fair number?** It is MacGym vs MuJoCo's fastest CPU path, the C rollout module on all 11 threads, on the same 2D walker model: 6.75M vs 165.1K env-steps/s (MuJoCo at its best batch size; MacGym at 16,384 envs) = 40.9×, and 22.0× at 4,096 envs. Physics is simpler on our side (planar 2D, impulse contacts, ground only).

**Where does the speed come from?** Same algorithm at 16,384 walkers: CPU array ops 135K → same ops on the GPU 498K (×3.7) → one compiled Metal kernel per robot 16.3M (×32.8) = 121×. Each GPU thread runs one robot's whole control step (8 substeps × 4 solver iterations, motors, joints, ground contacts) in one dispatch.

**Why not compare MJX or MuJoCo Warp?** Not run. They need an NVIDIA GPU (Warp) or a JAX GPU backend (MJX), and that is exactly the gap on a Mac. On this laptop MJX would run on CPU.

**Why does the PPO update dominate training?** 4 epochs × 4 minibatches of backprop + Adam through a 256×256 MLP per iteration is far more math per sample than stepping 7 capsules in 2D: PPO update 72%, physics 10%.

**Who pays?** Today nobody: it is open source and the first users are the DVHS Computer Science Club (I'm an officer) and my hackathons (Cove Hacks, HackPilot); next is a club workshop. Later: a hosted classroom version and edu licenses for RL / robotics courses. No users or revenue yet.

**ROCm / CUDA portability?** The approach is compile one robot into one GPU kernel. The generator emits Metal today; the same codegen could emit CUDA or HIP (ROCm). Not built yet.

**What did you personally decide?** I picked this idea after rejecting about 20 others, made it a framework instead of a demo, scrapped the first UI because you couldn't tell it was learning, and chose the experiments (foot-slip metric, ablation, MuJoCo validation, transfer, seeds). Agents wrote most of the code; I reviewed, tested and directed.

**What is the cheating ablation?** Warm-starting contacts alone stopped the skating (slip 0.063 m/s) but the walker still ran 6.64 m/s; the motor torque-speed curve was the real fix (walker 2.74 m/s). Speed = forward speed of the learned gait.

**Warm-started impulses, in plain words?** Each contact remembers how hard it pushed last step and starts there, so the solver settles in a few iterations. The DC-motor curve makes torque drop as a joint spins faster, so no infinite power.

**Does contact match MuJoCo?** On the walker (collapse, tilt, open-loop torques) it agrees within 5 cm for the first 0.6 to 0.8 s, then the contact models differ (hard impulses vs MuJoCo soft constraints); both end in the same pose.

**Does the policy transfer?** Sim-to-sim only (MacGym → MuJoCo, not hardware). n=20 per condition in MuJoCo: nominal 19.7 ± 0.9 m, 1/20 falls; friction_x0.8 19.7 ± 1.7 m, 1/20 falls; friction_x1.2 19.9 ± 0.6 m, 0/20 falls; mass_x0.8 19.9 ± 1.4 m, 1/20 falls; mass_x1.2 15.6 ± 5.2 m, 3/20 falls; init_random 20.1 ± 0.5 m, 0/20 falls. Mass +20% is the weak case.

**How long for a stranger?** Measured on a fresh venv: git clone → pip install -e . (10.6 s, warm pip cache) → macgym train walker (74.4 s) ≈ 87 s total, best training distance 54 m. Note: the clone was of the committed HEAD.
