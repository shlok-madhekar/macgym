# MacGym deck: speaker notes (round 3)

Open http://localhost:8765/deck/index.html. Keys: → / space / click = next, ← = back, F = fullscreen, P = notes overlay, **L = live playground in a new tab**, Home/End. Hash routing #1..#15. Slide 3 has 2 steps.
Static export: `?static=1&slide=N`. Exports in `export/` (slide-NN.png, anim-NN-[abc].png, contact.png). Evidence files in `data/` are re-polled every 12 s.

Spine: an ordinary MacBook (slide 2: the laptop on every desk, GPU idle) becomes a robot-learning lab (slide 15: the same laptop full of robots, 7.8 days of robot practice in 1:06).

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

### 06. The number · simulated robot time: 3.2 days of / experience. / 28 seconds.

That recording, frozen at twenty eight seconds: 4,096 tripods have already racked up three point two days of experience. The math is on the slide: robots times steps times the 0.02 second timestep. That is roughly half a million steps a second, with learning included.

### 07. It learns: From falling / to running. / In one minute.

These are real replays from one run. At zero seconds it falls. Ten seconds, it lunges and falls. Thirty seconds, fifteen metres. One minute, twenty one and a half metres in eight seconds. And so you know it is not one lucky run: I retrained from scratch with five seeds, and all five learned to walk within a minute.

### 08. The twist: My first robots / cheated. / Here’s how / we caught it.

My favorite part. The first walkers ran six metres a second. That was RL cheating: infinite motor power and feet that skated. We added a foot-slip metric to catch it, then fixed the physics: contacts that remember last step’s push, and a real DC motor curve. Now the walker does 2.7 metres a second and slip dropped fourteen times.

### 09. Is it real physics?: Checked against / MuJoCo.

Is it real physics? We ran the same setups in MuJoCo. Pendulum period within a hundredth of a percent of the textbook, the swing within a sixth of a degree of MuJoCo, the friction stop within millimetres, the drop lands on the same frame. And the hard one, a walker hitting the ground with contacts, which tracks MuJoCo for a while and then diverges, I will tell you exactly when.

### 10. Fast · same 2D walker model, MuJoCo on all 11 CPU cores: 60×

One number. Same walker model, MuJoCo on all eleven cores of this laptop, versus MacGym on the GPU. Where does it come from? Mostly from compiling each robot into one Metal kernel, about thirty times, and about four times from the GPU itself. And honestly, training is limited by the PPO update now, not physics. Our physics is simpler: planar 2D.

### 11. It’s a framework: Not a demo. / A library.

And it is a real library. This is real terminal output: macgym train hopper, four thousand robots, forty iterations, seventeen seconds, distance climbing from zero to three metres. Then one command exports it to CoreML, checked against the original to within four ten-millionths.

### 12. Built today: One day. / One laptop.

Built today. The first commit was at eleven thirty eight. Eighteen checks pass. And to be straight: I built it solo with AI coding agents. I picked the idea, designed the API and the experiments, and directed the agents. The cheating robots are an example: the agents were happy with six metres a second, I was not.

### 13. Who it’s for: RL classes / on MacBooks.

The wedge is students. My first users are the DVHS Computer Science Club, where I am an officer, and the hackathons I run, Cove Hacks and HackPilot. Why not Colab? This runs offline on the laptop you already own, no quotas, no disconnects, and you watch the robots learn live. It is open source; later a hosted classroom version and edu licenses.

### 14. Done · next: Policy transfer: / measured.

Done and next. We took the walker policy trained in MacGym and ran it inside MuJoCo, and that number is on the slide. Next: 3D with the same API, then a CoreML policy on an iPhone driving a real SO-101 arm. Today it is 2D only, and I want to be upfront about that.

### 15. Close: Elevate the ordinary.

The laptop on your desk is the most ordinary computer you own. MacGym turns it into a robot-learning lab: almost eight days of robot practice in a minute. Scan the code, it is all on GitHub. Thank you.

## Q&A appendix (round-2 judge questions)

**Where does the speed come from? Is the CPU baseline naive?**
Each GPU thread runs one robot's whole control step in a single generated Metal kernel: 8 substeps x 4 solver iterations, motors, joint constraints and ground contacts, with no trips back to memory between operations. The fair_bench baseline runs the same algorithm as vectorized MLX array ops. At 16,384 walkers: fused kernel 16.3M steps/s, same ops on the GPU 498K, same ops on the CPU 135K. So the GPU alone is about 3.7x and fusing into one kernel is about 33x. The CPU baseline is vectorized but not hand-tuned multithreaded C, which is why the headline is the same-walker MuJoCo comparison instead (60x vs MuJoCo on 11 processes, 7.6M vs 126K steps/s; MuJoCo via its standard Python mj_step loop).

**Contacts per thread?** Contacts are resolved inside the same thread: each capsule endpoint against the ground, a non-negative normal impulse plus a Coulomb-clamped friction impulse, iterated 4 times per substep. Ground-only contact, no self-collision, planar 2D.

**Why does the PPO update dominate training time?** Each iteration collects 98304 env-steps/iter and then runs 4 epochs x 4 minibatches of backprop + Adam through a 256x256 MLP. That is far more math per sample than stepping 7 capsules in 2D. Measured: PPO update 72%, physics 10%, end-to-end 295K steps/s.

**What did you write vs the agents?** Built solo today with AI coding agents. I picked the idea, designed the API (Robot / Task / make / train), decided which experiments to run (foot-slip metric, MuJoCo validation, policy transfer, seeds) and directed the agents. The agents wrote most of the code. The cheating robots are the clearest example: 6 m/s looked fine to the agents, not to me.

**Why not free Colab?** MacGym runs offline on the laptop you already own: no quotas, no disconnects, and a live viewer where you watch the robots learn. Colab is still great for big 3D work.

**First 10 users?** DVHS Computer Science Club (I'm an officer) and the Bay Area hackathons I organize (Cove Hacks, HackPilot). Plan: a club workshop and a hackathon track. No users yet; open source first, later a hosted classroom version and edu licenses.

**Warm-started impulses, in plain words?** Every contact remembers how hard it pushed last step and starts from that guess, so the solver settles in a few iterations and feet stop sliding. The DC-motor curve means a motor's torque drops as it spins faster, so the policy can't use infinite power. Together they took the walker from 6.16 to 2.74 m/s and cheetah foot slip from 0.735 to 0.053 m/s.

**Does the contact physics match MuJoCo?** Not exactly. On the walker itself (collapse, tilt, open-loop torques) MacGym tracks MuJoCo within 5 cm for the first 0.6 to 0.8 s, then diverges because the contact models differ (our hard impulses vs MuJoCo's soft constraints). Both end in the same pose.

**Do policies transfer?** Our trained walker policy, run unchanged in a MuJoCo mirror of the robot: 19.8 m vs 21.9 m in MacGym over 10 episodes of 8 s, 0 falls in either. Sim-to-real is not done.
