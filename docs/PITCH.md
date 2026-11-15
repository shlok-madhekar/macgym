# MacGym live pitch (60 seconds)

Have the Playground open with a walker run already training on the projector. Talk over it.

## Script

> (pointing at the screen) That's 4,096 robots learning to walk, live, on this MacBook. No NVIDIA card, no cloud.
>
> Every serious robot-learning simulator needs NVIDIA. But most students and hackers have a MacBook, and its GPU just sits there.
>
> So I built MacGym, a robot-learning framework for Apple Silicon. A robot is about 15 lines of Python. (tripod preset, add a 4th hip to `HIPS`, Cmd+Enter) I just gave this one a fourth leg. MacGym compiles that into its own Metal kernel, one GPU thread per robot, and trains it with PPO in MLX on the same chip.
>
> Through the normal API that's about 3.2 million env-steps a second with 4,096 walkers, and about 270 thousand a second including training. One command exports the policy to CoreML or ONNX, verified to under one in a million.
>
> It's pip installable, with a CLI, docs and a Gymnasium wrapper. It's for students learning RL and anyone iterating on a robot before paying for cloud GPUs. Your laptop is already a robot-learning rig. MacGym turns it on.

(About 160 spoken words, about 60 seconds at a brisk, natural pace. Stage directions in parentheses aren't spoken.)

## Likely judge questions

**1. Is that a real gait, or is it exploiting your physics?**
It's simplified planar physics, so I won't claim it's a real-world gait. But we specifically check for the classic exploits: we measure stance-foot slip, so a foot that's supposed to be planted can't slide, and we tuned contact friction so gaits don't skate. Joint limits and torque caps keep the motion plausible.

**2. Does any of this transfer?**
Out of MacGym, yes: CoreML and ONNX exports match the original network to under 1e-6, so the policy you ship is the policy you trained. To a real robot, not yet, that's future work. The next step is 3D bodies and a sim-to-sim test: train here, run in MuJoCo, see what survives.

**3. Who actually uses this?**
Students, hackers and researchers on MacBooks who can't get an NVIDIA GPU. That's a lot of people, since the MacBook is the default laptop in CS classrooms and at hackathons. Use cases: learning RL, and fast morphology or reward iteration locally before scaling on cloud.

**4. Your benchmark says you're 97.2x faster than MuJoCo. Is that fair?**
Not apples to apples, and we say so on the page. Both are planar, but MuJoCo uses generalized coordinates and a convex contact solver, we use maximal coordinates and a simpler impulse solver, and our baseline drives MuJoCo from Python. The ratio also swings a lot because the CPU baseline drops when the laptop is busy; a quieter run earlier today gave about 21x. The number I trust most is the end-to-end one: 272K env-steps a second of actual training on a laptop. And we measure through the public API, not a hand-tuned kernel.

**5. Why was walker the same speed as cheetah at small sizes?**
At around a thousand robots the GPU isn't the bottleneck. Every step pays a fixed Python and MLX dispatch cost no matter how heavy the robot is. Scale up to 16,384 and that cost gets spread out, the GPU saturates, and you start seeing the real physics cost.
