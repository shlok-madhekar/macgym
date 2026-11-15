# MacGym redesign spec (v2) — legibility first

User verdict on v1: "looks like Claude slop", "can't tell if it's doing anything, looks like a random number generator". Rule for v2: a viewer must SEE learning happen without reading a number. Every number shown must be in human units.

## New server -> client messages (additive; keep existing ones working)
1. metrics gains human-unit fields (computed over the last ~100 finished episodes, same window as mean_ep_return):
   - locomotion (walker, cheetah): "dist_mean" (m, torso x at episode end minus start x), "dist_best" (m, max in window), "unit":"m", "headline_label":"distance walked"
   - arm: "err_mean" (m, mean |ee - target| over the last 20 steps of finished episodes) , "unit":"cm"
   - cartpole: "balance_mean" (s, ep_len*dt), "unit":"s"
   - all: "train_seconds" (wall seconds since start of run)
   Generic fields the client can use for any env: "human_value" (float, in display unit: meters / centimeters / seconds), "human_best", "human_unit" ("m"|"cm"|"s"), "human_label" (e.g. "average distance walked", "average miss distance", "average time balanced"), "higher_is_better" (bool).
2. frames for the TRACK view: server sends R robots (client requests render_envs 256 for locomotion via {"cmd":"render","render_envs":256}). For locomotion, capsule x coords must be in a frame where each robot's episode START is x=0 (i.e. subtract per-env start x), so all robots can be overlaid on one shared runway. Add "dist": [R] current distance per robot, and "best_index": index of the robot with largest dist. For arm: all arms share base at origin already; include targets. For cartpole: x as is.
3. NEW "replay" message: at iteration 0 (untrained, BEFORE any update) and then at training wall time ~10 s, 30 s, 60 s, 120 s, 240 s (and on "showcase"/stop as "now"), run the CURRENT policy deterministically on a separate fresh small env instance (num_envs=1 for loco/cartpole, 1 for arm with a fixed target sequence) for up to 400 control steps (stop early at done), record capsules every 2nd step:
   {"type":"replay","env":..., "label":"0:00"|"0:10"|..., "train_seconds":s, "iter":i, "K":K, "frames":[[flat K*5 floats], ...], "targets":[[x,y],...] or null, "result": human value (meters reached / seconds balanced / final cm error), "result_text": e.g. "fell after 0.4 s · 0.2 m" or "ran 18.3 m in 8 s", "dt": control dt*2}
   Replays must not disturb training env state (separate env object, same class). Keep cost < 300 ms.
   On client connect, re-send all replays of the current run.
4. "info" sim_sps stays.

## Visual direction (frontend) — NOT the generic AI dashboard look
Banned: dark glassmorphism, neon glows, gradients, blurred panels, pill buttons with icons, emoji, "LIVE" badges, giant gradient numbers, cards-in-cards, rounded-2xl everything.
Direction: technical drawing / lab instrument / Swiss print. 
- Background warm paper #F3F0E8, ink #121212, rules #D9D3C7 (1px), muted ink #6B665C, ONE accent: signal orange #FF4D00 (best robot + current value), plus a quiet blue-gray #8A94A6 for older generations.
- Type: "Inter Tight" (600/700) for headings and big numbers with tabular-nums, "JetBrains Mono" 11-12px uppercase-tracked for small labels. Square corners (0-2px radius). Hairline 1px borders. Generous whitespace.
- Layout (1600x900 and 1920x1080, no scrolling):
  * Top bar (56px): wordmark "MacGym" + one-line subtitle "Train robots on a Mac GPU"; env tabs as plain text tabs with underline (Walker, Cheetah, Arm, Cartpole); right: robot-count select (1,024 / 4,096 / 16,384) and ONE primary button "Train" / "Stop" (solid ink, square). Secondary text button "Load trained".
  * HERO TRACK (~58% height): a long ground line with tick marks every 1 m labeled 0 m, 5 m, 10 m..., camera auto-scales so 0 m at left and the furthest robot + 15% margin fits (smoothly animated). All R=256 robots drawn overlaid as thin ink strokes (alpha ~0.12-0.18, capsule round caps), the best robot drawn solid orange on top with joints as small dots. Two vertical markers with labels: dashed ink "avg 6.2 m" and orange "best 14.8 m". Before training starts, show a calm state text "4,096 robots ready. Press Train." For arm: 256 arms overlaid from one base, target dot(s), orange best arm; markers replaced by "avg miss 3.1 cm". Cartpole: overlaid carts/poles, "avg balanced 7.4 s".
  * LOWER BAND (~32%): left 55%: PROGRESS chart: big headline value + label at top-left (e.g. "6.2 m  average distance walked", value in Inter Tight 56px, unit smaller), line chart of human_value vs train_seconds (x axis in m:ss), ink line, orange current dot, hairline grid, labeled axes. Right 45%: GENERATIONS strip: up to 5 small replay panels side by side (oldest -> newest) each looping its replay frames on its own little ground, label "0:00", "0:30"... in mono, and result_text under it ("fell · 0.2 m", "ran 18.3 m"). Newest panel has orange outline. Empty slots show a faint dashed placeholder "after 0:30".
  * Footer line (24px, mono, muted): "4,096 robots in parallel · Apple M3 Pro GPU (MLX) · sim 2.9M steps/s · training 284K steps/s · iteration 52" — honest small print.
- Motion: smooth 60fps interpolation between frames; no flashing, no bouncy UI animations.
- Must also run with ?demo=1 (fake data that shows learning: robots progressively going further, replays improving).
