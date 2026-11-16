# Deck v2 review round (4 reviewers: motion designer, typographer, 3 technical judges, 3 gut/story judges). Apply these.

## Decisions from the integrator (binding)
- Accuracy fixes:
  - Pendulum: our period is +0.008% from the exact analytic answer (MuJoCo also matches). Say "0.01% from the textbook answer", NOT "vs MuJoCo".
  - 22× must carry "same planar 2D walker · M3 Pro · 4,096 robots · vs MuJoCo C rollout on 11 CPU threads" in small text ON the slide.
  - 90%: "19.7 m vs 22.0 m, sim-to-sim (MuJoCo), 20 runs" on the slide.
  - Hardware spec "Apple M3 Pro, 18 GB" visible on slides 3/6/8.
  - Do NOT add any user counts or revenue claims. Remove "No users or revenue yet" from A5 and replace with the real plan: "First users: DVHS CS Club (I'm an officer), my hackathons (Cove Hacks, HackPilot)".
  - Bring one honest authorship line into the main deck (slide 9): "Built today with AI coding agents. I designed the API and directed and tested it."
  - Repo URL is https://github.com/shlok-madhekar/macgym (QR + text). Deck will be hosted publicly at a Vercel URL with /deck; L key should open the playground only when on localhost, otherwise open the GitHub repo.

## Motion (designer)
1. Slide 2→3 dive: ~1.5 s of grey blob noise. Hold headline until 60% of the camera move, cap point size ~40% in the dive, 0.9 s ease-in-out, kill the ghost "MACGYM" tracking.
2. Slide 1 holds too long (8 s identical): dur ~5.5 s, reveal sub line at 2.5 s, slow 3° camera track.
3. Slide 5: stagger counter then rows at 0.15 s; row type ≥14 px or cut to 2 rows; limit ghosts to 2 (35% / 15%), dark robot leads, old ghosts slide off left.
4. Slide 6: fixed-width numerals with decimal drawn from the start, easeOutExpo landing; stick-figure row: 6 robots, 2.5× larger, with margins (or remove).
5. Slide 7: remove "honest gait" tag; body ≥20 px; tidy red trail.
6. Slide 8: no counting through misleading intermediate values: fade/snap in or slow ramp; align three stats on one grid; captions ≥14 px.
7. Slide 9: terminal: text ≥16 px, lighter box (not a flat black slab), count lands on 87 without flashing 75.
8. Slide 4: replace the smeared-code "glitch" with a clean typewriter-out or mask wipe (≤0.25 s smear).
9. Variety: don't use the same dissolve-to-dust on every cut. Use 3-4 distinct transition styles (morph-stream, camera dive, wipe/scan, hard cut on beat).
10. Minimum text size 14 px at 1080p; drop repeated footers/corner tags except a tiny pager.

## Typography / copy
- One eyebrow baseline and one headline top across all slides; one left edge for all text.
- Pager must never collide with content (clear zone).
- Rule: gray = setup line, orange = payoff, max one orange element per slide.
- Slide 1 sub: "An ordinary MacBook. / A robot-learning lab." (no arrow)
- Slide 2: "EVERY LAPTOP HERE HAS A GPU." gray "FOR ROBOTS, IT SITS IDLE." body "Fast robot simulators need NVIDIA. Apple's GPU has no maintained robot-RL backend."
- Slide 3: "MACGYM." body "Write a robot in Python. Train 4,096 copies on your Mac. About a minute." tag "PLANAR 2D, BY CHOICE". Don't let the dot field overlap the headline.
- Slide 4: clear the area behind the headline (walker overlaps "TO"); show all 3 beats clearly.
- Slide 5: remove extra overlays; one aligned row: "0:00 falls · 0:10 falls · 0:30 15.1 m · 1:00 21.5 m"; keep "5 of 5 seeds" as one small line.
- Slide 6: caption "4,096 robots. Live counter from the playground." Plus a plain line for non-technical people: "Days of practice, squeezed into seconds."
- Slide 7: give the cheating story the full slide: "MY FIRST ROBOTS CHEATED." "6.16 → 2.74 m/s" body "Unlimited motor power. Feet that skated. Both fixed." Trim the strike line to the text.
- Slide 8: retitle (avoid "It's real physics" cliché), e.g. "CHECKED AGAINST THE PRO SIMULATOR." Fixed column widths for the 3 stats, pendulum not clipping, footnote "n=20 · MuJoCo 4.0.1 · sim-to-sim, not hardware". Plain-English sub-labels (non-technical judges): "matches the textbook", "faster than the industry standard", "still walks in the pro simulator".
- Slide 9: replace "NOT A DEMO. A LIBRARY." cliché with "ONE INSTALL. A REAL LIBRARY." or similar; drop iter noise line; 87 s in orange as the single big number; spacing between terminal and 87 s.
- Slide 10: drop the eyebrow; footer "github.com/shlok-madhekar/macgym"; QR finders in black (orange may not scan); test the QR decodes (e.g. with your repl browser + BarcodeDetector or python if available).
- Add a "what it's for" line somewhere early (slide 3 or 5): "Robots that learn to walk, grasp and balance: for RL classes, games and real robot arms."
- Appendix: remove the faint gray circle; consistent layout; wrap labels cleanly; A6 "Mine:" instead of "I decided".

## Story (gut judges)
- Show first, explain second: consider opening slide 1 directly on the robot falling then running (the learning arc) before the title slams.
- Cheating twist and the 3.2-days counter are the strongest beats; keep them punchy.
