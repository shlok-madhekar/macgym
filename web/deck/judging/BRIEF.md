# Judge brief

You are judging at Bay-Valley Hacks 2026 (NorCal high-school hackathon, ~400 hackers, 12-hour build, theme "ELEVATE THE ORDINARY"). Rubric (equal weight, score each 1-10): Creativity & Originality, Technical Difficulty, Functionality, UI/UX, Theme Relevance, Presentation & Pitch. Prizes: 1st/2nd/3rd overall, Best Solo, Best AI/ML, Best Fintech, Best Hardware.

Project: MacGym, built solo today by a 16-year-old. Robot-learning framework for Apple Silicon: define a robot in Python, MacGym compiles it into a Metal GPU kernel, trains 4,096 copies in parallel with PPO in Apple's MLX in about a minute, exports policies to CoreML/ONNX. Includes CLI, Gymnasium wrapper, browser code Playground with live training view, docs site. Physics is 2D (planar).

You are shown the PITCH DECK (the thing being refined): static slide images in /Users/shlok/macgym/web/deck/export/ (slide-NN.png, contact.png, anim-*.png mid-animation frames — the deck has a morphing halftone-dot particle animation between slides) and the speaker notes in /Users/shlok/macgym/web/deck/NOTES.md. Look at EVERY slide image with your read_file tool. Optionally open the live deck at http://localhost:8765/deck/index.html with your repl browser tool (arrow keys to advance) to see animations; close the tab after.

Judge each persona independently and honestly, in character. Don't be nice by default: a typical project gets 5-7. Score the PROJECT as presented by this deck + notes (as if this were the live pitch).

Return ONLY a JSON array (no prose outside it), one object per persona:
{"judge_id":int,"scores":{"creativity":n,"technical":n,"functionality":n,"uiux":n,"theme":n,"pitch":n},"total":n,"rank_guess":"1st|top3|top10|unplaced","first_impression":"<=20 words, in character","best_slide":int,"worst_slide":int,"confusing":["specific thing that confused or bored you"],"doubts":["claims you don't believe or want proof of"],"questions":["question you'd ask the presenter"],"fixes":[{"slide":int|null,"change":"concrete, specific change that would raise your score","criterion":"which rubric item it raises","impact":1-5}]}
