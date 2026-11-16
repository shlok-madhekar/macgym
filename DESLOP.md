# De-slop pass (binding). Deadline is tight: the hackathon submission closes at 10 PM and the user still has to record.

User verdict: "the app itself overall gives claude slop vibes, and kinda the slides too."

Why it reads that way (fix exactly these, by CUTTING, not rewording):
1. Cream paper + orange accent is literally the Claude/Anthropic look. Kill it.
2. Tiny helper notes everywhere explaining things ("where all 256 shown robots are right now", "one robot, fresh start, no exploration noise", "8 s replay", status strips, spec lines).
3. Mono type used for labels, captions and footers; uppercase tracked eyebrow labels; chains of " · " separators.
4. Pill tags / chips, boxed sections, hairline rules around everything, pagers like "07 / 10".
5. Too many things on screen at once. He wants: big, plain, confident, few elements.

## New visual system (both app and slides)
- Background pure white #FFFFFF. Text #0A0A0A. Secondary text #8A8A8A. Dividers #EDEDED (use sparingly; prefer whitespace over lines).
- ONE accent: green #1F9D55 (his own Syro deck used white/black/green). Used for exactly one thing per screen (the best robot / the key number). No orange, no cream, anywhere.
- Fonts: "Geist" for all UI and body text, "Geist Mono" ONLY for code and terminal text (never for labels/captions). Load from Google Fonts (families "Geist" and "Geist Mono"), fall back to local("Geist") then system-ui. Deck headlines stay Bebas Neue.
- Sentence case everywhere. No uppercase tracked labels. No " · " separator chains. No pills/chips/tags. Radius 8px on buttons/inputs, no borders on panels (whitespace separates them).
- Every number that stays must be big and have one plain label of 2-4 words. Delete every caption that explains another caption.
