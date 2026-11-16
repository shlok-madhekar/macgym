# Dark + Liquid Glass pass (binding, supersedes DESLOP colors). 10 MINUTES. Submission closes 10 PM.
User: "make it dark mode and use liquid glass and feel authentic. the text placement / UI/UX also feels AI. fix QUICKLY."

Authentic = it should look like a real native macOS 26 (Tahoe) app / an Apple keynote, not a web dashboard.

## Tokens
- Background #000000 (app window base #0B0B0D). Text primary rgba(255,255,255,.96), secondary rgba(255,255,255,.55), tertiary rgba(255,255,255,.3).
- One accent: macOS dark-mode system green #30D158. Nothing else colored.
- Fonts: -apple-system, "SF Pro Display", "SF Pro Text", system-ui for EVERYTHING (headlines too). Code/console only: "SF Mono", ui-monospace, Menlo. Remove Google font loads you no longer use.
- Type like Apple: semibold (600), tight tracking (-0.03em on large sizes, -0.01em on body), sentence case, no all-caps anywhere.
- Liquid Glass material (use for floating panels/capsules only, never for full-width bars):
  background: rgba(40,40,44,.42); backdrop-filter: blur(30px) saturate(180%); -webkit-backdrop-filter: same;
  border: 1px solid rgba(255,255,255,.14); border-radius: 22px (capsules: 999px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.22), inset 0 -1px 0 rgba(255,255,255,.04), 0 18px 50px rgba(0,0,0,.5);
  Glass must sit OVER moving content (robots/particles) so the blur is visible. No gradient blobs, no glows, no neon.
- Text placement rule: stop the "eyebrow + stacked headline + paragraph, top-left" template. Left-aligned, generous margins, one statement per screen, supporting text is small and sits away from the headline (not stacked under it).
