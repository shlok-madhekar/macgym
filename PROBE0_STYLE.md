# Probe0-style pass (binding; supersedes DESLOP.md and GLASS.md). The user's own Probe0 desktop app is the reference.
User: "you know the probe0 app UI? make it inspired off that. make it into an electron app (for both slides and main app and all), based on the probe0 design style. an actual window too." Submission closes 10 PM tonight; be fast and don't break what works.

Reference source (READ, it is his own code): ~/Projects/proxiscope/apps/desktop/src/renderer (App.tsx, components/chrome/*, components/primitives/*, views/Overview.tsx, styles/*) and src/main/window.ts.
Already copied for you into ~/macgym/web/probe0/: tokens.css, base.css, buttons.css, inputs.css, surfaces.css, shell.css, toggles.css, feel.css, chrome/*.css, fonts/Geist-Variable.woff2, fonts/GeistMono-Variable.woff2 (fix the @font-face url() paths in base.css to ./fonts/...; delete @imports of sheets that were not copied). Served by the MacGym server at http://localhost:8765/probe0/...

What "probe0 style" means (from his tokens.css / shell.css comments):
- A real native macOS window: titleBarStyle hiddenInset, traffic lights at {x:19,y:13}, vibrancy 'sidebar', visualEffectState 'active', backgroundColor '#00000000'. Vibrant translucent SIDEBAR drawn by AppKit + ONE opaque content pane (var(--window-bg)). html/body/sidebar transparent. NO backdrop-filter, no CSS glass, no gradients.
- Geist for UI, Geist Mono for values/code. Body 13px/16px. Native density: control height 28px, 4px grid, max 44px gaps. Titles 17px/22px semibold. No giant marketing type inside the app.
- Monochrome: accent is near-white in dark (var(--accent)); labels var(--label-primary/secondary/tertiary); hairlines var(--separator). No green, no orange. A single semantic color only where meaning needs it.
- FORCE DARK: set <html data-theme="dark"> everywhere (user asked for dark mode).
- Use his class names/patterns from the copied CSS (buttons, inputs, surfaces, sidebar rows with 36px height and selection pill) rather than inventing new ones.
