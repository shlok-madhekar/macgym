# Xcode pass (binding, on top of PROBE0_STYLE.md). 10 MINUTES. Submission closes 10 PM.
User: "this is good but make the main window more transparent, it feels mid, it needs to feel more like Xcode. also the UI elements aren't responsive and get kinda fried and look a bit odd."

## Contract between shell (web/shell.*, desktop/main.js) and playground iframe (web/playground.*)
- shell -> iframe:  iframe.contentWindow.postMessage({type:'macgym:cmd', cmd:'run'|'stop'|'setCopies', value?:number}, '*')
- iframe -> shell:  parent.postMessage({type:'macgym:status', view:'playground', state:'idle'|'building'|'training', robot:'tripod', detail:'9 bodies, 8 motors', copies:4096, headline:'2.9 days of practice', sub:'in 0:19', best:'12.4 m'}, '*')
  posted once on load, on every state change, and about once a second while training.
- In ?embed=1 the playground HIDES its own toolbar row (the shell's window toolbar owns Run/Stop/copies/status). Standalone (no embed) keeps its own toolbar.
- Transparency: in ?embed=1 the inner pages set html/body background: transparent, so the window material shows through. The shell's content pane provides the tint.

## Xcode feel
- Unified window toolbar (52px, translucent, draggable; controls are no-drag): right after the sidebar edge: Run (play triangle) and Stop (square) icon buttons, 28px, borderless toolbar style with hover fill; then a "scheme" control: "tripod  >  4,096 copies" (copies opens a small menu/select: 1,024 / 4,096 / 16,384); CENTER: the Xcode activity view: a rounded 28px-high field (min 280, max 460px) with subtle fill, one line 12px: idle "MacGym | Ready", building "Building tripod...", training "Training tripod: 2.9 days of practice in 0:19" with a thin 2px progress shimmer at its bottom while training.
- Editor area: a 28px jump bar above the code ("MacGym > robots > tripod.py", 11px secondary, chevrons), then the code; bottom "debug area" = console with a 28px header bar (hairline above, "Console" 11px secondary, clear button), resizable feel.
- Right side = the "canvas" (like Xcode's SwiftUI preview canvas): slightly different tint, robots track on top, Progress and Generations beneath it.
- Hairline separators (var(--separator)) between panes, never boxes/cards. 13px body, 11px secondary labels. Every control has hover + pressed states (feel.css).

## Responsive (must not break): test content-pane sizes 864x648, 1044x748, 1204x848, 1492x930 (and full window widths 1100..1728)
- CSS grid/flex with minmax(); no fixed pixel widths for panes; canvases sized with ResizeObserver + devicePixelRatio and redrawn on resize; text never overlaps or clips mid-word (ellipsis for long labels); below ~980px content width hide the Generations column (keep Progress); below ~760px collapse the debug area. Sidebar stays 236px (min 180).
