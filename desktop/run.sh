#!/bin/bash
# Launch MacGym as a native window.
#   ./run.sh         open the packaged MacGym.app if it exists, else run from source
#   ./run.sh --dev   always run main.js from this folder with the shared Electron
DIR="$(cd "$(dirname "$0")" && pwd)"
ELECTRON="${MACGYM_ELECTRON:-$HOME/Projects/proxiscope/apps/desktop/node_modules/.bin/electron}"

if [ "$1" != "--dev" ] && [ -d "$DIR/MacGym.app" ]; then
  # keep the bundle's copy of the main process in sync with the source
  cp "$DIR/main.js" "$DIR/package.json" "$DIR/MacGym.app/Contents/Resources/app/" 2>/dev/null
  exec open "$DIR/MacGym.app"
fi

[ "$1" = "--dev" ] && shift
if [ ! -x "$ELECTRON" ]; then
  echo "Electron not found at $ELECTRON (set MACGYM_ELECTRON)" >&2
  exit 1
fi
exec "$ELECTRON" "$DIR" "$@"
