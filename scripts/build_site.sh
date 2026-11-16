#!/usr/bin/env bash
# Assemble the static publish bundle in dist/ for Vercel (or any static host).
#   dist/            docs site (site/), landing page at /
#   dist/deck/       pitch deck (web/deck2/, no export/ renders)
#   dist/media/      web-encoded deck video + a real Playground recording
#   dist/data/       benchmark data referenced by the docs
# The live Playground needs the local Python server, so /playground links are
# rewritten to "Run locally" (quickstart) or to the recorded clip.
# Usage: bash scripts/build_site.sh   (from anywhere)
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/dist"
FFMPEG="${FFMPEG:-/opt/homebrew/bin/ffmpeg}"
[ -x "$FFMPEG" ] || FFMPEG="$(command -v ffmpeg || true)"

rm -rf "$DIST"
mkdir -p "$DIST/deck" "$DIST/media" "$DIST/data"

# 1. docs site at /
cp -R "$ROOT/site/." "$DIST/"

# 2. deck at /deck/ (skip big renders and internal spec/notes)
rsync -a --exclude 'export/' --exclude '*.md' --exclude '.DS_Store' \
  "$ROOT/web/deck2/" "$DIST/deck/"
# Pin relative URLs so /deck (no trailing slash, cleanUrls) still resolves assets.
if ! grep -q '<base href="/deck/">' "$DIST/deck/index.html"; then
  sed -i '' 's#<head>#<head>\n<base href="/deck/">#' "$DIST/deck/index.html"
fi

# 3. media
SRC_MP4="$ROOT/web/deck2/export/macgym-deck.mp4"
WEB_MP4="$ROOT/web/deck2/export/macgym-deck-web.mp4"
if [ ! -f "$WEB_MP4" ] && [ -f "$SRC_MP4" ] && [ -n "$FFMPEG" ]; then
  echo "encoding web copy of deck video (1280x720)..."
  "$FFMPEG" -y -loglevel error -i "$SRC_MP4" -vf "scale=1280:720:flags=lanczos,fps=30" \
    -c:v libx264 -preset slow -crf 33 -pix_fmt yuv420p -c:a aac -b:a 96k \
    -movflags +faststart "$WEB_MP4"
fi
[ -f "$WEB_MP4" ] && cp "$WEB_MP4" "$DIST/media/macgym-deck-web.mp4"
cp "$ROOT/web/deck/data/playground_tripod_4x.mp4" "$DIST/media/playground_tripod_4x.mp4"

# 4. data
cp "$ROOT/web/bench.json" "$DIST/data/bench.json"

# 5. rewrite local-only /playground links
for f in "$DIST"/*.html; do
  sed -i '' \
    -e 's#<a class="btn" href="/playground">Open Playground</a>#<a class="btn" href="quickstart.html">Run locally</a>#g' \
    -e 's#<a class="btn big" href="/playground">Try it in your browser</a>#<a class="btn big" href="media/playground_tripod_4x.mp4">Watch a real run</a>#g' \
    -e 's#<a class="arrow" href="/playground">Open the Playground and press Run &rarr;</a>#<a class="arrow" href="media/playground_tripod_4x.mp4">Watch a recorded Playground run \&rarr;</a>#g' \
    -e 's#<a href="/playground" class="">Playground</a>#<a href="quickstart.html\#train" class="">Playground (run locally)</a>#g' \
    -e 's#<a class="ln" href="/playground">Playground</a>#<a class="ln" href="media/playground_tripod_4x.mp4">Playground</a>#g' \
    -e 's#<a href="benchmarks.html">Benchmarks</a><a href="https://github.com#<a href="benchmarks.html">Benchmarks</a><a href="deck/">Deck</a><a href="https://github.com#g' \
    "$f"
done
if grep -l 'href="/playground' "$DIST"/*.html; then
  echo "error: unrewritten /playground links remain" >&2; exit 1
fi

# 6. host config
cp "$ROOT/vercel.json" "$DIST/vercel.json"

# 7. sanity: nothing over 50 MB
big=$(find "$DIST" -type f -size +50M)
[ -z "$big" ] || { echo "error: files over 50MB in dist: $big" >&2; exit 1; }
echo "dist ready: $(du -sh "$DIST" | cut -f1)"
