#!/bin/sh
# Serve the dataset gallery plus its label API (default port 8000):
#   ./data/gallery/run.sh [port]
# -> http://localhost:8000/data/gallery/
# Right-click labels persist to data/gallery/annotations.json; the header's
# "→ train.csv" button merges them via uv run apply-annotations.
cd "$(dirname "$0")/../.." || exit 1

if [ ! -f "data/gallery/manifest.json" ]; then
    uv run build-gallery || echo "warning: manifest not built (dataset missing?) — the page shows details on load" >&2
fi

exec uv run serve-gallery "${1:-8000}"
