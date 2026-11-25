#!/bin/sh
# Serve the dataset gallery from the repo root (default port 8000):
#   ./data/gallery/run.sh [port]
# -> http://localhost:8000/data/gallery/
cd "$(dirname "$0")/../.." || exit 1

# The manifest is generated (it is not tracked): rebuild it if missing.
if [ ! -f "data/gallery/manifest.json" ]; then
    uv run build-gallery
fi

exec uv run python -m http.server "${1:-8000}"
