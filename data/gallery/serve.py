"""Development server for the gallery: static files plus label persistence.

Extends the plain ``http.server`` used by the gallery with a tiny JSON API so
the webapp can persist its right-click labels to ``annotations.json`` and
merge them into ``train.csv``:

    GET  /api/annotations  -> {"<image id>": ["<label>", ...], ...}
    POST /api/annotations  body: {"<image id>": ["<label>", ...], ...}
    POST /api/apply        -> rewrites train.csv with the extra label column

Static files are still served from the repository root, exactly like
``python -m http.server``, so the gallery URLs do not change.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from data.gallery.apply_annotations import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_COLUMN,
    DEFAULT_TRAIN_CSV,
    apply_annotations,
)
from data.gallery.build_manifest import build_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
GALLERY_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = GALLERY_DIR / "manifest.json"
# URL path of the dataset as served from the repo root (the handler's directory)
DATASET_URL_PATH = "/" + DEFAULT_TRAIN_CSV.parent.relative_to(REPO_ROOT).as_posix()

_WRITE_LOCK = threading.Lock()


def read_annotations(path: Path = DEFAULT_ANNOTATIONS) -> dict[str, list[str]]:
    """Return the saved annotations as ``{image id: [labels]}``."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_annotations(annotations: dict[str, list[str]], path: Path = DEFAULT_ANNOTATIONS) -> None:
    """Persist the annotations atomically (temp file + replace)."""
    with _WRITE_LOCK:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(annotations, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)


class GalleryHandler(SimpleHTTPRequestHandler):
    """Serve the repo root statically, with ``/api`` endpoints on top."""

    annotations_path: Path = DEFAULT_ANNOTATIONS
    train_csv: Path = DEFAULT_TRAIN_CSV
    column: str = DEFAULT_COLUMN

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/annotations":
            self._send_json(read_annotations(self.annotations_path))
        elif path == "/api/dataset-root":
            # Tell the page where the images actually are, relative to this
            # server's root (the repo root), instead of letting it guess.
            self._send_json({"root": DATASET_URL_PATH})
        else:
            super().do_GET()

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if path == "/api/annotations":
            self._handle_save(body)
        elif path == "/api/apply":
            annotated = apply_annotations(self.annotations_path, self.train_csv, self.column)
            self._send_json({"ok": True, "annotated": annotated, "column": self.column})
        else:
            self._send_json({"error": f"unknown endpoint: {path}"}, status=404)

    def _handle_save(self, body: bytes) -> None:
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)
            return
        if not isinstance(data, dict) or not all(isinstance(tags, list) for tags in data.values()):
            self._send_json({"error": "expected {image id: [labels, ...]}"}, status=400)
            return
        annotations = {str(k): sorted({str(t) for t in tags}) for k, tags in data.items()}
        write_annotations(annotations, self.annotations_path)
        tagged = sum(1 for tags in annotations.values() if tags)
        self._send_json({"ok": True, "tagged": tagged})

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        """Keep the default access log but tag it, so API calls stand out."""
        print(f"[server] {self.address_string()} {format % args}", flush=True)


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    train_csv: Path = DEFAULT_TRAIN_CSV,
    column: str = DEFAULT_COLUMN,
) -> None:
    """Run the gallery + annotation API until interrupted."""
    if not MANIFEST_PATH.exists():
        try:
            count = build_manifest(output=MANIFEST_PATH)
            print(f"built {MANIFEST_PATH} ({count} entries)", flush=True)
        except Exception as err:  # dataset likely missing; the page explains this
            print(
                f"warning: could not build the manifest automatically ({err}).\n"
                "         Once the dataset is downloaded (uv run download-dataset), run uv run build-gallery.",
                flush=True,
            )
    GalleryHandler.annotations_path = annotations_path
    GalleryHandler.train_csv = train_csv
    GalleryHandler.column = column
    server = ThreadingHTTPServer((host, port), GalleryHandler)
    print(f"Serving {REPO_ROOT} at http://localhost:{port}/data/gallery/ (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print("\nstopped", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?", type=int, default=8000, help="port to listen on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: 127.0.0.1)")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help=f"Path to the annotations file (default: {DEFAULT_ANNOTATIONS}).",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help=f"train.csv updated by POST /api/apply (default: {DEFAULT_TRAIN_CSV}).",
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help=f"Name of the annotation column (default: {DEFAULT_COLUMN}).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    serve(args.host, args.port, args.annotations, args.train_csv, args.column)


if __name__ == "__main__":
    main()
