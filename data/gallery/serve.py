"""Development server for the gallery: static files, label persistence, and cluster API.

Extends the plain ``http.server`` used by the gallery with a tiny JSON API so
the webapp can persist its right-click labels to ``annotations.json``,
merge them into ``train.csv``, and serve/recluster multi-backbone embeddings:

    GET  /api/annotations                  -> {"<image id>": ["<label>", ...], ...}
    POST /api/annotations                  body: {"<image id>": ["<label>", ...], ...}
    POST /api/apply                        -> rewrites train.csv with the extra label column
    GET  /api/dataset-root                 -> {"root": "/artifacts/dataset"}
    GET  /api/clusters/meta                -> cluster catalog + sample arrays + spaces
    GET  /api/clusters/space?name=<id>     -> 2D coords, HDBSCAN clusters, preds, summaries
    GET  /api/clusters/neighbors?space=..  -> top-k cosine neighbors in PCA-50 space
    POST /api/clusters/recluster           -> live HDBSCAN re-clustering on 2D or PCA-k

Static files are still served from the repository root, exactly like
``python -m http.server``, so the gallery URLs do not change.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

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
DEFAULT_CLUSTERS_DIR = REPO_ROOT / "artifacts" / "clusters"
# URL path of the dataset as served from the repo root (the handler's directory)
DATASET_URL_PATH = "/" + DEFAULT_TRAIN_CSV.parent.relative_to(REPO_ROOT).as_posix()

_WRITE_LOCK = threading.Lock()


class _ClusterStore:
    """Thread-safe lazy cache for cluster metadata, view JSON bytes, and PCA-50 arrays."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._meta_dir: Path | None = None
        self._meta: dict | None = None
        self._meta_bytes: bytes | None = None
        self._valid_spaces: set[str] = set()
        self._space_view_bytes: dict[str, bytes] = {}
        self._space_arrays: dict[str, dict[str, np.ndarray]] = {}
        self._overlap_bytes: bytes | None = None

    def _ensure_meta(self, clusters_dir: Path) -> tuple[dict | None, bytes | None]:
        meta_path = clusters_dir / "meta.json"
        with self._lock:
            if self._meta_dir != clusters_dir:
                self._meta_dir = clusters_dir
                self._meta = None
                self._meta_bytes = None
                self._valid_spaces = set()
                self._space_view_bytes.clear()
                self._space_arrays.clear()
                self._overlap_bytes = None

            if self._meta is None and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["available"] = True
                self._meta = meta
                self._meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
                self._valid_spaces = {str(sp["id"]) for sp in meta.get("spaces", [])}
            return self._meta, self._meta_bytes

    def get_meta_bytes(self, clusters_dir: Path) -> bytes:
        meta, meta_bytes = self._ensure_meta(clusters_dir)
        if meta is None or meta_bytes is None:
            missing = {
                "available": False,
                "hint": "Run uv run build-clusters to generate embeddings and clusters.",
            }
            return json.dumps(missing).encode("utf-8")
        return meta_bytes

    def validate_space(self, clusters_dir: Path, space_id: str) -> bool:
        self._ensure_meta(clusters_dir)
        return space_id in self._valid_spaces

    def get_view_bytes(self, clusters_dir: Path, space_id: str) -> bytes | None:
        if not self.validate_space(clusters_dir, space_id):
            return None
        with self._lock:
            cached = self._space_view_bytes.get(space_id)
        if cached is not None:
            return cached
        view_path = clusters_dir / space_id / "view.json"
        if not view_path.exists():
            return None
        raw = view_path.read_bytes()
        with self._lock:
            self._space_view_bytes[space_id] = raw
        return raw

    def get_arrays(self, clusters_dir: Path, space_id: str) -> tuple[dict, dict[str, np.ndarray]] | None:
        meta, _ = self._ensure_meta(clusters_dir)
        if meta is None or space_id not in self._valid_spaces:
            return None
        with self._lock:
            cached = self._space_arrays.get(space_id)
        if cached is not None:
            return meta, cached

        sdir = clusters_dir / space_id
        if not (sdir / "pca50.npy").exists() or not (sdir / "tsne_2d.npy").exists():
            return None

        pca50_raw = np.load(sdir / "pca50.npy").astype(np.float32)
        norms = np.linalg.norm(pca50_raw, axis=1, keepdims=True)
        pca50_norm = (pca50_raw / np.maximum(norms, 1e-12)).astype(np.float32)
        coords_2d = np.load(sdir / "tsne_2d.npy").astype(np.float32)
        probs = np.load(sdir / "probs.npy").astype(np.float32)
        arrays = {
            "pca50_raw": pca50_raw,
            "pca50_norm": pca50_norm,
            "coords_2d": coords_2d,
            "probs": probs,
            "pred": probs.argmax(axis=1).astype(np.int32),
            "conf": probs.max(axis=1).astype(np.float32),
        }
        with self._lock:
            self._space_arrays[space_id] = arrays
        return meta, arrays

    def get_overlap_bytes(self, clusters_dir: Path) -> bytes:
        self._ensure_meta(clusters_dir)
        with self._lock:
            cached = self._overlap_bytes
        if cached is not None:
            return cached
        overlap_path = clusters_dir / "overlap.json"
        if not overlap_path.exists():
            missing = {
                "available": False,
                "hint": "Run uv run eval-overlap to compute cross-model k-NN and CKA overlap.",
            }
            return json.dumps(missing).encode("utf-8")
        payload = json.loads(overlap_path.read_text(encoding="utf-8"))
        payload["available"] = True
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        with self._lock:
            self._overlap_bytes = raw
        return raw


_CLUSTER_STORE = _ClusterStore()


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
    clusters_dir: Path = DEFAULT_CLUSTERS_DIR

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/api/annotations":
            self._send_json(read_annotations(self.annotations_path))
        elif path == "/api/dataset-root":
            # Tell the page where the images actually are, relative to this
            # server's root (the repo root), instead of letting it guess.
            self._send_json({"root": DATASET_URL_PATH})
        elif path == "/api/clusters/meta":
            self._send_raw_json(_CLUSTER_STORE.get_meta_bytes(self.clusters_dir))
        elif path == "/api/clusters/space":
            self._handle_cluster_space(parsed.query)
        elif path == "/api/clusters/neighbors":
            self._handle_cluster_neighbors(parsed.query)
        elif path == "/api/clusters/overlap":
            self._send_raw_json(_CLUSTER_STORE.get_overlap_bytes(self.clusters_dir))
        else:
            super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if path == "/api/annotations":
            self._handle_save(body)
        elif path == "/api/apply":
            annotated = apply_annotations(self.annotations_path, self.train_csv, self.column)
            self._send_json({"ok": True, "annotated": annotated, "column": self.column})
        elif path == "/api/clusters/recluster":
            self._handle_recluster(body)
        else:
            self._send_json({"error": f"unknown endpoint: {path}"}, status=404)

    def _handle_cluster_space(self, query: str) -> None:
        params = parse_qs(query)
        space_id = (params.get("name") or [""])[0]
        if not space_id:
            self._send_json({"error": "missing 'name' query parameter"}, status=400)
            return
        raw = _CLUSTER_STORE.get_view_bytes(self.clusters_dir, space_id)
        if raw is None:
            self._send_json({"error": f"unknown or missing cluster space '{space_id}'"}, status=404)
            return
        self._send_raw_json(raw)

    def _handle_cluster_neighbors(self, query: str) -> None:
        params = parse_qs(query)
        space_id = (params.get("space") or [""])[0]
        try:
            idx = int((params.get("index") or [""])[0])
            k = int((params.get("k") or ["48"])[0])
        except ValueError:
            self._send_json({"error": "invalid integer 'index' or 'k'"}, status=400)
            return

        loaded = _CLUSTER_STORE.get_arrays(self.clusters_dir, space_id)
        if loaded is None:
            self._send_json({"error": f"unknown or missing cluster space '{space_id}'"}, status=404)
            return
        meta, arrays = loaded
        n_samples = int(meta["n_samples"])
        if idx < 0 or idx >= n_samples:
            self._send_json({"error": f"index {idx} out of bounds [0, {n_samples - 1}]"}, status=400)
            return

        k = max(1, min(200, min(k, n_samples - 1)))
        pca_norm = arrays["pca50_norm"]
        sims = pca_norm @ pca_norm[idx]
        sims[idx] = -2.0
        top_unsorted = np.argpartition(-sims, k)[:k]
        top_idx = top_unsorted[np.argsort(-sims[top_unsorted])]

        probs = arrays["probs"][idx]
        self._send_json(
            {
                "space": space_id,
                "index": idx,
                "id": meta["samples"]["ids"][idx],
                "probs": [round(float(p), 4) for p in probs],
                "neighbors": [{"index": int(i), "sim": round(float(sims[i]), 4)} for i in top_idx],
            }
        )

    def _handle_recluster(self, body: bytes) -> None:
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)
            return

        space_id = str(payload.get("space", ""))
        loaded = _CLUSTER_STORE.get_arrays(self.clusters_dir, space_id)
        if loaded is None:
            self._send_json({"error": f"unknown or missing cluster space '{space_id}'"}, status=404)
            return
        meta, arrays = loaded

        cluster_on = str(payload.get("cluster_on", "2d"))
        scope = str(payload.get("scope", "all"))
        method = str(payload.get("method", "eom"))
        if cluster_on not in ("2d", "pca"):
            self._send_json({"error": "cluster_on must be '2d' or 'pca'"}, status=400)
            return
        if scope not in ("all", "train", "clean", "test"):
            self._send_json({"error": "scope must be 'all', 'train', 'clean', or 'test'"}, status=400)
            return
        if method not in ("eom", "leaf"):
            self._send_json({"error": "method must be 'eom' or 'leaf'"}, status=400)
            return

        try:
            min_cluster_size = max(5, min(2000, int(payload.get("min_cluster_size", 80))))
            min_samples = max(1, min(500, int(payload.get("min_samples", 15))))
            raw_max = payload.get("max_cluster_size", 3000)
            if raw_max is None or int(raw_max) <= 0:
                max_cluster_size = None
            else:
                max_cluster_size = max(50, min(20000, int(raw_max)))
            pca_dims = max(2, min(50, int(payload.get("pca_dims", 15))))
        except (TypeError, ValueError):
            self._send_json(
                {"error": "min_cluster_size, min_samples, max_cluster_size, and pca_dims must be integers"},
                status=400,
            )
            return

        from data.gallery.build_clusters import run_hdbscan_and_summarize

        t0 = time.perf_counter()
        res = run_hdbscan_and_summarize(
            coords_2d=arrays["coords_2d"],
            pca50=arrays["pca50_raw"],
            samples_meta=meta["samples"],
            pred=arrays["pred"],
            conf=arrays["conf"],
            cluster_on=cluster_on,
            scope=scope,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            max_cluster_size=max_cluster_size,
            method=method,
            pca_dims=pca_dims,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self._send_json({"ok": True, "space": space_id, "elapsed_ms": elapsed_ms, **res})

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
        self._send_raw_json(json.dumps(payload, separators=(",", ":")).encode("utf-8"), status=status)

    def _send_raw_json(self, data: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
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
    clusters_dir: Path = DEFAULT_CLUSTERS_DIR,
) -> None:
    """Run the gallery + annotation + cluster API until interrupted."""
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
    GalleryHandler.clusters_dir = clusters_dir
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
    parser.add_argument(
        "--clusters-dir",
        type=Path,
        default=DEFAULT_CLUSTERS_DIR,
        help=f"Directory containing cluster JSON/NPY artifacts (default: {DEFAULT_CLUSTERS_DIR}).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    serve(args.host, args.port, args.annotations, args.train_csv, args.column, args.clusters_dir)


if __name__ == "__main__":
    main()
