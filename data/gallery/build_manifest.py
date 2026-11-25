"""Build the ``manifest.json`` consumed by the HTML dataset browser.

Scans ``artifacts/dataset`` for images in the ``train`` and ``test`` splits,
merges the training labels from ``train.csv``, and writes a compact
``manifest.json`` next to the gallery page (``index.html``).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "manifest.json"

SPLITS = ("train", "test")


def build_manifest(dataset_dir: Path = DEFAULT_DATASET_DIR, output: Path = DEFAULT_OUTPUT) -> int:
    """Scan ``dataset_dir`` and write the gallery manifest; return the entry count."""
    with (dataset_dir / "train.csv").open(newline="") as f:
        labels = {r["id"]: int(r["label"]) for r in csv.DictReader(f)}

    rows = []
    for split in SPLITS:
        for image in sorted((dataset_dir / split).glob("*.png")):
            rows.append({"id": image.stem, "split": split, "label": labels.get(image.stem)})

    output.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    return len(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the manifest builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing the dataset splits (default: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path of the generated manifest (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    count = build_manifest(args.dataset_dir, args.output)
    print(f"wrote {args.output}: {count} entries")


if __name__ == "__main__":
    main()
