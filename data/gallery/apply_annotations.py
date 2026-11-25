"""Merge the gallery annotations into ``train.csv`` as an extra column.

``data/gallery/serve.py`` exposes this over ``POST /api/apply``; it is also a
CLI so annotations can be applied without running the server:

    uv run apply-annotations

Annotation ids that are not rows of ``train_csv`` (e.g. ``test_*`` ids) are
silently skipped, so the same ``annotations.json`` can also hold labels for
test images.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GALLERY_DIR = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS = GALLERY_DIR / "annotations.json"
DEFAULT_TRAIN_CSV = REPO_ROOT / "artifacts" / "dataset" / "train.csv"
DEFAULT_COLUMN = "extra_labels"


def apply_annotations(
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    train_csv: Path = DEFAULT_TRAIN_CSV,
    column: str = DEFAULT_COLUMN,
) -> int:
    """Write the annotation ``column`` into ``train_csv``; return rows updated.

    Every row gets the column: annotated rows carry their labels as a
    comma-separated string, all other rows an empty string. The rewrite is
    atomic (temp file + replace) and keeps the original line endings.
    """
    annotations: dict[str, list[str]] = {}
    if annotations_path.exists():
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))

    with train_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        raise ValueError(f"{train_csv} has no data rows")

    if column not in fieldnames:
        fieldnames.append(column)

    annotated = 0
    for row in rows:
        tags = annotations.get(row["id"])
        if tags:
            row[column] = ",".join(tags)
            annotated += 1
        else:
            row[column] = ""

    tmp = train_csv.with_suffix(train_csv.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(train_csv)
    return annotated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the annotation exporter."""
    parser = argparse.ArgumentParser(description=__doc__)
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
        help=f"Path to train.csv to update (default: {DEFAULT_TRAIN_CSV}).",
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help=f"Name of the annotation column to add (default: {DEFAULT_COLUMN}).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    annotated = apply_annotations(args.annotations, args.train_csv, args.column)
    print(f"wrote {args.train_csv}: annotated {annotated} rows in column '{args.column}'")


if __name__ == "__main__":
    main()
