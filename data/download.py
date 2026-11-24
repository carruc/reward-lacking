"""Download the Kaggle competition dataset into artifacts/dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub

COMPETITION = "an2dl2627"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "dataset"


def download_dataset(
    competition: str = COMPETITION,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> Path:
    """Download the latest competition dataset files into ``output_dir``.

    Preserves a ``.gitkeep`` placeholder if one exists in ``output_dir`` so the
    tracked directory structure stays intact in Git.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in output_dir.iterdir() if p.name != ".gitkeep"]
    if existing and not force:
        print(
            f"Dataset files already present in {output_dir} "
            "(pass --force to re-download)."
        )
        print("Path to competition files:", output_dir)
        return output_dir

    gitkeep = output_dir / ".gitkeep"
    had_gitkeep = gitkeep.exists()

    # kagglehub rejects any non-empty output_dir (including one with only .gitkeep)
    # when force_download=False, so temporarily remove .gitkeep before downloading.
    if had_gitkeep and not existing:
        gitkeep.unlink()

    try:
        path = kagglehub.competition_download(
            competition,
            output_dir=str(output_dir),
            force_download=force,
        )
    finally:
        if had_gitkeep:
            output_dir.mkdir(parents=True, exist_ok=True)
            gitkeep.touch()

    print("Path to competition files:", path)
    return Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the dataset download script."""
    parser = argparse.ArgumentParser(
        description="Download the Kaggle competition dataset into artifacts/dataset.",
    )
    parser.add_argument(
        "--competition",
        default=COMPETITION,
        help=f"Kaggle competition slug (default: {COMPETITION}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory for dataset files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-downloading and replacing existing files in output-dir.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    download_dataset(
        competition=args.competition,
        output_dir=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
