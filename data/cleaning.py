"""Clean the training dataset and assign leakage-free stratified group folds.

Excludes:
1. Images tagged ``alien`` in ``extra_labels``.
2. Blank/starless frames (``std_gray < 5.0``).
3. Overexposed label-flipped copies whose outer-border star field matches a
   normal-exposure training image (border-ring cosine similarity > 0.88).

Remaining near-duplicates (border-ring cosine similarity > 0.90) are linked via
union-find so ``StratifiedGroupKFold`` never splits them across train/val.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = REPO_ROOT / "artifacts" / "dataset" / "train.csv"
DEFAULT_TRAIN_DIR = REPO_ROOT / "artifacts" / "dataset" / "train"
DEFAULT_CLEAN_FOLDS = REPO_ROOT / "artifacts" / "dataset" / "clean_folds.csv"
DEFAULT_DROPPED_CSV = REPO_ROOT / "artifacts" / "dataset" / "dropped_samples.csv"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "artifacts" / "ablations" / "cleaning_summary.json"


def _border_ring_mask(size: int = 64, inner: int = 32) -> np.ndarray:
    """Boolean mask selecting the outer border ring outside an ``inner`` square."""
    margin = (size - inner) // 2
    mask = np.ones((size, size), dtype=bool)
    mask[margin : margin + inner, margin : margin + inner] = False
    return mask


def _extract_image_stats(img_path: Path, border_mask: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Read an image once in grayscale and compute sky median, std, max, and border vector."""
    with Image.open(img_path) as raw:
        img = raw.convert("L")
        arr = np.asarray(img, dtype=np.float32)
        small = np.asarray(img.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)

    sky_strips = np.concatenate([arr[:16, :].ravel(), arr[240:, :].ravel()])
    sky_med = float(np.median(sky_strips))
    std_gray = float(arr.std())
    max_val = float(arr.max())

    vec = small[border_mask]
    vec = (vec - vec.mean()) / (vec.std() + 1e-6)
    return sky_med, std_gray, max_val, vec


class _UnionFind:
    """Disjoint-set structure for grouping near-duplicate star fields."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_clean_folds(
    train_csv: Path = DEFAULT_TRAIN_CSV,
    train_dir: Path = DEFAULT_TRAIN_DIR,
    clean_folds_path: Path = DEFAULT_CLEAN_FOLDS,
    dropped_csv_path: Path = DEFAULT_DROPPED_CSV,
    summary_json_path: Path = DEFAULT_SUMMARY_JSON,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Filter corrupted/poisoned rows and assign 5 stratified group folds."""
    df = pd.read_csv(train_csv, keep_default_na=False)
    if "extra_labels" not in df.columns:
        df["extra_labels"] = ""

    is_alien = df["extra_labels"].str.contains("alien", na=False)
    dropped_records: list[dict] = []
    for row in df[is_alien].itertuples():
        dropped_records.append(
            {
                "id": row.id,
                "label": int(row.label),
                "drop_reason": "alien",
                "sky_med": np.nan,
                "std_gray": np.nan,
                "max_val": np.nan,
                "matched_id": "",
                "matched_label": np.nan,
                "border_sim": np.nan,
            }
        )

    candidates = df[~is_alien].copy().reset_index(drop=True)
    border_mask = _border_ring_mask(64, 32)

    sky_meds: list[float] = []
    std_grays: list[float] = []
    max_vals: list[float] = []
    border_vecs: list[np.ndarray] = []

    for row in candidates.itertuples():
        img_path = train_dir / f"{row.id}.png"
        sky_med, std_gray, max_val, vec = _extract_image_stats(img_path, border_mask)
        sky_meds.append(sky_med)
        std_grays.append(std_gray)
        max_vals.append(max_val)
        border_vecs.append(vec)

    candidates["sky_med"] = sky_meds
    candidates["std_gray"] = std_grays
    candidates["max_val"] = max_vals
    vecs_mat = np.stack(border_vecs, axis=0)

    is_blank = candidates["std_gray"] < 5.0
    for idx in np.where(is_blank.to_numpy())[0]:
        row = candidates.iloc[idx]
        dropped_records.append(
            {
                "id": row["id"],
                "label": int(row["label"]),
                "drop_reason": "blank",
                "sky_med": float(row["sky_med"]),
                "std_gray": float(row["std_gray"]),
                "max_val": float(row["max_val"]),
                "matched_id": "",
                "matched_label": np.nan,
                "border_sim": np.nan,
            }
        )

    non_blank_mask = ~is_blank.to_numpy()
    bright_idx = np.where(non_blank_mask & (candidates["sky_med"].to_numpy() > 52.0))[0]
    dark_idx = np.where(non_blank_mask & (candidates["sky_med"].to_numpy() <= 50.0))[0]

    drop_overexposed = np.zeros(len(candidates), dtype=bool)
    if len(bright_idx) > 0 and len(dark_idx) > 0:
        sim_bd = (vecs_mat[bright_idx] @ vecs_mat[dark_idx].T) / vecs_mat.shape[1]
        best_pos = sim_bd.argmax(axis=1)
        best_sim = sim_bd[np.arange(len(bright_idx)), best_pos]
        for k, bi in enumerate(bright_idx):
            sim_val = float(best_sim[k])
            if sim_val > 0.88:
                drop_overexposed[bi] = True
                row = candidates.iloc[bi]
                match_row = candidates.iloc[dark_idx[int(best_pos[k])]]
                dropped_records.append(
                    {
                        "id": row["id"],
                        "label": int(row["label"]),
                        "drop_reason": "overexposed_duplicate",
                        "sky_med": float(row["sky_med"]),
                        "std_gray": float(row["std_gray"]),
                        "max_val": float(row["max_val"]),
                        "matched_id": str(match_row["id"]),
                        "matched_label": int(match_row["label"]),
                        "border_sim": round(sim_val, 6),
                    }
                )

    keep_mask = non_blank_mask & ~drop_overexposed
    kept_idx = np.where(keep_mask)[0]
    kept_df = candidates.iloc[kept_idx].copy().reset_index(drop=True)
    kept_vecs = vecs_mat[kept_idx]

    # Link near-duplicate star fields within the kept set via union-find.
    sim_kk = (kept_vecs @ kept_vecs.T) / kept_vecs.shape[1]
    hi_i, hi_j = np.where(np.triu(sim_kk > 0.90, k=1))
    uf = _UnionFind(len(kept_df))
    for a, b in zip(hi_i.tolist(), hi_j.tolist(), strict=True):
        uf.union(a, b)

    roots = [uf.find(i) for i in range(len(kept_df))]
    unique_roots: dict[int, str] = {}
    dup_groups: list[str] = []
    for root in roots:
        if root not in unique_roots:
            unique_roots[root] = f"grp_{len(unique_roots):04d}"
        dup_groups.append(unique_roots[root])
    kept_df["dup_group"] = dup_groups

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(kept_df), -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(kept_df["id"], kept_df["label"], kept_df["dup_group"])):
        folds[val_idx] = fold_idx
    if (folds < 0).any():
        raise RuntimeError("Failed to assign fold to every kept sample.")
    kept_df["fold"] = folds

    clean_out = kept_df[["id", "label", "fold", "dup_group", "sky_med", "std_gray"]].copy()
    clean_out["label"] = clean_out["label"].astype(int)
    clean_out["fold"] = clean_out["fold"].astype(int)
    clean_out["sky_med"] = clean_out["sky_med"].round(3)
    clean_out["std_gray"] = clean_out["std_gray"].round(3)

    dropped_df = pd.DataFrame(dropped_records)
    if not dropped_df.empty:
        dropped_df = dropped_df.sort_values(["drop_reason", "id"]).reset_index(drop=True)

    _write_csv_atomic(clean_out, clean_folds_path)
    _write_csv_atomic(dropped_df, dropped_csv_path)

    drop_counts = dropped_df["drop_reason"].value_counts().to_dict() if not dropped_df.empty else {}
    group_sizes = clean_out["dup_group"].value_counts()
    multi_groups = int((group_sizes > 1).sum())

    fold_class_counts = (
        pd.crosstab(clean_out["fold"], clean_out["label"])
        .sort_index()
        .to_dict(orient="index")
    )
    fold_class_counts_str = {
        str(f): {str(c): int(cnt) for c, cnt in cls_map.items()}
        for f, cls_map in fold_class_counts.items()
    }

    summary = {
        "total_train_rows": int(len(df)),
        "kept_rows": int(len(clean_out)),
        "dropped_total": int(len(dropped_df)),
        "dropped_by_reason": {str(k): int(v) for k, v in drop_counts.items()},
        "n_splits": int(n_splits),
        "seed": int(seed),
        "unique_dup_groups": int(clean_out["dup_group"].nunique()),
        "multi_member_dup_groups": multi_groups,
        "near_dup_pairs_linked": int(len(hi_i)),
        "class_counts_clean": {str(int(k)): int(v) for k, v in clean_out["label"].value_counts().sort_index().items()},
        "fold_totals": {str(int(k)): int(v) for k, v in clean_out["fold"].value_counts().sort_index().items()},
        "fold_class_counts": fold_class_counts_str,
    }
    _write_json_atomic(summary, summary_json_path)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for dataset cleaning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--clean-folds", type=Path, default=DEFAULT_CLEAN_FOLDS)
    parser.add_argument("--dropped-csv", type=Path, default=DEFAULT_DROPPED_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    summary = build_clean_folds(
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        clean_folds_path=args.clean_folds,
        dropped_csv_path=args.dropped_csv,
        summary_json_path=args.summary_json,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
