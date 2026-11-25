"""Reusable visualization helpers for dataset exploration and diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


def _resolve_image_path(image_id: str, split_dir: Path) -> Path:
    """Resolve an image ID to a file path inside ``split_dir``."""
    candidate = split_dir / image_id
    if candidate.suffix == "":
        candidate = split_dir / f"{image_id}.png"
    return candidate


def _stretch_for_display(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1] for visualizing subtle spatial patterns."""
    arr_f = np.asarray(arr, dtype=np.float32)
    lo = float(arr_f.min())
    hi = float(arr_f.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr_f, dtype=np.float32)
    return (arr_f - lo) / (hi - lo)


def plot_class_distribution(
    labels: Sequence[int | str] | pd.Series,
    title: str = "Class Distribution",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot class counts and percentages with imbalance summary."""
    series = pd.Series(labels)
    counts = series.value_counts()
    try:
        counts = counts.sort_index(key=lambda idx: idx.astype(int))
    except (TypeError, ValueError):
        counts = counts.sort_index()

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.2), dpi=100)

    total = int(counts.sum())
    palette = sns.color_palette("viridis", n_colors=len(counts))
    bars = ax.bar(
        [str(k) for k in counts.index],
        counts.to_numpy(),
        color=palette,
        edgecolor="black",
        linewidth=0.6,
    )

    ymax = float(counts.max()) if len(counts) else 1.0
    ax.set_ylim(0, ymax * 1.18)

    for bar, cnt in zip(bars, counts.to_numpy(), strict=True):
        pct = 100.0 * cnt / total if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + ymax * 0.015,
            f"{cnt}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    imbalance = float(counts.max() / counts.min()) if len(counts) and counts.min() > 0 else float("inf")
    ax.set_title(f"{title} (n={total:,}, max/min={imbalance:.2f}x)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Image Count")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)
    return ax


def plot_image_grid(
    image_paths: Sequence[str | Path],
    titles: Sequence[str] | None = None,
    ncols: int = 6,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
) -> plt.Figure:
    """Display a grid of RGB images with optional per-tile titles."""
    paths = [Path(p) for p in image_paths]
    if not paths:
        fig, ax = plt.subplots(figsize=(5, 2), dpi=100)
        ax.text(0.5, 0.5, "No images to display", ha="center", va="center")
        ax.axis("off")
        if suptitle:
            fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        return fig

    ncols = max(1, min(ncols, len(paths)))
    nrows = int(np.ceil(len(paths) / ncols))
    if figsize is None:
        figsize = (ncols * 1.95, nrows * 2.15 + (0.4 if suptitle else 0.0))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=80)
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes_flat):
        if idx < len(paths):
            with Image.open(paths[idx]) as img:
                ax.imshow(img.convert("RGB"))
            if titles is not None and idx < len(titles):
                ax.set_title(str(titles[idx]), fontsize=8.5)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if suptitle else 1))
    return fig


def plot_class_samples(
    df: pd.DataFrame,
    image_dir: str | Path,
    samples_per_class: int = 6,
    seed: int = 42,
    id_col: str = "id",
    label_col: str = "label",
) -> plt.Figure:
    """Plot a grid with one row per class and ``samples_per_class`` random samples."""
    img_dir = Path(image_dir)
    classes = sorted(df[label_col].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    nrows = len(classes)
    ncols = samples_per_class

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 1.8, nrows * 1.9),
        dpi=75,
    )
    axes_2d = np.atleast_2d(axes)
    if nrows == 1:
        axes_2d = axes_2d.reshape(1, -1)
    if ncols == 1:
        axes_2d = axes_2d.reshape(-1, 1)

    for row_idx, cls in enumerate(classes):
        cls_df = df[df[label_col] == cls]
        n_take = min(samples_per_class, len(cls_df))
        sampled = cls_df.sample(n=n_take, random_state=seed + row_idx)

        for col_idx in range(ncols):
            ax = axes_2d[row_idx, col_idx]
            if col_idx < n_take:
                img_id = str(sampled.iloc[col_idx][id_col])
                path = _resolve_image_path(img_id, img_dir)
                with Image.open(path) as img:
                    ax.imshow(img.convert("RGB"))
                ax.set_title(img_id, fontsize=7.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col_idx == 0:
                ax.set_ylabel(
                    f"Class {cls}\n(n={len(cls_df)})",
                    fontsize=9.5,
                    fontweight="bold",
                    rotation=0,
                    labelpad=38,
                    va="center",
                )

    fig.suptitle(
        f"Representative Samples per Class ({samples_per_class} random samples/class, seed={seed})",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.985))
    return fig


def plot_distribution_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: Sequence[str],
    titles: Sequence[str] | None = None,
    ncols: int = 3,
    bins: int = 50,
) -> plt.Figure:
    """Plot overlaid density histograms comparing train vs. test feature distributions."""
    cols = list(columns)
    ncols = max(1, min(ncols, len(cols)))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.3, nrows * 3.2), dpi=100)
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes_flat):
        if idx >= len(cols):
            ax.axis("off")
            continue

        col = cols[idx]
        tr_vals = train_df[col].dropna()
        te_vals = test_df[col].dropna()
        use_kde = tr_vals.nunique() > 1 and te_vals.nunique() > 1

        combined = pd.concat([tr_vals, te_vals], ignore_index=True)
        lo, hi = float(combined.min()), float(combined.max())
        bin_edges = np.linspace(lo, hi, bins + 1) if hi > lo else 10

        sns.histplot(
            tr_vals,
            bins=bin_edges,
            stat="density",
            kde=use_kde,
            color="#1f77b4",
            alpha=0.45,
            label=f"Train (n={len(tr_vals):,})",
            ax=ax,
        )
        sns.histplot(
            te_vals,
            bins=bin_edges,
            stat="density",
            kde=use_kde,
            color="#ff7f0e",
            alpha=0.40,
            label=f"Test (n={len(te_vals):,})",
            ax=ax,
        )
        title = titles[idx] if titles is not None and idx < len(titles) else col
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, linestyle="--")
        sns.despine(ax=ax)

    fig.suptitle("Train vs. Test Feature Distributions", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def plot_duplicate_pairs(
    pairs_df: pd.DataFrame,
    train_dir: str | Path,
    test_dir: str | Path | None = None,
    max_pairs: int = 8,
) -> plt.Figure:
    """Plot candidate duplicate or near-duplicate pairs side-by-side with pixel diffs."""
    if pairs_df.empty:
        fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
        ax.text(0.5, 0.5, "No duplicate pairs found", ha="center", va="center")
        ax.axis("off")
        return fig

    tr_dir = Path(train_dir)
    te_dir = Path(test_dir) if test_dir is not None else tr_dir
    show_df = pairs_df.head(max_pairs).reset_index(drop=True)
    n_pairs = len(show_df)

    fig, axes = plt.subplots(n_pairs, 3, figsize=(8.2, n_pairs * 2.35), dpi=80)
    axes_2d = np.atleast_2d(axes)
    if n_pairs == 1:
        axes_2d = axes_2d.reshape(1, 3)

    for i, row in show_df.iterrows():
        split_a = str(row.get("split_a", "train"))
        split_b = str(row.get("split_b", "train"))
        dir_a = tr_dir if split_a == "train" else te_dir
        dir_b = tr_dir if split_b == "train" else te_dir

        path_a = _resolve_image_path(str(row["id_a"]), dir_a)
        path_b = _resolve_image_path(str(row["id_b"]), dir_b)

        with Image.open(path_a) as ia, Image.open(path_b) as ib:
            arr_a = np.asarray(ia.convert("RGB"), dtype=np.int16)
            arr_b = np.asarray(ib.convert("RGB"), dtype=np.int16)

        abs_diff = np.abs(arr_a - arr_b)
        max_diff = int(abs_diff.max())
        mean_diff = float(abs_diff.mean())
        diff_vis = abs_diff.mean(axis=2)

        lbl_a = f" (y={row['label_a']})" if "label_a" in row and pd.notna(row["label_a"]) else ""
        lbl_b = f" (y={row['label_b']})" if "label_b" in row and pd.notna(row["label_b"]) else ""
        ham_str = f" | dHash dist={int(row['hamming'])}" if "hamming" in row and pd.notna(row["hamming"]) else ""

        axes_2d[i, 0].imshow(arr_a.astype(np.uint8))
        axes_2d[i, 0].set_title(f"{split_a}: {row['id_a']}{lbl_a}", fontsize=8.5)
        axes_2d[i, 0].axis("off")

        axes_2d[i, 1].imshow(arr_b.astype(np.uint8))
        axes_2d[i, 1].set_title(f"{split_b}: {row['id_b']}{lbl_b}", fontsize=8.5)
        axes_2d[i, 1].axis("off")

        axes_2d[i, 2].imshow(diff_vis, cmap="inferno", vmin=0, vmax=max(1.0, float(diff_vis.max())))
        axes_2d[i, 2].set_title(f"|A - B| mean={mean_diff:.2f}, max={max_diff}{ham_str}", fontsize=8.5)
        axes_2d[i, 2].axis("off")

    fig.suptitle("Duplicate / Near-Duplicate Pairs Comparison", fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_spatial_maps(
    mean_img: np.ndarray,
    std_img: np.ndarray,
    fft_mag: np.ndarray | None = None,
    title_prefix: str = "",
) -> plt.Figure:
    """Plot dataset-wide spatial mean, per-pixel std, and optional 2D FFT spectrum."""
    ncols = 3 if fft_mag is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.1), dpi=95)
    axes_flat = np.atleast_1d(axes).ravel()

    prefix = f"{title_prefix} — " if title_prefix else ""

    mean_vis = _stretch_for_display(mean_img)
    axes_flat[0].imshow(mean_vis)
    axes_flat[0].set_title(
        f"{prefix}Pixel Mean (stretched)\nrange [{mean_img.min():.1f}, {mean_img.max():.1f}]",
        fontsize=10,
        fontweight="bold",
    )
    axes_flat[0].axis("off")

    std_vis = _stretch_for_display(std_img)
    axes_flat[1].imshow(std_vis)
    axes_flat[1].set_title(
        f"{prefix}Pixel Std (stretched)\nrange [{std_img.min():.1f}, {std_img.max():.1f}]",
        fontsize=10,
        fontweight="bold",
    )
    axes_flat[1].axis("off")

    if fft_mag is not None:
        im = axes_flat[2].imshow(fft_mag, cmap="magma")
        axes_flat[2].set_title(f"{prefix}Mean Log 2D FFT Spectrum", fontsize=10, fontweight="bold")
        axes_flat[2].axis("off")
        fig.colorbar(im, ax=axes_flat[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig
