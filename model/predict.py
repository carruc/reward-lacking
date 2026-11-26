"""GPU-batched 4x dihedral TTA test inference and verified Kaggle submission generator."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.cleaning import DEFAULT_CLEAN_FOLDS
from data.dataset import IMAGENET_MEAN, IMAGENET_STD, SKYLEVEL_STAGES, _outer_border_mask, preload_images
from model.cnn import ARCHITECTURES, build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_CSV = REPO_ROOT / "artifacts" / "dataset" / "sample_submission.csv"
DEFAULT_TEST_DIR = REPO_ROOT / "artifacts" / "dataset" / "test"
DEFAULT_CKPT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "stage3_5fold"
DEFAULT_ABLATIONS_DIR = REPO_ROOT / "artifacts" / "ablations" / "stage3_5fold"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "submissions"
DEFAULT_OUT_CSV = DEFAULT_OUT_DIR / "sub_stage3_5fold_tta4.csv"


def sky_level_batch(
    x: torch.Tensor,
    border_mask: torch.Tensor,
    target_sky: float = 33.0 / 255.0,
) -> torch.Tensor:
    """Vectorized GPU equivalent of ``SkyLevelPreprocess`` for ``(B, 3, H, W)`` in ``[0, 1]``."""
    sky = x[:, :, border_mask].median(dim=2).values[:, :, None, None]
    return (x - sky + target_sky).clamp(0.0, 1.0)


def discover_families(
    ckpt_dir: Path,
    allow_partial: bool = False,
) -> list[tuple[str, str, str, list[tuple[int, Path]]]]:
    """Discover ``(arch, init, stage, [(fold, path), ...])`` checkpoint families."""
    grouped: dict[tuple[str, str, str], dict[int, Path]] = defaultdict(dict)
    for pt_path in sorted(ckpt_dir.glob("*.pt")):
        parts = pt_path.stem.split("__")
        if len(parts) != 4 or not parts[3].startswith("fold"):
            continue
        arch, init, stage, fold_str = parts
        if arch not in ARCHITECTURES:
            continue
        try:
            fold_idx = int(fold_str.removeprefix("fold"))
        except ValueError:
            continue
        grouped[(arch, init, stage)][fold_idx] = pt_path

    families: list[tuple[str, str, str, list[tuple[int, Path]]]] = []
    for (arch, init, stage), fold_map in sorted(grouped.items()):
        has_all_five = all(k in fold_map for k in range(5))
        if not has_all_five and not allow_partial:
            continue
        fold_items = [(k, fold_map[k]) for k in sorted(fold_map)]
        if fold_items:
            families.append((arch, init, stage, fold_items))
    return families


def _dihedral_views(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return 4 deterministic dihedral views of ``(B, C, H, W)``."""
    return (
        x,
        torch.flip(x, dims=[-1]),
        torch.flip(x, dims=[-2]),
        torch.rot90(x, k=1, dims=[-2, -1]),
    )


@torch.no_grad()
def predict_family_tta4(
    arch: str,
    stage: str,
    fold_ckpts: list[tuple[int, Path]],
    test_images: torch.Tensor,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> np.ndarray:
    """Predict ``(N, 10)`` probabilities for one ``(arch, stage)`` family with 4x dihedral TTA."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    use_skylevel = stage in SKYLEVEL_STAGES

    border_mask = _outer_border_mask(256, 256, 16).to(device)
    mean_t = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    n_samples = len(test_images)
    family_probs = np.zeros((n_samples, 10), dtype=np.float64)

    for fold_idx, ckpt_path in fold_ckpts:
        model = build_model(arch=arch, num_classes=10, pretrained=False).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        fold_probs = np.empty((n_samples, 10), dtype=np.float32)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xb = test_images[start:end].to(device, non_blocking=True).float().div_(255.0)
            if use_skylevel:
                xb = sky_level_batch(xb, border_mask)
            xb = (xb - mean_t) / std_t

            probs_sum = torch.zeros((end - start, 10), dtype=torch.float32, device=device)
            for view in _dihedral_views(xb):
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(view)
                    probs_sum += torch.softmax(logits.float(), dim=1)
            fold_probs[start:end] = (probs_sum / 4.0).cpu().numpy()

        family_probs += fold_probs.astype(np.float64)
        print(f"  predicted {arch} ({stage}) fold {fold_idx} from {ckpt_path.name}", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    family_probs /= float(len(fold_ckpts))
    return family_probs.astype(np.float32)


def write_and_verify_submission(
    sample_df: pd.DataFrame,
    probs: np.ndarray,
    out_csv: Path,
    clean_folds_csv: Path | None = DEFAULT_CLEAN_FOLDS,
) -> pd.DataFrame:
    """Write ``id,label`` CSV atomically and assert submission invariants."""
    if probs.shape != (len(sample_df), 10):
        raise ValueError(f"Expected probs shape {(len(sample_df), 10)}, got {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("Predicted probabilities contain NaN or Inf.")

    preds = probs.argmax(axis=1).astype(np.int64)
    sub = pd.DataFrame({"id": sample_df["id"].astype(str), "label": preds})

    if len(sub) != len(sample_df):
        raise ValueError(f"Row count mismatch: {len(sub)} vs {len(sample_df)}")
    if not (sub["id"].to_numpy() == sample_df["id"].astype(str).to_numpy()).all():
        raise ValueError("Submission IDs do not exactly match sample_submission.csv")
    if sub["label"].isna().any() or not sub["label"].between(0, 9).all():
        raise ValueError("Invalid labels outside 0..9 or NaN values found.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    sub.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(out_csv)

    pred_shares = (sub["label"].value_counts().reindex(range(10), fill_value=0) / len(sub) * 100.0).round(2)
    train_shares = None
    if clean_folds_csv is not None and clean_folds_csv.exists():
        cf = pd.read_csv(clean_folds_csv)
        train_shares = (cf["label"].value_counts().reindex(range(10), fill_value=0) / len(cf) * 100.0).round(2)

    dist_df = pd.DataFrame({"pred_count": sub["label"].value_counts().reindex(range(10), fill_value=0), "pred_pct": pred_shares})
    if train_shares is not None:
        dist_df["clean_train_pct"] = train_shares
    print(f"\nVerified submission: {out_csv} ({len(sub)} rows)\n{dist_df.to_string()}", flush=True)
    return sub


def _parse_checkpoint_stem(checkpoint: Path) -> tuple[str, str, str, int]:
    """Extract ``(arch, init, stage, fold_idx)`` from ``<arch>__<init>__<stage>__fold<k>.pt``."""
    parts = checkpoint.stem.split("__")
    if len(parts) != 4 or not parts[3].startswith("fold") or parts[0] not in ARCHITECTURES:
        raise ValueError(
            f"Unexpected checkpoint filename '{checkpoint.name}'; "
            "expected '<arch>__<init>__<stage>__fold<k>.pt'."
        )
    arch, init, stage, fold_str = parts
    return arch, init, stage, int(fold_str.removeprefix("fold"))


def run_inference(
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    ablations_dir: Path = DEFAULT_ABLATIONS_DIR,
    test_csv: Path = DEFAULT_TEST_CSV,
    test_dir: Path = DEFAULT_TEST_DIR,
    clean_folds_csv: Path = DEFAULT_CLEAN_FOLDS,
    out_csv: Path = DEFAULT_OUT_CSV,
    batch_size: int = 128,
    allow_partial: bool = False,
    checkpoint: Path | None = None,
) -> dict:
    """Run 4x TTA inference over completed 5-fold families (or one checkpoint) and write submissions."""
    t0 = time.perf_counter()
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        arch, _init, stage, fold_idx = _parse_checkpoint_stem(checkpoint)
        families: list[tuple[str, str, str, list[tuple[int, Path]]]] = []
    else:
        families = discover_families(ckpt_dir=ckpt_dir, allow_partial=allow_partial)
        if not families:
            raise FileNotFoundError(f"No completed 5-fold checkpoint families found in {ckpt_dir}")

    sample_df = pd.read_csv(test_csv)
    t_load = time.perf_counter()
    test_images = preload_images(sample_df["id"].astype(str).tolist(), test_dir)
    print(
        f"Preloaded {len(test_images)} test images ({test_images.nbytes / 1e9:.2f} GB) "
        f"in {time.perf_counter() - t_load:.1f}s",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint is not None:
        print(f"Running 4x TTA inference for single checkpoint {checkpoint.name}...", flush=True)
        probs = predict_family_tta4(
            arch=arch,
            stage=stage,
            fold_ckpts=[(fold_idx, checkpoint)],
            test_images=test_images,
            batch_size=batch_size,
            device=device,
        )
        probs_path = out_dir / f"probs_{checkpoint.stem}_tta4.npy"
        tmp_npy = out_dir / f"probs_{checkpoint.stem}_tta4.tmp.npy"
        np.save(tmp_npy, probs)
        tmp_npy.replace(probs_path)
        write_and_verify_submission(sample_df, probs, out_csv, clean_folds_csv=clean_folds_csv)
        elapsed = time.perf_counter() - t0
        print(f"Single-checkpoint inference complete in {elapsed:.1f}s. Wrote: {out_csv}", flush=True)
        return {"checkpoint": str(checkpoint), "submissions": [str(out_csv)], "elapsed_sec": round(elapsed, 1)}

    family_probs_map: dict[str, np.ndarray] = {}
    for arch, _init, stage, fold_ckpts in families:
        key = f"{arch}__{stage}"
        print(f"Running 4x TTA inference for {key} ({len(fold_ckpts)} folds)...", flush=True)
        probs = predict_family_tta4(
            arch=arch,
            stage=stage,
            fold_ckpts=fold_ckpts,
            test_images=test_images,
            batch_size=batch_size,
            device=device,
        )
        probs_path = out_dir / f"probs_{key}_5fold_tta4.npy"
        tmp_npy = out_dir / f"probs_{key}_5fold_tta4.tmp.npy"
        np.save(tmp_npy, probs)
        tmp_npy.replace(probs_path)
        family_probs_map[key] = probs

    equal_probs = np.mean(list(family_probs_map.values()), axis=0).astype(np.float32)
    write_and_verify_submission(sample_df, equal_probs, out_csv, clean_folds_csv=clean_folds_csv)

    written_files = [str(out_csv)]
    oof_path = ablations_dir / "oof_summary.json"
    if oof_path.exists():
        oof_data = json.loads(oof_path.read_text(encoding="utf-8"))
        weighted_info = oof_data.get("oof_ensemble_weighted")
        equal_info = oof_data.get("oof_ensemble_equal")
        if weighted_info and equal_info:
            weights_map = weighted_info.get("weights", {})
            delta = float(weighted_info["weighted_f1"]) - float(equal_info["weighted_f1"])
            if set(weights_map.keys()) == set(family_probs_map.keys()) and delta > 0.001:
                w_probs = np.zeros_like(equal_probs, dtype=np.float64)
                for key, w in weights_map.items():
                    w_probs += float(w) * family_probs_map[key].astype(np.float64)
                w_csv = out_dir / f"{out_csv.stem}_weighted{out_csv.suffix}"
                write_and_verify_submission(
                    sample_df, w_probs.astype(np.float32), w_csv, clean_folds_csv=clean_folds_csv
                )
                written_files.append(str(w_csv))

    elapsed = time.perf_counter() - t0
    print(f"Inference complete in {elapsed:.1f}s. Wrote: {written_files}", flush=True)
    return {"families": list(family_probs_map.keys()), "submissions": written_files, "elapsed_sec": round(elapsed, 1)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for test inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--ablations-dir", type=Path, default=DEFAULT_ABLATIONS_DIR)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--clean-folds", type=Path, default=DEFAULT_CLEAN_FOLDS)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_inference(
        ckpt_dir=args.ckpt_dir,
        ablations_dir=args.ablations_dir,
        test_csv=args.test_csv,
        test_dir=args.test_dir,
        clean_folds_csv=args.clean_folds,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        allow_partial=args.allow_partial,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
