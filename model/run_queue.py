"""Resumable sequential queue runner for the 24-run Fold 0 CNN screening."""

from __future__ import annotations

import argparse
import gc
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from data.cleaning import DEFAULT_CLEAN_FOLDS, DEFAULT_TRAIN_DIR
from data.dataset import STAGES, preload_images
from eval.metrics import write_summary_tables
from model.cnn import ARCHITECTURES
from model.train import DEFAULT_CKPT_DIR, DEFAULT_OUT_DIR, TrainConfig, train_single_run

INITS = ("pretrained", "scratch")
DEFAULT_SCREEN_STAGES = ("stage0_clean", "stage1_geom", "stage2_mixup_blur")


def _log(msg: str, log_path: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_queue(
    inits: tuple[str, ...] = INITS,
    stages: tuple[str, ...] = DEFAULT_SCREEN_STAGES,
    archs: tuple[str, ...] = ARCHITECTURES,
    fold: int = 0,
    seed: int = 42,
    max_epochs: int | None = None,
    num_workers: int = 6,
    clean_folds_csv: Path = DEFAULT_CLEAN_FOLDS,
    train_dir: Path = DEFAULT_TRAIN_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
) -> list[TrainConfig]:
    """Construct the ordered list of Fold 0 runs (pretrained first, then scratch)."""
    queue: list[TrainConfig] = []
    for init in inits:
        for stage in stages:
            for arch in archs:
                queue.append(
                    TrainConfig(
                        arch=arch,
                        init=init,
                        stage=stage,
                        fold=fold,
                        seed=seed,
                        max_epochs=max_epochs,
                        num_workers=num_workers,
                        clean_folds_csv=clean_folds_csv,
                        train_dir=train_dir,
                        out_dir=out_dir,
                        ckpt_dir=ckpt_dir,
                    )
                )
    return queue


def run_queue(
    configs: list[TrainConfig],
    force: bool = False,
) -> dict:
    """Execute ``configs`` sequentially, skipping completed runs unless ``force=True``."""
    if not configs:
        return {"completed": 0, "skipped": 0, "failed": 0}

    out_dir = configs[0].out_dir
    ckpt_dir = configs[0].ckpt_dir
    clean_folds_csv = configs[0].clean_folds_csv
    train_dir = configs[0].train_dir
    log_path = out_dir / "queue.log"

    if not clean_folds_csv.exists():
        raise FileNotFoundError(
            f"{clean_folds_csv} does not exist. Run `uv run clean-dataset` first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device_desc = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    _log(
        f"Queue start: {len(configs)} jobs | device={device_desc} | force={force} | out={out_dir}",
        log_path,
    )

    folds_df = pd.read_csv(clean_folds_csv)
    t_load = time.perf_counter()
    images_cache = preload_images(folds_df["id"].tolist(), train_dir)
    _log(
        f"Preloaded {images_cache.shape[0]} images ({tuple(images_cache.shape)}, "
        f"{images_cache.nbytes / 1e6:.1f} MB) in {time.perf_counter() - t_load:.1f}s",
        log_path,
    )

    completed = 0
    skipped = 0
    failed = 0
    total = len(configs)
    queue_t0 = time.perf_counter()

    for idx, cfg in enumerate(configs, start=1):
        run_dir = cfg.out_dir / cfg.run_id
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and not force:
            skipped += 1
            _log(f"[{idx}/{total}] SKIP {cfg.run_id} (metrics.json already exists)", log_path)
            continue

        _log(f"[{idx}/{total}] START {cfg.run_id}", log_path)
        try:
            res = train_single_run(cfg, images_cache=images_cache, folds_df=folds_df)
            completed += 1
            write_summary_tables(out_dir)
            elapsed = time.perf_counter() - queue_t0
            hard_part = (
                f" hard_shifted_wf1={res['hard_shifted_val_weighted_f1']:.4f}"
                if "hard_shifted_val_weighted_f1" in res
                else ""
            )
            _log(
                f"[{idx}/{total}] DONE {cfg.run_id} | best_ep={res['best_epoch']} "
                f"wf1={res['best_val_weighted_f1']:.4f} mf1={res['best_val_macro_f1']:.4f} "
                f"acc={res['best_val_acc']:.4f} shifted_wf1={res['shifted_val_weighted_f1']:.4f}{hard_part} "
                f"run={res['train_time_sec']:.1f}s total_elapsed={elapsed / 60:.1f}m",
                log_path,
            )
        except Exception as exc:
            failed += 1
            run_dir.mkdir(parents=True, exist_ok=True)
            err_text = traceback.format_exc()
            (run_dir / "error.txt").write_text(err_text, encoding="utf-8")
            _log(f"[{idx}/{total}] FAIL {cfg.run_id}: {exc!r} (saved error.txt)", log_path)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_summary_tables(out_dir)
    total_min = (time.perf_counter() - queue_t0) / 60.0
    _log(
        f"Queue finished in {total_min:.1f}m | completed={completed} skipped={skipped} failed={failed}",
        log_path,
    )
    return {"completed": completed, "skipped": skipped, "failed": failed}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Fold 0 screening queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-inits", nargs="+", choices=INITS, default=list(INITS))
    parser.add_argument("--only-stages", nargs="+", choices=STAGES, default=list(DEFAULT_SCREEN_STAGES))
    parser.add_argument("--only-archs", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--clean-folds", type=Path, default=DEFAULT_CLEAN_FOLDS)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    configs = build_queue(
        inits=tuple(args.only_inits),
        stages=tuple(args.only_stages),
        archs=tuple(args.only_archs),
        fold=args.fold,
        seed=args.seed,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        clean_folds_csv=args.clean_folds,
        train_dir=args.train_dir,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
    )
    run_queue(configs, force=args.force)


if __name__ == "__main__":
    main()
