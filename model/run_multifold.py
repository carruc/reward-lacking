"""Time-budgeted 5-fold orchestrator and Out-Of-Fold evaluator for top Stage 3 CNNs."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.cleaning import DEFAULT_CLEAN_FOLDS, DEFAULT_TRAIN_DIR
from data.dataset import STAGES, preload_images
from eval.metrics import compute_classification_metrics, write_summary_tables
from model.cnn import ARCHITECTURES
from model.predict import run_inference
from model.train import TrainConfig, train_single_run

REPO_ROOT = Path(__file__).resolve().parents[1]
TOP_ARCHS = ("efficientnet_v2_s", "convnext_tiny")
FOLD0_OUT = REPO_ROOT / "artifacts" / "ablations" / "fold0_stage3"
FOLD0_CKPT = REPO_ROOT / "artifacts" / "checkpoints" / "fold0_stage3"
CV5_OUT = REPO_ROOT / "artifacts" / "ablations" / "stage3_5fold"
CV5_CKPT = REPO_ROOT / "artifacts" / "checkpoints" / "stage3_5fold"

SKY_CANDIDATES = ("stage3e_skylevel_photo", "stage3c_skylevel_mixup")


def _log(msg: str, log_path: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _run_job(
    cfg: TrainConfig,
    images_cache: torch.Tensor,
    folds_df: pd.DataFrame,
    log_path: Path,
    force: bool,
) -> dict | None:
    """Execute one training job with skip/error/GPU cleanup handling."""
    run_dir = cfg.out_dir / cfg.run_id
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        _log(f"SKIP {cfg.run_id} (metrics.json exists in {cfg.out_dir.name})", log_path)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    _log(f"START {cfg.run_id} -> {cfg.out_dir}", log_path)
    try:
        res = train_single_run(cfg, images_cache=images_cache, folds_df=folds_df)
        write_summary_tables(cfg.out_dir)
        _log(
            f"DONE {cfg.run_id} | best_ep={res['best_epoch']} "
            f"wf1={res['best_val_weighted_f1']:.4f} mf1={res['best_val_macro_f1']:.4f} "
            f"shifted_wf1={res['shifted_val_weighted_f1']:.4f} "
            f"hard_shifted_wf1={res['hard_shifted_val_weighted_f1']:.4f} "
            f"run={res['train_time_sec']:.1f}s",
            log_path,
        )
        return res
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _log(f"FAIL {cfg.run_id}: {exc!r} (saved error.txt)", log_path)
        return None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _seed_fold0_into_cv5(
    stage: str,
    archs: tuple[str, ...],
    fold0_out: Path,
    fold0_ckpt: Path,
    cv5_out: Path,
    cv5_ckpt: Path,
    log_path: Path,
) -> None:
    """Copy completed Fold 0 artifacts/checkpoints into the 5-fold workspace without modifying fold0_stage3."""
    copied_any = False
    for arch in archs:
        run_id = f"{arch}__pretrained__{stage}__fold0"
        src_dir = fold0_out / run_id
        dst_dir = cv5_out / run_id
        src_ckpt = fold0_ckpt / f"{run_id}.pt"
        dst_ckpt = cv5_ckpt / f"{run_id}.pt"

        if (dst_dir / "metrics.json").exists() and dst_ckpt.exists():
            continue
        if not (src_dir / "metrics.json").exists() or not src_ckpt.exists():
            _log(f"WARN cannot seed {run_id} (missing in {fold0_out})", log_path)
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
        for fname in ("metrics.json", "history.csv", "val_probs.npy", "hard_shifted_val_probs.npy"):
            src_f = src_dir / fname
            if src_f.exists():
                shutil.copy2(src_f, dst_dir / fname)
        shutil.copy2(src_ckpt, dst_ckpt)
        copied_any = True
        _log(f"SEEDED {run_id} from {fold0_out.name} into {cv5_out.name}", log_path)

    if copied_any:
        write_summary_tables(cv5_out)


def _select_best_sky_stage(
    fold0_out: Path,
    archs: tuple[str, ...],
    log_path: Path,
) -> str:
    """Pick the better sky-leveled Stage 3 recipe on Fold 0 by mean 0.5*(clean + hard_shifted) WF1."""
    scores: dict[str, float] = {}
    for stage in SKY_CANDIDATES:
        vals: list[float] = []
        details: list[str] = []
        for arch in archs:
            m_path = fold0_out / f"{arch}__pretrained__{stage}__fold0" / "metrics.json"
            if not m_path.exists():
                continue
            m = json.loads(m_path.read_text(encoding="utf-8"))
            clean_wf1 = float(m["best_val_weighted_f1"])
            hard_wf1 = float(m["hard_shifted_val_weighted_f1"])
            combined = 0.5 * clean_wf1 + 0.5 * hard_wf1
            vals.append(combined)
            details.append(f"{arch}: clean={clean_wf1:.4f}, hard={hard_wf1:.4f}")
        if len(vals) == len(archs) and vals:
            mean_score = float(np.mean(vals))
            scores[stage] = mean_score
            _log(
                f"Sky candidate {stage}: combined={mean_score:.4f} ({'; '.join(details)})",
                log_path,
            )

    if not scores:
        _log("WARN no complete sky-level Fold 0 pairs found; defaulting to stage3c_skylevel_mixup", log_path)
        return "stage3c_skylevel_mixup"

    best_stage = max(scores, key=scores.__getitem__)
    _log(f"Selected best sky-leveled stage: {best_stage} (combined={scores[best_stage]:.4f})", log_path)
    return best_stage


def _weight_simplex_grid(n_models: int, steps: int = 10) -> list[tuple[float, ...]]:
    """All non-negative weight tuples on a 0.1-resolution simplex summing to 1.0."""
    combos: list[tuple[float, ...]] = []

    def _rec(idx: int, remaining: int, cur: list[int]) -> None:
        if idx == n_models - 1:
            combos.append(tuple(v / steps for v in (*cur, remaining)))
            return
        for v in range(remaining + 1):
            _rec(idx + 1, remaining - v, [*cur, v])

    _rec(0, steps, [])
    return combos


def compute_oof_summary(
    cv5_out: Path,
    clean_folds_csv: Path = DEFAULT_CLEAN_FOLDS,
) -> dict:
    """Stitch full 5-fold OOF probabilities and compute single-family + ensemble metrics."""
    folds_df = pd.read_csv(clean_folds_csv)
    y_true = folds_df["label"].to_numpy(dtype=np.int64)
    fold_arr = folds_df["fold"].to_numpy(dtype=np.int64)
    n_total = len(folds_df)

    # Discover all (arch, init, stage) families present under cv5_out
    candidates: dict[tuple[str, str, str], dict[int, Path]] = {}
    for m_path in sorted(cv5_out.glob("*/metrics.json")):
        parts = m_path.parent.name.split("__")
        if len(parts) != 4 or not parts[3].startswith("fold"):
            continue
        arch, init, stage, fold_str = parts
        try:
            fold_idx = int(fold_str.removeprefix("fold"))
        except ValueError:
            continue
        probs_path = m_path.parent / "val_probs.npy"
        if probs_path.exists():
            candidates.setdefault((arch, init, stage), {})[fold_idx] = probs_path

    family_results: list[dict] = []
    oof_probs_map: dict[str, np.ndarray] = {}

    for (arch, init, stage), fold_map in sorted(candidates.items()):
        if not all(k in fold_map for k in range(5)):
            continue
        oof = np.zeros((n_total, 10), dtype=np.float32)
        for k in range(5):
            idx_k = np.where(fold_arr == k)[0]
            probs_k = np.load(fold_map[k])
            if len(probs_k) != len(idx_k):
                raise ValueError(f"OOF length mismatch for {arch}__{stage} fold {k}: {len(probs_k)} vs {len(idx_k)}")
            oof[idx_k] = probs_k

        key = f"{arch}__{stage}"
        np.save(cv5_out / f"oof_{key}.npy", oof)
        metrics = compute_classification_metrics(y_true, oof.argmax(axis=1), num_classes=10)
        oof_probs_map[key] = oof
        family_results.append(
            {
                "family": key,
                "arch": arch,
                "init": init,
                "stage": stage,
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "per_class_f1": metrics["per_class_f1"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )

    family_results.sort(key=lambda r: r["weighted_f1"], reverse=True)

    ensemble_equal = None
    ensemble_weighted = None
    keys = sorted(oof_probs_map.keys())
    if len(keys) >= 2:
        eq_probs = np.mean([oof_probs_map[k] for k in keys], axis=0)
        eq_m = compute_classification_metrics(y_true, eq_probs.argmax(axis=1), num_classes=10)
        ensemble_equal = {
            "families": keys,
            "weights": {k: round(1.0 / len(keys), 4) for k in keys},
            **eq_m,
        }

        if 2 <= len(keys) <= 4:
            probs_stack = [oof_probs_map[k].astype(np.float64) for k in keys]
            best_wf1 = -1.0
            best_weights = None
            best_probs = None
            for w_tuple in _weight_simplex_grid(len(keys), steps=10):
                blend = sum(w * p for w, p in zip(w_tuple, probs_stack, strict=True))
                m = compute_classification_metrics(y_true, blend.argmax(axis=1), num_classes=10)
                if m["weighted_f1"] > best_wf1:
                    best_wf1 = m["weighted_f1"]
                    best_weights = w_tuple
                    best_probs = blend
            if best_weights is not None and best_probs is not None:
                wm = compute_classification_metrics(y_true, best_probs.argmax(axis=1), num_classes=10)
                ensemble_weighted = {
                    "families": keys,
                    "weights": {k: float(w) for k, w in zip(keys, best_weights, strict=True)},
                    **wm,
                }

    payload = {
        "n_oof_samples": int(n_total),
        "completed_families": family_results,
        "oof_ensemble_equal": ensemble_equal,
        "oof_ensemble_weighted": ensemble_weighted,
    }
    _write_json_atomic(payload, cv5_out / "oof_summary.json")

    md = [f"# 5-Fold Out-Of-Fold (OOF) Summary (n={n_total})", ""]
    if not family_results:
        md.append("No complete 5-fold families yet.")
    else:
        md.extend(
            [
                "| Model Family | OOF Weighted F1 | OOF Macro F1 | OOF Accuracy |",
                "| :--- | :---: | :---: | :---: |",
            ]
        )
        for r in family_results:
            md.append(
                f"| `{r['family']}` | {r['weighted_f1']:.4f} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} |"
            )
        if ensemble_equal is not None:
            md.append(
                f"| **Equal Ensemble ({len(keys)} families)** | **{ensemble_equal['weighted_f1']:.4f}** | "
                f"**{ensemble_equal['macro_f1']:.4f}** | **{ensemble_equal['accuracy']:.4f}** |"
            )
        if ensemble_weighted is not None:
            w_desc = ", ".join(f"{k}={v:.1f}" for k, v in ensemble_weighted["weights"].items())
            md.append(
                f"| **OOF-Weighted Ensemble** ({w_desc}) | **{ensemble_weighted['weighted_f1']:.4f}** | "
                f"**{ensemble_weighted['macro_f1']:.4f}** | **{ensemble_weighted['accuracy']:.4f}** |"
            )
    _write_text_atomic("\n".join(md) + "\n", cv5_out / "oof_summary.md")
    return payload


def run_multifold(
    archs: tuple[str, ...] = TOP_ARCHS,
    folds: tuple[int, ...] = (1, 2, 3, 4),
    only_stages: tuple[str, ...] | None = None,
    time_budget_min: float = 155.0,
    max_epochs: int | None = None,
    num_workers: int = 6,
    clean_folds_csv: Path = DEFAULT_CLEAN_FOLDS,
    train_dir: Path = DEFAULT_TRAIN_DIR,
    fold0_out: Path = FOLD0_OUT,
    fold0_ckpt: Path = FOLD0_CKPT,
    cv5_out: Path = CV5_OUT,
    cv5_ckpt: Path = CV5_CKPT,
    run_predict: bool = False,
    force: bool = False,
) -> dict:
    """Run Phase A (missing Fold 0), Phase B/C (budget-guarded 5-fold CV), Phase D (OOF), and optional inference."""
    cv5_out.mkdir(parents=True, exist_ok=True)
    cv5_ckpt.mkdir(parents=True, exist_ok=True)
    log_path = cv5_out / "queue.log"

    device_desc = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    _log(
        f"Multifold start | device={device_desc} | budget={time_budget_min:.1f}m | "
        f"archs={archs} | folds={folds} | only_stages={only_stages}",
        log_path,
    )

    folds_df = pd.read_csv(clean_folds_csv)
    t_load = time.perf_counter()
    images_cache = preload_images(folds_df["id"].tolist(), train_dir)
    _log(
        f"Preloaded {images_cache.shape[0]} images ({images_cache.nbytes / 1e6:.1f} MB) "
        f"in {time.perf_counter() - t_load:.1f}s",
        log_path,
    )

    t0 = time.perf_counter()

    def _within_budget(label: str) -> bool:
        elapsed_min = (time.perf_counter() - t0) / 60.0
        if elapsed_min >= time_budget_min:
            _log(
                f"BUDGET REACHED ({elapsed_min:.1f}m >= {time_budget_min:.1f}m); stopping before {label}",
                log_path,
            )
            return False
        return True

    # Phase A: Fill missing Fold 0 cell (stage3e_skylevel_photo)
    if only_stages is None or "stage3e_skylevel_photo" in only_stages:
        _log("=== Phase A: Fold 0 stage3e_skylevel_photo ===", log_path)
        for arch in archs:
            if not _within_budget(f"{arch} stage3e fold0"):
                break
            cfg = TrainConfig(
                arch=arch,
                init="pretrained",
                stage="stage3e_skylevel_photo",
                fold=0,
                max_epochs=max_epochs,
                num_workers=num_workers,
                clean_folds_csv=clean_folds_csv,
                train_dir=train_dir,
                out_dir=fold0_out,
                ckpt_dir=fold0_ckpt,
            )
            _run_job(cfg, images_cache, folds_df, log_path, force=force)
        write_summary_tables(fold0_out)

    # Determine stages to train across remaining folds
    if only_stages is not None:
        stages_to_scale = list(only_stages)
    else:
        stages_to_scale = ["stage3a_photo"]
        best_sky = _select_best_sky_stage(fold0_out, archs, log_path)
        if best_sky not in stages_to_scale:
            stages_to_scale.append(best_sky)

    for phase_idx, stage in enumerate(stages_to_scale, start=1):
        _log(f"=== Phase {chr(65 + phase_idx)}: 5-Fold {stage} (folds {folds}) ===", log_path)
        _seed_fold0_into_cv5(stage, archs, fold0_out, fold0_ckpt, cv5_out, cv5_ckpt, log_path)
        for fold in folds:
            for arch in archs:
                if not _within_budget(f"{arch}__{stage}__fold{fold}"):
                    break
                cfg = TrainConfig(
                    arch=arch,
                    init="pretrained",
                    stage=stage,
                    fold=fold,
                    max_epochs=max_epochs,
                    num_workers=num_workers,
                    clean_folds_csv=clean_folds_csv,
                    train_dir=train_dir,
                    out_dir=cv5_out,
                    ckpt_dir=cv5_ckpt,
                )
                _run_job(cfg, images_cache, folds_df, log_path, force=force)
            else:
                continue
            break

    # Phase D: OOF aggregation
    _log("=== Phase D: Computing 5-Fold OOF summary ===", log_path)
    oof_payload = compute_oof_summary(cv5_out, clean_folds_csv=clean_folds_csv)
    _log(
        f"OOF completed for {len(oof_payload['completed_families'])} families -> {cv5_out / 'oof_summary.md'}",
        log_path,
    )

    # Optional test inference
    if run_predict:
        if oof_payload["completed_families"]:
            _log("=== Running 4x TTA Test Inference ===", log_path)
            inf_res = run_inference(ckpt_dir=cv5_ckpt, ablations_dir=cv5_out, clean_folds_csv=clean_folds_csv)
            _log(f"Test inference done: {inf_res}", log_path)
        else:
            _log("WARN skipping test inference: no completed 5-fold families available.", log_path)

    total_min = (time.perf_counter() - t0) / 60.0
    _log(f"Multifold pipeline finished in {total_min:.1f}m", log_path)
    return oof_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for multi-fold training orchestration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archs", nargs="+", choices=ARCHITECTURES, default=list(TOP_ARCHS))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--only-stages", nargs="+", choices=STAGES, default=None)
    parser.add_argument("--time-budget-min", type=float, default=155.0)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--clean-folds", type=Path, default=DEFAULT_CLEAN_FOLDS)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--fold0-out", type=Path, default=FOLD0_OUT)
    parser.add_argument("--fold0-ckpt", type=Path, default=FOLD0_CKPT)
    parser.add_argument("--cv5-out", type=Path, default=CV5_OUT)
    parser.add_argument("--cv5-ckpt", type=Path, default=CV5_CKPT)
    parser.add_argument("--run-predict", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_multifold(
        archs=tuple(args.archs),
        folds=tuple(args.folds),
        only_stages=tuple(args.only_stages) if args.only_stages is not None else None,
        time_budget_min=args.time_budget_min,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        clean_folds_csv=args.clean_folds,
        train_dir=args.train_dir,
        fold0_out=args.fold0_out,
        fold0_ckpt=args.fold0_ckpt,
        cv5_out=args.cv5_out,
        cv5_ckpt=args.cv5_ckpt,
        run_predict=args.run_predict,
        force=args.force,
    )


if __name__ == "__main__":
    main()
