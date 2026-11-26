"""Single-run PyTorch trainer for cleaned Galaxy10 CNN baselines and ablations."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.cleaning import DEFAULT_CLEAN_FOLDS, DEFAULT_TRAIN_DIR
from data.dataset import (
    MIXUP_STAGES,
    STAGE3_STAGES,
    STAGES,
    GalaxyTensorDataset,
    build_eval_transform,
    build_hard_shifted_eval_dataset,
    build_shifted_eval_dataset,
    build_train_transform,
    get_preprocess_for_stage,
    preload_images,
)
from eval.metrics import compute_classification_metrics
from model.cnn import ARCHITECTURES, build_model, default_hparams, get_param_groups

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "ablations" / "fold0_screen"
DEFAULT_CKPT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "fold0_screen"


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for a single training run."""

    arch: str
    init: str
    stage: str
    fold: int = 0
    seed: int = 42
    max_epochs: int | None = None
    num_workers: int = 6
    clean_folds_csv: Path = DEFAULT_CLEAN_FOLDS
    train_dir: Path = DEFAULT_TRAIN_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    ckpt_dir: Path = DEFAULT_CKPT_DIR

    @property
    def pretrained(self) -> bool:
        if self.init == "pretrained":
            return True
        if self.init == "scratch":
            return False
        raise ValueError(f"Unknown init '{self.init}'. Expected 'pretrained' or 'scratch'.")

    @property
    def run_id(self) -> str:
        return f"{self.arch}__{self.init}__{self.stage}__fold{self.fold}"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _build_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
    if epochs <= 1:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-6)
    effective_warmup = min(warmup_epochs, epochs - 1)
    if effective_warmup <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=effective_warmup
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - effective_warmup, eta_min=1e-6
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[effective_warmup]
    )


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict, np.ndarray]:
    """Run evaluation and return ``(mean_loss, metrics_dict, softmax_probs)``."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            probs = torch.softmax(logits.float(), dim=1)
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        all_targets.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    probs_mat = np.concatenate(all_probs, axis=0).astype(np.float32)
    y_pred = probs_mat.argmax(axis=1)
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=10)
    return total_loss / max(1, total_n), metrics, probs_mat


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def train_single_run(
    config: TrainConfig,
    images_cache: torch.Tensor | None = None,
    folds_df: pd.DataFrame | None = None,
) -> dict:
    """Train and evaluate one architecture/init/stage configuration on a single fold."""
    if config.arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{config.arch}'.")
    if config.stage not in STAGES:
        raise ValueError(f"Unknown stage '{config.stage}'.")

    _set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    if folds_df is None:
        folds_df = pd.read_csv(config.clean_folds_csv)
    if images_cache is None:
        images_cache = preload_images(folds_df["id"].tolist(), config.train_dir)

    labels_tensor = torch.tensor(folds_df["label"].to_numpy(dtype=np.int64), dtype=torch.long)
    train_idx = np.where(folds_df["fold"].to_numpy() != config.fold)[0]
    val_idx = np.where(folds_df["fold"].to_numpy() == config.fold)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"Empty train/val split for fold {config.fold}.")

    hparams = default_hparams(config.arch, pretrained=config.pretrained)
    is_stage3_pretrained = config.stage in STAGE3_STAGES and config.pretrained
    default_epochs = 40 if is_stage3_pretrained else int(hparams["epochs"])
    epochs = int(config.max_epochs if config.max_epochs is not None else default_epochs)
    batch_size = int(hparams["batch_size"])
    patience = 12 if is_stage3_pretrained else int(hparams["patience"])

    preprocess_fn = get_preprocess_for_stage(config.stage)
    train_ds = GalaxyTensorDataset(
        images=images_cache,
        labels=labels_tensor,
        indices=train_idx,
        transform=build_train_transform(config.stage),
    )
    val_ds = GalaxyTensorDataset(
        images=images_cache,
        labels=labels_tensor,
        indices=val_idx,
        transform=build_eval_transform(config.stage),
    )
    shifted_val_ds = build_shifted_eval_dataset(
        images=images_cache,
        labels=labels_tensor,
        indices=val_idx,
        preprocess_fn=preprocess_fn,
    )
    hard_shifted_val_ds = build_hard_shifted_eval_dataset(
        images=images_cache,
        labels=labels_tensor,
        indices=val_idx,
        preprocess_fn=preprocess_fn,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(config.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    shifted_val_loader = DataLoader(
        shifted_val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    hard_shifted_val_loader = DataLoader(
        hard_shifted_val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = build_model(
        arch=config.arch,
        num_classes=10,
        pretrained=config.pretrained,
        dropout=float(hparams["dropout"]),
    ).to(device)

    param_groups = get_param_groups(
        model=model,
        arch=config.arch,
        backbone_lr=float(hparams["backbone_lr"]),
        head_lr=float(hparams["head_lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = _build_scheduler(optimizer, epochs=epochs, warmup_epochs=int(hparams["warmup_epochs"]))

    run_dir = config.out_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.ckpt_dir / f"{config.run_id}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    history_rows: list[dict] = []
    use_mixup = config.stage in MIXUP_STAGES

    t0 = time.perf_counter()
    print(
        f"[{config.run_id}] start | device={device.type} epochs={epochs} "
        f"train={len(train_ds)} val={len(val_ds)}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            apply_mixup = use_mixup and (rng.random() < 0.5)
            if apply_mixup:
                lam = float(rng.beta(0.3, 0.3))
                perm = torch.randperm(x.size(0), device=device)
                x_in = lam * x + (1.0 - lam) * x[perm]
                y_b = y[perm]
            else:
                x_in = x
                y_b = y

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x_in)
                if apply_mixup:
                    loss = lam * F.cross_entropy(logits, y) + (1.0 - lam) * F.cross_entropy(logits, y_b)
                else:
                    loss = F.cross_entropy(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch} for {config.run_id}: {loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_n += bs

        scheduler.step()
        train_loss = train_loss_sum / max(1, train_n)

        val_loss, val_metrics, _ = _evaluate(model, val_loader, device=device, use_amp=use_amp)
        val_wf1 = float(val_metrics["weighted_f1"])
        val_mf1 = float(val_metrics["macro_f1"])
        val_acc = float(val_metrics["accuracy"])

        improved = (best_f1 < 0.0) or (val_wf1 > best_f1 + 1e-4)
        if improved:
            best_f1 = val_wf1
            best_epoch = epoch
            no_improve = 0
            tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
            torch.save(model.state_dict(), tmp_ckpt)
            tmp_ckpt.replace(ckpt_path)
        else:
            no_improve += 1

        ep_sec = time.perf_counter() - ep_t0
        lr_backbone = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_weighted_f1": val_wf1,
                "val_macro_f1": val_mf1,
                "val_acc": val_acc,
                "lr_backbone": lr_backbone,
                "epoch_sec": round(ep_sec, 2),
                "improved": improved,
            }
        )
        print(
            f"  ep {epoch:02d}/{epochs:02d} | tr_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"wf1={val_wf1:.4f} mf1={val_mf1:.4f} acc={val_acc:.4f} "
            f"({ep_sec:.1f}s){' *' if improved else ''}",
            flush=True,
        )

        if no_improve >= patience:
            print(f"[{config.run_id}] early stopping at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    train_time_sec = time.perf_counter() - t0

    # Reload best checkpoint for final clean + shifted validation diagnostics.
    best_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    best_val_loss, best_val_metrics, val_probs = _evaluate(model, val_loader, device=device, use_amp=use_amp)
    shifted_loss, shifted_metrics, _ = _evaluate(model, shifted_val_loader, device=device, use_amp=use_amp)
    hard_shifted_loss, hard_shifted_metrics, hard_shifted_probs = _evaluate(
        model, hard_shifted_val_loader, device=device, use_amp=use_amp
    )

    tmp_probs = run_dir / "val_probs.tmp.npy"
    np.save(tmp_probs, val_probs)
    tmp_probs.replace(run_dir / "val_probs.npy")

    tmp_hard_probs = run_dir / "hard_shifted_val_probs.tmp.npy"
    np.save(tmp_hard_probs, hard_shifted_probs)
    tmp_hard_probs.replace(run_dir / "hard_shifted_val_probs.npy")

    history_df = pd.DataFrame(history_rows)
    _write_csv_atomic(history_df, run_dir / "history.csv")

    metrics_payload = {
        "run_id": config.run_id,
        "arch": config.arch,
        "init": config.init,
        "stage": config.stage,
        "fold": config.fold,
        "seed": config.seed,
        "train_count": int(len(train_ds)),
        "val_count": int(len(val_ds)),
        "epochs_configured": epochs,
        "epochs_ran": int(len(history_rows)),
        "best_epoch": int(best_epoch),
        "best_val_loss": round(best_val_loss, 6),
        "best_val_weighted_f1": float(best_val_metrics["weighted_f1"]),
        "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
        "best_val_acc": float(best_val_metrics["accuracy"]),
        "best_val_per_class_f1": best_val_metrics["per_class_f1"],
        "best_val_confusion_matrix": best_val_metrics["confusion_matrix"],
        "shifted_val_loss": round(shifted_loss, 6),
        "shifted_val_weighted_f1": float(shifted_metrics["weighted_f1"]),
        "shifted_val_macro_f1": float(shifted_metrics["macro_f1"]),
        "shifted_val_acc": float(shifted_metrics["accuracy"]),
        "hard_shifted_val_loss": round(hard_shifted_loss, 6),
        "hard_shifted_val_weighted_f1": float(hard_shifted_metrics["weighted_f1"]),
        "hard_shifted_val_macro_f1": float(hard_shifted_metrics["macro_f1"]),
        "hard_shifted_val_acc": float(hard_shifted_metrics["accuracy"]),
        "train_time_sec": round(train_time_sec, 2),
        "hparams": {**hparams, "epochs": epochs, "patience": patience},
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
    }
    _write_json_atomic(metrics_payload, run_dir / "metrics.json")
    print(
        f"[{config.run_id}] done in {train_time_sec:.1f}s | best_ep={best_epoch} "
        f"wf1={metrics_payload['best_val_weighted_f1']:.4f} "
        f"shifted_wf1={metrics_payload['shifted_val_weighted_f1']:.4f} "
        f"hard_shifted_wf1={metrics_payload['hard_shifted_val_weighted_f1']:.4f}",
        flush=True,
    )
    return metrics_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for single-run training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHITECTURES, required=True)
    parser.add_argument("--init", choices=("pretrained", "scratch"), required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--clean-folds", type=Path, default=DEFAULT_CLEAN_FOLDS)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    cfg = TrainConfig(
        arch=args.arch,
        init=args.init,
        stage=args.stage,
        fold=args.fold,
        seed=args.seed,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        clean_folds_csv=args.clean_folds,
        train_dir=args.train_dir,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
    )
    train_single_run(cfg)


if __name__ == "__main__":
    main()
