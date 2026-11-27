"""DINOv2 Vision Transformer model factory, fine-tuning trainer, and feature helpers."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from data.cleaning import DEFAULT_CLEAN_FOLDS, DEFAULT_TRAIN_DIR
from data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MIXUP_STAGES,
    STAGES,
    GalaxyTensorDataset,
    _geom_pipeline,
    _stage3_photo_blur,
    build_hard_shifted_eval_dataset,
    build_shifted_eval_dataset,
    get_preprocess_for_stage,
    preload_images,
)
from eval.metrics import compute_classification_metrics, write_summary_tables

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_CACHE = REPO_ROOT / "artifacts" / "hf_cache"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "ablations" / "dinov2"
DEFAULT_CKPT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "dinov2"

DINOV2_VARIANTS: dict[str, dict] = {
    "vitb14_reg": {
        "timm_name": "vit_base_patch14_reg4_dinov2.lvd142m",
        "arch_id": "dinov2_vitb14_reg",
        "default_unfreeze": -1,
        "backbone_lr": 3e-5,
        "head_lr": 5e-4,
        "batch_size": 32,
        "weight_decay": 5e-2,
        "dropout": 0.2,
    },
    "vitl14_reg": {
        "timm_name": "vit_large_patch14_reg4_dinov2.lvd142m",
        "arch_id": "dinov2_vitl14_reg",
        "default_unfreeze": 6,
        "backbone_lr": 3e-5,
        "head_lr": 5e-4,
        "batch_size": 32,
        "weight_decay": 5e-2,
        "dropout": 0.2,
    },
    "vitg14_reg": {
        "timm_name": "vit_giant_patch14_reg4_dinov2.lvd142m",
        "arch_id": "dinov2_vitg14_reg",
        "default_unfreeze": 4,
        "backbone_lr": 2e-5,
        "head_lr": 5e-4,
        "batch_size": 16,
        "weight_decay": 5e-2,
        "dropout": 0.2,
    },
}


def configure_hf_cache(cache_dir: Path = DEFAULT_HF_CACHE, create: bool = False) -> Path:
    """Configure ``HF_HOME`` before importing ``timm`` so weights stay inside the workspace."""
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"])
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if create or cache_dir.exists():
        os.environ["HF_HOME"] = str(cache_dir)
    return cache_dir


class _Clamp01:
    """Picklable [0, 1] clamp transform after bicubic resizing."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)


class _ResizePreprocess:
    """Picklable stage preprocessor + bicubic resize wrapper for shifted eval datasets."""

    def __init__(
        self,
        stage_preprocess: Callable[[torch.Tensor], torch.Tensor] | None,
        img_size: int = 224,
    ) -> None:
        self.stage_preprocess = stage_preprocess
        self.img_size = int(img_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.stage_preprocess is not None:
            x = self.stage_preprocess(x)
        x = v2.functional.resize(
            x,
            [self.img_size, self.img_size],
            interpolation=v2.InterpolationMode.BICUBIC,
            antialias=True,
        )
        return x.clamp(0.0, 1.0)


def build_dinov2_eval_transform(stage: str | None = None, img_size: int = 224) -> v2.Compose:
    """Deterministic evaluation preprocessing resized to ``img_size`` for DINOv2."""
    ops: list[Callable] = [v2.ToDtype(torch.float32, scale=True)]
    preprocess_fn = get_preprocess_for_stage(stage)
    if preprocess_fn is not None:
        ops.append(preprocess_fn)
    ops.extend(
        [
            v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            _Clamp01(),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return v2.Compose(ops)


def build_dinov2_train_transform(stage: str, img_size: int = 224) -> v2.Compose:
    """Build stage-aligned training augmentations resized to ``img_size`` for DINOv2."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Expected one of {STAGES}.")
    if stage == "stage0_clean":
        return build_dinov2_eval_transform(stage, img_size=img_size)

    resize_tail = [
        v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        _Clamp01(),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if stage == "stage1_geom":
        return v2.Compose([*_geom_pipeline(), *resize_tail])
    if stage == "stage2_mixup_blur":
        return v2.Compose(
            [
                *_geom_pipeline(),
                v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.3, 1.3))], p=0.4),
                *resize_tail,
            ]
        )

    ops: list[Callable] = [*_geom_pipeline(), *_stage3_photo_blur()]
    preprocess_fn = get_preprocess_for_stage(stage)
    if preprocess_fn is not None:
        ops.append(preprocess_fn)
    ops.extend(resize_tail)
    return v2.Compose(ops)


def build_dinov2_model(
    variant: str = "vitl14_reg",
    num_classes: int = 10,
    pretrained: bool = True,
    dropout: float | None = None,
    img_size: int = 224,
    unfreeze_blocks: int | None = None,
    grad_checkpointing: bool = True,
) -> nn.Module:
    """Instantiate a register-augmented DINOv2 Vision Transformer with partial/full unfreezing."""
    if variant not in DINOV2_VARIANTS:
        raise ValueError(f"Unknown DINOv2 variant '{variant}'. Expected one of {tuple(DINOV2_VARIANTS)}.")
    spec = DINOV2_VARIANTS[variant]
    configure_hf_cache(create=pretrained)
    import timm

    drop = float(spec["dropout"] if dropout is None else dropout)
    model = timm.create_model(
        spec["timm_name"],
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop,
        img_size=img_size,
    )
    if grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)

    n_unfreeze = int(spec["default_unfreeze"] if unfreeze_blocks is None else unfreeze_blocks)
    if n_unfreeze >= 0:
        for param in model.parameters():
            param.requires_grad = False
        if n_unfreeze > 0 and hasattr(model, "blocks"):
            for blk in model.blocks[-n_unfreeze:]:
                for param in blk.parameters():
                    param.requires_grad = True
        for attr in ("norm", "fc_norm", "head"):
            mod = getattr(model, attr, None)
            if isinstance(mod, nn.Module):
                for param in mod.parameters():
                    param.requires_grad = True

    return model


def extract_dinov2_features_and_logits(
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pre_logits_embedding, class_logits)`` for an eval-mode DINOv2 model."""
    feats = model.forward_features(x)
    emb = model.forward_head(feats, pre_logits=True)
    logits = model.head(emb)
    return emb, logits


def get_dinov2_param_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float = 5e-2,
) -> list[dict]:
    """Split trainable DINOv2 parameters into backbone/head and decay/no-decay AdamW groups."""
    buckets: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (False, True): [],
        (False, False): [],
        (True, True): [],
        (True, False): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = name.startswith("head.")
        apply_decay = param.ndim > 1
        buckets[(is_head, apply_decay)].append(param)

    groups: list[dict] = []
    for (is_head, apply_decay), params in buckets.items():
        if not params:
            continue
        groups.append(
            {
                "params": params,
                "lr": head_lr if is_head else backbone_lr,
                "weight_decay": weight_decay if apply_decay else 0.0,
            }
        )
    return groups


@dataclass(frozen=True)
class DinoTrainConfig:
    """Configuration for a DINOv2 fine-tuning run."""

    variant: str = "vitl14_reg"
    stage: str = "stage3a_photo"
    fold: int = 0
    seed: int = 42
    epochs: int = 30
    warmup_epochs: int = 3
    patience: int = 8
    img_size: int = 224
    unfreeze_blocks: int | None = None
    batch_size: int | None = None
    backbone_lr: float | None = None
    head_lr: float | None = None
    weight_decay: float | None = None
    dropout: float | None = None
    num_workers: int = 6
    clean_folds_csv: Path = DEFAULT_CLEAN_FOLDS
    train_dir: Path = DEFAULT_TRAIN_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    ckpt_dir: Path = DEFAULT_CKPT_DIR

    @property
    def arch_id(self) -> str:
        return str(DINOV2_VARIANTS[self.variant]["arch_id"])

    @property
    def run_id(self) -> str:
        return f"{self.arch_id}__pretrained__{self.stage}__fold{self.fold}"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _build_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
    if epochs <= 1:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-7)
    effective_warmup = min(warmup_epochs, epochs - 1)
    if effective_warmup <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=effective_warmup
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - effective_warmup, eta_min=1e-7
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[effective_warmup]
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
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


def run_smoke_test(config: DinoTrainConfig) -> dict:
    """Download/load DINOv2 and verify a single CUDA forward+backward step & peak VRAM."""
    spec = DINOV2_VARIANTS[config.variant]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    bs = int(config.batch_size or spec["batch_size"])
    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)

    model = build_dinov2_model(
        variant=config.variant,
        num_classes=10,
        pretrained=True,
        dropout=config.dropout,
        img_size=config.img_size,
        unfreeze_blocks=config.unfreeze_blocks,
        grad_checkpointing=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    backbone_lr = float(config.backbone_lr or spec["backbone_lr"])
    head_lr = float(config.head_lr or spec["head_lr"])
    wd = float(config.weight_decay or spec["weight_decay"])
    optimizer = torch.optim.AdamW(get_dinov2_param_groups(model, backbone_lr, head_lr, wd))

    x = torch.randn(bs, 3, config.img_size, config.img_size, device=device)
    y = torch.randint(0, 10, (bs,), device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke-test loss: {loss.item()}")
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        emb, eval_logits = extract_dinov2_features_and_logits(model, x)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if use_amp else 0.0
    info = {
        "variant": config.variant,
        "timm_name": spec["timm_name"],
        "device": device.type,
        "batch_size": bs,
        "img_size": config.img_size,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "emb_shape": list(emb.shape),
        "logits_shape": list(eval_logits.shape),
        "loss": round(float(loss.item()), 4),
        "peak_vram_mb": round(float(peak_mb), 1),
    }
    print(json.dumps(info, indent=2), flush=True)
    return info


def train_dinov2(
    config: DinoTrainConfig,
    images_cache: torch.Tensor | None = None,
    folds_df: pd.DataFrame | None = None,
) -> dict:
    """Fine-tune DINOv2 on a single fold and record clean + shifted validation metrics."""
    if config.variant not in DINOV2_VARIANTS:
        raise ValueError(f"Unknown DINOv2 variant '{config.variant}'.")
    if config.stage not in STAGES:
        raise ValueError(f"Unknown stage '{config.stage}'.")

    _set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    spec = DINOV2_VARIANTS[config.variant]

    if folds_df is None:
        folds_df = pd.read_csv(config.clean_folds_csv)
    if images_cache is None:
        images_cache = preload_images(folds_df["id"].tolist(), config.train_dir)

    labels_tensor = torch.tensor(folds_df["label"].to_numpy(dtype=np.int64), dtype=torch.long)
    train_idx = np.where(folds_df["fold"].to_numpy() != config.fold)[0]
    val_idx = np.where(folds_df["fold"].to_numpy() == config.fold)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"Empty train/val split for fold {config.fold}.")

    batch_size = int(config.batch_size or spec["batch_size"])
    backbone_lr = float(config.backbone_lr or spec["backbone_lr"])
    head_lr = float(config.head_lr or spec["head_lr"])
    weight_decay = float(config.weight_decay or spec["weight_decay"])
    dropout = float(spec["dropout"] if config.dropout is None else config.dropout)
    unfreeze_blocks = int(spec["default_unfreeze"] if config.unfreeze_blocks is None else config.unfreeze_blocks)

    stage_preprocess = get_preprocess_for_stage(config.stage)
    eval_preprocess = _ResizePreprocess(stage_preprocess=stage_preprocess, img_size=config.img_size)

    train_ds = GalaxyTensorDataset(
        images=images_cache,
        labels=labels_tensor,
        indices=train_idx,
        transform=build_dinov2_train_transform(config.stage, img_size=config.img_size),
    )
    val_ds = GalaxyTensorDataset(
        images=images_cache,
        labels=labels_tensor,
        indices=val_idx,
        transform=build_dinov2_eval_transform(config.stage, img_size=config.img_size),
    )
    shifted_val_ds = build_shifted_eval_dataset(
        images=images_cache,
        labels=labels_tensor,
        indices=val_idx,
        preprocess_fn=eval_preprocess,
    )
    hard_shifted_val_ds = build_hard_shifted_eval_dataset(
        images=images_cache,
        labels=labels_tensor,
        indices=val_idx,
        preprocess_fn=eval_preprocess,
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
    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)

    model = build_dinov2_model(
        variant=config.variant,
        num_classes=10,
        pretrained=True,
        dropout=dropout,
        img_size=config.img_size,
        unfreeze_blocks=unfreeze_blocks,
        grad_checkpointing=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_groups = get_dinov2_param_groups(
        model=model,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = _build_scheduler(optimizer, epochs=config.epochs, warmup_epochs=config.warmup_epochs)

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
        f"[{config.run_id}] start | device={device.type} epochs={config.epochs} "
        f"trainable={trainable_params:,}/{total_params:,} unfreeze_blocks={unfreeze_blocks} "
        f"train={len(train_ds)} val={len(val_ds)}",
        flush=True,
    )

    for epoch in range(1, config.epochs + 1):
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
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
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
            f"  ep {epoch:02d}/{config.epochs:02d} | tr_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"wf1={val_wf1:.4f} mf1={val_mf1:.4f} acc={val_acc:.4f} "
            f"({ep_sec:.1f}s){' *' if improved else ''}",
            flush=True,
        )

        if no_improve >= config.patience:
            print(f"[{config.run_id}] early stopping at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    train_time_sec = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if use_amp else 0.0

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
        "arch": config.arch_id,
        "variant": config.variant,
        "timm_name": spec["timm_name"],
        "init": "pretrained",
        "stage": config.stage,
        "fold": config.fold,
        "seed": config.seed,
        "img_size": config.img_size,
        "unfreeze_blocks": unfreeze_blocks,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "peak_vram_mb": round(float(peak_vram_mb), 1),
        "train_count": int(len(train_ds)),
        "val_count": int(len(val_ds)),
        "epochs_configured": config.epochs,
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
        "hparams": {
            "batch_size": batch_size,
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "warmup_epochs": config.warmup_epochs,
            "epochs": config.epochs,
            "patience": config.patience,
        },
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
    }
    _write_json_atomic(metrics_payload, run_dir / "metrics.json")
    write_summary_tables(config.out_dir)
    print(
        f"[{config.run_id}] done in {train_time_sec:.1f}s | best_ep={best_epoch} "
        f"wf1={metrics_payload['best_val_weighted_f1']:.4f} "
        f"shifted_wf1={metrics_payload['shifted_val_weighted_f1']:.4f} "
        f"hard_shifted_wf1={metrics_payload['hard_shifted_val_weighted_f1']:.4f} "
        f"peak_vram={peak_vram_mb:.0f}MB",
        flush=True,
    )
    return metrics_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for DINOv2 fine-tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(DINOV2_VARIANTS), default="vitl14_reg")
    parser.add_argument("--stage", choices=STAGES, default="stage3a_photo")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--unfreeze-blocks", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--clean-folds", type=Path, default=DEFAULT_CLEAN_FOLDS)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.download_only:
        cache = configure_hf_cache(create=True)
        model = build_dinov2_model(
            variant=args.variant,
            num_classes=10,
            pretrained=True,
            dropout=args.dropout,
            img_size=args.img_size,
            unfreeze_blocks=args.unfreeze_blocks,
            grad_checkpointing=False,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Cached {args.variant} ({n_params:,} params) under HF_HOME={cache}", flush=True)
        return

    cfg = DinoTrainConfig(
        variant=args.variant,
        stage=args.stage,
        fold=args.fold,
        seed=args.seed,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        img_size=args.img_size,
        unfreeze_blocks=args.unfreeze_blocks,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        clean_folds_csv=args.clean_folds,
        train_dir=args.train_dir,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
    )
    if args.smoke_test:
        run_smoke_test(cfg)
        return

    train_dinov2(cfg)


if __name__ == "__main__":
    main()
