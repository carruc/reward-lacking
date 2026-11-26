"""CNN model factory, differential parameter groups, and architecture defaults."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

ARCHITECTURES = ("resnet50", "densenet121", "efficientnet_v2_s", "convnext_tiny")

_HEAD_PREFIXES = {
    "resnet50": "fc.",
    "densenet121": "classifier.",
    "efficientnet_v2_s": "classifier.",
    "convnext_tiny": "classifier.2.",
}


def build_model(
    arch: str,
    num_classes: int = 10,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Instantiate a CNN backbone with a 10-class dropout + linear head."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}'. Expected one of {ARCHITECTURES}.")

    if arch == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model

    if arch == "densenet121":
        weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model

    if arch == "efficientnet_v2_s":
        weights = tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model

    weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def get_param_groups(
    model: nn.Module,
    arch: str,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Split parameters into backbone/head and decay/no-decay groups for AdamW."""
    if arch not in _HEAD_PREFIXES:
        raise ValueError(f"Unknown architecture '{arch}'. Expected one of {ARCHITECTURES}.")
    head_prefix = _HEAD_PREFIXES[arch]

    buckets: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (False, True): [],
        (False, False): [],
        (True, True): [],
        (True, False): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = name.startswith(head_prefix)
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


def default_hparams(arch: str, pretrained: bool) -> dict:
    """Return architecture- and initialization-specific default hyperparameters."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}'. Expected one of {ARCHITECTURES}.")

    is_convnext = arch == "convnext_tiny"
    if pretrained:
        return {
            "epochs": 35,
            "warmup_epochs": 2,
            "patience": 10,
            "batch_size": 32,
            "dropout": 0.2,
            "backbone_lr": 5e-5 if is_convnext else 1e-4,
            "head_lr": 5e-4 if is_convnext else 1e-3,
            "weight_decay": 5e-2 if is_convnext else 1e-2,
        }
    return {
        "epochs": 70,
        "warmup_epochs": 5,
        "patience": 15,
        "batch_size": 32,
        "dropout": 0.2,
        "backbone_lr": 5e-4 if is_convnext else 1e-3,
        "head_lr": 5e-4 if is_convnext else 1e-3,
        "weight_decay": 5e-2 if is_convnext else 1e-2,
    }


def cache_pretrained_weights() -> None:
    """Download and cache ImageNet weights for all supported CNN architectures."""
    for arch in ARCHITECTURES:
        print(f"Caching pretrained weights for {arch}...", flush=True)
        model = build_model(arch, pretrained=True)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  cached {arch} ({n_params:,} params)", flush=True)
        del model
