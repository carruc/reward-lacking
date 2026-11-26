"""PyTorch datasets, image preloading, and stage-wise augmentations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STAGE3_STAGES = (
    "stage3a_photo",
    "stage3b_photo_mixup",
    "stage3c_skylevel_mixup",
    "stage3d_asinh_mixup",
    "stage3e_skylevel_photo",
)
STAGES = ("stage0_clean", "stage1_geom", "stage2_mixup_blur", *STAGE3_STAGES)
MIXUP_STAGES = (
    "stage2_mixup_blur",
    "stage3b_photo_mixup",
    "stage3c_skylevel_mixup",
    "stage3d_asinh_mixup",
)
SKYLEVEL_STAGES = ("stage3c_skylevel_mixup", "stage3e_skylevel_photo")


def _outer_border_mask(height: int = 256, width: int = 256, margin: int = 16) -> torch.Tensor:
    """Boolean mask selecting the outer ``margin`` pixels around the frame."""
    mask = torch.ones((height, width), dtype=torch.bool)
    mask[margin : height - margin, margin : width - margin] = False
    return mask


class SkyLevelPreprocess:
    """Level each RGB channel's outer-sky median to a constant target floor."""

    def __init__(self, target_sky: float = 33.0 / 255.0, margin: int = 16) -> None:
        self.target_sky = float(target_sky)
        self.border_mask = _outer_border_mask(256, 256, margin)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sky_rgb = x[:, self.border_mask].median(dim=1).values[:, None, None]
        return torch.clamp(x - sky_rgb + self.target_sky, 0.0, 1.0)


class AsinhStretchPreprocess:
    """Sky-subtract and apply an astronomical ``asinh`` dynamic-range stretch."""

    def __init__(
        self,
        k: float = 3.0,
        q: float = 0.995,
        min_scale: float = 0.15,
        margin: int = 16,
    ) -> None:
        self.k = float(k)
        self.q = float(q)
        self.min_scale = float(min_scale)
        self.denom = math.asinh(self.k)
        self.border_mask = _outer_border_mask(256, 256, margin)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sky_rgb = x[:, self.border_mask].median(dim=1).values[:, None, None]
        x_sub = torch.clamp(x - sky_rgb, min=0.0)
        sub = x_sub[:, ::2, ::2].reshape(-1)
        k_idx = max(1, min(sub.numel(), int(round(self.q * sub.numel()))))
        scale = torch.kthvalue(sub, k_idx).values.clamp(min=self.min_scale)
        return torch.clamp(torch.asinh(self.k * (x_sub / scale)) / self.denom, 0.0, 1.0)


def get_preprocess_for_stage(stage: str | None) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Return the deterministic tensor preprocessor associated with ``stage``, if any."""
    if stage in SKYLEVEL_STAGES:
        return SkyLevelPreprocess()
    if stage == "stage3d_asinh_mixup":
        return AsinhStretchPreprocess()
    return None


def preload_images(ids: Sequence[str], image_dir: Path) -> torch.Tensor:
    """Decode all RGB PNG images into a single contiguous ``uint8`` tensor ``(N, 3, H, W)``."""
    arrays: list[np.ndarray] = []
    for img_id in ids:
        path = image_dir / f"{img_id}.png"
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        arrays.append(np.transpose(arr, (2, 0, 1)))
    stacked = np.ascontiguousarray(np.stack(arrays, axis=0))
    return torch.from_numpy(stacked).share_memory_()


def _geom_pipeline() -> list[Callable]:
    """Rotation/flip/translation/scale pipeline with reflect padding to avoid black borders."""
    return [
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.Pad(padding=112, padding_mode="reflect"),
        v2.RandomAffine(
            degrees=180,
            translate=(0.06, 0.06),
            scale=(0.88, 1.12),
            interpolation=v2.InterpolationMode.BILINEAR,
        ),
        v2.CenterCrop(256),
    ]


def _stage3_photo_blur() -> list[Callable]:
    """Photometric jitter + stronger blur matched to observed test-set shifts."""
    return [
        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=(0.55, 1.45),
                    contrast=(0.70, 1.35),
                    saturation=(0.65, 1.35),
                    hue=0.35,
                )
            ],
            p=0.7,
        ),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.4, 2.2))], p=0.5),
    ]


def build_eval_transform(stage: str | None = None) -> v2.Compose:
    """Deterministic preprocessing for validation and clean evaluation."""
    ops: list[Callable] = [v2.ToDtype(torch.float32, scale=True)]
    preprocess_fn = get_preprocess_for_stage(stage)
    if preprocess_fn is not None:
        ops.append(preprocess_fn)
    ops.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return v2.Compose(ops)


def build_train_transform(stage: str) -> v2.Compose:
    """Build the training transform corresponding to an ablation stage."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Expected one of {STAGES}.")
    if stage == "stage0_clean":
        return build_eval_transform(stage)
    if stage == "stage1_geom":
        return v2.Compose(
            [
                *_geom_pipeline(),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    if stage == "stage2_mixup_blur":
        return v2.Compose(
            [
                *_geom_pipeline(),
                v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.3, 1.3))], p=0.4),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    ops: list[Callable] = [*_geom_pipeline(), *_stage3_photo_blur()]
    preprocess_fn = get_preprocess_for_stage(stage)
    if preprocess_fn is not None:
        ops.append(preprocess_fn)
    ops.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return v2.Compose(ops)


class GalaxyTensorDataset(Dataset):
    """In-memory indexed view over preloaded ``(N, 3, H, W)`` uint8 galaxy images."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        indices: np.ndarray,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        src_idx = int(self.indices[item])
        img = self.images[src_idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[src_idx].item())


def _finalize_eval_tensor(
    img: torch.Tensor,
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    if preprocess_fn is not None:
        img = preprocess_fn(img)
    return v2.functional.normalize(img, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))


class ShiftedGalaxyTensorDataset(Dataset):
    """Deterministic shifted validation dataset mimicking test blur, dimming, and hue shifts."""

    _BRIGHTNESS_FACTORS = (0.75, 0.85, 0.92)
    _HUE_SHIFTS = (-0.12, 0.0, 0.12)

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        indices: np.ndarray,
        preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        src_idx = int(self.indices[item])
        img = v2.functional.to_dtype(self.images[src_idx], torch.float32, scale=True)
        img = v2.functional.gaussian_blur(img, kernel_size=[5, 5], sigma=[0.9, 0.9])
        brightness = self._BRIGHTNESS_FACTORS[item % len(self._BRIGHTNESS_FACTORS)]
        hue = self._HUE_SHIFTS[item % len(self._HUE_SHIFTS)]
        img = v2.functional.adjust_brightness(img, brightness_factor=brightness)
        if hue != 0.0:
            img = v2.functional.adjust_hue(img, hue_factor=hue)
        img = _finalize_eval_tensor(img, self.preprocess_fn)
        return img, int(self.labels[src_idx].item())


class HardShiftedGalaxyTensorDataset(Dataset):
    """Deterministic hard-shifted validation set calibrated to observed test corruption quantiles."""

    _BRIGHTNESS_FACTORS = (0.58, 0.75, 1.0, 1.35, 0.65)
    _CONTRAST_FACTORS = (0.80, 1.0, 1.20)
    _HUE_SHIFTS = (-0.30, -0.15, 0.0, 0.15, 0.30, 0.0, 0.0)

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        indices: np.ndarray,
        preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        src_idx = int(self.indices[item])
        img = v2.functional.to_dtype(self.images[src_idx], torch.float32, scale=True)
        if item % 2 == 0:
            img = v2.functional.gaussian_blur(img, kernel_size=[7, 7], sigma=[1.8, 1.8])
        else:
            img = v2.functional.gaussian_blur(img, kernel_size=[5, 5], sigma=[0.8, 0.8])

        brightness = self._BRIGHTNESS_FACTORS[item % len(self._BRIGHTNESS_FACTORS)]
        contrast = self._CONTRAST_FACTORS[item % len(self._CONTRAST_FACTORS)]
        hue = self._HUE_SHIFTS[item % len(self._HUE_SHIFTS)]

        img = v2.functional.adjust_brightness(img, brightness_factor=brightness)
        if contrast != 1.0:
            img = v2.functional.adjust_contrast(img, contrast_factor=contrast)
        if hue != 0.0:
            img = v2.functional.adjust_hue(img, hue_factor=hue)
        img = _finalize_eval_tensor(img, self.preprocess_fn)
        return img, int(self.labels[src_idx].item())


def build_shifted_eval_dataset(
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> ShiftedGalaxyTensorDataset:
    """Create a deterministic domain-shifted validation dataset for robustness tracking."""
    return ShiftedGalaxyTensorDataset(
        images=images,
        labels=labels,
        indices=indices,
        preprocess_fn=preprocess_fn,
    )


def build_hard_shifted_eval_dataset(
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> HardShiftedGalaxyTensorDataset:
    """Create a deterministic hard-shifted validation dataset calibrated to test shifts."""
    return HardShiftedGalaxyTensorDataset(
        images=images,
        labels=labels,
        indices=indices,
        preprocess_fn=preprocess_fn,
    )
