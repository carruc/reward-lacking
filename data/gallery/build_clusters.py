"""Extract multi-backbone embeddings, 2D t-SNE projections, and HDBSCAN clusters.

Generates ``artifacts/clusters/meta.json`` plus per-space artifacts under
``artifacts/clusters/<space_id>/`` covering all 19,064 train + test images across:
- 5-fold fine-tuned EfficientNetV2-S (``stage3a_photo``)
- 5-fold fine-tuned ConvNeXt-Tiny (``stage3a_photo``)
- Fold 0 fine-tuned DINOv2-L/14 with registers (``stage3a_photo``)

Also exports ``run_hdbscan_and_summarize`` so ``data/gallery/serve.py`` can
re-cluster live on user-selected HDBSCAN hyperparameters with an identical schema.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data.cleaning import DEFAULT_CLEAN_FOLDS, DEFAULT_DROPPED_CSV, DEFAULT_TRAIN_CSV
from data.dataset import IMAGENET_MEAN, IMAGENET_STD, preload_images
from model.cnn import build_model
from model.dinov2 import build_dinov2_model, extract_dinov2_features_and_logits

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
DEFAULT_CLUSTERS_DIR = REPO_ROOT / "artifacts" / "clusters"
DEFAULT_CNN_CKPT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "stage3_5fold"
DEFAULT_CNN_ABL_DIR = REPO_ROOT / "artifacts" / "ablations" / "stage3_5fold"
DEFAULT_DINO_CKPT = (
    REPO_ROOT / "artifacts" / "checkpoints" / "dinov2" / "dinov2_vitl14_reg__pretrained__stage3a_photo__fold0.pt"
)
DEFAULT_DINO_METRICS = (
    REPO_ROOT
    / "artifacts"
    / "ablations"
    / "dinov2"
    / "dinov2_vitl14_reg__pretrained__stage3a_photo__fold0"
    / "metrics.json"
)

STATUS_NAMES = ("clean", "alien", "overexposed_duplicate", "blank", "test")
STATUS_TO_INT = {name: idx for idx, name in enumerate(STATUS_NAMES)}


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def _write_npy_atomic(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


def _compute_image_stats_batch(images: torch.Tensor, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Compute outer-16px sky median and full-frame grayscale std for ``(N, 3, 256, 256)`` uint8."""
    n = len(images)
    sky_meds = np.empty(n, dtype=np.float32)
    std_grays = np.empty(n, dtype=np.float32)
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = images[start:end].float()
        gray = (chunk * weights).sum(dim=1)  # (B, 256, 256)
        top = gray[:, :16, :].reshape(end - start, -1)
        bot = gray[:, 240:, :].reshape(end - start, -1)
        strips = torch.cat([top, bot], dim=1)
        sky_meds[start:end] = strips.median(dim=1).values.numpy()
        std_grays[start:end] = gray.reshape(end - start, -1).std(dim=1, unbiased=False).numpy()

    return sky_meds, std_grays


def load_sample_catalog(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    train_csv: Path = DEFAULT_TRAIN_CSV,
    clean_folds_csv: Path = DEFAULT_CLEAN_FOLDS,
    dropped_csv: Path = DEFAULT_DROPPED_CSV,
) -> tuple[dict, torch.Tensor, dict[str, int]]:
    """Load ordered train+test catalog, metadata arrays, and preloaded uint8 tensor."""
    train_paths = sorted((dataset_dir / "train").glob("*.png"))
    test_paths = sorted((dataset_dir / "test").glob("*.png"))
    train_ids = [p.stem for p in train_paths]
    test_ids = [p.stem for p in test_paths]
    all_ids = train_ids + test_ids
    id_to_row = {img_id: idx for idx, img_id in enumerate(all_ids)}

    train_df = pd.read_csv(train_csv, keep_default_na=False)
    train_labels = {str(r["id"]): int(r["label"]) for _, r in train_df.iterrows()}

    clean_df = pd.read_csv(clean_folds_csv)
    clean_fold_map = {str(r["id"]): int(r["fold"]) for _, r in clean_df.iterrows()}

    dropped_reason_map: dict[str, str] = {}
    if dropped_csv.exists():
        dropped_df = pd.read_csv(dropped_csv, keep_default_na=False)
        dropped_reason_map = {str(r["id"]): str(r["drop_reason"]) for _, r in dropped_df.iterrows()}

    splits: list[str] = []
    labels: list[int | None] = []
    statuses: list[int] = []
    folds: list[int] = []

    for img_id in train_ids:
        splits.append("train")
        labels.append(train_labels.get(img_id))
        if img_id in clean_fold_map:
            statuses.append(STATUS_TO_INT["clean"])
            folds.append(clean_fold_map[img_id])
        else:
            reason = dropped_reason_map.get(img_id, "alien")
            statuses.append(STATUS_TO_INT.get(reason, STATUS_TO_INT["alien"]))
            folds.append(-1)

    for _ in test_ids:
        splits.append("test")
        labels.append(None)
        statuses.append(STATUS_TO_INT["test"])
        folds.append(-1)

    t0 = time.perf_counter()
    train_tensor = preload_images(train_ids, dataset_dir / "train")
    test_tensor = preload_images(test_ids, dataset_dir / "test")
    all_images = torch.cat([train_tensor, test_tensor], dim=0)
    del train_tensor, test_tensor
    gc.collect()

    sky_meds, std_grays = _compute_image_stats_batch(all_images)
    print(
        f"Loaded {len(all_ids)} images ({all_images.nbytes / 1e9:.2f} GB) + sky stats "
        f"in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    samples_meta = {
        "ids": all_ids,
        "split": splits,
        "label": labels,
        "status": statuses,
        "fold": folds,
        "sky_med": [round(float(v), 1) for v in sky_meds],
        "std_gray": [round(float(v), 2) for v in std_grays],
    }
    return samples_meta, all_images, id_to_row


def _l2_normalize_np(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-12)).astype(np.float32)


@torch.no_grad()
def _extract_cnn_5fold(
    arch: str,
    ckpt_paths: list[Path],
    images: torch.Tensor,
    oof_probs_path: Path | None,
    clean_ids: list[str],
    id_to_row: dict[str, int],
    batch_size: int = 128,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract concatenated L2-normalized 5-fold CNN embeddings and probabilities."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    n_samples = len(images)

    mean_t = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    fold_embs: list[np.ndarray] = []
    probs_sum = np.zeros((n_samples, 10), dtype=np.float64)

    for ckpt_path in ckpt_paths:
        model = build_model(arch=arch, num_classes=10, pretrained=False).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        emb_chunks: list[np.ndarray] = []
        prob_chunks: list[np.ndarray] = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xb = images[start:end].to(device, non_blocking=True).float().div_(255.0)
            xb = (xb - mean_t) / std_t

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                pooled = model.avgpool(model.features(xb))
                if arch == "convnext_tiny":
                    emb = model.classifier[1](model.classifier[0](pooled))
                    logits = model.classifier[2](emb)
                else:
                    emb = torch.flatten(pooled, 1)
                    logits = model.classifier(emb)
                emb = F.normalize(emb.float(), p=2, dim=1)
                probs = torch.softmax(logits.float(), dim=1)

            emb_chunks.append(emb.cpu().numpy())
            prob_chunks.append(probs.cpu().numpy())

        fold_embs.append(np.concatenate(emb_chunks, axis=0).astype(np.float32))
        probs_sum += np.concatenate(prob_chunks, axis=0).astype(np.float64)
        print(f"  extracted {arch} from {ckpt_path.name}", flush=True)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_emb = np.concatenate(fold_embs, axis=1) / np.sqrt(float(len(ckpt_paths)))
    combined_emb = _l2_normalize_np(combined_emb)
    probs_mat = (probs_sum / float(len(ckpt_paths))).astype(np.float32)

    is_oof = np.zeros(n_samples, dtype=np.uint8)
    if oof_probs_path is not None and oof_probs_path.exists():
        oof_probs = np.load(oof_probs_path).astype(np.float32)
        if len(oof_probs) != len(clean_ids):
            raise ValueError(
                f"OOF rows mismatch for {arch}: {len(oof_probs)} vs {len(clean_ids)} clean_folds rows"
            )
        clean_rows = np.array([id_to_row[cid] for cid in clean_ids], dtype=np.int64)
        probs_mat[clean_rows] = oof_probs
        is_oof[clean_rows] = 1

    return combined_emb, probs_mat, is_oof


@torch.no_grad()
def _extract_dinov2(
    variant: str,
    ckpt_path: Path,
    images: torch.Tensor,
    fold_arr: np.ndarray,
    val_fold: int = 0,
    img_size: int = 224,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract L2-normalized DINOv2 post-norm CLS embeddings and softmax probabilities."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    n_samples = len(images)

    mean_t = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    model = build_dinov2_model(
        variant=variant,
        num_classes=10,
        pretrained=False,
        img_size=img_size,
        unfreeze_blocks=0,
        grad_checkpointing=False,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    emb_chunks: list[np.ndarray] = []
    prob_chunks: list[np.ndarray] = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        xb = images[start:end].to(device, non_blocking=True).float().div_(255.0)
        xb = F.interpolate(xb, size=(img_size, img_size), mode="bicubic", align_corners=False).clamp_(0.0, 1.0)
        xb = (xb - mean_t) / std_t

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            emb, logits = extract_dinov2_features_and_logits(model, xb)
            emb = F.normalize(emb.float(), p=2, dim=1)
            probs = torch.softmax(logits.float(), dim=1)

        emb_chunks.append(emb.cpu().numpy())
        prob_chunks.append(probs.cpu().numpy())

    print(f"  extracted DINOv2 ({variant}) from {ckpt_path.name}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embeddings = _l2_normalize_np(np.concatenate(emb_chunks, axis=0).astype(np.float32))
    probs_mat = np.concatenate(prob_chunks, axis=0).astype(np.float32)
    is_oof = (fold_arr == val_fold).astype(np.uint8)
    return embeddings, probs_mat, is_oof


def compute_pca_and_tsne(
    embeddings: np.ndarray,
    pca_components: int = 50,
    perplexity: float = 35.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute PCA-50 features and aspect-normalized 2D t-SNE coordinates in ``[-1, 1]``."""
    t0 = time.perf_counter()
    pca = PCA(n_components=pca_components, random_state=seed)
    pca50 = pca.fit_transform(embeddings).astype(np.float32)
    var_explained = float(pca.explained_variance_ratio_.sum())
    t_pca = time.perf_counter() - t0

    t1 = time.perf_counter()
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        metric="euclidean",
        random_state=seed,
        method="barnes_hut",
        n_jobs=-1,
    )
    coords = tsne.fit_transform(pca50).astype(np.float32)
    midpoint = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
    centered = coords - midpoint
    scale = float(np.max(np.abs(centered))) + 1e-8
    coords_norm = (centered / scale).astype(np.float32)
    print(
        f"  PCA({pca_components})={t_pca:.1f}s (var={var_explained:.3f}) | "
        f"t-SNE(2)={time.perf_counter() - t1:.1f}s",
        flush=True,
    )
    return pca50, coords_norm, var_explained


def run_hdbscan_and_summarize(
    coords_2d: np.ndarray,
    pca50: np.ndarray,
    samples_meta: dict,
    pred: np.ndarray,
    conf: np.ndarray,
    *,
    cluster_on: str = "2d",
    scope: str = "all",
    min_cluster_size: int = 80,
    min_samples: int = 15,
    max_cluster_size: int | None = 3000,
    method: str = "eom",
    pca_dims: int = 15,
) -> dict:
    """Cluster samples with HDBSCAN and compute rich per-cluster summary cards."""
    if cluster_on not in ("2d", "pca"):
        raise ValueError(f"Unsupported cluster_on='{cluster_on}'. Expected '2d' or 'pca'.")
    if scope not in ("all", "train", "clean", "test"):
        raise ValueError(f"Unsupported scope='{scope}'. Expected 'all', 'train', 'clean', or 'test'.")
    if method not in ("eom", "leaf"):
        raise ValueError(f"Unsupported HDBSCAN method='{method}'. Expected 'eom' or 'leaf'.")

    max_c = int(max_cluster_size) if (max_cluster_size is not None and int(max_cluster_size) > 0) else None
    n_samples = len(coords_2d)
    status_arr = np.asarray(samples_meta["status"], dtype=np.int32)
    label_arr = np.array([-1 if v is None else int(v) for v in samples_meta["label"]], dtype=np.int32)
    sky_arr = np.asarray(samples_meta["sky_med"], dtype=np.float32)
    pred_arr = np.asarray(pred, dtype=np.int32)
    conf_arr = np.asarray(conf, dtype=np.float32)

    test_int = STATUS_TO_INT["test"]
    clean_int = STATUS_TO_INT["clean"]
    alien_int = STATUS_TO_INT["alien"]
    over_int = STATUS_TO_INT["overexposed_duplicate"]
    blank_int = STATUS_TO_INT["blank"]

    if scope == "all":
        active_mask = np.ones(n_samples, dtype=bool)
    elif scope == "train":
        active_mask = status_arr != test_int
    elif scope == "clean":
        active_mask = status_arr == clean_int
    else:
        active_mask = status_arr == test_int

    active_idx = np.where(active_mask)[0]
    cluster_out = np.full(n_samples, -2, dtype=np.int32)
    prob_out = np.zeros(n_samples, dtype=np.float32)

    if len(active_idx) >= max(2, int(min_cluster_size)):
        x_feat = coords_2d[active_idx] if cluster_on == "2d" else pca50[active_idx, : int(pca_dims)]
        hdb = HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            max_cluster_size=max_c,
            cluster_selection_method=method,
            metric="euclidean",
            copy=True,
            n_jobs=-1,
        )
        raw_labels = hdb.fit_predict(x_feat)
        raw_probs = getattr(hdb, "probabilities_", np.ones(len(active_idx), dtype=np.float32))

        pos_ids = sorted(int(c) for c in np.unique(raw_labels) if c >= 0)
        pos_ids.sort(key=lambda c: (-int((raw_labels == c).sum()), c))
        remap = {old_id: new_id for new_id, old_id in enumerate(pos_ids)}

        mapped = np.array([remap[int(c)] if c >= 0 else -1 for c in raw_labels], dtype=np.int32)
        cluster_out[active_idx] = mapped
        prob_out[active_idx] = np.asarray(raw_probs, dtype=np.float32)
    elif len(active_idx) > 0:
        cluster_out[active_idx] = -1

    unique_clusters = sorted(int(c) for c in np.unique(cluster_out) if c >= 0)
    ordered_ids = [*unique_clusters]
    if (cluster_out == -1).any():
        ordered_ids.append(-1)

    clusters_summary: list[dict] = []
    for cid in ordered_ids:
        idxs = np.where(cluster_out == cid)[0]
        st = status_arr[idxs]
        lbl = label_arr[idxs]
        pr = pred_arr[idxs]

        train_mask = st != test_int
        train_count = int(train_mask.sum())
        test_count = int((st == test_int).sum())
        clean_count = int((st == clean_int).sum())
        alien_count = int((st == alien_int).sum())
        over_count = int((st == over_int).sum())
        blank_count = int((st == blank_int).sum())
        dropped_count = alien_count + over_count + blank_count

        class_hist = [int((lbl[train_mask] == c).sum()) for c in range(10)] if train_count > 0 else [0] * 10
        pred_hist = [int((pr == c).sum()) for c in range(10)]
        dominant_class = int(np.argmax(class_hist)) if train_count > 0 else None
        dominant_pred = int(np.argmax(pred_hist))
        purity = round(float(max(class_hist) / train_count), 4) if train_count > 0 else None

        cx = float(np.median(coords_2d[idxs, 0]))
        cy = float(np.median(coords_2d[idxs, 1]))
        dists = np.hypot(coords_2d[idxs, 0] - cx, coords_2d[idxs, 1] - cy)

        if cid >= 0:
            cprobs = prob_out[idxs]
            thresh = max(0.5, float(np.quantile(cprobs, 0.5)))
            core_pos = np.where(cprobs >= thresh)[0]
            if len(core_pos) < 8:
                core_pos = np.arange(len(idxs))
            order = core_pos[np.argsort(dists[core_pos])[:8]]
            exemplars = [int(idxs[k]) for k in order]
        else:
            sorted_pos = np.argsort(dists)
            pick = np.linspace(0, len(sorted_pos) - 1, num=min(8, len(sorted_pos)), dtype=int)
            exemplars = [int(idxs[sorted_pos[k]]) for k in pick]

        clusters_summary.append(
            {
                "id": int(cid),
                "count": int(len(idxs)),
                "train_count": train_count,
                "test_count": test_count,
                "clean_count": clean_count,
                "dropped_count": dropped_count,
                "alien_count": alien_count,
                "overexposed_count": over_count,
                "blank_count": blank_count,
                "cx": round(cx, 4),
                "cy": round(cy, 4),
                "class_hist": class_hist,
                "pred_hist": pred_hist,
                "dominant_class": dominant_class,
                "dominant_pred": dominant_pred,
                "purity": purity,
                "mean_conf": round(float(conf_arr[idxs].mean()), 4),
                "mean_sky_med": round(float(sky_arr[idxs].mean()), 1),
                "exemplars": exemplars,
            }
        )

    return {
        "cluster": [int(c) for c in cluster_out],
        "cluster_prob": [round(float(p), 3) for p in prob_out],
        "clusters": clusters_summary,
        "hdbscan_params": {
            "cluster_on": cluster_on,
            "scope": scope,
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
            "max_cluster_size": max_c,
            "method": method,
            "pca_dims": int(pca_dims),
        },
    }


def _build_and_write_view(
    space_dir: Path,
    pca50: np.ndarray,
    coords_2d: np.ndarray,
    probs: np.ndarray,
    is_oof: np.ndarray | list[int],
    samples_meta: dict,
    min_cluster_size: int = 80,
    min_samples: int = 15,
    max_cluster_size: int | None = 3000,
) -> dict:
    """Run HDBSCAN and atomically write ``view.json`` from cached space arrays."""
    pred = probs.argmax(axis=1).astype(np.int32)
    conf = probs.max(axis=1).astype(np.float32)
    clipped = np.clip(probs.astype(np.float64), 1e-7, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1).astype(np.float32)

    hdb_res = run_hdbscan_and_summarize(
        coords_2d=coords_2d,
        pca50=pca50,
        samples_meta=samples_meta,
        pred=pred,
        conf=conf,
        cluster_on="2d",
        scope="all",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        max_cluster_size=max_cluster_size,
        method="eom",
        pca_dims=15,
    )

    view_payload = {
        "x": [round(float(v), 5) for v in coords_2d[:, 0]],
        "y": [round(float(v), 5) for v in coords_2d[:, 1]],
        "cluster": hdb_res["cluster"],
        "cluster_prob": hdb_res["cluster_prob"],
        "pred": [int(v) for v in pred],
        "conf": [round(float(v), 4) for v in conf],
        "entropy": [round(float(v), 4) for v in entropy],
        "is_oof": [int(v) for v in is_oof],
        "hdbscan_params": hdb_res["hdbscan_params"],
        "clusters": hdb_res["clusters"],
    }
    _write_json_atomic(view_payload, space_dir / "view.json")
    return hdb_res


def _save_space_outputs(
    space_dir: Path,
    embeddings: np.ndarray,
    pca50: np.ndarray,
    coords_2d: np.ndarray,
    probs: np.ndarray,
    is_oof: np.ndarray,
    samples_meta: dict,
    min_cluster_size: int = 80,
    min_samples: int = 15,
    max_cluster_size: int | None = 3000,
) -> dict:
    """Persist space arrays + ``view.json`` and return cluster statistics."""
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Non-finite embeddings encountered for {space_dir.name}")
    if not np.isfinite(coords_2d).all():
        raise ValueError(f"Non-finite t-SNE coordinates encountered for {space_dir.name}")
    if not np.isfinite(probs).all():
        raise ValueError(f"Non-finite probabilities encountered for {space_dir.name}")

    _write_npy_atomic(embeddings.astype(np.float16), space_dir / "embeddings_fp16.npy")
    _write_npy_atomic(pca50.astype(np.float32), space_dir / "pca50.npy")
    _write_npy_atomic(coords_2d.astype(np.float32), space_dir / "tsne_2d.npy")
    _write_npy_atomic(probs.astype(np.float32), space_dir / "probs.npy")
    return _build_and_write_view(
        space_dir=space_dir,
        pca50=pca50,
        coords_2d=coords_2d,
        probs=probs,
        is_oof=is_oof,
        samples_meta=samples_meta,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        max_cluster_size=max_cluster_size,
    )


def build_clusters(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    clusters_dir: Path = DEFAULT_CLUSTERS_DIR,
    cnn_ckpt_dir: Path = DEFAULT_CNN_CKPT_DIR,
    cnn_abl_dir: Path = DEFAULT_CNN_ABL_DIR,
    dino_ckpt: Path = DEFAULT_DINO_CKPT,
    dino_metrics_path: Path = DEFAULT_DINO_METRICS,
    dino_variant: str = "vitl14_reg",
    perplexity: float = 35.0,
    min_cluster_size: int = 80,
    min_samples: int = 15,
    max_cluster_size: int | None = 3000,
    seed: int = 42,
) -> dict:
    """Run full feature extraction, PCA, t-SNE, and HDBSCAN across all 3 embedding spaces."""
    t_total = time.perf_counter()
    samples_meta, all_images, id_to_row = load_sample_catalog(
        dataset_dir=dataset_dir,
        train_csv=dataset_dir / "train.csv",
        clean_folds_csv=dataset_dir / "clean_folds.csv",
        dropped_csv=dataset_dir / "dropped_samples.csv",
    )
    clean_df = pd.read_csv(dataset_dir / "clean_folds.csv")
    clean_ids = clean_df["id"].astype(str).tolist()
    fold_arr = np.asarray(samples_meta["fold"], dtype=np.int32)

    oof_metrics_map: dict[str, float] = {}
    oof_summary_path = cnn_abl_dir / "oof_summary.json"
    if oof_summary_path.exists():
        oof_summary = json.loads(oof_summary_path.read_text(encoding="utf-8"))
        for fam in oof_summary.get("completed_families", []):
            oof_metrics_map[str(fam["arch"])] = float(fam["weighted_f1"])

    dino_wf1 = None
    if dino_metrics_path.exists():
        dino_metrics = json.loads(dino_metrics_path.read_text(encoding="utf-8"))
        dino_wf1 = float(dino_metrics["best_val_weighted_f1"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spaces_meta: list[dict] = []

    space_specs = [
        {
            "id": "efficientnet_v2_s",
            "name": "EfficientNetV2-S",
            "subtitle": "5-fold OOF · 6,400d",
            "arch": "efficientnet_v2_s",
            "stage": "stage3a_photo",
            "folds": [0, 1, 2, 3, 4],
            "val_wf1": oof_metrics_map.get("efficientnet_v2_s"),
            "kind": "cnn",
            "oof_path": cnn_abl_dir / "oof_efficientnet_v2_s__stage3a_photo.npy",
            "ckpts": [
                cnn_ckpt_dir / f"efficientnet_v2_s__pretrained__stage3a_photo__fold{k}.pt" for k in range(5)
            ],
        },
        {
            "id": "convnext_tiny",
            "name": "ConvNeXt-Tiny",
            "subtitle": "5-fold OOF · 3,840d",
            "arch": "convnext_tiny",
            "stage": "stage3a_photo",
            "folds": [0, 1, 2, 3, 4],
            "val_wf1": oof_metrics_map.get("convnext_tiny"),
            "kind": "cnn",
            "oof_path": cnn_abl_dir / "oof_convnext_tiny__stage3a_photo.npy",
            "ckpts": [
                cnn_ckpt_dir / f"convnext_tiny__pretrained__stage3a_photo__fold{k}.pt" for k in range(5)
            ],
        },
        {
            "id": f"dinov2_{dino_variant}",
            "name": "DINOv2-L/14 (reg)",
            "subtitle": "Fold 0 · 1,024d",
            "arch": f"dinov2_{dino_variant}",
            "stage": "stage3a_photo",
            "folds": [0],
            "val_wf1": dino_wf1,
            "kind": "dinov2",
            "ckpt": dino_ckpt,
        },
    ]

    for spec in space_specs:
        sid = spec["id"]
        print(f"\n=== Building cluster space: {sid} ===", flush=True)
        if spec["kind"] == "cnn":
            for p in spec["ckpts"]:
                if not p.exists():
                    raise FileNotFoundError(f"Missing CNN checkpoint: {p}")
            embeddings, probs, is_oof = _extract_cnn_5fold(
                arch=spec["arch"],
                ckpt_paths=spec["ckpts"],
                images=all_images,
                oof_probs_path=spec["oof_path"],
                clean_ids=clean_ids,
                id_to_row=id_to_row,
                batch_size=128,
                device=device,
            )
        else:
            if not spec["ckpt"].exists():
                raise FileNotFoundError(f"Missing DINOv2 checkpoint: {spec['ckpt']}")
            embeddings, probs, is_oof = _extract_dinov2(
                variant=dino_variant,
                ckpt_path=spec["ckpt"],
                images=all_images,
                fold_arr=fold_arr,
                val_fold=0,
                img_size=224,
                batch_size=64,
                device=device,
            )

        pca50, coords_2d, pca_var = compute_pca_and_tsne(
            embeddings=embeddings,
            pca_components=50,
            perplexity=perplexity,
            seed=seed,
        )
        hdb_res = _save_space_outputs(
            space_dir=clusters_dir / sid,
            embeddings=embeddings,
            pca50=pca50,
            coords_2d=coords_2d,
            probs=probs,
            is_oof=is_oof,
            samples_meta=samples_meta,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            max_cluster_size=max_cluster_size,
        )

        cl_list = hdb_res["clusters"]
        n_clusters = sum(1 for c in cl_list if c["id"] >= 0)
        noise_count = next((c["count"] for c in cl_list if c["id"] == -1), 0)
        print(
            f"  saved {sid} | emb={embeddings.shape} | clusters={n_clusters} "
            f"noise={noise_count} ({100.0 * noise_count / len(all_images):.1f}%)",
            flush=True,
        )

        spaces_meta.append(
            {
                "id": sid,
                "name": spec["name"],
                "subtitle": spec["subtitle"],
                "arch": spec["arch"],
                "stage": spec["stage"],
                "folds": spec["folds"],
                "embed_dim": int(embeddings.shape[1]),
                "pca_var_explained": round(pca_var, 4),
                "val_wf1": round(float(spec["val_wf1"]), 4) if spec["val_wf1"] is not None else None,
                "n_clusters": n_clusters,
                "noise_count": noise_count,
            }
        )

        del embeddings, pca50, coords_2d, probs, is_oof
        gc.collect()

    meta_payload = {
        "version": 1,
        "n_samples": len(samples_meta["ids"]),
        "status_names": list(STATUS_NAMES),
        "spaces": spaces_meta,
        "samples": samples_meta,
    }
    _write_json_atomic(meta_payload, clusters_dir / "meta.json")
    print(f"\nDone building all cluster spaces in {time.perf_counter() - t_total:.1f}s -> {clusters_dir}", flush=True)
    return meta_payload


def recluster_existing_spaces(
    clusters_dir: Path = DEFAULT_CLUSTERS_DIR,
    min_cluster_size: int = 80,
    min_samples: int = 15,
    max_cluster_size: int | None = 3000,
) -> dict:
    """Re-run HDBSCAN across already extracted spaces and update ``view.json`` + ``meta.json``."""
    meta_path = clusters_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} does not exist; run full build-clusters first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    samples_meta = meta["samples"]
    n_samples = int(meta["n_samples"])

    t0 = time.perf_counter()
    for sp in meta["spaces"]:
        sid = str(sp["id"])
        sdir = clusters_dir / sid
        pca50 = np.load(sdir / "pca50.npy").astype(np.float32)
        coords_2d = np.load(sdir / "tsne_2d.npy").astype(np.float32)
        probs = np.load(sdir / "probs.npy").astype(np.float32)
        prev_view = json.loads((sdir / "view.json").read_text(encoding="utf-8"))
        is_oof = prev_view["is_oof"]

        hdb_res = _build_and_write_view(
            space_dir=sdir,
            pca50=pca50,
            coords_2d=coords_2d,
            probs=probs,
            is_oof=is_oof,
            samples_meta=samples_meta,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            max_cluster_size=max_cluster_size,
        )
        cl_list = hdb_res["clusters"]
        n_clusters = sum(1 for c in cl_list if c["id"] >= 0)
        noise_count = next((c["count"] for c in cl_list if c["id"] == -1), 0)
        sp["n_clusters"] = n_clusters
        sp["noise_count"] = noise_count
        print(
            f"  reclustered {sid} | clusters={n_clusters} "
            f"noise={noise_count} ({100.0 * noise_count / n_samples:.1f}%)",
            flush=True,
        )

    _write_json_atomic(meta, meta_path)
    print(f"Reclustered {len(meta['spaces'])} spaces in {time.perf_counter() - t0:.1f}s", flush=True)
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for cluster data generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--clusters-dir", type=Path, default=DEFAULT_CLUSTERS_DIR)
    parser.add_argument("--cnn-ckpt-dir", type=Path, default=DEFAULT_CNN_CKPT_DIR)
    parser.add_argument("--cnn-abl-dir", type=Path, default=DEFAULT_CNN_ABL_DIR)
    parser.add_argument("--dino-ckpt", type=Path, default=DEFAULT_DINO_CKPT)
    parser.add_argument("--dino-metrics", type=Path, default=DEFAULT_DINO_METRICS)
    parser.add_argument("--dino-variant", default="vitl14_reg")
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--min-cluster-size", type=int, default=80)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--max-cluster-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recluster-only", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    max_c = args.max_cluster_size if args.max_cluster_size and args.max_cluster_size > 0 else None
    if args.recluster_only:
        recluster_existing_spaces(
            clusters_dir=args.clusters_dir,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            max_cluster_size=max_c,
        )
        return
    build_clusters(
        dataset_dir=args.dataset_dir,
        clusters_dir=args.clusters_dir,
        cnn_ckpt_dir=args.cnn_ckpt_dir,
        cnn_abl_dir=args.cnn_abl_dir,
        dino_ckpt=args.dino_ckpt,
        dino_metrics_path=args.dino_metrics,
        dino_variant=args.dino_variant,
        perplexity=args.perplexity,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        max_cluster_size=max_c,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
