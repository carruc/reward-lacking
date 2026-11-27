"""Measure cross-model embedding overlap via exact k-NN neighborhood sharing and CKA.

Computes:
1. Exact cosine k-NN shared-neighbor fractions (k in {10, 20, 50} on raw embeddings,
   plus k=20 on PCA-50) across EfficientNetV2-S, ConvNeXt-Tiny, and DINOv2-L/14 (reg),
   including pairwise, 3-way consensus, per-status, per-class, and per-sample arrays.
2. Exact Linear CKA (invariant to orthogonal transforms & isotropic scaling) across
   subsets (all, clean, test, dropped, alien, overexposed_duplicate) + PCA-50, and
   median-bandwidth RBF CKA on clean train samples (n=3,004).
3. HDBSCAN cluster agreement (ARI / NMI) across the three embedding spaces.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTERS_DIR = REPO_ROOT / "artifacts" / "clusters"
DEFAULT_SPACE_IDS = ("efficientnet_v2_s", "convnext_tiny", "dinov2_vitl14_reg")
CLASS_NAMES = (
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on no Bulge",
    "Edge-on with Bulge",
)


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _l2_normalize_f32(arr: np.ndarray) -> np.ndarray:
    arr_f = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr_f, axis=1, keepdims=True)
    return (arr_f / np.maximum(norms, 1e-12)).astype(np.float32)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Compute exact Linear CKA invariant to orthogonal transforms and isotropic scaling."""
    if len(x) != len(y):
        raise ValueError(f"Row count mismatch: {len(x)} vs {len(y)}")
    if len(x) < 2:
        return 0.0
    xc = np.asarray(x, dtype=np.float64) - np.mean(x, axis=0, dtype=np.float64)
    yc = np.asarray(y, dtype=np.float64) - np.mean(y, axis=0, dtype=np.float64)
    norm_xx = float(np.linalg.norm(xc.T @ xc, ord="fro"))
    norm_yy = float(np.linalg.norm(yc.T @ yc, ord="fro"))
    denom = norm_xx * norm_yy
    if denom <= 0.0:
        return 0.0
    hsic_xy = float(np.linalg.norm(yc.T @ xc, ord="fro") ** 2)
    return min(1.0, max(0.0, hsic_xy / denom))


def linear_cka_matrix(arrays: list[np.ndarray]) -> list[list[float]]:
    """Compute the symmetric Linear CKA matrix across a list of embedding matrices."""
    centered = [np.asarray(a, dtype=np.float64) - np.mean(a, axis=0, dtype=np.float64) for a in arrays]
    self_norms = [float(np.linalg.norm(c.T @ c, ord="fro")) for c in centered]
    m = len(arrays)
    mat = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            denom = self_norms[i] * self_norms[j]
            if denom <= 0.0:
                val = 0.0
            else:
                hsic = float(np.linalg.norm(centered[j].T @ centered[i], ord="fro") ** 2)
                val = min(1.0, max(0.0, hsic / denom))
            rval = round(val, 4)
            mat[i][j] = rval
            mat[j][i] = rval
    return mat


def _sq_dists_and_median_sigma_sq(x: np.ndarray) -> tuple[np.ndarray, float]:
    xf = np.asarray(x, dtype=np.float64)
    sq_norms = np.sum(xf * xf, axis=1, keepdims=True)
    sq_dists = np.maximum(sq_norms + sq_norms.T - 2.0 * (xf @ xf.T), 0.0)
    np.fill_diagonal(sq_dists, 0.0)
    iu = np.triu_indices(len(xf), k=1)
    upper = sq_dists[iu]
    pos = upper[upper > 0.0]
    med_sq = float(np.median(pos)) if len(pos) > 0 else 1.0
    return sq_dists, max(med_sq, 1e-12)


def _center_gram(k_mat: np.ndarray) -> np.ndarray:
    row_mean = k_mat.mean(axis=1, keepdims=True)
    col_mean = k_mat.mean(axis=0, keepdims=True)
    total_mean = k_mat.mean()
    return k_mat - row_mean - col_mean + total_mean


def rbf_cka(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Compute median-heuristic RBF-kernel CKA and return ``(cka, sigma_x, sigma_y)``."""
    if len(x) != len(y):
        raise ValueError(f"Row count mismatch: {len(x)} vs {len(y)}")
    if len(x) < 3:
        return 0.0, 0.0, 0.0
    sq_x, sig2_x = _sq_dists_and_median_sigma_sq(x)
    sq_y, sig2_y = _sq_dists_and_median_sigma_sq(y)
    kx = _center_gram(np.exp(-sq_x / (2.0 * sig2_x)))
    ky = _center_gram(np.exp(-sq_y / (2.0 * sig2_y)))
    denom = float(np.linalg.norm(kx, ord="fro") * np.linalg.norm(ky, ord="fro"))
    if denom <= 0.0:
        return 0.0, float(np.sqrt(sig2_x)), float(np.sqrt(sig2_y))
    val = min(1.0, max(0.0, float(np.sum(kx * ky) / denom)))
    return val, float(np.sqrt(sig2_x)), float(np.sqrt(sig2_y))


def rbf_cka_matrix(arrays: list[np.ndarray]) -> tuple[list[list[float]], list[float]]:
    """Compute the symmetric RBF-CKA matrix and per-space median sigmas."""
    centered_grams: list[np.ndarray] = []
    norms: list[float] = []
    sigmas: list[float] = []
    for arr in arrays:
        sq, sig2 = _sq_dists_and_median_sigma_sq(arr)
        kc = _center_gram(np.exp(-sq / (2.0 * sig2)))
        centered_grams.append(kc)
        norms.append(float(np.linalg.norm(kc, ord="fro")))
        sigmas.append(round(float(np.sqrt(sig2)), 4))

    m = len(arrays)
    mat = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            denom = norms[i] * norms[j]
            val = min(1.0, max(0.0, float(np.sum(centered_grams[i] * centered_grams[j]) / denom))) if denom > 0 else 0.0
            rval = round(val, 4)
            mat[i][j] = rval
            mat[j][i] = rval
    return mat, sigmas


@torch.no_grad()
def exact_topk_cosine(
    embeddings: np.ndarray,
    max_k: int = 50,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute exact top-``max_k`` cosine neighbor indices (excluding self) for every row."""
    normed = _l2_normalize_f32(embeddings)
    n = len(normed)
    if max_k >= n:
        raise ValueError(f"max_k={max_k} must be strictly less than sample count {n}.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_dev = torch.from_numpy(normed).to(device)
    out = np.empty((n, max_k), dtype=np.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = x_dev[start:end] @ x_dev.T
        local_rows = torch.arange(end - start, device=device)
        global_cols = torch.arange(start, end, device=device)
        sims[local_rows, global_cols] = -float("inf")
        _, idx = torch.topk(sims, k=max_k, dim=1, largest=True, sorted=True)
        out[start:end] = idx.cpu().numpy().astype(np.int32)

    del x_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _summarize_array(vals: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(vals)), 4),
        "median": round(float(np.median(vals)), 4),
        "std": round(float(np.std(vals)), 4),
        "p10": round(float(np.quantile(vals, 0.10)), 4),
        "p90": round(float(np.quantile(vals, 0.90)), 4),
    }


def _pairwise_overlap_fractions(
    topk_a: np.ndarray,
    topk_b: np.ndarray,
    k: int,
) -> np.ndarray:
    sets_a = [set(r.tolist()) for r in topk_a[:, :k]]
    sets_b = [set(r.tolist()) for r in topk_b[:, :k]]
    n = len(sets_a)
    counts = np.fromiter((len(sa & sb) for sa, sb in zip(sets_a, sets_b, strict=True)), dtype=np.float32, count=n)
    return counts / float(k)


def _threeway_overlap_fractions(
    topk_a: np.ndarray,
    topk_b: np.ndarray,
    topk_c: np.ndarray,
    k: int,
) -> np.ndarray:
    sets_a = [set(r.tolist()) for r in topk_a[:, :k]]
    sets_b = [set(r.tolist()) for r in topk_b[:, :k]]
    sets_c = [set(r.tolist()) for r in topk_c[:, :k]]
    n = len(sets_a)
    counts = np.fromiter(
        (len(sa & sb & sc) for sa, sb, sc in zip(sets_a, sets_b, sets_c, strict=True)),
        dtype=np.float32,
        count=n,
    )
    return counts / float(k)


def compute_knn_overlap(
    topk_map: dict[str, np.ndarray],
    space_ids: tuple[str, ...],
    ks: tuple[int, ...] = (10, 20, 50),
    status_arr: np.ndarray | None = None,
    status_names: tuple[str, ...] = (),
    label_arr: np.ndarray | None = None,
    record_k: int = 20,
) -> dict:
    """Compute exact pairwise & 3-way k-NN overlap matrices and demographic breakdowns."""
    if len(space_ids) != 3:
        raise ValueError(f"Expected 3 space_ids, got {space_ids}")
    s0, s1, s2 = space_ids
    n = len(topk_map[s0])

    by_k: dict[str, dict] = {}
    per_sample_k20: dict[str, list[float]] = {}

    for k in ks:
        ov_01 = _pairwise_overlap_fractions(topk_map[s0], topk_map[s1], k)
        ov_02 = _pairwise_overlap_fractions(topk_map[s0], topk_map[s2], k)
        ov_12 = _pairwise_overlap_fractions(topk_map[s1], topk_map[s2], k)
        ov_3 = _threeway_overlap_fractions(topk_map[s0], topk_map[s1], topk_map[s2], k)
        mean_pair = (ov_01 + ov_02 + ov_12) / 3.0

        mat = [
            [1.0, round(float(ov_01.mean()), 4), round(float(ov_02.mean()), 4)],
            [round(float(ov_01.mean()), 4), 1.0, round(float(ov_12.mean()), 4)],
            [round(float(ov_02.mean()), 4), round(float(ov_12.mean()), 4), 1.0],
        ]
        by_k[str(k)] = {
            "k": int(k),
            "random_baseline": round(float(k / max(1, n - 1)), 6),
            "matrix": mat,
            "pairs": {
                "eff_conv": _summarize_array(ov_01),
                "eff_dino": _summarize_array(ov_02),
                "conv_dino": _summarize_array(ov_12),
            },
            "mean_pairwise": _summarize_array(mean_pair),
            "three_way": _summarize_array(ov_3),
        }

        if k == record_k:
            per_sample_k20 = {
                "mean_pairwise": [round(float(v), 3) for v in mean_pair],
                "three_way": [round(float(v), 3) for v in ov_3],
                "eff_conv": [round(float(v), 3) for v in ov_01],
                "eff_dino": [round(float(v), 3) for v in ov_02],
                "conv_dino": [round(float(v), 3) for v in ov_12],
            }

            by_status: dict[str, dict] = {}
            if status_arr is not None and status_names:
                for st_idx, st_name in enumerate(status_names):
                    mask = status_arr == st_idx
                    if not mask.any():
                        continue
                    by_status[st_name] = {
                        "count": int(mask.sum()),
                        "mean_pairwise": round(float(mean_pair[mask].mean()), 4),
                        "three_way": round(float(ov_3[mask].mean()), 4),
                        "eff_conv": round(float(ov_01[mask].mean()), 4),
                        "eff_dino": round(float(ov_02[mask].mean()), 4),
                        "conv_dino": round(float(ov_12[mask].mean()), 4),
                    }

            by_class: list[dict] = []
            if status_arr is not None and label_arr is not None:
                clean_mask = status_arr == 0
                for c in range(10):
                    cmask = clean_mask & (label_arr == c)
                    if not cmask.any():
                        continue
                    by_class.append(
                        {
                            "class": int(c),
                            "name": CLASS_NAMES[c],
                            "count": int(cmask.sum()),
                            "mean_pairwise": round(float(mean_pair[cmask].mean()), 4),
                            "three_way": round(float(ov_3[cmask].mean()), 4),
                            "eff_conv": round(float(ov_01[cmask].mean()), 4),
                            "eff_dino": round(float(ov_02[cmask].mean()), 4),
                            "conv_dino": round(float(ov_12[cmask].mean()), 4),
                        }
                    )

            by_k[str(k)]["by_status"] = by_status
            by_k[str(k)]["by_class_clean"] = by_class

    return {"by_k": by_k, "per_sample_k20": per_sample_k20}


def compute_cluster_agreement(clusters_dir: Path, space_ids: tuple[str, ...]) -> dict:
    """Compute pairwise ARI and NMI across HDBSCAN clusterings of the 3 spaces."""
    labels_by_space = []
    for sid in space_ids:
        view = json.loads((clusters_dir / sid / "view.json").read_text(encoding="utf-8"))
        labels_by_space.append(np.asarray(view["cluster"], dtype=np.int32))

    m = len(space_ids)
    ari_all = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    nmi_all = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    ari_non_noise = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    nmi_non_noise = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    mutual_counts = [[int((labels_by_space[i] >= 0).sum()) if i == j else 0 for j in range(m)] for i in range(m)]

    for i in range(m):
        for j in range(i + 1, m):
            a, b = labels_by_space[i], labels_by_space[j]
            va = round(float(adjusted_rand_score(a, b)), 4)
            vn = round(float(normalized_mutual_info_score(a, b)), 4)
            ari_all[i][j] = ari_all[j][i] = va
            nmi_all[i][j] = nmi_all[j][i] = vn

            mask = (a >= 0) & (b >= 0)
            cnt = int(mask.sum())
            mutual_counts[i][j] = mutual_counts[j][i] = cnt
            if cnt >= 2:
                va_nn = round(float(adjusted_rand_score(a[mask], b[mask])), 4)
                vn_nn = round(float(normalized_mutual_info_score(a[mask], b[mask])), 4)
            else:
                va_nn = 0.0
                vn_nn = 0.0
            ari_non_noise[i][j] = ari_non_noise[j][i] = va_nn
            nmi_non_noise[i][j] = nmi_non_noise[j][i] = vn_nn

    return {
        "ari_all": ari_all,
        "nmi_all": nmi_all,
        "ari_non_noise": ari_non_noise,
        "nmi_non_noise": nmi_non_noise,
        "mutual_non_noise_counts": mutual_counts,
    }


def _format_matrix_md(title: str, names: list[str], mat: list[list[float]]) -> list[str]:
    lines = [f"### {title}", ""]
    header = "| Space | " + " | ".join(names) + " |"
    sep = "| :--- | " + " | ".join([":---:"] * len(names)) + " |"
    lines.extend([header, sep])
    for name, row in zip(names, mat, strict=True):
        lines.append("| " + name + " | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    lines.append("")
    return lines


def _build_markdown_report(payload: dict) -> str:
    names = [sp["name"] for sp in payload["spaces"]]
    lines = [
        f"# Cross-Model Embedding Overlap Report (n={payload['n_samples']:,})",
        "",
        "## 1. Exact k-NN Shared Neighborhood Overlap",
        "",
        "| k / Space | Random Baseline | EffNetV2 ↔ ConvNeXt | EffNetV2 ↔ DINOv2 | ConvNeXt ↔ DINOv2 | Mean Pairwise | 3-Way Consensus |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k_str in ("10", "20", "50"):
        d = payload["knn_raw"]["by_k"][k_str]
        p = d["pairs"]
        lines.append(
            f"| Raw k={k_str} | {d['random_baseline']:.5f} | "
            f"{p['eff_conv']['mean']:.4f} | {p['eff_dino']['mean']:.4f} | {p['conv_dino']['mean']:.4f} | "
            f"**{d['mean_pairwise']['mean']:.4f}** | **{d['three_way']['mean']:.4f}** |"
        )
    d_pca = payload["knn_pca50"]["by_k"]["20"]
    p_pca = d_pca["pairs"]
    lines.append(
        f"| PCA-50 k=20 | {d_pca['random_baseline']:.5f} | "
        f"{p_pca['eff_conv']['mean']:.4f} | {p_pca['eff_dino']['mean']:.4f} | {p_pca['conv_dino']['mean']:.4f} | "
        f"**{d_pca['mean_pairwise']['mean']:.4f}** | **{d_pca['three_way']['mean']:.4f}** |"
    )
    lines.append("")

    k20 = payload["knn_raw"]["by_k"]["20"]
    lines.extend(
        [
            "### k=20 Breakdown by Data Status",
            "",
            "| Status | Count | EffNetV2 ↔ ConvNeXt | EffNetV2 ↔ DINOv2 | ConvNeXt ↔ DINOv2 | Mean Pairwise | 3-Way |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for st_name, row in k20["by_status"].items():
        lines.append(
            f"| `{st_name}` | {row['count']:,} | {row['eff_conv']:.4f} | {row['eff_dino']:.4f} | "
            f"{row['conv_dino']:.4f} | **{row['mean_pairwise']:.4f}** | {row['three_way']:.4f} |"
        )
    lines.append("")

    lines.extend(
        [
            "### k=20 Breakdown by Clean Train Class",
            "",
            "| Class | Count | EffNetV2 ↔ ConvNeXt | EffNetV2 ↔ DINOv2 | ConvNeXt ↔ DINOv2 | Mean Pairwise | 3-Way |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for row in k20["by_class_clean"]:
        lines.append(
            f"| `{row['class']}` {row['name']} | {row['count']} | {row['eff_conv']:.4f} | "
            f"{row['eff_dino']:.4f} | {row['conv_dino']:.4f} | **{row['mean_pairwise']:.4f}** | {row['three_way']:.4f} |"
        )
    lines.append("")

    lines.extend(["## 2. Global Relational Similarity (CKA)", ""])
    for key, label in (
        ("linear_all", "Linear CKA — All Samples (n=19,064)"),
        ("linear_clean", "Linear CKA — Clean Train (n=3,004)"),
        ("linear_test", "Linear CKA — Test (n=14,736)"),
        ("linear_dropped", "Linear CKA — Dropped Train (n=1,324)"),
        ("linear_alien", "Linear CKA — Alien (n=327)"),
        ("linear_overexposed", "Linear CKA — Overexposed Duplicates (n=996)"),
        ("linear_pca50_all", "Linear CKA — PCA-50 All Samples (n=19,064)"),
        ("rbf_clean", "RBF-Kernel CKA — Clean Train (n=3,004, median heuristic)"),
    ):
        if key in payload["cka"]:
            lines.extend(_format_matrix_md(label, names, payload["cka"][key]["matrix"]))

    lines.extend(["## 3. HDBSCAN Cluster Agreement", ""])
    lines.extend(_format_matrix_md("Adjusted Rand Index (All Points)", names, payload["hdbscan_agreement"]["ari_all"]))
    lines.extend(_format_matrix_md("Normalized Mutual Info (All Points)", names, payload["hdbscan_agreement"]["nmi_all"]))
    lines.extend(
        _format_matrix_md(
            "Adjusted Rand Index (Mutual Non-Noise Points)",
            names,
            payload["hdbscan_agreement"]["ari_non_noise"],
        )
    )
    return "\n".join(lines) + "\n"


def compute_embedding_overlap(clusters_dir: Path = DEFAULT_CLUSTERS_DIR) -> dict:
    """Compute k-NN overlap, CKA matrices, and HDBSCAN agreement across the 3 spaces."""
    t0 = time.perf_counter()
    meta_path = clusters_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run `uv run build-clusters` first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    spaces_meta = meta["spaces"]
    space_ids = tuple(str(sp["id"]) for sp in spaces_meta)
    n_samples = int(meta["n_samples"])
    status_names = tuple(str(s) for s in meta["status_names"])
    status_arr = np.asarray(meta["samples"]["status"], dtype=np.int32)
    label_arr = np.array([-1 if v is None else int(v) for v in meta["samples"]["label"]], dtype=np.int32)

    raw_embs: list[np.ndarray] = []
    pca_embs: list[np.ndarray] = []
    for sid in space_ids:
        sdir = clusters_dir / sid
        raw = _l2_normalize_f32(np.load(sdir / "embeddings_fp16.npy"))
        pca = _l2_normalize_f32(np.load(sdir / "pca50.npy"))
        raw_embs.append(raw)
        pca_embs.append(pca)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing exact top-50 k-NN on {device.type} for {len(space_ids)} spaces (n={n_samples})...", flush=True)
    topk_raw = {sid: exact_topk_cosine(emb, max_k=50, device=device) for sid, emb in zip(space_ids, raw_embs, strict=True)}
    topk_pca = {sid: exact_topk_cosine(emb, max_k=20, device=device) for sid, emb in zip(space_ids, pca_embs, strict=True)}

    knn_raw = compute_knn_overlap(
        topk_map=topk_raw,
        space_ids=space_ids,
        ks=(10, 20, 50),
        status_arr=status_arr,
        status_names=status_names,
        label_arr=label_arr,
        record_k=20,
    )
    knn_pca = compute_knn_overlap(
        topk_map=topk_pca,
        space_ids=space_ids,
        ks=(20,),
        status_arr=None,
        label_arr=None,
        record_k=-1,
    )

    print("Computing exact Linear CKA and clean-split RBF CKA...", flush=True)
    status_to_idx = {name: i for i, name in enumerate(status_names)}
    clean_mask = status_arr == status_to_idx["clean"]
    test_mask = status_arr == status_to_idx["test"]
    alien_mask = status_arr == status_to_idx["alien"]
    over_mask = status_arr == status_to_idx["overexposed_duplicate"]
    dropped_mask = (status_arr != status_to_idx["clean"]) & (status_arr != status_to_idx["test"])

    cka_results: dict[str, dict] = {
        "linear_all": {
            "label": "All samples",
            "count": n_samples,
            "matrix": linear_cka_matrix(raw_embs),
        },
        "linear_clean": {
            "label": "Clean train",
            "count": int(clean_mask.sum()),
            "matrix": linear_cka_matrix([e[clean_mask] for e in raw_embs]),
        },
        "linear_test": {
            "label": "Test split",
            "count": int(test_mask.sum()),
            "matrix": linear_cka_matrix([e[test_mask] for e in raw_embs]),
        },
        "linear_dropped": {
            "label": "Dropped train",
            "count": int(dropped_mask.sum()),
            "matrix": linear_cka_matrix([e[dropped_mask] for e in raw_embs]),
        },
        "linear_alien": {
            "label": "Alien",
            "count": int(alien_mask.sum()),
            "matrix": linear_cka_matrix([e[alien_mask] for e in raw_embs]),
        },
        "linear_overexposed": {
            "label": "Overexposed dup",
            "count": int(over_mask.sum()),
            "matrix": linear_cka_matrix([e[over_mask] for e in raw_embs]),
        },
        "linear_pca50_all": {
            "label": "PCA-50 (all)",
            "count": n_samples,
            "matrix": linear_cka_matrix(pca_embs),
        },
    }

    rbf_mat, sigmas = rbf_cka_matrix([e[clean_mask] for e in raw_embs])
    cka_results["rbf_clean"] = {
        "label": "RBF CKA (clean)",
        "count": int(clean_mask.sum()),
        "matrix": rbf_mat,
        "sigmas": sigmas,
    }

    hdb_agreement = compute_cluster_agreement(clusters_dir, space_ids)

    payload = {
        "version": 1,
        "n_samples": n_samples,
        "spaces": [
            {
                "id": str(sp["id"]),
                "name": str(sp["name"]),
                "embed_dim": int(sp["embed_dim"]),
                "val_wf1": sp.get("val_wf1"),
            }
            for sp in spaces_meta
        ],
        "knn_raw": {"by_k": knn_raw["by_k"]},
        "knn_pca50": {"by_k": knn_pca["by_k"]},
        "per_sample_k20": knn_raw["per_sample_k20"],
        "cka": cka_results,
        "hdbscan_agreement": hdb_agreement,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

    out_json = clusters_dir / "overlap.json"
    out_md = clusters_dir / "overlap_summary.md"
    _write_json_atomic(payload, out_json)
    md_text = _build_markdown_report(payload)
    _write_text_atomic(md_text, out_md)
    print(md_text, flush=True)
    print(f"Saved {out_json} and {out_md} in {payload['elapsed_sec']}s", flush=True)
    return payload


def run_self_test() -> None:
    """Verify CKA invariance properties and exact k-NN overlap calculations."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(180, 32))
    y = rng.normal(size=(180, 24))
    q_mat, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    shift = rng.normal(size=(1, 32)) * 5.0
    x_trans = 7.3 * (x @ q_mat) + shift

    assert abs(linear_cka(x, x) - 1.0) < 1e-6, "Linear CKA self-similarity failed"
    assert abs(linear_cka(x, y) - linear_cka(y, x)) < 1e-12, "Linear CKA symmetry failed"
    assert abs(linear_cka(x, y) - linear_cka(x_trans, y)) < 1e-6, "Linear CKA orthogonal/scale/shift invariance failed"

    rbf_self, _, _ = rbf_cka(x, x)
    rbf_orig, _, _ = rbf_cka(x, y)
    rbf_trans, _, _ = rbf_cka(x_trans, y)
    assert abs(rbf_self - 1.0) < 1e-6, "RBF CKA self-similarity failed"
    assert abs(rbf_orig - rbf_trans) < 1e-6, "RBF CKA invariance failed"

    # Synthetic k-NN check
    z = _l2_normalize_f32(rng.normal(size=(60, 16)))
    z_rot = z @ np.linalg.qr(rng.normal(size=(16, 16)))[0].astype(np.float32)
    tk1 = exact_topk_cosine(z, max_k=10, device=torch.device("cpu"))
    tk2 = exact_topk_cosine(z_rot, max_k=10, device=torch.device("cpu"))
    assert not (tk1 == np.arange(60)[:, None]).any(), "Self included in top-k neighbors"
    ov = compute_knn_overlap(
        {"a": tk1, "b": tk2, "c": tk1},
        ("a", "b", "c"),
        ks=(5, 10),
        record_k=10,
    )
    assert abs(ov["by_k"]["10"]["three_way"]["mean"] - 1.0) < 1e-6, "Rotated k-NN overlap should be 1.0"
    print("Self-test PASSED: Linear CKA, RBF CKA invariance, and exact k-NN overlap verified.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for embedding overlap evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-dir", type=Path, default=DEFAULT_CLUSTERS_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    compute_embedding_overlap(clusters_dir=args.clusters_dir)


if __name__ == "__main__":
    main()
