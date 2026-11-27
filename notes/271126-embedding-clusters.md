# Embedding Clustering & Cross-Model Similarity (`n=19,064`)

## 1. Fine-Tuned Models (`stage3a_photo` on `clean_folds.csv`)

- **EfficientNetV2-S** (`efficientnet_v2_s`): 5-fold full fine-tune at `256×256` (`OOF WF1 0.8276`); embeddings are per-fold L2-normalized flattened `avgpool` vectors concatenated across 5 folds (`6,400d`).
- **ConvNeXt-Tiny** (`convnext_tiny`): 5-fold full fine-tune at `256×256` (`OOF WF1 0.8315`); embeddings are per-fold L2-normalized `LayerNorm2d + Flatten` features concatenated across 5 folds (`3,840d`).
- **DINOv2-L/14 (reg)** (`vit_large_patch14_reg4_dinov2.lvd142m`): Fold 0 fine-tune at `224×224` with the last 6 transformer blocks, `norm`, and `head` unfrozen (`Clean Val WF1 0.8111`, `Shifted 0.8065`, `Hard-Shifted 0.7927`); embeddings are L2-normalized post-norm CLS pre-logits (`1,024d`).

## 2. t-SNE & HDBSCAN Visualization (`data/gallery/`)

All `19,064` samples (`3,004` clean train, `1,324` dropped train, `14,736` test) are reduced with `PCA(50)` (`79.4–79.8%` variance explained) → 2D `t-SNE` (`perplexity=35`, `barnes_hut`) and clustered with `HDBSCAN(min_cluster_size=80, min_samples=15, max_cluster_size=3000, method="eom")`:
- **Cluster counts / noise**: EfficientNetV2-S `29` clusters (`28.3%` noise), ConvNeXt-Tiny `24` (`20.0%`), DINOv2-L/14 `25` (`32.3%`). All `327` labeled `alien` images isolate into a single `332`-image cluster (`test=0`) in every space.

## 3. Cross-Model Similarity (`eval/embedding_overlap.py`)

| Metric | EffNetV2 ↔ ConvNeXt | ConvNeXt ↔ DINOv2 | EffNetV2 ↔ DINOv2 | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Linear CKA** (`all`, `n=19,064`) | `0.9051` | `0.8838` | `0.8388` | `PCA-50`: `0.9023` / `0.8719` / `0.8353` |
| **Linear CKA** (`clean`, `n=3,004`) | `0.9078` | `0.8956` | `0.8492` | `dropped` (`1,324`): `0.9407` / `0.9340` / `0.9179` |
| **RBF-Kernel CKA** (`clean`, median $\sigma$) | `0.9162` | `0.9014` | `0.8612` | Centered RBF Gram matrices |
| **Exact k-NN@20 Overlap** (raw) | `0.2400` | `0.1456` | `0.1230` | Mean `0.1695`, 3-way `0.0588` (`rand=0.00105`); `alien` mean `0.5398` |
| **HDBSCAN Mutual Non-Noise ARI** | `0.7617` | `0.7393` | `0.7213` | All-point NMI: `0.5547` / `0.5025` / `0.4712` |
