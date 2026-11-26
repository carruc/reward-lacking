# Submission #1 — EfficientNetV2-S (Fold 0)

First public leaderboard submission: **`0.8328` Weighted F1** (Top 20; current #1: `0.8765`), trained on Fold 0 (`80%` of cleaned data) to establish a clean, shift-resistant baseline.

## Choices & Measured Effects (Fold 0)

| Axis | Choice | Rationale | Effect on Fold 0 Val (`Clean` / `Shifted` / `Hard-Shifted` WF1) |
| :--- | :--- | :--- | :--- |
| **Data Cleaning** | Keep `3,004 / 4,328` images (`-327` aliens, `-996` overexposed `2↔3`/`5↔6` swapped duplicates, `-1` blank); `StratifiedGroupKFold` over `4` near-dup pairs | Eliminates occluded/poisoned labels and cross-fold star-field leakage | Removes ~30% corrupted labels; Classes `2`/`3` reach `0.94–0.96` F1 |
| **Architecture & Init** | ImageNet-pretrained full fine-tune (`EfficientNetV2-S`) over from-scratch and `ResNet-50` / `DenseNet-121` | `~2.4k` train images/fold overfit from scratch; SE blocks help under channel shifts | Pretrained beat scratch across all 4 CNNs (`0.8361` vs `0.7990` max in Stage 2) and converged ~2× faster; `EfficientNetV2-S` topped Fold 0 (`ConvNeXt-Tiny` #2) |
| **Geometric Augs** | `H/V` flips, `112 px` reflect pad, ±180° rot, ±6% shift, isotropic zoom `0.88–1.12`, no shear/stretch | Matches galaxy rotational invariance and reflect-padded test borders without distorting ellipticity classes | `0.7802 → 0.8243` clean (`+4.4` pts) / `0.6732 → 0.7026` shifted / `0.5663` hard-shifted |
| **Photometric & Blur** | `ColorJitter` (`brightness 0.55–1.45`, `contrast 0.70–1.35`, `sat 0.65–1.35`, `hue 0.35`, `p=0.7`) + `GaussianBlur(k=7, σ 0.4–2.2, p=0.5)` | Directly matches observed test shifts (`32.5%` blurred, `18.5%` dimmed/brightened, `~10–15%` hue-rotated) | `0.8243 → 0.8450` clean / `0.7026 → 0.8338` shifted (`+13.1` pts) / `0.5663 → 0.8215` hard-shifted (`+25.5` pts) |
| **MixUp** | Dropped (`Beta(0.3, 0.3)` removed in `stage3a`/`stage3e`) | Redundant once strong photometric + blur jitter is active | Adding MixUp (`stage3b`) dropped clean `0.8450 → 0.8305` and hard-shifted `0.8215 → 0.7938` |
| **Preprocessing** | Per-channel outer-`16 px` sky-median leveling to `33/255` (applied post-corruption in train and on raw test); rejected `asinh` stretch | Cancels global sky/exposure offsets without compressing bulge-vs-disk contrast | Sky-leveling (`stage3e`) lifted clean `0.8450 → 0.8519` (`+0.7` pts, shifted `0.8093`, hard-shifted `0.8053`); `asinh` (`stage3d`) hurt clean to `0.8141` |
| **Inference** | Native `256×256` with 4× dihedral TTA (`id`, `H-flip`, `V-flip`, `90° rot`) | Averages predictions across exact spatial symmetries | Public LB **`0.8328`** from a single fold, tracking between clean (`0.8519`) and shifted (`0.8093`) local val |

## Next Focus
Finishing Folds `2–4` of `stage3e_skylevel_photo` (`EfficientNetV2-S` + `ConvNeXt-Tiny`) for a full 5-fold ensemble, probing remaining test artifacts (color bands/stripes), and addressing the primary confusion modes (Class `0` F1 `0.554`, Classes `6`/`7` `~0.75`).

---

## Technical Details

| Field | Value |
| :--- | :--- |
| **Date** | 2026-11-26 |
| **Public LB** | `0.8328` Weighted F1 (Top 20; top score `0.8765`) |
| **Local Fold 0 Val (`n=601`)** | Clean WF1 `0.8519`, Macro F1 `0.8311`, Acc `0.8536`; Shifted WF1 `0.8093`; Hard-Shifted WF1 `0.8053` |
| **Model** | `torchvision.models.efficientnet_v2_s` (`IMAGENET1K_V1`), `Dropout(0.2) + Linear(1280, 10)`, `~20.2M` params |
| **Training** | `AdamW` (`backbone_lr=1e-4`, `head_lr=1e-3`, `wd=1e-2` excl. 1D norms/biases), 2-ep warmup + cosine over 40 ep (`patience=12`, best ep `32`), `bs=32`, unweighted CE, grad clip `1.0`, `bf16` AMP, `seed=42` |
| **Split** | Fold `0` of `artifacts/dataset/clean_folds.csv` (`2,403` train / `601` val) |
| **Checkpoint** | `artifacts/checkpoints/stage3_5fold/efficientnet_v2_s__pretrained__stage3e_skylevel_photo__fold0.pt` |
| **Submission CSV** | `artifacts/submissions/sub_best_single_efficientnet_v2_s_stage3e_fold0_tta4.csv` (`14,736` rows) |
